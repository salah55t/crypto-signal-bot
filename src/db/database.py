"""
Database for Crypto Signal Bot - Statistics & Monitoring.

Supports BOTH:
  - PostgreSQL (Render free tier, 90 days then $7/month) — set DATABASE_URL env var
  - SQLite (default, file-based) — fallback when DATABASE_URL not set

Tables:
  - runs: every bot cycle (timestamp, duration, signals found)
  - recommendations: every recommendation ever made
  - strategy_signals: per-strategy signals for each recommendation
  - positions: open and closed positions with P&L
  - daily_stats: aggregated per-day stats
  - bottom_candidates: history of bottom scans

Auto-detects database type from connection string:
  - postgresql://... → PostgreSQL
  - default → SQLite (data/bot_stats.db)
"""
import os
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone
from config.settings import PROJECT_ROOT
from src.utils.logger import log

# Database configuration
DB_DIR = Path(os.getenv("DB_DIR", str(PROJECT_ROOT / "data")))
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(os.getenv("DB_PATH", str(DB_DIR / "bot_stats.db")))
DATABASE_URL = os.getenv("DATABASE_URL", "")  # Render PostgreSQL sets this automatically


# Detect database type
USE_POSTGRES = DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://")


# ============================================================
# SQL SCHEMAS (different syntax for SQLite vs PostgreSQL)
# ============================================================

SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    duration_seconds REAL,
    symbols_analyzed INTEGER,
    signals_passed INTEGER,
    recommendations_count INTEGER,
    bottom_candidates_count INTEGER,
    mode TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT,
    confidence REAL,
    weighted_score REAL,
    current_price REAL,
    expected_rise_pct REAL,
    stop_loss REAL,
    take_profit REAL,
    risk_reward_ratio REAL,
    atr_pct REAL,
    timeframe TEXT,
    boosted_from_bottom INTEGER DEFAULT 0,
    paper INTEGER DEFAULT 1,
    opened_position INTEGER DEFAULT 0,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE TABLE IF NOT EXISTS strategy_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id INTEGER,
    strategy_name TEXT NOT NULL,
    direction TEXT,
    score REAL,
    confidence REAL,
    reasons TEXT,
    FOREIGN KEY (recommendation_id) REFERENCES recommendations(id)
);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    direction TEXT,
    entry_price REAL,
    exit_price REAL,
    stop_loss REAL,
    take_profit REAL,
    size REAL,
    notional_usd REAL,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    pnl REAL,
    pnl_pct REAL,
    confidence REAL,
    paper INTEGER DEFAULT 1,
    buy_order_id TEXT,
    oco_order_id TEXT,
    close_reason TEXT,
    risk_updates_count INTEGER DEFAULT 0,
    risk_updates TEXT
);
CREATE TABLE IF NOT EXISTS daily_stats (
    date TEXT PRIMARY KEY,
    trades_opened INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    pnl REAL DEFAULT 0,
    starting_capital REAL,
    ending_capital REAL
);
CREATE TABLE IF NOT EXISTS bottom_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    score REAL,
    current_price REAL,
    recent_low REAL,
    distance_from_low_pct REAL,
    rsi REAL,
    atr_pct REAL,
    signals TEXT,
    patterns_detected TEXT
);
CREATE INDEX IF NOT EXISTS idx_rec_timestamp ON recommendations(timestamp);
CREATE INDEX IF NOT EXISTS idx_rec_symbol ON recommendations(symbol);
CREATE INDEX IF NOT EXISTS idx_pos_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_pos_status ON positions(exit_time);
CREATE INDEX IF NOT EXISTS idx_signals_strategy ON strategy_signals(strategy_name);
CREATE INDEX IF NOT EXISTS idx_bottom_timestamp ON bottom_candidates(timestamp);
"""

SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS runs (
    id SERIAL PRIMARY KEY,
    timestamp TEXT NOT NULL,
    duration_seconds REAL,
    symbols_analyzed INTEGER,
    signals_passed INTEGER,
    recommendations_count INTEGER,
    bottom_candidates_count INTEGER,
    mode TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES runs(id),
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT,
    confidence REAL,
    weighted_score REAL,
    current_price REAL,
    expected_rise_pct REAL,
    stop_loss REAL,
    take_profit REAL,
    risk_reward_ratio REAL,
    atr_pct REAL,
    timeframe TEXT,
    boosted_from_bottom INTEGER DEFAULT 0,
    paper INTEGER DEFAULT 1,
    opened_position INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS strategy_signals (
    id SERIAL PRIMARY KEY,
    recommendation_id INTEGER REFERENCES recommendations(id),
    strategy_name TEXT NOT NULL,
    direction TEXT,
    score REAL,
    confidence REAL,
    reasons TEXT
);
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    direction TEXT,
    entry_price REAL,
    exit_price REAL,
    stop_loss REAL,
    take_profit REAL,
    size REAL,
    notional_usd REAL,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    pnl REAL,
    pnl_pct REAL,
    confidence REAL,
    paper INTEGER DEFAULT 1,
    buy_order_id TEXT,
    oco_order_id TEXT,
    close_reason TEXT,
    risk_updates_count INTEGER DEFAULT 0,
    risk_updates TEXT
);
CREATE TABLE IF NOT EXISTS daily_stats (
    date TEXT PRIMARY KEY,
    trades_opened INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    pnl REAL DEFAULT 0,
    starting_capital REAL,
    ending_capital REAL
);
CREATE TABLE IF NOT EXISTS bottom_candidates (
    id SERIAL PRIMARY KEY,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    score REAL,
    current_price REAL,
    recent_low REAL,
    distance_from_low_pct REAL,
    rsi REAL,
    atr_pct REAL,
    signals TEXT,
    patterns_detected TEXT
);
CREATE INDEX IF NOT EXISTS idx_rec_timestamp ON recommendations(timestamp);
CREATE INDEX IF NOT EXISTS idx_rec_symbol ON recommendations(symbol);
CREATE INDEX IF NOT EXISTS idx_pos_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_pos_status ON positions(exit_time);
CREATE INDEX IF NOT EXISTS idx_signals_strategy ON strategy_signals(strategy_name);
CREATE INDEX IF NOT EXISTS idx_bottom_timestamp ON bottom_candidates(timestamp);
"""


class Database:
    """Database wrapper — supports PostgreSQL (Render) and SQLite (local)."""

    def __init__(self, db_url: str = None, db_path: Path = None):
        self.db_url = db_url or DATABASE_URL
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.use_postgres = self.db_url.startswith("postgresql://") or self.db_url.startswith("postgres://")
        self._conn: Optional[Any] = None
        # Auto-init on first use
        self._ensure_init()

    def init(self):
        """Public explicit initialization (alias of _ensure_init).

        web/app.py startup_event() calls db.init(); previously this raised
        "'Database' object has no attribute 'init'" at every startup
        (harmless - the constructor already lazy-inits - but noisy).
        """
        self._ensure_init()

    def _ensure_init(self):
        """Initialize DB schema on first use."""
        try:
            schema = SCHEMA_POSTGRES if self.use_postgres else SCHEMA_SQLITE
            with self._connect() as conn:
                if self.use_postgres:
                    # PostgreSQL supports executing multiple statements
                    cur = conn.cursor()
                    cur.execute(schema)
                    conn.commit()
                else:
                    # SQLite uses executescript
                    conn.executescript(schema)
                    conn.commit()
            db_type = "PostgreSQL" if self.use_postgres else "SQLite"
            location = self.db_url if self.use_postgres else str(self.db_path)
            log.info(f"[green]Database initialized[/] ({db_type}): {location[:50]}{'...' if len(location) > 50 else ''}")
        except Exception as e:
            log.error(f"Database init failed: {e}")
            # Fall back to SQLite if PostgreSQL fails
            if self.use_postgres:
                log.warning("[yellow]Falling back to SQLite[/]")
                self.use_postgres = False
                self.db_url = ""
                try:
                    with self._connect() as conn:
                        conn.executescript(SCHEMA_SQLITE)
                        conn.commit()
                    log.info(f"[green]SQLite fallback initialized[/] at {self.db_path}")
                except Exception as e2:
                    log.error(f"SQLite fallback also failed: {e2}")

    def _connect(self):
        """Get a database connection."""
        if self.use_postgres:
            import psycopg2
            conn = psycopg2.connect(self.db_url)
            conn.autocommit = False
            return conn
        else:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            return conn

    def _exec_queries(self, conn, sql: str, params: tuple = None):
        """Execute SQL with params — works for both DB types."""
        cur = conn.cursor()
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        return cur

    def _exec_many(self, conn, sql: str):
        """Execute script — works for both DB types."""
        if self.use_postgres:
            cur = conn.cursor()
            cur.execute(sql)
        else:
            conn.executescript(sql)

    def _fetch_all(self, cur) -> List[Dict]:
        """Fetch all rows as dicts — works for both DB types."""
        if self.use_postgres:
            # psycopg2: cur.description has column names
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        else:
            # sqlite3: Row factory returns Row objects
            return [dict(r) for r in cur.fetchall()]

    def _lastrowid(self, cur) -> int:
        """Get last inserted row ID — works for both DB types."""
        if self.use_postgres:
            # psycopg2: use RETURNING id
            try:
                return cur.fetchone()[0]
            except Exception:
                return 0
        else:
            return cur.lastrowid

    # ============================================
    # INSERT FUNCTIONS
    # ============================================

    def log_run(self, timestamp: str, duration_seconds: float,
                symbols_analyzed: int, signals_passed: int,
                recommendations_count: int, bottom_candidates_count: int,
                mode: str = "paper", error: str = None) -> int:
        """Log a bot run (analysis cycle). Returns the run ID."""
        with self._connect() as conn:
            # For PostgreSQL use RETURNING id; for SQLite use lastrowid
            if self.use_postgres:
                cur = conn.cursor()
                cur.execute(
                    """INSERT INTO runs (timestamp, duration_seconds, symbols_analyzed,
                        signals_passed, recommendations_count, bottom_candidates_count, mode, error)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                    (timestamp, duration_seconds, symbols_analyzed, signals_passed,
                     recommendations_count, bottom_candidates_count, mode, error)
                )
                run_id = cur.fetchone()[0]
            else:
                cur = conn.execute(
                    """INSERT INTO runs (timestamp, duration_seconds, symbols_analyzed,
                        signals_passed, recommendations_count, bottom_candidates_count, mode, error)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (timestamp, duration_seconds, symbols_analyzed, signals_passed,
                     recommendations_count, bottom_candidates_count, mode, error)
                )
                run_id = cur.lastrowid
            conn.commit()
            return run_id

    def log_recommendation(self, run_id: int, rec: Dict) -> int:
        """Log a recommendation. Returns the recommendation ID."""
        with self._connect() as conn:
            cur = conn.cursor()
            if self.use_postgres:
                cur.execute(
                    """INSERT INTO recommendations
                        (run_id, timestamp, symbol, direction, confidence, weighted_score,
                         current_price, expected_rise_pct, stop_loss, take_profit,
                         risk_reward_ratio, atr_pct, timeframe, boosted_from_bottom, paper)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                    (run_id, rec.get("analyzed_at") or rec.get("timestamp"),
                     rec.get("symbol"), rec.get("direction"),
                     rec.get("confidence", 0), rec.get("weighted_score", 0),
                     rec.get("current_price", 0), rec.get("expected_rise_pct", 0),
                     rec.get("stop_loss", 0), rec.get("take_profit", 0),
                     rec.get("risk_reward_ratio", 0), rec.get("atr_pct", 0),
                     rec.get("timeframe"), 1 if rec.get("boosted_from_bottom") else 0,
                     1 if rec.get("paper", True) else 0)
                )
                rec_id = cur.fetchone()[0]
            else:
                cur.execute(
                    """INSERT INTO recommendations
                        (run_id, timestamp, symbol, direction, confidence, weighted_score,
                         current_price, expected_rise_pct, stop_loss, take_profit,
                         risk_reward_ratio, atr_pct, timeframe, boosted_from_bottom, paper)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (run_id, rec.get("analyzed_at") or rec.get("timestamp"),
                     rec.get("symbol"), rec.get("direction"),
                     rec.get("confidence", 0), rec.get("weighted_score", 0),
                     rec.get("current_price", 0), rec.get("expected_rise_pct", 0),
                     rec.get("stop_loss", 0), rec.get("take_profit", 0),
                     rec.get("risk_reward_ratio", 0), rec.get("atr_pct", 0),
                     rec.get("timeframe"), 1 if rec.get("boosted_from_bottom") else 0,
                     1 if rec.get("paper", True) else 0)
                )
                rec_id = cur.lastrowid

            # Insert strategy signals
            placeholder = "%s" if self.use_postgres else "?"
            for sig in rec.get("signals", []):
                cur.execute(
                    f"""INSERT INTO strategy_signals
                        (recommendation_id, strategy_name, direction, score, confidence, reasons)
                        VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})""",
                    (rec_id, sig.get("strategy"), sig.get("direction"),
                     sig.get("score", 0), sig.get("confidence", 0),
                     json.dumps(sig.get("reasons", []), ensure_ascii=False))
                )
            conn.commit()
            return rec_id

    def log_position_opened(self, position: Dict) -> int:
        """Log when a position is opened. Returns the position ID."""
        placeholder = "%s" if self.use_postgres else "?"
        with self._connect() as conn:
            cur = conn.cursor()
            sql = f"""
                INSERT INTO positions
                    (symbol, direction, entry_price, stop_loss, take_profit,
                     size, notional_usd, entry_time, confidence, paper,
                     buy_order_id, oco_order_id, risk_updates_count, risk_updates)
                VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder}, {placeholder})
            """
            if self.use_postgres:
                sql += " RETURNING id"
            cur.execute(sql, (
                position.get("symbol"), position.get("direction"),
                position.get("entry_price"), position.get("stop_loss"),
                position.get("take_profit"), position.get("size", 0),
                position.get("notional_usd", 0),
                position.get("entry_time"), position.get("confidence", 0),
                1 if position.get("paper", True) else 0,
                position.get("buy_order_id"), position.get("oco_order_id"),
                len(position.get("risk_updates", [])),
                json.dumps(position.get("risk_updates", []), default=str)
            ))
            pos_id = cur.fetchone()[0] if self.use_postgres else cur.lastrowid
            conn.commit()
            return pos_id

    def log_position_closed(self, symbol: str, entry_time: str, exit_price: float,
                              exit_time: str, pnl: float, pnl_pct: float,
                              close_reason: str) -> int:
        """Update a position when it's closed."""
        placeholder = "%s" if self.use_postgres else "?"
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""UPDATE positions SET
                    exit_price = {placeholder}, exit_time = {placeholder}, pnl = {placeholder},
                    pnl_pct = {placeholder}, close_reason = {placeholder}
                    WHERE symbol = {placeholder} AND entry_time = {placeholder}""",
                (exit_price, exit_time, pnl, pnl_pct, close_reason, symbol, entry_time)
            )
            conn.commit()
            return cur.rowcount

    def log_position_risk_update(self, symbol: str, entry_time: str,
                                   updates_count: int, updates_json: str):
        """Update the risk_updates for a position."""
        placeholder = "%s" if self.use_postgres else "?"
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""UPDATE positions SET risk_updates_count = {placeholder}, risk_updates = {placeholder}
                    WHERE symbol = {placeholder} AND entry_time = {placeholder}""",
                (updates_count, updates_json, symbol, entry_time)
            )
            conn.commit()

    def log_bottom_candidate(self, candidate: Dict):
        """Log a bottom candidate."""
        placeholder = "%s" if self.use_postgres else "?"
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""INSERT INTO bottom_candidates
                    (timestamp, symbol, score, current_price, recent_low,
                     distance_from_low_pct, rsi, atr_pct, signals, patterns_detected)
                    VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder},
                            {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})""",
                (candidate.get("analyzed_at"),
                 candidate.get("symbol"), candidate.get("score", 0),
                 candidate.get("current_price"), candidate.get("recent_low"),
                 candidate.get("distance_from_low_pct", 0),
                 candidate.get("rsi"), candidate.get("atr_pct", 0),
                 json.dumps(candidate.get("signals", []), ensure_ascii=False, default=str),
                 json.dumps(candidate.get("patterns_detected", []), ensure_ascii=False))
            )
            conn.commit()

    def update_daily_stats(self, date: str, trades_opened: int = None,
                             wins: int = None, losses: int = None,
                             pnl_delta: float = None,
                             starting_capital: float = None,
                             ending_capital: float = None):
        """Insert or update daily stats (UPSERT)."""
        placeholder = "%s" if self.use_postgres else "?"
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT * FROM daily_stats WHERE date = {placeholder}", (date,)
            )
            row = cur.fetchone()
            if row:
                # Update existing
                new_trades = trades_opened if trades_opened is not None else (row["trades_opened"] if not self.use_postgres else row[1])
                new_wins = wins if wins is not None else (row["wins"] if not self.use_postgres else row[2])
                new_losses = losses if losses is not None else (row["losses"] if not self.use_postgres else row[3])
                current_pnl = row["pnl"] if not self.use_postgres else row[4]
                new_pnl = (current_pnl + pnl_delta) if pnl_delta is not None else current_pnl
                cur.execute(
                    f"""UPDATE daily_stats SET trades_opened = {placeholder}, wins = {placeholder},
                        losses = {placeholder}, pnl = {placeholder},
                        starting_capital = {placeholder}, ending_capital = {placeholder}
                        WHERE date = {placeholder}""",
                    (new_trades, new_wins, new_losses, new_pnl,
                     starting_capital or (row["starting_capital"] if not self.use_postgres else row[5]),
                     ending_capital or (row["ending_capital"] if not self.use_postgres else row[6]), date)
                )
            else:
                cur.execute(
                    f"""INSERT INTO daily_stats
                        (date, trades_opened, wins, losses, pnl, starting_capital, ending_capital)
                        VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})""",
                    (date, trades_opened or 0, wins or 0, losses or 0,
                     pnl_delta or 0, starting_capital, ending_capital)
                )
            conn.commit()

    # ============================================
    # QUERY FUNCTIONS
    # ============================================

    def get_runs_history(self, limit: int = 50) -> List[Dict]:
        placeholder = "%s" if self.use_postgres else "?"
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT * FROM runs ORDER BY timestamp DESC LIMIT {placeholder}", (limit,)
            )
            return self._fetch_all(cur)

    def get_strategy_performance(self, days: int = 30) -> List[Dict]:
        placeholder = "%s" if self.use_postgres else "?"
        with self._connect() as conn:
            cur = conn.cursor()
            if self.use_postgres:
                # PostgreSQL date arithmetic
                cur.execute(
                    f"""SELECT
                        strategy_name,
                        COUNT(*) as total_signals,
                        SUM(CASE WHEN strategy_signals.direction = 'bullish' THEN 1 ELSE 0 END) as bullish_signals,
                        SUM(CASE WHEN strategy_signals.direction = 'bearish' THEN 1 ELSE 0 END) as bearish_signals,
                        SUM(CASE WHEN strategy_signals.direction = 'neutral' THEN 1 ELSE 0 END) as neutral_signals,
                        AVG(strategy_signals.score) as avg_score,
                        AVG(strategy_signals.confidence) as avg_confidence
                        FROM strategy_signals
                        JOIN recommendations ON strategy_signals.recommendation_id = recommendations.id
                        WHERE recommendations.timestamp >= NOW() - INTERVAL %s
                        GROUP BY strategy_name
                        ORDER BY total_signals DESC""",
                    (f"{days} days",)
                )
            else:
                # SQLite date arithmetic
                cur.execute(
                    f"""SELECT
                        strategy_name,
                        COUNT(*) as total_signals,
                        SUM(CASE WHEN strategy_signals.direction = 'bullish' THEN 1 ELSE 0 END) as bullish_signals,
                        SUM(CASE WHEN strategy_signals.direction = 'bearish' THEN 1 ELSE 0 END) as bearish_signals,
                        SUM(CASE WHEN strategy_signals.direction = 'neutral' THEN 1 ELSE 0 END) as neutral_signals,
                        AVG(strategy_signals.score) as avg_score,
                        AVG(strategy_signals.confidence) as avg_confidence
                        FROM strategy_signals
                        JOIN recommendations ON strategy_signals.recommendation_id = recommendations.id
                        WHERE recommendations.timestamp >= datetime('now', ?)
                        GROUP BY strategy_name
                        ORDER BY total_signals DESC""",
                    (f"-{days} days",)
                )
            return self._fetch_all(cur)

    def get_recommendations_history(self, limit: int = 100,
                                       symbol: str = None) -> List[Dict]:
        placeholder = "%s" if self.use_postgres else "?"
        with self._connect() as conn:
            cur = conn.cursor()
            if symbol:
                cur.execute(
                    f"""SELECT * FROM recommendations
                        WHERE symbol LIKE {placeholder} ORDER BY timestamp DESC LIMIT {placeholder}""",
                    (f"%{symbol}%", limit)
                )
            else:
                cur.execute(
                    f"""SELECT * FROM recommendations ORDER BY timestamp DESC LIMIT {placeholder}""",
                    (limit,)
                )
            return self._fetch_all(cur)

    def get_positions_history(self, closed_only: bool = False,
                                limit: int = 50) -> List[Dict]:
        placeholder = "%s" if self.use_postgres else "?"
        with self._connect() as conn:
            cur = conn.cursor()
            if closed_only:
                cur.execute(
                    f"""SELECT * FROM positions
                        WHERE exit_time IS NOT NULL
                        ORDER BY exit_time DESC LIMIT {placeholder}""",
                    (limit,)
                )
            else:
                cur.execute(
                    f"""SELECT * FROM positions ORDER BY entry_time DESC LIMIT {placeholder}""",
                    (limit,)
                )
            return self._fetch_all(cur)

    def get_daily_stats(self, limit: int = 30) -> List[Dict]:
        placeholder = "%s" if self.use_postgres else "?"
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT * FROM daily_stats ORDER BY date DESC LIMIT {placeholder}",
                (limit,)
            )
            return self._fetch_all(cur)

    def get_bottom_candidates_history(self, limit: int = 50,
                                          min_score: float = 50) -> List[Dict]:
        placeholder = "%s" if self.use_postgres else "?"
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""SELECT * FROM bottom_candidates
                    WHERE score >= {placeholder}
                    ORDER BY timestamp DESC LIMIT {placeholder}""",
                (min_score, limit)
            )
            return self._fetch_all(cur)

    def get_summary_stats(self) -> Dict:
        """Get aggregated summary statistics."""
        with self._connect() as conn:
            cur = conn.cursor()
            # Total runs
            cur.execute("SELECT COUNT(*) as c FROM runs")
            row = cur.fetchone()
            total_runs = row[0] if self.use_postgres else row["c"]

            cur.execute("SELECT COUNT(*) as c FROM recommendations")
            row = cur.fetchone()
            total_recs = row[0] if self.use_postgres else row["c"]

            cur.execute("SELECT COUNT(*) as c FROM positions")
            row = cur.fetchone()
            total_positions = row[0] if self.use_postgres else row["c"]

            cur.execute("SELECT COUNT(*) as c FROM positions WHERE exit_time IS NOT NULL")
            row = cur.fetchone()
            closed_positions = row[0] if self.use_postgres else row["c"]

            cur.execute(
                """SELECT
                    COUNT(*) as total_closed,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                    SUM(pnl) as total_pnl,
                    AVG(pnl_pct) as avg_pnl_pct
                    FROM positions WHERE exit_time IS NOT NULL"""
            )
            row = cur.fetchone()
            if self.use_postgres:
                total_closed = row[0] or 0
                wins = row[1] or 0
                losses = row[2] or 0
                total_pnl = float(row[3] or 0)
                avg_pnl_pct = float(row[4] or 0)
            else:
                total_closed = row["total_closed"] or 0
                wins = row["wins"] or 0
                losses = row["losses"] or 0
                total_pnl = float(row["total_pnl"] or 0)
                avg_pnl_pct = float(row["avg_pnl_pct"] or 0)

            cur.execute("SELECT COUNT(*) as c FROM bottom_candidates")
            row = cur.fetchone()
            total_bottoms = row[0] if self.use_postgres else row["c"]

            cur.execute("SELECT * FROM runs ORDER BY timestamp DESC LIMIT 1")
            last_run_row = cur.fetchone()
            last_run = None
            if last_run_row:
                if self.use_postgres:
                    cols = [d[0] for d in cur.description]
                    last_run = dict(zip(cols, last_run_row))
                else:
                    last_run = dict(last_run_row)

            return {
                "total_runs": total_runs,
                "total_recommendations": total_recs,
                "total_positions": total_positions,
                "closed_positions": closed_positions,
                "open_positions": total_positions - closed_positions,
                "total_closed_trades": total_closed,
                "total_wins": wins,
                "total_losses": losses,
                "win_rate": (wins / total_closed * 100) if total_closed else 0,
                "total_pnl": total_pnl,
                "avg_pnl_pct": avg_pnl_pct,
                "total_bottom_candidates": total_bottoms,
                "last_run": last_run,
                "database_type": "PostgreSQL" if self.use_postgres else "SQLite",
            }

    def reset_all_history(self) -> Dict:
        """Delete ALL records from all tables (truncate). Use with caution."""
        tables = ["strategy_signals", "recommendations", "positions",
                  "daily_stats", "bottom_candidates", "runs"]
        deleted = {}
        with self._connect() as conn:
            cur = conn.cursor()
            for table in tables:
                try:
                    if self.use_postgres:
                        # PostgreSQL: use TRUNCATE with CASCADE for FK constraints
                        cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")
                    else:
                        cur.execute(f"DELETE FROM {table}")
                    deleted[table] = "truncated"
                except Exception as e:
                    deleted[table] = f"error: {e}"
            conn.commit()
        log.warning(f"[yellow]ALL HISTORY RESET[/] - tables: {list(deleted.keys())}")
        return {"status": "success", "tables_cleared": deleted}


# Singleton
db = Database()
