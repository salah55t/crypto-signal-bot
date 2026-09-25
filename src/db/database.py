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
import time
import sqlite3
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone, timedelta
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
    risk_updates TEXT,
    -- v5.11 persistent trade ledger
    trade_uid TEXT,
    status TEXT,
    initial_sl REAL,
    initial_tp REAL,
    tp2 REAL,
    initial_notional REAL,
    strategy TEXT,
    realized_pnl REAL,
    partial_count INTEGER,
    tp1_taken INTEGER,
    peak_price REAL,
    trough_price REAL,
    mfe_pct REAL,
    mae_pct REAL,
    entry_fee REAL,
    exit_fee REAL,
    total_pnl REAL,
    total_pnl_pct REAL,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS trade_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_uid TEXT,
    symbol TEXT,
    event_time TEXT NOT NULL,
    event_type TEXT NOT NULL,
    old_sl REAL,
    new_sl REAL,
    old_tp REAL,
    new_tp REAL,
    price REAL,
    pnl REAL,
    pnl_pct REAL,
    reason TEXT
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
-- NOTE: idx_pos_uid / idx_events_uid are created by _migrate() AFTER the
-- ledger columns exist (they must not run before the ALTER TABLEs on
-- pre-v5.11 databases).
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
    risk_updates TEXT,
    -- v5.11 persistent trade ledger
    trade_uid TEXT,
    status TEXT,
    initial_sl REAL,
    initial_tp REAL,
    tp2 REAL,
    initial_notional REAL,
    strategy TEXT,
    realized_pnl REAL,
    partial_count INTEGER,
    tp1_taken INTEGER,
    peak_price REAL,
    trough_price REAL,
    mfe_pct REAL,
    mae_pct REAL,
    entry_fee REAL,
    exit_fee REAL,
    total_pnl REAL,
    total_pnl_pct REAL,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS trade_events (
    id SERIAL PRIMARY KEY,
    trade_uid TEXT,
    symbol TEXT,
    event_time TEXT NOT NULL,
    event_type TEXT NOT NULL,
    old_sl REAL,
    new_sl REAL,
    old_tp REAL,
    new_tp REAL,
    price REAL,
    pnl REAL,
    pnl_pct REAL,
    reason TEXT
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
-- NOTE: idx_pos_uid / idx_events_uid are created by _migrate() AFTER the
-- ledger columns exist (they must not run before the ALTER TABLEs on
-- pre-v5.11 databases).
CREATE INDEX IF NOT EXISTS idx_signals_strategy ON strategy_signals(strategy_name);
CREATE INDEX IF NOT EXISTS idx_bottom_timestamp ON bottom_candidates(timestamp);
"""


# v5.11: idempotent column migration for EXISTING databases (fresh installs
# already get the columns from the schema above).
LEDGER_COLUMNS = {
    "trade_uid": "TEXT",
    "status": "TEXT",
    "initial_sl": "REAL",
    "initial_tp": "REAL",
    "tp2": "REAL",
    "initial_notional": "REAL",
    "strategy": "TEXT",
    "realized_pnl": "REAL",
    "partial_count": "INTEGER",
    "tp1_taken": "INTEGER",
    "peak_price": "REAL",
    "trough_price": "REAL",
    "mfe_pct": "REAL",
    "mae_pct": "REAL",
    "entry_fee": "REAL",
    "exit_fee": "REAL",
    "total_pnl": "REAL",
    "total_pnl_pct": "REAL",
    "updated_at": "TEXT",
}

TRADE_EVENTS_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS trade_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_uid TEXT,
    symbol TEXT,
    event_time TEXT NOT NULL,
    event_type TEXT NOT NULL,
    old_sl REAL, new_sl REAL, old_tp REAL, new_tp REAL,
    price REAL, pnl REAL, pnl_pct REAL, reason TEXT
)
"""

TRADE_EVENTS_DDL_PG = """
CREATE TABLE IF NOT EXISTS trade_events (
    id SERIAL PRIMARY KEY,
    trade_uid TEXT,
    symbol TEXT,
    event_time TEXT NOT NULL,
    event_type TEXT NOT NULL,
    old_sl REAL, new_sl REAL, old_tp REAL, new_tp REAL,
    price REAL, pnl REAL, pnl_pct REAL, reason TEXT
)
"""


class Database:
    """Database wrapper — supports PostgreSQL (Render) and SQLite (local)."""

    def __init__(self, db_url: str = None, db_path: Path = None):
        self.db_url = db_url or DATABASE_URL
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.use_postgres = self.db_url.startswith("postgresql://") or self.db_url.startswith("postgres://")
        self._conn: Optional[Any] = None
        self._initialized = False
        self._next_retry = 0.0
        # Init schema now; on PostgreSQL failure we do NOT silently fall back
        # to ephemeral SQLite (that wiped positions/stats on every redeploy).
        self._ensure_init()

    def init(self):
        """Public explicit initialization (alias of _ensure_init).

        web/app.py startup_event() calls db.init(); previously this raised
        "'Database' object has no attribute 'init'" at every startup
        (harmless - the constructor already lazy-inits - but noisy).
        """
        self._ensure_init()

    def _ensure_init(self):
        """Initialize DB schema (PostgreSQL with retries / SQLite direct).

        v5.1 hardening: when DATABASE_URL is configured but Postgres is
        unreachable (cold Render DB, restart, rotated password), retry a few
        times and keep retrying lazily on later use. The old behaviour fell
        back to a fresh ephemeral SQLite file, silently splitting data and
        losing open positions on the next deploy.
        """
        if self._initialized:
            return
        if self.use_postgres:
            last_err = None
            for attempt in range(3):
                try:
                    with self._connect_raw() as conn:
                        cur = conn.cursor()
                        cur.execute(SCHEMA_POSTGRES)
                        conn.commit()
                    self._initialized = True
                    self._migrate()
                    log.info("[green]Database initialized[/] (PostgreSQL) "
                             f"after {attempt + 1} attempt(s)")
                    return
                except Exception as e:
                    last_err = e
                    wait = 2 ** attempt * 2  # 2s, 4s, 8s (lets cold DBs wake)
                    log.warning(f"PostgreSQL init attempt {attempt + 1}/3 "
                                f"failed: {e} - retrying in {wait}s")
                    time.sleep(wait)
            self._next_retry = time.monotonic() + 60.0
            log.error("PostgreSQL init failed after retries: "
                      f"{last_err}. STAYING on PostgreSQL - will retry on "
                      "next use (no silent SQLite fallback; data written to "
                      "ephemeral SQLite is lost on redeploy). Check that "
                      "DATABASE_URL matches the current DB credentials.")
            return
        try:
            with self._connect_raw() as conn:
                conn.executescript(SCHEMA_SQLITE)
                conn.commit()
            self._initialized = True
            self._migrate()
            log.info(f"[green]Database initialized[/] (SQLite): {self.db_path}")
        except Exception as e:
            log.error(f"SQLite init failed: {e}")

    def _migrate(self):
        """v5.11 persistent trade ledger migration (idempotent).

        - Adds the ledger columns to an existing `positions` table
        - Creates the append-only `trade_events` audit table
        - Backfills status for legacy rows (exit_time IS NOT NULL -> closed)
        Runs on BOTH dialects after a successful schema init.
        """
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                # existing columns of `positions`
                if self.use_postgres:
                    cur.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'positions'"
                    )
                    cols = {r[0] for r in cur.fetchall()}
                else:
                    cur.execute("PRAGMA table_info(positions)")
                    cols = {r[1] for r in cur.fetchall()}
                added = []
                for col, typ in LEDGER_COLUMNS.items():
                    if col not in cols:
                        cur.execute(
                            f"ALTER TABLE positions ADD COLUMN {col} {typ}"
                        )
                        added.append(col)
                # trade_events audit table + indexes
                cur.execute(
                    TRADE_EVENTS_DDL_PG if self.use_postgres
                    else TRADE_EVENTS_DDL_SQLITE
                )
                for idx_sql in (
                    "CREATE INDEX IF NOT EXISTS idx_pos_uid ON positions(trade_uid)",
                    "CREATE INDEX IF NOT EXISTS idx_events_uid ON trade_events(trade_uid)",
                ):
                    try:
                        cur.execute(idx_sql)
                    except Exception:
                        pass  # PG re-run: index may already exist
                # backfill status for legacy rows
                cur.execute(
                    f"UPDATE positions SET status = 'closed' "
                    f"WHERE status IS NULL AND exit_time IS NOT NULL"
                )
                cur.execute(
                    f"UPDATE positions SET status = 'open' "
                    f"WHERE status IS NULL"
                )
                conn.commit()
            if added:
                log.info(
                    f"[green]Trade-ledger migration[/]: +{len(added)} "
                    f"column(s) on positions ({', '.join(added[:6])}"
                    f"{'...' if len(added) > 6 else ''})"
                )
        except Exception as e:
            # Never kill the bot over a migration issue - log and continue.
            log.warning(f"Ledger migration skipped: {e}")

    def _connect(self):
        """Get a database connection (re-attempts Postgres init lazily)."""
        if self.use_postgres:
            if not self._initialized and time.monotonic() >= self._next_retry:
                self._ensure_init()
            if not self._initialized:
                raise RuntimeError(
                    "PostgreSQL not initialized (init retry scheduled); "
                    "refusing to write anywhere else to avoid silent data loss")
            import psycopg2
            conn = psycopg2.connect(self.db_url, connect_timeout=10)
            conn.autocommit = False
            return conn
        else:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            return conn

    def _connect_raw(self):
        """Raw connection used BY the initializer (no init recursion)."""
        if self.use_postgres:
            import psycopg2
            conn = psycopg2.connect(self.db_url, connect_timeout=10)
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
        """Log when a position is opened. Returns the position ID.

        v5.11: also writes the persistent-ledger columns (trade_uid, initial
        SL/TP, status='open', ...) so entry prices survive restarts/redeploys
        even when the ephemeral JSON file is lost.
        """
        placeholder = "%s" if self.use_postgres else "?"
        with self._connect() as conn:
            cur = conn.cursor()
            sql = f"""
                INSERT INTO positions
                    (symbol, direction, entry_price, stop_loss, take_profit,
                     size, notional_usd, entry_time, confidence, paper,
                     buy_order_id, oco_order_id, risk_updates_count, risk_updates,
                     trade_uid, status, initial_sl, initial_tp, tp2,
                     initial_notional, strategy, entry_fee,
                     peak_price, trough_price, mfe_pct, mae_pct,
                     realized_pnl, partial_count, tp1_taken)
                VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder})
            """
            entry_price = position.get("entry_price")
            entry = entry_price if isinstance(entry_price, (int, float)) else 0
            sl = position.get("stop_loss")
            tp = position.get("take_profit")
            params = (
                position.get("symbol"), position.get("direction"),
                entry_price, sl,
                tp, position.get("size", 0),
                position.get("notional_usd", 0),
                position.get("entry_time"), position.get("confidence", 0),
                1 if position.get("paper", True) else 0,
                position.get("buy_order_id"), position.get("oco_order_id"),
                len(position.get("risk_updates", [])),
                json.dumps(position.get("risk_updates", []), default=str),
                # v5.11 ledger columns
                position.get("trade_uid"),
                "open",
                sl, tp,
                position.get("take_profit_2") or tp,
                position.get("initial_notional_usd", position.get("notional_usd", 0)),
                position.get("strategy"),
                position.get("entry_fee", 0),
                position.get("peak_price", entry) or entry,
                position.get("trough_price", entry) or entry,
                position.get("mfe_pct", 0.0),
                position.get("mae_pct", 0.0),
                0.0, 0, 0,
            )
            if self.use_postgres:
                sql += " RETURNING id"
            cur.execute(sql, params)
            pos_id = cur.fetchone()[0] if self.use_postgres else cur.lastrowid
            # append the OPEN event to the audit trail
            self._insert_trade_event(cur, {
                "trade_uid": position.get("trade_uid"),
                "symbol": position.get("symbol"),
                "event_time": position.get("entry_time"),
                "event_type": "OPEN",
                "new_sl": sl, "new_tp": tp,
                "price": entry_price,
                "reason": "position opened",
            })
            conn.commit()
            return pos_id

    def _insert_trade_event(self, cur, ev: Dict):
        """Append one row to the trade_events audit table (no commit here)."""
        placeholder = "%s" if self.use_postgres else "?"
        cur.execute(
            f"""INSERT INTO trade_events
                (trade_uid, symbol, event_time, event_type,
                 old_sl, new_sl, old_tp, new_tp, price, pnl, pnl_pct, reason)
                VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder}, {placeholder},
                        {placeholder}, {placeholder}, {placeholder}, {placeholder})""",
            (ev.get("trade_uid"), ev.get("symbol"), ev.get("event_time"),
             ev.get("event_type"), ev.get("old_sl"), ev.get("new_sl"),
             ev.get("old_tp"), ev.get("new_tp"), ev.get("price"),
             ev.get("pnl"), ev.get("pnl_pct"), ev.get("reason"))
        )

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

    # ============================================
    # v5.11 PERSISTENT TRADE LEDGER
    # The DB is the source of truth for entry prices and SL/TP levels.
    # Every level change appends a trade_events row (audit trail).
    # ============================================

    @staticmethod
    def _uid_where(placeholder: str, trade_uid: str = None,
                   symbol: str = None, entry_time: str = None):
        """WHERE clause matching by trade_uid, falling back to the legacy
        (symbol, entry_time) pair for pre-v5.11 rows."""
        if trade_uid:
            return f"WHERE trade_uid = {placeholder}", (trade_uid,)
        return (f"WHERE symbol = {placeholder} AND entry_time = {placeholder} "
                f"AND (trade_uid IS NULL OR trade_uid = '')",
                (symbol, entry_time))

    def attach_trade_uid(self, symbol: str, entry_time: str,
                         trade_uid: str) -> int:
        """Tag an existing (pre-v5.11) row with a trade_uid. Returns rowcount."""
        placeholder = "%s" if self.use_postgres else "?"
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""UPDATE positions SET trade_uid = {placeholder},
                    status = COALESCE(status, 'open')
                    WHERE symbol = {placeholder} AND entry_time = {placeholder}
                    AND (trade_uid IS NULL OR trade_uid = '')""",
                (trade_uid, symbol, entry_time)
            )
            conn.commit()
            return cur.rowcount

    def update_position_levels(self, trade_uid: str = None, symbol: str = None,
                               entry_time: str = None, stop_loss: float = None,
                               take_profit: float = None, old_sl: float = None,
                               old_tp: float = None, price: float = None,
                               reason: str = "", event_type: str = "RISK_UPDATE",
                               peak_price: float = None, trough_price: float = None,
                               mfe_pct: float = None, mae_pct: float = None,
                               notional_usd: float = None, size: float = None,
                               tp1_taken: bool = None):
        """v5.11: persist current SL/TP (+ excursion snapshot) to the DB and
        append a trade_events audit row. Called on EVERY SL/TP update."""
        placeholder = "%s" if self.use_postgres else "?"
        where, params_w = self._uid_where(placeholder, trade_uid, symbol, entry_time)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""UPDATE positions SET
                    stop_loss = COALESCE({placeholder}, stop_loss),
                    take_profit = COALESCE({placeholder}, take_profit),
                    peak_price = COALESCE({placeholder}, peak_price),
                    trough_price = COALESCE({placeholder}, trough_price),
                    mfe_pct = COALESCE({placeholder}, mfe_pct),
                    mae_pct = COALESCE({placeholder}, mae_pct),
                    notional_usd = COALESCE({placeholder}, notional_usd),
                    size = COALESCE({placeholder}, size),
                    tp1_taken = COALESCE({placeholder}, tp1_taken),
                    updated_at = {placeholder}
                    {where}""",
                (stop_loss, take_profit, peak_price, trough_price,
                 mfe_pct, mae_pct, notional_usd, size,
                 None if tp1_taken is None else (1 if tp1_taken else 0),
                 now, *params_w)
            )
            self._insert_trade_event(cur, {
                "trade_uid": trade_uid, "symbol": symbol,
                "event_time": now, "event_type": event_type,
                "old_sl": old_sl, "new_sl": stop_loss,
                "old_tp": old_tp, "new_tp": take_profit,
                "price": price, "reason": reason,
            })
            conn.commit()

    def record_partial_close(self, trade_uid: str = None, symbol: str = None,
                             entry_time: str = None, remaining_notional: float = 0,
                             remaining_size: float = 0, realized_total: float = 0,
                             partial_count: int = 0, tp1_taken: bool = False,
                             stop_loss: float = None, take_profit: float = None,
                             exit_price: float = 0, exit_time: str = "",
                             chunk_pnl: float = 0, chunk_pnl_pct: float = 0,
                             fraction: float = 0, reason: str = ""):
        """v5.11 fix: a PARTIAL (TP1) close no longer pretends the trade is
        closed. It ACCUMULATES realized_pnl, syncs the remaining notional and
        the new SL/TP levels, and appends a TP1_PARTIAL event. The legacy
        path overwrote the row's pnl/exit_time - the TP1 profit was erased
        by the final close and stats were wrong."""
        placeholder = "%s" if self.use_postgres else "?"
        where, params_w = self._uid_where(placeholder, trade_uid, symbol, entry_time)
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""UPDATE positions SET
                    notional_usd = {placeholder},
                    size = {placeholder},
                    realized_pnl = {placeholder},
                    partial_count = {placeholder},
                    tp1_taken = {placeholder},
                    stop_loss = COALESCE({placeholder}, stop_loss),
                    take_profit = COALESCE({placeholder}, take_profit),
                    risk_updates_count = COALESCE(risk_updates_count, 0) + 1,
                    updated_at = {placeholder}
                    {where}""",
                (remaining_notional, remaining_size, realized_total,
                 partial_count, 1 if tp1_taken else 0,
                 stop_loss, take_profit, exit_time, *params_w)
            )
            self._insert_trade_event(cur, {
                "trade_uid": trade_uid, "symbol": symbol,
                "event_time": exit_time, "event_type": "TP1_PARTIAL",
                "old_sl": None, "new_sl": stop_loss,
                "old_tp": None, "new_tp": take_profit,
                "price": exit_price, "pnl": chunk_pnl,
                "pnl_pct": chunk_pnl_pct,
                "reason": f"{fraction*100:.0f}% closed - {reason}",
            })
            conn.commit()

    def close_trade(self, trade_uid: str = None, symbol: str = None,
                    entry_time: str = None, exit_price: float = 0,
                    exit_time: str = "", final_pnl: float = 0,
                    final_pnl_pct: float = 0, total_pnl: float = None,
                    total_pnl_pct: float = None, close_reason: str = "",
                    exit_fee: float = None, peak_price: float = None,
                    trough_price: float = None, mfe_pct: float = None,
                    mae_pct: float = None):
        """v5.11: close a trade by trade_uid and store the WHOLE-trade result:
        pnl = final chunk, total_pnl = partials + final chunk (accurate stats)."""
        placeholder = "%s" if self.use_postgres else "?"
        where, params_w = self._uid_where(placeholder, trade_uid, symbol, entry_time)
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""UPDATE positions SET
                    exit_price = {placeholder}, exit_time = {placeholder},
                    pnl = {placeholder}, pnl_pct = {placeholder},
                    close_reason = {placeholder},
                    exit_fee = COALESCE({placeholder}, exit_fee),
                    total_pnl = COALESCE({placeholder}, total_pnl),
                    total_pnl_pct = COALESCE({placeholder}, total_pnl_pct),
                    peak_price = COALESCE({placeholder}, peak_price),
                    trough_price = COALESCE({placeholder}, trough_price),
                    mfe_pct = COALESCE({placeholder}, mfe_pct),
                    mae_pct = COALESCE({placeholder}, mae_pct),
                    status = 'closed',
                    updated_at = {placeholder}
                    {where}""",
                (exit_price, exit_time, final_pnl, final_pnl_pct,
                 close_reason, exit_fee, total_pnl, total_pnl_pct,
                 peak_price, trough_price, mfe_pct, mae_pct,
                 exit_time, *params_w)
            )
            self._insert_trade_event(cur, {
                "trade_uid": trade_uid, "symbol": symbol,
                "event_time": exit_time, "event_type": "CLOSE",
                "price": exit_price, "pnl": final_pnl,
                "pnl_pct": final_pnl_pct, "reason": close_reason,
            })
            conn.commit()
            return cur.rowcount

    def get_open_positions(self) -> List[Dict]:
        """v5.11: open positions straight from the ledger (restore source)."""
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM positions WHERE status = 'open' "
                "ORDER BY entry_time ASC"
            )
            return self._fetch_all(cur)

    def get_trade_events(self, trade_uid: str, limit: int = 200) -> List[Dict]:
        """Full audit timeline of one trade (oldest first)."""
        placeholder = "%s" if self.use_postgres else "?"
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT * FROM trade_events WHERE trade_uid = {placeholder} "
                f"ORDER BY event_time ASC, id ASC LIMIT {placeholder}",
                (trade_uid, limit)
            )
            return self._fetch_all(cur)

    def get_performance_stats(self, days: int = 90) -> Dict:
        """v5.11 creative layer: whole-trade performance analytics.

        Computed in Python from closed rows so the same code runs on SQLite
        and PostgreSQL (no dialect-specific date math). Uses total_pnl when
        available (partials included), falling back to pnl for legacy rows.
        """
        rows = self.get_positions_history(closed_only=True, limit=500)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        trades = []
        for r in rows:
            exit_time = r.get("exit_time") or ""
            if exit_time and exit_time < cutoff:
                continue
            trades.append(r)
        if not trades:
            return {"closed_trades": 0, "window_days": days}
        pnl_of = lambda r: float(r.get("total_pnl") if r.get("total_pnl") is not None else (r.get("pnl") or 0))
        wins = [r for r in trades if pnl_of(r) > 0]
        losses = [r for r in trades if pnl_of(r) <= 0]
        gross_win = sum(pnl_of(r) for r in wins)
        gross_loss = -sum(pnl_of(r) for r in losses)
        net = gross_win - gross_loss
        mfe_vals = [float(r.get("mfe_pct") or 0) for r in trades]
        mae_vals = [float(r.get("mae_pct") or 0) for r in trades]
        durations = []
        for r in trades:
            try:
                d = (datetime.fromisoformat(r["exit_time"])
                     - datetime.fromisoformat(r["entry_time"]))
                durations.append(max(0.0, d.total_seconds() / 3600.0))
            except Exception:
                pass
        # capture efficiency: how much of the peak move turned into PnL
        effs = []
        for r in trades:
            mfe = float(r.get("mfe_pct") or 0)
            if mfe > 0:
                notional = float(r.get("initial_notional") or r.get("notional_usd") or 0)
                if notional > 0:
                    effs.append(min(200.0, max(-200.0, pnl_of(r) / notional * 100 / mfe * 100)))
        return {
            "window_days": days,
            "closed_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(trades) * 100,
            "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else (gross_win if gross_win > 0 else 0),
            "net_pnl": net,
            "avg_pnl": net / len(trades),
            "expectancy": net / len(trades),
            "avg_win": (gross_win / len(wins)) if wins else 0,
            "avg_loss": (gross_loss / len(losses)) if losses else 0,
            "best_pnl": max(pnl_of(r) for r in trades),
            "worst_pnl": min(pnl_of(r) for r in trades),
            "avg_mfe_pct": (sum(mfe_vals) / len(mfe_vals)) if mfe_vals else 0,
            "avg_mae_pct": (sum(mae_vals) / len(mae_vals)) if mae_vals else 0,
            "avg_capture_efficiency": (sum(effs) / len(effs)) if effs else None,
            "avg_hold_hours": (sum(durations) / len(durations)) if durations else 0,
            "partial_closes_total": sum(int(r.get("partial_count") or 0) for r in trades),
        }

    def get_equity_curve(self, limit: int = 200) -> List[Dict]:
        """v5.11: cumulative realized PnL over closed trades (dashboard sparkline)."""
        rows = self.get_positions_history(closed_only=True, limit=limit)
        rows.sort(key=lambda r: r.get("exit_time") or "")
        curve, cum = [], 0.0
        for r in rows:
            pnl = r.get("total_pnl")
            if pnl is None:
                pnl = r.get("pnl") or 0
            cum += float(pnl)
            curve.append({
                "exit_time": r.get("exit_time"),
                "symbol": r.get("symbol"),
                "pnl": float(pnl),
                "cum_pnl": cum,
            })
        return curve

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
                # PostgreSQL date arithmetic.
                # NOTE: timestamp is TEXT (ISO-8601, shared schema with SQLite) —
                # cast to timestamptz or the comparison fails with
                # "operator does not exist: text >= timestamp with time zone".
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
                        WHERE recommendations.timestamp::timestamptz >= NOW() - INTERVAL %s
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
