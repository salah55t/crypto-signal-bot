"""
Unified FastAPI application for Render deployment.
Serves:
  - Static web dashboard (HTML/CSS/JS)
  - REST API endpoints for recommendations & positions
  - WebSocket for live updates (auto-broadcasts when data file changes)
  - In-process APScheduler that runs the bot every hour
  - Health check endpoint for Render

Usage on Render:
  uvicorn src.web.app:app --host 0.0.0.0 --port $PORT --workers 1

Locally:
  uvicorn src.web.app:app --reload --port 8080
"""
import os
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure project root on sys.path
import sys
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings, PROJECT_ROOT
from src.utils.logger import log
from src.utils.helpers import load_json, save_json, to_json_safe, now_utc

# ============================================================
# Paths
# ============================================================
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RECOMMENDATIONS_FILE = DATA_DIR / "recommendations.json"
POSITIONS_FILE = DATA_DIR / "open_positions.json"
STATS_FILE = DATA_DIR / "daily_stats.json"
DASHBOARD_DIR = PROJECT_ROOT / "web" / "dashboard" / "public"

# Ensure data files exist
for f in [RECOMMENDATIONS_FILE, POSITIONS_FILE, STATS_FILE]:
    if not f.exists():
        if f.name.endswith("positions") or f.name == "open_positions.json":
            save_json([], f)
        elif f.name == "daily_stats.json":
            save_json({}, f)
        else:
            save_json({"timestamp": None, "top_recommendations": []}, f)

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="Crypto Signal Bot",
    description="Cryptocurrency Spot Trading Signal Bot — Multi-Strategy Analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static dashboard
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")


# ============================================================
# WebSocket Connection Manager
# ============================================================
class ConnectionManager:
    """Manages active WebSocket connections and broadcasts updates."""

    def __init__(self):
        self.active: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)
        log.info(f"[WS] Client connected. Total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.discard(ws)
        log.info(f"[WS] Client disconnected. Total: {len(self.active)}")

    async def broadcast(self, message: Dict):
        """Send a message to all connected clients."""
        if not self.active:
            return
        text = json.dumps(message, default=str)
        # Send to all, drop failed connections
        dead = []
        for ws in list(self.active):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ============================================================
# REST API Endpoints
# ============================================================

@app.get("/")
async def root():
    """Serve the dashboard HTML."""
    index = DASHBOARD_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "Dashboard not found. Visit /docs for API."}, status_code=404)


@app.get("/api/health")
async def health():
    # v5.2: surface rate-limit state so "bot silent" is explainable
    # (shared-IP cooldown vs real outage) straight from the dashboard.
    # v5.3: + WebSocket feed status + pressure streak.
    try:
        from src.core.rate_limiter import rate_limiter
        rate_state = {
            "in_cooldown": rate_limiter.in_cooldown(),
            "cooldown_seconds_left": round(rate_limiter.cooldown_remaining(), 1),
            "used_weight_1m": int(rate_limiter.used_weight()),
            "budget_per_min": int(rate_limiter.budget),
            "pressure_streak": rate_limiter.pressure_streak(),
        }
    except Exception:
        rate_state = {"error": "unavailable"}
    try:
        from src.core.ws_feed import ws_feed
        ws_state = ws_feed.status()
    except Exception:
        ws_state = {"error": "unavailable"}
    return {
        "status": "ok",
        "service": "crypto-signal-bot",
        "version": "1.0.0",
        "timestamp": now_utc().isoformat(),
        "mode": settings.RUN_MODE,
        "rate_limit": rate_state,
        "ws_feed": ws_state,
    }


@app.get("/api/recommendations")
async def get_recommendations():
    """Get latest recommendations."""
    data = load_json(RECOMMENDATIONS_FILE, default={
        "timestamp": None, "top_recommendations": []
    })
    return data


@app.get("/api/positions")
async def get_positions():
    """Get open positions with real-time P&L + v5 tracking fields."""
    from src.risk.manager import risk_manager
    from src.core.data_fetcher import data_fetcher
    positions = risk_manager.open_positions
    if not positions:
        return []
    # Fetch current prices for all position symbols.
    # NOTE: get_batch_tickers hits /ticker/price whose rows have key "price"
    # (NOT "lastPrice" from /ticker/24hr) — reading lastPrice here silently
    # zeroed every price and the dashboard showed no P&L. get_batch_prices
    # handles both key shapes.
    symbols = list({p["symbol"] for p in positions})
    try:
        current_prices = data_fetcher.get_batch_prices(symbols)
    except Exception as e:
        log.error(f"Failed to fetch prices for positions: {e}")
        current_prices = {}
    return risk_manager.get_positions_with_pnl(current_prices)


@app.get("/api/pending")
async def get_pending_entries():
    """v5: armed pending LIMIT entries waiting for their entry zone."""
    from src.risk.manager import risk_manager
    return [
        {k: v for k, v in p.items() if k != "rec"}
        for p in risk_manager.pending_entries
    ]


@app.get("/api/market-map")
async def get_market_map(refresh: bool = False):
    """v5.2/v5.4: leader/follower groups + leader trends + market cycle.

    Serves the cached map (recomputed at most every MARKET_MAP_REFRESH_HOURS)
    plus the leader-coin market-cycle verdict (cached up to
    MARKET_CYCLE_REFRESH_MIN). Pass ?refresh=true to force a rebuild
    (blocking, ~1 API call per symbol).
    """
    from src.analysis.market_map import market_map
    try:
        market_map.get_map(force=refresh)  # rebuild now if refresh=True
        view = market_map.grouped_view()
        try:
            view["cycle"] = market_map.run_market_cycle(force=refresh)
        except Exception as ce:
            log.warning(f"Market cycle unavailable: {ce}")
            view["cycle"] = None
        return view
    except Exception as e:
        log.error(f"Market map error: {e}")
        return {"error": str(e), "groups": {}, "leaders": {}, "cycle": None}


@app.get("/api/market-groups-file", response_class=PlainTextResponse)
async def get_market_groups_file():
    """v5.4: the human-readable Arabic classification/decision file.

    Serves data/market_groups.txt (built on demand the first time).
    """
    from src.analysis.market_map import market_map
    path = Path(settings.MARKET_GROUPS_FILE)
    try:
        if not path.exists():
            market_map.get_map()
            market_map.run_market_cycle()
        if path.exists():
            return PlainTextResponse(
                path.read_text(encoding="utf-8"), media_type="text/plain; charset=utf-8"
            )
    except Exception as e:
        log.error(f"Groups file error: {e}")
    return PlainTextResponse(
        "الملف غير متاح بعد — يُبنى تلقائياً مع أول دورة تحليل للسوق.",
        media_type="text/plain; charset=utf-8",
    )


@app.get("/api/all-analyses")
async def get_all_analyses():
    """
    Return ALL analyzed symbols (not just recommendations) with their scores,
    signals, and rejection reasons.
    Useful for debugging and understanding why certain coins were rejected.
    """
    data = load_json(RECOMMENDATIONS_FILE, default={
        "all_results": [], "top_recommendations": []
    })
    all_results = data.get("all_results", [])
    # Add pass/reject status for each
    top_symbols = {r["symbol"] for r in data.get("top_recommendations", [])}
    enriched = []
    for r in all_results:
        symbol = r.get("symbol", "")
        is_recommended = symbol in top_symbols
        # Build rejection reasons list
        rejection_reasons = []
        if not is_recommended and not r.get("skip"):
            if r.get("confidence", 0) < settings.MIN_CONFIDENCE:
                rejection_reasons.append(
                    f"Confidence {r.get('confidence', 0):.1f}% < {settings.MIN_CONFIDENCE}%"
                )
            if r.get("expected_rise_pct", 0) < settings.MIN_EXPECTED_RISE:
                rejection_reasons.append(
                    f"Expected rise {r.get('expected_rise_pct', 0):.2f}% < {settings.MIN_EXPECTED_RISE}%"
                )
            if r.get("risk_reward_ratio", 0) < settings.MIN_RR_RATIO:
                rejection_reasons.append(
                    f"R/R {r.get('risk_reward_ratio', 0):.2f} < {settings.MIN_RR_RATIO}"
                )
            if r.get("direction") != "bullish":
                rejection_reasons.append(
                    f"Direction {r.get('direction', 'unknown')} (need bullish)"
                )
            if not rejection_reasons:
                rejection_reasons.append("Capped by MAX_RECOMMENDATIONS limit")

        # Extract per-strategy reasons
        strategy_breakdown = []
        for sig in r.get("signals", []):
            strategy_breakdown.append({
                "strategy": sig.get("strategy", ""),
                "direction": sig.get("direction", ""),
                "score": sig.get("score", 0),
                "confidence": sig.get("confidence", 0),
                "reasons": sig.get("reasons", []),
            })

        enriched.append({
            "symbol": symbol,
            "direction": r.get("direction", "neutral"),
            "confidence": r.get("confidence", 0),
            "weighted_score": r.get("weighted_score", 0),
            "current_price": r.get("current_price", 0),
            "expected_rise_pct": r.get("expected_rise_pct", 0),
            "stop_loss": r.get("stop_loss", 0),
            "take_profit": r.get("take_profit", 0),
            "risk_reward_ratio": r.get("risk_reward_ratio", 0),
            "atr_pct": r.get("atr_pct", 0),
            "is_recommended": is_recommended,
            "rejection_reasons": rejection_reasons,
            "strategy_breakdown": strategy_breakdown,
            "analyzed_at": r.get("analyzed_at"),
            "timeframe": r.get("timeframe"),
        })

    # Sort: recommended first, then by confidence desc
    enriched.sort(key=lambda x: (not x["is_recommended"], -x["confidence"]))

    return {
        "timestamp": data.get("timestamp"),
        "symbols_analyzed": data.get("symbols_analyzed", len(enriched)),
        "symbols_with_data": data.get("symbols_with_data", len(enriched)),
        "signals_passed": data.get("signals_passed", 0),
        "analysis_time_seconds": data.get("analysis_time_seconds", 0),
        "recommended_count": sum(1 for e in enriched if e["is_recommended"]),
        "rejected_count": sum(1 for e in enriched if not e["is_recommended"]),
        "analyses": enriched,
    }


@app.get("/api/closed-trades")
async def get_closed_trades():
    """Return closed positions history with P&L."""
    # Closed trades are stored in daily_stats or a separate file
    closed_file = DATA_DIR / "closed_trades.json"
    return load_json(closed_file, default=[])


@app.get("/api/bottom-candidates")
async def get_bottom_candidates():
    """
    Return coins near their recent lows with bounce signals.
    These are the top 20 bottom-reversal candidates.
    """
    bottom_file = DATA_DIR / "bottom_candidates.json"
    return load_json(bottom_file, default={
        "timestamp": None, "top_candidates": []
    })


@app.get("/api/config")
async def get_public_config():
    """
    v5.6: public admission knobs for the dashboard info banners
    (توصية vs صفقة مفتوحة + bottom-boost gates). Read-only, no secrets.
    """
    return {
        "min_confidence": settings.MIN_CONFIDENCE,
        "min_rr_ratio": settings.MIN_RR_RATIO,
        "min_harmony": settings.MIN_HARMONY,
        "max_open_positions": settings.MAX_OPEN_POSITIONS,
        "max_recommendations": settings.MAX_RECOMMENDATIONS,
        "trade_amount_usd": settings.TRADE_AMOUNT_USD,
        "bottom_boost_enabled": settings.BOTTOM_BOOST_ENABLED,
        "bottom_strong_score": settings.BOTTOM_STRONG_SCORE,
        "bottom_max_per_cycle": settings.BOTTOM_MAX_PER_CYCLE,
        "bottom_conf_cap": settings.BOTTOM_CONF_CAP,
        "timeframes": settings.TIMEFRAMES,
        "run_mode": settings.RUN_MODE,
        # v5.9: AI advisor status (NO key material - enabled/model only)
        "ai_advisor_enabled": bool(
            settings.AI_ADVISOR_ENABLED and settings.CODECRAFT_API_KEY
        ),
        "ai_advisor_model": settings.AI_ADVISOR_MODEL,
    }


@app.post("/api/scan-bottoms")
async def scan_bottoms_now():
    """Trigger an immediate bottom scan in background."""
    import threading
    log.info("[cyan]Manual bottom scan trigger received[/]")

    def run_scan():
        try:
            from src.analysis.bottom_scanner import bottom_scanner
            bottom_scanner.scan(max_candidates=20, limit=200)
        except Exception as e:
            log.error(f"Bottom scan error: {e}")

    thread = threading.Thread(target=run_scan, daemon=True)
    thread.start()
    return {
        "status": "started",
        "message": "Bottom scan started in background. Check /api/bottom-candidates in 30-60s.",
        "timestamp": now_utc().isoformat(),
    }


@app.get("/api/stats")
async def get_stats():
    return load_json(STATS_FILE, default={})


@app.post("/api/run-analysis")
async def run_analysis_now():
    """
    Trigger an immediate bot analysis cycle.
    Returns immediately with a job ID; the analysis runs in background.
    Frontend polls /api/recommendations to see results.
    """
    import threading
    log.info("[cyan]Manual analysis trigger received[/]")
    # Run in background to not block the request
    thread = threading.Thread(target=run_bot_cycle_sync, daemon=True)
    thread.start()
    return {
        "status": "started",
        "message": "Analysis cycle started in background. Check /api/recommendations in 30-60s.",
        "timestamp": now_utc().isoformat(),
    }


# ============================================
# DATABASE / STATS ENDPOINTS
# ============================================

@app.get("/api/stats/summary")
async def get_stats_summary():
    """Get aggregated summary statistics from the database."""
    try:
        from src.db.database import db
        return db.get_summary_stats()
    except Exception as e:
        log.error(f"Stats summary error: {e}")
        return {"error": str(e)}


@app.get("/api/stats/strategies")
async def get_strategy_stats(days: int = 30):
    """Get per-strategy performance metrics."""
    try:
        from src.db.database import db
        return {
            "days": days,
            "strategies": db.get_strategy_performance(days=days)
        }
    except Exception as e:
        log.error(f"Strategy stats error: {e}")
        return {"error": str(e), "strategies": []}


@app.get("/api/stats/runs")
async def get_runs_history(limit: int = 50):
    """Get history of bot runs (analysis cycles)."""
    try:
        from src.db.database import db
        return {"runs": db.get_runs_history(limit=limit)}
    except Exception as e:
        log.error(f"Runs history error: {e}")
        return {"error": str(e), "runs": []}


@app.get("/api/stats/recommendations")
async def get_recommendations_history(limit: int = 100, symbol: str = None):
    """Get history of all recommendations from the database."""
    try:
        from src.db.database import db
        recs = db.get_recommendations_history(limit=limit, symbol=symbol)
        return {"count": len(recs), "recommendations": recs}
    except Exception as e:
        log.error(f"Recommendations history error: {e}")
        return {"error": str(e), "count": 0, "recommendations": []}


@app.get("/api/stats/positions")
async def get_positions_history(closed_only: bool = False, limit: int = 50):
    """Get positions history from the database."""
    try:
        from src.db.database import db
        positions = db.get_positions_history(closed_only=closed_only, limit=limit)
        return {"count": len(positions), "positions": positions}
    except Exception as e:
        log.error(f"Positions history error: {e}")
        return {"error": str(e), "count": 0, "positions": []}


@app.get("/api/stats/daily")
async def get_daily_stats_history(limit: int = 30):
    """Get daily stats history from the database."""
    try:
        from src.db.database import db
        return {"daily": db.get_daily_stats(limit=limit)}
    except Exception as e:
        log.error(f"Daily stats error: {e}")
        return {"error": str(e), "daily": []}


@app.get("/api/stats/bottoms")
async def get_bottoms_history(limit: int = 50, min_score: float = 50):
    """Get bottom candidates history from the database."""
    try:
        from src.db.database import db
        candidates = db.get_bottom_candidates_history(limit=limit, min_score=min_score)
        return {"count": len(candidates), "candidates": candidates}
    except Exception as e:
        log.error(f"Bottoms history error: {e}")
        return {"error": str(e), "count": 0, "candidates": []}


@app.post("/api/reset-history")
async def reset_history():
    """Reset ALL bot history (database tables + JSON files). Use with caution."""
    try:
        from src.db.database import db
        from src.risk.manager import risk_manager
        # Clear DB
        db_result = db.reset_all_history()
        # Clear JSON files
        for filename in ["open_positions.json", "closed_trades.json",
                         "daily_stats.json", "recommendations.json",
                         "bottom_candidates.json"]:
            file_path = DATA_DIR / filename
            if file_path.exists():
                file_path.unlink()
        # Reset in-memory state
        risk_manager.open_positions = []
        from src.utils.helpers import save_json
        save_json([], POSITIONS_FILE)
        save_json({}, STATS_FILE)
        save_json({"timestamp": None, "top_recommendations": []},
                  RECOMMENDATIONS_FILE)
        log.warning("[red]ALL HISTORY RESET[/] - database + JSON files + in-memory")
        return {
            "status": "success",
            "message": "All history has been reset (DB + JSON + in-memory).",
            "database": db_result,
            "files_cleared": ["open_positions.json", "closed_trades.json",
                              "daily_stats.json", "recommendations.json",
                              "bottom_candidates.json"],
        }
    except Exception as e:
        log.error(f"Reset history error: {e}")
        return {"status": "error", "error": str(e)}


# ============================================================
# WebSocket Endpoint
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            # Keep connection alive, listen for ping
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
                if data.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


# ============================================================
# Background Tasks (file watcher + scheduler)
# ============================================================

async def watch_recommendations_file():
    """Watch recommendations.json for changes and broadcast to clients."""
    last_mtime = 0
    while True:
        try:
            if RECOMMENDATIONS_FILE.exists():
                mtime = RECOMMENDATIONS_FILE.stat().st_mtime
                if mtime != last_mtime:
                    last_mtime = mtime
                    data = load_json(RECOMMENDATIONS_FILE, default={})
                    await manager.broadcast({
                        "type": "recommendations",
                        "timestamp": now_utc().isoformat(),
                        "data": to_json_safe(data),
                    })
        except Exception as e:
            log.debug(f"File watch error: {e}")
        await asyncio.sleep(5)


def run_bot_cycle_sync():
    """Run one bot analysis cycle synchronously (called from scheduler).

    v5: delegates to the unified cycle module so the web dashboard and the
    standalone runner behave IDENTICALLY (the old duplicated copy here was
    missing v4.1 gates and v5 veteran management).
    """
    try:
        from src.core.cycle import run_analysis_cycle
        run_analysis_cycle()
    except Exception as e:
        log.exception(f"Bot cycle error: {e}")


def monitor_positions_sync():
    """
    Fast position monitor - runs every 1 minute.
    v5: delegates to the unified watcher: SL/TP/partial + chandelier trailing
    + MFE/MAE tracking + time stop + pending limit-entry fills. Price-only.
    """
    try:
        from src.core.cycle import run_position_watch
        run_position_watch()
    except Exception as e:
        log.debug(f"Position monitor error: {e}")


@app.on_event("startup")
async def startup_event():
    """Start background tasks on app startup."""
    # Initialize database
    try:
        from src.db.database import db
        db.init()
        log.info("[green]Database initialized successfully[/]")
    except Exception as e:
        log.error(f"Database initialization failed: {e}")

    # v5.3: WebSocket kline feed (zero REST weight for 1h candles)
    try:
        from src.core.ws_feed import ws_feed
        from src.analysis.analyzer import analyzer
        ws_feed.update_universe(analyzer.symbols)
        ws_feed.start()
    except Exception as e:
        log.warning(f"[yellow]WS feed startup skipped:[/] {e}")

    # File watcher (always on)
    asyncio.create_task(watch_recommendations_file())

    # In-process scheduler (APScheduler)
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger
        scheduler = AsyncIOScheduler(timezone="UTC")
        trigger = CronTrigger.from_crontab(settings.SCHEDULE_CRON)
        scheduler.add_job(
            run_bot_cycle_sync,
            trigger=trigger,
            id="analyze",
            name="market_analysis",
            misfire_grace_time=300,
        )
        # NEW: Fast position monitor - runs every 1 minute
        # Applies SAME logic (SL/TP check) to paper AND live positions
        scheduler.add_job(
            monitor_positions_sync,
            trigger=IntervalTrigger(minutes=1),
            id="position_monitor",
            name="position_monitor",
            misfire_grace_time=60,
        )
        scheduler.start()
        log.info(f"[green]Scheduler started[/] - analysis: '{settings.SCHEDULE_CRON}', monitor: every 1 min")

        # Optionally run an initial cycle on startup
        if os.getenv("RUN_ON_STARTUP", "false").lower() == "true":
            log.info("[cyan]Running initial analysis on startup...[/]")
            # Run in background thread to not block startup
            import threading
            threading.Thread(target=run_bot_cycle_sync, daemon=True).start()
    except Exception as e:
        log.error(f"Failed to start scheduler: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    log.info("[yellow]Shutting down app[/]")


# ============================================================
# Main (for direct execution)
# ============================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "src.web.app:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info",
    )
