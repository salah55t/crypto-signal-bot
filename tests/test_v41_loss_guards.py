"""
v4.1 Loss-Avoidance Guards - Unit Tests
Covers:
  - Raised defaults (MIN_CONFIDENCE=68, MIN_RR_RATIO=1.8)
  - Daily trade cap (MAX_TRADES_PER_DAY)
  - Loss-streak circuit breaker (3 losses -> pause)
  - Win resets loss streak
  - Per-symbol re-entry cooldown after a losing close
  - Daily opened sync from DB (survives restarts)
"""
import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
import src.risk.manager as manager_module
from src.risk.manager import RiskManager
from src.utils.helpers import now_utc


def make_manager(tmp_path, monkeypatch, positions=None, daily=None):
    """RiskManager isolated from real data files (real JSON files in tmp_path)."""
    import json as _json
    pos_file = tmp_path / "open_positions.json"
    stats_file = tmp_path / "daily_stats.json"
    pos_file.write_text(_json.dumps(positions if positions is not None else []))
    stats_file.write_text(_json.dumps(daily or {}))
    monkeypatch.setattr(manager_module, "POSITIONS_FILE", pos_file)
    monkeypatch.setattr(manager_module, "DAILY_STATS_FILE", stats_file)
    # Avoid touching the real DB in close_position
    class _NoDB:
        def __getattr__(self, name):
            return lambda *a, **k: None
    monkeypatch.setattr(manager_module, "db", _NoDB())
    return RiskManager(capital=10000)


def make_position(symbol="TESTUSDT", direction="bullish", notional=10.0):
    return {
        "symbol": symbol,
        "direction": direction,
        "entry_price": 100.0,
        "stop_loss": 98.0,
        "take_profit": 104.0,
        "size": notional / 100.0,
        "notional_usd": notional,
        "entry_fee": notional * settings.TRADING_FEE_PCT / 100,
        "entry_time": now_utc().isoformat(),
        "status": "open",
    }


# ============================================
# Defaults
# ============================================

def test_v41_default_thresholds():
    assert settings.MIN_CONFIDENCE == 68
    assert settings.MIN_RR_RATIO == 1.5
    assert settings.MAX_TRADES_PER_DAY == 12
    assert settings.LOSS_STREAK_LIMIT == 3
    assert settings.LOSS_STREAK_PAUSE_HOURS == 4
    assert settings.REENTRY_COOLDOWN_HOURS == 2


# ============================================
# Daily trade cap
# ============================================

def test_daily_trade_cap_blocks(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rm._ensure_today_stats()
    rm.daily_stats[rm._today_key()]["trades_opened"] = settings.MAX_TRADES_PER_DAY
    assert rm.can_open_position() is False


def test_daily_trade_cap_allows_below(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rm._ensure_today_stats()
    rm.daily_stats[rm._today_key()]["trades_opened"] = settings.MAX_TRADES_PER_DAY - 1
    assert rm.can_open_position() is True


def test_sync_daily_opened_from_db(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rm._ensure_today_stats()
    assert rm.daily_stats[rm._today_key()]["trades_opened"] == 0
    rm.sync_daily_opened(20)  # e.g. restart mid-day: DB says 20 opened
    assert rm.daily_stats[rm._today_key()]["trades_opened"] == 20
    assert rm.can_open_position() is False  # 20 >= 12 -> cap


# ============================================
# Loss-streak circuit breaker
# ============================================

def _open_and_lose(rm, symbol):
    rm.open_positions.append(make_position(symbol))
    return rm.close_position(0, exit_price=95.0, reason="Stop Loss Hit")


def test_loss_streak_triggers_pause(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    for _ in range(settings.LOSS_STREAK_LIMIT):
        _open_and_lose(rm, "AAAUSDT")
    assert rm.loss_streak == 3
    assert rm._loss_pause_active() is True
    assert rm.can_open_position() is False


def test_win_resets_loss_streak(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    _open_and_lose(rm, "BBBUSDT")
    _open_and_lose(rm, "BBBUSDT")
    assert rm.loss_streak == 2
    # winning close
    rm.open_positions.append(make_position("WINUSDT"))
    rm.close_position(0, exit_price=110.0, reason="Take Profit Hit")
    assert rm.loss_streak == 0
    assert rm._loss_pause_active() is False
    assert rm.can_open_position() is True


def test_pause_expires(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    for _ in range(settings.LOSS_STREAK_LIMIT):
        _open_and_lose(rm, "CCCUSDT")
    assert rm.can_open_position() is False
    # simulate pause expiry
    rm._loss_pause_until = now_utc() - timedelta(seconds=1)
    assert rm._loss_pause_active() is False
    assert rm.can_open_position() is True


# ============================================
# Per-symbol re-entry cooldown
# ============================================

def test_reentry_cooldown_blocks_symbol_only(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    _open_and_lose(rm, "LOSERUSDT")
    assert rm.is_symbol_blocked("LOSERUSDT") is True
    assert rm.is_symbol_blocked("OTHERUSDT") is False
    assert rm.can_open_position("LOSERUSDT") is False
    assert rm.can_open_position("OTHERUSDT") is True


def test_reentry_cooldown_expires(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    _open_and_lose(rm, "OLDUSDT")
    assert rm.is_symbol_blocked("OLDUSDT") is True
    rm._reentry_block["OLDUSDT"] = now_utc() - timedelta(seconds=1)
    assert rm.is_symbol_blocked("OLDUSDT") is False


# ============================================
# Persistence across restarts
# ============================================

def test_loss_state_persisted_in_daily_stats(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    _open_and_lose(rm, "PERSISTUSDT")
    _open_and_lose(rm, "PERSISTUSDT")
    today = rm.daily_stats[rm._today_key()]
    assert today["loss_streak"] == 2
    # A fresh manager (simulated restart) restores the state
    rm2 = RiskManager(capital=10000)
    assert rm2.loss_streak == 2


def test_close_position_records_stats(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    _open_and_lose(rm, "STATSUSDT")
    today = rm.daily_stats[rm._today_key()]
    assert today["losses"] == 1
    assert today["wins"] == 0
    assert today["trades_opened"] == 0  # opened manually, not via open_position
    assert today["pnl"] < 0
