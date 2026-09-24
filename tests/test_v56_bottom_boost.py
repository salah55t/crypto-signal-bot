"""
v5.6 Bottom-fishing admission - Unit Tests

User report: "لماذا البوت لا يفتح توصيات من عملات القاع" (why does the bot
never open positions from bottom coins?). Three stacked blockers killed the
whole bottom-boost channel:

  1) validate_recommendation's harmony gate: boosted recs carried no harmony
     key -> 0.0 < 0.45 -> "Harmony too low" -> 100% rejection at open time,
     even after a candidate became a visible recommendation.
  2) The confidence formula (45 + score/3, cap 72) was written for
     MIN_CONFIDENCE=60; v4.1's raise to 68 turned the documented
     "score >= 60" gate into a de-facto score >= 69 gate.
  3) Boosted recs were appended LAST and cut by the [:MAX_RECOMMENDATIONS]
     slice whenever 5+ strategy signals passed the filter.

Fixes covered here:
  - build_bottom_rec: bounce-scale confidence (40 + score/2, capped at
    BOTTOM_CONF_CAP), bounce-derived harmony, admission_confidence
  - validate_recommendation: boosted recs exempt from the strategy harmony
    gate (they carry their own layered bounce gate: score + bullish close + RR)
  - End-to-end: open_paper_position opens a harmony-less boosted rec
  - /api/config exposes the public admission knobs for the dashboard notes
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.analysis.analyzer import build_bottom_rec
from src.risk.manager import RiskManager

from tests.test_v5_veteran import make_manager


def _candidate(score=65.0, n_signals=3, symbol="BOTTOMUSDT"):
    """A bottom-scanner candidate as score_bottom_candidate returns it."""
    return {
        "symbol": symbol,
        "score": float(score),
        "current_price": 1.0,
        "recent_low": 0.98,
        "recent_high": 1.30,
        "distance_from_low_pct": 2.0,
        "position_in_range_pct": 6.7,
        "rsi": 28.0,
        "atr_pct": 1.2,
        # 1.2 ATR SL / 2.5 ATR TP -> RR ~2.08 (>= MIN_RR_RATIO 1.5)
        "stop_loss": 1.0 - 1.2 * 0.0144,
        "take_profit": 1.0 + 2.5 * 0.0144,
        "risk_reward_ratio": 2.083,
        "signals": [f"layer{i}" for i in range(n_signals)],
        "bb_percent_b": 0.03,
        "last_candle_bullish": True,
        "patterns_detected": ["hammer"],
        "analyzed_at": "2026-09-25T00:00:00+00:00",
    }


# ============================================
# Settings defaults
# ============================================

def test_settings_v56_defaults():
    assert settings.BOTTOM_BOOST_ENABLED is True
    assert settings.BOTTOM_STRONG_SCORE == 62.0
    assert settings.BOTTOM_MAX_PER_CYCLE == 2
    assert settings.BOTTOM_CONF_CAP == 72.0


# ============================================
# Confidence scale fix (blocker #2)
# ============================================

def test_old_formula_silently_required_score_69():
    """Regression arithmetic: under the old mapping a score-62 candidate
    (the documented 'score >= 60' intent) mapped to 65.7 confidence and
    died against MIN_CONFIDENCE=68. The gate was never 60 - it was 69."""
    old_conf = min(72.0, 45.0 + 62 / 3)
    assert old_conf < settings.MIN_CONFIDENCE  # the old silent kill
    assert 45.0 + 69 / 3 >= settings.MIN_CONFIDENCE  # de-facto floor was 69


def test_score62_now_maps_to_passing_confidence():
    """The new bounce-scale mapping: 40 + score/2 puts a score-62 candidate
    at 71% - above MIN_CONFIDENCE with room for the cap to matter."""
    rec = build_bottom_rec(_candidate(score=62.0))
    assert rec["confidence"] == pytest.approx(71.0)
    assert rec["confidence"] >= settings.MIN_CONFIDENCE
    assert rec["admission_confidence"] == rec["confidence"]


def test_confidence_capped_at_bottom_conf_cap():
    rec = build_bottom_rec(_candidate(score=95.0))
    assert rec["confidence"] == settings.BOTTOM_CONF_CAP


def test_build_bottom_rec_shape():
    rec = build_bottom_rec(_candidate())
    assert rec["symbol"] == "BOTTOMUSDT"
    assert rec["direction"] == "bullish"
    assert rec["boosted_from_bottom"] is True
    assert rec["weighted_score"] == 65.0
    # expected rise = max(MIN_EXPECTED_RISE, 1.8 x ATR%)
    assert rec["expected_rise_pct"] == pytest.approx(
        max(settings.MIN_EXPECTED_RISE, 1.2 * 1.8))
    # composite-ranking fields present (harmony + RR)
    assert "harmony" in rec and "risk_reward_ratio" in rec
    assert rec["signals"][0]["strategy"] == "bottom_scanner_boost"


# ============================================
# Bounce-derived harmony
# ============================================

def test_harmony_scales_with_bounce_layers():
    assert build_bottom_rec(_candidate(n_signals=1))["harmony"] == 0.35
    assert build_bottom_rec(_candidate(n_signals=3))["harmony"] == 0.55
    assert build_bottom_rec(_candidate(n_signals=6))["harmony"] == 0.85  # cap


# ============================================
# Harmony-gate exemption (blocker #1 - the 100% killer)
# ============================================

def test_boosted_rec_without_harmony_passes_validation(tmp_path, monkeypatch):
    """THE incident: a boosted rec with NO harmony key must open a position.
    Before v5.6 this was rejected 100% of the time with
    'Harmony too low (0.00 < 0.45)'."""
    rm = make_manager(tmp_path, monkeypatch)
    rec = build_bottom_rec(_candidate())
    rec.pop("harmony", None)  # simulate the pre-v5.6 rec shape
    valid, reasons = rm.validate_recommendation(rec)
    assert valid, f"boosted rec rejected: {reasons}"
    assert not any("Harmony" in r for r in reasons)


def test_non_boosted_rec_without_harmony_still_rejected(tmp_path, monkeypatch):
    """The exemption must NOT leak to normal strategy recs."""
    rm = make_manager(tmp_path, monkeypatch)
    rec = build_bottom_rec(_candidate())
    rec.pop("harmony", None)
    rec.pop("boosted_from_bottom", None)
    valid, reasons = rm.validate_recommendation(rec)
    assert not valid
    assert any("Harmony too low" in r for r in reasons)


def test_boosted_rec_still_gated_by_rr(tmp_path, monkeypatch):
    """Exemption from the harmony gate only - every other gate applies."""
    rm = make_manager(tmp_path, monkeypatch)
    rec = build_bottom_rec(_candidate())
    rec["risk_reward_ratio"] = 1.0  # < MIN_RR_RATIO 1.5
    valid, reasons = rm.validate_recommendation(rec)
    assert not valid
    assert any("R/R too low" in r for r in reasons)


# ============================================
# End-to-end: boosted rec opens a position
# ============================================

def test_open_paper_position_accepts_boosted_rec(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rec = build_bottom_rec(_candidate())
    result = rm.open_position(rec)
    assert result["status"] == "opened", result.get("reasons")
    assert len(rm.open_positions) == 1
    pos = rm.open_positions[0]
    assert pos["symbol"] == "BOTTOMUSDT"
    assert pos["harmony"] == 0.55  # bounce-derived harmony persisted


def test_open_paper_position_rejects_weak_bottom_candidate(tmp_path, monkeypatch):
    """A weak candidate (RR below floor) must still not open."""
    rm = make_manager(tmp_path, monkeypatch)
    rec = build_bottom_rec(_candidate(score=63.0))
    rec["risk_reward_ratio"] = 1.1
    result = rm.open_position(rec)
    assert result["status"] == "rejected"
    assert any("R/R" in r for r in result["reasons"])


# ============================================
# /api/config - dashboard info banners
# ============================================

def test_api_config_exposes_public_knobs():
    from fastapi.testclient import TestClient
    from src.web.app import app

    client = TestClient(app)
    r = client.get("/api/config")
    assert r.status_code == 200
    cfg = r.json()
    assert cfg["bottom_strong_score"] == settings.BOTTOM_STRONG_SCORE
    assert cfg["bottom_max_per_cycle"] == settings.BOTTOM_MAX_PER_CYCLE
    assert cfg["trade_amount_usd"] == settings.TRADE_AMOUNT_USD
    assert cfg["min_confidence"] == settings.MIN_CONFIDENCE
    # no secrets in the payload
    assert not any("key" in k.lower() or "secret" in k.lower() or
                   "token" in k.lower() for k in cfg)
