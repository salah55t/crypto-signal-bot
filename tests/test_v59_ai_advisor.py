"""
v5.9 AI Advisor tests - CodeCraft API (codecraftapi.com) integration.

Covers:
  - sanitize_comment: Telegram-Markdown safety + length cap
  - LLMAdvisor gating: no key -> disabled, explain/enrich are safe no-ops
  - _chat: mocked success / HTTP error / network exception
  - explain_recommendation: content -> sanitized comment, failure -> None
  - cache: hit avoids a second HTTP call, key buckets confidence by 5
  - enrich_recommendations: adds ai_comment + ai_model, parallel, bounded
  - Telegram card: AI section present/absent
  - /api/config: exposes ai_advisor_enabled/model, never the key
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai.llm_advisor import LLMAdvisor, sanitize_comment
import src.ai.llm_advisor as llm_mod
from config.settings import settings


# ============================================
# helpers
# ============================================

@pytest.fixture
def advisor(tmp_path, monkeypatch):
    """Fresh LLMAdvisor with isolated cache file + fake key."""
    monkeypatch.setattr(llm_mod, "AI_CACHE_FILE", tmp_path / "ai_cache.json")
    monkeypatch.setattr(settings, "AI_ADVISOR_ENABLED", True)
    monkeypatch.setattr(settings, "CODECRAFT_API_KEY", "cc_test")
    monkeypatch.setattr(settings, "AI_ADVISOR_MODEL", "test-model")
    monkeypatch.setattr(settings, "AI_ADVISOR_TIMEOUT", 5)
    monkeypatch.setattr(settings, "AI_ADVISOR_CACHE_HOURS", 4)
    return LLMAdvisor()


@pytest.fixture
def disabled_advisor(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod, "AI_CACHE_FILE", tmp_path / "ai_cache.json")
    monkeypatch.setattr(settings, "CODECRAFT_API_KEY", "")
    return LLMAdvisor()


def make_rec(**over):
    rec = {
        "symbol": "BTCUSDT",
        "direction": "bullish",
        "confidence": 76.0,
        "current_price": 65000.0,
        "stop_loss": 63500.0,
        "take_profit": 68000.0,
        "risk_reward_ratio": 1.8,
        "atr_pct": 1.2,
        "signals": [
            {"strategy": "TripleConfluenceTrend",
             "reasons": ["EMA50 above EMA200 (regime aligned (bullish))"]}
        ],
    }
    rec.update(over)
    return rec


class FakeResp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def chat_payload(content):
    return {"choices": [{"message": {"content": content}}],
            "model": "test-model", "usage": {"total_tokens": 42}}


# ============================================
# sanitize_comment
# ============================================

def test_sanitize_strips_markdown_chars():
    raw = "**شراء** جيد `حسب RSI` [x](y) #عام +1 =2 |a| {b}"
    out = sanitize_comment(raw)
    for ch in "*_`[]()#+=|{}":
        assert ch not in out
    assert "شراء" in out and "RSI" in out


def test_sanitize_collapses_whitespace():
    assert sanitize_comment("سطر  \n\n أول\tثان") == "سطر أول ثان"


def test_sanitize_caps_length():
    out = sanitize_comment("كلمة " * 500, max_len=100)
    assert len(out) <= 101  # 100 chars + ellipsis char


def test_sanitize_empty_safe():
    assert sanitize_comment("") == ""
    assert sanitize_comment(None) == ""


# ============================================
# gating
# ============================================

def test_disabled_without_key(disabled_advisor):
    assert disabled_advisor.enabled is False
    assert disabled_advisor.explain_recommendation(make_rec()) is None


def test_enrich_noop_when_disabled(disabled_advisor):
    recs = [make_rec()]
    assert disabled_advisor.enrich_recommendations(recs) == 0
    assert "ai_comment" not in recs[0]


def test_disabled_via_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod, "AI_CACHE_FILE", tmp_path / "ai_cache.json")
    monkeypatch.setattr(settings, "CODECRAFT_API_KEY", "cc_x")
    monkeypatch.setattr(settings, "AI_ADVISOR_ENABLED", False)
    adv = LLMAdvisor()
    assert adv.enabled is False


# ============================================
# _chat
# ============================================

def test_chat_success(advisor, monkeypatch):
    calls = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["url"] = url
        calls["auth"] = headers.get("Authorization", "")
        return FakeResp(payload=chat_payload("جواب عربي"))

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    out = advisor._chat("sys", "user")
    assert out == "جواب عربي"
    assert advisor.base_url in calls["url"]
    assert calls["url"].endswith("/v1/chat/completions")
    assert calls["auth"] == "Bearer cc_test"


def test_chat_http_error_returns_none(advisor, monkeypatch):
    monkeypatch.setattr(
        llm_mod.requests, "post",
        lambda *a, **k: FakeResp(status_code=401, text="unauthorized"))
    assert advisor._chat("s", "u") is None


def test_chat_network_exception_returns_none(advisor, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("timeout")
    monkeypatch.setattr(llm_mod.requests, "post", boom)
    assert advisor._chat("s", "u") is None


def test_chat_empty_choices(advisor, monkeypatch):
    monkeypatch.setattr(
        llm_mod.requests, "post",
        lambda *a, **k: FakeResp(payload={"choices": []}))
    assert advisor._chat("s", "u") is None


# ============================================
# explain + cache
# ============================================

def test_explain_success_and_cache(advisor, monkeypatch):
    n_calls = [0]

    def fake_post(*a, **k):
        n_calls[0] += 1
        return FakeResp(payload=chat_payload("الزخم صاعد والحجم يؤكد."))

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    rec = make_rec()
    c1 = advisor.explain_recommendation(rec)
    assert c1 == "الزخم صاعد والحجم يؤكد."
    # second call same setup -> served from cache, no new HTTP
    c2 = advisor.explain_recommendation(make_rec())
    assert c2 == c1
    assert n_calls[0] == 1


def test_explain_failure_returns_none(advisor, monkeypatch):
    monkeypatch.setattr(
        llm_mod.requests, "post",
        lambda *a, **k: FakeResp(status_code=500, text="boom"))
    assert advisor.explain_recommendation(make_rec()) is None


def test_cache_key_buckets_confidence(advisor):
    k1 = advisor._cache_key(make_rec(confidence=76.0))
    k2 = advisor._cache_key(make_rec(confidence=78.9))
    k3 = advisor._cache_key(make_rec(confidence=81.0))
    assert k1 == k2
    assert k1 != k3


def test_cache_ttl_expiry(advisor, monkeypatch):
    from datetime import datetime, timezone, timedelta
    monkeypatch.setattr(
        llm_mod.requests, "post",
        lambda *a, **k: FakeResp(payload=chat_payload("ج")))
    rec = make_rec()
    key = advisor._cache_key(rec)
    advisor._cache_put(key, "قديم")
    # force the timestamp to be older than TTL
    old = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    advisor._cache[key]["ts"] = old
    assert advisor._cache_get(key) is None  # expired


def test_prompt_contains_key_facts(advisor):
    p = advisor._user_prompt(make_rec(harmony=0.75, boosted_from_bottom=True))
    assert "BTCUSDT" in p
    assert "شرائي" in p
    assert "76" in p
    assert "0.75" in p or "75" in p
    assert "قاع" in p  # bottom-boost mention
    sys_p = advisor._system_prompt()
    assert "عربية" in sys_p and "Markdown" not in sys_p


# ============================================
# enrich_recommendations
# ============================================

def test_enrich_adds_comment_and_model(advisor, monkeypatch):
    monkeypatch.setattr(
        llm_mod.requests, "post",
        lambda *a, **k: FakeResp(payload=chat_payload("شرح عربي مختصر.")))
    recs = [make_rec(), make_rec(symbol="ETHUSDT")]
    n = advisor.enrich_recommendations(recs)
    assert n == 2
    for r in recs:
        assert r["ai_comment"] == "شرح عربي مختصر."
        assert r["ai_model"] == "test-model"


def test_enrich_respects_max_recs(advisor, monkeypatch):
    monkeypatch.setattr(settings, "AI_ADVISOR_MAX_RECS", 1)
    monkeypatch.setattr(
        llm_mod.requests, "post",
        lambda *a, **k: FakeResp(payload=chat_payload("شرح.")))
    recs = [make_rec(symbol="A"), make_rec(symbol="B")]
    advisor.enrich_recommendations(recs)
    assert "ai_comment" in recs[0]
    assert "ai_comment" not in recs[1]


def test_enrich_survives_total_failure(advisor, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("down")
    monkeypatch.setattr(llm_mod.requests, "post", boom)
    recs = [make_rec()]
    n = advisor.enrich_recommendations(recs)  # must not raise
    assert n == 0
    assert "ai_comment" not in recs[0]


# ============================================
# Telegram card
# ============================================

def test_telegram_card_includes_ai_section(monkeypatch):
    from src.notifications.telegram_bot import TelegramNotifier
    n = TelegramNotifier()
    rec = make_rec()
    rec["ai_comment"] = "زخم صاعد مع حجم مؤكد."
    card = n._format_recommendation(rec)
    assert "تحليل ذكي" in card
    assert "زخم صاعد مع حجم مؤكد." in card


def test_telegram_card_without_ai_section():
    from src.notifications.telegram_bot import TelegramNotifier
    n = TelegramNotifier()
    card = n._format_recommendation(make_rec())
    assert "تحليل ذكي" not in card


# ============================================
# /api/config
# ============================================

def test_api_config_exposes_ai_status_without_key():
    from fastapi.testclient import TestClient
    from src.web.app import app
    client = TestClient(app)
    r = client.get("/api/config")
    assert r.status_code == 200
    data = r.json()
    assert "ai_advisor_enabled" in data
    assert "ai_advisor_model" in data
    # the secret must never appear
    import json as _json
    raw = _json.dumps(data)
    assert "cc_" not in raw
    assert "CODECRAFT_API_KEY" not in raw
