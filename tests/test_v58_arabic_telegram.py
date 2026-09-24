"""
v5.8 - Arabic Telegram tests.

Covers:
  - i18n.tr(): exact map, regex rules, composites, pattern tokens,
    pass-through for unknown strings, numbers preserved
  - TelegramNotifier frames (header/footer/recommendation) are Arabic
  - cycle.py alert helpers (_notify_closed / _notify_updates) send Arabic
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.i18n import (  # noqa: E402
    tr, ar_direction, ar_mode, ar_entry_type, ar_ichimoku,
)
from src.notifications import telegram_bot as tg_module  # noqa: E402
from src.notifications.telegram_bot import TelegramNotifier  # noqa: E402
from src.core import cycle  # noqa: E402
from src.notifications import telegram_notifier  # noqa: E402


# ------------------------------------------------------------------
# i18n.tr() - strategy reasons
# ------------------------------------------------------------------
@pytest.mark.parametrize("src,must_contain,numbers", [
    # triple confluence trend
    ("Price above EMA200 (42110.1234)", "السعر فوق EMA200", ["42110.1234"]),
    ("FRESH golden cross (EMA50 x EMA200, 3 bars ago)", "تقاطع ذهبي طازج", ["3"]),
    ("RSI crossed above 50 (48.0 -> 52.3)", "RSI اخترق 50 صعوداً", ["48.0", "52.3"]),
    ("Strong volume (1.80x avg20)", "حجم قوي", ["1.80"]),
    ("OBV rising (accumulation)", "OBV صاعد", None),
    # volatility breakout
    ("Active BB squeeze (ratio 0.62)", "انضغاط بولينجر نشط", ["0.62"]),
    ("Volume spike on breakout (2.10x avg)", "قفزة حجم عند الاختراق", ["2.10"]),
    ("BUCKSHOT: close 1.2340 > upper BB 1.2200", "اختراق صاعد", ["1.2340", "1.2200"]),
    ("Strong + rising ADX (31.5, was 27.0)", "ADX قوي ومتزايد", ["31.5", "27.0"]),
    # bb mean reversion
    ("Lower BB touched (percent B = 0.03)", "لمس الباند السفلي", ["0.03"]),
    ("Deeply oversold (Stoch 8.1, RSI 22.4)", "تشبع بيعي عميق", ["8.1", "22.4"]),
    ("Fresh Stoch cross up in oversold zone (12.5/8.3 -> 18.2/14.0)",
     "تقاطع ستوكاستك صاعد طازج", ["18.2"]),
    ("Capitulation volume (1.90x avg20)", "حجم استسلام", ["1.90"]),
    # macd breakout
    ("FRESH MACD cross above signal line", "تقاطع MACD صاعد طازج", None),
    ("Zero-line cross (below zero - earliest entry)", "خط الصفر", None),
    ("Histogram flipped positive", "انقلب موجباً", None),
    # trend pullback
    ("Pullback to EMA 21 (1.2050) + recovered", "ارتداد إلى EMA21", ["1.2050"]),
    ("Bullish reversal candle: Hammer", "شمعة انعكاس صاعدة", ["المطرقة"]),
    ("MACD histogram rising", "هيستوجرام MACD متزايد", None),
    # liquidity sweep
    ("Reversal candle: Morning Star", "شمعة انعكاس", ["نجمة الصباح"]),
    ("RSI deeply oversold (21.3)", "تشبع بيعي عميق", ["21.3"]),
    ("Bullish Liquidity Sweep: swept low 1.1000, recovered to 1.1150 (+1.36%)",
     "كنس سيولة صاعد", ["1.1000", "1.1150"]),
    ("Wyckoff Spring: false breakdown below 1.0850", "زنبرك وايكوف", ["1.0850"]),
])
def test_tr_strategy_reasons(src, must_contain, numbers):
    out = tr(src)
    assert must_contain in out, f"{src} -> {out}"
    assert out != src, "should have changed"
    for n in (numbers or []):
        assert n in out, f"number {n} lost: {src} -> {out}"


# ------------------------------------------------------------------
# i18n.tr() - risk manager reasons
# ------------------------------------------------------------------
@pytest.mark.parametrize("src,must_contain", [
    ("Stop Loss Hit", "ضرب وقف الخسارة"),
    ("Stop Loss Hit (BE)", "بعد التعادل"),
    ("Take Profit Hit", "تحقق هدف جني الأرباح"),
    ("Take Profit 2 Hit", "TP2"),
    ("Trailing stop (profit +3.21%)", "وقف متتبع"),
    ("Lock +2% profit (current +2.40%)", "تثبيت ربح +2%"),
    ("Break-even (profit +1.10%)", "نقطة التعادل"),
    ("Chandelier trail 2.5xATR (peak 0.4321, +2.80%)", "وقف ثريا"),
    ("Ichimoku regime flipped bearish", "إيشيموكي"),
    ("Kijun defence (0.9950)", "كيجون"),
    ("Tenkan cross-down defence (1.0100)", "تينكان"),
    ("Opposite bearish signal (conf 62% >= 55%) - closing now",
     "إشارة هابطة معاكسة"),
    ("Time stop: stale trade (27.3h, -1.20% < 0.50%)", "وقف زمني"),
    ("Structural exit: Ichimoku regime flipped bearish", "خروج هيكلي"),
])
def test_tr_risk_reasons(src, must_contain):
    assert must_contain in tr(src)


def test_tr_composite_trailing_reason():
    out = tr("Trailing stop (profit +3.21%) + Extended TP (bullish continuation)"
             " + Locked 1.40% profit")
    assert "وقف متتبع" in out
    assert "تمديد الهدف (استمرار صاعد)" in out
    assert "تثبيت 1.40% من الربح" in out


def test_tr_direction_inside_capture():
    out = tr("Ichimoku regime aligned (bullish) x1.15")
    assert "(صاعد)" in out and "×1.15" in out


def test_tr_unknown_passthrough():
    assert tr("XYZZY totally unknown") == "XYZZY totally unknown"
    assert tr("") == ""
    assert tr(None) is None


# ------------------------------------------------------------------
# i18n helpers
# ------------------------------------------------------------------
def test_helpers():
    assert ar_direction("bullish") == "صاعد"
    assert ar_direction("bearish") == "هابط"
    assert ar_direction("neutral") == "محايد"
    assert "تجريبي" in ar_mode("PAPER")
    assert "حقيقي" in ar_mode("LIVE")
    assert ar_entry_type("limit") == "أمر محدد"
    assert ar_entry_type("market") == "سوق"
    assert ar_ichimoku("cloud_color", "green") == "خضراء"
    assert ar_ichimoku("price_vs_cloud", "above") == "فوق السحابة"
    assert ar_ichimoku("regime", "weird") == "weird"


# ------------------------------------------------------------------
# Telegram frames
# ------------------------------------------------------------------
def _rec():
    return {
        "symbol": "BTCUSDT", "direction": "bullish",
        "current_price": 43000.0, "confidence": 78.5,
        "expected_rise_pct": 4.2, "stop_loss": 41500.0,
        "take_profit": 45500.0, "take_profit_2": 47800.0,
        "risk_reward_ratio": 2.1, "weighted_score": 12.3, "atr_pct": 1.9,
        "entry_type": "limit", "entry_price": 42800.0,
        "entry_zone": {"low": 42500.0, "high": 43000.0},
        "entry_label": "Limit - Fibonacci 0.618 golden pocket",
        "ichimoku": {"regime": "bullish", "price_vs_cloud": "above",
                     "tk_state": "bullish", "cloud_color": "green"},
        "elliott": {"pattern": "wave3_up", "current_wave": "3",
                    "wave_confidence": 0.82,
                    "implication": "Wave 3 in progress - strongest phase of the cycle",
                    "projection": 46500.0},
        "decision": {"a_plus": True, "base_confidence": 64.0,
                     "adjustments": ["Ichimoku regime aligned (bullish) x1.15",
                                     "A+ SETUP +6 (regime + cycle + zones all aligned)"]},
        "signals": [{"strategy": "triple_trend",
                     "reasons": ["Price above EMA200 (42110.1234)",
                                 "RSI crossed above 50 (48.0 -> 52.3)"]}],
    }


@pytest.fixture
def notifier(tmp_path, monkeypatch):
    monkeypatch.setattr(tg_module, "TG_STATE_FILE", tmp_path / "tg_state.json")
    n = TelegramNotifier()
    n.enabled = True  # simulate configured token/chat
    return n


def test_format_recommendation_is_arabic(notifier):
    msg = notifier._format_recommendation(_rec(), position_num=1, total=2)
    for frame in ["السعر:", "الدخول (أمر محدد)", "منطقة الدخول",
                  "الارتفاع المتوقع:", "الثقة:", "وقف الخسارة:",
                  "الهدف الأول TP1", "الهدف الثاني TP2",
                  "العائد/المخاطرة", "الإشارات:", "صفقة 1/2"]:
        assert frame in msg, f"missing {frame}"
    # dynamic content translated
    assert "الجيب الذهبي" in msg
    assert "RSI اخترق 50 صعوداً" in msg
    assert "السعر فوق EMA200" in msg
    assert "إيشيموكي" in msg and "صاعد" in msg and "خضراء" in msg
    assert "الموجة 3" in msg and "أقوى مرحلة" in msg
    assert "إعداد A+ ممتاز" in msg
    assert "نظام إيشيموكي موافق (صاعد)" in msg
    # no English frame labels leaked
    for english in ["*Price:", "*Entry (", "*Stop Loss:", "*Signals:*",
                    "Position ", "Confidence:"]:
        assert english not in msg, f"english frame leaked: {english}"


def test_send_recommendations_frames_are_arabic(notifier, monkeypatch):
    sent_msgs = []
    monkeypatch.setattr(notifier, "send", lambda m: sent_msgs.append(m) or True)
    count = notifier.send_recommendations([_rec()])
    assert count == 1
    assert len(sent_msgs) == 3  # header + rec + footer
    header, rec_msg, footer = sent_msgs
    assert "أفضل 1 توصية" in header
    assert "الوضع:" in header
    assert "حجم الصفقة:" in header
    assert "الإجمالي: 1 صفقة" in footer
    assert "للاستخدام التعليمي فقط" in footer


# ------------------------------------------------------------------
# cycle.py alerts
# ------------------------------------------------------------------
def test_notify_closed_arabic(monkeypatch):
    captured = []
    monkeypatch.setattr(telegram_notifier, "enabled", True, raising=False)
    monkeypatch.setattr(telegram_notifier, "send",
                        lambda m: captured.append(m) or True)
    cycle._notify_closed([{
        "symbol": "ETHUSDT", "entry_price": 2200.0, "exit_price": 2260.0,
        "pnl": 5.4, "pnl_pct": 2.72, "paper": True,
        "reason": "Trailing stop (profit +2.65%)",
    }])
    assert len(captured) == 1
    msg = captured[0]
    assert "إغلاق صفقة ✅" in msg
    assert "العملة: ETHUSDT" in msg
    assert "السبب: وقف متتبع (الربح 2.65%)" in msg
    assert "تجريبي" in msg


def test_notify_closed_partial_arabic(monkeypatch):
    captured = []
    monkeypatch.setattr(telegram_notifier, "enabled", True, raising=False)
    monkeypatch.setattr(telegram_notifier, "send",
                        lambda m: captured.append(m) or True)
    cycle._notify_closed([{
        "symbol": "SOLUSDT", "status": "partial", "fraction": 0.5,
        "exit_price": 151.2, "pnl": 3.1, "pnl_pct": 2.1,
    }])
    msg = captured[0]
    assert "جني جزئي TP1" in msg
    assert "الكمية المتبقية تستهدف TP2" in msg


def test_notify_updates_arabic_watch_tag(monkeypatch):
    captured = []
    monkeypatch.setattr(telegram_notifier, "enabled", True, raising=False)
    monkeypatch.setattr(telegram_notifier, "send",
                        lambda m: captured.append(m) or True)
    cycle._notify_updates([{
        "symbol": "BNBUSDT", "reason": "Chandelier trail 2.5xATR (peak 310.5, +1.90%)",
        "old_sl": 300.0, "new_sl": 305.0, "old_tp": 330.0, "new_tp": 330.0,
    }], tag="(watch)")
    msg = captured[0]
    assert "تحديث مخاطر 🔧" in msg
    assert "(مراقبة)" in msg
    assert "وقف ثريا 2.5×ATR" in msg
    assert "وقف قديم: 300.0 → وقف جديد: 305.0" in msg


def test_notify_disabled_sends_nothing(monkeypatch):
    captured = []
    monkeypatch.setattr(telegram_notifier, "enabled", False, raising=False)
    monkeypatch.setattr(telegram_notifier, "send",
                        lambda m: captured.append(m) or True)
    cycle._notify_closed([{"symbol": "X", "pnl": 1}])
    cycle._notify_updates([{"symbol": "X"}])
    assert captured == []
