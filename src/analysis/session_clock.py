"""Session Clock - the fixed daily rhythm of every exchange (v5.16).

User request: "fix the logic of ALL strategies, respecting the fixed daily
movements - exchange opens and closes - and the weekend days, whose
volatility swings; and avoid quasi-stable coins whose price is pinned."

Crypto trades 24/7, but its liquidity is NOT flat: the fixed anchors of
traditional desks drive the rhythm of every day:
  00:00 UTC  daily open/close (funding-style flows, daily candle rollover)
  07:00 UTC  London open        -> volatility burst
  11-14 UTC  London close + US pre-market -> the CHOP WINDOW
  13:30 UTC  NYSE open          -> second burst
  21:00 UTC  US equity close    -> liquidity drains

DIAGNOSTIC (data/diagnose_v516.json, 171 live-logic trades, 90 days 1h):
  hours 11,12,13,14 UTC  -> 54 trades, -172 USD, PF 0.05..0.79  (the chop)
  hours 00-03 UTC        -> PF 2.6..78   (Asia drift, cleanest trends)
  Saturday breakouts fake out (weekend PF 1.9 vs weekday 2.95).

WHAT IT DOES (spot-only, entries only - never touches exits):
  - session_info(dt)      -> the session tag + flags for one bar timestamp
  - entry_gate(...)       -> (blocked, reason) hard rules per strategy
  - strategy_session_weight(...) -> soft ranking multiplier per strategy
Hard rules are conservative and data-backed:
  1. CHOP HOURS (11-14 UTC): no NEW entries at all (any strategy).
  2. SATURDAY (thinnest day): no NEW breakout/momentum entries
     (volatility_breakout / macd_breakout) unless the rec is a_plus.
  3. Weekly open gap (Mon 00-01 UTC): breakouts allowed, reversals blocked
     (weekend wicks unwind violently right after the weekly open).
Everything is settings-gated (SESSION_FILTER_ENABLED) and fails OPEN.
"""
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from config.settings import settings

# session tags -> Arabic label (dashboard/log display)
SESSIONS_AR = {
    "daily_open": "افتتاح اليوم (00:00 UTC)",
    "asia": "جلسة آسيا",
    "london_open": "افتتاح لندن",
    "london": "جلسة لندن",
    "chop": "فجوة الاختلاق (إغلاق لندن/ما قبل نيويورك)",
    "ny": "جلسة نيويورك",
    "us_late": "ما بعد إغلاق وول ستريت",
    "weekend": "نهاية الأسبوع",
}


def session_info(now: Optional[datetime] = None) -> Dict:
    """Classify one UTC timestamp into the fixed daily/weekly rhythm."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    h, wd = now.hour, now.weekday()  # Mon=0..Sun=6

    if wd == 6 and h >= 22 or wd == 4 and h >= 22 or wd == 5:
        weekend = True   # v5.13 window: Fri 22:00 -> Mon 00:00
    elif wd == 6:
        weekend = True   # all Sunday (the v5.13 router treats Sun as weekend)
    else:
        weekend = False

    if h == 0:
        session = "daily_open"
    elif h < 7:
        session = "asia"
    elif h == 7:
        session = "london_open"
    elif h < 11:
        session = "london"
    elif h < 15:
        session = "chop"
    elif h < 21:
        session = "ny"
    else:
        session = "us_late"

    return {
        "hour": h,
        "weekday": wd,              # 0=Mon .. 6=Sun
        "is_saturday": wd == 5,
        "is_weekend": weekend,      # v5.13 window (Fri 22:00 -> Mon 00:00)
        "session": session,
        "session_ar": SESSIONS_AR.get(session, session),
    }


def _blocked_hours() -> set:
    try:
        raw = str(getattr(settings, "SESSION_BLOCK_HOURS", "11,12,13,14"))
        return {int(x) for x in raw.split(",") if str(x).strip() != ""}
    except Exception:
        return {11, 12, 13, 14}


BREAKOUT_STRATS = {"volatility_breakout", "macd_breakout"}
REVERSAL_STRATS = {"liquidity_sweep_reversal", "bb_mean_reversion"}


def entry_gate(info: Dict, dominant_strategy: Optional[str],
               a_plus: bool = False) -> Tuple[bool, str]:
    """Hard NEW-ENTRY rules for one bar. Returns (blocked, reason).

    Fails OPEN on anything unexpected - the session layer may only ever
    skip entries; it must never crash an analysis cycle.
    """
    if not getattr(settings, "SESSION_FILTER_ENABLED", True):
        return False, ""
    try:
        s = info.get("session", "")
        h = int(info.get("hour", -1))
        strat = dominant_strategy or ""

        # 1) the chop window: London close -> US pre-market.
        #    Diagnostic: 54 trades, PF 0.05-0.79 - structurally losing time.
        if h in _blocked_hours():
            return True, (
                f"نافذة الاختلاق ({h}:00 UTC) - إغلاق لندن/ما قبل نيويورك: "
                f"دخول جديد معطل (بيانياً PF<0.8)")

        # 2) Saturday is the thinnest day - breakouts fake out.
        if (info.get("is_saturday") and strat in BREAKOUT_STRATS
                and not a_plus
                and getattr(settings, "SESSION_SATURDAY_BLOCK_BREAKOUTS",
                            True)):
            return True, (
                "سبت (أرفع يوم سيولة) - الكسور والزخم معطلان إلا لإشارة A+")

        # 3) the first hours after the weekly open unwind weekend wicks -
        #    reversal (knife-catching) entries are blocked, trends allowed.
        if (s == "daily_open" and info.get("weekday") == 0
                and strat in REVERSAL_STRATS
                and getattr(settings, "SESSION_MONDAY_BLOCK_REVERSALS", True)):
            return True, "افتتاح الأسبوع - الارتدادات معطلة (تصفية ذيول الأسبوع)"

        return False, ""
    except Exception:
        return False, ""


def strategy_session_weight(info: Dict, dominant_strategy: Optional[str]
                            ) -> float:
    """Soft ranking multiplier (0.6..1.25) - which strategy owns this hour.

    Diagnostic-backed (171 trades, 90d):
      asia (00-07)     PF 4.7  -> trends/breakouts like the Asia drift
      chop (11-14)     PF 0.4  -> (hard-blocked above anyway)
      ny (14-21)       PF 1.7  -> neutral
      us_late (21-24)  PF 10.7 -> thin drift: mean reversion shines
      london_open (07) burst   -> breakouts fire on the open impulse
    """
    if not getattr(settings, "SESSION_FILTER_ENABLED", True):
        return 1.0
    try:
        s = info.get("session", "")
        strat = dominant_strategy or ""
        w = 1.0
        if s in ("daily_open", "asia"):
            # Asia drift is the cleanest trend hours (PF 2.7-11)
            if strat in BREAKOUT_STRATS or strat in (
                    "trend_pullback", "triple_confluence_trend"):
                w *= 1.15
            w *= 1.10
        elif s == "london_open":
            # the 07:00 impulse is breakout territory
            if strat in BREAKOUT_STRATS:
                w *= 1.20
        elif s == "us_late":
            # post-close thin drift: fade extremes, don't chase momentum
            if strat in ("bb_mean_reversion", "liquidity_sweep_reversal"):
                w *= 1.20
            elif strat in BREAKOUT_STRATS:
                w *= 0.80
        elif s == "ny":
            if strat in BREAKOUT_STRATS:
                w *= 1.05   # the 13:30-15:00 burst lives here
        if info.get("is_weekend") and strat in BREAKOUT_STRATS:
            w *= 0.85       # thin weekend book on top of the v5.13 router
        return round(max(0.6, min(w, 1.25)), 3)
    except Exception:
        return 1.0


def describe(info: Dict) -> str:
    """One Arabic line for logs/dashboard."""
    try:
        return (f"{SESSIONS_AR.get(info.get('session', ''), '')} "
                f"({int(info.get('hour', 0)):02d}:00 UTC"
                f"{' · نهاية أسبوع' if info.get('is_weekend') else ''})")
    except Exception:
        return ""
