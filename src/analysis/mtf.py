"""Multi-Timeframe Context - the professional's first filter (v5.21).

Research base (professional spot-trading practice):
  "Use a daily chart as the higher timeframe and a 1-hour chart as the
   lower timeframe. Trade only in the direction of the higher timeframe."
  - EMA 50 vs EMA 200 (and price relative to them) is the canonical
    macro-trend read on any timeframe.
  - The 1h frame is the tactical/timing frame: EMA9/EMA21 + RSI 50-line.
  - Mean reversion collapses in regime breaks: fade band extremes only
    when the macro trend is not violently against the trade.

WHAT IT DOES (spot-only, read-only, fails OPEN):
  - mtf_context(symbol, multi_tf_data) -> one dict per symbol per cycle:
      htf_trend    "up" | "down" | "flat" | None (None = not available)
      ltf_momentum "up" | "down" | "flat" | None
  - Strategies receive it via the `mtf_ctx` kwarg. When the context is
    unavailable every strategy behaves EXACTLY as before (fail-open) -
    the pre-v5.21 contracts stay bit-identical.

DATA SOURCES (zero/cheap REST, ban-aware):
  - The multi_tf_data dict the analyzer already fetched (WS-cached, free).
  - 1d macro frame: fetched once per symbol and TTL-cached
    (MTF_DAILY_TTL_MIN, weight 2/call -> ~3 weight/min at 95 symbols).
    Skipped during REST bans and ws_only bursts (fail-open).
  - 1h tactical frame: served from the WS cache (settings.WS_INTERVALS
    always includes "1h") - zero REST weight.

WHY THE HIGHER TF MATTERS HERE: production TIMEFRAMES="4h" means the
strategies read the 4h frame; without this module the bot never consults
the daily macro trend at all. Every production bottom-scanner loss in the
v5.20 forensics happened against a macro tide the code could not see.
"""
import time
from typing import Dict, Optional

import pandas as pd

from config.settings import settings
from src.utils.logger import log

# canonical Binance interval order (used to pick the highest frame)
_INTERVAL_RANK = {
    "1m": 1, "3m": 2, "5m": 3, "15m": 4, "30m": 5,
    "1h": 6, "2h": 7, "4h": 8, "6h": 9, "8h": 10, "12h": 11,
    "1d": 12, "3d": 13, "1w": 14,
}

# TTL caches (per process; 1d candles barely move intraday)
_daily_cache: Dict[str, tuple] = {}


def _ban_active(threshold: float = 60.0) -> bool:
    """True while a Binance cooldown is in force - do NOT add REST load."""
    try:
        from src.core.rate_limiter import rate_limiter
        return rate_limiter.cooldown_remaining() > threshold
    except Exception:
        return False


def trend_from_df(df: Optional[pd.DataFrame]) -> Dict:
    """Classify one OHLCV frame into up/down/flat via EMA50/EMA200.

    Professional read: price above a rising EMA50 with EMA50 > EMA200 is
    an uptrend; the mirror is a downtrend; anything else is flat/range.
    With < 210 bars the EMA200 is statistically weak, so a degraded
    EMA50-slope read is used instead (degraded=True).
    """
    if df is None or len(df) < 60:
        return {"usable": False}
    try:
        from src.indicators.technical import ema
        close = df["close"].astype(float)
        price = float(close.iloc[-1])
        e50_s = ema(close, 50)
        e50 = float(e50_s.iloc[-1])
        e50_prev = float(e50_s.iloc[-6]) if len(e50_s) > 6 else e50
        if len(close) >= 210:
            e200 = float(ema(close, 200).iloc[-1])
            if price > e50 and e50 > e200:
                trend = "up"
            elif price < e50 and e50 < e200:
                trend = "down"
            else:
                trend = "flat"
            return {"usable": True, "trend": trend, "degraded": False,
                    "price": price, "ema50": e50, "ema200": e200}
        # degraded: EMA50 + slope only
        if price > e50 and e50 > e50_prev:
            trend = "up"
        elif price < e50 and e50 < e50_prev:
            trend = "down"
        else:
            trend = "flat"
        return {"usable": True, "trend": trend, "degraded": True,
                "price": price, "ema50": e50}
    except Exception:
        return {"usable": False}


def _daily(symbol: str) -> Optional[pd.DataFrame]:
    """1d OHLCV, TTL-cached, ban-aware (the macro frame professionals read).

    Cache policy: a successful fetch lives MTF_DAILY_TTL_MIN (1d candles
    change slowly); a failed/None fetch lives MTF_DAILY_FAIL_TTL_S so a
    ban or a hiccup does not turn into a per-cycle retry storm.
    """
    if not getattr(settings, "MTF_FETCH_DAILY", True):
        return None
    if _ban_active():
        return None
    now = time.monotonic()
    hit = _daily_cache.get(symbol)
    if hit:
        ts, df = hit
        ttl = (max(1, settings.MTF_DAILY_TTL_MIN) * 60
               if df is not None
               else max(30, settings.MTF_DAILY_FAIL_TTL_S))
        if now - ts < ttl:
            return df
    try:
        from src.core.data_fetcher import data_fetcher
        df = data_fetcher.get_candles(symbol, "1d", 250)
        if df is None or len(df) < 60:
            df = None
    except Exception as e:
        log.debug(f"MTF daily fetch failed for {symbol}: {e}")
        df = None
    _daily_cache[symbol] = (now, df)
    return df


def _ltf_1h(symbol: str) -> Optional[pd.DataFrame]:
    """1h tactical frame - WS-cached (free), None during bans/quiet."""
    try:
        from src.core.data_fetcher import data_fetcher
        return data_fetcher.get_candles(symbol, "1h", 120)
    except Exception:
        return None


def ltf_momentum_from_df(df: Optional[pd.DataFrame]) -> Dict:
    """1h timing read: EMA9/EMA21 alignment + RSI 50-line (v5.21)."""
    if df is None or len(df) < 30:
        return {"usable": False}
    try:
        from src.indicators.technical import ema, rsi
        close = df["close"].astype(float)
        e9 = float(ema(close, 9).iloc[-1])
        e21 = float(ema(close, 21).iloc[-1])
        r = float(rsi(close, 14).iloc[-1])
        if e9 > e21 and r > 50:
            mom = "up"
        elif e9 < e21 and r < 50:
            mom = "down"
        else:
            mom = "flat"
        return {"usable": True, "momentum": mom, "ema9": e9, "ema21": e21,
                "rsi": r}
    except Exception:
        return {"usable": False}


def mtf_context(symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                primary_tf: Optional[str] = None) -> Dict:
    """One professional multi-timeframe snapshot for a symbol.

    HTF resolution order (strictly HIGHER frames only - reading the
    primary frame as its own context would be duplication, not
    confluence):
      1. the highest TF present in multi_tf_data above primary_tf
      2. the 1d frame (TTL-cached REST, ban-aware)
    LTF: the 1h WS-cached frame, only when primary_tf != "1h".
    """
    out = {
        "available": False, "htf_trend": None, "htf_tf": None,
        "htf_degraded": None, "htf_detail": {},
        "ltf_available": False, "ltf_momentum": None, "ltf_detail": {},
    }
    if not getattr(settings, "MTF_ENABLED", True):
        return out
    primary_rank = _INTERVAL_RANK.get(str(primary_tf or ""), None)

    # --- 1. higher frame from the data we already have (free) ---
    htf_df, htf_tf = None, None
    if multi_tf_data:
        best_rank = primary_rank if primary_rank else 0
        for tf, df in (multi_tf_data or {}).items():
            r = _INTERVAL_RANK.get(str(tf))
            if r is not None and r > best_rank and df is not None:
                best_rank, htf_df, htf_tf = r, df, str(tf)

    # --- 2. the daily macro frame (cheap REST, TTL-cached, ban-aware) ---
    if htf_df is None and "1d" != str(primary_tf):
        d = _daily(symbol)
        if d is not None:
            htf_df, htf_tf = d, "1d"

    if htf_df is not None:
        t = trend_from_df(htf_df)
        if t.get("usable"):
            out.update({
                "available": True,
                "htf_trend": t["trend"],
                "htf_tf": htf_tf,
                "htf_degraded": t.get("degraded"),
                "htf_detail": t,
            })

    # --- 3. 1h tactical frame (free, timing only) ---
    if getattr(settings, "MTF_LTF_ENABLED", True) and \
            str(primary_tf or "") != "1h":
        m = ltf_momentum_from_df(_ltf_1h(symbol))
        if m.get("usable"):
            out.update({
                "ltf_available": True,
                "ltf_momentum": m["momentum"],
                "ltf_detail": m,
            })
    return out


# strategy-signal language ("bullish"/"bearish") <-> trend language
# ("up"/"down"/"flat")
_DIR_TO_TREND = {"bullish": "up", "bearish": "down"}


def htf_agrees(ctx: Optional[Dict], direction: str) -> bool:
    """True when the macro frame confirms the trade direction."""
    if not ctx or not ctx.get("available"):
        return False
    want = _DIR_TO_TREND.get(direction)
    return want is not None and ctx.get("htf_trend") == want


def htf_against(ctx: Optional[Dict], direction: str) -> bool:
    """True when the macro frame runs AGAINST the trade direction.

    "flat" is deliberately NOT against: a range macro is where mean
    reversion legitimately lives (the regime router agrees).
    """
    if not ctx or not ctx.get("available"):
        return False
    t = ctx.get("htf_trend")
    if direction == "bullish":
        return t == "down"
    if direction == "bearish":
        return t == "up"
    return False


def ltf_agrees(ctx: Optional[Dict], direction: str) -> bool:
    """True when the 1h tactical frame confirms the trade direction."""
    if not ctx or not ctx.get("ltf_available"):
        return False
    want = _DIR_TO_TREND.get(direction)
    return want is not None and ctx.get("ltf_momentum") == want
