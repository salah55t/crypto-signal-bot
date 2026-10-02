"""Micro-Scalper (v5.25): the user's strategy, VERBATIM, on true 15s/30s
candles - SPOT-ONLY and LONG-ONLY.

The v5.22 DoubleIndicatorStrategy implemented the same document ("دليل
استراتيجية التداول المؤشر المزدوج - Bollinger Bands & SuperTrend") but on
1m candles, because Binance spot klines have no 15s/30s intervals. This
module closes that gap "بحذافيرها" (to the letter):

  * Binance spot offers exactly ONE sub-minute interval: 1s klines.
    The scanner fetches 1s bars and RESAMPLES them locally into real
    15s or 30s candles (bucket = floor(open_time / N) * N, OHLCV aggregated,
    the still-forming bucket is dropped - signals are only ever computed on
    CLOSED candles).

  * Timeframe selection (the doc's own words):
      - 15s candles -> market fast and relatively calm (default)
      - 30s candles -> market very strong/fast
    Speed proxy: the absolute move of the 1s closes over the last
    SCALP_SPEED_WINDOW_S seconds >= SCALP_FAST_MOVE_PCT  ->  30s.

  * LONG signal (the ONLY direction - "عند الصعود فقط", spot has no shorts):
      1. >= 3 consecutive green candles (close > open) on the resampled TF,
      2. those candles sit ABOVE the SuperTrend line (ATR 2, mult 2 - the
         line is green/uptrend),
      3. the candles are VERY CLOSE to the UPPER Bollinger band (11, 3).
    Golden rule: "لا تستعجل الدخول في أي صفقة حتى تتحقق الشروط بنسبة 100%"
    - the checklist is binary. All of them, or no trade.

  * Holding: FIXED 1 minute (the doc's expiry). The strategy emits the
    position ladder only to satisfy the spot risk engine; the SL is a
    disaster stop (0.8% floor) and the real exit is the scalp time exit in
    risk manager (first watcher tick past SCALP_HOLD_SECONDS).

v5.27 FEE REALITY: the first live day produced 6/6 time-exit losses where
fees were 72% of the realized loss - every trade fired in markets whose
resampled-TF ATR was a fraction of the 0.2% spot round-trip fee, so the
60s time exit paid the exchange regardless of signal quality. The entry
gates are therefore now FEE-LINKED (settings keep the knobs):

  * effective ATR floor  = max(SCALP_MIN_ATR_PCT,
                               round_trip_fee * SCALP_FEE_COVER_MULT)
    (0.2% * 1.25 = 0.25% at the default 0.1%/side taker fee) - one resampled
    candle must be able to travel ~1.25x the fees for the hold to matter.
  * effective range floor: SCALP_MIN_RANGE_PCT (0.60% over the lookback).
  * TP1 floor = round_trip_fee + SCALP_MIN_NET_MOVE_PCT (>= 0.35%) - a
    partial that cannot clear fees is not a target.

The mirrored SELL setup (3 red candles below SuperTrend near the lower
band) is intentionally NOT implemented - "عند الصعود فقط".
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config.settings import settings
from src.indicators.technical import atr, bollinger_bands
from src.indicators.supertrend import st_trend_series
from src.strategies.base import BaseStrategy, Signal
from src.strategies.double_indicator_strategy import last_closed_bars


def resample_1s(df: pd.DataFrame, seconds: int) -> pd.DataFrame:
    """Aggregate 1s klines into true `seconds`-candles (15s / 30s).

    Version-proof manual bucketing (no resample() alias pitfalls):
    every open_time is floored to its bucket start, then OHLCV are
    aggregated inside each bucket. The index becomes the bucket start
    (UTC) and a `close_time` column carries the bucket END (exclusive),
    so last_closed_bars() drops the still-forming last bucket exactly the
    way it drops forming REST candles.
    """
    if df is None or len(df) == 0 or seconds <= 0:
        return df
    out = df.copy()
    idx = pd.to_datetime(out.index, utc=True)
    ep_s = idx.asi8 // 10**9
    bucket_start = (ep_s // seconds) * seconds
    grp = pd.Series(bucket_start, index=out.index)
    agg = out.groupby(grp).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    agg.index = pd.to_datetime(agg.index, unit="s", utc=True)
    agg["close_time"] = agg.index + pd.Timedelta(seconds=seconds)
    return agg.sort_index()


def choose_timeframe(df_1s: pd.DataFrame) -> int:
    """Doc's TF rule: 15s when fast-but-calm, 30s when very strong/fast.

    Speed = |last close - close N seconds ago| / close * 100 over the last
    SCALP_SPEED_WINDOW_S one-second bars. Fails OPEN to the calm TF."""
    win = max(60, int(settings.SCALP_SPEED_WINDOW_S))
    try:
        tail = df_1s.tail(win)
        if len(tail) < 60:
            return int(settings.SCALP_TF_CALM)
        first = float(tail["close"].iloc[0])
        last = float(tail["close"].iloc[-1])
        if first <= 0:
            return int(settings.SCALP_TF_CALM)
        move_pct = abs(last - first) / first * 100.0
        if move_pct >= settings.SCALP_FAST_MOVE_PCT:
            return int(settings.SCALP_TF_FAST)
        return int(settings.SCALP_TF_CALM)
    except Exception:
        return int(settings.SCALP_TF_CALM)


class MicroScalpStrategy(BaseStrategy):
    """v5.25: BB(11, 3) + SuperTrend(2, 2) on TRUE 15s/30s candles,
    aggregated from 1s klines - long-only, fixed 1-minute holding."""
    name = "micro_scalp"
    weight = 1.0  # channel strategy: not part of the 6-strategy stack vote

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None,
                mtf_ctx: Optional[Dict] = None) -> Signal:
        seconds = choose_timeframe(df)
        min_1s = max(60, seconds * 30)  # >= 30 closed resampled buckets
        if df is None or len(df) < min_1s:
            return self._neutral("Insufficient 1s data for the scalp TF")

        rf = resample_1s(df, seconds)
        rf = last_closed_bars(rf)  # drop the forming bucket (close_time > now)
        min_bars = max(settings.SCALP_BB_PERIOD,
                       settings.SCALP_ST_PERIOD * 5) + \
            settings.SCALP_MIN_GREEN + 5
        if rf is None or len(rf) < max(30, min_bars):
            return self._neutral(
                f"Insufficient closed {seconds}s candles "
                f"({0 if rf is None else len(rf)} < {max(30, min_bars)})")

        close = rf["close"]
        high = rf["high"]
        low = rf["low"]

        # === Exact indicator settings from the user's document ===
        bb = bollinger_bands(close, settings.SCALP_BB_PERIOD,
                             settings.SCALP_BB_DEV)
        st = st_trend_series(high, low, close, settings.SCALP_ST_PERIOD,
                             settings.SCALP_ST_MULT)
        atr_series = atr(high, low, close, 14)
        atr_val = float(atr_series.iloc[-1]) \
            if not pd.isna(atr_series.iloc[-1]) else 0.0

        last_close = float(close.iloc[-1])
        if last_close <= 0:
            return self._neutral("bad price")

        # === Anti-dead-market floors (resampled-TF scale) ===
        # v5.27: the floors are FEE-LINKED - spot round-trip fees are
        # 2 x TRADING_FEE_PCT; a market whose candle ATR (and whose 60-bar
        # range) cannot cover them turns the 60s time exit into a guaranteed
        # fee donation, so it is untradeable no matter how clean the setup.
        fee_rt_pct = 2.0 * float(settings.TRADING_FEE_PCT)
        min_atr_pct = max(settings.SCALP_MIN_ATR_PCT,
                          fee_rt_pct * settings.SCALP_FEE_COVER_MULT)
        if atr_val > 0:
            atr_pct = atr_val / last_close * 100.0
            if atr_pct < min_atr_pct:
                return self._neutral(
                    f"dead/fee-food micro-range (TF ATR {atr_pct:.3f}% < "
                    f"fee-linked floor {min_atr_pct:.2f}% = fees "
                    f"{fee_rt_pct:.2f}% x {settings.SCALP_FEE_COVER_MULT:g})")
        lb = min(60, len(rf))
        tail = rf.tail(lb)
        rng_pct = ((float(tail["high"].max()) - float(tail["low"].min()))
                   / last_close * 100.0)
        if rng_pct < settings.SCALP_MIN_RANGE_PCT:
            return self._neutral(
                f"flat range ({rng_pct:.2f}% < "
                f"{settings.SCALP_MIN_RANGE_PCT:.2f}% over {lb} bars)")

        n = settings.SCALP_MIN_GREEN
        run = rf.iloc[-n:]
        closes = run["close"]
        opens = run["open"]
        lows = run["low"]
        st_line_run = st["st_line"].iloc[-n:]
        trend_now = int(st["trend"].iloc[-1])
        bb_upper = float(bb["upper"].iloc[-1])

        near_dist_pct = (bb_upper - last_close) / last_close * 100.0
        details = {
            "bb_period": settings.SCALP_BB_PERIOD,
            "bb_dev": settings.SCALP_BB_DEV,
            "st_period": settings.SCALP_ST_PERIOD,
            "st_mult": settings.SCALP_ST_MULT,
            "tf_seconds": seconds,
            "timeframe": f"{seconds}s",
            "resampled_bars": int(len(rf)),
            "bb_upper": bb_upper,
            "bb_lower": float(bb["lower"].iloc[-1]),
            "st_line": float(st_line_run.iloc[-1])
            if not pd.isna(st_line_run.iloc[-1]) else None,
            "st_trend": trend_now,
            "green_run": int((closes.values > opens.values).sum()),
            "near_upper_dist_pct": round(near_dist_pct, 4),
            "atr": atr_val,
        }

        # === THE 100% CHECKLIST (all conditions, no partial pass) ===
        green = closes.values > opens.values
        cond_green = bool(green.all())
        cond_above_st = trend_now == 1 and bool(
            (lows.values > st_line_run.values).all())
        cond_near_band = near_dist_pct <= settings.SCALP_BB_NEAR_PCT

        if not (cond_green and cond_above_st and cond_near_band):
            reasons = []
            if not cond_green:
                reasons.append(
                    f"لا: التتابع الأخضر أقل من {n} شموع "
                    f"{seconds}ث مغلقة")
            if not cond_above_st:
                reasons.append(
                    "لا: الشموع ليست كلها فوق خط SuperTrend الصاعد")
            if not cond_near_band:
                reasons.append(
                    "لا: السعر ليس قريباً جداً من الحد العلوي "
                    f"(البعد {near_dist_pct:.3f}% > "
                    f"{settings.SCALP_BB_NEAR_PCT:.2f}%)")
            return self._neutral(
                "Scalp checklist incomplete: " + "; ".join(reasons), details)

        # === Quality ranking (valid signal only - never creates one) ===
        score = float(settings.SCALP_CONF_BASE)
        reasons = [
            f"{n} شموع {seconds}ث خضراء متتالية فوق SuperTrend("
            f"{settings.SCALP_ST_PERIOD},{settings.SCALP_ST_MULT:g})",
            "قريبة جداً من الحد العلوي BB("
            f"{settings.SCALP_BB_PERIOD},{settings.SCALP_BB_DEV:g}) "
            f"(البعد {near_dist_pct:.3f}%)",
        ]

        vol_avg = float(rf["volume"].iloc[-20:].mean()) \
            if len(rf) >= 20 else 0.0
        vol_now = float(rf["volume"].iloc[-1])
        vol_ratio = vol_now / max(vol_avg, 1e-9)
        details["volume_ratio"] = round(vol_ratio, 3)
        if vol_ratio >= 1.5:
            score += 3
            reasons.append(f"حجم قوي على شمعة الزخم ({vol_ratio:.2f}x)")
        elif vol_ratio >= 1.0:
            score += 2
            reasons.append(f"حجم مؤكد ({vol_ratio:.2f}x)")

        body_ratio = float(
            ((closes.values - opens.values) /
             np.maximum(run["high"].values - run["low"].values, 1e-12)).min())
        details["body_ratio"] = round(body_ratio, 3)
        if body_ratio >= 0.6:
            score += 2
            reasons.append("شموع الزخم ذات أجسام صلبة")

        if bool(st["rising"].iloc[-n:].all()):
            score += 1
            reasons.append("خط SuperTrend يتسع صعوداً")

        hh_ll = bool(
            (run["high"].values[1:] > run["high"].values[:-1]).all()
            and (run["low"].values[1:] > run["low"].values[:-1]).all())
        if hh_ll:
            score += 1
            reasons.append("قيعان وقمم متصاعدة")

        score = max(0.0, min(score, settings.SCALP_CONF_CAP))
        details["score"] = float(score)

        # === Spot-adaptation ladder: the TIME EXIT is the strategy; the SL
        # is a disaster stop with a floor wide enough to survive 60s noise,
        # TP2 = 2x the SL distance (RR >= 2 by construction), TP1 a quick
        # partial that may print inside the minute. v5.27: TP1 must clear
        # the round-trip fee plus a net minimum - a partial that pays the
        # exchange more than the trade is not a target. ===
        sl_dist = max(atr_val * settings.SCALP_SL_ATR_MULT,
                      last_close * settings.SCALP_SL_MIN_PCT / 100.0)
        tp2_dist = max(sl_dist * settings.SCALP_TP2_RR_MULT,
                       last_close * settings.SCALP_TP2_MIN_PCT / 100.0)
        tp1_floor_pct = max(settings.SCALP_TP1_MIN_PCT,
                            fee_rt_pct + settings.SCALP_MIN_NET_MOVE_PCT)
        tp1_dist = max(atr_val * settings.SCALP_TP1_ATR_MULT,
                       last_close * tp1_floor_pct / 100.0)
        details.update({
            "stop_loss": last_close - sl_dist,
            "take_profit": last_close + tp1_dist,
            "take_profit_2": last_close + tp2_dist,
            "tp1_pct": round(tp1_dist / last_close * 100, 3),
            "tp2_pct": round(tp2_dist / last_close * 100, 3),
            "sl_pct": round(sl_dist / last_close * 100, 3),
            # v5.27 fee telemetry for the dashboard / debugging
            "fee_rt_pct": round(fee_rt_pct, 3),
            "min_atr_floor_pct": round(min_atr_pct, 3),
            "atr_pct": round(atr_val / last_close * 100, 3)
            if last_close > 0 else 0.0,
        })

        # LONG-ONLY by design: there is no bearish branch to fall into.
        return self._bull(score, reasons, details)
