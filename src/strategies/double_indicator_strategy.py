"""Composite Strategy #7: DOUBLE INDICATOR (v5.22) - the user's strategy,
applied to the letter, SPOT-ONLY and LONG-ONLY.

Source: the user's document "دليل استراتيجية التداول المؤشر المزدوج
(Bollinger Bands & SuperTrend)". Its rules, verbatim:

  Indicators:
    Bollinger Bands: period 11, deviation 3 (the middle band is visually
      removed in the doc - we keep it only as the percent_B basis).
    SuperTrend: ATR length 2, multiplier 2.

  BUY signal (the only direction this bot trades - spot has no shorts and
  the user asked for the upward side only):
    1. 3 consecutive green candles or more (close > open),
    2. the candles sit ABOVE the SuperTrend line (the line is green/uptrend),
    3. the candles are very close to the UPPER Bollinger band.

  Golden rule: "لا تستعجل الدخول في أي صفقة حتى تتحقق الشروط بنسبة 100%"
  - the conditions are binary. All of them, or no trade. Quality bonuses
  below only RANK a valid signal; they can never create one from a partial
  setup.

Spot adaptations (documented, not deviations of the entry rules):
  * Timeframe: the doc's 15s/30s candles do not exist on Binance spot
    klines; the doc's own trade duration is 1 minute, so the scanner feeds
    1m candles (settings.DOUBLE_IND_TIMEFRAME).
  * Exits: the doc has none (binary options expire after 1 minute). For
    spot we attach the bot's standard ladder with percentage FLOORS so
    every target clears the 0.2% round-trip fee (the v5.18 lesson):
      SL  = max(1.5 x ATR, DOUBLE_IND_SL_MIN_PCT  % of price)
      TP1 = max(1.2 x ATR, DOUBLE_IND_TP1_MIN_PCT % of price)
      TP2 = max(2.5 x ATR, DOUBLE_IND_TP2_MIN_PCT % of price)
  * The mirrored SELL setup (3 red candles below SuperTrend near the lower
    band) is intentionally NOT implemented - "عند الصعود فقط".
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config.settings import settings
from src.indicators.technical import atr, bollinger_bands
from src.indicators.supertrend import st_trend_series
from src.strategies.base import BaseStrategy, Signal


def last_closed_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the still-forming candle (Binance returns it as the last row).

    A signal on a forming candle flickers (the candle can turn red before
    it closes) - the doc's "100% conditions" are only knowable on CLOSED
    candles. Frames without a close_time column (tests, WS cache) are
    assumed to be closed bars already.
    """
    if df is None or len(df) == 0 or "close_time" not in df.columns:
        return df
    try:
        from src.utils.helpers import now_utc
        now = now_utc()
        ct = pd.to_datetime(df["close_time"], utc=True)
        return df.loc[ct <= now]
    except Exception:
        return df


class DoubleIndicatorStrategy(BaseStrategy):
    """v5.22: Bollinger Bands (11, 3) + SuperTrend (2, 2) - long-only."""
    name = "double_indicator"
    weight = 1.0  # channel strategy: not part of the 6-strategy stack vote

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None,
                mtf_ctx: Optional[Dict] = None) -> Signal:
        df = last_closed_bars(df)
        min_bars = max(settings.DOUBLE_IND_BB_PERIOD,
                       settings.DOUBLE_IND_ST_PERIOD * 5) + \
            settings.DOUBLE_IND_MIN_GREEN + 5
        if df is None or len(df) < max(30, min_bars):
            return self._neutral("Insufficient data")

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # === Exact indicator settings from the user's document ===
        bb = bollinger_bands(close, settings.DOUBLE_IND_BB_PERIOD,
                             settings.DOUBLE_IND_BB_DEV)
        st = st_trend_series(high, low, close, settings.DOUBLE_IND_ST_PERIOD,
                             settings.DOUBLE_IND_ST_MULT)
        atr_series = atr(high, low, close, 14)
        atr_val = float(atr_series.iloc[-1]) \
            if not pd.isna(atr_series.iloc[-1]) else 0.0

        n = settings.DOUBLE_IND_MIN_GREEN
        run = df.iloc[-n:]
        closes = run["close"]
        opens = run["open"]
        lows = run["low"]
        last_close = float(closes.iloc[-1])
        st_line_run = st["st_line"].iloc[-n:]
        trend_now = int(st["trend"].iloc[-1])

        pct_b = bb["percent_b"].iloc[-n:].astype(float)
        pct_b_last = float(pct_b.iloc[-1])
        pct_b_avg = float(pct_b.mean())

        details = {
            "bb_period": settings.DOUBLE_IND_BB_PERIOD,
            "bb_dev": settings.DOUBLE_IND_BB_DEV,
            "st_period": settings.DOUBLE_IND_ST_PERIOD,
            "st_mult": settings.DOUBLE_IND_ST_MULT,
            "bb_upper": float(bb["upper"].iloc[-1]),
            "bb_lower": float(bb["lower"].iloc[-1]),
            "pct_b_last": round(pct_b_last, 4),
            "pct_b_avg": round(pct_b_avg, 4),
            "st_line": float(st_line_run.iloc[-1])
            if not pd.isna(st_line_run.iloc[-1]) else None,
            "st_trend": trend_now,
            "green_run": int((closes.values > opens.values).sum()),
            "atr": atr_val,
        }

        # === THE 100% CHECKLIST (all conditions, no partial pass) ===
        green = closes.values > opens.values
        cond_green = bool(green.all())
        cond_above_st = trend_now == 1 and bool(
            (lows.values > st_line_run.values).all())
        cond_near_band = (
            pct_b_last >= settings.DOUBLE_IND_BB_PCTB_MIN
            and pct_b_avg >= settings.DOUBLE_IND_BB_PCTB_AVG_MIN)

        if not (cond_green and cond_above_st and cond_near_band):
            reasons = []
            if not cond_green:
                reasons.append(
                    f"لا: التتابع الأخضر أقل من {n} شموع مغلقة")
            if not cond_above_st:
                reasons.append(
                    "لا: الشموع ليست كلها فوق خط SuperTrend الصاعد")
            if not cond_near_band:
                reasons.append(
                    "لا: السعر ليس قريباً جداً من الحد العلوي "
                    f"(percentB {pct_b_last:.2f})")
            return self._neutral(
                "Double-indicator checklist incomplete: " + "; ".join(reasons),
                details)

        # === Quality ranking (valid signal only - never creates one) ===
        score = float(settings.DOUBLE_IND_CONF_BASE)
        reasons = [
            f"{n} شموع خضراء متتالية فوق SuperTrend({settings.DOUBLE_IND_ST_PERIOD},"
            f"{settings.DOUBLE_IND_ST_MULT:g})",
            "قريبة جداً من الحد العلوي BB("
            f"{settings.DOUBLE_IND_BB_PERIOD},{settings.DOUBLE_IND_BB_DEV:g}) "
            f"percentB={pct_b_last:.2f}",
        ]

        vol_avg = float(df["volume"].iloc[-20:].mean()) if len(df) >= 20 else 0.0
        vol_now = float(df["volume"].iloc[-1])
        vol_ratio = vol_now / max(vol_avg, 1e-9)
        details["volume_ratio"] = round(vol_ratio, 3)
        if vol_ratio >= 1.5:
            score += 6
            reasons.append(f"حجم قوي على شمعة الزخم ({vol_ratio:.2f}x)")
        elif vol_ratio >= 1.0:
            score += 3
            reasons.append(f"حجم مؤكد ({vol_ratio:.2f}x)")
        elif vol_ratio < settings.DOUBLE_IND_MIN_VOL_RATIO:
            score -= 8
            reasons.append(f"حجم ضعيف ({vol_ratio:.2f}x) - زخم مريب")

        body_ratio = float(
            ((closes.values - opens.values) /
             np.maximum(run["high"].values - run["low"].values, 1e-12)).min())
        details["body_ratio"] = round(body_ratio, 3)
        if body_ratio >= 0.6:
            score += 4
            reasons.append("شموع الزخم ذات أجسام صلبة")

        if bool(st["rising"].iloc[-n:].all()):
            score += 3
            reasons.append("خط SuperTrend يتسع صعوداً")

        hh_ll = bool(
            (run["high"].values[1:] > run["high"].values[:-1]).all()
            and (run["low"].values[1:] > run["low"].values[:-1]).all())
        if hh_ll:
            score += 3
            reasons.append("قيعان وقمم متصاعدة")

        if last_close > float(bb["upper"].iloc[-1]):
            score -= 4
            reasons.append("الإغلاق خارج الحد - مطاردة متأخرة")

        score = max(0.0, min(score, settings.DOUBLE_IND_CONF_CAP))
        details["score"] = float(score)

        # === Fee-survival exit ladder (spot adaptation) ===
        # v5.27: TP1 is fee-linked and ATR-dominant. Live evidence (ZEC):
        # TP1 0.50% vs SL 0.55% with 0.2% round-trip fees = inverted
        # geometry (net win 0.30% vs net loss 0.75% -> breakeven win rate
        # 71%). TP1 floor = max(TP1_MIN_PCT, 3.5 x round-trip fee) and the
        # TP1 ATR multiple (1.8x) now exceeds the SL multiple (1.5x) so
        # RR-on-TP1 >= 1.2 by construction.
        fee_rt_pct = 2.0 * float(settings.TRADING_FEE_PCT)
        tp1_floor_pct = max(settings.DOUBLE_IND_TP1_MIN_PCT,
                            fee_rt_pct * settings.DOUBLE_IND_FEE_TP1_MULT)
        if atr_val > 0 and last_close > 0:
            sl_dist = max(atr_val * settings.DOUBLE_IND_SL_ATR_MULT,
                          last_close * settings.DOUBLE_IND_SL_MIN_PCT / 100.0)
            tp1_dist = max(atr_val * settings.DOUBLE_IND_TP1_ATR_MULT,
                           last_close * tp1_floor_pct / 100.0)
            tp2_dist = max(atr_val * settings.DOUBLE_IND_TP2_ATR_MULT,
                           last_close * settings.DOUBLE_IND_TP2_MIN_PCT / 100.0)
        else:
            sl_dist = last_close * settings.DOUBLE_IND_SL_MIN_PCT / 100.0
            tp1_dist = last_close * tp1_floor_pct / 100.0
            tp2_dist = last_close * settings.DOUBLE_IND_TP2_MIN_PCT / 100.0
        details.update({
            "stop_loss": last_close - sl_dist,
            "take_profit": last_close + tp1_dist,
            "take_profit_2": last_close + tp2_dist,
            "tp1_pct": round(tp1_dist / last_close * 100, 3),
            "tp2_pct": round(tp2_dist / last_close * 100, 3),
            "sl_pct": round(sl_dist / last_close * 100, 3),
            # v5.27 fee telemetry
            "fee_rt_pct": round(fee_rt_pct, 3),
        })

        # LONG-ONLY by design: there is no bearish branch to fall into.
        return self._bull(score, reasons, details)
