"""
v5.7 Composite Strategy #6: MACD BREAKOUT (user specification)

"التداول التلقائي مع انفجار الزخم والتغير المباشر في حركة السعر"

Signal Stack Framework (the golden rule - one indicator per class):
  DIRECTION      : EMA 50                (never trade against the mean path)
  MOMENTUM       : MACD (12, 26, 9)      (the breakout engine itself)
  LIQUIDITY      : Volume vs 20-avg      (real money behind the burst)

Automatic BUY conditions (all from the user spec):
  1. MACD line crosses ABOVE the signal line BELOW the zero line
     (or just above it - earliest momentum ignition)
  2. Histogram flips from negative to POSITIVE
  3. Price trades ABOVE EMA 50          (direction confirmation)
  Bonus: volume above average, EMA 50 sloping up

Automatic trade management (user spec: "Laddered Multi-TP + Trailing SL"):
  - Mapped 1:1 onto the bot's existing veteran machinery: TP1 banks 50% and
    moves SL to break-even, TP2 runs, chandelier ATR trailing follows price
    up. The MACD "bend down" exit below emits the bearish signal that the
    v5.5 opposite-signal machinery (conf >= 55 = immediate close) acts on.

Automatic SELL/EXIT conditions (emitted as bearish signals):
  1. FRESH MACD cross BELOW the signal line while price is above EMA 50
     (momentum bent down - the runner's job is done)
  2. Histogram flips negative (momentum officially negative)
"""
import pandas as pd
from typing import Dict, Optional
from src.indicators.technical import ema, macd, atr
from src.strategies.base import BaseStrategy, Signal

MIN_BARS = 80  # MACD(26+9) + EMA50 warmup
RECENT_CROSS_BARS = 3


class MACDBreakoutStrategy(BaseStrategy):
    """Composite #6: automatic momentum-burst riding with laddered exits."""
    name = "macd_breakout"

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < MIN_BARS:
            return self._neutral(
                f"Insufficient data for MACD/EMA50 ({len(df)}/{MIN_BARS})")

        close = df["close"]
        volume = df["volume"]

        # === Signal Stack: one indicator per class ===
        macd_df = macd(close, 12, 26, 9)
        line = macd_df["macd"]
        signal_line = macd_df["signal"]
        hist = macd_df["histogram"]
        ema50 = ema(close, 50)
        atr_val = float(atr(df["high"], df["low"], close, 14).iloc[-1])

        price = float(close.iloc[-1])
        e50 = float(ema50.iloc[-1])
        e50_rising = e50 > float(ema50.iloc[-4])
        line_now, line_prev = float(line.iloc[-1]), float(line.iloc[-2])
        sig_now, sig_prev = float(signal_line.iloc[-1]), float(signal_line.iloc[-2])
        hist_now, hist_prev = float(hist.iloc[-1]), float(hist.iloc[-2])
        vol_ratio = float(volume.iloc[-1] / max(
            float(volume.rolling(20).mean().iloc[-1]), 1e-9))

        details = {
            "stack": {"direction": "EMA50", "momentum": "MACD(12,26,9)",
                      "liquidity": "Volume"},
            "price": price, "ema50": e50, "ema50_rising": e50_rising,
            "macd": line_now, "macd_signal": sig_now, "macd_hist": hist_now,
            "volume_ratio": vol_ratio, "atr": atr_val,
        }

        # === EXIT SIDE (MACD bent down -> v5.5 immediate-close machinery) ===
        fresh_cross_down = (line_prev >= sig_prev) and (line_now < sig_now)
        if fresh_cross_down and price > e50:
            return self._bear(
                55, "MACD crossed below signal line - momentum bent down",
                details)
        if hist_now < 0 and hist_prev >= 0:
            return self._bear(
                55, "MACD histogram flipped negative - momentum lost",
                details)

        # === BUY CHECKLIST (Signal Stack) ===
        score = 0
        reasons = []

        # --- DIRECTION class (mandatory gate) ---
        # 3) price above EMA 50
        if price <= e50:
            return self._neutral(
                f"Price below EMA50 ({price:.4f} <= {e50:.4f})", details)
        score += 20
        reasons.append(f"Price above EMA50 ({e50:.4f})")

        # --- MOMENTUM class ---
        # 1) MACD x signal cross (fresh = trigger, recent = still valid)
        fresh_cross_up = (line_prev <= sig_prev) and (line_now > sig_now)
        recent_cross_up = self._crossed_within(line, signal_line,
                                               RECENT_CROSS_BARS)
        if fresh_cross_up:
            score += 30
            reasons.append("FRESH MACD cross above signal line")
        elif recent_cross_up:
            score += 22
            reasons.append(f"Recent MACD cross (<= {RECENT_CROSS_BARS} bars)")
        else:
            return self._neutral("No MACD bullish crossover", details)

        # cross location: below the zero line (earliest) or just above it
        if line_now < 0:
            score += 12
            reasons.append("Zero-line cross (below zero - earliest entry)")
        elif line_now <= 0.5 * max(atr_val, 1e-9):
            score += 8
            reasons.append("Cross just above zero line")
        # else: deep-positive cross = late; no bonus, no veto

        # 2) histogram flip negative -> positive
        if hist_now > 0 and hist_prev <= 0:
            score += 20
            reasons.append("Histogram flipped positive")
        elif hist_now > 0 and hist_now > hist_prev:
            score += 10
            reasons.append("Histogram positive and rising")
        elif hist_now > 0:
            score += 5
            reasons.append("Histogram positive")
        else:
            return self._neutral("Histogram still negative", details)

        # --- LIQUIDITY class ---
        if vol_ratio >= 1.3:
            score += 15
            reasons.append(f"Breakout volume ({vol_ratio:.2f}x avg20)")
        elif vol_ratio >= 1.0:
            score += 10
            reasons.append(f"Volume confirmed ({vol_ratio:.2f}x)")
        else:
            return self._neutral(
                f"No volume confirmation ({vol_ratio:.2f}x avg20)", details)

        # bonus: EMA 50 sloping up (path least resistance)
        if e50_rising:
            score += 8
            reasons.append("EMA50 sloping up")

        score = max(-100, min(100, score))

        if score >= 60:
            return self._bull(score, reasons, details)
        elif score >= 40:
            return self._bull(score * 0.5,
                              [f"Partial breakout: {r}" for r in reasons],
                              details)
        return self._neutral(f"Score too low ({score})", details)

    @staticmethod
    def _crossed_within(line: pd.Series, signal_line: pd.Series,
                        bars: int) -> bool:
        """True when line crossed ABOVE signal within the last `bars` bars."""
        diff = (line - signal_line).dropna()
        if len(diff) < bars + 1:
            return False
        tail = diff.iloc[-(bars + 1):]
        return bool(((tail.shift(1) <= 0) & (tail > 0)).any())
