"""
Multi-Strategy Signal Scorer
Aggregates signals from all strategies into a single confidence score (0-100)
and computes the expected rise %, entry price, stop loss, and take profit.

## Confidence model (v2 — strength x confluence)

The old formula `confidence = (weighted_score + 100) / 2` was fundamentally
broken:
  - A totally NEUTRAL market (score 0) produced 50% confidence.
  - ONE strategy firing at maximum strength produced only ~67%.
  - The 70% backtest threshold required near-unanimous confluence -> the
    historical backtest produced ZERO trades (see data/backtest_report.json).

New model:
  1. Each strategy votes: direction (bullish/bearish/neutral) + strength
     (|score| / 100).
  2. avg_strength  = weighted mean strength of strategies AGREEING with the
     final direction (how strong is the signal?).
  3. confluence    = weighted fraction of ALL strategy weight agreeing with
     the final direction (how unanimous is the signal?).
  4. confidence    = 100 * (0.65 * avg_strength + 0.35 * confluence)

 Calibration:
  - 1 strategy @ 80 strength alone          -> ~64%
  - 2 strategies @ 80                        -> ~76%
  - 3 strategies @ 80                        -> ~87%
  - neutral market                           -> 0% (was 50%!)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from config.settings import settings
from src.strategies import (
    # 3 powerful composite strategies only (clean & focused)
    TrendPullbackStrategy,
    LiquiditySweepReversalStrategy,
    VolatilityBreakoutStrategy,
    Signal
)
from src.indicators import technical as ta
from src.indicators.liquidity import find_support_resistance
from src.utils.logger import log


class SignalScorer:
    """Combines signals from multiple strategies into a final recommendation."""

    # Score threshold above which a strategy vote counts towards direction
    VOTE_THRESHOLD = 15.0
    # Net weighted score needed to call the market bullish / bearish
    DIRECTION_THRESHOLD = 8.0

    def __init__(self):
        self.strategies = [
            # 3 powerful composite strategies (clean & focused)
            TrendPullbackStrategy(weight=2.0),                # buy strength on pullbacks
            LiquiditySweepReversalStrategy(weight=2.0),       # join stop hunts reversal
            VolatilityBreakoutStrategy(weight=1.8),           # squeeze breakout with volume
        ]
        self.total_weight = sum(s.weight for s in self.strategies)
        log.info(
            f"[cyan]SignalScorer[/] initialized with {len(self.strategies)} composite strategies "
            f"(strength x confluence confidence model)"
        )

    # ------------------------------------------------------------------
    # Confidence model
    # ------------------------------------------------------------------
    def _compute_confidence(self, signals: List[Signal],
                            strategies: List) -> Dict:
        """Return dict with direction, weighted_score, confidence, confluence info."""
        total_w = sum(s.weight for s in strategies)

        weighted_score = sum(
            sig.score * strat.weight for sig, strat in zip(signals, strategies)
        ) / total_w

        if weighted_score > self.DIRECTION_THRESHOLD:
            direction = "bullish"
        elif weighted_score < -self.DIRECTION_THRESHOLD:
            direction = "bearish"
        else:
            direction = "neutral"

        agree_w = 0.0
        agree_strength_w = 0.0
        for sig, strat in zip(signals, strategies):
            if direction == "neutral":
                continue
            sig_dir = "bullish" if sig.score > self.VOTE_THRESHOLD else \
                      "bearish" if sig.score < -self.VOTE_THRESHOLD else "neutral"
            if sig_dir == direction:
                strength = min(abs(sig.score), 100) / 100.0
                agree_w += strat.weight
                agree_strength_w += strat.weight * strength

        if agree_w > 0:
            avg_strength = agree_strength_w / agree_w
            confluence = agree_w / total_w
        else:
            avg_strength = 0.0
            confluence = 0.0

        confidence = 100.0 * (0.65 * avg_strength + 0.35 * confluence)
        confidence = max(0.0, min(100.0, confidence))

        return {
            "direction": direction,
            "weighted_score": float(weighted_score),
            "confidence": float(confidence),
            "avg_strength": float(avg_strength),
            "confluence": float(confluence),
        }

    # ------------------------------------------------------------------
    # SL / TP (structure-aware)
    # ------------------------------------------------------------------
    def _compute_sl_tp(self, df: pd.DataFrame, direction: str,
                       current_price: float, atr_val: float,
                       sr: Dict) -> Dict:
        """Structure-aware SL/TP.

        SL (bullish): just below the 10-bar swing low (+0.4 ATR buffer),
        clamped to [0.9, 2.2] x ATR so it is never absurdly tight or wide.
        TP: at nearest resistance (structure) but never below the minimum
        R/R ratio; capped so targets stay realistic.
        """
        min_rr = max(1.2, settings.MIN_RR_RATIO)

        low = df["low"]
        high = df["high"]
        swing_low = float(low.iloc[-10:].min())
        swing_high = float(high.iloc[-10:].max())

        if direction == "bearish":
            struct_dist = (swing_high - current_price) + 0.4 * atr_val
            sl_dist = min(max(struct_dist, 0.9 * atr_val), 2.2 * atr_val)
            stop_loss = current_price + sl_dist

            res_level = sr.get("nearest_support")
            tp_struct = (res_level - current_price) if res_level and res_level < current_price else None
            tp_dist = max(tp_struct, min_rr * sl_dist) if tp_struct else min_rr * sl_dist
            tp_dist = min(tp_dist, 5.0 * sl_dist)
            take_profit = current_price - tp_dist
        elif direction == "bullish":
            struct_dist = (current_price - swing_low) + 0.4 * atr_val
            sl_dist = min(max(struct_dist, 0.9 * atr_val), 2.2 * atr_val)
            stop_loss = current_price - sl_dist

            res_level = sr.get("nearest_resistance")
            tp_struct = (res_level - current_price) if res_level and res_level > current_price else None
            tp_dist = max(tp_struct, min_rr * sl_dist) if tp_struct else min_rr * sl_dist
            tp_dist = min(tp_dist, 5.0 * sl_dist)
            take_profit = current_price + tp_dist
        else:
            sl_dist = 1.5 * atr_val
            stop_loss = current_price - sl_dist
            take_profit = current_price + 1.5 * sl_dist
            tp_dist = 1.5 * sl_dist

        rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0.0
        return {
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "risk_reward_ratio": float(rr_ratio),
            "sl_distance_pct": float(sl_dist / current_price * 100) if current_price > 0 else 0.0,
            "tp_distance_pct": float(tp_dist / current_price * 100) if current_price > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def analyze_symbol(self, df: pd.DataFrame, symbol: str,
                       multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                       order_book: Optional[Dict] = None) -> Dict:
        """
        Run all strategies on a symbol and produce a final score.
        Returns a comprehensive recommendation dict.
        """
        if df is None or len(df) < 60:
            return {"symbol": symbol, "skip": True, "reason": "Insufficient data"}

        signals: List[Signal] = []
        for strat in self.strategies:
            try:
                sig = strat.analyze(df, symbol, multi_tf_data, order_book)
                signals.append(sig)
            except Exception as e:
                log.warning(f"Strategy {strat.name} failed for {symbol}: {e}")
                signals.append(Signal(
                    strategy=strat.name,
                    direction="neutral",
                    score=0,
                    confidence=0,
                    reasons=[f"Strategy error: {e}"],
                ))

        verdict = self._compute_confidence(signals, self.strategies)
        direction = verdict["direction"]

        current_price = float(df["close"].iloc[-1])

        # ATR with NaN fallback (recent mean range)
        atr_series = ta.atr(df["high"], df["low"], df["close"], 14)
        atr_val = float(atr_series.iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            recent_range = (df["high"].iloc[-14:] - df["low"].iloc[-14:]).mean()
            atr_val = float(recent_range) if not np.isnan(recent_range) else 0.0
        atr_pct = atr_val / current_price if current_price > 0 else 0.0

        # Expected rise: ATR projected over the typical holding window,
        # scaled by signal strength (0..1). Much more realistic than the old
        # formula which under-estimated by ~3x.
        strength_norm = max(0.0, verdict["weighted_score"] / 100.0)
        expected_rise_pct = atr_pct * 100 * (1.2 + 1.8 * strength_norm)
        expected_rise_pct = float(min(expected_rise_pct, 30.0))

        sr = find_support_resistance(df, lookback=50)
        sl_tp = self._compute_sl_tp(df, direction, current_price, atr_val, sr)

        return {
            "symbol": symbol,
            "direction": direction,
            "weighted_score": verdict["weighted_score"],
            "confidence": verdict["confidence"],
            "avg_strength": verdict["avg_strength"],
            "confluence": verdict["confluence"],
            "current_price": current_price,
            "expected_rise_pct": expected_rise_pct,
            "stop_loss": sl_tp["stop_loss"],
            "take_profit": sl_tp["take_profit"],
            "risk_reward_ratio": sl_tp["risk_reward_ratio"],
            "sl_distance_pct": sl_tp["sl_distance_pct"],
            "tp_distance_pct": sl_tp["tp_distance_pct"],
            "atr": float(atr_val),
            "atr_pct": float(atr_pct * 100),
            "signals": [s.to_dict() for s in signals],
            "support_resistance": {
                "supports": sr.get("supports", []),
                "resistances": sr.get("resistances", []),
                "nearest_support": sr.get("nearest_support"),
                "nearest_resistance": sr.get("nearest_resistance"),
            },
        }

    def filter_signals(self, recommendations: List[Dict],
                       min_confidence: float = 60,
                       min_expected_rise: float = 1.0,
                       direction: str = "bullish") -> List[Dict]:
        """
        Filter and rank recommendations by criteria.
        Uses MIN_RR_RATIO from settings (no more hardcoded values).
        Ranking = confidence + bonus for R/R (capped at 3) so that a slightly
        lower-confidence signal with an excellent R/R can outrank a weaker
        setup — this directly improves expected yield per trade.
        """
        from config.settings import settings

        filtered = [
            r for r in recommendations
            if r.get("direction") == direction
            and r.get("confidence", 0) >= min_confidence
            and r.get("expected_rise_pct", 0) >= min_expected_rise
            and r.get("risk_reward_ratio", 0) >= settings.MIN_RR_RATIO
        ]
        # Composite rank: confidence + up to +5 for great R/R
        filtered.sort(
            key=lambda r: r.get("confidence", 0) + 5.0 * min(r.get("risk_reward_ratio", 0), 3.0) / 3.0,
            reverse=True,
        )
        return filtered


# Singleton
scorer = SignalScorer()
