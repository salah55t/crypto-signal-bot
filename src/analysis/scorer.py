"""
Multi-Strategy Signal Scorer
Aggregates signals from all strategies into a single confidence score (0-100)
and computes the expected rise %, entry price, stop loss, and take profit.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
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

    def __init__(self):
        self.strategies = [
            # 3 powerful composite strategies (clean & focused)
            TrendPullbackStrategy(weight=2.0),                # buy strength on pullbacks
            LiquiditySweepReversalStrategy(weight=2.0),       # join stop hunts reversal
            VolatilityBreakoutStrategy(weight=1.8),          # squeeze breakout with volume
        ]
        log.info(
            f"[cyan]SignalScorer[/] initialized with {len(self.strategies)} composite strategies "
            f"(clean code, smart trader logic)"
        )

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

        # Compute weighted score
        total_weight = sum(s_w.weight for s_w in self.strategies)
        weighted_score = sum(
            sig.score * strat.weight
            for sig, strat in zip(signals, self.strategies)
        ) / total_weight

        # Confidence 0-100
        confidence = (weighted_score + 100) / 2  # convert -100..100 -> 0..100
        confidence = max(0, min(100, confidence))

        # Direction
        # Lowered threshold: was >15, now >5 (much less strict)
        if weighted_score > 5:
            direction = "bullish"
        elif weighted_score < -5:
            direction = "bearish"
        else:
            direction = "neutral"

        # Compute expected rise (using ATR + score)
        atr_val = ta.atr(df["high"], df["low"], df["close"], 14).iloc[-1]
        current_price = float(df["close"].iloc[-1])
        atr_pct = atr_val / current_price if current_price > 0 else 0

        # Bullish expectation: higher score + higher ATR = larger expected move
        expected_rise_pct = max(0, weighted_score / 100) * atr_pct * 100 * 2

        # Compute SL/TP using ATR
        sr = find_support_resistance(df, lookback=50)
        nearest_support = sr.get("nearest_support") or (current_price * (1 - atr_pct * 1.5))
        nearest_resistance = sr.get("nearest_resistance") or (current_price * (1 + atr_pct * 1.5))

        # For bullish: SL below nearest support (or 1.5 ATR), TP at nearest resistance or based on R/R
        sl_distance = atr_val * 1.5
        tp_distance = atr_val * 3.0  # default R/R = 2:1

        if direction == "bullish":
            stop_loss = min(nearest_support, current_price - sl_distance)
            take_profit = max(nearest_resistance, current_price + tp_distance)
            take_profit = max(take_profit, current_price * (1 + expected_rise_pct / 100))
        elif direction == "bearish":
            stop_loss = max(sr.get("nearest_resistance") or current_price,
                            current_price + sl_distance)
            take_profit = min(sr.get("nearest_support") or current_price,
                              current_price - tp_distance)
        else:
            stop_loss = current_price - sl_distance
            take_profit = current_price + tp_distance

        rr_ratio = abs(take_profit - current_price) / abs(current_price - stop_loss) \
            if abs(current_price - stop_loss) > 0 else 0

        return {
            "symbol": symbol,
            "direction": direction,
            "weighted_score": float(weighted_score),
            "confidence": float(confidence),
            "current_price": float(current_price),
            "expected_rise_pct": float(expected_rise_pct),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "risk_reward_ratio": float(rr_ratio),
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
                       min_confidence: float = 70,
                       min_expected_rise: float = 3.0,
                       direction: str = "bullish") -> List[Dict]:
        """
        Filter and sort recommendations by criteria.
        Returns only bullish signals with confidence >= min_confidence
        and expected_rise >= min_expected_rise.
        """
        filtered = [
            r for r in recommendations
            if r.get("direction") == direction
            and r.get("confidence", 0) >= min_confidence
            and r.get("expected_rise_pct", 0) >= min_expected_rise
            and r.get("risk_reward_ratio", 0) >= 1.5
        ]
        # Sort by confidence desc
        filtered.sort(key=lambda r: r.get("confidence", 0), reverse=True)
        return filtered


# Singleton
scorer = SignalScorer()
