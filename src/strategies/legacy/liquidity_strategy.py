"""
Liquidity Strategy
Order book imbalance + support/resistance levels.
"""
import pandas as pd
from typing import Dict, Optional
from src.indicators import liquidity as liq
from src.strategies.base import BaseStrategy, Signal


class LiquidityStrategy(BaseStrategy):
    name = "liquidity_analysis"
    weight = 1.0

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if order_book is None:
            return self._neutral("No order book data")

        ob_metrics = liq.analyze_order_book(order_book)
        if not ob_metrics:
            return self._neutral("Order book analysis failed")

        sr = liq.find_support_resistance(df, lookback=50)
        current_price = sr.get("current_price") or df["close"].iloc[-1]
        nearest_support = sr.get("nearest_support")
        nearest_resistance = sr.get("nearest_resistance")

        score = 0.0
        reasons = []
        details = {
            "imbalance": ob_metrics["imbalance"],
            "bias": ob_metrics["bias"],
            "bid_pct": ob_metrics["bid_pct"],
            "ask_pct": ob_metrics["ask_pct"],
            "spread_pct": ob_metrics["spread_pct"],
            "bid_wall_price": ob_metrics["bid_wall_price"],
            "bid_wall_qty": ob_metrics["bid_wall_qty"],
            "ask_wall_price": ob_metrics["ask_wall_price"],
            "ask_wall_qty": ob_metrics["ask_wall_qty"],
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
        }

        # ============ Order Book Imbalance ============
        imb = ob_metrics["imbalance"]
        if imb > 0.3:
            score += 25
            reasons.append(f"Heavy buy walls (imbalance +{imb:.2f})")
        elif imb > 0.15:
            score += 15
            reasons.append(f"Moderate buy pressure (imbalance +{imb:.2f})")
        elif imb < -0.3:
            score -= 25
            reasons.append(f"Heavy sell walls (imbalance {imb:.2f})")
        elif imb < -0.15:
            score -= 15
            reasons.append(f"Moderate sell pressure (imbalance {imb:.2f})")

        # ============ Bid vs Ask Walls ============
        bid_wall_strength = ob_metrics["bid_wall_qty"]
        ask_wall_strength = ob_metrics["ask_wall_qty"]
        if bid_wall_strength > ask_wall_strength * 2:
            score += 10
            reasons.append("Bid wall 2x larger than ask wall")
        elif ask_wall_strength > bid_wall_strength * 2:
            score -= 10
            reasons.append("Ask wall 2x larger than bid wall")

        # ============ Support/Resistance proximity ============
        if nearest_support and nearest_support > 0:
            dist_to_support = (current_price - nearest_support) / current_price * 100
            details["dist_to_support_pct"] = float(dist_to_support)
            if dist_to_support < 2:
                score += 15
                reasons.append(f"Close to support ({dist_to_support:.2f}%)")
            elif dist_to_support > 8:
                score -= 5
                reasons.append(f"Far from support ({dist_to_support:.2f}%)")

        if nearest_resistance and nearest_resistance > 0:
            dist_to_resistance = (nearest_resistance - current_price) / current_price * 100
            details["dist_to_resistance_pct"] = float(dist_to_resistance)
            if dist_to_resistance < 2:
                score -= 15
                reasons.append(f"Close to resistance ({dist_to_resistance:.2f}%)")
            elif dist_to_resistance > 8:
                score += 5
                reasons.append(f"Far from resistance ({dist_to_resistance:.2f}%)")

        # ============ Wall location (price-action interaction) ============
        bid_wall_price = ob_metrics["bid_wall_price"]
        ask_wall_price = ob_metrics["ask_wall_price"]
        # If bid wall is just below current price = strong support
        if bid_wall_price < current_price and \
           (current_price - bid_wall_price) / current_price < 0.01:
            score += 10
            reasons.append("Strong bid wall near current price")

        score = max(-100, min(100, score))

        if score > 15:
            return self._bull(score, reasons, details)
        elif score < -15:
            return self._bear(abs(score), reasons, details)
        return self._neutral(f"Neutral liquidity ({score:+.1f})", details)
