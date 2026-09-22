"""
Fibonacci Confluence + Order Book Strategy.

Finds coins where:
1. Price is in the Fibonacci "golden zone" (0.618 - 0.786 retracement)
2. Order book shows heavy bid walls (buy-side imbalance)

This is a PROPRIETARY strategy combining technical levels with order flow
to identify high-probability bounces at deep retracements.
"""
import pandas as pd
from typing import Dict, Optional
from src.indicators.proprietary import fibonacci_confluence
from src.indicators.liquidity import analyze_order_book
from src.strategies.base import BaseStrategy, Signal


class FibonacciConfluenceStrategy(BaseStrategy):
    name = "fibonacci_confluence"
    weight = 1.3

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < 100:
            return self._neutral("Insufficient data for Fibonacci")

        fib_result = fibonacci_confluence(df, lookback=100)
        fib_signal = fib_result.get("signal", "none")

        score = 0
        reasons = []
        details = {"fibonacci": fib_result}

        # Fibonacci signal
        if fib_signal == "bullish":
            score += fib_result.get("strength", 0.5) * 60
            reasons.append(fib_result.get("reason", ""))

        # Order book confluence
        if order_book:
            ob_metrics = analyze_order_book(order_book)
            imbalance = ob_metrics.get("imbalance", 0)
            bid_pct = ob_metrics.get("bid_pct", 0.5)
            details["order_book"] = {
                "imbalance": imbalance,
                "bid_pct": bid_pct,
                "bias": ob_metrics.get("bias"),
            }
            # Heavy buy walls + Fibonacci = strong confluence
            if imbalance > 0.2 and fib_signal == "bullish":
                score += 40
                reasons.append(f"Heavy buy walls (imbalance +{imbalance:.2f}) at Fibonacci zone")
            elif imbalance > 0.15:
                score += 15
                reasons.append(f"Moderate buy pressure (imbalance +{imbalance:.2f})")

        if score > 25:
            return self._bull(score, reasons, details)
        return self._neutral(f"No confluence (fib={fib_signal})", details)
