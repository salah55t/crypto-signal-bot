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
from src.indicators.fibonacci import compute_fibonacci_levels, compute_entry_exit
from src.indicators.ichimoku import ichimoku_state
from src.indicators.elliott import detect_elliott_wave
from src.analysis.confluence import confluence_engine
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
    # Entry / SL / TP (Fibonacci + Support/Resistance aware, v3)
    # ------------------------------------------------------------------
    def _compute_sl_tp(self, df: pd.DataFrame, direction: str,
                       current_price: float, atr_val: float,
                       sr: Dict, fib: Optional[Dict] = None) -> Dict:
        """Fibonacci + S/R entry and exit points (v3).

        Entry:
          - UP impulse: golden pocket (0.5-0.618 retracement), prefer
            Fib x support confluence levels; "market" when price is
            already inside the pocket, "limit" otherwise.
          - DOWN impulse: market entry on the reversal bounce, with the
            shallow retracement pocket (0.236-0.382) as the pullback zone.
        SL: below/above the 10-bar swing structure + 0.4 ATR buffer,
        clamped to [0.9, 2.2] x ATR (v2 risk model preserved).
        TP1: first Fib extension / retracement / S/R level that satisfies
        the minimum R/R ratio. TP2: next structure level beyond TP1.
        Falls back to pure structure logic when no Fibonacci map exists
        (flat market, neutral direction).
        """
        min_rr = max(1.2, settings.MIN_RR_RATIO)

        low = df["low"]
        high = df["high"]
        swing_low = float(low.iloc[-10:].min())
        swing_high = float(high.iloc[-10:].max())

        return compute_entry_exit(
            direction=direction,
            current_price=current_price,
            atr_val=atr_val,
            sr=sr,
            fib=fib or {},
            swing_low=swing_low,
            swing_high=swing_high,
            min_rr=min_rr,
        )

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
        # v3: Fibonacci map (impulse, retracements, extensions, golden zone)
        # combined with S/R for entry / exit points.
        fib = compute_fibonacci_levels(df, lookback=100)
        sl_tp = self._compute_sl_tp(df, direction, current_price, atr_val, sr,
                                     fib=fib)

        # v4: Ichimoku regime gate + Elliott cycle timing + integrated
        # confluence rules. The Ichimoku gate can VETO the signal; the
        # Elliott layer adjusts confidence and may promote its wave-5
        # projection into TP2.
        icho = ichimoku_state(df)
        elliott = detect_elliott_wave(df)
        min_rr = max(1.2, settings.MIN_RR_RATIO)
        decision = confluence_engine.evaluate(
            direction=direction,
            confidence=verdict["confidence"],
            ichimoku=icho,
            elliott=elliott,
            fib=fib,
            sl_tp=sl_tp,
            current_price=current_price,
            min_rr=min_rr,
            veto_enabled=settings.CONFLUENCE_VETO_ENABLED,
            strategy_confluence=verdict.get("confluence", 0.0),
        )

        # v5 veteran filters: chaotic candles + dead markets are skipped
        # BEFORE they can become bad trades (ATR% of price, primary TF).
        atr_pct_total = atr_pct * 100
        volatility_extreme = atr_pct_total > settings.ATR_PCT_MAX
        dead_market = atr_pct_total < settings.ATR_PCT_MIN

        return {
            "symbol": symbol,
            "direction": direction,
            "weighted_score": verdict["weighted_score"],
            "confidence": decision["confidence"],
            "base_confidence": decision["base_confidence"],
            # v4 rule: boosts may RANK a signal higher but can never ADMIT it
            # past the quality threshold - only its base merit can. Penalties
            # (chop discount, late-cycle, counter-trend) CAN demote it out.
            "admission_confidence": float(
                min(decision["base_confidence"], decision["confidence"])
            ),
            # v5: layered harmony (regime+cycle+zone+strategies, 0..1)
            "harmony": decision.get("harmony", 0.0),
            "a_plus": decision.get("a_plus", False),
            "volatility_extreme": bool(volatility_extreme),
            "dead_market": bool(dead_market),
            "atr_pct_total": float(atr_pct_total),
            "avg_strength": verdict["avg_strength"],
            "confluence": verdict["confluence"],
            "current_price": current_price,
            "expected_rise_pct": expected_rise_pct,
            "entry_price": sl_tp["entry_price"],
            "entry_type": sl_tp["entry_type"],
            "entry_zone": sl_tp["entry_zone"],
            "entry_label": sl_tp.get("entry_label", ""),
            "stop_loss": sl_tp["stop_loss"],
            "take_profit": sl_tp["take_profit"],
            "take_profit_2": sl_tp["take_profit_2"],
            "tp2_rr": sl_tp.get("tp2_rr", 0.0),
            "tp1_label": sl_tp.get("tp1_label", ""),
            "tp2_label": sl_tp.get("tp2_label", ""),
            "risk_reward_ratio": sl_tp["risk_reward_ratio"],
            "sl_distance_pct": sl_tp["sl_distance_pct"],
            "tp_distance_pct": sl_tp["tp_distance_pct"],
            "tp2_distance_pct": sl_tp.get("tp2_distance_pct", 0.0),
            "fib_notes": sl_tp.get("fib_notes", []),
            "atr": float(atr_val),
            "atr_pct": float(atr_pct * 100),
            "signals": [s.to_dict() for s in signals],
            "fibonacci": {
                "impulse": fib.get("impulse"),
                "swing_high": fib.get("swing_high"),
                "swing_low": fib.get("swing_low"),
                "golden_zone": fib.get("golden_zone"),
                "retracements": fib.get("retracements", {}),
                "extensions": fib.get("extensions", {}),
            },
            "support_resistance": {
                "supports": sr.get("supports", []),
                "resistances": sr.get("resistances", []),
                "nearest_support": sr.get("nearest_support"),
                "nearest_resistance": sr.get("nearest_resistance"),
            },
            "ichimoku": icho or {},
            "elliott": elliott or {},
            "decision": {
                "vetoed": decision["vetoed"],
                "veto_reason": decision["veto_reason"],
                "adjustments": decision["adjustments"],
                "a_plus": decision["a_plus"],
                "base_confidence": decision["base_confidence"],
            },
        }

    def filter_signals(self, recommendations: List[Dict],
                       min_confidence: float = 60,
                       min_expected_rise: float = 1.0,
                       direction: str = "bullish") -> List[Dict]:
        """
        Filter and rank recommendations by criteria.
        Uses MIN_RR_RATIO from settings (no more hardcoded values).

        v5 veteran gates:
          - harmony >= MIN_HARMONY (layered agreement, not one loud layer)
          - volatility_extreme / dead_market symbols are dropped
        Ranking = confidence + R/R bonus (capped 5) + harmony bonus (capped 4)
        so complete setups outrank louder-but-lonesome signals.
        """
        from config.settings import settings

        filtered = [
            r for r in recommendations
            if r.get("direction") == direction
            and not r.get("decision", {}).get("vetoed", False)
            and r.get("admission_confidence",
                      r.get("confidence", 0)) >= min_confidence
            and r.get("expected_rise_pct", 0) >= min_expected_rise
            and r.get("risk_reward_ratio", 0) >= settings.MIN_RR_RATIO
            and r.get("harmony", 0.0) >= settings.MIN_HARMONY
            and not (settings.EXCLUDE_VOLATILITY_EXTREME
                     and r.get("volatility_extreme", False))
            and not r.get("dead_market", False)
        ]
        # Composite rank: confidence + up to +5 for great R/R + up to +4 harmony
        filtered.sort(
            key=lambda r: (
                r.get("confidence", 0)
                + 5.0 * min(r.get("risk_reward_ratio", 0), 3.0) / 3.0
                + 4.0 * min(r.get("harmony", 0.0), 1.0)
            ),
            reverse=True,
        )
        return filtered


# Singleton
scorer = SignalScorer()
