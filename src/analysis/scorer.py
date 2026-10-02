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
    # Original trio (calibration reference stack)
    TrendPullbackStrategy,
    LiquiditySweepReversalStrategy,
    VolatilityBreakoutStrategy,
    # v5.7 Signal Stack trio (user-specified)
    TripleConfluenceTrendStrategy,
    BBMeanReversionStrategy,
    MACDBreakoutStrategy,
    Signal
)
from src.indicators import technical as ta
from src.indicators.liquidity import find_support_resistance
from src.indicators.fibonacci import compute_fibonacci_levels, compute_entry_exit
from src.indicators.ichimoku import ichimoku_state
from src.indicators.elliott import detect_elliott_wave
from src.analysis.confluence import confluence_engine
from src.analysis.session_clock import entry_gate, session_info, \
    strategy_session_weight
from src.utils.logger import log


class SignalScorer:
    """Combines signals from multiple strategies into a final recommendation."""

    # Score threshold above which a strategy vote counts towards direction
    VOTE_THRESHOLD = 15.0
    # Net weighted score needed to call the market bullish / bearish
    DIRECTION_THRESHOLD = 8.0

    def __init__(self):
        # v5.28: every composite is settings-gated now. The defaults are
        # EVIDENCE-BASED (scripts/research/ baseline on 39 top-volume pairs
        # x ~500 days of real 4h data, production exit lifecycle, 0.24%
        # round-trip costs, 4 time-folds):
        #   volatility_breakout  +18.9 bps/trade, PF 1.58, all folds > 0 -> ON
        #   trend_pullback       -0.5 bps/trade, PF 0.99, folds mixed -> OFF
        #   liquidity_sweep      -6.9 bps/trade, PF 0.81, ALL folds < 0 -> OFF
        #   triple_confluence    +6.9 bps, macd_breakout +9.5, bb_mean_rev
        #                        +6.4 (thin but positive) -> ON (v5.7 flags)
        self.strategies = []
        if settings.STRATEGY_TREND_PULLBACK_ENABLED:
            self.strategies.append(TrendPullbackStrategy(weight=2.0))
        if settings.STRATEGY_LIQUIDITY_SWEEP_ENABLED:
            self.strategies.append(LiquiditySweepReversalStrategy(weight=2.0))
        if settings.STRATEGY_VOL_BREAKOUT_ENABLED:
            self.strategies.append(VolatilityBreakoutStrategy(weight=1.8))
        # v5.7 Signal Stack trio (user-specified, settings-gated)
        if settings.STRATEGY_TRIPLE_TREND_ENABLED:
            self.strategies.append(TripleConfluenceTrendStrategy(
                weight=settings.STRATEGY_TRIPLE_TREND_WEIGHT))
        if settings.STRATEGY_BB_MEAN_REV_ENABLED:
            self.strategies.append(BBMeanReversionStrategy(
                weight=settings.STRATEGY_BB_MEAN_REV_WEIGHT))
        if settings.STRATEGY_MACD_BREAKOUT_ENABLED:
            self.strategies.append(MACDBreakoutStrategy(
                weight=settings.STRATEGY_MACD_BREAKOUT_WEIGHT))
        self.total_weight = sum(s.weight for s in self.strategies)
        log.info(
            f"[cyan]SignalScorer[/] initialized with {len(self.strategies)} composite strategies "
            f"(strength x confluence confidence model, total weight {self.total_weight:.1f})"
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
            # v5.7: confluence is measured against the CALIBRATED reference
            # stack weight (5.8), NOT the live total. Adding strategies must
            # never dilute the confidence scale (with 6 strategies the live
            # total is 10.0 and a lone signal would sink 64% -> 58%, making
            # MIN_CONFIDENCE=68 silently demand 3-of-6 agreement). The
            # original stack is bit-identical (its max agree_w == 5.8);
            # extra strategies only ADD confluence when they agree, capped
            # at 1.0 so the model can never exceed full unanimity.
            confluence = min(
                1.0, agree_w / max(settings.STRATEGY_CONFLUENCE_REF_WEIGHT, 1e-9)
            )
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

        v5.16: pegged/quasi-stable symbols are hard-skipped before ANY
        strategy runs (fees > movement = guaranteed slow bleed).
        """
        if symbol in set(getattr(settings, "PEGGED_SYMBOLS", []) or []):
            return {"symbol": symbol, "skip": True,
                    "reason": "pegged/quasi-stable symbol (pinned price)"}
        if df is None or len(df) < 60:
            return {"symbol": symbol, "skip": True, "reason": "Insufficient data"}

        # === v5.21: one multi-timeframe snapshot per symbol ===
        # Daily macro trend + 1h tactical momentum, shared by every
        # strategy (the professional's first filter). The daily fetch is
        # gated on the live WS feed so unit tests stay hermetic; bans are
        # handled inside mtf._daily (fail-open -> strategies unchanged).
        mtf_ctx = None
        try:
            from src.core.ws_feed import ws_feed as _ws
            if getattr(settings, "MTF_ENABLED", True) and _ws._started:
                from src.analysis.mtf import mtf_context
                n_tf = len(settings.TIMEFRAMES or ["4h"])
                primary_tf = (settings.TIMEFRAMES[n_tf // 2]
                              if n_tf > 1 else settings.TIMEFRAMES[0])
                mtf_ctx = mtf_context(symbol, multi_tf_data,
                                      primary_tf=primary_tf)
        except Exception:
            mtf_ctx = None

        signals: List[Signal] = []
        for strat in self.strategies:
            try:
                sig = strat.analyze(df, symbol, multi_tf_data, order_book,
                                    mtf_ctx=mtf_ctx)
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

        # === v5.21: MTF alignment confidence bonus ===
        # Professionals pay a premium for confluence that includes the
        # macro tide. Applied only when >= 2 strategies vote the final
        # direction AND the higher-timeframe trend agrees - a single loud
        # strategy still cannot buy its way past MIN_CONFIDENCE.
        try:
            if mtf_ctx and float(getattr(settings, "MTF_CONF_BONUS", 0)) > 0:
                from src.analysis.mtf import htf_agrees
                if htf_agrees(mtf_ctx, direction):
                    voters = [s for s in signals
                              if s.direction == direction
                              and abs(float(s.score)) >= self.VOTE_THRESHOLD]
                    if len(voters) >= 2:
                        verdict["confidence"] = min(
                            100.0,
                            verdict["confidence"]
                            + float(settings.MTF_CONF_BONUS))
                        verdict["mtf_bonus"] = float(
                            settings.MTF_CONF_BONUS)
        except Exception:
            pass

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
        dead_reason = "atr_floor" if dead_market else ""

        # v5.16: STATISTICAL FLATNESS - the "semi-stable" coin killer.
        # A pinned coin can pass a small ATR floor on a high timeframe;
        # it can never pass a rolling RANGE check + per-bar movement check.
        flat_market = False
        rng_pct = None
        try:
            lb = max(20, int(getattr(settings, "FLAT_LOOKBACK", 48)))
            tail = df.tail(lb)
            rng_pct = ((float(tail["high"].max())
                        - float(tail["low"].min()))
                       / max(current_price, 1e-12)) * 100.0
            tf_hint = ""
            try:
                # primary timeframe of this window (live passes settings
                # TIMEFRAMES; the index only gives bar spacing)
                tf_hint = str(settings.TIMEFRAMES[0]) if settings.TIMEFRAMES else ""
            except Exception:
                tf_hint = ""
            flat_floor = float(
                (settings.FLAT_RANGE_PCT_BY_TF or {}).get(
                    tf_hint, settings.FLAT_RANGE_PCT_DEFAULT))
            rets = tail["close"].pct_change().dropna()
            mean_abs_ret = float(rets.abs().mean()) * 100.0 \
                if len(rets) else 100.0
            if rng_pct < flat_floor and \
                    mean_abs_ret < float(getattr(
                        settings, "FLAT_RETURN_ABS_MIN", 0.05)):
                flat_market = True
                dead_market = True
                dead_reason = "flat/pinned"
        except Exception:
            pass

        # v5.16: SESSION CLOCK - the fixed daily rhythm (exchange opens/
        # closes). Tags every rec with its hour's session; hard-blocks NEW
        # entries in the data-backed losing windows. Exits are never gated.
        sinfo = session_info()
        try:
            if df.index is not None and len(df.index) > 0:
                bar_ts = df.index[-1]
                sinfo = session_info(getattr(bar_ts, "to_pydatetime",
                                             lambda: bar_ts)())
        except Exception:
            pass
        dom_sig = ""
        try:
            agreeing = [
                s for s in signals
                if s.direction == direction
                and abs(float(s.score)) >= self.VOTE_THRESHOLD
            ]
            if agreeing:
                dom_sig = max(agreeing,
                              key=lambda s: abs(float(s.score))).strategy
        except Exception:
            dom_sig = ""
        a_plus_flag = bool(decision.get("a_plus", False))
        blocked, block_reason = entry_gate(sinfo, dom_sig, a_plus_flag)
        sess_weight = strategy_session_weight(sinfo, dom_sig)

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
            "dead_market_reason": dead_reason,
            "flat_market": bool(flat_market),
            "range_pct": (round(float(rng_pct), 3)
                          if rng_pct is not None else None),
            "session": {
                "hour": sinfo.get("hour"),
                "session": sinfo.get("session"),
                "session_ar": sinfo.get("session_ar"),
                "is_weekend": sinfo.get("is_weekend"),
                "is_saturday": sinfo.get("is_saturday"),
            },
            "session_blocked": bool(blocked),
            "session_block_reason": block_reason,
            "session_weight": float(sess_weight),
            "dominant_strategy": dom_sig,
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
            # v5.21: multi-timeframe context snapshot (dashboard/governance)
            "mtf": mtf_ctx or {},
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
        v5.16: session-blocked recs (chop window / Saturday breakouts /
        Monday-open reversals) are dropped; ranking is scaled by the
        session weight so the right strategy owns the right hour.
        Ranking = (confidence + R/R bonus (capped 5) + harmony bonus (capped 4))
                  x session_weight
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
            and not r.get("session_blocked", False)
        ]
        # Composite rank: confidence + up to +5 for great R/R + up to +4
        # harmony, scaled by the v5.16 session weight (right hour, right
        # strategy).
        filtered.sort(
            key=lambda r: (
                (r.get("confidence", 0)
                 + 5.0 * min(r.get("risk_reward_ratio", 0), 3.0) / 3.0
                 + 4.0 * min(r.get("harmony", 0.0), 1.0))
                * float(r.get("session_weight", 1.0) or 1.0)
            ),
            reverse=True,
        )
        return filtered


# Singleton
scorer = SignalScorer()
