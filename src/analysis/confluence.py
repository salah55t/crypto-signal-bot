"""
Confluence Nexus (v4) — Integrated Decision Layer

The single place where ALL layers meet. Each component has ONE role:

  Ichimoku   -> REGIME GATE   : never trade against the cloud; discount
                                signals when price is inside it.
  Elliott    -> CYCLE TIMING  : wave 3 = strongest phase (boost),
                                wave 4 = good R/R zone with a wave-5
                                projection target, mature wave 5 /
                                completed impulse = correction risk
                                (penalty), C complete = reversal setup.
  Fibonacci  -> ENTRY ZONES   : golden-pocket entries + extension
                                targets (v3 engine, unchanged).
  S/R        -> STRUCTURE     : zone confluence validation (v3).
  Strategies -> TRIGGER       : direction + strength votes (v2 model).

Decision rules:
  1. Ichimoku regime OPPOSES the signal direction      -> VETO (signal dies)
  2. Ichimoku regime ALIGNED                            -> confidence x1.10
  3. Price inside the cloud (neutral regime)            -> confidence x0.92
  4. Elliott favorable (wave 3 / wave 4 / C complete)   -> +8 / +6 / +6
  5. Elliott hostile (mature wave 5, completed impulse,
     counter-trend cycle)                               -> -10 .. -12
  6. A+ SETUP: regime aligned + elliott favorable +
     entry in the Fibonacci golden pocket + Fib x S/R
     confluence on entry                                -> +8 extra & flag

v5 HARMONY SCORE (0..1) — "would a veteran take this trade?":
  Weighted agreement across ALL layers:
    regime alignment            0.35
    elliott cycle favorability  0.25
    entry zone quality          0.20
    strategy confluence x 0.20
  The harmony score becomes a separate ADMISSION gate (MIN_HARMONY) and a
  ranking factor: a veteran demands LAYERED agreement, not one loud layer.

The adjusted confidence replaces the base confidence in the final
recommendation; vetoed signals are excluded from filtering, Telegram
and position management.
"""
from typing import Dict, Optional

# --- Ichimoku regime adjustments ---
REGIME_ALIGNED_MULT = 1.10   # regime agrees with signal direction
REGIME_NEUTRAL_MULT = 0.92   # price inside the cloud (chop discount)
REGIME_OPPOSING_MULT = 0.85  # applied instead of a veto when veto is disabled

# --- Elliott adjustments (confidence points) ---
ELLIOTT_WAVE3_BONUS = 8.0
ELLIOTT_WAVE4_BONUS = 6.0
ELLIOTT_ABC_BONUS = 6.0
ELLIOTT_CORRECTION_WAIT_PENALTY = -6.0
ELLIOTT_LATE_PENALTY = -12.0     # mature wave 5 / completed impulse
ELLIOTT_COUNTER_PENALTY = -10.0  # trading against the dominant cycle

# --- A+ setup bonus ---
A_PLUS_BONUS = 8.0
A_PLUS_MIN_ELLIOTT_CONF = 0.6
A_PLUS_MIN_HARMONY = 0.75

# --- v5 harmony weights (must sum to ~1.0) ---
H_REGIME_ALIGNED = 0.35
H_REGIME_NEUTRAL = 0.15
H_ELLIOTT_FAVORED = 0.25
H_ELLIOTT_WAVE4 = 0.20
H_ELLIOTT_UNCLEAR = 0.05
H_ZONE_QUALITY = 0.20
H_ZONE_DECENT = 0.10
H_STRATEGY_CONFLUENCE = 0.20


def _pattern_aligned(direction: str, pattern: str) -> bool:
    """True when a wave3 / partial-impulse pattern matches the signal direction."""
    if direction == "bullish":
        return pattern in ("wave3_up", "partial_impulse_up")
    if direction == "bearish":
        return pattern in ("wave3_down", "partial_impulse_down")
    return False


class ConfluenceEngine:
    """Applies the integrated v4 decision rules to one recommendation."""

    def evaluate(self, direction: str, confidence: float,
                 ichimoku: Optional[Dict], elliott: Optional[Dict],
                 fib: Optional[Dict], sl_tp: Dict,
                 current_price: float, min_rr: float = 1.5,
                 veto_enabled: bool = True,
                 strategy_confluence: float = 0.0) -> Dict:
        """
        Returns:
          base_confidence   original strategy-model confidence
          confidence        adjusted confidence (clamped 0-100)
          vetoed            True when the Ichimoku gate kills the signal
          veto_reason       why the signal was vetoed
          adjustments       human-readable list of applied adjustments
          a_plus            True when the setup qualifies as A+
          harmony           v5 layered-agreement score 0..1
          elliott_target    wave-5 projection promoted to TP2 (or None)
          elliott_target_label
        """
        base = float(confidence)
        adjusted = base
        adjustments: list = []
        vetoed = False
        veto_reason = None
        a_plus = False
        ell_target = None
        ell_target_label = None
        harmony = 0.0

        elliott = elliott or {}
        fib = fib or {}
        pattern = elliott.get("pattern", "unclear")
        wave = elliott.get("current_wave")
        ell_conf = float(elliott.get("wave_confidence", 0.0))

        # ---------------------------------------------------------
        # 0) HARMONY LAYER TRACKING (v5)
        # ---------------------------------------------------------
        regime_aligned = False
        regime_neutral = False
        elliott_favored = 0.0   # points 0..0.25
        zone_quality = 0.0      # points 0..0.20

        # ---------------------------------------------------------
        # 1) ICHIMOKU REGIME GATE
        # ---------------------------------------------------------
        if ichimoku:
            regime = ichimoku.get("regime")
            if regime == direction:
                adjusted *= REGIME_ALIGNED_MULT
                regime_aligned = True
                adjustments.append(
                    f"Ichimoku regime aligned ({regime}) "
                    f"x{REGIME_ALIGNED_MULT:.2f}"
                )
            elif regime == "neutral":
                adjusted *= REGIME_NEUTRAL_MULT
                regime_neutral = True
                adjustments.append(
                    f"Price inside Ichimoku cloud (chop) "
                    f"x{REGIME_NEUTRAL_MULT:.2f}"
                )
            elif veto_enabled:
                vetoed = True
                veto_reason = (
                    f"Ichimoku regime opposes signal "
                    f"({regime} vs {direction})"
                )
                adjustments.append(f"VETO: {veto_reason}")
            else:
                adjusted *= REGIME_OPPOSING_MULT
                adjustments.append(
                    f"Ichimoku regime opposes ({regime}) "
                    f"x{REGIME_OPPOSING_MULT:.2f}"
                )

        # ---------------------------------------------------------
        # 2) ELLIOTT CYCLE TIMING
        # ---------------------------------------------------------
        if not vetoed and pattern != "unclear":
            if pattern in ("wave3_up", "wave3_down"):
                if _pattern_aligned(direction, pattern):
                    adjusted += ELLIOTT_WAVE3_BONUS
                    elliott_favored = H_ELLIOTT_FAVORED
                    adjustments.append(
                        f"Elliott wave 3 in progress +"
                        f"{ELLIOTT_WAVE3_BONUS:.0f} (strongest phase)"
                    )
            elif pattern in ("partial_impulse_up", "partial_impulse_down"):
                if _pattern_aligned(direction, pattern) and wave == "4":
                    adjusted += ELLIOTT_WAVE4_BONUS
                    elliott_favored = H_ELLIOTT_WAVE4
                    adjustments.append(
                        f"Elliott wave 4 pullback +"
                        f"{ELLIOTT_WAVE4_BONUS:.0f} (wave 5 ahead)"
                    )
                elif _pattern_aligned(direction, pattern) and wave == "5":
                    maturity = elliott.get("maturity") or 0.0
                    if maturity >= 1.0:
                        adjusted += ELLIOTT_LATE_PENALTY
                        adjustments.append(
                            f"Elliott wave 5 mature "
                            f"{ELLIOTT_LATE_PENALTY:.0f} (correction risk)"
                        )
                    else:
                        adjustments.append(
                            "Elliott wave 5 in progress (final leg, no boost)"
                        )
            elif pattern == "impulse_up" or pattern == "impulse_down":
                adjusted += ELLIOTT_LATE_PENALTY
                adjustments.append(
                    f"Elliott impulse complete "
                    f"{ELLIOTT_LATE_PENALTY:.0f} (correction expected)"
                )
            elif pattern == "correction_after_up":
                # bullish reversal setup when C is complete
                if direction == "bullish" and wave == "C (complete)":
                    adjusted += ELLIOTT_ABC_BONUS
                    elliott_favored = H_ELLIOTT_FAVORED
                    adjustments.append(
                        f"ABC correction complete +"
                        f"{ELLIOTT_ABC_BONUS:.0f} (new cycle starting)"
                    )
                elif direction == "bullish":
                    adjusted += ELLIOTT_CORRECTION_WAIT_PENALTY
                    adjustments.append(
                        f"ABC correction in progress "
                        f"{ELLIOTT_CORRECTION_WAIT_PENALTY:.0f} (wait)"
                    )
                elif direction == "bearish" and wave == "C (complete)":
                    adjusted += ELLIOTT_LATE_PENALTY
                    adjustments.append(
                        f"ABC bounce complete "
                        f"{ELLIOTT_LATE_PENALTY:.0f} (down impulse next)"
                    )
            elif pattern == "correction_after_down":
                if direction == "bearish" and wave == "C (complete)":
                    adjusted += ELLIOTT_ABC_BONUS
                    elliott_favored = H_ELLIOTT_FAVORED
                    adjustments.append(
                        f"ABC bounce complete +"
                        f"{ELLIOTT_ABC_BONUS:.0f} (down cycle resuming)"
                    )
                elif direction == "bearish":
                    adjusted += ELLIOTT_CORRECTION_WAIT_PENALTY
                    adjustments.append(
                        f"ABC bounce in progress "
                        f"{ELLIOTT_CORRECTION_WAIT_PENALTY:.0f} (wait)"
                    )
                elif direction == "bullish" and wave == "C (complete)":
                    adjusted += ELLIOTT_LATE_PENALTY
                    adjustments.append(
                        f"Corrective bounce complete "
                        f"{ELLIOTT_LATE_PENALTY:.0f} (down impulse next)"
                    )

            # counter-trend: signal direction against the detected cycle
            counter = (
                (direction == "bullish" and
                 pattern in ("impulse_down", "wave3_down",
                             "partial_impulse_down"))
                or
                (direction == "bearish" and
                 pattern in ("impulse_up", "wave3_up",
                             "partial_impulse_up"))
            )
            if counter:
                adjusted += ELLIOTT_COUNTER_PENALTY
                adjustments.append(
                    f"Counter-cycle signal {ELLIOTT_COUNTER_PENALTY:.0f} "
                    f"({pattern})"
                )

            # Wave-5 projection as an extended TP2 candidate
            proj = elliott.get("projection")
            if proj and _pattern_aligned(direction, pattern) and wave == "4":
                ell_target = float(proj)
                ell_target_label = "Elliott wave 5 projection"

        # ---------------------------------------------------------
        # 3) A+ SETUP DETECTION (v5: also requires layered harmony)
        # ---------------------------------------------------------
        if not vetoed and ichimoku and pattern != "unclear":
            regime_aligned_a = ichimoku.get("regime") == direction
            elliott_ok = (
                ell_conf >= A_PLUS_MIN_ELLIOTT_CONF
                and (
                    (direction == "bullish" and
                     (pattern == "wave3_up"
                      or (pattern == "partial_impulse_up" and wave == "4")
                      or (pattern == "correction_after_up"
                          and wave == "C (complete)")))
                    or
                    (direction == "bearish" and
                     (pattern == "wave3_down"
                      or (pattern == "partial_impulse_down" and wave == "4")
                      or (pattern == "correction_after_down"
                          and wave == "C (complete)")))
                )
            )
            golden_ok = self._entry_in_golden_pocket(sl_tp, fib)
            sr_ok = "confluence" in (sl_tp.get("entry_label", "") or "")

            if regime_aligned_a and elliott_ok and (golden_ok or sr_ok):
                adjusted += A_PLUS_BONUS
                a_plus = True
                adjustments.append(
                    f"A+ SETUP +{A_PLUS_BONUS:.0f} "
                    f"(regime + cycle + zones all aligned)"
                )

        # ---------------------------------------------------------
        # 3b) HARMONY SCORE (v5) — layered agreement 0..1
        # ---------------------------------------------------------
        if not vetoed:
            zone_quality = (
                H_ZONE_QUALITY
                if (self._entry_in_golden_pocket(sl_tp, fib)
                    or "confluence" in (sl_tp.get("entry_label", "") or ""))
                else (H_ZONE_DECENT if sl_tp.get("entry_zone") else 0.0)
            )
            if pattern == "unclear":
                elliott_harmony = H_ELLIOTT_UNCLEAR
            else:
                elliott_harmony = elliott_favored
            harmony = (
                (H_REGIME_ALIGNED if regime_aligned
                 else (H_REGIME_NEUTRAL if regime_neutral else 0.0))
                + elliott_harmony
                + zone_quality
                + H_STRATEGY_CONFLUENCE * max(0.0, min(1.0, strategy_confluence))
            )
            harmony = max(0.0, min(1.0, harmony))
            if a_plus and harmony < A_PLUS_MIN_HARMONY:
                # An A+ label with weak layered agreement is an illusion:
                # keep the bonus out of admission decisions via harmony gate.
                adjustments.append(
                    f"Harmony {harmony:.2f} below A+ bar "
                    f"{A_PLUS_MIN_HARMONY:.2f} (setup is 2-layer, not 3-layer)"
                )

        # ---------------------------------------------------------
        # 4) Promote the Elliott projection to TP2 when it extends it
        # ---------------------------------------------------------
        if ell_target is not None and not vetoed:
            sl_tp_adj = self._apply_elliott_target(sl_tp, ell_target,
                                                   ell_target_label,
                                                   direction)
            if sl_tp_adj:
                adjustments.append(
                    "TP2 promoted to Elliott wave 5 projection"
                )

        adjusted = max(0.0, min(100.0, adjusted))

        return {
            "base_confidence": base,
            "confidence": float(adjusted),
            "vetoed": vetoed,
            "veto_reason": veto_reason,
            "adjustments": adjustments,
            "a_plus": a_plus,
            "harmony": float(harmony),
            "elliott_target": ell_target,
            "elliott_target_label": ell_target_label,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _entry_in_golden_pocket(sl_tp: Dict, fib: Dict) -> bool:
        """True when the entry zone overlaps the Fibonacci golden pocket."""
        gz = fib.get("golden_zone") or {}
        zone = sl_tp.get("entry_zone") or {}
        if not gz or not zone:
            return False
        gz_low, gz_high = gz.get("low"), gz.get("high")
        z_low, z_high = zone.get("low"), zone.get("high")
        if None in (gz_low, gz_high, z_low, z_high):
            return False
        return z_low <= gz_high and z_high >= gz_low

    @staticmethod
    def _apply_elliott_target(sl_tp: Dict, target: float, label: str,
                              direction: str) -> bool:
        """
        Replace TP2 with the Elliott projection when it EXTENDS the current
        TP2 (never shrinks it) and stays within a sane 8x SL-distance cap.
        Mutates sl_tp in place. Returns True when applied.
        """
        tp1 = sl_tp.get("take_profit")
        tp2 = sl_tp.get("take_profit_2")
        sl = sl_tp.get("stop_loss")
        entry = sl_tp.get("entry_price")
        if not all(isinstance(v, (int, float)) for v in
                   (tp1, tp2, sl, entry)) or entry is None:
            return False

        sl_dist = abs(entry - sl)
        if sl_dist <= 0:
            return False

        if direction == "bullish":
            if target > tp2 and target <= entry + 8.0 * sl_dist \
                    and target > tp1:
                sl_tp["take_profit_2"] = float(target)
                sl_tp["tp2_label"] = label
                sl_tp["tp2_rr"] = float((target - entry) / sl_dist)
                sl_tp["tp2_distance_pct"] = float(
                    (target - entry) / entry * 100)
                return True
        else:
            if target < tp2 and target >= entry - 8.0 * sl_dist \
                    and target < tp1:
                sl_tp["take_profit_2"] = float(target)
                sl_tp["tp2_label"] = label
                sl_tp["tp2_rr"] = float((entry - target) / sl_dist)
                sl_tp["tp2_distance_pct"] = float(
                    (entry - target) / entry * 100)
                return True
        return False


# Singleton
confluence_engine = ConfluenceEngine()
