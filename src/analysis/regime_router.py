"""Regime Router - pick the right strategy for the current market state (v5.13).

User request: "review the trade-acceptance logic across different volatility
regimes; the bot should use the strategy appropriate for the market state.
Market states come from the major coins, the Fear & Greed index, and weekends."

FOUR COMPONENTS (all cheap):
  1. LEADERS   - market_map.run_market_cycle() weighted 0-100 score of
                 BTC/ETH/SOL/XRP groups (cached 60 min, WS candles = free).
  2. FEAR&GREED- alternative.me free API (no key), cached FNG_CACHE_HOURS
                 (the index updates daily server-side). Unavailable -> neutral.
  3. WEEKEND   - UTC Friday 22:00 -> Monday 00:00 (thin liquidity, fakeouts).
  4. VOLATILITY- BTC 1h ATR14% now vs its own median (WS cache, free):
                 ratio >= 2.6 crisis / >= 1.8 elevated / <= 0.55 dead-choppy.

ROUTING OUTPUT (data/regime_state.json + memory, refreshed every
REGIME_REFRESH_MIN) - a policy dict consumed by three gates:
  - analyzer.apply_regime_routing(): re-ranks candidates by how well their
    strategies fit the regime (the "right strategy for the state")
  - risk.validate_recommendation(): regime-adjusted MIN_CONFIDENCE / MIN_RR
  - sizing: paper + live notional x size_multiplier (0.25..1.25)
  - cycle.open_new_positions(): full stand-aside flag (crisis freeze)

Spot-only: every rule only ever limits or re-weights long entries; nothing
here can open shorts or leverage.
"""
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config.settings import settings
from src.utils.logger import log
from src.utils.helpers import load_json, save_json, now_utc

REGIME_FILE = Path("data/regime_state.json")
FNG_FILE = Path("data/fng_cache.json")

# exact strategy.name values (src/strategies/*)
TREND_STRATS = {"trend_pullback", "triple_confluence_trend"}
BREAKOUT_STRATS = {"volatility_breakout", "macd_breakout"}
MEANREV_STRATS = {"bb_mean_reversion"}
REVERSAL_STRATS = {"liquidity_sweep_reversal"}
ALL_STRATS = TREND_STRATS | BREAKOUT_STRATS | MEANREV_STRATS | REVERSAL_STRATS

FNG_URL = "https://api.alternative.me/fng/?limit=1"


def is_weekend(now: Optional[datetime] = None) -> bool:
    """Crypto weekend window: Fri 22:00 UTC -> Mon 00:00 UTC.

    Traditional desks are closed, retail dominates, liquidity thins and
    breakouts fake out more. Configurable kill-switch WEEKEND_FILTER.
    """
    if not settings.WEEKEND_FILTER:
        return False
    now = now or datetime.now(timezone.utc)
    wd = now.weekday()  # Mon=0 .. Sun=6
    if wd == 4 and now.hour >= 22:   # Friday evening (US close)
        return True
    if wd in (5, 6):                  # Saturday, Sunday
        return True
    return False


class RegimeRouter:
    """Classify the market state and route strategies/gates accordingly."""

    def __init__(self):
        self._regime: Optional[Dict] = None
        self._regime_ts: float = 0.0

    # ------------------------------------------------------------------
    # components
    # ------------------------------------------------------------------
    def _leaders(self) -> Dict:
        """Leaders weighted score/verdict from the market cycle (cached)."""
        try:
            from src.analysis.market_map import market_map
            cyc = market_map.run_market_cycle() or {}
            m = cyc.get("market") or {}
            return {
                "score": float(m.get("score", 50) or 50),
                "verdict": str(m.get("verdict", "neutral") or "neutral"),
                "posture_ar": str(m.get("posture_ar", "") or ""),
            }
        except Exception as e:
            log.debug(f"Regime router: leaders read failed: {e}")
            return {"score": 50.0, "verdict": "neutral", "posture_ar": ""}

    def _fear_greed(self) -> Dict:
        """Fear & Greed index (alternative.me, free, no key) - file cached."""
        cached = load_json(FNG_FILE, default=None)
        if cached:
            try:
                fetched = datetime.fromisoformat(cached["fetched_at"])
                if now_utc() - fetched < timedelta(
                        hours=max(0.5, settings.FNG_CACHE_HOURS)):
                    return cached
            except Exception:
                pass
        if not settings.FNG_ENABLED:
            return {"value": 50, "label": "neutral", "source": "disabled"}
        try:
            import requests
            r = requests.get(FNG_URL, timeout=8,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            row = (r.json().get("data") or [{}])[0]
            val = int(row.get("value", 50))
            label = str(row.get("value_classification", "Neutral"))
            out = {
                "value": val,
                "label": label.lower(),
                "source": "alternative.me",
                "fetched_at": now_utc().isoformat(),
            }
            save_json(out, FNG_FILE)
            log.info(f"[cyan]Fear & Greed[/] index: {val} ({label})")
            return out
        except Exception as e:
            log.debug(f"Fear & Greed fetch failed ({e}) - neutral 50")
            return {"value": 50, "label": "neutral", "source": "unavailable"}

    def _btc_vol(self) -> Dict:
        """BTC 1h ATR14% now vs its own median (WS cache first - free)."""
        try:
            from src.core.data_fetcher import data_fetcher
            from src.indicators.technical import atr as atr_ind
            df = data_fetcher.get_candles("BTCUSDT", "1h", 240)
            if df is None or len(df) < 60:
                return {"level": "unknown"}
            atr_s = atr_ind(df["high"].astype(float),
                            df["low"].astype(float),
                            df["close"].astype(float), 14)
            atr_pct = (atr_s / df["close"].astype(float) * 100.0).dropna()
            atr_pct = atr_pct.replace([np.inf, -np.inf], np.nan).dropna()
            if len(atr_pct) < 40:
                return {"level": "unknown"}
            cur = float(atr_pct.iloc[-1])
            med = float(atr_pct.median())
            ratio = (cur / med) if med > 0 else 1.0
            return {
                "atr_pct": round(cur, 3),
                "median": round(med, 3),
                "ratio": round(ratio, 2),
                "level": ("crisis" if ratio >= 2.6 else
                          "elevated" if ratio >= 1.8 else
                          "dead" if ratio <= 0.55 else "normal"),
            }
        except Exception as e:
            log.debug(f"Regime router: BTC vol read failed: {e}")
            return {"level": "unknown"}

    # ------------------------------------------------------------------
    # policy composition
    # ------------------------------------------------------------------
    @staticmethod
    def _weights() -> Dict[str, float]:
        return {s: 1.0 for s in ALL_STRATS}

    def compose(self, leaders: Dict, fng: Dict, vol: Dict,
                weekend: bool) -> Dict:
        """Turn the four components into one acceptance/routing policy."""
        pol = {
            "min_confidence_adjust": 0.0,
            "min_rr_adjust": 0.0,
            "size_multiplier": 1.0,
            "strategy_weights": self._weights(),
            "allow_new_entries": True,
            "freeze_reason": "",
            "reasons_ar": [],
        }
        w = pol["strategy_weights"]
        s = float(leaders.get("score", 50) or 50)

        # ---- 1. base state from the leaders ----
        if s >= 62:
            state = "trending_bull"
            pol["reasons_ar"].append(
                f"العملات الرئيسة تشير لاتجاه صاعد ({s:.0f}/100)")
            w["trend_pullback"] *= 1.25
            w["triple_confluence_trend"] *= 1.25
            w["volatility_breakout"] *= 1.15
            w["macd_breakout"] *= 1.15
            w["bb_mean_reversion"] *= 0.75
        elif s <= 40:
            state = "trending_bear"
            pol["min_confidence_adjust"] += 3.0
            pol["size_multiplier"] *= 0.85
            pol["reasons_ar"].append(
                f"العملات الرئيسة تشير لاتجاه هابط ({s:.0f}/100) — تشديد القبول")
            w["trend_pullback"] *= 0.70
            w["triple_confluence_trend"] *= 0.70
            w["volatility_breakout"] *= 0.70
            w["macd_breakout"] *= 0.70
            w["bb_mean_reversion"] *= 1.15
            w["liquidity_sweep_reversal"] *= 1.10
        else:
            state = "ranging"
            pol["reasons_ar"].append(
                f"العملات الرئيسة في نطاق عرضي ({s:.0f}/100) — ارتدادات أفضل من كسور")
            w["bb_mean_reversion"] *= 1.20
            w["liquidity_sweep_reversal"] *= 1.10
            w["volatility_breakout"] *= 0.75
            w["macd_breakout"] *= 0.75
            w["trend_pullback"] *= 0.90

        # ---- 2. volatility overlay (BTC ATR% ratio) ----
        lvl = vol.get("level", "unknown")
        r = vol.get("ratio")
        if lvl == "crisis":
            pol["min_confidence_adjust"] += 6.0
            pol["min_rr_adjust"] += 0.4
            pol["size_multiplier"] *= 0.5
            w["volatility_breakout"] *= 0.5
            w["macd_breakout"] *= 0.5
            w["trend_pullback"] *= 0.8
            pol["reasons_ar"].append(
                f"تقلب شديد (ATR ×{r}) — خفض الحجم للنصف وتشديد البوابات")
            if state == "trending_bear":
                pol["allow_new_entries"] = False
                pol["freeze_reason"] = (
                    "crisis: high vol + bearish leaders - new entries frozen")
        elif lvl == "elevated":
            pol["min_rr_adjust"] += 0.2
            pol["size_multiplier"] *= 0.7
            w["volatility_breakout"] *= 0.7
            w["trend_pullback"] *= 1.1
            pol["reasons_ar"].append(
                f"تقلب مرتفع (ATR ×{r}) — حجم أصغر ووقف أوسع ضمنياً")
        elif lvl == "dead":
            w["bb_mean_reversion"] *= 1.25
            w["liquidity_sweep_reversal"] *= 1.10
            w["volatility_breakout"] *= 0.70
            w["macd_breakout"] *= 0.70
            pol["reasons_ar"].append(
                f"تقلب منخفض (ATR ×{r}) — النطاقات والارتدادات أفضل من الكسور")

        # ---- 3. Fear & Greed overlay ----
        f = int(fng.get("value", 50) or 50)
        flabel = fng.get("label", "neutral")
        if f <= 20:
            pol["min_confidence_adjust"] += 4.0
            pol["size_multiplier"] *= 0.85
            w["bb_mean_reversion"] *= 1.15
            w["liquidity_sweep_reversal"] *= 1.20
            w["volatility_breakout"] *= 0.70
            w["trend_pullback"] *= 0.90
            pol["reasons_ar"].append(
                f"خوف شديد (F&G {f}) — منطقة قيعان محتملة: أولوية للماسح السفلي والارتدادات بحذر")
        elif f >= 80:
            pol["min_confidence_adjust"] += 4.0
            pol["size_multiplier"] *= 0.75
            w["trend_pullback"] *= 1.10
            w["macd_breakout"] *= 0.90
            pol["reasons_ar"].append(
                f"طمع شديد (F&G {f}) — سوق متأخر: قبول أكثر صرامة وحجم أصغر")

        # ---- 4. weekend overlay ----
        if weekend:
            pol["min_rr_adjust"] += 0.2
            pol["size_multiplier"] *= 0.75
            w["volatility_breakout"] *= 0.60
            w["macd_breakout"] *= 0.60
            w["bb_mean_reversion"] *= 1.15
            w["trend_pullback"] *= 0.95
            pol["reasons_ar"].append(
                "نهاية أسبوع — سيولة خفيفة: كسور أقل موثوقية وحجم مخفّض")

        # ---- caps / sanity ----
        pol["min_confidence_adjust"] = round(
            min(pol["min_confidence_adjust"], 12.0), 1)
        pol["min_rr_adjust"] = round(min(pol["min_rr_adjust"], 0.6), 2)
        pol["size_multiplier"] = round(
            max(0.25, min(pol["size_multiplier"], 1.25)), 2)
        w.update({k: round(v, 2) for k, v in w.items()})

        pol["state"] = state
        pol["leaders"] = leaders
        pol["fear_greed"] = {"value": f, "label": flabel,
                             "source": fng.get("source", "")}
        pol["volatility"] = vol
        pol["weekend"] = weekend
        pol["state_ar"] = self._state_ar(state, f, flabel, lvl, weekend, s)
        return pol

    @staticmethod
    def _state_ar(state: str, f: int, flabel: str, lvl: str,
                  weekend: bool, s: float) -> str:
        base = {
            "trending_bull": "اتجاه صاعد",
            "trending_bear": "اتجاه هابط",
            "ranging": "نطاق عرضي",
        }.get(state, "محايد")
        vol_ar = {"crisis": "تقلب شديد", "elevated": "تقلب مرتفع",
                  "dead": "تقلب منخفض", "normal": "تقلب طبيعي",
                  "unknown": ""}.get(lvl, "")
        parts = [f"{base} ({s:.0f}/100)",
                 f"الخوف والطمع: {f} ({flabel})"]
        if vol_ar:
            parts.append(vol_ar)
        if weekend:
            parts.append("نهاية الأسبوع")
        return " · ".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get_regime(self, force: bool = False) -> Dict:
        """Current regime decision (memory + file cached, REFRESH_MIN TTL)."""
        age = time.time() - self._regime_ts
        if not force and self._regime is not None \
                and age < max(1, settings.REGIME_REFRESH_MIN) * 60:
            return self._regime
        # try the file cache first (survives restarts, cheap)
        if not force and self._regime is None:
            cached = load_json(REGIME_FILE, default=None)
            if cached:
                try:
                    ts = datetime.fromisoformat(cached["computed_at"])
                    if now_utc() - ts < timedelta(
                            minutes=max(1, settings.REGIME_REFRESH_MIN)):
                        self._regime = cached
                        self._regime_ts = time.time()
                        return cached
                except Exception:
                    pass
        try:
            regime = self.compose(self._leaders(), self._fear_greed(),
                                  self._btc_vol(), is_weekend())
            regime["computed_at"] = now_utc().isoformat()
            self._regime = regime
            self._regime_ts = time.time()
            save_json(regime, REGIME_FILE)
            entries = "مسموح" if regime["allow_new_entries"] else "مجمّد"
            log.info(
                f"[bold cyan]Regime[/] {regime['state_ar']} | "
                f"حجم المقترح ×{regime['size_multiplier']} | "
                f"ثقة {settings.MIN_CONFIDENCE + regime['min_confidence_adjust']:.0f}% "
                f"| RR ≥ {settings.MIN_RR_RATIO + regime['min_rr_adjust']:.2f} "
                f"| فتح صفقات: {entries}"
            )
            return regime
        except Exception as e:
            log.warning(f"[yellow]Regime router failed ({e}) - neutral policy[/]")
            neutral = self.compose(
                {"score": 50.0, "verdict": "neutral", "posture_ar": ""},
                {"value": 50, "label": "neutral", "source": "error"},
                {"level": "unknown"}, is_weekend())
            neutral["computed_at"] = now_utc().isoformat()
            self._regime = neutral
            self._regime_ts = time.time()
            return neutral

    def active_policy(self) -> Dict:
        """Cheap policy accessor for hot paths (validate/sizing/ranking)."""
        return self.get_regime()

    def apply_regime_routing(self, recs: List[Dict]) -> List[Dict]:
        """v5.13: re-rank candidates by strategy-regime fit (in place+return).

        Each rec carries its producing strategies in rec["signals"][i]
        ["strategy"]; the DOMINANT strategy's regime weight becomes the
        rec's fit factor applied to the composite ranking key, so in a
        ranging regime mean-reversion candidates outrank breakout
        candidates with identical raw scores - and vice versa.
        """
        if not recs:
            return recs
        pol = self.active_policy()
        w = pol.get("strategy_weights", {})
        for r in recs:
            names = [
                s.get("strategy") for s in (r.get("signals") or [])
                if isinstance(s, dict) and s.get("strategy")
            ]
            fits = [float(w.get(n, 1.0)) for n in names] or [1.0]
            fit = max(fits)
            r["regime_fit"] = fit
            r["regime_state"] = pol.get("state", "")
        recs.sort(
            key=lambda r: (
                (r.get("confidence", 0)
                 + 5.0 * min(r.get("risk_reward_ratio", 0), 3.0) / 3.0
                 + 4.0 * min(r.get("harmony", 0.0), 1.0))
                * r.get("regime_fit", 1.0)
            ),
            reverse=True,
        )
        return recs

    def status(self) -> Dict:
        """Dashboard-facing snapshot."""
        try:
            return self.get_regime()
        except Exception:
            return {"state": "unknown", "state_ar": "غير متاح"}


# Singleton
regime_router = RegimeRouter()
