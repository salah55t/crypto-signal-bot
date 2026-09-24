"""Market Map - leader/follower correlation classification (v5.2).

User observation (2026-09-24): many altcoins chart almost identically to a
major (BTC, SOL, XRP...). This module quantifies that:

1. CLASSIFY  - Pearson correlation of 1h returns over the lookback window
               assigns each symbol to its highest-correlated leader
               (BTC/ETH/SOL/XRP) if corr >= threshold, else "independent".
2. REGIME    - each leader's own trend (1h SMA50 + slope) is detected:
               bullish / bearish / neutral.
3. ACTION    - apply_regime() adjusts follower signals before filtering:
                 bearish leader -> confidence -= corr * PENALTY  (admission
                                  too - penalties may demote out, v4 rule)
                 bullish leader -> confidence += corr * BOOST    (rank-only,
                                  admission stays merit-based)

The map is cached in data/market_map.json and recomputed at most every
MARKET_MAP_REFRESH_HOURS (default 6h) to protect the API weight budget.
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import settings
from src.core.data_fetcher import data_fetcher
from src.utils.helpers import load_json, save_json, now_utc
from src.utils.logger import log

MARKET_MAP_FILE = Path("data/market_map.json")

INDEPENDENT = "independent"


class MarketMap:
    """Classify symbols to leader coins and apply leader-regime to signals."""

    def __init__(self):
        self._map: Optional[Dict] = None  # in-memory cache of the JSON artifact

    # ---------- cache ----------

    def _load_cache(self) -> Optional[Dict]:
        if self._map is not None:
            return self._map
        self._map = load_json(MARKET_MAP_FILE, default=None)
        return self._map

    def _is_stale(self, cache: Optional[Dict]) -> bool:
        if not cache or not cache.get("symbols"):
            return True
        try:
            updated = datetime.fromisoformat(cache["updated_at"])
        except (KeyError, ValueError, TypeError):
            return True
        age = now_utc() - updated
        return age > timedelta(hours=settings.MARKET_MAP_REFRESH_HOURS)

    def get_map(self, force: bool = False) -> Dict:
        """Return the current map, recomputing it if stale (or forced)."""
        cache = self._load_cache()
        if force or self._is_stale(cache):
            try:
                self._map = self._build()
                save_json(self._map, MARKET_MAP_FILE)
            except Exception as e:
                log.error(f"[red]Market map build failed:[/] {e}")
                # fall back to whatever cache exists (may be None)
        return self._map or {"updated_at": None, "leaders": {}, "symbols": {}}

    # ---------- classification ----------

    def _build(self) -> Dict:
        leaders = list(settings.MARKET_MAP_LEADERS)
        lookback = max(60, settings.MARKET_MAP_LOOKBACK_HOURS)

        leader_returns: Dict[str, pd.Series] = {}
        leader_meta: Dict[str, Dict] = {}
        for ld in leaders:
            df = data_fetcher.get_candles(ld, "1h", limit=lookback)
            if df is None or len(df) < 60:
                log.warning(f"[yellow]Leader {ld}: insufficient data[/]")
                continue
            closes = df["close"].astype(float)
            leader_returns[ld] = closes.pct_change().dropna()
            leader_meta[ld] = self._trend_from_closes(closes)

        if not leader_returns:
            raise RuntimeError("No leader data available for market map")

        symbols_meta: Dict[str, Dict] = {}
        for sym in self._universe():
            if sym in leader_returns:  # leaders classify as themselves
                symbols_meta[sym] = {
                    "leader": sym, "corr": 1.0, "beta": 1.0, "is_leader": True,
                }
                continue
            try:
                df = data_fetcher.get_candles(sym, "1h", limit=lookback)
                if df is None or len(df) < 60:
                    continue
                ret = df["close"].astype(float).pct_change().dropna()
            except Exception as e:
                log.debug(f"Market map: fetch failed for {sym}: {e}")
                continue

            best_ld, best_corr, best_beta = None, 0.0, None
            for ld, lret in leader_returns.items():
                corr, beta = self._corr_beta(ret, lret)
                if corr is not None and corr > best_corr:
                    best_ld, best_corr, best_beta = ld, corr, beta

            if best_ld and best_corr >= settings.MARKET_MAP_CORR_THRESHOLD:
                symbols_meta[sym] = {
                    "leader": best_ld, "corr": round(best_corr, 3),
                    "beta": round(best_beta, 3) if best_beta else None,
                    "is_leader": False,
                }
            else:
                symbols_meta[sym] = {
                    "leader": INDEPENDENT,
                    "corr": round(best_corr, 3) if best_ld else None,
                    "beta": round(best_beta, 3) if best_beta else None,
                    "is_leader": False,
                }

        return {
            "updated_at": now_utc().isoformat(),
            "corr_threshold": settings.MARKET_MAP_CORR_THRESHOLD,
            "lookback_hours": settings.MARKET_MAP_LOOKBACK_HOURS,
            "leaders": leader_meta,
            "symbols": symbols_meta,
        }

    def _universe(self) -> List[str]:
        """Symbols to classify: leaders + configured universe."""
        try:
            from src.analysis.analyzer import analyzer
            syms = list(analyzer.symbols)
        except Exception:
            syms = []
        out = list(dict.fromkeys(list(settings.MARKET_MAP_LEADERS) + syms))
        return out[: max(len(syms), len(settings.MARKET_MAP_LEADERS)) + 50]

    @staticmethod
    def _corr_beta(sym_ret: pd.Series, leader_ret: pd.Series):
        """Pearson corr + beta of aligned return series; (None, None) if degenerate."""
        joined = pd.concat([sym_ret, leader_ret], axis=1, join="inner").dropna()
        joined.columns = ["s", "l"]
        if len(joined) < 50:
            return None, None
        corr = float(joined["s"].corr(joined["l"]))
        var_l = float(np.var(joined["l"]))
        if not np.isfinite(corr) or var_l <= 0:
            return None, None
        beta = float(np.cov(joined["s"], joined["l"])[0][1] / var_l)
        return corr, beta

    # ---------- leader regime ----------

    @staticmethod
    def _trend_from_closes(closes: pd.Series) -> Dict:
        """Trend from price vs SMA50 + SMA50 slope over the last 24 bars."""
        c = closes.astype(float)
        if len(c) < 74:  # need SMA50 + 24h slope
            return {"trend": "neutral", "price": float(c.iloc[-1])}
        price = float(c.iloc[-1])
        sma50 = c.rolling(50).mean()
        sma_now = float(sma50.iloc[-1])
        sma_24h_ago = float(sma50.iloc[-25])
        slope_up = sma_now > sma_24h_ago
        chg_24h = (price / float(c.iloc[-25]) - 1.0) * 100.0

        if price > sma_now and slope_up:
            trend = "bullish"
        elif price < sma_now and not slope_up:
            trend = "bearish"
        else:
            trend = "neutral"
        return {
            "trend": trend,
            "price": round(price, 8),
            "sma50": round(sma_now, 8),
            "chg_24h_pct": round(chg_24h, 2),
        }

    def leader_trends(self) -> Dict[str, str]:
        """{leader: trend} from the current map (empty if map is fresh-less)."""
        m = self.get_map()
        return {ld: meta.get("trend", "neutral")
                for ld, meta in (m.get("leaders") or {}).items()}

    # ---------- the ACTION ----------

    def apply_regime(self, results: List[Dict]) -> List[Dict]:
        """Adjust bullish follower signals by their leader's trend (in place).

        - bearish leader: confidence -= corr * LEADER_BEARISH_CONF_PENALTY
          (applied to admission_confidence as well - penalties may demote out)
        - bullish leader: confidence += corr * LEADER_BULLISH_CONF_BOOST
          (rank-only boost - admission_confidence untouched, v4 rule)
        """
        if not results:
            return results
        try:
            m = self.get_map()
            leaders = m.get("leaders") or {}
            symbols = m.get("symbols") or {}
        except Exception as e:
            log.warning(f"[yellow]Regime filter skipped (no map): {e}[/]")
            return results

        penalized = boosted = 0
        for r in results:
            if r.get("direction") != "bullish":
                continue
            info = symbols.get(r.get("symbol"))
            if not info:
                continue
            leader = info.get("leader")
            corr = float(info.get("corr") or 0.0)
            r["leader"] = leader
            r["leader_corr"] = corr
            if not leader or leader == INDEPENDENT or info.get("is_leader"):
                r["leader_trend"] = "n/a"
                continue
            trend = (leaders.get(leader) or {}).get("trend", "neutral")
            r["leader_trend"] = trend

            if trend == "bearish" and corr > 0:
                penalty = min(corr * settings.LEADER_BEARISH_CONF_PENALTY, 25.0)
                r["confidence"] = round(max(0.0, r.get("confidence", 0) - penalty), 2)
                r["admission_confidence"] = round(
                    max(0.0, r.get("admission_confidence",
                                   r.get("confidence", 0)) - penalty), 2)
                r["regime_adj"] = -penalty
                penalized += 1
            elif trend == "bullish" and corr > 0:
                boost = min(corr * settings.LEADER_BULLISH_CONF_BOOST, 10.0)
                r["confidence"] = round(min(100.0, r.get("confidence", 0) + boost), 2)
                r["regime_adj"] = boost  # admission_confidence untouched (rank-only)
                boosted += 1
            else:
                r["regime_adj"] = 0.0

        if penalized or boosted:
            log.info(
                f"[cyan]Market regime:[/] {penalized} signals penalized "
                f"(bearish leader), {boosted} boosted (bullish leader)"
            )
        return results

    # ---------- dashboard view ----------

    def grouped_view(self) -> Dict:
        """Human-friendly groups: {leader: {trend, followers: [{symbol,corr,beta}]}}."""
        m = self.get_map()
        groups: Dict[str, List[Dict]] = {}
        for sym, info in (m.get("symbols") or {}).items():
            if info.get("is_leader"):
                continue  # leaders are represented by their own card
            ld = info.get("leader") or INDEPENDENT
            groups.setdefault(ld, []).append({
                "symbol": sym,
                "corr": info.get("corr"),
                "beta": info.get("beta"),
            })
        for ld in groups:
            groups[ld].sort(key=lambda x: -(x.get("corr") or 0))
        return {
            "updated_at": m.get("updated_at"),
            "corr_threshold": m.get("corr_threshold"),
            "lookback_hours": m.get("lookback_hours"),
            "leaders": m.get("leaders") or {},
            "groups": groups,
        }


# Singleton
market_map = MarketMap()
