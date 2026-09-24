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

v5.4 additions (user request):
  - INTEGRITY  - a build where too many universe fetches failed is REJECTED
                 instead of cached (a leaders-only map used to freeze the
                 dashboard groups tab empty for the whole 6h TTL).
  - CYCLE      - run_market_cycle(): a periodic deep read of the leader
                 coins (trend + RSI14 + momentum + SMA50/ATR) that emits a
                 market-wide verdict + per-group policy for decisions.
  - FILE       - save_groups_file(): human-readable Arabic classification
                 file (data/market_groups.txt) sorted by leader, written
                 after every cycle for easy decision-making.
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
MARKET_CYCLE_FILE = Path("data/market_cycle.json")

INDEPENDENT = "independent"

# v5.4: Arabic verdict labels (decision file + dashboard are Arabic-first)
VERDICT_AR = {
    "strong_bullish": "صاعد بقوة",
    "bullish": "صاعد",
    "neutral": "محايد",
    "bearish": "هابط",
    "strong_bearish": "هابط بقوة",
}

POSTURE_AR = {
    "strong_bullish": "بيئة صاعدة قوية — معظم التوابع تحت قادة صاعدين؛ الشراء له الأولوية",
    "bullish": "بيئة صاعدة — يُفضل الشراء على تابعي القادة الصاعدين",
    "neutral": "سوق متوازن — الالتزام بالإشارات القوية فقط دون انحياز",
    "bearish": "بيئة هابطة — يُنصح بتجنب الشراء على تابعي القادة الهابطين",
    "strong_bearish": "بيئة هابطة قوية — خصم الثقة يُطبَّق تلقائياً على تابعي القادة الهابطين؛ حذر شديد من الشراء",
}

# map score (0-100) -> verdict label
VERDICT_THRESHOLDS = ((68, "strong_bullish"), (56, "bullish"),
                      (44, "neutral"), (32, "bearish"), (0, "strong_bearish"))


class MarketMap:
    """Classify symbols to leader coins and apply leader-regime to signals."""

    def __init__(self):
        self._map: Optional[Dict] = None  # in-memory cache of the JSON artifact
        self._last_fail_ts = None  # v5.4: throttle repeated failed rebuilds

    # ---------- cache ----------

    def _load_cache(self) -> Optional[Dict]:
        if self._map is not None:
            return self._map
        self._map = load_json(MARKET_MAP_FILE, default=None)
        return self._map

    def _is_stale(self, cache: Optional[Dict]) -> bool:
        if not cache or not cache.get("symbols"):
            return True
        # v5.4 completeness guard (the empty-tab bug): a map holding ONLY
        # the leaders is worthless - grouped_view() hides leader entries so
        # the tab rendered empty until the whole 6h TTL expired.
        symbols = cache.get("symbols") or {}
        if not any(not info.get("is_leader") for info in symbols.values()):
            return True
        try:
            updated = datetime.fromisoformat(cache["updated_at"])
        except (KeyError, ValueError, TypeError):
            return True
        age = now_utc() - updated
        if age > timedelta(hours=settings.MARKET_MAP_REFRESH_HOURS):
            return True
        # universe grew notably since the build -> new symbols unclassified
        try:
            cached_size = int(cache.get("universe_size") or 0)
            if cached_size and len(self._universe()) > cached_size * 1.4:
                log.info("[cyan]Market map stale: universe grew "
                         f"({cached_size} -> {len(self._universe())})[/]")
                return True
        except Exception:
            pass
        return False

    def get_map(self, force: bool = False) -> Dict:
        """Return the current map, recomputing it if stale (or forced).

        v5.4: failed rebuilds are throttled to one attempt per 5 minutes so
        an incomplete build (rate pressure) cannot hammer the API every
        time the dashboard polls this endpoint.
        """
        cache = self._load_cache()
        if force or self._is_stale(cache):
            if (not force and self._last_fail_ts
                    and now_utc() - self._last_fail_ts < timedelta(minutes=5)):
                return self._map or {"updated_at": None, "leaders": {}, "symbols": {}}
            try:
                self._map = self._build()
                save_json(self._map, MARKET_MAP_FILE)
                self._last_fail_ts = None
                # v5.4: a SUCCESSFUL but degenerate build (leaders only -
                # universe not yet known) must also be throttled, otherwise
                # every dashboard poll triggers a full rebuild burst.
                if not any(not i.get("is_leader")
                           for i in (self._map.get("symbols") or {}).values()):
                    self._last_fail_ts = now_utc()
            except Exception as e:
                log.error(f"[red]Market map build failed:[/] {e}")
                self._last_fail_ts = now_utc()
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

        universe = self._universe()
        symbols_meta: Dict[str, Dict] = {}
        failed: List[str] = []
        for sym in universe:
            if sym in leader_returns:  # leaders classify as themselves
                symbols_meta[sym] = {
                    "leader": sym, "corr": 1.0, "beta": 1.0, "is_leader": True,
                }
                continue
            try:
                df = data_fetcher.get_candles(sym, "1h", limit=lookback)
                if df is None or len(df) < 60:
                    failed.append(sym)
                    continue
                ret = df["close"].astype(float).pct_change().dropna()
            except Exception as e:
                log.debug(f"Market map: fetch failed for {sym}: {e}")
                failed.append(sym)
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

        # v5.4 integrity gate: refuse to cache a holey map. Under shared-IP
        # rate pressure most follower fetches used to fail silently, a
        # "leaders-only" artifact was saved as fresh and the groups tab
        # stayed EMPTY for the whole 6h TTL. Better to fail here (get_map
        # keeps the previous good cache) and retry on a later call.
        followers_expected = len(universe) - len(leader_returns)
        fail_pct = (len(failed) / followers_expected) if followers_expected > 0 else 0.0
        if followers_expected > 0 and fail_pct > settings.MARKET_MAP_MAX_FAIL_PCT:
            raise RuntimeError(
                f"Incomplete market map: {len(failed)}/{followers_expected} "
                f"({fail_pct:.0%}) of the universe has no candle data "
                f"(max {settings.MARKET_MAP_MAX_FAIL_PCT:.0%}) - "
                "not caching a holey classification"
            )

        return {
            "updated_at": now_utc().isoformat(),
            "corr_threshold": settings.MARKET_MAP_CORR_THRESHOLD,
            "lookback_hours": settings.MARKET_MAP_LOOKBACK_HOURS,
            "universe_size": len(universe),
            "classified_count": len(symbols_meta),
            "failed_count": len(failed),
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
            "refresh_hours": settings.MARKET_MAP_REFRESH_HOURS,
            "universe_size": m.get("universe_size"),
            "failed_count": m.get("failed_count"),
            "leaders": m.get("leaders") or {},
            "groups": groups,
        }


    # ---------- v5.4: market cycle via leader coins ----------

    @staticmethod
    def _verdict_from_score(score: float) -> str:
        for threshold, verdict in VERDICT_THRESHOLDS:
            if score >= threshold:
                return verdict
        return "neutral"

    def _group_sizes(self, m: Dict) -> Dict[str, int]:
        """{leader: follower_count} (+ 'independent'), leaders excluded."""
        sizes: Dict[str, int] = {}
        for info in (m.get("symbols") or {}).values():
            if info.get("is_leader"):
                continue
            ld = info.get("leader") or INDEPENDENT
            sizes[ld] = sizes.get(ld, 0) + 1
        return sizes

    def _analyze_leader(self, ld: str) -> Optional[Dict]:
        """Deep 1h read of one leader coin: trend + RSI14 + momentum + ATR.

        Served from the WS candle cache when available (zero REST weight).
        Returns a dict with a 0-100 score and a verdict label, or None.
        """
        try:
            df = data_fetcher.get_candles(ld, "1h", limit=200)
        except Exception as e:
            log.warning(f"[yellow]Market cycle: leader fetch failed {ld}: {e}[/]")
            return None
        if df is None or len(df) < 74:
            return None

        from src.indicators.technical import rsi as rsi_ind, atr as atr_ind

        c = df["close"].astype(float)
        price = float(c.iloc[-1])

        rsi_series = rsi_ind(c, 14)
        rsi_now = float(rsi_series.iloc[-1])
        if not np.isfinite(rsi_now):
            rsi_now = 50.0

        trend_meta = self._trend_from_closes(c)
        trend = trend_meta.get("trend", "neutral")

        sma50 = float(c.rolling(50).mean().iloc[-1])
        dist_pct = (price / sma50 - 1.0) * 100.0 if sma50 > 0 else 0.0
        chg24 = (price / float(c.iloc[-25]) - 1.0) * 100.0
        chg48 = (price / float(c.iloc[-49]) - 1.0) * 100.0

        atr_series = atr_ind(df["high"].astype(float), df["low"].astype(float), c, 14)
        atr_val = float(atr_series.iloc[-1])
        atr_pct = (atr_val / price * 100.0) if price > 0 and np.isfinite(atr_val) else 0.0

        # where inside the 24h range does price sit? (0 = at the low)
        hi24 = float(df["high"].astype(float).iloc[-24:].max())
        lo24 = float(df["low"].astype(float).iloc[-24:].min())
        pos24 = ((price - lo24) / (hi24 - lo24) * 100.0) if hi24 > lo24 else 50.0

        # ---- composite 0-100 score ----
        score = 50.0
        if trend == "bullish":
            score += 15.0
        elif trend == "bearish":
            score -= 15.0
        score += max(-8.0, min(8.0, dist_pct * 2.0))            # SMA50 distance
        score += max(-12.0, min(12.0, (rsi_now - 50.0) * 0.5))  # RSI tilt
        score += max(-7.5, min(7.5, max(-5.0, min(5.0, chg24)) * 1.5))  # 24h mom.
        score += max(-5.0, min(5.0, max(-8.0, min(8.0, chg48)) * 0.6))  # 48h mom.
        score = max(0.0, min(100.0, score))
        verdict = self._verdict_from_score(score)

        return {
            "price": round(price, 8),
            "trend": trend,
            "rsi14": round(rsi_now, 1),
            "chg_24h_pct": round(chg24, 2),
            "chg_48h_pct": round(chg48, 2),
            "sma50_dist_pct": round(dist_pct, 2),
            "atr_pct": round(atr_pct, 2),
            "pos_in_24h_range": round(pos24, 0),
            "score": round(score, 1),
            "verdict": verdict,
            "verdict_ar": VERDICT_AR[verdict],
        }

    @staticmethod
    def _cycle_is_stale(cycle: Optional[Dict]) -> bool:
        if not cycle or not cycle.get("leaders") or not cycle.get("market"):
            return True
        try:
            updated = datetime.fromisoformat(cycle["updated_at"])
        except (KeyError, ValueError, TypeError):
            return True
        return (now_utc() - updated) > timedelta(
            minutes=max(1, settings.MARKET_CYCLE_REFRESH_MIN))

    def run_market_cycle(self, force: bool = False) -> Dict:
        """دورة تحليل السوق عبر العملات السيدة (v5.4).

        A periodic deep read of the leader coins that turns their state
        into an explicit market-wide verdict + per-group policy:

          1. ensure the classification map is fresh (cheap: WS candle cache)
          2. analyze each leader: trend + RSI14 + 24h/48h momentum +
             SMA50 distance + ATR% + 24h-range position -> 0-100 score
          3. weight leader scores by their group sizes -> market posture
          4. save data/market_cycle.json (machine) and rewrite the
             human-readable decision file data/market_groups.txt

        Refreshed at most every MARKET_CYCLE_REFRESH_MIN (default 60 min);
        the artifacts are what the user reads for decision-making.
        """
        if not force:
            cached = load_json(MARKET_CYCLE_FILE, default=None)
            if not self._cycle_is_stale(cached):
                return cached

        m = self.get_map()
        group_sizes = self._group_sizes(m)

        leaders_analysis: Dict[str, Dict] = {}
        for ld in (m.get("leaders") or {}):
            a = self._analyze_leader(ld)
            if a:
                a["followers"] = group_sizes.get(ld, 0)
                leaders_analysis[ld] = a

        policy: List[Dict] = []
        weighted = 0.0
        total_w = 0.0
        for ld, a in leaders_analysis.items():
            w = 1.0 + group_sizes.get(ld, 0)
            weighted += a["score"] * w
            total_w += w
            n = group_sizes.get(ld, 0)
            v = a["verdict"]
            if n == 0:
                continue  # no followers - the posture already reflects this leader
            if v in ("bearish", "strong_bearish"):
                policy.append({
                    "leader": ld, "action": "avoid_longs", "followers": n,
                    "text_ar": f"تجنب صفقات الشراء الجديدة على تابعي {ld} "
                               f"({n} عملة) — القائد {VERDICT_AR[v]}",
                })
            elif v in ("bullish", "strong_bullish"):
                policy.append({
                    "leader": ld, "action": "prioritize_longs", "followers": n,
                    "text_ar": f"أولوية الشراء على تابعي {ld} ({n} عملة) — "
                               f"القائد {VERDICT_AR[v]}",
                })
        if group_sizes.get(INDEPENDENT):
            n = group_sizes[INDEPENDENT]
            policy.append({
                "leader": INDEPENDENT, "action": "watch", "followers": n,
                "text_ar": f"{n} عملة مستقلة — تُقيَّم على جدارتها الخاصة فقط",
            })

        market_score = round(weighted / total_w, 1) if total_w else 50.0
        verdict = self._verdict_from_score(market_score)
        cycle = {
            "updated_at": now_utc().isoformat(),
            "refresh_minutes": settings.MARKET_CYCLE_REFRESH_MIN,
            "group_sizes": group_sizes,
            "leaders": leaders_analysis,
            "market": {
                "score": market_score,
                "verdict": verdict,
                "verdict_ar": VERDICT_AR[verdict],
                "posture_ar": POSTURE_AR[verdict],
                "policy": policy,
            },
        }
        save_json(cycle, MARKET_CYCLE_FILE)
        self.save_groups_file(cycle)
        log.info(
            f"[bold cyan]Market cycle[/] - {verdict} (score {market_score}) "
            f"| groups: " + ", ".join(
                f"{ld}={a['verdict']}({a['followers']}f)"
                for ld, a in leaders_analysis.items())
        )
        return cycle

    # ---------- v5.4: human-readable decision file ----------

    def save_groups_file(self, cycle: Optional[Dict] = None) -> Path:
        """اكتب ملف التصنيف العربي المقروء لتسهيل اتخاذ القرار.

        Layout: market-cycle verdict on top (posture + policies + per-leader
        readings), then the full classification sorted by leader with corr
        and beta per follower. Saved to settings.MARKET_GROUPS_FILE.
        """
        m = self._map or self.get_map()
        if cycle is None:
            try:
                cycle = load_json(MARKET_CYCLE_FILE, default=None)
            except Exception:
                cycle = None

        groups: Dict[str, List[Dict]] = {}
        for sym, info in (m.get("symbols") or {}).items():
            if info.get("is_leader"):
                continue
            ld = info.get("leader") or INDEPENDENT
            groups.setdefault(ld, []).append({
                "symbol": sym, "corr": info.get("corr"), "beta": info.get("beta"),
            })
        for ld in groups:
            groups[ld].sort(key=lambda x: -(x.get("corr") or 0))

        lines: List[str] = []
        lines.append("=" * 62)
        lines.append("🧭 ملف تصنيف مجموعات السوق — العملات السيدة والتوابع")
        lines.append(f"آخر تحديث: {now_utc().strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append(f"عتبة الارتباط: {m.get('corr_threshold')} · "
                     f"نافذة الحساب: {m.get('lookback_hours')} ساعة · "
                     f"حجم الكون: {m.get('universe_size', '?')} رمز")
        lines.append("=" * 62)
        lines.append("")
        lines.append("استخدام الملف: قبل فتح صفقة شراء على عملة، انظر لمجموعتها —")
        lines.append("إذا كان قائدها هابطاً فالاحتمال الأكبر أن تنزلق العملة معه.")
        lines.append("")

        mk = (cycle or {}).get("market") or {}
        if mk:
            lines.append(f"▶ حكم دورة السوق: {mk.get('verdict_ar', '-')} "
                         f"(النتيجة {mk.get('score', '-')}/100)")
            lines.append(f"  {mk.get('posture_ar', '')}")
            lines.append("")
            lines.append("— قراءة العملات السيدة (فريم 1h) —")
            for ld, a in ((cycle or {}).get("leaders") or {}).items():
                lines.append(
                    f"  {ld}: {a['verdict_ar']} · النتيجة {a['score']}/100 · "
                    f"RSI14 {a['rsi14']} · 24h {a['chg_24h_pct']:+.2f}% · "
                    f"48h {a['chg_48h_pct']:+.2f}% · عن SMA50 "
                    f"{a['sma50_dist_pct']:+.2f}% · ATR {a['atr_pct']:.2f}%"
                )
            lines.append("")
            lines.append("— سياسة المجموعات —")
            for p in mk.get("policy") or []:
                lines.append(f"  • {p.get('text_ar', '')}")
            lines.append("")

        lines.append("— التصنيف الكامل (الأعلى ارتباطاً أولاً) —")
        leaders = list((m.get("leaders") or {}).keys())
        order = [ld for ld in leaders if groups.get(ld)]
        order.extend(g for g in groups if g not in order)
        for ld in order:
            if ld == INDEPENDENT:
                lines.append(f"  [عملات مستقلة] — {len(groups[ld])} عملة "
                             "بدون قائد واضح:")
            else:
                trend = ((cycle or {}).get("leaders") or (m.get("leaders") or {}) or {}) \
                    .get(ld, {})
                label = trend.get("verdict_ar") or VERDICT_AR.get(
                    trend.get("trend"), "") if isinstance(trend, dict) else ""
                lines.append(f"  [مجموعة {ld}] {('— ' + label) if label else ''} — "
                             f"{len(groups[ld])} عملة تتبع القائد:")
            for f in groups[ld]:
                corr = f"{f['corr'] * 100:.0f}%" if f.get("corr") is not None else "—"
                beta = f"{f['beta']:.2f}" if f.get("beta") is not None else "—"
                lines.append(f"      {f['symbol']:<14} corr {corr:>4}  beta {beta}")
            lines.append("")

        path = Path(settings.MARKET_GROUPS_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info(f"[green]Market groups file saved[/] -> {path}")
        return path


# Singleton
market_map = MarketMap()
