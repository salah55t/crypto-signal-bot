"""
Risk Management Module
- Position sizing (Kelly fraction / fixed fractional)
- Daily max loss enforcement
- Risk/Reward check
- Stop loss / Take profit verification

v5 "Veteran Trader" management:
  - Partial TP: bank half at TP1, SL -> break-even+fees, run rest to TP2
  - MFE/MAE excursion tracking per position (peak/trough since entry)
  - ATR chandelier trailing (market-adaptive) on top of the % ladder
  - Structure exits: Ichimoku regime flip / opposite strong signal
  - Time stop: stale trades are dead capital
  - Pending LIMIT entries: buy the golden pocket, never chase
"""
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timezone, timedelta
import functools
import secrets
import threading
from config.settings import settings
from src.db.database import db
from src.utils.logger import log
from src.utils.helpers import load_json, save_json, now_utc

POSITIONS_FILE = Path("data/open_positions.json")
DAILY_STATS_FILE = Path("data/daily_stats.json")
PENDING_FILE = Path("data/pending_entries.json")


def _new_trade_uid() -> str:
    """v5.11: unique, human-readable trade identity (ledger primary key)."""
    return f"TRD-{now_utc().strftime('%Y%m%d')}-{secrets.token_hex(2).upper()}"


def _locked(fn):
    """v5.26: serialize RiskManager state access across threads.

    open_positions/daily_stats/pending_entries were mutated from FOUR
    concurrent contexts (analysis cron, 1-min watcher, manual POST
    triggers, /api/reset-history) with no synchronization: index-based
    close_position could pop the WRONG position after another thread
    shifted the list, and concurrent save_json calls could lose updates.
    Every public method that reads-then-writes trading state now runs
    under the instance RLock (re-entrant: check_open_positions ->
    close_position and open_* -> can_open_position nest safely).

    Known trade-off (accepted): open_* may fetch the BTC tide snapshot
    (cached 30 min, REST fetch only on stale cache) while holding the
    lock - the 1-min watcher may wait one fetch. Correctness wins over
    the race it eliminates. Lock order is always _lock -> transport
    (rate limiter/DB); no reverse path exists.
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return wrapper


def _primary_strategy(rec: Dict) -> str:
    """Best-scoring strategy name attached to the recommendation (for stats)."""
    sigs = rec.get("signals") or []
    try:
        best = max(sigs, key=lambda s: float(s.get("score", 0) or 0))
        if best and best.get("strategy"):
            return str(best["strategy"])
    except Exception:
        pass
    return str(rec.get("strategy") or "")


class RiskManager:
    """Enforces risk rules across the trading bot."""

    def __init__(self, capital: float = None):
        # v5.26: the lock must exist before ANY state is touched (the DB
        # restore below and every decorated method depend on it).
        self._lock = threading.RLock()
        self.capital = capital or settings.INITIAL_CAPITAL
        self.open_positions: List[Dict] = load_json(POSITIONS_FILE, default=[])
        self.daily_stats: Dict = load_json(DAILY_STATS_FILE, default={})
        self.pending_entries: List[Dict] = (
            load_json(PENDING_FILE, default=[]) if settings.PENDING_ENTRIES_ENABLED else []
        )
        # v4.1 loss-avoidance state
        self._reentry_block: Dict[str, datetime] = {}  # symbol -> blocked until
        # v5.11: reconcile the JSON working set with the persistent ledger
        # BEFORE anything else touches positions (entry prices survive even
        # when data/open_positions.json is wiped by a redeploy).
        self._restore_from_db()
        self._restore_loss_state()
        log.info(
            f"[cyan]RiskManager[/] initialized (v5) - "
            f"Capital: ${self.capital:,.2f} | "
            f"Open positions: {len(self.open_positions)} | "
            f"Pending limit entries: {len(self.pending_entries)}"
        )

    def _restore_loss_state(self):
        """v4.1: restore loss-streak / pause state persisted in today's stats."""
        try:
            today = self.daily_stats.get(self._today_key(), {})
            self._loss_streak = int(today.get("loss_streak", 0))
            pause_until = today.get("loss_pause_until")
            self._loss_pause_until = (
                datetime.fromisoformat(pause_until) if pause_until else None
            )
        except Exception:
            self._loss_streak = 0
            self._loss_pause_until = None

    @property
    def loss_streak(self) -> int:
        return getattr(self, "_loss_streak", 0)

    @loss_streak.setter
    def loss_streak(self, value: int):
        self._loss_streak = value

    def _set_loss_pause(self):
        """v4.1: activate anti-tilt pause after N consecutive losses."""
        self._loss_pause_until = now_utc() + timedelta(
            hours=settings.LOSS_STREAK_PAUSE_HOURS
        )
        self._persist_loss_state()
        log.warning(
            f"[red]Loss-streak circuit breaker:[/] {self._loss_streak} consecutive "
            f"losses - pausing new entries until {self._loss_pause_until.isoformat()}"
        )

    def _persist_loss_state(self):
        """Persist streak/pause inside today's daily stats (best effort)."""
        try:
            self._ensure_today_stats()
            stats = self.daily_stats[self._today_key()]
            stats["loss_streak"] = self._loss_streak
            stats["loss_pause_until"] = (
                self._loss_pause_until.isoformat() if self._loss_pause_until else None
            )
            save_json(self.daily_stats, DAILY_STATS_FILE)
        except Exception as e:
            log.debug(f"Persist loss state failed: {e}")

    def _loss_pause_active(self) -> bool:
        return (
            self._loss_pause_until is not None
            and now_utc() < self._loss_pause_until
        )

    def _in_reentry_cooldown(self, symbol: str) -> bool:
        """v4.1: symbol recently hit Stop Loss -> block re-entry for a while."""
        until = self._reentry_block.get(symbol)
        if until is None:
            return False
        if now_utc() >= until:
            del self._reentry_block[symbol]
            return False
        return True

    @_locked
    def sync_daily_opened(self, db_count: int):
        """v4.1: sync today's opened count from DB (survives process restarts)."""
        self._ensure_today_stats()
        stats = self.daily_stats[self._today_key()]
        if db_count > stats.get("trades_opened", 0):
            stats["trades_opened"] = int(db_count)
            save_json(self.daily_stats, DAILY_STATS_FILE)

    def _today_key(self) -> str:
        return now_utc().strftime("%Y-%m-%d")

    def _ensure_today_stats(self):
        today = self._today_key()
        if today not in self.daily_stats:
            self.daily_stats[today] = {
                "trades_opened": 0,
                "wins": 0,
                "losses": 0,
                "pnl": 0.0,
                "starting_capital": self.capital,
            }

    def daily_pnl_pct(self) -> float:
        """Today's P&L as % of starting capital."""
        self._ensure_today_stats()
        stats = self.daily_stats[self._today_key()]
        if stats.get("starting_capital", 0) == 0:
            return 0.0
        return stats["pnl"] / stats["starting_capital"] * 100

    def is_symbol_blocked(self, symbol: str) -> bool:
        """v4.1: public check for per-symbol re-entry cooldown (after a loss)."""
        return bool(symbol) and self._in_reentry_cooldown(symbol)

    @_locked
    def can_open_position(self, symbol: str = None) -> bool:
        """Check if we can open a new position (risk rules).
        v4.1 adds: daily trade cap, loss-streak pause, per-symbol re-entry cooldown.
        """
        if len(self.open_positions) >= settings.MAX_OPEN_POSITIONS:
            log.warning(f"Max open positions reached ({settings.MAX_OPEN_POSITIONS})")
            return False
        if self.daily_pnl_pct() <= -settings.DAILY_MAX_LOSS:
            log.warning(f"Daily max loss hit ({self.daily_pnl_pct():.2f}%)")
            return False
        # v4.1: daily trade count cap (fees from churn exceeded profits live)
        self._ensure_today_stats()
        opened_today = self.daily_stats[self._today_key()].get("trades_opened", 0)
        if opened_today >= settings.MAX_TRADES_PER_DAY:
            log.warning(
                f"Daily trade cap reached ({opened_today}/{settings.MAX_TRADES_PER_DAY})"
            )
            return False
        # v4.1: anti-tilt circuit breaker
        if self._loss_pause_active():
            log.warning(
                f"Loss-streak pause active until {self._loss_pause_until.isoformat()}"
            )
            return False
        # v4.1: per-symbol re-entry cooldown after a Stop Loss
        if symbol and self._in_reentry_cooldown(symbol):
            log.warning(
                f"Re-entry cooldown active for {symbol} "
                f"(last SL < {settings.REENTRY_COOLDOWN_HOURS}h ago)"
            )
            return False
        return True

    def _bottom_channel_ok(self, symbol: str) -> Tuple[bool, str]:
        """v5.20: clustering + tape caps for the bottom-boost channel.

        The 2026-09-29 burst put FIVE bottom longs into the market within
        6 hours on a falling tape: BOTTOM_MAX_PER_CYCLE only counts entries
        inside ONE cycle, so successive cycles stacked correlated risk.
        Admission-time checks (paper and live alike):
          1. max concurrent open bottom positions
          2. minimum spacing since the newest bottom entry
          3. no new bottom longs into a bearish 1h BTC regime (the whole
             6-trade burst was counter-tape)
        """
        bottoms = [p for p in self.open_positions
                   if p.get("boosted_from_bottom")]
        cap = int(settings.BOTTOM_MAX_OPEN_CONCURRENT)
        if len(bottoms) >= cap:
            return (False,
                    f"bottom concurrent cap ({len(bottoms)}/{cap} open)")
        last_ts = None
        for p in bottoms:
            try:
                t = datetime.fromisoformat(str(p.get("entry_time")))
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                if last_ts is None or t > last_ts:
                    last_ts = t
            except Exception:
                continue
        if last_ts is not None:
            gap_min = (now_utc() - last_ts).total_seconds() / 60.0
            if gap_min < settings.BOTTOM_ENTRY_SPACING_MIN:
                return (False,
                        f"bottom entry spacing ({gap_min:.0f}min < "
                        f"{settings.BOTTOM_ENTRY_SPACING_MIN:.0f}min)")
        if settings.BOTTOM_BTC_TIDE_GATE:
            regime, _score = self._tide_snapshot()
            if regime == "bearish":
                return (False,
                        "bottom tide gate (BTC 1h regime bearish) - "
                        "no counter-tape bounce longs")
        return (True, "")

    def _momentum_channel_ok(self, symbol: str) -> Tuple[bool, str]:
        """v5.22: clustering + tape caps for the double-indicator channel.

        Same three protections as the bottom channel (v5.20), sized for a
        1m momentum scalper whose failure mode is re-firing the same burst
        every cycle and stacking correlated longs on one tape:
          1. max concurrent open double-indicator positions
          2. minimum spacing since the newest momentum entry
          3. no new momentum longs into a bearish 1h BTC regime
        """
        moments = [p for p in self.open_positions
                   if p.get("boosted_from_momentum")]
        cap = int(settings.DOUBLE_IND_MAX_OPEN_CONCURRENT)
        if len(moments) >= cap:
            return (False,
                    f"momentum concurrent cap ({len(moments)}/{cap} open)")
        last_ts = None
        for p in moments:
            try:
                t = datetime.fromisoformat(str(p.get("entry_time")))
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                if last_ts is None or t > last_ts:
                    last_ts = t
            except Exception:
                continue
        if last_ts is not None:
            gap_min = (now_utc() - last_ts).total_seconds() / 60.0
            if gap_min < settings.DOUBLE_IND_ENTRY_SPACING_MIN:
                return (False,
                        f"momentum entry spacing ({gap_min:.0f}min < "
                        f"{settings.DOUBLE_IND_ENTRY_SPACING_MIN:.0f}min)")
        if settings.DOUBLE_IND_BTC_TIDE_GATE:
            regime, _score = self._tide_snapshot()
            if regime == "bearish":
                return (False,
                        "momentum tide gate (BTC 1h regime bearish) - "
                        "no counter-tape momentum longs")
        return (True, "")

    def _scalp_channel_ok(self, symbol: str) -> Tuple[bool, str]:
        """v5.25: clustering + tape caps for the micro-scalp channel.

        Same three protections as bottom/momentum (v5.20/v5.22), sized for
        a fixed-60s-hold scalper whose failure mode is machine-gunning the
        same burst tick after tick:
          1. max concurrent open micro-scalp positions
          2. minimum spacing since the newest scalp entry
          3. no new scalp longs into a bearish 1h BTC regime
        """
        scalps = [p for p in self.open_positions
                  if p.get("boosted_from_scalp")]
        cap = int(settings.SCALP_MAX_OPEN_CONCURRENT)
        if len(scalps) >= cap:
            return (False,
                    f"scalp concurrent cap ({len(scalps)}/{cap} open)")
        last_ts = None
        for p in scalps:
            try:
                t = datetime.fromisoformat(str(p.get("entry_time")))
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                if last_ts is None or t > last_ts:
                    last_ts = t
            except Exception:
                continue
        if last_ts is not None:
            gap_min = (now_utc() - last_ts).total_seconds() / 60.0
            if gap_min < settings.SCALP_ENTRY_SPACING_MIN:
                return (False,
                        f"scalp entry spacing ({gap_min:.1f}min < "
                        f"{settings.SCALP_ENTRY_SPACING_MIN:.1f}min)")
        if settings.SCALP_BTC_TIDE_GATE:
            regime, _score = self._tide_snapshot()
            if regime == "bearish":
                return (False,
                        "scalp tide gate (BTC 1h regime bearish) - "
                        "no counter-tape scalp longs")
        return (True, "")

    def position_size_notional(self, entry_price: float,
                                stop_loss: float) -> float:
        """
        v5.26: USD notional for a trade whose STOP hit loses exactly
        RISK_PER_TRADE% of capital (real fixed-fractional risk):

            notional = (capital * RISK_PER_TRADE%) / (|entry - SL| / entry)

        - regime size multiplier still applies (v5.13 contract)
        - hard-capped at 20% of capital (unchanged)
        - unusable entry/SL -> conservative fallback of 5x the risk amount
          (assumes a 2% stop distance), STILL capped - never the old
          unlinked 10x heuristic that ignored the stop entirely.
        """
        risk_amount = self.capital * (settings.RISK_PER_TRADE / 100.0)
        notional = 0.0
        try:
            entry = float(entry_price or 0)
            sl = float(stop_loss or 0)
        except (TypeError, ValueError):
            entry = sl = 0.0
        if entry > 0 and sl > 0:
            sl_pct = abs(entry - sl) / entry
            if sl_pct > 0:
                notional = risk_amount / sl_pct
        if notional <= 0:
            notional = risk_amount * 5.0
        notional *= self._regime_size_multiplier()
        return min(notional, self.capital * 0.20)

    def position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        Compute position size in base currency using fixed fractional risk.
        Risk = settings.RISK_PER_TRADE% of capital.
        Returns position size in units of base asset.
        v5.26: derived from position_size_notional (single source of truth
        shared with the live sizing path, including the 20% capital cap).
        """
        try:
            entry = float(entry_price or 0)
        except (TypeError, ValueError):
            return 0.0
        if entry <= 0:
            log.warning("Invalid entry_price for position_size")
            return 0.0
        return self.position_size_notional(entry_price, stop_loss) / entry

    @staticmethod
    def _regime_gates() -> tuple:
        """v5.13: (effective_min_confidence, effective_min_rr) from the regime
        router. Fails open to static settings on any error."""
        try:
            if not settings.REGIME_ENABLED:
                return (settings.MIN_CONFIDENCE, settings.MIN_RR_RATIO)
            from src.analysis.regime_router import regime_router
            pol = regime_router.active_policy()
            return (
                settings.MIN_CONFIDENCE + float(pol.get("min_confidence_adjust", 0)),
                settings.MIN_RR_RATIO + float(pol.get("min_rr_adjust", 0)),
            )
        except Exception:
            return (settings.MIN_CONFIDENCE, settings.MIN_RR_RATIO)

    @staticmethod
    def _regime_size_multiplier() -> float:
        """v5.13: regime size multiplier (0.25..1.25); 1.0 on any failure."""
        try:
            if not settings.REGIME_ENABLED:
                return 1.0
            from src.analysis.regime_router import regime_router
            return float(regime_router.active_policy().get("size_multiplier", 1.0))
        except Exception:
            return 1.0

    def validate_recommendation(self, rec: Dict) -> tuple:
        """
        Returns (is_valid, reasons).
        Validates R/R ratio, ATR sanity, etc.
        v5: harmony gate (layered agreement) + volatility sanity.
        v5.13: REGIME-ADJUSTED thresholds - the regime router (leaders +
        Fear & Greed + weekend + BTC volatility) shifts MIN_CONFIDENCE and
        MIN_RR up in hostile states (weekend, high vol, extreme fear/greed,
        bearish leaders) instead of using static settings only.
        """
        reasons = []
        # v5.13: regime-adjusted effective gates (cheap: cached policy)
        conf_gate, rr_gate = self._regime_gates()
        # v4: vetoed signals (Ichimoku regime opposition) can never open positions
        if rec.get("decision", {}).get("vetoed", False):
            reasons.append(
                f"Vetoed by confluence engine: "
                f"{rec.get('decision', {}).get('veto_reason', 'regime opposition')}"
            )
        rr = rec.get("risk_reward_ratio", 0)
        if rr < rr_gate:
            reasons.append(
                f"R/R too low ({rr:.2f} < {rr_gate:.2f} "
                f"[regime-adjusted])")
        if rec.get("admission_confidence",
                   rec.get("confidence", 0)) < conf_gate:
            reasons.append(
                f"Confidence too low ({rec['confidence']:.1f}% < "
                f"{conf_gate:.0f}% [regime-adjusted])")
        if rec.get("expected_rise_pct", 0) < settings.MIN_EXPECTED_RISE:
            reasons.append(f"Expected rise too low ({rec['expected_rise_pct']:.2f}%)")
        if rec.get("stop_loss", 0) <= 0:
            reasons.append("Invalid stop loss")
        # v5.17: geometry coherence - a long whose stop sits at/above the
        # price it would actually pay (and the mirror for shorts) would be
        # closed by the watcher on tick 1 ("Stop Loss Hit" at ~entry).
        # compute_entry_exit now anchors limit-rec SLs to the fill price;
        # this catch-all rejects any other path that still produces an
        # incoherent rec instead of opening a self-destructing position.
        _sl = float(rec.get("stop_loss") or 0)
        _cur = float(rec.get("current_price") or 0)
        if _sl > 0 and _cur > 0:
            if rec.get("direction") == "bullish" and _sl >= _cur:
                reasons.append(
                    f"Incoherent stop (SL {_sl:.6g} >= entry {_cur:.6g} "
                    f"for a long)")
            elif rec.get("direction") == "bearish" and _sl <= _cur:
                reasons.append(
                    f"Incoherent stop (SL {_sl:.6g} <= entry {_cur:.6g} "
                    f"for a short)")
        # v5: layered harmony gate - a veteran requires layered agreement.
        # v5.6: bottom-boosted recs are EXEMPT - they carry their own layered
        # gate (bounce score >= BOTTOM_STRONG_SCORE + bullish close + RR) and
        # a bounce-derived harmony. Without the exemption the strategy-scale
        # gate rejected EVERY bottom rec (no harmony key -> 0.0 < 0.45), which
        # is why the bot never opened a position from bottom coins.
        # v5.22: momentum-channel recs are exempt for the same contract
        # reason - their admission is the strategy's binary 100% checklist.
        # v5.25: micro-scalp recs are exempt too - their admission is the
        # same binary checklist on true 15s/30s candles.
        if (not (rec.get("boosted_from_bottom")
                 or rec.get("boosted_from_momentum")
                 or rec.get("boosted_from_scalp"))
                and float(rec.get("harmony", 0.0)) < settings.MIN_HARMONY):
            reasons.append(
                f"Harmony too low ({rec.get('harmony', 0.0):.2f} "
                f"< {settings.MIN_HARMONY:.2f})"
            )
        # v5: skip chaotic candles
        if settings.EXCLUDE_VOLATILITY_EXTREME and rec.get("volatility_extreme"):
            reasons.append(
                f"Volatility extreme (ATR {rec.get('atr_pct_total', 0):.2f}% "
                f"> {settings.ATR_PCT_MAX:.2f}%)"
            )
        if rec.get("dead_market"):
            reasons.append("Dead market (ATR% below floor)")
        return (len(reasons) == 0, reasons)

    @_locked
    def open_paper_position(self, rec: Dict) -> Dict:
        """Open a paper-trading position based on a recommendation.
        Uses TRADE_AMOUNT_USD ($10 default) for position sizing.
        Applies LOT_SIZE rules from Binance and trading fees (0.1%).
        """
        # Duplicate-symbol guard runs FIRST (cheap + more specific reason)
        if settings.SKIP_DUPLICATE_SYMBOLS and self.has_open_position(rec["symbol"]):
            return {"status": "rejected",
                    "reasons": [f"{rec['symbol']} already has an open position"]}

        valid, reasons = self.validate_recommendation(rec)
        if not valid:
            return {"status": "rejected", "reasons": reasons}

        if not self.can_open_position(rec.get("symbol")):
            return {"status": "rejected", "reasons": ["Risk limits reached"]}

        # v5.20: bottom-channel clustering + tape caps (concurrent bottom
        # positions, entry spacing, bearish-BTC tide gate).
        if rec.get("boosted_from_bottom"):
            _bok, _bwhy = self._bottom_channel_ok(rec.get("symbol", ""))
            if not _bok:
                return {"status": "rejected", "reasons": [_bwhy]}
        # v5.22: momentum-channel clustering + tape caps (paper path)
        if rec.get("boosted_from_momentum"):
            _mok, _mwhy = self._momentum_channel_ok(rec.get("symbol", ""))
            if not _mok:
                return {"status": "rejected", "reasons": [_mwhy]}
        # v5.25: micro-scalp channel clustering + tape caps (paper path)
        if rec.get("boosted_from_scalp"):
            _sok, _swhy = self._scalp_channel_ok(rec.get("symbol", ""))
            if not _sok:
                return {"status": "rejected", "reasons": [_swhy]}

        entry = rec["current_price"]
        sl = rec["stop_loss"]
        # Use fixed $10 trade amount (configurable via TRADE_AMOUNT_USD)
        # v5.13: scaled by the regime size multiplier (weekend / high-vol /
        # extreme fear-greed shrink the position, strong trend may grow it)
        regime_mult = self._regime_size_multiplier()
        notional_usd = round(settings.TRADE_AMOUNT_USD * regime_mult, 2)
        # Compute quantity (USD / price), apply LOT_SIZE rules
        raw_qty = notional_usd / entry if entry > 0 else 0
        # Round to LOT_SIZE step if possible (skip for paper without API)
        try:
            from src.core.binance_client import binance_client
            size = binance_client.round_quantity_to_lot(rec["symbol"], raw_qty)
            if size <= 0:
                # Min notional check failed — use raw qty as fallback
                size = round(raw_qty, 8)
        except Exception:
            size = round(raw_qty, 8)  # Fallback: simple rounding

        # Calculate entry fee (0.1% Binance spot fee)
        entry_fee = notional_usd * (settings.TRADING_FEE_PCT / 100)

        position = {
            "symbol": rec["symbol"],
            "direction": rec["direction"],
            "entry_price": entry,
            "stop_loss": sl,
            "take_profit": rec["take_profit"],
            # v3: Fibonacci + S/R entry/exit metadata (optional fields)
            "entry_type": rec.get("entry_type", "market"),
            "entry_zone": rec.get("entry_zone"),
            "take_profit_2": rec.get("take_profit_2", rec["take_profit"]),
            "size": size,
            "notional_usd": notional_usd,
            "entry_fee": entry_fee,
            "entry_time": now_utc().isoformat(),
            "confidence": rec["confidence"],
            "expected_rise_pct": rec["expected_rise_pct"],
            # v5: veteran trade management metadata
            "harmony": float(rec.get("harmony", 0.0)),
            "atr": float(rec.get("atr", 0) or 0),
            "initial_notional_usd": notional_usd,
            "initial_size": size,
            "tp1_taken": False,
            "partial_closes": [],
            "peak_price": entry,
            "trough_price": entry,
            "mfe_pct": 0.0,
            "mae_pct": 0.0,
            "status": "open",
            "paper": True,
            # v5.11 persistent ledger identity + original levels
            "trade_uid": _new_trade_uid(),
            "initial_sl": sl,
            "initial_tp": rec["take_profit"],
            "realized_pnl": 0.0,
            "partial_count": 0,
            "strategy": _primary_strategy(rec),
            # v5.18: entry-coherence snapshot. Bottom-fishing positions are
            # OPENED while the 4h Ichimoku regime is bearish (price under the
            # cloud at a low). Without this snapshot the next cycle's
            # structural exit read "still bearish" as "flipped bearish" and
            # killed every bottom trade ~9 minutes after entry (fees won,
            # trade lost - see XAUTUSDT/CRCLBUSDT/TRXUSDT on 2026-09-27).
            "entry_ichimoku_regime": (rec.get("ichimoku") or {}).get("regime"),
            "boosted_from_bottom": bool(rec.get("boosted_from_bottom")),
            "boosted_from_momentum": bool(rec.get("boosted_from_momentum")),
            "boosted_from_scalp": bool(rec.get("boosted_from_scalp")),
        }
        self.open_positions.append(position)
        save_json(self.open_positions, POSITIONS_FILE)
        self._ensure_today_stats()
        self.daily_stats[self._today_key()]["trades_opened"] += 1
        save_json(self.daily_stats, DAILY_STATS_FILE)
        # Log to database (ledger: entry price is now immutable in the DB)
        try:
            db.log_position_opened(position)
        except Exception as e:
            log.debug(f"DB log_position_opened failed: {e}")
        log.info(
            f"[green]PAPER position opened[/] {rec['symbol']} - "
            f"size={size:.6f} entry=${entry:.4f} "
            f"SL=${sl:.4f} TP=${rec['take_profit']:.4f} "
            f"notional=${notional_usd:.2f} fee=${entry_fee:.4f}"
        )
        return {"status": "opened", "position": position}

    @_locked
    def open_live_position(self, rec: Dict) -> Dict:
        """
        Open a REAL position on Binance Spot.
        - Places MARKET BUY with quoteOrderQty = position_size * entry_price
        - After buy fills, places OCO SELL order (TP + SL)
        - Records the order IDs for tracking

        ⚠️ This places REAL orders with REAL money. Use with caution.
        """
        from src.core.binance_client import binance_client

        # Safety checks
        if settings.USE_PUBLIC_ONLY:
            return {"status": "rejected", "reasons": ["No Binance API keys configured"]}
        if settings.RUN_MODE != "live":
            return {"status": "rejected", "reasons": [f"RUN_MODE is {settings.RUN_MODE}, not 'live'"]}
        if not rec.get("direction") == "bullish":
            return {"status": "rejected", "reasons": ["Only bullish positions supported for spot"]}

        # Duplicate-symbol guard runs FIRST (cheap + more specific reason)
        if settings.SKIP_DUPLICATE_SYMBOLS and self.has_open_position(rec["symbol"]):
            return {"status": "rejected",
                    "reasons": [f"{rec['symbol']} already has an open position"]}

        valid, reasons = self.validate_recommendation(rec)
        if not valid:
            return {"status": "rejected", "reasons": reasons}
        # v5.26 bugfix: the symbol was NOT passed here (the paper path passes
        # it) so the per-symbol re-entry cooldown after a losing close was
        # silently bypassed on the LIVE path - a just-stopped-out symbol was
        # immediately re-buyable with real money.
        if not self.can_open_position(rec.get("symbol")):
            return {"status": "rejected", "reasons": ["Risk limits reached"]}

        # v5.20: bottom-channel clustering + tape caps (live path too)
        if rec.get("boosted_from_bottom"):
            _bok, _bwhy = self._bottom_channel_ok(rec.get("symbol", ""))
            if not _bok:
                return {"status": "rejected", "reasons": [_bwhy]}
        # v5.22: momentum-channel clustering + tape caps (live path too)
        if rec.get("boosted_from_momentum"):
            _mok, _mwhy = self._momentum_channel_ok(rec.get("symbol", ""))
            if not _mok:
                return {"status": "rejected", "reasons": [_mwhy]}
        # v5.25: micro-scalp channel clustering + tape caps (live path too)
        if rec.get("boosted_from_scalp"):
            _sok, _swhy = self._scalp_channel_ok(rec.get("symbol", ""))
            if not _sok:
                return {"status": "rejected", "reasons": [_swhy]}

        symbol = rec["symbol"]
        entry = rec["current_price"]
        sl = rec["stop_loss"]
        tp = rec["take_profit"]

        # Position sizing (v5.26): notional is DERIVED from the actual SL
        # distance so hitting the stop loses exactly RISK_PER_TRADE% of
        # capital (regime multiplier still applies, capped at 20%). The old
        # "risk_amount * 10" heuristic ignored the stop: a 0.5%-SL trade
        # risked ~5x the intended 1% while a 10%-SL trade risked ~0.5x -
        # real per-trade loss was 0.5-2%+ of capital depending on stop
        # width, not the nominal RISK_PER_TRADE. See position_size_notional.
        notional_usd = round(self.position_size_notional(entry, sl), 2)
        if notional_usd < 10:
            return {"status": "rejected", "reasons": [f"Notional ${notional_usd:.2f} below Binance minimum"]}

        log.info(
            f"[bold red]LIVE TRADE STARTING[/] {symbol} - "
            f"notional=${notional_usd:.2f} entry≈{entry} SL={sl} TP={tp}"
        )

        try:
            # 1) Place MARKET BUY
            buy_order = binance_client.place_market_buy(symbol, notional_usd)
            buy_order_id = buy_order.get("orderId")
            # Actual fills might differ slightly from quoteOrderQty
            executed_qty = float(buy_order.get("executedQty", 0))
            cum_quote = float(buy_order.get("cummulativeQuoteQty", notional_usd))
            avg_price = cum_quote / executed_qty if executed_qty > 0 else entry
            log.info(
                f"[green]BUY FILLED[/] {symbol} - orderId={buy_order_id} "
                f"qty={executed_qty} avg_price={avg_price:.4f}"
            )

            # 2) Place OCO SELL (Take Profit + Stop Loss)
            oco_order = None
            oco_id = None
            try:
                # Round quantity and prices to symbol's LOT_SIZE and TICK_SIZE
                filters = binance_client.get_symbol_filters(symbol)
                step = filters.get("lot_size_step", 0.00000001)
                import math
                qty_rounded = math.floor(executed_qty / step) * step
                qty_rounded = round(qq if (qq := qty_rounded) > 0 else executed_qty, 8)
                tick = filters.get("tick_size", 0.00000001)
                tp_rounded = math.floor(tp / tick) * tick
                sl_rounded = math.floor(sl / tick) * tick
                sl_limit = math.floor((sl * 0.995) / tick) * tick  # 0.5% below SL for stop-limit

                oco_order = binance_client.place_oco_sell(
                    symbol=symbol,
                    quantity=qty_rounded,
                    take_profit_price=round(tp_rounded, 8),
                    stop_loss_price=round(sl_rounded, 8),
                    stop_limit_price=round(sl_limit, 8),
                )
                oco_id = oco_order.get("orderListId")
                log.info(f"[green]OCO SELL PLACED[/] {symbol} - orderListId={oco_id}")
            except Exception as oco_err:
                log.error(
                    f"[red]OCO placement failed[/] for {symbol}: {oco_err}. "
                    f"Position is OPEN without auto-exit. Manual monitoring required!"
                )

            # 3) Record the position
            position = {
                "symbol": symbol,
                "direction": rec["direction"],
                "entry_price": float(avg_price),
                "stop_loss": float(sl),
                "take_profit": float(tp),
                # v3: Fibonacci + S/R entry/exit metadata (optional fields)
                "entry_type": rec.get("entry_type", "market"),
                "entry_zone": rec.get("entry_zone"),
                "take_profit_2": rec.get("take_profit_2", tp),
                "size": float(executed_qty),
                "notional_usd": float(cum_quote),
                "entry_time": now_utc().isoformat(),
                "confidence": rec["confidence"],
                "expected_rise_pct": rec["expected_rise_pct"],
                "status": "open",
                "paper": False,
                "buy_order_id": buy_order_id,
                "oco_order_id": oco_id,
                "oco_order_response": oco_order,
                # v5.11 persistent ledger identity + original levels
                "trade_uid": _new_trade_uid(),
                "initial_sl": float(sl),
                "initial_tp": float(tp),
                "realized_pnl": 0.0,
                "partial_count": 0,
                "strategy": _primary_strategy(rec),
                # v5.18: entry-coherence snapshot (see open_paper_position)
                "entry_ichimoku_regime": (rec.get("ichimoku") or {}).get("regime"),
                "boosted_from_bottom": bool(rec.get("boosted_from_bottom")),
                "boosted_from_momentum": bool(rec.get("boosted_from_momentum")),
                "boosted_from_scalp": bool(rec.get("boosted_from_scalp")),
            }
            self.open_positions.append(position)
            save_json(self.open_positions, POSITIONS_FILE)
            self._ensure_today_stats()
            self.daily_stats[self._today_key()]["trades_opened"] += 1
            save_json(self.daily_stats, DAILY_STATS_FILE)
            # v5.11 bugfix: LIVE positions were never written to the DB -
            # a restart erased them completely. The ledger now records
            # real fills (avg entry price) exactly like paper trades.
            try:
                db.log_position_opened(position)
            except Exception as e:
                log.debug(f"DB log_position_opened (live) failed: {e}")

            return {"status": "opened", "position": position, "buy_order": buy_order, "oco_order": oco_order}

        except Exception as e:
            log.exception(f"[red]LIVE order failed[/] for {symbol}: {e}")
            return {"status": "error", "reasons": [str(e)]}

    def open_position(self, rec: Dict) -> Dict:
        """
        Open a position based on current RUN_MODE.
        - paper: opens virtual paper position
        - live: places REAL order on Binance
        """
        if settings.RUN_MODE == "live":
            return self.open_live_position(rec)
        return self.open_paper_position(rec)

    @_locked
    def close_position(self, idx: int, exit_price: float, reason: str = "",
                       fraction: float = 1.0) -> Dict:
        """Close an open position at the given exit price.

        v5: `fraction < 1.0` closes a PARTIAL chunk (e.g. TP1 banking).
        - P&L is realized on the closed fraction only
        - the position stays open with reduced notional/size
        - partial closes do NOT touch the win/loss streak counters (only
          full closes do) - they only add realized P&L

        v5.26: idx is only meaningful while the lock is held (it always is
        now - this method and every caller of it are @_locked). External
        callers that hold a position identity across an await/I-O gap must
        use close_position_by_uid instead of caching an index.
        """
        if idx >= len(self.open_positions):
            return {"status": "error", "reason": "Invalid index"}
        pos = self.open_positions[idx]
        fraction = max(0.0, min(1.0, float(fraction)))
        partial = fraction < 0.999
        if partial and fraction * (pos.get("notional_usd") or 0) < 1.0:
            # too small to be worth a partial close - close fully instead
            fraction = 1.0
            partial = False

        entry_price = pos["entry_price"]
        notional_full = pos.get("notional_usd", pos.get("size", 0) * entry_price)
        notional_chunk = notional_full * fraction
        entry_fee = pos.get("entry_fee", 0) * fraction

        # Calculate exit value (proportional to entry)
        if entry_price > 0:
            exit_value = notional_chunk * (exit_price / entry_price)
        else:
            exit_value = notional_chunk

        # Exit fee (0.1% of exit value)
        exit_fee = exit_value * (settings.TRADING_FEE_PCT / 100)
        # Net P&L = exit_value - entry_value - entry_fee - exit_fee
        gross_pnl = exit_value - notional_chunk
        net_pnl = gross_pnl - entry_fee - exit_fee
        pnl_pct = (net_pnl / notional_chunk * 100) if notional_chunk > 0 else 0

        if partial:
            # ---- v5 partial close: shrink position, keep it open ----
            pos["notional_usd"] = notional_full - notional_chunk
            pos["size"] = pos.get("size", 0) * (1.0 - fraction)
            pos["entry_fee"] = pos.get("entry_fee", 0) - entry_fee
            pos.setdefault("partial_closes", []).append({
                "time": now_utc().isoformat(),
                "price": exit_price,
                "fraction": fraction,
                "pnl": net_pnl,
                "pnl_pct": pnl_pct,
                "reason": reason,
            })
            # v5.11: accumulate the realized profit ON the trade itself -
            # the final close must report the WHOLE-trade result
            # (partials + final chunk), not just the last chunk.
            pos["realized_pnl"] = float(pos.get("realized_pnl", 0) or 0) + net_pnl
            pos["partial_count"] = len(pos.get("partial_closes", []))
            save_json(self.open_positions, POSITIONS_FILE)
            self._ensure_today_stats()
            self.daily_stats[self._today_key()]["pnl"] += net_pnl
            self.daily_stats[self._today_key()]["partials"] = (
                self.daily_stats[self._today_key()].get("partials", 0) + 1
            )
            save_json(self.daily_stats, DAILY_STATS_FILE)
            # v5.11 ledger: partial closes ACCUMULATE realized_pnl and keep
            # status='open'. The old code wrote exit_time here, so the DB
            # counted the trade as closed and the final close ERASED the
            # TP1 profit by overwriting the same row.
            try:
                db.record_partial_close(
                    trade_uid=pos.get("trade_uid"),
                    symbol=pos["symbol"],
                    entry_time=pos["entry_time"],
                    remaining_notional=pos["notional_usd"],
                    remaining_size=pos.get("size", 0),
                    realized_total=pos["realized_pnl"],
                    partial_count=pos["partial_count"],
                    tp1_taken=bool(pos.get("tp1_taken")),
                    stop_loss=pos["stop_loss"],
                    take_profit=pos["take_profit"],
                    exit_price=exit_price,
                    exit_time=now_utc().isoformat(),
                    chunk_pnl=net_pnl,
                    chunk_pnl_pct=pnl_pct,
                    fraction=fraction,
                    reason=reason,
                )
                db.update_daily_stats(date=self._today_key(), pnl_delta=net_pnl)
            except Exception as e:
                log.debug(f"DB record_partial_close failed: {e}")
            log.info(
                f"[green]PARTIAL close ({fraction*100:.0f}%)[/] {pos['symbol']} - "
                f"Net: ${net_pnl:+.4f} ({pnl_pct:+.2f}%) - {reason} | "
                f"remaining notional ${pos['notional_usd']:.2f}"
            )
            return {"status": "partial", "position": pos, "pnl": net_pnl,
                    "pnl_pct": pnl_pct, "reason": reason, "fraction": fraction}

        # ---- full close (original path, fraction == 1.0) ----
        # v5.11: WHOLE-trade accounting - the final chunk PLUS everything
        # already banked by partial closes (TP1). This is the number that
        # goes to the DB, Telegram and the stats engine.
        realized_prev = float(pos.get("realized_pnl", 0) or 0)
        trade_total_pnl = realized_prev + net_pnl
        initial_notional = float(pos.get("initial_notional_usd")
                                 or notional_full or 0)
        trade_total_pct = (trade_total_pnl / initial_notional * 100) \
            if initial_notional > 0 else pnl_pct
        mfe_val = float(pos.get("mfe_pct", 0) or 0)
        capture_eff = None
        if mfe_val > 0 and initial_notional > 0:
            capture_eff = max(-200.0, min(200.0,
                (trade_total_pnl / initial_notional * 100) / mfe_val * 100))
        closed = {**pos, "exit_price": exit_price, "exit_time": now_utc().isoformat(),
                  "pnl": net_pnl, "pnl_pct": pnl_pct, "reason": reason,
                  "status": "closed", "entry_fee": entry_fee, "exit_fee": exit_fee,
                  "gross_pnl": gross_pnl,
                  # v5.11 whole-trade result + autopsy metrics
                  "trade_uid": pos.get("trade_uid"),
                  "total_pnl": trade_total_pnl,
                  "total_pnl_pct": trade_total_pct,
                  "capture_efficiency": capture_eff,
                  "duration_hours": round(self._position_age_hours(pos), 2),
                  "partials_count": len(pos.get("partial_closes", [])),
                  "risk_updates_count": len(pos.get("risk_updates", []))}
        self.open_positions.pop(idx)
        save_json(self.open_positions, POSITIONS_FILE)
        # Update daily stats
        self._ensure_today_stats()
        self.daily_stats[self._today_key()]["pnl"] += net_pnl
        if net_pnl > 0:
            self.daily_stats[self._today_key()]["wins"] += 1
            # v4.1: winning close resets the loss streak
            self._loss_streak = 0
        else:
            self.daily_stats[self._today_key()]["losses"] += 1
            # v4.1: track consecutive losses -> circuit breaker + re-entry cooldown
            self._loss_streak = self.loss_streak + 1
            self._reentry_block[pos["symbol"]] = now_utc() + timedelta(
                hours=settings.REENTRY_COOLDOWN_HOURS
            )
            log.info(
                f"[yellow]Re-entry cooldown[/] {pos['symbol']} for "
                f"{settings.REENTRY_COOLDOWN_HOURS}h after a losing close"
            )
            if self._loss_streak >= settings.LOSS_STREAK_LIMIT:
                self._set_loss_pause()
            else:
                self._persist_loss_state()
        save_json(self.daily_stats, DAILY_STATS_FILE)
        # Log to database (v5.11 ledger: close by trade_uid, whole-trade PnL)
        try:
            db.close_trade(
                trade_uid=pos.get("trade_uid"),
                symbol=pos["symbol"],
                entry_time=pos["entry_time"],
                exit_price=exit_price,
                exit_time=closed["exit_time"],
                final_pnl=net_pnl,
                final_pnl_pct=pnl_pct,
                total_pnl=trade_total_pnl,
                total_pnl_pct=trade_total_pct,
                close_reason=reason,
                exit_fee=exit_fee,
                peak_price=pos.get("peak_price"),
                trough_price=pos.get("trough_price"),
                mfe_pct=pos.get("mfe_pct"),
                mae_pct=pos.get("mae_pct"),
            )
            # Update daily stats in DB (v5.11 fix: pnl_delta was always 0,
            # so the DB daily P&L never moved)
            db.update_daily_stats(
                date=self._today_key(),
                trades_opened=self.daily_stats[self._today_key()]["trades_opened"],
                wins=self.daily_stats[self._today_key()]["wins"],
                losses=self.daily_stats[self._today_key()]["losses"],
                pnl_delta=net_pnl,
            )
        except Exception as e:
            log.debug(f"DB close_trade failed: {e}")
        log.info(
            f"[yellow]Position closed[/] {pos['symbol']} - "
            f"chunk ${net_pnl:+.4f} + partials ${realized_prev:+.4f} = "
            f"TOTAL ${trade_total_pnl:+.4f} ({trade_total_pct:+.2f}%) - "
            f"fees ${entry_fee+exit_fee:.4f} - {reason}"
        )
        return closed

    # ============================================
    # v5: EXCURSION TRACKING + TIME STOP + STRUCTURE EXITS
    # ============================================

    @staticmethod
    def _position_age_hours(pos: Dict) -> float:
        try:
            entered = datetime.fromisoformat(pos["entry_time"])
            if entered.tzinfo is None:
                entered = entered.replace(tzinfo=timezone.utc)
            return (now_utc() - entered).total_seconds() / 3600.0
        except Exception:
            return 0.0

    @staticmethod
    def _position_age_seconds(pos: Dict) -> float:
        """v5.25: seconds-precision age for the micro-scalp fixed hold."""
        try:
            entered = datetime.fromisoformat(pos["entry_time"])
            if entered.tzinfo is None:
                entered = entered.replace(tzinfo=timezone.utc)
            return (now_utc() - entered).total_seconds()
        except Exception:
            return 0.0

    @staticmethod
    def track_excursions(pos: Dict, price: float) -> None:
        """v5: update peak/trough + MFE/MAE since entry (in-place, cheap)."""
        if not price or not pos.get("entry_price"):
            return
        entry = pos["entry_price"]
        peak = max(float(pos.get("peak_price") or entry), price)
        trough = min(float(pos.get("trough_price") or entry), price)
        pos["peak_price"] = peak
        pos["trough_price"] = trough
        if pos.get("direction") == "bullish":
            pos["mfe_pct"] = max(pos.get("mfe_pct", 0.0),
                                 (peak - entry) / entry * 100)
            pos["mae_pct"] = max(pos.get("mae_pct", 0.0),
                                 (entry - trough) / entry * 100)
        else:
            pos["mfe_pct"] = max(pos.get("mfe_pct", 0.0),
                                 (entry - trough) / entry * 100)
            pos["mae_pct"] = max(pos.get("mae_pct", 0.0),
                                 (peak - entry) / entry * 100)

    def _check_time_stop(self, pos: Dict, pnl_pct: float) -> Optional[str]:
        """v5: a trade that goes nowhere is dead capital.
        v5.20: bottom-channel STAGNATION exit. The 72h stale-trade horizon
        fits trend trades; a bounce that has not appeared within 6h with
        pnl < 0.2% and MFE < 0.6% never worked (INTCB: -2.59% over 11.6h
        with MFE 0.22%; ZAMA: -5.71% with MFE 0.11%). Close early, recycle
        the slot, keep the loss small.
        v5.25: micro-scalp FIXED holding window - the document's 1-minute
        expiry. Seconds-precision: the watcher ticks every minute, so a
        scalp closes on the first tick past SCALP_HOLD_SECONDS (60-120s
        real holding). The only earlier door is the disaster SL (checked
        before this in check_open_positions)."""
        if pos.get("boosted_from_scalp"):
            age_s = self._position_age_seconds(pos)
            hold_s = float(max(1, settings.SCALP_HOLD_SECONDS))
            if age_s >= hold_s:
                return (f"Scalp time exit: fixed {hold_s:.0f}s holding "
                        f"completed ({age_s:.0f}s, pnl {pnl_pct:+.2f}%)")
            return None
        age_h = self._position_age_hours(pos)
        if age_h >= settings.ABSOLUTE_MAX_TRADE_HOURS:
            return (f"Max holding time reached "
                    f"({age_h:.1f}h >= {settings.ABSOLUTE_MAX_TRADE_HOURS:.0f}h)")
        if (age_h >= settings.MAX_TRADE_HOURS
                and pnl_pct < settings.TIME_STOP_MIN_PNL_PCT):
            return (f"Time stop: stale trade ({age_h:.1f}h, "
                    f"{pnl_pct:+.2f}% < {settings.TIME_STOP_MIN_PNL_PCT:.2f}%)")
        stag_h = float(getattr(settings, "BOTTOM_STAGNATION_HOURS", 0) or 0)
        if (pos.get("boosted_from_bottom") and stag_h > 0
                and age_h >= stag_h
                and pnl_pct < settings.BOTTOM_STAGNATION_MAX_PNL_PCT
                and float(pos.get("mfe_pct") or 0.0)
                    < settings.BOTTOM_STAGNATION_MAX_MFE_PCT):
            return (f"Bottom stagnation exit ({age_h:.1f}h, "
                    f"pnl {pnl_pct:+.2f}%, MFE "
                    f"{float(pos.get('mfe_pct') or 0.0):.2f}% - "
                    f"no bounce appeared, recycling capital")
        return None

    @staticmethod
    def _in_structural_grace(pos: Dict) -> bool:
        """v5.18: True within STRUCTURAL_EXIT_GRACE_MIN of entry_time.

        Regime/Kijun/Tenkan structural exits are suppressed during the
        window; the hard SL and the >=55 opposite-signal exit stay armed.
        Bottom-fishing entries need a few cycles before the 4h regime is
        meaningful again - an instant next-cycle exit is churn, not risk
        management.
        """
        grace_min = float(
            getattr(settings, "STRUCTURAL_EXIT_GRACE_MIN", 30) or 0)
        if grace_min <= 0:
            return False
        try:
            et = pos.get("entry_time")
            if not et:
                return False
            entry_dt = datetime.fromisoformat(str(et))
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            return (now_utc() - entry_dt).total_seconds() < grace_min * 60.0
        except Exception:
            return False

    def evaluate_structural_exit(self, pos: Dict, sig: Dict,
                                 current_price: float) -> Tuple[str, Optional[str]]:
        """
        v5 market-aware exit decision from the latest analysis of this symbol.
        Returns (action, reason) where action is:
          "exit"        -> close the whole position
          "tighten"     -> raise the SL to pos["_structural_sl"] (set by caller)
          "none"
        Only called from the main analysis cycle (klines-backed signals).

        v5.5 graduated opposite-signal response (user rule: when the analysis
        of the trade turns bearish on a long, CLOSE IT NOW - never wait for
        the stop to be hit, and never "tighten" into a locked loss):
          conf >= OPPOSITE_SIGNAL_CONF (55)   -> exit immediately
          conf >= SIGNAL_TIGHTEN_CONF (40)    -> defend: SL 0.5% below price
        """
        if not settings.STRUCTURAL_EXITS_ENABLED or not sig:
            return ("none", None)
        # v5.25: micro-scalp positions have a FIXED holding window (the
        # document's 1-minute expiry). No structural/signal exit applies -
        # the watcher's disaster SL and the scalp time exit are the only
        # doors; an analysis-cycle "flip" on 4h candles is meaningless for
        # a trade that lives 60 seconds.
        if pos.get("boosted_from_scalp"):
            return ("none", None)
        icho = sig.get("ichimoku") or {}
        direction = pos.get("direction", "bullish")

        # 1) Opposite analysis signal -> graduated response
        sig_dir = sig.get("direction")
        sig_conf = float(sig.get("confidence", 0) or 0)
        if (sig_dir and sig_dir != direction
                and sig_dir in ("bullish", "bearish")):
            if sig_conf >= settings.OPPOSITE_SIGNAL_CONF:
                return ("exit",
                        f"Opposite {sig_dir} signal (conf {sig_conf:.0f}% >= "
                        f"{settings.OPPOSITE_SIGNAL_CONF:.0f}%) - closing now")
            if sig_conf >= settings.SIGNAL_TIGHTEN_CONF:
                if direction == "bullish":
                    level = current_price * 0.995
                else:
                    level = current_price * 1.005
                return ("tighten",
                        f"Opposite {sig_dir} pressure defence "
                        f"({level:.4f})")

        if not icho:
            return ("none", None)
        regime = icho.get("regime")
        kijun = icho.get("kijun")
        tenkan = icho.get("tenkan")

        # v5.18: entry-coherence guards. A structural regime exit must mean
        # "the thesis DIED", not "the thesis was never born": the regime has
        # to differ from the entry snapshot, and the grace window right
        # after entry suppresses regime/Kijun/Tenkan reactions entirely.
        in_grace = self._in_structural_grace(pos)
        entry_regime = pos.get("entry_ichimoku_regime")

        if direction == "bullish":
            # 2) Ichimoku regime flipped bearish -> thesis dead, exit.
            #    v5.18: only a genuine FLIP vs the entry regime exits - a
            #    bottom-fishing position opened under a bearish 4h regime
            #    (price below the cloud at the low) no longer dies on the
            #    next cycle just because the regime is still bearish.
            #    v5.19 PRODUCTION FORENSICS: that guard was dead code for
            #    the only strategy that trades - build_bottom_rec carries
            #    no ichimoku field, so entry_ichimoku_regime was always
            #    None, and None != "bearish" made EVERY bearish reading a
            #    "flip". All 16 closed trades (bottom_scanner_boost) died
            #    with this exact reason 30-40 min after entry = the grace
            #    expiry, winners included (INJ +1.63%, UNI +1.02% were
            #    killed mid-profit). A bottom is a PRICE thesis, not a
            #    trend thesis: price under the 4h cloud is its NATURAL
            #    state, so the regime label can never be its kill switch.
            #    Bottom trades now exit on: hard SL (1.2 ATR under entry,
            #    below the swing low), opposite signal >= 55 (still armed
            #    above), time stop, TP ladder - never on the cloud label.
            if regime == "bearish":
                if pos.get("boosted_from_bottom"):
                    return ("none", None)
                if entry_regime != "bearish" and not in_grace:
                    return ("exit", "Ichimoku regime flipped bearish")
                return ("none", None)
            # 3) Regime decayed to neutral + price lost Kijun -> tighten to Kijun
            if (regime == "neutral" and icho.get("price_vs_kijun") == "below"
                    and kijun and not in_grace):
                if kijun < current_price:
                    return ("tighten", f"Kijun defence ({kijun:.4f})")
            # 4) Fresh bearish TK cross -> tighten to Tenkan
            if (not in_grace
                    and icho.get("tk_cross_recent") == "bearish"
                    and icho.get("tk_state") == "bearish" and tenkan
                    and tenkan < current_price):
                return ("tighten", f"Tenkan cross-down defence ({tenkan:.4f})")
        else:
            # v5.19: mirror immunity for (future) short bottoms - same
            # price-thesis-not-trend-thesis semantics.
            if regime == "bullish":
                if pos.get("boosted_from_bottom"):
                    return ("none", None)
                if entry_regime != "bullish" and not in_grace:
                    return ("exit", "Ichimoku regime flipped bullish")
                return ("none", None)
            if (regime == "neutral" and icho.get("price_vs_kijun") == "above"
                    and kijun and not in_grace):
                if kijun > current_price:
                    return ("tighten", f"Kijun defence ({kijun:.4f})")
            if (not in_grace
                    and icho.get("tk_cross_recent") == "bullish"
                    and icho.get("tk_state") == "bullish" and tenkan
                    and tenkan > current_price):
                return ("tighten", f"Tenkan cross-up defence ({tenkan:.4f})")
        return ("none", None)

    @_locked
    def check_open_positions(self, prices: Dict[str, float]) -> List[Dict]:
        """Check open positions: SL / TP1-partial / TP2 / time stop.

        v5 veteran flow per position (bullish shown; bearish mirrored):
          1. MFE/MAE excursion tracking (peak/trough since entry)
          2. Hard SL hit -> full close
          3. Time stop (stale trade) -> full close
          4. TP1 not taken yet and price >= TP1:
             -> bank PARTIAL_TP_FRACTION at TP1, SL -> break-even+fees,
                promote TP to TP2 when it extends further
          5. Remaining runner hits the (possibly promoted) TP -> full close
        Returns list of full-close dicts (partials are returned too, tagged).
        """
        results = []
        for i in range(len(self.open_positions) - 1, -1, -1):
            pos = self.open_positions[i]
            price = prices.get(pos["symbol"])
            if not price:
                continue

            self.track_excursions(pos, price)

            direction = pos["direction"]
            if direction == "bullish":
                profit_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
                sl_hit = price <= pos["stop_loss"]
                tp_hit = price >= pos["take_profit"]
            else:
                profit_pct = (pos["entry_price"] - price) / pos["entry_price"] * 100
                sl_hit = price >= pos["stop_loss"]
                tp_hit = price <= pos["take_profit"]

            # 1) hard stop first (priority over everything)
            if sl_hit:
                results.append(self.close_position(
                    i, pos["stop_loss"], "Stop Loss Hit"))
                continue

            # 2) time stop (uses live pnl)
            stop_reason = self._check_time_stop(pos, profit_pct)
            if stop_reason:
                results.append(self.close_position(i, price, stop_reason))
                continue

            # 3) TP ladder: partial at TP1, runner to TP2
            if tp_hit and settings.PARTIAL_TP_ENABLED and not pos.get("tp1_taken"):
                fraction = max(0.1, min(0.9, settings.PARTIAL_TP_FRACTION))
                partial_res = self.close_position(
                    i, pos["take_profit"], "TP1 Partial", fraction=fraction)
                if partial_res.get("status") == "partial":
                    pos = self.open_positions[i]  # refreshed after partial
                    pos["tp1_taken"] = True
                    pre_sl = pos["stop_loss"]
                    pre_tp = pos["take_profit"]
                    # SL to break-even + fee buffer (never turns a winner red)
                    entry = pos["entry_price"]
                    buf = settings.TP1_FEE_BUFFER_PCT / 100.0
                    be_sl = (entry * (1 + buf) if direction == "bullish"
                             else entry * (1 - buf))
                    current_sl = pos["stop_loss"]
                    if (direction == "bullish" and be_sl > current_sl) or \
                       (direction == "bearish" and be_sl < current_sl):
                        pos["stop_loss"] = be_sl
                    # promote runner target to TP2 when it extends beyond TP1
                    tp2 = pos.get("take_profit_2")
                    if isinstance(tp2, (int, float)) and tp2:
                        if direction == "bullish" and tp2 > pos["take_profit"]:
                            pos["take_profit"] = tp2
                        elif direction == "bearish" and tp2 < pos["take_profit"]:
                            pos["take_profit"] = tp2
                    save_json(self.open_positions, POSITIONS_FILE)
                    # v5.11: the TP1 level promotion is persisted to the
                    # ledger too (stop_loss/take_profit columns + event)
                    self._sync_levels(
                        pos, pos["take_profit"], old_sl=pre_sl, old_tp=pre_tp,
                        reason="TP1: SL->breakeven+fees, TP->TP2",
                        event_type="TP1_LEVELS")
                    log.info(
                        f"[blue]Break-even lock[/] {pos['symbol']} "
                        f"SL -> {pos['stop_loss']:.4f} | "
                        f"runner TP -> {pos['take_profit']:.4f}"
                    )
                    results.append(partial_res)
                    # same tick may already reach the promoted runner TP
                    price = prices.get(pos["symbol"])
                    if not price:
                        continue
                    if direction == "bullish":
                        sl_hit = price <= pos["stop_loss"]
                        tp_hit = price >= pos["take_profit"]
                    else:
                        sl_hit = price >= pos["stop_loss"]
                        tp_hit = price <= pos["take_profit"]
                    if sl_hit:
                        results.append(self.close_position(
                            i, pos["stop_loss"], "Stop Loss Hit (BE)"))
                        continue
                    if tp_hit:
                        results.append(self.close_position(
                            i, pos["take_profit"], "Take Profit 2 Hit"))
                        continue
                elif partial_res.get("status") == "error":
                    continue

            elif tp_hit:
                # TP1 already taken (or partials disabled) -> final target
                results.append(self.close_position(
                    i, pos["take_profit"], "Take Profit Hit"))
        return results

    # ============================================
    # DYNAMIC SL/TP UPDATE (Trailing Stop + Break-Even)
    # ============================================

    @_locked
    def update_position_risk(self, idx: int, current_price: float,
                              new_sl: float = None, new_tp: float = None,
                              reason: str = "") -> Dict:
        """
        Update SL and/or TP for an open position.
        Rules (safety):
          - For LONG: new_sl must be > current_sl (only tighten, never loosen)
                       new_tp can be higher (extend) or equal (no change)
          - Records change in position["risk_updates"] history
        """
        if idx >= len(self.open_positions):
            return {"status": "error", "reason": "Invalid index"}
        pos = self.open_positions[idx]
        updates = pos.get("risk_updates", [])
        old_sl = pos["stop_loss"]
        old_tp = pos["take_profit"]

        # Safety rules
        if new_sl is not None:
            if pos["direction"] == "bullish" and new_sl <= old_sl:
                log.warning(f"Skipping SL update for {pos['symbol']}: new SL {new_sl} <= old SL {old_sl} (cannot loosen)")
                new_sl = None
            elif pos["direction"] == "bearish" and new_sl >= old_sl:
                log.warning(f"Skipping SL update for {pos['symbol']}: new SL {new_sl} >= old SL {old_sl}")
                new_sl = None

        if new_sl is not None or new_tp is not None:
            update_record = {
                "timestamp": now_utc().isoformat(),
                "price_at_update": current_price,
                "old_sl": old_sl,
                "new_sl": new_sl if new_sl is not None else old_sl,
                "old_tp": old_tp,
                "new_tp": new_tp if new_tp is not None else old_tp,
                "reason": reason,
            }
            updates.append(update_record)
            pos["risk_updates"] = updates
            if new_sl is not None:
                pos["stop_loss"] = new_sl
            if new_tp is not None:
                pos["take_profit"] = new_tp
            save_json(self.open_positions, POSITIONS_FILE)
            # v5.11: EVERY SL/TP update is mirrored into the database
            # (positions row + trade_events audit row). This is the piece
            # that was completely missing before - on Render the JSON file
            # is wiped on every redeploy and the updated levels were lost.
            self._sync_levels(pos, current_price, old_sl=old_sl, old_tp=old_tp,
                              reason=reason, event_type="RISK_UPDATE")
            log.info(
                f"[blue]Risk update[/] {pos['symbol']} - "
                f"SL: {old_sl:.4f} -> {pos['stop_loss']:.4f} | "
                f"TP: {old_tp:.4f} -> {pos['take_profit']:.4f} - {reason}"
            )
            return {"status": "updated", "position": pos, "update": update_record}
        return {"status": "no_change"}

    @_locked
    def has_open_position(self, symbol: str) -> bool:
        """Check if a position is already open for the given symbol."""
        return any(p.get("symbol") == symbol for p in self.open_positions)

    @_locked
    def index_of_uid(self, trade_uid: str) -> int:
        """v5.26: fresh index of an open position by its ledger uid (-1 if
        gone). ALWAYS re-resolve right before an index-based mutation when
        the position identity was captured across any I/O or scheduling gap
        - the watcher may have popped the list in between."""
        if not trade_uid:
            return -1
        for i, p in enumerate(self.open_positions):
            if p.get("trade_uid") == trade_uid:
                return i
        return -1

    @_locked
    def close_position_by_uid(self, trade_uid: str, exit_price: float,
                              reason: str = "", fraction: float = 1.0) -> Dict:
        """v5.26: uid-addressed close (the safe public entry point for
        callers outside the lock scope, e.g. the structural-exit loop in
        cycle.py). Resolves the index FRESH under the lock, so a concurrent
        watcher close can never make a stale index close the WRONG
        position; returns status=error when the position is already gone."""
        idx = self.index_of_uid(trade_uid)
        if idx < 0:
            return {"status": "error",
                    "reason": f"Position {trade_uid} not found (already closed?)"}
        return self.close_position(idx, exit_price, reason, fraction=fraction)

    @_locked
    def update_position_risk_by_uid(self, trade_uid: str,
                                     current_price: float, new_sl: float = None,
                                     new_tp: float = None,
                                     reason: str = "") -> Dict:
        """v5.26: uid-addressed SL/TP update (same contract as
        close_position_by_uid)."""
        idx = self.index_of_uid(trade_uid)
        if idx < 0:
            return {"status": "error", "reason": "Position not found"}
        return self.update_position_risk(idx, current_price, new_sl, new_tp,
                                         reason)

    @_locked
    def clear_all_state(self):
        """v5.26: locked wipe of the in-memory trading state (used by
        /api/reset-history). The old direct `risk_manager.open_positions = []`
        assignment from a request thread raced every other context AND left
        daily_stats/_reentry_block/loss-streak intact in memory, so the next
        save_json silently resurrected the pre-reset stats from RAM.
        Mirrors exactly what the endpoint deletes from disk (positions,
        daily stats, loss state); pending entries are kept (they are future
        orders, not history - unchanged endpoint behavior)."""
        self.open_positions = []
        self._reentry_block = {}
        self.daily_stats = {}
        self._loss_streak = 0
        self._loss_pause_until = None
        log.warning("[red]RiskManager in-memory state cleared[/] "
                    "(positions + daily stats + loss state)")

    # ============================================
    # v5.11: PERSISTENT TRADE LEDGER (source of truth = DB)
    # ============================================

    def _sync_levels(self, pos: Dict, price: float, old_sl: float = None,
                     old_tp: float = None, reason: str = "",
                     event_type: str = "RISK_UPDATE"):
        """Mirror the current SL/TP (+ excursion snapshot) into the ledger.
        Best-effort: a DB hiccup must never break trade management."""
        try:
            db.update_position_levels(
                trade_uid=pos.get("trade_uid"),
                symbol=pos.get("symbol"),
                entry_time=pos.get("entry_time"),
                stop_loss=pos.get("stop_loss"),
                take_profit=pos.get("take_profit"),
                old_sl=old_sl, old_tp=old_tp,
                price=price, reason=reason, event_type=event_type,
                peak_price=pos.get("peak_price"),
                trough_price=pos.get("trough_price"),
                mfe_pct=pos.get("mfe_pct"),
                mae_pct=pos.get("mae_pct"),
                notional_usd=pos.get("notional_usd"),
                size=pos.get("size"),
                tp1_taken=pos.get("tp1_taken") if pos.get("tp1_taken") else None,
            )
        except Exception as e:
            log.debug(f"Ledger level sync failed: {e}")

    @staticmethod
    def _position_from_row(row: Dict) -> Dict:
        """Rebuild a manageable position dict from a ledger row."""
        entry = float(row.get("entry_price") or 0)
        sl = float(row.get("stop_loss") or 0)
        tp = float(row.get("take_profit") or 0)
        tp2 = row.get("tp2") or tp
        return {
            "symbol": row.get("symbol"),
            "direction": row.get("direction") or "bullish",
            "entry_price": entry,
            "stop_loss": sl,
            "take_profit": tp,
            "take_profit_2": float(tp2) if tp2 else tp,
            "size": float(row.get("size") or 0),
            "notional_usd": float(row.get("notional_usd") or 0),
            "initial_notional_usd": float(
                row.get("initial_notional") or row.get("notional_usd") or 0),
            "entry_fee": float(row.get("entry_fee") or 0),
            "entry_time": row.get("entry_time"),
            "confidence": float(row.get("confidence") or 0),
            "status": "open",
            "paper": bool(row.get("paper", 1)),
            "trade_uid": row.get("trade_uid"),
            "initial_sl": row.get("initial_sl") if row.get("initial_sl") is not None else sl,
            "initial_tp": row.get("initial_tp") if row.get("initial_tp") is not None else tp,
            "tp1_taken": bool(row.get("tp1_taken")),
            "realized_pnl": float(row.get("realized_pnl") or 0),
            "partial_count": int(row.get("partial_count") or 0),
            "partial_closes": [],
            "peak_price": float(row.get("peak_price") or entry),
            "trough_price": float(row.get("trough_price") or entry),
            "mfe_pct": float(row.get("mfe_pct") or 0),
            "mae_pct": float(row.get("mae_pct") or 0),
            "strategy": row.get("strategy") or "",
            # v5.20: the ledger has no dedicated column for this flag, but
            # bottom-boost trades are the only ones whose _primary_strategy
            # is "bottom_scanner_boost". Restoring it matters: without the
            # flag a restart (Render redeploys constantly) silently strips
            # the v5.19 structural-exit immunity and disables the v5.20
            # stagnation exit / clustering caps for surviving positions.
            # v5.22: same contract for the momentum channel (strategy name
            # "double_indicator") - flag restore survives redeploys too.
            "boosted_from_bottom": (
                (row.get("strategy") or "") == "bottom_scanner_boost"),
            "boosted_from_momentum": (
                (row.get("strategy") or "") == "double_indicator"),
            # v5.25: same contract for the micro-scalp channel - a scalp
            # position restored after a redeploy keeps its fixed-hold time
            # exit and structural immunity.
            "boosted_from_scalp": (
                (row.get("strategy") or "") == "micro_scalp"),
            "risk_updates": [],
            "restored_from_db": True,
        }

    def _restore_from_db(self):
        """v5.11: reconcile open positions with the persistent ledger.

        Direction 1 - JSON lost (Render redeploy wiped the disk):
            DB rows with status='open' are restored, entry prices intact.
        Direction 2 - DB lost/empty (fresh DB or rotated credentials):
            JSON positions are written INTO the ledger so they survive.
        Direction 3 - legacy JSON positions without trade_uid:
            they get a uid and their existing DB row is tagged (or a row is
            created).
        """
        if not settings.TRADE_LEDGER_RESTORE:
            return
        try:
            db_rows = db.get_open_positions() or []
        except Exception as e:
            log.debug(f"Ledger restore unavailable: {e}")
            return
        changed = False
        json_uids = {p.get("trade_uid") for p in self.open_positions
                     if p.get("trade_uid")}
        json_symbols = {p.get("symbol") for p in self.open_positions}
        db_uids = {r.get("trade_uid") for r in db_rows if r.get("trade_uid")}

        # --- 1) legacy JSON positions: tag or insert into the ledger ---
        for p in self.open_positions:
            if p.get("trade_uid"):
                continue
            uid = _new_trade_uid()
            p["trade_uid"] = uid
            p.setdefault("realized_pnl", 0.0)
            p.setdefault("partial_count", 0)
            changed = True
            try:
                if (db.attach_trade_uid(p.get("symbol", ""),
                                        p.get("entry_time", ""), uid) or 0) < 1:
                    db.log_position_opened(p)
                db_uids.add(uid)
            except Exception as e:
                log.debug(f"Ledger tag/insert failed: {e}")

        # --- 2) uid-bearing JSON positions missing from the DB ---
        for p in self.open_positions:
            uid = p.get("trade_uid")
            if uid and uid not in db_uids:
                try:
                    db.log_position_opened(p)
                    db_uids.add(uid)
                    changed = True
                    log.info(
                        f"[blue]Ledger backfill[/] {p.get('symbol')} "
                        f"written to DB (was missing)")
                except Exception as e:
                    log.debug(f"Ledger backfill failed: {e}")

        # --- 3) DB open positions missing from JSON: RESTORE them ---
        restored = 0
        for r in db_rows:
            uid = r.get("trade_uid")
            if not uid or uid in json_uids or r.get("symbol") in json_symbols:
                continue
            try:
                pos = self._position_from_row(r)
            except Exception as e:
                log.debug(f"Restore parse failed for {r.get('symbol')}: {e}")
                continue
            self.open_positions.append(pos)
            json_symbols.add(pos["symbol"])
            json_uids.add(uid)
            restored += 1
        if restored:
            changed = True
            log.warning(
                f"[yellow]Trade-ledger restore[/]: {restored} open "
                f"position(s) recovered from the database "
                f"(entry prices preserved)")
        if changed:
            save_json(self.open_positions, POSITIONS_FILE)

    @_locked
    def apply_trailing_logic(self, current_prices: Dict[str, float],
                              market_signals: Dict[str, Dict] = None) -> List[Dict]:
        """
        Apply dynamic SL/TP adjustment based on price movement and market signals.

        v5 veteran trailing (LONG positions) - two cooperating mechanisms,
        the most protective valid level wins:
          1. Ladder (unchanged): +1% -> BE, +2% -> +1%, +3% -> +2%,
             +5% -> trail 1% below price
          2. ATR chandelier: SL trails CHANDELIER_ATR_MULT x ATR below the
             highest price seen since entry (peak_price) once profit >= 1%.
             Market-adaptive: wide in trends, tight in chop.
        Safety: SL never loosens, and never goes above (current - 0.1%)
        for longs (would close instantly).
          - On bearish signal (conf > 60): tighten SL to 0.5% below current
          - On strong bullish continuation (conf > 80): extend TP higher
        """
        updates = []
        market_signals = market_signals or {}

        for i, pos in enumerate(self.open_positions):
            symbol = pos["symbol"]
            entry = pos["entry_price"]
            current_sl = pos["stop_loss"]
            current_tp = pos["take_profit"]
            current = current_prices.get(symbol)
            if not current or not entry:
                continue

            # v5.25: fixed-hold scalps are never trailed - the position dies
            # at the 60s time exit anyway; ladder/chandelier tightening would
            # just close it early through the back door.
            if pos.get("boosted_from_scalp"):
                continue

            self.track_excursions(pos, current)

            if pos["direction"] == "bullish":
                profit_pct = (current - entry) / entry * 100
            else:
                profit_pct = (entry - current) / entry * 100

            new_sl = None
            new_tp = None
            reason = ""

            if pos["direction"] == "bullish":
                # --- Mechanism 1: ladder (compute TARGET SL for the profit
                # level, then take max(target, current_sl) so we always jump
                # straight to the highest earned level.
                # v5.20: settings-driven and finer at the bottom rung. The
                # old +1% -> flat break-even gave back the whole move (ZEC
                # peaked +1.95%, exited -0.20% after fees). Locking a third
                # of the move at +1% keeps noise exits profitable.
                target_sl = None
                if profit_pct >= settings.LADDER_TRAIL_PCT:
                    target_sl = current * 0.99   # trail 1% below price
                    reason = f"Trailing stop (profit +{profit_pct:.2f}%)"
                elif profit_pct >= settings.LADDER_LOCK3_PCT:
                    target_sl = entry * (1 + settings.LADDER_LOCK3_LEVEL_PCT / 100.0)
                    reason = (f"Lock +{settings.LADDER_LOCK3_LEVEL_PCT:.2f}% profit "
                              f"(current +{profit_pct:.2f}%)")
                elif profit_pct >= settings.LADDER_LOCK2_PCT:
                    target_sl = entry * (1 + settings.LADDER_LOCK2_LEVEL_PCT / 100.0)
                    reason = (f"Lock +{settings.LADDER_LOCK2_LEVEL_PCT:.2f}% profit "
                              f"(current +{profit_pct:.2f}%)")
                elif profit_pct >= settings.LADDER_LOCK1_PCT:
                    target_sl = entry * (1 + settings.LADDER_LOCK1_LEVEL_PCT / 100.0)
                    reason = (f"Lock +{settings.LADDER_LOCK1_LEVEL_PCT:.2f}% profit "
                              f"(current +{profit_pct:.2f}%)")

                if target_sl is not None and target_sl > current_sl:
                    new_sl = target_sl

                # --- Mechanism 2: ATR chandelier (v5) ---
                if settings.CHANDELIER_ENABLED and profit_pct >= 1.0:
                    atr_val = self._atr_for(symbol, market_signals, pos)
                    if atr_val and atr_val > 0:
                        peak = float(pos.get("peak_price") or current)
                        chand = peak - settings.CHANDELIER_ATR_MULT * atr_val
                        # only meaningful if it tightens and stays below price
                        chand = min(chand, current * 0.999)
                        if chand > (new_sl if new_sl is not None else current_sl):
                            new_sl = chand
                            reason = (
                                f"Chandelier trail {settings.CHANDELIER_ATR_MULT}xATR "
                                f"(peak {peak:.4f}, +{profit_pct:.2f}%)"
                            )

                # hard cap: never place SL at/above the current price
                if new_sl is not None:
                    new_sl = min(new_sl, current * 0.999)
                    if new_sl <= current_sl:
                        new_sl = None

                # Extend TP on bullish continuation - v5.5: TOGETHER with a
                # profit-lock SL raise, so an extension can never leave the
                # trade exposed to a full round-trip (the "3 updates then
                # negative" failure mode). Requirements:
                #   - fresh analysis still bullish with conf >= CONTINUATION_CONF
                #   - trade already in profit >= CONTINUATION_MIN_PROFIT_PCT
                signal_data = market_signals.get(symbol, {})
                if (signal_data.get("direction") == "bullish"
                        and signal_data.get("confidence", 0) >= settings.CONTINUATION_CONF
                        and profit_pct >= settings.CONTINUATION_MIN_PROFIT_PCT):
                    current_tp_distance = current_tp - current
                    if current_tp_distance > 0:
                        # Extend by 50% of current TP distance
                        new_tp = current_tp + current_tp_distance * 0.5
                        reason += " + Extended TP (bullish continuation)"
                        # lock a fraction of the CURRENT profit into the SL
                        locked_pct = profit_pct * settings.PROFIT_LOCK_FRACTION
                        lock_sl = entry * (1 + locked_pct / 100.0)
                        lock_sl = min(lock_sl, current * 0.999)  # never above price
                        floor_sl = new_sl if new_sl is not None else current_sl
                        if lock_sl > floor_sl:
                            new_sl = lock_sl
                            reason += f" + Locked {locked_pct:.2f}% profit"

            elif pos["direction"] == "bearish":
                # mirror ladder for bearish (chandelier mirrored, v5.20 rungs)
                target_sl = None
                if profit_pct >= settings.LADDER_TRAIL_PCT:
                    target_sl = current * 1.01
                    reason = f"Trailing stop (profit +{profit_pct:.2f}%)"
                elif profit_pct >= settings.LADDER_LOCK3_PCT:
                    target_sl = entry * (1 - settings.LADDER_LOCK3_LEVEL_PCT / 100.0)
                    reason = (f"Lock +{settings.LADDER_LOCK3_LEVEL_PCT:.2f}% profit "
                              f"(current +{profit_pct:.2f}%)")
                elif profit_pct >= settings.LADDER_LOCK2_PCT:
                    target_sl = entry * (1 - settings.LADDER_LOCK2_LEVEL_PCT / 100.0)
                    reason = (f"Lock +{settings.LADDER_LOCK2_LEVEL_PCT:.2f}% profit "
                              f"(current +{profit_pct:.2f}%)")
                elif profit_pct >= settings.LADDER_LOCK1_PCT:
                    target_sl = entry * (1 - settings.LADDER_LOCK1_LEVEL_PCT / 100.0)
                    reason = (f"Lock +{settings.LADDER_LOCK1_LEVEL_PCT:.2f}% profit "
                              f"(current +{profit_pct:.2f}%)")
                if target_sl is not None and (current_sl is None or target_sl < current_sl):
                    new_sl = target_sl

                if settings.CHANDELIER_ENABLED and profit_pct >= 1.0:
                    atr_val = self._atr_for(symbol, market_signals, pos)
                    if atr_val and atr_val > 0:
                        trough = float(pos.get("trough_price") or current)
                        chand = trough + settings.CHANDELIER_ATR_MULT * atr_val
                        chand = max(chand, current * 1.001)
                        floor_sl = new_sl if new_sl is not None else current_sl
                        if chand < floor_sl:
                            new_sl = chand
                            reason = (
                                f"Chandelier trail {settings.CHANDELIER_ATR_MULT}xATR "
                                f"(trough {trough:.4f}, +{profit_pct:.2f}%)"
                            )

                if new_sl is not None:
                    new_sl = max(new_sl, current * 1.001)
                    if current_sl is not None and new_sl >= current_sl:
                        new_sl = None

                # v5.5: mirrored continuation (short side). Bullish-signal
                # exits are handled by the graduated structural pass - the
                # old loss-locking "tighten" block is deliberately gone.
                signal_data = market_signals.get(symbol, {})
                if (signal_data.get("direction") == "bearish"
                        and signal_data.get("confidence", 0) >= settings.CONTINUATION_CONF
                        and profit_pct >= settings.CONTINUATION_MIN_PROFIT_PCT):
                    current_tp_distance = current - current_tp
                    if current_tp_distance > 0:
                        new_tp = current_tp - current_tp_distance * 0.5
                        reason += " + Extended TP (bearish continuation)"
                        locked_pct = profit_pct * settings.PROFIT_LOCK_FRACTION
                        lock_sl = entry * (1 - locked_pct / 100.0)
                        lock_sl = max(lock_sl, current * 1.001)
                        floor_sl = new_sl if new_sl is not None else current_sl
                        if floor_sl is None or lock_sl < floor_sl:
                            new_sl = lock_sl
                            reason += f" + Locked {locked_pct:.2f}% profit"

            if new_sl is not None or new_tp is not None:
                result = self.update_position_risk(i, current, new_sl, new_tp, reason)
                if result.get("status") == "updated":
                    updates.append(result["update"])

        return updates

    @staticmethod
    def _atr_for(symbol: str, market_signals: Dict[str, Dict],
                 pos: Dict) -> Optional[float]:
        """ATR source precedence: fresh analysis -> stored at entry."""
        sig = market_signals.get(symbol) or {}
        atr = sig.get("atr")
        if isinstance(atr, (int, float)) and atr > 0:
            return float(atr)
        atr = pos.get("atr")
        if isinstance(atr, (int, float)) and atr > 0:
            return float(atr)
        return None

    # ============================================
    # v5: PENDING LIMIT ENTRIES — "buy the pocket, never chase"
    # ============================================

    @_locked
    def add_pending_entry(self, rec: Dict, reason: str = "",
                          dist_atr: float = 0.0) -> Dict:
        """Arm a pending LIMIT entry at the golden-pocket/entry zone.

        Instead of chasing a market buy when price has already left the
        entry zone, the setup is parked here and filled ONLY if price comes
        back into the zone within PENDING_TTL_HOURS.

        v5.19: the TTL scales with how far the zone sits below the price
        (dist_atr, in ATRs). Production showed a 4h TTL on zones 4.9-5.0
        ATR away (AAVE/NVDABUSDT) - a 5-ATR pullback inside 4 hours is a
        crash, not a fill, so the orders were guaranteed to expire while
        the strategy looked broken. dist 1 ATR -> 1x TTL, 3 ATR -> 3x TTL,
        capped at 4x. Unreachable zones (> PENDING_REACH_MAX_ATR) never
        reach this method - cycle.py drops them.
        """
        symbol = rec["symbol"]
        # one pending per symbol
        self.pending_entries = [
            p for p in self.pending_entries if p.get("symbol") != symbol
        ]
        if len(self.pending_entries) >= settings.MAX_PENDING_ENTRIES:
            # drop the oldest
            self.pending_entries.sort(key=lambda p: p.get("created_at", ""))
            self.pending_entries.pop(0)
        zone = rec.get("entry_zone") or {}
        entry = rec.get("entry_price") or rec.get("current_price") or 0
        zone_low = float(zone.get("low") or entry)
        zone_high = float(zone.get("high") or entry)
        ttl_scale = min(4.0, max(1.0, float(dist_atr or 1.0)))
        ttl_hours = settings.PENDING_TTL_HOURS * ttl_scale
        pending = {
            "symbol": symbol,
            "direction": rec.get("direction", "bullish"),
            "zone_low": zone_low,
            "zone_high": zone_high,
            "ref_price": rec.get("current_price"),
            "atr": float(rec.get("atr", 0) or 0),
            "created_at": now_utc().isoformat(),
            "expires_at": (now_utc() + timedelta(hours=ttl_hours)).isoformat(),
            "reason": reason,
            "rec": rec,
        }
        self.pending_entries.append(pending)
        save_json(self.pending_entries, PENDING_FILE)
        dist_tag = f", dist {dist_atr:.1f} ATR" if dist_atr else ""
        log.info(
            f"[cyan]Pending LIMIT entry armed[/] {symbol} "
            f"zone [{zone_low:.4f} - {zone_high:.4f}] "
            f"(ttl {ttl_hours:.1f}h{dist_tag}) - {reason}"
        )
        return pending

    @_locked
    def check_pending_fills(self, prices: Dict[str, float]) -> List[Dict]:
        """Fill / cancel / expire pending entries (called by the 1-min watcher).

        Long logic:
          - price <= zone_high  -> FILL at current price (a real limit fill)
            (re-checks every risk gate before the fill)
          - price < zone_low - PENDING_INVALID_ATR x ATR -> CANCEL (zone broke)
          - expired -> drop
        """
        if not self.pending_entries:
            return []
        filled, kept = [], []
        now = now_utc()
        for pending in self.pending_entries:
            symbol = pending["symbol"]
            price = prices.get(symbol)
            expired = pending.get("expires_at") and now >= datetime.fromisoformat(
                pending["expires_at"])
            if not price or expired:
                if expired:
                    log.info(f"[yellow]Pending entry expired[/] {symbol}")
                continue  # drop silently when no price (stale data)

            if pending.get("direction") != "bullish":
                continue  # spot bot: longs only for now

            zone_low = pending["zone_low"]
            zone_high = pending["zone_high"]
            atr = pending.get("atr") or 0
            invalid_level = zone_low - settings.PENDING_INVALID_ATR * atr

            # zone broke: price collapsed THROUGH the pocket -> setup dead
            # (checked BEFORE the fill: a limit buy must not fill on a crash)
            if price < invalid_level:
                log.info(
                    f"[yellow]Pending entry cancelled[/] {symbol} - "
                    f"zone broken (price {price:.4f} < {invalid_level:.4f})"
                )
                continue  # cancel

            if price <= zone_high:
                # ---- FILL: price returned into the zone ----
                rec = dict(pending["rec"])
                rec["current_price"] = price  # realistic limit fill price
                rec["entry_price"] = price
                result = self.open_position(rec)
                if result.get("status") == "opened":
                    filled.append({
                        "symbol": symbol,
                        "fill_price": price,
                        "position": result["position"],
                    })
                    log.info(
                        f"[green]Pending entry FILLED[/] {symbol} @ {price:.4f} "
                        f"(zone [{zone_low:.4f}-{zone_high:.4f}])"
                    )
                    continue  # consumed
                else:
                    reasons = result.get("reasons", [])
                    log.info(
                        f"[yellow]Pending fill rejected[/] {symbol}: {reasons}"
                    )
                    # a rejected fill (risk limits) -> drop it, don't retry
                    continue

            kept.append(pending)

        if len(kept) != len(self.pending_entries):
            self.pending_entries = kept
            save_json(self.pending_entries, PENDING_FILE)
        return filled

    @_locked
    def cancel_pending(self, symbol: str) -> bool:
        """Manually cancel a pending entry (also used after a fill opens)."""
        before = len(self.pending_entries)
        self.pending_entries = [
            p for p in self.pending_entries if p.get("symbol") != symbol
        ]
        if len(self.pending_entries) != before:
            save_json(self.pending_entries, PENDING_FILE)
            return True
        return False

    # ============================================
    # v5: MARKET TIDE (BTC) FILTER
    # ============================================

    def _tide_snapshot(self) -> Tuple[Optional[str], float]:
        """v5.20: BTC 1h regime + score from the shared market_tide cache
        (data/market_tide.json), refreshed when stale. (None, 0.0) on any
        failure. Used by market_tide_blocked AND the v5.20 bottom-channel
        tide gate."""
        cache_file = Path("data/market_tide.json")
        cached = load_json(cache_file, default={})
        try:
            fetched_at = datetime.fromisoformat(cached["fetched_at"])
            age_min = (now_utc() - fetched_at).total_seconds() / 60
        except Exception:
            age_min = 1e9
        if age_min > settings.MARKET_FILTER_CACHE_MIN:
            try:
                from src.core.data_fetcher import data_fetcher
                from src.indicators.ichimoku import ichimoku_state
                df = data_fetcher.get_candles(
                    settings.MARKET_FILTER_SYMBOL,
                    "1h", limit=max(120, settings.CANDLE_LIMIT))
                icho = ichimoku_state(df)
                cached = {
                    "fetched_at": now_utc().isoformat(),
                    "regime": icho.get("regime"),
                    "score": icho.get("score", 0),
                }
                save_json(cached, cache_file)
            except Exception as e:
                log.warning(f"Market tide fetch failed: {e}")
                cached = cached or {"regime": None, "score": 0}
        return (cached.get("regime"), float(cached.get("score") or 0))

    def market_tide_blocked(self) -> Tuple[bool, str]:
        """Block NEW entries when the BTC regime is strongly bearish.
        Cached to data/market_tide.json for MARKET_FILTER_CACHE_MIN minutes.
        Open positions are ALWAYS still managed - this gate is entries-only.
        """
        if not settings.MARKET_FILTER_ENABLED:
            return (False, "")
        regime, score = self._tide_snapshot()
        if regime == "bearish" and score <= -40:
            return (True,
                    f"Market tide bearish ({settings.MARKET_FILTER_SYMBOL} "
                    f"score {score:.0f}) - new entries paused")
        return (False, "")

    @_locked
    def get_positions_with_pnl(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Return open positions with real-time P&L info (fees included)."""
        positions_with_pnl = []
        for i, pos in enumerate(self.open_positions):
            entry = pos["entry_price"]
            current = current_prices.get(pos["symbol"])
            notional_usd = pos.get("notional_usd", pos.get("size", 0) * entry)
            entry_fee = pos.get("entry_fee", 0)

            if not current:
                pnl = 0
                pnl_pct = 0
                current = None
                exit_fee = 0
                gross_pnl = 0
            else:
                # Current value of position
                current_value = notional_usd * (current / entry) if entry > 0 else notional_usd
                exit_fee = current_value * (settings.TRADING_FEE_PCT / 100)
                gross_pnl = current_value - notional_usd
                # Net P&L = gross - entry_fee - exit_fee
                pnl = gross_pnl - entry_fee - exit_fee
                pnl_pct = (pnl / notional_usd * 100) if notional_usd > 0 else 0

            # Distance to SL/TP in %
            sl = pos["stop_loss"]
            tp = pos["take_profit"]
            if current and entry:
                # Progress: 0% at SL, 50% at entry, 100% at TP
                if tp != sl:
                    # Position between SL (0%) and TP (100%)
                    progress = (current - sl) / (tp - sl) * 100
                else:
                    progress = 50
                if pos["direction"] == "bullish":
                    sl_dist = (current - sl) / current * 100
                    tp_dist = (tp - current) / current * 100
                else:
                    sl_dist = (sl - current) / current * 100
                    tp_dist = (current - tp) / current * 100
            else:
                sl_dist = 0
                tp_dist = 0
                progress = 50

            positions_with_pnl.append({
                **pos,
                "current_price": current,
                "current_pnl": float(pnl),
                "current_pnl_pct": float(pnl_pct),
                "gross_pnl": float(gross_pnl),
                "entry_fee": float(entry_fee),
                "exit_fee": float(exit_fee),
                "total_fees": float(entry_fee + exit_fee),
                "sl_distance_pct": float(sl_dist),
                "tp_distance_pct": float(tp_dist),
                "progress_pct": float(progress),
                "updates_count": len(pos.get("risk_updates", [])),
                # v5 real-time tracking fields
                "age_hours": round(self._position_age_hours(pos), 2),
                "stage": "runner" if pos.get("tp1_taken") else "entry",
                "peak_price": pos.get("peak_price"),
                "trough_price": pos.get("trough_price"),
                "mfe_pct": float(pos.get("mfe_pct", 0.0)),
                "mae_pct": float(pos.get("mae_pct", 0.0)),
                "partials_taken": len(pos.get("partial_closes", [])),
                # v5.11: WHOLE-trade P&L = banked partials + unrealized rest
                "realized_pnl": float(pos.get("realized_pnl", 0) or 0),
                "total_pnl": float(pos.get("realized_pnl", 0) or 0) + float(pnl),
                "total_pnl_pct": float(
                    ((float(pos.get("realized_pnl", 0) or 0) + float(pnl))
                     / (float(pos.get("initial_notional_usd") or notional_usd) or 1)
                     * 100)),
                # v5.5: full SL/TP update history (newest first, capped) so
                # the dashboard can show WHY each update happened
                "risk_updates": list(reversed(pos.get("risk_updates", [])))[:10],
                "time_stop_pending": bool(
                    self._position_age_hours(pos) >= settings.MAX_TRADE_HOURS
                    and pnl_pct < settings.TIME_STOP_MIN_PNL_PCT
                ),
            })
        return positions_with_pnl


# Singleton
risk_manager = RiskManager()
