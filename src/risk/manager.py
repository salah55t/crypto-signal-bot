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
from config.settings import settings
from src.db.database import db
from src.utils.logger import log
from src.utils.helpers import load_json, save_json, now_utc

POSITIONS_FILE = Path("data/open_positions.json")
DAILY_STATS_FILE = Path("data/daily_stats.json")
PENDING_FILE = Path("data/pending_entries.json")


class RiskManager:
    """Enforces risk rules across the trading bot."""

    def __init__(self, capital: float = None):
        self.capital = capital or settings.INITIAL_CAPITAL
        self.open_positions: List[Dict] = load_json(POSITIONS_FILE, default=[])
        self.daily_stats: Dict = load_json(DAILY_STATS_FILE, default={})
        self.pending_entries: List[Dict] = (
            load_json(PENDING_FILE, default=[]) if settings.PENDING_ENTRIES_ENABLED else []
        )
        # v4.1 loss-avoidance state
        self._reentry_block: Dict[str, datetime] = {}  # symbol -> blocked until
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

    def position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        Compute position size in base currency using fixed fractional risk.
        Risk = settings.RISK_PER_TRADE% of capital.
        Returns position size in units of base asset.
        """
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            log.warning("Invalid risk_per_unit (entry == stop_loss)")
            return 0.0
        risk_amount = self.capital * (settings.RISK_PER_TRADE / 100)
        return risk_amount / risk_per_unit

    def validate_recommendation(self, rec: Dict) -> tuple:
        """
        Returns (is_valid, reasons).
        Validates R/R ratio, ATR sanity, etc.
        v5: harmony gate (layered agreement) + volatility sanity.
        """
        reasons = []
        # v4: vetoed signals (Ichimoku regime opposition) can never open positions
        if rec.get("decision", {}).get("vetoed", False):
            reasons.append(
                f"Vetoed by confluence engine: "
                f"{rec.get('decision', {}).get('veto_reason', 'regime opposition')}"
            )
        rr = rec.get("risk_reward_ratio", 0)
        if rr < settings.MIN_RR_RATIO:
            reasons.append(f"R/R too low ({rr:.2f} < {settings.MIN_RR_RATIO})")
        if rec.get("admission_confidence",
                   rec.get("confidence", 0)) < settings.MIN_CONFIDENCE:
            reasons.append(f"Confidence too low ({rec['confidence']:.1f}%)")
        if rec.get("expected_rise_pct", 0) < settings.MIN_EXPECTED_RISE:
            reasons.append(f"Expected rise too low ({rec['expected_rise_pct']:.2f}%)")
        if rec.get("stop_loss", 0) <= 0:
            reasons.append("Invalid stop loss")
        # v5: layered harmony gate - a veteran requires layered agreement.
        # v5.6: bottom-boosted recs are EXEMPT - they carry their own layered
        # gate (bounce score >= BOTTOM_STRONG_SCORE + bullish close + RR) and
        # a bounce-derived harmony. Without the exemption the strategy-scale
        # gate rejected EVERY bottom rec (no harmony key -> 0.0 < 0.45), which
        # is why the bot never opened a position from bottom coins.
        if (not rec.get("boosted_from_bottom")
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

        entry = rec["current_price"]
        sl = rec["stop_loss"]
        # Use fixed $10 trade amount (configurable via TRADE_AMOUNT_USD)
        notional_usd = settings.TRADE_AMOUNT_USD
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
        }
        self.open_positions.append(position)
        save_json(self.open_positions, POSITIONS_FILE)
        self._ensure_today_stats()
        self.daily_stats[self._today_key()]["trades_opened"] += 1
        save_json(self.daily_stats, DAILY_STATS_FILE)
        # Log to database
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
        if not self.can_open_position():
            return {"status": "rejected", "reasons": ["Risk limits reached"]}

        symbol = rec["symbol"]
        entry = rec["current_price"]
        sl = rec["stop_loss"]
        tp = rec["take_profit"]

        # Position sizing: compute USD amount to risk
        risk_amount = self.capital * (settings.RISK_PER_TRADE / 100)
        # USD notional to spend: 2x risk amount (gives reasonable position size)
        # Adjust so position size matches risk_per_trade model
        notional_usd = min(risk_amount * 10, self.capital * 0.20)  # cap at 20% of capital
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
            }
            self.open_positions.append(position)
            save_json(self.open_positions, POSITIONS_FILE)
            self._ensure_today_stats()
            self.daily_stats[self._today_key()]["trades_opened"] += 1
            save_json(self.daily_stats, DAILY_STATS_FILE)

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

    def close_position(self, idx: int, exit_price: float, reason: str = "",
                       fraction: float = 1.0) -> Dict:
        """Close an open position at the given exit price.

        v5: `fraction < 1.0` closes a PARTIAL chunk (e.g. TP1 banking).
        - P&L is realized on the closed fraction only
        - the position stays open with reduced notional/size
        - partial closes do NOT touch the win/loss streak counters (only
          full closes do) - they only add realized P&L
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
            save_json(self.open_positions, POSITIONS_FILE)
            self._ensure_today_stats()
            self.daily_stats[self._today_key()]["pnl"] += net_pnl
            self.daily_stats[self._today_key()]["partials"] = (
                self.daily_stats[self._today_key()].get("partials", 0) + 1
            )
            save_json(self.daily_stats, DAILY_STATS_FILE)
            try:
                db.log_position_closed(
                    symbol=pos["symbol"],
                    entry_time=pos["entry_time"],
                    exit_price=exit_price,
                    exit_time=now_utc().isoformat(),
                    pnl=net_pnl,
                    pnl_pct=pnl_pct,
                    close_reason=reason,
                )
            except Exception as e:
                log.debug(f"DB log partial close failed: {e}")
            log.info(
                f"[green]PARTIAL close ({fraction*100:.0f}%)[/] {pos['symbol']} - "
                f"Net: ${net_pnl:+.4f} ({pnl_pct:+.2f}%) - {reason} | "
                f"remaining notional ${pos['notional_usd']:.2f}"
            )
            return {"status": "partial", "position": pos, "pnl": net_pnl,
                    "pnl_pct": pnl_pct, "reason": reason, "fraction": fraction}

        # ---- full close (original path, fraction == 1.0) ----
        closed = {**pos, "exit_price": exit_price, "exit_time": now_utc().isoformat(),
                  "pnl": net_pnl, "pnl_pct": pnl_pct, "reason": reason,
                  "status": "closed", "entry_fee": entry_fee, "exit_fee": exit_fee,
                  "gross_pnl": gross_pnl}
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
        # Log to database
        try:
            db.log_position_closed(
                symbol=pos["symbol"],
                entry_time=pos["entry_time"],
                exit_price=exit_price,
                exit_time=closed["exit_time"],
                pnl=net_pnl,
                pnl_pct=pnl_pct,
                close_reason=reason,
            )
            # Update daily stats in DB
            db.update_daily_stats(
                date=self._today_key(),
                trades_opened=self.daily_stats[self._today_key()]["trades_opened"],
                wins=self.daily_stats[self._today_key()]["wins"],
                losses=self.daily_stats[self._today_key()]["losses"],
                pnl_delta=0,  # already updated above
            )
        except Exception as e:
            log.debug(f"DB log_position_closed failed: {e}")
        log.info(
            f"[yellow]Position closed[/] {pos['symbol']} - "
            f"Gross P&L: ${gross_pnl:+.4f} - Fees: ${entry_fee+exit_fee:.4f} "
            f"= Net: ${net_pnl:+.4f} ({pnl_pct:+.2f}%) - {reason}"
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
        """v5: a trade that goes nowhere is dead capital."""
        age_h = self._position_age_hours(pos)
        if age_h >= settings.ABSOLUTE_MAX_TRADE_HOURS:
            return (f"Max holding time reached "
                    f"({age_h:.1f}h >= {settings.ABSOLUTE_MAX_TRADE_HOURS:.0f}h)")
        if (age_h >= settings.MAX_TRADE_HOURS
                and pnl_pct < settings.TIME_STOP_MIN_PNL_PCT):
            return (f"Time stop: stale trade ({age_h:.1f}h, "
                    f"{pnl_pct:+.2f}% < {settings.TIME_STOP_MIN_PNL_PCT:.2f}%)")
        return None

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

        if direction == "bullish":
            # 2) Ichimoku regime flipped bearish -> thesis dead, exit
            if regime == "bearish":
                return ("exit", "Ichimoku regime flipped bearish")
            # 3) Regime decayed to neutral + price lost Kijun -> tighten to Kijun
            if regime == "neutral" and icho.get("price_vs_kijun") == "below" and kijun:
                if kijun < current_price:
                    return ("tighten", f"Kijun defence ({kijun:.4f})")
            # 4) Fresh bearish TK cross -> tighten to Tenkan
            if (icho.get("tk_cross_recent") == "bearish"
                    and icho.get("tk_state") == "bearish" and tenkan
                    and tenkan < current_price):
                return ("tighten", f"Tenkan cross-down defence ({tenkan:.4f})")
        else:
            if regime == "bullish":
                return ("exit", "Ichimoku regime flipped bullish")
            if regime == "neutral" and icho.get("price_vs_kijun") == "above" and kijun:
                if kijun > current_price:
                    return ("tighten", f"Kijun defence ({kijun:.4f})")
            if (icho.get("tk_cross_recent") == "bullish"
                    and icho.get("tk_state") == "bullish" and tenkan
                    and tenkan > current_price):
                return ("tighten", f"Tenkan cross-up defence ({tenkan:.4f})")
        return ("none", None)

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
            log.info(
                f"[blue]Risk update[/] {pos['symbol']} - "
                f"SL: {old_sl:.4f} -> {pos['stop_loss']:.4f} | "
                f"TP: {old_tp:.4f} -> {pos['take_profit']:.4f} - {reason}"
            )
            return {"status": "updated", "position": pos, "update": update_record}
        return {"status": "no_change"}

    def has_open_position(self, symbol: str) -> bool:
        """Check if a position is already open for the given symbol."""
        return any(p.get("symbol") == symbol for p in self.open_positions)

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
                target_sl = None
                if profit_pct >= 5.0:
                    target_sl = current * 0.99   # trail 1% below price
                    reason = f"Trailing stop (profit +{profit_pct:.2f}%)"
                elif profit_pct >= 3.0:
                    target_sl = entry * 1.02
                    reason = f"Lock +2% profit (current +{profit_pct:.2f}%)"
                elif profit_pct >= 2.0:
                    target_sl = entry * 1.01
                    reason = f"Lock +1% profit (current +{profit_pct:.2f}%)"
                elif profit_pct >= 1.0:
                    target_sl = entry
                    reason = f"Break-even (profit +{profit_pct:.2f}%)"

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
                # mirror ladder for bearish (chandelier mirrored)
                target_sl = None
                if profit_pct >= 5.0:
                    target_sl = current * 1.01
                    reason = f"Trailing stop (profit +{profit_pct:.2f}%)"
                elif profit_pct >= 3.0:
                    target_sl = entry * 0.98
                    reason = f"Lock +2% profit (current +{profit_pct:.2f}%)"
                elif profit_pct >= 2.0:
                    target_sl = entry * 0.99
                    reason = f"Lock +1% profit (current +{profit_pct:.2f}%)"
                elif profit_pct >= 1.0:
                    target_sl = entry
                    reason = f"Break-even (profit +{profit_pct:.2f}%)"
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

    def add_pending_entry(self, rec: Dict, reason: str = "") -> Dict:
        """Arm a pending LIMIT entry at the golden-pocket/entry zone.

        Instead of chasing a market buy when price has already left the
        entry zone, the setup is parked here and filled ONLY if price comes
        back into the zone within PENDING_TTL_HOURS.
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
        pending = {
            "symbol": symbol,
            "direction": rec.get("direction", "bullish"),
            "zone_low": zone_low,
            "zone_high": zone_high,
            "ref_price": rec.get("current_price"),
            "atr": float(rec.get("atr", 0) or 0),
            "created_at": now_utc().isoformat(),
            "expires_at": (now_utc() + timedelta(
                hours=settings.PENDING_TTL_HOURS)).isoformat(),
            "reason": reason,
            "rec": rec,
        }
        self.pending_entries.append(pending)
        save_json(self.pending_entries, PENDING_FILE)
        log.info(
            f"[cyan]Pending LIMIT entry armed[/] {symbol} "
            f"zone [{zone_low:.4f} - {zone_high:.4f}] "
            f"(ttl {settings.PENDING_TTL_HOURS:.0f}h) - {reason}"
        )
        return pending

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

    def market_tide_blocked(self) -> Tuple[bool, str]:
        """Block NEW entries when the BTC regime is strongly bearish.
        Cached to data/market_tide.json for MARKET_FILTER_CACHE_MIN minutes.
        Open positions are ALWAYS still managed - this gate is entries-only.
        """
        if not settings.MARKET_FILTER_ENABLED:
            return (False, "")
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

        regime = cached.get("regime")
        score = float(cached.get("score") or 0)
        if regime == "bearish" and score <= -40:
            return (True,
                    f"Market tide bearish ({settings.MARKET_FILTER_SYMBOL} "
                    f"score {score:.0f}) - new entries paused")
        return (False, "")

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
