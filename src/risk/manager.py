"""
Risk Management Module
- Position sizing (Kelly fraction / fixed fractional)
- Daily max loss enforcement
- Risk/Reward check
- Stop loss / Take profit verification
"""
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, timezone, timedelta
from config.settings import settings
from src.db.database import db
from src.utils.logger import log
from src.utils.helpers import load_json, save_json, now_utc

POSITIONS_FILE = Path("data/open_positions.json")
DAILY_STATS_FILE = Path("data/daily_stats.json")


class RiskManager:
    """Enforces risk rules across the trading bot."""

    def __init__(self, capital: float = None):
        self.capital = capital or settings.INITIAL_CAPITAL
        self.open_positions: List[Dict] = load_json(POSITIONS_FILE, default=[])
        self.daily_stats: Dict = load_json(DAILY_STATS_FILE, default={})
        # v4.1 loss-avoidance state
        self._reentry_block: Dict[str, datetime] = {}  # symbol -> blocked until
        self._restore_loss_state()
        log.info(
            f"[cyan]RiskManager[/] initialized - "
            f"Capital: ${self.capital:,.2f} | "
            f"Open positions: {len(self.open_positions)}"
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
        return (len(reasons) == 0, reasons)

    def open_paper_position(self, rec: Dict) -> Dict:
        """Open a paper-trading position based on a recommendation.
        Uses TRADE_AMOUNT_USD ($10 default) for position sizing.
        Applies LOT_SIZE rules from Binance and trading fees (0.1%).
        """
        valid, reasons = self.validate_recommendation(rec)
        if not valid:
            return {"status": "rejected", "reasons": reasons}

        if not self.can_open_position(rec.get("symbol")):
            return {"status": "rejected", "reasons": ["Risk limits reached"]}

        # Duplicate-symbol guard (second line of defense)
        if settings.SKIP_DUPLICATE_SYMBOLS and self.has_open_position(rec["symbol"]):
            return {"status": "rejected", "reasons": [f"{rec['symbol']} already has an open position"]}

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

        valid, reasons = self.validate_recommendation(rec)
        if not valid:
            return {"status": "rejected", "reasons": reasons}
        if not self.can_open_position():
            return {"status": "rejected", "reasons": ["Risk limits reached"]}
        if settings.SKIP_DUPLICATE_SYMBOLS and self.has_open_position(rec["symbol"]):
            return {"status": "rejected", "reasons": [f"{rec['symbol']} already has an open position"]}

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

    def close_position(self, idx: int, exit_price: float, reason: str = "") -> Dict:
        """Close an open position at the given exit price.
        P&L = (exit_value - entry_value) - entry_fee - exit_fee
        Both paper and live positions use SAME fee calculation.
        """
        if idx >= len(self.open_positions):
            return {"status": "error", "reason": "Invalid index"}
        pos = self.open_positions[idx]
        entry_price = pos["entry_price"]
        size = pos.get("size", 0)
        notional_usd = pos.get("notional_usd", size * entry_price)
        entry_fee = pos.get("entry_fee", 0)

        # Calculate exit value (proportional to entry)
        if entry_price > 0:
            exit_value = notional_usd * (exit_price / entry_price)
        else:
            exit_value = notional_usd

        # Exit fee (0.1% of exit value)
        exit_fee = exit_value * (settings.TRADING_FEE_PCT / 100)
        # Net P&L = exit_value - entry_value - entry_fee - exit_fee
        gross_pnl = exit_value - notional_usd
        net_pnl = gross_pnl - entry_fee - exit_fee
        pnl_pct = (net_pnl / notional_usd * 100) if notional_usd > 0 else 0

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

    def check_open_positions(self, prices: Dict[str, float]) -> List[Dict]:
        """Check open positions for SL/TP hits."""
        closed = []
        for i in range(len(self.open_positions) - 1, -1, -1):
            pos = self.open_positions[i]
            price = prices.get(pos["symbol"])
            if not price:
                continue
            if pos["direction"] == "bullish":
                if price <= pos["stop_loss"]:
                    closed.append(self.close_position(i, pos["stop_loss"], "Stop Loss Hit"))
                elif price >= pos["take_profit"]:
                    closed.append(self.close_position(i, pos["take_profit"], "Take Profit Hit"))
            else:  # bearish
                if price >= pos["stop_loss"]:
                    closed.append(self.close_position(i, pos["stop_loss"], "Stop Loss Hit"))
                elif price <= pos["take_profit"]:
                    closed.append(self.close_position(i, pos["take_profit"], "Take Profit Hit"))
        return closed

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

        Trailing Stop Ladder (for LONG positions) - v2 jumps DIRECTLY to the
        highest level reached (the old elif-chain lagged one level per cycle,
        so it never caught up with fast pumps on 10-minute cycles):
          - +1.0% profit: SL -> entry (break-even)
          - +2.0% profit: SL -> +1.0%
          - +3.0% profit: SL -> +2.0%
          - +5.0%+ profit: SL trails 1% below current price
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

            if pos["direction"] == "bullish":
                profit_pct = (current - entry) / entry * 100
            else:
                profit_pct = (entry - current) / entry * 100

            new_sl = None
            new_tp = None
            reason = ""

            if pos["direction"] == "bullish":
                # --- Trailing ladder: compute TARGET SL for the profit level,
                # then take the max(target, current_sl) so we always jump
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

                # Extend TP if strong bullish continuation
                signal_data = market_signals.get(symbol, {})
                if signal_data.get("direction") == "bullish" and signal_data.get("confidence", 0) > 80:
                    current_tp_distance = current_tp - current
                    if current_tp_distance > 0:
                        # Extend by 50% of current TP distance
                        new_tp = current_tp + current_tp_distance * 0.5
                        reason += " + Extended TP (strong bullish signal)"

                # Tighten SL if bearish signal appears (overrides ladder)
                if signal_data.get("direction") == "bearish" and signal_data.get("confidence", 0) > 60:
                    tighten_sl = current * 0.995  # 0.5% below current
                    if tighten_sl > current_sl:
                        new_sl = tighten_sl
                        reason = f"Tightened SL (bearish signal detected, conf={signal_data['confidence']:.0f}%)"

            if new_sl is not None or new_tp is not None:
                result = self.update_position_risk(i, current, new_sl, new_tp, reason)
                if result.get("status") == "updated":
                    updates.append(result["update"])

        return updates

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
            })
        return positions_with_pnl


# Singleton
risk_manager = RiskManager()
