"""
Backtesting Engine (v5)
Replays historical OHLCV data through the SignalScorer to evaluate
strategy performance over time.

v2: realistic simulation (fees, settings thresholds, expectancy).
v5: "Veteran Trader" simulation - can be toggled with `v5=True/False`
    for clean A/B comparison against the v4.1 behaviour:
      - Partial TP at TP1 (bank fraction, SL -> BE+fees, runner -> TP2)
      - ATR chandelier trailing (mult x ATR below peak since entry)
      - Structure exits (Ichimoku regime flip / opposite signal)
      - Time stop (stale trades)
      - Pending LIMIT entries (no chasing: fill only inside the zone,
        cancel when the zone breaks, expire after TTL)
      - v5 admission gates (harmony / volatility / dead market)

    Trade records keep the FULL lifecycle (partial P&L chunks are
    accumulated into the position's final record) so metrics stay
    directly comparable with v4.1 runs.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import time
from config.settings import settings
from src.analysis.scorer import scorer
from src.indicators.ichimoku import ichimoku_state
from src.indicators.technical import atr as ta_atr
from src.utils.logger import log
from src.utils.helpers import save_json, to_json_safe, fmt_pct


class Backtester:
    """
    Walks historical data bar-by-bar, simulates recommendations,
    tracks performance with paper positions and SL/TP rules.
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.positions: List[Dict] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.pending: List[Dict] = []

    # ------------------------------------------------------------------
    def _open_position(self, rec: Dict, timestamp: str,
                       entry_price: Optional[float] = None) -> Dict:
        entry = float(entry_price if entry_price is not None
                      else rec["current_price"])
        sl = rec["stop_loss"]
        # Position sizing: 5% of capital per trade
        notional = self.capital * 0.05
        size = notional / entry if entry > 0 else 0
        pos = {
            "symbol": rec["symbol"],
            "entry_price": entry,
            "stop_loss": sl,
            "take_profit": rec["take_profit"],
            "take_profit_2": rec.get("take_profit_2", rec["take_profit"]),
            "size": size,
            "notional": notional,
            "initial_notional": notional,
            "entry_fee": notional * (settings.TRADING_FEE_PCT / 100),
            "entry_time": timestamp,
            "confidence": rec["confidence"],
            "expected_rise_pct": rec["expected_rise_pct"],
            "direction": rec["direction"],
            # v5 state
            "tp1_taken": False,
            "peak_price": entry,
            "trough_price": entry,
            "realized_pnl": 0.0,
            "realized_chunks": 0,
            "bars_held": 0,
            "atr": float(rec.get("atr", 0) or 0),
            "entry_type": rec.get("entry_type", "market"),
        }
        self.positions.append(pos)
        return pos

    # ------------------------------------------------------------------
    def _realize(self, pos: Dict, exit_price: float, fraction: float,
                 reason: str) -> float:
        """Realize P&L on a chunk of the position; returns chunk net pnl."""
        notional_full = pos["notional"]
        chunk_notional = notional_full * fraction
        entry_fee = pos["entry_fee"] * fraction
        exit_value = chunk_notional * (exit_price / pos["entry_price"])
        exit_fee = exit_value * (settings.TRADING_FEE_PCT / 100)
        pnl = exit_value - chunk_notional - entry_fee - exit_fee
        pos["realized_pnl"] += pnl
        pos["realized_chunks"] += 1
        pos["notional"] -= chunk_notional
        pos["size"] *= (1.0 - fraction)
        pos["entry_fee"] -= entry_fee
        self.capital += pnl
        pos["_last_reason"] = reason
        return pnl

    def _finish_trade(self, pos: Dict, exit_price: float, timestamp: str,
                      reason: str) -> Dict:
        """Emit one aggregated trade record for the full lifecycle."""
        # remaining chunk
        self._realize(pos, exit_price, 1.0, reason)
        total_pnl = pos["realized_pnl"]
        initial_notional = pos.get("initial_notional") or 1e-9
        rec = {
            "symbol": pos["symbol"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "entry_time": pos["entry_time"],
            "exit_time": timestamp,
            "pnl": total_pnl,
            "pnl_pct": total_pnl / initial_notional * 100,
            "reason": reason,
            "partials": pos["realized_chunks"] - 1,
            "bars_held": pos["bars_held"],
            "direction": pos["direction"],
            "confidence": pos["confidence"],
            "entry_type": pos.get("entry_type", "market"),
        }
        return rec

    # ------------------------------------------------------------------
    def _check_exits_v5(self, bar, atr_val: float, window_df, timestamp: str,
                        tf_minutes: int) -> List[Dict]:
        """v5 exit pass over open positions for one bar."""
        closed = []
        current_price = float(bar["close"])
        max_bars = int(settings.MAX_TRADE_HOURS * 60 / max(tf_minutes, 1))
        abs_bars = int(settings.ABSOLUTE_MAX_TRADE_HOURS * 60 / max(tf_minutes, 1))

        for pos in self.positions[:]:
            pos["bars_held"] += 1
            # excursions
            pos["peak_price"] = max(pos["peak_price"], current_price)
            pos["trough_price"] = min(pos["trough_price"], current_price)
            if pos["direction"] == "bullish":
                profit_pct = (current_price - pos["entry_price"]) / pos["entry_price"] * 100
                sl_hit = current_price <= pos["stop_loss"]
                tp_hit = current_price >= pos["take_profit"]
            else:
                profit_pct = (pos["entry_price"] - current_price) / pos["entry_price"] * 100
                sl_hit = current_price >= pos["stop_loss"]
                tp_hit = current_price <= pos["take_profit"]

            if sl_hit:
                rec = self._finish_trade(pos, pos["stop_loss"], timestamp, "SL")
                self.positions.remove(pos)
                closed.append(rec)
                continue

            # time stop
            if pos["bars_held"] >= abs_bars or (
                    pos["bars_held"] >= max_bars
                    and profit_pct < settings.TIME_STOP_MIN_PNL_PCT):
                reason = ("MaxHolding" if pos["bars_held"] >= abs_bars
                          else "TimeStop")
                rec = self._finish_trade(pos, current_price, timestamp, reason)
                self.positions.remove(pos)
                closed.append(rec)
                continue

            # TP1 partial
            if tp_hit and settings.PARTIAL_TP_ENABLED and not pos["tp1_taken"]:
                frac = max(0.1, min(0.9, settings.PARTIAL_TP_FRACTION))
                self._realize(pos, pos["take_profit"], frac, "TP1")
                pos["tp1_taken"] = True
                entry = pos["entry_price"]
                buf = settings.TP1_FEE_BUFFER_PCT / 100.0
                be = (entry * (1 + buf) if pos["direction"] == "bullish"
                      else entry * (1 - buf))
                if pos["direction"] == "bullish":
                    pos["stop_loss"] = max(pos["stop_loss"], be)
                    if pos["take_profit_2"] > pos["take_profit"]:
                        pos["take_profit"] = pos["take_profit_2"]
                    tp_hit = current_price >= pos["take_profit"]
                else:
                    pos["stop_loss"] = min(pos["stop_loss"], be)
                    if pos["take_profit_2"] < pos["take_profit"]:
                        pos["take_profit"] = pos["take_profit_2"]
                    tp_hit = current_price <= pos["take_profit"]
                if tp_hit:
                    rec = self._finish_trade(pos, pos["take_profit"], timestamp, "TP2")
                    self.positions.remove(pos)
                    closed.append(rec)
                    continue

            elif tp_hit:
                rec = self._finish_trade(pos, pos["take_profit"], timestamp, "TP")
                self.positions.remove(pos)
                closed.append(rec)
                continue

            # chandelier trailing
            if (settings.CHANDELIER_ENABLED and atr_val and atr_val > 0
                    and profit_pct >= settings.CHANDELIER_ACTIVATE_PCT):
                if pos["direction"] == "bullish":
                    chand = pos["peak_price"] - settings.CHANDELIER_ATR_MULT * atr_val
                    chand = min(chand, current_price * 0.999)
                    if chand > pos["stop_loss"]:
                        pos["stop_loss"] = chand
                else:
                    chand = pos["trough_price"] + settings.CHANDELIER_ATR_MULT * atr_val
                    chand = max(chand, current_price * 1.001)
                    if chand < pos["stop_loss"]:
                        pos["stop_loss"] = chand

            # structure exit (Ichimoku flip) - only with fresh klines context
            if settings.STRUCTURAL_EXITS_ENABLED and window_df is not None:
                icho = ichimoku_state(window_df)
                if icho:
                    if (pos["direction"] == "bullish"
                            and icho.get("regime") == "bearish"):
                        rec = self._finish_trade(
                            pos, current_price, timestamp, "StructFlip")
                        self.positions.remove(pos)
                        closed.append(rec)
                        continue
                    if (pos["direction"] == "bearish"
                            and icho.get("regime") == "bullish"):
                        rec = self._finish_trade(
                            pos, current_price, timestamp, "StructFlip")
                        self.positions.remove(pos)
                        closed.append(rec)
                        continue
        return closed

    def _check_exits(self, current_price: float, timestamp: str) -> List[Dict]:
        """v4.1 (baseline) exit pass: full SL/TP closes only."""
        closed = []
        for pos in self.positions:
            exit_price = None
            reason = None
            if pos["direction"] == "bullish":
                if current_price <= pos["stop_loss"]:
                    exit_price, reason = current_price, "SL"
                elif current_price >= pos["take_profit"]:
                    exit_price, reason = pos["take_profit"], "TP"
            else:  # bearish
                if current_price >= pos["stop_loss"]:
                    exit_price, reason = current_price, "SL"
                elif current_price <= pos["take_profit"]:
                    exit_price, reason = pos["take_profit"], "TP"

            if exit_price is None:
                continue

            notional = pos.get("notional", pos["size"] * pos["entry_price"])
            exit_value = notional * (exit_price / pos["entry_price"])
            exit_fee = exit_value * (settings.TRADING_FEE_PCT / 100)
            pnl = exit_value - notional - pos.get("entry_fee", 0) - exit_fee
            pos["exit_price"] = exit_price
            pos["exit_time"] = timestamp
            pos["pnl"] = pnl
            pos["pnl_pct"] = pnl / notional * 100 if notional > 0 else 0
            pos["reason"] = reason
            closed.append(pos)
            self.capital += pnl
        for pos in closed:
            self.positions.remove(pos)
        self.trades.extend(closed)
        return closed

    # ------------------------------------------------------------------
    def _passes_gates(self, rec: Dict, v5: bool) -> bool:
        if (rec.get("direction") != "bullish"
                or rec.get("decision", {}).get("vetoed", False)):
            return False
        if (rec.get("admission_confidence",
                    rec.get("confidence", 0)) < settings.MIN_CONFIDENCE):
            return False
        if rec.get("risk_reward_ratio", 0) < settings.MIN_RR_RATIO:
            return False
        if v5:
            if rec.get("harmony", 0.0) < settings.MIN_HARMONY:
                return False
            if settings.EXCLUDE_VOLATILITY_EXTREME and rec.get("volatility_extreme"):
                return False
            if rec.get("dead_market"):
                return False
        return True

    def _process_pending(self, bar, atr_val: float, i: int,
                         ttl_bars: int, timestamp: str):
        """v5 pending limit entries: fill inside zone / cancel on break."""
        if not settings.PENDING_ENTRIES_ENABLED:
            return
        bar_open = float(bar["open"])
        bar_low = float(bar["low"])
        still = []
        for p in self.pending:
            if i - p["armed_idx"] > ttl_bars:
                continue  # expired
            zone_high, zone_low = p["zone_high"], p["zone_low"]
            invalid = zone_low - settings.PENDING_INVALID_ATR * (p["atr"] or atr_val or 0)
            if bar_open < invalid or bar_low < invalid:
                continue  # zone broke -> cancel
            if bar_low <= zone_high:
                fill = min(bar_open, zone_high)
                self._open_position(p["rec"], timestamp, entry_price=fill)
                continue
            still.append(p)
        self.pending = still

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame, symbol: str,
            window: int = 100, step: int = 1,
            warmup: int = 100, v5: bool = True,
            rec_cache: Optional[Dict[int, Dict]] = None) -> Dict:
        """
        Run backtest on a single symbol's OHLCV data.
        - window: bars to use per analysis window
        - step: bars to advance per iteration
        - warmup: initial bars to skip (need full indicators)
        - v5: enable veteran features (partial TP / chandelier / structure
              exits / time stop / pending entries / harmony gates)
        - rec_cache: {bar_index: rec} precomputed signals (fast A/B reruns)
        """
        log.info(
            f"[cyan]Backtesting[/] {symbol} - {len(df)} bars, "
            f"window={window}, step={step}, warmup={warmup}, v5={v5}"
        )
        if len(df) < warmup + window:
            return {"symbol": symbol, "status": "failed",
                    "reason": "Insufficient data"}

        self.capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.pending = []

        # bar duration in minutes (for time-stop conversion)
        try:
            idx = df.index
            diffs = (idx[1:] - idx[:-1]).total_seconds() / 60
            tf_minutes = float(np.median(diffs)) if len(diffs) else 60.0
        except Exception:
            tf_minutes = 60.0
        ttl_bars = max(1, int(settings.PENDING_TTL_HOURS * 60 / max(tf_minutes, 1)))

        # precomputed ATR series for the chandelier
        try:
            atr_series = ta_atr(df["high"], df["low"], df["close"], 14)
        except Exception:
            atr_series = pd.Series(np.nan, index=df.index)

        start_idx = warmup
        end_idx = len(df) - 1

        for i in range(start_idx, end_idx, step):
            window_df = df.iloc[i - window + 1: i + 1]
            if len(window_df) < window:
                continue
            bar = df.iloc[i]
            timestamp = str(bar.name)
            atr_val = atr_series.iloc[i]
            atr_val = float(atr_val) if not np.isnan(atr_val) else 0.0

            # --- exits first ---
            if v5:
                closed = self._check_exits_v5(bar, atr_val, window_df,
                                              timestamp, tf_minutes)
                for rec in closed:
                    self.trades.append(rec)
                # --- pending limit fills ---
                self._process_pending(bar, atr_val, i, ttl_bars, timestamp)
            else:
                self._check_exits(float(bar["close"]), timestamp)

            # --- try to open a new position if we have capacity ---
            if len(self.positions) < 3:  # max 3 simultaneous
                try:
                    if rec_cache is not None and i in rec_cache:
                        rec = rec_cache[i]
                    else:
                        rec = scorer.analyze_symbol(window_df, symbol)
                    if rec and self._passes_gates(rec, v5):
                        if v5 and settings.PENDING_ENTRIES_ENABLED \
                                and rec.get("entry_type") == "limit":
                            strong_momentum = (
                                rec.get("a_plus", False)
                                or float(rec.get(
                                    "admission_confidence",
                                    rec.get("confidence", 0)) or 0)
                                >= settings.PENDING_MOMENTUM_CONF
                            )
                            zone = rec.get("entry_zone") or {}
                            zone_high = zone.get("high")
                            price = rec.get("current_price") or 0
                            chase = settings.PENDING_CHASE_ATR * (rec.get("atr") or atr_val or 0)
                            if (zone_high and price and not strong_momentum
                                    and price > float(zone_high) + chase):
                                self.pending.append({
                                    "rec": rec,
                                    "armed_idx": i,
                                    "zone_high": float(zone_high),
                                    "zone_low": float(zone.get("low", zone_high)),
                                    "atr": float(rec.get("atr", 0) or atr_val),
                                })
                            else:
                                self._open_position(rec, timestamp)
                        else:
                            self._open_position(rec, timestamp)
                except Exception as e:
                    log.debug(f"Backtest bar {i} error: {e}")

            # Track equity
            open_pnl = sum(
                (float(bar["close"]) - p["entry_price"]) * p["size"]
                if p["direction"] == "bullish"
                else (p["entry_price"] - float(bar["close"])) * p["size"]
                for p in self.positions
            )
            self.equity_curve.append({
                "timestamp": timestamp,
                "capital": float(self.capital),
                "open_pnl": float(open_pnl),
                "equity": float(self.capital + open_pnl),
            })

        # Close any remaining positions at last price
        last_price = float(df.iloc[-1]["close"])
        last_ts = str(df.index[-1])
        if v5:
            for pos in self.positions[:]:
                rec = self._finish_trade(pos, last_price, last_ts, "EndOfData")
                self.trades.append(rec)
            self.positions = []
        else:
            self._check_exits(last_price, last_ts)

        # Compute metrics
        total_trades = len(self.trades)
        wins = [t for t in self.trades if t.get("pnl", 0) > 0]
        losses = [t for t in self.trades if t.get("pnl", 0) <= 0]
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
        gross_win = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = (
            gross_win / gross_loss
            if losses and gross_loss != 0 else 0
        )
        # Expectancy: average P&L per trade in % of notional
        expectancy_pct = (
            np.mean([t.get("pnl_pct", 0) for t in self.trades])
            if total_trades > 0 else 0
        )
        # Payoff ratio: avg win / avg loss (in % terms)
        avg_win_pct = np.mean([t.get("pnl_pct", 0) for t in wins]) if wins else 0
        avg_loss_pct = abs(np.mean([t.get("pnl_pct", 0) for t in losses])) if losses else 0
        payoff_ratio = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 0
        # Buy & hold comparison
        first_price = float(df.iloc[warmup]["close"])
        last_price = float(df.iloc[-1]["close"])
        buy_hold_return = (last_price - first_price) / first_price * 100

        max_drawdown = 0
        peak = self.initial_capital
        for point in self.equity_curve:
            if point["equity"] > peak:
                peak = point["equity"]
            dd = (peak - point["equity"]) / peak * 100
            if dd > max_drawdown:
                max_drawdown = dd
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100

        # v5 stats
        partial_trades = sum(1 for t in self.trades if t.get("partials", 0) > 0)
        exit_reasons = {}
        for t in self.trades:
            r = t.get("reason", "?")
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        result = {
            "symbol": symbol,
            "status": "success",
            "engine": "v5" if v5 else "v4.1-baseline",
            "initial_capital": float(self.initial_capital),
            "final_capital": float(self.capital),
            "total_return_pct": float(total_return),
            "buy_hold_return_pct": float(buy_hold_return),
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "avg_win_pct": float(avg_win_pct),
            "avg_loss_pct": float(avg_loss_pct),
            "payoff_ratio": float(payoff_ratio),
            "expectancy_pct_per_trade": float(expectancy_pct),
            "profit_factor": float(profit_factor),
            "max_drawdown_pct": float(max_drawdown),
            "fee_pct_per_side": float(settings.TRADING_FEE_PCT),
            "partial_tp_trades": partial_trades,
            "exit_reasons": exit_reasons,
            "equity_curve": self.equity_curve,
            "trades": self.trades,
        }
        log.info(
            f"[green]Backtest {symbol} (v5={v5}) complete[/] - "
            f"Return: {fmt_pct(total_return)} | "
            f"Trades: {total_trades} | "
            f"Win rate: {win_rate:.1f}% | "
            f"PF: {profit_factor:.2f} | "
            f"Max DD: {max_drawdown:.2f}%"
        )
        return result

    def save_report(self, result: Dict, output_file: Path = None) -> Path:
        """Save backtest report to JSON."""
        if output_file is None:
            output_file = Path("data/backtest_report.json")
        save_json(to_json_safe(result), output_file)
        log.info(f"Backtest report saved to {output_file}")
        return output_file


# Singleton
backtester = Backtester()
