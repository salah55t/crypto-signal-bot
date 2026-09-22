"""
Backtesting Engine
Replays historical OHLCV data through the SignalScorer to evaluate
strategy performance over time.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import time
from src.analysis.scorer import scorer
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

    def _open_position(self, rec: Dict, timestamp: str) -> Dict:
        entry = rec["current_price"]
        sl = rec["stop_loss"]
        # Position sizing: 5% of capital per trade
        notional = self.capital * 0.05
        size = notional / entry if entry > 0 else 0
        pos = {
            "symbol": rec["symbol"],
            "entry_price": entry,
            "stop_loss": sl,
            "take_profit": rec["take_profit"],
            "size": size,
            "entry_time": timestamp,
            "confidence": rec["confidence"],
            "expected_rise_pct": rec["expected_rise_pct"],
            "direction": rec["direction"],
        }
        self.positions.append(pos)
        return pos

    def _check_exits(self, current_price: float, timestamp: str) -> List[Dict]:
        """Check all open positions for SL/TP hits and close them."""
        closed = []
        remaining = []
        for pos in self.positions:
            if pos["direction"] == "bullish":
                if current_price <= pos["stop_loss"]:
                    pnl = (current_price - pos["entry_price"]) * pos["size"]
                    pos["exit_price"] = current_price
                    pos["exit_time"] = timestamp
                    pos["pnl"] = pnl
                    pos["reason"] = "SL"
                    closed.append(pos)
                    self.capital += pnl
                    continue
                elif current_price >= pos["take_profit"]:
                    pnl = (current_price - pos["entry_price"]) * pos["size"]
                    pos["exit_price"] = current_price
                    pos["exit_time"] = timestamp
                    pos["pnl"] = pnl
                    pos["reason"] = "TP"
                    closed.append(pos)
                    self.capital += pnl
                    continue
            else:  # bearish
                if current_price >= pos["stop_loss"]:
                    pnl = (pos["entry_price"] - current_price) * pos["size"]
                    pos["exit_price"] = current_price
                    pos["exit_time"] = timestamp
                    pos["pnl"] = pnl
                    pos["reason"] = "SL"
                    closed.append(pos)
                    self.capital += pnl
                    continue
                elif current_price <= pos["take_profit"]:
                    pnl = (pos["entry_price"] - current_price) * pos["size"]
                    pos["exit_price"] = current_price
                    pos["exit_time"] = timestamp
                    pos["pnl"] = pnl
                    pos["reason"] = "TP"
                    closed.append(pos)
                    self.capital += pnl
                    continue
            remaining.append(pos)
        self.positions = remaining
        self.trades.extend(closed)
        return closed

    def run(self, df: pd.DataFrame, symbol: str,
            window: int = 100, step: int = 1,
            warmup: int = 100) -> Dict:
        """
        Run backtest on a single symbol's OHLCV data.
        - window: bars to use per analysis window
        - step: bars to advance per iteration
        - warmup: initial bars to skip (need full indicators)
        """
        log.info(
            f"[cyan]Backtesting[/] {symbol} - {len(df)} bars, "
            f"window={window}, step={step}, warmup={warmup}"
        )
        if len(df) < warmup + window:
            return {"symbol": symbol, "status": "failed",
                    "reason": "Insufficient data"}

        self.capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []

        start_idx = warmup
        end_idx = len(df) - 1

        for i in range(start_idx, end_idx, step):
            window_df = df.iloc[i - window + 1: i + 1]
            if len(window_df) < window:
                continue
            current_bar = df.iloc[i]
            current_price = float(current_bar["close"])
            timestamp = str(current_bar.name)

            # Check exits first
            self._check_exits(current_price, timestamp)

            # Try to open a new position if we have capacity
            if len(self.positions) < 3:  # max 3 simultaneous
                try:
                    rec = scorer.analyze_symbol(window_df, symbol)
                    if (rec.get("direction") == "bullish"
                            and rec.get("confidence", 0) >= 70
                            and rec.get("risk_reward_ratio", 0) >= 2.0):
                        self._open_position(rec, timestamp)
                except Exception as e:
                    log.debug(f"Backtest bar {i} error: {e}")

            # Track equity
            open_pnl = sum(
                (current_price - p["entry_price"]) * p["size"]
                if p["direction"] == "bullish"
                else (p["entry_price"] - current_price) * p["size"]
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
        self._check_exits(last_price, last_ts)

        # Compute metrics
        total_trades = len(self.trades)
        wins = [t for t in self.trades if t.get("pnl", 0) > 0]
        losses = [t for t in self.trades if t.get("pnl", 0) <= 0]
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
        profit_factor = (
            sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses))
            if losses and sum(t["pnl"] for t in losses) != 0 else 0
        )
        max_drawdown = 0
        peak = self.initial_capital
        for point in self.equity_curve:
            if point["equity"] > peak:
                peak = point["equity"]
            dd = (peak - point["equity"]) / peak * 100
            if dd > max_drawdown:
                max_drawdown = dd
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100

        result = {
            "symbol": symbol,
            "status": "success",
            "initial_capital": float(self.initial_capital),
            "final_capital": float(self.capital),
            "total_return_pct": float(total_return),
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "max_drawdown_pct": float(max_drawdown),
            "equity_curve": self.equity_curve,
            "trades": self.trades,
        }
        log.info(
            f"[green]Backtest {symbol} complete[/] - "
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
