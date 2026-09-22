"""
File Logger - saves recommendations to CSV and JSON for archival and review.
"""
import csv
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from config.settings import settings
from src.utils.logger import log
from src.utils.helpers import save_json, now_utc

LOG_DIR = Path("data/logs")
CSV_FILE = LOG_DIR / "recommendations.csv"
JSON_FILE = Path("data/recommendations_history.json")


class FileLogger:
    """Persistent logger for recommendations."""

    def __init__(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        # Initialize CSV header if not exists
        if not CSV_FILE.exists():
            with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "direction", "current_price",
                    "confidence", "weighted_score", "expected_rise_pct",
                    "stop_loss", "take_profit", "risk_reward_ratio",
                    "atr_pct", "timeframe", "top_reasons"
                ])
        log.info("[green]FileLogger initialized[/] - CSV + JSON archival")

    def log_recommendations(self, recommendations: List[Dict]) -> int:
        """Append recommendations to CSV file."""
        if not recommendations:
            return 0
        ts = now_utc().isoformat()
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for rec in recommendations:
                top_reasons = "; ".join(
                    r for sig in rec.get("signals", [])
                    for r in sig.get("reasons", [])[:1]
                )[:200]
                writer.writerow([
                    ts,
                    rec.get("symbol", ""),
                    rec.get("direction", ""),
                    rec.get("current_price", 0),
                    rec.get("confidence", 0),
                    rec.get("weighted_score", 0),
                    rec.get("expected_rise_pct", 0),
                    rec.get("stop_loss", 0),
                    rec.get("take_profit", 0),
                    rec.get("risk_reward_ratio", 0),
                    rec.get("atr_pct", 0),
                    rec.get("timeframe", ""),
                    top_reasons,
                ])

        # Append to JSON history
        from src.utils.helpers import load_json, to_json_safe
        history = load_json(JSON_FILE, default=[])
        history.append({
            "timestamp": ts,
            "recommendations": to_json_safe(recommendations),
        })
        # Cap to last 1000 entries
        if len(history) > 1000:
            history = history[-1000:]
        save_json(history, JSON_FILE)
        log.info(f"[green]Logged {len(recommendations)} recommendations to CSV/JSON[/]")
        return len(recommendations)


# Singleton
file_logger = FileLogger()
