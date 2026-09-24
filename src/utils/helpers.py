"""Helper utilities."""
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np

from src.utils.logger import log


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)


def to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


def save_json(data: Any, path: Path) -> None:
    """Save data as JSON to a file (creates parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(data), f, ensure_ascii=False, indent=2, default=str)


def load_json(path: Path, default: Any = None) -> Any:
    """Load JSON from file. Returns default if file does not exist."""
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def fmt_price(p: float) -> str:
    """Format a price based on its magnitude."""
    if p is None or pd.isna(p):
        return "-"
    if p >= 1000:
        return f"{p:,.2f}"
    if p >= 1:
        return f"{p:,.4f}"
    if p >= 0.01:
        return f"{p:.6f}"
    return f"{p:.10f}"


def fmt_pct(p: float) -> str:
    """Format a percentage with sign."""
    if p is None or pd.isna(p):
        return "-"
    return f"{p:+.2f}%"


def retry_on_failure(func, retries: int = 3, delay: float = 1.0, exceptions=(Exception,)):
    """Retry a function on failure. Works as decorator (preserves args).

    v5: Binance 429 (RateLimitError) gets special treatment - the rate
    limiter already applied a global cooldown, so retry FEWER times with a
    LONG delay instead of hammering the API into an IP ban.
    """
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from src.core.rate_limiter import RateLimitError, rate_limiter
        last_exc = None
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except RateLimitError as e:
                # Cooldown is already active inside the limiter; wait it out.
                last_exc = e
                cooldown_left = rate_limiter.cooldown_remaining()
                if cooldown_left > 130.0:
                    # v5.2: hard backoff (418 IP ban >= 15 min) - sleeping 45s
                    # per fetch would just limp the whole cycle for nothing.
                    # Abort fast; the next cron tick retries when the ban lifts.
                    log.warning(
                        f"Rate-limited on {getattr(func, '__name__', 'call')} "
                        f"with {cooldown_left:.0f}s ban left - aborting fast"
                    )
                    raise
                backoff = max(15.0, cooldown_left + 1.0)
                log.warning(
                    f"Rate-limited on {getattr(func, '__name__', 'call')} "
                    f"(attempt {attempt + 1}/{retries}) - backing off {backoff:.0f}s"
                )
                if attempt < retries - 1:
                    time.sleep(min(backoff, 45.0))
            except exceptions as e:
                last_exc = e
                time.sleep(delay * (attempt + 1))
        raise last_exc
    return wrapper
