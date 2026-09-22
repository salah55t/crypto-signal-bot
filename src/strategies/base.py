"""
Base Strategy Class
All strategies inherit from this and produce a signal with score.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class Signal:
    """Standardized signal output."""
    strategy: str
    direction: str = "neutral"  # bullish | bearish | neutral
    score: float = 0.0  # -100 (strong bearish) to +100 (strong bullish)
    confidence: float = 0.0  # 0-1
    reasons: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    name: str = "base"
    weight: float = 1.0  # relative weight in final score

    def __init__(self, weight: float = None):
        if weight is not None:
            self.weight = weight

    @abstractmethod
    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        """
        Run the strategy on the given OHLCV DataFrame.
        Returns a Signal object.
        """
        pass

    def _bull(self, score: float, reason: str, details: Dict = None) -> Signal:
        return Signal(
            strategy=self.name,
            direction="bullish",
            score=score,
            confidence=abs(score) / 100.0,
            reasons=[reason] if isinstance(reason, str) else reason,
            details=details or {},
        )

    def _bear(self, score: float, reason: str, details: Dict = None) -> Signal:
        return Signal(
            strategy=self.name,
            direction="bearish",
            score=-score,
            confidence=abs(score) / 100.0,
            reasons=[reason] if isinstance(reason, str) else reason,
            details=details or {},
        )

    def _neutral(self, reason: str, details: Dict = None) -> Signal:
        return Signal(
            strategy=self.name,
            direction="neutral",
            score=0.0,
            confidence=0.0,
            reasons=[reason] if isinstance(reason, str) else reason,
            details=details or {},
        )
