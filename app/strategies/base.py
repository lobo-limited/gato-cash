"""Abstract base class for all prediction strategies.

Every strategy must produce ranked PredictionResult objects that include
per-position probability distributions over digits 0-9.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.models.draw import Draw


@dataclass
class PredictionResult:
    """A single prediction candidate with confidence and probability detail.

    Attributes
    ----------
    digits : list[int]
        Predicted digits, e.g. [3, 7, 1] (Daily 3) or [3, 7, 1, 5] (Daily 4).
    confidence : float
        Overall confidence score in the range [0.0, 1.0].
    digit_probabilities : list[list[float]]
        Per-position probability distributions.  Each inner list has exactly
        10 elements (one per digit 0-9) that sum to 1.0.
    metadata : dict
        Strategy-specific extra information (e.g. window size, parameters).
    """

    digits: list[int]
    confidence: float
    digit_probabilities: list[list[float]]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for pos_idx, probs in enumerate(self.digit_probabilities):
            if len(probs) != 10:
                raise ValueError(
                    f"digit_probabilities[{pos_idx}] must have exactly 10 elements, "
                    f"got {len(probs)}"
                )


class BaseStrategy(ABC):
    """Abstract base class that all prediction strategies must implement."""

    name: str = "base"

    @abstractmethod
    def predict(
        self,
        game_type: str,
        draw_time: str,
        history: list[Draw],
    ) -> list[PredictionResult]:
        """Generate ranked prediction candidates.

        Parameters
        ----------
        game_type : str
            ``"daily3"`` or ``"daily4"``.
        draw_time : str
            ``"midday"`` or ``"evening"``.
        history : list[Draw]
            Historical draws ordered **most-recent first**.

        Returns
        -------
        list[PredictionResult]
            Candidates sorted by descending confidence.
        """
        ...

    @abstractmethod
    def train(self, game_type: str, history: list[Draw]) -> None:
        """Train or update the strategy on historical data.

        For purely statistical strategies this can be a no-op; the method
        exists so that ML-based strategies can override it.
        """
        ...

    @staticmethod
    def get_digit_count(game_type: str) -> int:
        """Return the number of digit positions for a game type."""
        return 3 if game_type == "daily3" else 4

    @staticmethod
    def extract_digits(draw: Draw, game_type: str) -> list[int]:
        """Extract digit values from a Draw as a plain list."""
        digits = [draw.digit_1, draw.digit_2, draw.digit_3]
        if game_type != "daily3":
            digits.append(draw.digit_4)  # type: ignore[arg-type]
        return digits
