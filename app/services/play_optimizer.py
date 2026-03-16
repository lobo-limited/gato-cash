"""Play type optimizer that recommends the best play type based on
predicted digit probabilities and expected prize values.
"""

from __future__ import annotations

import math
from collections import Counter


class PlayTypeOptimizer:
    """Recommends optimal play type (straight, box, straight/box, combo)
    by computing expected values for each option.

    Supported game types: ``daily3`` (3 digits) and ``daily4`` (4 digits).
    """

    # Default prizes when historical averages are not available.
    DEFAULT_PRIZES: dict[str, dict[str, float]] = {
        "daily3": {"straight": 500.0, "box": 80.0},
        "daily4": {"straight": 5000.0, "box": 200.0},
    }

    # Cost per play.
    PLAY_COST = 1.0

    def recommend(
        self,
        digits: list[int],
        game_type: str,
        digit_probabilities: list[list[float]],
        avg_straight_prize: float | None = None,
        avg_box_prize: float | None = None,
    ) -> dict:
        """Compute expected values and recommend the best play type.

        Parameters
        ----------
        digits : list[int]
            Predicted digit sequence (e.g. [3, 7, 1]).
        game_type : str
            ``"daily3"`` or ``"daily4"``.
        digit_probabilities : list[list[float]]
            Per-position probability distributions (10 floats per position).
        avg_straight_prize : float | None
            Average straight-match prize; uses defaults when *None*.
        avg_box_prize : float | None
            Average box-match prize; uses defaults when *None*.

        Returns
        -------
        dict
            ``{"recommended_play", "expected_values", "probabilities", "reasoning"}``
        """
        defaults = self.DEFAULT_PRIZES.get(game_type, self.DEFAULT_PRIZES["daily3"])
        straight_prize = avg_straight_prize if avg_straight_prize is not None else defaults["straight"]
        box_prize = avg_box_prize if avg_box_prize is not None else defaults["box"]

        num_positions = 3 if game_type == "daily3" else 4

        # ------------------------------------------------------------------
        # Probability calculations
        # ------------------------------------------------------------------
        p_straight = self._p_straight(digits, digit_probabilities, num_positions)
        box_multiplier = self._box_multiplier(digits, num_positions)
        p_box = p_straight * box_multiplier

        # ------------------------------------------------------------------
        # Expected values
        # ------------------------------------------------------------------
        ev_straight = p_straight * straight_prize - self.PLAY_COST
        ev_box = p_box * box_prize - self.PLAY_COST

        # Straight/box costs $1 and pays half for each component.
        ev_straight_box = (
            p_straight * (straight_prize / 2.0)
            + p_box * (box_prize / 2.0)
            - self.PLAY_COST
        )

        # Combo costs $box_multiplier and pays straight prize on any arrangement.
        combo_cost = float(box_multiplier) * self.PLAY_COST
        ev_combo = p_box * straight_prize - combo_cost

        expected_values = {
            "straight": round(ev_straight, 6),
            "box": round(ev_box, 6),
            "straight_box": round(ev_straight_box, 6),
            "combo": round(ev_combo, 6),
        }

        probabilities = {
            "straight": round(p_straight, 8),
            "box": round(p_box, 8),
        }

        # ------------------------------------------------------------------
        # Recommendation: pick the play with the best (least negative) EV.
        # ------------------------------------------------------------------
        best_play = max(expected_values, key=lambda k: expected_values[k])

        reasoning = self._build_reasoning(
            digits, game_type, box_multiplier,
            probabilities, expected_values, best_play,
            straight_prize, box_prize,
        )

        return {
            "recommended_play": best_play,
            "expected_values": expected_values,
            "probabilities": probabilities,
            "reasoning": reasoning,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _p_straight(
        digits: list[int],
        digit_probabilities: list[list[float]],
        num_positions: int,
    ) -> float:
        """P(straight) = product of P(digit_i at position_i)."""
        p = 1.0
        for pos in range(num_positions):
            d = digits[pos]
            if pos < len(digit_probabilities) and d < len(digit_probabilities[pos]):
                p *= digit_probabilities[pos][d]
            else:
                p *= 0.1  # fallback: uniform
        return p

    @staticmethod
    def _box_multiplier(digits: list[int], num_positions: int) -> int:
        """Number of distinct permutations of the digit sequence.

        Daily 3 (3 digits):
            - All unique:    3! / 1       = 6
            - One pair:      3! / 2!      = 3
            - Triple:        3! / 3!      = 1

        Daily 4 (4 digits):
            - All unique:    4! / 1       = 24
            - One pair:      4! / 2!      = 12
            - Two pairs:     4! / (2!*2!) = 6
            - Triple:        4! / 3!      = 4
            - Quad:          4! / 4!      = 1

        General formula: n! / product(count_i!) for each distinct digit.
        """
        counts = Counter(digits[:num_positions])
        numerator = math.factorial(num_positions)
        denominator = 1
        for c in counts.values():
            denominator *= math.factorial(c)
        return numerator // denominator

    @staticmethod
    def _build_reasoning(
        digits: list[int],
        game_type: str,
        box_multiplier: int,
        probabilities: dict[str, float],
        expected_values: dict[str, float],
        best_play: str,
        straight_prize: float,
        box_prize: float,
    ) -> str:
        digit_str = "-".join(str(d) for d in digits)
        counts = Counter(digits)
        num_unique = len(counts)
        num_digits = len(digits)

        if num_digits == 3:
            if num_unique == 3:
                pattern_desc = "all unique digits"
            elif num_unique == 2:
                pattern_desc = "one pair"
            else:
                pattern_desc = "triple (all same)"
        else:  # daily4
            if num_unique == 4:
                pattern_desc = "all unique digits"
            elif num_unique == 3:
                pattern_desc = "one pair"
            elif num_unique == 2:
                max_count = max(counts.values())
                if max_count == 3:
                    pattern_desc = "triple + singleton"
                else:
                    pattern_desc = "two pairs"
            else:
                pattern_desc = "quad (all same)"

        lines = [
            f"Prediction {digit_str} ({game_type}) has {pattern_desc} "
            f"({box_multiplier} distinct arrangements).",
            f"P(straight) = {probabilities['straight']:.6f}, "
            f"P(box) = {probabilities['box']:.6f}.",
            f"Expected values -- straight: ${expected_values['straight']:.4f}, "
            f"box: ${expected_values['box']:.4f}, "
            f"straight/box: ${expected_values['straight_box']:.4f}, "
            f"combo: ${expected_values['combo']:.4f}.",
            f"Using avg prizes: straight=${straight_prize:.0f}, box=${box_prize:.0f}.",
            f"Recommended play: {best_play.upper()} "
            f"(best EV = ${expected_values[best_play]:.4f}).",
        ]
        return " ".join(lines)
