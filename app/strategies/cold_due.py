"""Cold / Due prediction strategy.

Predicts digits that are most "overdue" at each position -- i.e., the digit
with the longest gap (most draws since its last appearance).  The underlying
heuristic is the gambler's-fallacy assumption that an absent digit is "due"
to appear.

Gaps are converted to pseudo-probabilities so the output conforms to the
standard PredictionResult contract.
"""

from __future__ import annotations

import numpy as np

from app.models.draw import Draw
from app.strategies.base import BaseStrategy, PredictionResult


class ColdDueStrategy(BaseStrategy):
    """Predict the digits with the longest absence at each position."""

    name: str = "cold_due"

    def __init__(self, window: int = 200) -> None:
        """
        Parameters
        ----------
        window : int
            Maximum number of recent draws to scan for gaps.
        """
        self.window = window

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def train(self, game_type: str, history: list[Draw]) -> None:
        """No-op for this purely statistical strategy."""

    def predict(
        self,
        game_type: str,
        draw_time: str,
        history: list[Draw],
    ) -> list[PredictionResult]:
        """Generate predictions based on digit gap / overdue analysis.

        Returns up to 5 candidates ranked by confidence.
        """
        n_pos = self.get_digit_count(game_type)
        draws = history[: self.window]
        n_draws = len(draws)

        if n_draws == 0:
            return []

        # Compute current gap per (position, digit)
        # gap = number of draws since the digit last appeared at that position
        # If never seen, gap = n_draws (maximum)
        gap = np.full((n_pos, 10), n_draws, dtype=np.float64)

        for idx, draw in enumerate(draws):
            digits = self.extract_digits(draw, game_type)
            for pos, d in enumerate(digits):
                if gap[pos, d] == n_draws:
                    # First (most recent) occurrence
                    gap[pos, d] = idx

        # Compute average gap per (position, digit) for confidence scoring
        # Average gap ~ expected interval = n_draws / count
        count = np.zeros((n_pos, 10), dtype=np.float64)
        for draw in draws:
            digits = self.extract_digits(draw, game_type)
            for pos, d in enumerate(digits):
                count[pos, d] += 1

        avg_gap = np.where(count > 0, n_draws / count, float(n_draws))

        # Convert gaps to pseudo-probabilities: longer gap -> higher "probability"
        # Use softmax over gaps so the distribution sums to 1
        prob = np.zeros((n_pos, 10), dtype=np.float64)
        for pos in range(n_pos):
            # Temperature-scaled softmax; using gap values directly
            g = gap[pos]
            g_shifted = g - g.max()  # numerical stability
            exp_g = np.exp(g_shifted)
            prob[pos] = exp_g / exp_g.sum()

        # Generate candidates
        results: list[PredictionResult] = []
        ranked = np.argsort(-prob, axis=1)  # descending probability
        top_k = 5

        for candidate_idx in range(top_k):
            chosen = ranked[:, 0].tolist()

            if candidate_idx > 0:
                swap_pos = (candidate_idx - 1) % n_pos
                chosen[swap_pos] = int(ranked[swap_pos, 1])

            # Confidence: how overdue the chosen digits are relative to their
            # average gaps.  ratio > 1 means overdue.
            ratios = []
            for pos in range(n_pos):
                current = float(gap[pos, chosen[pos]])
                avg = float(avg_gap[pos, chosen[pos]])
                if avg > 0:
                    ratios.append(current / avg)
                else:
                    ratios.append(0.0)

            # Normalise: ratio of 2 => very overdue, cap at 1.0 confidence
            raw_conf = float(np.mean(ratios)) / 2.0
            confidence = min(max(raw_conf, 0.0), 1.0)

            results.append(
                PredictionResult(
                    digits=chosen,
                    confidence=round(confidence, 4),
                    digit_probabilities=[prob[p].tolist() for p in range(n_pos)],
                    metadata={
                        "strategy": self.name,
                        "window": self.window,
                        "draws_used": n_draws,
                        "candidate_rank": candidate_idx + 1,
                        "chosen_gaps": [
                            float(gap[pos, chosen[pos]]) for pos in range(n_pos)
                        ],
                        "chosen_avg_gaps": [
                            round(float(avg_gap[pos, chosen[pos]]), 2)
                            for pos in range(n_pos)
                        ],
                    },
                )
            )

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results
