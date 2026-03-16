"""Frequency-based prediction strategy.

Predicts the most frequently occurring digit at each position within a
configurable recent-draw window.  Confidence is derived from how much the
top digit's observed frequency exceeds the uniform expectation of 10 %.
"""

from __future__ import annotations

import numpy as np

from app.models.draw import Draw
from app.strategies.base import BaseStrategy, PredictionResult


class FrequencyStrategy(BaseStrategy):
    """Pick the historically most-frequent digit at every position."""

    name: str = "frequency"

    def __init__(self, window: int = 100) -> None:
        """
        Parameters
        ----------
        window : int
            Number of most-recent draws to consider.
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
        """Generate predictions from positional digit frequencies.

        Returns up to 5 candidates ranked by confidence.
        """
        n_pos = self.get_digit_count(game_type)
        draws = history[: self.window]
        n_draws = len(draws)

        if n_draws == 0:
            return []

        # Count frequencies per position: shape (n_pos, 10)
        freq = np.zeros((n_pos, 10), dtype=np.float64)
        for draw in draws:
            digits = self.extract_digits(draw, game_type)
            for pos, d in enumerate(digits):
                freq[pos, d] += 1

        # Normalise to probability distributions
        prob = freq / freq.sum(axis=1, keepdims=True)

        # Generate multiple candidates by picking top-k digits per position
        results: list[PredictionResult] = []
        top_k = 5

        # Rank digits at each position by probability (descending)
        ranked = np.argsort(-prob, axis=1)  # shape (n_pos, 10)

        # Candidate 0: argmax at every position (the "best" prediction)
        # Candidates 1-4: swap one position at a time to its 2nd-best digit
        for candidate_idx in range(top_k):
            chosen = ranked[:, 0].tolist()  # start from top-1 everywhere

            if candidate_idx > 0:
                # Swap position (candidate_idx - 1) to its 2nd best digit
                swap_pos = (candidate_idx - 1) % n_pos
                chosen[swap_pos] = int(ranked[swap_pos, 1])

            # Confidence: average excess probability above uniform 0.1
            conf_scores = []
            for pos in range(n_pos):
                top_prob = float(prob[pos, chosen[pos]])
                excess = max(0.0, top_prob - 0.1)  # excess above uniform
                conf_scores.append(excess / 0.9)  # normalise: max excess is 0.9
            confidence = float(np.mean(conf_scores))
            confidence = min(max(confidence, 0.0), 1.0)

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
                    },
                )
            )

        # Sort by confidence descending (first candidate should already be best)
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results
