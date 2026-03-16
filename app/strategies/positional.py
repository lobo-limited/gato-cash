"""Positional Trend prediction strategy using Exponentially Weighted Moving Average.

For each position, the strategy one-hot encodes each digit's appearances
across the draw history and computes an EWMA (alpha = 0.1 by default).
The digit with the highest EWMA value at each position is the prediction.
Confidence reflects how dominant that digit is relative to the rest of the
EWMA distribution.
"""

from __future__ import annotations

import numpy as np

from app.models.draw import Draw
from app.strategies.base import BaseStrategy, PredictionResult


class PositionalTrendStrategy(BaseStrategy):
    """Predict digits using exponentially weighted moving average of per-position frequencies."""

    name: str = "positional_trend"

    def __init__(self, alpha: float = 0.1) -> None:
        """
        Parameters
        ----------
        alpha : float
            EWMA smoothing factor.  Closer to 1.0 gives more weight to recent
            draws; closer to 0.0 gives a longer memory.
        """
        self.alpha = alpha

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
        """Generate predictions from EWMA-smoothed positional digit frequencies.

        Returns up to 5 candidates ranked by confidence.
        """
        n_pos = self.get_digit_count(game_type)
        n_draws = len(history)

        if n_draws == 0:
            return []

        # We iterate chronologically (oldest first) so the EWMA reflects
        # recency at the end.
        ordered = list(reversed(history))

        # Compute EWMA for each (position, digit) pair
        # ewma[pos][digit] tracks the running average
        ewma = np.full((n_pos, 10), 0.1, dtype=np.float64)  # init to uniform

        for draw in ordered:
            digits = self.extract_digits(draw, game_type)
            for pos in range(n_pos):
                # One-hot for this draw
                one_hot = np.zeros(10, dtype=np.float64)
                one_hot[digits[pos]] = 1.0
                ewma[pos] = self.alpha * one_hot + (1.0 - self.alpha) * ewma[pos]

        # Normalise each position to a proper probability distribution
        prob = ewma / ewma.sum(axis=1, keepdims=True)

        # Generate candidates
        results: list[PredictionResult] = []
        ranked = np.argsort(-prob, axis=1)
        top_k = 5

        for candidate_idx in range(top_k):
            chosen = ranked[:, 0].tolist()

            if candidate_idx > 0:
                swap_pos = (candidate_idx - 1) % n_pos
                chosen[swap_pos] = int(ranked[swap_pos, 1])

            # Confidence: how dominant the top digit is
            # Measured as (top_prob - second_prob) averaged across positions
            dominance_scores = []
            for pos in range(n_pos):
                sorted_probs = np.sort(prob[pos])[::-1]
                top_p = sorted_probs[0]
                second_p = sorted_probs[1]
                # Dominance: gap between 1st and 2nd, normalised to [0, 1]
                dominance_scores.append((top_p - second_p) / max(top_p, 1e-9))

            confidence = float(np.mean(dominance_scores))
            confidence = min(max(confidence, 0.0), 1.0)

            results.append(
                PredictionResult(
                    digits=chosen,
                    confidence=round(confidence, 4),
                    digit_probabilities=[prob[p].tolist() for p in range(n_pos)],
                    metadata={
                        "strategy": self.name,
                        "alpha": self.alpha,
                        "draws_used": n_draws,
                        "candidate_rank": candidate_idx + 1,
                    },
                )
            )

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results
