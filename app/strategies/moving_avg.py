"""Moving Average prediction strategy.

Computes a rolling mean of digit *values* at each position over a recent
window.  The rounded mean is the prediction.  A Gaussian-like probability
distribution centred on the mean (with standard deviation derived from
the data) provides the per-position probability vector.

Low variance at a position => high confidence; high variance => low
confidence.
"""

from __future__ import annotations

import numpy as np

from app.models.draw import Draw
from app.strategies.base import BaseStrategy, PredictionResult


class MovingAverageStrategy(BaseStrategy):
    """Predict using rolling mean of digit values per position."""

    name: str = "moving_average"

    def __init__(self, window: int = 50) -> None:
        """
        Parameters
        ----------
        window : int
            Number of most-recent draws used for the rolling mean.
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
        """Generate predictions from rolling mean digit values.

        Returns up to 5 candidates ranked by confidence.
        """
        n_pos = self.get_digit_count(game_type)
        draws = history[: self.window]
        n_draws = len(draws)

        if n_draws == 0:
            return []

        # Collect digit values per position: shape (n_draws, n_pos)
        values = np.zeros((n_draws, n_pos), dtype=np.float64)
        for i, draw in enumerate(draws):
            digits = self.extract_digits(draw, game_type)
            for pos in range(n_pos):
                values[i, pos] = digits[pos]

        # Rolling mean and standard deviation per position
        means = values.mean(axis=0)  # shape (n_pos,)
        stds = values.std(axis=0, ddof=1) if n_draws > 1 else np.zeros(n_pos)

        # Build Gaussian-like probability distribution centred on the mean
        prob = np.zeros((n_pos, 10), dtype=np.float64)
        for pos in range(n_pos):
            mu = means[pos]
            sigma = max(stds[pos], 0.5)  # floor to avoid degenerate distributions
            for d in range(10):
                # Un-normalised Gaussian density
                prob[pos, d] = np.exp(-0.5 * ((d - mu) / sigma) ** 2)
            # Normalise
            total = prob[pos].sum()
            if total > 0:
                prob[pos] /= total
            else:
                prob[pos] = np.ones(10) / 10.0

        # Primary prediction: rounded mean at each position
        primary = [int(np.clip(round(m), 0, 9)) for m in means]

        # Generate candidates
        results: list[PredictionResult] = []
        ranked = np.argsort(-prob, axis=1)
        top_k = 5

        for candidate_idx in range(top_k):
            if candidate_idx == 0:
                chosen = primary[:]
            else:
                # Variations: use the (candidate_idx+1)-th most probable digit
                # at one position
                chosen = primary[:]
                swap_pos = (candidate_idx - 1) % n_pos
                chosen[swap_pos] = int(ranked[swap_pos, 1])

            # Confidence from variance: low std => high confidence
            # Max std for digits 0-9 is ~2.87; normalise inversely
            max_std = 2.87
            variance_scores = []
            for pos in range(n_pos):
                inv_var = 1.0 - min(stds[pos] / max_std, 1.0)
                variance_scores.append(inv_var)
            confidence = float(np.mean(variance_scores))
            confidence = min(max(confidence, 0.0), 1.0)

            # Slightly reduce confidence for non-primary candidates
            if candidate_idx > 0:
                confidence *= 0.9

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
                        "position_means": [round(float(m), 2) for m in means],
                        "position_stds": [round(float(s), 2) for s in stds],
                    },
                )
            )

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results
