"""Pair Analysis prediction strategy.

Builds frequency matrices for adjacent digit pairs at each consecutive
position pair (1-2, 2-3, 3-4) and chains the most-frequent pairs together
to form a complete prediction.

For example, if the most frequent pair at positions 1-2 is (3, 7), the
strategy then looks for the most frequent pair starting with 7 at positions
2-3 to pick the third digit, and so on.
"""

from __future__ import annotations

import numpy as np

from app.models.draw import Draw
from app.strategies.base import BaseStrategy, PredictionResult


class PairAnalysisStrategy(BaseStrategy):
    """Predict by chaining the highest-frequency adjacent digit pairs."""

    name: str = "pair_analysis"

    def __init__(self, window: int = 150) -> None:
        """
        Parameters
        ----------
        window : int
            Number of most-recent draws to consider when building pair matrices.
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
        """Generate predictions by chaining high-frequency adjacent pairs.

        Returns up to 5 candidates ranked by confidence.
        """
        n_pos = self.get_digit_count(game_type)
        draws = history[: self.window]
        n_draws = len(draws)

        if n_draws == 0:
            return []

        # Build pair frequency matrices: pair_freq[adj_idx] is a 10x10 matrix
        # where pair_freq[adj_idx][a][b] = count of (digit_a, digit_b) at
        # positions (adj_idx, adj_idx+1)
        n_adj = n_pos - 1
        pair_freq = np.zeros((n_adj, 10, 10), dtype=np.float64)

        for draw in draws:
            digits = self.extract_digits(draw, game_type)
            for adj in range(n_adj):
                pair_freq[adj, digits[adj], digits[adj + 1]] += 1

        # Normalise each adjacency matrix to joint probabilities
        pair_prob = np.zeros_like(pair_freq)
        for adj in range(n_adj):
            total = pair_freq[adj].sum()
            if total > 0:
                pair_prob[adj] = pair_freq[adj] / total

        # Also compute marginal (per-position) probabilities for output
        # Derive from pair matrices: position 0 from adj 0 row sums,
        # position i from adj (i-1) column sums (or adj i row sums)
        pos_prob = np.zeros((n_pos, 10), dtype=np.float64)
        # Position 0: row sums of adj 0
        pos_prob[0] = pair_freq[0].sum(axis=1)
        # Middle positions: average of adj (i-1) col sums and adj i row sums
        for pos in range(1, n_pos - 1):
            from_prev = pair_freq[pos - 1].sum(axis=0)
            from_next = pair_freq[pos].sum(axis=1)
            pos_prob[pos] = (from_prev + from_next) / 2.0
        # Last position: column sums of last adjacency
        pos_prob[n_pos - 1] = pair_freq[n_adj - 1].sum(axis=0)
        # Normalise
        for pos in range(n_pos):
            total = pos_prob[pos].sum()
            if total > 0:
                pos_prob[pos] /= total
            else:
                pos_prob[pos] = np.ones(10) / 10.0

        # Average pair frequency for confidence baseline
        avg_pair_freq = n_draws / 100.0  # each of 100 pairs expected this often

        # Generate candidates using different starting pairs
        results: list[PredictionResult] = []

        # Get top-k starting pairs at adjacency 0
        flat_indices = np.argsort(pair_freq[0].ravel())[::-1]
        top_k = min(5, len(flat_indices))

        for candidate_idx in range(top_k):
            flat_idx = flat_indices[candidate_idx]
            d0, d1 = divmod(int(flat_idx), 10)
            chain = [d0, d1]
            chain_freqs = [float(pair_freq[0, d0, d1])]

            # Extend the chain for remaining positions
            for adj in range(1, n_adj):
                prev_digit = chain[-1]
                # Pick the most frequent pair starting with prev_digit
                next_digit = int(np.argmax(pair_freq[adj, prev_digit]))
                chain.append(next_digit)
                chain_freqs.append(float(pair_freq[adj, prev_digit, next_digit]))

            # Confidence: how much the chosen pairs exceed the average
            if avg_pair_freq > 0:
                excess_ratios = [f / avg_pair_freq for f in chain_freqs]
                raw_conf = float(np.mean(excess_ratios)) / 5.0  # normalise
            else:
                raw_conf = 0.0
            confidence = min(max(raw_conf, 0.0), 1.0)

            results.append(
                PredictionResult(
                    digits=chain,
                    confidence=round(confidence, 4),
                    digit_probabilities=[pos_prob[p].tolist() for p in range(n_pos)],
                    metadata={
                        "strategy": self.name,
                        "window": self.window,
                        "draws_used": n_draws,
                        "candidate_rank": candidate_idx + 1,
                        "chain_pair_frequencies": chain_freqs,
                    },
                )
            )

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results
