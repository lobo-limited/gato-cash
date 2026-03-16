"""Random Forest prediction strategy.

Uses scikit-learn RandomForestClassifier with a separate model per digit
position.  Features are built by the shared ``feature_engineering`` module.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app.models.draw import Draw
from app.services.feature_engineering import build_feature_matrix, build_features
from app.strategies.base import BaseStrategy, PredictionResult

logger = logging.getLogger(__name__)

_MIN_DRAWS = 200  # minimum history length to train


class RandomForestStrategy(BaseStrategy):
    """Per-position RandomForestClassifier (10-class digit classification)."""

    name: str = "random_forest"

    def __init__(self) -> None:
        # Mapping: game_type -> list of fitted classifiers (one per position)
        self._models: dict[str, list[RandomForestClassifier]] = {}
        self._is_trained: dict[str, bool] = {}

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, game_type: str, history: list[Draw]) -> None:
        """Fit one RandomForestClassifier per digit position.

        ``history`` is expected **most-recent first** (matching the API
        contract in ``BaseStrategy``).  We reverse it to chronological order
        before feature construction.
        """
        chrono = list(reversed(history))
        n = len(chrono)

        if n < _MIN_DRAWS:
            logger.warning(
                "RandomForest: only %d draws for %s (need %d) — skipping training",
                n, game_type, _MIN_DRAWS,
            )
            self._is_trained[game_type] = False
            return

        X, y = build_feature_matrix(chrono, game_type, min_history=100)
        if X.shape[0] == 0:
            logger.warning("RandomForest: no training samples for %s", game_type)
            self._is_trained[game_type] = False
            return

        n_pos = self.get_digit_count(game_type)
        models: list[RandomForestClassifier] = []
        for pos in range(n_pos):
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            clf.fit(X, y[:, pos])
            models.append(clf)
            logger.info(
                "RandomForest: trained position %d for %s  (samples=%d, features=%d)",
                pos, game_type, X.shape[0], X.shape[1],
            )

        self._models[game_type] = models
        self._is_trained[game_type] = True

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(
        self,
        game_type: str,
        draw_time: str,
        history: list[Draw],
    ) -> list[PredictionResult]:
        n_pos = self.get_digit_count(game_type)

        # Auto-train if not already done
        if not self._is_trained.get(game_type, False):
            self.train(game_type, history)

        # If still not trained (insufficient data), return uniform prediction
        if not self._is_trained.get(game_type, False):
            return self._uniform_prediction(n_pos, game_type)

        # Build feature vector for the *next* draw.
        # We fabricate a lightweight "target draw" that carries the draw_time
        # and date metadata the feature builder needs.  The simplest approach
        # is to use history in chronological order and treat index=len as the
        # prediction target — but build_features needs draws[index] for day-of-
        # week and draw_time.  We re-use the most-recent draw as a proxy (same
        # date/time info will be close enough for the next draw).
        chrono = list(reversed(history))
        # Append a copy of the latest draw as the "to-predict" placeholder
        chrono.append(history[0])  # most-recent draw
        idx = len(chrono) - 1

        feat = build_features(chrono, idx, game_type).reshape(1, -1)

        # Predict probabilities per position
        probs: list[list[float]] = []
        models = self._models[game_type]
        for pos in range(n_pos):
            prob_all = models[pos].predict_proba(feat)[0]
            # predict_proba may not return all 10 classes if some were missing
            full_prob = np.zeros(10, dtype=np.float64)
            for cls_idx, cls_label in enumerate(models[pos].classes_):
                full_prob[int(cls_label)] = prob_all[cls_idx]
            # Re-normalise just in case
            total = full_prob.sum()
            if total > 0:
                full_prob /= total
            else:
                full_prob[:] = 0.1
            probs.append(full_prob.tolist())

        # Generate top candidates
        return self._build_candidates(probs, n_pos, game_type)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_candidates(
        self,
        probs: list[list[float]],
        n_pos: int,
        game_type: str,
    ) -> list[PredictionResult]:
        """Build ranked PredictionResult candidates from probability arrays."""
        prob_arr = np.array(probs)  # shape (n_pos, 10)
        ranked = np.argsort(-prob_arr, axis=1)

        results: list[PredictionResult] = []
        top_k = 5
        for c in range(top_k):
            chosen = ranked[:, 0].tolist()
            if c > 0:
                swap_pos = (c - 1) % n_pos
                chosen[swap_pos] = int(ranked[swap_pos, 1])

            conf_scores = [float(prob_arr[p, chosen[p]]) for p in range(n_pos)]
            confidence = float(np.mean(conf_scores))
            confidence = min(max(confidence, 0.0), 1.0)

            results.append(
                PredictionResult(
                    digits=chosen,
                    confidence=round(confidence, 4),
                    digit_probabilities=[probs[p] for p in range(n_pos)],
                    metadata={
                        "strategy": self.name,
                        "game_type": game_type,
                        "candidate_rank": c + 1,
                    },
                )
            )
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    @staticmethod
    def _uniform_prediction(n_pos: int, game_type: str) -> list[PredictionResult]:
        uniform = [0.1] * 10
        return [
            PredictionResult(
                digits=[0] * n_pos,
                confidence=0.0,
                digit_probabilities=[uniform[:] for _ in range(n_pos)],
                metadata={
                    "strategy": "random_forest",
                    "game_type": game_type,
                    "note": "insufficient training data",
                },
            )
        ]
