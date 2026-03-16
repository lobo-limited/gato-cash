"""XGBoost prediction strategy.

Uses XGBClassifier with a separate model per digit position and the same
feature set as RandomForestStrategy (via the shared feature_engineering
module).
"""

from __future__ import annotations

import logging

import numpy as np
from xgboost import XGBClassifier

from app.models.draw import Draw
from app.services.feature_engineering import build_feature_matrix, build_features
from app.strategies.base import BaseStrategy, PredictionResult

logger = logging.getLogger(__name__)

_MIN_DRAWS = 200


class XGBoostStrategy(BaseStrategy):
    """Per-position XGBClassifier (multi-class softprob digit prediction)."""

    name: str = "xgboost"

    def __init__(self) -> None:
        self._models: dict[str, list[XGBClassifier]] = {}
        self._is_trained: dict[str, bool] = {}

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, game_type: str, history: list[Draw]) -> None:
        """Train one XGBClassifier per digit position.

        ``history`` comes most-recent first (BaseStrategy contract).
        """
        chrono = list(reversed(history))
        n = len(chrono)

        if n < _MIN_DRAWS:
            logger.warning(
                "XGBoost: only %d draws for %s (need %d) — skipping training",
                n, game_type, _MIN_DRAWS,
            )
            self._is_trained[game_type] = False
            return

        X, y = build_feature_matrix(chrono, game_type, min_history=100)
        if X.shape[0] == 0:
            logger.warning("XGBoost: no training samples for %s", game_type)
            self._is_trained[game_type] = False
            return

        n_pos = self.get_digit_count(game_type)
        models: list[XGBClassifier] = []
        for pos in range(n_pos):
            clf = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective="multi:softprob",
                num_class=10,
                eval_metric="mlogloss",
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            )
            clf.fit(X, y[:, pos])
            models.append(clf)
            logger.info(
                "XGBoost: trained position %d for %s  (samples=%d, features=%d)",
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

        # Auto-train if needed
        if not self._is_trained.get(game_type, False):
            self.train(game_type, history)

        if not self._is_trained.get(game_type, False):
            return self._uniform_prediction(n_pos, game_type)

        # Build features for the next draw
        chrono = list(reversed(history))
        chrono.append(history[0])  # placeholder for target draw metadata
        idx = len(chrono) - 1
        feat = build_features(chrono, idx, game_type).reshape(1, -1)

        probs: list[list[float]] = []
        models = self._models[game_type]
        for pos in range(n_pos):
            prob_raw = models[pos].predict_proba(feat)[0]
            # XGBClassifier with num_class=10 always outputs 10 columns
            full_prob = np.zeros(10, dtype=np.float64)
            for cls_idx, cls_label in enumerate(models[pos].classes_):
                full_prob[int(cls_label)] = prob_raw[cls_idx]
            total = full_prob.sum()
            if total > 0:
                full_prob /= total
            else:
                full_prob[:] = 0.1
            probs.append(full_prob.tolist())

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
        prob_arr = np.array(probs)
        ranked = np.argsort(-prob_arr, axis=1)

        results: list[PredictionResult] = []
        for c in range(5):
            chosen = ranked[:, 0].tolist()
            if c > 0:
                swap_pos = (c - 1) % n_pos
                chosen[swap_pos] = int(ranked[swap_pos, 1])

            conf_scores = [float(prob_arr[p, chosen[p]]) for p in range(n_pos)]
            confidence = round(min(max(float(np.mean(conf_scores)), 0.0), 1.0), 4)

            results.append(
                PredictionResult(
                    digits=chosen,
                    confidence=confidence,
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
                    "strategy": "xgboost",
                    "game_type": game_type,
                    "note": "insufficient training data",
                },
            )
        ]
