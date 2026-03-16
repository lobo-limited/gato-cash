"""LSTM prediction strategy.

Uses a TensorFlow/Keras model with two LSTM layers to learn sequential
patterns in draw history.  A single model outputs softmax probabilities for
all digit positions simultaneously.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from app.models.draw import Draw
from app.strategies.base import BaseStrategy, PredictionResult

logger = logging.getLogger(__name__)

_SEQ_LEN = 50       # number of past draws per input sequence
_MIN_DRAWS = 500     # minimum history length to train (need enough for sequences)
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def _extract_digits(draw: Draw, game_type: str) -> list[int]:
    digits = [draw.digit_1, draw.digit_2, draw.digit_3]
    if game_type != "daily3":
        digits.append(draw.digit_4)  # type: ignore[arg-type]
    return digits


def _one_hot_draw(draw: Draw, game_type: str) -> np.ndarray:
    """One-hot encode a single draw: shape ``(digit_count * 10,)``."""
    n_pos = 3 if game_type == "daily3" else 4
    vec = np.zeros(n_pos * 10, dtype=np.float32)
    digits = _extract_digits(draw, game_type)
    for pos, d in enumerate(digits):
        vec[pos * 10 + d] = 1.0
    return vec


class LSTMStrategy(BaseStrategy):
    """Two-layer LSTM predicting all digit positions simultaneously."""

    name: str = "lstm"

    def __init__(self) -> None:
        self._models: dict[str, object] = {}  # game_type -> keras Model
        self._is_trained: dict[str, bool] = {}

    # ------------------------------------------------------------------ #
    # Model construction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_model(n_pos: int) -> "object":
        """Construct and compile a fresh Keras model.

        Returns a ``tf.keras.Model`` (typed as ``object`` so the module can be
        imported even if TensorFlow is slow to load).
        """
        # Delay import so TF startup cost is paid only when actually needed
        import tensorflow as tf

        input_dim = n_pos * 10
        inputs = tf.keras.Input(shape=(_SEQ_LEN, input_dim), name="seq_input")

        x = tf.keras.layers.LSTM(64, return_sequences=True, name="lstm_1")(inputs)
        x = tf.keras.layers.LSTM(64, name="lstm_2")(x)

        # One Dense(10, softmax) head per digit position
        outputs = []
        for pos in range(n_pos):
            head = tf.keras.layers.Dense(10, activation="softmax", name=f"pos_{pos}")(x)
            outputs.append(head)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lstm_lottery")
        model.compile(
            optimizer="adam",
            loss=["sparse_categorical_crossentropy"] * n_pos,
            metrics=[["accuracy"] for _ in range(n_pos)],
        )
        return model

    # ------------------------------------------------------------------ #
    # Data preparation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_sequences(
        draws: list[Draw],
        game_type: str,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Build input sequences and label arrays.

        Parameters
        ----------
        draws:
            **Chronological** order (oldest first).

        Returns
        -------
        X : np.ndarray
            Shape ``(n_samples, SEQ_LEN, input_dim)``.
        ys : list[np.ndarray]
            One array per position, each shape ``(n_samples,)``.
        """
        n_pos = 3 if game_type == "daily3" else 4
        input_dim = n_pos * 10

        # Pre-encode all draws
        encoded = np.array([_one_hot_draw(d, game_type) for d in draws], dtype=np.float32)

        n_samples = len(draws) - _SEQ_LEN
        X = np.zeros((n_samples, _SEQ_LEN, input_dim), dtype=np.float32)
        y_lists: list[list[int]] = [[] for _ in range(n_pos)]

        for i in range(n_samples):
            X[i] = encoded[i : i + _SEQ_LEN]
            target_digits = _extract_digits(draws[i + _SEQ_LEN], game_type)
            for pos in range(n_pos):
                y_lists[pos].append(target_digits[pos])

        ys = [np.array(yl, dtype=np.int64) for yl in y_lists]
        return X, ys

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, game_type: str, history: list[Draw]) -> None:
        """Train the LSTM model.

        ``history`` arrives most-recent first (BaseStrategy contract).
        """
        chrono = list(reversed(history))
        n = len(chrono)

        if n < _MIN_DRAWS:
            logger.warning(
                "LSTM: only %d draws for %s (need %d) — skipping training",
                n, game_type, _MIN_DRAWS,
            )
            self._is_trained[game_type] = False
            return

        n_pos = self.get_digit_count(game_type)
        X, ys = self._build_sequences(chrono, game_type)
        if X.shape[0] == 0:
            logger.warning("LSTM: no training samples for %s", game_type)
            self._is_trained[game_type] = False
            return

        logger.info(
            "LSTM: training on %d sequences for %s (seq_len=%d, input_dim=%d)",
            X.shape[0], game_type, _SEQ_LEN, X.shape[2],
        )

        model = self._build_model(n_pos)
        model.fit(
            X,
            ys,
            epochs=20,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
        )

        self._models[game_type] = model
        self._is_trained[game_type] = True

        # Persist to disk
        self._save_model(game_type, model)

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

        # Try to load a saved model if we haven't trained in this session
        if not self._is_trained.get(game_type, False):
            loaded = self._load_model(game_type, n_pos)
            if loaded is not None:
                self._models[game_type] = loaded
                self._is_trained[game_type] = True
            else:
                self.train(game_type, history)

        if not self._is_trained.get(game_type, False):
            return self._uniform_prediction(n_pos, game_type)

        # Build the most recent sequence of SEQ_LEN draws
        # history is most-recent first; we need chronological order
        chrono = list(reversed(history))
        if len(chrono) < _SEQ_LEN:
            return self._uniform_prediction(n_pos, game_type)

        last_seq = chrono[-_SEQ_LEN:]  # last SEQ_LEN draws in chronological order
        input_dim = n_pos * 10
        X = np.array(
            [[_one_hot_draw(d, game_type) for d in last_seq]],
            dtype=np.float32,
        )  # shape (1, SEQ_LEN, input_dim)

        model = self._models[game_type]
        raw_outputs = model.predict(X, verbose=0)  # list of arrays, one per position

        probs: list[list[float]] = []
        for pos in range(n_pos):
            p = raw_outputs[pos][0]  # shape (10,)
            p = np.asarray(p, dtype=np.float64)
            total = p.sum()
            if total > 0:
                p /= total
            else:
                p[:] = 0.1
            probs.append(p.tolist())

        return self._build_candidates(probs, n_pos, game_type)

    # ------------------------------------------------------------------ #
    # Model persistence
    # ------------------------------------------------------------------ #

    def _save_model(self, game_type: str, model: object) -> None:
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = _MODELS_DIR / f"lstm_{game_type}.keras"
        try:
            model.save(str(path))
            logger.info("LSTM: saved model to %s", path)
        except Exception:
            logger.exception("LSTM: failed to save model to %s", path)

    def _load_model(self, game_type: str, n_pos: int) -> object | None:
        path = _MODELS_DIR / f"lstm_{game_type}.keras"
        if not path.exists():
            return None
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(str(path))
            logger.info("LSTM: loaded saved model from %s", path)
            return model
        except Exception:
            logger.exception("LSTM: failed to load model from %s", path)
            return None

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
                    "strategy": "lstm",
                    "game_type": game_type,
                    "note": "insufficient training data or model not loaded",
                },
            )
        ]
