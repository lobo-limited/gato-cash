"""Reusable feature engineering for ML prediction strategies.

Computes a consistent feature vector from draw history for any position-based
digit prediction model (RandomForest, XGBoost, etc.).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.models.draw import Draw

logger = logging.getLogger(__name__)


def _extract_digits(draw: "Draw", game_type: str) -> list[int]:
    digits = [draw.digit_1, draw.digit_2, draw.digit_3]
    if game_type != "daily3":
        digits.append(draw.digit_4)  # type: ignore[arg-type]
    return digits


def build_features(
    draws: list["Draw"],
    index: int,
    game_type: str,
) -> np.ndarray:
    """Build a feature vector for predicting ``draws[index]`` using history.

    ``draws`` must be in **chronological order** (oldest first).  The feature
    vector is constructed from ``draws[0:index]`` — i.e. only past data
    relative to the target draw.

    Parameters
    ----------
    draws:
        Full chronological list of draws (oldest at position 0).
    index:
        Position of the target draw whose digits we want to predict.
    game_type:
        ``"daily3"`` or ``"daily4"``.

    Returns
    -------
    np.ndarray
        1-D float64 feature vector.
    """
    n_pos = 3 if game_type == "daily3" else 4
    history = draws[:index]  # only past draws
    n_hist = len(history)

    features: list[float] = []

    # --- 1. Per-position digit frequencies for windows of 10, 30, 100 -------
    for window in (10, 30, 100):
        recent = history[-window:] if n_hist >= window else history
        n = len(recent)
        freq = np.zeros((n_pos, 10), dtype=np.float64)
        for d in recent:
            digs = _extract_digits(d, game_type)
            for pos, digit in enumerate(digs):
                freq[pos, digit] += 1
        if n > 0:
            freq /= n
        features.extend(freq.ravel().tolist())  # n_pos * 10 features per window

    # --- 2. Gap since last appearance for each digit at each position --------
    gap = np.full((n_pos, 10), n_hist, dtype=np.float64)  # default: never seen
    for lookback, d in enumerate(reversed(history)):
        digs = _extract_digits(d, game_type)
        for pos, digit in enumerate(digs):
            if gap[pos, digit] == n_hist:  # first (most recent) occurrence
                gap[pos, digit] = lookback
    # Normalise gap by history length to keep scale bounded
    if n_hist > 0:
        gap /= n_hist
    features.extend(gap.ravel().tolist())  # n_pos * 10 features

    # --- 3. Day of week — one-hot encoded (7 features) ----------------------
    target_draw = draws[index]
    dow = target_draw.draw_date.weekday()  # Monday=0 .. Sunday=6
    dow_onehot = [0.0] * 7
    dow_onehot[dow] = 1.0
    features.extend(dow_onehot)

    # --- 4. Draw time (midday=0, evening=1) ----------------------------------
    draw_time_val = 0.0 if target_draw.draw_time == "midday" else 1.0
    features.append(draw_time_val)

    # --- 5. Sum of digits in the last draw -----------------------------------
    if n_hist > 0:
        last_digits = _extract_digits(history[-1], game_type)
        digit_sum = sum(last_digits) / (9.0 * n_pos)  # normalised to [0, 1]
    else:
        digit_sum = 0.5  # neutral
    features.append(digit_sum)

    # --- 6. Pair frequencies for adjacent positions in last 30 draws ---------
    recent_30 = history[-30:] if n_hist >= 30 else history
    n_30 = len(recent_30)
    pair_feats: list[float] = []
    for pos in range(n_pos - 1):
        pair_count = np.zeros((10, 10), dtype=np.float64)
        for d in recent_30:
            digs = _extract_digits(d, game_type)
            pair_count[digs[pos], digs[pos + 1]] += 1
        # Use entropy-like measure: max pair freq / total  (how concentrated?)
        if n_30 > 0:
            pair_probs = pair_count / n_30
            max_pair_freq = float(pair_probs.max())
        else:
            max_pair_freq = 0.01
        pair_feats.append(max_pair_freq)
    features.extend(pair_feats)  # (n_pos - 1) features

    return np.array(features, dtype=np.float64)


def build_feature_matrix(
    draws: list["Draw"],
    game_type: str,
    min_history: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Build X (features) and y (labels) matrices for all trainable draws.

    Parameters
    ----------
    draws:
        **Chronological** list (oldest first).
    game_type:
        ``"daily3"`` or ``"daily4"``.
    min_history:
        Minimum number of preceding draws before we create a training sample.

    Returns
    -------
    X : np.ndarray
        Shape ``(n_samples, n_features)``.
    y : np.ndarray
        Shape ``(n_samples, n_positions)`` — digit labels per position.
    """
    n_pos = 3 if game_type == "daily3" else 4
    X_rows: list[np.ndarray] = []
    y_rows: list[list[int]] = []

    for idx in range(min_history, len(draws)):
        feat = build_features(draws, idx, game_type)
        digits = _extract_digits(draws[idx], game_type)
        X_rows.append(feat)
        y_rows.append(digits)

    if not X_rows:
        return np.empty((0, 0)), np.empty((0, 0))

    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=np.int64)
    logger.info(
        "Built feature matrix: X %s, y %s for %s", X.shape, y.shape, game_type,
    )
    return X, y


def feature_count(game_type: str) -> int:
    """Return the total number of features for a given game type.

    Useful for pre-allocating or validating array shapes.
    """
    n_pos = 3 if game_type == "daily3" else 4
    count = 0
    count += n_pos * 10 * 3   # freq windows (10, 30, 100)
    count += n_pos * 10        # gap features
    count += 7                 # day of week one-hot
    count += 1                 # draw time
    count += 1                 # last draw digit sum
    count += n_pos - 1         # pair features
    return count
