"""Comprehensive statistical analysis service for CA Lottery draw data.

Provides frequency analysis, hot/cold detection, gap analysis, pair/triple
analysis, positional trends, sum distribution, and pattern recognition for
both Daily 3 and Daily 4 games.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

from app.models.draw import Draw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num_positions(game_type: str) -> int:
    """Return the number of digit positions for a game type."""
    return 3 if game_type == "daily3" else 4


def _extract_digits(draw: Draw, game_type: str) -> list[int]:
    """Return the digit list for a draw according to game type."""
    digits = [draw.digit_1, draw.digit_2, draw.digit_3]
    if _num_positions(game_type) == 4:
        digits.append(draw.digit_4)  # type: ignore[arg-type]
    return digits


# ---------------------------------------------------------------------------
# a) Digit Frequency Analysis
# ---------------------------------------------------------------------------

def frequency_analysis(
    draws: list[Draw],
    game_type: str,
    position: int | None = None,
) -> dict[str, Any]:
    """Count frequency of each digit (0-9) overall and per position.

    Parameters
    ----------
    draws : list[Draw]
        Draw objects to analyse (should already be filtered/windowed).
    game_type : str
        ``"daily3"`` or ``"daily4"``.
    position : int | None
        If given (1-based), restrict analysis to that single position.
        ``None`` means return stats for *every* position plus an overall total.

    Returns
    -------
    dict with keys ``total_draws``, ``game_type``, ``positions`` (list of
    per-position dicts), and ``overall`` aggregate.
    """
    n_pos = _num_positions(game_type)
    total = len(draws)
    if total == 0:
        return {
            "game_type": game_type,
            "total_draws": 0,
            "positions": [],
            "overall": [],
        }

    expected_pct = 10.0  # each digit should appear ~10% of the time

    # positions to analyse (1-based)
    pos_range = [position] if position else list(range(1, n_pos + 1))

    positions_result: list[dict[str, Any]] = []
    overall_counter: Counter[int] = Counter()

    for pos in pos_range:
        counter: Counter[int] = Counter()
        for d in draws:
            digit = _extract_digits(d, game_type)[pos - 1]
            counter[digit] += 1
            overall_counter[digit] += 1

        digits_info = []
        for digit in range(10):
            count = counter.get(digit, 0)
            pct = (count / total) * 100 if total else 0.0
            digits_info.append({
                "digit": digit,
                "count": count,
                "percentage": round(pct, 2),
                "deviation": round(pct - expected_pct, 2),
            })

        positions_result.append({
            "position": pos,
            "digits": digits_info,
        })

    # Overall across all analysed positions
    total_slots = total * len(pos_range)
    overall_info = []
    for digit in range(10):
        count = overall_counter.get(digit, 0)
        pct = (count / total_slots) * 100 if total_slots else 0.0
        overall_info.append({
            "digit": digit,
            "count": count,
            "percentage": round(pct, 2),
            "deviation": round(pct - expected_pct, 2),
        })

    return {
        "game_type": game_type,
        "total_draws": total,
        "positions": positions_result,
        "overall": overall_info,
    }


# ---------------------------------------------------------------------------
# b) Hot / Cold Numbers
# ---------------------------------------------------------------------------

def hot_cold_analysis(
    draws: list[Draw],
    game_type: str,
    window: int = 50,
) -> dict[str, Any]:
    """Identify digits appearing above/below expected frequency.

    Hot: frequency > expected + 1 std dev.
    Cold: frequency < expected - 1 std dev.

    Parameters
    ----------
    draws : list[Draw]
        Already windowed draws (caller trims to *window* from DB).
    game_type : str
        ``"daily3"`` or ``"daily4"``.
    window : int
        Informational; the actual trimming is done by the caller.

    Returns
    -------
    dict with ``hot``, ``cold``, and ``neutral`` digit lists.
    """
    n_pos = _num_positions(game_type)
    total = len(draws)
    if total == 0:
        return {"game_type": game_type, "total_draws": 0, "hot": [], "cold": [], "neutral": []}

    counter: Counter[int] = Counter()
    for d in draws:
        for digit in _extract_digits(d, game_type):
            counter[digit] += 1

    total_slots = total * n_pos
    expected = total_slots / 10.0
    # Standard deviation for a multinomial proportion
    std_dev = math.sqrt(total_slots * 0.1 * 0.9)

    hot: list[dict[str, Any]] = []
    cold: list[dict[str, Any]] = []
    neutral: list[dict[str, Any]] = []

    for digit in range(10):
        count = counter.get(digit, 0)
        deviation = (count - expected) / std_dev if std_dev else 0.0
        entry = {
            "digit": digit,
            "count": count,
            "expected": round(expected, 2),
            "std_dev_away": round(deviation, 2),
            "percentage": round((count / total_slots) * 100, 2) if total_slots else 0.0,
        }
        if count > expected + std_dev:
            hot.append(entry)
        elif count < expected - std_dev:
            cold.append(entry)
        else:
            neutral.append(entry)

    hot.sort(key=lambda x: x["std_dev_away"], reverse=True)
    cold.sort(key=lambda x: x["std_dev_away"])

    return {
        "game_type": game_type,
        "total_draws": total,
        "expected_count": round(expected, 2),
        "std_dev": round(std_dev, 2),
        "hot": hot,
        "cold": cold,
        "neutral": neutral,
    }


# ---------------------------------------------------------------------------
# c) Gap Analysis
# ---------------------------------------------------------------------------

def gap_analysis(
    draws: list[Draw],
    game_type: str,
) -> dict[str, Any]:
    """For each digit at each position compute gap statistics.

    Draws **must** be ordered most-recent first (draw_date desc).

    Returns current gap, average gap, max gap, and whether the digit is
    overdue (current gap > average gap).
    """
    n_pos = _num_positions(game_type)
    total = len(draws)
    if total == 0:
        return {"game_type": game_type, "total_draws": 0, "positions": []}

    positions_result = []

    for pos_idx in range(n_pos):
        digits_result = []
        for target_digit in range(10):
            gaps: list[int] = []
            current_gap: int | None = None
            last_seen_idx: int | None = None

            for idx, draw in enumerate(draws):
                digit = _extract_digits(draw, game_type)[pos_idx]
                if digit == target_digit:
                    if last_seen_idx is None:
                        current_gap = idx  # gap from "now" to first appearance
                    else:
                        gaps.append(idx - last_seen_idx)
                    last_seen_idx = idx

            # If the digit was never seen, current gap is the full window
            if last_seen_idx is None:
                current_gap = total
            # The gap *after* the last appearance extends to end of data
            if last_seen_idx is not None and last_seen_idx < total - 1:
                gaps.append(total - last_seen_idx)

            avg_gap = round(float(np.mean(gaps)), 2) if gaps else 0.0
            max_gap = max(gaps) if gaps else 0
            c_gap = current_gap if current_gap is not None else 0

            digits_result.append({
                "digit": target_digit,
                "current_gap": c_gap,
                "average_gap": avg_gap,
                "max_gap": max_gap,
                "overdue": c_gap > avg_gap if avg_gap > 0 else False,
            })

        positions_result.append({
            "position": pos_idx + 1,
            "digits": digits_result,
        })

    return {
        "game_type": game_type,
        "total_draws": total,
        "positions": positions_result,
    }


# ---------------------------------------------------------------------------
# d) Pair / Triple Analysis
# ---------------------------------------------------------------------------

def pair_analysis(
    draws: list[Draw],
    game_type: str,
) -> dict[str, Any]:
    """Analyse positionally adjacent pairs (and triples for Daily 3).

    For each adjacency (pos 1-2, 2-3, 3-4) counts frequency of all 100
    possible two-digit combinations.  Returns the top 10 most frequent
    pairs per adjacency plus full counts.

    For Daily 3, also computes triple frequency (all 1000 possible combos).
    """
    n_pos = _num_positions(game_type)
    total = len(draws)
    if total == 0:
        return {"game_type": game_type, "total_draws": 0, "adjacencies": [], "triples": None}

    adjacencies_result: list[dict[str, Any]] = []

    for start in range(n_pos - 1):
        counter: Counter[str] = Counter()
        for d in draws:
            digits = _extract_digits(d, game_type)
            pair = f"{digits[start]}{digits[start + 1]}"
            counter[pair] += 1

        top10 = counter.most_common(10)
        adjacencies_result.append({
            "positions": f"{start + 1}-{start + 2}",
            "top_pairs": [
                {"pair": p, "count": c, "percentage": round((c / total) * 100, 2)}
                for p, c in top10
            ],
            "total_unique": len(counter),
            "full_counts": dict(counter.most_common()),
        })

    # Triples for Daily 3
    triples_result = None
    if game_type == "daily3":
        triple_counter: Counter[str] = Counter()
        for d in draws:
            digits = _extract_digits(d, game_type)
            triple = f"{digits[0]}{digits[1]}{digits[2]}"
            triple_counter[triple] += 1

        top10_triples = triple_counter.most_common(10)
        triples_result = {
            "top_triples": [
                {"triple": t, "count": c, "percentage": round((c / total) * 100, 2)}
                for t, c in top10_triples
            ],
            "total_unique": len(triple_counter),
        }

    return {
        "game_type": game_type,
        "total_draws": total,
        "adjacencies": adjacencies_result,
        "triples": triples_result,
    }


# ---------------------------------------------------------------------------
# e) Positional Trends
# ---------------------------------------------------------------------------

def positional_trends(
    draws: list[Draw],
    game_type: str,
    window: int = 30,
) -> dict[str, Any]:
    """Compute exponentially weighted moving average of digit frequency per position.

    Draws must be ordered most-recent first.  We reverse internally so the
    EMA is computed chronologically (oldest -> newest) and the last value
    reflects the most recent trend.

    Returns per-position, per-digit EMA, trend direction, and the
    distribution of digits in the last *window* draws.
    """
    n_pos = _num_positions(game_type)
    total = len(draws)
    if total == 0:
        return {"game_type": game_type, "total_draws": 0, "positions": []}

    # Reverse so index 0 = oldest
    ordered_draws = list(reversed(draws))

    span = min(window, total)
    alpha = 2.0 / (span + 1)

    positions_result: list[dict[str, Any]] = []

    for pos_idx in range(n_pos):
        # Build a Series of digit values chronologically
        digit_series = [_extract_digits(d, game_type)[pos_idx] for d in ordered_draws]

        # One-hot encode each digit and compute EMA
        digit_ema: dict[int, float] = {}
        digit_trend: dict[int, str] = {}

        for target in range(10):
            binary = pd.Series([1.0 if v == target else 0.0 for v in digit_series])
            ema = binary.ewm(span=span, adjust=False).mean()
            current_ema = float(ema.iloc[-1])
            mid_ema = float(ema.iloc[len(ema) // 2]) if len(ema) > 1 else current_ema

            digit_ema[target] = round(current_ema, 4)
            diff = current_ema - mid_ema
            if abs(diff) < 0.01:
                digit_trend[target] = "stable"
            elif diff > 0:
                digit_trend[target] = "increasing"
            else:
                digit_trend[target] = "decreasing"

        # Last-window distribution
        recent = digit_series[-span:]
        dist_counter = Counter(recent)
        distribution = {d: dist_counter.get(d, 0) for d in range(10)}

        positions_result.append({
            "position": pos_idx + 1,
            "ema": digit_ema,
            "trend": digit_trend,
            "last_window_distribution": distribution,
        })

    return {
        "game_type": game_type,
        "total_draws": total,
        "window": span,
        "positions": positions_result,
    }


# ---------------------------------------------------------------------------
# f) Sum Analysis
# ---------------------------------------------------------------------------

def sum_analysis(
    draws: list[Draw],
    game_type: str,
) -> dict[str, Any]:
    """Distribution of digit sums across draws.

    Returns histogram, mean, median, standard deviation, and the current
    streak of consecutive draws where the sum has been above or below the
    mean.
    """
    total = len(draws)
    if total == 0:
        return {
            "game_type": game_type,
            "total_draws": 0,
            "histogram": {},
            "mean": 0,
            "median": 0,
            "std_dev": 0,
            "current_streak": {"direction": "none", "length": 0},
        }

    sums = [sum(_extract_digits(d, game_type)) for d in draws]
    arr = np.array(sums, dtype=float)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr, ddof=1)) if total > 1 else 0.0

    # Histogram
    hist_counter = Counter(sums)
    n_pos = _num_positions(game_type)
    max_sum = 9 * n_pos
    histogram = {s: hist_counter.get(s, 0) for s in range(max_sum + 1)}

    # Current streak (draws are most-recent first)
    streak_dir: str = "none"
    streak_len: int = 0
    if sums:
        first_above = sums[0] >= mean_val
        streak_dir = "above_or_equal" if first_above else "below"
        streak_len = 1
        for s in sums[1:]:
            if (s >= mean_val) == first_above:
                streak_len += 1
            else:
                break

    return {
        "game_type": game_type,
        "total_draws": total,
        "histogram": histogram,
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "std_dev": round(std_val, 2),
        "current_streak": {"direction": streak_dir, "length": streak_len},
    }


# ---------------------------------------------------------------------------
# g) Repeat / Pattern Analysis
# ---------------------------------------------------------------------------

def pattern_analysis(
    draws: list[Draw],
    game_type: str,
) -> dict[str, Any]:
    """Detect repeat and pattern statistics.

    - How often consecutive draws share 0, 1, 2, 3+ digits in the same
      position.
    - Frequency of repeated digits within a single draw (pairs, triples,
      quads).
    - Most common full draw patterns.
    """
    n_pos = _num_positions(game_type)
    total = len(draws)
    if total == 0:
        return {
            "game_type": game_type,
            "total_draws": 0,
            "positional_matches": {},
            "within_draw_repeats": {},
            "most_common_draws": [],
        }

    # --- Positional matches between consecutive draws ---
    # Draws are most-recent first, so consecutive = index i and i+1
    match_counts: Counter[int] = Counter()  # number of matching positions
    for i in range(total - 1):
        d1 = _extract_digits(draws[i], game_type)
        d2 = _extract_digits(draws[i + 1], game_type)
        matches = sum(1 for a, b in zip(d1, d2) if a == b)
        match_counts[matches] += 1

    consecutive_pairs = total - 1 if total > 1 else 1
    positional_matches = {}
    for m in range(n_pos + 1):
        count = match_counts.get(m, 0)
        positional_matches[str(m)] = {
            "count": count,
            "percentage": round((count / consecutive_pairs) * 100, 2),
        }

    # --- Within-draw repeated digits ---
    repeat_types: Counter[str] = Counter()
    for d in draws:
        digits = _extract_digits(d, game_type)
        digit_counts = Counter(digits)
        max_repeat = max(digit_counts.values())
        if max_repeat == 1:
            repeat_types["all_unique"] += 1
        elif max_repeat == 2:
            # Could be one pair or two pairs (daily4)
            num_pairs = sum(1 for c in digit_counts.values() if c == 2)
            if num_pairs >= 2:
                repeat_types["double_pair"] += 1
            else:
                repeat_types["single_pair"] += 1
        elif max_repeat == 3:
            repeat_types["triple"] += 1
        elif max_repeat >= 4:
            repeat_types["quad_or_more"] += 1

    within_draw_repeats = {
        k: {"count": v, "percentage": round((v / total) * 100, 2)}
        for k, v in repeat_types.items()
    }

    # --- Most common full draws ---
    draw_counter: Counter[str] = Counter()
    for d in draws:
        digits = _extract_digits(d, game_type)
        key = "-".join(str(x) for x in digits)
        draw_counter[key] += 1

    most_common_draws = [
        {"draw": k, "count": v, "percentage": round((v / total) * 100, 2)}
        for k, v in draw_counter.most_common(15)
    ]

    return {
        "game_type": game_type,
        "total_draws": total,
        "positional_matches": positional_matches,
        "within_draw_repeats": within_draw_repeats,
        "most_common_draws": most_common_draws,
    }


# ---------------------------------------------------------------------------
# h) Combined Summary
# ---------------------------------------------------------------------------

def get_full_summary(
    draws: list[Draw],
    game_type: str,
    window: int = 100,
) -> dict[str, Any]:
    """Call every analysis function and return a combined dict for the dashboard.

    Parameters
    ----------
    draws : list[Draw]
        Already windowed draws from the DB.
    game_type : str
        ``"daily3"`` or ``"daily4"``.
    window : int
        Passed through to sub-functions that need a window value.
    """
    return {
        "game_type": game_type,
        "total_draws": len(draws),
        "frequency": frequency_analysis(draws, game_type),
        "hot_cold": hot_cold_analysis(draws, game_type, window=window),
        "gaps": gap_analysis(draws, game_type),
        "pairs": pair_analysis(draws, game_type),
        "trends": positional_trends(draws, game_type, window=window),
        "sums": sum_analysis(draws, game_type),
        "patterns": pattern_analysis(draws, game_type),
    }
