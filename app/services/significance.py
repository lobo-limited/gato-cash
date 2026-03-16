"""Statistical significance tests for comparing prediction strategies.

Provides non-parametric tests suitable for binary hit/miss outcomes:
- McNemar's test for paired strategy comparison
- Wilson confidence interval for hit rates
- Friedman test for multi-strategy comparison
"""

from __future__ import annotations

import math


class SignificanceTests:
    """Collection of statistical significance tests for backtest comparisons."""

    @staticmethod
    def mcnemar_test(hits_a: list[bool], hits_b: list[bool]) -> dict:
        """McNemar's test for paired comparison of two strategies.

        Compares whether two strategies have statistically different hit rates
        on the **same** set of draws.

        Parameters
        ----------
        hits_a, hits_b : list[bool]
            Per-draw hit indicators for strategy A and B respectively.
            Must have equal length.

        Returns
        -------
        dict
            Keys: ``b`` (A miss / B hit), ``c`` (A hit / B miss),
            ``chi2``, ``p_value``, ``significant`` (at alpha=0.05).
        """
        if len(hits_a) != len(hits_b):
            raise ValueError("Hit lists must have equal length.")

        # Contingency counts
        # b = A wrong, B right  (discordant)
        # c = A right, B wrong  (discordant)
        b = sum(1 for a, bv in zip(hits_a, hits_b) if not a and bv)
        c = sum(1 for a, bv in zip(hits_a, hits_b) if a and not bv)

        bc_sum = b + c
        if bc_sum == 0:
            return {
                "b": b,
                "c": c,
                "chi2": 0.0,
                "p_value": 1.0,
                "significant": False,
            }

        # McNemar's chi-squared with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / bc_sum if bc_sum > 0 else 0.0

        # p-value from chi2 distribution with 1 df
        p_value = _chi2_sf(chi2, df=1)

        return {
            "b": b,
            "c": c,
            "chi2": round(chi2, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < 0.05,
        }

    @staticmethod
    def wilson_confidence_interval(
        hits: int, total: int, alpha: float = 0.05
    ) -> tuple[float, float]:
        """Wilson score confidence interval for a binomial proportion.

        More accurate than the normal approximation for small samples or
        extreme proportions (close to 0 or 1).

        Parameters
        ----------
        hits : int
            Number of successes.
        total : int
            Total trials.
        alpha : float
            Significance level (default 0.05 for 95% CI).

        Returns
        -------
        tuple[float, float]
            (lower_bound, upper_bound) of the confidence interval.
        """
        if total == 0:
            return (0.0, 0.0)

        p_hat = hits / total
        z = _normal_ppf(1 - alpha / 2)
        z2 = z * z

        denom = 1 + z2 / total
        centre = p_hat + z2 / (2 * total)
        spread = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * total)) / total)

        lower = (centre - spread) / denom
        upper = (centre + spread) / denom

        return (round(max(lower, 0.0), 6), round(min(upper, 1.0), 6))

    @staticmethod
    def friedman_test(hit_matrix: list[list[bool]]) -> dict:
        """Friedman test for comparing multiple strategies across the same draws.

        The Friedman test is a non-parametric repeated-measures ANOVA on ranks.

        Parameters
        ----------
        hit_matrix : list[list[bool]]
            A list of *k* strategies, each containing *n* boolean hit
            indicators for the same *n* draws.

        Returns
        -------
        dict
            Keys: ``statistic``, ``p_value``, ``significant``.
        """
        k = len(hit_matrix)
        if k < 2:
            return {"statistic": 0.0, "p_value": 1.0, "significant": False}

        n = len(hit_matrix[0])
        if n == 0 or any(len(row) != n for row in hit_matrix):
            raise ValueError("All strategy hit lists must have the same non-zero length.")

        # Rank strategies within each draw (column-wise)
        rank_sums = [0.0] * k
        for j in range(n):
            # Collect hits for this draw across strategies
            values = [int(hit_matrix[i][j]) for i in range(k)]
            ranks = _rank_with_ties(values)
            for i in range(k):
                rank_sums[i] += ranks[i]

        # Friedman statistic
        mean_rank = sum(rank_sums) / k
        ss = sum((rs - mean_rank) ** 2 for rs in rank_sums)
        chi2 = (12 * n / (k * (k + 1))) * ss / n if n > 0 else 0.0

        # Correct formula: chi2_F = [12 / (n*k*(k+1))] * sum(R_i^2) - 3*n*(k+1)
        sum_r2 = sum(rs ** 2 for rs in rank_sums)
        chi2 = (12 / (n * k * (k + 1))) * sum_r2 - 3 * n * (k + 1)
        chi2 = max(chi2, 0.0)

        p_value = _chi2_sf(chi2, df=k - 1)

        return {
            "statistic": round(chi2, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < 0.05,
            "rank_sums": [round(rs, 2) for rs in rank_sums],
        }


# ------------------------------------------------------------------ #
# Pure-Python math helpers (avoid scipy import for lightweight usage)
# ------------------------------------------------------------------ #

def _normal_ppf(p: float) -> float:
    """Approximate inverse CDF of the standard normal (Abramowitz & Stegun 26.2.23)."""
    if p <= 0:
        return -8.0
    if p >= 1:
        return 8.0

    if p < 0.5:
        return -_rational_approx(math.sqrt(-2.0 * math.log(p)))
    else:
        return _rational_approx(math.sqrt(-2.0 * math.log(1 - p)))


def _rational_approx(t: float) -> float:
    """Rational approximation helper for inverse normal CDF."""
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)


def _chi2_sf(x: float, df: int) -> float:
    """Survival function (1 - CDF) of the chi-squared distribution.

    Uses the regularised incomplete gamma function via a series expansion.
    Accurate enough for our purposes (df typically 1..10).
    """
    if x <= 0 or df <= 0:
        return 1.0
    return 1.0 - _lower_incomplete_gamma_reg(df / 2.0, x / 2.0)


def _lower_incomplete_gamma_reg(a: float, x: float) -> float:
    """Regularised lower incomplete gamma function P(a, x) via series expansion."""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0

    # Series expansion: P(a, x) = e^{-x} * x^a * sum_{n=0}^{inf} x^n / Gamma(a+n+1)
    # Equivalently: sum of terms t_n where t_0 = 1/a, t_{n} = t_{n-1} * x / (a+n)
    ap = a
    total = 1.0 / a
    delta = 1.0 / a
    for _ in range(300):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-14:
            break

    log_gamma_a = math.lgamma(a)
    return total * math.exp(-x + a * math.log(x) - log_gamma_a)


def _rank_with_ties(values: list[int]) -> list[float]:
    """Compute fractional ranks for a list of integer values (higher = better rank)."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n

    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
            j += 1
        # Assign average rank (1-based)
        avg_rank = (i + j) / 2.0 + 1.0
        for k_idx in range(i, j + 1):
            ranks[indexed[k_idx]] = avg_rank
        i = j + 1

    return ranks
