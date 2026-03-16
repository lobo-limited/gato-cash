"""Self-calibration engine for the CA Lottery prediction system.

Computes rolling performance metrics, recalibrates ensemble weights using an
exponential-weights algorithm, and detects strategy degradation via z-score
analysis.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.performance import EnsembleWeight, ModelPerformance
from app.models.prediction import Prediction
from app.strategies import get_all_strategies

logger = logging.getLogger(__name__)


class CalibrationService:
    """Tracks performance, recalibrates weights, and detects degradation."""

    # Maps human-friendly window labels to approximate draw counts.
    # Daily 3 has ~2 draws/day, so 7 days ~ 14 draws, etc.
    WINDOWS: dict[str, int] = {
        "7d": 14,
        "30d": 60,
        "90d": 180,
        "180d": 360,
        "365d": 730,
    }

    # Softmax temperature for weight calculation
    TEMPERATURE = 0.5

    # Weight guardrails
    WEIGHT_FLOOR = 0.00125   # 0.125% per strategy (1% / 8)
    WEIGHT_CAP = 0.50        # 50%
    MAX_WEIGHT_SHIFT = 0.20  # max 20% change from previous weights

    # ------------------------------------------------------------------ #
    # Rolling performance                                                  #
    # ------------------------------------------------------------------ #

    def compute_rolling_performance(
        self, db: Session, game_type: str
    ) -> list[ModelPerformance]:
        """Compute performance metrics for each strategy across all windows.

        Only considers *scored* predictions (scored_at IS NOT NULL) that are
        not back-test predictions.

        Returns the list of ModelPerformance rows created/updated.
        """
        strategy_names = self._get_strategy_names()
        results: list[ModelPerformance] = []

        for strategy_name in strategy_names:
            for window_label, draw_count in self.WINDOWS.items():
                perf = self._compute_window(
                    db, game_type, strategy_name, window_label, draw_count,
                )
                if perf is not None:
                    results.append(perf)

            # Also compute an "all" window covering every scored prediction.
            perf_all = self._compute_window(
                db, game_type, strategy_name, "all", draw_count=None,
            )
            if perf_all is not None:
                results.append(perf_all)

        db.commit()
        logger.info(
            "Computed rolling performance: %d records for game_type=%s",
            len(results), game_type,
        )
        return results

    def _compute_window(
        self,
        db: Session,
        game_type: str,
        strategy_name: str,
        window_label: str,
        draw_count: int | None,
    ) -> ModelPerformance | None:
        """Compute metrics for a single strategy + window combination."""
        # Base query: scored, non-backtest predictions for this strategy/game.
        stmt = (
            select(Prediction)
            .where(
                Prediction.game_type == game_type,
                Prediction.strategy_name == strategy_name,
                Prediction.scored_at.isnot(None),
                Prediction.is_backtest == False,  # noqa: E712
            )
            .order_by(Prediction.scored_at.desc())
        )

        if draw_count is not None:
            stmt = stmt.limit(draw_count)

        predictions = db.execute(stmt).scalars().all()

        total = len(predictions)
        if total == 0:
            return None

        straight_hits = sum(1 for p in predictions if p.straight_hit)
        box_hits = sum(1 for p in predictions if p.box_hit)
        straight_hit_rate = straight_hits / total
        box_hit_rate = box_hits / total
        avg_confidence = sum(p.confidence for p in predictions) / total

        # Calibration score: 1 - |avg_confidence - actual_hit_rate|
        # Use box_hit_rate as the "actual hit rate" (more common than straight).
        calibration_score = 1.0 - abs(avg_confidence - box_hit_rate)

        now = datetime.utcnow()

        # Upsert: find existing record or create new one.
        existing = db.execute(
            select(ModelPerformance).where(
                ModelPerformance.strategy_name == strategy_name,
                ModelPerformance.game_type == game_type,
                ModelPerformance.window_type == window_label,
            )
        ).scalar_one_or_none()

        if existing:
            existing.total_predictions = total
            existing.straight_hits = straight_hits
            existing.box_hits = box_hits
            existing.straight_hit_rate = straight_hit_rate
            existing.box_hit_rate = box_hit_rate
            existing.avg_confidence = avg_confidence
            existing.calibration_score = calibration_score
            existing.computed_at = now
            return existing
        else:
            perf = ModelPerformance(
                strategy_name=strategy_name,
                game_type=game_type,
                window_type=window_label,
                total_predictions=total,
                straight_hits=straight_hits,
                box_hits=box_hits,
                straight_hit_rate=straight_hit_rate,
                box_hit_rate=box_hit_rate,
                avg_confidence=avg_confidence,
                calibration_score=calibration_score,
                computed_at=now,
            )
            db.add(perf)
            return perf

    # ------------------------------------------------------------------ #
    # Recalibrate weights                                                  #
    # ------------------------------------------------------------------ #

    def recalibrate_weights(
        self, db: Session, game_type: str
    ) -> dict[str, float]:
        """Recalculate ensemble weights using exponential-weights algorithm.

        Score formula per strategy:
            score = 0.4 * norm_box_hit_rate_30d
                  + 0.3 * norm_straight_hit_rate_30d
                  + 0.2 * norm_calibration_score_30d
                  + 0.1 * norm_avg_confidence_30d

        Weights = softmax(scores / temperature) with guardrails applied.

        Returns the new weight mapping {strategy_name: weight}.
        """
        # First, recompute rolling performance to have fresh data.
        self.compute_rolling_performance(db, game_type)

        strategy_names = self._get_strategy_names()
        if not strategy_names:
            logger.warning("No strategies found for recalibration.")
            return {}

        # Fetch 30d performance metrics.
        metrics_30d: dict[str, ModelPerformance] = {}
        for name in strategy_names:
            perf = db.execute(
                select(ModelPerformance).where(
                    ModelPerformance.strategy_name == name,
                    ModelPerformance.game_type == game_type,
                    ModelPerformance.window_type == "30d",
                )
            ).scalar_one_or_none()
            if perf is not None:
                metrics_30d[name] = perf

        if not metrics_30d:
            logger.warning("No 30d metrics available; returning equal weights.")
            return self._equal_weights(strategy_names)

        # Extract raw values for normalization.
        box_rates = {n: m.box_hit_rate for n, m in metrics_30d.items()}
        straight_rates = {n: m.straight_hit_rate for n, m in metrics_30d.items()}
        cal_scores = {
            n: m.calibration_score if m.calibration_score is not None else 0.0
            for n, m in metrics_30d.items()
        }
        avg_confs = {n: m.avg_confidence for n, m in metrics_30d.items()}

        # Normalize each metric to [0, 1] using min-max scaling.
        norm_box = self._min_max_normalize(box_rates)
        norm_straight = self._min_max_normalize(straight_rates)
        norm_cal = self._min_max_normalize(cal_scores)
        norm_conf = self._min_max_normalize(avg_confs)

        # Compute composite score for each strategy.
        scores: dict[str, float] = {}
        for name in strategy_names:
            if name in metrics_30d:
                scores[name] = (
                    0.4 * norm_box.get(name, 0.0)
                    + 0.3 * norm_straight.get(name, 0.0)
                    + 0.2 * norm_cal.get(name, 0.0)
                    + 0.1 * norm_conf.get(name, 0.0)
                )
            else:
                # Strategy has no 30d data: assign a minimal score.
                scores[name] = 0.0

        # Softmax to get raw weights.
        raw_weights = self._softmax(scores, self.TEMPERATURE)

        # Get previous weights for damping.
        prev_weights = self.get_current_weights(db, game_type)

        # Apply guardrails.
        final_weights = self._apply_guardrails(raw_weights, prev_weights)

        # Store new weights in DB.
        self._store_weights(db, game_type, final_weights, reason="auto_recalibration")

        db.commit()
        logger.info("Recalibrated weights for %s: %s", game_type, final_weights)
        return final_weights

    # ------------------------------------------------------------------ #
    # Degradation detection                                                #
    # ------------------------------------------------------------------ #

    def detect_degradation(
        self, db: Session, game_type: str
    ) -> list[dict]:
        """Check for strategy degradation using z-score analysis.

        Compares 7-day performance to 90-day baseline for each strategy.
        Returns a list of alert dicts.
        """
        strategy_names = self._get_strategy_names()
        alerts: list[dict] = []

        for name in strategy_names:
            # Get 7d and 90d performance.
            perf_7d = db.execute(
                select(ModelPerformance).where(
                    ModelPerformance.strategy_name == name,
                    ModelPerformance.game_type == game_type,
                    ModelPerformance.window_type == "7d",
                )
            ).scalar_one_or_none()

            perf_90d = db.execute(
                select(ModelPerformance).where(
                    ModelPerformance.strategy_name == name,
                    ModelPerformance.game_type == game_type,
                    ModelPerformance.window_type == "90d",
                )
            ).scalar_one_or_none()

            if perf_7d is None or perf_90d is None:
                continue

            if perf_7d.total_predictions == 0 or perf_90d.total_predictions == 0:
                continue

            # Check box_hit_rate degradation.
            alert = self._check_metric_degradation(
                strategy_name=name,
                metric_name="box_hit_rate",
                current_value=perf_7d.box_hit_rate,
                baseline_value=perf_90d.box_hit_rate,
                n_current=perf_7d.total_predictions,
                n_baseline=perf_90d.total_predictions,
            )
            if alert:
                alerts.append(alert)

            # Check straight_hit_rate degradation.
            alert = self._check_metric_degradation(
                strategy_name=name,
                metric_name="straight_hit_rate",
                current_value=perf_7d.straight_hit_rate,
                baseline_value=perf_90d.straight_hit_rate,
                n_current=perf_7d.total_predictions,
                n_baseline=perf_90d.total_predictions,
            )
            if alert:
                alerts.append(alert)

            # Check calibration_score degradation.
            cal_7d = perf_7d.calibration_score if perf_7d.calibration_score is not None else 0.0
            cal_90d = perf_90d.calibration_score if perf_90d.calibration_score is not None else 0.0
            alert = self._check_metric_degradation(
                strategy_name=name,
                metric_name="calibration_score",
                current_value=cal_7d,
                baseline_value=cal_90d,
                n_current=perf_7d.total_predictions,
                n_baseline=perf_90d.total_predictions,
            )
            if alert:
                alerts.append(alert)

        if alerts:
            logger.warning(
                "Degradation detected for %s: %d alerts", game_type, len(alerts),
            )
        else:
            logger.info("No degradation detected for %s.", game_type)

        return alerts

    def _check_metric_degradation(
        self,
        strategy_name: str,
        metric_name: str,
        current_value: float,
        baseline_value: float,
        n_current: int,
        n_baseline: int,
    ) -> dict | None:
        """Compute z-score for a metric and return an alert if degraded.

        z = (mean_7d - mean_90d) / (std_90d / sqrt(n_7d))

        For hit rates, std is estimated from binomial distribution:
            std = sqrt(p * (1 - p))
        """
        if n_current == 0 or n_baseline == 0:
            return None

        # Estimate std from the baseline using binomial std deviation.
        p = baseline_value
        std_baseline = math.sqrt(p * (1 - p)) if 0 < p < 1 else 0.0

        if std_baseline == 0:
            # Cannot compute z-score when baseline has zero variance.
            return None

        se = std_baseline / math.sqrt(n_current)
        z_score = (current_value - baseline_value) / se

        severity = None
        if z_score < -3.0:
            severity = "CRITICAL"
        elif z_score < -2.0:
            severity = "WARNING"

        if severity is None:
            return None

        return {
            "strategy": strategy_name,
            "severity": severity,
            "z_score": round(z_score, 3),
            "metric": metric_name,
            "baseline": round(baseline_value, 6),
            "current": round(current_value, 6),
        }

    # ------------------------------------------------------------------ #
    # Current weights & history                                            #
    # ------------------------------------------------------------------ #

    def get_current_weights(
        self, db: Session, game_type: str
    ) -> dict[str, float]:
        """Get current ensemble weights. Falls back to equal weights."""
        stmt = (
            select(EnsembleWeight)
            .where(
                EnsembleWeight.game_type == game_type,
                EnsembleWeight.effective_to.is_(None),
            )
        )
        rows = db.execute(stmt).scalars().all()

        if not rows:
            return self._equal_weights(self._get_strategy_names())

        return {row.strategy_name: row.weight for row in rows}

    def get_weight_history(
        self, db: Session, game_type: str, days: int = 90
    ) -> list[dict]:
        """Get weight changes over time for charting.

        Returns a list of dicts sorted by effective_from ascending:
            [{date, strategy_name, weight, reason}, ...]
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        stmt = (
            select(EnsembleWeight)
            .where(
                EnsembleWeight.game_type == game_type,
                EnsembleWeight.effective_from >= cutoff,
            )
            .order_by(EnsembleWeight.effective_from.asc())
        )
        rows = db.execute(stmt).scalars().all()

        return [
            {
                "date": row.effective_from.isoformat(),
                "strategy_name": row.strategy_name,
                "weight": round(row.weight, 6),
                "reason": row.reason,
            }
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_strategy_names() -> list[str]:
        """Return names of all registered strategies."""
        try:
            strategies = get_all_strategies()
            return [s.name for s in strategies]
        except Exception:
            logger.warning("Could not load strategies; returning empty list.")
            return []

    @staticmethod
    def _equal_weights(strategy_names: list[str]) -> dict[str, float]:
        """Return equal weights for all strategies."""
        n = len(strategy_names)
        if n == 0:
            return {}
        w = 1.0 / n
        return {name: w for name in strategy_names}

    @staticmethod
    def _min_max_normalize(values: dict[str, float]) -> dict[str, float]:
        """Min-max normalize a dict of values to [0, 1].

        If all values are the same, returns 0.5 for each.
        """
        if not values:
            return {}
        vals = list(values.values())
        v_min = min(vals)
        v_max = max(vals)
        span = v_max - v_min
        if span == 0:
            return {k: 0.5 for k in values}
        return {k: (v - v_min) / span for k, v in values.items()}

    @staticmethod
    def _softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
        """Compute softmax of scores with given temperature.

        Returns a dict of weights that sum to 1.0.
        """
        if not scores:
            return {}

        # Scale by temperature.
        scaled = {k: v / temperature for k, v in scores.items()}

        # Subtract max for numerical stability.
        max_val = max(scaled.values())
        exp_vals = {k: math.exp(v - max_val) for k, v in scaled.items()}
        total = sum(exp_vals.values())

        if total == 0:
            n = len(scores)
            return {k: 1.0 / n for k in scores}

        return {k: v / total for k, v in exp_vals.items()}

    def _apply_guardrails(
        self,
        raw_weights: dict[str, float],
        prev_weights: dict[str, float],
    ) -> dict[str, float]:
        """Apply weight floor, cap, and change-damping guardrails.

        1. Weight floor: min WEIGHT_FLOOR per strategy.
        2. Weight cap: max WEIGHT_CAP per strategy.
        3. Change damping: max MAX_WEIGHT_SHIFT from previous weights.
        4. Re-normalize to sum to 1.0.
        """
        if not raw_weights:
            return {}

        weights = dict(raw_weights)

        # Step 1 & 2: Apply floor and cap.
        for name in weights:
            weights[name] = max(weights[name], self.WEIGHT_FLOOR)
            weights[name] = min(weights[name], self.WEIGHT_CAP)

        # Step 3: Damping — limit change from previous weights.
        if prev_weights:
            for name in weights:
                if name in prev_weights:
                    prev = prev_weights[name]
                    diff = weights[name] - prev
                    if abs(diff) > self.MAX_WEIGHT_SHIFT:
                        weights[name] = prev + self.MAX_WEIGHT_SHIFT * (1 if diff > 0 else -1)

        # Step 4: Re-normalize to sum to 1.0.
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _store_weights(
        self,
        db: Session,
        game_type: str,
        weights: dict[str, float],
        reason: str = "recalibration",
    ) -> None:
        """Store new weights in the ensemble_weights table.

        Marks previous active weights as expired (sets effective_to).
        """
        now = datetime.utcnow()

        # Expire existing active weights for this game type.
        existing = db.execute(
            select(EnsembleWeight).where(
                EnsembleWeight.game_type == game_type,
                EnsembleWeight.effective_to.is_(None),
            )
        ).scalars().all()

        for row in existing:
            row.effective_to = now

        # Insert new weights.
        for strategy_name, weight in weights.items():
            ew = EnsembleWeight(
                game_type=game_type,
                strategy_name=strategy_name,
                weight=round(weight, 6),
                effective_from=now,
                effective_to=None,
                reason=reason,
            )
            db.add(ew)
