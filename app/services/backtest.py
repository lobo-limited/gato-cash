"""Walk-forward backtesting engine for lottery prediction strategies.

Runs walk-forward simulations where, for each target draw D[t]:
  1. The strategy is trained on D[max(0, t-window) : t]  (no future data)
  2. A prediction is generated for D[t]
  3. The prediction is scored against the actual D[t]
  4. Results are persisted in BacktestDetail rows

After all draws are processed, summary metrics are computed and stored in a
BacktestRun row.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.backtest import BacktestDetail, BacktestRun
from app.models.draw import Draw
from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# ----- Payout tables (CA Lottery approximate fixed payouts per $1 wager) ----
# Daily 3: straight $500, box 3-way $160, box 6-way $80
# Daily 4: straight $5000, box varies
PAYOUTS = {
    "daily3": {"straight": 500.0, "box_3way": 160.0, "box_6way": 80.0},
    "daily4": {"straight": 5000.0, "box_4way": 2500.0, "box_6way": 800.0,
               "box_12way": 400.0, "box_24way": 200.0},
}


class BacktestEngine:
    """Walk-forward backtesting engine."""

    # ------------------------------------------------------------------ #
    # Main entry points
    # ------------------------------------------------------------------ #

    def run(
        self,
        db: Session,
        strategy: BaseStrategy,
        game_type: str,
        training_window: int = 200,
        start_index: int | None = None,
        end_index: int | None = None,
        step: int = 1,
        name: str | None = None,
    ) -> BacktestRun:
        """Execute a walk-forward backtest and persist results.

        Parameters
        ----------
        db : Session
            Active SQLAlchemy session.
        strategy : BaseStrategy
            The strategy instance to evaluate.
        game_type : str
            ``"daily3"`` or ``"daily4"``.
        training_window : int
            Number of most-recent draws to feed to the strategy.
        start_index : int | None
            0-based index into the chronologically-sorted draw list where
            backtesting begins.  Defaults to ``training_window`` so that
            the strategy always has at least one full window of history.
        end_index : int | None
            0-based index where backtesting ends (exclusive).
            Defaults to the total number of draws.
        step : int
            Evaluate every *step*-th draw (1 = every draw).
        name : str | None
            Human-readable label for this run.

        Returns
        -------
        BacktestRun
            Fully populated ORM object (already committed).
        """
        t0 = time.perf_counter()

        # Fetch draws ordered oldest-first (chronological).
        all_draws: list[Draw] = list(
            db.execute(
                select(Draw)
                .where(Draw.game_type == game_type)
                .order_by(Draw.draw_date.asc(), Draw.draw_number.asc())
            ).scalars().all()
        )

        total = len(all_draws)
        if total == 0:
            raise ValueError(f"No draws found for game_type={game_type!r}")

        if start_index is None:
            start_index = min(training_window, total)
        if end_index is None:
            end_index = total

        start_index = max(start_index, 1)  # need at least 1 draw of history
        end_index = min(end_index, total)

        if name is None:
            name = f"{strategy.name} | {game_type} | win={training_window}"

        logger.info(
            "Backtest %s: draws %d..%d (step %d) out of %d total",
            name, start_index, end_index, step, total,
        )

        # Determine which draw indices to evaluate
        eval_indices = list(range(start_index, end_index, step))
        n_eval = len(eval_indices)
        if n_eval == 0:
            raise ValueError("No draws in the evaluation range after applying step.")

        n_pos = BaseStrategy.get_digit_count(game_type)

        # Track when we last trained (for ML efficiency: retrain every 100)
        _last_train_idx: int | None = None
        _ML_RETRAIN_INTERVAL = 100

        details: list[BacktestDetail] = []

        for progress_i, t in enumerate(eval_indices):
            if progress_i % 50 == 0:
                logger.info("  progress: %d / %d  (draw index %d)", progress_i, n_eval, t)

            target_draw = all_draws[t]
            actual_digits = BaseStrategy.extract_digits(target_draw, game_type)

            # Build history window: draws *before* t, most-recent first
            window_start = max(0, t - training_window)
            history = list(reversed(all_draws[window_start:t]))

            # Train (or retrain) strategy
            need_train = (
                _last_train_idx is None
                or (t - _last_train_idx) >= _ML_RETRAIN_INTERVAL
            )
            if need_train:
                try:
                    strategy.train(game_type, history)
                except Exception:
                    logger.exception("Strategy.train() failed at index %d", t)
                _last_train_idx = t

            # Generate prediction
            try:
                preds = strategy.predict(game_type, target_draw.draw_time, history)
            except Exception:
                logger.exception("Strategy.predict() failed at index %d", t)
                continue

            if not preds:
                continue

            top_pred = preds[0]
            predicted_digits = top_pred.digits[:n_pos]

            # Score
            straight_hit = predicted_digits == actual_digits
            box_hit = sorted(predicted_digits) == sorted(actual_digits)

            detail = BacktestDetail(
                draw_id=target_draw.id,
                predicted_digits=predicted_digits,
                actual_digits=actual_digits,
                straight_hit=straight_hit,
                box_hit=box_hit,
                confidence=top_pred.confidence,
            )
            details.append(detail)

        # Compute summary metrics
        metrics = self.compute_metrics(details, game_type)

        # Determine draw number range
        first_draw = all_draws[eval_indices[0]]
        last_draw = all_draws[eval_indices[-1]]

        run = BacktestRun(
            name=name,
            game_type=game_type,
            strategy_name=strategy.name,
            strategy_params={"training_window": training_window, "step": step},
            start_draw_number=first_draw.draw_number,
            end_draw_number=last_draw.draw_number,
            training_window=training_window,
            total_predictions=metrics["total_predictions"],
            straight_hits=metrics["straight_hits"],
            box_hits=metrics["box_hits"],
            straight_hit_rate=metrics["straight_hit_rate"],
            box_hit_rate=metrics["box_hit_rate"],
            avg_payout_per_play=metrics.get("avg_payout_per_play"),
            roi=metrics.get("roi"),
            run_duration_seconds=round(time.perf_counter() - t0, 2),
        )
        db.add(run)
        db.flush()  # get run.id

        for d in details:
            d.backtest_run_id = run.id
        db.add_all(details)
        db.commit()
        db.refresh(run)

        logger.info(
            "Backtest complete: %s | %d predictions | straight %.4f | box %.4f | %.1fs",
            run.name,
            run.total_predictions,
            run.straight_hit_rate,
            run.box_hit_rate,
            run.run_duration_seconds or 0,
        )

        return run

    def run_comparison(
        self,
        db: Session,
        strategies: list[BaseStrategy],
        game_type: str,
        **kwargs,
    ) -> list[BacktestRun]:
        """Run the same backtest configuration across multiple strategies."""
        runs: list[BacktestRun] = []
        for strat in strategies:
            logger.info("Comparison: running strategy %r ...", strat.name)
            run = self.run(db, strat, game_type, **kwargs)
            runs.append(run)
        return runs

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #

    def compute_metrics(
        self,
        details: list[BacktestDetail],
        game_type: str = "daily3",
    ) -> dict:
        """Compute comprehensive metrics from backtest detail rows.

        Returns a dict with all scalar metrics.
        """
        n = len(details)
        if n == 0:
            return {
                "total_predictions": 0,
                "straight_hits": 0,
                "box_hits": 0,
                "straight_hit_rate": 0.0,
                "box_hit_rate": 0.0,
                "avg_payout_per_play": 0.0,
                "roi": 0.0,
                "sharpe_ratio": 0.0,
                "calibration_error": 0.0,
                "partial_match_scores": [],
                "avg_partial_match_score": 0.0,
                "top_n_hit_rates": {},
            }

        straight_hits = sum(1 for d in details if d.straight_hit)
        box_hits = sum(1 for d in details if d.box_hit)
        straight_rate = straight_hits / n
        box_rate = box_hits / n

        # Baselines (random chance)
        n_pos = BaseStrategy.get_digit_count(game_type)
        baseline_straight = (1 / 10) ** n_pos  # 0.001 for D3, 0.0001 for D4
        baseline_box_approx = baseline_straight * math.factorial(n_pos)  # rough

        # Partial match scoring
        partial_scores = []
        for d in details:
            score = self._partial_match_score(d.predicted_digits, d.actual_digits)
            partial_scores.append(score)

        avg_partial = sum(partial_scores) / n if n else 0.0

        # Payout / ROI calculation
        payouts = PAYOUTS.get(game_type, PAYOUTS["daily3"])
        total_payout = 0.0
        per_play_returns: list[float] = []
        for d in details:
            payout = 0.0
            if d.straight_hit:
                payout = payouts["straight"]
            elif d.box_hit:
                payout = self._box_payout(d.actual_digits, game_type)
            total_payout += payout
            per_play_returns.append(payout - 1.0)  # net return per $1 bet

        avg_payout = total_payout / n
        roi = (total_payout - n) / n  # (total_won - total_wagered) / total_wagered

        # Sharpe ratio (annualised ~ 730 draws/year for twice-daily)
        mean_r = sum(per_play_returns) / n
        if n > 1:
            var_r = sum((r - mean_r) ** 2 for r in per_play_returns) / (n - 1)
            std_r = math.sqrt(var_r)
        else:
            std_r = 0.0
        sharpe = (mean_r / std_r * math.sqrt(730)) if std_r > 0 else 0.0

        # Calibration error: |avg_confidence - actual_hit_rate|
        avg_conf = sum(d.confidence for d in details) / n
        calibration_error = abs(avg_conf - straight_rate)

        # Top-N hit rates (using the top-1 prediction only, so these are the same
        # as straight/box when we only store 1 prediction per detail).
        # In future, if we store ranked lists, we can expand this.
        top_n_hit_rates = {
            "top_1_straight": straight_rate,
            "top_1_box": box_rate,
        }

        return {
            "total_predictions": n,
            "straight_hits": straight_hits,
            "box_hits": box_hits,
            "straight_hit_rate": round(straight_rate, 6),
            "box_hit_rate": round(box_rate, 6),
            "baseline_straight": round(baseline_straight, 6),
            "baseline_box": round(baseline_box_approx, 6),
            "avg_payout_per_play": round(avg_payout, 4),
            "roi": round(roi, 4),
            "sharpe_ratio": round(sharpe, 4),
            "calibration_error": round(calibration_error, 6),
            "partial_match_scores": partial_scores,
            "avg_partial_match_score": round(avg_partial, 4),
            "top_n_hit_rates": top_n_hit_rates,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _partial_match_score(predicted: list[int], actual: list[int]) -> float:
        """Score a prediction with partial credit.

        - 1.0 per digit in the correct position
        - 0.3 per digit present but in the wrong position
        Normalised by the number of positions.
        """
        n = min(len(predicted), len(actual))
        if n == 0:
            return 0.0

        score = 0.0
        used_actual = [False] * n

        # First pass: exact position matches
        for i in range(n):
            if predicted[i] == actual[i]:
                score += 1.0
                used_actual[i] = True

        # Second pass: digit-in-wrong-position
        for i in range(n):
            if predicted[i] == actual[i]:
                continue  # already scored
            for j in range(n):
                if not used_actual[j] and predicted[i] == actual[j]:
                    score += 0.3
                    used_actual[j] = True
                    break

        return score / n

    @staticmethod
    def _box_payout(actual_digits: list[int], game_type: str) -> float:
        """Determine the box payout based on the number of unique digit arrangements."""
        payouts = PAYOUTS[game_type]
        n = len(actual_digits)
        unique = len(set(actual_digits))

        if game_type == "daily3":
            if unique == 1:
                # All same digits (e.g., 1-1-1): no box payout distinct from straight
                return payouts["straight"]
            elif unique == 2:
                return payouts["box_3way"]
            else:
                return payouts["box_6way"]
        else:  # daily4
            if unique == 1:
                return payouts["straight"]
            elif unique == 2:
                # Could be 4-way (AAAB) or 6-way (AABB)
                from collections import Counter
                counts = sorted(Counter(actual_digits).values(), reverse=True)
                if counts[0] == 3:
                    return payouts["box_4way"]
                else:
                    return payouts["box_6way"]
            elif unique == 3:
                return payouts["box_12way"]
            else:
                return payouts["box_24way"]
