"""Prediction service that orchestrates strategy execution, ensemble
combination, database storage, and scoring.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.draw import Draw
from app.models.prediction import Prediction
from app.services.play_optimizer import PlayTypeOptimizer
from app.strategies.base import PredictionResult
from app.strategies.ensemble import EnsembleStrategy, StrategyWeight

logger = logging.getLogger(__name__)

# Number of recent draws to feed to strategies.
HISTORY_LIMIT = 500


def _get_all_strategies():
    """Lazy import so the module can load even if strategy files are not
    yet on disk (another agent is creating them concurrently)."""
    try:
        from app.strategies import get_all_strategies
        return get_all_strategies()
    except (ImportError, AttributeError):
        logger.warning("get_all_strategies not available; returning empty list")
        return []


class PredictionService:
    """High-level service that generates, stores, and scores predictions."""

    def __init__(self) -> None:
        self.optimizer = PlayTypeOptimizer()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_predictions(
        self,
        db: Session,
        game_type: str,
        draw_time: str,
    ) -> list[Prediction]:
        """Run all strategies + ensemble, store predictions in the DB.

        Returns the list of Prediction ORM objects that were persisted.
        """
        strategies = _get_all_strategies()
        if not strategies:
            logger.info("No strategies available; skipping prediction generation.")
            return []

        # Fetch historical draws for the requested game/time.
        history = self._fetch_history(db, game_type, draw_time)
        if len(history) < 10:
            logger.warning(
                "Only %d draws in history for %s/%s; predictions may be poor.",
                len(history), game_type, draw_time,
            )

        # Determine the target draw date/time.
        target_date = self._next_draw_date(history)

        # Compute average prizes from recent draws (for play optimizer).
        avg_straight_prize, avg_box_prize = self._avg_prizes(history, game_type)

        saved_predictions: list[Prediction] = []

        # ------------------------------------------------------------------
        # Individual strategy predictions
        # ------------------------------------------------------------------
        strategy_results: list[tuple[str, PredictionResult]] = []

        for strategy in strategies:
            try:
                results = strategy.predict(game_type, draw_time, history)
                if results:
                    best = results[0]
                    strategy_results.append((strategy.name, best))
                    pred = self._save_prediction(
                        db, best, strategy.name, game_type, draw_time,
                        target_date, avg_straight_prize, avg_box_prize,
                    )
                    saved_predictions.append(pred)
            except Exception:
                logger.exception("Strategy %s failed", strategy.name)

        # ------------------------------------------------------------------
        # Ensemble prediction
        # ------------------------------------------------------------------
        if strategy_results:
            ensemble = self._build_ensemble(strategies, db, game_type)
            try:
                ensemble_results = ensemble.predict(game_type, draw_time, history)
                for idx, result in enumerate(ensemble_results):
                    name = "ensemble" if idx == 0 else f"ensemble_alt_{idx}"
                    pred = self._save_prediction(
                        db, result, name, game_type, draw_time,
                        target_date, avg_straight_prize, avg_box_prize,
                    )
                    saved_predictions.append(pred)
            except Exception:
                logger.exception("Ensemble strategy failed")

        db.commit()
        return saved_predictions

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_predictions(self, db: Session, draw: Draw) -> int:
        """Score all unscored predictions whose target matches the given draw.

        Returns the number of predictions scored.
        """
        draw_digits = [draw.digit_1, draw.digit_2, draw.digit_3]
        if draw.digit_4 is not None:
            draw_digits.append(draw.digit_4)

        # Find unscored predictions targeting this draw date/time/game.
        stmt = (
            select(Prediction)
            .where(
                Prediction.game_type == draw.game_type,
                Prediction.target_draw_time == draw.draw_time,
                Prediction.scored_at.is_(None),
                Prediction.is_backtest == False,  # noqa: E712
            )
        )
        unscored = db.execute(stmt).scalars().all()

        count = 0
        for pred in unscored:
            pred_digits = [pred.digit_1, pred.digit_2, pred.digit_3]
            if pred.digit_4 is not None:
                pred_digits.append(pred.digit_4)

            # Straight: exact positional match.
            pred.straight_hit = pred_digits == draw_digits

            # Box: same digits in any order.
            pred.box_hit = sorted(pred_digits) == sorted(draw_digits)

            pred.actual_draw_id = draw.id
            pred.scored_at = datetime.utcnow()
            count += 1

        db.commit()
        return count

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_latest_predictions(
        self,
        db: Session,
        game_type: str,
        draw_time: str | None = None,
    ) -> dict:
        """Return the most recent batch of predictions, including scoring info.

        Returns a dict suitable for JSON serialisation.
        """
        stmt = (
            select(Prediction)
            .where(
                Prediction.game_type == game_type,
                Prediction.is_backtest == False,  # noqa: E712
            )
            .order_by(Prediction.created_at.desc())
            .limit(50)
        )
        if draw_time:
            stmt = stmt.where(Prediction.target_draw_time == draw_time)

        predictions = db.execute(stmt).scalars().all()

        # Group by creation batch (all predictions created within 60s).
        if not predictions:
            return {"predictions": [], "ensemble": None, "strategies": []}

        latest_time = predictions[0].created_at
        batch = [
            p for p in predictions
            if abs((p.created_at - latest_time).total_seconds()) < 60
        ]

        ensemble_pred = None
        strategy_preds: list[dict] = []

        for p in batch:
            info = self._prediction_to_dict(p)
            if p.strategy_name.startswith("ensemble"):
                if ensemble_pred is None or p.strategy_name == "ensemble":
                    ensemble_pred = info
            else:
                strategy_preds.append(info)

        return {
            "predictions": [self._prediction_to_dict(p) for p in batch],
            "ensemble": ensemble_pred,
            "strategies": strategy_preds,
        }

    def get_prediction_history(
        self,
        db: Session,
        game_type: str,
        limit: int = 20,
    ) -> list[dict]:
        """Return scored predictions for the history table."""
        stmt = (
            select(Prediction)
            .where(
                Prediction.game_type == game_type,
                Prediction.is_backtest == False,  # noqa: E712
            )
            .order_by(Prediction.created_at.desc())
            .limit(limit)
        )
        predictions = db.execute(stmt).scalars().all()
        return [self._prediction_to_dict(p) for p in predictions]

    def get_prediction_detail(self, db: Session, prediction_id: int) -> dict | None:
        """Return full details for a single prediction."""
        pred = db.get(Prediction, prediction_id)
        if pred is None:
            return None
        info = self._prediction_to_dict(pred)

        # Attach the actual draw digits if scored.
        if pred.actual_draw_id:
            draw = db.get(Draw, pred.actual_draw_id)
            if draw:
                actual_digits = [draw.digit_1, draw.digit_2, draw.digit_3]
                if draw.digit_4 is not None:
                    actual_digits.append(draw.digit_4)
                info["actual_digits"] = actual_digits

        return info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_history(
        self, db: Session, game_type: str, draw_time: str,
    ) -> list[Draw]:
        stmt = (
            select(Draw)
            .where(Draw.game_type == game_type)
            .order_by(Draw.draw_date.desc(), Draw.draw_number.desc())
            .limit(HISTORY_LIMIT)
        )
        return list(db.execute(stmt).scalars().all())

    @staticmethod
    def _next_draw_date(history: list[Draw]) -> datetime:
        """Estimate the next draw date from history."""
        if history:
            latest = history[0].draw_date
            return latest + timedelta(days=1)
        return datetime.utcnow() + timedelta(days=1)

    @staticmethod
    def _avg_prizes(history: list[Draw], game_type: str) -> tuple[float, float]:
        """Average straight and box prizes from the last 30 draws."""
        recent = history[:30]
        straight_prizes = [d.straight_prize for d in recent if d.straight_prize]
        box_prizes = [d.box_prize for d in recent if d.box_prize]

        defaults = PlayTypeOptimizer.DEFAULT_PRIZES.get(
            game_type, PlayTypeOptimizer.DEFAULT_PRIZES["daily3"]
        )
        avg_s = sum(straight_prizes) / len(straight_prizes) if straight_prizes else defaults["straight"]
        avg_b = sum(box_prizes) / len(box_prizes) if box_prizes else defaults["box"]
        return avg_s, avg_b

    def _build_ensemble(
        self,
        strategies: list,
        db: Session,
        game_type: str,
    ) -> EnsembleStrategy:
        """Create an EnsembleStrategy with equal weights for each strategy."""
        n = len(strategies)
        default_weight = 1.0 / n if n else 1.0

        sw_list = [
            StrategyWeight(strategy=s, weight=default_weight)
            for s in strategies
        ]

        ensemble = EnsembleStrategy(strategy_weights=sw_list)

        # Try to load custom weights from the database.
        try:
            from app.models.performance import EnsembleWeight
            stmt = (
                select(EnsembleWeight)
                .where(
                    EnsembleWeight.game_type == game_type,
                    EnsembleWeight.effective_to.is_(None),
                )
            )
            custom_weights = db.execute(stmt).scalars().all()
            if custom_weights:
                weight_map = {cw.strategy_name: cw.weight for cw in custom_weights}
                ensemble.set_weights(weight_map)
        except Exception:
            logger.debug("Could not load custom ensemble weights; using defaults.")

        return ensemble

    def _save_prediction(
        self,
        db: Session,
        result: PredictionResult,
        strategy_name: str,
        game_type: str,
        draw_time: str,
        target_date: datetime,
        avg_straight_prize: float,
        avg_box_prize: float,
    ) -> Prediction:
        """Persist a PredictionResult as a Prediction row."""
        digits = result.digits
        num_positions = 3 if game_type == "daily3" else 4

        # Pad or truncate digits.
        while len(digits) < num_positions:
            digits.append(0)

        # Run the play optimizer.
        play_rec = self.optimizer.recommend(
            digits=digits[:num_positions],
            game_type=game_type,
            digit_probabilities=result.digit_probabilities,
            avg_straight_prize=avg_straight_prize,
            avg_box_prize=avg_box_prize,
        )

        best_play = play_rec["recommended_play"]
        best_ev = play_rec["expected_values"].get(best_play, 0.0)

        pred = Prediction(
            game_type=game_type,
            target_draw_date=target_date,
            target_draw_time=draw_time,
            strategy_name=strategy_name,
            digit_1=digits[0],
            digit_2=digits[1],
            digit_3=digits[2],
            digit_4=digits[3] if num_positions == 4 else None,
            confidence=round(result.confidence, 4),
            recommended_play_type=best_play,
            expected_value=round(best_ev, 6),
            is_backtest=False,
            metadata_json={
                "digit_probabilities": [
                    [round(p, 6) for p in pos_probs]
                    for pos_probs in result.digit_probabilities
                ],
                "play_recommendation": play_rec,
                **(result.metadata or {}),
            },
        )
        db.add(pred)
        return pred

    @staticmethod
    def _prediction_to_dict(pred: Prediction) -> dict:
        """Convert a Prediction ORM object to a JSON-friendly dict."""
        digits = [pred.digit_1, pred.digit_2, pred.digit_3]
        if pred.digit_4 is not None:
            digits.append(pred.digit_4)

        return {
            "id": pred.id,
            "game_type": pred.game_type,
            "target_draw_date": pred.target_draw_date.isoformat() if pred.target_draw_date else None,
            "target_draw_time": pred.target_draw_time,
            "strategy_name": pred.strategy_name,
            "digits": digits,
            "confidence": pred.confidence,
            "recommended_play_type": pred.recommended_play_type,
            "expected_value": pred.expected_value,
            "straight_hit": pred.straight_hit,
            "box_hit": pred.box_hit,
            "scored_at": pred.scored_at.isoformat() if pred.scored_at else None,
            "created_at": pred.created_at.isoformat() if pred.created_at else None,
            "metadata": pred.metadata_json,
        }
