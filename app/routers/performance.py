"""Performance tracking and calibration API router.

Provides endpoints for viewing strategy performance metrics, ensemble weights,
calibration curves, degradation alerts, and manual admin triggers.
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.backtest import BacktestRun
from app.models.draw import Draw
from app.models.performance import EnsembleWeight, ModelPerformance
from app.models.prediction import Prediction
from app.services.calibration import CalibrationService
from app.services.scheduler import lottery_scheduler

logger = logging.getLogger(__name__)

router = APIRouter(tags=["performance"])

_calibration = CalibrationService()


# ------------------------------------------------------------------ #
# Performance endpoints                                                #
# ------------------------------------------------------------------ #

@router.get("/api/performance/strategies")
def get_strategy_performance(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    db: Session = Depends(get_db),
):
    """Get current performance metrics for all strategies across all windows."""
    stmt = (
        select(ModelPerformance)
        .where(ModelPerformance.game_type == game_type)
        .order_by(ModelPerformance.strategy_name, ModelPerformance.window_type)
    )
    rows = db.execute(stmt).scalars().all()

    if not rows:
        # Try computing fresh performance data.
        rows = _calibration.compute_rolling_performance(db, game_type)

    strategies: dict[str, dict] = {}
    for row in rows:
        if row.strategy_name not in strategies:
            strategies[row.strategy_name] = {"strategy_name": row.strategy_name, "windows": {}}
        strategies[row.strategy_name]["windows"][row.window_type] = {
            "total_predictions": row.total_predictions,
            "straight_hits": row.straight_hits,
            "box_hits": row.box_hits,
            "straight_hit_rate": round(row.straight_hit_rate, 6),
            "box_hit_rate": round(row.box_hit_rate, 6),
            "avg_confidence": round(row.avg_confidence, 6),
            "calibration_score": round(row.calibration_score, 6) if row.calibration_score is not None else None,
            "computed_at": row.computed_at.isoformat() if row.computed_at else None,
        }

    return {"game_type": game_type, "strategies": list(strategies.values())}


@router.get("/api/performance/weights")
def get_current_weights(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    db: Session = Depends(get_db),
):
    """Get current ensemble weights for all strategies."""
    weights = _calibration.get_current_weights(db, game_type)
    return {
        "game_type": game_type,
        "weights": weights,
        "total": round(sum(weights.values()), 6) if weights else 0,
    }


@router.get("/api/performance/weights/history")
def get_weight_history(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    days: int = Query(90, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """Get weight changes over time for charting."""
    history = _calibration.get_weight_history(db, game_type, days)
    return {"game_type": game_type, "days": days, "history": history}


@router.get("/api/performance/calibration")
def get_calibration_curve(
    strategy_name: str = Query(..., min_length=1),
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    db: Session = Depends(get_db),
):
    """Get calibration curve data for a specific strategy.

    Returns binned confidence levels vs. actual hit rates.
    Bins predictions by confidence into 10 buckets (0-10%, 10-20%, etc.)
    and computes the actual hit rate in each bucket.
    """
    stmt = (
        select(Prediction)
        .where(
            Prediction.game_type == game_type,
            Prediction.strategy_name == strategy_name,
            Prediction.scored_at.isnot(None),
            Prediction.is_backtest == False,  # noqa: E712
        )
        .order_by(Prediction.scored_at.desc())
        .limit(500)
    )
    predictions = db.execute(stmt).scalars().all()

    if not predictions:
        return {
            "strategy_name": strategy_name,
            "game_type": game_type,
            "bins": [],
            "total_predictions": 0,
        }

    # Bin predictions by confidence level.
    bins: list[dict] = []
    for i in range(10):
        low = i * 0.1
        high = (i + 1) * 0.1
        bucket = [
            p for p in predictions
            if low <= p.confidence < high or (i == 9 and p.confidence == 1.0)
        ]
        if bucket:
            n = len(bucket)
            box_hits = sum(1 for p in bucket if p.box_hit)
            straight_hits = sum(1 for p in bucket if p.straight_hit)
            bins.append({
                "confidence_range": f"{low:.1f}-{high:.1f}",
                "confidence_midpoint": round((low + high) / 2, 2),
                "count": n,
                "box_hit_rate": round(box_hits / n, 6),
                "straight_hit_rate": round(straight_hits / n, 6),
                "avg_confidence": round(sum(p.confidence for p in bucket) / n, 6),
            })
        else:
            bins.append({
                "confidence_range": f"{low:.1f}-{high:.1f}",
                "confidence_midpoint": round((low + high) / 2, 2),
                "count": 0,
                "box_hit_rate": 0.0,
                "straight_hit_rate": 0.0,
                "avg_confidence": 0.0,
            })

    return {
        "strategy_name": strategy_name,
        "game_type": game_type,
        "bins": bins,
        "total_predictions": len(predictions),
    }


@router.get("/api/performance/degradation")
def get_degradation_alerts(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    db: Session = Depends(get_db),
):
    """Check for strategy degradation alerts."""
    # Ensure performance is up to date.
    _calibration.compute_rolling_performance(db, game_type)
    alerts = _calibration.detect_degradation(db, game_type)
    return {"game_type": game_type, "alerts": alerts}


# ------------------------------------------------------------------ #
# Admin endpoints                                                      #
# ------------------------------------------------------------------ #

@router.post("/api/admin/calibrate")
def trigger_calibration(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    db: Session = Depends(get_db),
):
    """Trigger manual weight recalibration."""
    weights = _calibration.recalibrate_weights(db, game_type)
    alerts = _calibration.detect_degradation(db, game_type)
    return {
        "game_type": game_type,
        "new_weights": weights,
        "alerts": alerts,
        "recalibrated_at": datetime.utcnow().isoformat(),
    }


@router.post("/api/admin/train")
def trigger_training(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    db: Session = Depends(get_db),
):
    """Trigger model retraining for all strategies."""
    from app.strategies import get_all_strategies

    strategies = get_all_strategies()

    # Fetch full history for training.
    stmt = (
        select(Draw)
        .where(Draw.game_type == game_type)
        .order_by(Draw.draw_date.desc(), Draw.draw_number.desc())
    )
    history = list(db.execute(stmt).scalars().all())

    trained: list[str] = []
    errors: list[dict] = []

    for strategy in strategies:
        try:
            strategy.train(game_type, history)
            trained.append(strategy.name)
        except Exception as exc:
            logger.exception("Training failed for strategy %s", strategy.name)
            errors.append({"strategy": strategy.name, "error": str(exc)})

    return {
        "game_type": game_type,
        "trained": trained,
        "errors": errors,
        "trained_at": datetime.utcnow().isoformat(),
    }


@router.post("/api/admin/compute-performance")
def trigger_compute_performance(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    db: Session = Depends(get_db),
):
    """Trigger manual rolling performance computation."""
    results = _calibration.compute_rolling_performance(db, game_type)
    return {
        "game_type": game_type,
        "records_computed": len(results),
        "computed_at": datetime.utcnow().isoformat(),
    }


# ------------------------------------------------------------------ #
# Health endpoint                                                      #
# ------------------------------------------------------------------ #

@router.get("/api/health")
def health_check(db: Session = Depends(get_db)):
    """System health: last fetch time, DB stats, scheduler status."""
    # DB stats.
    d3_draws = db.execute(
        select(func.count()).select_from(Draw).where(Draw.game_type == "daily3")
    ).scalar() or 0

    d4_draws = db.execute(
        select(func.count()).select_from(Draw).where(Draw.game_type == "daily4")
    ).scalar() or 0

    total_predictions = db.execute(
        select(func.count()).select_from(Prediction).where(
            Prediction.is_backtest == False  # noqa: E712
        )
    ).scalar() or 0

    scored_predictions = db.execute(
        select(func.count()).select_from(Prediction).where(
            Prediction.is_backtest == False,  # noqa: E712
            Prediction.scored_at.isnot(None),
        )
    ).scalar() or 0

    total_backtests = db.execute(
        select(func.count()).select_from(BacktestRun)
    ).scalar() or 0

    # Latest draw dates.
    latest_d3 = db.execute(
        select(Draw.draw_date)
        .where(Draw.game_type == "daily3")
        .order_by(Draw.draw_date.desc())
        .limit(1)
    ).scalar()

    latest_d4 = db.execute(
        select(Draw.draw_date)
        .where(Draw.game_type == "daily4")
        .order_by(Draw.draw_date.desc())
        .limit(1)
    ).scalar()

    # Scheduler status.
    scheduler_status = lottery_scheduler.get_status()

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": {
            "daily3_draws": d3_draws,
            "daily4_draws": d4_draws,
            "total_predictions": total_predictions,
            "scored_predictions": scored_predictions,
            "total_backtests": total_backtests,
            "latest_daily3_draw": latest_d3.isoformat() if latest_d3 else None,
            "latest_daily4_draw": latest_d4.isoformat() if latest_d4 else None,
        },
        "scheduler": scheduler_status,
    }
