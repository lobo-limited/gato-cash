"""Predictions API router.

Exposes endpoints for generating predictions, retrieving the latest
predictions, viewing prediction history, and getting detailed breakdowns.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/api/predictions", tags=["predictions"])

_service = PredictionService()


@router.post("/generate")
def generate_predictions(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    draw_time: str = Query("evening", pattern="^(midday|evening)$"),
    db: Session = Depends(get_db),
):
    """Trigger prediction generation for the next draw."""
    predictions = _service.generate_predictions(db, game_type, draw_time)
    return {
        "count": len(predictions),
        "predictions": [_service._prediction_to_dict(p) for p in predictions],
    }


@router.get("/next")
def get_next_predictions(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    draw_time: str = Query("evening", pattern="^(midday|evening)$"),
    db: Session = Depends(get_db),
):
    """Get the most recent predictions (generate if none exist)."""
    data = _service.get_latest_predictions(db, game_type, draw_time)

    # If no predictions exist yet, generate them.
    if not data["predictions"]:
        predictions = _service.generate_predictions(db, game_type, draw_time)
        if predictions:
            data = _service.get_latest_predictions(db, game_type, draw_time)

    return data


@router.get("/history")
def get_prediction_history(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get past predictions with scoring results."""
    return _service.get_prediction_history(db, game_type, limit)


@router.get("/{prediction_id}")
def get_prediction_detail(
    prediction_id: int,
    db: Session = Depends(get_db),
):
    """Get detailed view of a single prediction including strategy breakdown."""
    detail = _service.get_prediction_detail(db, prediction_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return detail
