from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BacktestRunCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    game_type: str = Field(..., pattern="^(daily3|daily4)$")
    strategy_name: str = Field(..., min_length=1, max_length=100)
    strategy_params: dict[str, Any] | None = None
    start_draw_number: int = Field(..., gt=0)
    end_draw_number: int = Field(..., gt=0)
    training_window: int = Field(..., gt=0)


class BacktestDetailRead(BaseModel):
    id: int
    backtest_run_id: int
    draw_id: int
    predicted_digits: list[int]
    actual_digits: list[int]
    straight_hit: bool
    box_hit: bool
    confidence: float

    model_config = {"from_attributes": True}


class BacktestRunRead(BaseModel):
    id: int
    user_id: int | None = None
    name: str
    game_type: str
    strategy_name: str
    strategy_params: dict[str, Any] | None = None
    start_draw_number: int
    end_draw_number: int
    training_window: int
    total_predictions: int
    straight_hits: int
    box_hits: int
    straight_hit_rate: float
    box_hit_rate: float
    avg_payout_per_play: float | None = None
    roi: float | None = None
    run_duration_seconds: float | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
