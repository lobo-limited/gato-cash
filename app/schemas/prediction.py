from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PredictionBase(BaseModel):
    game_type: str = Field(..., pattern="^(daily3|daily4)$")
    target_draw_date: datetime
    target_draw_time: str = Field(..., pattern="^(midday|evening)$")
    strategy_name: str = Field(..., min_length=1, max_length=100)
    digit_1: int = Field(..., ge=0, le=9)
    digit_2: int = Field(..., ge=0, le=9)
    digit_3: int = Field(..., ge=0, le=9)
    digit_4: int | None = Field(None, ge=0, le=9)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    recommended_play_type: str | None = None
    expected_value: float | None = None


class PredictionCreate(PredictionBase):
    is_backtest: bool = False
    metadata_json: dict[str, Any] | None = None


class PredictionRead(PredictionBase):
    id: int
    user_id: int | None = None
    is_backtest: bool
    metadata_json: dict[str, Any] | None = None
    actual_draw_id: int | None = None
    straight_hit: bool | None = None
    box_hit: bool | None = None
    scored_at: datetime | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
