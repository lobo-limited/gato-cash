from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class DrawBase(BaseModel):
    game_type: str = Field(..., pattern="^(daily3|daily4)$")
    draw_number: int = Field(..., gt=0)
    draw_date: datetime
    draw_time: str = Field(..., pattern="^(midday|evening)$")
    digit_1: int = Field(..., ge=0, le=9)
    digit_2: int = Field(..., ge=0, le=9)
    digit_3: int = Field(..., ge=0, le=9)
    digit_4: int | None = Field(None, ge=0, le=9)

    @field_validator("digit_4")
    @classmethod
    def validate_digit_4(cls, v, info):
        game_type = info.data.get("game_type")
        if game_type == "daily3" and v is not None:
            raise ValueError("digit_4 must be null for daily3")
        if game_type == "daily4" and v is None:
            raise ValueError("digit_4 is required for daily4")
        return v


class DrawCreate(DrawBase):
    straight_prize: float | None = None
    box_prize: float | None = None
    straight_winners: int | None = None
    box_winners: int | None = None


class DrawRead(DrawCreate):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class DrawList(BaseModel):
    draws: list[DrawRead]
    total: int
    page: int
    page_size: int
