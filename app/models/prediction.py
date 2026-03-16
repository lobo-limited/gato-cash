from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    game_type: Mapped[str] = mapped_column(String(10), nullable=False)
    target_draw_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    target_draw_time: Mapped[str] = mapped_column(String(10), nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False)

    digit_1: Mapped[int] = mapped_column(Integer, nullable=False)
    digit_2: Mapped[int] = mapped_column(Integer, nullable=False)
    digit_3: Mapped[int] = mapped_column(Integer, nullable=False)
    digit_4: Mapped[int | None] = mapped_column(Integer, nullable=True)

    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    recommended_play_type: Mapped[str | None] = mapped_column(String(20), nullable=True)  # 'straight', 'box', etc.
    expected_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_backtest: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)

    actual_draw_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("draws.id"), nullable=True)
    straight_hit: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    box_hit: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    scored_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="predictions")
    actual_draw = relationship("Draw", back_populates="predictions", foreign_keys=[actual_draw_id])

    __table_args__ = (
        Index("ix_predictions_target_lookup", "game_type", "target_draw_date", "target_draw_time"),
        Index("ix_predictions_strategy", "strategy_name", "game_type"),
        Index("ix_predictions_user", "user_id"),
        Index("ix_predictions_backtest", "is_backtest"),
    )

    def __repr__(self) -> str:
        digits = f"{self.digit_1}-{self.digit_2}-{self.digit_3}"
        if self.digit_4 is not None:
            digits += f"-{self.digit_4}"
        return f"<Prediction(id={self.id}, {self.strategy_name}: {digits}, conf={self.confidence:.2f})>"
