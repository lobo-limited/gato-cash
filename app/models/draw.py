from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Float, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Draw(Base):
    __tablename__ = "draws"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_type: Mapped[str] = mapped_column(String(10), nullable=False)  # 'daily3' or 'daily4'
    draw_number: Mapped[int] = mapped_column(Integer, nullable=False)
    draw_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    draw_time: Mapped[str] = mapped_column(String(10), nullable=False)  # 'midday' or 'evening'

    digit_1: Mapped[int] = mapped_column(Integer, nullable=False)
    digit_2: Mapped[int] = mapped_column(Integer, nullable=False)
    digit_3: Mapped[int] = mapped_column(Integer, nullable=False)
    digit_4: Mapped[int | None] = mapped_column(Integer, nullable=True)  # NULL for daily3

    straight_prize: Mapped[float | None] = mapped_column(Float, nullable=True)
    box_prize: Mapped[float | None] = mapped_column(Float, nullable=True)
    straight_winners: Mapped[int | None] = mapped_column(Integer, nullable=True)
    box_winners: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    predictions = relationship("Prediction", back_populates="actual_draw", foreign_keys="Prediction.actual_draw_id")
    backtest_details = relationship("BacktestDetail", back_populates="draw")

    __table_args__ = (
        UniqueConstraint("game_type", "draw_number", name="uq_game_draw_number"),
        Index("ix_draws_game_date_desc", "game_type", draw_date.desc()),
        Index("ix_draws_game_time", "game_type", "draw_time"),
        CheckConstraint("game_type IN ('daily3', 'daily4')", name="ck_draws_game_type"),
        CheckConstraint("draw_time IN ('midday', 'evening')", name="ck_draws_draw_time"),
        CheckConstraint("digit_1 BETWEEN 0 AND 9", name="ck_draws_digit_1"),
        CheckConstraint("digit_2 BETWEEN 0 AND 9", name="ck_draws_digit_2"),
        CheckConstraint("digit_3 BETWEEN 0 AND 9", name="ck_draws_digit_3"),
        CheckConstraint("digit_4 IS NULL OR digit_4 BETWEEN 0 AND 9", name="ck_draws_digit_4"),
    )

    def __repr__(self) -> str:
        digits = f"{self.digit_1}-{self.digit_2}-{self.digit_3}"
        if self.digit_4 is not None:
            digits += f"-{self.digit_4}"
        return f"<Draw(id={self.id}, {self.game_type} #{self.draw_number}: {digits})>"
