"""Model for tracking user-played numbers and their accuracy."""

from datetime import datetime

from sqlalchemy import (
    Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index,
    Integer, String, Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class NumbersPlayed(Base):
    __tablename__ = "numbers_played"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    game_type: Mapped[str] = mapped_column(String(10), nullable=False)
    play_type: Mapped[str] = mapped_column(String(20), nullable=False)  # straight, box, straight_box, combo
    digit_1: Mapped[int] = mapped_column(Integer, nullable=False)
    digit_2: Mapped[int] = mapped_column(Integer, nullable=False)
    digit_3: Mapped[int] = mapped_column(Integer, nullable=False)
    digit_4: Mapped[int | None] = mapped_column(Integer, nullable=True)

    target_draw_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    target_draw_time: Mapped[str] = mapped_column(String(10), nullable=False)  # midday, evening
    amount_wagered: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Scoring — filled after draw occurs
    matched_draw_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("draws.id"), nullable=True)
    straight_hit: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    box_hit: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    amount_won: Mapped[float | None] = mapped_column(Float, nullable=True)
    scored_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", backref="numbers_played")
    matched_draw = relationship("Draw", backref="numbers_played")

    __table_args__ = (
        Index("ix_np_user_game", "user_id", "game_type"),
        Index("ix_np_target_date", "target_draw_date", "target_draw_time"),
        Index("ix_np_created", "created_at"),
        CheckConstraint("game_type IN ('daily3', 'daily4')", name="ck_np_game_type"),
        CheckConstraint("play_type IN ('straight', 'box', 'straight_box', 'combo')", name="ck_np_play_type"),
        CheckConstraint("digit_1 BETWEEN 0 AND 9", name="ck_np_digit_1"),
        CheckConstraint("digit_2 BETWEEN 0 AND 9", name="ck_np_digit_2"),
        CheckConstraint("digit_3 BETWEEN 0 AND 9", name="ck_np_digit_3"),
        CheckConstraint("digit_4 IS NULL OR digit_4 BETWEEN 0 AND 9", name="ck_np_digit_4"),
    )

    @property
    def digits(self) -> list[int]:
        d = [self.digit_1, self.digit_2, self.digit_3]
        if self.digit_4 is not None:
            d.append(self.digit_4)
        return d

    def __repr__(self) -> str:
        return f"<NumbersPlayed(id={self.id}, {self.game_type} {self.digits} {self.play_type})>"
