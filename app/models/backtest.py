from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    game_type: Mapped[str] = mapped_column(String(10), nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False)
    strategy_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    start_draw_number: Mapped[int] = mapped_column(Integer, nullable=False)
    end_draw_number: Mapped[int] = mapped_column(Integer, nullable=False)
    training_window: Mapped[int] = mapped_column(Integer, nullable=False)

    total_predictions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    straight_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    box_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    straight_hit_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    box_hit_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_payout_per_play: Mapped[float | None] = mapped_column(Float, nullable=True)
    roi: Mapped[float | None] = mapped_column(Float, nullable=True)
    run_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="backtest_runs")
    details = relationship(
        "BacktestDetail",
        back_populates="backtest_run",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )

    __table_args__ = (
        Index("ix_backtest_runs_strategy", "strategy_name", "game_type"),
        Index("ix_backtest_runs_user", "user_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<BacktestRun(id={self.id}, name={self.name!r}, "
            f"strategy={self.strategy_name}, "
            f"straight_rate={self.straight_hit_rate:.4f})>"
        )


class BacktestDetail(Base):
    __tablename__ = "backtest_details"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    backtest_run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False
    )
    draw_id: Mapped[int] = mapped_column(Integer, ForeignKey("draws.id"), nullable=False)
    predicted_digits: Mapped[dict] = mapped_column(JSON, nullable=False)
    actual_digits: Mapped[dict] = mapped_column(JSON, nullable=False)
    straight_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    box_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    backtest_run = relationship("BacktestRun", back_populates="details")
    draw = relationship("Draw", back_populates="backtest_details")

    __table_args__ = (
        Index("ix_backtest_details_run", "backtest_run_id"),
        Index("ix_backtest_details_draw", "draw_id"),
    )

    def __repr__(self) -> str:
        hit = "STRAIGHT" if self.straight_hit else ("BOX" if self.box_hit else "miss")
        return f"<BacktestDetail(run={self.backtest_run_id}, draw={self.draw_id}, {hit})>"
