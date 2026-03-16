from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ModelPerformance(Base):
    __tablename__ = "model_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False)
    game_type: Mapped[str] = mapped_column(String(10), nullable=False)
    window_type: Mapped[str] = mapped_column(String(10), nullable=False)  # '7d','30d','90d','180d','365d','all'

    total_predictions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    straight_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    box_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    straight_hit_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    box_hit_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    calibration_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_model_perf_strategy_game", "strategy_name", "game_type"),
        Index("ix_model_perf_window", "window_type"),
    )

    def __repr__(self) -> str:
        return (
            f"<ModelPerformance(strategy={self.strategy_name!r}, "
            f"game={self.game_type}, window={self.window_type}, "
            f"straight_rate={self.straight_hit_rate:.4f})>"
        )


class EnsembleWeight(Base):
    __tablename__ = "ensemble_weights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    game_type: Mapped[str] = mapped_column(String(10), nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(100), nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    effective_from: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    effective_to: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    reason: Mapped[str | None] = mapped_column(String(500), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="ensemble_weights")

    __table_args__ = (
        Index("ix_ensemble_weights_game_strategy", "game_type", "strategy_name"),
        Index("ix_ensemble_weights_user", "user_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<EnsembleWeight(strategy={self.strategy_name!r}, "
            f"game={self.game_type}, weight={self.weight:.3f})>"
        )


class StatSnapshot(Base):
    __tablename__ = "stat_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_type: Mapped[str] = mapped_column(String(10), nullable=False)
    snapshot_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    stat_type: Mapped[str] = mapped_column(String(50), nullable=False)
    parameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    data: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_stat_snapshots_game_type", "game_type", "stat_type"),
        Index("ix_stat_snapshots_date", "snapshot_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<StatSnapshot(game={self.game_type}, "
            f"type={self.stat_type!r}, date={self.snapshot_date})>"
        )
