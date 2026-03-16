"""API endpoints for statistical analysis of lottery draw data."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.draw import Draw
from app.schemas.analysis import (
    FrequencyResponse,
    GapResponse,
    HotColdResponse,
    PairResponse,
    PatternResponse,
    SumResponse,
    SummaryResponse,
    TrendsResponse,
)
from app.services.statistics import (
    frequency_analysis,
    gap_analysis,
    get_full_summary,
    hot_cold_analysis,
    pair_analysis,
    pattern_analysis,
    positional_trends,
    sum_analysis,
)

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _fetch_draws(
    db: Session,
    game_type: str,
    window: int,
) -> list[Draw]:
    """Query draws from the DB filtered by game_type, ordered newest first."""
    query = (
        select(Draw)
        .where(Draw.game_type == game_type)
        .order_by(Draw.draw_date.desc(), Draw.draw_number.desc())
        .limit(window)
    )
    return list(db.execute(query).scalars().all())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/frequency", response_model=FrequencyResponse)
def get_frequency(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    window: int = Query(100, ge=10, le=5000),
    position: int | None = Query(None, ge=1, le=4, description="1-based position filter"),
    db: Session = Depends(get_db),
) -> FrequencyResponse:
    """Digit frequency analysis overall and per position."""
    draws = _fetch_draws(db, game_type, window)
    result = frequency_analysis(draws, game_type, position=position)
    return FrequencyResponse(**result)


@router.get("/hot-cold", response_model=HotColdResponse)
def get_hot_cold(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    window: int = Query(50, ge=10, le=5000),
    db: Session = Depends(get_db),
) -> HotColdResponse:
    """Hot and cold digit identification."""
    draws = _fetch_draws(db, game_type, window)
    result = hot_cold_analysis(draws, game_type, window=window)
    return HotColdResponse(**result)


@router.get("/gaps", response_model=GapResponse)
def get_gaps(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    window: int = Query(500, ge=10, le=5000),
    db: Session = Depends(get_db),
) -> GapResponse:
    """Gap analysis for each digit at each position."""
    draws = _fetch_draws(db, game_type, window)
    result = gap_analysis(draws, game_type)
    return GapResponse(**result)


@router.get("/pairs", response_model=PairResponse)
def get_pairs(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    window: int = Query(200, ge=10, le=5000),
    db: Session = Depends(get_db),
) -> PairResponse:
    """Pair and triple frequency analysis."""
    draws = _fetch_draws(db, game_type, window)
    result = pair_analysis(draws, game_type)
    return PairResponse(**result)


@router.get("/trends", response_model=TrendsResponse)
def get_trends(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    window: int = Query(30, ge=10, le=5000),
    db: Session = Depends(get_db),
) -> TrendsResponse:
    """Positional digit trend analysis using exponentially weighted moving average."""
    draws = _fetch_draws(db, game_type, window)
    result = positional_trends(draws, game_type, window=window)
    return TrendsResponse(**result)


@router.get("/sum", response_model=SumResponse)
def get_sum(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    window: int = Query(100, ge=10, le=5000),
    db: Session = Depends(get_db),
) -> SumResponse:
    """Sum distribution analysis across draws."""
    draws = _fetch_draws(db, game_type, window)
    result = sum_analysis(draws, game_type)
    return SumResponse(**result)


@router.get("/patterns", response_model=PatternResponse)
def get_patterns(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    window: int = Query(100, ge=10, le=5000),
    db: Session = Depends(get_db),
) -> PatternResponse:
    """Repeat and pattern analysis."""
    draws = _fetch_draws(db, game_type, window)
    result = pattern_analysis(draws, game_type)
    return PatternResponse(**result)


@router.get("/summary", response_model=SummaryResponse)
def get_summary(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    window: int = Query(100, ge=10, le=5000),
    db: Session = Depends(get_db),
) -> SummaryResponse:
    """Full combined analysis summary for the dashboard."""
    draws = _fetch_draws(db, game_type, window)
    result = get_full_summary(draws, game_type, window=window)
    return SummaryResponse(**result)
