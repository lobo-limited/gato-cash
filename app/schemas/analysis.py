"""Pydantic response models for the statistical analysis endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Frequency Analysis
# ---------------------------------------------------------------------------

class DigitFrequency(BaseModel):
    digit: int
    count: int
    percentage: float
    deviation: float = Field(description="Deviation from expected 10%")


class PositionFrequency(BaseModel):
    position: int
    digits: list[DigitFrequency]


class FrequencyResponse(BaseModel):
    game_type: str
    total_draws: int
    positions: list[PositionFrequency]
    overall: list[DigitFrequency]


# ---------------------------------------------------------------------------
# Hot / Cold Analysis
# ---------------------------------------------------------------------------

class HotColdDigit(BaseModel):
    digit: int
    count: int
    expected: float
    std_dev_away: float
    percentage: float


class HotColdResponse(BaseModel):
    game_type: str
    total_draws: int
    expected_count: float
    std_dev: float
    hot: list[HotColdDigit]
    cold: list[HotColdDigit]
    neutral: list[HotColdDigit]


# ---------------------------------------------------------------------------
# Gap Analysis
# ---------------------------------------------------------------------------

class DigitGap(BaseModel):
    digit: int
    current_gap: int
    average_gap: float
    max_gap: int
    overdue: bool


class PositionGap(BaseModel):
    position: int
    digits: list[DigitGap]


class GapResponse(BaseModel):
    game_type: str
    total_draws: int
    positions: list[PositionGap]


# ---------------------------------------------------------------------------
# Pair / Triple Analysis
# ---------------------------------------------------------------------------

class PairEntry(BaseModel):
    pair: str
    count: int
    percentage: float


class AdjacencyResult(BaseModel):
    positions: str = Field(description="e.g. '1-2', '2-3'")
    top_pairs: list[PairEntry]
    total_unique: int
    full_counts: dict[str, int]


class TripleEntry(BaseModel):
    triple: str
    count: int
    percentage: float


class TripleResult(BaseModel):
    top_triples: list[TripleEntry]
    total_unique: int


class PairResponse(BaseModel):
    game_type: str
    total_draws: int
    adjacencies: list[AdjacencyResult]
    triples: TripleResult | None = None


# ---------------------------------------------------------------------------
# Positional Trends
# ---------------------------------------------------------------------------

class PositionTrend(BaseModel):
    position: int
    ema: dict[int, float] = Field(description="Digit -> EMA value")
    trend: dict[int, str] = Field(description="Digit -> increasing/decreasing/stable")
    last_window_distribution: dict[int, int] = Field(description="Digit -> count in last window")


class TrendsResponse(BaseModel):
    game_type: str
    total_draws: int
    window: int
    positions: list[PositionTrend]


# ---------------------------------------------------------------------------
# Sum Analysis
# ---------------------------------------------------------------------------

class StreakInfo(BaseModel):
    direction: str
    length: int


class SumResponse(BaseModel):
    game_type: str
    total_draws: int
    histogram: dict[int, int]
    mean: float
    median: float
    std_dev: float
    current_streak: StreakInfo


# ---------------------------------------------------------------------------
# Pattern Analysis
# ---------------------------------------------------------------------------

class MatchInfo(BaseModel):
    count: int
    percentage: float


class RepeatInfo(BaseModel):
    count: int
    percentage: float


class CommonDraw(BaseModel):
    draw: str
    count: int
    percentage: float


class PatternResponse(BaseModel):
    game_type: str
    total_draws: int
    positional_matches: dict[str, MatchInfo]
    within_draw_repeats: dict[str, RepeatInfo]
    most_common_draws: list[CommonDraw]


# ---------------------------------------------------------------------------
# Full Summary
# ---------------------------------------------------------------------------

class SummaryResponse(BaseModel):
    game_type: str
    total_draws: int
    frequency: FrequencyResponse
    hot_cold: HotColdResponse
    gaps: GapResponse
    pairs: PairResponse
    trends: TrendsResponse
    sums: SumResponse
    patterns: PatternResponse
