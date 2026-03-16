"""Prediction strategies for the CA Lottery prediction app.

Each strategy implements ``BaseStrategy`` and produces ``PredictionResult``
objects that include per-position probability distributions.
"""

from app.strategies.base import BaseStrategy, PredictionResult
from app.strategies.cold_due import ColdDueStrategy
from app.strategies.frequency import FrequencyStrategy
from app.strategies.lstm_model import LSTMStrategy
from app.strategies.moving_avg import MovingAverageStrategy
from app.strategies.pair_analysis import PairAnalysisStrategy
from app.strategies.positional import PositionalTrendStrategy
from app.strategies.random_forest import RandomForestStrategy
from app.strategies.xgboost_model import XGBoostStrategy

__all__ = [
    "BaseStrategy",
    "PredictionResult",
    "FrequencyStrategy",
    "ColdDueStrategy",
    "PositionalTrendStrategy",
    "PairAnalysisStrategy",
    "MovingAverageStrategy",
    "RandomForestStrategy",
    "XGBoostStrategy",
    "LSTMStrategy",
    "get_all_strategies",
    "get_statistical_strategies",
    "get_ml_strategies",
]


def get_statistical_strategies() -> list[BaseStrategy]:
    """Return fresh instances of purely statistical (non-ML) strategies."""
    return [
        FrequencyStrategy(),
        ColdDueStrategy(),
        PositionalTrendStrategy(),
        PairAnalysisStrategy(),
        MovingAverageStrategy(),
    ]


def get_ml_strategies() -> list[BaseStrategy]:
    """Return fresh instances of ML-based strategies."""
    return [
        RandomForestStrategy(),
        XGBoostStrategy(),
        LSTMStrategy(),
    ]


def get_all_strategies(include_ml: bool = True) -> list[BaseStrategy]:
    """Return fresh instances of every registered prediction strategy.

    Parameters
    ----------
    include_ml:
        When ``True`` (default), ML strategies (RandomForest, XGBoost, LSTM)
        are appended to the list.  Set to ``False`` for fast, stats-only
        predictions.
    """
    strategies = get_statistical_strategies()
    if include_ml:
        strategies.extend(get_ml_strategies())
    return strategies
