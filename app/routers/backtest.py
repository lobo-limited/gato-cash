"""Backtesting API router.

Endpoints for launching backtests, viewing results, and comparing strategies.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.backtest import BacktestDetail, BacktestRun
from app.models.draw import Draw
from app.services.backtest import BacktestEngine
from app.services.significance import SignificanceTests
from app.strategies import get_all_strategies

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/backtest", tags=["backtest"])

_engine = BacktestEngine()


def _strategy_by_name(name: str):
    """Look up a strategy instance by its name attribute."""
    for s in get_all_strategies():
        if s.name == name:
            return s
    return None


def _run_to_dict(run: BacktestRun) -> dict:
    return {
        "id": run.id,
        "name": run.name,
        "game_type": run.game_type,
        "strategy_name": run.strategy_name,
        "strategy_params": run.strategy_params,
        "start_draw_number": run.start_draw_number,
        "end_draw_number": run.end_draw_number,
        "training_window": run.training_window,
        "total_predictions": run.total_predictions,
        "straight_hits": run.straight_hits,
        "box_hits": run.box_hits,
        "straight_hit_rate": run.straight_hit_rate,
        "box_hit_rate": run.box_hit_rate,
        "avg_payout_per_play": run.avg_payout_per_play,
        "roi": run.roi,
        "run_duration_seconds": run.run_duration_seconds,
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


def _detail_to_dict(detail: BacktestDetail) -> dict:
    return {
        "id": detail.id,
        "draw_id": detail.draw_id,
        "predicted_digits": detail.predicted_digits,
        "actual_digits": detail.actual_digits,
        "straight_hit": detail.straight_hit,
        "box_hit": detail.box_hit,
        "confidence": detail.confidence,
    }


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@router.post("/run")
def start_backtest(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    strategy_name: str = Query("frequency"),
    training_window: int = Query(200, ge=10, le=500),
    start_index: Optional[int] = Query(None, ge=0),
    end_index: Optional[int] = Query(None, ge=1),
    step: int = Query(1, ge=1, le=50),
    name: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Launch a walk-forward backtest for a single strategy.

    If ``strategy_name`` is ``"all"``, runs a comparison across every
    registered strategy.
    """
    if strategy_name == "all":
        strategies = get_all_strategies()
        runs = _engine.run_comparison(
            db,
            strategies,
            game_type,
            training_window=training_window,
            start_index=start_index,
            end_index=end_index,
            step=step,
        )
        return {"runs": [_run_to_dict(r) for r in runs]}

    strategy = _strategy_by_name(strategy_name)
    if strategy is None:
        available = [s.name for s in get_all_strategies()]
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy {strategy_name!r}. Available: {available}",
        )

    run = _engine.run(
        db,
        strategy,
        game_type,
        training_window=training_window,
        start_index=start_index,
        end_index=end_index,
        step=step,
        name=name,
    )
    return _run_to_dict(run)


@router.get("/runs")
def list_runs(
    game_type: Optional[str] = Query(None, pattern="^(daily3|daily4)$"),
    db: Session = Depends(get_db),
):
    """List all backtest runs, optionally filtered by game type."""
    query = select(BacktestRun).order_by(BacktestRun.created_at.desc())
    if game_type:
        query = query.where(BacktestRun.game_type == game_type)
    runs = db.execute(query).scalars().all()
    return [_run_to_dict(r) for r in runs]


@router.get("/runs/{run_id}")
def get_run(run_id: int, db: Session = Depends(get_db)):
    """Get detailed summary for a single backtest run, including recomputed metrics."""
    run = db.get(BacktestRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Backtest run not found")

    # Also compute extended metrics from the details
    details = list(
        db.execute(
            select(BacktestDetail)
            .where(BacktestDetail.backtest_run_id == run_id)
            .order_by(BacktestDetail.id.asc())
        ).scalars().all()
    )
    extended_metrics = _engine.compute_metrics(details, run.game_type)

    # Wilson CI for straight and box hit rates
    straight_ci = SignificanceTests.wilson_confidence_interval(
        run.straight_hits, run.total_predictions
    )
    box_ci = SignificanceTests.wilson_confidence_interval(
        run.box_hits, run.total_predictions
    )

    result = _run_to_dict(run)
    result["extended_metrics"] = {
        "sharpe_ratio": extended_metrics.get("sharpe_ratio"),
        "calibration_error": extended_metrics.get("calibration_error"),
        "avg_partial_match_score": extended_metrics.get("avg_partial_match_score"),
        "baseline_straight": extended_metrics.get("baseline_straight"),
        "baseline_box": extended_metrics.get("baseline_box"),
        "top_n_hit_rates": extended_metrics.get("top_n_hit_rates"),
    }
    result["straight_ci"] = list(straight_ci)
    result["box_ci"] = list(box_ci)

    # Rolling hit rate for chart (cumulative hit rate over time)
    rolling_straight: list[float] = []
    rolling_box: list[float] = []
    s_total = 0
    b_total = 0
    for i, d in enumerate(details, 1):
        s_total += int(d.straight_hit)
        b_total += int(d.box_hit)
        rolling_straight.append(round(s_total / i, 6))
        rolling_box.append(round(b_total / i, 6))
    result["rolling_straight_rate"] = rolling_straight
    result["rolling_box_rate"] = rolling_box

    return result


@router.get("/runs/{run_id}/details")
def get_run_details(
    run_id: int,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Get individual prediction details for a backtest run (paginated)."""
    run = db.get(BacktestRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Backtest run not found")

    total = db.execute(
        select(func.count())
        .select_from(BacktestDetail)
        .where(BacktestDetail.backtest_run_id == run_id)
    ).scalar() or 0

    details = db.execute(
        select(BacktestDetail)
        .where(BacktestDetail.backtest_run_id == run_id)
        .order_by(BacktestDetail.id.asc())
        .offset(offset)
        .limit(limit)
    ).scalars().all()

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "details": [_detail_to_dict(d) for d in details],
    }


@router.get("/compare")
def compare_runs(
    run_ids: str = Query(..., description="Comma-separated run IDs, e.g. 1,2,3"),
    db: Session = Depends(get_db),
):
    """Side-by-side comparison of multiple backtest runs.

    Includes McNemar and Friedman significance tests.
    """
    ids = [int(x.strip()) for x in run_ids.split(",") if x.strip()]
    if len(ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 run IDs to compare")
    if len(ids) > 6:
        raise HTTPException(status_code=400, detail="Maximum 6 runs for comparison")

    runs_data = []
    all_details: dict[int, list[BacktestDetail]] = {}

    for rid in ids:
        run = db.get(BacktestRun, rid)
        if not run:
            raise HTTPException(status_code=404, detail=f"Run {rid} not found")
        runs_data.append(_run_to_dict(run))
        all_details[rid] = list(
            db.execute(
                select(BacktestDetail)
                .where(BacktestDetail.backtest_run_id == rid)
                .order_by(BacktestDetail.id.asc())
            ).scalars().all()
        )

    # Build aligned draw-level hit arrays for significance tests
    # Find common draws across all runs
    draw_id_sets = [
        {d.draw_id for d in all_details[rid]} for rid in ids
    ]
    common_draw_ids = set.intersection(*draw_id_sets) if draw_id_sets else set()
    common_draws = sorted(common_draw_ids)

    comparisons = []
    if len(common_draws) > 0:
        # Build per-run hit vectors aligned to common draws
        hit_vectors_straight: dict[int, list[bool]] = {}
        hit_vectors_box: dict[int, list[bool]] = {}
        for rid in ids:
            detail_map = {d.draw_id: d for d in all_details[rid]}
            hit_vectors_straight[rid] = [detail_map[did].straight_hit for did in common_draws]
            hit_vectors_box[rid] = [detail_map[did].box_hit for did in common_draws]

        # Pairwise McNemar tests
        for i_idx in range(len(ids)):
            for j_idx in range(i_idx + 1, len(ids)):
                rid_a, rid_b = ids[i_idx], ids[j_idx]
                mcn_straight = SignificanceTests.mcnemar_test(
                    hit_vectors_straight[rid_a], hit_vectors_straight[rid_b]
                )
                mcn_box = SignificanceTests.mcnemar_test(
                    hit_vectors_box[rid_a], hit_vectors_box[rid_b]
                )
                comparisons.append({
                    "run_a": rid_a,
                    "run_b": rid_b,
                    "mcnemar_straight": mcn_straight,
                    "mcnemar_box": mcn_box,
                })

        # Friedman test (if 3+ runs)
        if len(ids) >= 3:
            hit_matrix = [hit_vectors_straight[rid] for rid in ids]
            friedman = SignificanceTests.friedman_test(hit_matrix)
        else:
            friedman = None
    else:
        friedman = None

    return {
        "runs": runs_data,
        "common_draws_count": len(common_draws),
        "pairwise_comparisons": comparisons,
        "friedman_test": friedman,
    }


@router.get("/strategies")
def list_strategies():
    """Return the list of available strategy names."""
    return [{"name": s.name} for s in get_all_strategies()]
