"""calottery-predict — kernel-composed sibling to app/main.py.

Runs on KERNEL_PORT (default 8211) so we can verify parity against the
production :8201 server without disrupting traffic. Cuts over by editing
the calottery-predict service ExecStart once parity is confirmed.

Per proposal §6.6, this migration is INTENTIONALLY MINIMAL: kernel injects
only at the FastAPI factory + DB + observability layer. The ML pipeline
(`app/strategies/*`, `app/services/{ingestion,prediction,calibration,backtest}`,
APScheduler job definitions) is months of tuning — touching it risks
silent regression on real revenue. Kept untouched.

What is wired through @gato/* kernel packages:
  - gato_core         — FastAPI factory, GatoSettings parent, structlog mw
  - gato_observability — Prometheus /metrics, request logging mw, InflightTracker

What stays unchanged (preserves the live lottery.db schema + ML pipeline):
  - app/database.py   — engine, Base, get_db, WAL pragmas
  - app/config.py     — Settings (DATABASE_URL, CA_LOTTERY_API_BASE, etc.)
  - app/models/*      — 7 ORM models (Draw, Prediction, BacktestRun, etc.)
  - app/routers/*     — 7 routers (draws, auth, analysis, predictions,
                         performance, backtest, numbers_played)
  - app/services/*    — 11 services including the scheduler + ML strategies
  - app/strategies/*  — 9 prediction strategies (DO NOT TOUCH)
  - alembic/          — schema migrations
  - templates/        — 9 Jinja2 dashboard pages

Critical safety: the kernel sibling does NOT start the APScheduler. Two
schedulers writing to the same lottery.db would cause double-ingestion +
double-scoring. The legacy production process keeps owning the schedule.
"""

from __future__ import annotations

import os
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from gato_core import GatoSettings, create_app, get_logger
from gato_observability import (
    InflightTracker,
    request_logging_middleware,
    setup_metrics,
)

from app.config import settings as legacy_settings
from app.database import Base, engine, get_db
from app.models import (  # noqa: F401 — register all models with Base
    BacktestDetail,
    BacktestRun,
    Draw,
    EnsembleWeight,
    ModelPerformance,
    Prediction,
    StatSnapshot,
    User,
)
from app.routers import (
    analysis,
    auth,
    backtest,
    draws,
    numbers_played,
    predictions,
)
from app.routers import performance as performance_router
from app.services.ingestion import IngestionService
from app.services.scheduler import lottery_scheduler

_log = get_logger("calottery-predict")

# When the kernel sibling is the production process (KERNEL_PORT unset, PORT
# is 8201 from systemd), it MUST own the APScheduler so draws keep ingesting,
# predictions keep scoring, and ensemble weights keep recalibrating.
#
# When the kernel sibling is in parallel-test mode (KERNEL_PORT explicitly
# set so a sibling port is used), it MUST NOT start the scheduler — the
# legacy production process is still running and owns the schedule, and two
# schedulers writing the same lottery.db would cause duplicate ingestion +
# double-scoring.
#
# The signal: KERNEL_PORT presence = parallel-test mode = NO scheduler.
KERNEL_IS_PRODUCTION = os.getenv("KERNEL_PORT") is None
BASE_DIR = Path(__file__).resolve().parent


class CalotterySettings(GatoSettings):
    """Wraps GatoSettings; the legacy app.config.Settings still owns the
    ML/CALottery-specific config (DATABASE_URL, CA_LOTTERY_API_BASE, game
    ids, etc.). Kernel only needs CORS + log + internal-token + standard
    /version endpoint."""

    app_name: str = "calottery-predict"
    app_version: str = "0.2.0-kernel"
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True


@asynccontextmanager
async def kernel_lifespan(_app: FastAPI):
    Base.metadata.create_all(bind=engine)
    if KERNEL_IS_PRODUCTION:
        lottery_scheduler.start()
        scheduler_state = "STARTED — kernel is production"
    else:
        scheduler_state = "DISABLED — KERNEL_PORT is set, parallel-test mode (legacy owns the schedule)"
    _log.info(
        "kernel_lifespan_ready",
        database_url=legacy_settings.DATABASE_URL,
        app_mode=legacy_settings.APP_MODE,
        scheduler=scheduler_state,
    )
    yield
    if KERNEL_IS_PRODUCTION:
        lottery_scheduler.shutdown()
    _log.info("kernel_lifespan_shutdown")


# ── Compose ─────────────────────────────────────────────────────────────────

settings = CalotterySettings()
app: FastAPI = create_app(settings=settings, lifespan=kernel_lifespan)
app.middleware("http")(request_logging_middleware)
setup_metrics(app)

# Per-user inflight tracker (visible to admin/future rate-limit hooks).
inflight = InflightTracker()
app.state.inflight = inflight

# Strip gato_core's generic /health so the legacy contract wins (calottery-
# predict has no router-owned /health today, so there's nothing to override —
# we just leave gato_core's default in place).

# ── Templates + static ──────────────────────────────────────────────────────

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ── Mount existing routers (unchanged) ──────────────────────────────────────

app.include_router(draws.router)
app.include_router(auth.router)
app.include_router(analysis.router)
app.include_router(predictions.router)
app.include_router(performance_router.router)
app.include_router(backtest.router)
app.include_router(numbers_played.router)


# ── Page handlers (preserved verbatim from app/main.py) ─────────────────────

def _draw_to_dict(draw: Draw) -> dict:
    digits = [draw.digit_1, draw.digit_2, draw.digit_3]
    if draw.digit_4 is not None:
        digits.append(draw.digit_4)
    return {
        "draw_number": draw.draw_number,
        "draw_date": draw.draw_date,
        "draw_time": draw.draw_time,
        "digits": digits,
        "straight_prize": draw.straight_prize,
        "box_prize": draw.box_prize,
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    d3_count = db.execute(
        select(func.count()).select_from(Draw).where(Draw.game_type == "daily3")
    ).scalar() or 0
    d4_count = db.execute(
        select(func.count()).select_from(Draw).where(Draw.game_type == "daily4")
    ).scalar() or 0

    latest_d3 = db.execute(
        select(Draw).where(Draw.game_type == "daily3")
        .order_by(Draw.draw_date.desc(), Draw.draw_number.desc()).limit(1)
    ).scalar_one_or_none()
    latest_d4 = db.execute(
        select(Draw).where(Draw.game_type == "daily4")
        .order_by(Draw.draw_date.desc(), Draw.draw_number.desc()).limit(1)
    ).scalar_one_or_none()

    recent_d3 = db.execute(
        select(Draw).where(Draw.game_type == "daily3")
        .order_by(Draw.draw_date.desc(), Draw.draw_number.desc()).limit(10)
    ).scalars().all()
    recent_d4 = db.execute(
        select(Draw).where(Draw.game_type == "daily4")
        .order_by(Draw.draw_date.desc(), Draw.draw_number.desc()).limit(10)
    ).scalars().all()

    freq_draws = db.execute(
        select(Draw).where(Draw.game_type == "daily3")
        .order_by(Draw.draw_date.desc(), Draw.draw_number.desc()).limit(100)
    ).scalars().all()
    digit_counts: Counter = Counter()
    for d in freq_draws:
        digit_counts[d.digit_1] += 1
        digit_counts[d.digit_2] += 1
        digit_counts[d.digit_3] += 1
    freq_data = {
        "labels": [str(i) for i in range(10)],
        "counts": [digit_counts.get(i, 0) for i in range(10)],
    }

    # Starlette 1.0+ deprecated the positional `TemplateResponse(name, ctx)`
    # signature; legacy app/main.py uses it because it has starlette<1. The
    # kernel venv pulls starlette 1.0 transitively. Use request-first form
    # which works in both versions.
    return templates.TemplateResponse(request, "dashboard.html", {
        "active_page": "dashboard",
        "stats": {"daily3_count": d3_count, "daily4_count": d4_count},
        "latest_daily3": _draw_to_dict(latest_d3) if latest_d3 else None,
        "latest_daily4": _draw_to_dict(latest_d4) if latest_d4 else None,
        "recent_daily3": [_draw_to_dict(d) for d in recent_d3],
        "recent_daily4": [_draw_to_dict(d) for d in recent_d4],
        "freq_data": freq_data,
    })


@app.get("/analysis", response_class=HTMLResponse)
async def analysis_page(request: Request):
    return templates.TemplateResponse(request, "analysis.html", {"active_page": "analysis"})


@app.get("/predictions", response_class=HTMLResponse)
async def predictions_page(request: Request):
    return templates.TemplateResponse(request, "predictions.html", {"active_page": "predictions"})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {})


@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    return templates.TemplateResponse(request, "backtest.html", {"active_page": "backtest"})


@app.get("/performance", response_class=HTMLResponse)
async def performance_page(request: Request):
    return templates.TemplateResponse(request, "performance.html", {"active_page": "performance"})


@app.get("/numbers-played", response_class=HTMLResponse)
async def numbers_played_page(request: Request):
    return templates.TemplateResponse(request, "numbers_played.html", {"active_page": "numbers_played"})


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse(request, "settings.html", {
        "active_page": "settings",
        "config": {
            "database_url": legacy_settings.DATABASE_URL,
            "app_mode": legacy_settings.APP_MODE,
            "api_base": legacy_settings.CA_LOTTERY_API_BASE,
            "daily3_game_id": legacy_settings.DAILY3_GAME_ID,
            "daily4_game_id": legacy_settings.DAILY4_GAME_ID,
            "api_page_size": legacy_settings.API_PAGE_SIZE,
            "fetch_retry_attempts": legacy_settings.FETCH_RETRY_ATTEMPTS,
            "kernel": True,
        },
    })


@app.post("/api/draws/refresh")
async def refresh_draws(db: Session = Depends(get_db)):
    """Fetch latest draws from CA Lottery API. Same contract as legacy."""
    svc = IngestionService(db_session=db)
    d3_count = await svc.ingest_draws("daily3")
    d4_count = await svc.ingest_draws("daily4")
    return {"daily3_new": d3_count, "daily4_new": d4_count}


# ── Entrypoint ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn

    # KERNEL_PORT = parallel-test override; PORT = systemd convention.
    port = int(os.getenv("KERNEL_PORT") or os.getenv("PORT") or "8211")
    _log.info(
        "calottery_kernel_boot",
        port=port,
        legacy_port=8201,
        database=legacy_settings.DATABASE_URL,
    )
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None)
