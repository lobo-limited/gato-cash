from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.models import (  # noqa: F401 — ensure all models are registered with Base
    BacktestDetail,
    BacktestRun,
    Draw,
    EnsembleWeight,
    ModelPerformance,
    Prediction,
    StatSnapshot,
    User,
)
from app.config import settings
from app.routers import analysis, auth, backtest, draws, numbers_played, predictions
from app.routers import performance as performance_router
from app.services.ingestion import CALotteryAPIClient, CALotteryScraper, IngestionService
from app.services.scheduler import lottery_scheduler

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    lottery_scheduler.start()
    yield
    lottery_scheduler.shutdown()


app = FastAPI(
    title="CA Lottery Predictor",
    description="Prediction engine for California Daily 3 and Daily 4 lottery games",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

app.include_router(draws.router)
app.include_router(auth.router)
app.include_router(analysis.router)
app.include_router(predictions.router)
app.include_router(performance_router.router)
app.include_router(backtest.router)
app.include_router(numbers_played.router)


def _draw_to_dict(draw: Draw) -> dict:
    """Convert a Draw ORM object to a template-friendly dict."""
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
    # Counts
    d3_count = db.execute(
        select(func.count()).select_from(Draw).where(Draw.game_type == "daily3")
    ).scalar() or 0
    d4_count = db.execute(
        select(func.count()).select_from(Draw).where(Draw.game_type == "daily4")
    ).scalar() or 0

    # Latest draws
    latest_d3 = db.execute(
        select(Draw).where(Draw.game_type == "daily3")
        .order_by(Draw.draw_date.desc(), Draw.draw_number.desc()).limit(1)
    ).scalar_one_or_none()

    latest_d4 = db.execute(
        select(Draw).where(Draw.game_type == "daily4")
        .order_by(Draw.draw_date.desc(), Draw.draw_number.desc()).limit(1)
    ).scalar_one_or_none()

    # Recent draws (last 10)
    recent_d3 = db.execute(
        select(Draw).where(Draw.game_type == "daily3")
        .order_by(Draw.draw_date.desc(), Draw.draw_number.desc()).limit(10)
    ).scalars().all()

    recent_d4 = db.execute(
        select(Draw).where(Draw.game_type == "daily4")
        .order_by(Draw.draw_date.desc(), Draw.draw_number.desc()).limit(10)
    ).scalars().all()

    # Frequency data (last 100 Daily 3 draws)
    freq_draws = db.execute(
        select(Draw).where(Draw.game_type == "daily3")
        .order_by(Draw.draw_date.desc(), Draw.draw_number.desc()).limit(100)
    ).scalars().all()

    digit_counts = Counter()
    for d in freq_draws:
        digit_counts[d.digit_1] += 1
        digit_counts[d.digit_2] += 1
        digit_counts[d.digit_3] += 1

    freq_data = {
        "labels": [str(i) for i in range(10)],
        "counts": [digit_counts.get(i, 0) for i in range(10)],
    }

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
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
    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "active_page": "analysis",
    })


@app.get("/predictions", response_class=HTMLResponse)
async def predictions_page(request: Request):
    return templates.TemplateResponse("predictions.html", {
        "request": request,
        "active_page": "predictions",
    })


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    return templates.TemplateResponse("backtest.html", {
        "request": request,
        "active_page": "backtest",
    })


@app.get("/performance", response_class=HTMLResponse)
async def performance_page(request: Request):
    return templates.TemplateResponse("performance.html", {
        "request": request,
        "active_page": "performance",
    })


@app.get("/numbers-played", response_class=HTMLResponse)
async def numbers_played_page(request: Request):
    return templates.TemplateResponse("numbers_played.html", {
        "request": request,
        "active_page": "numbers_played",
    })


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    # Mask the database URL for display (hide file path details in production).
    db_url = settings.DATABASE_URL
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "active_page": "settings",
        "config": {
            "database_url": db_url,
            "app_mode": settings.APP_MODE,
            "api_base": settings.CA_LOTTERY_API_BASE,
            "daily3_game_id": settings.DAILY3_GAME_ID,
            "daily4_game_id": settings.DAILY4_GAME_ID,
            "api_page_size": settings.API_PAGE_SIZE,
            "fetch_retry_attempts": settings.FETCH_RETRY_ATTEMPTS,
        },
    })


@app.post("/api/draws/refresh")
async def refresh_draws(db: Session = Depends(get_db)):
    """Fetch latest draws from CA Lottery API."""
    svc = IngestionService(db_session=db)
    d3_count = await svc.ingest_draws("daily3")
    d4_count = await svc.ingest_draws("daily4")
    return {"daily3_new": d3_count, "daily4_new": d4_count}
