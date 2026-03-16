"""Background scheduler for automated lottery prediction pipeline.

Manages periodic jobs for data fetching, prediction scoring, prediction
generation, and weight recalibration using APScheduler.

Dependency chain:
    1. Fetch draws every 30 minutes
    2. Score predictions 5 minutes after each successful fetch
    3. Generate predictions daily at 11 AM and 5 PM
    4. Recalibrate weights weekly on Sunday at 3 AM
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from apscheduler.triggers.date import DateTrigger

from app.database import SessionLocal
from app.models.draw import Draw
from app.services.calibration import CalibrationService
from app.services.ingestion import IngestionService
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)


class LotteryScheduler:
    """Manages background jobs for the lottery prediction pipeline.

    Job dependency chain:
        1. Fetch draws (every 30 minutes)
        2. Score predictions (5 minutes after fetch completes)
        3. Generate predictions (daily at 11 AM and 5 PM)
        4. Recalibrate weights (weekly on Sunday at 3 AM)
    """

    def __init__(self) -> None:
        self.scheduler = AsyncIOScheduler()
        self._calibration = CalibrationService()
        self._prediction_service = PredictionService()
        self._last_fetch_time: datetime | None = None
        self._last_score_time: datetime | None = None
        self._last_generate_time: datetime | None = None
        self._last_calibrate_time: datetime | None = None
        self._job_errors: list[dict] = []

    @property
    def is_running(self) -> bool:
        return self.scheduler.running

    def start(self) -> None:
        """Register all jobs and start the scheduler."""
        # Job 1: Fetch new draws every 30 minutes.
        self.scheduler.add_job(
            self._fetch_draws,
            "interval",
            minutes=30,
            id="fetch_draws",
            name="Fetch Draws",
            replace_existing=True,
        )

        # Job 2: Score predictions is triggered 5 minutes after each
        # successful fetch (see _fetch_draws). We also add a fallback
        # interval schedule in case the chained trigger is missed.
        self.scheduler.add_job(
            self._score_predictions,
            "interval",
            minutes=35,
            id="score_predictions",
            name="Score Predictions",
            replace_existing=True,
        )

        # Job 3: Generate predictions daily at 11 AM and 5 PM.
        self.scheduler.add_job(
            self._generate_predictions,
            "cron",
            hour="11,17",
            id="generate_predictions",
            name="Generate Predictions",
            replace_existing=True,
        )

        # Job 4: Recalibrate weights weekly on Sunday at 3 AM.
        self.scheduler.add_job(
            self._recalibrate,
            "cron",
            day_of_week="sun",
            hour=3,
            id="recalibrate",
            name="Recalibrate Weights",
            replace_existing=True,
        )

        # Listen for job events.
        self.scheduler.add_listener(
            self._on_job_executed, EVENT_JOB_EXECUTED,
        )
        self.scheduler.add_listener(
            self._on_job_error, EVENT_JOB_ERROR,
        )

        self.scheduler.start()
        logger.info("LotteryScheduler started with 4 jobs.")

    def shutdown(self) -> None:
        """Gracefully shut down the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("LotteryScheduler shut down.")

    def get_status(self) -> dict:
        """Return scheduler status including job next-run times."""
        jobs_info = []
        if self.scheduler.running:
            for job in self.scheduler.get_jobs():
                jobs_info.append({
                    "id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                })

        return {
            "running": self.scheduler.running,
            "jobs": jobs_info,
            "last_fetch": self._last_fetch_time.isoformat() if self._last_fetch_time else None,
            "last_score": self._last_score_time.isoformat() if self._last_score_time else None,
            "last_generate": self._last_generate_time.isoformat() if self._last_generate_time else None,
            "last_calibrate": self._last_calibrate_time.isoformat() if self._last_calibrate_time else None,
            "recent_errors": self._job_errors[-10:],  # Last 10 errors
        }

    # ------------------------------------------------------------------ #
    # Job implementations                                                  #
    # ------------------------------------------------------------------ #

    async def _fetch_draws(self) -> None:
        """Fetch latest draws from CA Lottery for both game types.

        On success, schedules a one-shot scoring job to run 5 minutes later,
        ensuring predictions are scored against the newly fetched draws.
        """
        logger.info("Scheduler: fetching draws...")
        db = SessionLocal()
        try:
            svc = IngestionService(db_session=db)
            d3 = await svc.ingest_draws("daily3")
            d4 = await svc.ingest_draws("daily4")
            self._last_fetch_time = datetime.utcnow()
            logger.info(
                "Scheduler: fetched %d Daily 3 and %d Daily 4 draws.", d3, d4,
            )

            # Chain: schedule scoring 5 minutes after this fetch completes.
            if d3 > 0 or d4 > 0:
                score_time = datetime.utcnow() + timedelta(minutes=5)
                self.scheduler.add_job(
                    self._score_predictions,
                    trigger=DateTrigger(run_date=score_time),
                    id=f"score_after_fetch_{score_time.strftime('%H%M%S')}",
                    name="Score After Fetch",
                    replace_existing=False,
                    misfire_grace_time=300,
                )
                logger.info(
                    "Scheduler: chained scoring job at %s", score_time.isoformat(),
                )
        except Exception:
            logger.exception("Scheduler: fetch_draws failed.")
            raise
        finally:
            db.close()

    async def _score_predictions(self) -> None:
        """Score unscored predictions against recently fetched draws."""
        logger.info("Scheduler: scoring predictions...")
        db = SessionLocal()
        try:
            from sqlalchemy import select as sa_select
            # Score for both game types against recent draws.
            total_scored = 0
            for game_type in ("daily3", "daily4"):
                stmt = (
                    sa_select(Draw)
                    .where(Draw.game_type == game_type)
                    .order_by(Draw.draw_date.desc(), Draw.draw_number.desc())
                    .limit(20)
                )
                recent_draws = db.execute(stmt).scalars().all()
                for draw in recent_draws:
                    count = self._prediction_service.score_predictions(db, draw)
                    total_scored += count

            self._last_score_time = datetime.utcnow()
            logger.info("Scheduler: scored %d predictions.", total_scored)
        except Exception:
            logger.exception("Scheduler: score_predictions failed.")
            raise
        finally:
            db.close()

    async def _generate_predictions(self) -> None:
        """Generate predictions for upcoming draws."""
        logger.info("Scheduler: generating predictions...")
        db = SessionLocal()
        try:
            total = 0
            for game_type in ("daily3", "daily4"):
                draw_times = ["midday", "evening"] if game_type == "daily3" else ["evening"]
                for draw_time in draw_times:
                    preds = self._prediction_service.generate_predictions(
                        db, game_type, draw_time,
                    )
                    total += len(preds)

            self._last_generate_time = datetime.utcnow()
            logger.info("Scheduler: generated %d predictions.", total)
        except Exception:
            logger.exception("Scheduler: generate_predictions failed.")
            raise
        finally:
            db.close()

    async def _recalibrate(self) -> None:
        """Recalibrate ensemble weights for both game types."""
        logger.info("Scheduler: recalibrating weights...")
        db = SessionLocal()
        try:
            for game_type in ("daily3", "daily4"):
                self._calibration.recalibrate_weights(db, game_type)
                alerts = self._calibration.detect_degradation(db, game_type)
                if alerts:
                    logger.warning(
                        "Degradation alerts for %s: %s", game_type, alerts,
                    )
            self._last_calibrate_time = datetime.utcnow()
            logger.info("Scheduler: recalibration complete.")
        except Exception:
            logger.exception("Scheduler: recalibrate failed.")
            raise
        finally:
            db.close()

    # ------------------------------------------------------------------ #
    # Event listeners                                                      #
    # ------------------------------------------------------------------ #

    def _on_job_executed(self, event) -> None:
        """Log successful job execution."""
        logger.debug("Job %s executed successfully.", event.job_id)

    def _on_job_error(self, event) -> None:
        """Record job errors."""
        error_info = {
            "job_id": event.job_id,
            "time": datetime.utcnow().isoformat(),
            "error": str(event.exception) if event.exception else "Unknown error",
        }
        self._job_errors.append(error_info)
        # Keep only last 50 errors.
        if len(self._job_errors) > 50:
            self._job_errors = self._job_errors[-50:]
        logger.error("Job %s failed: %s", event.job_id, event.exception)


# Module-level singleton so the scheduler can be shared across the app.
lottery_scheduler = LotteryScheduler()
