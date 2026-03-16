#!/usr/bin/env python3
"""
Seed the database with ALL historical CA Lottery draws.

Fetches every Daily 3 and Daily 4 draw ever recorded via the CA Lottery
API and inserts them into the local database.  Safe to re-run — existing
draws are de-duplicated by (game_type, draw_number).

Usage:
    python scripts/seed_historical.py

Environment variables (optional, via .env):
    DATABASE_URL   – defaults to sqlite:///./lottery.db
    CA_LOTTERY_API_BASE – defaults to https://www.calottery.com/api/DrawGameApi
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path so "app" is importable regardless
# of where the script is invoked from.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.database import Base, SessionLocal, engine  # noqa: E402
from app.models import Draw  # noqa: E402  – registers the model with Base
from app.services.ingestion import (  # noqa: E402
    CALotteryAPIClient,
    CALotteryScraper,
    IngestionService,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("seed_historical")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _seed() -> None:
    """Run the full historical ingestion for both games."""

    # Create tables if they don't exist yet.
    logger.info("Ensuring database tables exist …")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        api_client = CALotteryAPIClient()
        scraper = CALotteryScraper()
        svc = IngestionService(db_session=db, api_client=api_client, scraper=scraper)

        total_inserted = 0
        errors: list[str] = []

        for game_type in ("daily3", "daily4"):
            logger.info("=" * 60)
            logger.info("Ingesting %s …", game_type.upper())
            logger.info("=" * 60)
            t0 = time.monotonic()

            try:
                count = await svc.ingest_all_historical(game_type)
                elapsed = time.monotonic() - t0
                total_inserted += count
                logger.info(
                    "%s: inserted %d new draws in %.1fs.",
                    game_type.upper(), count, elapsed,
                )
            except Exception as exc:
                elapsed = time.monotonic() - t0
                msg = f"{game_type}: failed after {elapsed:.1f}s — {exc}"
                logger.exception(msg)
                errors.append(msg)

        # Summary
        logger.info("=" * 60)
        logger.info("SEED COMPLETE")
        logger.info("  Total new draws inserted: %d", total_inserted)

        total_in_db = db.query(Draw).count()
        logger.info("  Total draws now in DB:    %d", total_in_db)

        daily3_count = db.query(Draw).filter(Draw.game_type == "daily3").count()
        daily4_count = db.query(Draw).filter(Draw.game_type == "daily4").count()
        logger.info("    Daily 3: %d", daily3_count)
        logger.info("    Daily 4: %d", daily4_count)

        if errors:
            logger.warning("  Errors encountered:")
            for e in errors:
                logger.warning("    - %s", e)
        else:
            logger.info("  Errors: none")
        logger.info("=" * 60)

    finally:
        db.close()


def main() -> None:
    asyncio.run(_seed())


if __name__ == "__main__":
    main()
