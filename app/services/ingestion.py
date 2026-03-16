"""
CA Lottery data ingestion service.

Fetches draw results from the CA Lottery API and (as a fallback) scrapes
the CA Lottery website.  Normalises every draw into a dict that maps 1-to-1
onto the ``draws`` table defined in ``app.models.draw``.

API details (discovered 2026-03-16 by hitting the live endpoints):

  URL pattern
    https://www.calottery.com/api/DrawGameApi/DrawGamePastDrawResults/{gameId}/{page}/{count}

  Game IDs
    9  = Daily 3   (3 digits, two draws per day – midday & evening)
    14 = Daily 4   (4 digits, one draw per day – evening only)

  Response shape (JSON)
    {
      "DrawGameId": int,
      "TotalPreviousDraws": int,          # total draws across all pages
      "PreviousDraws": [                   # list of draw objects
        {
          "DrawNumber": int,
          "DrawDate": "YYYY-MM-DDTHH:MM:SS",
          "WinningNumbers": {
            "1": {"Number": "d", "IsSpecial": false, "Name": null},
            ...
          },
          "Prizes": {
            "1": {"PrizeTypeDescription": str, "Count": int, "Amount": int, ...},
            ...
          },
          "RaceTime": null                 # always null – not useful
        }
      ]
    }

  Midday vs. evening (Daily 3 only)
    The API returns two draws per day with the same DrawDate.  Within each
    pair the *higher* DrawNumber is the evening draw and the *lower* is
    midday.  This was confirmed against the CA Lottery website which labels
    draws explicitly.  Daily 4 has only one draw per day (evening).
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import defaultdict
from datetime import datetime
from typing import Any

import httpx
from bs4 import BeautifulSoup
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.draw import Draw

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAME_CONFIG: dict[str, dict[str, Any]] = {
    "daily3": {
        "game_id": settings.DAILY3_GAME_ID,   # 9
        "digit_count": 3,
        "draws_per_day": 2,                    # midday + evening
        "url_slug": "daily-3",
    },
    "daily4": {
        "game_id": settings.DAILY4_GAME_ID,   # 14
        "digit_count": 4,
        "draws_per_day": 1,                    # evening only
        "url_slug": "daily-4",
    },
}

# Map game_type to its game_id and vice-versa for quick look-ups.
GAME_ID_TO_TYPE: dict[int, str] = {v["game_id"]: k for k, v in GAME_CONFIG.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_draw(raw: dict[str, Any], game_type: str) -> dict[str, Any]:
    """Convert a single raw API draw object into a standardised dict.

    Returns a dict with exactly the fields needed for the ``Draw`` model:
      draw_number, draw_date, draw_time, digits, digit_1..digit_N,
      straight_prize, box_prize, straight_winners, box_winners
    """
    cfg = GAME_CONFIG[game_type]
    digit_count = cfg["digit_count"]

    # --- Digits ---
    winning = raw.get("WinningNumbers", {})
    digits: list[int] = []
    for i in range(1, digit_count + 1):
        entry = winning.get(str(i), {})
        num_str = entry.get("Number", "")
        if num_str == "" or not num_str.isdigit():
            raise ValueError(
                f"Draw #{raw.get('DrawNumber')}: invalid digit at position {i}: {num_str!r}"
            )
        d = int(num_str)
        if d < 0 or d > 9:
            raise ValueError(
                f"Draw #{raw.get('DrawNumber')}: digit {d} out of range 0-9"
            )
        digits.append(d)

    if len(digits) != digit_count:
        raise ValueError(
            f"Draw #{raw.get('DrawNumber')}: expected {digit_count} digits, got {len(digits)}"
        )

    # --- Prizes ---
    prizes_raw = raw.get("Prizes", {})
    straight_prize: float | None = None
    box_prize: float | None = None
    straight_winners: int | None = None
    box_winners: int | None = None

    for _key, prize in prizes_raw.items():
        desc = (prize.get("PrizeTypeDescription") or "").lower()
        amount = prize.get("Amount")
        count = prize.get("Count")

        if desc == "straight":
            straight_prize = float(amount) if amount is not None else None
            straight_winners = int(count) if count is not None else None
        elif desc in ("box", "box only"):
            # Both "Box" and "Box Only" represent the box prize.  The API
            # sometimes uses one, sometimes the other.  We take the first
            # one we encounter (they carry the same per-ticket amount for
            # a pure box bet).
            if box_prize is None:
                box_prize = float(amount) if amount is not None else None
                box_winners = int(count) if count is not None else None

    # --- Date ---
    draw_date_str = raw.get("DrawDate", "")
    try:
        draw_date = datetime.fromisoformat(draw_date_str)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Draw #{raw.get('DrawNumber')}: cannot parse DrawDate {draw_date_str!r}"
        ) from exc

    return {
        "draw_number": int(raw["DrawNumber"]),
        "draw_date": draw_date,
        "draw_time": None,  # resolved later by _assign_draw_times
        "digits": digits,
        "straight_prize": straight_prize,
        "box_prize": box_prize,
        "straight_winners": straight_winners,
        "box_winners": box_winners,
    }


def _assign_draw_times(draws: list[dict[str, Any]], game_type: str) -> None:
    """Assign ``draw_time`` ('midday' or 'evening') in-place.

    Rules (derived from live data and the CA Lottery website):
    - Daily 4: always 'evening' (one draw per day).
    - Daily 3: two draws per day.  Within a pair sharing the same date the
      higher DrawNumber is 'evening', the lower is 'midday'.
    """
    if game_type == "daily4":
        for d in draws:
            d["draw_time"] = "evening"
        return

    # Daily 3 – group by date, then assign by draw_number ordering.
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in draws:
        date_key = d["draw_date"].strftime("%Y-%m-%d")
        by_date[date_key].append(d)

    for _date_key, group in by_date.items():
        group.sort(key=lambda x: x["draw_number"])
        if len(group) == 2:
            group[0]["draw_time"] = "midday"
            group[1]["draw_time"] = "evening"
        elif len(group) == 1:
            # Edge case: only one draw fetched for this date (e.g. evening
            # draw hasn't happened yet, or single-draw day).  Fall back to
            # odd/even heuristic: odd = evening based on observed data where
            # draw numbers started at 1 and midday was added later.
            group[0]["draw_time"] = "evening" if group[0]["draw_number"] % 2 == 1 else "midday"
        else:
            # More than 2 draws on a single date – shouldn't happen, but
            # handle gracefully with the odd/even heuristic.
            for item in group:
                item["draw_time"] = "evening" if item["draw_number"] % 2 == 1 else "midday"


# ---------------------------------------------------------------------------
# CALotteryAPIClient
# ---------------------------------------------------------------------------

class CALotteryAPIClient:
    """Async client for the CA Lottery Draw Game API."""

    def __init__(
        self,
        base_url: str = settings.CA_LOTTERY_API_BASE,
        retry_attempts: int = settings.FETCH_RETRY_ATTEMPTS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.retry_attempts = retry_attempts

    # -- low-level --------------------------------------------------------

    async def fetch_draws(
        self, game_id: int, page: int, count: int = settings.API_PAGE_SIZE
    ) -> dict[str, Any]:
        """Fetch one page of draw results from the API.

        Returns the full JSON response dict (including ``TotalPreviousDraws``
        and ``PreviousDraws``).

        Retries with exponential back-off on transient HTTP errors.
        """
        url = f"{self.base_url}/DrawGamePastDrawResults/{game_id}/{page}/{count}"
        last_exc: Exception | None = None

        for attempt in range(1, self.retry_attempts + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0),
                    follow_redirects=True,
                ) as client:
                    logger.debug("GET %s (attempt %d/%d)", url, attempt, self.retry_attempts)
                    resp = await client.get(url)
                    resp.raise_for_status()
                    return resp.json()
            except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < self.retry_attempts:
                    wait = min(2 ** attempt, 30)  # 2, 4, 8 … capped at 30s
                    logger.warning(
                        "Attempt %d/%d for %s failed (%s). Retrying in %ds …",
                        attempt, self.retry_attempts, url, exc, wait,
                    )
                    await asyncio.sleep(wait)

        raise RuntimeError(
            f"Failed to fetch {url} after {self.retry_attempts} attempts"
        ) from last_exc

    # -- high-level -------------------------------------------------------

    async def fetch_all_draws(self, game_id: int) -> list[dict[str, Any]]:
        """Paginate through *all* historical draws for a game.

        Returns a list of standardised draw dicts (see ``_parse_draw``).
        """
        game_type = GAME_ID_TO_TYPE.get(game_id)
        if game_type is None:
            raise ValueError(f"Unknown game_id: {game_id}")

        page_size = settings.API_PAGE_SIZE
        # Fetch page 1 to learn total count.
        first_page = await self.fetch_draws(game_id, page=1, count=page_size)
        total = first_page.get("TotalPreviousDraws", 0)
        all_raw: list[dict] = list(first_page.get("PreviousDraws", []))
        logger.info(
            "Game %s (id=%d): %d total draws, page_size=%d",
            game_type, game_id, total, page_size,
        )

        total_pages = math.ceil(total / page_size) if total else 1

        # Fetch remaining pages sequentially (be polite to the API).
        for page in range(2, total_pages + 1):
            logger.info("Fetching page %d/%d for %s …", page, total_pages, game_type)
            data = await self.fetch_draws(game_id, page=page, count=page_size)
            draws_page = data.get("PreviousDraws", [])
            if not draws_page:
                logger.info("Page %d returned 0 draws – stopping.", page)
                break
            all_raw.extend(draws_page)
            # Small delay between pages to avoid hammering the server.
            await asyncio.sleep(0.25)

        logger.info("Fetched %d raw draws for %s.", len(all_raw), game_type)

        parsed = [_parse_draw(d, game_type) for d in all_raw]
        _assign_draw_times(parsed, game_type)
        return parsed

    async def fetch_recent_draws(
        self, game_id: int, pages: int = 1
    ) -> list[dict[str, Any]]:
        """Fetch the most recent draw results (first *pages* pages).

        Returns standardised draw dicts.
        """
        game_type = GAME_ID_TO_TYPE.get(game_id)
        if game_type is None:
            raise ValueError(f"Unknown game_id: {game_id}")

        all_raw: list[dict] = []
        for page in range(1, pages + 1):
            data = await self.fetch_draws(game_id, page=page, count=settings.API_PAGE_SIZE)
            draws_page = data.get("PreviousDraws", [])
            if not draws_page:
                break
            all_raw.extend(draws_page)
            if page < pages:
                await asyncio.sleep(0.15)

        parsed = [_parse_draw(d, game_type) for d in all_raw]
        _assign_draw_times(parsed, game_type)
        return parsed


# ---------------------------------------------------------------------------
# CALotteryScraper  (fallback)
# ---------------------------------------------------------------------------

class CALotteryScraper:
    """Fallback scraper that pulls results from the CA Lottery website HTML.

    The website at ``https://www.calottery.com/draw-games/{slug}`` renders
    the latest draw results as server-side HTML.  This scraper extracts the
    most recent draws displayed on the page.

    NOTE: The website loads historical results via the same JSON API that
    ``CALotteryAPIClient`` uses, so this scraper can only retrieve the draws
    that are statically rendered on the page (typically the 1-2 most recent).
    It exists as a last-resort fallback.
    """

    SITE_BASE = "https://www.calottery.com"

    async def scrape_draws(self, game_type: str) -> list[dict[str, Any]]:
        """Scrape draw results from the CA Lottery website.

        Returns a list of standardised draw dicts (same shape as the API
        client produces).
        """
        cfg = GAME_CONFIG.get(game_type)
        if cfg is None:
            raise ValueError(f"Unknown game_type: {game_type!r}")

        url = f"{self.SITE_BASE}/draw-games/{cfg['url_slug']}"
        logger.info("Scraping %s from %s", game_type, url)

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                follow_redirects=True,
                headers={"User-Agent": "CalotteryPredict/1.0"},
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.error("Scraper failed for %s: %s", game_type, exc)
            raise

        soup = BeautifulSoup(resp.text, "html.parser")
        return self._parse_html(soup, game_type)

    def _parse_html(
        self, soup: BeautifulSoup, game_type: str
    ) -> list[dict[str, Any]]:
        """Extract draw results from the parsed HTML.

        The CA Lottery pages render results in sections with class names
        containing the draw info.  The structure observed (2026-03-16):
        - Draw results appear in sections with draw date/time headers
        - Winning numbers are rendered in list items inside a number container
        - Prize tables follow each result section

        This parser is intentionally defensive: if the HTML structure changes,
        it returns an empty list and logs a warning rather than crashing.
        """
        cfg = GAME_CONFIG[game_type]
        digit_count = cfg["digit_count"]
        results: list[dict[str, Any]] = []

        try:
            # Look for draw result sections.  The page uses
            # "past-draw-result" or "draw-result" class patterns.
            result_sections = soup.select(
                ".draw-results-detail, .past-results, [class*='draw-result']"
            )
            if not result_sections:
                # Fallback: try to find winning number containers anywhere.
                result_sections = [soup]

            for section in result_sections:
                # Find winning number elements.  They appear as list items
                # or spans with a single digit inside a numbers container.
                number_els = section.select(
                    ".winning-numbers li, .draw-number, "
                    "[class*='winning'] .number, [class*='number-ball']"
                )

                digits: list[int] = []
                for el in number_els:
                    text = el.get_text(strip=True)
                    if text.isdigit() and len(text) == 1:
                        digits.append(int(text))

                if len(digits) < digit_count:
                    continue

                # Take exactly the number of digits we expect.
                digits = digits[:digit_count]

                # Try to extract draw number from text like "#20923".
                draw_number: int | None = None
                draw_header = section.find(
                    string=lambda s: s and "#" in s
                )
                if draw_header:
                    match = re.search(r"#(\d+)", str(draw_header))
                    if match:
                        draw_number = int(match.group(1))

                # Try to extract draw time (midday/evening) from text.
                draw_time: str | None = None
                section_text = section.get_text(separator=" ").lower()
                if "evening" in section_text:
                    draw_time = "evening"
                elif "midday" in section_text:
                    draw_time = "midday"

                # Try to extract date.
                draw_date: datetime | None = None
                date_match = re.search(
                    r"(\w{3})/(\w{3})\s+(\d{1,2}),?\s+(\d{4})", section_text
                )
                if date_match:
                    try:
                        date_str = f"{date_match.group(2)} {date_match.group(3)} {date_match.group(4)}"
                        draw_date = datetime.strptime(date_str, "%b %d %Y")
                    except ValueError:
                        pass

                # Try to extract prize data from a table.
                straight_prize: float | None = None
                box_prize: float | None = None
                straight_winners: int | None = None
                box_winners: int | None = None

                rows = section.select("tr, .prize-row")
                for row in rows:
                    cells = row.select("td, .prize-cell")
                    row_text = row.get_text(separator="|").lower()
                    if "straight" in row_text and "box" not in row_text:
                        amounts = re.findall(r"\$?([\d,]+(?:\.\d+)?)", row_text)
                        if len(amounts) >= 2:
                            straight_winners = int(amounts[0].replace(",", ""))
                            straight_prize = float(amounts[1].replace(",", ""))
                    elif "box" in row_text and "straight" not in row_text:
                        amounts = re.findall(r"\$?([\d,]+(?:\.\d+)?)", row_text)
                        if len(amounts) >= 2:
                            box_winners = int(amounts[0].replace(",", ""))
                            box_prize = float(amounts[1].replace(",", ""))

                if draw_number is not None and draw_date is not None:
                    results.append({
                        "draw_number": draw_number,
                        "draw_date": draw_date,
                        "draw_time": draw_time or ("evening" if game_type == "daily4" else "evening"),
                        "digits": digits,
                        "straight_prize": straight_prize,
                        "box_prize": box_prize,
                        "straight_winners": straight_winners,
                        "box_winners": box_winners,
                    })

        except Exception:
            logger.exception("HTML parsing failed for %s", game_type)

        if not results:
            logger.warning(
                "Scraper extracted 0 results for %s. The site HTML may have changed.",
                game_type,
            )

        return results


# ---------------------------------------------------------------------------
# IngestionService
# ---------------------------------------------------------------------------

class IngestionService:
    """Orchestrates fetching draw data and persisting it to the database.

    Typical usage::

        async with get_db_session() as db:
            svc = IngestionService(db)
            count = await svc.ingest_draws("daily3")
            print(f"Inserted {count} new draws.")
    """

    def __init__(
        self,
        db_session: Session,
        api_client: CALotteryAPIClient | None = None,
        scraper: CALotteryScraper | None = None,
    ) -> None:
        self.db = db_session
        self.api_client = api_client or CALotteryAPIClient()
        self.scraper = scraper or CALotteryScraper()

    # ------------------------------------------------------------------

    async def ingest_draws(self, game_type: str) -> int:
        """Fetch the most recent draws and insert any new ones into the DB.

        Tries the API first; falls back to scraping if the API is down.

        Returns the number of newly inserted draws.
        """
        cfg = GAME_CONFIG.get(game_type)
        if cfg is None:
            raise ValueError(f"Unknown game_type: {game_type!r}")

        draws: list[dict[str, Any]] = []

        # Try API first.
        try:
            logger.info("Ingesting recent %s draws via API …", game_type)
            draws = await self.api_client.fetch_recent_draws(
                game_id=cfg["game_id"], pages=1
            )
            logger.info("API returned %d draws for %s.", len(draws), game_type)
        except Exception:
            logger.exception(
                "API fetch failed for %s – falling back to scraper.", game_type
            )
            try:
                draws = await self.scraper.scrape_draws(game_type)
                logger.info("Scraper returned %d draws for %s.", len(draws), game_type)
            except Exception:
                logger.exception("Scraper also failed for %s.", game_type)
                return 0

        return self._upsert_draws(draws, game_type)

    async def ingest_all_historical(self, game_type: str) -> int:
        """Fetch ALL historical draws and insert new ones into the DB.

        This can take a while for games with tens of thousands of draws.

        Returns the number of newly inserted draws.
        """
        cfg = GAME_CONFIG.get(game_type)
        if cfg is None:
            raise ValueError(f"Unknown game_type: {game_type!r}")

        logger.info("Starting full historical ingestion for %s …", game_type)
        draws = await self.api_client.fetch_all_draws(game_id=cfg["game_id"])
        logger.info("Fetched %d total draws for %s.", len(draws), game_type)

        return self._upsert_draws(draws, game_type)

    # ------------------------------------------------------------------

    def _upsert_draws(self, draws: list[dict[str, Any]], game_type: str) -> int:
        """De-duplicate and insert draws into the database.

        Draws are de-duplicated on ``(game_type, draw_number)`` — the same
        unique constraint enforced by the DB schema.

        Returns the count of newly inserted rows.
        """
        if not draws:
            return 0

        # Fetch existing draw numbers in one query to avoid N+1.
        draw_numbers = [d["draw_number"] for d in draws]
        existing_stmt = (
            select(Draw.draw_number)
            .where(Draw.game_type == game_type)
            .where(Draw.draw_number.in_(draw_numbers))
        )
        existing_numbers: set[int] = set(
            self.db.execute(existing_stmt).scalars().all()
        )

        new_count = 0
        for d in draws:
            if d["draw_number"] in existing_numbers:
                continue

            digits = d["digits"]
            draw = Draw(
                game_type=game_type,
                draw_number=d["draw_number"],
                draw_date=d["draw_date"],
                draw_time=d["draw_time"],
                digit_1=digits[0],
                digit_2=digits[1],
                digit_3=digits[2],
                digit_4=digits[3] if len(digits) > 3 else None,
                straight_prize=d.get("straight_prize"),
                box_prize=d.get("box_prize"),
                straight_winners=d.get("straight_winners"),
                box_winners=d.get("box_winners"),
            )
            self.db.add(draw)
            new_count += 1

        if new_count:
            self.db.commit()
            logger.info(
                "Inserted %d new %s draws (%d duplicates skipped).",
                new_count, game_type, len(draws) - new_count,
            )
        else:
            logger.info("No new %s draws to insert.", game_type)

        return new_count
