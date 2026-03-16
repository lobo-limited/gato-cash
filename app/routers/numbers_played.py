"""Numbers Played API — track user picks and measure accuracy."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import cast, func, select, and_, Date
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.draw import Draw
from app.models.numbers_played import NumbersPlayed

router = APIRouter(prefix="/api/numbers-played", tags=["numbers-played"])

RETENTION_DAYS = 548  # 18 months


# ---------- Schemas ----------

class NumbersPlayedCreate(BaseModel):
    game_type: str = Field(pattern="^(daily3|daily4)$")
    play_type: str = Field(pattern="^(straight|box|straight_box|combo)$")
    digit_1: int = Field(ge=0, le=9)
    digit_2: int = Field(ge=0, le=9)
    digit_3: int = Field(ge=0, le=9)
    digit_4: int | None = Field(None, ge=0, le=9)
    target_draw_date: str  # YYYY-MM-DD
    target_draw_time: str = Field(pattern="^(midday|evening)$")
    amount_wagered: float = 1.0
    notes: str | None = None


class NumbersPlayedOut(BaseModel):
    id: int
    game_type: str
    play_type: str
    digits: list[int]
    target_draw_date: str
    target_draw_time: str
    amount_wagered: float
    notes: str | None
    straight_hit: bool | None
    box_hit: bool | None
    amount_won: float | None
    scored_at: str | None
    created_at: str


class AccuracyStats(BaseModel):
    total_plays: int
    total_scored: int
    straight_hits: int
    box_hits: int
    straight_hit_rate: float
    box_hit_rate: float
    total_wagered: float
    total_won: float
    net_profit: float
    roi_percent: float
    best_streak: int
    current_streak: int
    by_play_type: dict
    by_game_type: dict


# ---------- Helpers ----------

def _to_out(np: NumbersPlayed) -> dict:
    return {
        "id": np.id,
        "game_type": np.game_type,
        "play_type": np.play_type,
        "digits": np.digits,
        "target_draw_date": np.target_draw_date.strftime("%Y-%m-%d"),
        "target_draw_time": np.target_draw_time,
        "amount_wagered": np.amount_wagered,
        "notes": np.notes,
        "straight_hit": np.straight_hit,
        "box_hit": np.box_hit,
        "amount_won": np.amount_won,
        "scored_at": np.scored_at.isoformat() if np.scored_at else None,
        "created_at": np.created_at.isoformat(),
    }


def _score_entry(entry: NumbersPlayed, draw: Draw) -> None:
    """Score a played entry against an actual draw."""
    played = entry.digits
    actual = [draw.digit_1, draw.digit_2, draw.digit_3]
    if draw.digit_4 is not None:
        actual.append(draw.digit_4)

    entry.straight_hit = played == actual
    entry.box_hit = sorted(played) == sorted(actual)
    entry.matched_draw_id = draw.id
    entry.scored_at = datetime.now(timezone.utc)

    # Calculate winnings
    if entry.straight_hit and entry.play_type in ("straight", "straight_box", "combo"):
        prize = draw.straight_prize or (500.0 if entry.game_type == "daily3" else 5000.0)
        if entry.play_type == "straight_box":
            entry.amount_won = prize * 0.5
        else:
            entry.amount_won = prize
    elif entry.box_hit and entry.play_type in ("box", "straight_box", "combo"):
        prize = draw.box_prize or (80.0 if entry.game_type == "daily3" else 200.0)
        if entry.play_type == "straight_box":
            entry.amount_won = prize * 0.5
        else:
            entry.amount_won = prize
    else:
        entry.amount_won = 0.0


# ---------- Endpoints ----------

@router.post("/", status_code=201)
def add_play(play: NumbersPlayedCreate, db: Session = Depends(get_db)):
    """Record numbers you played."""
    if play.game_type == "daily4" and play.digit_4 is None:
        raise HTTPException(400, "digit_4 required for Daily 4")
    if play.game_type == "daily3" and play.digit_4 is not None:
        raise HTTPException(400, "digit_4 must be null for Daily 3")

    entry = NumbersPlayed(
        game_type=play.game_type,
        play_type=play.play_type,
        digit_1=play.digit_1,
        digit_2=play.digit_2,
        digit_3=play.digit_3,
        digit_4=play.digit_4,
        target_draw_date=datetime.strptime(play.target_draw_date, "%Y-%m-%d"),
        target_draw_time=play.target_draw_time,
        amount_wagered=play.amount_wagered,
        notes=play.notes,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    # Try to score immediately if the draw already exists
    draw = db.execute(
        select(Draw).where(
            Draw.game_type == play.game_type,
            cast(Draw.draw_date, Date) == cast(entry.target_draw_date, Date),
            Draw.draw_time == play.target_draw_time,
        ).order_by(Draw.draw_number.desc()).limit(1)
    ).scalar_one_or_none()

    if draw:
        _score_entry(entry, draw)
        db.commit()

    return _to_out(entry)


@router.get("/")
def list_plays(
    game_type: str | None = Query(None, pattern="^(daily3|daily4)$"),
    scored_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List played numbers with optional filters."""
    q = select(NumbersPlayed).order_by(NumbersPlayed.target_draw_date.desc(), NumbersPlayed.id.desc())
    count_q = select(func.count()).select_from(NumbersPlayed)

    if game_type:
        q = q.where(NumbersPlayed.game_type == game_type)
        count_q = count_q.where(NumbersPlayed.game_type == game_type)
    if scored_only:
        q = q.where(NumbersPlayed.scored_at.isnot(None))
        count_q = count_q.where(NumbersPlayed.scored_at.isnot(None))

    total = db.execute(count_q).scalar() or 0
    entries = db.execute(q.offset(offset).limit(limit)).scalars().all()

    return {
        "total": total,
        "plays": [_to_out(e) for e in entries],
    }


@router.delete("/{play_id}")
def delete_play(play_id: int, db: Session = Depends(get_db)):
    """Delete a played entry."""
    entry = db.get(NumbersPlayed, play_id)
    if not entry:
        raise HTTPException(404, "Entry not found")
    db.delete(entry)
    db.commit()
    return {"deleted": play_id}


@router.post("/score-all")
def score_all(db: Session = Depends(get_db)):
    """Score all unscored entries against available draw results."""
    unscored = db.execute(
        select(NumbersPlayed).where(NumbersPlayed.scored_at.is_(None))
    ).scalars().all()

    scored_count = 0
    for entry in unscored:
        draw = db.execute(
            select(Draw).where(
                Draw.game_type == entry.game_type,
                cast(Draw.draw_date, Date) == cast(entry.target_draw_date, Date),
                Draw.draw_time == entry.target_draw_time,
            ).order_by(Draw.draw_number.desc()).limit(1)
        ).scalar_one_or_none()

        if draw:
            _score_entry(entry, draw)
            scored_count += 1

    if scored_count:
        db.commit()
    return {"scored": scored_count, "remaining_unscored": len(unscored) - scored_count}


@router.get("/accuracy")
def get_accuracy(
    game_type: str | None = Query(None, pattern="^(daily3|daily4)$"),
    days: int | None = Query(None, ge=1, le=RETENTION_DAYS),
    db: Session = Depends(get_db),
):
    """Get accuracy statistics for played numbers."""
    q = select(NumbersPlayed).where(NumbersPlayed.scored_at.isnot(None))
    if game_type:
        q = q.where(NumbersPlayed.game_type == game_type)
    if days:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        q = q.where(NumbersPlayed.created_at >= cutoff)

    entries = db.execute(q.order_by(NumbersPlayed.target_draw_date.asc())).scalars().all()

    if not entries:
        return AccuracyStats(
            total_plays=0, total_scored=0, straight_hits=0, box_hits=0,
            straight_hit_rate=0, box_hit_rate=0, total_wagered=0, total_won=0,
            net_profit=0, roi_percent=0, best_streak=0, current_streak=0,
            by_play_type={}, by_game_type={},
        )

    straight_hits = sum(1 for e in entries if e.straight_hit)
    box_hits = sum(1 for e in entries if e.box_hit)
    total_wagered = sum(e.amount_wagered for e in entries)
    total_won = sum(e.amount_won or 0 for e in entries)
    n = len(entries)

    # Streaks (any hit = straight or box)
    best_streak = 0
    current_streak = 0
    streak = 0
    for e in entries:
        if e.straight_hit or e.box_hit:
            streak += 1
            best_streak = max(best_streak, streak)
        else:
            streak = 0
    # Current streak from the end
    for e in reversed(entries):
        if e.straight_hit or e.box_hit:
            current_streak += 1
        else:
            break

    # Breakdown by play type
    by_play_type = {}
    for pt in ("straight", "box", "straight_box", "combo"):
        pt_entries = [e for e in entries if e.play_type == pt]
        if pt_entries:
            pt_s = sum(1 for e in pt_entries if e.straight_hit)
            pt_b = sum(1 for e in pt_entries if e.box_hit)
            pt_n = len(pt_entries)
            by_play_type[pt] = {
                "plays": pt_n,
                "straight_hits": pt_s,
                "box_hits": pt_b,
                "straight_rate": round(pt_s / pt_n * 100, 2) if pt_n else 0,
                "box_rate": round(pt_b / pt_n * 100, 2) if pt_n else 0,
                "wagered": sum(e.amount_wagered for e in pt_entries),
                "won": sum(e.amount_won or 0 for e in pt_entries),
            }

    # Breakdown by game type
    by_game_type = {}
    for gt in ("daily3", "daily4"):
        gt_entries = [e for e in entries if e.game_type == gt]
        if gt_entries:
            gt_s = sum(1 for e in gt_entries if e.straight_hit)
            gt_b = sum(1 for e in gt_entries if e.box_hit)
            gt_n = len(gt_entries)
            by_game_type[gt] = {
                "plays": gt_n,
                "straight_hits": gt_s,
                "box_hits": gt_b,
                "straight_rate": round(gt_s / gt_n * 100, 2) if gt_n else 0,
                "box_rate": round(gt_b / gt_n * 100, 2) if gt_n else 0,
                "wagered": sum(e.amount_wagered for e in gt_entries),
                "won": sum(e.amount_won or 0 for e in gt_entries),
            }

    return AccuracyStats(
        total_plays=n,
        total_scored=n,
        straight_hits=straight_hits,
        box_hits=box_hits,
        straight_hit_rate=round(straight_hits / n * 100, 2) if n else 0,
        box_hit_rate=round(box_hits / n * 100, 2) if n else 0,
        total_wagered=round(total_wagered, 2),
        total_won=round(total_won, 2),
        net_profit=round(total_won - total_wagered, 2),
        roi_percent=round((total_won - total_wagered) / total_wagered * 100, 2) if total_wagered else 0,
        best_streak=best_streak,
        current_streak=current_streak,
        by_play_type=by_play_type,
        by_game_type=by_game_type,
    )


@router.post("/cleanup")
def cleanup_old_entries(db: Session = Depends(get_db)):
    """Remove entries older than 18 months (548 days)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
    old = db.execute(
        select(NumbersPlayed).where(NumbersPlayed.created_at < cutoff)
    ).scalars().all()
    count = len(old)
    for entry in old:
        db.delete(entry)
    if count:
        db.commit()
    return {"deleted": count, "retention_days": RETENTION_DAYS}
