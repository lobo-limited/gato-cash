from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.draw import Draw
from app.schemas.draw import DrawCreate, DrawList, DrawRead

router = APIRouter(prefix="/api/draws", tags=["draws"])


@router.get("/", response_model=DrawList)
def list_draws(
    game_type: str = Query("daily3", pattern="^(daily3|daily4)$"),
    draw_time: str | None = Query(None, pattern="^(midday|evening)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    query = select(Draw).where(Draw.game_type == game_type)
    count_query = select(func.count()).select_from(Draw).where(Draw.game_type == game_type)

    if draw_time:
        query = query.where(Draw.draw_time == draw_time)
        count_query = count_query.where(Draw.draw_time == draw_time)

    total = db.execute(count_query).scalar() or 0

    query = (
        query.order_by(Draw.draw_date.desc(), Draw.draw_number.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    draws = db.execute(query).scalars().all()

    return DrawList(
        draws=[DrawRead.model_validate(d) for d in draws],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{draw_id}", response_model=DrawRead)
def get_draw(draw_id: int, db: Session = Depends(get_db)):
    draw = db.get(Draw, draw_id)
    if not draw:
        raise HTTPException(status_code=404, detail="Draw not found")
    return DrawRead.model_validate(draw)


@router.post("/", response_model=DrawRead, status_code=201)
def create_draw(draw_in: DrawCreate, db: Session = Depends(get_db)):
    existing = db.execute(
        select(Draw).where(
            Draw.game_type == draw_in.game_type,
            Draw.draw_number == draw_in.draw_number,
        )
    ).scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Draw #{draw_in.draw_number} for {draw_in.game_type} already exists",
        )

    draw = Draw(**draw_in.model_dump())
    db.add(draw)
    db.commit()
    db.refresh(draw)
    return DrawRead.model_validate(draw)


@router.get("/latest/{game_type}", response_model=DrawRead)
def get_latest_draw(
    game_type: str,
    draw_time: str | None = Query(None, pattern="^(midday|evening)$"),
    db: Session = Depends(get_db),
):
    query = (
        select(Draw)
        .where(Draw.game_type == game_type)
        .order_by(Draw.draw_date.desc(), Draw.draw_number.desc())
    )
    if draw_time:
        query = query.where(Draw.draw_time == draw_time)

    draw = db.execute(query).scalars().first()
    if not draw:
        raise HTTPException(status_code=404, detail="No draws found")
    return DrawRead.model_validate(draw)
