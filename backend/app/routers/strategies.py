"""Strategy management endpoints (DB-backed, no dummy data)."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.database import db
from app.models import Strategy
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/strategies", tags=["strategies"])


class StrategyOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    strategy_type: str
    enabled: bool
    parameters: Optional[str] = None
    performance: float
    trades_count: int

    class Config:
        from_attributes = True


class StrategyCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    strategy_type: str = Field(..., description="technical|ai|custom")
    enabled: bool = False
    parameters: Optional[Dict[str, Any]] = None


class StrategyUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1)
    description: Optional[str] = None
    strategy_type: Optional[str] = None
    enabled: Optional[bool] = None
    parameters: Optional[Dict[str, Any]] = None
    performance: Optional[float] = None
    trades_count: Optional[int] = None


@router.get("", response_model=List[StrategyOut])
async def list_strategies(enabled: Optional[bool] = None):
    try:
        async for session in db.get_session():
            q = select(Strategy)
            if enabled is not None:
                q = q.where(Strategy.enabled == bool(enabled))
            q = q.order_by(Strategy.id.asc())
            res = await session.execute(q)
            rows = res.scalars().all()
            return rows
    except Exception as e:
        logger.error(f"Failed to list strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list strategies: {str(e)}")


@router.post("", response_model=StrategyOut)
async def create_strategy(payload: StrategyCreate):
    import json

    try:
        async for session in db.get_session():
            s = Strategy(
                name=payload.name,
                description=payload.description,
                strategy_type=payload.strategy_type,
                enabled=bool(payload.enabled),
                parameters=json.dumps(payload.parameters or {}),
                performance=0.0,
                trades_count=0,
            )
            session.add(s)
            await session.commit()
            await session.refresh(s)
            return s
    except Exception as e:
        logger.error(f"Failed to create strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create strategy: {str(e)}")


@router.put("/{strategy_id}", response_model=StrategyOut)
async def update_strategy(strategy_id: int, payload: StrategyUpdate):
    import json

    try:
        async for session in db.get_session():
            res = await session.execute(select(Strategy).where(Strategy.id == strategy_id))
            s = res.scalar_one_or_none()
            if s is None:
                raise HTTPException(status_code=404, detail="Strategy not found")

            if payload.name is not None:
                s.name = payload.name
            if payload.description is not None:
                s.description = payload.description
            if payload.strategy_type is not None:
                s.strategy_type = payload.strategy_type
            if payload.enabled is not None:
                s.enabled = bool(payload.enabled)
            if payload.parameters is not None:
                s.parameters = json.dumps(payload.parameters)
            if payload.performance is not None:
                s.performance = float(payload.performance)
            if payload.trades_count is not None:
                s.trades_count = int(payload.trades_count)

            await session.commit()
            await session.refresh(s)
            return s
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update strategy: {str(e)}")


@router.delete("/{strategy_id}")
async def delete_strategy(strategy_id: int):
    try:
        async for session in db.get_session():
            res = await session.execute(select(Strategy).where(Strategy.id == strategy_id))
            s = res.scalar_one_or_none()
            if s is None:
                raise HTTPException(status_code=404, detail="Strategy not found")

            await session.delete(s)
            await session.commit()
            return {"status": "success", "deleted_id": strategy_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete strategy: {str(e)}")
