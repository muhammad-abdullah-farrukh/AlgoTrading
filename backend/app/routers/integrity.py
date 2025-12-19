"""Dataset integrity endpoints."""
from fastapi import APIRouter, HTTPException
from typing import Optional
from app.services.integrity import integrity_service
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/integrity", tags=["integrity"])


@router.post("/enforce/{symbol}")
async def enforce_rolling_window(symbol: str):
    """
    Manually enforce rolling window limit for a symbol.
    
    Deletes oldest rows if the symbol exceeds the maximum row limit.
    """
    try:
        deleted, error = await integrity_service.enforce_rolling_window(symbol)
        
        if error:
            raise HTTPException(status_code=500, detail=error)
        
        return {
            "symbol": symbol,
            "rows_deleted": deleted,
            "message": f"Deleted {deleted} oldest rows" if deleted > 0 else "No rows deleted (within limit)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enforce rolling window: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to enforce rolling window: {str(e)}")


@router.get("/check/{symbol}")
async def check_integrity(symbol: str):
    """
    Check dataset integrity for a symbol.
    
    Returns comprehensive integrity check including:
    - Total rows and rows per source
    - Date range
    - Duplicate detection
    - Null value detection
    - Limit compliance
    - Integrity score (0-100)
    """
    try:
        result = await integrity_service.check_integrity(symbol)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Integrity check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Integrity check failed: {str(e)}")


@router.post("/metadata/{symbol}")
async def update_metadata(symbol: str):
    """
    Manually update dataset metadata for a symbol.
    
    Metadata includes row counts, date ranges, and source information.
    """
    try:
        error = await integrity_service.update_metadata(symbol)
        
        if error:
            raise HTTPException(status_code=500, detail=error)
        
        return {
            "symbol": symbol,
            "message": "Metadata updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update metadata: {str(e)}")


@router.get("/stats")
async def get_dataset_stats(symbol: Optional[str] = None):
    """
    Get dataset statistics.
    
    Returns overall statistics or statistics for a specific symbol.
    """
    try:
        result = await integrity_service.get_dataset_stats(symbol)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get dataset stats: {str(e)}")
