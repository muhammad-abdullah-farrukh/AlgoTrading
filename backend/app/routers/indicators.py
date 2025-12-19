"""Technical indicators endpoints."""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from app.services.indicators import indicators_service
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/indicators", tags=["indicators"])


class IndicatorParams(BaseModel):
    """Parameters for indicator calculations."""
    ema: Optional[Dict[str, int]] = Field(None, description="EMA parameters: {period: 20}")
    rsi: Optional[Dict[str, int]] = Field(None, description="RSI parameters: {period: 14}")
    macd: Optional[Dict[str, int]] = Field(None, description="MACD parameters: {fast: 12, slow: 26, signal: 9}")
    volume: Optional[Dict[str, str]] = Field(None, description="Volume aggregation: {period: '1D'}")


class IndicatorsRequest(BaseModel):
    """Request model for calculating indicators."""
    symbol: str = Field(..., description="Trading symbol")
    indicators: List[str] = Field(..., description="List of indicators: ema, rsi, macd, volume")
    params: Optional[IndicatorParams] = Field(None, description="Optional parameters for indicators")
    use_cache: bool = Field(True, description="Whether to use cached results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "indicators": ["ema", "rsi", "macd"],
                "params": {
                    "ema": {"period": 20},
                    "rsi": {"period": 14},
                    "macd": {"fast": 12, "slow": 26, "signal": 9}
                }
            }
        }


@router.post("/calculate")
async def calculate_indicators(request: IndicatorsRequest):
    """
    Calculate technical indicators for a symbol.
    
    Supported indicators:
    - **EMA**: Exponential Moving Average
    - **RSI**: Relative Strength Index (0-100)
    - **MACD**: Moving Average Convergence Divergence
    - **Volume**: Volume aggregation
    
    All calculations are deterministic and match standard implementations.
    """
    try:
        # Validate indicators
        valid_indicators = {'ema', 'rsi', 'macd', 'volume'}
        invalid = set(request.indicators) - valid_indicators
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid indicators: {invalid}. Valid indicators: {valid_indicators}"
            )
        
        # Convert params to dict format
        params_dict = {}
        if request.params:
            if request.params.ema:
                params_dict['ema'] = request.params.ema
            if request.params.rsi:
                params_dict['rsi'] = request.params.rsi
            if request.params.macd:
                params_dict['macd'] = request.params.macd
            if request.params.volume:
                params_dict['volume'] = request.params.volume
        
        result = await indicators_service.calculate_indicators(
            request.symbol,
            request.indicators,
            params_dict if params_dict else None,
            request.use_cache
        )
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to calculate indicators: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate indicators: {str(e)}")


@router.get("/calculate/{symbol}")
async def calculate_indicators_get(
    symbol: str,
    indicators: str = Query(..., description="Comma-separated list: ema,rsi,macd,volume"),
    ema_period: Optional[int] = Query(20, description="EMA period"),
    rsi_period: Optional[int] = Query(14, description="RSI period"),
    macd_fast: Optional[int] = Query(12, description="MACD fast period"),
    macd_slow: Optional[int] = Query(26, description="MACD slow period"),
    macd_signal: Optional[int] = Query(9, description="MACD signal period"),
    volume_period: Optional[str] = Query("1D", description="Volume aggregation period"),
    use_cache: bool = Query(True, description="Use cached results")
):
    """
    Calculate indicators via GET request (simpler interface).
    
    Example: /api/indicators/calculate/AAPL?indicators=ema,rsi&ema_period=20&rsi_period=14
    """
    try:
        indicator_list = [i.strip().lower() for i in indicators.split(',')]
        
        # Build params dict
        params_dict = {}
        if 'ema' in indicator_list:
            params_dict['ema'] = {'period': ema_period}
        if 'rsi' in indicator_list:
            params_dict['rsi'] = {'period': rsi_period}
        if 'macd' in indicator_list:
            params_dict['macd'] = {'fast': macd_fast, 'slow': macd_slow, 'signal': macd_signal}
        if 'volume' in indicator_list:
            params_dict['volume'] = {'period': volume_period}
        
        result = await indicators_service.calculate_indicators(
            symbol,
            indicator_list,
            params_dict if params_dict else None,
            use_cache
        )
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to calculate indicators: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate indicators: {str(e)}")


@router.delete("/cache/{symbol}")
async def clear_cache(symbol: str):
    """
    Clear indicator cache for a symbol.
    """
    try:
        count = indicators_service.clear_cache(symbol)
        return {
            "symbol": symbol,
            "cache_entries_cleared": count,
            "message": f"Cleared {count} cache entries for {symbol}"
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.delete("/cache")
async def clear_all_cache():
    """
    Clear all indicator cache.
    """
    try:
        count = indicators_service.clear_cache()
        return {
            "cache_entries_cleared": count,
            "message": f"Cleared all {count} cache entries"
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/info")
async def get_indicator_info():
    """
    Get information about available indicators and their parameters.
    """
    return {
        "indicators": {
            "ema": {
                "name": "Exponential Moving Average",
                "description": "Weighted moving average that gives more weight to recent prices",
                "parameters": {
                    "period": {
                        "type": "int",
                        "default": 20,
                        "description": "EMA period (number of periods)"
                    }
                },
                "output": "Series of EMA values",
                "range": "Price range"
            },
            "rsi": {
                "name": "Relative Strength Index",
                "description": "Momentum oscillator that measures speed and magnitude of price changes",
                "parameters": {
                    "period": {
                        "type": "int",
                        "default": 14,
                        "description": "RSI period (number of periods)"
                    }
                },
                "output": "Series of RSI values (0-100)",
                "range": "0-100 (typically 30-70 for normal range)"
            },
            "macd": {
                "name": "Moving Average Convergence Divergence",
                "description": "Trend-following momentum indicator",
                "parameters": {
                    "fast": {
                        "type": "int",
                        "default": 12,
                        "description": "Fast EMA period"
                    },
                    "slow": {
                        "type": "int",
                        "default": 26,
                        "description": "Slow EMA period"
                    },
                    "signal": {
                        "type": "int",
                        "default": 9,
                        "description": "Signal line EMA period"
                    }
                },
                "output": {
                    "macd_line": "MACD line values",
                    "signal_line": "Signal line values",
                    "histogram": "Histogram values (MACD - Signal)"
                }
            },
            "volume": {
                "name": "Volume Aggregation",
                "description": "Aggregate volume over time periods",
                "parameters": {
                    "period": {
                        "type": "string",
                        "default": "1D",
                        "description": "Aggregation period (e.g., '1D', '1H', '1W')"
                    }
                },
                "output": "Aggregated volume values"
            }
        },
        "cache": {
            "enabled": True,
            "ttl_seconds": indicators_service._cache_ttl,
            "description": "Results are cached for 5 minutes to improve performance"
        }
    }
