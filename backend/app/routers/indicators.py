"""Technical indicators endpoints."""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from app.services.indicators import indicators_service
from app.utils.logging import get_logger
import pandas as pd

from app.services import get_mt5_client
from app.services.mock_mt5 import MockMT5Client

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


@router.get("/ohlcv")
async def get_ohlcv(
    symbol: str = Query(..., description="Trading symbol"),
    timeframe: str = Query("1d", description="Timeframe (e.g., 1m,5m,1h,1d)"),
    limit: int = Query(200, ge=1, le=10000, description="Maximum number of bars")
):
    """Get OHLCV data for charts (real data only).

    Preference order:
    1) Real MT5 connection (returns latest bars)
    2) Database OHLCV table
    """
    try:
        symbol = symbol.strip().upper()
        timeframe = timeframe.strip()

        def _parse_requested_timeframe(tf: str) -> Dict[str, object]:
            tf = str(tf or '').strip()
            if not tf:
                return {"raw": "1d", "kind": "day", "n": 1}

            # Months are expressed in the UI as '1M', '3M', ... (uppercase M)
            if tf.endswith('M') and tf[:-1].isdigit():
                return {"raw": tf, "kind": "month", "n": int(tf[:-1])}

            low = tf.lower()
            if low.endswith('m') and low[:-1].isdigit():
                return {"raw": low, "kind": "minute", "n": int(low[:-1])}
            if low.endswith('h') and low[:-1].isdigit():
                return {"raw": low, "kind": "hour", "n": int(low[:-1])}
            if low.endswith('d') and low[:-1].isdigit():
                return {"raw": low, "kind": "day", "n": int(low[:-1])}
            if low.endswith('w') and low[:-1].isdigit():
                return {"raw": low, "kind": "week", "n": int(low[:-1])}

            return {"raw": low, "kind": "day", "n": 1}

        def _pick_mt5_base_tf(parsed: Dict[str, object]) -> Dict[str, object]:
            kind = parsed["kind"]
            n = int(parsed["n"])  # type: ignore[arg-type]

            # Supported by our MT5 client wrapper
            if kind == 'month':
                # Use D1 + resample for months to avoid extremely long MN1 lookbacks.
                # This keeps the endpoint responsive and prevents empty responses.
                return {"mt5_tf": "D1", "mult": max(1, n * 30), "resample": f"{n}MS"}

            if kind == 'minute':
                if n in (1, 5, 15, 30):
                    return {"mt5_tf": f"M{n}", "mult": 1, "resample": None}
                if n == 45:
                    return {"mt5_tf": "M15", "mult": 3, "resample": "45min"}
                # Generic minute aggregation
                return {"mt5_tf": "M1", "mult": n, "resample": f"{n}min"}

            if kind == 'hour':
                if n == 4:
                    return {"mt5_tf": "H4", "mult": 1, "resample": None}
                return {"mt5_tf": "H1", "mult": n, "resample": f"{n}H"}

            if kind == 'week':
                return {"mt5_tf": "W1", "mult": n, "resample": f"{n}W"}

            # day (default)
            return {"mt5_tf": "D1", "mult": n, "resample": f"{n}D" if n != 1 else None}

        def _resample_ohlcv(df: 'pd.DataFrame', rule: str) -> 'pd.DataFrame':
            if df is None or df.empty:
                return df
            if 'timestamp' not in df.columns:
                return df

            tmp = df.copy()
            tmp = tmp.sort_values('timestamp')
            tmp = tmp.set_index('timestamp')

            agg = tmp.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            })

            agg = agg.dropna(subset=['open', 'high', 'low', 'close']).reset_index()
            return agg

        # Try MT5 first if real connected
        mt5_client = get_mt5_client()
        if mt5_client.is_connected and not isinstance(mt5_client, MockMT5Client):
            parsed = _parse_requested_timeframe(timeframe)
            base = _pick_mt5_base_tf(parsed)
            mt5_tf = str(base['mt5_tf'])
            mult = int(base['mult'])
            resample_rule = base.get('resample')

            from datetime import datetime, timedelta
            end = datetime.utcnow()
            max_lookback = timedelta(days=3650)
            # Approximate window for requested limit (fetch enough base bars if we need aggregation)
            base_limit = max(1, int(limit) * max(1, mult) + 10)
            if mt5_tf.startswith('M'):
                minutes = int(mt5_tf[1:]) if len(mt5_tf) > 1 else 1
                start = end - timedelta(minutes=minutes * base_limit)
            elif mt5_tf.startswith('H'):
                hours = int(mt5_tf[1:]) if len(mt5_tf) > 1 else 1
                start = end - timedelta(hours=hours * base_limit)
            elif mt5_tf == 'D1':
                start = end - timedelta(days=base_limit)
            elif mt5_tf == 'W1':
                start = end - timedelta(weeks=base_limit)
            elif mt5_tf == 'MN1':
                start = end - timedelta(days=30 * base_limit)
            else:
                start = end - timedelta(days=base_limit)

            # Prevent massive lookbacks (e.g., 12M with limit=200 -> ~200 years).
            if start < end - max_lookback:
                start = end - max_lookback

            df = mt5_client.get_ohlcv(symbol, mt5_tf, start=start, end=end)
            if df is not None and not df.empty:
                if resample_rule:
                    df = _resample_ohlcv(df, str(resample_rule))
                df = df.tail(limit)
                return {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'source': 'mt5',
                    'count': int(len(df)),
                    'ohlcv': [
                        {
                            'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume']),
                        }
                        for _, row in df.iterrows()
                    ],
                }

        # Fallback: database
        parsed = _parse_requested_timeframe(timeframe)
        kind = str(parsed.get('kind') or 'day')
        n = int(parsed.get('n') or 1)

        # DB OHLCV data is often daily (scraped). Resampling only makes sense when
        # the requested timeframe is COARSER than the DB resolution.
        resample_rule = None
        if kind == 'week':
            resample_rule = f"{n}W"
            base_limit = int(limit) * max(1, 7 * n) + 10
        elif kind == 'month':
            resample_rule = f"{n}MS"
            base_limit = int(limit) * max(1, 31 * n) + 10
        elif kind == 'day' and n > 1:
            resample_rule = f"{n}D"
            base_limit = int(limit) * n + 10
        else:
            base_limit = int(limit) + 10

        df = await indicators_service.get_ohlcv_data(symbol, limit=base_limit)
        if df.empty:
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'source': 'database',
                'count': 0,
                'ohlcv': [],
            }

        effective_timeframe = timeframe
        note: Optional[str] = None

        try:
            if len(df) >= 3 and 'timestamp' in df.columns:
                diffs = (
                    pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                    .sort_values()
                    .diff()
                    .dropna()
                )
                if not diffs.empty:
                    base_seconds = float(diffs.dt.total_seconds().median())
                    req_seconds = 86400.0
                    if kind == 'minute':
                        req_seconds = float(max(1, n) * 60)
                    elif kind == 'hour':
                        req_seconds = float(max(1, n) * 3600)
                    elif kind == 'day':
                        req_seconds = float(max(1, n) * 86400)
                    elif kind == 'week':
                        req_seconds = float(max(1, n) * 7 * 86400)
                    elif kind == 'month':
                        req_seconds = float(max(1, n) * 30 * 86400)

                    # If requested timeframe is FINER than DB resolution, don't resample.
                    # Returning DB data avoids empty charts.
                    if req_seconds < base_seconds:
                        resample_rule = None
                        effective_timeframe = '1d' if base_seconds >= 86400 else timeframe
                        note = (
                            f"Database OHLCV resolution (~{int(base_seconds)}s) is coarser than requested "
                            f"timeframe (~{int(req_seconds)}s). Returning base data without resampling."
                        )
        except Exception:
            # Keep endpoint resilient; fallback continues without note.
            pass

        if resample_rule:
            df = _resample_ohlcv(df, str(resample_rule))
        df = df.tail(limit)

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'effective_timeframe': effective_timeframe,
            'source': 'database',
            'count': int(len(df)),
            'note': note,
            'ohlcv': [
                {
                    'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                    'open': float(o),
                    'high': float(h),
                    'low': float(l),
                    'close': float(c),
                    'volume': float(v),
                }
                for ts, o, h, l, c, v in zip(
                    df['timestamp'].tolist(),
                    df['open'].tolist(),
                    df['high'].tolist(),
                    df['low'].tolist(),
                    df['close'].tolist(),
                    df['volume'].tolist(),
                )
            ],
        }
    except Exception as e:
        logger.error(f"Failed to fetch OHLCV for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch OHLCV: {str(e)}")


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
