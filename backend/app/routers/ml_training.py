"""
ML Training and Retraining Endpoints

Provides API endpoints for:
- Manual model retraining
- Online learning
- Retraining status checks
- Model export
- Signal generation
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import asyncio
import io

from app.ai.retraining_service import retraining_service
from app.ai.ai_config import ai_config
from app.ai.model_export import model_export_service
from app.ai.signal_generator import signal_generator
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/ml", tags=["ml-training"])

# Signal caching for instant responses
_signal_cache: Dict[str, Dict] = {}
_cache_ttl = 3  # Cache signals for 3 seconds (ULTRA fast updates)
_dataset_cache: Optional[Dict] = None
_dataset_cache_time: Optional[datetime] = None
_dataset_cache_ttl = 300  # Cache normalized dataset for 5 minutes


def _get_cached_dataset():
    """Get cached normalized dataset or load fresh."""
    global _dataset_cache, _dataset_cache_time
    
    # Check if cache is valid
    if _dataset_cache is not None and _dataset_cache_time is not None:
        age = (datetime.utcnow() - _dataset_cache_time).total_seconds()
        if age < _dataset_cache_ttl:
            logger.debug(f"Using cached dataset (age: {age:.1f}s)")
            return _dataset_cache.copy()
    
    # Load and normalize dataset
    from app.ai.dataset_manager import dataset_manager
    import pandas as pd
    
    try:
        trained_files = list(dataset_manager.trained_dir.glob("*.csv"))
        if not trained_files:
            dataset_files = list(dataset_manager.datasets_dir.glob("*.csv"))
            if not dataset_files:
                logger.warning("No CSV files found in datasets or trained directories")
                return None
            data_file = sorted(dataset_files, key=lambda p: p.stat().st_size, reverse=True)[0]
        else:
            data_file = sorted(trained_files, key=lambda p: p.stat().st_size, reverse=True)[0]
        
        logger.info(f"Loading dataset: {data_file.name} ({data_file.stat().st_size / 1024:.1f} KB)")
        
        # Try to load with multiple encodings
        df = None
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                df = pd.read_csv(data_file, encoding=encoding, nrows=500)  # Limit to 500 rows for MAXIMUM speed
                if encoding != 'utf-8':
                    logger.debug(f"Loaded with {encoding} encoding")
                break
            except (UnicodeDecodeError, pd.errors.EmptyDataError):
                continue
            except Exception as e:
                logger.warning(f"Error loading with {encoding}: {str(e)}")
                continue
        
        if df is None or df.empty:
            logger.error(f"Failed to load dataset: {data_file.name}")
            return None
        
        logger.debug(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Normalize if needed
        has_currency_pair = any('currency_pair' in col.lower() for col in df.columns)
        has_close_price = any('close' in col.lower() and 'price' in col.lower() for col in df.columns)
        
        if not (has_currency_pair and has_close_price):
            logger.debug("Normalizing dataset format...")
            from app.ai.dataset_adapter import dataset_adapter
            try:
                df, _ = dataset_adapter.normalize_dataset(df, data_file.name)
            except Exception as e:
                logger.error(f"Dataset normalization failed: {str(e)}")
                return None
        
        df.columns = [col.lower().strip() for col in df.columns]
        
        if 'date' not in df.columns:
            for date_col in ['timestamp', 'time', 'datetime', 'time serie', 'time_serie']:
                if date_col in df.columns:
                    df['date'] = df[date_col]
                    break
        
        if 'date' not in df.columns:
            logger.error("No date column found in dataset")
            return None
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        if df.empty:
            logger.error("Dataset became empty after date processing")
            return None
        
        logger.debug(f"Processed dataset: {len(df)} rows, currency pairs: {df['currency_pair'].nunique() if 'currency_pair' in df.columns else 0}")
        
        # Cache the normalized dataset
        _dataset_cache = df.copy()
        _dataset_cache_time = datetime.utcnow()
        
        return df.copy()
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}", exc_info=True)
        return None


class RetrainRequest(BaseModel):
    """Request model for manual retraining."""
    timeframe: Optional[str] = Field(
        default="1d",
        description="Timeframe for training (e.g., '1h', '1d', '1w')"
    )
    force: Optional[bool] = Field(
        default=False,
        description="Force retraining even if not needed"
    )


class OnlineLearnRequest(BaseModel):
    """Request model for online learning."""
    timeframe: Optional[str] = Field(
        default="1d",
        description="Timeframe for feature engineering"
    )


@router.post("/retrain")
async def retrain_model(
    request: RetrainRequest,
    background_tasks: BackgroundTasks
):
    """
    Manually trigger model retraining.
    
    Performs full retraining of the model using all available datasets.
    Can be forced to retrain even if accuracy is above threshold.
    
    Args:
        request: RetrainRequest with timeframe and force flag
        background_tasks: Background tasks for async execution
        
    Returns:
        Dictionary with retraining status
    """
    try:
        timeframe = request.timeframe or "1d"
        force = request.force or False
        
        logger.info(f"Manual retraining requested: timeframe={timeframe}, force={force}")
        
        # Perform retraining (can run in background for long operations)
        result = retraining_service.retrain_model(
            timeframe=timeframe,
            force=force
        )
        
        if result.get('success'):
            return {
                "status": "success",
                "message": "Model retraining completed",
                "accuracy": result.get('accuracy'),
                "train_size": result.get('train_size'),
                "test_size": result.get('test_size'),
                "timestamp": result.get('timestamp'),
                "reason": result.get('reason')
            }
        else:
            if result.get('skipped'):
                return {
                    "status": "skipped",
                    "message": "Retraining not needed",
                    "reason": result.get('reason')
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail=result.get('error', 'Retraining failed')
                )
                
    except Exception as e:
        logger.error(f"Failed to retrain model: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


@router.post("/online-learn")
async def online_learn(request: OnlineLearnRequest):
    """
    Perform online/incremental learning.
    
    Updates the model incrementally with new data without full retraining.
    Requires online_learning_enabled=True in config.
    
    Args:
        request: OnlineLearnRequest with timeframe
        
    Returns:
        Dictionary with online learning status
    """
    try:
        timeframe = request.timeframe or "1d"
        
        logger.info(f"Online learning requested: timeframe={timeframe}")
        
        # Perform online learning
        result = retraining_service.online_learn(timeframe=timeframe)
        
        if result.get('success'):
            return {
                "status": "success",
                "message": "Online learning completed",
                "accuracy": result.get('accuracy'),
                "samples_added": result.get('samples_added'),
                "timestamp": result.get('timestamp'),
                "note": result.get('note')
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Online learning failed')
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to perform online learning: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Online learning failed: {str(e)}")


@router.get("/status")
async def get_training_status():
    """
    Get current model training status.
    
    Returns:
        Dictionary with model status, accuracy, and retraining info
    """
    try:
        # Force load model if not already loaded
        if retraining_service.model.model is None:
            logger.info("Model not loaded, attempting to load from disk...")
            loaded = retraining_service.model.load_model()
            if not loaded:
                logger.warning("No trained model found on disk")
        
        # Get current accuracy (try to load from latest model)
        current_accuracy = retraining_service.get_current_accuracy('1d')
        
        # Check if retraining is needed
        should_retrain, reason = retraining_service.should_retrain('1d')
        
        # Get model metadata (ensure it's loaded)
        metadata = retraining_service.model.get_metadata()
        if metadata is None:
            # Try to reload model to get metadata
            if retraining_service.model.model is not None:
                logger.debug("Model loaded but metadata missing, reloading...")
                retraining_service.model.load_model()
                metadata = retraining_service.model.get_metadata()
        
        # Build status with real data (no mock data)
        status = {
            "model_exists": retraining_service.model.model is not None,
            "current_accuracy": current_accuracy,  # Real accuracy or None
            "retrain_threshold": ai_config.retrain_threshold,
            "auto_retrain_enabled": ai_config.auto_retrain_enabled,
            "online_learning_enabled": ai_config.online_learning_enabled,
            "should_retrain": should_retrain,
            "retrain_reason": reason
        }
        
        # Only include model_info if we have real metadata
        if metadata:
            status["model_info"] = {
                "timeframe": metadata.get('timeframe'),
                "last_trained": metadata.get('timestamp'),
                "last_retrained": metadata.get('last_retrained'),
                "train_size": metadata.get('train_size'),
                "test_size": metadata.get('test_size'),
                "sample_size": metadata.get('sample_size'),
                "feature_count": metadata.get('feature_count'),
                "feature_names": metadata.get('feature_names', []),
                "datasets_used": metadata.get('datasets_used', [])
            }
        else:
            # Return null if no model exists (not mock data)
            status["model_info"] = None
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get training status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/performance")
async def get_model_performance():
    """Return chart points for model performance (accuracy/loss), plus a point updated on latest executed trade."""
    try:
        # Load model metadata
        if retraining_service.model.model is None:
            retraining_service.model.load_model()

        metadata = retraining_service.model.get_metadata() or {}
        accuracy = metadata.get('accuracy')
        loss = metadata.get('loss')
        trained_at = metadata.get('timestamp')

        if accuracy is None or not trained_at:
            return {"performance": [], "count": 0}

        # Normalize
        acc_pct = float(accuracy) * 100.0 if float(accuracy) <= 1.0 else float(accuracy)
        loss_val = None
        try:
            if loss is not None:
                loss_val = float(loss)
        except Exception:
            loss_val = None
        if loss_val is None:
            loss_val = max(0.0, min(1.0, 1.0 - (acc_pct / 100.0)))

        points = [{"date": str(trained_at), "accuracy": float(acc_pct), "loss": float(loss_val)}]

        # Ensure at least 2 points so line chart renders reliably.
        if len(points) == 1:
            now_ts = datetime.utcnow().isoformat()
            if now_ts != str(trained_at):
                points.append({"date": now_ts, "accuracy": float(acc_pct), "loss": float(loss_val)})

        # Add an update point on latest trade execution timestamp (so chart updates after trades)
        try:
            from app.database import db
            from app.models import Trade
            from sqlalchemy import select, desc

            async for session in db.get_session():
                result = await session.execute(select(Trade).order_by(desc(Trade.timestamp)).limit(1))
                last_trade = result.scalars().first()
                break
            if last_trade and getattr(last_trade, 'timestamp', None):
                last_ts = last_trade.timestamp.isoformat() if hasattr(last_trade.timestamp, 'isoformat') else str(last_trade.timestamp)
                if last_ts != str(trained_at):
                    points.append({"date": last_ts, "accuracy": float(acc_pct), "loss": float(loss_val)})
        except Exception:
            pass

        return {"performance": points, "count": len(points)}
    except Exception as e:
        logger.error(f"Failed to get model performance: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")


@router.post("/check-retrain")
async def check_and_retrain():
    """
    Check if retraining is needed and perform it if so.
    
    This endpoint checks the current model accuracy against the threshold
    and automatically retrains if needed (if auto_retrain_enabled=True).
    
    Returns:
        Dictionary with check/retrain results
    """
    try:
        result = retraining_service.check_and_retrain_if_needed()
        
        return {
            "status": "success",
            "checked": result.get('checked'),
            "retrained": result.get('retrained'),
            "reason": result.get('reason'),
            "retrain_result": result.get('result') if result.get('retrained') else None
        }
        
    except Exception as e:
        logger.error(f"Failed to check and retrain: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Check and retrain failed: {str(e)}")


@router.post("/export")
async def export_model():
    """
    Export model weights and metadata to CSV files.
    
    Exports:
    - Model weights (coefficients) to CSV
    - Model metadata (accuracy, sample size, timestamps) to CSV
    
    Returns:
        Dictionary with export paths and info
    """
    try:
        logger.info("Model export requested via API")
        
        result = model_export_service.export_all(include_loss=False)
        
        if result.get('success'):
            return {
                "status": "success",
                "message": "Model exported successfully",
                "weights_path": result.get('weights_path'),
                "metadata_path": result.get('metadata_path'),
                "weights_info": result.get('weights_info'),
                "metadata_info": result.get('metadata_info'),
                "timestamp": result.get('timestamp')
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Export failed')
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export model: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/export/xlsx")
async def export_model_xlsx():
    try:
        logger.info("Model XLSX export requested via API")
        result = model_export_service.export_all(include_loss=False)
        if not result.get('success'):
            raise HTTPException(status_code=500, detail=result.get('error', 'Export failed'))

        import pandas as pd
        weights_path = result.get('weights_path')
        metadata_path = result.get('metadata_path')
        if not weights_path or not metadata_path:
            raise HTTPException(status_code=500, detail="Export failed: missing export paths")

        df_weights = pd.read_csv(weights_path)
        df_metadata = pd.read_csv(metadata_path)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df_weights.to_excel(writer, sheet_name='weights', index=False)
            df_metadata.to_excel(writer, sheet_name='metadata', index=False)

        buf.seek(0)

        timeframe = 'unknown'
        try:
            meta = model_export_service.model.get_metadata() or {}
            timeframe = str(meta.get('timeframe') or 'unknown')
        except Exception:
            timeframe = 'unknown'

        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"ai_model_export_{timeframe}_{ts}.xlsx"

        return StreamingResponse(
            buf,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export model XLSX: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export XLSX failed: {str(e)}")


@router.get("/signals")
async def get_ai_signals(
    timeframe: str = Query("1d", description="Timeframe for signal generation"),
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols to analyze"),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
):
    """
    Get AI trading signals for multiple currency pairs.
    
    This endpoint generates BUY/SELL signals for available currency pairs
    using the trained logistic regression model. Results are cached for 30 seconds
    for instant responses.
    
    Args:
        timeframe: Timeframe for signal generation (default: '1d')
        symbols: Comma-separated list of symbols to analyze (default: all available)
        min_confidence: Minimum confidence threshold (default: 0.5)
    
    Returns:
        Dictionary with signals array, count, timeframe, and model info
    """
    import pandas as pd
    start_time = datetime.utcnow()
    raw_timeframe = str(timeframe or '').strip() or '1d'
    requested_timeframe = raw_timeframe if (raw_timeframe.endswith('M') and raw_timeframe[:-1].isdigit()) else raw_timeframe.lower()
    
    try:
        # ULTRA-FAST: Check cache FIRST before any other operations
        cache_key = f"{requested_timeframe}_{symbols}_{min_confidence}"
        
        # Check cache - only use if fresh (age < TTL) AND timeframe matches
        if cache_key in _signal_cache:
            cache_entry = _signal_cache[cache_key]
            cached_timeframe = cache_entry['data'].get('timeframe')
            age = (datetime.utcnow() - cache_entry['timestamp']).total_seconds()
            
            # Only use cache if timeframe matches AND it's fresh (strict check)
            if cached_timeframe == requested_timeframe and age < _cache_ttl:
                logger.debug(f"Returning cached signals INSTANTLY (age: {age:.1f}s < ttl: {_cache_ttl}s)")
                return cache_entry['data']
            else:
                # Cache expired or timeframe changed - force refresh
                logger.debug(f"Cache expired (age: {age:.1f}s >= ttl: {_cache_ttl}s) - generating fresh")
                if cache_key in _signal_cache:
                    del _signal_cache[cache_key]
        
        # Clear any other cache entries with different timeframes (keep cache small)
        keys_to_remove = [k for k in list(_signal_cache.keys()) if not k.startswith(f"{requested_timeframe}_")]
        for key in keys_to_remove:
            del _signal_cache[key]
        
        # Keep cache size small (max 2 entries)
        if len(_signal_cache) > 2:
            oldest_key = min(_signal_cache.keys(), key=lambda k: _signal_cache[k]['timestamp'])
            del _signal_cache[oldest_key]

        model_timeframe = signal_generator.resolve_model_timeframe(requested_timeframe)

        # If a per-timeframe model doesn't exist, fallback to the nearest available model.
        # This prevents 1m/5m/etc. from returning empty when only a 1d model exists.
        if not model_timeframe:
            logger.debug(f"No trained model available for requested timeframe: {requested_timeframe} - returning empty signals")
            return {
                "signals": [],
                "count": 0,
                "timeframe": requested_timeframe,
                "model_timeframe": None,
                "model_available": False,
                "model_accuracy": None
            }
        
        logger.debug(f"Generating fresh signals for timeframe: {requested_timeframe} (model_timeframe={model_timeframe})")
        
        # Get cached dataset (optimized - already cached in memory)
        try:
            df = _get_cached_dataset()
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            return {
                "signals": [],
                "count": 0,
                "timeframe": requested_timeframe,
                "model_timeframe": model_timeframe,
                "model_available": True,
                "model_accuracy": retraining_service.get_current_accuracy(model_timeframe)
            }
        
        if df is None or df.empty:
            logger.debug("No dataset available - returning empty signals")
            return {
                "signals": [],
                "count": 0,
                "timeframe": requested_timeframe,
                "model_timeframe": model_timeframe,
                "model_available": True,
                "model_accuracy": retraining_service.get_current_accuracy(model_timeframe)
            }
        
        # Get unique currency pairs (optimized - skip if column missing)
        if 'currency_pair' not in df.columns:
            logger.debug("Dataset missing 'currency_pair' column - returning empty")
            return {
                "signals": [],
                "count": 0,
                "timeframe": requested_timeframe,
                "model_timeframe": model_timeframe,
                "model_available": True,
                "model_accuracy": retraining_service.get_current_accuracy(model_timeframe)
            }
        
        available_pairs = df['currency_pair'].unique().tolist()
        
        # Filter by requested symbols if provided
        if symbols:
            requested_symbols = [s.strip().upper() for s in symbols.split(',')]
            available_pairs = [pair for pair in available_pairs if pair in requested_symbols]
        
        # Limit to top 1 pair for maximum speed
        available_pairs = available_pairs[:1] if available_pairs else []
        
        if not available_pairs:
            logger.debug("No currency pairs available")
            return {
                "signals": [],
                "count": 0,
                "timeframe": requested_timeframe,
                "model_timeframe": model_timeframe,
                "model_available": True,
                "model_accuracy": retraining_service.get_current_accuracy(model_timeframe)
            }
        
        # Generate signals for each pair (optimized: use only last 50 rows for faster processing)
        signals = []
        signal_id_counter = 1
        
        # Sort pairs for consistent ordering
        available_pairs = sorted(available_pairs)
        
        for currency_pair in available_pairs:
            try:
                # Try to get REAL-TIME data from database/MT5 first
                pair_data = None
                try:
                    from app.services import get_mt5_client
                    from app.database import db
                    from app.models import OHLCV
                    from sqlalchemy import select, desc
                    
                    mt5_client = get_mt5_client()
                    
                    # Try MT5 first for real-time data
                    if mt5_client.is_connected:
                        try:
                            def _parse_tf(tf: str):
                                tf = str(tf or '').strip()
                                if tf.endswith('M') and tf[:-1].isdigit():
                                    return ('month', int(tf[:-1]))
                                low = tf.lower()
                                if low.endswith('m') and low[:-1].isdigit():
                                    return ('minute', int(low[:-1]))
                                if low.endswith('h') and low[:-1].isdigit():
                                    return ('hour', int(low[:-1]))
                                if low.endswith('d') and low[:-1].isdigit():
                                    return ('day', int(low[:-1]))
                                if low.endswith('w') and low[:-1].isdigit():
                                    return ('week', int(low[:-1]))
                                return ('day', 1)

                            def _pick_base(kind: str, n: int):
                                if kind == 'month':
                                    return ('MN1', n, f"{n}MS")
                                if kind == 'minute':
                                    if n in (1, 5, 15, 30):
                                        return (f"M{n}", 1, None)
                                    if n == 45:
                                        return ('M15', 3, '45min')
                                    return ('M1', n, f"{n}min")
                                if kind == 'hour':
                                    if n == 4:
                                        return ('H4', 1, None)
                                    return ('H1', n, f"{n}H")
                                if kind == 'week':
                                    return ('W1', n, f"{n}W")
                                return ('D1', n, f"{n}D" if n != 1 else None)

                            def _resample(df: 'pd.DataFrame', rule: str) -> 'pd.DataFrame':
                                if df is None or df.empty or 'timestamp' not in df.columns:
                                    return df
                                tmp = df.copy().sort_values('timestamp').set_index('timestamp')
                                agg = tmp.resample(rule).agg({
                                    'open': 'first',
                                    'high': 'max',
                                    'low': 'min',
                                    'close': 'last',
                                    'volume': 'sum',
                                })
                                return agg.dropna(subset=['open', 'high', 'low', 'close']).reset_index()

                            kind, n = _parse_tf(requested_timeframe)
                            mt5_tf, mult, resample_rule = _pick_base(kind, n)
                            
                            # Get latest bars from MT5 (REAL-TIME)
                            from datetime import timedelta
                            end_time = datetime.utcnow()
                            # Calculate start time based on timeframe
                            base_limit = max(30, 300 * max(1, int(mult)))
                            if mt5_tf.startswith('M'):
                                minutes = int(mt5_tf[1:]) if len(mt5_tf) > 1 else 1
                                start_time = end_time - timedelta(minutes=minutes * base_limit)
                            elif mt5_tf.startswith('H'):
                                hours = int(mt5_tf[1:]) if len(mt5_tf) > 1 else 1
                                start_time = end_time - timedelta(hours=hours * base_limit)
                            elif mt5_tf == 'D1':
                                start_time = end_time - timedelta(days=base_limit)
                            elif mt5_tf == 'W1':
                                start_time = end_time - timedelta(weeks=base_limit)
                            elif mt5_tf == 'MN1':
                                start_time = end_time - timedelta(days=30 * base_limit)
                            else:
                                start_time = end_time - timedelta(hours=base_limit)
                            
                            ohlcv_df = mt5_client.get_ohlcv(currency_pair, mt5_tf, start=start_time, end=end_time)
                            
                            if not ohlcv_df.empty:
                                if resample_rule:
                                    ohlcv_df = _resample(ohlcv_df, str(resample_rule))
                                # Convert to expected format
                                ohlcv_df = ohlcv_df.rename(columns={'timestamp': 'date', 'close': 'close_price'})
                                ohlcv_df['currency_pair'] = currency_pair
                                pair_data = ohlcv_df.copy()
                                logger.debug(f"Using REAL-TIME MT5 data for {currency_pair}")
                        except Exception as e:
                            logger.debug(f"MT5 data fetch failed for {currency_pair}: {str(e)}")
                    
                    # Fallback to database if MT5 not available
                    if pair_data is None or pair_data.empty:
                        async for session in db.get_session():
                            query = (
                                select(OHLCV)
                                .where(OHLCV.symbol == currency_pair)
                                .order_by(desc(OHLCV.timestamp))
                                .limit(300)
                            )
                            result = await session.execute(query)
                            rows = result.scalars().all()
                            
                            if rows:
                                import pandas as pd
                                data = []
                                for row in rows:
                                    data.append({
                                        'date': row.timestamp,
                                        'close_price': row.close,
                                        'open': row.open,
                                        'high': row.high,
                                        'low': row.low,
                                        'volume': row.volume,
                                        'currency_pair': currency_pair
                                    })
                                pair_data = pd.DataFrame(data)
                                pair_data = pair_data.sort_values('date')
                                logger.debug(f"Using database data for {currency_pair}")
                            break
                    
                except Exception as e:
                    logger.debug(f"Real-time data fetch failed: {str(e)}, using CSV fallback")
                
                # Fallback to CSV data if real-time not available
                if pair_data is None or pair_data.empty:
                    pair_data = df[df['currency_pair'] == currency_pair].copy()
                    if len(pair_data) == 0:
                        continue
                    pair_data = pair_data.sort_values('date').tail(300)
                
                if len(pair_data) < 4:
                    continue
                
                # Generate signal using the model (with timeout protection)
                try:
                    # Run prediction in executor to avoid blocking
                    def generate_signal():
                        return signal_generator.predict_next_period(
                            input_data=pair_data,
                            timeframe=requested_timeframe,
                            min_confidence=min_confidence
                        )
                    
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, generate_signal),
                        timeout=0.8  # 0.8 second timeout per pair (MAXIMUM speed)
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Signal generation timeout for {currency_pair}")
                    continue
                except Exception as e:
                    logger.debug(f"Signal generation error for {currency_pair}: {str(e)}")
                    continue
                
                if result.get('error'):
                    logger.debug(f"Signal generation failed for {currency_pair}: {result['error']}")
                    continue
                
                # Only add signals that are not HOLD
                signal_type = result['signal']
                if signal_type == signal_generator.SIGNAL_HOLD:
                    continue
                
                # Get latest price (REAL-TIME)
                latest_price = float(pair_data['close_price'].iloc[-1])
                confidence = result['confidence'] * 100  # Convert to percentage
                
                # Get REAL-TIME current price from MT5 if available
                try:
                    from app.services import get_mt5_client
                    mt5_client = get_mt5_client()
                    if mt5_client.is_connected:
                        symbol_info = mt5_client.get_symbol_info(currency_pair)
                        if symbol_info and symbol_info.get('bid'):
                            latest_price = float(symbol_info['bid'])  # Use REAL-TIME price
                except:
                    pass  # Use historical price if MT5 not available
                
                # Generate reasoning
                if signal_type == 'BUY':
                    reason = f"Model predicts price increase (confidence: {confidence:.1f}%)"
                elif signal_type == 'SELL':
                    reason = f"Model predicts price decrease (confidence: {confidence:.1f}%)"
                else:
                    reason = f"Low confidence signal (confidence: {confidence:.1f}%)"
                
                # ALWAYS use current timestamp for REAL-TIME signals
                current_timestamp = datetime.utcnow().isoformat()
                
                signals.append({
                    'id': signal_id_counter,
                    'symbol': currency_pair,
                    'signal': signal_type,
                    'confidence': round(confidence, 1),
                    'reason': reason,
                    'price': round(latest_price, 4),
                    'timeframe': requested_timeframe,
                    'model_timeframe': result.get('model_timeframe') or model_timeframe,
                    'timestamp': current_timestamp,  # REAL-TIME timestamp
                    '_cache_key': f"{requested_timeframe}_{currency_pair}_{datetime.utcnow().timestamp()}"  # Force unique key
                })
                signal_id_counter += 1
                
            except Exception as e:
                logger.debug(f"Error generating signal for {currency_pair}: {str(e)}")
                continue
        
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        logger.debug(f"Generated {len(signals)} signals for {timeframe} in {elapsed:.3f}s")
        
        # Build response
        response_data = {
            "signals": signals,
            "count": len(signals),
            "timeframe": requested_timeframe,
            "model_timeframe": model_timeframe,
            "model_available": True,
            "model_accuracy": retraining_service.get_current_accuracy(model_timeframe)
        }
        
        # Cache the results with current timestamp
        _signal_cache[cache_key] = {
            'data': response_data,
            'timestamp': datetime.utcnow()
        }
        
        # Clean old cache entries (keep only last 2)
        if len(_signal_cache) > 2:
            oldest_key = min(_signal_cache.keys(), key=lambda k: _signal_cache[k]['timestamp'])
            del _signal_cache[oldest_key]
        
        logger.info(f"Cached signals for {timeframe} (cache size: {len(_signal_cache)})")
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        logger.error(f"Failed to generate AI signals after {elapsed:.2f}s: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Signal generation failed: {str(e)}")


@router.get("/weights")
async def get_model_weights():
    """
    Get model weights (feature importance) for the trained model.
    
    Returns:
        Dictionary with feature names and their weights
    """
    try:
        # Load model if not already loaded
        if retraining_service.model.model is None:
            if not retraining_service.model.load_model():
                raise HTTPException(
                    status_code=404,
                    detail="No trained model found"
                )
        
        # Get feature names and weights
        feature_names = retraining_service.model.feature_names
        if not feature_names:
            metadata = retraining_service.model.get_metadata()
            if metadata:
                feature_names = metadata.get('feature_names', [])
        
        if not feature_names or retraining_service.model.model is None:
            raise HTTPException(
                status_code=404,
                detail="Model weights not available"
            )
        
        # Get coefficients
        coefficients = retraining_service.model.model.coef_[0]
        
        # Create feature importance list
        feature_importance = [
            {
                'feature': name,
                'weight': float(weight),
                'abs_weight': float(abs(weight))
            }
            for name, weight in zip(feature_names, coefficients)
        ]
        
        # Sort by absolute weight (descending)
        feature_importance.sort(key=lambda x: x['abs_weight'], reverse=True)
        
        return {
            "status": "success",
            "feature_count": len(feature_names),
            "features": feature_importance
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model weights: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get model weights: {str(e)}")

