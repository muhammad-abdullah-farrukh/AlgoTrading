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
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import asyncio
from app.ai.retraining_service import retraining_service
from app.ai.ai_config import ai_config
from app.ai.model_export import model_export_service
from app.ai.signal_generator import signal_generator
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/ml", tags=["ml-training"])

# Signal caching for instant responses
_signal_cache: Dict[str, Dict] = {}
_cache_ttl = 30  # Cache signals for 30 seconds
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
                df = pd.read_csv(data_file, encoding=encoding, nrows=10000)  # Limit to 10k rows for speed
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
    
    try:
        # Check cache first (but always respect timeframe changes)
        cache_key = f"{timeframe}_{symbols}_{min_confidence}"
        
        # Always invalidate cache if timeframe changed (clear all entries with different timeframe)
        if cache_key in _signal_cache:
            cache_entry = _signal_cache[cache_key]
            cached_timeframe = cache_entry['data'].get('timeframe')
            age = (datetime.utcnow() - cache_entry['timestamp']).total_seconds()
            
            # Only use cache if timeframe matches AND it's fresh
            if cached_timeframe == timeframe and age < _cache_ttl:
                logger.debug(f"Returning cached signals (age: {age:.1f}s, timeframe: {timeframe})")
                return cache_entry['data']
            else:
                # Invalidate - timeframe changed or cache expired
                logger.debug(f"Cache invalidated (timeframe: {cached_timeframe} -> {timeframe} or age: {age:.1f}s)")
                del _signal_cache[cache_key]
        
        # Also clear any other cache entries with different timeframes
        keys_to_remove = [k for k in _signal_cache.keys() if not k.startswith(f"{timeframe}_")]
        for key in keys_to_remove:
            del _signal_cache[key]
            logger.debug(f"Removed stale cache entry: {key}")
        
        logger.info(f"AI signals requested for timeframe: {timeframe}, symbols: {symbols}")
        
        # Check if model is available (fast check)
        if not signal_generator.is_model_available(timeframe):
            logger.warning(f"No trained model found for timeframe: {timeframe}")
            raise HTTPException(
                status_code=404,
                detail=f"No trained model found for timeframe: {timeframe}"
            )
        
        # Get cached or fresh dataset (with timeout protection)
        logger.debug("Loading dataset...")
        try:
            df = _get_cached_dataset()
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load dataset: {str(e)}"
            )
        
        if df is None or df.empty:
            logger.warning("No forex data available for signal generation")
            raise HTTPException(
                status_code=404,
                detail="No forex data available for signal generation"
            )
        
        # Get unique currency pairs
        if 'currency_pair' not in df.columns:
            logger.error("Dataset missing 'currency_pair' column")
            raise HTTPException(
                status_code=500,
                detail="Failed to identify currency pairs in dataset"
            )
        
        available_pairs = df['currency_pair'].unique().tolist()
        logger.debug(f"Found {len(available_pairs)} currency pairs in dataset")
        
        # Filter by requested symbols if provided
        if symbols:
            requested_symbols = [s.strip().upper() for s in symbols.split(',')]
            available_pairs = [pair for pair in available_pairs if pair in requested_symbols]
            logger.debug(f"Filtered to {len(available_pairs)} requested pairs")
        
        # Limit to top 6 pairs for faster performance (reduced from 10)
        available_pairs = available_pairs[:6]
        
        if not available_pairs:
            logger.warning("No currency pairs available after filtering")
            return {
                "signals": [],
                "count": 0,
                "timeframe": timeframe,
                "model_available": True,
                "model_accuracy": retraining_service.get_current_accuracy(timeframe)
            }
        
        logger.info(f"Generating signals for {len(available_pairs)} currency pairs: {available_pairs}")
        
        # Generate signals for each pair (optimized: use only last 50 rows for faster processing)
        signals = []
        signal_id_counter = 1
        
        # Sort pairs for consistent ordering
        available_pairs = sorted(available_pairs)
        
        for currency_pair in available_pairs:
            try:
                # Get data for this pair - use only last 50 rows for faster processing
                pair_data = df[df['currency_pair'] == currency_pair].copy()
                if len(pair_data) == 0:
                    logger.debug(f"No data found for {currency_pair}")
                    continue
                    
                pair_data = pair_data.sort_values('date').tail(50)  # Reduced from 100 to 50
                
                if len(pair_data) < 20:
                    logger.debug(f"Insufficient data for {currency_pair} ({len(pair_data)} rows)")
                    continue
                
                # Generate signal using the model (with timeout protection)
                try:
                    # Run prediction in executor to avoid blocking
                    # Use a function instead of lambda to avoid closure issues
                    def generate_signal():
                        return signal_generator.predict_next_period(
                            input_data=pair_data,
                            timeframe=timeframe,
                            min_confidence=min_confidence
                        )
                    
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, generate_signal),
                        timeout=5.0  # 5 second timeout per pair
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
                
                # Get latest price
                latest_price = float(pair_data['close_price'].iloc[-1])
                confidence = result['confidence'] * 100  # Convert to percentage
                
                # Generate reasoning
                if signal_type == 'BUY':
                    reason = f"Model predicts price increase (confidence: {confidence:.1f}%)"
                elif signal_type == 'SELL':
                    reason = f"Model predicts price decrease (confidence: {confidence:.1f}%)"
                else:
                    reason = f"Low confidence signal (confidence: {confidence:.1f}%)"
                
                signals.append({
                    'id': signal_id_counter,
                    'symbol': currency_pair,
                    'signal': signal_type,
                    'confidence': round(confidence, 1),
                    'reason': reason,
                    'price': round(latest_price, 4),
                    'timeframe': timeframe,
                    'timestamp': datetime.utcnow().isoformat()
                })
                signal_id_counter += 1
                
            except Exception as e:
                logger.debug(f"Error generating signal for {currency_pair}: {str(e)}")
                continue
        
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Generated {len(signals)} AI signals for timeframe {timeframe} in {elapsed:.2f}s")
        
        # Build response
        response_data = {
            "signals": signals,
            "count": len(signals),
            "timeframe": timeframe,
            "model_available": True,
            "model_accuracy": retraining_service.get_current_accuracy(timeframe)
        }
        
        # Cache the results
        _signal_cache[cache_key] = {
            'data': response_data,
            'timestamp': datetime.utcnow()
        }
        
        # Clean old cache entries (keep only last 10)
        if len(_signal_cache) > 10:
            oldest_key = min(_signal_cache.keys(), key=lambda k: _signal_cache[k]['timestamp'])
            del _signal_cache[oldest_key]
        
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

