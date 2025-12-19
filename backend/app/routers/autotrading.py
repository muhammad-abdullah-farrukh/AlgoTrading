"""Autotrading control endpoints."""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
from pydantic import BaseModel, Field
from app.services.autotrading import autotrading_service
from app.services.paper_trading import paper_trading_service
from app.utils.logging import get_logger
import asyncio

logger = get_logger(__name__)
router = APIRouter(prefix="/api/autotrading", tags=["autotrading"])

# Background task for autotrading loop
_autotrading_task: Optional[asyncio.Task] = None
_autotrading_running = False


class AutotradingSettingsRequest(BaseModel):
    """Request model for updating autotrading settings."""
    enabled: Optional[bool] = Field(None, description="Enable/disable autotrading")
    emergency_stop: Optional[bool] = Field(None, description="Emergency stop flag")
    stop_loss_percent: Optional[float] = Field(None, ge=0, le=100, description="Stop loss percentage (e.g., 2.0 for 2%)")
    take_profit_percent: Optional[float] = Field(None, ge=0, le=100, description="Take profit percentage (e.g., 5.0 for 5%)")
    max_daily_loss: Optional[float] = Field(None, ge=0, description="Maximum daily loss amount")
    position_size: Optional[float] = Field(None, gt=0, description="Default position size")
    auto_mode: Optional[bool] = Field(None, description="Auto vs manual mode")
    selected_strategy_id: Optional[int] = Field(None, description="Selected strategy ID")


@router.get("/settings")
async def get_settings():
    """
    Get current autotrading settings.
    
    Returns all autotrading controls and their current values.
    """
    try:
        settings = await autotrading_service.get_settings()
        
        if not settings:
            raise HTTPException(status_code=404, detail="Autotrading settings not found")
        
        return settings
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get autotrading settings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get settings: {str(e)}")


@router.put("/settings")
async def update_settings(request: AutotradingSettingsRequest):
    """
    Update autotrading settings.
    
    Updates one or more autotrading controls. Only provided fields are updated.
    """
    try:
        settings_dict = request.dict(exclude_unset=True)
        
        if not settings_dict:
            raise HTTPException(status_code=400, detail="No settings provided to update")
        
        updated_settings, error = await autotrading_service.update_settings(settings_dict)
        
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        return {
            "status": "success",
            "message": "Settings updated successfully",
            "settings": updated_settings
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update autotrading settings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")


@router.post("/enable")
async def enable_autotrading():
    """
    Enable autotrading.
    """
    try:
        settings, error = await autotrading_service.update_settings({'enabled': True})
        
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        return {
            "status": "success",
            "message": "Autotrading enabled",
            "settings": settings
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enable autotrading: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to enable autotrading: {str(e)}")


@router.post("/disable")
async def disable_autotrading():
    """
    Disable autotrading.
    """
    try:
        settings, error = await autotrading_service.update_settings({'enabled': False})
        
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        return {
            "status": "success",
            "message": "Autotrading disabled",
            "settings": settings
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disable autotrading: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to disable autotrading: {str(e)}")


@router.post("/emergency-stop")
async def emergency_stop():
    """
    Activate emergency stop.
    
    Immediately stops all autotrading and prevents new trades.
    """
    try:
        settings, error = await autotrading_service.update_settings({
            'emergency_stop': True,
            'enabled': False
        })
        
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        logger.warning("Emergency stop activated")
        
        return {
            "status": "success",
            "message": "Emergency stop activated",
            "settings": settings
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to activate emergency stop: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to activate emergency stop: {str(e)}")


@router.post("/emergency-stop/clear")
async def clear_emergency_stop():
    """
    Clear emergency stop.
    
    Clears the emergency stop flag. Autotrading must be manually enabled again.
    """
    try:
        settings, error = await autotrading_service.update_settings({'emergency_stop': False})
        
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        logger.info("Emergency stop cleared")
        
        return {
            "status": "success",
            "message": "Emergency stop cleared",
            "settings": settings
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear emergency stop: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear emergency stop: {str(e)}")


@router.post("/reset-daily-loss")
async def reset_daily_loss():
    """
    Reset daily loss tracking.
    
    Resets the daily loss counter to zero and updates the reset date.
    """
    try:
        success, error = await autotrading_service.reset_daily_loss()
        
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        return {
            "status": "success",
            "message": "Daily loss reset successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset daily loss: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reset daily loss: {str(e)}")


@router.post("/check-triggers")
async def check_triggers(background_tasks: BackgroundTasks):
    """
    Check stop loss and take profit triggers.
    
    Checks all open positions and closes those that hit stop loss or take profit.
    Runs in background to avoid blocking.
    """
    try:
        positions_to_close = await autotrading_service.check_stop_loss_take_profit()
        
        if not positions_to_close:
            return {
                "status": "success",
                "message": "No positions need to be closed",
                "positions_closed": 0
            }
        
        # Close positions in background
        closed_count = 0
        errors = []
        
        for pos_info in positions_to_close:
            try:
                position, error = await paper_trading_service.close_position(pos_info['position_id'])
                if error:
                    errors.append(f"Position {pos_info['position_id']}: {error}")
                else:
                    closed_count += 1
                    logger.info(
                        f"Closed position {pos_info['position_id']} ({pos_info['symbol']}) "
                        f"due to {pos_info['reason']}"
                    )
            except Exception as e:
                errors.append(f"Position {pos_info['position_id']}: {str(e)}")
        
        return {
            "status": "success",
            "message": f"Checked triggers, closed {closed_count} positions",
            "positions_closed": closed_count,
            "errors": errors if errors else None
        }
        
    except Exception as e:
        logger.error(f"Failed to check triggers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check triggers: {str(e)}")


@router.get("/status")
async def get_status():
    """
    Get autotrading status.
    
    Returns current status including enabled state, daily loss, and control settings.
    """
    try:
        status = await autotrading_service.get_status()
        
        if 'error' in status:
            raise HTTPException(status_code=500, detail=status['error'])
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get autotrading status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.post("/validate-trade")
async def validate_trade(symbol: str, trade_type: str, quantity: float):
    """
    Validate if a trade would be allowed.
    
    Checks if a trade would pass all autotrading controls without executing it.
    """
    try:
        allowed, reason = await autotrading_service.check_trade_allowed(symbol, trade_type, quantity)
        
        return {
            "allowed": allowed,
            "reason": reason if not allowed else None,
            "symbol": symbol,
            "trade_type": trade_type,
            "quantity": quantity
        }
        
    except Exception as e:
        logger.error(f"Failed to validate trade: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to validate trade: {str(e)}")


@router.post("/execute-strategy")
async def execute_strategy(background_tasks: BackgroundTasks):
    """
    Execute trading strategy based on AI signals.
    
    Monitors AI signals and executes trades automatically when signals meet criteria.
    """
    try:
        # Check if autotrading is enabled
        settings = await autotrading_service.get_settings()
        if not settings:
            raise HTTPException(status_code=404, detail="Autotrading settings not found")
        
        if not settings.get('enabled', False):
            return {
                "status": "skipped",
                "message": "Autotrading is disabled",
                "enabled": False
            }
        
        if settings.get('emergency_stop', False):
            return {
                "status": "skipped",
                "message": "Emergency stop is active",
                "enabled": False
            }
        
        logger.info("[Autotrading] Executing strategy based on AI signals...")
        
        # Get AI signals
        from app.ai.signal_generator import signal_generator
        from app.routers.ml_training import _get_cached_dataset
        import pandas as pd
        
        # Get timeframe from settings or use default
        timeframe = "1d"  # Default timeframe
        
        # Check if model is available
        if not signal_generator.is_model_available(timeframe):
            return {
                "status": "skipped",
                "message": f"No trained model found for timeframe: {timeframe}",
                "enabled": True
            }
        
        # Get dataset
        df = _get_cached_dataset()
        if df is None or df.empty:
            return {
                "status": "skipped",
                "message": "No market data available",
                "enabled": True
            }
        
        # Get unique currency pairs (limit to 5 for performance)
        available_pairs = df['currency_pair'].unique().tolist()[:5]
        
        executed_trades = []
        skipped_signals = []
        
        from app.services.paper_trading import paper_trading_service
        
        for currency_pair in available_pairs:
            try:
                # Get data for this pair
                pair_data = df[df['currency_pair'] == currency_pair].copy()
                pair_data = pair_data.sort_values('date').tail(50)
                
                if len(pair_data) < 20:
                    continue
                
                # Generate signal
                result = signal_generator.predict_next_period(
                    input_data=pair_data,
                    timeframe=timeframe,
                    min_confidence=0.6  # Higher confidence for autotrading
                )
                
                if result.get('error') or result.get('signal') == 'HOLD':
                    continue
                
                signal_type = result['signal']
                confidence = result['confidence']
                
                # Only execute high-confidence signals
                if confidence < 0.6:
                    skipped_signals.append({
                        'symbol': currency_pair,
                        'signal': signal_type,
                        'confidence': confidence,
                        'reason': 'Low confidence'
                    })
                    continue
                
                # Get position size from settings
                position_size = settings.get('position_size', 0.1)  # Default 0.1 lots
                
                # Check if we already have a position for this symbol
                positions = await paper_trading_service.get_positions('open')
                existing_position = next(
                    (p for p in positions if p['symbol'] == currency_pair),
                    None
                )
                
                # Skip if we already have a position (avoid over-trading)
                if existing_position:
                    skipped_signals.append({
                        'symbol': currency_pair,
                        'signal': signal_type,
                        'confidence': confidence,
                        'reason': 'Position already exists'
                    })
                    continue
                
                # Execute trade
                trade_type = 'buy' if signal_type == 'BUY' else 'sell'
                
                position, error = await paper_trading_service.open_position(
                    currency_pair,
                    trade_type,
                    position_size,
                    None,  # Use market price
                    skip_controls=False  # Apply autotrading controls
                )
                
                if error:
                    skipped_signals.append({
                        'symbol': currency_pair,
                        'signal': signal_type,
                        'confidence': confidence,
                        'reason': error
                    })
                else:
                    executed_trades.append({
                        'symbol': currency_pair,
                        'type': trade_type,
                        'quantity': position_size,
                        'confidence': confidence,
                        'position_id': position.get('id') if position else None
                    })
                    logger.info(
                        f"[Autotrading] Trade executed: {trade_type} {position_size} {currency_pair} "
                        f"(confidence: {confidence:.2f})"
                    )
                    
            except Exception as e:
                logger.error(f"[Autotrading] Error processing {currency_pair}: {str(e)}")
                continue
        
        return {
            "status": "success",
            "message": f"Strategy executed: {len(executed_trades)} trades, {len(skipped_signals)} skipped",
            "enabled": True,
            "executed_trades": executed_trades,
            "skipped_signals": skipped_signals,
            "timeframe": timeframe
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute strategy: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to execute strategy: {str(e)}")


async def _autotrading_loop():
    """Background loop that periodically executes trading strategy."""
    global _autotrading_running
    
    while _autotrading_running:
        try:
            # Check if autotrading is still enabled
            settings = await autotrading_service.get_settings()
            if not settings or not settings.get('enabled', False):
                logger.debug("[Autotrading] Loop paused - autotrading disabled")
                await asyncio.sleep(60)  # Check every minute
                continue
            
            # Execute strategy
            try:
                from app.ai.signal_generator import signal_generator
                from app.routers.ml_training import _get_cached_dataset
                import pandas as pd
                
                timeframe = "1d"
                
                if not signal_generator.is_model_available(timeframe):
                    await asyncio.sleep(300)  # Wait 5 minutes if no model
                    continue
                
                df = _get_cached_dataset()
                if df is None or df.empty:
                    await asyncio.sleep(300)  # Wait 5 minutes if no data
                    continue
                
                # Execute trades (same logic as execute_strategy endpoint)
                available_pairs = df['currency_pair'].unique().tolist()[:5]
                
                for currency_pair in available_pairs:
                    try:
                        pair_data = df[df['currency_pair'] == currency_pair].copy()
                        pair_data = pair_data.sort_values('date').tail(50)
                        
                        if len(pair_data) < 20:
                            continue
                        
                        result = signal_generator.predict_next_period(
                            input_data=pair_data,
                            timeframe=timeframe,
                            min_confidence=0.6
                        )
                        
                        if result.get('error') or result.get('signal') == 'HOLD':
                            continue
                        
                        signal_type = result['signal']
                        confidence = result['confidence']
                        
                        if confidence < 0.6:
                            continue
                        
                        position_size = settings.get('position_size', 0.1)
                        
                        positions = await paper_trading_service.get_positions('open')
                        existing_position = next(
                            (p for p in positions if p['symbol'] == currency_pair),
                            None
                        )
                        
                        if existing_position:
                            continue
                        
                        trade_type = 'buy' if signal_type == 'BUY' else 'sell'
                        
                        position, error = await paper_trading_service.open_position(
                            currency_pair,
                            trade_type,
                            position_size,
                            None,
                            skip_controls=False
                        )
                        
                        if not error and position:
                            logger.info(
                                f"[Autotrading] Auto-trade executed: {trade_type} {position_size} {currency_pair} "
                                f"(confidence: {confidence:.2f})"
                            )
                            
                    except Exception as e:
                        logger.debug(f"[Autotrading] Error processing {currency_pair}: {str(e)}")
                        continue
                
            except Exception as e:
                logger.error(f"[Autotrading] Strategy execution error: {str(e)}")
            
            # Wait 5 minutes before next execution
            await asyncio.sleep(300)
            
        except asyncio.CancelledError:
            logger.info("[Autotrading] Loop cancelled")
            break
        except Exception as e:
            logger.error(f"[Autotrading] Loop error: {str(e)}")
            await asyncio.sleep(60)  # Wait 1 minute on error


@router.post("/start-loop")
async def start_autotrading_loop():
    """
    Start the background autotrading loop.
    
    The loop will periodically check AI signals and execute trades automatically.
    """
    global _autotrading_task, _autotrading_running
    
    try:
        if _autotrading_running:
            return {
                "status": "already_running",
                "message": "Autotrading loop is already running"
            }
        
        settings = await autotrading_service.get_settings()
        if not settings or not settings.get('enabled', False):
            raise HTTPException(
                status_code=400,
                detail="Autotrading must be enabled before starting the loop"
            )
        
        _autotrading_running = True
        _autotrading_task = asyncio.create_task(_autotrading_loop())
        
        logger.info("[Autotrading] Background loop started")
        
        return {
            "status": "started",
            "message": "Autotrading loop started successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start autotrading loop: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start loop: {str(e)}")


@router.post("/stop-loop")
async def stop_autotrading_loop():
    """
    Stop the background autotrading loop.
    """
    global _autotrading_task, _autotrading_running
    
    try:
        if not _autotrading_running:
            return {
                "status": "not_running",
                "message": "Autotrading loop is not running"
            }
        
        _autotrading_running = False
        
        if _autotrading_task:
            _autotrading_task.cancel()
            try:
                await _autotrading_task
            except asyncio.CancelledError:
                pass
        
        logger.info("[Autotrading] Background loop stopped")
        
        return {
            "status": "stopped",
            "message": "Autotrading loop stopped successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to stop autotrading loop: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop loop: {str(e)}")
