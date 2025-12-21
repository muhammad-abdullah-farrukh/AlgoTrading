"""Autotrading control endpoints."""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from sqlalchemy import select
from app.database import db
from app.models import TradingSettings
from app.services.autotrading import autotrading_service
from app.services.paper_trading import paper_trading_service
from app.services.live_trade_logger import live_trade_logger, LiveTradeEvent
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
    timeframe: Optional[str] = Field(None, description="Preferred timeframe for AI signals (e.g., 1d, 1h, 15m)")
    auto_mode: Optional[bool] = Field(None, description="Auto vs manual mode")
    selected_strategy_id: Optional[int] = Field(None, description="Selected strategy ID")


async def _get_live_mode_requested() -> bool:
    async for session in db.get_session():
        result = await session.execute(select(TradingSettings).limit(1))
        settings = result.scalar_one_or_none()
        return bool(settings.live_mode) if settings else False


def _is_demo_server_name(server: Optional[str]) -> bool:
    if not server:
        return False
    return "demo" in server.lower()


def _is_demo_account(account_info: dict) -> bool:
    try:
        trade_mode = account_info.get('trade_mode')
        if isinstance(trade_mode, int) and trade_mode == 0:
            return True
    except Exception:
        pass
    return _is_demo_server_name(account_info.get('server'))


def _mt5_can_execute_live_orders(mt5_client) -> bool:
    try:
        from app.config import settings

        status = mt5_client.get_detailed_status() if hasattr(mt5_client, "get_detailed_status") else {}
        if not bool((status or {}).get('trading_enabled')):
            return False

        if bool(settings.mt5_real_trading):
            return True

        account_info = (status or {}).get('account_info') or {}
        return _is_demo_account(account_info)
    except Exception:
        return False


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
    Enable autotrading and automatically start the trading loop.
    """
    global _autotrading_task, _autotrading_running
    
    try:
        settings, error = await autotrading_service.update_settings({'enabled': True})
        
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        # Automatically start the loop when autotrading is enabled
        # Check if loop is already running to avoid duplicates
        if _autotrading_task is None or _autotrading_task.done():
            _autotrading_running = True
            _autotrading_task = asyncio.create_task(_autotrading_loop())
            logger.info("[Autotrading] Auto-started background loop on enable")
        elif not _autotrading_running:
            # Task exists but flag is False - restart it
            _autotrading_running = True
            logger.info("[Autotrading] Restarting background loop (flag was False)")
        else:
            logger.debug("[Autotrading] Loop already running, skipping start")
        
        return {
            "status": "success",
            "message": "Autotrading enabled and loop started automatically",
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
    Disable autotrading and automatically stop the trading loop.
    """
    global _autotrading_task, _autotrading_running
    
    try:
        settings, error = await autotrading_service.update_settings({'enabled': False})
        
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        # Automatically stop the loop when autotrading is disabled
        if _autotrading_running:
            _autotrading_running = False
            if _autotrading_task:
                _autotrading_task.cancel()
                try:
                    await _autotrading_task
                except asyncio.CancelledError:
                    pass
            logger.info("[Autotrading] Auto-stopped background loop on disable")
        
        return {
            "status": "success",
            "message": "Autotrading disabled and loop stopped automatically",
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
        
        # Get timeframe from settings or use default (respect user selection)
        timeframe = settings.get('timeframe', "1d")  # Use setting or default
        model_timeframe = signal_generator.resolve_model_timeframe(timeframe)

        if not model_timeframe:
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
        
        live_mode_requested = await _get_live_mode_requested()
        from app.services import get_mt5_client
        from app.services.mock_mt5 import MockMT5Client
        mt5_client = get_mt5_client()
        mt5_real_connected = mt5_client.is_connected and not isinstance(mt5_client, MockMT5Client)
        can_execute_live = bool(live_mode_requested and mt5_real_connected and _mt5_can_execute_live_orders(mt5_client))

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
                    timeframe=model_timeframe,
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
                
                position = None
                error = None
                if can_execute_live:
                    comment = f"ATW2-AUTO-{trade_type[:1].upper()}-{currency_pair}"[:31]
                    success, order_result, order_error = mt5_client.place_order(
                        currency_pair,
                        trade_type,
                        position_size,
                        None,
                        comment=comment,
                    )
                    if not success:
                        error = order_error or "MT5 order failed"
                    else:
                        try:
                            live_trade_logger.record_trade(
                                LiveTradeEvent(
                                    timestamp=datetime.utcnow().isoformat(),
                                    symbol=currency_pair,
                                    side=trade_type,
                                    quantity=float(position_size),
                                    price=float((order_result or {}).get('price') or 0.0),
                                    order_id=str((order_result or {}).get('order')) if (order_result or {}).get('order') is not None else None,
                                    comment=comment,
                                    source='mt5_autotrading',
                                    timeframe=str(timeframe),
                                    confidence=float(confidence),
                                )
                            )
                        except Exception:
                            pass
                        position, _ = await paper_trading_service.open_position(
                            currency_pair,
                            trade_type,
                            position_size,
                            (order_result or {}).get('price'),
                            skip_controls=False,
                            external_order_id=str((order_result or {}).get('order')) if (order_result or {}).get('order') is not None else None
                        )
                else:
                    position, error = await paper_trading_service.open_position(
                        currency_pair,
                        trade_type,
                        position_size,
                        None,
                        skip_controls=False
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
                        'position_id': position.get('id') if position else None,
                        'mt5_order': order_result if can_execute_live else None,
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
            "timeframe": timeframe,
            "model_timeframe": model_timeframe,
            "live_mode": bool(can_execute_live)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute strategy: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to execute strategy: {str(e)}")


async def _autotrading_loop():
    """
    Background loop that periodically executes trading strategy.
    This loop runs independently and persists even if frontend disconnects.
    It checks database settings directly, not just the global flag.
    """
    global _autotrading_running
    
    logger.info("[Autotrading] Background loop started - will persist until explicitly disabled")
    
    while True:  # Run indefinitely, check database for enabled state
        try:
            # ALWAYS check database settings (not just global flag)
            settings = await autotrading_service.get_settings()
            
            # Check if we should stop (global flag OR database says disabled)
            if not _autotrading_running:
                logger.info("[Autotrading] Loop stopping - global flag set to False")
                break
            
            if not settings or not settings.get('enabled', False):
                logger.debug("[Autotrading] Loop paused - autotrading disabled in database")
                await asyncio.sleep(60)  # Check every minute
                continue
            
            # Execute strategy
            try:
                from app.ai.signal_generator import signal_generator
                from app.routers.ml_training import _get_cached_dataset
                import pandas as pd
                
                # Get timeframe from settings or use default
                timeframe = settings.get('timeframe', "1d") if settings else "1d"
                model_timeframe = signal_generator.resolve_model_timeframe(timeframe)

                if not model_timeframe:
                    await asyncio.sleep(300)  # Wait 5 minutes if no model
                    continue
                
                df = _get_cached_dataset()
                if df is None or df.empty:
                    await asyncio.sleep(300)  # Wait 5 minutes if no data
                    continue
                
                # Execute trades (optimized for speed)
                available_pairs = df['currency_pair'].unique().tolist()[:3]  # Only 3 pairs for speed

                live_mode_requested = await _get_live_mode_requested()
                from app.services import get_mt5_client
                from app.services.mock_mt5 import MockMT5Client
                mt5_client = get_mt5_client()
                mt5_real_connected = mt5_client.is_connected and not isinstance(mt5_client, MockMT5Client)
                can_execute_live = bool(live_mode_requested and mt5_real_connected and _mt5_can_execute_live_orders(mt5_client))
                
                for currency_pair in available_pairs:
                    try:
                        # Try to get REAL-TIME data from MT5/database first
                        pair_data = None
                        try:
                            from app.services import get_mt5_client
                            from app.database import db
                            from app.models import OHLCV
                            from sqlalchemy import select, desc
                            from datetime import timedelta
                            
                            mt5_client = get_mt5_client()
                            
                            # Try MT5 first for real-time data
                            if mt5_client.is_connected:
                                try:
                                    timeframe_map = {
                                        '1m': 'M1', '3m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30', '45m': 'M15',
                                        '1h': 'H1', '2h': 'H1', '3h': 'H1', '4h': 'H4',
                                        '1d': 'D1', '1w': 'W1',
                                        '1M': 'MN1', '3M': 'MN1', '6M': 'MN1', '12M': 'MN1'
                                    }
                                    mt5_tf = timeframe_map.get(timeframe, timeframe_map.get(timeframe.lower(), 'H1'))
                                    
                                    end_time = datetime.utcnow()
                                    if mt5_tf.startswith('M'):
                                        minutes = int(mt5_tf[1:]) if len(mt5_tf) > 1 else 1
                                        start_time = end_time - timedelta(minutes=minutes * 20)
                                    elif mt5_tf.startswith('H'):
                                        hours = int(mt5_tf[1:]) if len(mt5_tf) > 1 else 1
                                        start_time = end_time - timedelta(hours=hours * 20)
                                    elif mt5_tf == 'D1':
                                        start_time = end_time - timedelta(days=20)
                                    else:
                                        start_time = end_time - timedelta(hours=20)
                                    
                                    ohlcv_df = mt5_client.get_ohlcv(currency_pair, mt5_tf, start=start_time, end=end_time)
                                    
                                    if not ohlcv_df.empty:
                                        ohlcv_df = ohlcv_df.rename(columns={'timestamp': 'date', 'close': 'close_price'})
                                        ohlcv_df['currency_pair'] = currency_pair
                                        pair_data = ohlcv_df.copy()
                                except Exception as e:
                                    logger.debug(f"MT5 data fetch failed: {str(e)}")
                            
                            # Fallback to database
                            if pair_data is None or pair_data.empty:
                                async for session in db.get_session():
                                    query = (
                                        select(OHLCV)
                                        .where(OHLCV.symbol == currency_pair)
                                        .order_by(desc(OHLCV.timestamp))
                                        .limit(20)
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
                                    break
                        except Exception as e:
                            logger.debug(f"Real-time data fetch failed: {str(e)}")
                        
                        # Fallback to CSV
                        if pair_data is None or pair_data.empty:
                            pair_data = df[df['currency_pair'] == currency_pair].copy()
                            if len(pair_data) == 0:
                                continue
                            pair_data = pair_data.sort_values('date').tail(20)
                        
                        if len(pair_data) < 10:
                            continue
                        
                        # Use async executor for faster prediction
                        def generate_signal():
                            return signal_generator.predict_next_period(
                                input_data=pair_data,
                                timeframe=model_timeframe,
                                min_confidence=0.6
                            )
                        
                        loop = asyncio.get_event_loop()
                        result = await asyncio.wait_for(
                            loop.run_in_executor(None, generate_signal),
                            timeout=2.0  # 2 second timeout
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
                        
                        position = None
                        error = None
                        if can_execute_live:
                            comment = f"ATW2-AUTO-{trade_type[:1].upper()}-{currency_pair}"[:31]
                            success, order_result, order_error = mt5_client.place_order(
                                currency_pair,
                                trade_type,
                                position_size,
                                None,
                                comment=comment,
                            )
                            if not success:
                                error = order_error or "MT5 order failed"
                            else:
                                try:
                                    live_trade_logger.record_trade(
                                        LiveTradeEvent(
                                            timestamp=datetime.utcnow().isoformat(),
                                            symbol=currency_pair,
                                            side=trade_type,
                                            quantity=float(position_size),
                                            price=float((order_result or {}).get('price') or 0.0),
                                            order_id=str((order_result or {}).get('order')) if (order_result or {}).get('order') is not None else None,
                                            comment=comment,
                                            source='mt5_autotrading',
                                            timeframe=str(timeframe),
                                            confidence=float(confidence),
                                        )
                                    )
                                except Exception:
                                    pass
                                position, _ = await paper_trading_service.open_position(
                                    currency_pair,
                                    trade_type,
                                    position_size,
                                    (order_result or {}).get('price'),
                                    skip_controls=False,
                                    external_order_id=str((order_result or {}).get('order')) if (order_result or {}).get('order') is not None else None
                                )
                        else:
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
                            
                            # Send WebSocket notification for autotrading trade
                            try:
                                from app.websocket import manager
                                notification_data = {
                                    'type': 'autotrading_trade',
                                    'trade': {
                                        'symbol': currency_pair,
                                        'action': trade_type.upper(),
                                        'quantity': position_size,
                                        'price': position.get('average_price', 0),
                                        'confidence': confidence,
                                        'timestamp': datetime.utcnow().isoformat()
                                    }
                                }
                                await manager.broadcast(notification_data)
                                logger.debug(f"[Autotrading] Notification sent for {trade_type} {currency_pair}")
                            except Exception as notif_error:
                                logger.debug(f"[Autotrading] Failed to send notification: {str(notif_error)}")
                            
                    except Exception as e:
                        logger.debug(f"[Autotrading] Error processing {currency_pair}: {str(e)}")
                        continue
                
            except Exception as e:
                logger.error(f"[Autotrading] Strategy execution error: {str(e)}")
            
            # Wait 1 minute before next execution (faster trading)
            await asyncio.sleep(60)
            
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
