"""Autotrading service - controls and safety mechanisms."""
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from app.database import db
from app.models import AutotradingSettings, Position, Trade
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AutotradingService:
    """Service for managing autotrading controls and safety mechanisms."""
    
    async def get_settings(self) -> Optional[Dict]:
        """
        Get current autotrading settings.
        
        Returns:
            Dictionary with settings or None if not found
        """
        try:
            async for session in db.get_session():
                query = select(AutotradingSettings).limit(1)
                result = await session.execute(query)
                settings = result.scalar_one_or_none()
                
                if not settings:
                    # Create default settings
                    settings = AutotradingSettings(
                        enabled=False,
                        emergency_stop=False,
                        auto_mode=False
                    )
                    session.add(settings)
                    await session.commit()
                    await session.refresh(settings)
                
                return {
                    'id': settings.id,
                    'enabled': settings.enabled,
                    'emergency_stop': settings.emergency_stop,
                    'stop_loss_percent': float(settings.stop_loss_percent) if settings.stop_loss_percent else None,
                    'take_profit_percent': float(settings.take_profit_percent) if settings.take_profit_percent else None,
                    'max_daily_loss': float(settings.max_daily_loss) if settings.max_daily_loss else None,
                    'position_size': float(settings.position_size) if settings.position_size else None,
                    'timeframe': str(settings.timeframe) if getattr(settings, 'timeframe', None) else '1d',
                    'auto_mode': settings.auto_mode,
                    'selected_strategy_id': settings.selected_strategy_id,
                    'daily_loss_reset_date': settings.daily_loss_reset_date.isoformat() if settings.daily_loss_reset_date else None,
                    'daily_loss_amount': float(settings.daily_loss_amount),
                    'updated_at': settings.updated_at.isoformat()
                }
                
                break  # Exit session context
                
        except Exception as e:
            logger.error(f"Failed to get autotrading settings: {str(e)}")
            return None
    
    async def update_settings(self, settings_dict: Dict) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Update autotrading settings.
        
        Args:
            settings_dict: Dictionary with settings to update
            
        Returns:
            Tuple of (updated_settings_dict, error_message)
        """
        try:
            async for session in db.get_session():
                query = select(AutotradingSettings).limit(1)
                result = await session.execute(query)
                settings = result.scalar_one_or_none()
                
                if not settings:
                    settings = AutotradingSettings()
                    session.add(settings)
                
                # Update fields
                if 'enabled' in settings_dict:
                    settings.enabled = bool(settings_dict['enabled'])
                
                if 'emergency_stop' in settings_dict:
                    settings.emergency_stop = bool(settings_dict['emergency_stop'])
                    # If emergency stop is enabled, also disable autotrading
                    if settings.emergency_stop:
                        settings.enabled = False
                
                if 'stop_loss_percent' in settings_dict:
                    value = settings_dict['stop_loss_percent']
                    settings.stop_loss_percent = float(value) if value is not None else None
                
                if 'take_profit_percent' in settings_dict:
                    value = settings_dict['take_profit_percent']
                    settings.take_profit_percent = float(value) if value is not None else None
                
                if 'max_daily_loss' in settings_dict:
                    value = settings_dict['max_daily_loss']
                    settings.max_daily_loss = float(value) if value is not None else None
                
                if 'position_size' in settings_dict:
                    value = settings_dict['position_size']
                    settings.position_size = float(value) if value is not None else None

                if 'timeframe' in settings_dict:
                    value = settings_dict['timeframe']
                    settings.timeframe = str(value) if value is not None else None
                
                if 'auto_mode' in settings_dict:
                    settings.auto_mode = bool(settings_dict['auto_mode'])
                
                if 'selected_strategy_id' in settings_dict:
                    value = settings_dict['selected_strategy_id']
                    settings.selected_strategy_id = int(value) if value is not None else None
                
                settings.updated_at = datetime.utcnow()
                
                await session.commit()
                await session.refresh(settings)
                
                return {
                    'id': settings.id,
                    'enabled': settings.enabled,
                    'emergency_stop': settings.emergency_stop,
                    'stop_loss_percent': float(settings.stop_loss_percent) if settings.stop_loss_percent else None,
                    'take_profit_percent': float(settings.take_profit_percent) if settings.take_profit_percent else None,
                    'max_daily_loss': float(settings.max_daily_loss) if settings.max_daily_loss else None,
                    'position_size': float(settings.position_size) if settings.position_size else None,
                    'timeframe': str(settings.timeframe) if getattr(settings, 'timeframe', None) else '1d',
                    'auto_mode': settings.auto_mode,
                    'selected_strategy_id': settings.selected_strategy_id,
                    'daily_loss_reset_date': settings.daily_loss_reset_date.isoformat() if settings.daily_loss_reset_date else None,
                    'daily_loss_amount': float(settings.daily_loss_amount),
                    'updated_at': settings.updated_at.isoformat()
                }, None
                
                break  # Exit session context
                
        except Exception as e:
            error = f"Failed to update autotrading settings: {str(e)}"
            logger.error(error)
            return None, error
    
    async def reset_daily_loss(self) -> Tuple[bool, Optional[str]]:
        """
        Reset daily loss tracking.
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            async for session in db.get_session():
                query = select(AutotradingSettings).limit(1)
                result = await session.execute(query)
                settings = result.scalar_one_or_none()
                
                if not settings:
                    return False, "Autotrading settings not found"
                
                settings.daily_loss_amount = 0.0
                settings.daily_loss_reset_date = datetime.utcnow()
                settings.updated_at = datetime.utcnow()
                
                await session.commit()
                
                logger.info("Daily loss reset")
                return True, None
                
                break  # Exit session context
                
        except Exception as e:
            error = f"Failed to reset daily loss: {str(e)}"
            logger.error(error)
            return False, error
    
    async def update_daily_loss(self, loss_amount: float) -> None:
        """
        Update daily loss amount.
        
        Args:
            loss_amount: Loss amount to add (positive value)
        """
        try:
            async for session in db.get_session():
                query = select(AutotradingSettings).limit(1)
                result = await session.execute(query)
                settings = result.scalar_one_or_none()
                
                if not settings:
                    return
                
                # Check if we need to reset (new day)
                today = date.today()
                if settings.daily_loss_reset_date:
                    reset_date = settings.daily_loss_reset_date.date()
                    if reset_date != today:
                        settings.daily_loss_amount = 0.0
                        settings.daily_loss_reset_date = datetime.utcnow()
                
                settings.daily_loss_amount += loss_amount
                settings.updated_at = datetime.utcnow()
                
                await session.commit()
                
                break  # Exit session context
                
        except Exception as e:
            logger.error(f"Failed to update daily loss: {str(e)}")
    
    async def check_trade_allowed(self, symbol: str, trade_type: str, quantity: float) -> Tuple[bool, Optional[str]]:
        """
        Check if a trade is allowed based on autotrading controls.
        
        Args:
            symbol: Trading symbol
            trade_type: 'buy' or 'sell'
            quantity: Trade quantity (lot size)
            
        Returns:
            Tuple of (allowed, reason_if_not_allowed)
        """
        try:
            logger.info(f"[Autotrading] Checking trade allowance: {trade_type} {quantity} {symbol}")
            
            settings_dict = await self.get_settings()
            if not settings_dict:
                logger.warning("[Autotrading] Settings not found")
                return False, "Autotrading settings not found"
            
            # Check emergency stop
            if settings_dict.get('emergency_stop', False):
                logger.warning("[Autotrading] Trade blocked: Emergency stop is active")
                return False, "Emergency stop is active"
            
            # Check if autotrading is enabled (only for auto mode)
            if settings_dict.get('auto_mode', False) and not settings_dict.get('enabled', False):
                logger.warning("[Autotrading] Trade blocked: Autotrading is disabled")
                return False, "Autotrading is disabled"
            
            # Check max daily loss
            max_daily_loss = settings_dict.get('max_daily_loss')
            daily_loss = settings_dict.get('daily_loss_amount', 0.0)
            
            if max_daily_loss is not None and daily_loss >= max_daily_loss:
                logger.warning(f"[Autotrading] Trade blocked: Max daily loss reached ({daily_loss} >= {max_daily_loss})")
                return False, f"Maximum daily loss reached: {daily_loss} >= {max_daily_loss}"
            
            # Check position size limit
            position_size = settings_dict.get('position_size')
            if position_size is not None and quantity > position_size:
                logger.warning(f"[Autotrading] Trade blocked: Position size exceeds limit ({quantity} > {position_size})")
                return False, f"Position size exceeds limit: {quantity} > {position_size}"
            
            logger.info(f"[Autotrading] Trade allowed: {trade_type} {quantity} {symbol}")
            return True, None
            
        except Exception as e:
            logger.error(f"[Autotrading] Failed to check trade allowance: {str(e)}")
            return False, f"Error checking trade allowance: {str(e)}"
    
    async def check_stop_loss_take_profit(self) -> List[Dict]:
        """
        Check all open positions for stop loss and take profit triggers.
        
        Returns:
            List of positions that should be closed
        """
        try:
            settings_dict = await self.get_settings()
            if not settings_dict:
                return []
            
            stop_loss_percent = settings_dict.get('stop_loss_percent')
            take_profit_percent = settings_dict.get('take_profit_percent')
            
            if stop_loss_percent is None and take_profit_percent is None:
                return []
            
            positions_to_close = []
            
            async for session in db.get_session():
                query = select(Position).where(Position.status == 'open')
                result = await session.execute(query)
                positions = result.scalars().all()
                
                for position in positions:
                    if not position.current_price or not position.average_price:
                        continue
                    
                    # Calculate price change percentage
                    if position.quantity > 0:  # Long position
                        price_change_percent = ((position.current_price - position.average_price) / position.average_price) * 100
                    else:  # Short position
                        price_change_percent = ((position.average_price - position.current_price) / position.average_price) * 100
                    
                    # Check stop loss
                    if stop_loss_percent is not None and price_change_percent <= -stop_loss_percent:
                        positions_to_close.append({
                            'position_id': position.id,
                            'symbol': position.symbol,
                            'reason': 'stop_loss',
                            'price_change_percent': price_change_percent,
                            'stop_loss_percent': stop_loss_percent
                        })
                        continue
                    
                    # Check take profit
                    if take_profit_percent is not None and price_change_percent >= take_profit_percent:
                        positions_to_close.append({
                            'position_id': position.id,
                            'symbol': position.symbol,
                            'reason': 'take_profit',
                            'price_change_percent': price_change_percent,
                            'take_profit_percent': take_profit_percent
                        })
                
                break  # Exit session context
            
            return positions_to_close
            
        except Exception as e:
            logger.error(f"Failed to check stop loss/take profit: {str(e)}")
            return []
    
    async def get_status(self) -> Dict:
        """
        Get autotrading status and statistics.
        
        Returns:
            Dictionary with status information
        """
        try:
            settings_dict = await self.get_settings()
            if not settings_dict:
                return {'error': 'Settings not found'}
            
            # Get position statistics
            async for session in db.get_session():
                from sqlalchemy import func
                from app.models import Position
                
                open_positions_query = select(func.count(Position.id)).where(Position.status == 'open')
                open_result = await session.execute(open_positions_query)
                open_count = open_result.scalar() or 0
                
                total_pnl_query = select(func.sum(Position.unrealized_pnl + Position.realized_pnl))
                pnl_result = await session.execute(total_pnl_query)
                total_pnl = pnl_result.scalar() or 0.0
                
                break  # Exit session context
            
            return {
                'enabled': settings_dict.get('enabled', False),
                'emergency_stop': settings_dict.get('emergency_stop', False),
                'auto_mode': settings_dict.get('auto_mode', False),
                'daily_loss': settings_dict.get('daily_loss_amount', 0.0),
                'max_daily_loss': settings_dict.get('max_daily_loss'),
                'daily_loss_limit_reached': (
                    settings_dict.get('max_daily_loss') is not None and
                    settings_dict.get('daily_loss_amount', 0.0) >= settings_dict.get('max_daily_loss', 0.0)
                ),
                'open_positions': open_count,
                'total_pnl': float(total_pnl) if total_pnl else 0.0,
                'controls': {
                    'stop_loss_percent': settings_dict.get('stop_loss_percent'),
                    'take_profit_percent': settings_dict.get('take_profit_percent'),
                    'position_size': settings_dict.get('position_size')
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get autotrading status: {str(e)}")
            return {'error': str(e)}


# Global autotrading service instance
autotrading_service = AutotradingService()
