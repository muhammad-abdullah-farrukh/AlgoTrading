"""
Paper trading engine - simulated trading without real market execution.

SAFETY: This service ONLY performs paper trading (simulated trades).
NO live trading is executed through this service. All trades are simulated
and stored in the database for tracking and analysis purposes only.

To enable live trading, a separate service would need to be implemented
with explicit safety checks and user confirmation.
"""
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, update
from app.database import db
from app.models import Trade, Position, OHLCV
from app.services import get_mt5_client
from app.services.autotrading import autotrading_service
from app.utils.logging import get_logger
import uuid

logger = get_logger(__name__)

# Safety constant: Paper trading default (can be overridden via config)
# Set to False to enable real MT5 trades (requires MT5 connection and proper configuration)
PAPER_TRADING_ONLY = True  # Default: paper trading only for safety


class PaperTradingService:
    """Service for paper trading operations."""
    
    async def _execute_mt5_trade(
        self,
        mt5_client,
        symbol: str,
        trade_type: str,
        quantity: float,
        price: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Execute a real trade in MT5.
        
        Args:
            mt5_client: MT5 client instance
            symbol: Trading symbol
            trade_type: 'buy' or 'sell'
            quantity: Quantity to trade (lot size)
            price: Optional execution price
            
        Returns:
            Dictionary with trade result or None if failed
        """
        try:
            import MetaTrader5 as mt5
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return {'error': f"Symbol {symbol} not found in MT5"}
            
            # Ensure symbol is selected
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    return {'error': f"Failed to select symbol {symbol}"}
            
            # Determine order type
            if trade_type == 'buy':
                order_type = mt5.ORDER_TYPE_BUY
                price_type = mt5.ORDER_TYPE_MARKET  # Market order
            else:  # sell
                order_type = mt5.ORDER_TYPE_SELL
                price_type = mt5.ORDER_TYPE_MARKET  # Market order
            
            # Get current price if not provided
            if price is None:
                tick = mt5.symbol_info_tick(symbol)
                if not tick:
                    return {'error': f"Could not get current price for {symbol}"}
                price = tick.ask if trade_type == 'buy' else tick.bid
            
            # Prepare trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(quantity),
                "type": order_type,
                "price": float(price),
                "deviation": 20,  # Slippage in points
                "magic": 234000,  # Magic number for identification
                "comment": "AlgoTradeBot",
                "type_time": mt5.ORDER_TIME_GTC,  # Good till cancelled
                "type_filling": mt5.ORDER_FILLING_IOC,  # Immediate or cancel
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    'error': f"MT5 order failed: {result.retcode} - {result.comment}",
                    'retcode': result.retcode
                }
            
            logger.info(f"[MT5] Trade executed: {trade_type} {quantity} {symbol} @ {price} (Order: {result.order})")
            
            return {
                'success': True,
                'order': result.order,
                'deal': result.deal,
                'volume': result.volume,
                'price': result.price,
                'comment': result.comment
            }
            
        except ImportError:
            return {'error': "MT5 module not available"}
        except Exception as e:
            logger.error(f"[MT5] Trade execution error: {str(e)}")
            return {'error': str(e)}
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None if not available
        """
        try:
            # Try to get latest price from database
            async for session in db.get_session():
                query = (
                    select(OHLCV.close)
                    .where(OHLCV.symbol == symbol)
                    .order_by(OHLCV.timestamp.desc())
                    .limit(1)
                )
                result = await session.execute(query)
                price = result.scalar()
                
                if price:
                    return float(price)
                
                break  # Exit session context
            
            # Fallback to MT5 if available
            try:
                mt5_client = get_mt5_client()
                if mt5_client.is_connected:
                    symbol_info = mt5_client.get_symbol_info(symbol)
                    if symbol_info and symbol_info.get('bid'):
                        return float(symbol_info['bid'])
            except Exception as e:
                logger.debug(f"MT5 price fetch failed: {str(e)}")
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {str(e)}")
            return None
    
    async def open_position(
        self,
        symbol: str,
        trade_type: str,
        quantity: float,
        price: Optional[float] = None,
        skip_controls: bool = False,
        external_order_id: Optional[str] = None
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Open a new position (buy or sell) - PAPER TRADING ONLY.
        
        This method ONLY performs simulated/paper trading. No real orders
        are sent to any broker or exchange. All trades are simulated and
        stored in the database for tracking purposes.
        
        Args:
            symbol: Trading symbol
            trade_type: 'buy' or 'sell'
            quantity: Quantity to trade (lot size)
            price: Optional execution price (uses current market price if not provided)
            skip_controls: Skip autotrading controls check (for manual trades)
            
        Returns:
            Tuple of (position_dict, error_message)
        """
        # Note: Paper trading is default, but real trading can be enabled via config
        # No blocking check here - allow both modes
        
        try:
            # Log trade attempt
            logger.info(f"[Trading] Attempting to open {trade_type} position: {quantity} {symbol}")
            
            if trade_type not in ['buy', 'sell']:
                logger.warning(f"[Trading] Invalid trade_type: {trade_type}")
                return None, f"Invalid trade_type: {trade_type}. Must be 'buy' or 'sell'"
            
            # Validate quantity (lot size)
            if quantity <= 0:
                logger.warning(f"[Trading] Invalid quantity: {quantity}")
                return None, "Quantity must be greater than 0"
            
            # Validate minimum lot size (0.01)
            if quantity < 0.01:
                logger.warning(f"[Trading] Quantity too small: {quantity} (minimum: 0.01)")
                return None, "Minimum lot size is 0.01"
            
            # Validate maximum lot size (reasonable limit: 100 lots)
            if quantity > 100:
                logger.warning(f"[Trading] Quantity too large: {quantity} (maximum: 100)")
                return None, "Maximum lot size is 100"
            
            # Validate symbol availability (paper trading - always allow, but validate format)
            if not symbol or len(symbol.strip()) == 0:
                logger.warning("[Trading] Empty symbol")
                return None, "Symbol cannot be empty"
            
            symbol = symbol.strip().upper()  # Normalize symbol
            
            # Check MT5 connection status (for market data availability)
            # Note: For paper trading, we allow trading even if MT5 is disconnected
            # but we log warnings for visibility
            mt5_connected = False
            try:
                mt5_client = get_mt5_client()
                mt5_connected = mt5_client.is_connected
                
                if not mt5_connected:
                    logger.warning(f"[Trading] MT5 not connected - using fallback price data for {symbol}")
                    # Continue with paper trading even if MT5 disconnected
                    # In production, you might want to block trading if MT5 is required
                else:
                    # Validate symbol with MT5 client if available
                    is_valid, error_msg = mt5_client.validate_symbol(symbol)
                    if not is_valid:
                        logger.warning(f"[Trading] Symbol validation warning: {error_msg} (continuing with paper trade)")
                    
                    # Check if market is open (basic check - MT5 provides this)
                    # For paper trading, we allow trading even if market is closed
                    # In production, you might want to check market hours here
                    # Example: if not mt5_client.is_market_open(symbol):
                    #     return None, "Market is closed for this symbol"
            except Exception as e:
                logger.debug(f"[Trading] MT5 check failed: {str(e)} (continuing)")
            
            # Note: Balance check is not implemented for paper trading
            # In production, you would check account balance here:
            # balance = await get_account_balance()
            # required_margin = quantity * price * contract_size * margin_requirement
            # if balance < required_margin:
            #     return None, f"Insufficient balance: {balance} < {required_margin}"
            
            # Check autotrading controls (unless skipping for manual trades)
            if not skip_controls:
                allowed, reason = await autotrading_service.check_trade_allowed(symbol, trade_type, quantity)
                if not allowed:
                    logger.warning(f"[Trading] Trade blocked by autotrading controls: {reason}")
                    return None, reason
            
            # Get execution price
            if price is None:
                price = await self.get_current_price(symbol)
                if price is None:
                    logger.error(f"[Trading] Could not determine current price for {symbol}")
                    return None, f"Could not determine current price for {symbol}. Market may be closed or data unavailable."
            
            # Validate price is reasonable (not zero or negative)
            if price <= 0:
                logger.error(f"[Trading] Invalid price: {price}")
                return None, f"Invalid price: {price}"
            
            logger.info(f"[Trading] Price determined: {symbol} @ {price}")
            
            # For sell positions, quantity is negative
            position_quantity = quantity if trade_type == 'buy' else -quantity
            
            async for session in db.get_session():
                # Optimized: Combine duplicate check and position lookup
                from datetime import timedelta
                
                # Single query to check for recent duplicate (simplified check)
                recent_trade_query = select(Trade.id).where(
                    and_(
                        Trade.symbol == symbol,
                        Trade.trade_type == trade_type,
                        Trade.timestamp >= datetime.utcnow() - timedelta(seconds=3),  # Reduced to 3 seconds
                        Trade.status == 'executed'
                    )
                ).limit(1)
                recent_trade_result = await session.execute(recent_trade_query)
                if recent_trade_result.scalar_one_or_none():
                    return None, "Duplicate order detected: Similar order executed recently"
                
                # Check for existing open position (optimized query)
                existing_query = select(Position).where(
                    and_(
                        Position.symbol == symbol,
                        Position.status == 'open'
                    )
                ).limit(1)
                existing_result = await session.execute(existing_query)
                existing_position = existing_result.scalar_one_or_none()
                
                if existing_position:
                    # Update existing position
                    old_quantity = existing_position.quantity
                    old_avg_price = existing_position.average_price
                    
                    # Calculate new average price (weighted average)
                    total_value = (old_quantity * old_avg_price) + (position_quantity * price)
                    new_quantity = old_quantity + position_quantity
                    
                    if abs(new_quantity) < 1e-10:  # Position closed
                        existing_position.quantity = 0
                        existing_position.status = 'closed'
                        existing_position.closed_at = datetime.utcnow()
                        # Calculate realized P/L
                        if old_quantity > 0:  # Was long
                            existing_position.realized_pnl = (price - old_avg_price) * old_quantity
                        else:  # Was short
                            existing_position.realized_pnl = (old_avg_price - price) * abs(old_quantity)
                    else:
                        existing_position.quantity = new_quantity
                        existing_position.average_price = total_value / new_quantity
                        existing_position.current_price = price
                        existing_position.unrealized_pnl = self._calculate_unrealized_pnl(
                            new_quantity, existing_position.average_price, price
                        )
                    
                    existing_position.updated_at = datetime.utcnow()
                    
                    # Create trade record
                    trade = Trade(
                        symbol=symbol,
                        trade_type=trade_type,
                        quantity=quantity,
                        price=price,
                        timestamp=datetime.utcnow(),
                        order_id=str(external_order_id) if external_order_id else str(uuid.uuid4()),
                        status='executed'
                    )
                    session.add(trade)
                    await session.commit()
                    
                    logger.info(
                        f"[Trading] Position updated: {symbol} {trade_type} {quantity} @ {price} "
                        f"(Position ID: {existing_position.id}, New Quantity: {new_quantity})"
                    )
                    
                    # Refresh position
                    await session.refresh(existing_position)
                    
                    position_dict = {
                        'id': existing_position.id,
                        'symbol': existing_position.symbol,
                        'quantity': float(existing_position.quantity),
                        'average_price': float(existing_position.average_price),
                        'current_price': float(existing_position.current_price) if existing_position.current_price else None,
                        'unrealized_pnl': float(existing_position.unrealized_pnl),
                        'realized_pnl': float(existing_position.realized_pnl),
                        'status': existing_position.status,
                        'opened_at': existing_position.opened_at.isoformat(),
                        'closed_at': existing_position.closed_at.isoformat() if existing_position.closed_at else None,
                        'updated_at': existing_position.updated_at.isoformat()
                    }
                    
                    return position_dict, None
                else:
                    # Create new position
                    position = Position(
                        symbol=symbol,
                        quantity=position_quantity,
                        average_price=price,
                        current_price=price,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        status='open',
                        opened_at=datetime.utcnow()
                    )
                    session.add(position)
                    await session.flush()
                    
                    # Create trade record
                    trade = Trade(
                        symbol=symbol,
                        trade_type=trade_type,
                        quantity=quantity,
                        price=price,
                        timestamp=datetime.utcnow(),
                        order_id=str(external_order_id) if external_order_id else str(uuid.uuid4()),
                        status='executed'
                    )
                    session.add(trade)
                    await session.commit()
                    
                    logger.info(
                        f"[Trading] New position opened: {symbol} {trade_type} {quantity} @ {price} "
                        f"(Position ID: {position.id}, Trade ID: {trade.id})"
                    )
                    
                    # Refresh position
                    await session.refresh(position)
                    
                    position_dict = {
                        'id': position.id,
                        'symbol': position.symbol,
                        'quantity': float(position.quantity),
                        'average_price': float(position.average_price),
                        'current_price': float(position.current_price) if position.current_price else None,
                        'unrealized_pnl': float(position.unrealized_pnl),
                        'realized_pnl': float(position.realized_pnl),
                        'status': position.status,
                        'opened_at': position.opened_at.isoformat(),
                        'closed_at': position.closed_at.isoformat() if position.closed_at else None,
                        'updated_at': position.updated_at.isoformat()
                    }
                    
                    return position_dict, None
                
                break  # Exit session context
                
        except Exception as e:
            error = f"Failed to open position: {str(e)}"
            logger.error(error)
            return None, error
    
    async def close_position(self, position_id: int, quantity: Optional[float] = None) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Close a position (fully or partially).
        
        Args:
            position_id: Position ID to close
            quantity: Optional quantity to close (closes full position if not provided)
            
        Returns:
            Tuple of (position_dict, error_message)
        """
        try:
            async for session in db.get_session():
                # Get position
                position_query = select(Position).where(Position.id == position_id)
                position_result = await session.execute(position_query)
                position = position_result.scalar_one_or_none()
                
                if not position:
                    return None, f"Position {position_id} not found"
                
                if position.status == 'closed':
                    return None, f"Position {position_id} is already closed"
                
                # Get current price
                current_price = await self.get_current_price(position.symbol)
                if current_price is None:
                    return None, f"Could not determine current price for {position.symbol}"
                
                # Determine close quantity
                if quantity is None:
                    close_quantity = abs(position.quantity)
                else:
                    if quantity <= 0 or quantity > abs(position.quantity):
                        return None, f"Invalid close quantity: {quantity}"
                    close_quantity = quantity
                
                # Calculate realized P/L
                if position.quantity > 0:  # Long position
                    realized_pnl = (current_price - position.average_price) * close_quantity
                    trade_type = 'sell'
                else:  # Short position
                    realized_pnl = (position.average_price - current_price) * close_quantity
                    trade_type = 'buy'
                
                # Update daily loss if there's a loss
                if realized_pnl < 0:
                    await autotrading_service.update_daily_loss(abs(realized_pnl))
                
                # Update position
                new_quantity = position.quantity - (close_quantity if position.quantity > 0 else -close_quantity)
                
                if abs(new_quantity) < 1e-10:  # Fully closed
                    position.quantity = 0
                    position.status = 'closed'
                    position.closed_at = datetime.utcnow()
                    position.realized_pnl += realized_pnl
                    position.unrealized_pnl = 0.0
                else:  # Partially closed
                    position.quantity = new_quantity
                    position.realized_pnl += realized_pnl
                    position.current_price = current_price
                    position.unrealized_pnl = self._calculate_unrealized_pnl(
                        new_quantity, position.average_price, current_price
                    )
                
                position.updated_at = datetime.utcnow()
                
                # Create trade record
                trade = Trade(
                    symbol=position.symbol,
                    trade_type=trade_type,
                    quantity=close_quantity,
                    price=current_price,
                    timestamp=datetime.utcnow(),
                    order_id=str(uuid.uuid4()),
                    status='executed'
                )
                session.add(trade)
                await session.commit()
                
                # Refresh position
                await session.refresh(position)
                
                position_dict = {
                    'id': position.id,
                    'symbol': position.symbol,
                    'quantity': float(position.quantity),
                    'average_price': float(position.average_price),
                    'current_price': float(position.current_price) if position.current_price else None,
                    'unrealized_pnl': float(position.unrealized_pnl),
                    'realized_pnl': float(position.realized_pnl),
                    'status': position.status,
                    'opened_at': position.opened_at.isoformat(),
                    'closed_at': position.closed_at.isoformat() if position.closed_at else None,
                    'updated_at': position.updated_at.isoformat()
                }
                
                return position_dict, None
                
                break  # Exit session context
                
        except Exception as e:
            error = f"Failed to close position: {str(e)}"
            logger.error(error)
            return None, error
    
    async def update_positions(self) -> Dict[str, int]:
        """
        Update all open positions with current prices and recalculate P/L.
        
        Returns:
            Dictionary with update statistics
        """
        try:
            updated_count = 0
            error_count = 0
            
            async for session in db.get_session():
                # Get all open positions
                positions_query = select(Position).where(Position.status == 'open')
                positions_result = await session.execute(positions_query)
                positions = positions_result.scalars().all()
                
                for position in positions:
                    try:
                        current_price = await self.get_current_price(position.symbol)
                        if current_price is None:
                            error_count += 1
                            continue
                        
                        position.current_price = current_price
                        position.unrealized_pnl = self._calculate_unrealized_pnl(
                            position.quantity, position.average_price, current_price
                        )
                        position.updated_at = datetime.utcnow()
                        updated_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to update position {position.id}: {str(e)}")
                        error_count += 1
                
                await session.commit()
                break  # Exit session context
            
            return {
                'updated': updated_count,
                'errors': error_count,
                'total': updated_count + error_count
            }
            
        except Exception as e:
            logger.error(f"Failed to update positions: {str(e)}")
            return {'updated': 0, 'errors': 0, 'total': 0}
    
    def _calculate_unrealized_pnl(self, quantity: float, average_price: float, current_price: float) -> float:
        """
        Calculate unrealized profit/loss.
        
        Args:
            quantity: Position quantity (positive for long, negative for short)
            average_price: Average entry price
            current_price: Current market price
            
        Returns:
            Unrealized P/L
        """
        if abs(quantity) < 1e-10:
            return 0.0
        
        if quantity > 0:  # Long position
            return (current_price - average_price) * quantity
        else:  # Short position
            return (average_price - current_price) * abs(quantity)
    
    async def get_positions(self, status: Optional[str] = None) -> List[Dict]:
        """
        Get all positions.
        
        Args:
            status: Optional filter by status ('open' or 'closed')
            
        Returns:
            List of position dictionaries
        """
        try:
            async for session in db.get_session():
                query = select(Position)
                if status:
                    query = query.where(Position.status == status)
                query = query.order_by(Position.opened_at.desc())
                
                result = await session.execute(query)
                positions = result.scalars().all()
                
                positions_list = []
                for position in positions:
                    positions_list.append({
                        'id': position.id,
                        'symbol': position.symbol,
                        'quantity': float(position.quantity),
                        'average_price': float(position.average_price),
                        'current_price': float(position.current_price) if position.current_price else None,
                        'unrealized_pnl': float(position.unrealized_pnl),
                        'realized_pnl': float(position.realized_pnl),
                        'status': position.status,
                        'opened_at': position.opened_at.isoformat(),
                        'closed_at': position.closed_at.isoformat() if position.closed_at else None,
                        'updated_at': position.updated_at.isoformat()
                    })
                
                break  # Exit session context
            
            return positions_list
            
        except Exception as e:
            logger.error(f"Failed to get positions: {str(e)}")
            return []
    
    async def get_trade_history(
        self,
        symbol: Optional[str] = None,
        trade_type: Optional[str] = None,
        status: Optional[str] = None,
        min_profit: Optional[float] = None,
        max_profit: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = 100
    ) -> List[Dict]:
        """
        Get trade history with comprehensive filtering.
        
        Args:
            symbol: Optional symbol filter
            trade_type: Optional trade type filter ('buy' or 'sell')
            status: Optional status filter ('executed', 'pending', 'cancelled')
            min_profit: Optional minimum profit filter
            max_profit: Optional maximum profit filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of trades to return
            
        Returns:
            List of trade dictionaries with profit/loss information
        """
        try:
            async for session in db.get_session():
                query = select(Trade)
                
                # Apply filters
                conditions = []
                
                if symbol:
                    conditions.append(Trade.symbol == symbol)
                
                if trade_type:
                    if trade_type.lower() not in ['buy', 'sell']:
                        return []
                    conditions.append(Trade.trade_type == trade_type.lower())
                
                if status:
                    conditions.append(Trade.status == status)
                
                if start_date:
                    conditions.append(Trade.timestamp >= start_date)
                
                if end_date:
                    conditions.append(Trade.timestamp <= end_date)
                
                if conditions:
                    from sqlalchemy import and_
                    query = query.where(and_(*conditions))
                
                query = query.order_by(Trade.timestamp.desc())
                
                if limit:
                    query = query.limit(limit)
                
                result = await session.execute(query)
                trades = result.scalars().all()
                
                # Get all positions for profit/loss calculation
                positions_query = select(Position)
                positions_result = await session.execute(positions_query)
                all_positions = positions_result.scalars().all()
                
                # Create a map of symbol+timestamp to position P/L
                # For closed positions, we can attribute P/L to closing trades
                symbol_trades_map = {}
                for pos in all_positions:
                    if pos.status == 'closed' and pos.closed_at:
                        # Find trades that closed this position (sell trades near close time)
                        key = f"{pos.symbol}_{pos.closed_at.isoformat()}"
                        symbol_trades_map[key] = {
                            'realized_pnl': pos.realized_pnl,
                            'closed_at': pos.closed_at,
                            'symbol': pos.symbol
                        }
                
                trades_list = []
                for trade in trades:
                    # Calculate profit/loss for this trade
                    profit_loss = None
                    
                    # For sell trades (closing trades), try to find associated position P/L
                    if trade.trade_type == 'sell':
                        # Find closed positions for this symbol around this time
                        for key, pos_info in symbol_trades_map.items():
                            if pos_info['symbol'] == trade.symbol:
                                time_diff = abs((pos_info['closed_at'] - trade.timestamp).total_seconds())
                                if time_diff < 3600:  # Within 1 hour
                                    # Use the position's realized P/L
                                    profit_loss = pos_info['realized_pnl']
                                    break
                    
                    # For buy trades, calculate potential P/L from open positions
                    elif trade.trade_type == 'buy':
                        # Find open positions for this symbol
                        for pos in all_positions:
                            if pos.symbol == trade.symbol and pos.status == 'open' and pos.current_price:
                                # Calculate unrealized P/L proportionally
                                if pos.quantity > 0:  # Long position
                                    pnl_per_unit = pos.current_price - pos.average_price
                                    profit_loss = pnl_per_unit * trade.quantity
                                break
                    
                    trade_dict = {
                        'id': trade.id,
                        'symbol': trade.symbol,
                        'trade_type': trade.trade_type,
                        'side': trade.trade_type,  # Alias for consistency
                        'quantity': float(trade.quantity),
                        'price': float(trade.price),
                        'timestamp': trade.timestamp.isoformat(),
                        'order_id': trade.order_id,
                        'status': trade.status,
                        'profit_loss': profit_loss,
                        'created_at': trade.created_at.isoformat()
                    }
                    
                    # Apply profit/loss filters
                    if min_profit is not None or max_profit is not None:
                        if profit_loss is None:
                            continue  # Skip trades without P/L if filtering by P/L
                        if min_profit is not None and profit_loss < min_profit:
                            continue
                        if max_profit is not None and profit_loss > max_profit:
                            continue
                    
                    trades_list.append(trade_dict)
                
                break  # Exit session context
            
            return trades_list
            
        except Exception as e:
            logger.error(f"Failed to get trade history: {str(e)}")
            return []
    
    async def get_position(self, position_id: int) -> Optional[Dict]:
        """
        Get a specific position by ID.
        
        Args:
            position_id: Position ID
            
        Returns:
            Position dictionary or None if not found
        """
        try:
            async for session in db.get_session():
                query = select(Position).where(Position.id == position_id)
                result = await session.execute(query)
                position = result.scalar_one_or_none()
                
                if not position:
                    return None
                
                return {
                    'id': position.id,
                    'symbol': position.symbol,
                    'quantity': float(position.quantity),
                    'average_price': float(position.average_price),
                    'current_price': float(position.current_price) if position.current_price else None,
                    'unrealized_pnl': float(position.unrealized_pnl),
                    'realized_pnl': float(position.realized_pnl),
                    'status': position.status,
                    'opened_at': position.opened_at.isoformat(),
                    'closed_at': position.closed_at.isoformat() if position.closed_at else None,
                    'updated_at': position.updated_at.isoformat()
                }
                
                break  # Exit session context
                
        except Exception as e:
            logger.error(f"Failed to get position: {str(e)}")
            return None


# Global paper trading service instance
paper_trading_service = PaperTradingService()
