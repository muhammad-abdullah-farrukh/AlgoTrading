"""
WebSocket endpoints for real-time data streaming.

IMPORTANT: These are WebSocket endpoints, NOT HTTP endpoints.

⚠️  WebSocket routes CANNOT be accessed via HTTP GET requests.
Swagger UI displays them for documentation purposes only.

✅ Correct Usage:
   - Use WebSocket client libraries (e.g., JavaScript WebSocket API)
   - Connect using ws:// or wss:// protocol
   - Example: ws://localhost:8000/ws/ticks/EURUSD

❌ Incorrect Usage:
   - HTTP GET requests will return 404 or error messages
   - Swagger UI "Try it out" will not work for WebSocket endpoints

Available Endpoints:
- ws://localhost:8000/ws/ticks/{symbol} - Live tick data
- ws://localhost:8000/ws/positions - Position updates
- ws://localhost:8000/ws/trades - Trade updates
- ws://localhost:8000/ws/general - Bidirectional communication
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Optional
from datetime import datetime
import json
import asyncio
from app.websocket import manager
from app.utils.logging import get_logger
from app.services import get_mt5_client
from app.database import db
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import Trade, Position
from sqlalchemy import select

logger = get_logger(__name__)
router = APIRouter(include_in_schema=True)


# HTTP GET guards for WebSocket paths - return clear error messages
@router.get("/ws/ticks/{symbol}")
async def http_guard_ticks(symbol: str):
    """
    HTTP guard for WebSocket endpoint.
    
    This endpoint is a WebSocket and cannot be accessed via HTTP GET.
    Use WebSocket protocol: ws://localhost:8000/ws/ticks/{symbol}
    """
    raise HTTPException(
        status_code=400,
        detail={
            "error": "This endpoint is a WebSocket and cannot be accessed via HTTP",
            "message": f"Use WebSocket protocol instead: ws://localhost:8000/ws/ticks/{symbol}",
            "correct_usage": "Connect using a WebSocket client (e.g., JavaScript WebSocket API)",
            "example": f"const ws = new WebSocket('ws://localhost:8000/ws/ticks/{symbol}');"
        }
    )


@router.get("/ws/positions")
async def http_guard_positions():
    """
    HTTP guard for WebSocket endpoint.
    
    This endpoint is a WebSocket and cannot be accessed via HTTP GET.
    Use WebSocket protocol: ws://localhost:8000/ws/positions
    """
    raise HTTPException(
        status_code=400,
        detail={
            "error": "This endpoint is a WebSocket and cannot be accessed via HTTP",
            "message": "Use WebSocket protocol instead: ws://localhost:8000/ws/positions",
            "correct_usage": "Connect using a WebSocket client (e.g., JavaScript WebSocket API)",
            "example": "const ws = new WebSocket('ws://localhost:8000/ws/positions');"
        }
    )


@router.get("/ws/trades")
async def http_guard_trades():
    """
    HTTP guard for WebSocket endpoint.
    
    This endpoint is a WebSocket and cannot be accessed via HTTP GET.
    Use WebSocket protocol: ws://localhost:8000/ws/trades
    """
    raise HTTPException(
        status_code=400,
        detail={
            "error": "This endpoint is a WebSocket and cannot be accessed via HTTP",
            "message": "Use WebSocket protocol instead: ws://localhost:8000/ws/trades",
            "correct_usage": "Connect using a WebSocket client (e.g., JavaScript WebSocket API)",
            "example": "const ws = new WebSocket('ws://localhost:8000/ws/trades');"
        }
    )


@router.get("/ws/general")
async def http_guard_general():
    """
    HTTP guard for WebSocket endpoint.
    
    This endpoint is a WebSocket and cannot be accessed via HTTP GET.
    Use WebSocket protocol: ws://localhost:8000/ws/general
    """
    raise HTTPException(
        status_code=400,
        detail={
            "error": "This endpoint is a WebSocket and cannot be accessed via HTTP",
            "message": "Use WebSocket protocol instead: ws://localhost:8000/ws/general",
            "correct_usage": "Connect using a WebSocket client (e.g., JavaScript WebSocket API)",
            "example": "const ws = new WebSocket('ws://localhost:8000/ws/general');"
        }
    )


@router.websocket("/ws/ticks/{symbol}")
async def websocket_ticks(websocket: WebSocket, symbol: str):
    """
    WebSocket endpoint for streaming live tick data.
    
    **Purpose**: Streams real-time tick-by-tick market data for a specific trading symbol.
    This includes bid/ask prices, volume, and timestamps for each market tick.
    
    **Message Types**:
    - `connected`: Initial connection confirmation with stream type and symbol
    - `tick`: Real-time tick data with bid, ask, volume, and timestamp
    - `error`: Error messages if streaming fails
    
    **Update Frequency**: Updates are sent approximately every 1 second when new ticks are available.
    Only new ticks (not previously sent) are transmitted to minimize bandwidth.
    
    **Supported Symbols**: EURUSD, BTCUSDT, AAPL, and other symbols available via MT5.
    
    Args:
        websocket: WebSocket connection
        symbol: Trading symbol (e.g., EURUSD, BTCUSDT, AAPL)
    """
    await manager.connect(websocket, stream_type='ticks')
    
    try:
        # Normalize symbol (uppercase, strip whitespace)
        symbol = symbol.upper().strip()
        
        # Get MT5 client
        mt5_client = get_mt5_client()
        
        # Validate symbol if MT5 is connected (mock always allows)
        if mt5_client.is_connected:
            is_valid, error_msg = mt5_client.validate_symbol(symbol)
            if not is_valid:
                await manager.send_personal_message({
                    'type': 'error',
                    'message': f"Invalid symbol: {error_msg}",
                    'symbol': symbol,
                }, websocket)
                logger.warning(f"Invalid symbol requested: {symbol} - {error_msg}")
                return
        
        # Send initial connection confirmation with normalized symbol
        await manager.send_personal_message({
            'type': 'connected',
            'stream': 'ticks',
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
        }, websocket)
        
        # Stream ticks
        last_tick_time = None
        last_tick_hash = None  # Hash of tick data to prevent duplicates
        last_status_log = datetime.utcnow()
        ticks_sent_count = 0
        
        while True:
            # Check connection state before processing
            if websocket.client_state.name == 'DISCONNECTED':
                break
                
            try:
                # Get latest ticks from MT5
                from datetime import timedelta
                start_time = datetime.utcnow() - timedelta(seconds=5)
                
                ticks = mt5_client.get_ticks(symbol, start=start_time, count=100)
                
                if not ticks.empty and len(ticks) > 0:
                    # Get the latest tick
                    latest_tick = ticks.iloc[-1]
                    
                    # Convert timestamp to datetime for comparison
                    tick_time = latest_tick['time']
                    if hasattr(tick_time, 'to_pydatetime'):
                        tick_time = tick_time.to_pydatetime()
                    elif hasattr(tick_time, 'timestamp'):
                        tick_time = datetime.fromtimestamp(tick_time.timestamp())
                    
                    # Create hash of tick data to prevent duplicates (bid+ask+time)
                    tick_hash = hash((float(latest_tick['bid']), float(latest_tick['ask']), str(tick_time)))
                    
                    # Only send if it's a new tick (different time or different price)
                    is_new_tick = (
                        last_tick_time is None or 
                        tick_time > last_tick_time or
                        (tick_time == last_tick_time and tick_hash != last_tick_hash)
                    )
                    
                    if is_new_tick:
                        # Ensure timestamp is in ISO format with timezone
                        if hasattr(tick_time, 'isoformat'):
                            tick_timestamp = tick_time.isoformat()
                        elif hasattr(tick_time, 'timestamp'):
                            tick_timestamp = datetime.fromtimestamp(tick_time.timestamp()).isoformat()
                        else:
                            tick_timestamp = datetime.utcnow().isoformat()
                        
                        tick_message = {
                            'type': 'tick',
                            'symbol': symbol,  # Use normalized symbol
                            'timestamp': tick_timestamp,
                            'bid': float(latest_tick['bid']),
                            'ask': float(latest_tick['ask']),
                            'volume': int(latest_tick['volume']),
                        }
                        
                        # Check connection before sending
                        if websocket.client_state.name == 'CONNECTED':
                            await manager.send_personal_message(tick_message, websocket)
                            last_tick_time = tick_time
                            last_tick_hash = tick_hash
                            ticks_sent_count += 1
                            
                            # Log new tick detection periodically (every 30 seconds)
                            now = datetime.utcnow()
                            if (now - last_status_log).total_seconds() >= 30:
                                logger.debug(f"Tick stream active for {symbol}: {ticks_sent_count} ticks sent in last 30s")
                                last_status_log = now
                                ticks_sent_count = 0
                
                # Wait before next update
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                # Normal cancellation during shutdown/disconnect - exit immediately
                logger.debug(f"WebSocket tick stream cancelled: {symbol}")
                break  # Exit loop immediately, do NOT restart polling
            except WebSocketDisconnect:
                # Client disconnected - exit loop immediately
                logger.debug(f"WebSocket disconnect detected in tick stream: {symbol}")
                break
            except Exception as e:
                logger.error(f"Error streaming ticks for {symbol}: {str(e)}")
                # Try to send error message, but exit if connection is closed
                try:
                    await manager.send_personal_message({
                        'type': 'error',
                        'message': f"Error streaming ticks: {str(e)}",
                    }, websocket)
                except (WebSocketDisconnect, asyncio.CancelledError):
                    # Connection closed or cancelled - exit immediately
                    break
                except Exception:
                    # Other send errors - connection likely closed, exit
                    break
                
                # Wait before retrying, but exit if cancelled
                try:
                    await asyncio.sleep(5)  # Wait before retrying
                except asyncio.CancelledError:
                    # Cancelled during retry wait - exit immediately
                    break
                
    except asyncio.CancelledError:
        # Normal cancellation - log as debug, not error
        logger.debug(f"WebSocket cancelled: ticks/{symbol}")
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: ticks/{symbol}")
    except Exception as e:
        logger.error(f"WebSocket error for ticks/{symbol}: {str(e)}")
    finally:
        try:
            await manager.disconnect(websocket)
        except Exception:
            # Silently ignore disconnect cleanup errors (connection already closed)
            pass


@router.websocket("/ws/positions")
async def websocket_positions(websocket: WebSocket):
    """
    WebSocket endpoint for streaming position updates.
    
    **Purpose**: Streams real-time updates about open trading positions.
    Provides position details including quantity, average price, current price,
    unrealized P/L, and realized P/L.
    
    **Message Types**:
    - `connected`: Initial connection confirmation with stream type
    - `positions`: Array of position objects with full position details
    - `error`: Error messages if streaming fails
    
    **Update Frequency**: Updates are sent every 2 seconds, but only when positions
    have changed (new positions opened, existing positions closed, or P/L updated).
    
    **Position Data Includes**:
    - Position ID, symbol, quantity
    - Average entry price and current market price
    - Unrealized and realized profit/loss
    - Position status and open timestamp
    """
    await manager.connect(websocket, stream_type='positions')
    
    try:
        # Send initial connection confirmation
        await manager.send_personal_message({
            'type': 'connected',
            'stream': 'positions',
            'timestamp': datetime.utcnow().isoformat(),
        }, websocket)
        
        # Stream position updates
        last_update_hash = None
        while True:
            # Check connection state before processing
            if websocket.client_state.name == 'DISCONNECTED':
                break
                
            try:
                # Get positions from database - each iteration gets a fresh session
                # Session is created, used, and closed within this try block
                positions_data = []
                db_error = None
                
                try:
                    # Create a new session for this iteration only
                    async for session in db.get_session():
                        try:
                            # Execute query with timeout protection
                            result = await session.execute(
                                select(Position).where(Position.status == 'open')
                            )
                            # Materialize all results while session is active
                            positions = result.scalars().all()
                            
                            # Convert to dict immediately - fully materialize data
                            # This ensures all DB access happens while session is open
                            positions_data = [{
                                'id': pos.id,
                                'symbol': pos.symbol,
                                'quantity': float(pos.quantity),
                                'average_price': float(pos.average_price),
                                'current_price': float(pos.current_price) if pos.current_price else None,
                                'unrealized_pnl': float(pos.unrealized_pnl),
                                'realized_pnl': float(pos.realized_pnl),
                                'status': pos.status,
                                'opened_at': pos.opened_at.isoformat() if pos.opened_at else None,
                            } for pos in positions]
                            
                            # Data is fully materialized - session can close safely
                        finally:
                            # Session context manager ensures cleanup
                            # Exit immediately after data is fetched
                            break
                except Exception as db_ex:
                    # Isolate DB errors - don't kill WebSocket
                    db_error = db_ex
                    logger.warning(f"Database error in positions stream (will retry): {str(db_ex)}")
                    # positions_data remains empty, will skip this frame
                
                # Session is guaranteed to be closed here
                # Only proceed if DB query succeeded
                if db_error is None:
                    # Create hash of entire position data to detect any changes (including P/L updates)
                    # Sort by ID for consistent hashing
                    positions_sorted = sorted(positions_data, key=lambda p: p['id'])
                    positions_hash = hash(tuple(
                        (p['id'], p['quantity'], p['current_price'], p['unrealized_pnl'], p['realized_pnl'])
                        for p in positions_sorted
                    ))
                    
                    # Only send if positions changed (new positions, closed positions, or P/L updates)
                    if last_update_hash != positions_hash:
                        # Check connection before sending
                        if websocket.client_state.name == 'CONNECTED':
                            try:
                                await manager.send_personal_message({
                                    'type': 'positions',
                                    'data': positions_data,
                                    'timestamp': datetime.utcnow().isoformat(),
                                }, websocket)
                                last_update_hash = positions_hash
                            except Exception as send_err:
                                # Send error - connection may be closed
                                logger.debug(f"Failed to send positions update: {str(send_err)}")
                                raise  # Will be caught by outer handler
                
                # Wait before next update (session is definitely closed)
                await asyncio.sleep(2)
                
            except asyncio.CancelledError:
                # Normal cancellation during shutdown/disconnect - exit immediately
                logger.debug("WebSocket positions stream cancelled")
                break  # Exit loop immediately, do NOT restart polling
            except WebSocketDisconnect:
                # Client disconnected - exit loop immediately
                logger.debug("WebSocket disconnect detected in positions stream")
                break
            except Exception as e:
                # Non-DB errors (e.g., WebSocket send errors)
                logger.error(f"Error in positions stream: {str(e)}")
                # Try to send error message, but exit if connection is closed
                try:
                    await manager.send_personal_message({
                        'type': 'error',
                        'message': f"Error streaming positions: {str(e)}",
                    }, websocket)
                except (WebSocketDisconnect, asyncio.CancelledError):
                    # Connection closed or cancelled - exit immediately
                    break
                except Exception:
                    # Other send errors - connection likely closed, exit
                    break
                
                # Wait before retrying, but exit if cancelled
                try:
                    await asyncio.sleep(5)  # Wait before retrying
                except asyncio.CancelledError:
                    # Cancelled during retry wait - exit immediately
                    break
                
    except asyncio.CancelledError:
        # Normal cancellation - log as debug, not error
        logger.debug("WebSocket cancelled: positions")
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: positions")
    except Exception as e:
        logger.error(f"WebSocket error for positions: {str(e)}")
    finally:
        try:
            await manager.disconnect(websocket)
        except Exception:
            # Silently ignore disconnect cleanup errors (connection already closed)
            pass


@router.websocket("/ws/trades")
async def websocket_trades(websocket: WebSocket):
    """
    WebSocket endpoint for streaming trade updates.
    
    **Purpose**: Streams real-time updates about executed trades.
    Provides trade execution details including symbol, trade type (buy/sell),
    quantity, price, and execution status.
    
    **Message Types**:
    - `connected`: Initial connection confirmation with stream type
    - `trades`: Array of trade objects with full trade execution details
    - `error`: Error messages if streaming fails
    
    **Update Frequency**: Updates are sent every 1 second, but only when new trades
    are detected. The stream includes the 10 most recent trades.
    
    **Trade Data Includes**:
    - Trade ID, symbol, trade type (buy/sell)
    - Quantity and execution price
    - Order ID and execution status
    - Trade timestamp
    """
    await manager.connect(websocket, stream_type='trades')
    
    try:
        # Send initial connection confirmation
        await manager.send_personal_message({
            'type': 'connected',
            'stream': 'trades',
            'timestamp': datetime.utcnow().isoformat(),
        }, websocket)
        
        # Stream trade updates
        last_trade_hash = None
        while True:
            # Check connection state before processing
            if websocket.client_state.name == 'DISCONNECTED':
                break
                
            try:
                # Get latest trades from database - each iteration gets a fresh session
                # Session is created, used, and closed within this try block
                trades_data = []
                db_error = None
                
                try:
                    # Create a new session for this iteration only
                    async for session in db.get_session():
                        try:
                            # Execute query - materialize results while session is active
                            query = select(Trade).order_by(Trade.timestamp.desc()).limit(10)
                            result = await session.execute(query)
                            trades = result.scalars().all()
                            
                            # Always convert trades to dict (even if no new trades, send current state)
                            # All DB access happens while session is open
                            trades_data = [{
                                'id': trade.id,
                                'symbol': trade.symbol,
                                'trade_type': trade.trade_type,
                                'quantity': float(trade.quantity),
                                'price': float(trade.price),
                                'timestamp': trade.timestamp.isoformat() if trade.timestamp else None,
                                'order_id': trade.order_id,
                                'status': trade.status,
                            } for trade in trades]
                            
                            # Data is fully materialized - session can close safely
                        finally:
                            # Session context manager ensures cleanup
                            # Exit immediately after data is fetched
                            break
                except Exception as db_ex:
                    # Isolate DB errors - don't kill WebSocket
                    db_error = db_ex
                    logger.warning(f"Database error in trades stream (will retry): {str(db_ex)}")
                    # trades_data remains empty, will skip this frame
                
                # Session is guaranteed to be closed here
                # Only proceed if DB query succeeded
                if db_error is None:
                    # Create hash of trade IDs to detect new trades
                    # Send if trades changed (new trades added)
                    if trades_data:
                        trades_hash = hash(tuple(sorted([t['id'] for t in trades_data])))
                        if last_trade_hash != trades_hash:
                            # Check connection before sending
                            if websocket.client_state.name == 'CONNECTED':
                                try:
                                    await manager.send_personal_message({
                                        'type': 'trades',
                                        'data': trades_data,
                                        'timestamp': datetime.utcnow().isoformat(),
                                    }, websocket)
                                    last_trade_hash = trades_hash
                                except Exception as send_err:
                                    # Send error - connection may be closed
                                    logger.debug(f"Failed to send trades update: {str(send_err)}")
                                    raise  # Will be caught by outer handler
                    elif last_trade_hash is not None:
                        # No trades but we had trades before - send empty array to notify client
                        if websocket.client_state.name == 'CONNECTED':
                            try:
                                await manager.send_personal_message({
                                    'type': 'trades',
                                    'data': [],
                                    'timestamp': datetime.utcnow().isoformat(),
                                }, websocket)
                                last_trade_hash = None
                            except Exception as send_err:
                                logger.debug(f"Failed to send empty trades update: {str(send_err)}")
                
                # Wait before next update (session is definitely closed)
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                # Normal cancellation during shutdown/disconnect - exit immediately
                logger.debug("WebSocket trades stream cancelled")
                break  # Exit loop immediately, do NOT restart polling
            except WebSocketDisconnect:
                # Client disconnected - exit loop immediately
                logger.debug("WebSocket disconnect detected in trades stream")
                break
            except Exception as e:
                # Non-DB errors (e.g., WebSocket send errors)
                logger.error(f"Error in trades stream: {str(e)}")
                # Try to send error message, but exit if connection is closed
                try:
                    await manager.send_personal_message({
                        'type': 'error',
                        'message': f"Error streaming trades: {str(e)}",
                    }, websocket)
                except (WebSocketDisconnect, asyncio.CancelledError):
                    # Connection closed or cancelled - exit immediately
                    break
                except Exception:
                    # Other send errors - connection likely closed, exit
                    break
                
                # Wait before retrying, but exit if cancelled
                try:
                    await asyncio.sleep(5)  # Wait before retrying
                except asyncio.CancelledError:
                    # Cancelled during retry wait - exit immediately
                    break
                
    except asyncio.CancelledError:
        # Normal cancellation - log as debug, not error
        logger.debug("WebSocket cancelled: trades")
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: trades")
    except Exception as e:
        logger.error(f"WebSocket error for trades: {str(e)}")
    finally:
        try:
            await manager.disconnect(websocket)
        except Exception:
            # Silently ignore disconnect cleanup errors (connection already closed)
            pass


@router.websocket("/ws/general")
async def websocket_general(websocket: WebSocket):
    """
    General WebSocket endpoint for bidirectional communication.
    
    **Purpose**: Provides a general-purpose WebSocket connection for bidirectional
    communication. Supports heartbeat/ping-pong messages and custom message handling.
    Useful for connection health checks and custom client-server communication.
    
    **Message Types (Received)**:
    - `heartbeat`: Client heartbeat - server responds with `heartbeat_ack`
    - `ping`: Client ping - server responds with `pong`
    - Any other type: Server responds with `message_received` acknowledgment
    
    **Message Types (Sent)**:
    - `connected`: Initial connection confirmation with stream type
    - `heartbeat_ack`: Response to client heartbeat messages
    - `pong`: Response to client ping messages
    - `message_received`: Acknowledgment for other message types
    - `error`: Error messages for invalid JSON or other errors
    
    **Update Frequency**: Server automatically sends heartbeat messages every 30 seconds
    to keep the connection alive. Client can send heartbeat/ping messages at any time.
    
    **Use Cases**:
    - Connection health monitoring
    - Custom bidirectional messaging
    - Testing WebSocket connectivity
    """
    await manager.connect(websocket, stream_type='general')
    
    try:
        # Send initial connection confirmation
        await manager.send_personal_message({
            'type': 'connected',
            'stream': 'general',
            'timestamp': datetime.utcnow().isoformat(),
        }, websocket)
        
        # Listen for messages
        while True:
            # Check connection state before processing
            if websocket.client_state.name == 'DISCONNECTED':
                break
                
            try:
                # Receive message from client
                data = await websocket.receive_json()
                
                message_type = data.get('type')
                
                # Handle heartbeat response
                if message_type == 'heartbeat':
                    await manager.send_personal_message({
                        'type': 'heartbeat_ack',
                        'timestamp': datetime.utcnow().isoformat(),
                    }, websocket)
                
                # Handle ping
                elif message_type == 'ping':
                    await manager.send_personal_message({
                        'type': 'pong',
                        'timestamp': datetime.utcnow().isoformat(),
                    }, websocket)
                
                # Handle other message types
                else:
                    await manager.send_personal_message({
                        'type': 'message_received',
                        'original_type': message_type,
                        'timestamp': datetime.utcnow().isoformat(),
                    }, websocket)
                    
            except asyncio.CancelledError:
                # Normal cancellation during shutdown/disconnect - exit immediately
                logger.debug("WebSocket general stream cancelled")
                break  # Exit loop immediately, do NOT restart polling
            except WebSocketDisconnect:
                # Client disconnected - exit loop immediately
                logger.debug("WebSocket disconnect detected in general stream")
                break
            except json.JSONDecodeError:
                logger.warning("Invalid JSON received from WebSocket")
                # Try to send error message, but exit if connection is closed
                try:
                    await manager.send_personal_message({
                        'type': 'error',
                        'message': 'Invalid JSON format',
                    }, websocket)
                except (WebSocketDisconnect, asyncio.CancelledError):
                    # Connection closed or cancelled - exit immediately
                    break
                except Exception:
                    # Other send errors - connection likely closed, exit
                    break
                
    except asyncio.CancelledError:
        # Normal cancellation - log as debug, not error
        logger.debug("WebSocket cancelled: general")
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: general")
    except Exception as e:
        logger.error(f"WebSocket error for general: {str(e)}")
    finally:
        try:
            await manager.disconnect(websocket)
        except Exception:
            # Silently ignore disconnect cleanup errors (connection already closed)
            pass
