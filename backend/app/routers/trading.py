"""Paper trading endpoints."""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from typing import Optional, Tuple
from pydantic import BaseModel, Field
from app.services.paper_trading import paper_trading_service
from app.services.live_trade_logger import live_trade_logger, LiveTradeEvent
from app.utils.logging import get_logger
import csv
import json
import io
from datetime import datetime
from app.database import db
from app.models import TradingSettings
from sqlalchemy import select

logger = get_logger(__name__)
router = APIRouter(prefix="/api/trade", tags=["trading"])


class BuyRequest(BaseModel):
    """Request model for buy order."""
    symbol: str = Field(..., description="Trading symbol")
    quantity: float = Field(..., gt=0, description="Quantity to buy")
    price: Optional[float] = Field(None, gt=0, description="Optional execution price (uses market price if not provided)")


class SellRequest(BaseModel):
    """Request model for sell order."""
    symbol: str = Field(..., description="Trading symbol")
    quantity: float = Field(..., gt=0, description="Quantity to sell")
    price: Optional[float] = Field(None, gt=0, description="Optional execution price (uses market price if not provided)")


class CloseRequest(BaseModel):
    """Request model for closing position."""
    quantity: Optional[float] = Field(None, gt=0, description="Quantity to close (closes full position if not provided)")


class TradingModeResponse(BaseModel):
    live_mode: bool
    updated_at: Optional[str] = None


class TradingModeRequest(BaseModel):
    live_mode: bool = Field(..., description="True for live mode, False for demo/paper mode")


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


def _mt5_can_execute_live_orders(mt5_client, server_settings) -> Tuple[bool, Optional[str]]:
    try:
        status = mt5_client.get_detailed_status() if hasattr(mt5_client, "get_detailed_status") else {}
        account_info = (status or {}).get('account_info') or {}

        trading_enabled = bool((status or {}).get('trading_enabled'))
        if not trading_enabled:
            server_name = account_info.get('server')
            login = account_info.get('login')
            return False, (
                "MT5 trading is not enabled for this account/session. "
                f"(trade_allowed=false, login={login}, server={server_name})"
            )

        server_name = account_info.get('server')
        is_demo = _is_demo_account(account_info)

        if bool(server_settings.mt5_real_trading):
            return True, None

        if is_demo:
            return True, None

        return False, "MT5 real trading is disabled by server configuration (set MT5_REAL_TRADING=true)."
    except Exception:
        return False, "Failed to validate MT5 account status."


async def _get_or_create_trading_settings() -> TradingSettings:
    async for session in db.get_session():
        result = await session.execute(select(TradingSettings).limit(1))
        settings = result.scalar_one_or_none()
        if settings is None:
            settings = TradingSettings(live_mode=False)
            session.add(settings)
            await session.commit()
            await session.refresh(settings)
        return settings



@router.post("/buy")
async def buy_order(request: BuyRequest):
    """
    Place a buy order (supports both live MT5 and paper trading).
    
    Opens a long position or adds to existing position.
    Creates a trade record and updates position tracking.
    Note: Manual trades skip autotrading controls.
    """
    try:
        logger.info(f"[API] Buy order request: {request.quantity} {request.symbol}")
        
        # Backend-authoritative trading mode
        trading_settings = await _get_or_create_trading_settings()
        live_mode_requested = bool(trading_settings.live_mode)

        # Check if MT5 is connected for live trading
        from app.services import get_mt5_client
        from app.services.mock_mt5 import MockMT5Client
        from app.config import settings
        
        mt5_client = get_mt5_client()
        mt5_real_connected = mt5_client.is_connected and not isinstance(mt5_client, MockMT5Client)

        can_execute_live, live_block_reason = (False, None)
        if live_mode_requested and mt5_real_connected:
            can_execute_live, live_block_reason = _mt5_can_execute_live_orders(mt5_client, settings)

        is_live_mode = live_mode_requested and mt5_real_connected and bool(can_execute_live)

        if live_mode_requested and not mt5_real_connected:
            raise HTTPException(
                status_code=400,
                detail="Live mode is enabled but MT5 is not connected (real). Connect MT5 or switch to demo mode."
            )

        if live_mode_requested and mt5_real_connected and not can_execute_live:
            raise HTTPException(
                status_code=400,
                detail=(live_block_reason or "Live trading is not available."),
            )

        if is_live_mode:
            # Execute on real MT5
            logger.info(f"[API] Executing LIVE buy order on MT5: {request.quantity} {request.symbol}")
            comment = f"ATW2-UI-B-{request.symbol}"[:31]
            success, order_result, error = mt5_client.place_order(
                request.symbol,
                'buy',
                request.quantity,
                request.price,
                comment=comment
            )
            
            if not success:
                logger.warning(f"[API] MT5 buy order failed: {error}")
                raise HTTPException(status_code=400, detail=error or "MT5 order failed")
            
            logger.info(f"[API] LIVE buy order executed on MT5: {order_result}")

            try:
                live_trade_logger.record_trade(
                    LiveTradeEvent(
                        timestamp=datetime.utcnow().isoformat(),
                        symbol=request.symbol,
                        side='buy',
                        quantity=float(request.quantity),
                        price=float(order_result.get('price', request.price) or 0.0),
                        order_id=str(order_result.get('order')) if order_result else None,
                        comment=comment,
                        source='mt5_ui',
                    )
                )
            except Exception:
                pass
            
            # Also record in paper trading for tracking
            position, _ = await paper_trading_service.open_position(
                request.symbol,
                'buy',
                request.quantity,
                order_result.get('price', request.price),
                skip_controls=True,
                external_order_id=str(order_result.get('order')) if order_result and order_result.get('order') is not None else None
            )
            
            return {
                "status": "success",
                "message": f"LIVE buy order executed on MT5 for {request.quantity} {request.symbol}",
                "position": position,
                "mt5_order": order_result,
                "live_mode": True,
                "live_mode_requested": True
            }
        else:
            # Paper trading mode
            position, error = await paper_trading_service.open_position(
                request.symbol,
                'buy',
                request.quantity,
                request.price,
                skip_controls=True
            )
            
            if error:
                logger.warning(f"[API] Buy order rejected: {error}")
                raise HTTPException(status_code=400, detail=error)
            
            logger.info(f"[API] Paper buy order executed: {request.quantity} {request.symbol}")
            
            return {
                "status": "success",
                "message": f"Buy order executed for {request.quantity} {request.symbol}",
                "position": position,
                "live_mode": False,
                "live_mode_requested": live_mode_requested
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Buy order failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Buy order failed: {str(e)}")


@router.post("/sell")
async def sell_order(request: SellRequest):
    """
    Place a sell order (supports both live MT5 and paper trading).
    
    Opens a short position or adds to existing position.
    Creates a trade record and updates position tracking.
    Note: Manual trades skip autotrading controls.
    """
    try:
        logger.info(f"[API] Sell order request: {request.quantity} {request.symbol}")

        # Backend-authoritative trading mode
        trading_settings = await _get_or_create_trading_settings()
        live_mode_requested = bool(trading_settings.live_mode)

        # Check if MT5 is connected for live trading
        from app.services import get_mt5_client
        from app.services.mock_mt5 import MockMT5Client
        from app.config import settings

        mt5_client = get_mt5_client()
        mt5_real_connected = mt5_client.is_connected and not isinstance(mt5_client, MockMT5Client)

        can_execute_live, live_block_reason = (False, None)
        if live_mode_requested and mt5_real_connected:
            can_execute_live, live_block_reason = _mt5_can_execute_live_orders(mt5_client, settings)

        is_live_mode = live_mode_requested and mt5_real_connected and bool(can_execute_live)

        if live_mode_requested and not mt5_real_connected:
            raise HTTPException(
                status_code=400,
                detail="Live mode is enabled but MT5 is not connected (real). Connect MT5 or switch to demo mode.",
            )

        if live_mode_requested and mt5_real_connected and not can_execute_live:
            raise HTTPException(
                status_code=400,
                detail=(live_block_reason or "Live trading is not available."),
            )

        if is_live_mode:
            # Execute on real MT5
            logger.info(f"[API] Executing LIVE sell order on MT5: {request.quantity} {request.symbol}")
            comment = f"ATW2-UI-S-{request.symbol}"[:31]
            success, order_result, error = mt5_client.place_order(
                request.symbol,
                'sell',
                request.quantity,
                request.price,
                comment=comment
            )

            if not success:
                logger.warning(f"[API] MT5 sell order failed: {error}")
                raise HTTPException(status_code=400, detail=error or "MT5 order failed")

            logger.info(f"[API] LIVE sell order executed on MT5: {order_result}")

            try:
                live_trade_logger.record_trade(
                    LiveTradeEvent(
                        timestamp=datetime.utcnow().isoformat(),
                        symbol=request.symbol,
                        side='sell',
                        quantity=float(request.quantity),
                        price=float(order_result.get('price', request.price) or 0.0),
                        order_id=str(order_result.get('order')) if order_result else None,
                        comment=comment,
                        source='mt5_ui',
                    )
                )
            except Exception:
                pass

            # Also record in paper trading for tracking
            position, _ = await paper_trading_service.open_position(
                request.symbol,
                'sell',
                request.quantity,
                order_result.get('price', request.price),
                skip_controls=True,
                external_order_id=str(order_result.get('order')) if order_result and order_result.get('order') is not None else None
            )

            return {
                "status": "success",
                "message": f"LIVE sell order executed on MT5 for {request.quantity} {request.symbol}",
                "position": position,
                "mt5_order": order_result,
                "live_mode": True,
                "live_mode_requested": True
            }

        # Paper trading mode
        position, error = await paper_trading_service.open_position(
            request.symbol,
            'sell',
            request.quantity,
            request.price,
            skip_controls=True
        )

        if error:
            logger.warning(f"[API] Sell order rejected: {error}")
            raise HTTPException(status_code=400, detail=error)

        logger.info(f"[API] Paper sell order executed: {request.quantity} {request.symbol}")

        return {
            "status": "success",
            "message": f"Sell order executed for {request.quantity} {request.symbol}",
            "position": position,
            "live_mode": False,
            "live_mode_requested": live_mode_requested
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Sell order failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sell order failed: {str(e)}")


@router.get("/mode", response_model=TradingModeResponse)
async def get_trading_mode():
    """Get backend-authoritative trading mode (live vs demo)."""
    settings = await _get_or_create_trading_settings()
    return TradingModeResponse(
        live_mode=bool(settings.live_mode),
        updated_at=settings.updated_at.isoformat() if getattr(settings, 'updated_at', None) else None,
    )


@router.put("/mode", response_model=TradingModeResponse)
async def set_trading_mode(request: TradingModeRequest):
    """Set backend-authoritative trading mode (live vs demo)."""
    # If enabling live mode, require real MT5 connection AND server config
    from app.services import get_mt5_client
    from app.services.mock_mt5 import MockMT5Client
    from app.config import settings

    mt5_client = get_mt5_client()
    mt5_real_connected = mt5_client.is_connected and not isinstance(mt5_client, MockMT5Client)

    if request.live_mode and not mt5_real_connected:
        raise HTTPException(
            status_code=400,
            detail="Cannot enable live mode: MT5 is not connected (real).",
        )

    if request.live_mode and mt5_real_connected:
        can_execute_live, live_block_reason = _mt5_can_execute_live_orders(mt5_client, settings)
        if not can_execute_live:
            raise HTTPException(
                status_code=400,
                detail=(live_block_reason or "Cannot enable live mode."),
            )

    async for session in db.get_session():
        result = await session.execute(select(TradingSettings).limit(1))
        settings = result.scalar_one_or_none()
        if settings is None:
            settings = TradingSettings(live_mode=bool(request.live_mode))
            session.add(settings)
        else:
            settings.live_mode = bool(request.live_mode)
        await session.commit()
        await session.refresh(settings)
        return TradingModeResponse(
            live_mode=bool(settings.live_mode),
            updated_at=settings.updated_at.isoformat() if getattr(settings, 'updated_at', None) else None,
        )



@router.post("/close/{position_id}")
async def close_position(position_id: int, request: Optional[CloseRequest] = None):
    """
    Close a position (fully or partially).
    
    Closes the position and calculates realized P/L.
    Creates a trade record for the closing transaction.
    """
    try:
        quantity = request.quantity if request else None
        position, error = await paper_trading_service.close_position(position_id, quantity)
        
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        return {
            "status": "success",
            "message": f"Position {position_id} closed",
            "position": position
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Close position failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Close position failed: {str(e)}")


@router.get("/positions")
async def get_positions(status: Optional[str] = Query(None, description="Filter by status: 'open' or 'closed'")):
    """
    Get all positions.
    
    Returns list of positions with current P/L calculations.
    """
    try:
        if status and status not in ['open', 'closed']:
            raise HTTPException(status_code=400, detail="Status must be 'open' or 'closed'")
        
        positions = await paper_trading_service.get_positions(status)
        
        return {
            "positions": positions,
            "count": len(positions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get positions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get positions: {str(e)}")


@router.get("/positions/{position_id}")
async def get_position(position_id: int):
    """
    Get a specific position by ID.
    
    Returns position details including current P/L.
    """
    try:
        position = await paper_trading_service.get_position(position_id)
        
        if not position:
            raise HTTPException(status_code=404, detail=f"Position {position_id} not found")
        
        return position
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get position: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get position: {str(e)}")


@router.get("/history")
async def get_trade_history(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    side: Optional[str] = Query(None, description="Filter by side: 'buy' or 'sell'"),
    status: Optional[str] = Query(None, description="Filter by status: 'executed', 'pending', 'cancelled'"),
    min_profit: Optional[float] = Query(None, description="Minimum profit/loss filter"),
    max_profit: Optional[float] = Query(None, description="Maximum profit/loss filter"),
    start_date: Optional[str] = Query(None, description="Start date filter (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"),
    end_date: Optional[str] = Query(None, description="End date filter (ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"),
    limit: Optional[int] = Query(100, ge=1, le=10000, description="Maximum number of trades")
):
    """
    Get trade history with comprehensive filtering.
    
    Supports filtering by:
    - Symbol
    - Side (buy/sell)
    - Status
    - Profit/loss range
    - Date range
    
    Returns list of executed trades with full details including profit/loss.
    """
    try:
        from datetime import datetime
        
        # Parse date strings
        start_dt = None
        end_dt = None
        
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                try:
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid start_date format: {start_date}")
        
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                try:
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    # Set to end of day
                    end_dt = end_dt.replace(hour=23, minute=59, second=59)
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid end_date format: {end_date}")
        
        trades = await paper_trading_service.get_trade_history(
            symbol=symbol,
            trade_type=side,
            status=status,
            min_profit=min_profit,
            max_profit=max_profit,
            start_date=start_dt,
            end_date=end_dt,
            limit=limit
        )
        
        return {
            "trades": trades,
            "count": len(trades),
            "filters": {
                "symbol": symbol,
                "side": side,
                "status": status,
                "min_profit": min_profit,
                "max_profit": max_profit,
                "start_date": start_date,
                "end_date": end_date
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trade history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get trade history: {str(e)}")


@router.post("/update")
async def update_positions():
    """
    Update all open positions with current market prices.
    
    Recalculates unrealized P/L for all open positions.
    """
    try:
        result = await paper_trading_service.update_positions()
        
        return {
            "status": "success",
            "message": f"Updated {result['updated']} positions",
            "statistics": result
        }
        
    except Exception as e:
        logger.error(f"Failed to update positions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update positions: {str(e)}")


@router.get("/history/export")
async def export_trade_history(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    side: Optional[str] = Query(None, description="Filter by side: 'buy' or 'sell'"),
    status: Optional[str] = Query(None, description="Filter by status"),
    min_profit: Optional[float] = Query(None, description="Minimum profit/loss filter"),
    max_profit: Optional[float] = Query(None, description="Maximum profit/loss filter"),
    start_date: Optional[str] = Query(None, description="Start date filter (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date filter (ISO format)"),
    format: str = Query("csv", description="Export format: 'csv' or 'json'"),
    limit: Optional[int] = Query(10000, ge=1, le=100000, description="Maximum number of trades")
):
    """
    Export trade history with filtering support.
    
    Exports trades in CSV or JSON format with all applied filters.
    """
    try:
        from datetime import datetime
        
        # Parse date strings
        start_dt = None
        end_dt = None
        
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                try:
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid start_date format: {start_date}")
        
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                try:
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    end_dt = end_dt.replace(hour=23, minute=59, second=59)
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid end_date format: {end_date}")
        
        trades = await paper_trading_service.get_trade_history(
            symbol=symbol,
            trade_type=side,
            status=status,
            min_profit=min_profit,
            max_profit=max_profit,
            start_date=start_dt,
            end_date=end_dt,
            limit=limit
        )
        
        if format.lower() == 'csv':
            # Generate CSV
            output = io.StringIO()
            if trades:
                writer = csv.DictWriter(output, fieldnames=trades[0].keys())
                writer.writeheader()
                writer.writerows(trades)
            
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                }
            )
        
        elif format.lower() == 'json':
            # Generate JSON
            json_data = json.dumps({
                "trades": trades,
                "count": len(trades),
                "exported_at": datetime.utcnow().isoformat(),
                "filters": {
                    "symbol": symbol,
                    "side": side,
                    "status": status,
                    "min_profit": min_profit,
                    "max_profit": max_profit,
                    "start_date": start_date,
                    "end_date": end_date
                }
            }, indent=2)
            
            return Response(
                content=json_data,
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                }
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format}. Use 'csv' or 'json'")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export trade history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.post("/live-trades/flush")
async def flush_live_trades_to_dataset_queue():
    try:
        flushed, path, err = live_trade_logger.flush_to_datasets()
        if err:
            raise HTTPException(status_code=500, detail=err)
        return {
            "status": "success",
            "rows_flushed": int(flushed),
            "path": path,
            "message": (
                f"Flushed {flushed} live trade(s) to dataset queue"
                if flushed
                else "No live trades to flush"
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to flush live trades: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to flush live trades: {str(e)}")


@router.get("/summary")
async def get_trading_summary():
    """
    Get trading summary statistics.
    
    Returns overall P/L, position counts, and trade statistics.
    """
    try:
        positions = await paper_trading_service.get_positions()
        trades = await paper_trading_service.get_trade_history(limit=1000)
        
        open_positions = [p for p in positions if p['status'] == 'open']
        closed_positions = [p for p in positions if p['status'] == 'closed']
        
        total_unrealized_pnl = sum(p['unrealized_pnl'] for p in open_positions)
        total_realized_pnl = sum(p['realized_pnl'] for p in positions)
        total_pnl = total_unrealized_pnl + total_realized_pnl
        
        buy_trades = [t for t in trades if t['trade_type'] == 'buy']
        sell_trades = [t for t in trades if t['trade_type'] == 'sell']
        
        return {
            "summary": {
                "total_pnl": total_pnl,
                "unrealized_pnl": total_unrealized_pnl,
                "realized_pnl": total_realized_pnl,
                "open_positions": len(open_positions),
                "closed_positions": len(closed_positions),
                "total_trades": len(trades),
                "buy_trades": len(buy_trades),
                "sell_trades": len(sell_trades)
            },
            "positions": {
                "open": open_positions,
                "closed": closed_positions
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get trading summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get trading summary: {str(e)}")
