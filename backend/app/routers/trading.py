"""Paper trading endpoints."""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from typing import Optional
from pydantic import BaseModel, Field
from app.services.paper_trading import paper_trading_service
from app.utils.logging import get_logger
import csv
import json
import io
from datetime import datetime

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


@router.post("/buy")
async def buy_order(request: BuyRequest):
    """
    Place a buy order (paper trading).
    
    Opens a long position or adds to existing position.
    Creates a trade record and updates position tracking.
    Note: Manual trades skip autotrading controls.
    """
    try:
        logger.info(f"[API] Buy order request: {request.quantity} {request.symbol}")
        
        position, error = await paper_trading_service.open_position(
            request.symbol,
            'buy',
            request.quantity,
            request.price,
            skip_controls=True  # Manual trades skip controls
        )
        
        if error:
            logger.warning(f"[API] Buy order rejected: {error}")
            raise HTTPException(status_code=400, detail=error)
        
        logger.info(f"[API] Buy order executed successfully: {request.quantity} {request.symbol}")
        
        return {
            "status": "success",
            "message": f"Buy order executed for {request.quantity} {request.symbol}",
            "position": position
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Buy order failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Buy order failed: {str(e)}")


@router.post("/sell")
async def sell_order(request: SellRequest):
    """
    Place a sell order (paper trading).
    
    Opens a short position or adds to existing position.
    Creates a trade record and updates position tracking.
    Note: Manual trades skip autotrading controls.
    """
    try:
        logger.info(f"[API] Sell order request: {request.quantity} {request.symbol}")
        
        position, error = await paper_trading_service.open_position(
            request.symbol,
            'sell',
            request.quantity,
            request.price,
            skip_controls=True  # Manual trades skip controls
        )
        
        if error:
            logger.warning(f"[API] Sell order rejected: {error}")
            raise HTTPException(status_code=400, detail=error)
        
        logger.info(f"[API] Sell order executed successfully: {request.quantity} {request.symbol}")
        
        return {
            "status": "success",
            "message": f"Sell order executed for {request.quantity} {request.symbol}",
            "position": position
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Sell order failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sell order failed: {str(e)}")


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
        raise HTTPException(status_code=500, detail=f"Failed to export trade history: {str(e)}")


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
