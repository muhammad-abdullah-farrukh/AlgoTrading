"""Health check and status router."""
from fastapi import APIRouter, HTTPException
from typing import Dict
from datetime import datetime
from app.database import db
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint.
    
    Returns:
        Status dictionary with "ok" status
    """
    return {"status": "ok"}


@router.get("/status")
async def get_status() -> Dict:
    """
    Comprehensive system status endpoint.
    
    Returns detailed status of all components:
    - Database connection
    - MT5 client status
    - WebSocket connections
    - Services status
    - System information
    """
    try:
        status = {
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Database status
        try:
            db_status = {
                "connected": db.is_connected,
                "database_url": "configured" if db.is_connected else "not connected"
            }
            status["components"]["database"] = db_status
        except Exception as e:
            status["components"]["database"] = {
                "connected": False,
                "error": str(e)
            }
        
        # MT5 Client status
        try:
            from app.services import get_mt5_client
            from app.services.mock_mt5 import MockMT5Client
            
            mt5_client = get_mt5_client()
            is_mock = isinstance(mt5_client, MockMT5Client)
            
            # Get detailed status
            detailed_status = mt5_client.get_detailed_status()
            
            mt5_status = {
                "connected": mt5_client.is_connected,
                "mode": "mock" if is_mock else ("real" if mt5_client.is_connected else "unavailable"),
                "terminal_installed": detailed_status.get('terminal_installed', False),
                "logged_in": detailed_status.get('logged_in', False),
                "trading_enabled": detailed_status.get('trading_enabled', False),
            }
            
            # Add account info if available (but don't expose sensitive data)
            if detailed_status.get('account_info'):
                account = detailed_status['account_info']
                mt5_status["account"] = {
                    "login": account.get('login'),
                    "server": account.get('server'),
                    "company": account.get('company'),
                    "balance": account.get('balance'),
                    "equity": account.get('equity'),
                    "margin": account.get('margin'),
                    "free_margin": account.get('free_margin'),
                    "trade_allowed": account.get('trade_allowed'),
                    "trade_expert": account.get('trade_expert'),
                }
            
            status["components"]["mt5"] = mt5_status
        except Exception as e:
            status["components"]["mt5"] = {
                "connected": False,
                "mode": "unavailable",
                "terminal_installed": False,
                "logged_in": False,
                "trading_enabled": False,
                "error": str(e)
            }
        
        # WebSocket Manager status
        try:
            from app.websocket import manager
            ws_status = {
                "active_connections": manager.get_connection_count(),
                "connections_by_type": {
                    stream_type: manager.get_connection_count(stream_type)
                    for stream_type in ['ticks', 'positions', 'trades', 'general']
                }
            }
            status["components"]["websocket"] = ws_status
        except Exception as e:
            status["components"]["websocket"] = {
                "active_connections": 0,
                "error": str(e)
            }
        
        # Scraper Service status
        try:
            from app.services.scraper import scraper_service
            scraper_status = {
                "available": True,
                "active_jobs": len(scraper_service._progress)
            }
            status["components"]["scraper"] = scraper_status
        except Exception as e:
            status["components"]["scraper"] = {
                "available": False,
                "error": str(e)
            }
        
        # Trading Services status
        try:
            from app.services.paper_trading import paper_trading_service
            from app.services.autotrading import autotrading_service
            
            # Get position and trade counts
            positions = await paper_trading_service.get_positions()
            trades = await paper_trading_service.get_trade_history(limit=1)
            
            trading_status = {
                "paper_trading": {
                    "available": True,
                    "open_positions": len([p for p in positions if p['status'] == 'open']),
                    "total_positions": len(positions)
                },
                "autotrading": {
                    "available": True,
                    "settings": await autotrading_service.get_settings() or {}
                }
            }
            status["components"]["trading"] = trading_status
        except Exception as e:
            status["components"]["trading"] = {
                "available": False,
                "error": str(e)
            }
        
        # Indicators Service status
        try:
            from app.services.indicators import indicators_service
            indicators_status = {
                "available": True,
                "cache_entries": len(indicators_service._cache)
            }
            status["components"]["indicators"] = indicators_status
        except Exception as e:
            status["components"]["indicators"] = {
                "available": False,
                "error": str(e)
            }
        
        # Determine overall status
        critical_components = ['database']
        has_critical_errors = any(
            not status["components"].get(comp, {}).get("connected", False)
            for comp in critical_components
        )
        
        if has_critical_errors:
            status["status"] = "degraded"
        else:
            status["status"] = "operational"
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system status: {str(e)}"
        )
