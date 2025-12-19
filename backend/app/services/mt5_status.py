"""MT5 connection status persistence service."""
from typing import Optional, Dict
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import db
from app.models import MT5ConnectionStatus
from app.utils.logging import get_logger

logger = get_logger(__name__)


class MT5StatusService:
    """Service for persisting MT5 connection status."""
    
    async def update_status(
        self,
        connected: bool,
        mode: str = "mock",
        login: Optional[int] = None,
        server: Optional[str] = None
    ) -> None:
        """
        Update MT5 connection status in database.
        
        Args:
            connected: Whether MT5 is connected
            mode: Connection mode ('real' or 'mock')
            login: MT5 login ID if connected
            server: MT5 server name if connected
        """
        try:
            async for session in db.get_session():
                query = select(MT5ConnectionStatus).limit(1)
                result = await session.execute(query)
                status = result.scalar_one_or_none()
                
                if not status:
                    status = MT5ConnectionStatus(
                        connected=False,
                        mode="mock"
                    )
                    session.add(status)
                
                # Update status
                old_connected = status.connected
                status.connected = connected
                status.mode = mode
                status.login = login if connected else None
                status.server = server if connected else None
                
                if connected and not old_connected:
                    status.last_connected_at = datetime.utcnow()
                    logger.info(f"MT5 connection status persisted: connected=True, mode={mode}")
                elif not connected and old_connected:
                    status.last_disconnected_at = datetime.utcnow()
                    logger.info(f"MT5 connection status persisted: connected=False")
                
                status.updated_at = datetime.utcnow()
                
                await session.commit()
                logger.debug(f"MT5 status updated in database: connected={connected}, mode={mode}")
                
                break  # Exit session context
                
        except Exception as e:
            logger.error(f"Failed to update MT5 connection status: {str(e)}")
    
    async def get_status(self) -> Optional[Dict]:
        """
        Get current MT5 connection status from database.
        
        Returns:
            Dictionary with status information or None if not found
        """
        try:
            async for session in db.get_session():
                query = select(MT5ConnectionStatus).limit(1)
                result = await session.execute(query)
                status = result.scalar_one_or_none()
                
                if not status:
                    # Create default status
                    status = MT5ConnectionStatus(
                        connected=False,
                        mode="mock"
                    )
                    session.add(status)
                    await session.commit()
                    await session.refresh(status)
                
                return {
                    'connected': status.connected,
                    'mode': status.mode,
                    'login': status.login,
                    'server': status.server,
                    'last_connected_at': status.last_connected_at.isoformat() if status.last_connected_at else None,
                    'last_disconnected_at': status.last_disconnected_at.isoformat() if status.last_disconnected_at else None,
                    'updated_at': status.updated_at.isoformat()
                }
                
                break  # Exit session context
                
        except Exception as e:
            logger.error(f"Failed to get MT5 connection status: {str(e)}")
            return None


# Global MT5 status service instance
mt5_status_service = MT5StatusService()



