"""WebSocket connection manager."""
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect
from app.utils.logging import get_logger
import json
import asyncio
from datetime import datetime
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError, ConnectionClosed

logger = get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""
    
    def __init__(self):
        # Active connections by stream type
        self._connections: Dict[str, Set[WebSocket]] = {
            'ticks': set(),
            'positions': set(),
            'trades': set(),
            'general': set(),
        }
        # Connection metadata
        self._connection_metadata: Dict[WebSocket, dict] = {}
        # Heartbeat tasks
        self._heartbeat_tasks: Dict[WebSocket, asyncio.Task] = {}
    
    async def connect(self, websocket: WebSocket, stream_type: str = 'general') -> None:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            stream_type: Type of stream (ticks, positions, trades, general)
        """
        await websocket.accept()
        
        if stream_type not in self._connections:
            stream_type = 'general'
        
        self._connections[stream_type].add(websocket)
        self._connection_metadata[websocket] = {
            'stream_type': stream_type,
            'connected_at': datetime.utcnow(),
            'last_heartbeat': datetime.utcnow(),
        }
        
        # Start heartbeat task
        self._heartbeat_tasks[websocket] = asyncio.create_task(
            self._heartbeat_loop(websocket)
        )
        
        logger.info(f"WebSocket connected: {stream_type} (total: {len(self._connections[stream_type])})")
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """
        Disconnect a WebSocket connection.
        
        Args:
            websocket: WebSocket connection to disconnect
        """
        # Cancel heartbeat task safely
        if websocket in self._heartbeat_tasks:
            task = self._heartbeat_tasks[websocket]
            try:
                task.cancel()
                # Wait for task to finish cancellation (with timeout)
                try:
                    await asyncio.wait_for(task, timeout=0.5)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    # Task cancelled or timeout - both are fine
                    pass
            except Exception as e:
                logger.debug(f"Error cancelling heartbeat task: {str(e)}")
            finally:
                del self._heartbeat_tasks[websocket]
        
        # Remove from all stream types
        for stream_type, connections in self._connections.items():
            connections.discard(websocket)
        
        # Remove metadata
        if websocket in self._connection_metadata:
            metadata = self._connection_metadata.pop(websocket)
            # Log at DEBUG level to reduce noise (normal disconnects are expected)
            logger.debug(f"WebSocket disconnected: {metadata.get('stream_type', 'unknown')}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket) -> bool:
        """
        Send a message to a specific WebSocket connection.
        
        Args:
            message: Message dictionary to send
            websocket: Target WebSocket connection
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        # Check if websocket is still in our tracking (avoids sending to disconnected sockets)
        if websocket not in self._connection_metadata:
            return False
        
        try:
            await websocket.send_json(message)
            return True
        except asyncio.CancelledError:
            # Normal cancellation - don't log as error
            return False
        except (ConnectionClosedOK, ConnectionClosedError, ConnectionClosed):
            # Normal disconnect (1000, 1001 codes) - clean up silently
            try:
                await self.disconnect(websocket)
            except Exception:
                pass
            return False
        except WebSocketDisconnect:
            # Client disconnected - clean up silently
            try:
                await self.disconnect(websocket)
            except Exception:
                pass
            return False
        except RuntimeError as e:
            # Handle "Response already sent" and other runtime errors silently
            if "Response already sent" in str(e) or "closed" in str(e).lower():
                try:
                    await self.disconnect(websocket)
                except Exception:
                    pass
                return False
            # Log unexpected runtime errors
            logger.warning(f"Unexpected error sending to WebSocket: {str(e)}")
            try:
                await self.disconnect(websocket)
            except Exception:
                pass
            return False
        except Exception as e:
            # Log truly unexpected errors only
            error_str = str(e).lower()
            if "going away" not in error_str and "closed" not in error_str:
                logger.warning(f"Failed to send message to WebSocket: {str(e)}")
            try:
                await self.disconnect(websocket)
            except Exception:
                pass
            return False
    
    async def broadcast(self, message: dict, stream_type: str = None) -> int:
        """
        Broadcast a message to all connections of a specific stream type.
        
        Args:
            message: Message dictionary to broadcast
            stream_type: Stream type to broadcast to (None = all)
            
        Returns:
            int: Number of connections that received the message
        """
        sent_count = 0
        disconnected = []
        
        # Determine which connections to send to
        if stream_type:
            connections = self._connections.get(stream_type, set())
        else:
            # Send to all connections
            connections = set()
            for conn_set in self._connections.values():
                connections.update(conn_set)
        
        # Send to all connections
        for websocket in connections.copy():
            # Skip if already disconnected
            if websocket not in self._connection_metadata:
                disconnected.append(websocket)
                continue
            
            try:
                await websocket.send_json(message)
                sent_count += 1
            except (ConnectionClosedOK, ConnectionClosedError, ConnectionClosed, WebSocketDisconnect):
                # Normal disconnect - no logging needed
                disconnected.append(websocket)
            except asyncio.CancelledError:
                # Task cancelled - no logging needed
                disconnected.append(websocket)
            except RuntimeError as e:
                # Only log unexpected runtime errors
                error_str = str(e).lower()
                if "going away" not in error_str and "closed" not in error_str and "response already sent" not in error_str:
                    logger.warning(f"Unexpected error broadcasting to WebSocket: {str(e)}")
                disconnected.append(websocket)
            except Exception as e:
                # Log truly unexpected errors only
                error_str = str(e).lower()
                if "going away" not in error_str and "closed" not in error_str:
                    logger.warning(f"Failed to broadcast to WebSocket: {str(e)}")
                disconnected.append(websocket)
        
        # Clean up disconnected connections
        for websocket in disconnected:
            try:
                await self.disconnect(websocket)
            except Exception:
                # Silently ignore errors during cleanup
                pass
        
        return sent_count
    
    async def _heartbeat_loop(self, websocket: WebSocket) -> None:
        """
        Send periodic heartbeat messages to keep connection alive.
        
        Args:
            websocket: WebSocket connection
        """
        try:
            while True:
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                
                # Check if connection is still active
                if websocket not in self._connection_metadata:
                    break
                
                # Update last heartbeat time
                self._connection_metadata[websocket]['last_heartbeat'] = datetime.utcnow()
                
                # Send heartbeat
                heartbeat = {
                    'type': 'heartbeat',
                    'timestamp': datetime.utcnow().isoformat(),
                }
                
                try:
                    await websocket.send_json(heartbeat)
                except (ConnectionClosedOK, ConnectionClosedError, ConnectionClosed, WebSocketDisconnect, asyncio.CancelledError):
                    # Normal disconnect - exit silently
                    break
                except RuntimeError as e:
                    # Handle runtime errors (closed connections)
                    if "closed" in str(e).lower() or "going away" in str(e).lower():
                        break
                    # Log unexpected errors
                    logger.debug(f"Heartbeat send error: {str(e)}")
                    break
                except Exception as e:
                    # Log unexpected errors only
                    error_str = str(e).lower()
                    if "going away" not in error_str and "closed" not in error_str:
                        logger.debug(f"Heartbeat error: {str(e)}")
                    break
                    
        except asyncio.CancelledError:
            # Task was cancelled (normal disconnect)
            pass
        except Exception as e:
            # Only log truly unexpected errors
            error_str = str(e).lower()
            if "going away" not in error_str and "closed" not in error_str:
                logger.error(f"Heartbeat loop error: {str(e)}")
        finally:
            try:
                if websocket in self._connection_metadata:
                    await self.disconnect(websocket)
            except Exception:
                # Silently ignore cleanup errors
                pass
    
    async def shutdown_all(self, timeout: float = 5.0) -> None:
        """
        Shutdown all WebSocket connections gracefully.
        
        Args:
            timeout: Maximum time to wait for shutdown (seconds)
        """
        try:
            logger.info("Shutting down all WebSocket connections...")
            
            # Get all connections first
            all_connections = []
            for connections in self._connections.values():
                all_connections.extend(list(connections))
            
            if not all_connections:
                logger.info("No active WebSocket connections to close")
                return
            
            logger.info(f"Closing {len(all_connections)} active WebSocket connections...")
            
            # Cancel all heartbeat tasks with timeout
            heartbeat_tasks = list(self._heartbeat_tasks.values())
            for task in heartbeat_tasks:
                try:
                    task.cancel()
                except Exception as e:
                    logger.debug(f"Error cancelling heartbeat task: {str(e)}")
            
            # Wait for heartbeat tasks to finish cancellation (with timeout)
            if heartbeat_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*heartbeat_tasks, return_exceptions=True),
                        timeout=min(timeout * 0.3, 1.0)  # Use 30% of timeout for heartbeat cleanup
                    )
                except asyncio.TimeoutError:
                    logger.warning("Some heartbeat tasks did not cancel within timeout")
                except Exception as e:
                    logger.debug(f"Error waiting for heartbeat tasks: {str(e)}")
            
            # Disconnect all connections with timeout
            disconnect_tasks = [self.disconnect(ws) for ws in all_connections]
            try:
                await asyncio.wait_for(
                    asyncio.gather(*disconnect_tasks, return_exceptions=True),
                    timeout=timeout * 0.7  # Use 70% of timeout for disconnects
                )
            except asyncio.TimeoutError:
                logger.warning("Some WebSocket disconnects did not complete within timeout")
            except Exception as e:
                logger.debug(f"Error during disconnect: {str(e)}")
            
            # Clear all tracking structures
            self._connections.clear()
            self._connection_metadata.clear()
            self._heartbeat_tasks.clear()
            
            logger.info(f"WebSocket shutdown complete: {len(all_connections)} connections closed")
            
        except Exception as e:
            logger.error(f"Error during WebSocket shutdown: {str(e)}")
            # Force clear on error to prevent hanging
            self._connections.clear()
            self._connection_metadata.clear()
            self._heartbeat_tasks.clear()
    
    def get_connection_count(self, stream_type: str = None) -> int:
        """
        Get the number of active connections.
        
        Args:
            stream_type: Stream type to count (None = all)
            
        Returns:
            int: Number of active connections
        """
        if stream_type:
            return len(self._connections.get(stream_type, set()))
        return sum(len(conns) for conns in self._connections.values())


# Global connection manager instance
manager = ConnectionManager()
