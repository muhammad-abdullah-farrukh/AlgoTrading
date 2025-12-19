"""MetaTrader5 client for real market data."""
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import MetaTrader5 as mt5
from app.utils.logging import get_logger

logger = get_logger(__name__)


class MT5Client:
    """MetaTrader5 client wrapper."""
    
    def __init__(self):
        self._connected = False
        self._login: Optional[int] = None
        self._password: Optional[str] = None
        self._server: Optional[str] = None
    
    def connect(self, login: Optional[int] = None, password: Optional[str] = None, 
                server: Optional[str] = None) -> bool:
        """
        Initialize and connect to MT5.
        
        Args:
            login: MT5 account login (optional)
            password: MT5 account password (optional)
            server: MT5 server name (optional)
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        if self._connected:
            logger.debug("MT5 already connected (reusing existing connection)")
            return True
        
        try:
            # Initialize MT5 - this checks if terminal is installed
            if not mt5.initialize():
                error = mt5.last_error()
                logger.info(f"MT5 terminal not available: {error}")
                # Shutdown to clean up failed initialization
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                return False
            
            logger.info("MT5 terminal detected and initialized")
            
            # Login if credentials provided
            logged_in = False
            if login and password and server:
                self._login = login
                self._password = password
                self._server = server
                
                if not mt5.login(login, password=password, server=server):
                    error = mt5.last_error()
                    logger.info(f"MT5 login failed: {error}")
                    logger.info("Continuing without login (demo account)")
                else:
                    logged_in = True
                    logger.info(f"MT5 logged in: {login}@{server}")
            
            # Get account info to check login status and trading enabled
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"MT5 account: {account_info.login}, Balance: {account_info.balance}")
                # Check if trading is enabled on account
                trading_allowed = account_info.trade_allowed
                if not trading_allowed:
                    logger.info("MT5 account trading is disabled (read-only mode)")
                else:
                    logger.info("MT5 account trading is enabled")
            else:
                logger.info("MT5 account info not available (not logged in)")
            
            self._connected = True
            return True
            
        except ImportError:
            # MT5 module not installed - this is expected in some environments
            logger.info("MT5 module not available (expected in mock mode)")
            return False
        except Exception as e:
            logger.info(f"MT5 connection error (falling back to mock): {str(e)}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown MT5 connection."""
        if self._connected:
            try:
                mt5.shutdown()
                self._connected = False
                logger.info("MT5 shutdown successfully")
            except Exception as e:
                logger.error(f"MT5 shutdown error: {str(e)}")
                self._connected = False
    
    def get_ticks(self, symbol: str, start: Optional[datetime] = None, 
                  count: int = 1000) -> pd.DataFrame:
        """
        Get tick data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            start: Start datetime (default: now - count seconds)
            count: Number of ticks to retrieve
            
        Returns:
            pd.DataFrame: DataFrame with columns [time, bid, ask, volume, flags]
        """
        if not self._connected:
            raise RuntimeError("MT5 not connected. Call connect() first.")
        
        try:
            # Default start time if not provided
            if start is None:
                start = datetime.now() - timedelta(seconds=count)
            
            # Copy ticks from MT5
            ticks = mt5.copy_ticks_from(symbol, start, count, mt5.COPY_TICKS_ALL)
            
            if ticks is None or len(ticks) == 0:
                # Don't log every empty result - this is normal when no new ticks available
                return pd.DataFrame()
            
            # Convert to DataFrame
            ticks_df = pd.DataFrame(ticks)
            
            # Normalize timestamp to UTC
            ticks_df['time'] = pd.to_datetime(ticks_df['time'], unit='s', utc=True)
            
            # Don't log every retrieval - only log warnings/errors
            return ticks_df
            
        except Exception as e:
            logger.error(f"Error getting ticks for {symbol}: {str(e)}")
            raise
    
    def get_ohlcv(self, symbol: str, timeframe: str, start: Optional[datetime] = None,
                  end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe string (e.g., 'M1', 'M5', 'H1', 'D1')
            start: Start datetime (default: 1000 bars ago)
            end: End datetime (default: now)
            
        Returns:
            pd.DataFrame: DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        if not self._connected:
            raise RuntimeError("MT5 not connected. Call connect() first.")
        
        try:
            # Map timeframe string to MT5 constant
            timeframe_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1,
                'W1': mt5.TIMEFRAME_W1,
                'MN1': mt5.TIMEFRAME_MN1,
            }
            
            mt5_timeframe = timeframe_map.get(timeframe.upper())
            if mt5_timeframe is None:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            # Default end time
            if end is None:
                end = datetime.now()
            
            # Default start time (1000 bars ago)
            if start is None:
                # Approximate start based on timeframe
                if timeframe.upper().startswith('M'):
                    minutes = int(timeframe.upper()[1:])
                    start = end - timedelta(minutes=minutes * 1000)
                elif timeframe.upper().startswith('H'):
                    hours = int(timeframe.upper()[1:])
                    start = end - timedelta(hours=hours * 1000)
                elif timeframe.upper() == 'D1':
                    start = end - timedelta(days=1000)
                else:
                    start = end - timedelta(days=1000)
            
            # Copy rates from MT5
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start, end)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No OHLCV data found for {symbol} {timeframe}")
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert to DataFrame
            rates_df = pd.DataFrame(rates)
            
            # Normalize timestamp to UTC
            rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s', utc=True)
            
            # Rename and reorder columns to match unified schema
            rates_df = rates_df.rename(columns={'time': 'timestamp'})
            rates_df = rates_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Retrieved {len(rates_df)} OHLCV bars for {symbol} {timeframe}")
            return rates_df
            
        except Exception as e:
            logger.error(f"Error getting OHLCV for {symbol} {timeframe}: {str(e)}")
            raise
    
    @property
    def is_connected(self) -> bool:
        """Check if MT5 is connected."""
        return self._connected
    
    def get_detailed_status(self) -> dict:
        """
        Get detailed MT5 connection status.
        
        Returns:
            Dictionary with connection details including:
            - terminal_installed: Whether MT5 terminal is installed
            - connected: Whether connected to MT5
            - logged_in: Whether user is logged in
            - trading_enabled: Whether trading is enabled on account
            - account_info: Account information if available
        """
        status = {
            'terminal_installed': False,
            'connected': self._connected,
            'logged_in': False,
            'trading_enabled': False,
            'account_info': None
        }
        
        if not self._connected:
            return status
        
        try:
            # Check if terminal is installed (initialize check)
            status['terminal_installed'] = mt5.initialize()
            if not status['terminal_installed']:
                mt5.shutdown()
                return status
            
            # Get account info
            account_info = mt5.account_info()
            if account_info:
                status['logged_in'] = True
                status['trading_enabled'] = bool(account_info.trade_allowed)
                status['account_info'] = {
                    'login': account_info.login,
                    'balance': float(account_info.balance),
                    'equity': float(account_info.equity),
                    'margin': float(account_info.margin),
                    'free_margin': float(account_info.margin_free),
                    'server': account_info.server,
                    'company': account_info.company,
                    'trade_allowed': bool(account_info.trade_allowed),
                    'trade_expert': bool(account_info.trade_expert),
                }
            else:
                status['logged_in'] = False
            
        except Exception as e:
            logger.debug(f"Error getting detailed MT5 status: {str(e)}")
        
        return status
    
    def validate_symbol(self, symbol: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if a symbol is available for trading.
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self._connected:
            return False, "MT5 not connected"
        
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False, f"Symbol {symbol} not found in MT5"
            
            # Check if symbol is visible/selectable
            if not symbol_info.visible:
                return False, f"Symbol {symbol} is not visible in MT5"
            
            # Check if symbol is selectable
            if not symbol_info.select:
                return False, f"Symbol {symbol} is not selectable in MT5"
            
            return True, None
            
        except Exception as e:
            return False, f"Error validating symbol: {str(e)}"
    
    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """
        Get symbol information.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            dict: Symbol information or None if not found
        """
        if not self._connected:
            return None
        
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                return {
                    'name': symbol_info.name,
                    'bid': symbol_info.bid,
                    'ask': symbol_info.ask,
                    'spread': symbol_info.spread,
                    'digits': symbol_info.digits,
                    'point': symbol_info.point,
                }
            return None
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {str(e)}")
            return None
