"""Mock MT5 client for testing and fallback when MT5 is unavailable."""
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from app.utils.logging import get_logger

logger = get_logger(__name__)


class MockMT5Client:
    """Mock MT5 client that generates realistic OHLCV data."""
    
    # Base prices for different symbols
    BASE_PRICES = {
        'EURUSD': 1.0850,
        'BTCUSDT': 43000.0,
        'AAPL': 185.0,
    }
    
    # Price volatility ranges
    VOLATILITY = {
        'EURUSD': 0.0010,
        'BTCUSDT': 500.0,
        'AAPL': 2.0,
    }
    
    def __init__(self):
        self._connected = False
    
    def connect(self, login: Optional[int] = None, password: Optional[str] = None,
                server: Optional[str] = None) -> bool:
        """
        Mock connection (always succeeds).
        
        Args:
            login: Ignored in mock mode
            password: Ignored in mock mode
            server: Ignored in mock mode
            
        Returns:
            bool: Always True
        """
        if not self._connected:
            self._connected = True
            logger.info("Mock MT5 client connected (simulation mode)")
        else:
            logger.debug("Mock MT5 client already connected (reusing existing connection)")
        return True
    
    def shutdown(self) -> None:
        """Mock shutdown."""
        self._connected = False
        logger.info("Mock MT5 client shutdown")
    
    def get_ticks(self, symbol: str, start: Optional[datetime] = None,
                  count: int = 1000) -> pd.DataFrame:
        """
        Generate mock tick data.
        
        Args:
            symbol: Trading symbol
            start: Start datetime
            count: Number of ticks
            
        Returns:
            pd.DataFrame: DataFrame with columns [time, bid, ask, volume, flags]
        """
        if not self._connected:
            raise RuntimeError("Mock MT5 not connected. Call connect() first.")
        
        if start is None:
            start = datetime.now() - timedelta(seconds=count)
        
        base_price = self.BASE_PRICES.get(symbol, 100.0)
        volatility = self.VOLATILITY.get(symbol, 1.0)
        
        # Generate realistic tick data
        np.random.seed(hash(symbol) % 2**32)
        price_changes = np.random.normal(0, volatility * 0.1, count)
        prices = base_price + np.cumsum(price_changes)
        
        # Generate timestamps
        timestamps = [start + timedelta(seconds=i) for i in range(count)]
        
        # Create DataFrame
        ticks_df = pd.DataFrame({
            'time': pd.to_datetime(timestamps, utc=True),
            'bid': prices - (volatility * 0.0001),
            'ask': prices + (volatility * 0.0001),
            'volume': np.random.randint(1, 100, count),
            'flags': np.random.randint(0, 4, count),
        })
        
        # Don't log every generation - only log warnings/errors
        return ticks_df
    
    def get_ohlcv(self, symbol: str, timeframe: str, start: Optional[datetime] = None,
                  end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Generate mock OHLCV data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string (e.g., 'M1', 'H1', 'D1')
            start: Start datetime
            end: End datetime
            
        Returns:
            pd.DataFrame: DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        if not self._connected:
            raise RuntimeError("Mock MT5 not connected. Call connect() first.")
        
        # Parse timeframe
        timeframe_map = {
            'M1': timedelta(minutes=1),
            'M5': timedelta(minutes=5),
            'M15': timedelta(minutes=15),
            'M30': timedelta(minutes=30),
            'H1': timedelta(hours=1),
            'H4': timedelta(hours=4),
            'D1': timedelta(days=1),
            'W1': timedelta(weeks=1),
            'MN1': timedelta(days=30),
        }
        
        bar_duration = timeframe_map.get(timeframe.upper(), timedelta(hours=1))
        
        # Default time range
        if end is None:
            end = datetime.now()
        if start is None:
            start = end - (bar_duration * 1000)  # 1000 bars
        
        # Generate timestamps
        timestamps = []
        current = start
        while current <= end:
            timestamps.append(current)
            current += bar_duration
        
        if not timestamps:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Get base price and volatility
        base_price = self.BASE_PRICES.get(symbol, 100.0)
        volatility = self.VOLATILITY.get(symbol, 1.0)
        
        # Generate realistic OHLCV data using random walk
        np.random.seed(hash(f"{symbol}{timeframe}") % 2**32)
        num_bars = len(timestamps)
        
        # Generate price changes
        price_changes = np.random.normal(0, volatility * 0.1, num_bars)
        closes = base_price + np.cumsum(price_changes)
        
        # Generate OHLC from close prices
        ohlcv_data = []
        for i, (ts, close) in enumerate(zip(timestamps, closes)):
            if i == 0:
                open_price = base_price
            else:
                open_price = closes[i - 1]
            
            # High and low with some randomness
            high = max(open_price, close) + abs(np.random.normal(0, volatility * 0.05))
            low = min(open_price, close) - abs(np.random.normal(0, volatility * 0.05))
            
            # Volume (realistic range)
            if symbol == 'EURUSD':
                volume = np.random.uniform(100000, 1000000)
            elif symbol == 'BTCUSDT':
                volume = np.random.uniform(10, 1000)
            else:
                volume = np.random.uniform(1000, 100000)
            
            ohlcv_data.append({
                'timestamp': pd.to_datetime(ts, utc=True),
                'open': round(open_price, 5 if symbol == 'EURUSD' else 2),
                'high': round(high, 5 if symbol == 'EURUSD' else 2),
                'low': round(low, 5 if symbol == 'EURUSD' else 2),
                'close': round(close, 5 if symbol == 'EURUSD' else 2),
                'volume': round(volume, 2),
            })
        
        df = pd.DataFrame(ohlcv_data)
        # Don't log every generation - only log warnings/errors
        return df
    
    @property
    def is_connected(self) -> bool:
        """Check if mock MT5 is connected."""
        return self._connected
    
    def get_detailed_status(self) -> dict:
        """
        Get detailed mock MT5 status.
        
        Returns:
            Dictionary with mock status information
        """
        return {
            'terminal_installed': False,
            'connected': self._connected,
            'logged_in': False,
            'trading_enabled': False,
            'account_info': None,
            'mode': 'mock'
        }
    
    def validate_symbol(self, symbol: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if a symbol is available (mock - always returns True for known symbols).
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self._connected:
            return False, "Mock MT5 not connected"
        
        # In mock mode, accept any symbol (no real validation needed)
        # But we can check against known symbols for consistency
        known_symbols = list(self.BASE_PRICES.keys())
        if symbol not in known_symbols:
            # Still allow it, but log for awareness
            logger.debug(f"Symbol {symbol} not in known mock symbols, but allowing")
        
        return True, None
    
    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        """
        Get mock symbol information.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            dict: Mock symbol information
        """
        if not self._connected:
            return None
        
        base_price = self.BASE_PRICES.get(symbol, 100.0)
        spread = self.VOLATILITY.get(symbol, 1.0) * 0.0001
        
        return {
            'name': symbol,
            'bid': base_price - spread / 2,
            'ask': base_price + spread / 2,
            'spread': spread,
            'digits': 5 if symbol == 'EURUSD' else 2,
            'point': 0.00001 if symbol == 'EURUSD' else 0.01,
        }
