"""Technical indicators service - deterministic mathematical calculations."""
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from functools import lru_cache
from app.database import db
from app.models import OHLCV
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.utils.logging import get_logger

logger = get_logger(__name__)


class IndicatorsService:
    """Service for calculating technical indicators."""
    
    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
    
    def _get_cache_key(self, symbol: str, indicator: str, params: Dict) -> str:
        """Generate cache key for indicator calculation."""
        params_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
        return f"{symbol}_{indicator}_{params_str}"
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid."""
        if 'timestamp' not in cache_entry:
            return False
        age = (datetime.utcnow() - cache_entry['timestamp']).total_seconds()
        return age < self._cache_ttl
    
    async def get_ohlcv_data(self, symbol: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get OHLCV data from database for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Optional limit on number of rows
            
        Returns:
            DataFrame with OHLCV data
        """
        async for session in db.get_session():
            query = (
                select(OHLCV)
                .where(OHLCV.symbol == symbol)
                .order_by(OHLCV.timestamp.desc())
            )
            
            if limit:
                query = query.limit(limit)
            
            result = await session.execute(query)
            rows = result.scalars().all()
            
            if not rows:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for row in rows:
                data.append({
                    'timestamp': row.timestamp,
                    'open': row.open,
                    'high': row.high,
                    'low': row.low,
                    'close': row.close,
                    'volume': row.volume
                })
            
            df = pd.DataFrame(data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            break  # Exit session context
        
        return df
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        Formula: EMA = (Price - EMA_prev) * (2 / (period + 1)) + EMA_prev
        
        Args:
            prices: Series of prices (typically close prices)
            period: EMA period
            
        Returns:
            Series of EMA values
        """
        if len(prices) < period:
            return pd.Series([np.nan] * len(prices), index=prices.index)
        
        # Calculate multiplier
        multiplier = 2.0 / (period + 1)
        
        # Initialize EMA with SMA for first value
        ema = pd.Series(index=prices.index, dtype=float)
        ema.iloc[period - 1] = prices.iloc[:period].mean()
        
        # Calculate EMA for remaining values
        for i in range(period, len(prices)):
            ema.iloc[i] = (prices.iloc[i] - ema.iloc[i - 1]) * multiplier + ema.iloc[i - 1]
        
        return ema
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Formula:
        - RS = Average Gain / Average Loss
        - RSI = 100 - (100 / (1 + RS))
        
        Args:
            prices: Series of prices (typically close prices)
            period: RSI period (default: 14)
            
        Returns:
            Series of RSI values (0-100)
        """
        if len(prices) < period + 1:
            return pd.Series([np.nan] * len(prices), index=prices.index)
        
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        # Calculate average gains and losses using Wilder's smoothing
        avg_gains = gains.rolling(window=period, min_periods=period).mean()
        avg_losses = losses.rolling(window=period, min_periods=period).mean()
        
        # For the first period, use simple average
        if period < len(gains):
            avg_gains.iloc[period] = gains.iloc[1:period+1].mean()
            avg_losses.iloc[period] = losses.iloc[1:period+1].mean()
        
        # Calculate subsequent averages using Wilder's smoothing
        for i in range(period + 1, len(gains)):
            avg_gains.iloc[i] = (avg_gains.iloc[i - 1] * (period - 1) + gains.iloc[i]) / period
            avg_losses.iloc[i] = (avg_losses.iloc[i - 1] * (period - 1) + losses.iloc[i]) / period
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(
        self, 
        prices: pd.Series, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Formula:
        - MACD Line = EMA(fast) - EMA(slow)
        - Signal Line = EMA(MACD Line)
        - Histogram = MACD Line - Signal Line
        
        Args:
            prices: Series of prices (typically close prices)
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal EMA period (default: 9)
            
        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' series
        """
        if len(prices) < slow_period + signal_period:
            nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
            return {
                'macd': nan_series,
                'signal': nan_series,
                'histogram': nan_series
            }
        
        # Calculate EMAs
        ema_fast = self.calculate_ema(prices, fast_period)
        ema_slow = self.calculate_ema(prices, slow_period)
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD line)
        signal_line = self.calculate_ema(macd_line, signal_period)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def aggregate_volume(self, df: pd.DataFrame, period: str = '1D') -> pd.Series:
        """
        Aggregate volume over a time period.
        
        Args:
            df: DataFrame with 'timestamp' and 'volume' columns
            period: Aggregation period (e.g., '1D', '1H', '1W')
            
        Returns:
            Series of aggregated volumes
        """
        if 'timestamp' not in df.columns or 'volume' not in df.columns:
            return pd.Series()
        
        # Set timestamp as index
        df_indexed = df.set_index('timestamp')
        
        # Resample and sum volume
        volume_agg = df_indexed['volume'].resample(period).sum()
        
        return volume_agg
    
    async def calculate_indicators(
        self,
        symbol: str,
        indicators: List[str],
        params: Optional[Dict[str, Dict]] = None,
        use_cache: bool = True
    ) -> Dict:
        """
        Calculate multiple indicators for a symbol.
        
        Args:
            symbol: Trading symbol
            indicators: List of indicators to calculate (ema, rsi, macd, volume)
            params: Optional parameters for each indicator
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary with indicator results
        """
        if params is None:
            params = {}
        
        # Check cache
        cache_key = self._get_cache_key(symbol, "_".join(sorted(indicators)), params)
        if use_cache and cache_key in self._cache:
            if self._is_cache_valid(self._cache[cache_key]):
                logger.debug(f"Returning cached indicators for {symbol}")
                return self._cache[cache_key]['data']
        
        # Get OHLCV data
        df = await self.get_ohlcv_data(symbol)
        
        if df.empty:
            return {'error': f'No data found for symbol {symbol}'}
        
        results = {
            'symbol': symbol,
            'timestamp': df['timestamp'].tolist(),
            'indicators': {}
        }
        
        # Calculate requested indicators
        if 'ema' in indicators:
            ema_params = params.get('ema', {'period': 20})
            period = ema_params.get('period', 20)
            ema = self.calculate_ema(df['close'], period)
            results['indicators']['ema'] = {
                'period': period,
                'values': ema.tolist()
            }
        
        if 'rsi' in indicators:
            rsi_params = params.get('rsi', {'period': 14})
            period = rsi_params.get('period', 14)
            rsi = self.calculate_rsi(df['close'], period)
            results['indicators']['rsi'] = {
                'period': period,
                'values': rsi.tolist()
            }
        
        if 'macd' in indicators:
            macd_params = params.get('macd', {'fast': 12, 'slow': 26, 'signal': 9})
            fast = macd_params.get('fast', 12)
            slow = macd_params.get('slow', 26)
            signal = macd_params.get('signal', 9)
            macd_result = self.calculate_macd(df['close'], fast, slow, signal)
            results['indicators']['macd'] = {
                'fast': fast,
                'slow': slow,
                'signal': signal,
                'macd_line': macd_result['macd'].tolist(),
                'signal_line': macd_result['signal'].tolist(),
                'histogram': macd_result['histogram'].tolist()
            }
        
        if 'volume' in indicators:
            volume_params = params.get('volume', {'period': '1D'})
            period = volume_params.get('period', '1D')
            volume_agg = self.aggregate_volume(df, period)
            results['indicators']['volume'] = {
                'period': period,
                'timestamps': volume_agg.index.tolist(),
                'values': volume_agg.tolist()
            }
        
        # Cache results
        if use_cache:
            self._cache[cache_key] = {
                'data': results,
                'timestamp': datetime.utcnow()
            }
        
        return results
    
    def clear_cache(self, symbol: Optional[str] = None) -> int:
        """
        Clear indicator cache.
        
        Args:
            symbol: Optional symbol to clear cache for (None = clear all)
            
        Returns:
            Number of cache entries cleared
        """
        if symbol:
            # Clear cache entries for specific symbol
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{symbol}_")]
            for key in keys_to_remove:
                del self._cache[key]
            return len(keys_to_remove)
        else:
            # Clear all cache
            count = len(self._cache)
            self._cache.clear()
            return count


# Global indicators service instance
indicators_service = IndicatorsService()
