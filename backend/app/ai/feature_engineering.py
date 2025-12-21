"""
Feature Engineering Module for Currency Price Prediction

Generates technical features from OHLCV data for ML model training.
Ensures no data leakage by only using past data for feature generation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from app.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Feature engineering for currency price prediction.
    
    Generates:
    - Lagged returns (1, 2, 3, 5, 10 periods)
    - Moving averages (SMA, EMA for multiple windows)
    - Price differences (high-low, close-open, etc.)
    - Target variable (1 = price goes up, 0 = price goes down)
    
    Ensures no data leakage by only using past data.
    """
    
    # Supported timeframes mapping (1m to 12w)
    TIMEFRAME_MAP = {
        '1m': timedelta(minutes=1),
        '3m': timedelta(minutes=3),
        '5m': timedelta(minutes=5),
        '6m': timedelta(minutes=6),
        '12m': timedelta(minutes=12),
        '15m': timedelta(minutes=15),
        '30m': timedelta(minutes=30),
        '45m': timedelta(minutes=45),
        '1h': timedelta(hours=1),
        '2h': timedelta(hours=2),
        '3h': timedelta(hours=3),
        '4h': timedelta(hours=4),
        '1d': timedelta(days=1),
        '1w': timedelta(weeks=1),
        '1M': timedelta(days=30),
        '3M': timedelta(days=90),
        '6M': timedelta(days=180),
        '12M': timedelta(days=365),
        '2w': timedelta(weeks=2),
        '4w': timedelta(weeks=4),
        '8w': timedelta(weeks=8),
        '12w': timedelta(weeks=12),
    }
    
    # Lag periods for returns
    LAG_PERIODS = [1, 2, 3, 5, 10]
    
    # Moving average windows
    MA_WINDOWS = [5, 10, 20, 50, 100]
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        logger.debug("FeatureEngineer initialized")
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate that DataFrame has required columns.
        
        Args:
            df: Input DataFrame
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ['date', 'close_price']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if len(df) == 0:
            raise ValueError("DataFrame is empty")
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Ensure close_price is numeric
        if not pd.api.types.is_numeric_dtype(df['close_price']):
            df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
    
    def _validate_timeframe(self, timeframe: str) -> None:
        """
        Validate timeframe string.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d', '1w', '12w')
            
        Raises:
            ValueError: If timeframe is not supported
        """
        if timeframe not in self.TIMEFRAME_MAP:
            supported = ', '.join(sorted(self.TIMEFRAME_MAP.keys()))
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported timeframes: {supported}"
            )
    
    def _generate_lagged_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate lagged returns (percentage change).
        
        Returns are calculated as: (price[t] - price[t-n]) / price[t-n]
        
        Args:
            df: DataFrame with 'close_price' column
            
        Returns:
            DataFrame with lagged return columns added
        """
        df = df.copy()
        
        for lag in self.LAG_PERIODS:
            # Calculate return: (current - lagged) / lagged
            df[f'return_lag_{lag}'] = df['close_price'].pct_change(periods=lag)
        
        return df
    
    def _generate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Simple Moving Averages (SMA) and Exponential Moving Averages (EMA).
        
        Args:
            df: DataFrame with 'close_price' column
            
        Returns:
            DataFrame with moving average columns added
        """
        df = df.copy()
        
        for window in self.MA_WINDOWS:
            # Simple Moving Average
            df[f'sma_{window}'] = df['close_price'].rolling(window=window, min_periods=1).mean()
            
            # Exponential Moving Average
            df[f'ema_{window}'] = df['close_price'].ewm(span=window, adjust=False).mean()
            
            # Price relative to moving average (normalized)
            df[f'price_to_sma_{window}'] = df['close_price'] / df[f'sma_{window}'] - 1.0
            df[f'price_to_ema_{window}'] = df['close_price'] / df[f'ema_{window}'] - 1.0
        
        return df
    
    def _generate_price_differences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate price difference features.
        
        Features:
        - High-Low spread (if available)
        - Close-Open difference (if available)
        - Price range (high-low) / close (if available)
        - Body size (close-open) / close (if available)
        
        Args:
            df: DataFrame with price columns
            
        Returns:
            DataFrame with price difference columns added
        """
        df = df.copy()
        
        # High-Low spread (if high and low columns exist)
        if 'high' in df.columns and 'low' in df.columns:
            df['high_low_spread'] = df['high'] - df['low']
            df['high_low_spread_pct'] = df['high_low_spread'] / df['close_price']
        
        # Close-Open difference (if open column exists)
        if 'open' in df.columns:
            df['close_open_diff'] = df['close_price'] - df['open']
            df['close_open_diff_pct'] = df['close_open_diff'] / df['open']
            
            # Body size (absolute)
            df['body_size'] = abs(df['close_open_diff'])
            df['body_size_pct'] = df['body_size'] / df['open']
        
        # Price change from previous period
        df['price_change'] = df['close_price'].diff()
        df['price_change_pct'] = df['close_price'].pct_change()
        
        # Volatility (rolling standard deviation of returns)
        if 'price_change_pct' in df.columns:
            for window in [5, 10, 20]:
                df[f'volatility_{window}'] = df['price_change_pct'].rolling(
                    window=window, min_periods=1
                ).std()
        
        return df
    
    def _generate_target(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Generate target variable for prediction.
        
        Target definition:
        - 1 = price goes up in the next period
        - 0 = price goes down in the next period
        
        Uses forward-looking data (shifted backward) to create target.
        This ensures no data leakage - target is based on future price movement.
        
        Args:
            df: DataFrame with 'close_price' column
            timeframe: Timeframe string (for logging)
            
        Returns:
            DataFrame with 'target' column added
        """
        df = df.copy()
        
        # Calculate next period's price (shifted backward)
        # This represents what the price will be in the future
        df['next_close_price'] = df['close_price'].shift(-1)
        
        # Target: 1 if price goes up, 0 if price goes down
        # Price goes up if next_close > current_close
        df['target'] = (df['next_close_price'] > df['close_price']).astype(int)
        
        # Drop the helper column
        df = df.drop(columns=['next_close_price'])
        
        # Log target distribution
        if 'target' in df.columns:
            target_counts = df['target'].value_counts()
            logger.debug(
                f"Target distribution for timeframe {timeframe}: "
                f"Up (1)={target_counts.get(1, 0)}, "
                f"Down (0)={target_counts.get(0, 0)}"
            )
        
        return df
    
    def _remove_data_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with NaN values that could cause data leakage.
        
        After feature generation, some rows will have NaN values:
        - First N rows (due to lagged features)
        - Last row (due to target generation, if target exists)
        
        Args:
            df: DataFrame with features and optionally target
            
        Returns:
            DataFrame with NaN rows removed
        """
        df = df.copy()
        
        # Count NaN values before cleaning
        nan_count_before = df.isna().sum().sum()
        
        # Drop rows with NaN in target (last row, future data) - only if target column exists
        if 'target' in df.columns:
            df = df.dropna(subset=['target'])
        
        # Drop rows with NaN in critical features
        # Keep rows where at least some features are available
        # (Some features may be NaN due to window requirements)
        critical_features = ['close_price', 'return_lag_1']
        # Only check features that exist in the dataframe
        critical_features = [f for f in critical_features if f in df.columns]
        if critical_features:
            df = df.dropna(subset=critical_features)
        
        # Fill remaining NaN values with 0 (for features that couldn't be calculated)
        # This is safe because these are typically edge cases (first few rows)
        feature_cols = [col for col in df.columns 
                       if col not in ['date', 'currency_pair', 'close_price', 'target', 
                                     'open', 'high', 'low', 'volume']]
        if feature_cols:
            df[feature_cols] = df[feature_cols].fillna(0)
        
        nan_count_after = df.isna().sum().sum()
        
        if nan_count_before > 0:
            logger.debug(
                f"Cleaned {nan_count_before - nan_count_after} NaN values"
            )
        
        return df
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        timeframe: str,
        include_target: bool = True
    ) -> pd.DataFrame:
        """
        Prepare features from raw OHLCV data.
        
        This is the main function for feature engineering.
        It generates all features and ensures no data leakage.
        
        Args:
            df: DataFrame with columns: date, close_price, (optional: open, high, low, volume)
            timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d', '1w', '12w')
            include_target: Whether to generate target variable (default: True)
            
        Returns:
            DataFrame with features and target (if include_target=True)
            
        Raises:
            ValueError: If DataFrame is invalid or timeframe is unsupported
        """
        logger.info(f"Preparing features for timeframe: {timeframe}")
        
        # Validate inputs
        self._validate_timeframe(timeframe)
        self._validate_dataframe(df)
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Ensure data is sorted by date (ascending)
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.debug(f"Input data shape: {df.shape}")
        logger.debug(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Generate features (order matters - no data leakage)
        # 1. Lagged returns (uses only past data)
        print("      > Generating lagged returns (5 features)...")
        df = self._generate_lagged_returns(df)
        logger.debug("Generated lagged returns")
        
        # 2. Moving averages (uses only past data)
        print("      > Generating moving averages (20 features)...")
        df = self._generate_moving_averages(df)
        logger.debug("Generated moving averages")
        
        # 3. Price differences (uses only current/past data)
        print("      > Generating price differences and volatility...")
        df = self._generate_price_differences(df)
        logger.debug("Generated price differences")
        
        # 4. Target variable (uses future data, but shifted correctly)
        if include_target:
            print("      > Generating target variable...")
            df = self._generate_target(df, timeframe)
            logger.debug("Generated target variable")
        
        # 5. Remove data leakage (drop NaN rows)
        print("      > Removing data leakage (dropping NaN rows)...")
        df = self._remove_data_leakage(df)
        logger.debug("Removed data leakage")
        
        logger.info(
            f"Feature engineering complete. Output shape: {df.shape} "
            f"({df.shape[0]} rows, {df.shape[1]} columns)"
        )
        
        return df
    
    def get_feature_names(self, include_target: bool = False) -> List[str]:
        """
        Get list of feature names that will be generated.
        
        Args:
            include_target: Whether to include target in the list
            
        Returns:
            List of feature column names
        """
        features = []
        
        # Lagged returns
        for lag in self.LAG_PERIODS:
            features.append(f'return_lag_{lag}')
        
        # Moving averages
        for window in self.MA_WINDOWS:
            features.extend([
                f'sma_{window}',
                f'ema_{window}',
                f'price_to_sma_{window}',
                f'price_to_ema_{window}',
            ])
        
        # Price differences (conditional)
        features.extend([
            'high_low_spread',
            'high_low_spread_pct',
            'close_open_diff',
            'close_open_diff_pct',
            'body_size',
            'body_size_pct',
            'price_change',
            'price_change_pct',
        ])
        
        # Volatility
        for window in [5, 10, 20]:
            features.append(f'volatility_{window}')
        
        if include_target:
            features.append('target')
        
        return features
    
    def get_supported_timeframes(self) -> List[str]:
        """
        Get list of supported timeframes.
        
        Returns:
            List of supported timeframe strings
        """
        return sorted(self.TIMEFRAME_MAP.keys())


# Global feature engineer instance
feature_engineer = FeatureEngineer()

