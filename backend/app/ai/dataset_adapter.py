"""
Dataset Adapter for Format Normalization

Handles conversion of wide-format FX datasets to long format.
Automatically detects dataset format and normalizes as needed.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime
from app.utils.logging import get_logger

logger = get_logger(__name__)


class DatasetAdapter:
    """
    Adapts datasets from various formats to the standard long format.
    
    Supports:
    - Wide format FX data (Date + one column per currency pair)
    - Long format data (already normalized)
    - Automatic format detection
    """
    
    # Date column aliases
    DATE_COLUMNS = ['date', 'timestamp', 'time', 'datetime', 'time serie', 'time_serie']
    
    # Exclude columns that are not currency pairs
    EXCLUDE_COLUMNS = ['unnamed: 0', 'unnamed_0', 'index', 'id']
    
    def __init__(self, processed_dir: Optional[Path] = None):
        """
        Initialize DatasetAdapter.
        
        Args:
            processed_dir: Directory to save processed/normalized datasets
        """
        # Get processed directory (default: app/ai/data/Processed)
        if processed_dir is None:
            current_file = Path(__file__).resolve()
            processed_dir = current_file.parent / "data" / "Processed"
        
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"DatasetAdapter initialized. Processed directory: {self.processed_dir}")
    
    def _detect_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detect the date/timestamp column in the dataset.
        
        Args:
            df: DataFrame to inspect
            
        Returns:
            Column name if found, None otherwise
        """
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in self.DATE_COLUMNS:
                return col
        
        return None
    
    def _is_wide_format(self, df: pd.DataFrame) -> bool:
        """
        Detect if dataset is in wide format (one column per currency pair).
        
        Wide format characteristics:
        - Has a date/time column
        - Has multiple numeric columns (potential currency pairs)
        - Does NOT have both currency_pair AND close_price columns
        
        Args:
            df: DataFrame to inspect
            
        Returns:
            True if dataset appears to be in wide format
        """
        # STRICT CHECK: Only consider long format if BOTH required columns exist
        # Check for currency_pair column (exact match or very specific pattern)
        has_currency_pair = any(
            col.lower().strip() == 'currency_pair' or
            col.lower().strip() == 'symbol' or
            (col.lower().strip() == 'pair' and len(col.strip()) <= 10)  # Short column name
            for col in df.columns
        )
        
        # Check for close_price column (exact match or very specific pattern)
        has_close_price = any(
            col.lower().strip() == 'close_price' or
            col.lower().strip() == 'closing_price' or
            (col.lower().strip() == 'close' and len(col.strip()) <= 10)  # Short column name
            for col in df.columns
        )
        
        # If BOTH exist, it's already in long format - do NOT convert
        if has_currency_pair and has_close_price:
            logger.info("Dataset already in long format (has currency_pair and close_price columns)")
            return False
        
        # Check for date column
        date_col = self._detect_date_column(df)
        if not date_col:
            logger.debug("No date column found - cannot be wide format")
            return False
        
        # Count numeric columns (potential currency pairs)
        # Exclude date column and index columns
        numeric_cols = []
        for col in df.columns:
            col_lower = col.lower().strip()
            # Skip date column and excluded columns
            if col == date_col or col_lower in self.EXCLUDE_COLUMNS:
                continue
            
            # Check if column is numeric (including object columns that can be converted)
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            
            # Also check if column contains numeric values (even if dtype is object)
            if not is_numeric and df[col].dtype == 'object':
                # Try to convert a sample to numeric to check if it's actually numeric data
                sample = df[col].dropna().head(20)  # Check first 20 non-null values
                if len(sample) > 0:
                    try:
                        pd.to_numeric(sample, errors='raise')
                        is_numeric = True  # Column contains numeric values
                    except (ValueError, TypeError):
                        pass  # Not numeric
            
            if is_numeric:
                numeric_cols.append(col)
        
        # Fallback: If no numeric columns detected, check for currency pair column names
        # (columns with "/US$" or "/USD" pattern)
        if len(numeric_cols) == 0:
            logger.debug("No numeric columns detected, checking for currency pair column patterns...")
            for col in df.columns:
                col_lower = col.lower().strip()
                if col == date_col or col_lower in self.EXCLUDE_COLUMNS:
                    continue
                # Check if column name suggests it's a currency pair (contains /US$ or /USD)
                if "/US$" in col or "/USD" in col.upper():
                    numeric_cols.append(col)
                    logger.debug(f"  Found currency pair column by name pattern: {col}")
        
        # Wide format: multiple numeric columns (currency pairs) + date column
        # AND no currency_pair/close_price columns
        # Require at least 2 numeric columns to be considered wide format
        is_wide = len(numeric_cols) >= 2
        
        if is_wide:
            logger.info(
                f"Detected WIDE format: {len(numeric_cols)} numeric columns "
                f"(potential currency pairs), date column: '{date_col}'"
            )
            logger.info(f"  Numeric columns (first 5): {numeric_cols[:5]}")
            if len(numeric_cols) > 5:
                logger.info(f"  ... and {len(numeric_cols) - 5} more")
        else:
            logger.warning(
                f"Not detected as wide format: numeric_cols={len(numeric_cols)}, "
                f"has_currency_pair={has_currency_pair}, has_close_price={has_close_price}"
            )
        
        return is_wide
    
    def _extract_currency_pair_name(self, column_name: str) -> str:
        """
        Extract currency pair name from column header.
        
        Examples:
        - "EURO AREA - EURO/US$" -> "EURUSD"
        - "JAPAN - YEN/US$" -> "JPYUSD"
        - "AUSTRALIA - AUSTRALIAN DOLLAR/US$" -> "AUDUSD"
        
        Args:
            column_name: Original column name
            
        Returns:
            Normalized currency pair name
        """
        # Remove common prefixes and suffixes
        name = column_name.strip()
        
        # Try to extract currency codes
        # Pattern: "COUNTRY - CURRENCY/US$" -> extract currency
        if "/US$" in name or "/USD" in name.upper():
            # Extract currency before /US$
            parts = name.split("/")
            if len(parts) > 0:
                currency_part = parts[0].strip()
                # Remove country prefix (e.g., "EURO AREA - ")
                if " - " in currency_part:
                    currency_part = currency_part.split(" - ")[-1].strip()
                
                # Map common currency names to codes
                currency_map = {
                    'euro': 'EUR',
                    'yen': 'JPY',
                    'australian dollar': 'AUD',
                    'australian': 'AUD',
                    'new zealand dollar': 'NZD',
                    'new zeland dollar': 'NZD',  # Handle typo in data
                    'united kingdom pound': 'GBP',
                    'pound': 'GBP',
                    'real': 'BRL',
                    'canadian dollar': 'CAD',
                    'yuan': 'CNY',
                    'hong kong dollar': 'HKD',
                    'indian rupee': 'INR',
                    'won': 'KRW',
                    'mexican peso': 'MXN',
                    'rand': 'ZAR',
                    'singapore dollar': 'SGD',
                    'danish krone': 'DKK',
                    'ringgit': 'MYR',
                    'norwegian krone': 'NOK',
                    'krona': 'SEK',
                    'sri lankan rupee': 'LKR',
                    'franc': 'CHF',
                    'new taiwan dollar': 'TWD',
                    'baht': 'THB',
                }
                
                currency_lower = currency_part.lower()
                if currency_lower in currency_map:
                    return f"{currency_map[currency_lower]}USD"
                
                # Try to extract 3-letter code if present
                if len(currency_part) >= 3:
                    # Take first 3 letters if uppercase
                    if currency_part[:3].isupper():
                        return f"{currency_part[:3]}USD"
        
        # Fallback: use simplified version of column name
        # Remove special characters and take first part
        simplified = ''.join(c for c in name if c.isalnum() or c.isspace())
        words = simplified.split()
        if words:
            # Take first significant word
            return words[0].upper()[:3] + "USD" if len(words[0]) >= 3 else "UNKNOWN"
        
        return "UNKNOWN"
    
    def _convert_wide_to_long(
        self,
        df: pd.DataFrame,
        date_column: str
    ) -> pd.DataFrame:
        """
        Convert wide format dataset to long format.
        
        Wide format: Date + multiple currency pair columns
        Long format: Date + currency_pair + close_price
        
        Args:
            df: DataFrame in wide format
            date_column: Name of the date column
            
        Returns:
            DataFrame in long format
        """
        logger.info(f"Converting wide format to long format...")
        logger.info(f"  Date column: {date_column}")
        logger.info(f"  Input shape: {df.shape}")
        
        # Identify currency pair columns (numeric columns excluding date and excluded)
        currency_pair_cols = []
        for col in df.columns:
            col_lower = col.lower().strip()
            if col == date_column or col_lower in self.EXCLUDE_COLUMNS:
                continue
            
            # Check if column is numeric (including object columns that can be converted)
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            
            # Also check if column contains numeric values (even if dtype is object)
            if not is_numeric and df[col].dtype == 'object':
                # Try to convert a sample to numeric to check if it's actually numeric data
                sample = df[col].dropna().head(20)  # Check first 20 non-null values
                if len(sample) > 0:
                    try:
                        pd.to_numeric(sample, errors='raise')
                        is_numeric = True  # Column contains numeric values
                    except (ValueError, TypeError):
                        pass  # Not numeric
            
            # Fallback: Check if column name suggests it's a currency pair
            if not is_numeric:
                if "/US$" in col or "/USD" in col.upper():
                    is_numeric = True  # Treat as currency pair column by name pattern
            
            if is_numeric:
                currency_pair_cols.append(col)
        
        logger.info(f"  Found {len(currency_pair_cols)} currency pair columns")
        
        if not currency_pair_cols:
            raise ValueError("No currency pair columns found in wide format dataset")
        
        # Prepare data for melting
        # Keep date column and all currency pair columns
        melt_cols = [date_column] + currency_pair_cols
        df_melt = df[melt_cols].copy()
        
        # Parse date column
        df_melt[date_column] = pd.to_datetime(df_melt[date_column], errors='coerce')
        
        # Remove rows with invalid dates
        df_melt = df_melt.dropna(subset=[date_column])
        
        # Melt the dataframe: wide to long
        # id_vars: date column (stays as is)
        # value_vars: currency pair columns (will become rows)
        df_long = pd.melt(
            df_melt,
            id_vars=[date_column],
            value_vars=currency_pair_cols,
            var_name='currency_pair_raw',
            value_name='close_price'
        )
        
        # Extract normalized currency pair names
        logger.info("  Extracting currency pair names...")
        df_long['currency_pair'] = df_long['currency_pair_raw'].apply(
            self._extract_currency_pair_name
        )
        
        # Remove rows with missing prices
        df_long = df_long.dropna(subset=['close_price'])
        
        # Select and rename columns to standard format
        df_normalized = pd.DataFrame({
            'date': df_long[date_column],
            'currency_pair': df_long['currency_pair'],
            'close_price': pd.to_numeric(df_long['close_price'], errors='coerce')
        })
        
        # Remove any remaining NaN values
        df_normalized = df_normalized.dropna()
        
        # Sort by date and currency pair
        df_normalized = df_normalized.sort_values(['date', 'currency_pair']).reset_index(drop=True)
        
        logger.info(f"  Output shape: {df_normalized.shape}")
        logger.info(f"  Currency pairs: {df_normalized['currency_pair'].unique().tolist()}")
        logger.info(f"  Date range: {df_normalized['date'].min()} to {df_normalized['date'].max()}")
        
        return df_normalized
    
    def normalize_dataset(
        self,
        df: pd.DataFrame,
        source_filename: str
    ) -> Tuple[pd.DataFrame, Optional[Path]]:
        """
        Normalize dataset to standard long format.
        
        Automatically detects format and converts if needed.
        
        Args:
            df: Input DataFrame
            source_filename: Original filename (for logging and saving)
            
        Returns:
            Tuple of (normalized_dataframe, processed_file_path)
            processed_file_path is None if no normalization was needed
        """
        logger.info("=" * 60)
        logger.info(f"Normalizing dataset: {source_filename}")
        logger.info(f"  Input shape: {df.shape}")
        logger.info(f"  Input columns ({len(df.columns)}): {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}")
        
        # Check if already in long format (STRICT: must have BOTH columns)
        has_currency_pair = any(col.lower().strip() == 'currency_pair' for col in df.columns)
        has_close_price = any(col.lower().strip() == 'close_price' for col in df.columns)
        
        if has_currency_pair and has_close_price:
            logger.info("  Dataset is already in long format (has currency_pair and close_price)")
            logger.info("  No conversion needed")
            logger.info("=" * 60)
            return df, None
        
        # Check for date column
        date_col = self._detect_date_column(df)
        if not date_col:
            raise ValueError("No date column found in dataset")
        
        logger.info(f"  Date column detected: '{date_col}'")
        
        # Check if wide format
        if self._is_wide_format(df):
            logger.info("  ✓ Detected WIDE format - converting to long format...")
            
            # Convert wide to long
            df_normalized = self._convert_wide_to_long(df, date_col)
            
            # Verify output has required columns
            if 'currency_pair' not in df_normalized.columns or 'close_price' not in df_normalized.columns:
                raise ValueError(
                    f"Normalization failed: output missing required columns. "
                    f"Got: {list(df_normalized.columns)}"
                )
            
            # Log normalization results
            currency_pairs = df_normalized['currency_pair'].unique()
            logger.info(f"  ✓ Conversion complete:")
            logger.info(f"    - Output shape: {df_normalized.shape}")
            logger.info(f"    - Currency pairs normalized: {len(currency_pairs)}")
            logger.info(f"    - Sample pairs: {currency_pairs[:5].tolist()}")
            logger.info(f"    - Date range: {df_normalized['date'].min()} to {df_normalized['date'].max()}")
            
            # Save normalized version to staging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            processed_filename = f"normalized_{timestamp}_{source_filename}"
            processed_path = self.processed_dir / processed_filename
            
            df_normalized.to_csv(processed_path, index=False)
            logger.info(f"  ✓ Saved normalized dataset: {processed_path}")
            logger.info("=" * 60)
            
            return df_normalized, processed_path
        else:
            logger.warning("  Dataset format unclear - treating as long format")
            logger.warning("  If validation fails, dataset may need manual conversion")
            logger.info("=" * 60)
            return df, None
    
    def normalize_file(
        self,
        file_path: Path
    ) -> Tuple[pd.DataFrame, Optional[Path]]:
        """
        Normalize a CSV file to standard long format.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Tuple of (normalized_dataframe, processed_file_path)
        """
        # Load CSV with encoding fallback
        df = None
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                if encoding != 'utf-8':
                    logger.debug(f"Loaded {file_path.name} with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError(f"Failed to load {file_path} with any encoding")
        
        return self.normalize_dataset(df, file_path.name)


# Global dataset adapter instance
dataset_adapter = DatasetAdapter()

