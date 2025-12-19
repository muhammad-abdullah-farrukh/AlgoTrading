"""
Dataset Manager for AI Training

Manages dataset loading, validation, and FIFO queue logic for ML training.
"""
import pandas as pd
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import os
from app.utils.logging import get_logger
from app.ai.dataset_adapter import dataset_adapter

logger = get_logger(__name__)


class DatasetManager:
    """
    Manages datasets for AI training with FIFO queue logic.
    
    Responsibilities:
    - Load CSV datasets from Datasets/ directory
    - Validate required columns
    - Handle missing values safely
    - Implement FIFO queue (oldest first)
    - Move datasets to TrainedDS/ after training
    - Support appending new datasets
    """
    
    # Required columns for dataset validation
    REQUIRED_COLUMNS = {
        'date': ['date', 'timestamp', 'time', 'datetime', 'time serie', 'time_serie'],
        'currency_pair': ['symbol', 'currency_pair', 'pair', 'instrument'],
        'close_price': ['close', 'close_price', 'price', 'closing_price']
    }
    
    def __init__(self, datasets_dir: Optional[Path] = None, trained_dir: Optional[Path] = None):
        """
        Initialize DatasetManager.
        
        Args:
            datasets_dir: Path to Datasets directory (default: project root/Datasets)
            trained_dir: Path to TrainedDS directory (default: app/ai/data/TrainedDS)
        """
        # Get project root path - go up from backend/app/ai to project root
        # __file__ is at: backend/app/ai/dataset_manager.py
        # Project root is: go up 4 levels (ai -> app -> backend -> AlgoTradeWeb2 -> AlgoTradeWeb2)
        current_file = Path(__file__).resolve()
        # Go up: backend/app/ai -> backend/app -> backend -> AlgoTradeWeb2 (nested) -> AlgoTradeWeb2 (project root)
        project_root = current_file.parent.parent.parent.parent
        
        # Default to project root Datasets folder if not specified
        self.datasets_dir = datasets_dir or (project_root / "Datasets")
        # Keep TrainedDS in backend/app/ai/data/TrainedDS for organization
        ai_data_path = current_file.parent / "data" / "TrainedDS"
        self.trained_dir = trained_dir or ai_data_path
        
        # Resolve to absolute paths for consistency
        self.datasets_dir = Path(self.datasets_dir).resolve()
        self.trained_dir = Path(self.trained_dir).resolve()
        
        # Ensure directories exist
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.trained_dir.mkdir(parents=True, exist_ok=True)
        
        # Track loaded files in current session to prevent infinite loops
        self._loaded_files: set[str] = set()
        
        # Log absolute paths for verification
        logger.info(f"DatasetManager initialized:")
        logger.info(f"  Datasets directory (absolute): {self.datasets_dir}")
        logger.info(f"  TrainedDS directory (absolute): {self.trained_dir}")
        logger.info(f"  Datasets directory exists: {self.datasets_dir.exists()}")
        logger.info(f"  TrainedDS directory exists: {self.trained_dir.exists()}")
    
    def _normalize_column_name(self, col_name: str) -> Optional[str]:
        """
        Normalize column name to match required columns.
        
        Args:
            col_name: Original column name
            
        Returns:
            Normalized column name or None if not recognized
        """
        col_lower = col_name.lower().strip()
        
        # Check each required column category
        for required, aliases in self.REQUIRED_COLUMNS.items():
            if col_lower in aliases:
                return required
        
        return None
    
    def _validate_columns(self, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Validate that required columns exist in the dataset.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Get normalized column mapping
        column_mapping = {}
        for col in df.columns:
            normalized = self._normalize_column_name(col)
            if normalized:
                column_mapping[normalized] = col
        
        # Check all required columns are present
        missing = []
        for required_col in self.REQUIRED_COLUMNS.keys():
            if required_col not in column_mapping:
                missing.append(required_col)
        
        if missing:
            return False, f"Missing required columns: {', '.join(missing)}"
        
        return True, None
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame column names and structure.
        
        Args:
            df: Original DataFrame
            
        Returns:
            Normalized DataFrame with standard column names
        """
        # Create normalized column mapping
        column_mapping = {}
        for col in df.columns:
            normalized = self._normalize_column_name(col)
            if normalized:
                column_mapping[col] = normalized
        
        # Rename columns
        df_normalized = df.rename(columns=column_mapping).copy()
        
        # Ensure date column is datetime
        if 'date' in df_normalized.columns:
            df_normalized['date'] = pd.to_datetime(df_normalized['date'], errors='coerce')
        
        # Ensure close_price is numeric
        if 'close_price' in df_normalized.columns:
            df_normalized['close_price'] = pd.to_numeric(df_normalized['close_price'], errors='coerce')
        
        # Ensure currency_pair is string
        if 'currency_pair' in df_normalized.columns:
            df_normalized['currency_pair'] = df_normalized['currency_pair'].astype(str)
        
        return df_normalized
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values safely.
        
        Strategy:
        - Drop rows with missing required columns (date, currency_pair, close_price)
        - Forward fill for optional columns (open, high, low, volume)
        - Log warnings for dropped rows
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        original_rows = len(df)
        
        # Drop rows with missing required columns
        required_cols = ['date', 'currency_pair', 'close_price']
        df_clean = df.dropna(subset=required_cols).copy()
        
        dropped_rows = original_rows - len(df_clean)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows with missing required columns (out of {original_rows} total)")
        
        # Forward fill optional columns if they exist
        optional_cols = ['open', 'high', 'low', 'volume']
        for col in optional_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].ffill().bfill()
        
        # Sort by date to ensure chronological order
        if 'date' in df_clean.columns:
            df_clean = df_clean.sort_values('date').reset_index(drop=True)
        
        return df_clean
    
    def _get_dataset_files(self) -> List[Tuple[Path, datetime]]:
        """
        Get all dataset files sorted by creation time (oldest first - FIFO).
        
        Returns:
            List of (file_path, creation_time) tuples, sorted by creation time
        """
        csv_files = []
        
        # Check if directory exists
        if not self.datasets_dir.exists():
            logger.warning(f"Datasets directory does not exist: {self.datasets_dir}")
            return csv_files
        
        # Find all CSV files
        try:
            csv_paths = list(self.datasets_dir.glob("*.csv"))
            logger.debug(f"Found {len(csv_paths)} CSV files in {self.datasets_dir}")
        except Exception as e:
            logger.error(f"Failed to scan Datasets directory: {str(e)}")
            return csv_files
        
        # Get file metadata and sort by creation time (FIFO)
        for file_path in csv_paths:
            try:
                # Get file creation time (OS-independent)
                stat = file_path.stat()
                creation_time = datetime.fromtimestamp(stat.st_ctime)
                csv_files.append((file_path, creation_time))
            except Exception as e:
                logger.warning(f"Failed to get creation time for {file_path.name}: {str(e)}")
                # Fall back to modification time
                try:
                    stat = file_path.stat()
                    creation_time = datetime.fromtimestamp(stat.st_mtime)
                    csv_files.append((file_path, creation_time))
                except Exception as e2:
                    logger.error(f"Failed to get file time for {file_path.name}: {str(e2)}")
        
        # Sort by creation time (oldest first - FIFO)
        csv_files.sort(key=lambda x: x[1])
        
        if csv_files:
            logger.debug(f"Dataset files sorted (FIFO order): {[f.name for f, _ in csv_files]}")
        
        return csv_files
    
    def load_next_dataset(self) -> Optional[Tuple[pd.DataFrame, str]]:
        """
        Load the next dataset from the FIFO queue (oldest first).
        
        Automatically normalizes wide-format datasets to long format.
        Tracks loaded files to prevent infinite loops.
        
        Returns:
            Tuple of (DataFrame, filename) if dataset found, None otherwise
        """
        dataset_files = self._get_dataset_files()
        
        if not dataset_files:
            logger.info("No datasets available in Datasets/ directory")
            return None
        
        # Filter out already loaded files to prevent infinite loops
        available_files = [
            (fp, ct) for fp, ct in dataset_files 
            if fp.name not in self._loaded_files
        ]
        
        if not available_files:
            logger.info("All available datasets have been loaded in this session")
            return None
        
        # Get oldest dataset (first in queue) that hasn't been loaded yet
        file_path, creation_time = available_files[0]
        filename = file_path.name
        
        # Mark as loaded immediately to prevent reloading
        self._loaded_files.add(filename)
        
        logger.info(f"Loading dataset: {filename} (created: {creation_time.isoformat()})")
        
        try:
            # Load CSV with encoding fallback (UTF-8 first, then latin-1)
            print(f"   → Reading CSV file: {filename}...")
            df = None
            encoding_errors = []
            
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    # Try to load with current encoding
                    df = pd.read_csv(file_path, encoding=encoding)
                    if encoding != 'utf-8':
                        logger.debug(f"Loaded {filename} with {encoding} encoding (UTF-8 failed or not optimal)")
                    break
                except UnicodeDecodeError as e:
                    # Encoding error - try next encoding
                    encoding_errors.append(f"{encoding}: {str(e)}")
                    continue
                except Exception as e:
                    # Other errors (not encoding-related) - re-raise
                    raise
            
            if df is None:
                logger.error(f"Failed to load {filename} with any encoding. Errors: {encoding_errors}")
                return None
            
            if df.empty:
                logger.warning(f"Dataset {filename} is empty")
                return None
            
            # Log dataset shape before processing
            logger.info(f"Dataset {filename} loaded: {len(df)} rows, {len(df.columns)} columns")
            logger.debug(f"Dataset {filename} columns: {list(df.columns)}")
            print(f"   > Loaded: {len(df):,} rows, {len(df.columns)} columns")
            
            # STEP 1: Normalize dataset format (wide to long conversion if needed)
            logger.info(f"Normalizing dataset format for {filename}...")
            print(f"   > Normalizing dataset format...")
            try:
                df_normalized, processed_path = dataset_adapter.normalize_dataset(df, filename)
                
                if processed_path:
                    logger.info(f"OK Dataset normalized successfully")
                    logger.info(f"  Normalized dataset saved to: {processed_path}")
                    logger.info(f"  Normalized shape: {df_normalized.shape}")
                    logger.info(f"  Normalized columns: {list(df_normalized.columns)}")
                    print(f"   > Converted wide format to long format")
                else:
                    logger.info("Dataset already in correct format, no normalization needed")
                    logger.debug(f"  Dataset columns: {list(df_normalized.columns)}")
                    print(f"   > Dataset already in correct format")
            except Exception as e:
                logger.error(f"ERROR Dataset normalization failed: {str(e)}", exc_info=True)
                logger.error("Cannot proceed without normalization - dataset format is invalid")
                return None
            
            # STEP 2: Validate columns (after normalization)
            print(f"   > Validating columns...")
            is_valid, error_msg = self._validate_columns(df_normalized)
            if not is_valid:
                logger.error(f"Dataset {filename} validation failed after normalization: {error_msg}")
                logger.error("Required columns: date, currency_pair, close_price")
                return None
            
            # STEP 3: Normalize DataFrame column names
            print(f"   > Normalizing column names...")
            df_normalized = self._normalize_dataframe(df_normalized)
            
            # STEP 4: Handle missing values
            print(f"   > Handling missing values...")
            df_clean = self._handle_missing_values(df_normalized)
            
            if df_clean.empty:
                logger.warning(f"Dataset {filename} became empty after cleaning")
                return None
            
            # Log final shape and currency pairs
            currency_pairs = df_clean['currency_pair'].unique().tolist() if 'currency_pair' in df_clean.columns else []
            logger.info(
                f"Successfully loaded dataset {filename}: "
                f"{len(df_clean)} rows, {len(df_clean.columns)} columns, "
                f"{len(currency_pairs)} currency pair(s)"
            )
            if currency_pairs:
                logger.info(f"Currency pairs: {', '.join(currency_pairs[:10])}{'...' if len(currency_pairs) > 10 else ''}")
            logger.debug(f"Dataset {filename} shape: ({len(df_clean)}, {len(df_clean.columns)})")
            print(f"   [OK] Dataset ready: {len(df_clean):,} rows, {len(currency_pairs)} currency pair(s)")
            
            return df_clean, filename
            
        except Exception as e:
            logger.error(f"Failed to load dataset {filename}: {str(e)}", exc_info=True)
            # Keep file in tracking to prevent infinite retries of broken files
            # But log a warning
            logger.warning(f"Dataset {filename} will be skipped for this training session due to loading error")
            return None
    
    def reset_loaded_files(self) -> None:
        """
        Reset the loaded files tracking (useful for new training sessions).
        """
        self._loaded_files.clear()
        logger.debug("Reset loaded files tracking")
    
    def mark_dataset_as_trained(self, filename: str) -> bool:
        """
        Move dataset from Datasets/ to TrainedDS/ after training.
        
        Also removes the file from loaded files tracking.
        
        Args:
            filename: Name of the dataset file to move
            
        Returns:
            True if successful, False otherwise
        """
        source_path = self.datasets_dir / filename
        
        if not source_path.exists():
            logger.error(f"Dataset file not found: {filename}")
            return False
        
        try:
            # Remove from loaded files tracking
            self._loaded_files.discard(filename)
            
            # Create destination path with timestamp to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_filename = f"{timestamp}_{filename}"
            dest_path = self.trained_dir / dest_filename
            
            # Move file
            shutil.move(str(source_path), str(dest_path))
            
            logger.info(f"Moved dataset {filename} to TrainedDS/ as {dest_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to move dataset {filename} to TrainedDS/: {str(e)}")
            return False
    
    def append_dataset(self, df: pd.DataFrame, filename: Optional[str] = None) -> bool:
        """
        Append a new dataset to the Datasets/ directory.
        
        Args:
            df: DataFrame to save as CSV
            filename: Optional filename (default: auto-generated with timestamp)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate columns
            is_valid, error_msg = self._validate_columns(df)
            if not is_valid:
                logger.error(f"Dataset validation failed: {error_msg}")
                return False
            
            # Normalize DataFrame
            df_normalized = self._normalize_dataframe(df)
            
            # Handle missing values
            df_clean = self._handle_missing_values(df_normalized)
            
            if df_clean.empty:
                logger.error("Dataset became empty after cleaning")
                return False
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Try to get symbol from data
                symbol = df_clean['currency_pair'].iloc[0] if 'currency_pair' in df_clean.columns else "unknown"
                filename = f"dataset_{symbol}_{timestamp}.csv"
            
            # Ensure .csv extension
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            # Save to Datasets directory
            file_path = self.datasets_dir / filename
            df_clean.to_csv(file_path, index=False)
            
            logger.info(f"Appended new dataset: {filename} ({len(df_clean)} rows)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to append dataset: {str(e)}")
            return False
    
    def get_dataset_count(self) -> int:
        """
        Get the number of datasets available in Datasets/ directory.
        
        Returns:
            Number of CSV files in Datasets/
        """
        return len(list(self.datasets_dir.glob("*.csv")))
    
    def get_trained_dataset_count(self) -> int:
        """
        Get the number of trained datasets in TrainedDS/ directory.
        
        Returns:
            Number of CSV files in TrainedDS/
        """
        return len(list(self.trained_dir.glob("*.csv")))
    
    def list_datasets(self) -> List[Dict[str, any]]:
        """
        List all datasets in Datasets/ directory with metadata.
        
        Returns:
            List of dictionaries with dataset information
        """
        datasets = []
        
        for file_path, creation_time in self._get_dataset_files():
            try:
                # Get file size
                file_size = file_path.stat().st_size
                
                # Try to get row count (quick check)
                try:
                    # Try multiple encodings for line counting
                    line_count = None
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                                line_count = sum(1 for _ in f) - 1  # Subtract header
                            break
                        except:
                            continue
                except:
                    line_count = None
                
                datasets.append({
                    'filename': file_path.name,
                    'creation_time': creation_time.isoformat(),
                    'file_size': file_size,
                    'estimated_rows': line_count,
                    'path': str(file_path)
                })
            except Exception as e:
                logger.warning(f"Failed to get metadata for {file_path.name}: {str(e)}")
        
        return datasets
    
    def verify_dataset_pipeline(self) -> Dict[str, any]:
        """
        Verify dataset loading pipeline at runtime.
        
        Performs comprehensive checks:
        - Verifies directory paths
        - Lists discovered CSV files
        - Attempts to load at least one dataset
        - Logs all findings
        
        Returns:
            Dictionary with verification results
        """
        logger.info("=" * 60)
        logger.info("Dataset Loading Pipeline Verification")
        logger.info("=" * 60)
        
        results = {
            'datasets_dir_exists': False,
            'datasets_dir_path': str(self.datasets_dir),
            'datasets_dir_absolute': str(self.datasets_dir.resolve()),
            'csv_files_found': [],
            'csv_files_count': 0,
            'load_test_successful': False,
            'load_test_dataset': None,
            'load_test_shape': None,
            'errors': []
        }
        
        # Check 1: Verify directory exists
        logger.info(f"[Check 1] Verifying Datasets directory path...")
        logger.info(f"  Relative path: {self.datasets_dir}")
        logger.info(f"  Absolute path: {self.datasets_dir.resolve()}")
        
        if self.datasets_dir.exists():
            results['datasets_dir_exists'] = True
            logger.info(f"  ✓ Directory exists")
        else:
            error_msg = f"Datasets directory does not exist: {self.datasets_dir}"
            results['errors'].append(error_msg)
            logger.error(f"  ✗ {error_msg}")
            logger.info("=" * 60)
            return results
        
        # Check 2: Discover CSV files
        logger.info(f"[Check 2] Discovering CSV files...")
        try:
            csv_files = self._get_dataset_files()
            results['csv_files_count'] = len(csv_files)
            
            if csv_files:
                logger.info(f"  ✓ Found {len(csv_files)} CSV file(s):")
                for file_path, creation_time in csv_files:
                    file_info = {
                        'filename': file_path.name,
                        'absolute_path': str(file_path.resolve()),
                        'creation_time': creation_time.isoformat(),
                        'size_bytes': file_path.stat().st_size
                    }
                    results['csv_files_found'].append(file_info)
                    logger.info(f"    - {file_path.name} ({file_path.stat().st_size} bytes, created: {creation_time.isoformat()})")
            else:
                logger.warning(f"  ⚠ No CSV files found in {self.datasets_dir}")
                results['errors'].append("No CSV files found in Datasets directory")
        except Exception as e:
            error_msg = f"Failed to discover CSV files: {str(e)}"
            results['errors'].append(error_msg)
            logger.error(f"  ✗ {error_msg}")
        
        # Check 3: Attempt to load at least one dataset
        logger.info(f"[Check 3] Testing dataset loading...")
        if csv_files:
            try:
                # Try to load the oldest dataset (FIFO)
                result = self.load_next_dataset()
                
                if result:
                    df, filename = result
                    results['load_test_successful'] = True
                    results['load_test_dataset'] = filename
                    results['load_test_shape'] = (len(df), len(df.columns))
                    
                    logger.info(f"  ✓ Successfully loaded dataset: {filename}")
                    logger.info(f"    Shape: {len(df)} rows × {len(df.columns)} columns")
                    logger.info(f"    Columns: {list(df.columns)}")
                    
                    # Log sample data
                    if len(df) > 0:
                        logger.debug(f"    First row date: {df['date'].iloc[0] if 'date' in df.columns else 'N/A'}")
                        logger.debug(f"    Last row date: {df['date'].iloc[-1] if 'date' in df.columns else 'N/A'}")
                        logger.debug(f"    Currency pair: {df['currency_pair'].iloc[0] if 'currency_pair' in df.columns else 'N/A'}")
                else:
                    error_msg = "Failed to load dataset (validation or processing error)"
                    results['errors'].append(error_msg)
                    logger.error(f"  ✗ {error_msg}")
            except Exception as e:
                error_msg = f"Exception during dataset loading: {str(e)}"
                results['errors'].append(error_msg)
                logger.error(f"  ✗ {error_msg}", exc_info=True)
        else:
            logger.warning(f"  ⚠ Skipping load test (no CSV files available)")
            results['errors'].append("Cannot test loading: no CSV files found")
        
        # Summary
        logger.info("=" * 60)
        logger.info("Verification Summary:")
        logger.info(f"  Directory exists: {results['datasets_dir_exists']}")
        logger.info(f"  CSV files found: {results['csv_files_count']}")
        logger.info(f"  Load test successful: {results['load_test_successful']}")
        if results['load_test_shape']:
            logger.info(f"  Loaded dataset shape: {results['load_test_shape'][0]} rows × {results['load_test_shape'][1]} columns")
        if results['errors']:
            logger.warning(f"  Errors: {len(results['errors'])}")
            for error in results['errors']:
                logger.warning(f"    - {error}")
        logger.info("=" * 60)
        
        return results


# Global dataset manager instance
dataset_manager = DatasetManager()

