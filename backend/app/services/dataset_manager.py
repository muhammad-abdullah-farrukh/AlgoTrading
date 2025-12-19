"""
Dataset Queue Management System

Manages dataset queue using FIFO (First In First Out) principle.
- Datasets folder contains untrained datasets
- TrainedDS folder contains trained datasets
- When retrain is triggered, uses first dataset from Datasets folder
- After training, moves dataset to TrainedDS folder
"""
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
from app.utils.logging import get_logger

logger = get_logger(__name__)


class DatasetManager:
    """Manages dataset queue and training workflow."""
    
    def __init__(self, datasets_dir: str = "Datasets", trained_dir: str = "TrainedDS"):
        """
        Initialize dataset manager.
        
        Args:
            datasets_dir: Directory containing untrained datasets
            trained_dir: Directory for trained datasets
        """
        # Get project root (parent of backend directory)
        project_root = Path(__file__).parent.parent.parent
        self.datasets_dir = project_root / datasets_dir
        self.trained_dir = project_root / trained_dir
        
        # Create directories if they don't exist
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.trained_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Dataset manager initialized: datasets={self.datasets_dir}, trained={self.trained_dir}")
    
    def get_next_dataset(self) -> Optional[Tuple[str, Path]]:
        """
        Get the next dataset in FIFO order (oldest first).
        
        Returns:
            Tuple of (filename, full_path) or None if no datasets available
        """
        try:
            # Get all CSV files in datasets directory
            csv_files = list(self.datasets_dir.glob("*.csv"))
            
            if not csv_files:
                logger.warning("No CSV datasets found in Datasets folder")
                return None
            
            # Sort by modification time (oldest first - FIFO)
            csv_files.sort(key=lambda p: p.stat().st_mtime)
            
            oldest_file = csv_files[0]
            filename = oldest_file.name
            
            logger.info(f"Next dataset for training: {filename} (FIFO order)")
            return (filename, oldest_file)
            
        except Exception as e:
            logger.error(f"Failed to get next dataset: {str(e)}")
            return None
    
    def move_to_trained(self, filename: str) -> bool:
        """
        Move a dataset from Datasets folder to TrainedDS folder.
        
        Args:
            filename: Name of the dataset file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            source_path = self.datasets_dir / filename
            target_path = self.trained_dir / filename
            
            if not source_path.exists():
                logger.error(f"Dataset file not found: {source_path}")
                return False
            
            # Move file to trained directory
            shutil.move(str(source_path), str(target_path))
            
            logger.info(f"Moved dataset to trained folder: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to move dataset to trained folder: {str(e)}")
            return False
    
    def list_datasets(self) -> List[str]:
        """
        List all datasets in the Datasets folder.
        
        Returns:
            List of dataset filenames
        """
        try:
            csv_files = list(self.datasets_dir.glob("*.csv"))
            return sorted([f.name for f in csv_files])
        except Exception as e:
            logger.error(f"Failed to list datasets: {str(e)}")
            return []
    
    def list_trained_datasets(self) -> List[str]:
        """
        List all datasets in the TrainedDS folder.
        
        Returns:
            List of trained dataset filenames
        """
        try:
            csv_files = list(self.trained_dir.glob("*.csv"))
            return sorted([f.name for f in csv_files])
        except Exception as e:
            logger.error(f"Failed to list trained datasets: {str(e)}")
            return []
    
    def add_dataset(self, source_path: str, filename: Optional[str] = None) -> bool:
        """
        Add a new dataset to the Datasets folder.
        
        Args:
            source_path: Path to the source dataset file
            filename: Optional custom filename (uses source filename if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            source = Path(source_path)
            if not source.exists():
                logger.error(f"Source dataset not found: {source_path}")
                return False
            
            target_filename = filename or source.name
            target_path = self.datasets_dir / target_filename
            
            # Copy file to datasets directory
            shutil.copy2(str(source), str(target_path))
            
            logger.info(f"Added dataset to queue: {target_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add dataset: {str(e)}")
            return False
    
    def get_dataset_path(self, filename: str) -> Optional[Path]:
        """
        Get full path to a dataset file.
        
        Args:
            filename: Name of the dataset file
            
        Returns:
            Path object or None if not found
        """
        # Check in datasets folder first
        path = self.datasets_dir / filename
        if path.exists():
            return path
        
        # Check in trained folder
        path = self.trained_dir / filename
        if path.exists():
            return path
        
        return None


# Global dataset manager instance
dataset_manager = DatasetManager()

