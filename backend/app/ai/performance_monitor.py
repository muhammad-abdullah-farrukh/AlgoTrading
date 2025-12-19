"""
Model Performance Monitoring Service

Tracks and stores model performance metrics over time for auditability.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import db
from app.models.model_performance import ModelPerformance
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceMonitor:
    """
    Service for monitoring and tracking ML model performance over time.
    
    Features:
    - Record performance metrics after training
    - Track accuracy over time
    - Generate model version hashes
    - Store metrics in database
    - Export performance history to CSV
    """
    
    def __init__(self):
        """Initialize PerformanceMonitor."""
        logger.debug("PerformanceMonitor initialized")
    
    def _calculate_model_hash(self, model_path: Path) -> str:
        """
        Calculate SHA-256 hash of model file for version tracking.
        
        Args:
            model_path: Path to model file
            
        Returns:
            SHA-256 hash string (64 characters)
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(model_path, "rb") as f:
                # Read file in chunks to handle large files
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate model hash: {str(e)}", exc_info=True)
            # Return timestamp-based hash as fallback
            return hashlib.sha256(str(datetime.now().isoformat()).encode()).hexdigest()
    
    async def record_performance(
        self,
        model_type: str,
        model_path: str,
        timeframe: str,
        accuracy: float,
        sample_size: int,
        train_size: int,
        test_size: int,
        feature_count: int,
        dataset_sources: List[str],
        metadata_path: Optional[str] = None,
        loss: Optional[float] = None,
        additional_metadata: Optional[Dict] = None,
        trained_at: Optional[datetime] = None
    ) -> ModelPerformance:
        """
        Record model performance metrics in database.
        
        Args:
            model_type: Type of model (e.g., 'logistic_regression')
            model_path: Path to model file
            timeframe: Timeframe used (e.g., '1d')
            accuracy: Model accuracy (0.0-1.0)
            sample_size: Total sample size
            train_size: Training set size
            test_size: Test set size
            feature_count: Number of features
            dataset_sources: List of dataset filenames used
            metadata_path: Optional path to metadata file
            loss: Optional loss value
            additional_metadata: Optional additional metrics (classification report, etc.)
            trained_at: Optional training timestamp (defaults to now)
            
        Returns:
            ModelPerformance record
        """
        try:
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                raise ValueError(f"Model file not found: {model_path}")
            
            # Calculate model version hash
            model_version_hash = self._calculate_model_hash(model_path_obj)
            
            # Calculate error rate
            error_rate = 1.0 - accuracy if accuracy is not None else None
            
            # Create performance record
            performance_record = ModelPerformance(
                model_type=model_type,
                model_version_hash=model_version_hash,
                timeframe=timeframe,
                accuracy=float(accuracy),
                loss=float(loss) if loss is not None else None,
                error_rate=float(error_rate) if error_rate is not None else None,
                sample_size=int(sample_size),
                train_size=int(train_size),
                test_size=int(test_size),
                feature_count=int(feature_count),
                dataset_sources=dataset_sources,
                model_path=str(model_path),
                metadata_path=str(metadata_path) if metadata_path else None,
                additional_metadata=additional_metadata,
                trained_at=trained_at or datetime.now()
            )
            
            # Save to database
            async with db.get_session() as session:
                session.add(performance_record)
                await session.commit()
                await session.refresh(performance_record)
            
            logger.info(
                f"Performance recorded: {model_type} ({timeframe}) - "
                f"Accuracy: {accuracy:.4f}, Hash: {model_version_hash[:16]}..."
            )
            
            return performance_record
            
        except Exception as e:
            error_msg = f"Failed to record performance: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    async def get_performance_history(
        self,
        model_type: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: int = 100
    ) -> List[ModelPerformance]:
        """
        Get performance history for models.
        
        Args:
            model_type: Optional filter by model type
            timeframe: Optional filter by timeframe
            limit: Maximum number of records to return
            
        Returns:
            List of ModelPerformance records (newest first)
        """
        try:
            async with db.get_session() as session:
                query = select(ModelPerformance)
                
                if model_type:
                    query = query.where(ModelPerformance.model_type == model_type)
                if timeframe:
                    query = query.where(ModelPerformance.timeframe == timeframe)
                
                query = query.order_by(desc(ModelPerformance.recorded_at)).limit(limit)
                
                result = await session.execute(query)
                records = result.scalars().all()
                
                logger.debug(f"Retrieved {len(records)} performance records")
                return list(records)
                
        except Exception as e:
            error_msg = f"Failed to get performance history: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    async def get_latest_performance(
        self,
        model_type: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> Optional[ModelPerformance]:
        """
        Get latest performance record for a model.
        
        Args:
            model_type: Optional filter by model type
            timeframe: Optional filter by timeframe
            
        Returns:
            Latest ModelPerformance record or None
        """
        records = await self.get_performance_history(
            model_type=model_type,
            timeframe=timeframe,
            limit=1
        )
        return records[0] if records else None
    
    async def get_accuracy_trend(
        self,
        model_type: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get accuracy trend over time.
        
        Args:
            model_type: Optional filter by model type
            timeframe: Optional filter by timeframe
            limit: Maximum number of records
            
        Returns:
            List of dicts with 'recorded_at' and 'accuracy'
        """
        records = await self.get_performance_history(
            model_type=model_type,
            timeframe=timeframe,
            limit=limit
        )
        
        return [
            {
                'recorded_at': record.recorded_at.isoformat() if record.recorded_at else None,
                'accuracy': record.accuracy,
                'sample_size': record.sample_size,
                'model_version_hash': record.model_version_hash[:16] + '...'
            }
            for record in reversed(records)  # Oldest first for trend
        ]
    
    def export_performance_history_to_csv(
        self,
        records: List[ModelPerformance],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export performance history to CSV file.
        
        Args:
            records: List of ModelPerformance records
            output_path: Optional output path (default: auto-generated)
            
        Returns:
            Path to exported CSV file
        """
        try:
            import pandas as pd
            
            if not records:
                raise ValueError("No records to export")
            
            # Convert records to dictionaries
            data = [record.to_dict() for record in records]
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Flatten dataset_sources (JSON array to comma-separated string)
            if 'dataset_sources' in df.columns:
                df['dataset_sources'] = df['dataset_sources'].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else str(x)
                )
            
            # Generate output path
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                current_file = Path(__file__).resolve()
                export_dir = current_file.parent / "data" / "Pipeline"
                export_dir.mkdir(parents=True, exist_ok=True)
                output_path = export_dir / f"performance_history_{timestamp}.csv"
            
            # Export to CSV
            df.to_csv(output_path, index=False)
            logger.info(f"Performance history exported to: {output_path}")
            logger.info(f"  Records: {len(df)}")
            
            return output_path
            
        except Exception as e:
            error_msg = f"Failed to export performance history: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e


# Global performance monitor instance
performance_monitor = PerformanceMonitor()

