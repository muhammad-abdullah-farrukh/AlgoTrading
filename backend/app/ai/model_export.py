"""
Model Export Service

Exports model weights and metadata to CSV files for analysis and tracking.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
from app.utils.logging import get_logger
from app.ai.models.logistic_regression import logistic_model

logger = get_logger(__name__)


class ModelExportService:
    """
    Service for exporting model weights and metadata to CSV files.
    
    Features:
    - Export model weights (coefficients) to CSV
    - Export metadata (accuracy, loss, sample size, timestamps) to CSV
    - Store exports in pipeline directory
    - Ensure files are accurate and complete
    """
    
    def __init__(self, export_dir: Optional[Path] = None):
        """
        Initialize ModelExportService.
        
        Args:
            export_dir: Directory to save exports (default: app/ai/data/Pipeline)
        """
        # Get export directory path
        if export_dir is None:
            current_file = Path(__file__).resolve()
            export_dir = current_file.parent / "data" / "Pipeline"
        
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"ModelExportService initialized. Export directory: {self.export_dir}")
    
    def _calculate_loss(self, X: np.ndarray, y: np.ndarray) -> Optional[float]:
        """
        Calculate log loss for the model.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Log loss value or None if calculation fails
        """
        try:
            if self.model.model is None:
                return None
            
            # Get prediction probabilities
            y_proba = self.model.model.predict_proba(X)
            
            # Calculate log loss
            loss = log_loss(y, y_proba)
            return float(loss)
        except Exception as e:
            logger.warning(f"Failed to calculate loss: {str(e)}")
            return None
    
    def export_weights(
        self,
        model_path: Optional[Path] = None,
        output_filename: Optional[str] = None
    ) -> Tuple[Optional[Path], Dict]:
        """
        Export model weights (coefficients) to CSV.
        
        Args:
            model_path: Path to model file (default: load latest)
            output_filename: Optional output filename (default: auto-generated)
            
        Returns:
            Tuple of (output_path, export_info):
            - output_path: Path to exported CSV file
            - export_info: Dictionary with export details
        """
        try:
            # Load model if not already loaded
            if self.model.model is None:
                if not self.model.load_model(model_path):
                    raise RuntimeError("No model available to export")
            
            if self.model.model is None:
                raise RuntimeError("Model not loaded")
            
            # Get feature names
            feature_names = self.model.feature_names
            if not feature_names:
                # Try to get from metadata
                metadata = self.model.get_metadata()
                if metadata:
                    feature_names = metadata.get('feature_names', [])
            
            if not feature_names:
                # Fallback: use generic names
                n_features = self.model.model.coef_.shape[1]
                feature_names = [f"feature_{i}" for i in range(n_features)]
                logger.warning(f"Feature names not available, using generic names")
            
            # Get coefficients and intercept
            coefficients = self.model.model.coef_[0]  # Flatten to 1D array
            intercept = self.model.model.intercept_[0]
            
            # Create DataFrame
            weights_data = {
                'feature_name': feature_names + ['intercept'],
                'weight': np.append(coefficients, intercept),
                'abs_weight': np.append(np.abs(coefficients), np.abs(intercept))
            }
            
            df_weights = pd.DataFrame(weights_data)
            
            # Sort by absolute weight (descending) for easier analysis
            df_weights = df_weights.sort_values('abs_weight', ascending=False).reset_index(drop=True)
            
            # Generate output filename
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                metadata = self.model.get_metadata()
                if metadata:
                    timeframe = metadata.get('timeframe', 'unknown')
                    output_filename = f"model_weights_{timeframe}_{timestamp}.csv"
                else:
                    output_filename = f"model_weights_{timestamp}.csv"
            
            output_path = self.export_dir / output_filename
            
            # Export to CSV
            df_weights.to_csv(output_path, index=False)
            logger.info(f"Model weights exported to: {output_path}")
            logger.info(f"  Features: {len(feature_names)}, Total weights: {len(df_weights)}")
            
            export_info = {
                'weights_path': str(output_path),
                'feature_count': len(feature_names),
                'total_weights': len(df_weights),
                'timestamp': datetime.now().isoformat()
            }
            
            return output_path, export_info
            
        except Exception as e:
            error_msg = f"Failed to export model weights: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def export_metadata(
        self,
        model_path: Optional[Path] = None,
        output_filename: Optional[str] = None,
        include_loss: bool = True
    ) -> Tuple[Optional[Path], Dict]:
        """
        Export model metadata to CSV.
        
        Exports key metrics: accuracy, loss (if available), sample size, timestamps.
        
        Args:
            model_path: Path to model file (default: load latest)
            output_filename: Optional output filename (default: auto-generated)
            include_loss: Whether to calculate and include loss (default: True)
            
        Returns:
            Tuple of (output_path, export_info):
            - output_path: Path to exported CSV file
            - export_info: Dictionary with export details
        """
        try:
            # Load model if not already loaded
            if self.model.model is None:
                if not self.model.load_model(model_path):
                    raise RuntimeError("No model available to export")
            
            # Get metadata
            metadata = self.model.get_metadata()
            if not metadata:
                raise RuntimeError("Model metadata not available")
            
            # Extract key metrics
            metrics_data = {
                'metric': [],
                'value': [],
                'description': []
            }
            
            # Accuracy
            accuracy = metadata.get('accuracy')
            if accuracy is not None:
                metrics_data['metric'].append('accuracy')
                metrics_data['value'].append(f"{accuracy:.6f}")
                metrics_data['description'].append("Model accuracy on test set")
            
            # Loss (calculate if requested)
            loss = None
            if include_loss:
                # Try to get loss from metadata first
                loss = metadata.get('loss')
                
                # If not in metadata, try to calculate (requires test data)
                # For now, we'll note that loss calculation requires test data
                if loss is None:
                    metrics_data['metric'].append('loss')
                    metrics_data['value'].append('N/A')
                    metrics_data['description'].append("Log loss (requires test data for calculation)")
                else:
                    metrics_data['metric'].append('loss')
                    metrics_data['value'].append(f"{loss:.6f}")
                    metrics_data['description'].append("Log loss on test set")
            
            # Sample size
            sample_size = metadata.get('sample_size')
            if sample_size is not None:
                metrics_data['metric'].append('sample_size')
                metrics_data['value'].append(str(sample_size))
                metrics_data['description'].append("Total number of samples used for training")
            
            # Train size
            train_size = metadata.get('train_size')
            if train_size is not None:
                metrics_data['metric'].append('train_size')
                metrics_data['value'].append(str(train_size))
                metrics_data['description'].append("Number of samples in training set")
            
            # Test size
            test_size = metadata.get('test_size')
            if test_size is not None:
                metrics_data['metric'].append('test_size')
                metrics_data['value'].append(str(test_size))
                metrics_data['description'].append("Number of samples in test set")
            
            # Feature count
            feature_count = metadata.get('feature_count')
            if feature_count is not None:
                metrics_data['metric'].append('feature_count')
                metrics_data['value'].append(str(feature_count))
                metrics_data['description'].append("Number of features used in model")
            
            # Timestamps
            timestamp = metadata.get('timestamp')
            if timestamp:
                metrics_data['metric'].append('last_trained')
                metrics_data['value'].append(timestamp)
                metrics_data['description'].append("Timestamp when model was last trained")
            
            last_retrained = metadata.get('last_retrained')
            if last_retrained:
                metrics_data['metric'].append('last_retrained')
                metrics_data['value'].append(last_retrained)
                metrics_data['description'].append("Timestamp when model was last retrained")
            
            # Model info
            model_type = metadata.get('model_type', 'unknown')
            metrics_data['metric'].append('model_type')
            metrics_data['value'].append(model_type)
            metrics_data['description'].append("Type of ML model")
            
            timeframe = metadata.get('timeframe', 'unknown')
            metrics_data['metric'].append('timeframe')
            metrics_data['value'].append(timeframe)
            metrics_data['description'].append("Timeframe used for training")
            
            # Dataset source (Phase V-1 requirement)
            datasets_used = metadata.get('datasets_used', [])
            if datasets_used:
                metrics_data['metric'].append('dataset_source')
                metrics_data['value'].append(', '.join(datasets_used))
                metrics_data['description'].append("Dataset files used for training")
            
            # Feature list (Phase V-1 requirement)
            feature_names = metadata.get('feature_names', [])
            if feature_names:
                metrics_data['metric'].append('feature_list')
                metrics_data['value'].append(', '.join(feature_names))
                metrics_data['description'].append("List of features used in the model")
            
            # Preprocessing steps (Phase V-1 requirement)
            preprocessing_steps = [
                "Normalized dataset format (wide to long if needed)",
                "Feature engineering: lagged returns (1, 2, 3, 5, 10 periods)",
                "Feature engineering: moving averages (SMA/EMA for windows 5, 10, 20, 50, 100)",
                "Feature engineering: price differences (high-low, close-open)",
                "Target variable: binary classification (1=price up, 0=price down)",
                "Train/test split (80/20 default)",
                "Class balancing (balanced class weights)"
            ]
            metrics_data['metric'].append('preprocessing_steps')
            metrics_data['value'].append('; '.join(preprocessing_steps))
            metrics_data['description'].append("Preprocessing and feature engineering steps applied")
            
            # Create DataFrame
            df_metadata = pd.DataFrame(metrics_data)
            
            # Generate output filename
            if output_filename is None:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"model_metadata_{timeframe}_{timestamp_str}.csv"
            
            output_path = self.export_dir / output_filename
            
            # Export to CSV
            df_metadata.to_csv(output_path, index=False)
            logger.info(f"Model metadata exported to: {output_path}")
            logger.info(f"  Metrics exported: {len(df_metadata)}")
            
            export_info = {
                'metadata_path': str(output_path),
                'metrics_count': len(df_metadata),
                'timestamp': datetime.now().isoformat(),
                'accuracy': accuracy,
                'loss': loss,
                'sample_size': sample_size
            }
            
            return output_path, export_info
            
        except Exception as e:
            error_msg = f"Failed to export model metadata: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def export_all(
        self,
        model_path: Optional[Path] = None,
        include_loss: bool = True
    ) -> Dict:
        """
        Export both weights and metadata.
        
        Args:
            model_path: Path to model file (default: load latest)
            include_loss: Whether to include loss in metadata (default: True)
            
        Returns:
            Dictionary with export results:
            {
                'weights_path': str,
                'metadata_path': str,
                'weights_info': dict,
                'metadata_info': dict,
                'timestamp': str
            }
        """
        logger.info("=" * 60)
        logger.info("Starting Model Export")
        logger.info("=" * 60)
        
        try:
            # Export weights
            weights_path, weights_info = self.export_weights(model_path=model_path)
            
            # Export metadata
            metadata_path, metadata_info = self.export_metadata(
                model_path=model_path,
                include_loss=include_loss
            )
            
            logger.info("=" * 60)
            logger.info("Model Export Complete")
            logger.info("=" * 60)
            
            return {
                'success': True,
                'weights_path': str(weights_path),
                'metadata_path': str(metadata_path),
                'weights_info': weights_info,
                'metadata_info': metadata_info,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Model export failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'success': False,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
    
    @property
    def model(self):
        """Get the logistic model instance."""
        return logistic_model


# Global export service instance
model_export_service = ModelExportService()

