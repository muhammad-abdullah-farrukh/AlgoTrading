"""
Logistic Regression Model for Trading Signals

Implements logistic regression for binary classification (price up/down prediction).
Model training is explicitly triggered - no automatic training on startup.
"""
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from app.utils.logging import get_logger
from app.ai.feature_engineering import feature_engineer
from app.ai.dataset_manager import dataset_manager

logger = get_logger(__name__)


class LogisticRegressionModel:
    """
    Logistic Regression model for currency price prediction.
    
    Features:
    - Train/test split
    - Model persistence (save/load)
    - Training metadata tracking
    - Evaluation metrics
    
    Important: Training is NOT automatic - must be explicitly triggered.
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize LogisticRegressionModel.
        
        Args:
            models_dir: Directory to save/load models (default: app/ai/models)
        """
        # Get models directory path
        if models_dir is None:
            current_file = Path(__file__).resolve()
            models_dir = current_file.parent
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model state
        self.model: Optional[LogisticRegression] = None
        self.metadata: Optional[Dict] = None
        self.feature_names: List[str] = []
        
        logger.debug(f"LogisticRegressionModel initialized. Models directory: {self.models_dir}")
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get feature column names (exclude non-feature columns).
        
        Args:
            df: DataFrame with features
            
        Returns:
            List of feature column names
        """
        exclude_cols = {'date', 'currency_pair', 'close_price', 'target', 
                       'open', 'high', 'low', 'volume'}
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def train(
        self,
        timeframe: str,
        test_size: float = 0.2,
        random_state: int = 42,
        max_iter: int = 1000
    ) -> Dict:
        """
        Train logistic regression model.
        
        This method:
        1. Loads datasets from Datasets/ directory (FIFO order)
        2. Applies feature engineering
        3. Splits into train/test sets
        4. Trains logistic regression
        5. Calculates accuracy
        6. Saves model and metadata
        
        Args:
            timeframe: Timeframe string (e.g., '1h', '1d', '1w')
            test_size: Proportion of data for testing (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
            max_iter: Maximum iterations for logistic regression (default: 1000)
            
        Returns:
            Dictionary with training results:
            {
                'success': bool,
                'accuracy': float,
                'train_size': int,
                'test_size': int,
                'model_path': str,
                'metadata_path': str,
                'datasets_used': List[str],
                'timeframe': str,
                'timestamp': str
            }
            
        Raises:
            ValueError: If no datasets available or invalid timeframe
            RuntimeError: If training fails
        """
        logger.info("=" * 60)
        logger.info("STARTING LOGISTIC REGRESSION MODEL TRAINING")
        logger.info("=" * 60)
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Test size: {test_size}")
        print("\n" + "=" * 60)
        print("STARTING LOGISTIC REGRESSION MODEL TRAINING")
        print("=" * 60)
        print(f"Timeframe: {timeframe}")
        print(f"Test size: {test_size}")
        print("=" * 60 + "\n")
        
        try:
            # Reset loaded files tracking for new training session
            dataset_manager.reset_loaded_files()
            
            # 1. Load all available datasets (FIFO order)
            logger.info("Loading datasets from Datasets/ directory...")
            print("📂 Step 1/6: Loading datasets...")
            all_datasets = []
            datasets_used = []
            dataset_count = 0
            max_datasets = 100  # Safety limit to prevent infinite loops
            
            while dataset_count < max_datasets:
                result = dataset_manager.load_next_dataset()
                if result is None:
                    break
                
                df, filename = result
                all_datasets.append((df, filename))
                datasets_used.append(filename)
                dataset_count += 1
                total_rows = len(df)
                logger.info(f"  Loaded dataset: {filename} ({total_rows} rows)")
                print(f"   [OK] Loaded dataset #{dataset_count}: {filename} ({total_rows:,} rows)")
            
            if not all_datasets:
                raise ValueError("No datasets available in Datasets/ directory. Cannot train model.")
            
            if dataset_count >= max_datasets:
                logger.warning(f"Reached maximum dataset limit ({max_datasets}). Stopping dataset loading.")
                print(f"   [WARNING] Reached maximum dataset limit ({max_datasets}). Using {len(all_datasets)} dataset(s).")
            
            print(f"   [OK] Total datasets loaded: {dataset_count}")
            
            # Combine all datasets
            logger.info(f"Combining {len(all_datasets)} dataset(s)...")
            print(f"📊 Step 2/6: Combining {len(all_datasets)} dataset(s)...")
            combined_df = pd.concat([df for df, _ in all_datasets], ignore_index=True)
            
            # Remove duplicates based on date and currency_pair
            if 'currency_pair' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(
                    subset=['date', 'currency_pair'],
                    keep='last'
                ).sort_values('date').reset_index(drop=True)
            else:
                combined_df = combined_df.drop_duplicates(
                    subset=['date'],
                    keep='last'
                ).sort_values('date').reset_index(drop=True)
            
            logger.info(f"Combined dataset shape: {combined_df.shape}")
            print(f"   [OK] Combined dataset: {combined_df.shape[0]:,} rows, {combined_df.shape[1]} columns")
            
            # 2. Apply feature engineering
            logger.info("Applying feature engineering...")
            print("🔧 Step 3/6: Applying feature engineering...")
            print("   → Generating lagged returns...")
            df_features = feature_engineer.prepare_features(
                df=combined_df,
                timeframe=timeframe,
                include_target=True
            )
            print("   [OK] Feature engineering complete")
            
            if len(df_features) == 0:
                raise ValueError("Feature engineering produced empty dataset. Cannot train model.")
            
            # 3. Prepare features and target
            feature_cols = self._get_feature_columns(df_features)
            if not feature_cols:
                raise ValueError("No feature columns found after feature engineering.")
            
            X = df_features[feature_cols].values
            y = df_features['target'].values
            
            self.feature_names = feature_cols
            
            logger.info(f"Features: {len(feature_cols)}")
            logger.info(f"Total samples: {len(X)}")
            logger.info(f"Target distribution: Up={np.sum(y==1)}, Down={np.sum(y==0)}")
            print(f"   [OK] Features generated: {len(feature_cols)}")
            print(f"   [OK] Total samples: {len(X):,}")
            print(f"   [OK] Target distribution: Up={np.sum(y==1):,} ({np.sum(y==1)/len(y)*100:.1f}%), Down={np.sum(y==0):,} ({np.sum(y==0)/len(y)*100:.1f}%)")
            
            # 4. Split into train/test sets
            logger.info(f"Splitting data (train: {1-test_size:.1%}, test: {test_size:.1%})...")
            print(f"✂️  Step 4/6: Splitting data (train: {1-test_size:.1%}, test: {test_size:.1%})...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y  # Maintain class distribution
            )
            
            train_size = len(X_train)
            test_size_actual = len(X_test)
            
            logger.info(f"Train set: {train_size} samples")
            logger.info(f"Test set: {test_size_actual} samples")
            print(f"   [OK] Train set: {train_size:,} samples")
            print(f"   [OK] Test set: {test_size_actual:,} samples")
            
            # 5. Train logistic regression
            logger.info("Training logistic regression model...")
            print(f"🤖 Step 5/6: Training logistic regression model...")
            print(f"   → Using LBFGS solver (max_iter={max_iter})...")
            print(f"   → This may take a few seconds to a minute depending on dataset size...")
            self.model = LogisticRegression(
                max_iter=max_iter,
                random_state=random_state,
                solver='lbfgs',  # Good for small-medium datasets
                class_weight='balanced'  # Handle class imbalance
            )
            
            self.model.fit(X_train, y_train)
            logger.info("Model training complete")
            print("   [OK] Model training complete!")
            
            # 6. Evaluate on test set
            logger.info("Evaluating model on test set...")
            print(f"📈 Step 6/6: Evaluating model on test set...")
            print("   → Generating predictions...")
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"   [OK] Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Additional metrics
            print("   → Calculating additional metrics...")
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            logger.debug(f"Classification report:\n{classification_report(y_test, y_pred)}")
            logger.debug(f"Confusion matrix:\n{cm}")
            
            # 7. Save model and metadata
            print("\n💾 Saving model and metadata...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"logistic_regression_{timeframe}_{timestamp}.pkl"
            metadata_filename = f"logistic_regression_{timeframe}_{timestamp}_metadata.json"
            
            model_path = self.models_dir / model_filename
            metadata_path = self.models_dir / metadata_filename
            
            # Save model
            print("   → Saving model file...")
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved: {model_path}")
            print(f"   [OK] Model saved: {model_path.name}")
            
            # Save metadata
            print("   → Saving metadata...")
            self.metadata = {
                'model_type': 'logistic_regression',
                'timeframe': timeframe,
                'accuracy': float(accuracy),
                'train_size': int(train_size),
                'test_size': int(test_size_actual),
                'sample_size': int(len(X)),
                'feature_count': len(feature_cols),
                'feature_names': feature_cols,
                'datasets_used': datasets_used,
                'timestamp': datetime.now().isoformat(),
                'model_path': str(model_path),
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'training_params': {
                    'test_size': test_size,
                    'random_state': random_state,
                    'max_iter': max_iter
                },
                'last_retrained': datetime.now().isoformat(),  # Track retraining time
                'retrain_reason': 'Initial training'  # Will be updated by retraining service
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"Metadata saved: {metadata_path}")
            print(f"   [OK] Metadata saved: {metadata_path.name}")
            
            # Also update in-memory metadata for immediate access
            self.metadata = self.metadata.copy()
            
            # 8. Mark datasets as trained (move to TrainedDS/)
            print("\n📦 Marking datasets as trained...")
            for idx, (_, filename) in enumerate(all_datasets, 1):
                dataset_manager.mark_dataset_as_trained(filename)
                logger.debug(f"  Moved {filename} to TrainedDS/")
                print(f"   [OK] Moved dataset {idx}/{len(all_datasets)}: {filename}")
            
            # 9. Export model weights and metadata to CSV (Phase ML-6)
            print("\n📤 Exporting model weights and metadata to CSV...")
            try:
                from app.ai.model_export import model_export_service
                logger.info("Exporting model weights and metadata...")
                print("   → Exporting weights...")
                export_result = model_export_service.export_all(model_path=model_path, include_loss=False)
                if export_result.get('success'):
                    logger.info(f"  ✓ Weights exported: {export_result.get('weights_path')}")
                    logger.info(f"  ✓ Metadata exported: {export_result.get('metadata_path')}")
                    weights_file = Path(export_result.get('weights_path', '')).name
                    metadata_file = Path(export_result.get('metadata_path', '')).name
                    print(f"   [OK] Weights exported: {weights_file}")
                    print(f"   [OK] Metadata exported: {metadata_file}")
                else:
                    logger.warning(f"  ⚠ Export failed: {export_result.get('error')}")
                    print(f"   [WARNING] Export failed: {export_result.get('error')}")
            except Exception as e:
                logger.warning(f"  [WARNING] Model export failed (non-critical): {str(e)}")
                print(f"   [WARNING] Model export failed (non-critical): {str(e)}")
            
            logger.info("=" * 60)
            logger.info("MODEL TRAINING COMPLETE!")
            logger.info("=" * 60)
            print("\n" + "=" * 60)
            print("[SUCCESS] MODEL TRAINING COMPLETE!")
            print("=" * 60)
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Train Size: {train_size:,} samples")
            print(f"Test Size: {test_size_actual:,} samples")
            print(f"Total Samples: {len(X):,}")
            print(f"Timeframe: {timeframe}")
            print(f"Model Saved: {model_path}")
            print(f"Metadata Saved: {metadata_path}")
            print(f"Datasets Used: {len(datasets_used)} file(s)")
            for ds in datasets_used:
                print(f"   - {ds}")
            print("=" * 60 + "\n")
            
            return {
                'success': True,
                'accuracy': float(accuracy),
                'train_size': int(train_size),
                'test_size': int(test_size_actual),
                'sample_size': int(len(X)),
                'model_path': str(model_path),
                'metadata_path': str(metadata_path),
                'datasets_used': datasets_used,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model on given data.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with evaluation metrics:
            {
                'accuracy': float,
                'classification_report': dict,
                'confusion_matrix': list
            }
            
        Raises:
            RuntimeError: If model is not trained
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        logger.info("Evaluating model...")
        
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        report = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)
        
        logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return {
            'accuracy': float(accuracy),
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
    
    def load_model(self, model_path: Optional[Path] = None) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to model file (default: load latest)
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if model_path is None:
                # Find latest model file
                model_files = list(self.models_dir.glob("logistic_regression_*.pkl"))
                if not model_files:
                    logger.warning("No model files found")
                    return False
                
                # Sort by modification time (newest first)
                model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                model_path = model_files[0]
            
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info(f"Model loaded: {model_path}")
            
            # Try to load corresponding metadata
            metadata_path = model_path.parent / model_path.name.replace('.pkl', '_metadata.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.feature_names = self.metadata.get('feature_names', [])
                logger.info(f"Metadata loaded: {metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            return False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels (0 or 1)
            
        Raises:
            RuntimeError: If model is not trained
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load_model() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability matrix (n_samples, 2) where columns are [P(down), P(up)]
            
        Raises:
            RuntimeError: If model is not trained
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load_model() first.")
        
        return self.model.predict_proba(X)
    
    def get_metadata(self) -> Optional[Dict]:
        """
        Get training metadata.
        
        Returns:
            Dictionary with metadata or None if not available
        """
        return self.metadata


# Global model instance (not automatically trained)
logistic_model = LogisticRegressionModel()

