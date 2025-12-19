"""
AI Signal Generator for Trading

Generates BUY/SELL trading signals from trained ML models.
Supports multiple timeframes and provides confidence scores.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from app.utils.logging import get_logger
# Lazy import to avoid breaking if sklearn is not installed
try:
    from app.ai.models.logistic_regression import LogisticRegressionModel, logistic_model
except ImportError:
    # sklearn not available - will raise error when used
    LogisticRegressionModel = None
    logistic_model = None

from app.ai.feature_engineering import feature_engineer

logger = get_logger(__name__)


class SignalGenerator:
    """
    Generates trading signals from trained ML models.
    
    Features:
    - BUY/SELL signal generation
    - Confidence score calculation
    - Multiple timeframe support
    - Model existence validation
    """
    
    # Signal types
    SIGNAL_BUY = "BUY"
    SIGNAL_SELL = "SELL"
    SIGNAL_HOLD = "HOLD"  # Used when model unavailable or low confidence
    
    def __init__(self):
        """Initialize SignalGenerator."""
        # Check if logistic_model is available (not a placeholder)
        if logistic_model is None or not hasattr(logistic_model, 'model'):
            try:
                # Try to access an attribute to trigger __getattr__ if it's a placeholder
                _ = logistic_model.model
            except (ImportError, AttributeError):
                raise ImportError(
                    "scikit-learn is not installed. "
                    "Please install it with: pip install scikit-learn"
                )
        self.model = logistic_model
        logger.debug("SignalGenerator initialized")
    
    def _check_model_exists(self, timeframe: str) -> bool:
        """
        Check if a trained model exists for the given timeframe.
        
        Args:
            timeframe: Timeframe string (e.g., '1h', '1d', '1w')
            
        Returns:
            True if model exists, False otherwise
        """
        # Get models directory
        current_file = Path(__file__).resolve()
        models_dir = current_file.parent / "models"
        
        # Look for model files matching the timeframe
        model_files = list(models_dir.glob(f"logistic_regression_{timeframe}_*.pkl"))
        
        if not model_files:
            logger.warning(f"No trained model found for timeframe: {timeframe}")
            return False
        
        # Sort by modification time (newest first)
        model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_model = model_files[0]
        
        logger.debug(f"Found model for timeframe {timeframe}: {latest_model.name}")
        return True
    
    def _prepare_features_from_data(
        self,
        df: pd.DataFrame,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """
        Prepare features from raw OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data (must have date, close_price)
            timeframe: Timeframe string
            
        Returns:
            DataFrame with features, or None if preparation fails
        """
        try:
            # Apply feature engineering (without target)
            df_features = feature_engineer.prepare_features(
                df=df,
                timeframe=timeframe,
                include_target=False  # No target needed for prediction
            )
            
            if len(df_features) == 0:
                logger.error("Feature engineering produced empty dataset")
                return None
            
            return df_features
            
        except Exception as e:
            logger.error(f"Failed to prepare features: {str(e)}", exc_info=True)
            return None
    
    def _get_latest_features(self, df_features: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Get the latest row of features for prediction.
        
        Uses the exact feature names the model was trained with to ensure compatibility.
        
        Args:
            df_features: DataFrame with features
            
        Returns:
            Feature vector (1D array) for the latest row, or None if unavailable
        """
        if len(df_features) == 0:
            return None
        
        # Get the feature names the model was trained with
        model_feature_names = self.model.feature_names
        
        if not model_feature_names:
            # Fallback: use all non-excluded columns
            logger.warning("Model feature names not available, using all features")
            exclude_cols = {'date', 'currency_pair', 'close_price', 'target',
                           'open', 'high', 'low', 'volume'}
            feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        else:
            # Use exact features the model expects
            feature_cols = model_feature_names
            
            # Check if all required features are available
            missing_features = [f for f in feature_cols if f not in df_features.columns]
            if missing_features:
                logger.error(f"Missing required features: {missing_features}")
                # Fill missing features with 0 as fallback
                for f in missing_features:
                    df_features[f] = 0.0
                logger.warning(f"Filled {len(missing_features)} missing features with 0")
        
        if not feature_cols:
            logger.error("No feature columns found")
            return None
        
        # Get the latest row (most recent data) with only the required features
        latest_features = df_features[feature_cols].iloc[-1].values
        
        # Reshape to 2D array (required by sklearn)
        latest_features = latest_features.reshape(1, -1)
        
        logger.debug(f"Extracted {len(feature_cols)} features for prediction")
        
        return latest_features
    
    def _model_prediction_to_signal(
        self,
        prediction: int,
        probability: float
    ) -> Tuple[str, float]:
        """
        Convert model prediction (0/1) to trading signal (BUY/SELL).
        
        Model output:
        - 1 = price goes up → BUY signal
        - 0 = price goes down → SELL signal
        
        Args:
            prediction: Model prediction (0 or 1)
            probability: Prediction probability (0.0 to 1.0)
            
        Returns:
            Tuple of (signal, confidence_score)
            - signal: "BUY" or "SELL"
            - confidence_score: Probability value (0.0 to 1.0)
        """
        if prediction == 1:
            signal = self.SIGNAL_BUY
            # Confidence is the probability of class 1 (price up)
            confidence = probability
        else:  # prediction == 0
            signal = self.SIGNAL_SELL
            # Confidence is the probability of class 0 (price down)
            confidence = 1.0 - probability
        
        return signal, float(confidence)
    
    def predict_next_period(
        self,
        input_data: pd.DataFrame,
        timeframe: str,
        min_confidence: float = 0.5
    ) -> Dict:
        """
        Predict next period and generate trading signal.
        
        This is the main function for signal generation.
        
        Args:
            input_data: DataFrame with OHLCV data (must have date, close_price)
            timeframe: Timeframe string (e.g., '1h', '1d', '1w')
            min_confidence: Minimum confidence threshold (default: 0.5)
                           Signals below this threshold return HOLD
                           
        Returns:
            Dictionary with prediction results:
            {
                'signal': str,  # 'BUY', 'SELL', or 'HOLD'
                'confidence': float,  # 0.0 to 1.0
                'prediction': int,  # 0 (down) or 1 (up)
                'probability': float,  # Probability of price going up
                'timeframe': str,
                'model_available': bool,
                'error': Optional[str]
            }
        """
        logger.info(f"Generating signal for timeframe: {timeframe}")
        
        result = {
            'signal': self.SIGNAL_HOLD,
            'confidence': 0.0,
            'prediction': None,
            'probability': None,
            'timeframe': timeframe,
            'model_available': False,
            'error': None
        }
        
        try:
            # 1. Check if model exists
            if not self._check_model_exists(timeframe):
                result['error'] = f"No trained model found for timeframe: {timeframe}"
                logger.warning(result['error'])
                return result
            
            result['model_available'] = True
            
            # 2. Load model if not already loaded
            if self.model.model is None:
                logger.info(f"Loading model for timeframe: {timeframe}")
                # Try to load the latest model for this timeframe
                if not self.model.load_model():
                    result['error'] = "Failed to load model"
                    logger.error(result['error'])
                    return result
            
            # 3. Prepare features from input data
            logger.debug("Preparing features from input data...")
            df_features = self._prepare_features_from_data(input_data, timeframe)
            
            if df_features is None:
                result['error'] = "Failed to prepare features"
                logger.error(result['error'])
                return result
            
            # 4. Get latest features for prediction
            latest_features = self._get_latest_features(df_features)
            
            if latest_features is None:
                result['error'] = "Failed to extract features"
                logger.error(result['error'])
                return result
            
            # 5. Make prediction
            logger.debug("Making prediction...")
            prediction = self.model.predict(latest_features)[0]  # Get single prediction
            probabilities = self.model.predict_proba(latest_features)[0]  # Get probabilities
            
            # Probability of class 1 (price going up)
            prob_up = float(probabilities[1])
            
            result['prediction'] = int(prediction)
            result['probability'] = prob_up
            
            # 6. Convert to signal
            signal, confidence = self._model_prediction_to_signal(prediction, prob_up)
            
            result['signal'] = signal
            result['confidence'] = confidence
            
            # 7. Apply confidence threshold
            if confidence < min_confidence:
                logger.info(
                    f"Confidence {confidence:.2f} below threshold {min_confidence:.2f}, "
                    f"returning HOLD signal"
                )
                result['signal'] = self.SIGNAL_HOLD
                result['confidence'] = confidence
            
            logger.info(
                f"Signal generated: {result['signal']} "
                f"(confidence: {result['confidence']:.2f}, "
                f"probability up: {result['probability']:.2f})"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Signal generation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result['error'] = error_msg
            return result
    
    def predict_next_period_from_features(
        self,
        input_features: np.ndarray,
        timeframe: str,
        min_confidence: float = 0.5
    ) -> Dict:
        """
        Predict next period from pre-computed features.
        
        This is an alternative interface that accepts features directly
        instead of raw OHLCV data.
        
        Args:
            input_features: Feature vector (1D or 2D array)
            timeframe: Timeframe string
            min_confidence: Minimum confidence threshold (default: 0.5)
            
        Returns:
            Dictionary with prediction results (same format as predict_next_period)
        """
        logger.info(f"Generating signal from features for timeframe: {timeframe}")
        
        result = {
            'signal': self.SIGNAL_HOLD,
            'confidence': 0.0,
            'prediction': None,
            'probability': None,
            'timeframe': timeframe,
            'model_available': False,
            'error': None
        }
        
        try:
            # 1. Check if model exists
            if not self._check_model_exists(timeframe):
                result['error'] = f"No trained model found for timeframe: {timeframe}"
                logger.warning(result['error'])
                return result
            
            result['model_available'] = True
            
            # 2. Load model if not already loaded
            if self.model.model is None:
                logger.info(f"Loading model for timeframe: {timeframe}")
                if not self.model.load_model():
                    result['error'] = "Failed to load model"
                    logger.error(result['error'])
                    return result
            
            # 3. Ensure features are in correct shape (2D array)
            if input_features.ndim == 1:
                input_features = input_features.reshape(1, -1)
            
            # 4. Make prediction
            logger.debug("Making prediction from features...")
            prediction = self.model.predict(input_features)[0]
            probabilities = self.model.predict_proba(input_features)[0]
            
            prob_up = float(probabilities[1])
            
            result['prediction'] = int(prediction)
            result['probability'] = prob_up
            
            # 5. Convert to signal
            signal, confidence = self._model_prediction_to_signal(prediction, prob_up)
            
            result['signal'] = signal
            result['confidence'] = confidence
            
            # 6. Apply confidence threshold
            if confidence < min_confidence:
                logger.info(
                    f"Confidence {confidence:.2f} below threshold {min_confidence:.2f}, "
                    f"returning HOLD signal"
                )
                result['signal'] = self.SIGNAL_HOLD
                result['confidence'] = confidence
            
            logger.info(
                f"Signal generated: {result['signal']} "
                f"(confidence: {result['confidence']:.2f}, "
                f"probability up: {result['probability']:.2f})"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Signal generation from features failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result['error'] = error_msg
            return result
    
    def get_available_timeframes(self) -> List[str]:
        """
        Get list of timeframes with trained models available.
        
        Returns:
            List of timeframe strings that have trained models
        """
        current_file = Path(__file__).resolve()
        models_dir = current_file.parent / "models"
        
        # Find all model files
        model_files = list(models_dir.glob("logistic_regression_*.pkl"))
        
        # Extract unique timeframes
        timeframes = set()
        for model_file in model_files:
            # Parse filename: logistic_regression_{timeframe}_{timestamp}.pkl
            parts = model_file.stem.split('_')
            if len(parts) >= 3:
                timeframe = parts[2]  # timeframe is the third part
                timeframes.add(timeframe)
        
        return sorted(list(timeframes))
    
    def is_model_available(self, timeframe: str) -> bool:
        """
        Check if a model is available for the given timeframe.
        
        Args:
            timeframe: Timeframe string
            
        Returns:
            True if model exists, False otherwise
        """
        return self._check_model_exists(timeframe)


# Global signal generator instance
signal_generator = SignalGenerator()

