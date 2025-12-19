"""
AI module for machine learning integration.

This module provides the foundation for ML-based trading strategies,
including dataset management, model training, and prediction capabilities.
"""

__version__ = "0.1.0"

# Import dataset manager for easy access
from app.ai.dataset_manager import DatasetManager, dataset_manager
from app.ai.ai_config import AIConfig, ai_config
from app.ai.feature_engineering import FeatureEngineer, feature_engineer

# Lazy import of ML models to avoid breaking startup if sklearn is not installed
try:
    from app.ai.models.logistic_regression import LogisticRegressionModel, logistic_model
    from app.ai.signal_generator import SignalGenerator, signal_generator
    from app.ai.retraining_service import RetrainingService, retraining_service
    _ml_available = True
except ImportError as e:
    if 'sklearn' in str(e) or 'scikit-learn' in str(e):
        # sklearn not available - create placeholders
        _ml_available = False
        class LogisticRegressionModel:
            def __init__(self, *args, **kwargs):
                # Don't raise here - allow instantiation
                pass
            def __getattr__(self, name):
                # Raise error when any method/attribute is accessed
                raise ImportError(
                    "scikit-learn is not installed. "
                    "Please install it with: pip install scikit-learn"
                )
        # Create instance without calling __init__ that would raise
        logistic_model = object.__new__(LogisticRegressionModel)
        
        class SignalGenerator:
            def __init__(self, *args, **kwargs):
                # Don't raise here - allow instantiation
                pass
            def __getattr__(self, name):
                # Raise error when any method/attribute is accessed
                raise ImportError(
                    "scikit-learn is not installed. "
                    "Please install it with: pip install scikit-learn"
                )
        # Create instance without calling __init__ that would raise
        signal_generator = object.__new__(SignalGenerator)
        
        class RetrainingService:
            def __init__(self, *args, **kwargs):
                pass
            def __getattr__(self, name):
                raise ImportError(
                    "scikit-learn is not installed. "
                    "Please install it with: pip install scikit-learn"
                )
        retraining_service = object.__new__(RetrainingService)
    else:
        raise

__all__ = [
    'DatasetManager',
    'dataset_manager',
    'AIConfig',
    'ai_config',
    'FeatureEngineer',
    'feature_engineer',
    'LogisticRegressionModel',
    'logistic_model',
    'SignalGenerator',
    'signal_generator',
    'RetrainingService',
    'retraining_service',
]
