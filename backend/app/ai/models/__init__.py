"""
AI models module.

This module contains trained ML models for trading predictions.
Models are stored here after training and can be loaded for inference.
"""
# Lazy import to avoid breaking startup if sklearn is not installed
# Models will only be imported when explicitly needed
try:
    from app.ai.models.logistic_regression import LogisticRegressionModel, logistic_model
    __all__ = ['LogisticRegressionModel', 'logistic_model']
except ImportError as e:
    # sklearn not available - models cannot be used but won't break imports
    import sys
    if 'sklearn' in str(e) or 'scikit-learn' in str(e):
        # Create placeholder classes to allow imports to succeed
        # These will raise ImportError when methods are called, not on instantiation
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
        __all__ = ['LogisticRegressionModel', 'logistic_model']
    else:
        raise

