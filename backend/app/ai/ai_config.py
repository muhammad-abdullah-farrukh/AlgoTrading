"""
AI Configuration Module

Central configuration for AI/ML training and inference settings.
All AI-related parameters are defined here.
"""
from typing import List
from pydantic import BaseModel, Field


class AIConfig(BaseModel):
    """
    AI configuration settings.
    
    Controls when and how models are trained, retrained, and used for predictions.
    """
    
    # Retraining settings
    retrain_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Performance threshold below which model should be retrained (0.0-1.0)"
    )
    
    auto_retrain_enabled: bool = Field(
        default=False,
        description="Enable automatic retraining when performance drops below threshold"
    )
    
    # Learning settings
    online_learning_enabled: bool = Field(
        default=False,
        description="Enable online/incremental learning (updates model with new data without full retrain)"
    )
    
    # Timeframe support (1m to 12w)
    supported_timeframes: List[str] = Field(
        default_factory=lambda: ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "2w", "4w", "8w", "12w"],
        description="List of supported timeframes for model training and prediction (1m to 12w)"
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "retrain_threshold": 0.6,
                "auto_retrain_enabled": False,
                "online_learning_enabled": False,
                "supported_timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "2w", "4w", "8w", "12w"]
            }
        }


# Global AI configuration instance
# This can be loaded from environment variables or database in the future
ai_config = AIConfig()

