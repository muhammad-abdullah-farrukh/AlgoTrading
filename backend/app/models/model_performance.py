"""
Model Performance Monitoring Database Model

Tracks model performance metrics over time for auditability and trustworthiness.
"""
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, JSON
from sqlalchemy.sql import func
from app.models import Base


class ModelPerformance(Base):
    """
    Database model for tracking ML model performance metrics over time.
    
    Tracks:
    - Accuracy over time
    - Sample size used
    - Dataset sources
    - Training timestamps
    - Model version hash
    - Loss / error rate
    """
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Model identification
    model_type = Column(String(100), nullable=False, index=True)  # e.g., 'logistic_regression'
    model_version_hash = Column(String(64), nullable=False, index=True)  # SHA-256 hash of model file
    timeframe = Column(String(20), nullable=False, index=True)  # e.g., '1d', '1h'
    
    # Performance metrics
    accuracy = Column(Float, nullable=False)  # Model accuracy (0.0-1.0)
    loss = Column(Float, nullable=True)  # Loss value (if available)
    error_rate = Column(Float, nullable=True)  # Error rate (1 - accuracy)
    
    # Training data info
    sample_size = Column(Integer, nullable=False)  # Total samples used
    train_size = Column(Integer, nullable=False)  # Training set size
    test_size = Column(Integer, nullable=False)  # Test set size
    feature_count = Column(Integer, nullable=False)  # Number of features
    
    # Dataset sources (JSON array of dataset filenames)
    dataset_sources = Column(JSON, nullable=False)  # List of dataset filenames used
    
    # Timestamps
    trained_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    recorded_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    
    # Model file paths
    model_path = Column(String(500), nullable=False)  # Path to model file
    metadata_path = Column(String(500), nullable=True)  # Path to metadata file
    
    # Additional metadata (JSON)
    additional_metadata = Column(JSON, nullable=True)  # Extra metrics, classification report, etc.
    
    def to_dict(self) -> dict:
        """Convert model performance record to dictionary."""
        return {
            'id': self.id,
            'model_type': self.model_type,
            'model_version_hash': self.model_version_hash,
            'timeframe': self.timeframe,
            'accuracy': self.accuracy,
            'loss': self.loss,
            'error_rate': self.error_rate,
            'sample_size': self.sample_size,
            'train_size': self.train_size,
            'test_size': self.test_size,
            'feature_count': self.feature_count,
            'dataset_sources': self.dataset_sources,
            'trained_at': self.trained_at.isoformat() if self.trained_at else None,
            'recorded_at': self.recorded_at.isoformat() if self.recorded_at else None,
            'model_path': self.model_path,
            'metadata_path': self.metadata_path,
            'additional_metadata': self.additional_metadata
        }

