# AI Module Documentation

## Overview

This module provides the foundation for machine learning-based trading strategies. It manages datasets, model training, and prediction capabilities in a structured, maintainable way.

## Directory Structure

```
ai/
├── __init__.py              # Module initialization
├── ai_config.py             # AI configuration settings
├── dataset_manager.py       # Dataset loading and FIFO queue management
├── README.md                 # This file
├── models/                   # Trained ML models
│   └── __init__.py
└── data/                     # Dataset management
    ├── __init__.py
    ├── Datasets/             # Raw datasets ready for training
    └── TrainedDS/            # Datasets that have been used for training
```

## Dataset Flow

### Datasets → TrainedDS Pipeline

1. **Datasets/** (Input)
   - Raw datasets are placed here from the main database/scraper
   - Datasets are prepared and validated before training
   - Each dataset should contain OHLCV data with proper formatting

2. **Training Process**
   - Training is **ONLY** triggered by the `retrain` flag
   - When `retrain=True`, the system:
     - Loads datasets from `Datasets/`
     - Trains/retrains the model
     - Saves the trained model to `models/`
     - Moves used datasets to `TrainedDS/`

3. **TrainedDS/** (Archive)
   - Datasets that have been used for training are moved here
   - Maintains a history of training data
   - Used for model versioning and audit trails

## FIFO Dataset Queue Logic

The system uses a **First-In-First-Out (FIFO)** approach for dataset management:

1. **New datasets** are added to `Datasets/` directory
2. **Oldest datasets** are processed first when training is triggered
3. **After training**, datasets are moved to `TrainedDS/` in chronological order
4. **Storage limits** can be enforced (oldest datasets in `TrainedDS/` can be archived/deleted)

### Example Flow

```
Time T0: Dataset1.csv → Datasets/
Time T1: Dataset2.csv → Datasets/
Time T2: Dataset3.csv → Datasets/

[Retrain flag triggered]

Training Process:
  1. Load Dataset1.csv (oldest)
  2. Load Dataset2.csv
  3. Load Dataset3.csv (newest)
  4. Train model with all datasets
  5. Save model to models/
  6. Move Dataset1.csv → TrainedDS/
  7. Move Dataset2.csv → TrainedDS/
  8. Move Dataset3.csv → TrainedDS/

Result:
  - Datasets/ is empty
  - TrainedDS/ contains Dataset1.csv, Dataset2.csv, Dataset3.csv
  - models/ contains the newly trained model
```

## Training Trigger

### Important: Training is ONLY triggered by retrain flag

- **Manual retrain**: Set `retrain=True` explicitly (via API or admin interface)
- **Auto retrain**: If `auto_retrain_enabled=True` and performance drops below `retrain_threshold`
- **No automatic training**: Models are NOT trained automatically on new data
- **Explicit control**: Training must be explicitly requested

### Retrain Conditions

1. **Manual Retrain**
   - User/admin explicitly sets retrain flag
   - Used for initial training or forced retraining

2. **Automatic Retrain** (if enabled)
   - Model performance drops below `retrain_threshold`
   - Requires `auto_retrain_enabled=True` in config
   - Performance is measured against validation metrics

3. **Online Learning** (if enabled)
   - Updates model incrementally with new data
   - Does NOT trigger full retrain
   - Requires `online_learning_enabled=True` in config

## Configuration

All AI settings are defined in `ai_config.py`:

- `retrain_threshold`: Performance threshold for auto-retrain (0.0-1.0)
- `auto_retrain_enabled`: Enable automatic retraining
- `online_learning_enabled`: Enable incremental learning
- `supported_timeframes`: List of timeframes for model training

## Future Enhancements

- Model versioning and rollback
- Dataset validation and quality checks
- Distributed training support
- Model performance monitoring
- A/B testing framework for models

## Dataset Manager

The `DatasetManager` class provides dataset loading and management functionality:

### Key Functions

- `load_next_dataset()`: Loads the oldest dataset from the FIFO queue
- `mark_dataset_as_trained(filename)`: Moves a dataset to TrainedDS/ after training
- `append_dataset(df, filename)`: Adds a new dataset to the queue
- `get_dataset_count()`: Returns the number of available datasets
- `list_datasets()`: Lists all datasets with metadata

### Usage Example

```python
from app.ai import dataset_manager

# Load next dataset (FIFO - oldest first)
result = dataset_manager.load_next_dataset()
if result:
    df, filename = result
    # Use df for training
    # ...
    
    # Mark as trained (moves to TrainedDS/)
    dataset_manager.mark_dataset_as_trained(filename)

# Append new dataset
new_df = pd.DataFrame({...})  # Must have date, currency_pair, close_price
dataset_manager.append_dataset(new_df, "my_dataset.csv")
```

### Column Validation

Required columns (with aliases):
- **date**: `date`, `timestamp`, `time`, `datetime`
- **currency_pair**: `symbol`, `currency_pair`, `pair`, `instrument`
- **close_price**: `close`, `close_price`, `price`, `closing_price`

Optional columns:
- `open`, `high`, `low`, `volume` (forward-filled if missing)

### Missing Value Handling

- Rows with missing required columns are dropped
- Optional columns are forward-filled then back-filled
- Warnings are logged for dropped rows

## Notes

- Phase ML-0: Foundation setup (complete)
- Phase ML-1: Dataset loader & FIFO queue (complete)
- No models are trained yet
- No API routes are added yet
- Frontend integration comes in later phases

