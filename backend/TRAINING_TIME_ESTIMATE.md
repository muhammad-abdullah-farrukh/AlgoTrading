# Model Training Time Estimates

## Training Time Breakdown

### Factors Affecting Training Time

1. **Dataset Size** (number of rows)
2. **Number of Features** (generated during feature engineering)
3. **Model Complexity** (Logistic Regression with LBFGS solver)
4. **Hardware** (CPU speed, RAM)

### Time Estimates by Dataset Size

#### Small Dataset (< 1,000 samples)
- **Dataset Loading**: 1-2 seconds
- **Feature Engineering**: 2-5 seconds
- **Model Training**: 1-3 seconds
- **Evaluation & Saving**: 1 second
- **Total**: **5-10 seconds**

#### Medium Dataset (1,000 - 10,000 samples)
- **Dataset Loading**: 2-5 seconds
- **Feature Engineering**: 5-15 seconds
- **Model Training**: 3-10 seconds
- **Evaluation & Saving**: 1-2 seconds
- **Total**: **15-35 seconds**

#### Large Dataset (10,000 - 50,000 samples)
- **Dataset Loading**: 5-10 seconds
- **Feature Engineering**: 15-45 seconds
- **Model Training**: 10-30 seconds
- **Evaluation & Saving**: 2-3 seconds
- **Total**: **30 seconds - 1.5 minutes**

#### Very Large Dataset (50,000 - 100,000 samples)
- **Dataset Loading**: 10-20 seconds
- **Feature Engineering**: 45-90 seconds
- **Model Training**: 30-60 seconds
- **Evaluation & Saving**: 3-5 seconds
- **Total**: **1.5 - 3 minutes**

### Feature Engineering Time

The feature engineering step generates:
- **5 lagged returns** (return_lag_1, return_lag_2, etc.)
- **20 moving average features** (5 windows × 4 features each: SMA, EMA, price_to_sma, price_to_ema)
- **Price difference features** (high-low spread, close-open diff, volatility)
- **Total**: ~30-40 features

**Time per 1,000 rows**: ~1-2 seconds

### Model Training Time

Logistic Regression with:
- **Solver**: LBFGS (efficient for small-medium datasets)
- **Max Iterations**: 1000 (default)
- **Class Weight**: Balanced (handles class imbalance)

**Time per 1,000 samples**: ~0.5-1 second

### Typical Scenarios

#### Scenario 1: Single Small Dataset (500 rows)
- **Estimated Time**: **5-8 seconds**

#### Scenario 2: Single Medium Dataset (5,000 rows)
- **Estimated Time**: **20-30 seconds**

#### Scenario 3: Multiple Datasets (3 datasets, ~10,000 total rows)
- **Estimated Time**: **30-45 seconds**

#### Scenario 4: Large Dataset (50,000 rows)
- **Estimated Time**: **1-2 minutes**

### Bottlenecks

1. **Feature Engineering** (especially moving averages with large windows)
2. **Dataset Loading** (if files are very large or many files)
3. **Model Training** (if dataset is very large)

### Optimization Tips

1. **Reduce max_iter**: If training is slow, reduce `max_iter` from 1000 to 500
2. **Limit dataset size**: Use MAX_ROWS = 10,000 limit (already implemented)
3. **Reduce feature count**: Fewer moving average windows = faster feature engineering
4. **Use smaller datasets**: Train on recent data only

### Real-World Example

For a typical dataset with **10,000 rows**:
- Loading: 3 seconds
- Feature Engineering: 12 seconds
- Training: 8 seconds
- Evaluation: 1 second
- Saving: 1 second
- **Total: ~25 seconds**


