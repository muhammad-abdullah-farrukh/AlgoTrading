# 🚀 AlgoTradeWeb2 - Startup Guide

## Quick Start

### 1. Start Backend Server
```bash
cd AlgoTradeWeb2/backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
============================================================
Starting Trading Application Backend
============================================================
[OK] scikit-learn available (version: 1.3.2)
[OK] AI dataset pipeline verified successfully
[OK] Found 1 trained model(s)
[OK] Model loaded successfully
  Timeframe: 1d
  Accuracy: 0.7858080876692478
  Sample Size: 110414
  Last Trained: 2025-12-19T04:09:12.352795
[OK] AI model integration verified
[OK] Database connected and tables initialized
[OK] Using mock MT5 client
[OK] WebSocket manager ready (connections: 0)
[OK] Scraper service ready
[OK] Paper trading service ready
[OK] Autotrading service ready
[OK] Indicators service ready
============================================================
[OK] All components initialized successfully
Application ready on 0.0.0.0:8000
============================================================
```

### 2. Start Frontend
```bash
cd AlgoTradeWeb2
npm run dev
```

### 3. Access Application
- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

---

## 🧪 Verify AI Signals

### Option 1: Via Frontend
1. Navigate to **Trading** page
2. Look at **AI Signals** panel (right side)
3. You should see signals like:
   - AUDUSD: SELL (53.5%)
   - BRLUSD: SELL (57.1%)
   - CADUSD: SELL (54.2%)
   - ... (up to 10 pairs)

### Option 2: Via API
```bash
curl "http://localhost:8000/api/ml/signals?timeframe=1d&min_confidence=0.5"
```

**Expected Response:**
```json
{
  "signals": [
    {
      "id": 1,
      "symbol": "AUDUSD",
      "signal": "SELL",
      "confidence": 53.5,
      "reason": "Model predicts price decrease (confidence: 53.5%)",
      "price": 0.6234,
      "timeframe": "1d"
    }
  ],
  "count": 10,
  "timeframe": "1d",
  "model_available": true,
  "model_accuracy": 0.7858080876692478
}
```

### Option 3: Via Python Test
```bash
cd AlgoTradeWeb2/backend
python FINAL_VERIFICATION.py
```

---

## 🎯 Key Features Verified

### ✅ AI/ML System
- [x] Logistic Regression model trained (78.58% accuracy)
- [x] 30 technical features generated automatically
- [x] Signal generation for 22 currency pairs
- [x] BUY/SELL/HOLD signals with confidence scores
- [x] Real-time predictions using trained model

### ✅ Data Pipeline
- [x] Wide-format dataset normalization (FX data)
- [x] FIFO queue for dataset processing
- [x] Automatic feature engineering
- [x] Data validation and cleaning
- [x] Dataset tracking (Datasets → TrainedDS)

### ✅ Trading System
- [x] Paper trading with position tracking
- [x] Trade history storage in database
- [x] Auto-trading with strategy selection
- [x] MT5 integration (mock mode working)
- [x] Real-time position updates via WebSocket

### ✅ Frontend Integration
- [x] Real-time AI signals display
- [x] Model dashboard with real metrics
- [x] Trading interface with live data
- [x] Auto-trading controls
- [x] Trade history with filtering

---

## 🔧 Troubleshooting

### AI Signals Not Showing?

**Check 1: Is the backend running?**
```bash
curl http://localhost:8000/health
# Expected: {"status":"ok"}
```

**Check 2: Is the model trained?**
```bash
cd AlgoTradeWeb2/backend
python -c "from app.ai.models.logistic_regression import logistic_model; print('Model loaded:', logistic_model.load_model())"
# Expected: Model loaded: True
```

**Check 3: Test signal generation**
```bash
cd AlgoTradeWeb2/backend
python FINAL_VERIFICATION.py
# Expected: ALL TESTS PASSED!
```

**Check 4: Check browser console**
- Open DevTools (F12)
- Look for errors in Console tab
- Check Network tab for failed API calls

### WebSocket Warnings?

**Normal Behavior:**
- "going away" (1001) is normal when closing tabs
- These are now handled silently

**If you see floods of warnings:**
- Restart the backend server
- Clear browser cache
- Check for multiple browser tabs open

### Training Not Working?

**Check datasets:**
```bash
# Windows
dir "AlgoTradeWeb2\Datasets\*.csv"

# Check if datasets exist
cd AlgoTradeWeb2/backend
python -c "from app.ai.dataset_manager import dataset_manager; print('Datasets dir:', dataset_manager.datasets_dir); import os; print('Exists:', os.path.exists(dataset_manager.datasets_dir))"
```

**Train manually:**
```bash
cd AlgoTradeWeb2/backend
python train_model.py 1d --force
```

---

## 📊 Current System Status

### Trained Model
- **Type:** Logistic Regression
- **Accuracy:** 78.58%
- **Timeframe:** 1 day (1d)
- **Features:** 30 technical indicators
- **Training Samples:** 110,414
- **Currency Pairs:** 22 pairs
- **Last Trained:** 2025-12-19 04:09:12

### Available Endpoints
- ✅ Health Check: `/health`
- ✅ ML Status: `/api/ml/status`
- ✅ AI Signals: `/api/ml/signals`
- ✅ Model Weights: `/api/ml/weights`
- ✅ Retrain Model: `/api/ml/retrain`
- ✅ Trading: `/api/trade/*`
- ✅ Autotrading: `/api/autotrading/*`
- ✅ WebSockets: `/ws/*`

### Data Files
- ✅ Model: `logistic_regression_1d_20251219_040912.pkl`
- ✅ Metadata: `logistic_regression_1d_20251219_040912_metadata.json`
- ✅ Weights Export: `model_weights_1d_20251219_040912.csv`
- ✅ Metadata Export: `model_metadata_1d_20251219_040912.csv`
- ✅ Training Dataset: `20251219_040912_Foreign_Exchange_Rates.csv`

---

## 🎉 Success Criteria - ALL MET ✅

- [x] Backend starts without errors
- [x] AI model loads at startup
- [x] Signals generate successfully
- [x] Frontend displays real AI signals
- [x] No WebSocket warning floods
- [x] No Unicode encoding errors
- [x] All API endpoints working
- [x] Database connections stable
- [x] No linting errors
- [x] Clean, production-ready code

---

## 📝 Next Steps

### To Add More Timeframes:
```bash
# Train for 1 hour timeframe
python train_model.py 1h --force

# Train for 4 hour timeframe
python train_model.py 4h --force
```

### To Retrain Existing Model:
```bash
python train_model.py 1d --force
```

### To Export Model Data:
```python
from app.ai.model_export import model_export_service
result = model_export_service.export_all()
print(f"Exported to: {result['weights_path']}")
```

---

## ✨ What's Working

**Your AlgoTradeWeb2 application is now fully functional with:**

1. **Real AI Predictions** - 78.58% accuracy logistic regression model
2. **Live Trading Signals** - BUY/SELL/HOLD for 22 currency pairs
3. **Automatic Updates** - Signals refresh every 30 seconds
4. **Clean Integration** - Frontend ↔ Backend ↔ AI seamless
5. **Production Ready** - Proper error handling, logging, validation

**Navigate to the Trading page and see your AI signals in action! 🎯**


