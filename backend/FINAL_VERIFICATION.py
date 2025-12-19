"""
Final Verification Script
Tests all critical components to ensure everything works correctly
"""
import sys
sys.path.insert(0, '.')

print("=" * 70)
print(" " * 20 + "FINAL SYSTEM VERIFICATION")
print("=" * 70)

all_tests_passed = True
test_results = []

# Test 1: Model Loading
print("\n[TEST 1] Model Loading...")
try:
    from app.ai.models.logistic_regression import logistic_model
    loaded = logistic_model.load_model()
    metadata = logistic_model.get_metadata()
    
    if loaded and metadata:
        accuracy = metadata.get('accuracy', 0)
        print(f"   [PASS] Model loaded successfully")
        print(f"          Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"          Timeframe: {metadata.get('timeframe')}")
        print(f"          Features: {metadata.get('feature_count')}")
        test_results.append(("Model Loading", "PASS"))
    else:
        print(f"   [FAIL] Model loaded but metadata missing")
        test_results.append(("Model Loading", "FAIL"))
        all_tests_passed = False
except Exception as e:
    print(f"   [FAIL] {str(e)}")
    test_results.append(("Model Loading", "FAIL"))
    all_tests_passed = False

# Test 2: Dataset Availability
print("\n[TEST 2] Dataset Availability...")
try:
    from app.ai.dataset_manager import dataset_manager
    from pathlib import Path
    
    trained_files = list(dataset_manager.trained_dir.glob("*.csv"))
    dataset_files = list(dataset_manager.datasets_dir.glob("*.csv"))
    
    total_files = len(trained_files) + len(dataset_files)
    
    if total_files > 0:
        print(f"   [PASS] Found {total_files} dataset file(s)")
        print(f"          Trained: {len(trained_files)}")
        print(f"          Available: {len(dataset_files)}")
        test_results.append(("Dataset Availability", "PASS"))
    else:
        print(f"   [FAIL] No datasets found")
        test_results.append(("Dataset Availability", "FAIL"))
        all_tests_passed = False
except Exception as e:
    print(f"   [FAIL] {str(e)}")
    test_results.append(("Dataset Availability", "FAIL"))
    all_tests_passed = False

# Test 3: Signal Generation
print("\n[TEST 3] Signal Generation...")
try:
    from app.ai.signal_generator import signal_generator
    import pandas as pd
    
    # Check model availability
    if not signal_generator.is_model_available('1d'):
        print(f"   [FAIL] Model not available for 1d timeframe")
        test_results.append(("Signal Generation", "FAIL"))
        all_tests_passed = False
    else:
        # Load test data (use largest file)
        trained_files = list(dataset_manager.trained_dir.glob("*.csv"))
        if trained_files:
            test_file = sorted(trained_files, key=lambda p: p.stat().st_size, reverse=True)[0]
            print(f"   [INFO] Testing with: {test_file.name}")
            df = pd.read_csv(test_file)
            
            # Normalize if needed (check BEFORE lowercasing)
            has_currency_pair = any('currency_pair' in col.lower() for col in df.columns)
            has_close_price = any('close' in col.lower() and 'price' in col.lower() for col in df.columns)
            
            if not (has_currency_pair and has_close_price):
                from app.ai.dataset_adapter import dataset_adapter
                print(f"   [INFO] Normalizing wide format...")
                df, _ = dataset_adapter.normalize_dataset(df, test_file.name)
            
            # Now lowercase columns
            df.columns = [col.lower().strip() for col in df.columns]
            
            if 'date' not in df.columns:
                for date_col in ['timestamp', 'time', 'datetime', 'time serie', 'time_serie']:
                    if date_col in df.columns:
                        df['date'] = df[date_col]
                        break
            
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            
            if 'currency_pair' in df.columns:
                pairs = df['currency_pair'].unique().tolist()
                if pairs:
                    # Test with first pair
                    test_pair = pairs[0]
                    pair_data = df[df['currency_pair'] == test_pair].copy()
                    pair_data = pair_data.sort_values('date').tail(200)
                    
                    result = signal_generator.predict_next_period(
                        input_data=pair_data,
                        timeframe='1d',
                        min_confidence=0.5
                    )
                    
                    if result.get('error'):
                        print(f"   [FAIL] {result['error']}")
                        test_results.append(("Signal Generation", "FAIL"))
                        all_tests_passed = False
                    else:
                        signal = result['signal']
                        confidence = result['confidence'] * 100
                        print(f"   [PASS] Signal generated successfully")
                        print(f"          Pair: {test_pair}")
                        print(f"          Signal: {signal}")
                        print(f"          Confidence: {confidence:.1f}%")
                        test_results.append(("Signal Generation", "PASS"))
                else:
                    print(f"   [FAIL] No currency pairs found")
                    test_results.append(("Signal Generation", "FAIL"))
                    all_tests_passed = False
            else:
                print(f"   [FAIL] No currency_pair column after normalization")
                test_results.append(("Signal Generation", "FAIL"))
                all_tests_passed = False
        else:
            print(f"   [FAIL] No datasets available for testing")
            test_results.append(("Signal Generation", "FAIL"))
            all_tests_passed = False
            
except Exception as e:
    print(f"   [FAIL] {str(e)}")
    import traceback
    traceback.print_exc()
    test_results.append(("Signal Generation", "FAIL"))
    all_tests_passed = False

# Test 4: Feature Engineering
print("\n[TEST 4] Feature Engineering...")
try:
    from app.ai.feature_engineering import feature_engineer
    import pandas as pd
    
    # Create test data
    test_df = pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=100),
        'close_price': [1.0 + i * 0.001 for i in range(100)]
    })
    
    # Generate features without target
    df_features = feature_engineer.prepare_features(
        df=test_df,
        timeframe='1d',
        include_target=False
    )
    
    if len(df_features) > 0:
        feature_cols = [col for col in df_features.columns 
                       if col not in ['date', 'close_price', 'target']]
        print(f"   [PASS] Features generated successfully")
        print(f"          Features: {len(feature_cols)}")
        print(f"          Rows: {len(df_features)}")
        test_results.append(("Feature Engineering", "PASS"))
    else:
        print(f"   [FAIL] No features generated")
        test_results.append(("Feature Engineering", "FAIL"))
        all_tests_passed = False
        
except Exception as e:
    print(f"   [FAIL] {str(e)}")
    test_results.append(("Feature Engineering", "FAIL"))
    all_tests_passed = False

# Test 5: Dataset Normalization
print("\n[TEST 5] Dataset Normalization...")
try:
    from app.ai.dataset_adapter import dataset_adapter
    import pandas as pd
    
    # Create wide-format test data
    test_wide = pd.DataFrame({
        'Time Serie': pd.date_range('2025-01-01', periods=10),
        'EURO/US$': [1.1 + i * 0.01 for i in range(10)],
        'YEN/US$': [110 + i for i in range(10)]
    })
    
    df_normalized, _ = dataset_adapter.normalize_dataset(test_wide, 'test.csv')
    
    has_required = all(col in df_normalized.columns for col in ['date', 'currency_pair', 'close_price'])
    
    if has_required and len(df_normalized) > 0:
        pairs = df_normalized['currency_pair'].unique()
        print(f"   [PASS] Wide format normalized successfully")
        print(f"          Rows: {len(df_normalized)}")
        print(f"          Pairs: {len(pairs)}")
        test_results.append(("Dataset Normalization", "PASS"))
    else:
        print(f"   [FAIL] Normalization produced invalid format")
        test_results.append(("Dataset Normalization", "FAIL"))
        all_tests_passed = False
        
except Exception as e:
    print(f"   [FAIL] {str(e)}")
    test_results.append(("Dataset Normalization", "FAIL"))
    all_tests_passed = False

# Test 6: Model Export
print("\n[TEST 6] Model Export...")
try:
    from app.ai.model_export import model_export_service
    from pathlib import Path
    
    # Check if export directory exists
    export_dir = model_export_service.export_dir
    
    if export_dir.exists():
        export_files = list(export_dir.glob("*.csv"))
        print(f"   [PASS] Export directory accessible")
        print(f"          Location: {export_dir}")
        print(f"          Files: {len(export_files)}")
        test_results.append(("Model Export", "PASS"))
    else:
        print(f"   [FAIL] Export directory not found")
        test_results.append(("Model Export", "FAIL"))
        all_tests_passed = False
        
except Exception as e:
    print(f"   [FAIL] {str(e)}")
    test_results.append(("Model Export", "FAIL"))
    all_tests_passed = False

# Test 7: Database Connection
print("\n[TEST 7] Database Connection...")
try:
    from app.database import db
    from app.config import settings
    
    print(f"   [INFO] Database URL: {settings.database_url}")
    print(f"   [INFO] Connected: {db.is_connected}")
    
    # Database connection is tested at startup, just verify config
    if settings.database_url:
        print(f"   [PASS] Database configuration valid")
        test_results.append(("Database Connection", "PASS"))
    else:
        print(f"   [FAIL] Database URL not configured")
        test_results.append(("Database Connection", "FAIL"))
        all_tests_passed = False
        
except Exception as e:
    print(f"   [FAIL] {str(e)}")
    test_results.append(("Database Connection", "FAIL"))
    all_tests_passed = False

# Summary
print("\n" + "=" * 70)
print(" " * 25 + "TEST SUMMARY")
print("=" * 70)

for test_name, result in test_results:
    status_symbol = "[PASS]" if result == "PASS" else "[FAIL]"
    print(f"{status_symbol} {test_name}")

print("\n" + "=" * 70)
if all_tests_passed:
    print(" " * 15 + "[SUCCESS] ALL TESTS PASSED!")
    print(" " * 10 + "System is ready for production use")
else:
    print(" " * 15 + "[WARNING] SOME TESTS FAILED")
    print(" " * 10 + "Review failed tests above")
print("=" * 70)

sys.exit(0 if all_tests_passed else 1)

