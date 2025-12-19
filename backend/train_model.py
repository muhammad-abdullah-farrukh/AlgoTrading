"""
Train Logistic Regression Model

Simple script to train the logistic regression model using datasets
from the Datasets folder.

Usage:
    python train_model.py [timeframe]

Examples:
    python train_model.py 1d
    python train_model.py 1h
    python train_model.py 1w
"""
import sys
import argparse
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.ai.models.logistic_regression import logistic_model
from app.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Train the logistic regression model."""
    parser = argparse.ArgumentParser(
        description="Train logistic regression model using datasets from Datasets folder"
    )
    parser.add_argument(
        "timeframe",
        nargs="?",
        default="1d",
        help="Timeframe for training (e.g., '1h', '1d', '1w'). Default: '1d'"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing (default: 0.2)"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for logistic regression (default: 1000)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("STARTING MODEL TRAINING")
    print("=" * 60)
    print(f"Timeframe: {args.timeframe}")
    print(f"Test Size: {args.test_size}")
    print(f"Max Iterations: {args.max_iter}")
    print("=" * 60 + "\n")
    
    try:
        # Train the model
        result = logistic_model.train(
            timeframe=args.timeframe,
            test_size=args.test_size,
            max_iter=args.max_iter
        )
        
        if result.get('success'):
            print("\n" + "=" * 60)
            print("TRAINING SUCCESSFUL!")
            print("=" * 60)
            print(f"Accuracy: {result.get('accuracy'):.4f} ({result.get('accuracy')*100:.2f}%)")
            print(f"Train Size: {result.get('train_size'):,} samples")
            print(f"Test Size: {result.get('test_size'):,} samples")
            print(f"Total Samples: {result.get('sample_size'):,}")
            print(f"Model Path: {result.get('model_path')}")
            print(f"Metadata Path: {result.get('metadata_path')}")
            print(f"Datasets Used: {len(result.get('datasets_used', []))} file(s)")
            for ds in result.get('datasets_used', []):
                print(f"   - {ds}")
            print("=" * 60 + "\n")
            return 0
        else:
            print("\n" + "=" * 60)
            print("TRAINING FAILED")
            print("=" * 60)
            print(f"Error: {result.get('error', 'Unknown error')}")
            print("=" * 60 + "\n")
            return 1
            
    except Exception as e:
        print("\n" + "=" * 60)
        print("TRAINING FAILED")
        print("=" * 60)
        print(f"Error: {str(e)}")
        print("=" * 60 + "\n")
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

