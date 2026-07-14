import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import lightgbm as lgb

# Add root folder to sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "src"))

from src.feature_engine import FeatureEngine

MODEL_PATH = root_dir / "predictions" / "lightgbm_model.pkl"

def main():
    # Load 2-digit loto data
    csv_path = root_dir / 'data' / 'xsmb-2-digits.csv'
    if not csv_path.exists():
        print(f"❌ Error: data file not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    total_days = len(df)
    print(f"Loaded {total_days} days of data.")
    
    # We start training dataset from day 100 to allow sufficient history for feature calculations
    start_idx = 100
    end_idx = total_days
    
    engine = FeatureEngine()
    print("⏳ Building feature dataset...")
    X, y = engine.build_dataset_range(df, start_idx, end_idx)
    
    feature_cols = [c for c in X.columns if c not in ['target_num', 'date']]
    print(f"Feature matrix shape: {X[feature_cols].shape}")
    print(f"Positive samples rate: {y.mean() * 100:.2f}%")
    
    print("🤖 Training LightGBM model...")
    clf = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=100,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=30,
        random_state=42,
        verbose=-1
    )
    clf.fit(X[feature_cols], y)
    
    # Save the model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
        
    print(f"✅ Successfully trained and saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
