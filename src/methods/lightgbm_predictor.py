import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import lightgbm as lgb
from . import BasePredictor
from feature_engine import FeatureEngine

class LightGBMPredictor(BasePredictor):
    def __init__(self, model_path: str = None, train_window: int = 180):
        super().__init__("LightGBM Meta Predictor")
        self.train_window = train_window
        self.model_path = model_path
        self.model = None
        self.feature_engine = FeatureEngine()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            # print(f"✅ Loaded LightGBM model from {model_path}")
        except Exception as e:
            print(f"⚠️ Error loading LightGBM model from {model_path}: {e}")

    def save_model(self, model_path: str) -> None:
        if self.model is not None:
            try:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                with open(model_path, "wb") as f:
                    pickle.dump(self.model, f)
                # print(f"✅ Saved LightGBM model to {model_path}")
            except Exception as e:
                print(f"⚠️ Error saving LightGBM model: {e}")

    def train_on_history(self, history_df: pd.DataFrame, S: np.ndarray = None) -> None:
        """
        Train LightGBM on the recent history_df.
        We use train_window days of history to create training samples.
        """
        N = len(history_df)
        if N < 50:
            # Not enough data to train
            return

        # Determine train range
        start_idx = max(30, N - self.train_window)
        end_idx = N

        # Build feature matrix and targets
        X_train, y_train = self.feature_engine.build_dataset_range(history_df, start_idx, end_idx)
        
        # Drop columns not used for training
        feature_cols = [c for c in X_train.columns if c not in ['target_num', 'date']]
        
        # Train LightGBM classifier
        clf = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=60,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=30,
            random_state=42,
            verbose=-1
        )
        clf.fit(X_train[feature_cols], y_train)
        self.model = clf

    def predict_proba(self, history_df: pd.DataFrame, S: np.ndarray = None) -> np.ndarray:
        """
        Predict probabilities of appearance for all 100 numbers.
        Returns a numpy array of shape (100,) with probability values.
        """
        N = len(history_df)
        if N == 0:
            return np.ones(100) * 0.27

        # 1. Train model if not loaded
        if self.model is None:
            # print("🤖 Model not found. Training on the fly...")
            self.train_on_history(history_df, S)

        if self.model is None:
            # Fallback if training failed
            return np.ones(100) * 0.27

        # 2. Build features for the next day (prediction target)
        # S matrix for history
        if S is None:
            prize_cols = [c for c in history_df.columns if c != 'date']
            arr = history_df[prize_cols].values.astype(int)
            S = np.zeros((N, 100), dtype=np.int8)
            rows = np.repeat(np.arange(N), arr.shape[1])
            cols = arr.flatten()
            valid = (cols >= 0) & (cols < 100)
            S[rows[valid], cols[valid]] = 1

        # We assume the next date is 1 day after the last date in history
        last_row = history_df.iloc[-1]
        last_date = pd.to_datetime(last_row['date'])
        target_date = last_date + pd.Timedelta(days=1)

        # Build feature DataFrame for target date
        df_feat = self.feature_engine.build_features_for_day(history_df, S, target_date)
        feature_cols = [c for c in df_feat.columns if c not in ['target_num', 'date']]

        # 3. Predict probabilities using trained model
        probs = self.model.predict_proba(df_feat[feature_cols])[:, 1]
        return probs

    def predict(self, history_df: pd.DataFrame, top_k: int = 5, S: np.ndarray = None) -> list[int]:
        probs = self.predict_proba(history_df, S)
        # Return top_k numbers with highest probabilities
        sorted_nums = np.argsort(probs)[::-1]
        return [int(x) for x in sorted_nums[:top_k]]
