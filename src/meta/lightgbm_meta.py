"""
META LEARNING LAYER — src/meta/lightgbm_meta.py
LightGBM Meta Learner — "Bộ não" của XPIS.

Input:  78 FeatureStore features + 10 model probabilities = 88 features
Output: P(number xuất hiện) đã học từ lịch sử walk-forward

Nguyên tắc:
- Là nơi DUY NHẤT kết hợp các model thành phần.
- Không tự tính feature, không đọc raw data.
- Walk-forward: retrain mỗi N ngày để tránh data leakage.
- Versioned model files để đảm bảo reproducibility.
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _lgb_available = True
except ImportError:
    _lgb_available = False
    print("[LightGBMMetaLearner] Warning: lightgbm not installed. pip install lightgbm")

from .base import BaseMetaLearner
from .calibration import ProbabilityCalibrator

MODEL_VERSION = "v1.0"
MODELS_DIR = Path("predictions/models")


class LightGBMMetaLearner(BaseMetaLearner):
    """
    Meta Learner dùng LightGBM để học cách kết hợp 88 features → P(number).

    Walk-forward protocol:
    - Train trên [t-train_window, t-1]
    - Predict ngày t
    - Retrain mỗi retrain_every ngày

    Versioning:
    - Model được lưu tại: predictions/models/lgbm_{version}_{date}.pkl
    - metadata.json ghi lại: train_date, feature_count, n_samples, hyperparams
    """

    def __init__(
        self,
        train_window: int = 365,       # Số ngày dùng để train (rolling)
        retrain_every: int = 30,       # Retrain mỗi N ngày
        calibrate: bool = True,        # Có dùng Isotonic calibration không
        model_path: Optional[str] = None,
    ):
        if not _lgb_available:
            raise ImportError("lightgbm không được cài. Chạy: pip install lightgbm")

        self._train_window = train_window
        self._retrain_every = retrain_every
        self._calibrate = calibrate
        self._model: Optional[lgb.LGBMClassifier] = None
        self._calibrator: Optional[ProbabilityCalibrator] = None
        self._feature_cols: list[str] = []
        self._is_trained = False
        self._train_date: Optional[str] = None

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if model_path and Path(model_path).exists():
            self.load(model_path)

    @property
    def name(self) -> str:
        return "lightgbm_meta"

    @property
    def version(self) -> str:
        return MODEL_VERSION

    def is_trained(self) -> bool:
        return self._is_trained and self._model is not None

    # ---------------------------------------------------------------- Training

    def build_training_data(
        self,
        feature_snapshots: list[pd.DataFrame],
        labels: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Xây dựng training matrix từ danh sách FeatureStore snapshots.

        Args:
            feature_snapshots: List của DataFrame (100 rows × N_features)
                               Mỗi phần tử ứng với một ngày
            labels:            List của np.ndarray (100,) — 1 nếu số ra, 0 nếu không

        Returns:
            X: (n_days × 100, N_features)
            y: (n_days × 100,)
        """
        X_parts, y_parts = [], []
        for df_feat, y in zip(feature_snapshots, labels):
            meta_cols = [c for c in df_feat.columns if c not in ("number", "date")]
            if not self._feature_cols:
                self._feature_cols = meta_cols
            X_parts.append(df_feat[meta_cols].values.astype(np.float32))
            y_parts.append(y.astype(np.float32))

        X = np.vstack(X_parts)   # (n_days * 100, n_features)
        y = np.concatenate(y_parts)   # (n_days * 100,)
        return X, y

    def train(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """
        Huấn luyện LightGBM và tự động hiệu chuẩn (Calibration) qua cross-validation.
        """
        from sklearn.calibration import CalibratedClassifierCV

        if feature_names:
            self._feature_cols = feature_names

        # Convert to DataFrame if we have feature names
        if isinstance(X, np.ndarray) and self._feature_cols:
            X = pd.DataFrame(X, columns=self._feature_cols)

        base_clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=80,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )

        if self._calibrate:
            # Sử dụng 5-fold cross-validation để thu thập dự báo out-of-sample và hiệu chuẩn isotonic
            clf = CalibratedClassifierCV(estimator=base_clf, method="isotonic", cv=5)
        else:
            clf = base_clf

        clf.fit(X, y)
        self._model = clf
        self._is_trained = True
        self._train_date = datetime.now().strftime("%Y-%m-%d")

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Dự báo xác suất đã được hiệu chuẩn cho mỗi sample.
        """
        if not self.is_trained():
            return np.full(X.shape[0] if hasattr(X, 'shape') else 100, 0.27)

        # Convert numpy array to DataFrame
        if isinstance(X, np.ndarray) and self._feature_cols:
            X = pd.DataFrame(X, columns=self._feature_cols)

        # Handle NaN/Inf
        if isinstance(X, pd.DataFrame):
            X = X.fillna(0.0)
        else:
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

        # CalibratedClassifierCV.predict_proba trả ra matrix shape (n_samples, 2)
        # Cột index 1 là xác suất lớp positive (1)
        proba = self._model.predict_proba(X)[:, 1]
        return proba

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Trả về top-N feature importances."""
        if not self.is_trained():
            return pd.DataFrame()
        
        # Nếu model được bọc bởi CalibratedClassifierCV, chúng ta lấy trung bình từ các classifier thành phần của mỗi fold
        if hasattr(self._model, 'calibrated_classifiers_'):
            importances = []
            for cc in self._model.calibrated_classifiers_:
                # cc.base_estimator là model cơ sở đã fit cho fold đó
                base_est = cc.base_estimator
                importances.append(base_est.feature_importances_)
            imp = np.mean(importances, axis=0)
        else:
            imp = self._model.feature_importances_
            
        names = self._feature_cols if self._feature_cols else [f"f{i}" for i in range(len(imp))]
        df = pd.DataFrame({"feature": names, "importance": imp})
        return df.sort_values("importance", ascending=False).head(top_n)

    # ---------------------------------------------------------------- Save / Load

    def save(self, path: Optional[str] = None) -> Path:
        """Lưu model + calibrator + metadata."""
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(MODELS_DIR / f"lgbm_{MODEL_VERSION}_{ts}.pkl")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self._model,
            "calibrator": self._calibrator,
            "feature_cols": self._feature_cols,
            "train_date": self._train_date,
            "version": MODEL_VERSION,
        }
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)

        # Save human-readable metadata
        meta_path = save_path.with_suffix(".json")
        meta = {
            "version": MODEL_VERSION,
            "train_date": self._train_date,
            "n_features": len(self._feature_cols),
            "feature_cols": self._feature_cols[:10],  # first 10 for preview
            "calibrated": self._calibrate,
            "hyperparams": {
                "n_estimators": 100,
                "num_leaves": 20,
                "learning_rate": 0.05,
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return save_path

    def load(self, path: str) -> None:
        """Tải model từ file pickle."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._model = payload["model"]
        self._calibrator = payload.get("calibrator")
        self._feature_cols = payload.get("feature_cols", [])
        self._train_date = payload.get("train_date")
        self._is_trained = self._model is not None
