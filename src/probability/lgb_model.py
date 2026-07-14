"""
PROBABILITY MODEL LAYER — src/probability/lgb_model.py
Model 11: LightGBM Machine Learning Classifier

Lớp học máy này học trực tiếp trên tập đặc trưng của FeatureStore (78 đặc trưng).
Nhiệm vụ duy nhất: Trả ra mảng xác suất thô (100,) đại diện cho khả năng nổ của mỗi số.
Không thực hiện hiệu chuẩn (Calibration) tại đây (Calibration thuộc về Layer 5).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _lgb_available = True
except ImportError:
    _lgb_available = False

from .base import BaseProbabilityModel
try:
    from ..registry.model_registry import ModelRegistry
except ImportError:
    from registry.model_registry import ModelRegistry

MODEL_DIR = Path("predictions/models")


class LightGBMProbabilityModel(BaseProbabilityModel):
    """
    Model 11: Machine Learning Model dùng LightGBM Classifier.
    Tự động đọc tham số huấn luyện từ ModelRegistry.
    """

    def __init__(self, model_path: Optional[str] = None):
        if not _lgb_available:
            raise ImportError("Vui lòng cài đặt lightgbm: pip install lightgbm")

        self._model: Optional[lgb.LGBMClassifier] = None
        self._feature_cols: list[str] = []
        self._is_trained = False

        # Load parameters from Registry
        registry = ModelRegistry()
        meta = registry.get_model_meta("lightgbm_classifier")
        self._params = meta.get("parameters") if meta else {
            "n_estimators": 80,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 30
        }

        if model_path and Path(model_path).exists():
            self.load(model_path)

    @property
    def name(self) -> str:
        return "lightgbm_classifier"

    @property
    def version(self) -> str:
        return "1.0"

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray, feature_names: Optional[list[str]] = None) -> None:
        """Huấn luyện mô hình cơ sở."""
        if isinstance(X, np.ndarray) and feature_names:
            self._feature_cols = feature_names
            X = pd.DataFrame(X, columns=feature_names)
        elif isinstance(X, pd.DataFrame):
            self._feature_cols = list(X.columns)

        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=self._params.get("n_estimators", 80),
            learning_rate=self._params.get("learning_rate", 0.05),
            num_leaves=self._params.get("num_leaves", 15),
            min_child_samples=self._params.get("min_child_samples", 30),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )

        clf.fit(X, y)
        self._model = clf
        self._is_trained = True

    def predict_proba(
        self,
        df_features: pd.DataFrame,
        df_history: Optional[pd.DataFrame] = None,
        S_history: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Dự báo xác suất thô (chưa hiệu chuẩn) cho 100 số."""
        if not self._is_trained or self._model is None:
            # Fallback nếu chưa train
            return np.full(len(df_features), 0.27)

        # Lọc các cột đặc trưng phi thời gian/phi danh tính
        meta_cols = [c for c in df_features.columns if c not in ("number", "date")]
        X = df_features[meta_cols]

        # Dự báo xác suất thô
        probs = self._model.predict_proba(X)[:, 1]
        return probs

    def save(self, path: str) -> None:
        """Lưu model weights."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "feature_cols": self._feature_cols,
            "is_trained": self._is_trained
        }
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> None:
        """Đọc model weights."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._model = payload["model"]
        self._feature_cols = payload.get("feature_cols", [])
        self._is_trained = payload.get("is_trained", False)

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Trả về top-N feature importances."""
        if not self._is_trained or self._model is None:
            return pd.DataFrame({"feature": [], "importance": []})
        names = self._feature_cols if self._feature_cols else [f"feat_{i}" for i in range(self._model.n_features_in_)]
        imp = self._model.feature_importances_
        df = pd.DataFrame({"feature": names, "importance": imp})
        return df.sort_values("importance", ascending=False).head(top_n)
