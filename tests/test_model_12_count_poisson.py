"""
TEST SUITE — tests/test_model_12_count_poisson.py
Kiểm thử tích hợp Mô hình 12 (CountEWMAPoissonPredictor).
"""
import numpy as np
import pandas as pd
import pytest

from src.probability import get_all_models, CountEWMAPoissonPredictor
from src.registry.model_registry import ModelRegistry


def test_model_12_in_all_models():
    """Đảm bảo Model 12 xuất hiện trong danh sách get_all_models()."""
    models = get_all_models()
    assert len(models) == 12, "Phải có đủ 12 mô hình trong get_all_models()"
    model_names = [m.name for m in models]
    assert "count_ewma_poisson" in model_names, "count_ewma_poisson phải nằm trong danh sách models"


def test_count_ewma_poisson_predict_proba():
    """Kiểm tra dự báo xác suất của CountEWMAPoissonPredictor."""
    model = CountEWMAPoissonPredictor()
    df_feat = pd.DataFrame({"number": np.arange(100)})
    
    # Giả lập lịch sử 100 ngày x 27 giải
    rows = []
    for day in range(100):
        row = {"date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)}
        for p in range(27):
            row[f"prize_{p}"] = 83 if p == 0 else (p + day) % 100
        rows.append(row)
    df_hist = pd.DataFrame(rows)

    proba = model.predict_proba(df_feat, df_history=df_hist)

    assert isinstance(proba, np.ndarray)
    assert proba.shape == (100,)
    assert np.all((proba >= 0.0) & (proba <= 1.0)), "Xác suất phải nằm trong khoảng [0, 1]"
    assert proba[83] > proba[0], "Số 83 nổ liên tục 100 ngày phải có xác suất cao hơn"


def test_model_12_registry():
    """Kiểm tra ModelRegistry ghi nhận đúng thông tin Model 12."""
    registry = ModelRegistry()
    meta = registry.get_model_meta("count_ewma_poisson")
    assert meta is not None, "Model 12 phải tồn tại trong registry"
    assert meta["status"] == "active", "Model 12 phải active"
    assert meta["type"] == "statistical"
