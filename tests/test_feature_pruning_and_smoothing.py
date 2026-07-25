"""
TEST SUITE — tests/test_feature_pruning_and_smoothing.py
Kiểm thử tính năng Cắt tỉa đặc trưng rác (delay_sq) và Làm mịn EWMA 3 ngày (cond_prob_yesterday).
"""
import numpy as np
import pandas as pd
import pytest

from src.features.delay_features import DelayFeatureExtractor
from src.features.bayesian_features import BayesianFeatureExtractor
from src.registry.feature_registry import FeatureRegistry


def test_delay_sq_is_pruned():
    """Đảm bảo delay_sq không xuất hiện trong kết quả trích xuất đặc trưng."""
    df_evidence = pd.DataFrame([
        {"number": 0, "current_delay": 5, "historical_delays": [3, 4, 5]},
        {"number": 1, "current_delay": 12, "historical_delays": [10, 12]},
    ])

    extractor = DelayFeatureExtractor()
    df_feat = extractor.extract(df_evidence)

    assert "delay" in df_feat.columns
    assert "delay_sq" not in df_feat.columns, "delay_sq đáng lẽ phải bị cắt tỉa!"
    assert "delay_zscore" in df_feat.columns


def test_bayesian_ewma_smoothing():
    """Kiểm thử tính năng EWMA 3 ngày cho cond_prob_yesterday."""
    # Tạo ma trận S giả lập 5 ngày x 100 số
    S_history = np.zeros((5, 100), dtype=np.int8)
    S_history[0, :27] = 1
    S_history[1, 10:37] = 1
    S_history[2, 20:47] = 1
    S_history[3, 30:57] = 1
    S_history[4, 40:67] = 1

    df_evidence = pd.DataFrame([
        {"number": i, "inverted": 0, "mirror": 0, "freq_30d": 0.27}
        for i in range(100)
    ])

    extractor = BayesianFeatureExtractor(S_history=S_history)
    df_feat = extractor.extract(df_evidence)

    assert "cond_prob_yesterday" in df_feat.columns
    assert not df_feat["cond_prob_yesterday"].isna().any(), "Xác suất không được chứa giá trị NaN!"

    # Kiểm tra các giá trị nằm trong khoảng [0, 1]
    values = df_feat["cond_prob_yesterday"].values
    assert np.all(values >= 0.0) and np.all(values <= 1.0)


def test_feature_registry_updates():
    """Kiểm tra FeatureRegistry cập nhật đúng trạng thái."""
    registry = FeatureRegistry()

    # delay_sq phải ở trạng thái deprecated
    assert not registry.is_valid("delay_sq"), "delay_sq không được active!"

    # cond_prob_yesterday phải ở trạng thái active
    assert registry.is_valid("cond_prob_yesterday")
    feat_meta = registry._data["features"]["cond_prob_yesterday"]
    assert feat_meta["version"] == "1.1"
    assert "EWMA" in feat_meta["formula"]
