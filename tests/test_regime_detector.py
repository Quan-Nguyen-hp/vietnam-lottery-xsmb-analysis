"""
TEST SUITE — tests/test_regime_detector.py
Kiểm thử Bộ phát hiện Trạng thái Thị trường (RegimeDetector) và MetaFusion Regime Boost.
"""
import numpy as np
import pytest

from src.meta.regime_detector import RegimeDetector, MarketRegime
from src.meta.fusion import MetaFusion


def test_regime_detector_repeat():
    """Kiểm thử phát hiện REPEAT regime khi tỷ lệ lô rơi cao."""
    # Tạo S_history 35 ngày x 100 số, trong đó các số nổ liên tục 2 ngày liên tiếp
    S_history = np.zeros((35, 100), dtype=np.int8)
    for day in range(35):
        # 10 số cố định nổ mỗi ngày (tỷ lệ rơi cực cao)
        S_history[day, :10] = 1
        S_history[day, 10:27] = 1 if day % 2 == 0 else 0

    detector = RegimeDetector(repeat_threshold=0.28, khan_threshold=0.15)
    regime, metrics = detector.detect(S_history, window_days=30)

    assert regime == MarketRegime.REPEAT, "Tỷ lệ rơi cao phải phân loại là REPEAT regime"
    assert metrics["repeat_rate"] >= 0.28
    assert "khan_rate" in metrics


def test_regime_detector_khan():
    """Kiểm thử phát hiện KHAN regime khi các số khan >= 15 ngày nổ."""
    S_history = np.zeros((45, 100), dtype=np.int8)
    # 30 ngày đầu luân phiên hai tập số (0-26 vào ngày chẵn, 27-53 vào ngày lẻ) -> repeat_rate = 0
    for day in range(30):
        if day % 2 == 0:
            S_history[day, :27] = 1
        else:
            S_history[day, 27:54] = 1

    # 15 ngày sau, các cụm số khan lâu ngày (55-94) nổ rải rác
    for day_idx, start_num in enumerate(range(55, 95, 4)):
        target_day = 30 + (day_idx % 15)
        S_history[target_day, start_num:start_num + 4] = 1

    detector = RegimeDetector(repeat_threshold=0.28, khan_threshold=0.08)
    regime, metrics = detector.detect(S_history, window_days=30)

    assert regime == MarketRegime.KHAN, f"Tỷ lệ nổ số khan cao phải phân loại là KHAN regime, thu được: {regime}, metrics={metrics}"
    assert metrics["khan_rate"] >= 0.08


def test_meta_fusion_regime_boost():
    """Kiểm thử việc tăng trọng số thích ứng trong MetaFusion."""
    fusion = MetaFusion()
    initial_weights = {
        "loto_repeat": 0.10,
        "markov_chain": 0.10,
        "max_delay": 0.10,
        "poisson_estimator": 0.10,
        "count_ewma_poisson": 0.10,
        "bayesian_predictor": 0.50,
    }

    # Áp dụng boost cho REPEAT regime
    boosted_repeat = fusion.apply_regime_boost(initial_weights, MarketRegime.REPEAT, boost_factor=0.30)

    assert boosted_repeat["loto_repeat"] > initial_weights["loto_repeat"]
    assert boosted_repeat["markov_chain"] > initial_weights["markov_chain"]
    assert pytest.approx(sum(boosted_repeat.values()), 1e-5) == 1.0

    # Áp dụng boost cho KHAN regime
    boosted_khan = fusion.apply_regime_boost(initial_weights, MarketRegime.KHAN, boost_factor=0.30)

    assert boosted_khan["max_delay"] > initial_weights["max_delay"]
    assert boosted_khan["count_ewma_poisson"] > initial_weights["count_ewma_poisson"]
    assert pytest.approx(sum(boosted_khan.values()), 1e-5) == 1.0
