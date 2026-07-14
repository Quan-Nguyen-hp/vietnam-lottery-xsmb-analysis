"""
DECISION INTELLIGENCE — src/decision/risk_filters.py
Quản lý rủi ro danh mục (Portfolio Risk Management) Layer 6.
Triển khai Normalized Mutual Information (NMI) Edge List thay cho Pearson.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

# File path
MI_EDGES_PATH = Path("predictions/mi_edges.json")


class RiskFilters:
    """
    Bộ quản lý và tối ưu hóa danh mục cược loto.
    Ngăn chặn việc phân bổ quá nhiều vốn vào các nhóm số bị trùng lặp rủi ro.
    """

    def __init__(
        self,
        max_head_exposure: float = 0.20,
        max_tail_exposure: float = 0.20,
        min_diversification: float = 0.85,
    ):
        self._max_head = max_head_exposure
        self._max_tail = max_tail_exposure
        self._min_div = min_diversification
        self._mi_matrix: np.ndarray = np.zeros((100, 100))

    def build_empirical_correlation(self, S_history: np.ndarray) -> None:
        """
        Tính toán ma trận tương quan thực nghiệm Normalized Mutual Information (NMI)
        trên cửa sổ trượt 1800 ngày gần nhất và xuất danh sách Edge List.
        """
        # Giới hạn 1800 ngày gần nhất
        S = S_history[-1800:] if len(S_history) > 1800 else S_history
        N = S.shape[0]
        if N < 50:
            self._mi_matrix = np.eye(100)
            return

        N_active = S.sum(axis=0).astype(float)
        
        # Joint counts (100, 100)
        N_11 = (S.T @ S).astype(float)
        N_10 = N_active[:, np.newaxis] - N_11
        N_01 = N_active[np.newaxis, :] - N_11
        N_00 = N - N_active[:, np.newaxis] - N_active[np.newaxis, :] + N_11
        
        # Marginals
        P_1 = N_active / N
        P_0 = 1.0 - P_1
        
        P_X1_Y1 = P_1[:, np.newaxis] * P_1[np.newaxis, :]
        P_X1_Y0 = P_1[:, np.newaxis] * P_0[np.newaxis, :]
        P_X0_Y1 = P_0[:, np.newaxis] * P_1[np.newaxis, :]
        P_X0_Y0 = P_0[:, np.newaxis] * P_0[np.newaxis, :]
        
        def safe_term(P_joint, P_marginal):
            ratio = np.divide(P_joint, P_marginal + 1e-12, out=np.zeros_like(P_joint), where=P_joint > 0)
            return np.where(P_joint > 0, P_joint * np.log(ratio + 1e-12), 0.0)

        I_XY = (
            safe_term(N_11 / N, P_X1_Y1) +
            safe_term(N_10 / N, P_X1_Y0) +
            safe_term(N_01 / N, P_X0_Y1) +
            safe_term(N_00 / N, P_X0_Y0)
        )
        
        # Entropy
        H = - (P_1 * np.log(P_1 + 1e-12) + P_0 * np.log(P_0 + 1e-12))
        H_geom = np.sqrt(H[:, np.newaxis] * H[np.newaxis, :])
        
        NMI = np.divide(I_XY, H_geom + 1e-12, out=np.zeros_like(I_XY), where=H_geom > 0)
        np.fill_diagonal(NMI, 1.0)
        
        self._mi_matrix = np.clip(NMI, 0.0, 1.0)

        # Xuất Edge List ra file JSON để lưu vết tri thức
        edge_list = []
        for i in range(100):
            for j in range(i + 1, 100):
                w = float(self._mi_matrix[i, j])
                if w > 0.005:  # Chỉ giữ lại các mối liên kết có tín hiệu đáng kể
                    edge_list.append({
                        "from": i,
                        "to": j,
                        "weight": round(w, 4)
                    })
                    
        edge_list = sorted(edge_list, key=lambda x: x["weight"], reverse=True)
        
        MI_EDGES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MI_EDGES_PATH, "w", encoding="utf-8") as f:
            json.dump(edge_list, f, ensure_ascii=False, indent=2)

    def get_correlation(self, num1: int, num2: int) -> float:
        """Trả về tương quan NMI thực nghiệm giữa 2 số."""
        return float(self._mi_matrix[num1, num2])

    def compute_diversification_score(self, numbers: list[int]) -> float:
        """
        Tính điểm đa dạng hóa danh mục cược.
        """
        n = len(numbers)
        if n <= 1:
            return 1.0

        correlations = []
        for i in range(n):
            for j in range(i + 1, n):
                correlations.append(self.get_correlation(numbers[i], numbers[j]))

        mean_corr = np.mean(correlations)
        diversification = 1.0 - mean_corr
        return float(diversification)

    def optimize_allocations(
        self,
        raw_kelly: np.ndarray,
        top_numbers: list[int],
    ) -> np.ndarray:
        """
        Tối ưu hóa phân bổ vốn Kelly dựa trên rủi ro tương quan NMI thực nghiệm và giới hạn nhóm.
        """
        allocations = raw_kelly.copy()
        
        # 1. Điều chỉnh giảm quy mô vốn cho các cặp có tương quan cao (NMI > 0.15)
        n_numbers = len(top_numbers)
        for i in range(n_numbers):
            for j in range(i + 1, n_numbers):
                n1, n2 = top_numbers[i], top_numbers[j]
                corr = self.get_correlation(n1, n2)
                if corr > 0.15:  # NMI thường nhỏ hơn Pearson nên ngưỡng tương quan nhạy bén hơn
                    factor = 1.0 - (corr * 0.50)
                    allocations[n1] *= factor
                    allocations[n2] *= factor

        # 2. Đầu số Exposure Limit
        head_totals = {}
        for num in top_numbers:
            h = num // 10
            head_totals[h] = head_totals.get(h, 0.0) + allocations[num]

        for h, total in head_totals.items():
            if total > self._max_head:
                scale = self._max_head / total
                for num in top_numbers:
                    if num // 10 == h:
                        allocations[num] *= scale

        # 3. Đuôi số Exposure Limit
        tail_totals = {}
        for num in top_numbers:
            t = num % 10
            tail_totals[t] = tail_totals.get(t, 0.0) + allocations[num]

        for t, total in tail_totals.items():
            if total > self._max_tail:
                scale = self._max_tail / total
                for num in top_numbers:
                    if num % 10 == t:
                        allocations[num] *= scale

        return np.clip(allocations, 0.0, 1.0)
