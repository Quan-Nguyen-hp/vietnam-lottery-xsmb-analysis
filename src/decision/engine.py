"""
DECISION INTELLIGENCE — src/decision/engine.py
DecisionEngine: tổng hợp Probability + Confidence + Risk → Decision.
Triển khai Output Contract, Pipeline Metadata, Spearman Rank PSI và Phân rã Giải thích.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np
import pandas as pd

from .confidence import ConfidenceEngine
from .kelly import KellyCriterion
from .risk_filters import RiskFilters


@dataclass
class NumberDecision:
    """Quyết định đầu ra chuẩn (Output Contract) cho một số loto."""
    number: int
    probability: float
    confidence: float
    risk: str            # "LOW" | "MEDIUM" | "HIGH"
    allocation: float    # Kelly adjusted fraction
    decision: str        # "BET" | "SKIP" | "WATCH"
    explanation: dict    # Phân rã: {"feature_states": {...}, "approximate_contributions": {...}}


@dataclass
class DayDecision:
    """Tổng hợp quyết định cho một ngày dự báo."""
    date: str
    run_id: str
    decisions: list[NumberDecision] = field(default_factory=list)
    diversification_score: float = 1.0
    rank_stability_index: float = 1.0  # Spearman Rank Correlation (PSI)

    # Pipeline Metadata (MLOps)
    feature_version: str = "v1"
    model_version: str = "v1.0"
    belief_version: str = "v1.0"
    evidence_version: str = "v1.0"
    git_commit: str = "abc123x"
    random_seed: int = 42

    @property
    def bets(self) -> list[NumberDecision]:
        return [d for d in self.decisions if d.decision == "BET"]

    @property
    def top_numbers(self) -> list[int]:
        return [d.number for d in self.bets]

    def to_dict(self) -> dict:
        """Đặc tả khớp 100% với Data Contract và Pipeline Metadata."""
        return {
            "pipeline_metadata": {
                "run_id": self.run_id,
                "date": self.date,
                "feature_version": self.feature_version,
                "model_version": self.model_version,
                "belief_version": self.belief_version,
                "evidence_version": self.evidence_version,
                "git_commit": self.git_commit,
                "random_seed": self.random_seed
            },
            "diversification_score": round(self.diversification_score, 4),
            "rank_stability_index": round(self.rank_stability_index, 4),
            "bets": [
                {
                    "number": int(d.number),
                    "probability": round(d.probability, 4),
                    "confidence": round(d.confidence, 4),
                    "risk": d.risk,
                    "allocation": round(d.allocation, 4),
                    "decision": d.decision,
                    "explanation": d.explanation
                }
                for d in self.decisions if d.decision == "BET"
            ]
        }


class DecisionEngine:
    """
    Tích hợp bộ lọc tối ưu danh mục rủi ro của XPIS v1.2.
    """

    DECISION_VERSION = "v1.2"

    def __init__(
        self,
        min_probability: float = 0.31,
        min_confidence: float = 0.45,
        top_k: int = 10,
        kelly_odds: float = 3.666,
        kelly_fraction: float = 0.20,
        min_diversification: float = 0.85,
    ):
        self._min_prob = min_probability
        self._min_conf = min_confidence
        self._top_k = top_k
        self._min_div = min_diversification
        self._confidence_engine = ConfidenceEngine()
        self._kelly = KellyCriterion(
            odds=kelly_odds,
            kelly_fraction=kelly_fraction,
            min_prob=min_probability - 0.02
        )
        self._risk_filters = RiskFilters(min_diversification=min_diversification)
        
        # Lưu trữ dự báo ngày hôm trước để tính Spearman PSI
        self._prev_proba: Optional[np.ndarray] = None

    def decide(
        self,
        date: str,
        meta_proba: np.ndarray,
        model_probas: Optional[dict[str, np.ndarray]] = None,
        S_history: Optional[np.ndarray] = None,
        df_features: Optional[pd.DataFrame] = None,          # 100 rows × 78 features
        feature_importance_df: Optional[pd.DataFrame] = None, # columns: [feature, importance]
        feature_version: str = "v1.0",
        model_version: str = "v1.0",
        belief_version: str = "v1.0",
        evidence_version: str = "v1.0",
        git_commit: str = "abc123x",
        random_seed: int = 42,
    ) -> DayDecision:
        run_id = f"RUN_{date.replace('-', '')}_{uuid.uuid4().hex[:6]}"

        # 1. Tính toán ma trận tương quan thực nghiệm (NMI) động từ lịch sử
        if S_history is not None:
            self._risk_filters.build_empirical_correlation(S_history)

        # 2. Tính Spearman Rank Correlation (PSI) so với hôm qua
        rank_stability = 1.0
        if self._prev_proba is not None:
            # argsort 2 lần trả ra xếp hạng từ 0 đến 99
            rank_curr = np.argsort(np.argsort(meta_proba))
            rank_prev = np.argsort(np.argsort(self._prev_proba))
            d_i = rank_curr - rank_prev
            rank_stability = float(1.0 - (6.0 * np.sum(d_i ** 2)) / (100 * (100 ** 2 - 1)))
        
        self._prev_proba = meta_proba.copy()

        # 3. Tính confidence score
        if model_probas:
            confidence = self._confidence_engine.compute(model_probas, meta_proba)
        else:
            confidence = np.full(100, 0.5)

        # 4. Tìm kiếm ứng viên cược sơ bộ
        candidate_bets = []
        ranks = np.argsort(meta_proba)[::-1]

        for rank_idx, num in enumerate(ranks):
            p = float(meta_proba[num])
            c = float(confidence[num])
            if p >= self._min_prob and c >= self._min_conf:
                candidate_bets.append(int(num))

        candidate_bets = candidate_bets[:self._top_k]

        # 5. Tính Diversification Score
        div_score = self._risk_filters.compute_diversification_score(candidate_bets)
        reject_by_diversification = len(candidate_bets) > 1 and div_score < self._min_div

        # 6. Tính Kelly và Tối ưu hóa phân bổ vốn
        raw_kelly = self._kelly.compute(meta_proba, confidence)
        if not reject_by_diversification:
            optimized_kelly = self._risk_filters.optimize_allocations(raw_kelly, candidate_bets)
        else:
            optimized_kelly = np.zeros(100)

        # 7. Đóng gói danh mục cược + Giải thích (Explainability)
        decisions = []
        for rank_idx, num in enumerate(ranks):
            num_int = int(num)
            p = float(meta_proba[num_int])
            c = float(confidence[num_int])
            k = float(optimized_kelly[num_int])

            # Phân loại rủi ro (Risk)
            # LOW nếu tương quan tối đa với các số khác < 0.05, MEDIUM nếu < 0.15, HIGH nếu >= 0.15
            max_corr = 0.0
            if len(candidate_bets) > 1 and num_int in candidate_bets:
                corrs = [self._risk_filters.get_correlation(num_int, other) for other in candidate_bets if other != num_int]
                max_corr = max(corrs) if corrs else 0.0
            
            risk_level = "LOW"
            if max_corr >= 0.15:
                risk_level = "HIGH"
            elif max_corr >= 0.05:
                risk_level = "MEDIUM"

            # Giải thích lý do lựa chọn (Approximate Explainability)
            explanation = {"feature_states": {}, "approximate_contributions": {}}
            if df_features is not None:
                # Trích xuất Z-score (Feature State) và Contribution cho các đặc trưng lớn
                # Tìm các đặc trưng của số loto này
                # df_features có 100 dòng ứng với 100 số
                row_idx = df_features[df_features['number'] == num_int].index
                if len(row_idx) > 0:
                    row_feat = df_features.loc[row_idx[0]]
                    
                    # Lấy danh sách 5 đặc trưng quan trọng nhất để làm rõ giải thích
                    important_feats = ["delay", "delay_percentile", "freq_30d", "freq_momentum_short", "markov_order1"]
                    if feature_importance_df is not None:
                        important_feats = list(feature_importance_df.head(5)['feature'].values)
                    
                    for feat in important_feats:
                        if feat in df_features.columns and feat not in ("number", "date"):
                            val = float(row_feat[feat])
                            mean_val = float(df_features[feat].mean())
                            std_val = float(df_features[feat].std()) + 1e-8
                            
                            # Feature State (Z-score)
                            z_score = (val - mean_val) / std_val
                            explanation["feature_states"][feat] = f"{z_score:+.2f}σ"
                            
                            # Approximate Feature Contribution
                            importance = 1.0
                            if feature_importance_df is not None:
                                imp_row = feature_importance_df[feature_importance_df['feature'] == feat]
                                if len(imp_row) > 0:
                                    importance = float(imp_row.iloc[0]['importance'])
                            
                            approx_contrib = (val - mean_val) * (importance / 1000.0) # scaling factor
                            explanation["approximate_contributions"][feat] = f"{approx_contrib:+.4f}"

            # Phân loại hành động (Decision action)
            if num_int in candidate_bets and not reject_by_diversification and k > 0:
                action = "BET"
            elif reject_by_diversification and num_int in candidate_bets:
                action = "SKIP"
            elif num_int in candidate_bets and k == 0:
                action = "SKIP"
            elif p >= self._min_prob and c < self._min_conf:
                action = "SKIP"
            elif p < self._min_prob:
                action = "SKIP"
            else:
                action = "WATCH"

            decisions.append(NumberDecision(
                number=num_int,
                probability=p,
                confidence=c,
                risk=risk_level,
                allocation=k,
                decision=action,
                explanation=explanation
            ))

        return DayDecision(
            date=date,
            run_id=run_id,
            decisions=decisions,
            diversification_score=div_score,
            rank_stability_index=rank_stability,
            feature_version=feature_version,
            model_version=model_version,
            belief_version=belief_version,
            evidence_version=evidence_version,
            git_commit=git_commit,
            random_seed=random_seed
        )
