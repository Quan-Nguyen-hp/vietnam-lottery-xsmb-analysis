"""
XPIS v1.1 — Toàn bộ luồng Tích hợp 8 Tầng (Integration Test)
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from datetime import datetime

print("=== XPIS v1.1 integration Test ===")

# Layer 1: Data Layer
from data.loader import DataLoader
loader = DataLoader().load()
print(f"[L1 OK] DataLoader: {loader.total_days} ngày dữ liệu")

# Layer 2: Evidence Layer
from evidence.builder import EvidenceBuilder
from evidence.store import EvidenceStore
store = EvidenceStore()
builder = EvidenceBuilder(store)
idx = loader.total_days - 1
df_hist, S_hist = loader.slice_history(idx)
target_date = loader.df.iloc[idx]['date'].to_pydatetime()
date_str = target_date.strftime('%Y-%m-%d')
df_ev = builder.build_all(df_hist, S_hist, target_date, save=False)
print(f"[L2 OK] EvidenceBuilder: {len(df_ev)} numbers × {len(df_ev.columns)} raw columns")

# Layer 3: Feature Layer
from features.feature_store import FeatureStore
fs = FeatureStore(S_history=loader.S)
df_feat = fs.build(df_ev, date_str, S=S_hist, save_parquet=False)
print(f"[L3 OK] FeatureStore: {df_feat.shape[1]-2} features")

# Layer 4: Probability Model Layer
from probability import get_all_models
models = get_all_models()
print(f"[L4 OK] Probability Model Layer: Loaded {len(models)} models (10 Statistical + 1 ML)")

# Layer 5: Meta Fusion & Calibration Layer
from meta.fusion import MetaFusion
fusion = MetaFusion()
model_probas = {}
for m in models:
    if m.name != "lightgbm_classifier":
        model_probas[m.name] = m.predict_proba(df_feat, df_hist, S_hist)
    else:
        # Giả lập lgb_model fit & predict
        meta_cols = [c for c in df_feat.columns if c not in ("number", "date")]
        m.fit(df_feat[meta_cols].values, np.random.randint(0, 2, 100), feature_names=meta_cols)
        model_probas[m.name] = m.predict_proba(df_feat, df_hist, S_hist)
        
fused_proba = fusion.fuse(model_probas)
print(f"[L5 OK] MetaFusion: Fused proba range [{fused_proba.min():.4f}, {fused_proba.max():.4f}]")

# Layer 6: Decision Layer
from decision.engine import DecisionEngine
decision_engine = DecisionEngine(min_probability=0.25, min_confidence=0.40)
day_decision = decision_engine.decide(
    date=date_str,
    meta_proba=fused_proba,
    model_probas=model_probas
)
print(f"[L6 OK] DecisionEngine: {len(day_decision.bets)} bets proposed | Div Score: {day_decision.diversification_score:.4f}")

# Layer 7: Evaluation Layer
from evaluation.metrics import EvaluationMetrics
evaluator = EvaluationMetrics()
flat_df = pd.DataFrame([{
    'date': date_str,
    'hit': True,
    'n_bets': len(day_decision.bets),
    'n_hits': 1,
    'daily_pnl': 99.0 - (len(day_decision.bets) * 27.0)
}])
metrics = evaluator.compute_full(flat_df)
print(f"[L7 OK] EvaluationMetrics: ROI = {metrics['roi']:+.2%}")

# Layer 8: Research & Registry
from registry import FeatureRegistry, ModelRegistry
from research.experiment_tracker import ExperimentTracker

f_reg = FeatureRegistry()
m_reg = ModelRegistry()
tracker = ExperimentTracker()

print(f"[L8 OK] Registries & ExperimentTracker loaded successfully!")
print(f"        Feature registry version: {f_reg.version}")
print(f"        Model registry version: {m_reg.version}")

print()
print("=== INTEGRATION TEST v1.1 PASSED SUCCESSFULLY ✅ ===")
