"""
Test Model 9 (Bayesian) và Model 10 (EWMA) trên dữ liệu thực.
Đo: tốc độ, output shape, phân phối xác suất.
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import time
from data.loader import DataLoader
from evidence.builder import EvidenceBuilder
from evidence.store import EvidenceStore
from features.feature_store import FeatureStore
from probability.bayesian import BayesianPredictor
from probability.ewma_prob import EWMAPredictor

# Setup
loader = DataLoader().load()
idx = loader.total_days - 1
df_hist, S_hist = loader.slice_history(idx)
target_date = loader.df.iloc[idx]['date'].to_pydatetime()
date_str = target_date.strftime('%Y-%m-%d')

print(f"Test date: {date_str} | History: {len(df_hist)} days")
print()

# Build Evidence + Features
store = EvidenceStore()
builder = EvidenceBuilder(store)
df_ev = builder.build_all(df_hist, S_hist, target_date, save=False)
fs = FeatureStore(S_history=S_hist)
df_feat = fs.build(df_ev, date_str, S=S_hist)
print(f"FeatureVector: {df_feat.shape} | cols: {list(df_feat.columns[:8])}...")
print()

# ===== Model 9: Bayesian Predictor =====
print("=" * 50)
print("Model 9: Bayesian Predictor")
model9 = BayesianPredictor()

t0 = time.time()
proba9 = model9.predict_proba(df_feat, df_hist, S_hist)
t1 = time.time()

print(f"  Time: {(t1-t0)*1000:.1f}ms")
print(f"  Shape: {proba9.shape}")
print(f"  Range: [{proba9.min():.4f}, {proba9.max():.4f}]")
print(f"  Mean:  {proba9.mean():.4f} (expected ~0.27)")
print(f"  Std:   {proba9.std():.4f}")
top10_9 = model9.top_k(proba9, 10)
print(f"  Top 10: {[f'{n:02d}' for n in top10_9]}")
print()

# ===== Model 10: EWMA Predictor =====
print("=" * 50)
print("Model 10: EWMA Predictor (Multi-scale)")
model10 = EWMAPredictor(multi_scale=True)

t0 = time.time()
proba10 = model10.predict_proba(df_feat, df_hist, S_hist)
t1 = time.time()

print(f"  Time: {(t1-t0)*1000:.1f}ms")
print(f"  Shape: {proba10.shape}")
print(f"  Range: [{proba10.min():.4f}, {proba10.max():.4f}]")
print(f"  Mean:  {proba10.mean():.4f} (expected ~0.27)")
print(f"  Std:   {proba10.std():.4f}")
top10_10 = model10.top_k(proba10, 10)
print(f"  Top 10: {[f'{n:02d}' for n in top10_10]}")
print()

# All scales breakdown
all_scales = model10.compute_all_scales(S_hist)
print("  Multi-scale breakdown:")
for col in all_scales.columns:
    vals = all_scales[col].values
    print(f"    {col}: mean={vals.mean():.4f} std={vals.std():.4f} max={vals.max():.4f}")
print()

# Actual result for this day
prize_cols = loader.prize_cols()
actual = loader.df.iloc[idx][prize_cols].dropna().astype(int).tolist()
actual = [n for n in actual if 0 <= n <= 99]
print(f"Actual numbers: {[f'{n:02d}' for n in sorted(set(actual))]}")
print(f"  Model 9 hit:  {any(n in top10_9 for n in actual)}")
print(f"  Model 10 hit: {any(n in top10_10 for n in actual)}")
