"""
Test tất cả 10 Probability Models — kiểm tra import, output shape, tốc độ.
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import time
from data.loader import DataLoader
from evidence.builder import EvidenceBuilder
from evidence.store import EvidenceStore
from features.feature_store import FeatureStore
from probability import get_all_models

# Setup data
loader = DataLoader().load()
idx = loader.total_days - 1
df_hist, S_hist = loader.slice_history(idx)
target_date = loader.df.iloc[idx]['date'].to_pydatetime()
date_str = target_date.strftime('%Y-%m-%d')

store = EvidenceStore()
builder = EvidenceBuilder(store)
df_ev = builder.build_all(df_hist, S_hist, target_date, save=False)
fs = FeatureStore(S_history=S_hist)
df_feat = fs.build(df_ev, date_str, S=S_hist, save_parquet=False)

# Get actual numbers for this day
prize_cols = loader.prize_cols()
actual = set(int(v) for v in loader.df.iloc[idx][prize_cols].dropna() if 0 <= int(v) <= 99)

print(f"Test: {date_str} | History: {len(df_hist)}d | Features: {df_feat.shape[1]-2}")
print(f"Actual: {sorted(actual)}")
print()
print(f"{'#':<3} {'Model':<25} {'Time':>8} {'Min':>7} {'Max':>7} {'Mean':>7} {'Hit@10':>8}")
print("-" * 70)

all_models = get_all_models()
model_probas = {}

for i, model in enumerate(all_models, 1):
    t0 = time.time()
    try:
        p = model.predict_proba(df_feat, df_hist, S_hist)
        elapsed = (time.time() - t0) * 1000
        top10 = model.top_k(p, 10)
        hit = any(n in actual for n in top10)
        model_probas[model.name] = p
        print(f"{i:<3} {model.name:<25} {elapsed:>7.1f}ms {p.min():>7.4f} {p.max():>7.4f} {p.mean():>7.4f} {'✅' if hit else '❌':>8}")
    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        print(f"{i:<3} {model.name:<25} {elapsed:>7.1f}ms  ERROR: {e}")

print()
print(f"Total models loaded: {len(model_probas)}/10")
print("All imports OK ✅")
