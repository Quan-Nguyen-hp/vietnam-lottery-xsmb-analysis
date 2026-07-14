"""
Mini backtest 90 ngày: so sánh Model 9 (Bayesian) vs Model 10 (EWMA).
Dùng top-10, tính hit rate và ROI.
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import time

from data.loader import DataLoader
from evidence.builder import EvidenceBuilder
from evidence.store import EvidenceStore
from features.feature_store import FeatureStore
from probability.bayesian import BayesianPredictor
from probability.ewma_prob import EWMAPredictor

BACKTEST_DAYS = 90
TOP_K = 10
ODDS = 70.0
MIN_HISTORY = 500

loader = DataLoader().load()
evidence_store = EvidenceStore()
builder = EvidenceBuilder(evidence_store)
model9 = BayesianPredictor()
model10 = EWMAPredictor(multi_scale=True)

total = loader.total_days
start_idx = total - BACKTEST_DAYS

results9 = []
results10 = []
t_start = time.time()

print(f"Backtest {BACKTEST_DAYS} ngày | Top-{TOP_K}")
print("-" * 50)

for idx in range(start_idx, total):
    df_hist, S_hist = loader.slice_history(idx)
    if len(df_hist) < MIN_HISTORY:
        continue

    target_row = loader.df.iloc[idx]
    target_date = target_row['date'].to_pydatetime()
    date_str = target_date.strftime('%Y-%m-%d')

    # Actual result
    prize_cols = loader.prize_cols()
    actual = [int(v) for v in target_row[prize_cols].dropna() if 0 <= int(v) <= 99]
    actual_set = set(actual)

    # Build features (dùng cache nếu có)
    df_ev = builder.build_all(df_hist, S_hist, target_date, save=False)
    fs = FeatureStore(S_history=S_hist)
    df_feat = fs.build(df_ev, date_str, S=S_hist, save_parquet=False)

    # Model 9: Bayesian
    p9 = model9.predict_proba(df_feat, df_hist, S_hist)
    top9 = model9.top_k(p9, TOP_K)
    hits9 = len([n for n in top9 if n in actual_set])
    results9.append({
        'date': date_str,
        'hit': hits9 > 0,
        'n_hits': hits9,
        'n_bets': TOP_K,
        'daily_pnl': hits9 * ODDS - TOP_K,
    })

    # Model 10: EWMA
    p10 = model10.predict_proba(df_feat, df_hist, S_hist)
    top10 = model10.top_k(p10, TOP_K)
    hits10 = len([n for n in top10 if n in actual_set])
    results10.append({
        'date': date_str,
        'hit': hits10 > 0,
        'n_hits': hits10,
        'n_bets': TOP_K,
        'daily_pnl': hits10 * ODDS - TOP_K,
    })

elapsed = time.time() - t_start

df9 = pd.DataFrame(results9)
df10 = pd.DataFrame(results10)

def summarize(df, name):
    n = len(df)
    hit_rate = df['hit'].mean()
    total_bets = df['n_bets'].sum()
    total_hits = df['n_hits'].sum()
    total_cost = total_bets
    total_win = total_hits * ODDS
    roi = (total_win - total_cost) / total_cost
    pnl_cum = df['daily_pnl'].cumsum().iloc[-1]
    print(f"  {name}:")
    print(f"    Ngày: {n} | Hit Rate: {hit_rate:.1%} | ROI: {roi:+.1%} | Cum PnL: {pnl_cum:+.0f}")
    print(f"    Tổng đặt: {total_bets} | Tổng trúng: {total_hits}")

print(f"\nKết quả (elapsed: {elapsed:.1f}s):")
summarize(df9, "Model 9 — Bayesian Predictor")
summarize(df10, "Model 10 — EWMA Multi-scale")
print()
print("Tham chiếu uniform random:")
print(f"  Expected hit rate top-10: {1-(90/100)**10:.1%} (nếu random)")
