"""
Test và xác thực Meta Learner (Layer 5) cùng Decision Engine (Layer 6).
Quy trình:
1. Load 100 ngày lịch sử.
2. Xây dựng Evidence + FeatureStore cho từng ngày.
3. Chạy 10 model thành phần để tạo feature xác suất (10 cột mới).
4. Ghép toàn bộ thành X_train (88 features), y_train (nhãn thực tế).
5. Train LightGBMMetaLearner + Calibrator.
6. Dự báo ngày kế tiếp, tính Confidence và Kelly → BET/SKIP.
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import time
from pathlib import Path

from data.loader import DataLoader
from evidence.builder import EvidenceBuilder
from evidence.store import EvidenceStore
from features.feature_store import FeatureStore
from probability import get_all_models
from meta.lightgbm_meta import LightGBMMetaLearner
from decision.engine import DecisionEngine

# 1. Load Data
loader = DataLoader().load()
total_days = loader.total_days
train_days = 60
start_idx = total_days - train_days - 1
end_idx = total_days - 1

print(f"--- Bắt đầu Integration Test Meta Learner ---")
print(f"Tổng số ngày trong CSDL: {total_days}")
print(f"Khoảng ngày huấn luyện: {start_idx} -> {end_idx-1} ({train_days} ngày)")
print(f"Ngày cần dự báo (out-of-sample): {end_idx}")

evidence_store = EvidenceStore()
evidence_builder = EvidenceBuilder(evidence_store)
feature_store = FeatureStore(S_history=loader.S)
models = get_all_models()

# 2. Xây dựng dataset cho training
snapshots = []
labels = []

t0 = time.time()
for idx in range(start_idx, end_idx):
    df_hist, S_hist = loader.slice_history(idx)
    target_row = loader.df.iloc[idx]
    target_date = target_row['date'].to_pydatetime()
    date_str = target_date.strftime('%Y-%m-%d')
    
    # 2a. Build Evidence
    df_ev = evidence_builder.build_all(df_hist, S_hist, target_date, save=False)
    
    # 2b. Build Features
    df_feat = feature_store.build(df_ev, date_str, S=S_hist, save_parquet=False)
    
    # 2c. Add 10 model probabilities as features
    # Copy df_feat để tránh mutate cache gốc
    df_feat_extended = df_feat.copy()
    for model in models:
        proba = model.predict_proba(df_feat, df_hist, S_hist)
        df_feat_extended[f"model_{model.name}"] = proba
        
    snapshots.append(df_feat_extended)
    
    # Label thực tế (1 nếu số xuất hiện ngày hôm đó, 0 nếu không)
    prize_cols = loader.prize_cols()
    actual = set(int(v) for v in target_row[prize_cols].dropna() if 0 <= int(v) <= 99)
    y = np.zeros(100)
    for num in actual:
        y[num] = 1
    labels.append(y)

print(f"Đã trích xuất xong đặc trưng. Thời gian: {time.time()-t0:.2f}s")

# 3. Khởi tạo và huấn luyện Meta Learner
meta_learner = LightGBMMetaLearner(calibrate=True)
X_train, y_train = meta_learner.build_training_data(snapshots, labels)
print(f"Kích thước X_train: {X_train.shape} (mỗi ngày 100 dòng × 88 features)")
print(f"Kích thước y_train: {y_train.shape}")

meta_learner.train(X_train, y_train)
print(f"Huấn luyện Meta Learner thành công: {meta_learner.is_trained()}")

# Lưu thử model
model_file = meta_learner.save()
print(f"Đã lưu model tại: {model_file}")

# 4. Dự báo cho ngày mới (out-of-sample)
idx_test = end_idx
df_hist_test, S_hist_test = loader.slice_history(idx_test)
target_row_test = loader.df.iloc[idx_test]
target_date_test = target_row_test['date'].to_pydatetime()
date_str_test = target_date_test.strftime('%Y-%m-%d')

df_ev_test = evidence_builder.build_all(df_hist_test, S_hist_test, target_date_test, save=False)
df_feat_test = feature_store.build(df_ev_test, date_str_test, S=S_hist_test, save_parquet=False)

# Thêm xác suất thành phần làm feature
df_feat_test_extended = df_feat_test.copy()
model_probas = {}
for model in models:
    proba = model.predict_proba(df_feat_test, df_hist_test, S_hist_test)
    df_feat_test_extended[f"model_{model.name}"] = proba
    model_probas[model.name] = proba

# Predict
meta_cols = [c for c in df_feat_test_extended.columns if c not in ("number", "date")]
X_test = df_feat_test_extended[meta_cols].values.astype(np.float32)
meta_proba = meta_learner.predict_proba(X_test)

print(f"\nDự báo cho ngày {date_str_test}:")
print(f"Range xác suất sau hiệu chuẩn: [{meta_proba.min():.4f}, {meta_proba.max():.4f}]")

# 5. Chạy qua Decision Engine
decision_engine = DecisionEngine(min_probability=0.35, min_confidence=0.5)
day_decision = decision_engine.decide(
    date=date_str_test,
    meta_proba=meta_proba,
    model_probas=model_probas,
    feature_version=feature_store.version,
    model_version=meta_learner.version
)

print(f"\nQuyết định từ Decision Engine (Layer 6):")
print(f"Số lượng cược đề xuất: {len(day_decision.bets)}")
for d in day_decision.bets:
    print(f"  Số {d.number:02d} | Xác suất: {d.probability:.4f} | Confidence: {d.confidence:.4f} | Vốn Kelly: {d.capital_pct:.2f}% | Action: {d.action}")

# Clean up
if Path(model_file).exists():
    Path(model_file).unlink()
    Path(model_file).with_suffix('.json').unlink()

print("\n--- Test Meta Learner OK ---")
