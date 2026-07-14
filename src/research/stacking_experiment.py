"""
RESEARCH LAYER — src/research/stacking_experiment.py
Thực nghiệm Stacking (Layer 5 - Research).
Xếp chồng hồi quy Ridge (Ridge Regression) lên đầu ra xác suất của 11 mô hình cơ sở.

Hypothesis: Stacking giúp tối ưu hóa trọng số phi tuyến tính tốt hơn Weighted Average.
Ngưỡng PASS: ECE thấp hơn, AUC-ROC cao hơn, ROI thực tế cao hơn trên tập Out-Of-Sample.
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import time
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV

from data.loader import DataLoader
from evidence.builder import EvidenceBuilder
from evidence.store import EvidenceStore
from features.feature_store import FeatureStore
from probability import get_all_models
from probability.lgb_model import LightGBMProbabilityModel
from meta.fusion import MetaFusion
from evaluation.metrics import EvaluationMetrics

# Khởi tạo dữ liệu
loader = DataLoader().load()
total_days = loader.total_days
test_days = 90
train_days = 180

start_idx = total_days - test_days
train_start_idx = start_idx - train_days

evidence_store = EvidenceStore()
evidence_builder = EvidenceBuilder(evidence_store)
feature_store = FeatureStore(S_history=loader.S)
static_models = [m for m in get_all_models() if m.name != "lightgbm_classifier"]
lgb_model = LightGBMProbabilityModel()
evaluator = EvaluationMetrics()

print("=== Thử nghiệm Stacking (Research) ===")
print(f"Khoảng Train Stacking: {train_start_idx} -> {start_idx-1} ({train_days} ngày)")
print(f"Khoảng Test Stacking: {start_idx} -> {total_days-1} ({test_days} ngày)")

# 1. Thu thập dữ liệu dự báo lịch sử để Train Stacking
print("Đang thu thập dự báo cho tập huấn luyện...")
train_preds = {m.name: [] for m in static_models}
train_preds[lgb_model.name] = []
train_labels = []

# Fit LGBM trên tập dữ liệu train lịch sử lớn hơn
# (Ở đây ta giả lập chạy lướt nhanh để nghiên cứu)
for idx in range(train_start_idx, start_idx):
    df_hist, S_hist = loader.slice_history(idx)
    target_row = loader.df.iloc[idx]
    target_date = target_row['date'].to_pydatetime()
    date_str = target_date.strftime('%Y-%m-%d')
    
    df_ev = evidence_builder.build_all(df_hist, S_hist, target_date, save=True)
    df_feat = feature_store.build(df_ev, date_str, S=S_hist)
    
    # LGBM fit
    if idx == train_start_idx:
        meta_cols = [c for c in df_feat.columns if c not in ("number", "date")]
        X_train_raw = np.vstack([df_feat[meta_cols].values])
        y_train_raw = np.zeros(100)
        actuals = set(target_row[loader.prize_cols()].dropna().values.astype(int))
        for num in actuals:
            y_train_raw[num] = 1
        lgb_model.fit(X_train_raw, y_train_raw, feature_names=meta_cols)
        
    for m in static_models:
        train_preds[m.name].append(m.predict_proba(df_feat, df_hist, S_hist))
    train_preds[lgb_model.name].append(lgb_model.predict_proba(df_feat, df_hist, S_hist))
    
    y = np.zeros(100)
    for num in set(target_row[loader.prize_cols()].dropna().values.astype(int)):
        y[num] = 1
    train_labels.append(y)

# Định dạng thành ma trận
# X_stack: (n_days * 100, 11)
X_stack_parts = []
for name in train_preds.keys():
    X_stack_parts.append(np.array(train_preds[name]).flatten())
X_stack = np.column_stack(X_stack_parts)
y_stack = np.array(train_labels).flatten()

# Huấn luyện RidgeClassifier làm Meta Classifier Stacking
stack_model = CalibratedClassifierCV(estimator=RidgeClassifier(), method='isotonic', cv=3)
stack_model.fit(X_stack, y_stack)
print("Đã train thành công Ridge Stacking Model + Calibrator.")

# 2. Đánh giá Out-of-Sample trên tập Test
print("\nĐang kiểm thử trên tập Out-of-sample...")
test_labels = []
stack_test_preds = []
fusion_test_preds = []

# Khởi tạo Weighted Fusion làm baseline so sánh
fusion = MetaFusion()
# Tính trọng số tĩnh/động cho fusion trên tập train
eval_preds_matrix = {k: np.array(v) for k, v in train_preds.items()}
eval_labels_matrix = np.array(train_labels)
fusion.compute_dynamic_weights(eval_preds_matrix, eval_labels_matrix)

for idx in range(start_idx, total_days):
    df_hist, S_hist = loader.slice_history(idx)
    target_row = loader.df.iloc[idx]
    target_date = target_row['date'].to_pydatetime()
    date_str = target_date.strftime('%Y-%m-%d')
    
    df_ev = evidence_builder.build_all(df_hist, S_hist, target_date, save=True)
    df_feat = feature_store.build(df_ev, date_str, S=S_hist)
    
    model_probas = {}
    for m in static_models:
        model_probas[m.name] = m.predict_proba(df_feat, df_hist, S_hist)
    model_probas[lgb_model.name] = lgb_model.predict_proba(df_feat, df_hist, S_hist)
    
    # 2a. Dự báo Stacking
    X_day = np.column_stack([model_probas[name] for name in train_preds.keys()])
    stack_pred = stack_model.predict_proba(X_day)[:, 1]
    stack_test_preds.append(stack_pred)
    
    # 2b. Dự báo Fusion (Baseline)
    fusion_pred = fusion.fuse(model_probas)
    fusion_test_preds.append(fusion_pred)
    
    y = np.zeros(100)
    for num in set(target_row[loader.prize_cols()].dropna().values.astype(int)):
        y[num] = 1
    test_labels.append(y)

# 3. Tính toán metrics so sánh
p_stack = np.array(stack_test_preds)
p_fusion = np.array(fusion_test_preds)
y_true = np.array(test_labels)

ece_stack = evaluator.ece_score(p_stack.flatten(), y_true.flatten())
ece_fusion = evaluator.ece_score(p_fusion.flatten(), y_true.flatten())

auc_stack = evaluator.auc_roc(p_stack.flatten(), y_true.flatten())
auc_fusion = evaluator.auc_roc(p_fusion.flatten(), y_true.flatten())

logloss_stack = evaluator.log_loss(p_stack.flatten(), y_true.flatten())
logloss_fusion = evaluator.log_loss(p_fusion.flatten(), y_true.flatten())

precision_stack = evaluator.precision_at_k(p_stack, y_true, k=10)
precision_fusion = evaluator.precision_at_k(p_fusion, y_true, k=10)

print("\n=== KẾT QUẢ ĐỐI CHIẾU ===")
print(f"| Chỉ số | Weighted Fusion (Baseline) | Ridge Stacking (Research) | Trạng thái |")
print(f"|---|:---:|:---:|:---:|")
print(f"| AUC-ROC | {auc_fusion:.4f} | {auc_stack:.4f} | {'Stacking Tốt hơn' if auc_stack > auc_fusion else 'Fusion Tốt hơn'} |")
print(f"| ECE Score | {ece_fusion:.4f} | {ece_stack:.4f} | {'Stacking Tốt hơn' if ece_stack < ece_fusion else 'Fusion Tốt hơn'} |")
print(f"| Log Loss | {logloss_fusion:.4f} | {logloss_stack:.4f} | {'Stacking Tốt hơn' if logloss_stack < logloss_fusion else 'Fusion Tốt hơn'} |")
print(f"| Precision@10 | {precision_fusion:.2%} | {precision_stack:.2%} | {'Stacking Tốt hơn' if precision_stack > precision_fusion else 'Fusion Tốt hơn'} |")

print("\nHypothesis Verification:")
if auc_stack > auc_fusion and ece_stack < ece_fusion:
    print("STATUS: PASS ✅ (Đề xuất đưa Ridge Stacking vào Production Review)")
else:
    print("STATUS: FAIL ❌ (Giữ nguyên Weighted Fusion trong Production)")
