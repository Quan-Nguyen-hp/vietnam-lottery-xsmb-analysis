"""
daily_predict.py — Dự đoán XSMB hàng ngày theo đặc tả XPIS v1.2 APPROVED.
Kết nối đầy đủ 8 tầng kiến trúc, MLOps Pipeline Metadata và Decision Output Contract.

Cách dùng:
  python daily_predict.py                      # Dự đoán cho ngày hôm nay (mặc định)
  python daily_predict.py --date 2026-07-14    # Cho một ngày cụ thể
  python daily_predict.py --dry-run            # Chạy thử nghiệm, không ghi log
"""
import sys
import json
import argparse
import hashlib
import platform
import subprocess
import sklearn
import lightgbm
from datetime import datetime, date, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir / "src"))

import numpy as np
import pandas as pd

from data.loader import DataLoader
from evidence.builder import EvidenceBuilder
from evidence.store import EvidenceStore
from features.feature_store import FeatureStore
from probability import get_all_models
from probability.count_expectation import EWMAHitCountChallenger
from probability.lgb_model import LightGBMProbabilityModel
from meta.fusion import MetaFusion
from decision.engine import DecisionEngine
from decision.edge_gate import EdgeGate
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator

DATA_CSV = root_dir / "data" / "xsmb-2-digits.csv"
PRED_LOG = root_dir / "predictions" / "prediction_log.json"
TZ = ZoneInfo("Asia/Ho_Chi_Minh")

COST_PER_BET = 27.0
PAYOUT_PER_HIT = 99.0


def get_git_commit() -> str:
    """Đọc hash commit git hiện tại, hoặc trả về mã build nếu thất bại."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "build_1.3.2"


def get_file_hash(path: Path) -> str:
    """Tính toán SHA256 hash của tệp cấu hình để lưu vào manifest kiểm toán."""
    if path.exists():
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    return "none"


def load_log() -> list:
    if PRED_LOG.exists():
        with open(PRED_LOG, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []


def save_log(log: list) -> None:
    PRED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PRED_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def fetch_today_data():
    """Tự động fetch dữ liệu mới nhất từ web (nếu có)."""
    try:
        from lottery import Lottery
        from datetime import timedelta

        print("🌐 Đang cập nhật dữ liệu mới nhất từ web…")
        lottery = Lottery()
        lottery.load()
        now = datetime.now(TZ)
        last_date = now.date()
        if now.time() < dtime(18, 35):
            last_date -= timedelta(days=1)

        begin = lottery.get_last_date()
        delta = (last_date - begin).days
        if delta > 0:
            for i in range(1, delta + 1):
                d = begin + timedelta(days=i)
                print(f"  Fetch ngày {d}…")
                lottery.fetch(d)
            lottery.generate_dataframes()
            lottery.dump()
            print(f"✅ Đã cập nhật đến {last_date}")
        else:
            print("✅ Dữ liệu đã cập nhật mới nhất")
    except Exception as e:
        print(f"⚠️  Không thể fetch dữ liệu mới: {e}")
        print("   Tiếp tục dùng dữ liệu CSV hiện có")


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Tính Expected Calibration Error (ECE) thu nhỏ."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    return float(ece)


def run_shared_prediction_pipeline(
    target_date: date,
    top_k: int = 2,
    prior_entries: list[dict] | None = None,
) -> dict:
    """Pipeline production dùng chung; ``prior_entries`` phục vụ walk-forward backtest."""

    target_date_str = str(target_date)

    # Đọc log lịch sử để lấy trọng số ngày hôm trước phục vụ EMA smoothing
    log = prior_entries if prior_entries is not None else load_log()
    prev_weights = {}
    if log:
        # Lấy bản ghi cuối cùng có dynamic_weights
        for entry in reversed(log):
            meta = entry.get("pipeline_metadata", {})
            if "dynamic_weights" in meta:
                prev_weights = meta["dynamic_weights"]
                break

    loader = DataLoader(DATA_CSV).load()
    df_full = loader.df
    
    target_dt = pd.to_datetime(target_date).to_pydatetime()
    
    # Định vị ngày trong dữ liệu lịch sử
    history_indices = df_full[df_full['date'] < pd.Timestamp(target_date)].index.tolist()
    if not history_indices:
        raise ValueError(f"Không tìm thấy lịch sử trước ngày {target_date_str}")
        
    # slice_history dùng mốc exclusive; +1 để bao gồm đúng ngày lịch sử gần nhất.
    history_end_idx = history_indices[-1] + 1
    df_hist, S_hist = loader.slice_history(history_end_idx)

    # 1. Evidence & Feature Engine
    evidence_store = EvidenceStore()
    evidence_builder = EvidenceBuilder(evidence_store)
    feature_store = FeatureStore(S_history=loader.S, evidence_store=evidence_store)

    df_ev = evidence_builder.build_all(df_hist, S_hist, target_dt, save=True)
    df_feat = feature_store.build(df_ev, target_date_str, S=S_hist)

    # 2. Layer 4: Load 11 models
    static_models = [m for m in get_all_models() if m.name != "lightgbm_classifier"]
    lgb_model = LightGBMProbabilityModel()
    
    # Train lgb_model trên 365 ngày lịch sử (kết thúc trước 90 ngày validation để tránh optimistic bias - D3 Point 1)
    train_end_idx = history_end_idx - 90
    train_start_idx = max(50, train_end_idx - 365)
    
    train_snapshots = []
    train_labels = []
    for t_idx in range(train_start_idx, train_end_idx):
        t_df_hist, t_S_hist = loader.slice_history(t_idx)
        t_row = df_full.iloc[t_idx]
        t_date = t_row['date'].to_pydatetime()
        
        t_df_ev = evidence_builder.build_all(t_df_hist, t_S_hist, t_date, save=True)
        t_df_feat = feature_store.build(t_df_ev, t_date.strftime('%Y-%m-%d'), S=t_S_hist)
        train_snapshots.append(t_df_feat)
        
        t_actual = set(t_row[loader.prize_cols()].dropna().values.astype(int).tolist())
        t_y = np.zeros(100)
        for num in t_actual:
            t_y[num] = 1
        train_labels.append(t_y)
        
    meta_cols = [c for c in train_snapshots[0].columns if c not in ("number", "date")]
    X_train = np.vstack([df[meta_cols].values for df in train_snapshots])
    y_train = np.concatenate(train_labels)
    lgb_model.fit(X_train, y_train, feature_names=meta_cols)

    # 3. Layer 4 Predict Proba
    model_probas = {}
    for m in static_models:
        model_probas[m.name] = m.predict_proba(df_feat, df_hist, S_hist)
    model_probas[lgb_model.name] = lgb_model.predict_proba(df_feat, df_hist, S_hist)

    # 4. Layer 5: Prediction Intelligence & Calibration
    # Phân tách cửa sổ 90 ngày validation thành 2 nửa không gối đầu (Nested Split - D3 Point 1)
    # Nửa đầu (45 ngày): Dùng để fit các bộ hiệu chuẩn CalibratedClassifierCV
    # Nửa sau (45 ngày): Dùng để tuyển chọn bộ hiệu chuẩn tối ưu nhất và tính toán trọng số động Meta Fusion
    
    val_snapshots = []
    val_labels = []
    
    # 4.1. Thu thập dữ liệu toàn bộ cửa sổ validation
    for t_idx in range(history_end_idx - 90, history_end_idx):
        t_df_hist, t_S_hist = loader.slice_history(t_idx)
        t_date = df_full.iloc[t_idx]['date'].to_pydatetime()
        t_df_feat = feature_store.build(evidence_builder.build_all(t_df_hist, t_S_hist, t_date, save=True), t_date.strftime('%Y-%m-%d'), S=t_S_hist)
        val_snapshots.append(t_df_feat)
        
        t_row = df_full.iloc[t_idx]
        t_actual = set(t_row[loader.prize_cols()].dropna().values.astype(int).tolist())
        t_y = np.zeros(100)
        for num in t_actual:
            t_y[num] = 1
        val_labels.append(t_y)
        
    X_val = np.vstack([df[meta_cols].values for df in val_snapshots])
    X_val_df = pd.DataFrame(X_val, columns=meta_cols)
    y_val = np.concatenate(val_labels)
    
    # 4.2. Huấn luyện các bộ hiệu chuẩn trên nửa đầu (45 ngày đầu, tương ứng 4500 mẫu)
    # Cắt lát dữ liệu nửa đầu
    half_samples = 45 * 100
    X_cal_fit = X_val_df.iloc[:half_samples]
    y_cal_fit = y_val[:half_samples]
    
    # Cắt lát dữ liệu nửa sau để tuyển chọn hiệu chuẩn và train Meta Fusion weights
    X_cal_sel = X_val_df.iloc[half_samples:]
    y_cal_sel = y_val[half_samples:]
    
    # Sigmoid Calibration Fit & Composite Score (D3 Point B)
    cal_sig = CalibratedClassifierCV(estimator=FrozenEstimator(lgb_model._model), method="sigmoid")
    cal_sig.fit(X_cal_fit, y_cal_fit)
    p_sig_sel = cal_sig.predict_proba(X_cal_sel)[:, 1]
    brier_sig = float(np.mean((p_sig_sel - y_cal_sel) ** 2))
    p_sig_clipped = np.clip(p_sig_sel, 1e-7, 1.0 - 1e-7)
    logloss_sig = float(-np.mean(y_cal_sel * np.log(p_sig_clipped) + (1.0 - y_cal_sel) * np.log(1.0 - p_sig_clipped)))
    ece_sig = compute_ece(p_sig_sel, y_cal_sel)
    score_sig = 0.5 * brier_sig + 0.3 * logloss_sig + 0.2 * ece_sig
    
    # Isotonic Calibration Fit & Composite Score
    cal_iso = CalibratedClassifierCV(estimator=FrozenEstimator(lgb_model._model), method="isotonic")
    cal_iso.fit(X_cal_fit, y_cal_fit)
    p_iso_sel = cal_iso.predict_proba(X_cal_sel)[:, 1]
    brier_iso = float(np.mean((p_iso_sel - y_cal_sel) ** 2))
    p_iso_clipped = np.clip(p_iso_sel, 1e-7, 1.0 - 1e-7)
    logloss_iso = float(-np.mean(y_cal_sel * np.log(p_iso_clipped) + (1.0 - y_cal_sel) * np.log(1.0 - p_iso_clipped)))
    ece_iso = compute_ece(p_iso_sel, y_cal_sel)
    score_iso = 0.5 * brier_iso + 0.3 * logloss_iso + 0.2 * ece_iso
    
    # Chọn winner dựa trên tiêu chí đa mục tiêu tổng hợp (D3 Point B)
    if score_sig <= score_iso:
        best_calibrator = cal_sig
        best_cal_method = "sigmoid"
        brier_winner = brier_sig
        brier_loser = brier_iso
    else:
        best_calibrator = cal_iso
        best_cal_method = "isotonic"
        brier_winner = brier_iso
        brier_loser = brier_sig
        
    # Tính toán xác suất hiệu chuẩn của LightGBM trên target day
    calibrated_lgb_probs = best_calibrator.predict_proba(df_feat[meta_cols])[:, 1]
    
    # 4.3. Huấn luyện lại trọng số dynamic weight trên xác suất ĐÃ HIỆU CHUẨN (D3 Point 2)
    eval_predictions_sel = {m.name: [] for m in static_models}
    eval_predictions_sel[lgb_model.name] = []
    
    # LGBM đã hiệu chuẩn dự báo cho 45 ngày selection
    calibrated_lgbm_sel_preds = best_calibrator.predict_proba(X_cal_sel)[:, 1].reshape(45, 100)
    eval_predictions_sel[lgb_model.name] = list(calibrated_lgbm_sel_preds)
    
    for i, t_idx in enumerate(range(history_end_idx - 45, history_end_idx)):
        t_df_hist, t_S_hist = loader.slice_history(t_idx)
        t_df_feat = val_snapshots[45 + i]
        for m in static_models:
            eval_predictions_sel[m.name].append(m.predict_proba(t_df_feat, t_df_hist, t_S_hist))
            
    eval_preds_matrix = {k: np.array(v) for k, v in eval_predictions_sel.items()}
    eval_labels_matrix = np.array(val_labels[45:])
    
    fusion = MetaFusion()
    new_weights = fusion.compute_dynamic_weights(eval_preds_matrix, eval_labels_matrix)
    
    # Áp dụng EMA smoothing có warm-up dựa trên số lượng mẫu thực tế (D3 Point 1)
    n_samples = len(log)
    alpha = min(0.9, n_samples / 50.0)
    
    if prev_weights and n_samples >= 5:
        smoothed_weights = {}
        for name in new_weights:
            old_w = prev_weights.get(name, 1.0 / len(new_weights))
            smoothed_weights[name] = alpha * old_w + (1.0 - alpha) * new_weights[name]
        
        # Chuẩn hóa tổng weights = 1.0
        total_w = sum(smoothed_weights.values())
        if total_w > 0:
            smoothed_weights = {k: v / total_w for k, v in smoothed_weights.items()}
        fusion._weights = smoothed_weights
    else:
        fusion._weights = new_weights
    
    # Đồng bộ LGB đã hiệu chuẩn vào đầu vào fusion hôm nay
    model_probas[lgb_model.name] = calibrated_lgb_probs
    meta_proba = fusion.fuse(model_probas)

    # 5. Layer 6: Decision Intelligence & Portfolio Optimizer
    decision_engine = DecisionEngine(
        min_probability=0.31,
        min_confidence=0.45,
        top_k=top_k,
        kelly_odds=PAYOUT_PER_HIT / COST_PER_BET,
        kelly_fraction=0.20,
        # Kelly chỉ phân bổ vốn; không được biến số đã đạt p/confidence thành SKIP.
        kelly_selects_bets=False,
    )

    # §7 Kelly transparency: đọc khai báo vốn từ evaluation_policy.json
    kelly_policy_path = root_dir / "predictions" / "evaluation_policy.json"
    kelly_capital_declared = False
    kelly_capital_points = 0.0
    kelly_point_value_vnd = 0.0
    if kelly_policy_path.exists():
        with open(kelly_policy_path, "r", encoding="utf-8") as kf:
            kpol = json.load(kf)
            kp = kpol.get("kelly_policy", {})
            kelly_capital_declared = kp.get("declared", False)
            kelly_capital_points = kp.get("starting_capital_points", 0.0) or 0.0
            kelly_point_value_vnd = kp.get("point_value_vnd", 0.0) or 0.0

    day_decision = decision_engine.decide(
        date=target_date_str,
        meta_proba=meta_proba,
        model_probas=model_probas,
        S_history=S_hist,
        df_features=df_feat,
        feature_importance_df=lgb_model.feature_importance()
    )

    # §7: Truyền Kelly capital info vào DayDecision
    day_decision.kelly_capital_declared = kelly_capital_declared
    day_decision.kelly_capital_points = kelly_capital_points
    day_decision.kelly_point_value_vnd = kelly_point_value_vnd

    decision_dict = day_decision.to_dict()

    # Shadow challenger: dự báo E[số nháy], chỉ ghi log nghiên cứu và không đổi bets production.
    count_challenger = EWMAHitCountChallenger(top_k=top_k)
    count_forecast = count_challenger.predict(df_hist, loader.prize_cols())
    decision_dict["count_challenger"] = count_challenger.to_shadow_dict(count_forecast)

    # ── Edge Gate check (§6): Block BET khi gate FAIL ──────────────────
    edge_gate = EdgeGate()
    gate_result = edge_gate.check()
    decision_dict["edge_gate"] = gate_result

    if not gate_result["pass"]:
        # Force tất cả BET → PAPER_TRADE
        for bet in decision_dict["bets"]:
            bet["decision"] = "PAPER_TRADE"
            bet["original_decision"] = "BET"
        decision_dict["decision_summary"]["edge_gate_blocked"] = True
        decision_dict["decision_summary"]["edge_gate_status"] = gate_result["status"]
        decision_dict["decision_summary"]["edge_gate_action"] = gate_result["action"]
    else:
        decision_dict["decision_summary"]["edge_gate_blocked"] = False
        decision_dict["decision_summary"]["edge_gate_status"] = "PASS"
        decision_dict["decision_summary"]["edge_gate_action"] = "BET"
    
    # Save raw probabilities for authentic scientific audit & schema version tagging
    decision_dict["pipeline_metadata"]["schema_version"] = "2.0"
    decision_dict["pipeline_metadata"]["prediction_engine_version"] = "1.3.2"
    decision_dict["pipeline_metadata"]["belief_engine_version"] = "1.3.1"
    decision_dict["ensemble_proba"] = [float(p) for p in meta_proba]
    decision_dict["model_probas"] = {
        name: [float(p) for p in probs] for name, probs in model_probas.items()
    }
    
    # Save both uncalibrated and calibrated LightGBM probabilities separately (D3 Point 1)
    # Lấy xác suất raw từ mô hình LightGBM thô ban đầu để đối chiếu
    raw_lgb_probs = lgb_model.predict_proba(df_feat, df_hist, S_hist)
    decision_dict["raw_lgb_proba"] = [float(p) for p in raw_lgb_probs]
    decision_dict["calibrated_lgb_proba"] = [float(p) for p in calibrated_lgb_probs]
    decision_dict["pipeline_metadata"]["best_calibration_method"] = best_cal_method
    decision_dict["pipeline_metadata"]["validation_brier_winner"] = round(brier_winner, 6)
    decision_dict["pipeline_metadata"]["validation_brier_loser"] = round(brier_loser, 6)
    decision_dict["pipeline_metadata"]["dynamic_weights"] = fusion._weights

    # Thêm Experiment Manifest làm mốc kiểm chứng cấu hình (D3 Point 3 & 5)
    feature_hash = get_file_hash(root_dir / "src" / "feature_engine.py")
    belief_hash = get_file_hash(root_dir / "predictions" / "belief_registry.json")
    dataset_hash = get_file_hash(DATA_CSV)
    
    decision_dict["pipeline_metadata"]["experiment"] = {
        "id": "XPIS-EVM-1",
        "git_commit": get_git_commit(),
        "feature_catalog_hash": feature_hash,
        "belief_registry_hash": belief_hash,
        "config_hash": get_file_hash(root_dir / "pyproject.toml"),
        "dataset_hash": dataset_hash,
        "training_window": f"{train_start_idx}-{train_end_idx}",
        "validation_window": f"{history_end_idx - 90}-{history_end_idx}",
        "prediction_date": target_date_str,
        "feature_count": len(meta_cols),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "random_seed": int(decision_dict["pipeline_metadata"].get("random_seed", 42))
        }
    }

    return decision_dict


def run_predict(target_date: date, top_k: int = 2) -> dict:
    """Compatibility wrapper cho CLI production."""
    return run_shared_prediction_pipeline(target_date, top_k=top_k)


def print_prediction(entry: dict) -> None:
    meta = entry["pipeline_metadata"]
    bets = entry["bets"]
    div_score = entry["diversification_score"]
    stability = entry["rank_stability_index"]

    # Edge Gate status
    gate = entry.get("edge_gate", {})
    gate_blocked = entry.get("decision_summary", {}).get("edge_gate_blocked", False)
    gate_status = gate.get("status", "UNKNOWN")

    # §7 Kelly transparency
    kelly_trans = entry.get("kelly_transparency", {})
    kelly_declared = kelly_trans.get("capital_declared", False)

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print(f"║  🎯  DỰ ĐOÁN XSMB — {meta['date']}                    ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Mã Dự án : XPIS v1.2 APPROVED                       ║")
    print(f"║  Run ID   : {meta['run_id']:<41}║")
    print(f"║  Độ đa dạng (Div): {div_score:.4f} | Spearman PSI: {stability:.4f} ║")
    if gate_blocked:
        print(f"║  ⛔ EDGE GATE: {gate_status:<9} → PAPER_TRADE MODE         ║")
    else:
        print(f"║  ✅ EDGE GATE: {gate_status:<9}                            ║")
    # §7: Hiển thị Kelly capital info
    if kelly_declared:
        cap_pts = kelly_trans.get("capital_points", 0)
        pt_val = kelly_trans.get("point_value_vnd", 0)
        print(f"║  💰 Vốn: {cap_pts:.0f} điểm × {pt_val:,.0f}đ/điểm = {cap_pts * pt_val:,.0f}đ  ║")
    else:
        print("║  💰 Vốn: CHƯA KHAI BÁO — allocation là điểm giả định    ║")
    print("╠══════════════════════════════════════════════════════╣")
    if len(bets) == 0:
        print("║  👉  HÀNH ĐỘNG: SKIP (Bộ lọc rủi ro chặn toàn bộ)     ║")
        summary = entry.get("decision_summary", {})
        if summary:
            print(
                "║  Chẩn đoán: "
                f"P={summary.get('probability_qualified', 0)} | "
                f"C={summary.get('confidence_qualified', 0)} | "
                f"Eligible={summary.get('candidates_before_top_k', 0)} | "
                f"TopK={summary.get('candidates_after_probability_and_confidence', 0)} | "
                f"Kelly+={summary.get('candidates_with_positive_kelly', 0)}"
            )
    elif gate_blocked:
        print("║  📋  HÀNH ĐỘNG: PAPER_TRADE (Edge Gate chặn BET)       ║")
        print("║  ⚠️  Dự báo dưới đây chỉ để theo dõi, KHÔNG đặt tiền   ║")
        for b in bets[:10]:
            num_str = f"{b['number']:02d}"
            prob_str = f"{b['probability'] * 100:.1f}%"
            conf_str = f"{b['confidence']:.2f}"
            alloc_str = f"{b['allocation'] * 100:.1f}%"
            state_delay = b["explanation"]["feature_states"].get("delay", "0.0σ")
            approx_contrib = b["explanation"]["approximate_contributions"].get("delay", "+0.0000")
            print(f"║  📋 Số {num_str} | P: {prob_str} | Conf: {conf_str} | Vốn Kelly: {alloc_str:<5} ║")
            print(f"║    [Audit] Delay: {state_delay:<6} | Approx Contrib: {approx_contrib:<8}      ║")
    else:
        print("║  🎯  HÀNH ĐỘNG: BET (Danh mục đề xuất tối ưu)        ║")
        for b in bets:
            num_str = f"{b['number']:02d}"
            prob_str = f"{b['probability'] * 100:.1f}%"
            conf_str = f"{b['confidence']:.2f}"
            alloc_frac = f"{b['allocation'] * 100:.1f}%"

            # §7: Tách điểm vốn và tiền quy đổi
            if kelly_declared and b.get("allocation_points") is not None:
                alloc_display = f"{b['allocation_points']:.1f}đ ({b['allocation_vnd']:,.0f}đ)"
            else:
                alloc_display = f"{alloc_frac} (giả định)"

            # Show explainability
            state_delay = b["explanation"]["feature_states"].get("delay", "0.0σ")
            approx_contrib = b["explanation"]["approximate_contributions"].get("delay", "+0.0000")

            print(f"║  - Số {num_str} | P: {prob_str} | Conf: {conf_str} | Kelly: {alloc_display:<16} ║")
            print(f"║    [Audit] Delay: {state_delay:<6} | Approx Contrib: {approx_contrib:<8}      ║")
    print("╚══════════════════════════════════════════════════════╝")
    challenger = entry.get("count_challenger", {})
    shadow_picks = challenger.get("picks", [])
    if shadow_picks:
        shadow_text = ", ".join(
            f"{pick['number']:02d}(μ={pick['expected_count']:.3f},LCB={pick['lower_bound_95']:.3f})"
            for pick in shadow_picks
        )
        print(f"  🧪 SHADOW COUNT — PAPER-TRADE ONLY: {shadow_text}")
    else:
        print("  🧪 SHADOW COUNT — SKIP (không LCB nào vượt hòa vốn)")
    # §7: Disclaimer khi chưa khai báo vốn
    if not kelly_declared:
        print("  ⚠️  CHƯA KHAI BÁO VỐN — allocation Kelly là điểm giả định, không phải tiền thật")
    print()


def main():
    parser = argparse.ArgumentParser(description="Dự đoán XSMB hằng ngày XPIS v1.2 APPROVED")
    parser.add_argument("--date", type=str, default=None, help="Ngày dự đoán (YYYY-MM-DD), mặc định là hôm nay")
    parser.add_argument("--top-k", type=int, default=2, help="Số lượng số tối đa chọn (mặc định 2)")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in, không lưu log")
    parser.add_argument("--no-fetch", action="store_true", help="Bỏ qua bước fetch dữ liệu mới")
    args = parser.parse_args()

    # Định ngày dự toán
    if args.date:
        target_date = date.fromisoformat(args.date)
    else:
        now = datetime.now(TZ)
        target_date = now.date()

    if not args.no_fetch:
        fetch_today_data()

    # Kiểm tra log trùng lặp hỗ trợ cả cấu trúc v1.0 và v1.2
    log = load_log()
    existing = []
    for e in log:
        d_val = e["pipeline_metadata"]["date"] if "pipeline_metadata" in e else e.get("date")
        if d_val == str(target_date):
            existing.append(e)
            
    if existing and not args.dry_run:
        print(f"\n⚠️  Đã có dự đoán cho ngày {target_date} trong log.")
        if "pipeline_metadata" in existing[0]:
            print_prediction(existing[0])
        else:
            # Fallback in cũ
            print(f"Bản ghi v1.0: {existing[0]}")
        return

    print(f"\n🔮 Đang tính toán dự báo định lượng cho ngày {target_date}...")
    try:
        entry = run_predict(target_date, args.top_k)
        print_prediction(entry)

        if not args.dry_run:
            cleaned_log = [e for e in log if (e["pipeline_metadata"]["date"] if "pipeline_metadata" in e else e.get("date")) != str(target_date)]
            cleaned_log.append(entry)
            save_log(cleaned_log)
            print(f"✅ Đã lưu dự đoán vào log tại {PRED_LOG}")
    except Exception as e:
        print(f"❌ Lỗi thực thi dự báo: {e}")
        import traceback
        traceback.print_exc()
        # Báo lỗi cho auto_runner/GitHub Actions; không được ghi nhận nhầm là SKIP.
        raise SystemExit(1)


if __name__ == "__main__":
    main()
