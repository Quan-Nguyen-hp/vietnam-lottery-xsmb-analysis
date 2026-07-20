"""
XPIS v1.1 — Walk-Forward Backtesting Engine
Kiểm thử toàn diện kiến trúc nâng cấp v1.1.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator

# Thêm src vào sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "src"))

from src.data.loader import DataLoader
from src.evidence.builder import EvidenceBuilder
from src.evidence.store import EvidenceStore
from src.features.feature_store import FeatureStore
from src.probability import get_all_models
from src.probability.lgb_model import LightGBMProbabilityModel
from src.meta.fusion import MetaFusion
from src.meta.component_calibration import ComponentCalibrationManager
from src.meta.constrained_stacking import ConstrainedStacking
from src.decision.engine import DecisionEngine
from src.evaluation.metrics import EvaluationMetrics
from daily_predict import run_shared_prediction_pipeline

# Cấu hình tài chính
COST_PER_BET = 27.0
PAYOUT_PER_HIT = 99.0
RETRAIN_INTERVAL = 30
TRAIN_WINDOW_DAYS = 365
CALIBRATION_WINDOW_DAYS = 90
CALIBRATION_FIT_DAYS = 45
BOOTSTRAP_RESAMPLES = 20_000
PERMUTATION_RESAMPLES = 5_000


def _calibration_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Cùng hàm mục tiêu với production: Brier, log loss và ECE."""
    brier = float(np.mean((probs - labels) ** 2))
    clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
    logloss = float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (probs >= lower) & (probs < upper)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(labels[mask])) - float(np.mean(probs[mask])))
    return 0.5 * brier + 0.3 * logloss + 0.2 * ece


def _bootstrap_roi_interval(results: pd.DataFrame) -> tuple[float, float, float]:
    """Khoảng tin cậy bootstrap theo ngày, giữ nguyên các ngày SKIP trong mẫu."""
    pnl = results["daily_pnl"].to_numpy(dtype=float)
    cost = results["n_bets"].to_numpy(dtype=float) * COST_PER_BET
    if len(pnl) == 0 or cost.sum() <= 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(pnl), size=(BOOTSTRAP_RESAMPLES, len(pnl)))
    sampled_cost = cost[indices].sum(axis=1)
    sampled_roi = np.divide(
        pnl[indices].sum(axis=1),
        sampled_cost,
        out=np.zeros(BOOTSTRAP_RESAMPLES),
        where=sampled_cost > 0,
    )
    lower, upper = np.quantile(sampled_roi, [0.025, 0.975])
    probability_positive = float(np.mean(sampled_roi > 0.0))
    return float(lower), float(upper), probability_positive


def _paired_bootstrap_roi_delta(
    challenger: pd.DataFrame,
    champion: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """Bootstrap theo ngày cho chênh lệch ROI khi hai policy có cùng exposure."""
    challenger_cost = challenger["n_bets"].to_numpy(dtype=float) * COST_PER_BET
    champion_cost = champion["n_bets"].to_numpy(dtype=float) * COST_PER_BET
    if not np.array_equal(challenger_cost, champion_cost):
        raise ValueError("Paired ROI delta yêu cầu exposure từng ngày giống hệt nhau")
    if challenger_cost.sum() <= 0:
        return 0.0, 0.0, 0.0, 0.0

    pnl_delta = (
        challenger["daily_pnl"].to_numpy(dtype=float)
        - champion["daily_pnl"].to_numpy(dtype=float)
    )
    observed = float(pnl_delta.sum() / challenger_cost.sum())
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(challenger), size=(BOOTSTRAP_RESAMPLES, len(challenger)))
    sampled_cost = challenger_cost[indices].sum(axis=1)
    sampled_delta = np.divide(
        pnl_delta[indices].sum(axis=1),
        sampled_cost,
        out=np.zeros(BOOTSTRAP_RESAMPLES),
        where=sampled_cost > 0,
    )
    lower, upper = np.quantile(sampled_delta, [0.025, 0.975])
    return observed, float(lower), float(upper), float(np.mean(sampled_delta > 0.0))


def _permutation_test(picks_by_day: list[list[int]], actuals_by_day: list[list[int]]) -> dict[str, float]:
    """Kiểm định nhãn hoán vị: liệu số XPIS chọn có khớp kết quả hơn ngẫu nhiên không?"""
    n_days = len(picks_by_day)
    picks = np.zeros((n_days, 100), dtype=np.int8)
    actual_counts = np.zeros((n_days, 100), dtype=np.int8)
    for day, numbers in enumerate(picks_by_day):
        picks[day, numbers] = 1
    for day, numbers in enumerate(actuals_by_day):
        np.add.at(actual_counts[day], numbers, 1)

    observed_hits = int(np.sum(picks * actual_counts))
    total_cost = int(np.sum(picks) * COST_PER_BET)
    observed_pnl = observed_hits * PAYOUT_PER_HIT - total_cost
    rng = np.random.default_rng(42)
    permuted_pnl = np.empty(PERMUTATION_RESAMPLES, dtype=float)
    for i in range(PERMUTATION_RESAMPLES):
        permuted_hits = int(np.sum(picks * actual_counts[rng.permutation(n_days)]))
        permuted_pnl[i] = permuted_hits * PAYOUT_PER_HIT - total_cost

    return {
        "observed_pnl": float(observed_pnl),
        "mean_pnl": float(np.mean(permuted_pnl)),
        "lower_pnl": float(np.quantile(permuted_pnl, 0.025)),
        "upper_pnl": float(np.quantile(permuted_pnl, 0.975)),
        "p_value": float((np.sum(permuted_pnl >= observed_pnl) + 1) / (PERMUTATION_RESAMPLES + 1)),
    }


def run_exact_production_backtest(
    n_test_days: int = 30,
    top_k: int = 2,
    report_name: str | None = None,
) -> pd.DataFrame:
    """Walk-forward gọi trực tiếp pipeline production, không sao chép logic model.

    Production recalibrate mỗi ngày, vì vậy chế độ exact chậm hơn backtest nhanh.
    Mặc định 30 ngày để smoke-test; tăng ``n_test_days`` khi chạy kiểm định đầy đủ.
    """
    loader = DataLoader().load()
    if n_test_days <= 0 or n_test_days >= loader.total_days:
        raise ValueError("n_test_days phải lớn hơn 0 và nhỏ hơn tổng số ngày dữ liệu")

    prior_entries: list[dict] = []
    records = []
    prize_cols = loader.prize_cols()
    for idx in range(loader.total_days - n_test_days, loader.total_days):
        row = loader.df.iloc[idx]
        target_date = pd.Timestamp(row["date"]).date()
        entry = run_shared_prediction_pipeline(target_date, top_k=top_k, prior_entries=prior_entries)
        picks = [bet["number"] for bet in entry["bets"]]
        actual = row[prize_cols].dropna().values.astype(int).tolist()
        hits = sum(actual.count(number) for number in picks)
        records.append({
            "date": str(target_date),
            "n_bets": len(picks),
            "n_hits": hits,
            "hit": hits > 0,
            "daily_pnl": hits * PAYOUT_PER_HIT - len(picks) * COST_PER_BET,
            "diversification_score": entry["diversification_score"],
        })
        # Không ghi log production: chỉ chuyển lịch sử xuất hiện đến thời điểm t cho EMA.
        prior_entries.append(entry)

    results = pd.DataFrame(records)
    if report_name:
        metrics = EvaluationMetrics(odds=PAYOUT_PER_HIT / COST_PER_BET, cost_per_bet=COST_PER_BET).compute_full(results)
        roi_lower, roi_upper, probability_positive = _bootstrap_roi_interval(results)
        report_path = root_dir / "backtests" / "results" / report_name
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as report:
            report.write("# Exact Production Pipeline Backtest\n\n")
            report.write(
                f"- **Kỳ kiểm thử**: {n_test_days} ngày ({results.iloc[0]['date']} đến {results.iloc[-1]['date']})\n"
            )
            report.write("- **Pipeline**: gọi trực tiếp `daily_predict.run_shared_prediction_pipeline()` mỗi ngày\n")
            report.write(f"- **Top K**: {top_k}\n\n")
            report.write("| Chỉ số | Giá trị |\n|---|---:|\n")
            report.write(f"| Số lượt cược | {metrics['total_bets']} |\n")
            report.write(f"| Số nháy trúng | {metrics['total_hits']} |\n")
            report.write(f"| ROI | {metrics['roi']:+.2%} |\n")
            report.write(f"| ROI bootstrap 95% | [{roi_lower:+.2%}, {roi_upper:+.2%}] |\n")
            report.write(f"| Xác suất ROI dương (bootstrap) | {probability_positive:.1%} |\n")
            report.write(
                "\n> Đây là exact-mode: calibration, fusion, EMA và quyết định dùng đúng code production; "
                "không ghi prediction log production.\n"
            )
    return results

def run_xpis_backtest(
    n_test_days: int = 180,
    min_prob: float = 0.31,
    min_conf: float = 0.45,
    top_k: int = 4,
    report_name: str = "xpis_backtest_report.md",
    kelly_selects_bets: bool = True,
    apply_diversification: bool = True,
    calibrate_all_models: bool = False,
    constrained_stacking: bool = False,
    uniform_fusion: bool = False,
    gated_uniform_ranking: bool = False,
):
    print("=== Khởi chạy Backtest Walk-Forward XPIS v1.1 ===")
    
    loader = DataLoader().load()
    total_days = loader.total_days
    start_idx = total_days - n_test_days
    
    evidence_store = EvidenceStore()
    evidence_builder = EvidenceBuilder(evidence_store)
    feature_store = FeatureStore(S_history=loader.S, evidence_store=evidence_store)
    
    # Layer 4 Models
    static_models = [m for m in get_all_models() if m.name != "lightgbm_classifier"]
    lgb_model = LightGBMProbabilityModel()
    
    # Layer 5 & 6 & 7
    fusion = MetaFusion()
    decision_engine = DecisionEngine(
        min_probability=min_prob,
        min_confidence=min_conf,
        top_k=top_k,
        kelly_odds=PAYOUT_PER_HIT / COST_PER_BET,
        kelly_fraction=0.20,
        min_diversification=0.85,
        kelly_selects_bets=kelly_selects_bets,
        apply_diversification=apply_diversification,
    )
    evaluator = EvaluationMetrics(odds=PAYOUT_PER_HIT / COST_PER_BET, cost_per_bet=COST_PER_BET)
    
    flat_bets_log = []
    champion_flat_bets_log = []
    random_baseline_log = []
    frequency_baseline_log = []
    random_baseline_rng = np.random.default_rng(42)
    model_benchmark_logs = {model.name: [] for model in static_models}
    model_benchmark_logs[lgb_model.name] = []
    kelly_bankroll = 10000.0
    kelly_bankroll_history = [kelly_bankroll]
    daily_results = []
    selected_numbers_history: list[list[int]] = []
    actual_lotos_history: list[list[int]] = []
    rank_topk_log = []
    uniform_rank_topk_log = []
    meta_probability_history = []
    uniform_probability_history = []
    binary_label_history = []
    calibration_methods = []
    best_calibrator = None
    component_calibrators = None
    component_calibration_methods: dict[str, str] = {}
    stacking_model = None
    enabled_fusion_experiments = sum(
        (calibrate_all_models, constrained_stacking, uniform_fusion, gated_uniform_ranking)
    )
    if enabled_fusion_experiments > 1:
        raise ValueError(
            "Chỉ được bật một thử nghiệm fusion/ranking tại một thời điểm"
        )
    
    # Cần lưu lịch sử dự báo để tính toán trọng số động
    history_model_preds = {m.name: [] for m in static_models}
    history_model_preds[lgb_model.name] = []
    history_labels = []
    
    t_start = time.time()
    
    for step, idx in enumerate(range(start_idx, total_days)):
        current_row = loader.df.iloc[idx]
        current_date = pd.to_datetime(current_row['date'])
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Nhãn
        prize_cols = loader.prize_cols()
        actual_lotos = current_row[prize_cols].dropna().values.astype(int).tolist()
        actual_set = set(actual_lotos)
        y = np.zeros(100)
        for num in actual_set:
            y[num] = 1
            
        # 1. Huấn luyện rolling, hiệu chuẩn và tính lại trọng số động.
        # Giống cấu trúc production; chu kỳ 30 ngày chỉ là tối ưu chi phí backtest.
        if step % RETRAIN_INTERVAL == 0:
            print(f"🔄 Ngày {step}/{n_test_days} ({date_str}): Huấn luyện lại LGBM và cập nhật trọng số Fusion...")
            
            # Huấn luyện mô hình LGBM ở Layer 4
            train_end_idx = idx - CALIBRATION_WINDOW_DAYS
            train_start_idx = max(50, train_end_idx - TRAIN_WINDOW_DAYS)
            train_snapshots = []
            train_labels = []
            
            # Tuyệt đối không đưa 90 ngày calibration/selection vào tập train.
            for t_idx in range(train_start_idx, train_end_idx):
                t_df_hist, t_S_hist = loader.slice_history(t_idx)
                t_row = loader.df.iloc[t_idx]
                t_date = t_row['date'].to_pydatetime()
                t_date_str = t_date.strftime('%Y-%m-%d')
                
                t_df_ev = evidence_builder.build_all(t_df_hist, t_S_hist, t_date, save=True)
                t_df_feat = feature_store.build(t_df_ev, t_date_str, S=t_S_hist)
                train_snapshots.append(t_df_feat)
                
                t_actual = set(t_row[prize_cols].dropna().values.astype(int).tolist())
                t_y = np.zeros(100)
                for num in t_actual:
                    t_y[num] = 1
                train_labels.append(t_y)
                
            # Fit LGBM
            # Khởi tạo ma trận X_train từ snapshots
            meta_cols = [c for c in train_snapshots[0].columns if c not in ("number", "date")]
            X_train = np.vstack([df[meta_cols].values for df in train_snapshots])
            y_train = np.concatenate(train_labels)
            
            lgb_model.fit(X_train, y_train, feature_names=meta_cols)
            
            # Validation 90 ngày không chồng lấn: 45 ngày fit calibrator, 45 ngày chọn calibrator/fusion.
            val_snapshots = []
            val_labels = []
            for t_idx in range(train_end_idx, idx):
                t_df_hist, t_S_hist = loader.slice_history(t_idx)
                t_row = loader.df.iloc[t_idx]
                t_date = t_row['date'].to_pydatetime()
                t_date_str = t_date.strftime('%Y-%m-%d')
                t_df_ev = evidence_builder.build_all(t_df_hist, t_S_hist, t_date, save=True)
                t_df_feat = feature_store.build(t_df_ev, t_date_str, S=t_S_hist)
                val_snapshots.append(t_df_feat)
                t_actual = set(t_row[prize_cols].dropna().values.astype(int).tolist())
                y_val = np.zeros(100)
                for num in t_actual:
                    y_val[num] = 1
                val_labels.append(y_val)

            X_val = np.vstack([df[meta_cols].values for df in val_snapshots])
            X_val_df = pd.DataFrame(X_val, columns=meta_cols)
            y_val = np.concatenate(val_labels)
            split = CALIBRATION_FIT_DAYS * 100
            if calibrate_all_models:
                raw_val_predictions = {m.name: [] for m in static_models}
                raw_val_predictions[lgb_model.name] = []
                for i, t_idx in enumerate(range(train_end_idx, idx)):
                    t_df_hist, t_S_hist = loader.slice_history(t_idx)
                    t_df_feat = val_snapshots[i]
                    for m in static_models:
                        raw_val_predictions[m.name].append(m.predict_proba(t_df_feat, t_df_hist, t_S_hist))
                    raw_val_predictions[lgb_model.name].append(
                        lgb_model.predict_proba(t_df_feat, t_df_hist, t_S_hist)
                    )
                raw_fit = {
                    name: np.asarray(values[:CALIBRATION_FIT_DAYS])
                    for name, values in raw_val_predictions.items()
                }
                raw_selection = {
                    name: np.asarray(values[CALIBRATION_FIT_DAYS:])
                    for name, values in raw_val_predictions.items()
                }
                component_calibrators = ComponentCalibrationManager().fit(
                    raw_fit,
                    np.asarray(val_labels[:CALIBRATION_FIT_DAYS]),
                    raw_selection,
                    np.asarray(val_labels[CALIBRATION_FIT_DAYS:]),
                )
                component_calibration_methods = component_calibrators.methods.copy()
                calibration_methods.append("all_components")
                eval_predictions = component_calibrators.calibrate_dict(raw_selection)
            else:
                cal_sig = CalibratedClassifierCV(estimator=FrozenEstimator(lgb_model._model), method="sigmoid")
                cal_iso = CalibratedClassifierCV(estimator=FrozenEstimator(lgb_model._model), method="isotonic")
                cal_sig.fit(X_val_df.iloc[:split], y_val[:split])
                cal_iso.fit(X_val_df.iloc[:split], y_val[:split])
                p_sig = cal_sig.predict_proba(X_val_df.iloc[split:])[:, 1]
                p_iso = cal_iso.predict_proba(X_val_df.iloc[split:])[:, 1]
                if _calibration_score(p_sig, y_val[split:]) <= _calibration_score(p_iso, y_val[split:]):
                    best_calibrator = cal_sig
                    calibration_methods.append("sigmoid")
                else:
                    best_calibrator = cal_iso
                    calibration_methods.append("isotonic")

                eval_predictions = {m.name: [] for m in static_models}
                eval_predictions[lgb_model.name] = list(
                    best_calibrator.predict_proba(X_val_df.iloc[split:])[:, 1].reshape(CALIBRATION_FIT_DAYS, 100)
                )
                for i, t_idx in enumerate(range(train_end_idx + CALIBRATION_FIT_DAYS, idx)):
                    t_df_hist, t_S_hist = loader.slice_history(t_idx)
                    t_df_feat = val_snapshots[CALIBRATION_FIT_DAYS + i]
                    for m in static_models:
                        eval_predictions[m.name].append(m.predict_proba(t_df_feat, t_df_hist, t_S_hist))

            if constrained_stacking:
                stack_fit_predictions = {m.name: [] for m in static_models}
                stack_fit_predictions[lgb_model.name] = []
                for i in range(CALIBRATION_FIT_DAYS):
                    t_idx = train_end_idx + i
                    t_df_hist, t_S_hist = loader.slice_history(t_idx)
                    t_df_feat = val_snapshots[i]
                    for m in static_models:
                        stack_fit_predictions[m.name].append(m.predict_proba(t_df_feat, t_df_hist, t_S_hist))
                    stack_fit_predictions[lgb_model.name].append(
                        best_calibrator.predict_proba(t_df_feat[meta_cols])[:, 1]
                    )
                stacking_model = ConstrainedStacking(regularization=0.10).fit(
                    {name: np.asarray(values) for name, values in stack_fit_predictions.items()},
                    np.asarray(val_labels[:CALIBRATION_FIT_DAYS]),
                )
                fusion._weights = stacking_model.weights.copy()
            elif not uniform_fusion:
                fusion.compute_dynamic_weights(
                    {name: np.array(predictions) for name, predictions in eval_predictions.items()},
                    np.array(val_labels[CALIBRATION_FIT_DAYS:]),
                )
            feature_store.clear_ram_cache()
            
        # 2. Dự báo ngày hiện tại
        df_hist, S_hist = loader.slice_history(idx)
        df_ev = evidence_builder.build_all(df_hist, S_hist, current_date.to_pydatetime(), save=True)
        df_feat = feature_store.build(df_ev, date_str, S=S_hist)
        
        # Lấy xác suất của tất cả 11 models
        raw_model_probas = {}
        for m in static_models:
            raw_model_probas[m.name] = m.predict_proba(df_feat, df_hist, S_hist)
        raw_model_probas[lgb_model.name] = lgb_model.predict_proba(df_feat, df_hist, S_hist)
        if calibrate_all_models:
            model_probas = component_calibrators.calibrate_dict(raw_model_probas)
        else:
            model_probas = dict(raw_model_probas)
            model_probas[lgb_model.name] = best_calibrator.predict_proba(df_feat[meta_cols])[:, 1]
        
        # 3. Fusion xác suất
        uniform_meta_proba = np.mean(np.stack(list(model_probas.values()), axis=0), axis=0)
        if uniform_fusion:
            meta_proba = uniform_meta_proba
        elif constrained_stacking and stacking_model is not None:
            meta_proba = stacking_model.fuse(model_probas)
        else:
            meta_proba = fusion.fuse(model_probas)
        meta_probability_history.append(meta_proba.copy())
        uniform_probability_history.append(uniform_meta_proba.copy())
        binary_label_history.append(y.copy())

        # Ranking-only diagnostic: đo chất lượng xếp hạng độc lập với threshold/kelly.
        rank_picks = np.argsort(meta_proba)[::-1][:top_k].astype(int).tolist()
        rank_hits = sum(actual_lotos.count(number) for number in rank_picks)
        rank_topk_log.append({
            "date": date_str,
            "n_bets": top_k,
            "n_hits": rank_hits,
            "hit": rank_hits > 0,
            "daily_pnl": rank_hits * PAYOUT_PER_HIT - top_k * COST_PER_BET,
        })
        uniform_rank_picks = np.argsort(uniform_meta_proba)[::-1][:top_k].astype(int).tolist()
        uniform_rank_hits = sum(actual_lotos.count(number) for number in uniform_rank_picks)
        uniform_rank_topk_log.append({
            "date": date_str,
            "n_bets": top_k,
            "n_hits": uniform_rank_hits,
            "hit": uniform_rank_hits > 0,
            "daily_pnl": uniform_rank_hits * PAYOUT_PER_HIT - top_k * COST_PER_BET,
        })
        
        # 4. Ra quyết định cược (Layer 6)
        day_decision = decision_engine.decide(
            date=date_str,
            meta_proba=meta_proba,
            model_probas=model_probas,
            S_history=S_hist,
            feature_version=feature_store.version,
            model_version=lgb_model.version,
            evidence_version="v1.0"
        )
        
        # 5. Đánh giá kết quả giao dịch
        bets = day_decision.bets
        n_bets = len(bets)
        champion_numbers = [int(b.number) for b in bets]
        evaluation_numbers = (
            np.argsort(uniform_meta_proba)[::-1][:n_bets].astype(int).tolist()
            if gated_uniform_ranking
            else champion_numbers
        )
        selected_numbers_history.append(evaluation_numbers)
        actual_lotos_history.append(actual_lotos)

        # Ablation: mỗi model chọn cùng số lượng số mà XPIS đã exposure trong ngày.
        for model_name, probabilities in model_probas.items():
            model_picks = np.argsort(probabilities)[::-1][:n_bets]
            model_hits = sum(actual_lotos.count(int(number)) for number in model_picks)
            model_benchmark_logs[model_name].append({
                "date": date_str,
                "hit": model_hits > 0,
                "n_bets": n_bets,
                "n_hits": model_hits,
                "daily_pnl": model_hits * PAYOUT_PER_HIT - n_bets * COST_PER_BET,
            })
        n_hits = sum(actual_lotos.count(number) for number in evaluation_numbers)
        pnl_flat = n_hits * PAYOUT_PER_HIT - n_bets * COST_PER_BET
        champion_hits = sum(actual_lotos.count(number) for number in champion_numbers)
        champion_flat_bets_log.append({
            "date": date_str,
            "hit": champion_hits > 0,
            "n_bets": n_bets,
            "n_hits": champion_hits,
            "daily_pnl": champion_hits * PAYOUT_PER_HIT - n_bets * COST_PER_BET,
        })

        # Baseline có cùng số cược từng ngày để so sánh công bằng với XPIS.
        random_picks = random_baseline_rng.choice(100, size=n_bets, replace=False) if n_bets else []
        frequency_picks = np.argsort(S_hist.sum(axis=0))[::-1][:n_bets] if n_bets else []
        random_hits = sum(actual_lotos.count(int(number)) for number in random_picks)
        frequency_hits = sum(actual_lotos.count(int(number)) for number in frequency_picks)
        random_baseline_log.append({
            "date": date_str,
            "hit": random_hits > 0,
            "n_bets": n_bets,
            "n_hits": random_hits,
            "daily_pnl": random_hits * PAYOUT_PER_HIT - n_bets * COST_PER_BET,
        })
        frequency_baseline_log.append({
            "date": date_str,
            "hit": frequency_hits > 0,
            "n_bets": n_bets,
            "n_hits": frequency_hits,
            "daily_pnl": frequency_hits * PAYOUT_PER_HIT - n_bets * COST_PER_BET,
        })
                
        # Đánh giá Kelly
        day_kelly_pnl = 0.0
        active_bets_info = []
        
        if n_bets > 0 and kelly_bankroll > 500.0:
            bet_allocations = {}
            for b in bets:
                # b.allocation là % vốn (ví dụ: 0.02 = 2% vốn)
                bet_amount = b.allocation * kelly_bankroll
                if bet_amount >= 5.0:
                    bet_allocations[b.number] = bet_amount
                    
            total_bet = sum(bet_allocations.values())
            if total_bet > 0.50 * kelly_bankroll:
                scale = (0.50 * kelly_bankroll) / total_bet
                bet_allocations = {n: a * scale for n, a in bet_allocations.items()}
                total_bet = sum(bet_allocations.values())
                
            winnings = 0.0
            for num, amt in bet_allocations.items():
                matches = actual_lotos.count(num)
                winnings += matches * amt * (PAYOUT_PER_HIT / COST_PER_BET)
                active_bets_info.append(f"{num:02d}({amt:.0f}đ)")
                
            day_kelly_pnl = winnings - total_bet
            kelly_bankroll += day_kelly_pnl
            
        kelly_bankroll_history.append(kelly_bankroll)
        
        flat_bets_log.append({
            'date': date_str,
            'hit': n_hits > 0,
            'n_bets': n_bets,
            'n_hits': n_hits,
            'daily_pnl': pnl_flat
        })
        
        daily_results.append({
            'date': date_str,
            'bets': [f"{number:02d}" for number in evaluation_numbers],
            'hits': n_hits,
            'pnl_flat': pnl_flat,
            'kelly_pnl': day_kelly_pnl,
            'kelly_bankroll': kelly_bankroll,
            'kelly_bets': ", ".join(active_bets_info) if active_bets_info else "SKIP",
            'div_score': day_decision.diversification_score,
            'actual': [f"{n:02d}" for n in sorted(actual_set)],
            'top_k_limited': day_decision.decision_summary['top_k_limited'],
        })
        
    # Kết thúc backtest
    elapsed = time.time() - t_start
    print(f"\n✅ Backtest XPIS v1.1 hoàn tất sau {elapsed:.1f} giây!")
    
    # 6. Ghi báo cáo kết quả
    df_flat = pd.DataFrame(flat_bets_log)
    df_champion_flat = pd.DataFrame(champion_flat_bets_log)
    df_random = pd.DataFrame(random_baseline_log)
    df_frequency = pd.DataFrame(frequency_baseline_log)
    metrics_summary = evaluator.compute_full(df_flat)
    roi_ci_lower, roi_ci_upper, roi_probability_positive = _bootstrap_roi_interval(df_flat)
    paired_roi_delta = _paired_bootstrap_roi_delta(df_flat, df_champion_flat)
    champion_metrics = evaluator.compute_full(df_champion_flat)
    champion_roi_lower, champion_roi_upper, _ = _bootstrap_roi_interval(df_champion_flat)
    random_metrics = evaluator.compute_full(df_random)
    random_roi_lower, random_roi_upper, _ = _bootstrap_roi_interval(df_random)
    frequency_metrics = evaluator.compute_full(df_frequency)
    frequency_roi_lower, frequency_roi_upper, _ = _bootstrap_roi_interval(df_frequency)
    df_rank = pd.DataFrame(rank_topk_log)
    rank_metrics = evaluator.compute_full(df_rank)
    rank_roi_lower, rank_roi_upper, rank_p_positive = _bootstrap_roi_interval(df_rank)
    rank_mean_hits = float(df_rank["n_hits"].mean()) if len(df_rank) else 0.0
    df_uniform_rank = pd.DataFrame(uniform_rank_topk_log)
    uniform_rank_metrics = evaluator.compute_full(df_uniform_rank)
    uniform_rank_lower, uniform_rank_upper, uniform_rank_p_positive = _bootstrap_roi_interval(df_uniform_rank)

    meta_probability_matrix = np.asarray(meta_probability_history)
    uniform_probability_matrix = np.asarray(uniform_probability_history)
    binary_label_matrix = np.asarray(binary_label_history)
    probability_quality = {
        "fusion": {
            "brier": evaluator.brier_score(meta_probability_matrix, binary_label_matrix),
            "ece": evaluator.ece_score(meta_probability_matrix.reshape(-1), binary_label_matrix.reshape(-1)),
            "precision_k": evaluator.precision_at_k(meta_probability_matrix, binary_label_matrix, k=top_k),
        },
        "uniform": {
            "brier": evaluator.brier_score(uniform_probability_matrix, binary_label_matrix),
            "ece": evaluator.ece_score(
                uniform_probability_matrix.reshape(-1),
                binary_label_matrix.reshape(-1),
            ),
            "precision_k": evaluator.precision_at_k(
                uniform_probability_matrix,
                binary_label_matrix,
                k=top_k,
            ),
        },
    }
    permutation = _permutation_test(selected_numbers_history, actual_lotos_history)
    model_leaderboard = []
    for model_name, rows in model_benchmark_logs.items():
        model_results = pd.DataFrame(rows)
        model_metrics = evaluator.compute_full(model_results)
        model_lower, model_upper, _ = _bootstrap_roi_interval(model_results)
        model_leaderboard.append({
            "name": model_name,
            "roi": model_metrics["roi"],
            "roi_lower": model_lower,
            "roi_upper": model_upper,
            "total_bets": model_metrics["total_bets"],
        })
    model_leaderboard.sort(key=lambda row: row["roi"], reverse=True)
    
    results_dir = root_dir / 'backtests' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / report_name
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Báo cáo XPIS v1.2 — Walk-forward (Calibrated LightGBM + Dynamic Fusion)\n\n")
        f.write(
            f"- **Kỳ kiểm thử**: {n_test_days} ngày "
            f"({daily_results[0]['date']} đến {daily_results[-1]['date']})\n"
        )
        f.write(f"- **Hệ thống**: XPIS v1.2 (11 Models + calibrated LightGBM + Dynamic Weighted Fusion + Portfolio Risk Manager)\n")
        f.write(f"- **Tham số**: Top K: {top_k} | Min Prob: {min_prob:.2f} | Min Conf: {min_conf:.2f} | Min Diversification: 0.85\n")
        f.write(f"- **Thời gian chạy**: {elapsed:.1f}s\n")
        f.write(
            f"- **Ablation**: Kelly chọn cược = {'Có' if kelly_selects_bets else 'Không'}; "
            f"diversification = {'Bật' if apply_diversification else 'Tắt'}; "
            f"calibration toàn bộ models = {'Bật' if calibrate_all_models else 'Tắt'}; "
            f"constrained stacking = {'Bật' if constrained_stacking else 'Tắt'}; "
            f"uniform fusion policy = {'Bật' if uniform_fusion else 'Tắt'}; "
            f"Meta gate + uniform ranking = {'Bật' if gated_uniform_ranking else 'Tắt'}\n\n"
        )
        f.write(
            f"> Phạm vi: calibration dùng cùng cấu trúc 90 ngày/45+45 ngày với production, "
            f"nhưng làm mới mỗi {RETRAIN_INTERVAL} ngày để giới hạn chi phí backtest.\n\n"
        )
        
        f.write("## 1. Kết quả Tổng Hợp\n\n")
        f.write("| Chỉ số | Cược Cố Định (Flat Bet 27k) | Cược Kelly (Đa dạng hóa rủi ro) |\n")
        f.write("|---|:---:|:---:|\n")
        
        flat_total_cost = df_flat['n_bets'].sum() * COST_PER_BET
        flat_total_pnl = df_flat['daily_pnl'].sum()
        flat_roi = (flat_total_pnl / flat_total_cost) * 100 if flat_total_cost > 0 else 0.0
        
        f.write(f"| Tổng vốn chi | {flat_total_cost*1000:,.0f}đ | Thay đổi theo ngày |\n")
        f.write(f"| Lợi nhuận ròng | **{flat_total_pnl*1000:+,.0f}đ** | **{(kelly_bankroll - 10000.0)*COST_PER_BET*1000:+,.0f}đ** |\n")
        f.write(f"| ROI tổng | **{flat_roi:+.2f}%** | **{((kelly_bankroll / 10000.0) - 1.0)*100:+.2f}%** (Tăng trưởng vốn) |\n")
        f.write(f"| ROI bootstrap 95% | **[{roi_ci_lower:+.2%}, {roi_ci_upper:+.2%}]** | — |\n")
        f.write(f"| Xác suất ROI dương (bootstrap) | **{roi_probability_positive:.1%}** | — |\n")
        f.write(f"| Win Rate ngày | **{metrics_summary['hit_rate']:.1%}** ({metrics_summary['total_hits']}/{n_test_days} ngày có trúng) | — |\n")
        f.write(f"| Tổng số lần cược | {df_flat['n_bets'].sum()} số | {sum(1 for r in daily_results if r['bets'])} ngày cược |\n")
        f.write(f"| Ngày bị Top K cắt bớt ứng viên | {sum(1 for r in daily_results if r['top_k_limited'])}/{n_test_days} | — |\n")
        f.write(f"| Vốn cuối kỳ | — | **{kelly_bankroll*COST_PER_BET*1000:,.0f}đ** ({kelly_bankroll:,.1f} điểm) |\n\n")
        f.write(f"- **Calibration được chọn**: sigmoid {calibration_methods.count('sigmoid')} lần; isotonic {calibration_methods.count('isotonic')} lần.\n\n")
        if calibrate_all_models:
            method_counts = {
                method: list(component_calibration_methods.values()).count(method)
                for method in ("identity", "platt", "isotonic")
            }
            f.write(
                "- **Component calibration ở lần retrain cuối**: "
                f"identity={method_counts['identity']}, platt={method_counts['platt']}, "
                f"isotonic={method_counts['isotonic']}.\n\n"
            )
        if constrained_stacking and stacking_model is not None:
            f.write("- **Constrained stacking weights ở lần retrain cuối**:\n")
            for name, weight in sorted(stacking_model.weights.items(), key=lambda item: item[1], reverse=True):
                f.write(f"  - {name}: {weight:.4%}\n")
            f.write(f"- **Stacking objective ở lần retrain cuối**: {stacking_model.objective_value:.6f}\n\n")
        edge_gate = "PASS" if roi_ci_lower > 0.0 else "FAIL"
        f.write(
            f"- **Statistical Edge Gate**: **{edge_gate}** — chỉ PASS khi cận dưới ROI bootstrap 95% > 0. "
            "Khi FAIL, kết quả chỉ phù hợp để theo dõi/paper-trade, không phải bằng chứng tăng vốn.\n\n"
        )
        if gated_uniform_ranking:
            f.write(
                "### So sánh ghép cặp với MetaFusion champion\n\n"
                "> MetaFusion quyết định ngày và số lượng cược; uniform fusion chỉ thay thứ tự chọn số. "
                "Hai policy có exposure từng ngày giống hệt nhau.\n\n"
            )
            f.write("| Cấu hình | Lượt cược | ROI | CI95 |\n|---|---:|---:|---:|\n")
            f.write(
                f"| Gated uniform ranking | {metrics_summary['total_bets']} | {metrics_summary['roi']:+.2%} | "
                f"[{roi_ci_lower:+.2%}, {roi_ci_upper:+.2%}] |\n"
            )
            f.write(
                f"| MetaFusion champion | {champion_metrics['total_bets']} | {champion_metrics['roi']:+.2%} | "
                f"[{champion_roi_lower:+.2%}, {champion_roi_upper:+.2%}] |\n\n"
            )
            f.write(
                f"- **ΔROI challenger − champion**: {paired_roi_delta[0]:+.2%}, "
                f"paired CI95 [{paired_roi_delta[1]:+.2%}, {paired_roi_delta[2]:+.2%}], "
                f"P(ΔROI>0)={paired_roi_delta[3]:.1%}.\n\n"
            )

        f.write("## 2. So sánh baseline cùng mức exposure\n\n")
        f.write("| Chiến lược | Lượt cược | ROI | ROI bootstrap 95% |\n")
        f.write("|---|---:|---:|---:|\n")
        f.write(f"| XPIS | {metrics_summary['total_bets']} | {metrics_summary['roi']:+.2%} | [{roi_ci_lower:+.2%}, {roi_ci_upper:+.2%}] |\n")
        f.write(f"| Ngẫu nhiên (seed 42) | {random_metrics['total_bets']} | {random_metrics['roi']:+.2%} | [{random_roi_lower:+.2%}, {random_roi_upper:+.2%}] |\n")
        f.write(f"| Tần suất lịch sử | {frequency_metrics['total_bets']} | {frequency_metrics['roi']:+.2%} | [{frequency_roi_lower:+.2%}, {frequency_roi_upper:+.2%}] |\n")
        f.write("| Không cược | 0 | 0.00% | [0.00%, 0.00%] |\n\n")

        f.write("## 2b. Ranking-only diagnostic (không threshold, không Kelly)\n\n")
        f.write("> Mỗi ngày lấy đúng Top-K xác suất cao nhất để tách chất lượng xếp hạng khỏi cổng BET/SKIP.\n\n")
        f.write("| Chỉ số | Giá trị |\n|---|---:|\n")
        f.write(f"| Bets giả lập | {rank_metrics['total_bets']} |\n")
        f.write(f"| Hits | {rank_metrics['total_hits']} |\n")
        f.write(f"| Hits trung bình/ngày | {rank_mean_hits:.4f} |\n")
        f.write(f"| ROI ranking-only | {rank_metrics['roi']:+.2%} |\n")
        f.write(f"| Bootstrap CI95 | [{rank_roi_lower:+.2%}, {rank_roi_upper:+.2%}] |\n")
        f.write(f"| P(ROI>0) | {rank_p_positive:.1%} |\n\n")

        f.write("## 2c. So sánh Fusion với Uniform Fusion\n\n")
        f.write("| Cấu hình | Brier ↓ | ECE ↓ | Precision@K ↑ | Top-K hits | Ranking ROI | CI95 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        f.write(
            f"| {'Uniform fusion policy' if uniform_fusion else ('Constrained stacking' if constrained_stacking else 'MetaFusion')} | "
            f"{probability_quality['fusion']['brier']:.6f} | "
            f"{probability_quality['fusion']['ece']:.6f} | "
            f"{probability_quality['fusion']['precision_k']:.4f} | "
            f"{rank_metrics['total_hits']} | {rank_metrics['roi']:+.2%} | "
            f"[{rank_roi_lower:+.2%}, {rank_roi_upper:+.2%}] |\n"
        )
        f.write(
            f"| Uniform fusion | {probability_quality['uniform']['brier']:.6f} | "
            f"{probability_quality['uniform']['ece']:.6f} | "
            f"{probability_quality['uniform']['precision_k']:.4f} | "
            f"{uniform_rank_metrics['total_hits']} | {uniform_rank_metrics['roi']:+.2%} | "
            f"[{uniform_rank_lower:+.2%}, {uniform_rank_upper:+.2%}] |\n\n"
        )
        f.write(f"- Uniform fusion P(ROI>0): {uniform_rank_p_positive:.1%}\n\n")

        f.write("## 3. Permutation test (5.000 lần)\n\n")
        f.write(
            "Giữ nguyên các số XPIS đã chọn, xáo kết quả giữa các ngày. "
            "p-value là tỷ lệ hoán vị có PnL lớn hơn hoặc bằng PnL quan sát.\n\n"
        )
        f.write("| Chỉ số | Giá trị |\n|---|---:|\n")
        f.write(f"| PnL quan sát | {permutation['observed_pnl'] * 1000:+,.0f}đ |\n")
        f.write(f"| PnL hoán vị trung bình | {permutation['mean_pnl'] * 1000:+,.0f}đ |\n")
        f.write(f"| PnL hoán vị 95% | [{permutation['lower_pnl'] * 1000:+,.0f}đ, {permutation['upper_pnl'] * 1000:+,.0f}đ] |\n")
        f.write(f"| p-value một phía | {permutation['p_value']:.4f} |\n\n")

        f.write("## 4. Ablation leaderboard cùng mức exposure\n\n")
        f.write("| Model | Lượt cược | ROI | ROI bootstrap 95% |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in model_leaderboard:
            f.write(
                f"| {row['name']} | {row['total_bets']} | {row['roi']:+.2%} | "
                f"[{row['roi_lower']:+.2%}, {row['roi_upper']:+.2%}] |\n"
            )
        f.write("\n")
        
        # In trọng số của Layer 5 đang được đánh giá.
        f.write("## 5. Trọng số Fusion hiện tại của 11 Models (Layer 5)\n\n")
        f.write("| Model | Trọng số |\n")
        f.write("|---|:---:|\n")
        report_weights = (
            {name: 1.0 / len(model_probas) for name in model_probas}
            if uniform_fusion
            else fusion.weights
        )
        for k, v in sorted(report_weights.items(), key=lambda x: x[1], reverse=True):
            f.write(f"| {k} | {v:.2%} |\n")
            
        f.write("\n## 6. Nhật ký 15 ngày cược tiêu biểu (Tối ưu danh mục)\n\n")
        f.write("| Ngày | Số cược | Nháy | PnL Flat (đ) | Chi tiết cược Kelly | PnL Kelly (đ) | Diversification |\n")
        f.write("|---|---|:---:|---|---|---|:---:|\n")
        
        sorted_daily = sorted(daily_results, key=lambda x: x['pnl_flat'], reverse=True)
        for r in sorted_daily[:15]:
            bets_str = ", ".join(r['bets']) if r['bets'] else "SKIP"
            f.write(f"| {r['date']} | {bets_str} | {r['hits']} | {r['pnl_flat']*1000:+,.0f}đ | {r['kelly_bets']} | {r['kelly_pnl']*COST_PER_BET*1000:+,.0f}đ | {r['div_score']:.2f} |\n")
            
    print(f"✅ Đã cập nhật báo cáo tại: {report_path}")


if __name__ == "__main__":
    # Dùng cấu hình tối ưu quét được (Prob=0.31, Conf=0.45)
    run_xpis_backtest(n_test_days=180, min_prob=0.31, min_conf=0.45, top_k=4)
