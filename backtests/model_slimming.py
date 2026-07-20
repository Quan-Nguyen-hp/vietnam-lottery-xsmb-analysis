"""
backtests/model_slimming.py — Ablation toàn diện từng model + đánh giá pruned ensemble.

Mục đích (ROADMAP §5):
  1. Lập bảng đóng góp ngoài mẫu của từng model: Brier, LogLoss, Precision@K, ROI
  2. Loại hoặc giảm trọng số model không vượt baseline
  3. So sánh full ensemble vs pruned ensemble trên cùng cửa sổ

Chạy:
  python backtests/model_slimming.py                          # Default: 365 ngày
  python backtests/model_slimming.py --days 180              # Cửa sổ tùy ý
  python backtests/model_slimming.py --top-k 4              # top_k tùy ý
  python backtests/model_slimming.py --skip-pruned           # Chỉ ablation, không chạy pruned

Output: backtests/results/model_slimming_report.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from src.data.loader import DataLoader
from src.evidence.builder import EvidenceBuilder
from src.evidence.store import EvidenceStore
from src.features.feature_store import FeatureStore
from src.probability import get_all_models
from src.probability.lgb_model import LightGBMProbabilityModel
from src.probability.count_poisson import CountEWMAPoissonPredictor
from src.meta.fusion import MetaFusion
from src.meta.calibration import ProbabilityCalibrator
from src.decision.engine import DecisionEngine
from src.decision.confidence import ConfidenceEngine
from src.evaluation.metrics import EvaluationMetrics

# ── Cấu hình ──────────────────────────────────────────────────────────────────
COST_PER_BET = 27.0
PAYOUT_PER_HIT = 99.0
RETRAIN_INTERVAL = 30
TRAIN_WINDOW_DAYS = 365
CALIBRATION_WINDOW_DAYS = 90
CALIBRATION_FIT_DAYS = 45
BOOTSTRAP_RESAMPLES = 20_000


# ── Metrics helpers ────────────────────────────────────────────────────────────

def _brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def _logloss(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    ece = 0.0
    n = len(labels)
    for lower, upper in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        mask = (probs >= lower) & (probs < upper)
        if mask.sum() == 0:
            continue
        ece += float(np.mean(mask)) * abs(float(np.mean(labels[mask])) - float(np.mean(probs[mask])))
    return ece


def _precision_at_k(proba: np.ndarray, actual: set[int], k: int) -> float:
    top_k = np.argsort(proba)[::-1][:k]
    hits = len([n for n in top_k if n in actual])
    return hits / float(k)


def _bootstrap_roi_interval(daily_pnl: np.ndarray, daily_cost: np.ndarray) -> tuple[float, float, float]:
    if len(daily_pnl) == 0 or daily_cost.sum() <= 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(daily_pnl), size=(BOOTSTRAP_RESAMPLES, len(daily_pnl)))
    sampled_cost = daily_cost[indices].sum(axis=1)
    sampled_roi = np.divide(
        daily_pnl[indices].sum(axis=1),
        sampled_cost,
        out=np.zeros(BOOTSTRAP_RESAMPLES),
        where=sampled_cost > 0,
    )
    lower, upper = np.quantile(sampled_roi, [0.025, 0.975])
    prob_positive = float(np.mean(sampled_roi > 0.0))
    return float(lower), float(upper), prob_positive


def _paired_bootstrap_roi_delta(
    candidate_pnl: np.ndarray,
    baseline_pnl: np.ndarray,
    shared_cost: np.ndarray,
) -> tuple[float, float, float, float]:
    """Bootstrap ghép cặp chênh lệch ROI trên đúng các ngày và cùng exposure."""
    if len(candidate_pnl) == 0 or shared_cost.sum() <= 0:
        return 0.0, 0.0, 0.0, 0.0
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(candidate_pnl), size=(BOOTSTRAP_RESAMPLES, len(candidate_pnl)))
    sampled_cost = shared_cost[indices].sum(axis=1)
    sampled_delta = np.divide(
        (candidate_pnl - baseline_pnl)[indices].sum(axis=1),
        sampled_cost,
        out=np.full(BOOTSTRAP_RESAMPLES, np.nan),
        where=sampled_cost > 0,
    )
    sampled_delta = sampled_delta[np.isfinite(sampled_delta)]
    if len(sampled_delta) == 0:
        return 0.0, 0.0, 0.0, 0.0
    observed_delta = float((candidate_pnl - baseline_pnl).sum() / shared_cost.sum())
    lower, upper = np.quantile(sampled_delta, [0.025, 0.975])
    prob_positive = float(np.mean(sampled_delta > 0.0))
    return observed_delta, float(lower), float(upper), prob_positive


# ── Core functions ─────────────────────────────────────────────────────────────

def run_model_slimming(
    n_test_days: int = 365,
    top_k: int = 4,
    min_prob: float = 0.31,
    min_conf: float = 0.45,
    skip_pruned: bool = False,
    report_name: str = "model_slimming_report.md",
    kelly_selects_bets: bool = False,
    forced_removed_names: tuple[str, ...] = (),
    add_count_ewma: bool = False,
) -> None:
    print("=" * 60)
    print("  §5 — MODEL SLIMMING: Ablation toàn diện ngoài mẫu")
    print("=" * 60)
    print(f"  Cửa sổ: {n_test_days} ngày | top_k={top_k} | p≥{min_prob} | conf≥{min_conf}")
    print()

    loader = DataLoader().load()
    total_days = loader.total_days
    start_idx = total_days - n_test_days

    evidence_store = EvidenceStore()
    evidence_builder = EvidenceBuilder(evidence_store)
    feature_store = FeatureStore(S_history=loader.S, evidence_store=evidence_store)

    # Tất cả 11 models
    all_models = get_all_models()
    if add_count_ewma:
        all_models.append(CountEWMAPoissonPredictor())
    model_names = [m.name for m in all_models]

    prize_cols = loader.prize_cols()

    # ── Accumulators ──────────────────────────────────────────────────────────
    # Per-model metrics
    model_accum = {m.name: {"probs": [], "labels": []} for m in all_models}

    # Per-model bet tracking (same exposure as XPIS decisions)
    model_bet_log = {m.name: [] for m in all_models}

    # Baseline tracking
    random_log = []
    frequency_log = []

    # Ensemble (full 11) tracking
    ensemble_log = []

    # Daily model probas + actual results — dùng cho pruned comparison (tránh re-predict)
    daily_prediction_log = []

    # LGBM calibrator state
    lgb_model = LightGBMProbabilityModel()
    best_calibrator = None
    meta_cols = []

    # Decision engine for determining XPIS bets each day
    decision_engine = DecisionEngine(
        min_probability=min_prob,
        min_confidence=min_conf,
        top_k=top_k,
        kelly_odds=PAYOUT_PER_HIT / COST_PER_BET,
        kelly_fraction=0.20,
        min_diversification=0.85,
        kelly_selects_bets=kelly_selects_bets,
    )
    confidence_engine = ConfidenceEngine()
    fusion = MetaFusion()

    rng = np.random.default_rng(42)
    t_start = time.time()

    print(f"  Bắt đầu walk-forward từ index {start_idx}...")

    for step, idx in enumerate(range(start_idx, total_days)):
        if step % 50 == 0:
            print(f"  [{step}/{n_test_days}]")

        current_row = loader.df.iloc[idx]
        current_date = pd.to_datetime(current_row["date"])
        date_str = current_date.strftime("%Y-%m-%d")

        # Label thực tế
        actual_lotos = current_row[prize_cols].dropna().values.astype(int).tolist()
        actual_set = set(actual_lotos)
        y = np.zeros(100)
        for num in actual_set:
            y[num] = 1

        # ── Retrain LGBM mỗi 30 ngày ───────────────────────────────────────
        if step % RETRAIN_INTERVAL == 0:
            train_end_idx = idx - CALIBRATION_WINDOW_DAYS
            train_start_idx = max(50, train_end_idx - TRAIN_WINDOW_DAYS)

            train_snapshots, train_labels = [], []
            # Không đưa cửa sổ calibration/selection vào tập huấn luyện.
            for t_idx in range(train_start_idx, train_end_idx):
                t_df_hist, t_S_hist = loader.slice_history(t_idx)
                t_row = loader.df.iloc[t_idx]
                t_date = t_row["date"].to_pydatetime()
                t_df_ev = evidence_builder.build_all(t_df_hist, t_S_hist, t_date, save=True)
                t_df_feat = feature_store.build(t_df_ev, t_date.strftime("%Y-%m-%d"), S=t_S_hist)
                train_snapshots.append(t_df_feat)
                t_actual = set(t_row[prize_cols].dropna().values.astype(int).tolist())
                t_y = np.zeros(100)
                for num in t_actual:
                    t_y[num] = 1
                train_labels.append(t_y)

            meta_cols = [c for c in train_snapshots[0].columns if c not in ("number", "date")]
            X_train = np.vstack([df[meta_cols].values for df in train_snapshots])
            y_train = np.concatenate(train_labels)
            lgb_model.fit(X_train, y_train, feature_names=meta_cols)

            # Calibration: nested split 45/45
            from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator

            val_snapshots, val_labels = [], []
            for t_idx in range(train_end_idx, idx):
                t_df_hist, t_S_hist = loader.slice_history(t_idx)
                t_row = loader.df.iloc[t_idx]
                t_date = t_row["date"].to_pydatetime()
                t_df_ev = evidence_builder.build_all(t_df_hist, t_S_hist, t_date, save=True)
                t_df_feat = feature_store.build(t_df_ev, t_date.strftime("%Y-%m-%d"), S=t_S_hist)
                val_snapshots.append(t_df_feat)
                t_actual = set(t_row[prize_cols].dropna().values.astype(int).tolist())
                t_y = np.zeros(100)
                for num in t_actual:
                    t_y[num] = 1
                val_labels.append(t_y)

            X_val = np.vstack([df[meta_cols].values for df in val_snapshots])
            X_val_df = pd.DataFrame(X_val, columns=meta_cols)
            y_val_arr = np.concatenate(val_labels)

            split = CALIBRATION_FIT_DAYS * 100
            cal_sig = CalibratedClassifierCV(estimator=FrozenEstimator(lgb_model._model), method="sigmoid")
            cal_iso = CalibratedClassifierCV(estimator=FrozenEstimator(lgb_model._model), method="isotonic")
            cal_sig.fit(X_val_df.iloc[:split], y_val_arr[:split])
            cal_iso.fit(X_val_df.iloc[:split], y_val_arr[:split])

            p_sig = cal_sig.predict_proba(X_val_df.iloc[split:])[:, 1]
            p_iso = cal_iso.predict_proba(X_val_df.iloc[split:])[:, 1]
            y_sel = y_val_arr[split:]

            score_sig = 0.5 * _brier(p_sig, y_sel) + 0.3 * _logloss(p_sig, y_sel) + 0.2 * _ece(p_sig, y_sel)
            score_iso = 0.5 * _brier(p_iso, y_sel) + 0.3 * _logloss(p_iso, y_sel) + 0.2 * _ece(p_iso, y_sel)
            best_calibrator = cal_sig if score_sig <= score_iso else cal_iso

            # Cập nhật fusion weights trên nửa sau
            eval_preds = {}
            for m in all_models:
                if m.name == "lightgbm_classifier":
                    eval_preds[m.name] = list(
                        best_calibrator.predict_proba(X_val_df.iloc[split:])[:, 1].reshape(CALIBRATION_FIT_DAYS, 100)
                    )
                else:
                    preds = []
                    for i in range(CALIBRATION_FIT_DAYS, len(val_snapshots)):
                        t_df_hist, t_S_hist = loader.slice_history(train_end_idx + i)
                        preds.append(m.predict_proba(val_snapshots[i], t_df_hist, t_S_hist))
                    eval_preds[m.name] = preds

            y_sel_matrix = y_sel.reshape(CALIBRATION_FIT_DAYS, 100)
            fusion.compute_dynamic_weights(
                {k: np.array(v) for k, v in eval_preds.items()},
                y_sel_matrix,
            )
            feature_store.clear_ram_cache()

        # ── Predict ngày hiện tại ──────────────────────────────────────────
        df_hist, S_hist = loader.slice_history(idx)
        df_ev = evidence_builder.build_all(df_hist, S_hist, current_date.to_pydatetime(), save=True)
        df_feat = feature_store.build(df_ev, date_str, S=S_hist)

        model_probas = {}
        for m in all_models:
            if m.name == "lightgbm_classifier":
                model_probas[m.name] = best_calibrator.predict_proba(df_feat[meta_cols])[:, 1]
            else:
                model_probas[m.name] = m.predict_proba(df_feat, df_hist, S_hist)

        # ── XPIS Decision (full 11-model ensemble) ────────────────────────
        meta_proba = fusion.fuse(model_probas)
        day_decision = decision_engine.decide(
            date=date_str,
            meta_proba=meta_proba,
            model_probas=model_probas,
            S_history=S_hist,
        )
        xpis_bets = [b.number for b in day_decision.bets]
        n_bets_xpis = len(xpis_bets)

        # Lưu predictions cho pruned comparison (không cần re-predict)
        daily_prediction_log.append({
            "model_probas": {k: v.copy() for k, v in model_probas.items()},
            "actual_lotos": actual_lotos,
            "date": date_str,
            "S_history": S_hist,
        })

        # ── Tích lũy metrics cho mỗi model ────────────────────────────────
        for m in all_models:
            proba = model_probas[m.name]
            model_accum[m.name]["probs"].append(proba.copy())
            model_accum[m.name]["labels"].append(y.copy())

            # Same exposure: top n_bets_xpis from this model
            if n_bets_xpis > 0:
                model_picks = list(np.argsort(proba)[::-1][:n_bets_xpis])
                model_hits = sum(actual_lotos.count(int(n)) for n in model_picks)
                model_bet_log[m.name].append({
                    "n_bets": n_bets_xpis,
                    "n_hits": model_hits,
                    "daily_pnl": model_hits * PAYOUT_PER_HIT - n_bets_xpis * COST_PER_BET,
                })
            else:
                model_bet_log[m.name].append({"n_bets": 0, "n_hits": 0, "daily_pnl": 0.0})

        # ── Baselines ──────────────────────────────────────────────────────
        if n_bets_xpis > 0:
            random_picks = rng.choice(100, size=n_bets_xpis, replace=False)
            random_hits = sum(actual_lotos.count(int(n)) for n in random_picks)
            random_log.append({
                "n_bets": n_bets_xpis, "n_hits": random_hits,
                "daily_pnl": random_hits * PAYOUT_PER_HIT - n_bets_xpis * COST_PER_BET,
            })
            frequency_picks = list(np.argsort(S_hist.sum(axis=0))[::-1][:n_bets_xpis])
            freq_hits = sum(actual_lotos.count(int(n)) for n in frequency_picks)
            frequency_log.append({
                "n_bets": n_bets_xpis, "n_hits": freq_hits,
                "daily_pnl": freq_hits * PAYOUT_PER_HIT - n_bets_xpis * COST_PER_BET,
            })

        # ── Ensemble result ──────────────────────────────────────────────
        xpis_hits = sum(actual_lotos.count(n) for n in xpis_bets)
        ensemble_log.append({
            "n_bets": n_bets_xpis, "n_hits": xpis_hits,
            "daily_pnl": xpis_hits * PAYOUT_PER_HIT - n_bets_xpis * COST_PER_BET,
        })

    elapsed = time.time() - t_start
    print(f"\n  ✅ Walk-forward hoàn tất sau {elapsed:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # PHÂN TÍCH KẾT QUẢ
    # ══════════════════════════════════════════════════════════════════════

    print("\n  📊 Phân tích kết quả...")

    # Tính metrics cho mỗi model
    model_results = []
    for m in all_models:
        name = m.name
        probs_flat = np.concatenate(model_accum[name]["probs"])
        labels_flat = np.concatenate(model_accum[name]["labels"])

        brier = _brier(probs_flat, labels_flat)
        logloss_val = _logloss(probs_flat, labels_flat)
        ece_val = _ece(probs_flat, labels_flat)

        # Precision@K per day
        prec_k_list = []
        for i in range(len(model_accum[name]["probs"])):
            actual_set_i = set(np.where(model_accum[name]["labels"][i] > 0)[0].tolist())
            prec_k_list.append(_precision_at_k(model_accum[name]["probs"][i], actual_set_i, top_k))
        precision_k = float(np.mean(prec_k_list))

        # ROI from bet log
        bet_df = pd.DataFrame(model_bet_log[name])
        total_cost = bet_df["n_bets"].sum() * COST_PER_BET
        total_pnl = bet_df["daily_pnl"].sum()
        roi = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

        # Bootstrap CI
        pnl_arr = bet_df["daily_pnl"].values.astype(float)
        cost_arr = bet_df["n_bets"].values.astype(float) * COST_PER_BET
        roi_lower, roi_upper, prob_pos = _bootstrap_roi_interval(pnl_arr, cost_arr)

        model_results.append({
            "name": name,
            "brier": brier,
            "logloss": logloss_val,
            "ece": ece_val,
            "precision_k": precision_k,
            "total_bets": int(bet_df["n_bets"].sum()),
            "total_hits": int(bet_df["n_hits"].sum()),
            "roi": roi,
            "roi_lower": roi_lower,
            "roi_upper": roi_upper,
            "prob_positive": prob_pos,
        })

    # Baselines
    baseline_results = {}
    for bl_name, bl_log in [
        ("random", random_log),
        ("frequency", frequency_log),
        (f"ensemble_{len(all_models)}", ensemble_log),
    ]:
        bl_df = pd.DataFrame(bl_log)
        total_cost = bl_df["n_bets"].sum() * COST_PER_BET
        total_pnl = bl_df["daily_pnl"].sum()
        roi = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0
        pnl_arr = bl_df["daily_pnl"].values.astype(float)
        cost_arr = bl_df["n_bets"].values.astype(float) * COST_PER_BET
        roi_lower, roi_upper, prob_pos = _bootstrap_roi_interval(pnl_arr, cost_arr)
        baseline_results[bl_name] = {
            "total_bets": int(bl_df["n_bets"].sum()),
            "total_hits": int(bl_df["n_hits"].sum()),
            "roi": roi,
            "roi_lower": roi_lower,
            "roi_upper": roi_upper,
            "prob_positive": prob_pos,
        }

    # Median Brier (ngưỡng phân loại)
    brier_values = [r["brier"] for r in model_results]
    median_brier = float(np.median(brier_values))
    frequency_roi = baseline_results["frequency"]["roi"]

    # Phân loại models
    for r in model_results:
        r["brier_above_median"] = r["brier"] > median_brier
        r["roi_below_freq"] = r["roi"] < frequency_roi
        # CI hoàn toàn nằm dưới frequency baseline?
        r["ci_below_freq"] = r["roi_upper"] < frequency_roi
        # Tiêu chí loại: ROI < freq AND Brier > median
        r["prune_candidate"] = r["roi_below_freq"] and r["brier_above_median"]
        # Tiêu chí loại mạnh: CI hoàn toàn dưới freq
        r["prune_strong"] = r["ci_below_freq"]

    # Sort by ROI descending
    model_results.sort(key=lambda x: x["roi"], reverse=True)

    pruned_names = list(forced_removed_names) if forced_removed_names else [
        r["name"] for r in model_results if r["prune_candidate"]
    ]
    strong_prune_names = [r["name"] for r in model_results if r["prune_strong"]]
    retained_names = [r["name"] for r in model_results if r["name"] not in pruned_names]

    print(f"  Median Brier: {median_brier:.6f}")
    print(f"  Frequency baseline ROI: {frequency_roi:+.2f}%")
    print(f"  Prune candidates (ROI<freq AND Brier>median): {pruned_names}")
    print(f"  Strong prune (CI hoàn toàn < freq): {strong_prune_names}")
    print(f"  Retained models: {retained_names}")

    # ── Báo cáo Markdown ──────────────────────────────────────────────────
    results_dir = root_dir / "backtests" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / report_name

    # Lấy date range
    first_date = loader.df.iloc[start_idx]["date"]
    last_date = loader.df.iloc[total_days - 1]["date"]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Báo cáo Model Slimming — Ablation ngoài mẫu toàn diện\n\n")
        f.write(f"- **Kỳ kiểm thử**: {n_test_days} ngày ({first_date} đến {last_date})\n")
        f.write(f"- **Tham số**: top_k={top_k} | min_prob={min_prob} | min_conf={min_conf}\n")
        f.write(f"- **Thời gian chạy**: {elapsed:.1f}s\n\n")

        # Bảng tổng hợp
        f.write("## 1. Bảng đóng góp ngoài mẫu từng model\n\n")
        f.write("| Model | Brier ↓ | LogLoss ↓ | ECE ↓ | Prec@K ↑ | Bets | Hits | ROI | CI95 | P(ROI>0) | Brier>Med | ROI<Freq | Trạng thái |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|:---:|:---:|:---:|\n")
        for r in model_results:
            prune_mark = "⚠️ ỨNG VIÊN" if r["prune_candidate"] else "✅ GIỮ"
            f.write(
                f"| {r['name']} | {r['brier']:.6f} | {r['logloss']:.6f} | {r['ece']:.6f} | "
                f"{r['precision_k']:.4f} | {r['total_bets']} | {r['total_hits']} | "
                f"{r['roi']:+.2f}% | [{r['roi_lower']:+.2%}, {r['roi_upper']:+.2%}] | "
                f"{r['prob_positive']:.1%} | "
                f"{'Yes' if r['brier_above_median'] else 'No'} | "
                f"{'Yes' if r['roi_below_freq'] else 'No'} | "
                f"{prune_mark} |\n"
            )

        f.write(f"\n- **Median Brier**: {median_brier:.6f}\n")
        f.write(f"- **Frequency baseline ROI**: {frequency_roi:+.2f}%\n\n")

        f.write("### Tiêu chí sàng lọc model\n")
        f.write(f"1. **ROI < frequency baseline** ({frequency_roi:+.2f}%) — không tốt hơn chọn số theo tần suất\n")
        f.write("2. **Brier > median Brier** — xác suất kém chất lượng hơn trung bình\n")
        f.write("3. Cả 2 điều kiện chỉ tạo **ứng viên loại**; chỉ áp dụng khi CI95 của chênh lệch ROI ghép cặp Pruned-fixed − Full > 0.\n\n")

        f.write(f"### Models ứng viên loại: **{pruned_names}**\n\n")
        f.write(f"### Models đề xuất giữ: **{retained_names}**\n\n")

        # Baselines
        f.write("## 2. So sánh baselines\n\n")
        f.write("| Baseline | Bets | Hits | ROI | CI95 |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for bl_name, bl in baseline_results.items():
            f.write(
                f"| {bl_name} | {bl['total_bets']} | {bl['total_hits']} | "
                f"{bl['roi']:+.2f}% | [{bl['roi_lower']:+.2%}, {bl['roi_upper']:+.2%}] |\n"
            )

        # Phân tích Spearman rank correlation giữa Brier và ROI
        from scipy.stats import spearmanr

        briers = [r["brier"] for r in model_results]
        rois = [r["roi"] for r in model_results]
        corr, p_val = spearmanr(briers, rois)
        f.write(f"\n## 3. Phân tích tương quan\n\n")
        f.write(f"- **Spearman rank (Brier vs ROI)**: ρ = {corr:.3f}, p = {p_val:.4f}\n")
        if p_val < 0.05:
            direction = "nghịch" if corr < 0 else "thuận"
            f.write(f"- Có tương quan {direction} có ý nghĩa thống kê giữa Brier và ROI trong mẫu này.\n\n")
        else:
            f.write("- Không có bằng chứng về tương quan có ý nghĩa thống kê giữa Brier và ROI trong mẫu này.\n\n")

        # Trọng số fusion hiện tại
        f.write("## 4. Trọng số Fusion hiện tại\n\n")
        f.write("| Model | Trọng số | Giữ/Loại |\n")
        f.write("|---|:---:|:---:|\n")
        weights = fusion.weights
        for r in model_results:
            w = weights.get(r["name"], 0.0)
            status = "Giữ" if r["name"] in retained_names else "Ứng viên loại"
            f.write(f"| {r['name']} | {w:.2%} | {status} |\n")

    print(f"\n  ✅ Báo cáo: {report_path}")

    # ── Pruned ensemble backtest ──────────────────────────────────────────
    if skip_pruned or not pruned_names:
        print("\n  ⏭️ Bỏ qua pruned ensemble backtest.")
        return

    print(f"\n  🔬 Chạy pruned ensemble ({len(retained_names)} models) vs full ({len(model_names)} models)...")

    pruned_models = [m for m in all_models if m.name in retained_names]

    # Chạy pruned comparison
    # Truyền ensemble_log (đã có từ main loop) làm baseline Full, chỉ cần chạy Pruned
    _run_pruned_comparison(
        all_models, pruned_models, retained_names,
        prize_cols, top_k, min_prob, min_conf,
        rng, report_path,
        daily_prediction_log=daily_prediction_log,
        saved_fusion_weights=fusion.weights.copy(),
        ensemble_log=ensemble_log,  # baseline từ main loop — chính xác 100%
        kelly_selects_bets=kelly_selects_bets,
    )


def _run_pruned_comparison(
    all_models, pruned_models, retained_names,
    prize_cols, top_k, min_prob, min_conf,
    rng, report_path: Path,
    daily_prediction_log: list[dict],
    saved_fusion_weights: dict | None = None,
    ensemble_log: list[dict] | None = None,
    kelly_selects_bets: bool = False,
) -> None:
    """So sánh full ensemble vs pruned ensemble trên cùng cửa sổ.

    Full baseline: dùng ensemble_log từ main loop (chính xác 100%).
    Pruned: fuse lại từ saved model_probas, chỉ giữ models đã chọn.
    """

    decision_engine_pruned = DecisionEngine(
        min_probability=min_prob, min_confidence=min_conf,
        top_k=top_k, kelly_odds=PAYOUT_PER_HIT / COST_PER_BET,
        kelly_fraction=0.20, min_diversification=0.85,
        kelly_selects_bets=kelly_selects_bets,
    )

    # Pruned fusion: redistribute weights của models giữ lại, tổng = 1
    fusion_pruned = MetaFusion()
    if saved_fusion_weights:
        total_pruned_w = sum(saved_fusion_weights.get(m.name, 0.0) for m in pruned_models)
        if total_pruned_w > 0:
            fusion_pruned._weights = {
                m.name: saved_fusion_weights.get(m.name, 0.0) / total_pruned_w
                for m in pruned_models
            }
        else:
            fusion_pruned._weights = {m.name: 1.0 / len(pruned_models) for m in pruned_models}
    else:
        fusion_pruned._weights = {m.name: 1.0 / len(pruned_models) for m in pruned_models}

    pruned_log = []
    # Pruned fixed-exposure: giới hạn bets = full ensemble mỗi ngày
    pruned_fixed_log = []

    for i, entry in enumerate(daily_prediction_log):
        model_probas = entry["model_probas"]
        actual_lotos = entry["actual_lotos"]
        date_str = entry["date"]
        S_hist = entry.get("S_history")

        # Pruned ensemble — chỉ dùng models giữ lại
        model_probas_pruned = {m.name: model_probas[m.name] for m in pruned_models}
        meta_proba_pruned = fusion_pruned.fuse(model_probas_pruned)
        day_pruned = decision_engine_pruned.decide(
            date=date_str, meta_proba=meta_proba_pruned,
            model_probas=model_probas_pruned, S_history=S_hist,
        )
        pruned_bets = [b.number for b in day_pruned.bets]
        pruned_hits = sum(actual_lotos.count(n) for n in pruned_bets)
        pruned_log.append({
            "n_bets": len(pruned_bets), "n_hits": pruned_hits,
            "daily_pnl": pruned_hits * PAYOUT_PER_HIT - len(pruned_bets) * COST_PER_BET,
        })

        # Pruned fixed-exposure: top-N từ pruned fusion, N = số bets của full ensemble ngày đó
        full_n = ensemble_log[i]["n_bets"] if ensemble_log and i < len(ensemble_log) else 0
        if full_n > 0:
            fixed_bets = list(np.argsort(meta_proba_pruned)[::-1][:full_n])
            fixed_hits = sum(actual_lotos.count(n) for n in fixed_bets)
            pruned_fixed_log.append({
                "n_bets": full_n, "n_hits": fixed_hits,
                "daily_pnl": fixed_hits * PAYOUT_PER_HIT - full_n * COST_PER_BET,
            })
        else:
            pruned_fixed_log.append({"n_bets": 0, "n_hits": 0, "daily_pnl": 0.0})

    # Full baseline: dùng ensemble_log từ main loop
    full_log = ensemble_log if ensemble_log else []

    # Tính & in kết quả
    results = {}
    for label, log_data in [
        (f"Full ({len(all_models)} models)", full_log),
        (f"Pruned ({len(retained_names)} models)", pruned_log),
        (f"Pruned-fixed ({len(retained_names)} models, same exposure)", pruned_fixed_log),
    ]:
        df = pd.DataFrame(log_data)
        total_cost = df["n_bets"].sum() * COST_PER_BET
        total_pnl = df["daily_pnl"].sum()
        roi = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0
        pnl_arr = df["daily_pnl"].values.astype(float)
        cost_arr = df["n_bets"].values.astype(float) * COST_PER_BET
        roi_lower, roi_upper, prob_pos = _bootstrap_roi_interval(pnl_arr, cost_arr)
        results[label] = {
            "total_bets": int(df["n_bets"].sum()),
            "total_hits": int(df["n_hits"].sum()),
            "roi": roi, "roi_lower": roi_lower, "roi_upper": roi_upper, "prob_pos": prob_pos,
        }
        print(f"  {label}: ROI={roi:+.2f}% CI=[{roi_lower:+.2%}, {roi_upper:+.2%}] Bets={int(df['n_bets'].sum())} Hits={int(df['n_hits'].sum())}")

    full_df = pd.DataFrame(full_log)
    fixed_df = pd.DataFrame(pruned_fixed_log)
    delta_roi, delta_lower, delta_upper, delta_prob_positive = _paired_bootstrap_roi_delta(
        fixed_df["daily_pnl"].to_numpy(dtype=float),
        full_df["daily_pnl"].to_numpy(dtype=float),
        full_df["n_bets"].to_numpy(dtype=float) * COST_PER_BET,
    )
    # Nhánh pruned dùng trọng số cuối kỳ cố định, nên chỉ là diagnostic.
    # Không dùng kết quả này để tự động xác nhận thay đổi production.
    pruning_validated = False

    # Append to report
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n## 5. So sánh Pruned vs Full Ensemble\n\n")
        f.write("> **Phương pháp**: Full baseline từ main walk-forward loop. Pruned dùng trọng số cuối kỳ rồi redistribute trên models giữ lại.\n")
        f.write("> **Cảnh báo**: Pruned/Pruned-fixed chỉ là diagnostic fixed-weight; cần hai lần chạy walk-forward độc lập để so sánh cấu hình.\n\n")

        for label, stats in results.items():
            r = stats
            f.write(f"### {label}\n\n")
            f.write(f"| Metric | Giá trị |\n|---|---:|\n")
            f.write(f"| Tổng cược | {r['total_bets']} |\n")
            f.write(f"| Tổng nháy | {r['total_hits']} |\n")
            f.write(f"| ROI | {r['roi']:+.2f}% |\n")
            f.write(f"| CI95 | [{r['roi_lower']:+.2%}, {r['roi_upper']:+.2%}] |\n")
            f.write(f"| P(ROI>0) | {r['prob_pos']:.1%} |\n\n")

        f.write("### Kiểm định chênh lệch ghép cặp Pruned-fixed − Full\n\n")
        f.write("| Chỉ số | Giá trị |\n|---|---:|\n")
        f.write(f"| ΔROI quan sát | {delta_roi:+.2%} |\n")
        f.write(f"| Bootstrap CI95 của ΔROI | [{delta_lower:+.2%}, {delta_upper:+.2%}] |\n")
        f.write(f"| P(ΔROI>0) | {delta_prob_positive:.1%} |\n")
        f.write(f"| Quyết định loại model | {'ĐẠT' if pruning_validated else 'KHÔNG ĐẠT'} |\n\n")
        f.write(
            "- Diagnostic fixed-weight không đủ quyền xác nhận thay đổi production; "
            "cần chạy hai cấu hình walk-forward độc lập.\n\n"
        )

        # Phân tích exposure gap
        full_bets = results[f"Full ({len(all_models)} models)"]["total_bets"]
        pruned_bets = results[f"Pruned ({len(retained_names)} models)"]["total_bets"]
        fixed_bets = results[f"Pruned-fixed ({len(retained_names)} models, same exposure)"]["total_bets"]
        f.write("### Phân tích exposure gap\n\n")
        f.write(f"- Full ensemble cược trung bình {full_bets}/{len(full_log)} ngày = {full_bets/max(len(full_log),1):.2f} bets/ngày\n")
        f.write(f"- Pruned (tự do) cược {pruned_bets}/{len(pruned_log)} ngày = {pruned_bets/max(len(pruned_log),1):.2f} bets/ngày\n")
        f.write(f"- Pruned-fixed (cùng exposure) cược {fixed_bets}/{len(pruned_fixed_log)} ngày\n")
        removed_label = ", ".join(m.name for m in all_models if m.name not in retained_names)
        f.write(f"- Bỏ ứng viên `{removed_label}` làm fusion probability spread khác → confidence/probability thresholds cho qua nhiều hơn → số bets tăng {pruned_bets - full_bets} lượt.\n")
        f.write("- Pruned-fixed chỉ đo xếp hạng với trọng số cuối kỳ cố định; không dùng để promote model.\n\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="§5 Model Slimming — Ablation toàn diện ngoài mẫu")
    parser.add_argument("--days", type=int, default=365, help="Số ngày backtest (default: 365)")
    parser.add_argument("--top-k", type=int, default=4, help="Top K số cược (default: 4)")
    parser.add_argument("--min-prob", type=float, default=0.31, help="Min probability (default: 0.31)")
    parser.add_argument("--min-conf", type=float, default=0.45, help="Min confidence (default: 0.45)")
    parser.add_argument("--skip-pruned", action="store_true", help="Chỉ ablation, không chạy pruned ensemble")
    parser.add_argument("--report-name", default="model_slimming_report.md", help="Tên file báo cáo Markdown")
    parser.add_argument(
        "--kelly-selects-bets",
        action="store_true",
        help="Chế độ cũ: Kelly bằng 0 sẽ loại số; mặc định Kelly chỉ sizing",
    )
    parser.add_argument(
        "--remove-model",
        action="append",
        default=[],
        help="Model cố định muốn loại khi chạy comparison (có thể lặp; ví dụ loto_repeat)",
    )
    parser.add_argument(
        "--add-count-ewma",
        action="store_true",
        help="Thêm CountEWMAPoissonPredictor chỉ cho research backtest",
    )
    args = parser.parse_args()

    run_model_slimming(
        n_test_days=args.days,
        top_k=args.top_k,
        min_prob=args.min_prob,
        min_conf=args.min_conf,
        skip_pruned=args.skip_pruned,
        report_name=args.report_name,
        kelly_selects_bets=args.kelly_selects_bets,
        forced_removed_names=tuple(args.remove_model),
        add_count_ewma=args.add_count_ewma,
    )
