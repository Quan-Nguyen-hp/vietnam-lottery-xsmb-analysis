"""Walk-forward research cho E[số nháy], tách khỏi P(xuất hiện >= 1 lần).

Script này chỉ dùng dữ liệu trước ngày dự báo và không thay đổi production policy.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.loader import DataLoader
from src.evaluation.metrics import EvaluationMetrics

COST_PER_BET = 27.0
PAYOUT_PER_HIT = 99.0
BOOTSTRAP_RESAMPLES = 20_000
PERMUTATION_RESAMPLES = 5_000
SEED = 42


@dataclass(frozen=True)
class EstimatorSpec:
    name: str
    kind: str
    window: int | None = None
    half_life: float | None = None
    prior_strength: float = 0.0


ESTIMATORS = (
    EstimatorSpec("uniform_027", "uniform"),
    EstimatorSpec("rolling_30", "rolling", window=30),
    EstimatorSpec("rolling_90", "rolling", window=90),
    EstimatorSpec("rolling_180", "rolling", window=180),
    EstimatorSpec("rolling_365", "rolling", window=365),
    EstimatorSpec("ewma_hl30", "ewma", window=180, half_life=30),
    EstimatorSpec("ewma_hl90", "ewma", window=540, half_life=90),
    EstimatorSpec("empirical_bayes_365", "rolling", window=365, prior_strength=365),
)


def _count_matrix(loader: DataLoader) -> np.ndarray:
    draws = loader.df[loader.prize_cols()].to_numpy(dtype=int)
    counts = np.zeros((len(draws), 100), dtype=np.int8)
    for number in range(100):
        counts[:, number] = np.sum(draws == number, axis=1)
    return counts


def _forecast(history: np.ndarray, spec: EstimatorSpec) -> tuple[np.ndarray, float]:
    global_rate = history.shape[1] * 0.0 + 27.0 / 100.0
    if spec.kind == "uniform":
        return np.full(100, 0.27), float("inf")

    window = min(spec.window or len(history), len(history))
    sample = history[-window:].astype(float)
    if spec.kind == "rolling":
        total = sample.sum(axis=0)
        if spec.prior_strength > 0:
            mu = (total + spec.prior_strength * global_rate) / (window + spec.prior_strength)
            return mu, float(window + spec.prior_strength)
        return sample.mean(axis=0), float(window)

    ages = np.arange(window - 1, -1, -1, dtype=float)
    weights = np.exp(-np.log(2.0) * ages / float(spec.half_life))
    weights /= weights.sum()
    mu = weights @ sample
    effective_n = float(1.0 / np.sum(weights ** 2))
    return mu, effective_n


def _bootstrap_roi(daily_bets: np.ndarray, daily_hits: np.ndarray) -> tuple[float, float, float, float]:
    cost = daily_bets.astype(float) * COST_PER_BET
    pnl = daily_hits.astype(float) * PAYOUT_PER_HIT - cost
    total_cost = cost.sum()
    if total_cost <= 0:
        return 0.0, 0.0, 0.0, 0.0
    observed = float(pnl.sum() / total_cost)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(pnl), size=(BOOTSTRAP_RESAMPLES, len(pnl)))
    sampled_cost = cost[idx].sum(axis=1)
    sampled_roi = np.divide(
        pnl[idx].sum(axis=1),
        sampled_cost,
        out=np.full(BOOTSTRAP_RESAMPLES, np.nan),
        where=sampled_cost > 0,
    )
    sampled_roi = sampled_roi[np.isfinite(sampled_roi)]
    lower, upper = np.quantile(sampled_roi, [0.025, 0.975])
    return observed, float(lower), float(upper), float(np.mean(sampled_roi > 0))


def _permutation_p_value(picks: np.ndarray, actual_counts: np.ndarray) -> float:
    observed_hits = int(np.sum(picks * actual_counts))
    rng = np.random.default_rng(SEED)
    permuted_hits = np.empty(PERMUTATION_RESAMPLES, dtype=int)
    for i in range(PERMUTATION_RESAMPLES):
        permuted_hits[i] = int(np.sum(picks * actual_counts[rng.permutation(len(actual_counts))]))
    return float((np.sum(permuted_hits >= observed_hits) + 1) / (PERMUTATION_RESAMPLES + 1))


def _strategy_stats(
    forecasts: np.ndarray,
    actual_counts: np.ndarray,
    top_k: int,
    threshold: float | None,
    effective_ns: np.ndarray | None = None,
    run_permutation: bool = True,
) -> dict:
    picks = np.zeros_like(actual_counts, dtype=np.int8)
    for day, mu in enumerate(forecasts):
        score = mu.copy()
        if effective_ns is not None:
            # Cận dưới gần đúng cho mean count; dùng để kiểm tra độ chắc chắn, không phải tune.
            score = mu - 1.96 * np.sqrt(np.clip(mu, 1e-9, None) / effective_ns[day])
        ranked = np.argsort(score)[::-1]
        if threshold is not None:
            ranked = ranked[score[ranked] > threshold]
        picks[day, ranked[:top_k]] = 1

    daily_bets = picks.sum(axis=1)
    daily_hits = (picks * actual_counts).sum(axis=1)
    roi, lower, upper, p_positive = _bootstrap_roi(daily_bets, daily_hits)
    return {
        "bets": int(daily_bets.sum()),
        "bet_days": int(np.sum(daily_bets > 0)),
        "hits": int(daily_hits.sum()),
        "roi": roi,
        "lower": lower,
        "upper": upper,
        "p_positive": p_positive,
        "permutation_p": (
            _permutation_p_value(picks, actual_counts)
            if run_permutation and daily_bets.sum() > 0
            else 1.0
        ),
    }


def run_research(n_test_days: int = 365, top_k: int = 4, report_name: str = "count_expectation_research_365d.md") -> Path:
    loader = DataLoader().load()
    if n_test_days <= 0 or n_test_days >= loader.total_days:
        raise ValueError("n_test_days phải lớn hơn 0 và nhỏ hơn tổng số ngày dữ liệu")
    counts = _count_matrix(loader)
    start = loader.total_days - n_test_days
    actual = counts[start:]
    metrics = EvaluationMetrics(
        odds=PAYOUT_PER_HIT / COST_PER_BET,
        cost_per_bet=COST_PER_BET,
        payout_per_hit=PAYOUT_PER_HIT,
    )
    break_even = metrics.break_even_expected_count
    rng = np.random.default_rng(SEED)

    rows = []
    forecast_store: dict[str, np.ndarray] = {}
    effective_store: dict[str, np.ndarray] = {}
    for spec in ESTIMATORS:
        forecasts = []
        effective_ns = []
        for idx in range(start, loader.total_days):
            mu, effective_n = _forecast(counts[:idx], spec)
            forecasts.append(mu)
            effective_ns.append(np.full(100, effective_n))
        forecast_matrix = np.asarray(forecasts)
        effective_matrix = np.asarray(effective_ns)
        forecast_store[spec.name] = forecast_matrix
        effective_store[spec.name] = effective_matrix

        count_metrics = metrics.count_forecast_metrics(forecast_matrix, actual)
        p_any = 1.0 - np.exp(-np.clip(forecast_matrix, 0.0, None))
        binary_brier = metrics.brier_score(p_any, (actual > 0).astype(float))
        top4 = _strategy_stats(forecast_matrix, actual, top_k, threshold=None)
        ev_gate = _strategy_stats(forecast_matrix, actual, top_k, threshold=break_even)
        lcb_gate = _strategy_stats(
            forecast_matrix,
            actual,
            top_k,
            threshold=break_even,
            effective_ns=effective_matrix,
        )
        rows.append({
            "name": spec.name,
            **count_metrics,
            "binary_brier": binary_brier,
            "top4": top4,
            "ev_gate": ev_gate,
            "lcb_gate": lcb_gate,
        })

    # Random Top-K cùng exposure để neo kết quả ranking.
    random_picks = np.zeros_like(actual, dtype=np.int8)
    for day in range(n_test_days):
        random_picks[day, rng.choice(100, size=top_k, replace=False)] = 1
    random_daily_bets = random_picks.sum(axis=1)
    random_daily_hits = (random_picks * actual).sum(axis=1)
    random_roi, random_lower, random_upper, random_p_pos = _bootstrap_roi(random_daily_bets, random_daily_hits)

    report_path = Path(__file__).parent / "results" / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    first_date = pd.Timestamp(loader.df.iloc[start]["date"]).date()
    last_date = pd.Timestamp(loader.df.iloc[-1]["date"]).date()
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Nghiên cứu kỳ vọng số nháy — walk-forward\n\n")
        f.write(f"- **Cửa sổ**: {n_test_days} ngày ({first_date} đến {last_date})\n")
        f.write(f"- **Ngưỡng hòa vốn**: E[count] > {break_even:.6f} (= {COST_PER_BET:.0f}/{PAYOUT_PER_HIT:.0f})\n")
        f.write("- **Nhãn**: count 0–N cho từng số/ngày; không dùng nhãn nhị phân để tính EV.\n")
        f.write("- **Chống leakage**: mọi forecast ngày t chỉ dùng count trước ngày t.\n\n")

        f.write("## 1. Chất lượng dự báo số nháy\n\n")
        f.write("| Estimator | Mean μ | Mean actual | MAE | RMSE | Poisson deviance | Count calib error | Binary Brier* |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['name']} | {row['mean_expected_count']:.6f} | {row['mean_observed_count']:.6f} | "
                f"{row['count_mae']:.6f} | {row['count_rmse']:.6f} | {row['poisson_deviance']:.6f} | "
                f"{row['count_calibration_error']:.6f} | {row['binary_brier']:.6f} |\n"
            )
        f.write("\n\\* Binary Brier chỉ là kiểm tra phụ, với P(any) xấp xỉ `1-exp(-μ)`.\n\n")

        f.write("## 2. Ranking Top-K cố định — cùng exposure\n\n")
        f.write("| Estimator | Bets | Hits | ROI | CI95 | P(ROI>0) | Permutation p |\n")
        f.write("|---|---:|---:|---:|---|---:|---:|\n")
        for row in rows:
            s = row["top4"]
            f.write(
                f"| {row['name']} | {s['bets']} | {s['hits']} | {s['roi']:+.2%} | "
                f"[{s['lower']:+.2%}, {s['upper']:+.2%}] | {s['p_positive']:.1%} | {s['permutation_p']:.4f} |\n"
            )
        f.write(
            f"| random_seed42 | {int(random_daily_bets.sum())} | {int(random_daily_hits.sum())} | "
            f"{random_roi:+.2%} | [{random_lower:+.2%}, {random_upper:+.2%}] | {random_p_pos:.1%} | — |\n\n"
        )

        f.write("## 3. Cổng EV theo point estimate\n\n")
        f.write("| Estimator | Bets | Ngày cược | Hits | ROI | CI95 | Permutation p |\n")
        f.write("|---|---:|---:|---:|---:|---|---:|\n")
        for row in rows:
            s = row["ev_gate"]
            f.write(
                f"| {row['name']} | {s['bets']} | {s['bet_days']} | {s['hits']} | {s['roi']:+.2%} | "
                f"[{s['lower']:+.2%}, {s['upper']:+.2%}] | {s['permutation_p']:.4f} |\n"
            )

        f.write("\n## 4. Cổng bảo thủ: lower confidence bound của E[count] > hòa vốn\n\n")
        f.write("| Estimator | Bets | Ngày cược | Hits | ROI | CI95 | Permutation p |\n")
        f.write("|---|---:|---:|---:|---:|---|---:|\n")
        for row in rows:
            s = row["lcb_gate"]
            f.write(
                f"| {row['name']} | {s['bets']} | {s['bet_days']} | {s['hits']} | {s['roi']:+.2%} | "
                f"[{s['lower']:+.2%}, {s['upper']:+.2%}] | {s['permutation_p']:.4f} |\n"
            )

        if n_test_days >= 3 * 365:
            f.write("\n## 5. Độ ổn định theo ba epoch 365 ngày — cổng LCB\n\n")
            f.write("| Estimator | Epoch 1 ROI [CI95] | Epoch 2 ROI [CI95] | Epoch 3 ROI [CI95] |\n")
            f.write("|---|---|---|---|\n")
            for row in rows:
                epoch_cells = []
                forecasts = forecast_store[row["name"]]
                effective_ns = effective_store[row["name"]]
                for epoch in range(3):
                    lo = epoch * 365
                    hi = (epoch + 1) * 365
                    stats = _strategy_stats(
                        forecasts[lo:hi],
                        actual[lo:hi],
                        top_k,
                        threshold=break_even,
                        effective_ns=effective_ns[lo:hi],
                        run_permutation=False,
                    )
                    epoch_cells.append(
                        f"{stats['roi']:+.2%} [{stats['lower']:+.2%}, {stats['upper']:+.2%}]"
                    )
                f.write(f"| {row['name']} | {' | '.join(epoch_cells)} |\n")

        f.write("\n## 6. Diễn giải\n\n")
        f.write("- Chỉ coi ranking có tín hiệu khi vượt random/frequency và permutation p đủ nhỏ.\n")
        f.write("- Chỉ coi chiến lược có edge khi cận dưới bootstrap ROI 95% > 0.\n")
        f.write("- Point-estimate EV có thể cược vào nhiễu; cổng LCB gần đúng là kiểm tra bảo thủ, chưa điều chỉnh multiple testing.\n")
        f.write("- Đây là nghiên cứu trước holdout, không tự động thay model hoặc production policy.\n")
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nghiên cứu walk-forward E[số nháy]")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--report-name", default="count_expectation_research_365d.md")
    args = parser.parse_args()
    print(run_research(args.days, args.top_k, args.report_name))
