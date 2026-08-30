from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.count_v2.contracts import (
    CountForecast,
    CountHistory,
    CountOutcome,
    DatasetValidationError,
    EvaluationIntegrityError,
    RawDrawBatch,
)
from src.count_v2.dataset import count_matrix_from_raw, raw_draw_batch_from_frame
from src.count_v2.models import (
    CountModel,
    DirichletShrinkageMultinomialModel,
    EWMACountModel,
    RollingCountModel,
    UniformCountModel,
)
from src.data.loader import DataLoader
from src.evaluation.metrics import EvaluationMetrics

EVIDENCE_CLASS: str = "DEVELOPMENT / EXPLORATORY"
DEVELOPMENT_ARTIFACT_ROOT: Path = (
    ROOT_DIR / "research_artifacts" / "xpis_v2_count_first" / "development"
)


@dataclass(frozen=True)
class DevelopmentRunConfig:
    """Configuration for an exploratory count model research run."""

    test_rows: int
    rolling_window: int
    ewma_half_life: float
    dirichlet_window: int
    prior_strength: float
    top_k: int

    def __post_init__(self) -> None:
        for field_name, val in (
            ("test_rows", self.test_rows),
            ("rolling_window", self.rolling_window),
            ("dirichlet_window", self.dirichlet_window),
            ("top_k", self.top_k),
        ):
            if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

        if self.top_k > 100:
            raise ValueError(f"top_k must be <= 100, got {self.top_k}")

        for field_name, val in (
            ("ewma_half_life", self.ewma_half_life),
            ("prior_strength", self.prior_strength),
        ):
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or math.isnan(val)
                or math.isinf(val)
                or val <= 0
            ):
                raise ValueError(f"{field_name} must be a positive finite float")


@dataclass(frozen=True)
class DevelopmentForecastSet:
    """Matrix of walk-forward point forecasts and observed outcomes."""

    target_dates: np.ndarray
    observed_counts: np.ndarray
    model_identities: tuple[str, ...]
    expected_counts: np.ndarray


@dataclass(frozen=True)
class DevelopmentRunResult:
    """Complete results from an exploratory evaluation run."""

    forecast_set: DevelopmentForecastSet
    summary: dict[str, Any]


def load_legacy_raw_draw_batch(csv_path: Path | None = None) -> RawDrawBatch:
    """Ingest raw draws using DataLoader.df, discarding legacy binary matrix S."""
    loader = DataLoader(csv_path=csv_path) if csv_path is not None else DataLoader()
    loader.load()
    draw_columns = loader.prize_cols()
    return raw_draw_batch_from_frame(loader.df, draw_columns=draw_columns)


def build_models(config: DevelopmentRunConfig) -> tuple[CountModel, ...]:
    """Construct exactly the four canonical models in fixed order (B0, B1, M1, M2)."""
    return (
        UniformCountModel(),
        RollingCountModel(window=config.rolling_window),
        EWMACountModel(half_life=config.ewma_half_life),
        DirichletShrinkageMultinomialModel(
            window=config.dirichlet_window,
            prior_strength=config.prior_strength,
        ),
    )


def generate_walk_forward_forecasts(
    batch: RawDrawBatch,
    config: DevelopmentRunConfig,
) -> DevelopmentForecastSet:
    """Generate strictly prior-history walk-forward count forecasts for all models."""
    counts = count_matrix_from_raw(batch)
    total_rows = len(batch.dates)
    start = total_rows - config.test_rows
    minimum_history = max(1, config.rolling_window, config.dirichlet_window)
    if start < minimum_history:
        raise ValueError(
            f"test_rows ({config.test_rows}) leave insufficient prior draw rows ({start}) for configured windows ({minimum_history})"
        )

    models = build_models(config)
    expected_counts = np.empty((len(models), config.test_rows, 100), dtype=np.float64)

    for target_offset, target_index in enumerate(range(start, total_rows)):
        history = CountHistory(batch.dates[:target_index], counts[:target_index])
        target_date = str(batch.dates[target_index])
        for model_index, model in enumerate(models):
            forecast = model.predict_count(history, target_date)
            expected_counts[model_index, target_offset] = forecast.expected_count

    return DevelopmentForecastSet(
        target_dates=batch.dates[start:],
        observed_counts=counts[start:],
        model_identities=tuple(model.model_identity for model in models),
        expected_counts=expected_counts,
    )


def _economic_summary(
    mu: np.ndarray, observed: np.ndarray, top_k: int
) -> dict[str, float | int]:
    """Compute empirical payoff preserving draw multiplicity."""
    picks = np.stack([np.argsort(-row, kind="stable")[:top_k] for row in mu])
    hits = int(sum(observed[day, picks[day]].sum() for day in range(len(observed))))
    bets = int(len(observed) * top_k)
    total_cost = float(27.0 * bets)
    total_payout = float(99.0 * hits)
    pnl = total_payout - total_cost
    roi = float(pnl / total_cost) if total_cost > 0 else 0.0
    return {
        "top_k": top_k,
        "total_bets": bets,
        "total_hits": hits,
        "total_cost": total_cost,
        "total_payout": total_payout,
        "pnl": pnl,
        "roi": roi,
    }


def evaluate_forecast(
    forecast: CountForecast, outcome: CountOutcome
) -> dict[str, float]:
    """Evaluate a single point forecast against observed outcome."""
    if forecast.target_date != outcome.target_date:
        raise EvaluationIntegrityError("forecast and outcome target dates must match")
    return EvaluationMetrics(
        cost_per_bet=27.0,
        payout_per_hit=99.0,
    ).count_forecast_metrics(forecast.expected_count, outcome.observed_counts)


def evaluate_development_run(
    forecast_set: DevelopmentForecastSet,
    config: DevelopmentRunConfig,
) -> DevelopmentRunResult:
    """Aggregate global count metrics and economic summaries across all target dates."""
    models_summary: dict[str, dict[str, float | int]] = {}
    evaluator = EvaluationMetrics(cost_per_bet=27.0, payout_per_hit=99.0)

    for model_idx, model_identity in enumerate(forecast_set.model_identities):
        mu_all = forecast_set.expected_counts[model_idx]
        obs_all = forecast_set.observed_counts

        count_metrics = evaluator.count_forecast_metrics(mu_all, obs_all)
        econ_metrics = _economic_summary(mu_all, obs_all, config.top_k)

        combined: dict[str, float | int] = {
            "count_mae": count_metrics.get("count_mae", 0.0),
            "count_rmse": count_metrics.get("count_rmse", 0.0),
            "poisson_deviance": count_metrics.get("poisson_deviance", 0.0),
            "count_calibration_error": count_metrics.get("count_calibration_error", 0.0),
            "break_even_expected_count": count_metrics.get(
                "break_even_expected_count", 27.0 / 99.0
            ),
            **econ_metrics,
        }
        models_summary[model_identity] = combined

    summary = {
        "schema_version": "xpis_v2_count_first_development_v1",
        "evidence_class": EVIDENCE_CLASS,
        "target_rows": len(forecast_set.target_dates),
        "model_identities": list(forecast_set.model_identities),
        "models": models_summary,
    }
    return DevelopmentRunResult(forecast_set=forecast_set, summary=summary)


def write_development_artifacts(
    result: DevelopmentRunResult,
    config: DevelopmentRunConfig,
    *,
    run_id: str,
    artifact_root: Path = DEVELOPMENT_ARTIFACT_ROOT,
) -> Path:
    """Persist exploratory development summary.json and forecasts.npz."""
    run_dir = Path(artifact_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    target_dates = result.forecast_set.target_dates
    date_range = {
        "start": str(target_dates[0]) if len(target_dates) > 0 else "",
        "end": str(target_dates[-1]) if len(target_dates) > 0 else "",
    }

    summary_payload = {
        "schema_version": "xpis_v2_count_first_development_v1",
        "evidence_class": EVIDENCE_CLASS,
        "run_id": run_id,
        "config": {
            "test_rows": config.test_rows,
            "rolling_window": config.rolling_window,
            "ewma_half_life": config.ewma_half_life,
            "dirichlet_window": config.dirichlet_window,
            "prior_strength": config.prior_strength,
            "top_k": config.top_k,
        },
        "date_range": date_range,
        "target_rows": len(target_dates),
        "model_identities": list(result.forecast_set.model_identities),
        "models": result.summary.get("models", result.summary),
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    forecasts_path = run_dir / "forecasts.npz"
    np.savez_compressed(
        forecasts_path,
        target_dates=result.forecast_set.target_dates,
        observed_counts=result.forecast_set.observed_counts,
        model_identities=np.array(result.forecast_set.model_identities),
        expected_counts=result.forecast_set.expected_counts,
    )

    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for exploratory development backtests."""
    parser = argparse.ArgumentParser(description="XPIS v2 Count-First Development Runner")
    parser.add_argument(
        "--test-rows",
        type=int,
        required=True,
        help="Number of trailing draw rows to evaluate",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=30,
        help="Window for B1 rolling baseline",
    )
    parser.add_argument(
        "--ewma-half-life",
        type=float,
        default=14.0,
        help="Half-life for M1 EWMA model",
    )
    parser.add_argument(
        "--dirichlet-window",
        type=int,
        default=45,
        help="Window for M2 Dirichlet model",
    )
    parser.add_argument(
        "--prior-strength",
        type=float,
        default=20.0,
        help="Prior strength beta for M2 Dirichlet model",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K count candidates for economic summary",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Optional path to custom lottery CSV",
    )
    parser.add_argument(
        "--artifact-root",
        type=str,
        default=str(DEVELOPMENT_ARTIFACT_ROOT),
        help="Artifact root directory",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id",
    )

    args = parser.parse_args(argv)

    config = DevelopmentRunConfig(
        test_rows=args.test_rows,
        rolling_window=args.rolling_window,
        ewma_half_life=args.ewma_half_life,
        dirichlet_window=args.dirichlet_window,
        prior_strength=args.prior_strength,
        top_k=args.top_k,
    )

    csv_path = Path(args.csv_path) if args.csv_path else None
    batch = load_legacy_raw_draw_batch(csv_path)
    forecast_set = generate_walk_forward_forecasts(batch, config)
    result = evaluate_development_run(forecast_set, config)

    run_id = (
        args.run_id
        or f"DEV_RUN_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    artifact_root = Path(args.artifact_root)
    run_dir = write_development_artifacts(
        result, config, run_id=run_id, artifact_root=artifact_root
    )

    print(f"Artifacts written to: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
