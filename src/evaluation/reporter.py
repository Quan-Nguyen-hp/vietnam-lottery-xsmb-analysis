"""
EVALUATION LAYER — src/evaluation/reporter.py
Tạo báo cáo Daily/Weekly/Monthly dạng Markdown.
Mỗi báo cáo ghi rõ version của từng layer để đảm bảo reproducibility.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .metrics import EvaluationMetrics


class ReportGenerator:
    """
    Tạo báo cáo đánh giá XPIS dạng Markdown.
    Hỗ trợ Daily, Weekly, Monthly reports.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        odds: float = 70.0,
    ):
        self._output_dir = output_dir or Path("backtests/results")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._metrics = EvaluationMetrics(odds=odds)

    def daily_report(
        self,
        date: str,
        decisions: list[dict],
        actual_numbers: list[int],
        meta_proba: np.ndarray,
        version_info: Optional[dict] = None,
    ) -> str:
        """Tạo báo cáo ngày."""
        hits = [d["number"] for d in decisions if int(d["number"]) in actual_numbers]
        hit = len(hits) > 0

        version_info = version_info or {}
        ev_ver = version_info.get("evidence_version", "v1.0")
        ft_ver = version_info.get("feature_version", "v1")
        mo_ver = version_info.get("model_version", "v1.0")
        de_ver = version_info.get("decision_version", "v1.0")

        top_numbers = [int(d["number"]) for d in decisions[:10]]
        top_proba = [float(meta_proba[n]) for n in top_numbers]

        lines = [
            f"# Báo cáo ngày {date}",
            f"",
            f"## Versioning",
            f"| Layer | Version |",
            f"|---|---|",
            f"| Evidence | {ev_ver} |",
            f"| Feature | {ft_ver} |",
            f"| Model | {mo_ver} |",
            f"| Decision | {de_ver} |",
            f"",
            f"## Kết quả",
            f"- **Hit**: {'✅ CÓ' if hit else '❌ KHÔNG'}",
            f"- **Trúng**: {[f'{n:02d}' for n in hits] or 'Không có'}",
            f"- **Thực tế ra**: {[f'{n:02d}' for n in sorted(actual_numbers)]}",
            f"",
            f"## Dự báo",
            f"| Số | Xác suất | Confidence | Action |",
            f"|---|---|---|---|",
        ]

        for d in decisions[:10]:
            n = int(d["number"])
            lines.append(
                f"| {n:02d} | {d.get('probability', 0):.3f} | "
                f"{d.get('confidence', 0):.3f} | {d.get('action', 'N/A')} |"
            )

        report = "\n".join(lines)
        return report

    def summary_report(
        self,
        results: pd.DataFrame,
        title: str = "XPIS Backtest Summary",
        version_info: Optional[dict] = None,
    ) -> Path:
        """
        Tạo báo cáo tổng hợp dạng Markdown.

        Args:
            results: DataFrame với cột: date, hit, n_bets, n_hits, daily_pnl
            title: Tiêu đề báo cáo
        """
        metrics = self._metrics.compute_full(results)
        version_info = version_info or {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self._output_dir / f"xpis_report_{timestamp}.md"

        lines = [
            f"# {title}",
            f"",
            f"**Thời gian tạo**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## Versioning",
            f"| Layer | Version |",
            f"|---|---|",
        ]
        for k, v in version_info.items():
            lines.append(f"| {k} | {v} |")

        lines += [
            f"",
            f"## Kết quả tổng hợp",
            f"| Metric | Giá trị |",
            f"|---|---|",
            f"| Số ngày backtest | {metrics['n_days']} |",
            f"| Hit Rate | {metrics['hit_rate']:.1%} |",
            f"| ROI | {metrics['roi']:+.1%} |",
            f"| Tổng số lần đặt | {metrics['total_bets']} |",
            f"| Tổng số lần trúng | {metrics['total_hits']} |",
        ]

        if "brier_score" in metrics:
            lines.append(f"| Brier Score | {metrics['brier_score']:.4f} |")
            lines.append(f"| Log Loss | {metrics['log_loss']:.4f} |")

        if "max_drawdown" in metrics:
            lines.append(f"| Max Drawdown | {metrics['max_drawdown']:.1%} |")

        lines.append(f"| Chuỗi thắng dài nhất | {metrics['max_win_streak']} ngày |")

        report = "\n".join(lines)
        filename.write_text(report, encoding="utf-8")
        return filename
