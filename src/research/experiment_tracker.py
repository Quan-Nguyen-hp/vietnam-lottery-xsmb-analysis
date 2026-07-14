"""
RESEARCH LAYER — src/research/experiment_tracker.py
Hệ thống quản lý và theo dõi thực nghiệm (Experiment Tracking).
Tự động lưu log cấu hình, giả thuyết và kết quả đo đạc ra predictions/experiments/.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class ExperimentTracker:
    """
    Quản lý nhật ký thí nghiệm.
    Đầu ra lưu tại predictions/experiments/EXP_XXX_[name].json.
    """

    def __init__(self, exp_dir: Optional[Path] = None):
        self._dir = exp_dir or Path(__file__).parent.parent.parent / "predictions" / "experiments"
        self._dir.mkdir(parents=True, exist_ok=True)

    def log_experiment(
        self,
        exp_id: str,
        name: str,
        hypothesis: str,
        parameters: dict[str, Any],
        results: dict[str, Any],
        status: str,
        recommendation: str,
    ) -> Path:
        """Ghi nhận thí nghiệm mới."""
        payload = {
            "experiment_id": exp_id,
            "name": name,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "hypothesis": hypothesis,
            "parameters": parameters,
            "results": results,
            "status": status,
            "recommendation": recommendation,
        }
        
        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        filename = self._dir / f"{exp_id}_{safe_name}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            
        return filename

    def list_experiments(self) -> list[dict]:
        """Liệt kê toàn bộ các thí nghiệm đã thực hiện."""
        exps = []
        for p in self._dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    exps.append(json.load(f))
            except Exception:
                pass
        return sorted(exps, key=lambda x: x.get("experiment_id", ""))
