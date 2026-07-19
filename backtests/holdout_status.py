"""Báo cáo trạng thái holdout prospective, không sửa policy và không tune model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))
POLICY_PATH = ROOT_DIR / "predictions" / "evaluation_policy.json"
LOG_PATH = ROOT_DIR / "predictions" / "prediction_log.json"
REPORT_PATH = ROOT_DIR / "backtests" / "results" / "locked_holdout_status.md"
STATUS_JSON_PATH = ROOT_DIR / "backtests" / "results" / "locked_holdout_status.json"


def _entry_date(entry: dict) -> pd.Timestamp | None:
    metadata = entry.get("pipeline_metadata", {})
    value = metadata.get("date", entry.get("date"))
    if not value:
        return None
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError):
        return None


def build_status() -> dict:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    holdout = policy["locked_holdout"]
    start = pd.Timestamp(holdout["start_date"])
    minimum_days = int(holdout["minimum_days"])

    from src.data.loader import DataLoader

    loader = DataLoader().load()
    available_dates = pd.to_datetime(loader.df["date"])
    holdout_dates = available_dates[available_dates >= start]
    log_entries = json.loads(LOG_PATH.read_text(encoding="utf-8")) if LOG_PATH.exists() else []
    dated_entries = [(idx, _entry_date(entry), entry) for idx, entry in enumerate(log_entries)]
    pre_holdout = [idx for idx, date, _ in dated_entries if date is not None and date < start]
    in_holdout = [
        (idx, date, entry)
        for idx, date, entry in dated_entries
        if date is not None and date >= start
    ]
    completed_in_holdout = [
        (idx, date)
        for idx, date, entry in in_holdout
        if entry.get("actual_results") is not None
    ]
    return {
        "start_date": start.date().isoformat(),
        "minimum_days": minimum_days,
        "available_days": int(len(holdout_dates)),
        "available_start": holdout_dates.iloc[0].date().isoformat() if len(holdout_dates) else None,
        "available_end": holdout_dates.iloc[-1].date().isoformat() if len(holdout_dates) else None,
        "ready": len(holdout_dates) >= minimum_days,
        "pre_holdout_log_entries": pre_holdout,
        "holdout_log_entries": len(in_holdout),
        "completed_holdout_log_entries": len(completed_in_holdout),
        "policy_mode": policy.get("mode"),
    }


def write_report(status: dict) -> Path:
    status_label = "READY" if status["ready"] else "PENDING"
    contamination = (
        "Có bản ghi log trước ngày bắt đầu holdout; các bản ghi này bị loại khỏi đánh giá."
        if status["pre_holdout_log_entries"]
        else "Không phát hiện bản ghi log trước ngày bắt đầu holdout."
    )
    lines = [
        "# Locked Holdout Status",
        "",
        f"- **Trạng thái**: **{status_label}**",
        f"- **Holdout bắt đầu**: {status['start_date']}",
        f"- **Yêu cầu tối thiểu**: {status['minimum_days']} ngày",
        f"- **Dữ liệu hiện có**: {status['available_days']} ngày "
        f"({status['available_start'] or 'chưa có'} → {status['available_end'] or 'chưa có'})",
        f"- **Prediction log trong holdout**: {status['holdout_log_entries']} bản ghi",
        f"- **Bản ghi đã có actual result**: {status['completed_holdout_log_entries']}",
        "",
        f"> {contamination}",
        "",
        "> Báo cáo này chỉ đọc dữ liệu. Nó không thay đổi model, threshold, top_k, policy hoặc Edge Gate.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    STATUS_JSON_PATH.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return REPORT_PATH


def main() -> int:
    status = build_status()
    report = write_report(status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(f"Đã ghi báo cáo: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
