"""
DECISION INTELLIGENCE — src/decision/edge_gate.py
Edge Gate: kiểm tra lợi thế thống kê trước khi cho phép BET.

Nguyên tắc:
- Chỉ cho phép BET khi bằng chứng thống kê đủ mạnh.
- Khi Edge Gate FAIL → force tất cả decisions sang PAPER_TRADE.
- State được lưu trong evaluation_policy.json để duy trì liên tục.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class EdgeGate:
    """
    Kiểm tra Edge Gate trước khi cho phép đặt cược.

    Edge Gate PASS khi:
    - bootstrap_roi_lower_95 > 0 (cận dưới ROI bootstrap 95% dương)

    Khi FAIL:
    - Tất cả quyết định BET → PAPER_TRADE
    - Hệ thống vẫn dự báo, nhưng KHÔNG khuyến nghị đặt tiền.
    """

    DEFAULT_POLICY_PATH = Path("predictions/evaluation_policy.json")

    def __init__(self, policy_path: Optional[Path] = None):
        self._path = policy_path or self.DEFAULT_POLICY_PATH
        self._policy: dict = {}
        self._gate: dict = {}
        self.load_state()

    def load_state(self) -> None:
        """Đọc Edge Gate state từ evaluation_policy.json."""
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                self._policy = json.load(f)
            self._gate = self._policy.get("edge_gate", {})
        else:
            self._gate = {"status": "PENDING", "roi_lower_95": None}

    def check(self) -> dict:
        """
        Kiểm tra Edge Gate hiện tại.

        Returns:
            dict với keys:
            - pass (bool): True nếu gate passes
            - status (str): "PASS" | "FAIL" | "PENDING"
            - action (str): "BET" | "PAPER_TRADE" | "WATCH"
            - roi_lower_95 (float | None): Cận dưới bootstrap 95%
            - last_evaluated (str | None): Ngày đánh giá cuối
            - evaluation_period (str | None): Kỳ đánh giá
            - required (str): Tiêu chí yêu cầu
        """
        status = self._gate.get("status", "PENDING")
        roi_lower = self._gate.get("roi_lower_95")

        if status == "PASS":
            return {
                "pass": True,
                "status": "PASS",
                "action": "BET",
                "roi_lower_95": roi_lower,
                "last_evaluated": self._gate.get("last_evaluated"),
                "evaluation_period": self._gate.get("evaluation_period"),
                "required": self._gate.get("required", "bootstrap_roi_lower_95_gt_zero"),
            }
        elif status == "FAIL":
            return {
                "pass": False,
                "status": "FAIL",
                "action": "PAPER_TRADE",
                "roi_lower_95": roi_lower,
                "last_evaluated": self._gate.get("last_evaluated"),
                "evaluation_period": self._gate.get("evaluation_period"),
                "required": self._gate.get("required", "bootstrap_roi_lower_95_gt_zero"),
            }
        else:
            # PENDING — chưa đủ dữ liệu để đánh giá
            return {
                "pass": False,
                "status": "PENDING",
                "action": "PAPER_TRADE",
                "roi_lower_95": None,
                "last_evaluated": None,
                "evaluation_period": None,
                "required": self._gate.get("required", "bootstrap_roi_lower_95_gt_zero"),
            }

    def update(
        self,
        roi_lower_95: float,
        evaluation_period: str,
        n_days: int,
        total_bets: int,
        total_hits: int,
        roi: float,
    ) -> dict:
        """
        Cập nhật Edge Gate state sau khi có kết quả backtest mới.

        Args:
            roi_lower_95: Cận dưới bootstrap 95% ROI
            evaluation_period: Mô tả kỳ (VD: "2025-07-12 to 2026-07-15")
            n_days: Số ngày backtest
            total_bets: Tổng lượt cược
            total_hits: Tổng nháy trúng
            roi: ROI tổng

        Returns:
            dict: gate state sau cập nhật
        """
        now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        passes = roi_lower_95 > 0.0

        self._gate = {
            "status": "PASS" if passes else "FAIL",
            "roi_lower_95": round(roi_lower_95, 6),
            "roi": round(roi, 6),
            "n_days": n_days,
            "total_bets": total_bets,
            "total_hits": total_hits,
            "last_evaluated": now_str,
            "evaluation_period": evaluation_period,
            "required": "bootstrap_roi_lower_95_gt_zero",
        }

        self._save_state()
        return self.check()

    def force_fail(self, reason: str = "manual_override") -> None:
        """Force Edge Gate về FAIL (dùng khi cần khóa manual)."""
        self._gate["status"] = "FAIL"
        self._gate["manual_override"] = reason
        self._gate["last_evaluated"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        self._save_state()

    def force_pending(self) -> None:
        """Reset Edge Gate về PENDING."""
        self._gate = {"status": "PENDING", "roi_lower_95": None}
        self._save_state()

    def _save_state(self) -> None:
        """Ghi Edge Gate state vào evaluation_policy.json."""
        self._policy["edge_gate"] = self._gate
        # Cập nhật mode dựa trên gate status
        if self._gate.get("status") == "PASS":
            self._policy["mode"] = "live"
        else:
            self._policy["mode"] = "paper_trade"

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._policy, f, indent=2, ensure_ascii=False)
