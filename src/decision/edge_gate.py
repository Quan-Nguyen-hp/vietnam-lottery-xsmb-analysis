"""
DECISION INTELLIGENCE — src/decision/edge_gate.py
Edge Gate: kiểm tra lợi thế thống kê 2 tầng trước khi cho phép BET.

Kiến trúc 2 tầng (Dual-Tier Edge Gate):
- Tầng 1 (Tier 1 - Statistical Model Gate):
  * Kiểm định Brier Score & ECE: delta_brier_upper_95 < 0 (CI95 không chứa 0) và ece_score <= 0.0800.
  * Khi Tier 1 PASS → Mô hình chứng minh có tri thức thống kê thực sự, cho phép PAPER_TRADE_APPROVED.
- Tầng 2 (Tier 2 - Capital Execution Gate):
  * Kiểm định ROI: bootstrap_roi_lower_95 > 0.0.
  * Khi cả Tier 1 & Tier 2 PASS → Mới cho phép BET cược tiền thật.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class EdgeGate:
    """
    Kiểm tra Edge Gate 2 tầng (Dual-Tier) trước khi cho phép đặt cược.

    Tầng 1 (Statistical Model Gate):
    - delta_brier_upper_95 < 0.0 (Cận trên CI95 của ΔBrier < 0, tức CI95 không chứa 0)
    - ece_score <= 0.0800 (Mức hiệu chuẩn ECE chuẩn)

    Tầng 2 (Capital Execution Gate):
    - roi_lower_95 > 0.0 (Cận dưới ROI bootstrap 95% dương)
    - tier1_pass == True

    Trạng thái:
    - Tier 1 PASS, Tier 2 FAIL -> PAPER_TRADE_APPROVED (Tri thức thống kê OK, theo dõi giấy)
    - Tier 1 PASS, Tier 2 PASS -> BET (Cược tiền thật)
    - Tier 1 FAIL -> PAPER_TRADE (Chưa chứng minh được tín hiệu)
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
            self._gate = {
                "status": "PENDING",
                "tier1_status": "PENDING",
                "tier2_status": "PENDING",
                "roi_lower_95": None,
                "delta_brier_upper_95": None,
                "ece_score": None,
            }

    def check(self) -> dict:
        """
        Kiểm tra Edge Gate hiện tại.

        Returns:
            dict chứa thông tin trạng thái 2 tầng:
            - pass (bool): True nếu CẢ 2 tầng đều PASS (Live BET)
            - tier1_pass (bool): True nếu Tầng 1 PASS (Tri thức ML OK)
            - tier2_pass (bool): True nếu Tầng 2 PASS (Vốn OK)
            - status (str): "PASS" | "FAIL" | "PENDING" | "TIER1_PASS"
            - action (str): "BET" | "PAPER_TRADE" | "PAPER_TRADE_APPROVED"
        """
        status = self._gate.get("status", "PENDING")
        tier1_status = self._gate.get("tier1_status", "PENDING")
        tier2_status = self._gate.get("tier2_status", "PENDING")

        roi_lower = self._gate.get("roi_lower_95")
        delta_brier_upper = self._gate.get("delta_brier_upper_95")
        ece = self._gate.get("ece_score")
        brier = self._gate.get("brier_score")

        tier1_pass = tier1_status == "PASS"
        tier2_pass = tier2_status == "PASS" or status == "PASS"

        # Nếu chưa phân tầng trong JSON cũ nhưng status == "PASS"
        if status == "PASS" and not tier1_pass:
            tier1_pass = True

        overall_pass = tier1_pass and tier2_pass

        if overall_pass:
            action = "BET"
            effective_status = "PASS"
        elif tier1_pass:
            action = "PAPER_TRADE_APPROVED"
            effective_status = "TIER1_PASS"
        elif status == "FAIL" or tier1_status == "FAIL":
            action = "PAPER_TRADE"
            effective_status = "FAIL"
        else:
            action = "PAPER_TRADE"
            effective_status = "PENDING"

        return {
            "pass": overall_pass,
            "tier1_pass": tier1_pass,
            "tier2_pass": tier2_pass,
            "status": effective_status,
            "action": action,
            "roi_lower_95": roi_lower,
            "delta_brier_upper_95": delta_brier_upper,
            "ece_score": ece,
            "brier_score": brier,
            "last_evaluated": self._gate.get("last_evaluated"),
            "evaluation_period": self._gate.get("evaluation_period"),
            "required": self._gate.get("required", "tier1_delta_brier_ci95_gt_zero_and_tier2_roi_gt_zero"),
        }

    def update(
        self,
        roi_lower_95: float,
        evaluation_period: str,
        n_days: int,
        total_bets: int,
        total_hits: int,
        roi: float,
        delta_brier_upper_95: Optional[float] = None,
        ece_score: Optional[float] = None,
        brier_score: Optional[float] = None,
    ) -> dict:
        """
        Cập nhật trạng thái Edge Gate 2 tầng sau khi có kết quả đánh giá.

        Tier 1 PASS khi: delta_brier_upper_95 < 0.0 VÀ (ece_score is None hoặc ece_score <= 0.0800)
        Tier 2 PASS khi: roi_lower_95 > 0.0 VÀ Tier 1 PASS
        """
        now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Kiểm định Tầng 1 (Statistical Model Gate)
        if delta_brier_upper_95 is not None:
            t1_passes = (delta_brier_upper_95 < 0.0) and (ece_score is None or ece_score <= 0.0800)
            tier1_status = "PASS" if t1_passes else "FAIL"
        else:
            tier1_status = "PENDING"
            t1_passes = False

        # Kiểm định Tầng 2 (Capital Execution Gate)
        t2_passes = (roi_lower_95 > 0.0) and (t1_passes or delta_brier_upper_95 is None)
        tier2_status = "PASS" if t2_passes else "FAIL"

        overall_status = "PASS" if (t1_passes or delta_brier_upper_95 is None) and t2_passes else (
            "TIER1_PASS" if t1_passes else "FAIL"
        )

        self._gate = {
            "status": overall_status,
            "tier1_status": tier1_status,
            "tier2_status": tier2_status,
            "roi_lower_95": round(roi_lower_95, 6),
            "delta_brier_upper_95": round(delta_brier_upper_95, 6) if delta_brier_upper_95 is not None else None,
            "ece_score": round(ece_score, 6) if ece_score is not None else None,
            "brier_score": round(brier_score, 6) if brier_score is not None else None,
            "roi": round(roi, 6),
            "n_days": n_days,
            "total_bets": total_bets,
            "total_hits": total_hits,
            "last_evaluated": now_str,
            "evaluation_period": evaluation_period,
            "required": "tier1_delta_brier_ci95_gt_zero_and_tier2_roi_gt_zero",
        }

        self._save_state()
        return self.check()

    def force_fail(self, reason: str = "manual_override") -> None:
        """Force Edge Gate về FAIL (dùng khi cần khóa manual)."""
        self._gate["status"] = "FAIL"
        self._gate["tier1_status"] = "FAIL"
        self._gate["tier2_status"] = "FAIL"
        self._gate["manual_override"] = reason
        self._gate["last_evaluated"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        self._save_state()

    def force_pending(self) -> None:
        """Reset Edge Gate về PENDING."""
        self._gate = {
            "status": "PENDING",
            "tier1_status": "PENDING",
            "tier2_status": "PENDING",
            "roi_lower_95": None,
            "delta_brier_upper_95": None,
            "ece_score": None,
        }
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

