"""
REGISTRY MODULE — src/registry/belief_registry.py
Quản lý các tiên đề, niềm tin khoa học (Belief Knowledge Graph) được rút ra từ thực nghiệm.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


class BeliefRegistry:
    """
    Knowledge Engine quản lý các Belief định lượng của hệ thống.
    Đầu ra lưu tại predictions/belief_registry.json.
    """

    DEFAULT_JSON = Path(__file__).parent.parent.parent / "predictions" / "belief_registry.json"

    def __init__(self, json_path: Optional[Path] = None):
        self._path = json_path or self.DEFAULT_JSON
        self._data: dict = {}
        self.load()

    def load(self) -> None:
        """Tải dữ liệu từ file JSON đăng ký Belief."""
        if not self._path.exists():
            self._data = {"version": "1.0", "last_updated": "", "beliefs": {}}
            return
        with open(self._path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

    def save(self) -> None:
        """Ghi cấu hình hiện tại xuống file JSON."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def register(
        self,
        belief_id: str,
        title: str,
        hypothesis: str,
        status: str = "Experimental",
        owner: str = "Researcher",
        version: str = "1.0.0",
        tags: Optional[list[str]] = None,
        support_exps: Optional[list[str]] = None,
        contradict_exps: Optional[list[str]] = None,
        confidence: float = 0.50,
    ) -> None:
        """Đăng ký hoặc cập nhật Belief mới."""
        from datetime import datetime
        now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

        existing = self._data.get("beliefs", {}).get(belief_id, {})
        created_at = existing.get("created_at", now_str)

        self._data["beliefs"][belief_id] = {
            "id": belief_id,
            "version": version,
            "status": status,
            "created_at": created_at,
            "updated_at": now_str,
            "owner": owner,
            "tags": tags or [],
            "title": title,
            "hypothesis": hypothesis,
            "support": support_exps or [],
            "contradict": contradict_exps or [],
            "confidence": confidence,
            "evidence_count": len(support_exps or []) + len(contradict_exps or []),
        }
        self._data["last_updated"] = now_str
        self.save()

    def get_belief(self, belief_id: str) -> Optional[dict]:
        """Lấy thông tin của một Belief."""
        return self._data.get("beliefs", {}).get(belief_id)

    def get_active_beliefs(self, status: str = "Validated") -> list[dict]:
        """Lấy danh sách các Belief theo trạng thái."""
        beliefs = []
        for b in self._data.get("beliefs", {}).values():
            if status and b.get("status") != status:
                continue
            beliefs.append(b)
        return sorted(beliefs, key=lambda x: x.get("id", ""))
