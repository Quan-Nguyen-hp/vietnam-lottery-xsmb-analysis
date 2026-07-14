"""
REGISTRY MODULE — src/registry/model_registry.py
Quản lý phiên bản, tham số huấn luyện và siêu tham số của từng mô hình trong XPIS v1.1.
Đảm bảo tính tái lập (reproducibility) trong thực nghiệm.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class ModelRegistry:
    """
    Quản lý danh sách các mô hình trong hệ thống.
    Đọc dữ liệu từ predictions/model_registry.json.
    """

    DEFAULT_JSON = Path(__file__).parent.parent.parent / "predictions" / "model_registry.json"

    def __init__(self, json_path: Optional[Path] = None):
        self._path = json_path or self.DEFAULT_JSON
        self._data: dict = {}
        self.load()

    def load(self) -> None:
        """Tải dữ liệu từ file JSON đăng ký mô hình."""
        if not self._path.exists():
            self._data = {"version": "1.0", "last_updated": "", "models": {}}
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
        name: str,
        model_type: str,
        version: str,
        description: str,
        parameters: Optional[dict] = None,
        status: str = "active",
    ) -> None:
        """Đăng ký mô hình mới."""
        from datetime import datetime
        self._data["models"][name] = {
            "name": name,
            "type": model_type,
            "version": version,
            "description": description,
            "parameters": parameters or {},
            "status": status,
        }
        self._data["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        self.save()

    def get_model_meta(self, name: str) -> Optional[dict]:
        """Lấy thông tin siêu dữ liệu của một mô hình cụ thể."""
        return self._data.get("models", {}).get(name)

    def get_active_models(self, model_type: Optional[str] = None) -> list[str]:
        """Lấy danh sách các mô hình đang hoạt động."""
        models = []
        for name, meta in self._data.get("models", {}).items():
            if meta.get("status") != "active":
                continue
            if model_type and meta.get("type") != model_type:
                continue
            models.append(name)
        return sorted(models)

    @property
    def version(self) -> str:
        return self._data.get("version", "1.0")
