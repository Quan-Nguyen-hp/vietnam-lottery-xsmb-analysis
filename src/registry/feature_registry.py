"""
REGISTRY MODULE — src/registry/feature_registry.py
Quản lý vòng đời và cấu hình của các đặc trưng (Feature Catalog).
Giúp audit và đảm bảo tính nhất quán của các đặc trưng đầu vào.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class FeatureRegistry:
    """
    Quản lý danh sách các đặc trưng trong hệ thống.
    Đọc dữ liệu từ predictions/feature_registry.json.
    """

    DEFAULT_JSON = Path(__file__).parent.parent.parent / "predictions" / "feature_registry.json"

    def __init__(self, json_path: Optional[Path] = None):
        self._path = json_path or self.DEFAULT_JSON
        self._data: dict = {}
        self.load()

    def load(self) -> None:
        """Tải dữ liệu từ file JSON đăng ký đặc trưng."""
        if not self._path.exists():
            self._data = {"version": "1.0", "last_updated": "", "features": {}}
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
        group: str,
        formula: str,
        description: str,
        version: str = "1.0",
        status: str = "active",
    ) -> None:
        """Đăng ký đặc trưng mới."""
        self._data["features"][name] = {
            "name": name,
            "group": group,
            "formula": formula,
            "description": description,
            "version": version,
            "status": status,
        }
        self._data["last_updated"] = Path(self._path).stat().st_mtime if self._path.exists() else ""
        self.save()

    def get_features(self, group: Optional[str] = None, status: str = "active") -> list[str]:
        """Lấy danh sách tên đặc trưng thỏa mãn bộ lọc."""
        features = []
        for name, meta in self._data.get("features", {}).items():
            if status and meta.get("status") != status:
                continue
            if group and meta.get("group") != group:
                continue
            features.append(name)
        return sorted(features)

    def is_valid(self, name: str) -> bool:
        """Kiểm tra đặc trưng có nằm trong registry và đang active không."""
        feat = self._data.get("features", {}).get(name)
        return feat is not None and feat.get("status") == "active"

    @property
    def version(self) -> str:
        return self._data.get("version", "1.0")
