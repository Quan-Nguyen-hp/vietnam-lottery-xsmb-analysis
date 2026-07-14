"""
EVIDENCE LAYER — src/evidence/store.py
Lưu trữ và đọc EvidenceObject dạng Parquet snapshot.
Evidence là bất biến: một khi đã tạo, không bao giờ thay đổi.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

EVIDENCE_VERSION = "v1.0"


class EvidenceStore:
    """
    Quản lý lưu trữ Evidence theo ngày dạng Parquet.

    Cấu trúc thư mục:
        EvidenceStore/
          metadata.json             # schema version, created_at
          YYYY-MM-DD/
            evidence.parquet        # 100 rows × raw evidence cols

    Nguyên tắc:
    - Một khi đã ghi, không bao giờ ghi đè (immutable).
    - Là nguồn dữ liệu duy nhất cho Feature Layer.
    """

    def __init__(self, store_root: Optional[Path] = None):
        if store_root is None:
            store_root = Path(__file__).parent.parent.parent / "predictions" / "EvidenceStore"
        self._root = store_root
        self._root.mkdir(parents=True, exist_ok=True)
        self._ensure_metadata()

    def _ensure_metadata(self) -> None:
        meta_path = self._root / "metadata.json"
        if not meta_path.exists():
            meta = {
                "evidence_version": EVIDENCE_VERSION,
                "description": "XPIS Evidence Layer - Raw observations per number per day",
                "schema": "100 rows × [number, date, + raw evidence columns]",
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

    def day_path(self, date_str: str) -> Path:
        """Trả về đường dẫn thư mục cho một ngày cụ thể."""
        return self._root / date_str

    def evidence_path(self, date_str: str) -> Path:
        return self.day_path(date_str) / "evidence.parquet"

    def exists(self, date_str: str) -> bool:
        """Kiểm tra xem Evidence của ngày này đã được lưu chưa."""
        return self.evidence_path(date_str).exists()

    def save(self, date_str: str, df_evidence: pd.DataFrame, overwrite: bool = False) -> Path:
        """
        Lưu DataFrame evidence (100 rows) vào Parquet.
        Mặc định không ghi đè nếu đã tồn tại (immutable principle).
        """
        path = self.evidence_path(date_str)
        if path.exists() and not overwrite:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        df_evidence.to_parquet(path, index=False)
        return path

    def load(self, date_str: str) -> Optional[pd.DataFrame]:
        """Đọc Evidence DataFrame cho một ngày. Trả None nếu chưa có."""
        path = self.evidence_path(date_str)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def list_dates(self) -> list[str]:
        """Danh sách tất cả các ngày đã có Evidence."""
        dates = []
        for d in self._root.iterdir():
            if d.is_dir() and (d / "evidence.parquet").exists():
                dates.append(d.name)
        return sorted(dates)
