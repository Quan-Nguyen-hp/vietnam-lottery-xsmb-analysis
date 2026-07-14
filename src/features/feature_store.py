"""
FEATURE LAYER — src/features/feature_store.py
Hybrid FeatureStore: RAM cache (daily prediction) + Parquet snapshot (backtest).
Hỗ trợ versioning độc lập để đảm bảo reproducibility.

Cấu trúc thư mục:
    predictions/FeatureStore/
        current_version.txt         # "v1" — version đang dùng
        v1/
            metadata.json           # {version, features, extractor_versions, created_at}
            snapshots/
                2026-07-14.parquet  # 100 rows × N_features cols
        v2/
            metadata.json
            snapshots/
                ...
"""
from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseFeatureExtractor
from .delay_features import DelayFeatureExtractor
from .frequency_features import FrequencyFeatureExtractor
from .markov_features import MarkovFeatureExtractor
from .bayesian_features import BayesianFeatureExtractor
from .pair_features import PairFeatureExtractor
from .time_features import TimeFeatureExtractor
try:
    from ..evidence.store import EvidenceStore
except ImportError:
    from evidence.store import EvidenceStore

FEATURE_STORE_VERSION = "v1"


class FeatureStore:
    """
    Hybrid FeatureStore: tích hợp RAM cache và Parquet snapshot.

    - Prediction hằng ngày: tính → RAM cache → predict (< 1s)
    - Backtest lần đầu: tính → Parquet snapshot
    - Backtest lần sau: load Parquet (không tính lại)
    - Mỗi phiên bản feature schema có thư mục riêng (v1/, v2/, ...)
    """

    def __init__(
        self,
        store_root: Optional[Path] = None,
        version: str = FEATURE_STORE_VERSION,
        S_history: Optional[np.ndarray] = None,
        evidence_store: Optional[EvidenceStore] = None,
    ):
        if store_root is None:
            store_root = (
                Path(__file__).parent.parent.parent / "predictions" / "FeatureStore"
            )
        self._root = store_root
        self._version = version
        self._evidence_store = evidence_store or EvidenceStore()
        self._S = S_history  # may be None at load-only time

        # RAM cache: {date_str: DataFrame(100 rows)}
        self._ram_cache: dict[str, pd.DataFrame] = {}

        # Extractors that do NOT need S_history
        self._static_extractors: list[BaseFeatureExtractor] = [
            DelayFeatureExtractor(),
            FrequencyFeatureExtractor(),
            PairFeatureExtractor(),
            TimeFeatureExtractor(),
        ]
        # Extractors that need S_history — initialized lazily
        self._dynamic_extractors: list[BaseFeatureExtractor] = []
        self._S_hash: Optional[str] = None

        self._version_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_version_metadata()
        self._write_current_version()

    # ------------------------------------------------------------------ paths

    @property
    def _version_dir(self) -> Path:
        return self._root / self._version

    @property
    def _snapshots_dir(self) -> Path:
        return self._version_dir / "snapshots"

    def snapshot_path(self, date_str: str) -> Path:
        return self._snapshots_dir / f"{date_str}.parquet"

    # --------------------------------------------------------------- metadata

    def _ensure_version_metadata(self) -> None:
        meta_path = self._version_dir / "metadata.json"
        if not meta_path.exists():
            extractors = self._static_extractors + self._dynamic_extractors
            meta = {
                "feature_version": self._version,
                "extractor_versions": {e.name: e.version for e in extractors},
                "description": "XPIS Feature Layer snapshot",
                "created_at": datetime.now().isoformat(),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

    def _write_current_version(self) -> None:
        current = self._root / "current_version.txt"
        current.write_text(self._version, encoding="utf-8")

    # -------------------------------------------- extractor initialization

    def _init_dynamic_extractors(self, S: np.ndarray) -> None:
        """Khởi tạo extractors cần S_history (gọi lại khi S thay đổi)."""
        new_hash = hashlib.md5(S.tobytes()).hexdigest()[:8]
        if new_hash == self._S_hash:
            return
        self._S_hash = new_hash
        self._S = S
        self._dynamic_extractors = [
            MarkovFeatureExtractor(S),
            BayesianFeatureExtractor(S),
        ]

    # ------------------------------------------------------ build feature vector

    def build(
        self,
        df_evidence: pd.DataFrame,
        date_str: str,
        S: Optional[np.ndarray] = None,
        use_cache: bool = True,
        save_parquet: bool = True,
    ) -> pd.DataFrame:
        """
        Xây dựng FeatureVector (100 rows × N_features cols) cho một ngày.

        Priority:
        1. RAM cache (nếu đã tính trong session này)
        2. Parquet snapshot (nếu đã tính ở session trước)
        3. Tính mới từ Evidence

        Args:
            df_evidence:  Evidence DataFrame (100 rows) từ EvidenceStore.
            date_str:     'YYYY-MM-DD'
            S:            Ma trận nhị phân (N, 100) cho Markov & Bayesian.
            use_cache:    Có dùng RAM cache không.
            save_parquet: Có lưu Parquet snapshot không.

        Returns:
            DataFrame 100 rows × N_features cols.
            Luôn có cột 'number' và 'date' ở đầu.
        """
        # 1. RAM cache
        if use_cache and date_str in self._ram_cache:
            return self._ram_cache[date_str]

        # 2. Parquet cache
        parquet = self.snapshot_path(date_str)
        if parquet.exists():
            df = pd.read_parquet(parquet)
            if use_cache:
                self._ram_cache[date_str] = df
            return df

        # 3. Compute from scratch
        if S is not None:
            self._init_dynamic_extractors(S)

        all_extractors = self._static_extractors + self._dynamic_extractors
        feature_frames = []

        for ext in all_extractors:
            try:
                df_feat = ext.extract(df_evidence)
                feature_frames.append(df_feat)
            except Exception as e:
                print(f"[FeatureStore] Warning: {ext.name} failed: {e}")

        # Assemble
        df_all = pd.concat(
            [df_evidence[["number", "date"]]] + feature_frames, axis=1
        )

        # Save to parquet
        if save_parquet:
            parquet.parent.mkdir(parents=True, exist_ok=True)
            df_all.to_parquet(parquet, index=False)

        # RAM cache
        if use_cache:
            self._ram_cache[date_str] = df_all

        return df_all

    def load(self, date_str: str) -> Optional[pd.DataFrame]:
        """Đọc FeatureVector từ Parquet hoặc RAM cache."""
        if date_str in self._ram_cache:
            return self._ram_cache[date_str]
        path = self.snapshot_path(date_str)
        if path.exists():
            return pd.read_parquet(path)
        return None

    def exists(self, date_str: str) -> bool:
        return date_str in self._ram_cache or self.snapshot_path(date_str).exists()

    def list_dates(self) -> list[str]:
        dates = sorted([p.stem for p in self._snapshots_dir.glob("*.parquet")])
        return dates

    def clear_ram_cache(self) -> None:
        """Giải phóng RAM cache sau mỗi backtest day để tiết kiệm bộ nhớ."""
        self._ram_cache.clear()

    @property
    def version(self) -> str:
        return self._version

    def feature_count(self, sample_date: str) -> Optional[int]:
        df = self.load(sample_date)
        if df is None:
            return None
        return df.shape[1] - 2  # trừ 'number' và 'date'
