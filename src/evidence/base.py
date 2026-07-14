"""
EVIDENCE LAYER — src/evidence/base.py
Layer 2: Tạo bằng chứng thô. Không tính điểm, không dự báo.
EvidenceObject là snapshot trạng thái của một số tại một thời điểm.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class EvidenceObject:
    """
    Bằng chứng thô của một số loto tại một thời điểm.

    Nguyên tắc:
    - Chỉ chứa raw observation, không phải computed feature.
    - Mọi giá trị đều có nguồn gốc trực tiếp từ lịch sử.
    - Là đầu vào chuẩn cho Feature Layer.
    """
    # Định danh
    number: int                   # Số loto (0–99)
    date: str                     # Ngày dự đoán (YYYY-MM-DD)
    history_days: int             # Số ngày lịch sử được sử dụng

    # Delay / Lô Khan
    current_delay: int            # Số ngày kể từ lần xuất hiện cuối
    historical_delays: list[int]  # Toàn bộ lịch sử khoảng trễ

    # Tần suất xuất hiện thô (count trong mỗi window)
    count_3d: int
    count_7d: int
    count_14d: int
    count_30d: int
    count_60d: int
    count_90d: int
    count_120d: int
    count_180d: int
    count_365d: int
    count_all: int

    # Trạng thái Markov
    state_yesterday: int          # 0 hoặc 1 (có xuất hiện hôm qua không)
    state_day_before: int         # 0 hoặc 1 (ngày kia)

    # Đặc điểm số học
    head: int                     # Chữ số đầu (0–9)
    tail: int                     # Chữ số cuối (0–9)
    is_twin: bool                 # Số kép (00, 11, ..., 99)
    inverted: int                 # Số lộn (12 → 21)
    mirror: int                   # Số gương (12 → 67)

    # Ngữ cảnh hôm trước
    yesterday_actives: list[int]  # Các số đã về hôm qua
    yesterday_head_cam: list[int] # Đầu câm hôm qua (chưa về)
    yesterday_tail_cam: list[int] # Đuôi câm hôm qua

    # Thông tin thời gian
    weekday: int                  # 0=Mon ... 6=Sun
    day_of_month: int
    month: int
    is_weekend: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "EvidenceObject":
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "EvidenceObject":
        return cls.from_dict(json.loads(json_str))
