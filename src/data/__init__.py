"""
DATA LAYER — src/data/__init__.py
Layer 1 của XPIS v1.0: Tầng duy nhất được phép đọc dữ liệu gốc.
Không chứa bất kỳ logic dự báo nào.
"""
from .loader import DataLoader

__all__ = ["DataLoader"]
