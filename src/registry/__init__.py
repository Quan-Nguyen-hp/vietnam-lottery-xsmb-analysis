"""
REGISTRY MODULE — src/registry
Quản lý Feature Registry và Model Registry cho hệ thống XPIS v1.1.
"""
from .feature_registry import FeatureRegistry
from .model_registry import ModelRegistry
from .belief_registry import BeliefRegistry

__all__ = ["FeatureRegistry", "ModelRegistry", "BeliefRegistry"]
