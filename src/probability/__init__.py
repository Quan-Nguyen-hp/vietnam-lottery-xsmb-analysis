"""
PROBABILITY MODEL LAYER — src/probability/__init__.py
10 Models độc lập — mỗi model có thể backtest riêng.
"""
from .base import BaseProbabilityModel

# Models 1–8 (migrated from src/methods/)
from .max_delay import MaxDelayPredictor
from .conditional import ConditionalPredictor
from .markov import MarkovPredictor
from .momentum import MomentumPredictor
from .poisson import PoissonPredictor
from .repeat import RepeatPredictor
from .inverted_pairs import InvertedPairsPredictor
from .day_of_week import DayOfWeekPredictor

# Models 9–10 (new)
from .bayesian import BayesianPredictor
from .ewma_prob import EWMAPredictor

# Model 11 (Machine Learning Model)
from .lgb_model import LightGBMProbabilityModel


def get_all_models() -> list[BaseProbabilityModel]:
    """Trả về tất cả 11 models theo thứ tự chuẩn."""
    return [
        MaxDelayPredictor(),
        ConditionalPredictor(),
        MarkovPredictor(),
        MomentumPredictor(),
        PoissonPredictor(),
        RepeatPredictor(),
        InvertedPairsPredictor(),
        DayOfWeekPredictor(),
        BayesianPredictor(),
        EWMAPredictor(multi_scale=True),
        LightGBMProbabilityModel(),
    ]


__all__ = [
    "BaseProbabilityModel",
    "MaxDelayPredictor",
    "ConditionalPredictor",
    "MarkovPredictor",
    "MomentumPredictor",
    "PoissonPredictor",
    "RepeatPredictor",
    "InvertedPairsPredictor",
    "DayOfWeekPredictor",
    "BayesianPredictor",
    "EWMAPredictor",
    "LightGBMProbabilityModel",
    "get_all_models",
]

