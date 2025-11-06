"""Configuration objects shared across the A-share workflow modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass
class SegmentConfig:
    """Time segments used to split the dataset."""

    train: tuple[str, str] = ("2008-01-01", "2017-12-31")
    valid: tuple[str, str] = ("2018-01-01", "2020-12-31")
    test: tuple[str, str] = ("2021-01-01", "2023-12-31")


@dataclass
class DataRangeConfig:
    """Configuration of the data handler time ranges."""

    start_time: str = "2008-01-01"
    end_time: str = "2023-12-31"
    fit_start_time: str = "2008-01-01"
    fit_end_time: str = "2023-06-30"


@dataclass
class LightGBMConfig:
    """Configuration of the LightGBM model hyper-parameters."""

    loss: str = "mse"
    learning_rate: float = 0.05
    num_leaves: int = 128
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    lambda_l1: float = 200.0
    lambda_l2: float = 600.0
    max_depth: int = 8
    min_data_in_leaf: int = 50
    num_threads: int = 16

    def as_dict(self) -> Dict[str, Any]:
        """Return the configuration as a plain dictionary."""

        return {
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "num_threads": self.num_threads,
        }


@dataclass
class ExperimentConfig:
    """High level configuration for the LightGBM workflow."""

    exp_name: str = "ashare_lightgbm"
    provider_uri: Optional[str] = None
    instruments: str = "csi300"
    data_range: DataRangeConfig = field(default_factory=DataRangeConfig)
    segments: SegmentConfig = field(default_factory=SegmentConfig)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    analysis_overrides: Mapping[str, Any] = field(default_factory=dict)
