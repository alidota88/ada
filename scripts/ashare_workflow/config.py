"""Configuration objects shared across the A-share workflow modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass
class SegmentConfig:
    """Time segments used to split the dataset."""

    train: tuple[str, str] = ("2008-01-01", "2017-12-31")
    valid: tuple[str, str] = ("2018-01-01", "2020-12-31")
    test: tuple[str, str] = ("2021-01-01", "2023-12-31")


@dataclass
class ExperimentConfig:
    """High level configuration for the LightGBM workflow."""

    exp_name: str = "ashare_lightgbm"
    provider_uri: Optional[str] = None
    instruments: str = "csi300"
    segments: SegmentConfig = field(default_factory=SegmentConfig)
    analysis_overrides: Mapping[str, Any] = field(default_factory=dict)
