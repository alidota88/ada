"""Data-handler configuration helpers."""

from __future__ import annotations

from typing import Any, Dict


from .config import DataRangeConfig


def build_data_handler_config(instruments: str, data_range: DataRangeConfig) -> Dict[str, Any]:
    """Return the Alpha158 handler configuration for the specified instruments."""

    return {
        "start_time": data_range.start_time,
        "end_time": data_range.end_time,
        "fit_start_time": data_range.fit_start_time,
        "fit_end_time": data_range.fit_end_time,
        "instruments": instruments,
        "infer_processors": [
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature"}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ],
        "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
    }
