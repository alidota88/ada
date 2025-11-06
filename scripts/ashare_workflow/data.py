"""Data-handler configuration helpers."""

from __future__ import annotations

from typing import Any, Dict


def build_data_handler_config(instruments: str) -> Dict[str, Any]:
    """Return the Alpha158 handler configuration for the specified instruments."""

    return {
        "start_time": "2008-01-01",
        "end_time": "2023-12-31",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2023-06-30",
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
