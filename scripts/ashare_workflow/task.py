"""Task-construction utilities for the LightGBM workflow."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .config import LightGBMConfig, SegmentConfig


def build_task_config(
    data_handler_config: Mapping[str, Any],
    segments: SegmentConfig,
    lightgbm: LightGBMConfig,
) -> Dict[str, Any]:
    """Create the Qlib task configuration used for training and evaluation."""

    return {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                **lightgbm.as_dict(),
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": dict(data_handler_config),
                },
                "segments": {
                    "train": segments.train,
                    "valid": segments.valid,
                    "test": segments.test,
                },
            },
        },
    }
