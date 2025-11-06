"""Task-construction utilities for the LightGBM workflow."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .config import SegmentConfig


def build_task_config(data_handler_config: Mapping[str, Any], segments: SegmentConfig) -> Dict[str, Any]:
    """Create the Qlib task configuration used for training and evaluation."""

    return {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "learning_rate": 0.05,
                "num_leaves": 128,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "lambda_l1": 200.0,
                "lambda_l2": 600.0,
                "max_depth": 8,
                "min_data_in_leaf": 50,
                "num_threads": 16,
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
