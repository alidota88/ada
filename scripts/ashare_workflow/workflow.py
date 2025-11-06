"""High-level orchestration helpers for the A-share workflow."""

from __future__ import annotations

from .analysis import run_analysis
from .config import ExperimentConfig
from .data import build_data_handler_config
from .environment import init_qlib_env
from .task import build_task_config
from .training import run_training

from qlib.workflow import R


def run_experiment(config: ExperimentConfig) -> None:
    """Execute the complete workflow using the provided configuration."""

    init_qlib_env(config.provider_uri)

    data_handler_config = build_data_handler_config(config.instruments, config.data_range)
    task_config = build_task_config(data_handler_config, config.segments, config.lightgbm)

    R.start(exp_name=config.exp_name)
    with R.start(experiment_name=config.exp_name):
        model, dataset, recorder = run_training(task_config)
        run_analysis(model, dataset, recorder, config.analysis_overrides)
