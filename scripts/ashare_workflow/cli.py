"""Command-line interface helpers for the A-share workflow."""

from __future__ import annotations

import argparse

from .config import DataRangeConfig, ExperimentConfig, LightGBMConfig, SegmentConfig
from .workflow import run_experiment


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the workflow CLI."""

    parser = argparse.ArgumentParser(description="Run the Qlib LightGBM workflow on A-share data.")
    parser.add_argument("--exp-name", default="ashare_lightgbm", help="Name of the Qlib experiment to create.")
    parser.add_argument(
        "--provider-uri",
        default=None,
        help="Path to the prepared Qlib CN data directory (defaults to ~/.qlib/qlib_data/cn_data).",
    )
    parser.add_argument("--instruments", default="csi300", help="Instrument universe to train on (e.g. csi500).")
    parser.add_argument("--data-start", default="2008-01-01", help="Start date for the data handler (YYYY-MM-DD).")
    parser.add_argument("--data-end", default="2023-12-31", help="End date for the data handler (YYYY-MM-DD).")
    parser.add_argument("--fit-start", default="2008-01-01", help="Start date used for data fitting (YYYY-MM-DD).")
    parser.add_argument("--fit-end", default="2023-06-30", help="End date used for data fitting (YYYY-MM-DD).")
    parser.add_argument("--train-start", default="2008-01-01", help="Training segment start date (YYYY-MM-DD).")
    parser.add_argument("--train-end", default="2017-12-31", help="Training segment end date (YYYY-MM-DD).")
    parser.add_argument("--valid-start", default="2018-01-01", help="Validation segment start date (YYYY-MM-DD).")
    parser.add_argument("--valid-end", default="2020-12-31", help="Validation segment end date (YYYY-MM-DD).")
    parser.add_argument("--test-start", default="2021-01-01", help="Test segment start date (YYYY-MM-DD).")
    parser.add_argument("--test-end", default="2023-12-31", help="Test segment end date (YYYY-MM-DD).")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="LightGBM learning rate.")
    parser.add_argument("--num-leaves", type=int, default=128, help="LightGBM number of leaves.")
    parser.add_argument("--max-depth", type=int, default=8, help="LightGBM tree max depth.")
    parser.add_argument("--feature-fraction", type=float, default=0.8, help="LightGBM feature fraction.")
    parser.add_argument("--bagging-fraction", type=float, default=0.8, help="LightGBM bagging fraction.")
    parser.add_argument("--bagging-freq", type=int, default=5, help="LightGBM bagging frequency.")
    parser.add_argument("--lambda-l1", type=float, default=200.0, help="LightGBM L1 regularisation strength.")
    parser.add_argument("--lambda-l2", type=float, default=600.0, help="LightGBM L2 regularisation strength.")
    parser.add_argument("--min-data-in-leaf", type=int, default=50, help="LightGBM minimum data in leaf nodes.")
    parser.add_argument("--num-threads", type=int, default=16, help="Number of threads used by LightGBM.")
    parser.add_argument(
        "--analysis-topk",
        type=int,
        default=50,
        help="Override the topk parameter used by the portfolio analysis strategy.",
    )
    parser.add_argument(
        "--analysis-drop",
        type=int,
        default=5,
        help="Override the n_drop parameter used by the portfolio analysis strategy.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    """Parse CLI arguments to build an :class:`ExperimentConfig`."""

    parser = build_parser()
    args = parser.parse_args(argv)
    analysis_overrides = {
        "strategy": {"kwargs": {"topk": args.analysis_topk, "n_drop": args.analysis_drop}}
    }

    segments = SegmentConfig(
        train=(args.train_start, args.train_end),
        valid=(args.valid_start, args.valid_end),
        test=(args.test_start, args.test_end),
    )

    data_range = DataRangeConfig(
        start_time=args.data_start,
        end_time=args.data_end,
        fit_start_time=args.fit_start,
        fit_end_time=args.fit_end,
    )

    lightgbm = LightGBMConfig(
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        min_data_in_leaf=args.min_data_in_leaf,
        num_threads=args.num_threads,
    )

    return ExperimentConfig(
        exp_name=args.exp_name,
        provider_uri=args.provider_uri,
        instruments=args.instruments,
        data_range=data_range,
        segments=segments,
        lightgbm=lightgbm,
        analysis_overrides=analysis_overrides,
    )


def main(argv: list[str] | None = None) -> None:
    """Execute the workflow using CLI arguments."""

    run_experiment(parse_args(argv))
