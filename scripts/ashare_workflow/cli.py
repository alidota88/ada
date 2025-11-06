"""Command-line interface helpers for the A-share workflow."""

from __future__ import annotations

import argparse

from .config import ExperimentConfig
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

    return ExperimentConfig(
        exp_name=args.exp_name,
        provider_uri=args.provider_uri,
        instruments=args.instruments,
        analysis_overrides=analysis_overrides,
    )


def main(argv: list[str] | None = None) -> None:
    """Execute the workflow using CLI arguments."""

    run_experiment(parse_args(argv))
