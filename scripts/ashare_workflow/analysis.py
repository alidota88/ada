"""Portfolio and signal analysis utilities."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, MutableMapping

from qlib.config import C
from qlib.workflow.record_temp import PortAnaRecord, SignalRecord

try:  # pragma: no cover - optional dependency
    from qlib.workflow.record_temp import SigAnaRecord
except ImportError:  # pragma: no cover
    SigAnaRecord = None


def build_analysis_config(overrides: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Return the portfolio analysis configuration with optional overrides."""

    analysis_config = deepcopy(C.port_analysis_config)
    analysis_config["strategy"]["kwargs"].update({"topk": 50, "n_drop": 5})
    analysis_config["backtest"]["kwargs"].update(
        {
            "start_time": "2021-01-01",
            "end_time": "2023-12-31",
            "benchmark": "SH000300",
            "account": 1e9,
        }
    )
    exchange_kwargs = analysis_config["backtest"]["kwargs"].setdefault("exchange", {})
    exchange_kwargs.update(
        {
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        }
    )

    for key, value in overrides.items():
        if isinstance(value, Mapping) and key in analysis_config:
            analysis_config[key] = {**analysis_config[key], **value}
        else:
            analysis_config[key] = value

    return analysis_config


def run_analysis(model, dataset, recorder, overrides: Mapping[str, Any]) -> None:
    """Generate signal, portfolio, and (optionally) signal analysis."""

    SignalRecord(model=model, dataset=dataset, recorder=recorder).generate()
    analysis_config = build_analysis_config(overrides)
    PortAnaRecord(recorder=recorder, config=analysis_config).generate()

    if SigAnaRecord is not None:
        SigAnaRecord(model=model, dataset=dataset, recorder=recorder).generate()
