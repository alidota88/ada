"""End-to-end Qlib workflow example for A-share (CN) market.

This script follows the workflow outlined in the provided tutorial but
uses the mainland China A-share data bundle shipped with Qlib instead of
Hong Kong equities. It performs the following steps:

1. Initialize Qlib with the CN binary data provider.
2. Configure the Alpha158 data handler to build model features/labels.
3. Define a LightGBM model task with train/valid/test segments in the
   A-share calendar.
4. Fit the model and record the artifacts.
5. Run signal and portfolio analysis to inspect the results.

Before running the script make sure you have prepared the CN data bundle
by executing:

    python scripts/dump_bin.py dump_all --csv-path ~/.qlib/csv_data/cn_data --qlib-dir ~/.qlib/qlib_data/cn_data --freq day

You can customise the csv-path parameter to the directory containing your
raw A-share CSV files.
"""

from copy import deepcopy
from pathlib import Path

import qlib
from qlib.config import C, REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SignalRecord

# Optional: enable SigAnaRecord if qlib>=0.8.0
try:  # pragma: no cover - optional dependency
    from qlib.workflow.record_temp import SigAnaRecord
except ImportError:  # pragma: no cover
    SigAnaRecord = None


def prepare_experiment(exp_name: str = "ashare_lightgbm") -> None:
    """Run the A-share LightGBM workflow experiment."""

    provider_uri = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")
    qlib.init(provider_uri=provider_uri, region=REG_CN, exp_manager=None)

    data_handler_config = {
        "start_time": "2008-01-01",
        "end_time": "2023-12-31",
        "fit_start_time": "2008-01-01",
        "fit_end_time": "2023-06-30",
        "instruments": "csi300",
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

    task = {
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
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": ("2008-01-01", "2017-12-31"),
                    "valid": ("2018-01-01", "2020-12-31"),
                    "test": ("2021-01-01", "2023-12-31"),
                },
            },
        },
    }

    R.start(exp_name=exp_name)
    with R.start(experiment_name=exp_name):
        model = init_instance_by_config(task["model"])
        dataset = init_instance_by_config(task["dataset"])
        model.fit(dataset.prepare("train"))
        recorder = R.get_recorder()
        recorder.save_objects(model=model, dataset=dataset)

        SignalRecord(model=model, dataset=dataset, recorder=recorder).generate()

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

        PortAnaRecord(recorder=recorder, config=analysis_config).generate()

        if SigAnaRecord is not None:
            SigAnaRecord(model=model, dataset=dataset, recorder=recorder).generate()


if __name__ == "__main__":
    prepare_experiment()
