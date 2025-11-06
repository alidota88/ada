# A-share LightGBM Workflow Example

This repository contains a modular Python implementation of the Qlib stock
selection workflow for the mainland China (A-share) market.

## Prerequisites

1. Install [Qlib](https://github.com/microsoft/qlib) and its
   dependencies.
2. Prepare the A-share binary data bundle:

   ```bash
   python scripts/dump_bin.py dump_all \
       --csv-path ~/.qlib/csv_data/cn_data \
       --qlib-dir ~/.qlib/qlib_data/cn_data \
       --freq day
   ```

   Adjust the `--csv-path` argument to the folder containing your raw
   daily A-share CSV files.

## Running the workflow

Execute the script after the data bundle is ready:

```bash
python scripts/ashare_lightgbm.py
```

The script exposes a CLI so each stage of the workflow can be customised
in isolation. For example, to run on the CSI 500 universe with different
portfolio analysis parameters:

```bash
python scripts/ashare_lightgbm.py --instruments csi500 --analysis-topk 30 --analysis-drop 2
```

By default the program will:

1. Initialise Qlib with the CN region configuration.
2. Build factors with the Alpha158 handler.
3. Train a LightGBM model on CSI 300 constituents.
4. Generate signals, run backtests, and produce analysis reports.

Experiment artefacts (signals, reports, charts) will be saved inside
Qlib's default experiment directory, typically under
`~/.qlib/qlib_data/cn_data/exp/`.

## Reusing the modules

The reusable building blocks live under `scripts/ashare_workflow/` and
can be imported from Python to compose custom pipelines. Example:

```python
from ashare_workflow import (
    ExperimentConfig,
    build_data_handler_config,
    build_task_config,
    init_qlib_env,
    run_analysis,
    run_training,
)

config = ExperimentConfig(instruments="csi500")
init_qlib_env(config.provider_uri)
handler_cfg = build_data_handler_config(config.instruments)
task_cfg = build_task_config(handler_cfg, config.segments)
model, dataset, recorder = run_training(task_cfg)
run_analysis(model, dataset, recorder, config.analysis_overrides)
```

This layout allows further extension by swapping out the task, dataset,
or analysis modules without modifying the CLI entry point.
