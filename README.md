# A-share LightGBM Workflow Example

This repository contains a Python script that replicates the Qlib stock
selection workflow described in the tutorial, but it is configured for
the mainland China (A-share) market instead of Hong Kong equities.

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

The script will:

1. Initialize Qlib with the CN region configuration.
2. Build factors with the Alpha158 handler.
3. Train a LightGBM model on CSI 300 constituents.
4. Generate signals, run backtests, and produce analysis reports.

Experiment artifacts (signals, reports, charts) will be saved inside
Qlib's default experiment directory, typically under
`~/.qlib/qlib_data/cn_data/exp/`.
