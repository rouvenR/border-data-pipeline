# Root Project
This project is part of the broader BORDER extension project. Refer to the [Orchestration Project](https://github.com/rouvenR/BORDER-grid5k-orchestration) for more context.

# Docs
This project contains python scripts to prepare and postprocess experiments with the extended BORDER framework.

## Before experiments
`categorial_latin_hypercube_sampling.py`: Samples configurations with a wide spread for randomised experiments.

```bash
python3 categorial_latin_hypercube_sampling.py --samples 50 --clients_qos0 "50,100,150" --throughput_qos0 "100,4375,8750,13125,17500" --size_qos0 "100,250,500,750,1000" --clients_qos1 "50,100,150" --throughput_qos1 "100,4375,8750,13125,17500" --size_qos1 "100,250,500,750,1000" --clients_qos2 "50,100,150" --throughput_qos2 "100,1250,2500,3750,5000" --size_qos2 "100,250,500,750,1000" --cpu "2,4,8,16"
```

Parameters:
- `--samples` (required): Number of samples to generate.
- `--clients_qos0` (required): Comma-separated category values for QoS 0 client count.
- `--throughput_qos0` (required): Comma-separated category values for QoS 0 target throughput.
- `--size_qos0` (required): Comma-separated category values for QoS 0 message size.
- `--clients_qos1` (required): Comma-separated category values for QoS 1 client count.
- `--throughput_qos1` (required): Comma-separated category values for QoS 1 target throughput.
- `--size_qos1` (required): Comma-separated category values for QoS 1 message size.
- `--clients_qos2` (required): Comma-separated category values for QoS 2 client count.
- `--throughput_qos2` (required): Comma-separated category values for QoS 2 target throughput.
- `--size_qos2` (required): Comma-separated category values for QoS 2 message size.
- `--cpu` (required): Comma-separated category values for CPU core allocation.
- `--seed` (optional): Random seed for reproducible sampling.
- `--output` (optional): Output CSV path. If omitted, samples are printed to stdout.

## After experiments
`combine_split_subscriber_files.py`: Experiments create hundreds of MQTT trace files. This script combines them into one file for further processing.

```bash
python3 combine_split_subscriber_files.py \
	--timestamp 20260512145503 \
	--input-dir inputs/result_data/experiments
```

Parameters:
- `--timestamp` (required): Base timestamp used to select split files (for example `20260512145503`).
- `--input-dir` (optional): Directory containing split files. Defaults to `inputs/result_data/experiments` relative to this project.

`visualize_individual_experiment.py`: Based on the MQTT traces (files starting by "conn_\*" and "e2e_\*") and hardware traces ("\*_stats.txt" files), creates a plot visualising an individual experiment.

```bash
python3 visualize_individual_experiment.py \
	--timestamp 20260512145503 \
	--pretty
```

Parameters:
- `--timestamp` (optional): Timestamp used to match files named like `*_<TIMESTAMP>*_stats.txt`. If omitted, the script uses all discoverable runs.
- `--pretty` (optional flag): Excludes `mn.sub.0` and `mn.pub.0` containers from the visualization.

`compute_throughput_metrics.py`: Based on the MQTT traces (files starting by "conn_\*" and "e2e_\*") and hardware traces ("\*_stats.txt" files), computes metrics (mean CPU utilisation, mean RAM utilisation, etc.) under "/inputs/training_data/\*_metrics.csv". These metrics are the basis for further training and validation.

```bash
python3 compute_throughput_metrics.py \
	--timestamp 20260512145503 \
	--broker-name jorammq
```

Parameters:
- `--timestamp` (required): Timestamp used to discover files matching `*_<TIMESTAMP>*_stats.txt`.
- `--broker-name` (required): Broker container base name (without prefix/suffix). Rows are filtered to `mn.<broker_name>0`.

## Additive Model
`train_throughput_regression.py`: Based on a metrics file ("/inputs/training_data/\*_metrics.csv"), creates a regression of an individual target value. This regression is saved as a json file ("outputs/plots/regressions/*.json") and a visualisation plot ("outputs/plots/regressions/*.png")

```bash
python3 train_throughput_regression.py \
	--timestamp 20260512145503 \
	--variable-column message_size \
	--target received_throughput \
	--metrics-dir inputs/training_data \
	--base-load 0.0
```

Parameters:
- `--timestamp` (required): Timestamp used to locate `training_data/<TIMESTAMP>_metrics.csv`.
- `--variable-column` (required): Feature column to regress, for example `message_size`, `number_of_messages`, or `quality_of_service`.
- `--target` (optional): Target metric. `received_throughput` maps to `received_throughput_mean`; `sent_throughput` maps to `sent_throughput_mean`. Default: `received_throughput`.
- `--metrics-dir` (optional): Directory containing `*_metrics.csv` files. Default: script training-data directory.
- `--base-load` (optional): Constant subtracted from model predictions before clipping to 0. Default: `0.0`.
- `--no-regression` (optional flag): Disables plotting of the regression line.

`predict_additive_model.py`: Combines several regressions ("outputs/plots/regressions/*.json") into a holistic additive prediction model.

```bash
python3 predict_additive_model.py \
	--model-dir "regressions/*.json" \
	--validate-against inputs/training_data/20260512145503_metrics.csv \
	--base-load 0.0
```

Parameters:
- `--model-dir` (optional): Glob pattern for component regression JSON files. Default: `regressions/*.json`.
- `--validate-against` (required): Path to a `*_metrics.csv` file used to compute validation metrics (MAE, RMSE, MaxError).
- `--include-capacity-limited` (optional flag): Includes rows where CPU or RAM reached maximum capacity.
- `--base-load` (optional): Baseline subtracted from each component output before clipping to 0. Default: `0.0`.

## Machine Learning Models
`train_random_forest.py`: Based on a metrics file ("/inputs/training_data/\*_metrics.csv"), trains a random forest prediction model and applies cross-validation.

```bash
python3 train_random_forest.py \
	--timestamp 20260512145503 \
	--validate-against inputs/training_data/20260512153000_metrics.csv \
	--random-search-iterations 50
```

Parameters:
- `--timestamp` (required): Timestamp used to locate `training_data/<TIMESTAMP>_metrics.csv`.
- `--include-capacity-limited` (optional flag): Includes rows where CPU or RAM reached maximum capacity.
- `--validate-against` (optional): Path to a second metrics CSV used as holdout validation. If omitted, the script uses cross-validation only.
- `--random-search-iterations` (optional): Number of `RandomizedSearchCV` iterations for tuning `n_estimators` and `max_features`.

`train_svm_regression.py`: Based on a metrics file ("/inputs/training_data/\*_metrics.csv"), trains a support vector regression prediction model and applies cross-validation.

```bash
python3 train_svm_regression.py \
	--timestamp 20260512145503 \
	--include-capacity-limited \
	--random-search-iterations 100
```

Parameters:
- `--timestamp` (required): Timestamp used to locate `training_data/<TIMESTAMP>_metrics.csv`.
- `--include-capacity-limited` (optional flag): Includes rows where CPU or RAM reached maximum capacity.
- `--validate-against` (optional): Path to a second metrics CSV used as holdout validation. If omitted, the script uses cross-validation only.
- `--random-search-iterations` (optional): Number of `RandomizedSearchCV` iterations for SVM hyperparameter tuning. Must be at least `1` when provided.