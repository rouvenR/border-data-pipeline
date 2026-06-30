# Docs
This project contains python scripts to prepare and postprocess experiments with the TODO (link to main project) framework.

## Before experiments
`categorial_latin_hypercube_sampling.py`: Samples configurations with a wide spread for randomised experiments.

## After experiments
`combine_split_subscriber_files.py`: Experiments create hundreds of MQTT trace files. This script combines them into one file for further processing.

`visualize_individual_experiment.py`: Based on the MQTT traces (files starting by "conn_\*" and "e2e_\*") and hardware traces ("\*_stats.txt" files), creates a plot visualising an individual experiment.

`compute_throughput_metrics.py`: Based on the MQTT traces (files starting by "conn_\*" and "e2e_\*") and hardware traces ("\*_stats.txt" files), computes metrics (mean CPU utilisation, mean RAM utilisation, etc.) under "/inputs/training_data/\*_metrics.csv". These metrics are the basis for further training and validation.

## Additive Model
`train_throughput_regression.py`: Based on a metrics file ("/inputs/training_data/\*_metrics.csv"), creates a regression of an individual target value. This regression is saved as a json file ("outputs/plots/regressions/*.json") and a visualisation plot ("outputs/plots/regressions/*.png")

`predict_additive_model.py`: Combines several regressions ("outputs/plots/regressions/*.json") into a holistic additive prediction model.

## Machine Learning Models
`train_random_forest.py`: Based on a metrics file ("/inputs/training_data/\*_metrics.csv"), trains a random forest prediction model and applies cross-validation.

`train_svm_regression.py`: Based on a metrics file ("/inputs/training_data/\*_metrics.csv"), trains a support vector regression prediction model and applies cross-validation.