#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from __helpers.metrics_common import build_dataset, load_and_validate_metrics_rows

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_METRICS_DIR = BASE_DIR / "inputs" / "training_data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "regressions" / "svm"

FEATURE_COLUMNS = [
    "cpu",
    "ram",
    "message_size_qos0",
    "message_size_qos1",
    "message_size_qos2",
    "number_of_clients_qos0",
    "number_of_clients_qos1",
    "number_of_clients_qos2",
    "incoming_throughput_qos0",
    "incoming_throughput_qos1",
    "incoming_throughput_qos2",
]

TARGET_COLUMNS = [
    "received_throughput_mean",
    "cpu_mean_consumption",
    "ram_mean_consumption",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train SVM regressors from one metrics CSV identified by "
            "timestamp and report cross-validated MAE/RMSE."
        )
    )
    parser.add_argument(
        "--timestamp",
        required=True,
        help="Timestamp used to locate training_data/<TIMESTAMP>_metrics.csv.",
    )
    parser.add_argument(
        "--include-capacity-limited",
        action="store_true",
        help="Include rows where CPU or RAM reached maximum capacity.",
    )
    parser.add_argument(
        "--validate-against",
        help=(
            "Optional path to a second metrics CSV used as a holdout validation "
            "dataset for reporting MAE/RMSE/MaxError. When omitted, the script "
            "uses cross-validation on the training metrics file."
        ),
    )
    return parser.parse_args()


def get_default_params_by_target(target: str) -> Dict[str, Any]:
    defaults: Dict[str, Dict[str, Any]] = {
        "received_throughput_mean": {
            "kernel": "poly",
            "C": 1000.0,
            "epsilon": 0.1,
            "gamma": 0.1,
            "degree": 4,
            "coef0": 1.0,
        },
        "cpu_mean_consumption": {
            "kernel": "poly",
            "C": 10.0,
            "epsilon": 0.5,
            "gamma": "scale",
            "degree": 2,
            "coef0": 1.0,
        },
        "ram_mean_consumption": {
            "kernel": "poly",
            "C": 10.0,
            "epsilon": 0.1,
            "gamma": 0.1,
            "degree": 2,
            "coef0": 0.5,
        },
    }
    if target not in defaults:
        raise SystemExit(f"Unknown target column: {target}")
    return dict(defaults[target])


def train_svm_with_defaults(
    x_rows_scaled: Any,
    y_values: List[float],
    target: str,
) -> Tuple[Any, Dict[str, Any]]:
    try:
        from sklearn.svm import SVR  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "scikit-learn is required for train_svm_regression.py. "
            "Install it in the active environment and rerun the script."
        ) from exc

    default_params = get_default_params_by_target(target)
    default_model = SVR(**default_params)
    default_model.fit(x_rows_scaled, y_values)
    return default_model, default_params


def _mape(y_true, y_pred) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    total = sum(
        abs(predicted - actual) / abs(actual)
        for actual, predicted in zip(y_true, y_pred)
        if actual != 0.0
    )
    return (total / n) * 100.0


def evaluate_target(
    x_rows: List[List[float]],
    y_values: List[float],
    target: str,
) -> Dict[str, Any]:
    try:
        from sklearn.metrics import make_scorer, max_error  # type: ignore
        from sklearn.model_selection import KFold, cross_validate  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "scikit-learn is required for train_svm_regression.py. "
            "Install it in the active environment and rerun the script."
        ) from exc

    if len(x_rows) < 2:
        raise SystemExit("Not enough matched rows to evaluate a model (need at least 2 rows).")

    n_splits = min(5, len(x_rows))
    if n_splits < 2:
        raise SystemExit("Cross-validation requires at least 2 rows.")

    # Apply feature scaling
    scaler = StandardScaler()
    x_rows_scaled = scaler.fit_transform(x_rows)

    best_model, best_params = train_svm_with_defaults(
        x_rows_scaled,
        y_values,
        target,
    )

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        best_model,
        x_rows_scaled,
        y_values,
        cv=splitter,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "max_error": make_scorer(max_error, greater_is_better=False),
            "mape": make_scorer(_mape, greater_is_better=False),
        },
        n_jobs=-1,
    )

    mae_values = [-value for value in scores["test_mae"]]
    rmse_values = [-value for value in scores["test_rmse"]]
    max_error_values = [-value for value in scores["test_max_error"]]
    mape_values = [-value for value in scores["test_mape"]]

    return {
        "folds": float(n_splits),
        "mae_mean": sum(mae_values) / len(mae_values),
        "mae_min": min(mae_values),
        "mae_max": max(mae_values),
        "rmse_mean": sum(rmse_values) / len(rmse_values),
        "rmse_min": min(rmse_values),
        "rmse_max": max(rmse_values),
        "max_error_mean": sum(max_error_values) / len(max_error_values),
        "max_error_min": min(max_error_values),
        "max_error_max": max(max_error_values),
        "mape_mean": sum(mape_values) / len(mape_values),
        "mape_min": min(mape_values),
        "mape_max": max(mape_values),
        "best_kernel": str(best_params.get("kernel", "linear")),
        "best_c": float(best_params["C"]),
        "best_epsilon": float(best_params["epsilon"]),
        "best_gamma": best_params.get("gamma"),
        "best_degree": best_params.get("degree"),
        "best_coef0": best_params.get("coef0"),
    }


def fit_svm_model(
    x_rows: List[List[float]],
    y_values: List[float],
    target: str,
) -> Tuple[Any, Any, Dict[str, Any]]:
    try:
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "scikit-learn is required for train_svm_regression.py. "
            "Install it in the active environment and rerun the script."
        ) from exc

    if len(x_rows) < 2:
        raise SystemExit("Not enough matched rows to train an SVM model (need at least 2 rows).")

    scaler = StandardScaler()
    x_rows_scaled = scaler.fit_transform(x_rows)
    model, best_params = train_svm_with_defaults(
        x_rows_scaled,
        y_values,
        target,
    )
    return scaler, model, best_params


def evaluate_against_validation_set(
    training_x_rows: List[List[float]],
    training_y_values: List[float],
    validation_x_rows: List[List[float]],
    validation_y_values: List[float],
    target: str,
) -> Dict[str, Any]:
    if len(validation_x_rows) < 1:
        raise SystemExit("Not enough matched rows to evaluate validation set (need at least 1 row).")

    scaler, model, best_params = fit_svm_model(
        training_x_rows,
        training_y_values,
        target,
    )
    validation_x_scaled = scaler.transform(validation_x_rows)
    predictions = model.predict(validation_x_scaled)

    abs_errors = [abs(pred - actual) for pred, actual in zip(predictions, validation_y_values)]
    squared_errors = [(pred - actual) ** 2 for pred, actual in zip(predictions, validation_y_values)]
    n_val = len(validation_y_values)
    mape_mean = (
        sum(
            abs(pred - actual) / abs(actual)
            for pred, actual in zip(predictions, validation_y_values)
            if actual != 0.0
        )
        / n_val
        * 100.0
    ) if n_val > 0 else 0.0

    return {
        "rows": float(len(validation_x_rows)),
        "mae_mean": sum(abs_errors) / len(abs_errors),
        "mae_min": min(abs_errors),
        "mae_max": max(abs_errors),
        "rmse_mean": (sum(squared_errors) / len(squared_errors)) ** 0.5,
        "rmse_min": min(error ** 0.5 for error in squared_errors),
        "rmse_max": max(error ** 0.5 for error in squared_errors),
        "max_error_mean": sum(abs_errors) / len(abs_errors),
        "max_error_min": min(abs_errors),
        "max_error_max": max(abs_errors),
        "mape_mean": mape_mean,
        "best_kernel": str(best_params.get("kernel", "linear")),
        "best_c": float(best_params["C"]),
        "best_epsilon": float(best_params["epsilon"]),
        "best_gamma": best_params.get("gamma"),
        "best_degree": best_params.get("degree"),
        "best_coef0": best_params.get("coef0"),
    }


def main() -> None:
    
    # ==== INPUTS ====
    args = parse_args()
    metrics_path = DEFAULT_METRICS_DIR / f"{args.timestamp}_metrics.csv"
    required_columns = list(FEATURE_COLUMNS) + list(TARGET_COLUMNS)
    rows = load_and_validate_metrics_rows(
        metrics_path,
        required_columns,
        file_label="Metrics",
        columns_label="metrics CSV",
    )
    # ==== END INPUTS ====



    # ==== DATA PARSING ====
    x_rows, target_values, skipped_capacity_rows, skipped_missing_rows = build_dataset(
        rows,
        args.include_capacity_limited,
        FEATURE_COLUMNS,
        TARGET_COLUMNS,
    )

    if len(x_rows) < 2:
        raise SystemExit(
            "Not enough usable rows to train/evaluate SVM models after filtering."
        )

    # Warning for small sample size
    if len(x_rows) < 10:
        print(
            "WARNING: Small training set size ({} rows). "
            "Results may be unreliable.".format(len(x_rows)),
            flush=True,
        )
    # ==== END DATA PARSING ====


    # ==== VALIDATION DATA INPUT & PARSING ====
    validation_rows: List[Dict[str, str]] = []
    validation_x_rows: List[List[float]] = []
    validation_target_values: Dict[str, List[float]] = {target: [] for target in TARGET_COLUMNS}
    validation_skipped_capacity_rows = 0
    validation_skipped_missing_rows = 0
    if args.validate_against:
        validation_path = Path(args.validate_against)
        validation_rows = load_and_validate_metrics_rows(
            validation_path,
            required_columns,
            file_label="Validation metrics",
            columns_label="validation metrics CSV",
        )

        (
            validation_x_rows,
            validation_target_values,
            validation_skipped_capacity_rows,
            validation_skipped_missing_rows,
        ) = build_dataset(
            validation_rows,
            args.include_capacity_limited,
            FEATURE_COLUMNS,
            TARGET_COLUMNS,
        )
    # ==== END VALIDATION DATA INPUT & PARSING ====



    # ==== MODEL FITTING & VALIDATION ====
    target_reports: Dict[str, Dict[str, Any]] = {}
    for target in TARGET_COLUMNS:
        if args.validate_against:
            target_reports[target] = evaluate_against_validation_set(
                x_rows,
                target_values[target],
                validation_x_rows,
                validation_target_values[target],
                target,
            )
        else:
            target_reports[target] = evaluate_target(
                x_rows,
                target_values[target],
                target,
            )
    # ==== END MODEL FITTING & VALIDATION ====



    # ==== REPORTING ====
    print(f"Metrics file: {metrics_path}")
    if args.validate_against:
        print(f"Validation metrics file: {args.validate_against}")
    print(f"Include capacity-limited rows: {args.include_capacity_limited}")
    print(f"Feature columns: {', '.join(FEATURE_COLUMNS)}")
    print(f"Target columns: {', '.join(TARGET_COLUMNS)}")
    print(f"Total rows: {len(rows)}")
    print(f"Used rows: {len(x_rows)}")
    print(f"Skipped capacity-limited rows: {skipped_capacity_rows}")
    print(f"Skipped rows with missing values: {skipped_missing_rows}")
    if args.validate_against:
        print(f"Validation rows total: {len(validation_rows)}")
        print(f"Validation rows used: {len(validation_x_rows)}")
        print(f"Validation skipped capacity-limited rows: {validation_skipped_capacity_rows}")
        print(f"Validation skipped rows with missing values: {validation_skipped_missing_rows}")
        print("Validation results:")
    else:
        print("Cross-validation results:")
    for target in TARGET_COLUMNS:
        report = target_reports[target]
        if args.validate_against:
            print(
                f"  {target}: "
                f"MAE={report['mae_mean']:.6f}, "
                f"MAPE={report['mape_mean']:.2f}%, "
                f"RMSE={report['rmse_mean']:.6f}, "
                f"MaxError={report['max_error_max']:.6f}, "
                f"rows={int(report['rows'])}, "
                f"best_kernel={report['best_kernel']}, "
                f"best_C={report['best_c']:.4f}, "
                f"best_epsilon={report['best_epsilon']:.4f}"
            )
        else:
            print(
                f"  {target}: "
                f"MAE mean={report['mae_mean']:.6f}, "
                f"MAPE mean={report['mape_mean']:.2f}%, "
                f"RMSE mean={report['rmse_mean']:.6f}, "
                f"MaxError max={report['max_error_max']:.6f}, "
                f"folds={int(report['folds'])}, "
                f"best_kernel={report['best_kernel']}, "
                f"best_C={report['best_c']:.4f}, "
                f"best_epsilon={report['best_epsilon']:.4f}"
            )

        if report.get("best_gamma") is not None:
            print(f"    best_gamma={report['best_gamma']}")
        if report.get("best_degree") is not None:
            print(f"    best_degree={report['best_degree']}")
        if report.get("best_coef0") is not None:
            print(f"    best_coef0={report['best_coef0']}")
    # ==== END REPORTING ====


if __name__ == "__main__":
    main()
