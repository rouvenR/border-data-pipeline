#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    parser.add_argument(
        "--random-search-iterations",
        type=int,
        help=(
            "Optional number of RandomizedSearchCV iterations for SVM hyperparameter "
            "tuning across kernels. When omitted, the script uses the existing "
            "linear-kernel grid search."
        ),
    )
    args = parser.parse_args()
    if args.random_search_iterations is not None and args.random_search_iterations < 1:
        raise SystemExit("--random-search-iterations must be at least 1.")
    return args


def parse_number(value: str) -> Optional[float]:
    if value is None:
        return None

    text = value.strip()
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        pass

    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        return float(match.group(1))
    return None


def parse_bool_flag(value: str) -> bool:
    return value.strip().lower() == "true" if value else False


def parse_cpu(config: str, csv_value: str) -> Optional[float]:
    csv_num = parse_number(csv_value)
    if csv_num is not None:
        return csv_num

    patterns = [
        r"(?:^|_)CPU([0-9]+(?:\.[0-9]+)?)(?:_|$)",
        r"(?:^|_)([0-9]+(?:\.[0-9]+)?)cpu(?:_|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, config, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def parse_ram_to_gib(config: str, csv_value: str) -> Optional[float]:
    token = csv_value.strip() if csv_value else ""
    if not token:
        patterns = [
            r"(?:^|_)RAM([^_]+)(?:_|$)",
            r"(?:^|_)([^_]+)ram(?:_|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, config, re.IGNORECASE)
            if match:
                token = match.group(1)
                break

    if not token:
        return None

    token = token.strip().lower()
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)([a-z]*)$", token)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2)
    factors_to_gib = {
        "": 1.0,
        "g": 1.0,
        "gb": 1.0,
        "gib": 1.0,
        "gi": 1.0,
        "m": 1.0 / 1024.0,
        "mb": 1.0 / 1024.0,
        "mib": 1.0 / 1024.0,
        "mi": 1.0 / 1024.0,
        "k": 1.0 / (1024.0 * 1024.0),
        "kb": 1.0 / (1024.0 * 1024.0),
        "kib": 1.0 / (1024.0 * 1024.0),
        "ki": 1.0 / (1024.0 * 1024.0),
        "b": 1.0 / (1024.0 * 1024.0 * 1024.0),
    }

    factor = factors_to_gib.get(unit)
    if factor is None:
        return None
    return value * factor


def split_run_tag(run_tag: str) -> Tuple[str, str]:
    if "__" in run_tag:
        name, config = run_tag.split("__", 1)
        return name, config
    return "", run_tag


def load_rows(metrics_path: Path) -> List[Dict[str, str]]:
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_feature_row(row: Dict[str, str], feature_columns: Sequence[str]) -> Optional[List[float]]:
    run_tag = row.get("run_tag", "")
    _, config = split_run_tag(run_tag)

    feature_values: List[float] = []
    for column in feature_columns:
        if column == "cpu":
            value = parse_cpu(config, row.get(column, ""))
        elif column == "ram":
            value = parse_ram_to_gib(config, row.get(column, ""))
        else:
            value = parse_number(row.get(column, ""))

        if value is None:
            return None
        feature_values.append(value)

    return feature_values


def build_dataset(
    rows: Sequence[Dict[str, str]],
    include_capacity_limited: bool,
) -> Tuple[List[List[float]], Dict[str, List[float]], int, int]:
    x_rows: List[List[float]] = []
    target_values: Dict[str, List[float]] = {target: [] for target in TARGET_COLUMNS}
    skipped_capacity_rows = 0
    skipped_missing_rows = 0

    for row in rows:
        is_capacity_limited = (
            parse_bool_flag(row.get("cpu_reached_maximum_capacity", ""))
            or parse_bool_flag(row.get("ram_reached_maximum_capacity", ""))
        )
        if is_capacity_limited and not include_capacity_limited:
            skipped_capacity_rows += 1
            continue

        feature_row = build_feature_row(row, FEATURE_COLUMNS)
        if feature_row is None:
            skipped_missing_rows += 1
            continue

        parsed_targets: List[float] = []
        for target in TARGET_COLUMNS:
            target_value = parse_number(row.get(target, ""))
            if target_value is None:
                parsed_targets = []
                break
            parsed_targets.append(target_value)

        if not parsed_targets:
            skipped_missing_rows += 1
            continue

        x_rows.append(feature_row)
        for index, target in enumerate(TARGET_COLUMNS):
            target_values[target].append(parsed_targets[index])

    return x_rows, target_values, skipped_capacity_rows, skipped_missing_rows


def get_linear_param_grid() -> Dict[str, List[float]]:
    return {
        "C": [0.1, 1.0, 10.0, 100.0],
        "epsilon": [0.01, 0.1, 0.5, 1.0],
    }


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


def get_kernel_param_distributions() -> List[Dict[str, List[Any]]]:
    return [
        {
            "kernel": ["linear"],
            "C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "epsilon": [0.001, 0.01, 0.1, 0.5, 1.0],
        },
        {
            "kernel": ["rbf"],
            "C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "epsilon": [0.001, 0.01, 0.1, 0.5, 1.0],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
        },
        {
            "kernel": ["poly"],
            "C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "epsilon": [0.001, 0.01, 0.1, 0.5, 1.0],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "degree": [2, 3, 4],
            "coef0": [0.0, 0.5, 1.0],
        },
        {
            "kernel": ["sigmoid"],
            "C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "epsilon": [0.001, 0.01, 0.1, 0.5, 1.0],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
            "coef0": [0.0, 0.5, 1.0],
        },
    ]


def train_svm_with_optional_random_search(
    x_rows_scaled: Any,
    y_values: List[float],
    n_splits: int,
    target: str,
    random_search_iterations: Optional[int],
) -> Tuple[Any, Dict[str, Any], Optional[Dict[str, Any]]]:
    try:
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # type: ignore
        from sklearn.svm import SVR  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "scikit-learn is required for train_svm_regression.py. "
            "Install it in the active environment and rerun the script."
        ) from exc

    if random_search_iterations is None:
        default_params = get_default_params_by_target(target)
        default_model = SVR(**default_params)
        default_model.fit(x_rows_scaled, y_values)
        return default_model, default_params, None

    random_search = RandomizedSearchCV(
        SVR(),
        param_distributions=get_kernel_param_distributions(),
        n_iter=random_search_iterations,
        cv=n_splits,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=42,
    )
    random_search.fit(x_rows_scaled, y_values)

    sampled_results: List[Tuple[float, Dict[str, Any]]] = []
    mean_test_scores = list(random_search.cv_results_["mean_test_score"])
    sampled_params = list(random_search.cv_results_["params"])
    for mean_score, params in zip(mean_test_scores, sampled_params):
        sampled_results.append((-float(mean_score), dict(params)))

    best_sample = min(sampled_results, key=lambda item: item[0])
    worst_sample = max(sampled_results, key=lambda item: item[0])
    summary = {
        "iterations": random_search_iterations,
        "best_sample_mae": best_sample[0],
        "best_sample_params": best_sample[1],
        "worst_sample_mae": worst_sample[0],
        "worst_sample_params": worst_sample[1],
    }

    return random_search.best_estimator_, dict(random_search.best_params_), summary


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
    random_search_iterations: Optional[int],
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

    best_model, best_params, random_search_summary = train_svm_with_optional_random_search(
        x_rows_scaled,
        y_values,
        n_splits,
        target,
        random_search_iterations,
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
        "random_search": random_search_summary,
    }


def fit_svm_model(
    x_rows: List[List[float]],
    y_values: List[float],
    target: str,
    random_search_iterations: Optional[int],
) -> Tuple[Any, Any, Dict[str, Any], Optional[Dict[str, Any]]]:
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
    model, best_params, random_search_summary = train_svm_with_optional_random_search(
        x_rows_scaled,
        y_values,
        min(5, len(x_rows)),
        target,
        random_search_iterations,
    )
    return scaler, model, best_params, random_search_summary


def evaluate_against_validation_set(
    training_x_rows: List[List[float]],
    training_y_values: List[float],
    validation_x_rows: List[List[float]],
    validation_y_values: List[float],
    target: str,
    random_search_iterations: Optional[int],
) -> Dict[str, Any]:
    if len(validation_x_rows) < 1:
        raise SystemExit("Not enough matched rows to evaluate validation set (need at least 1 row).")

    scaler, model, best_params, random_search_summary = fit_svm_model(
        training_x_rows,
        training_y_values,
        target,
        random_search_iterations,
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
        "random_search": random_search_summary,
    }


def main() -> None:
    args = parse_args()
    metrics_path = DEFAULT_METRICS_DIR / f"{args.timestamp}_metrics.csv"
    if not metrics_path.exists():
        raise SystemExit(f"Metrics file does not exist: {metrics_path}")

    rows = load_rows(metrics_path)
    if not rows:
        raise SystemExit(f"Metrics file is empty: {metrics_path}")

    sample_row = rows[0]
    required_columns = list(FEATURE_COLUMNS) + list(TARGET_COLUMNS)
    missing_columns = [column for column in required_columns if column not in sample_row]
    if missing_columns:
        raise SystemExit(
            "Missing required columns in metrics CSV: " + ", ".join(sorted(missing_columns))
        )

    x_rows, target_values, skipped_capacity_rows, skipped_missing_rows = build_dataset(
        rows,
        args.include_capacity_limited,
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

    validation_rows: List[Dict[str, str]] = []
    validation_x_rows: List[List[float]] = []
    validation_target_values: Dict[str, List[float]] = {target: [] for target in TARGET_COLUMNS}
    validation_skipped_capacity_rows = 0
    validation_skipped_missing_rows = 0
    if args.validate_against:
        validation_path = Path(args.validate_against)
        if not validation_path.exists():
            raise SystemExit(f"Validation metrics file does not exist: {validation_path}")

        validation_rows = load_rows(validation_path)
        if not validation_rows:
            raise SystemExit(f"Validation metrics file is empty: {validation_path}")

        validation_sample_row = validation_rows[0]
        validation_missing_columns = [
            column for column in required_columns if column not in validation_sample_row
        ]
        if validation_missing_columns:
            raise SystemExit(
                "Missing required columns in validation metrics CSV: "
                + ", ".join(sorted(validation_missing_columns))
            )

        (
            validation_x_rows,
            validation_target_values,
            validation_skipped_capacity_rows,
            validation_skipped_missing_rows,
        ) = build_dataset(validation_rows, args.include_capacity_limited)

    target_reports: Dict[str, Dict[str, Any]] = {}
    for target in TARGET_COLUMNS:
        if args.validate_against:
            target_reports[target] = evaluate_against_validation_set(
                x_rows,
                target_values[target],
                validation_x_rows,
                validation_target_values[target],
                target,
                args.random_search_iterations,
            )
        else:
            target_reports[target] = evaluate_target(
                x_rows,
                target_values[target],
                target,
                args.random_search_iterations,
            )

    print(f"Metrics file: {metrics_path}")
    if args.validate_against:
        print(f"Validation metrics file: {args.validate_against}")
    print(f"Random search iterations: {args.random_search_iterations}")
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

        random_search = report.get("random_search")
        if random_search is not None:
            print(f"    random_search_iterations={random_search['iterations']}")
            print(
                "    random_search_best_sample: "
                f"MAE={random_search['best_sample_mae']:.6f}, "
                f"params={json.dumps(random_search['best_sample_params'], sort_keys=True)}"
            )
            print(
                "    random_search_worst_sample: "
                f"MAE={random_search['worst_sample_mae']:.6f}, "
                f"params={json.dumps(random_search['worst_sample_params'], sort_keys=True)}"
            )


if __name__ == "__main__":
    main()
