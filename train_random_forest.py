#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_METRICS_DIR = BASE_DIR / "inputs" / "training_data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "regressions" / "random_forest"

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
            "Train random forest regressors from one metrics CSV identified by "
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
) -> Dict[str, object]:
    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore
        from sklearn.metrics import make_scorer, max_error  # type: ignore
        from sklearn.model_selection import KFold, cross_validate  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "scikit-learn is required for train_random_forest.py. "
            "Install it in the active environment and rerun the script."
        ) from exc

    if len(x_rows) < 2:
        raise SystemExit("Not enough matched rows to evaluate a model (need at least 2 rows).")

    n_splits = min(5, len(x_rows))
    if n_splits < 2:
        raise SystemExit("Cross-validation requires at least 2 rows.")

    base_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = cross_validate(
        base_model,
        x_rows,
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

    report: Dict[str, object] = {
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
    }
    return report


def fit_random_forest(
    x_rows: List[List[float]],
    y_values: List[float],
):
    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "scikit-learn is required for train_random_forest.py. "
            "Install it in the active environment and rerun the script."
        ) from exc

    if len(x_rows) < 2:
        raise SystemExit("Not enough matched rows to train a model (need at least 2 rows).")

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_rows, y_values)
    return model


def evaluate_against_validation_set(
    training_x_rows: List[List[float]],
    training_y_values: List[float],
    validation_x_rows: List[List[float]],
    validation_y_values: List[float],
) -> Dict[str, object]:
    if len(validation_x_rows) < 1:
        raise SystemExit("Not enough matched rows to evaluate validation set (need at least 1 row).")

    model = fit_random_forest(
        training_x_rows,
        training_y_values,
    )
    predictions = model.predict(validation_x_rows)

    abs_errors = [abs(pred - actual) for pred, actual in zip(predictions, validation_y_values)]
    squared_errors = [(pred - actual) ** 2 for pred, actual in zip(predictions, validation_y_values)]
    max_errors = abs_errors
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

    report: Dict[str, object] = {
        "rows": float(len(validation_x_rows)),
        "mae_mean": sum(abs_errors) / len(abs_errors),
        "mae_min": min(abs_errors),
        "mae_max": max(abs_errors),
        "rmse_mean": (sum(squared_errors) / len(squared_errors)) ** 0.5,
        "rmse_min": min(error ** 0.5 for error in squared_errors),
        "rmse_max": max(error ** 0.5 for error in squared_errors),
        "max_error_mean": sum(max_errors) / len(max_errors),
        "max_error_min": min(max_errors),
        "max_error_max": max(max_errors),
        "mape_mean": mape_mean,
    }
    return report


def main() -> None:

    # ==== INPUTS ====
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
    # ==== END INPUTS ====



    # ==== DATA PARSING ====
    x_rows, target_values, skipped_capacity_rows, skipped_missing_rows = build_dataset(
        rows,
        args.include_capacity_limited,
    )

    if len(x_rows) < 2:
        raise SystemExit(
            "Not enough usable rows to train/evaluate random forest models after filtering."
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
    # ==== END VALIDATION DATA INPUT & PARSING ====



    # ==== MODEL FITTING & VALIDATION ====
    target_reports: Dict[str, Dict[str, float]] = {}
    for target in TARGET_COLUMNS:
        if args.validate_against:
            target_reports[target] = evaluate_against_validation_set(
                x_rows,
                target_values[target],
                validation_x_rows,
                validation_target_values[target],
            )
        else:
            target_reports[target] = evaluate_target(
                x_rows,
                target_values[target],
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
                f"rows={int(report['rows'])}"
            )
        else:
            print(
                f"  {target}: "
                f"MAE mean={report['mae_mean']:.6f}, "
                f"MAPE mean={report['mape_mean']:.2f}%, "
                f"RMSE mean={report['rmse_mean']:.6f}, "
                f"MaxError max={report['max_error_max']:.6f}, "
                f"folds={int(report['folds'])}"
            )
    # ==== END REPORTING ====


if __name__ == "__main__":
    main()