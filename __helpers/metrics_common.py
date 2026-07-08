from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


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


def split_run_tag(run_tag: str) -> Tuple[str, str]:
    if "__" in run_tag:
        name, config = run_tag.split("__", 1)
        return name, config
    return "", run_tag


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


def load_csv_rows(metrics_path: Path) -> List[Dict[str, str]]:
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_and_validate_metrics_rows(
    metrics_path: Path,
    required_columns: Sequence[str],
    file_label: str,
    columns_label: str,
) -> List[Dict[str, str]]:
    if not metrics_path.exists():
        raise SystemExit(f"{file_label} file does not exist: {metrics_path}")

    rows = load_csv_rows(metrics_path)
    if not rows:
        raise SystemExit(f"{file_label} file is empty: {metrics_path}")

    sample_row = rows[0]
    missing_columns = [column for column in required_columns if column not in sample_row]
    if missing_columns:
        raise SystemExit(
            f"Missing required columns in {columns_label}: " + ", ".join(sorted(missing_columns))
        )

    return rows


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
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
) -> Tuple[List[List[float]], Dict[str, List[float]], int, int]:
    x_rows: List[List[float]] = []
    target_values: Dict[str, List[float]] = {target: [] for target in target_columns}
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

        feature_row = build_feature_row(row, feature_columns)
        if feature_row is None:
            skipped_missing_rows += 1
            continue

        parsed_targets: List[float] = []
        for target in target_columns:
            target_value = parse_number(row.get(target, ""))
            if target_value is None:
                parsed_targets = []
                break
            parsed_targets.append(target_value)

        if not parsed_targets:
            skipped_missing_rows += 1
            continue

        x_rows.append(feature_row)
        for index, target in enumerate(target_columns):
            target_values[target].append(parsed_targets[index])

    return x_rows, target_values, skipped_capacity_rows, skipped_missing_rows
