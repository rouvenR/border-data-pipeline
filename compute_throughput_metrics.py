#!/usr/bin/env python3

import argparse
import csv
import re
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "inputs/result_data"
DEFAULT_EXPERIMENTS_DIR = DEFAULT_INPUT_DIR / "experiments"
DEFAULT_OUTPUT_DIR = BASE_DIR / "inputs/training_data"
N_CLIENTS_MULTIPLIER = 5  # Multiplier to estimate total clients from qos0 clients, based on experiment design.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute throughput metrics in the active window and write one CSV "
            "per timestamp to inputs/training_data/{TIMESTAMP}_metrics.csv"
        )
    )
    parser.add_argument(
        "--timestamp",
        required=True,
        help="Timestamp to find all matching files: *_{TIMESTAMP}*_stats.txt",
    )
    parser.add_argument(
        "--broker-name",
        required=True,
        help="Broker container name (without prefix/suffix). Stats rows are filtered to 'mn.<broker_name>0'.",
    )
    return parser.parse_args()


def discover_run_tags(timestamp: str) -> List[str]:
    files = sorted({path for path in DEFAULT_INPUT_DIR.glob(f"{timestamp}_*_stats.txt")})
    if not files:
        raise SystemExit(f"No files found in {DEFAULT_INPUT_DIR} for timestamp: {timestamp}")

    run_tags: List[str] = []
    for path in files:
        stem = path.stem
        run_tag = stem.replace("_stats", "")
        run_tags.append(run_tag)

    return run_tags


FIELDNAMES = [
    "run_tag",
    "number_of_clients_qos0",
    "delay_qos0",
    "number_of_messages_qos0",
    "message_size_qos0",
    "number_of_clients_qos1",
    "delay_qos1",
    "number_of_messages_qos1",
    "message_size_qos1",
    "number_of_clients_qos2",
    "delay_qos2",
    "number_of_messages_qos2",
    "message_size_qos2",
    "cpu",
    "ram",
    "active_window_start",
    "active_window_end",
    "active_window_seconds",
    "throughput_window_bins",
    "sent_throughput_mean",
    "sent_throughput_median",
    "received_throughput_mean",
    "received_throughput_completion_percentage",
    "received_throughput_median",
    "message_loss",
    "malformed_e2e_rows_skipped",
    "cpu_mean_consumption",
    "ram_mean_consumption",
    "cpu_reached_maximum_capacity",
    "ram_reached_maximum_capacity",
    "incoming_throughput_qos0",
    "incoming_throughput_qos1",
    "incoming_throughput_qos2",
    "cpu_factor_compared_to_min",
    "ram_factor_compared_to_min",
    "throughput_reached_maximum_capacity",
    "message_loss_threshold_exceeded",
]


def parse_input_tokens(run_tag: str) -> Dict[str, str]:
    # Parse only the suffix after "__" as requested.
    parse_segment = run_tag.split("__", 1)[1] if "__" in run_tag else run_tag

    # Match token forms like 10d, 6000m, 1c, 0q, 100s, cpu2/2cpu, ram1g/1gram,
    # and new per-qos tokens like C010_D020_M015000_S0100.
    number = r"[0-9]+(?:\.[0-9]+)?"

    def match(pattern: str) -> str:
        found = re.search(pattern, parse_segment, re.IGNORECASE)
        return found.group(1) if found else ""

    def match_any(pattern: str) -> str:
        found = re.search(pattern, parse_segment, re.IGNORECASE)
        if not found:
            return ""
        for group in found.groups():
            if group:
                return group
        return ""

    return {
        "number_of_clients": match_any(rf"(?:^|_)({number})c(?:_|$)|(?:^|_)c({number})(?:_|$)"),
        "delay": match_any(rf"(?:^|_)({number})d(?:_|$)|(?:^|_)d({number})(?:_|$)"),
        "number_of_messages": match_any(rf"(?:^|_)({number})m(?:_|$)|(?:^|_)m({number})(?:_|$)"),
        "quality_of_service": match_any(rf"(?:^|_)({number})q(?:_|$)|(?:^|_)q({number})(?:_|$)"),
        "message_size": match_any(rf"(?:^|_)({number})s(?:_|$)|(?:^|_)s({number})(?:_|$)"),
        "number_of_clients_qos0": match_any(rf"(?:^|_)c0({number})(?:_|$)|(?:^|_)({number})c0(?:_|$)"),
        "delay_qos0": match_any(rf"(?:^|_)d0({number})(?:_|$)|(?:^|_)({number})d0(?:_|$)"),
        "number_of_messages_qos0": match_any(rf"(?:^|_)m0({number})(?:_|$)|(?:^|_)({number})m0(?:_|$)"),
        "message_size_qos0": match_any(rf"(?:^|_)s0({number})(?:_|$)|(?:^|_)({number})s0(?:_|$)"),
        "number_of_clients_qos1": match_any(rf"(?:^|_)c1({number})(?:_|$)|(?:^|_)({number})c1(?:_|$)"),
        "delay_qos1": match_any(rf"(?:^|_)d1({number})(?:_|$)|(?:^|_)({number})d1(?:_|$)"),
        "number_of_messages_qos1": match_any(rf"(?:^|_)m1({number})(?:_|$)|(?:^|_)({number})m1(?:_|$)"),
        "message_size_qos1": match_any(rf"(?:^|_)s1({number})(?:_|$)|(?:^|_)({number})s1(?:_|$)"),
        "number_of_clients_qos2": match_any(rf"(?:^|_)c2({number})(?:_|$)|(?:^|_)({number})c2(?:_|$)"),
        "delay_qos2": match_any(rf"(?:^|_)d2({number})(?:_|$)|(?:^|_)({number})d2(?:_|$)"),
        "number_of_messages_qos2": match_any(rf"(?:^|_)m2({number})(?:_|$)|(?:^|_)({number})m2(?:_|$)"),
        "message_size_qos2": match_any(rf"(?:^|_)s2({number})(?:_|$)|(?:^|_)({number})s2(?:_|$)"),
        "cpu": match_any(rf"(?:^|_)({number})cpu(?:_|$)|(?:^|_)cpu({number})(?:_|$)"),
        "ram": match_any(r"(?:^|_)ram([^_]+)(?:_|$)|(?:^|_)([^_]+)ram(?:_|$)"),
    }


def normalize_ram_token(raw_ram: str) -> str:
    return raw_ram.strip().lower()


def to_epoch_millis(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def load_stats_data(
    run_tag: str,
    start_time: datetime,
    end_time: datetime,
    broker_name: str,
) -> Tuple[float, float, List[float], List[float]]:
    stats_path = DEFAULT_INPUT_DIR / f"{run_tag}_stats.txt"
    if not stats_path.exists():
        return 0.0, 0.0, [], []

    cpu_values: List[float] = []
    mem_percent_values: List[float] = []

    ansi_re = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    
    with stats_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            clean = ansi_re.sub("", raw).strip()

            # Skip header rows
            if not clean:
                continue
            parts = [segment.strip() for segment in clean.split(";")]
            if len(parts) < 6:
                continue
            if parts[1].upper() == "CONTAINER" or parts[2].upper() == "NAME":
                continue

            container_name = parts[2]
            if container_name != f"mn.{broker_name}0":
                continue

            try:
                timestamp_text = parts[0].strip('"')
                timestamp = datetime.strptime(timestamp_text, "%Y-%m-%d-%H:%M:%S")
                if not (start_time <= timestamp <= end_time):
                    continue

                cpu_percent = float(parts[3].replace("%", ""))
                mem_percent = float(parts[5].replace("%", ""))
                cpu_values.append(cpu_percent)
                mem_percent_values.append(mem_percent)
            except (ValueError, IndexError):
                continue

    cpu_mean = statistics.fmean(cpu_values) if cpu_values else 0.0
    ram_mean = statistics.fmean(mem_percent_values) if mem_percent_values else 0.0

    return cpu_mean, ram_mean, cpu_values, mem_percent_values


def reached_maximum_capacity(utilization_percent_values: List[float]) -> str:
    if not utilization_percent_values:
        return "false"

    above_threshold_count = sum(1 for value in utilization_percent_values if value > 95.0)
    share_above_threshold = above_threshold_count / len(utilization_percent_values)
    return "true" if share_above_threshold > 0.10 else "false"


def load_e2e_data(run_tag: str) -> Tuple[Dict[datetime, int], Dict[datetime, int], int, List[datetime], List[datetime]]:
    e2e_path = DEFAULT_EXPERIMENTS_DIR / f"e2e_b0_{run_tag}_SUB_0.txt"
    if not e2e_path.exists():
        return {}, {}, 0, [], []

    sent_counts: Dict[datetime, int] = defaultdict(int)
    recv_counts: Dict[datetime, int] = defaultdict(int)
    malformed_rows = 0
    sent_events: List[datetime] = []
    recv_events: List[datetime] = []

    with e2e_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) < 8:
                # Malformed Row (missing fields)
                malformed_rows += 1
                continue

            if parts[0].lower() == "receiver_brk":
                # Header Row
                continue

            try:
                sent_ms = int(parts[4])
                recv_ms = int(parts[6])
            except ValueError:
                malformed_rows += 1
                continue

            sent_dt = datetime.fromtimestamp(sent_ms / 1000.0)
            recv_dt = datetime.fromtimestamp(recv_ms / 1000.0)
            sent_events.append(sent_dt)
            recv_events.append(recv_dt)

            sent_bin = sent_dt.replace(microsecond=0)
            recv_bin = recv_dt.replace(microsecond=0)
            sent_counts[sent_bin] += 1
            recv_counts[recv_bin] += 1

    return sent_counts, recv_counts, malformed_rows, sent_events, recv_events


def compute_series_in_window(
    sent_counts: Dict[datetime, int],
    recv_counts: Dict[datetime, int],
    start: datetime,
    end: datetime,
) -> Tuple[List[float], List[float], List[datetime]]:
    all_timestamps = set(sent_counts.keys()) | set(recv_counts.keys())
    all_active_window_timestamps_sorted = sorted(
        timestamp
        for timestamp in all_timestamps
        if start <= timestamp <= end
    )
    sent_series = [float(sent_counts.get(timestamp, 0)) for timestamp in all_active_window_timestamps_sorted]
    recv_series = [float(recv_counts.get(timestamp, 0)) for timestamp in all_active_window_timestamps_sorted]
    return sent_series, recv_series, all_active_window_timestamps_sorted


def compute_stats(series: List[float]) -> Dict[str, float]:
    if not series:
        return {
            "mean": 0.0,
            "median": 0.0,
        }

    return {
        "mean": statistics.fmean(series),
        "median": statistics.median(series),
    }


def _to_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _incoming_throughput_from_values(delay_value: str, clients_value: str, messages_value: str) -> float:
    delay = _to_float(delay_value)
    clients = _to_float(clients_value)
    messages = _to_float(messages_value)
    if delay is None or clients is None or messages is None:
        return 0.0
    if delay <= 0 or clients <= 0 or messages <= 0:
        return 0.0
    
    # Parsing to int because this happens in the batch script that launches MZBench as well
    return (int((1000.0 * 60 / delay)) / 60) * clients * N_CLIENTS_MULTIPLIER


def compute_run_metrics(run_tag: str, broker_name: str) -> Optional[Dict[str, object]]:
    sent_counts, recv_counts, malformed_rows, sent_events, recv_events = load_e2e_data(run_tag)
    if not sent_events or not recv_events:
        return None

    active_start = min(sent_events).replace(microsecond=0)
    active_end = max(recv_events).replace(microsecond=0)
    if active_end < active_start:
        return None

    sent_series, recv_series, window_bins = compute_series_in_window(
        sent_counts,
        recv_counts,
        active_start,
        active_end,
    )
    if not window_bins:
        return None

    sent_stats = compute_stats(sent_series)
    recv_stats = compute_stats(recv_series)
    sent_total = sum(sent_series)
    recv_total = sum(recv_series)

    duration_seconds = (active_end - active_start).total_seconds() + 1.0
    inputs = parse_input_tokens(run_tag)

    incoming_qos0 = _incoming_throughput_from_values(
        inputs.get("delay_qos0", ""),
        inputs.get("number_of_clients_qos0", ""),
        inputs.get("number_of_messages_qos0", ""),
    )
    incoming_qos1 = _incoming_throughput_from_values(
        inputs.get("delay_qos1", ""),
        inputs.get("number_of_clients_qos1", ""),
        inputs.get("number_of_messages_qos1", ""),
    )
    incoming_qos2 = _incoming_throughput_from_values(
        inputs.get("delay_qos2", ""),
        inputs.get("number_of_clients_qos2", ""),
        inputs.get("number_of_messages_qos2", ""),
    )

    ram_value = inputs["ram"]
    if ram_value:
        ram_value = normalize_ram_token(ram_value)

    cpu_mean_cons, ram_mean_cons, cpu_values, ram_values = load_stats_data(
        run_tag,
        active_start + timedelta(seconds=30),
        active_end,
        broker_name,
    )

    # Docker CPU% is per-core (100% == 1 core). Normalize by configured cores.
    cpu_cores = 0.0
    if inputs["cpu"]:
        try:
            cpu_cores = float(inputs["cpu"])
        except ValueError:
            cpu_cores = 0.0
    if cpu_cores > 0:
        cpu_mean_cons = cpu_mean_cons / cpu_cores

    normalized_cpu_values = cpu_values
    if cpu_cores > 0:
        normalized_cpu_values = [value / cpu_cores for value in cpu_values]

    cpu_reached_maximum_capacity = reached_maximum_capacity(normalized_cpu_values)
    ram_reached_maximum_capacity = reached_maximum_capacity(ram_values)

    # TODO due to rounding in bash files, the actual expected result can be slightly different
    message_loss = (int(inputs["number_of_messages_qos0"]) * int(inputs["number_of_clients_qos0"]) + int(inputs["number_of_messages_qos1"]) * int(inputs["number_of_clients_qos1"]) + int(inputs["number_of_messages_qos2"]) * int(inputs["number_of_clients_qos2"])) * N_CLIENTS_MULTIPLIER - recv_total

    if message_loss != 0:
        print(f"WARNING: computed message_loss={message_loss} for run_tag={run_tag} based on inputs and recv_total={recv_total}. This may indicate malformed input tokens or e2e data.")

    expected_throughput = incoming_qos0 + incoming_qos1 + incoming_qos2
    received_throughput_completion_percentage = (recv_stats["mean"] / expected_throughput) if expected_throughput > 0 else 0.0

    return {
        "run_tag": run_tag,
        "number_of_clients_qos0": inputs["number_of_clients_qos0"],
        "delay_qos0": inputs["delay_qos0"],
        "number_of_messages_qos0": inputs["number_of_messages_qos0"],
        "message_size_qos0": inputs["message_size_qos0"],
        "number_of_clients_qos1": inputs["number_of_clients_qos1"],
        "delay_qos1": inputs["delay_qos1"],
        "number_of_messages_qos1": inputs["number_of_messages_qos1"],
        "message_size_qos1": inputs["message_size_qos1"],
        "number_of_clients_qos2": inputs["number_of_clients_qos2"],
        "delay_qos2": inputs["delay_qos2"],
        "number_of_messages_qos2": inputs["number_of_messages_qos2"],
        "message_size_qos2": inputs["message_size_qos2"],
        "incoming_throughput_qos0": incoming_qos0,
        "incoming_throughput_qos1": incoming_qos1,
        "incoming_throughput_qos2": incoming_qos2,
        "cpu": inputs["cpu"],
        "ram": ram_value,
        "active_window_start": to_epoch_millis(active_start),
        "active_window_end": to_epoch_millis(active_end),
        "active_window_seconds": duration_seconds,
        "throughput_window_bins": len(window_bins),
        "sent_throughput_mean": sent_stats["mean"],
        "sent_throughput_median": sent_stats["median"],
        "received_throughput_mean": recv_stats["mean"],
        "received_throughput_median": recv_stats["median"],
        "received_throughput_completion_percentage": received_throughput_completion_percentage,
        "message_loss": message_loss,
        "malformed_e2e_rows_skipped": malformed_rows,
        "cpu_mean_consumption": cpu_mean_cons,
        "ram_mean_consumption": ram_mean_cons,
        "cpu_reached_maximum_capacity": cpu_reached_maximum_capacity,
        "ram_reached_maximum_capacity": ram_reached_maximum_capacity,
        "throughput_reached_maximum_capacity": received_throughput_completion_percentage < 0.95, # TODO parameter
        "message_loss_threshold_exceeded": message_loss > 4000 or message_loss < -4000, # TODO parameter # TODO research too many messges anomaly (due to resends?)
    }


def write_metrics_csv(output_dir: Path, timestamp: str, rows: List[Dict[str, object]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{timestamp}_metrics.csv"

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main() -> None:
    args = parse_args()
    run_tags = discover_run_tags(args.timestamp)

    rows_to_write: List[Dict[str, object]] = []
    skipped = 0
    for run_tag in run_tags:
        metrics = compute_run_metrics(run_tag, args.broker_name)
        if metrics is None:
            skipped += 1
            print(f"SKIP: no valid active-window data for run_tag={run_tag}")
            continue

        rows_to_write.append(metrics)

    if rows_to_write:
        cpu_means = [float(row["cpu_mean_consumption"]) for row in rows_to_write]
        positive_cpu_means = [value for value in cpu_means if value > 0.0]
        min_cpu_mean = min(positive_cpu_means) if positive_cpu_means else 0.0

        ram_means = [float(row["ram_mean_consumption"]) for row in rows_to_write]
        positive_ram_means = [value for value in ram_means if value > 0.0]
        min_ram_mean = min(positive_ram_means) if positive_ram_means else 0.0

        for metrics in rows_to_write:
            cpu_mean = float(metrics["cpu_mean_consumption"])
            if min_cpu_mean > 0.0:
                metrics["cpu_factor_compared_to_min"] = str(cpu_mean / min_cpu_mean)
            else:
                metrics["cpu_factor_compared_to_min"] = "0.0"

            ram_mean = float(metrics["ram_mean_consumption"])
            if min_ram_mean > 0.0:
                metrics["ram_factor_compared_to_min"] = str(ram_mean / min_ram_mean)
            else:
                metrics["ram_factor_compared_to_min"] = "0.0"

    if rows_to_write:
        output = write_metrics_csv(DEFAULT_OUTPUT_DIR, args.timestamp, rows_to_write)
        print(f"WROTE: {output}")
    else:
        print("WROTE: no CSV because no valid run_tag rows were produced")

    print(f"Processed run_tags: {len(run_tags)}")
    print(f"Written rows: {len(rows_to_write)}")
    print(f"Skipped run_tags: {skipped}")


if __name__ == "__main__":
    main()
