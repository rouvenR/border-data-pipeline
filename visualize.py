#!/usr/bin/env python3

import argparse
import re
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt


ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "inputs/result_data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs/plots"
DEFAULT_EXPERIMENTS_DIR = DEFAULT_INPUT_DIR / "experiments"


@dataclass
class StatRow:
	timestamp: datetime
	name: str
	run_tag: str
	cpu_percent: float
	mem_used_bytes: float
	mem_limit_bytes: float
	mem_percent: float


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Visualize docker stats fields NAME;CPU %;MEM USAGE / LIMIT;MEM % over time"
	)
	parser.add_argument(
		"--timestamp",
		default=None,
		help="Timestamp to match run tags (e.g., 20260420144333). Matches all files *_{TIMESTAMP}*_stats.txt",
	)
	parser.add_argument(
		"--pretty",
		action="store_true",
		help="Exclude mn.sub.0 and mn.pub.0 containers from visualization",
	)
	return parser.parse_args()


def unit_to_bytes(value: float, unit: str) -> float:
	factor = {
		"B": 1,
		"KB": 1000,
		"MB": 1000**2,
		"GB": 1000**3,
		"TB": 1000**4,
		"KIB": 1024,
		"MIB": 1024**2,
		"GIB": 1024**3,
		"TIB": 1024**4,
	}
	return value * factor[unit.upper()]


def parse_cpu_cores(run_tag: str) -> Optional[int]:
	"""Extract CPU core count from run_tag (e.g., CPU2, CPU4, CPU8)."""
	for pattern in [r"(?:^|_)CPU([0-9]+)(?:_|$)", r"(?:^|_)([0-9]+)cpu(?:_|$)"]:
		match = re.search(pattern, run_tag, re.IGNORECASE)
		if match:
			return int(match.group(1))
	return None


def parse_mem_usage(mem_usage: str) -> Tuple[float, float]:
	# Expected format: "80.15MiB / 187.5GiB"
	left, right = [part.strip() for part in mem_usage.split("/")]

	left_match = re.match(r"([0-9]+(?:\.[0-9]+)?)([A-Za-z]+)", left)
	right_match = re.match(r"([0-9]+(?:\.[0-9]+)?)([A-Za-z]+)", right)
	if not left_match or not right_match:
		raise ValueError(f"Invalid MEM USAGE / LIMIT field: {mem_usage}")

	used = unit_to_bytes(float(left_match.group(1)), left_match.group(2))
	limit = unit_to_bytes(float(right_match.group(1)), right_match.group(2))
	return used, limit


def parse_line(line: str, run_tag: str) -> Optional[StatRow]:
	clean = ANSI_RE.sub("", line).strip()
	if not clean:
		return None

	# Timestamp + data are separated by '; '
	parts = [segment.strip() for segment in clean.split(";")]
	if len(parts) < 6:
		return None

	# Skip header/status rows.
	if parts[1].upper() == "CONTAINER" or parts[2].upper() == "NAME":
		return None

	timestamp_text = parts[0].strip('"')
	try:
		timestamp = datetime.strptime(timestamp_text, "%Y-%m-%d-%H:%M:%S")
	except ValueError:
		return None

	name = parts[2]
	cpu_text = parts[3].replace("%", "")
	mem_usage = parts[4]
	mem_percent_text = parts[5].replace("%", "")

	try:
		cpu_percent = float(cpu_text)
		mem_used, mem_limit = parse_mem_usage(mem_usage)
		mem_percent = float(mem_percent_text)
	except ValueError:
		return None

	return StatRow(
		timestamp=timestamp,
		name=name,
		run_tag=run_tag,
		cpu_percent=cpu_percent,
		mem_used_bytes=mem_used,
		mem_limit_bytes=mem_limit,
		mem_percent=mem_percent,
	)


def load_rows(path: Path, run_tag: str) -> List[StatRow]:
	rows: List[StatRow] = []
	seen: Set[Tuple[datetime, str, str, float, float, float]] = set()

	with path.open("r", encoding="utf-8", errors="ignore") as handle:
		for raw in handle:
			row = parse_line(raw, run_tag)
			if row is None:
				continue

			key = (
				row.timestamp,
				row.name,
				row.run_tag,
				row.cpu_percent,
				row.mem_used_bytes,
				row.mem_percent,
			)
			# docker stats output contains repeated redraw lines; dedupe exact repeats.
			if key in seen:
				continue
			seen.add(key)
			rows.append(row)

	rows.sort(key=lambda r: (r.timestamp, r.name))
	return rows


def discover_input_files(args: argparse.Namespace) -> List[Tuple[Path, str]]:
	if args.timestamp:
		patterns = [
			f"{args.timestamp}_*_stats.txt",
			f"*_{args.timestamp}*_stats.txt",
		]
		files = sorted({path for pattern in patterns for path in DEFAULT_INPUT_DIR.glob(pattern)})
		if not files:
			raise SystemExit(
				f"No files found in {DEFAULT_INPUT_DIR} for timestamp: {args.timestamp}"
			)
		result: List[Tuple[Path, str]] = []
		for path in files:
			stem = path.stem
			run_tag = stem[: -len("_stats")] if stem.endswith("_stats") else stem
			result.append((path, run_tag))
		return result

	raise SystemExit("Please provide --timestamp (e.g., 20260420144333)")


def load_connack_timestamps(run_tag: str) -> List[datetime]:
	conn_path = DEFAULT_EXPERIMENTS_DIR / f"conn_b0_{run_tag}_SUB_0.txt"
	if not conn_path.exists():
		return []

	values: Set[datetime] = set()
	with conn_path.open("r", encoding="utf-8", errors="ignore") as handle:
		for raw in handle:
			parts = [p.strip() for p in raw.split(",")]
			if len(parts) < 4:
				continue
			if parts[0].lower() == "broker" or not parts[3]:
				continue

			try:
				connack_ms = int(parts[3])
			except ValueError:
				continue

			values.add(datetime.fromtimestamp(connack_ms / 1000.0))

	return sorted(values)


def load_e2e_throughput(run_tag: str) -> Tuple[List[datetime], List[float], List[float], int]:
	e2e_path = DEFAULT_EXPERIMENTS_DIR / f"e2e_b0_{run_tag}_SUB_0.txt"
	if not e2e_path.exists():
		return [], [], [], 0

	sent_counts: Dict[datetime, int] = defaultdict(int)
	received_counts: Dict[datetime, int] = defaultdict(int)
	malformed_rows = 0

	with e2e_path.open("r", encoding="utf-8", errors="ignore") as handle:
		for raw in handle:
			parts = [p.strip() for p in raw.split(",")]
			if len(parts) < 8:
				malformed_rows += 1
				continue

			if parts[0].lower() == "receiver_brk":
				continue

			try:
				sent_ms = int(parts[4])
				received_ms = int(parts[6])
			except ValueError:
				malformed_rows += 1
				continue

			sent_ts = datetime.fromtimestamp(sent_ms / 1000.0).replace(microsecond=0)
			received_ts = datetime.fromtimestamp(received_ms / 1000.0).replace(microsecond=0)
			sent_counts[sent_ts] += 1
			received_counts[received_ts] += 1

	all_bins = sorted(set(sent_counts.keys()) | set(received_counts.keys()))
	sent_series = [float(sent_counts.get(ts, 0)) for ts in all_bins]
	received_series = [float(received_counts.get(ts, 0)) for ts in all_bins]
	return all_bins, sent_series, received_series, malformed_rows


def load_publisher_first_send(run_tag: str) -> Dict[int, datetime]:
	"""Extract first send timestamp per publisher client_num from e2e data."""
	e2e_path = DEFAULT_EXPERIMENTS_DIR / f"e2e_b0_{run_tag}_SUB_0.txt"
	if not e2e_path.exists():
		return {}

	first_send: Dict[int, datetime] = {}

	with e2e_path.open("r", encoding="utf-8", errors="ignore") as handle:
		for raw in handle:
			parts = [p.strip() for p in raw.split(",")]
			if len(parts) < 8:
				continue

			if parts[0].lower() == "receiver_brk":
				continue

			try:
				client_num = int(parts[3])
				sent_ms = int(parts[4])
			except ValueError:
				continue

			sent_ts = datetime.fromtimestamp(sent_ms / 1000.0)
			if client_num not in first_send or sent_ts < first_send[client_num]:
				first_send[client_num] = sent_ts

	return first_send


def load_client_last_received(run_tag: str) -> Dict[int, datetime]:
	"""Extract last received timestamp per client_num from e2e data."""
	e2e_path = DEFAULT_EXPERIMENTS_DIR / f"e2e_b0_{run_tag}_SUB_0.txt"
	if not e2e_path.exists():
		return {}

	last_recv: Dict[int, datetime] = {}

	with e2e_path.open("r", encoding="utf-8", errors="ignore") as handle:
		for raw in handle:
			parts = [p.strip() for p in raw.split(",")]
			if len(parts) < 8:
				continue

			if parts[0].lower() == "receiver_brk":
				continue

			try:
				client_num = int(parts[3])
				received_ms = int(parts[6])
			except ValueError:
				continue

			recv_ts = datetime.fromtimestamp(received_ms / 1000.0)
			if client_num not in last_recv or recv_ts > last_recv[client_num]:
				last_recv[client_num] = recv_ts

	return last_recv


def build_series(rows: List[StatRow], run_tag: str) -> Dict[str, Dict[str, List[Union[float, datetime]]]]:
	cpu_cores = parse_cpu_cores(run_tag)
	data: Dict[str, Dict[str, List[Union[float, datetime]]]] = {}
	for row in rows:
		series_name = f"{row.name} [{row.run_tag}]"
		if series_name not in data:
			data[series_name] = {
				"t": [],
				"cpu": [],
				"mem_used_mib": [],
				"mem_limit_mib": [],
				"mem_percent": [],
			}
		data[series_name]["t"].append(row.timestamp)
		# Normalize CPU percentage by dividing by core count (docker stats uses 100% per core)
		normalized_cpu = row.cpu_percent / cpu_cores if cpu_cores else row.cpu_percent
		data[series_name]["cpu"].append(normalized_cpu)
		data[series_name]["mem_used_mib"].append(row.mem_used_bytes / (1024**2))
		data[series_name]["mem_limit_mib"].append(row.mem_limit_bytes / (1024**2))
		data[series_name]["mem_percent"].append(row.mem_percent)
	return data


def plot_metric(
	series: Dict[str, Dict[str, List[Union[float, datetime]]]],
	metric: str,
	ylabel: str,
	output: Path,
	connack_by_run: Dict[str, List[datetime]],
	e2e_throughput_by_run: Dict[str, Dict[str, List[Union[datetime, float]]]],
	pub_first_send_by_run: Dict[str, Dict[int, datetime]],
	client_last_recv_by_run: Dict[str, Dict[int, datetime]],
) -> None:
	# Collect all timestamps to find the minimum for reference point
	all_timestamps: List[datetime] = []
	for values in series.values():
		all_timestamps.extend(values["t"])
	for timestamps in connack_by_run.values():
		all_timestamps.extend(timestamps)
	for pub_sends in pub_first_send_by_run.values():
		all_timestamps.extend(pub_sends.values())
	for last_recvs in client_last_recv_by_run.values():
		all_timestamps.extend(last_recvs.values())
	for throughput in e2e_throughput_by_run.values():
		all_timestamps.extend(throughput.get("t", []))

	if not all_timestamps:
		return

	# Use the earliest timestamp as the reference point (0 minutes)
	time_ref = min(all_timestamps)

	def to_minutes(ts: datetime) -> float:
		"""Convert datetime to minutes elapsed since time_ref."""
		delta = ts - time_ref
		return delta.total_seconds() / 60.0

	# Derive x-axis window in minutes
	all_first_sends = [ts for sends in pub_first_send_by_run.values() for ts in sends.values()]
	all_last_recvs = [ts for recvs in client_last_recv_by_run.values() for ts in recvs.values()]
	if all_first_sends and all_last_recvs:
		x_min_min = to_minutes(min(all_first_sends) - timedelta(minutes=2))
		x_max_min = to_minutes(max(all_last_recvs) + timedelta(minutes=1))
	else:
		x_min_min = x_max_min = None

	plt.figure(figsize=(14, 7))
	ax = plt.gca()
	for name, values in sorted(series.items()):
		t_minutes = [to_minutes(ts) for ts in values["t"]]
		if "mn.sub0" in name:
			name = "Subscriber"
		elif "mn.pub0" in name:
			name = "Publisher"
		elif "mn.jorammq0" in name:
			name = "Broker"
		ax.plot(t_minutes, values[metric], label=name + " Resource Utilisation", linewidth=1.4)

	for run_tag, timestamps in sorted(connack_by_run.items()):
		for idx, connack_ts in enumerate(timestamps):
			label = f"Subscriber Connection" if idx == 0 else None
			ax.axvline(
				to_minutes(connack_ts),
				color="green",
				linestyle="--",
				alpha=0.2,
				linewidth=1.0,
				label=label,
			)

	for run_tag, pub_sends in sorted(pub_first_send_by_run.items()):
		for idx, (client_num, send_ts) in enumerate(sorted(pub_sends.items())):
			label = f"Publisher First Send" if idx == 0 else None
			ax.axvline(
				to_minutes(send_ts),
				color="blue",
				linestyle="--",
				alpha=0.2,
				linewidth=1.0,
				label=label,
			)

	for run_tag, last_recvs in sorted(client_last_recv_by_run.items()):
		for idx, (client_num, recv_ts) in enumerate(sorted(last_recvs.items())):
			label = f"Client Last Receive" if idx == 0 else None
			ax.axvline(
				to_minutes(recv_ts),
				color="violet",
				linestyle="--",
				alpha=0.2,
				linewidth=1.0,
				label=label,
			)

	ax2 = ax.twinx()
	has_throughput = False
	for run_tag, throughput in sorted(e2e_throughput_by_run.items()):
		timestamps = throughput.get("t", [])
		sent_series = throughput.get("sent", [])
		received_series = throughput.get("received", [])
		if not timestamps:
			continue
		has_throughput = True
		t_minutes = [to_minutes(ts) for ts in timestamps]
		ax2.plot(
			t_minutes,
			sent_series,
			linestyle="-.",
			linewidth=1.8,
			label=f"Incoming Throughput [msg/s]",
		)
		ax2.plot(
			t_minutes,
			received_series,
			linestyle="-.",
			linewidth=1.8,
			label=f"Outgoing Throughput [msg/s]",
			color="green"
		)

	ax.set_xlabel("Time [minutes]")
	ax.set_ylabel(ylabel)
	ax.set_title(f"{ylabel} over time")
	ax.grid(alpha=0.25)
	# if args.pretty:
	# 	ax.set_ylim(0, 100)
	if x_min_min is not None and x_max_min is not None:
		ax.set_xlim(x_min_min, x_max_min)
	if has_throughput:
		ax2.set_ylabel("e2e throughput [msg/s]")

	handles1, labels1 = ax.get_legend_handles_labels()
	handles2, labels2 = ax2.get_legend_handles_labels()
	ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=8)
	plt.tight_layout()
	plt.savefig(output, dpi=150)
	plt.close()


def main() -> None:
	args = parse_args()
	inputs = discover_input_files(args)

	base_output_dir = DEFAULT_OUTPUT_DIR / "raw" / args.timestamp
	base_output_dir.mkdir(parents=True, exist_ok=True)

	# Group data by run_tag
	data_by_run: Dict[str, Dict] = {}
	for input_path, run_tag in inputs:
		if run_tag not in data_by_run:
			data_by_run[run_tag] = {
				"rows": [],
				"connack": [],
				"e2e_throughput": {"t": [], "sent": [], "received": []},
				"pub_first_send": {},
				"client_last_recv": {},
				"malformed_e2e": 0,
			}

		rows = load_rows(input_path, run_tag)
		# Filter out mn.sub.0 and mn.pub.0 if --pretty flag is set
		if args.pretty:
			rows = [r for r in rows if r.name not in ("mn.sub0", "mn.pub0")]
		data_by_run[run_tag]["rows"].extend(rows)
		data_by_run[run_tag]["connack"] = load_connack_timestamps(run_tag)
		t, sent_series, recv_series, malformed_rows = load_e2e_throughput(run_tag)
		data_by_run[run_tag]["e2e_throughput"] = {
			"t": t,
			"sent": sent_series,
			"received": recv_series,
		}
		data_by_run[run_tag]["pub_first_send"] = load_publisher_first_send(run_tag)
		data_by_run[run_tag]["client_last_recv"] = load_client_last_received(run_tag)
		data_by_run[run_tag]["malformed_e2e"] += malformed_rows

	# Generate plots for each run_tag
	for run_tag, data in data_by_run.items():
		if not data["rows"]:
			print(f"WARNING: No valid rows for run_tag {run_tag}")
			continue

		# Create subdirectory for this run_tag
		run_output_dir = base_output_dir / run_tag
		run_output_dir.mkdir(parents=True, exist_ok=True)

		series = build_series(data["rows"], run_tag)

		# Build aggregated data structures for this run_tag
		connack_by_run = {run_tag: data["connack"]}
		e2e_throughput_by_run = {run_tag: data["e2e_throughput"]}
		pub_first_send_by_run = {run_tag: data["pub_first_send"]}
		client_last_recv_by_run = {run_tag: data["client_last_recv"]}

		plot_metric(series, "cpu", "CPU [%]", run_output_dir / "cpu_percent_over_time.png", connack_by_run, e2e_throughput_by_run, pub_first_send_by_run, client_last_recv_by_run)
		plot_metric(series, "mem_used_mib", "MEM USAGE [MiB]", run_output_dir / "mem_usage_over_time.png", connack_by_run, e2e_throughput_by_run, pub_first_send_by_run, client_last_recv_by_run)
		plot_metric(series, "mem_percent", "MEM % [%]", run_output_dir / "mem_percent_over_time.png", connack_by_run, e2e_throughput_by_run, pub_first_send_by_run, client_last_recv_by_run)

		print(f"Run tag: {run_tag}")
		print(f"  Parsed rows: {len(data['rows'])}")
		print(f"  Containers: {len(series)}")
		if data["malformed_e2e"] > 0:
			print(f"  WARNING: malformed e2e rows skipped: {data['malformed_e2e']}")
		print(f"  Plots saved to: {run_output_dir}")

	print(f"\nTotal run_tags processed: {len(data_by_run)}")
	print(f"Base output directory: {base_output_dir}")


if __name__ == "__main__":
	main()
