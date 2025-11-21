#!/usr/bin/env python3
"""Collect validation cer_ctc values from ESPnet train.log files and plot them."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

SCRIPT_ROOT = Path(__file__).resolve().parent
MPL_CACHE_DIR = SCRIPT_ROOT / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (needs Agg backend first)


EPOCH_LINE = re.compile(
    r"INFO:\s*(\d+)epoch results:.*?\[valid].*?cer_ctc=(?P<value>[0-9.eE+-]+|nan)",
    re.IGNORECASE,
)


@dataclass
class LogGroup:
    source: Path
    log_files: List[Path]


@dataclass
class CERSeries:
    label: str
    log_path: Path
    points: List[tuple[int, float]]

    @property
    def epoch_to_value(self) -> dict[int, float]:
        return {epoch: value for epoch, value in self.points}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse one or more ESPnet train.log files, extract validation cer_ctc, "
            "write a CSV summary, and plot all series in one figure. "
            "Directories that contain staged logs (train.log, train.1.log, ...) are supported."
        )
    )
    parser.add_argument(
        "logs",
        nargs="+",
        help=(
            "Paths to train.log files or directories containing train*.log fragments "
            "(train.log, train.1.log, ...)."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional custom labels matching the number of logs.",
    )
    parser.add_argument(
        "--output-csv",
        default="local/cer_ctc_summary.csv",
        help="Where to save the aggregated table (default: %(default)s).",
    )
    parser.add_argument(
        "--figure",
        default="local/cer_ctc_plot.png",
        help="Where to save the plot (default: %(default)s).",
    )
    return parser.parse_args()


def _log_sort_key(path: Path) -> tuple[int, Union[int, str]]:
    name = path.name
    if name == "train.log":
        return (0, 0)
    match = re.fullmatch(r"train\.(\d+)\.log", name)
    if match:
        return (1, int(match.group(1)))
    return (2, name)


def resolve_log_group(path_str: str) -> LogGroup:
    path = Path(path_str)
    if path.is_file():
        return LogGroup(source=path, log_files=[path])
    if path.is_dir():
        log_files: List[Path] = []
        candidate = path / "train.log"
        if candidate.is_file():
            log_files.append(candidate)
        log_files.extend(
            log_path
            for log_path in path.glob("train.*.log")
            if log_path.is_file() and log_path.name != "train.log"
        )
        log_files = sorted(set(log_files), key=_log_sort_key)
        if not log_files:
            raise FileNotFoundError(f"No train.*.log files found in {path}")
        return LogGroup(source=path, log_files=log_files)
    raise FileNotFoundError(f"Path not found: {path}")


def parse_log(path: Path) -> List[tuple[int, float]]:
    points: List[tuple[int, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = EPOCH_LINE.search(line)
            if not match:
                continue
            epoch = int(match.group(1))
            raw_value = match.group("value")
            value = float(raw_value) if raw_value.lower() != "nan" else math.nan
            points.append((epoch, value))
    return points


def parse_log_group(paths: Sequence[Path]) -> List[tuple[int, float]]:
    epoch_map: dict[int, float] = {}
    for path in paths:
        for epoch, value in parse_log(path):
            epoch_map[epoch] = value
    points = sorted(epoch_map.items(), key=lambda item: item[0])
    return points


def create_series(logs: Sequence[str], labels: Optional[Sequence[str]]) -> List[CERSeries]:
    log_groups = [resolve_log_group(log) for log in logs]
    if labels:
        if len(labels) != len(log_groups):
            raise ValueError("--labels must match the number of logs")
        series_labels = list(labels)
    else:
        series_labels = []
        for group in log_groups:
            if group.source.is_file():
                series_labels.append(group.source.parent.name or group.source.name)
            else:
                series_labels.append(group.source.name)
    series: List[CERSeries] = []
    for label, group in zip(series_labels, log_groups):
        points = parse_log_group(group.log_files)
        if not points:
            raise ValueError(f"No cer_ctc entries found in {group.source}")
        series.append(CERSeries(label=label, log_path=group.source, points=points))
    return series


def write_csv(series_list: Iterable[CERSeries], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    series_list = list(series_list)
    all_epochs = sorted({epoch for series in series_list for epoch, _ in series.points})
    header = ["epoch"] + [series.label for series in series_list]
    with destination.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for epoch in all_epochs:
            row: List[Union[str, float, int]] = [epoch]
            for series in series_list:
                value = series.epoch_to_value.get(epoch, "")
                if isinstance(value, float) and math.isnan(value):
                    value = ""
                row.append(value)
            writer.writerow(row)


def plot_series(series_list: Iterable[CERSeries], destination: Path) -> None:
    series_list = list(series_list)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for series in series_list:
        epochs: List[int] = []
        values: List[float] = []
        for epoch, value in series.points:
            if math.isnan(value):
                continue
            epochs.append(epoch)
            values.append(value)
        if not epochs:
            continue
        ax.plot(
            epochs,
            values,
            marker="o",
            linewidth=1.5,
            label=series.label,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation CER (cer_ctc)")
    ax.set_title("Validation cer_ctc per epoch")
    ax.grid(True, linewidth=0.1, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    try:
        series_list = create_series(args.logs, args.labels)
        # for series in series_list:
        #     series.points = series.points[:70]
    except (FileNotFoundError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
    csv_path = Path(args.output_csv)
    figure_path = Path(args.figure)
    write_csv(series_list, csv_path)
    plot_series(series_list, figure_path)
    for series in series_list:
        print(
            f"{series.label}: collected {len(series.points)} cer_ctc points from {series.log_path}"
        )
    print(f"CSV summary saved to {csv_path}")
    print(f"Plot saved to {figure_path}")


if __name__ == "__main__":
    main()
