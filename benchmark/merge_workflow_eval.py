#!/usr/bin/env python3
"""Merge workflow validation (Role C: C5).

This script validates the assignment's pre-aggregate + merge workflow using
the locally built DuckDB CLI, which matches the locally built `quack`
extension.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from duckdb_cli_runner import DuckDBCliRunner, detect_default_duckdb_binary, ensure_tpch_data


@dataclass
class MergeWorkflowResult:
    scale_factor: int
    partition_column: str
    partition_count: int
    exact_distinct: int
    merge_estimate: int
    single_scan_estimate: int
    merge_relative_error: float
    single_scan_relative_error: float
    merge_vs_single_relative_diff: float
    pre_agg_time_s: float
    merge_query_median_s: float
    merge_query_std_s: float
    single_query_median_s: float
    single_query_std_s: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_db_dir = repo_root / "benchmark" / "out" / "databases"
    default_out = repo_root / "benchmark" / "out" / "merge_workflow_eval"
    default_duckdb = detect_default_duckdb_binary(repo_root)
    default_threads = min(4, os.cpu_count() or 1)

    parser = argparse.ArgumentParser(description="Validate pre-aggregate + merge workflow for HLL sketches.")
    parser.add_argument("--scale-factor", type=int, default=1, help="TPC-H scale factor for dbgen.")
    parser.add_argument("--iterations", type=int, default=5, help="Warm runs for query timing (cold run discarded).")
    parser.add_argument("--threads", type=int, default=default_threads, help="PRAGMA threads value for benchmark runs.")
    parser.add_argument("--error-threshold", type=float, default=0.02, help="Relative error threshold for assertions.")
    parser.add_argument(
        "--consistency-threshold",
        type=float,
        default=0.01,
        help="Max relative diff between merge estimate and single-scan estimate.",
    )
    parser.add_argument("--partition-column", type=str, default="l_shipdate", help="lineitem partition column.")
    parser.add_argument("--database-dir", type=Path, default=default_db_dir, help="Directory holding TPC-H databases.")
    parser.add_argument("--output-dir", type=Path, default=default_out, help="Output directory for results.")
    parser.add_argument(
        "--duckdb-binary",
        type=Path,
        default=default_duckdb,
        help="Path to the locally built DuckDB CLI binary that includes the quack extension.",
    )
    parser.add_argument("--force-dbgen", action="store_true", help="Force regenerate TPCH data.")
    return parser.parse_args()


def sanitize_identifier(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Unsafe SQL identifier: {name}")
    return name


def summarize(times: Sequence[float]) -> tuple[float, float]:
    if not times:
        return math.nan, math.nan
    if len(times) == 1:
        return times[0], 0.0
    return statistics.median(times), statistics.stdev(times)


def write_outputs(result: MergeWorkflowResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "merge_workflow_eval.csv"
    json_path = output_dir / "merge_workflow_eval.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MergeWorkflowResult.__annotations__.keys()))
        writer.writeheader()
        writer.writerow(asdict(result))

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2)

    print(f"[info] Wrote CSV:  {csv_path}")
    print(f"[info] Wrote JSON: {json_path}")


def plot_outputs(result: MergeWorkflowResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    full_labels = ["pre_agg", "merge_query", "single_scan"]
    full_values = [
        result.pre_agg_time_s,
        result.merge_query_median_s,
        result.single_query_median_s,
    ]

    query_labels = ["merge_query", "single_scan"]
    query_values = [
        result.merge_query_median_s,
        result.single_query_median_s,
    ]

    colors = ["#4c72b0", "#55a868", "#c44e52"]
    fig, (ax_full, ax_zoom) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.8),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    full_bars = ax_full.bar(full_labels, full_values, color=colors)
    ax_full.set_ylabel("Time (seconds)")
    ax_full.set_title(f"Merge Workflow Timing (SF {result.scale_factor})")

    query_bars = ax_zoom.bar(query_labels, query_values, color=colors[1:])
    ax_zoom.set_ylabel("Time (seconds)")
    ax_zoom.set_title("Query-Time Zoom")

    # Add value labels so very small bars remain readable even when they are
    # visually compressed by the full workflow scale.
    ax_full.bar_label(full_bars, fmt="%.3fs", padding=3, fontsize=9)
    ax_zoom.bar_label(query_bars, fmt="%.3fs", padding=3, fontsize=9)

    # Tighten the zoomed axis around query-time values so merge_query is easy
    # to see even when pre-aggregation dominates the full figure.
    max_query_value = max(query_values)
    ax_zoom.set_ylim(0, max_query_value * 1.18 if max_query_value > 0 else 1.0)

    fig.suptitle("Pre-Aggregate + Merge Workflow Validation", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    chart_path = output_dir / "merge_workflow_timing_bar.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    print(f"[info] Wrote chart: {chart_path}")


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")

    partition_column = sanitize_identifier(args.partition_column)
    args.database_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    database_path = args.database_dir / f"tpch_sf{args.scale_factor}_eval.duckdb"
    ensure_tpch_data(database_path, args.scale_factor, args.force_dbgen)
    runner = DuckDBCliRunner(args.duckdb_binary, database_path)
    pre_statements = [f"PRAGMA threads={args.threads}"]

    pre_agg_sql = (
        "CREATE TABLE pre_agg_sketches AS "
        f"SELECT {partition_column}, hll_create_agg(CAST(l_orderkey AS VARCHAR)) AS sketch "
        "FROM lineitem GROUP BY 1"
    )

    print(f"[info] Running pre-aggregation by {partition_column} for sf={args.scale_factor} ...")
    timed_setup = runner.run_timed_statements(
        [
            "DROP TABLE IF EXISTS pre_agg_sketches",
            pre_agg_sql,
        ],
        pre_statements=pre_statements,
    )
    pre_agg_time = timed_setup[-1]

    partition_count = int(runner.query_scalar("SELECT COUNT(*) FROM pre_agg_sketches", pre_statements=pre_statements))

    exact_sql = "SELECT COUNT(DISTINCT l_orderkey)::BIGINT FROM lineitem"
    merge_sql = "SELECT hll_estimate(hll_merge_agg(sketch))::BIGINT FROM pre_agg_sketches"
    single_sql = "SELECT hll_estimate(hll_create_agg(CAST(l_orderkey AS VARCHAR)))::BIGINT FROM lineitem"

    exact_count = int(runner.query_scalar(exact_sql, pre_statements=pre_statements))
    merge_run = runner.run_scalar_query_repeated(merge_sql, repeats=args.iterations + 1, pre_statements=pre_statements)
    single_run = runner.run_scalar_query_repeated(single_sql, repeats=args.iterations + 1, pre_statements=pre_statements)

    merge_values = {int(value) for value in merge_run.values}
    single_values = {int(value) for value in single_run.values}
    if len(merge_values) != 1:
        raise RuntimeError(f"Merge query returned inconsistent values: {sorted(merge_values)}")
    if len(single_values) != 1:
        raise RuntimeError(f"Single-scan query returned inconsistent values: {sorted(single_values)}")

    merge_estimate = merge_values.pop()
    single_estimate = single_values.pop()

    merge_warm_times = merge_run.times_s[1:]
    single_warm_times = single_run.times_s[1:]
    merge_median, merge_std = summarize(merge_warm_times)
    single_median, single_std = summarize(single_warm_times)

    merge_relative_error = abs(merge_estimate - exact_count) / exact_count if exact_count else 0.0
    single_relative_error = abs(single_estimate - exact_count) / exact_count if exact_count else 0.0
    merge_vs_single_relative_diff = abs(merge_estimate - single_estimate) / exact_count if exact_count else 0.0

    result = MergeWorkflowResult(
        scale_factor=args.scale_factor,
        partition_column=partition_column,
        partition_count=partition_count,
        exact_distinct=exact_count,
        merge_estimate=merge_estimate,
        single_scan_estimate=single_estimate,
        merge_relative_error=merge_relative_error,
        single_scan_relative_error=single_relative_error,
        merge_vs_single_relative_diff=merge_vs_single_relative_diff,
        pre_agg_time_s=pre_agg_time,
        merge_query_median_s=merge_median,
        merge_query_std_s=merge_std,
        single_query_median_s=single_median,
        single_query_std_s=single_std,
    )

    print(
        "[result] "
        f"sf={args.scale_factor}, exact={exact_count}, merge={merge_estimate}, single={single_estimate}, "
        f"merge_err={merge_relative_error:.4%}, single_err={single_relative_error:.4%}, "
        f"merge_vs_single={merge_vs_single_relative_diff:.4%}, pre_agg={pre_agg_time:.4f}s"
    )

    if merge_relative_error > args.error_threshold:
        raise AssertionError(
            f"Merge workflow relative error {merge_relative_error:.4%} exceeds threshold {args.error_threshold:.2%}"
        )
    if single_relative_error > args.error_threshold:
        raise AssertionError(
            f"Single-scan HLL relative error {single_relative_error:.4%} exceeds threshold {args.error_threshold:.2%}"
        )
    if merge_vs_single_relative_diff > args.consistency_threshold:
        raise AssertionError(
            f"Merge vs single relative diff {merge_vs_single_relative_diff:.4%} exceeds threshold "
            f"{args.consistency_threshold:.2%}"
        )

    write_outputs(result, args.output_dir)
    plot_outputs(result, args.output_dir)
    print("[done] Merge workflow evaluation finished.")


if __name__ == "__main__":
    main()
