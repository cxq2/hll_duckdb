#!/usr/bin/env python3
"""TPC-H HLL accuracy and performance evaluation (Role C: C1/C2/C3/C4).

This script keeps TPC-H data generation and HLL benchmarking separate:
1. Use the Python `duckdb` package to generate or reuse TPC-H databases.
2. Use the locally built DuckDB CLI for exact/HLL benchmarking so the
   runtime matches the locally built `quack` extension.
3. Record warm-run medians/stddevs, write CSV/JSON outputs, and generate
   charts required by the assignment report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

# Keep matplotlib cache writable in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from duckdb_cli_runner import DuckDBCliRunner, detect_default_duckdb_binary, ensure_tpch_data, split_target


DEFAULT_TARGETS = [
    "lineitem.l_orderkey",
    "orders.o_custkey",
    "partsupp.ps_partkey",
]


@dataclass
class MetricRow:
    scale_factor: int
    table_name: str
    column_name: str
    exact_count: int
    hll_estimate: int
    relative_error: float
    exact_median_s: float
    exact_std_s: float
    hll_median_s: float
    hll_std_s: float
    speedup_exact_over_hll: float


@dataclass
class ScaleSummaryRow:
    scale_factor: int
    target_count: int
    mean_exact_median_s: float
    mean_hll_median_s: float
    mean_relative_error: float
    max_relative_error: float
    mean_speedup_exact_over_hll: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_db_dir = repo_root / "benchmark" / "out" / "databases"
    default_out = repo_root / "benchmark" / "out" / "tpch_hll_eval"
    default_duckdb = detect_default_duckdb_binary(repo_root)
    default_threads = min(4, os.cpu_count() or 1)

    parser = argparse.ArgumentParser(
        description="TPC-H HLL benchmarking (exact COUNT DISTINCT vs quack HLL via the local DuckDB CLI)."
    )
    parser.add_argument(
        "--scale-factors",
        nargs="+",
        type=int,
        default=[1],
        help="TPC-H scale factors to benchmark (e.g. --scale-factors 1 10 100).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of warm runs used for statistics (first run is discarded as cold).",
    )
    parser.add_argument("--error-threshold", type=float, default=0.02, help="Max allowed relative error.")
    parser.add_argument("--threads", type=int, default=default_threads, help="PRAGMA threads value for benchmark runs.")
    parser.add_argument("--database-dir", type=Path, default=default_db_dir, help="Directory holding TPC-H databases.")
    parser.add_argument("--output-dir", type=Path, default=default_out, help="Directory for CSV/JSON/charts.")
    parser.add_argument(
        "--duckdb-binary",
        type=Path,
        default=default_duckdb,
        help="Path to the locally built DuckDB CLI binary that includes the quack extension.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help="Target columns in table.column format (e.g., lineitem.l_orderkey).",
    )
    parser.add_argument(
        "--force-dbgen",
        action="store_true",
        help="Force regenerate each requested TPC-H database even if tables already exist.",
    )
    return parser.parse_args()


def summarize(times: Sequence[float]) -> tuple[float, float]:
    if not times:
        return math.nan, math.nan
    if len(times) == 1:
        return times[0], 0.0
    return statistics.median(times), statistics.stdev(times)


def summarize_by_scale_factor(rows: Sequence[MetricRow]) -> list[ScaleSummaryRow]:
    summaries: list[ScaleSummaryRow] = []
    for scale_factor in sorted({row.scale_factor for row in rows}):
        grouped = [row for row in rows if row.scale_factor == scale_factor]
        summaries.append(
            ScaleSummaryRow(
                scale_factor=scale_factor,
                target_count=len(grouped),
                mean_exact_median_s=statistics.mean(row.exact_median_s for row in grouped),
                mean_hll_median_s=statistics.mean(row.hll_median_s for row in grouped),
                mean_relative_error=statistics.mean(row.relative_error for row in grouped),
                max_relative_error=max(row.relative_error for row in grouped),
                mean_speedup_exact_over_hll=statistics.mean(row.speedup_exact_over_hll for row in grouped),
            )
        )
    return summaries


def write_outputs(rows: Sequence[MetricRow], output_dir: Path) -> list[ScaleSummaryRow]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = summarize_by_scale_factor(rows)

    csv_path = output_dir / "tpch_hll_eval_results.csv"
    summary_csv_path = output_dir / "tpch_hll_eval_scale_summary.csv"
    json_path = output_dir / "tpch_hll_eval_results.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MetricRow.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ScaleSummaryRow.__annotations__.keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(asdict(row))

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "rows": [asdict(row) for row in rows],
                "scale_factor_summary": [asdict(row) for row in summary_rows],
            },
            handle,
            indent=2,
        )

    print(f"[info] Wrote CSV:         {csv_path}")
    print(f"[info] Wrote summary CSV: {summary_csv_path}")
    print(f"[info] Wrote JSON:        {json_path}")
    return summary_rows


def plot_outputs(rows: Sequence[MetricRow], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    ordered_rows = sorted(rows, key=lambda row: (row.scale_factor, row.table_name, row.column_name))
    labels = [f"SF{row.scale_factor}\n{row.table_name}.{row.column_name}" for row in ordered_rows]
    exact_times = [row.exact_median_s for row in ordered_rows]
    hll_times = [row.hll_median_s for row in ordered_rows]

    fig, ax = plt.subplots(figsize=(13, 6))
    x_positions = range(len(labels))
    width = 0.38
    ax.bar([index - width / 2 for index in x_positions], exact_times, width=width, label="Exact COUNT(DISTINCT)")
    ax.bar([index + width / 2 for index in x_positions], hll_times, width=width, label="HLL (quack)")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Median Time (seconds)")
    ax.set_title("TPC-H Query Time: Exact vs HLL")
    ax.legend()
    fig.tight_layout()
    time_bar_path = output_dir / "tpch_time_comparison_bar.png"
    fig.savefig(time_bar_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for scale_factor in sorted({row.scale_factor for row in ordered_rows}):
        subset = sorted(
            (row for row in ordered_rows if row.scale_factor == scale_factor),
            key=lambda row: row.exact_count,
        )
        ax.plot(
            [row.exact_count for row in subset],
            [row.relative_error * 100.0 for row in subset],
            marker="o",
            linewidth=2.0,
            label=f"SF {scale_factor}",
        )
    ax.axhline(2.0, color="red", linestyle="--", linewidth=1.5, label="2% threshold")
    ax.set_xscale("log")
    ax.set_xlabel("Exact Distinct Count")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("HLL Relative Error vs Cardinality")
    ax.legend()
    fig.tight_layout()
    error_line_path = output_dir / "tpch_relative_error_line.png"
    fig.savefig(error_line_path, dpi=160)
    plt.close(fig)

    summary_rows = summarize_by_scale_factor(ordered_rows)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x_positions = range(len(summary_rows))
    width = 0.35
    ax.bar(
        [index - width / 2 for index in x_positions],
        [row.mean_exact_median_s for row in summary_rows],
        width=width,
        label="Exact mean median time",
    )
    ax.bar(
        [index + width / 2 for index in x_positions],
        [row.mean_hll_median_s for row in summary_rows],
        width=width,
        label="HLL mean median time",
    )
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([f"SF {row.scale_factor}" for row in summary_rows])
    ax.set_ylabel("Average Median Time Across Targets (seconds)")
    ax.set_title("Scalability by TPC-H Scale Factor")
    ax.legend()
    fig.tight_layout()
    scalability_path = output_dir / "tpch_scalability_grouped.png"
    fig.savefig(scalability_path, dpi=160)
    plt.close(fig)

    print(f"[info] Wrote chart: {time_bar_path}")
    print(f"[info] Wrote chart: {error_line_path}")
    print(f"[info] Wrote chart: {scalability_path}")


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")

    args.database_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[MetricRow] = []
    pre_statements = [f"PRAGMA threads={args.threads}"]

    for scale_factor in args.scale_factors:
        database_path = args.database_dir / f"tpch_sf{scale_factor}_eval.duckdb"
        ensure_tpch_data(database_path, scale_factor, args.force_dbgen)
        runner = DuckDBCliRunner(args.duckdb_binary, database_path)

        print(f"[info] Running benchmarks for scale factor {scale_factor} using {database_path.name}")
        for target in args.targets:
            table_name, column_name = split_target(target)
            print(f"[info] Benchmarking target: SF{scale_factor} {table_name}.{column_name}")

            exact_sql = f"SELECT COUNT(DISTINCT {column_name})::BIGINT FROM {table_name}"
            hll_sql = (
                f"SELECT hll_estimate(hll_create_agg(CAST({column_name} AS VARCHAR)))::BIGINT "
                f"FROM {table_name}"
            )

            exact_run = runner.run_scalar_query_repeated(
                exact_sql,
                repeats=args.iterations + 1,
                pre_statements=pre_statements,
            )
            hll_run = runner.run_scalar_query_repeated(
                hll_sql,
                repeats=args.iterations + 1,
                pre_statements=pre_statements,
            )

            exact_values = {int(value) for value in exact_run.values}
            hll_values = {int(value) for value in hll_run.values}
            if len(exact_values) != 1:
                raise RuntimeError(f"Exact query returned inconsistent values: {sorted(exact_values)}")
            if len(hll_values) != 1:
                raise RuntimeError(f"HLL query returned inconsistent values: {sorted(hll_values)}")

            exact_count = exact_values.pop()
            hll_estimate = hll_values.pop()

            exact_warm_times = exact_run.times_s[1:]
            hll_warm_times = hll_run.times_s[1:]
            exact_median, exact_std = summarize(exact_warm_times)
            hll_median, hll_std = summarize(hll_warm_times)

            relative_error = abs(hll_estimate - exact_count) / exact_count if exact_count else 0.0
            speedup = (exact_median / hll_median) if hll_median > 0 else math.inf

            row = MetricRow(
                scale_factor=scale_factor,
                table_name=table_name,
                column_name=column_name,
                exact_count=exact_count,
                hll_estimate=hll_estimate,
                relative_error=relative_error,
                exact_median_s=exact_median,
                exact_std_s=exact_std,
                hll_median_s=hll_median,
                hll_std_s=hll_std,
                speedup_exact_over_hll=speedup,
            )
            rows.append(row)

            print(
                "[result] "
                f"sf={scale_factor}, exact={exact_count}, hll={hll_estimate}, "
                f"rel_error={relative_error:.4%}, exact_med={exact_median:.4f}s, "
                f"hll_med={hll_median:.4f}s, speedup={speedup:.2f}x"
            )

            if relative_error > args.error_threshold:
                raise AssertionError(
                    f"Relative error {relative_error:.4%} exceeds threshold "
                    f"{args.error_threshold:.2%} for SF{scale_factor} {table_name}.{column_name}"
                )

    write_outputs(rows, args.output_dir)
    plot_outputs(rows, args.output_dir)
    print("[done] TPC-H HLL evaluation finished.")


if __name__ == "__main__":
    main()
