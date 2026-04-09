#!/usr/bin/env python3
"""Merge workflow validation (Role C: C5).

This script validates the key workflow:
1) Pre-aggregate per partition into sketches
2) Merge sketches at query time via hll_merge_agg
3) Compare against single-scan HLL and exact distinct baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence, Tuple

import duckdb

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class MergeWorkflowResult:
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
    default_db = repo_root / "benchmark" / "out" / "tpch_sf1_eval.duckdb"
    default_out = repo_root / "benchmark" / "out" / "merge_workflow_eval"
    default_ext = repo_root / "build" / "release" / "extension" / "quack" / "quack.duckdb_extension"

    parser = argparse.ArgumentParser(description="Validate pre-aggregate + merge workflow for HLL sketches.")
    parser.add_argument("--scale-factor", type=int, default=1, help="TPC-H scale factor for dbgen.")
    parser.add_argument("--iterations", type=int, default=5, help="Warm runs for query timing (cold run discarded).")
    parser.add_argument("--error-threshold", type=float, default=0.02, help="Relative error threshold for assertions.")
    parser.add_argument(
        "--consistency-threshold",
        type=float,
        default=0.01,
        help="Max relative diff between merge estimate and single-scan estimate.",
    )
    parser.add_argument("--partition-column", type=str, default="l_shipdate", help="lineitem partition column.")
    parser.add_argument("--database", type=Path, default=default_db, help="DuckDB database file.")
    parser.add_argument("--output-dir", type=Path, default=default_out, help="Output directory for results.")
    parser.add_argument("--extension-path", type=Path, default=default_ext, help="Path to quack extension binary.")
    parser.add_argument("--force-dbgen", action="store_true", help="Force regenerate TPCH data.")
    return parser.parse_args()


def sanitize_identifier(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Unsafe SQL identifier: {name}")
    return name


def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    row = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = current_schema()
          AND table_name = ?
        """,
        [table_name],
    ).fetchone()
    return bool(row and row[0] > 0)


def ensure_tpch_data(con: duckdb.DuckDBPyConnection, sf: int, force_dbgen: bool) -> None:
    required_tables = {"lineitem", "orders", "partsupp"}
    has_all = all(table_exists(con, t) for t in required_tables)

    if has_all and not force_dbgen:
        print("[info] Reusing existing TPC-H tables.")
        return

    print(f"[info] Generating TPC-H tables with CALL dbgen(sf={sf}) ...")
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")

    if force_dbgen:
        for t in required_tables:
            con.execute(f"DROP TABLE IF EXISTS {t}")

    start = time.perf_counter()
    con.execute(f"CALL dbgen(sf={sf})")
    elapsed = time.perf_counter() - start
    print(f"[info] dbgen completed in {elapsed:.3f}s")


def load_quack_extension(con: duckdb.DuckDBPyConnection, extension_path: Path) -> None:
    extension_path = extension_path.resolve()
    if not extension_path.exists():
        raise FileNotFoundError(f"Extension binary not found: {extension_path}")

    try:
        con.execute(f"LOAD '{extension_path.as_posix()}'")
    except Exception as ex:
        raise RuntimeError(
            "Failed to load quack extension. This is usually a DuckDB version mismatch "
            "between Python package and quack.duckdb_extension. "
            f"Extension path: {extension_path}\nOriginal error: {ex}"
        ) from ex


def timed_scalar_query(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    warm_runs: int,
    cold_runs: int = 1,
) -> Tuple[int, List[float]]:
    result: int | None = None
    warm_times: List[float] = []

    total = cold_runs + warm_runs
    for i in range(total):
        start = time.perf_counter()
        result = int(con.execute(sql).fetchone()[0])
        elapsed = time.perf_counter() - start
        if i >= cold_runs:
            warm_times.append(elapsed)

    if result is None:
        raise RuntimeError("Query returned no result")
    return result, warm_times


def summarize(times: Sequence[float]) -> Tuple[float, float]:
    if not times:
        return math.nan, math.nan
    if len(times) == 1:
        return times[0], 0.0
    return statistics.median(times), statistics.stdev(times)


def write_outputs(result: MergeWorkflowResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "merge_workflow_eval.csv"
    json_path = output_dir / "merge_workflow_eval.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(result).keys()))
        writer.writeheader()
        writer.writerow(asdict(result))

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"[info] Wrote CSV:  {csv_path}")
    print(f"[info] Wrote JSON: {json_path}")


def plot_outputs(result: MergeWorkflowResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    labels = ["pre_agg", "merge_query", "single_scan"]
    values = [
        result.pre_agg_time_s,
        result.merge_query_median_s,
        result.single_query_median_s,
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(labels, values)
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Merge Workflow Timing")
    fig.tight_layout()

    chart_path = output_dir / "merge_workflow_timing_bar.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    print(f"[info] Wrote chart: {chart_path}")


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")

    partition_col = sanitize_identifier(args.partition_column)

    args.database.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(args.database), config={"allow_unsigned_extensions": "true"})

    ensure_tpch_data(con, args.scale_factor, args.force_dbgen)
    load_quack_extension(con, args.extension_path)

    con.execute("DROP TABLE IF EXISTS pre_agg_sketches")

    pre_agg_sql = (
        "CREATE TABLE pre_agg_sketches AS "
        f"SELECT {partition_col}, hll_create_agg(CAST(l_orderkey AS VARCHAR)) AS sketch "
        "FROM lineitem GROUP BY 1"
    )

    print(f"[info] Running pre-aggregation by {partition_col} ...")
    t0 = time.perf_counter()
    con.execute(pre_agg_sql)
    pre_agg_time = time.perf_counter() - t0

    partition_count = int(con.execute("SELECT COUNT(*) FROM pre_agg_sketches").fetchone()[0])

    exact_sql = "SELECT COUNT(DISTINCT l_orderkey)::BIGINT FROM lineitem"
    merge_sql = "SELECT hll_estimate(hll_merge_agg(sketch))::BIGINT FROM pre_agg_sketches"
    single_sql = "SELECT hll_estimate(hll_create_agg(CAST(l_orderkey AS VARCHAR)))::BIGINT FROM lineitem"

    exact_count = int(con.execute(exact_sql).fetchone()[0])
    merge_estimate, merge_times = timed_scalar_query(con, merge_sql, warm_runs=args.iterations)
    single_estimate, single_times = timed_scalar_query(con, single_sql, warm_runs=args.iterations)

    merge_median, merge_std = summarize(merge_times)
    single_median, single_std = summarize(single_times)

    merge_rel_err = abs(merge_estimate - exact_count) / exact_count if exact_count else 0.0
    single_rel_err = abs(single_estimate - exact_count) / exact_count if exact_count else 0.0
    merge_vs_single_diff = abs(merge_estimate - single_estimate) / exact_count if exact_count else 0.0

    result = MergeWorkflowResult(
        partition_column=partition_col,
        partition_count=partition_count,
        exact_distinct=exact_count,
        merge_estimate=merge_estimate,
        single_scan_estimate=single_estimate,
        merge_relative_error=merge_rel_err,
        single_scan_relative_error=single_rel_err,
        merge_vs_single_relative_diff=merge_vs_single_diff,
        pre_agg_time_s=pre_agg_time,
        merge_query_median_s=merge_median,
        merge_query_std_s=merge_std,
        single_query_median_s=single_median,
        single_query_std_s=single_std,
    )

    print(
        "[result] "
        f"exact={exact_count}, merge={merge_estimate}, single={single_estimate}, "
        f"merge_err={merge_rel_err:.4%}, single_err={single_rel_err:.4%}, "
        f"merge_vs_single={merge_vs_single_diff:.4%}, pre_agg={pre_agg_time:.4f}s"
    )

    if merge_rel_err > args.error_threshold:
        raise AssertionError(
            f"Merge workflow relative error {merge_rel_err:.4%} exceeds threshold {args.error_threshold:.2%}"
        )
    if single_rel_err > args.error_threshold:
        raise AssertionError(
            f"Single-scan HLL relative error {single_rel_err:.4%} exceeds threshold {args.error_threshold:.2%}"
        )
    if merge_vs_single_diff > args.consistency_threshold:
        raise AssertionError(
            f"Merge vs single relative diff {merge_vs_single_diff:.4%} exceeds threshold {args.consistency_threshold:.2%}"
        )

    write_outputs(result, args.output_dir)
    plot_outputs(result, args.output_dir)

    con.close()
    print("[done] Merge workflow evaluation finished.")


if __name__ == "__main__":
    main()
