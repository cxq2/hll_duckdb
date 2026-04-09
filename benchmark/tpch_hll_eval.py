#!/usr/bin/env python3
"""TPC-H HLL accuracy & performance evaluation (Role C: C1/C2/C3/C4).

This script:
1. Generates TPC-H data via CALL dbgen(sf=1) (or reuses existing tables)
2. Runs exact COUNT(DISTINCT ...) vs HLL queries using quack extension
3. Uses warm-run timing (discard first cold run), reports median/stddev
4. Computes relative error and asserts it is <= threshold (default 2%)
5. Writes CSV/JSON outputs and generates charts
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence, Tuple

import duckdb

# Keep matplotlib cache writable in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_TARGETS = [
    "lineitem.l_orderkey",
    "orders.o_custkey",
    "partsupp.ps_partkey",
]


@dataclass
class MetricRow:
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


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_db = repo_root / "benchmark" / "out" / "tpch_sf1_eval.duckdb"
    default_out = repo_root / "benchmark" / "out" / "tpch_hll_eval"
    default_ext = repo_root / "build" / "release" / "extension" / "quack" / "quack.duckdb_extension"

    parser = argparse.ArgumentParser(description="TPC-H HLL benchmarking (exact vs quack HLL).")
    parser.add_argument("--scale-factor", type=int, default=1, help="TPC-H scale factor for CALL dbgen(sf=?).")
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of warm runs used for statistics (first run is discarded as cold).",
    )
    parser.add_argument("--error-threshold", type=float, default=0.02, help="Max allowed relative error.")
    parser.add_argument("--database", type=Path, default=default_db, help="DuckDB database file path.")
    parser.add_argument("--output-dir", type=Path, default=default_out, help="Directory for CSV/JSON/charts.")
    parser.add_argument(
        "--extension-path",
        type=Path,
        default=default_ext,
        help="Path to quack.duckdb_extension built artifact.",
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
        help="Force regenerate TPC-H tables even if they already exist.",
    )
    return parser.parse_args()


def split_target(target: str) -> Tuple[str, str]:
    if "." not in target:
        raise ValueError(f"Invalid target '{target}', expected table.column")
    table, col = target.split(".", 1)
    if not table or not col:
        raise ValueError(f"Invalid target '{target}', expected table.column")
    return table, col


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


def timed_query(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    warm_runs: int,
    cold_runs: int = 1,
) -> Tuple[int, List[float], List[float]]:
    cold_times: List[float] = []
    warm_times: List[float] = []
    result: int | None = None

    total = cold_runs + warm_runs
    for i in range(total):
        start = time.perf_counter()
        result = int(con.execute(sql).fetchone()[0])
        elapsed = time.perf_counter() - start
        if i < cold_runs:
            cold_times.append(elapsed)
        else:
            warm_times.append(elapsed)

    if result is None:
        raise RuntimeError("Query returned no result")
    return result, cold_times, warm_times


def summarize(times: Sequence[float]) -> Tuple[float, float]:
    if not times:
        return math.nan, math.nan
    if len(times) == 1:
        return times[0], 0.0
    return statistics.median(times), statistics.stdev(times)


def write_outputs(rows: Sequence[MetricRow], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "tpch_hll_eval_results.csv"
    json_path = output_dir / "tpch_hll_eval_results.json"

    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)

    print(f"[info] Wrote CSV:  {csv_path}")
    print(f"[info] Wrote JSON: {json_path}")


def plot_outputs(rows: Sequence[MetricRow], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    labels = [f"{r.table_name}.{r.column_name}" for r in rows]
    exact_times = [r.exact_median_s for r in rows]
    hll_times = [r.hll_median_s for r in rows]
    rel_errors_pct = [r.relative_error * 100.0 for r in rows]

    # Bar chart: exact vs HLL median query time
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = range(len(labels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], exact_times, width=width, label="Exact COUNT(DISTINCT)")
    ax.bar([i + width / 2 for i in x], hll_times, width=width, label="HLL (quack)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Median Time (seconds)")
    ax.set_title("TPC-H Query Time: Exact vs HLL")
    ax.legend()
    fig.tight_layout()
    bar_path = output_dir / "tpch_time_comparison_bar.png"
    fig.savefig(bar_path, dpi=160)
    plt.close(fig)

    # Line chart: relative error by target
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(labels, rel_errors_pct, marker="o", linewidth=2.0)
    ax.axhline(2.0, color="red", linestyle="--", linewidth=1.5, label="2% threshold")
    ax.set_ylabel("Relative Error (%)")
    ax.set_xlabel("Target Column")
    ax.set_title("HLL Relative Error by Target Column")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    line_path = output_dir / "tpch_relative_error_line.png"
    fig.savefig(line_path, dpi=160)
    plt.close(fig)

    print(f"[info] Wrote chart: {bar_path}")
    print(f"[info] Wrote chart: {line_path}")


def main() -> None:
    args = parse_args()

    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")

    args.database.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(args.database), config={"allow_unsigned_extensions": "true"})

    ensure_tpch_data(con, args.scale_factor, args.force_dbgen)
    load_quack_extension(con, args.extension_path)

    rows: List[MetricRow] = []

    for target in args.targets:
        table_name, column_name = split_target(target)
        print(f"[info] Benchmarking target: {table_name}.{column_name}")

        exact_sql = f"SELECT COUNT(DISTINCT {column_name})::BIGINT FROM {table_name}"
        hll_sql = (
            f"SELECT hll_estimate(hll_create_agg(CAST({column_name} AS VARCHAR)))::BIGINT "
            f"FROM {table_name}"
        )

        exact_count, _, exact_warm_times = timed_query(con, exact_sql, warm_runs=args.iterations)
        hll_estimate, _, hll_warm_times = timed_query(con, hll_sql, warm_runs=args.iterations)

        exact_median, exact_std = summarize(exact_warm_times)
        hll_median, hll_std = summarize(hll_warm_times)

        rel_error = abs(hll_estimate - exact_count) / exact_count if exact_count else 0.0
        speedup = (exact_median / hll_median) if hll_median > 0 else math.inf

        row = MetricRow(
            table_name=table_name,
            column_name=column_name,
            exact_count=exact_count,
            hll_estimate=hll_estimate,
            relative_error=rel_error,
            exact_median_s=exact_median,
            exact_std_s=exact_std,
            hll_median_s=hll_median,
            hll_std_s=hll_std,
            speedup_exact_over_hll=speedup,
        )
        rows.append(row)

        print(
            "[result] "
            f"exact={exact_count}, hll={hll_estimate}, "
            f"rel_error={rel_error:.4%}, exact_med={exact_median:.4f}s, "
            f"hll_med={hll_median:.4f}s, speedup={speedup:.2f}x"
        )

        if rel_error > args.error_threshold:
            raise AssertionError(
                f"Relative error {rel_error:.4%} exceeds threshold {args.error_threshold:.2%} "
                f"for {table_name}.{column_name}"
            )

    write_outputs(rows, args.output_dir)
    plot_outputs(rows, args.output_dir)

    con.close()
    print("[done] TPC-H HLL evaluation finished.")


if __name__ == "__main__":
    main()
