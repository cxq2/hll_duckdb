#!/usr/bin/env python3
"""Small demo for showing sketches as explicit SQL objects.

This script creates:
1. A raw event table with duplicated user IDs across days
2. A sketch table that stores one HLL BLOB per day
3. A short Markdown summary plus a compact PNG figure for demos/slides
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import duckdb

from duckdb_cli_runner import DuckDBCliRunner, detect_default_duckdb_binary


@dataclass
class QueryWindowResult:
    label: str
    raw_rows: int
    sketch_rows: int
    exact_distinct: int
    merged_estimate: int
    relative_error: float


@dataclass
class DemoSummary:
    raw_rows: int
    sketch_rows: int
    sketch_sql_type: str
    total_sketch_bytes: int
    total_exact_distinct: int
    total_hll_from_raw: int
    total_hll_from_sketches: int
    total_relative_error: float
    windows: list[QueryWindowResult]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_output_dir = repo_root / "benchmark" / "out" / "sketch_workflow_demo"
    default_database = default_output_dir / "sketch_workflow_demo.duckdb"
    default_duckdb = detect_default_duckdb_binary(repo_root)

    parser = argparse.ArgumentParser(description="Generate a small sketch-workflow demo artifact.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help="Directory for demo outputs.")
    parser.add_argument("--database", type=Path, default=default_database, help="DuckDB database used for the demo.")
    parser.add_argument(
        "--duckdb-binary",
        type=Path,
        default=default_duckdb,
        help="Path to the local DuckDB CLI binary that includes the quack extension.",
    )
    return parser.parse_args()


def parse_csv_rows(stdout: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in stdout.splitlines():
        if not raw_line.strip():
            continue
        rows.append(next(csv.reader([raw_line])))
    return rows


def query_rows(runner: DuckDBCliRunner, sql: str) -> list[list[str]]:
    completed = runner.run_script([sql])
    return parse_csv_rows(completed.stdout)


def query_scalar_int(runner: DuckDBCliRunner, sql: str) -> int:
    rows = query_rows(runner, sql)
    if len(rows) != 1 or len(rows[0]) != 1:
        raise RuntimeError(f"Expected a single scalar result for query: {sql}")
    return int(rows[0][0])


def query_scalar_str(runner: DuckDBCliRunner, sql: str) -> str:
    rows = query_rows(runner, sql)
    if len(rows) != 1 or len(rows[0]) != 1:
        raise RuntimeError(f"Expected a single scalar result for query: {sql}")
    return rows[0][0]


def prepare_demo_database(runner: DuckDBCliRunner) -> None:
    runner.run_script(
        [
            "DROP TABLE IF EXISTS raw_events",
            "DROP TABLE IF EXISTS daily_sketches",
            """
            CREATE TABLE raw_events AS
            WITH base AS (
                SELECT
                    i,
                    CAST(i % 7 AS INTEGER) AS day_idx
                FROM range(100000) t(i)
            )
            SELECT
                DATE '2024-01-01' + day_idx AS event_day,
                CONCAT(
                    'user_',
                    CAST(((((i * 37) % 7000) + (day_idx * 2000)) % 15000) AS VARCHAR)
                ) AS user_id,
                CONCAT('event_', CAST(i AS VARCHAR)) AS event_id
            FROM base
            """,
            """
            CREATE TABLE daily_sketches AS
            SELECT
                event_day,
                hll_create_agg(user_id) AS user_sketch
            FROM raw_events
            GROUP BY 1
            ORDER BY 1
            """,
        ]
    )


def summarize_window(runner: DuckDBCliRunner, label: str, start_date: str, end_date: str) -> QueryWindowResult:
    date_predicate = f"event_day BETWEEN DATE '{start_date}' AND DATE '{end_date}'"
    sketch_predicate = f"event_day BETWEEN DATE '{start_date}' AND DATE '{end_date}'"

    raw_rows = query_scalar_int(runner, f"SELECT COUNT(*) FROM raw_events WHERE {date_predicate}")
    sketch_rows = query_scalar_int(runner, f"SELECT COUNT(*) FROM daily_sketches WHERE {sketch_predicate}")
    exact_distinct = query_scalar_int(runner, f"SELECT COUNT(DISTINCT user_id) FROM raw_events WHERE {date_predicate}")
    merged_estimate = query_scalar_int(
        runner,
        f"SELECT hll_estimate(hll_merge_agg(user_sketch)) FROM daily_sketches WHERE {sketch_predicate}",
    )

    relative_error = abs(merged_estimate - exact_distinct) / exact_distinct if exact_distinct else 0.0
    return QueryWindowResult(
        label=label,
        raw_rows=raw_rows,
        sketch_rows=sketch_rows,
        exact_distinct=exact_distinct,
        merged_estimate=merged_estimate,
        relative_error=relative_error,
    )


def build_summary(runner: DuckDBCliRunner) -> DemoSummary:
    raw_rows = query_scalar_int(runner, "SELECT COUNT(*) FROM raw_events")
    sketch_rows = query_scalar_int(runner, "SELECT COUNT(*) FROM daily_sketches")
    sketch_sql_type = query_scalar_str(runner, "SELECT typeof(user_sketch) FROM daily_sketches LIMIT 1")
    total_sketch_bytes = query_scalar_int(runner, "SELECT SUM(octet_length(user_sketch)) FROM daily_sketches")
    total_exact_distinct = query_scalar_int(runner, "SELECT COUNT(DISTINCT user_id) FROM raw_events")
    total_hll_from_raw = query_scalar_int(runner, "SELECT hll_estimate(hll_create_agg(user_id)) FROM raw_events")
    total_hll_from_sketches = query_scalar_int(runner, "SELECT hll_estimate(hll_merge_agg(user_sketch)) FROM daily_sketches")
    total_relative_error = (
        abs(total_hll_from_sketches - total_exact_distinct) / total_exact_distinct if total_exact_distinct else 0.0
    )

    windows = [
        summarize_window(runner, "Jan 02 - Jan 04", "2024-01-02", "2024-01-04"),
        summarize_window(runner, "Jan 05 - Jan 07", "2024-01-05", "2024-01-07"),
    ]

    return DemoSummary(
        raw_rows=raw_rows,
        sketch_rows=sketch_rows,
        sketch_sql_type=sketch_sql_type,
        total_sketch_bytes=total_sketch_bytes,
        total_exact_distinct=total_exact_distinct,
        total_hll_from_raw=total_hll_from_raw,
        total_hll_from_sketches=total_hll_from_sketches,
        total_relative_error=total_relative_error,
        windows=windows,
    )


def write_summary(summary: DemoSummary, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "sketch_workflow_demo.json"
    md_path = output_dir / "sketch_workflow_demo.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **asdict(summary),
                "windows": [asdict(window) for window in summary.windows],
            },
            handle,
            indent=2,
        )

    lines = [
        "# Sketch Workflow Demo",
        "",
        "## Why this matters",
        "",
        "- `daily_sketches.user_sketch` is a real SQL `BLOB` column.",
        "- We create sketches once, store them in a table, and reuse them across multiple queries.",
        "- Merge queries touch only a few sketch rows instead of rescanning the full raw table.",
        "",
        "## Stored sketch table",
        "",
        f"- raw table rows: `{summary.raw_rows}`",
        f"- sketch table rows: `{summary.sketch_rows}`",
        f"- sketch SQL type: `{summary.sketch_sql_type}`",
        f"- total sketch bytes: `{summary.total_sketch_bytes}`",
        "",
        "## Whole-table result",
        "",
        f"- exact distinct users: `{summary.total_exact_distinct}`",
        f"- HLL from raw table: `{summary.total_hll_from_raw}`",
        f"- HLL from stored sketches: `{summary.total_hll_from_sketches}`",
        f"- relative error: `{summary.total_relative_error:.4%}`",
        "",
        "## Reused query windows",
        "",
        "| Window | Raw Rows | Sketch Rows | Exact Distinct | Merged Estimate | Relative Error |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for window in summary.windows:
        lines.append(
            f"| {window.label} | {window.raw_rows} | {window.sketch_rows} | "
            f"{window.exact_distinct} | {window.merged_estimate} | {window.relative_error:.4%} |"
        )

    lines.extend(
        [
            "",
            "## Demo line to say out loud",
            "",
            "We are not only returning an approximate distinct count. We are materializing sketches as SQL BLOB objects, "
            "storing them in a table, and reusing them later with `hll_merge_agg`.",
        ]
    )

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"[info] Wrote JSON: {json_path}")
    print(f"[info] Wrote Markdown: {md_path}")


def plot_summary(summary: DemoSummary, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_window = summary.windows[0]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    axes[0].bar(["raw_events", "daily_sketches"], [summary.raw_rows, summary.sketch_rows], color=["#4c72b0", "#55a868"])
    axes[0].set_title("Stored Objects")
    axes[0].set_ylabel("Row Count")

    axes[1].bar(
        ["raw range", "sketch range"],
        [primary_window.raw_rows, primary_window.sketch_rows],
        color=["#c44e52", "#8172b3"],
    )
    axes[1].set_title(f"Rows Touched\n{primary_window.label}")
    axes[1].set_ylabel("Rows Read")

    axes[2].bar(
        ["exact", "merged sketch"],
        [primary_window.exact_distinct, primary_window.merged_estimate],
        color=["#64b5cd", "#dd8452"],
    )
    axes[2].set_title(f"Distinct Result\nerror={primary_window.relative_error:.2%}")
    axes[2].set_ylabel("Distinct Users")

    for axis in axes:
        axis.bar_label(axis.containers[0], padding=3, fontsize=9)

    fig.suptitle("Sketches as Explicit SQL Objects", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    chart_path = output_dir / "sketch_workflow_demo.png"
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)
    print(f"[info] Wrote chart: {chart_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.database.parent.mkdir(parents=True, exist_ok=True)

    if args.database.exists():
        args.database.unlink()

    duckdb.connect(str(args.database)).close()
    runner = DuckDBCliRunner(args.duckdb_binary, args.database)

    prepare_demo_database(runner)
    summary = build_summary(runner)
    write_summary(summary, args.output_dir)
    plot_summary(summary, args.output_dir)

    print(
        "[done] "
        f"Created raw table ({summary.raw_rows} rows), sketch table ({summary.sketch_rows} rows), "
        f"and reusable sketch demo artifacts in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
