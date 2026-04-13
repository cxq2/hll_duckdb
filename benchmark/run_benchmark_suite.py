#!/usr/bin/env python3
"""Run the full Role C benchmark suite with one command."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from duckdb_cli_runner import detect_default_duckdb_binary


DEFAULT_TARGETS = [
    "lineitem.l_orderkey",
    "orders.o_custkey",
    "customer.c_custkey",
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_output_root = repo_root / "benchmark" / "out"
    default_database_dir = default_output_root / "databases"
    default_duckdb = detect_default_duckdb_binary(repo_root)
    default_threads = min(4, os.cpu_count() or 1)

    parser = argparse.ArgumentParser(description="Run Role C TPC-H evaluation and merge workflow benchmarks.")
    parser.add_argument("--scale-factors", nargs="+", type=int, default=[1, 10, 100], help="Scale factors for the main evaluation.")
    parser.add_argument(
        "--merge-scale-factors",
        nargs="+",
        type=int,
        default=[10, 100],
        help="Scale factors for merge-workflow validation.",
    )
    parser.add_argument("--iterations", type=int, default=3, help="Warm-run iterations for each benchmark query.")
    parser.add_argument("--threads", type=int, default=default_threads, help="PRAGMA threads value used by the local DuckDB CLI.")
    parser.add_argument("--database-dir", type=Path, default=default_database_dir, help="Directory holding TPC-H databases.")
    parser.add_argument("--output-root", type=Path, default=default_output_root, help="Root directory for benchmark outputs.")
    parser.add_argument(
        "--duckdb-binary",
        type=Path,
        default=default_duckdb,
        help="Path to the locally built DuckDB CLI binary that includes the quack extension.",
    )
    parser.add_argument("--force-dbgen", action="store_true", help="Force regenerate TPC-H databases before benchmarking.")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help="Target columns used by the main evaluation script.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print(f"[run] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    benchmark_dir = Path(__file__).resolve().parent
    eval_script = benchmark_dir / "tpch_hll_eval.py"
    merge_script = benchmark_dir / "merge_workflow_eval.py"
    python_exe = Path(sys.executable).resolve()

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.database_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = args.output_root / "tpch_hll_eval_report"
    eval_command = [
        str(python_exe),
        str(eval_script),
        "--scale-factors",
        *[str(scale_factor) for scale_factor in args.scale_factors],
        "--iterations",
        str(args.iterations),
        "--threads",
        str(args.threads),
        "--database-dir",
        str(args.database_dir),
        "--output-dir",
        str(eval_output_dir),
        "--duckdb-binary",
        str(args.duckdb_binary),
        "--targets",
        *args.targets,
    ]
    if args.force_dbgen:
        eval_command.append("--force-dbgen")
    run_command(eval_command)

    for scale_factor in args.merge_scale_factors:
        merge_output_dir = args.output_root / f"merge_workflow_sf{scale_factor}"
        merge_command = [
            str(python_exe),
            str(merge_script),
            "--scale-factor",
            str(scale_factor),
            "--iterations",
            str(args.iterations),
            "--threads",
            str(args.threads),
            "--database-dir",
            str(args.database_dir),
            "--output-dir",
            str(merge_output_dir),
            "--duckdb-binary",
            str(args.duckdb_binary),
        ]
        if args.force_dbgen:
            merge_command.append("--force-dbgen")
        run_command(merge_command)

    print("[done] Role C benchmark suite finished.")


if __name__ == "__main__":
    main()
