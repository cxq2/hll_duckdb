# Benchmark Guide

This folder contains the Role C benchmark scripts for the `quack` HLL
extension.

## Prerequisites

1. Build the extension and local DuckDB CLI:

```bash
PATH="/opt/homebrew/bin:$PATH" make -j4
```

2. Install Python dependencies:

```bash
python3 -m pip install -r benchmark/requirements.txt
```

3. Make sure the machine has network on first run.

The scripts use DuckDB `tpch` extension for `CALL dbgen(...)`. DuckDB may need
to download it the first time.

## Quick Verification

Run a small end-to-end test:

```bash
python3 benchmark/tpch_hll_eval.py --scale-factors 1 --iterations 1 --threads 1
python3 benchmark/merge_workflow_eval.py --scale-factor 1 --iterations 1 --threads 1
```

## Small Demo

Generate a small demo for rehearsal, recording, or slides:

```bash
python3 benchmark/sketch_workflow_demo.py
```

This will create:

- `benchmark/out/sketch_workflow_demo/sketch_workflow_demo.md`
- `benchmark/out/sketch_workflow_demo/sketch_workflow_demo.json`
- `benchmark/out/sketch_workflow_demo/sketch_workflow_demo.png`

This demo does not use `dbgen` or external download. It uses a small synthetic
table to show the workflow:

- create sketches
- store sketches in a SQL table as `BLOB`
- reuse the stored sketches across multiple queries
- merge sketches instead of rescanning the raw table

## Full Suite

Run the full benchmark entry:

```bash
python3 benchmark/run_benchmark_suite.py
```

Useful flags:

- `--scale-factors 1 10 100`
- `--merge-scale-factors 10 100`
- `--iterations 3`
- `--threads 4`
- `--force-dbgen`
- `--duckdb-binary /path/to/duckdb`

## Notes

- The scripts auto-detect the local DuckDB CLI binary from `build/release`.
- Benchmark outputs are written under `benchmark/out/`.
- Do not run multiple benchmark scripts against the same database file at the
  same time; DuckDB will reject concurrent writers.
