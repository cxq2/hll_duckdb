# Benchmark Guide

This folder contains the Role C benchmark scripts used to evaluate the `quack`
HLL extension.

## Prerequisites

1. Build the extension and local DuckDB CLI:

```bash
PATH="/opt/homebrew/bin:$PATH" make -j4
```

2. Install the Python dependencies:

```bash
python3 -m pip install -r benchmark/requirements.txt
```

3. Ensure the machine has network access on first run.

The benchmark scripts use DuckDB's `tpch` extension to execute
`CALL dbgen(...)`. DuckDB may need to download that extension the first time.

## Quick Verification

Run a minimal end-to-end smoke test:

```bash
python3 benchmark/tpch_hll_eval.py --scale-factors 1 --iterations 1 --threads 1
python3 benchmark/merge_workflow_eval.py --scale-factor 1 --iterations 1 --threads 1
```

## Small Demo

Generate a compact demo artifact for rehearsal, screen recording, or slides:

```bash
python3 benchmark/sketch_workflow_demo.py
```

This creates:

- `benchmark/out/sketch_workflow_demo/sketch_workflow_demo.md`
- `benchmark/out/sketch_workflow_demo/sketch_workflow_demo.json`
- `benchmark/out/sketch_workflow_demo/sketch_workflow_demo.png`

The demo avoids `dbgen` and external downloads. It uses a small synthetic table
to show the core workflow:

- create sketches
- store sketches in a SQL table as `BLOB`
- reuse the stored sketches across multiple queries
- merge sketches instead of rescanning the raw table

## Full Suite

Run the combined benchmark entrypoint:

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

- The scripts auto-detect a local DuckDB CLI binary from `build/release`.
- Benchmark outputs are written under `benchmark/out/`.
- Do not run multiple benchmark scripts against the same database file at the
  same time; DuckDB will reject concurrent writers.
