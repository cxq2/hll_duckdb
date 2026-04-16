# DuckDB HLL Extension

This repository is our DuckDB extension project for reusable HyperLogLog
(HLL) sketches.

The main point is different from DuckDB built-in approximate distinct.
Built-in function gives one final number. Our extension gives sketch objects as
`BLOB`, so they can be:

- created with `hll_create_agg(...)`
- stored in tables
- merged later with `hll_merge(...)` or `hll_merge_agg(...)`
- converted back to estimates with `hll_estimate(...)`

This repo includes:

- the HLL extension implementation
- SQL regression tests for Role B
- benchmark scripts for Role C
- a small standalone demo showing sketch reuse across queries

## Repository Layout

- `src/quack_extension.cpp`
  DuckDB function registration and bindings.
- `src/hll/`
  HLL core code used by the extension.
- `test/sql/quack.test`
  SQL tests, including sparse/dense merge cases.
- `benchmark/tpch_hll_eval.py`
  Accuracy and runtime test against `COUNT(DISTINCT)`.
- `benchmark/merge_workflow_eval.py`
  Pre-aggregate and merge workflow test.
- `benchmark/sketch_workflow_demo.py`
  Small demo for reusable sketch `BLOB` objects.

## Prerequisites

Required tools:

- C++ toolchain with CMake and Make
- Python 3
- Git submodule support

After clone, initialize the DuckDB submodule:

```bash
git submodule update --init --recursive
```

If you clone from scratch, this is the safer way:

```bash
git clone --recurse-submodules <repo-url>
```

Install Python packages for benchmark and demo:

```bash
python3 -m pip install -r benchmark/requirements.txt
```

## Build

Run from repository root:

```bash
make -j4
```

This will generate:

- `build/release/duckdb`
- `build/release/extension/quack/quack.duckdb_extension`

## Verify The Extension

Run SQL regression tests:

```bash
make test
```

Expected result: all assertions pass.

You can also do a small manual check:

```bash
./build/release/duckdb -unsigned
```

Then run this SQL:

```sql
SELECT hll_estimate(hll_create_agg(v::VARCHAR))
FROM (SELECT * FROM range(1000)) t(v);
```

## Benchmark And Evaluation

Benchmark details are in:

- `benchmark/README.md`

Quick benchmark check:

```bash
python3 benchmark/tpch_hll_eval.py --scale-factors 1 --iterations 1 --threads 1
python3 benchmark/merge_workflow_eval.py --scale-factor 1 --iterations 1 --threads 1
```

## Small Demo For Presentation

To make a small demo for presentation:

```bash
python3 benchmark/sketch_workflow_demo.py
```

This creates a raw table, a sketch table, and output files in:

- `benchmark/out/sketch_workflow_demo/`

The demo shows:

- create sketches once
- store them as `BLOB` values in SQL tables
- reuse and merge them later across different queries

## Notes For Reproducibility

- The extension itself does not require OpenSSL or other external crypto
  libraries.
- The benchmark scripts use DuckDB's `tpch` extension to run `CALL dbgen(...)`.
  On first run, DuckDB may need network access to download that extension.
- Do not run multiple benchmark scripts against the same `.duckdb` database file
  at the same time.
