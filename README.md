# DuckDB HLL Extension

This repository contains a DuckDB extension that adds reusable HyperLogLog
(HLL) sketches as SQL-level objects.

The key difference from DuckDB's built-in approximate distinct support and the main innovation idea is that
this extension exposes sketches as `BLOB` values that can be:

- created with `hll_create_agg(...)`
- stored in tables
- merged later with `hll_merge(...)` or `hll_merge_agg(...)`
- converted back to estimates with `hll_estimate(...)`

This repository includes:

- the HLL extension implementation
- SQL regression tests for Role B
- benchmark scripts for Role C
- a small standalone demo showing sketch reuse across queries

## Repository Layout

- `src/quack_extension.cpp`
  DuckDB function registration and aggregate/scalar function bindings.
- `src/hll/`
  HLL core implementation used by the extension.
- `test/sql/quack.test`
  SQL regression tests, including sparse/dense merge cases.
- `benchmark/tpch_hll_eval.py`
  Accuracy and runtime evaluation against `COUNT(DISTINCT)`.
- `benchmark/merge_workflow_eval.py`
  Pre-aggregate and merge-workflow validation.
- `benchmark/sketch_workflow_demo.py`
  Small synthetic demo showing sketches as reusable SQL `BLOB` objects.

## Prerequisites

Required tools:

- C++ toolchain with CMake and Make
- Python 3
- Git submodule support

Quick check:

```bash
cmake --version
python3 --version
```

`cmake` should be installed and available in `PATH`.

If you are using macOS with Homebrew and `cmake` is installed but not found,
you may need:

```bash
export PATH="/opt/homebrew/bin:$PATH"
```

After finishing clone, you can initialize the DuckDB submodule:

```bash
git submodule update --init --recursive
```

If you are cloning from scratch, the safest option is:

```bash
git clone --recurse-submodules <repo-url>
```

Python packages for benchmark and demo scripts:

```bash
python3 -m pip install -r benchmark/requirements.txt
```

## Build

From the repository root:

```bash
make -j4
```

This can produce:

- `build/release/duckdb`
- `build/release/extension/quack/quack.duckdb_extension`

## Verify The Extension

Run the SQL regression test suite:

```bash
make test
```

The expected result is that all assertions pass.

You can also do a minimal manual check:

```bash
./build/release/duckdb -unsigned
```

Then run:

```sql
SELECT hll_estimate(hll_create_agg(v::VARCHAR))
FROM (SELECT * FROM range(1000)) t(v);
```

## Benchmark And Evaluation

Benchmark instructions are located in:

- `benchmark/README.md`

Typical quick verification commands:

```bash
python3 benchmark/tpch_hll_eval.py --scale-factors 1 --iterations 1 --threads 1
python3 benchmark/merge_workflow_eval.py --scale-factor 1 --iterations 1 --threads 1
```

## Small Demo For Presentation

To generate a small local demo that shows why stored sketches matter:

```bash
python3 benchmark/sketch_workflow_demo.py
```

This creates a raw table, a stored sketch table, and output artifacts under:

- `benchmark/out/sketch_workflow_demo/`

The demo shows that one can:

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
