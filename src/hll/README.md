# HLL — Standalone HyperLogLog Library

A C++17 HyperLogLog implementation combining **Sparse-Dense adaptive representation** (HLL++, Heule et al. 2013) with **LogLog-Beta bias correction** (Qin et al. 2016).

- **Precision**: p=14 (16,384 registers)
- **Memory**: adaptive — a few hundred bytes (sparse) to ~16 KB (dense)
- **Expected error**: < 2% relative error for typical cardinalities
- **Hash**: MurmurHash3_x64_128 (vendored, header-only)
- **Tests**: 50 unit tests (Catch2)

## Build

```bash
cd hll
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build && ctest --output-on-failure   # run 50 unit tests
```

Requires: CMake 3.14+, C++17 compiler (Clang/GCC).

---

## API Quick Reference

```cpp
#include "hll/hll.h"
using hll::HyperLogLog;

HyperLogLog sketch;   // starts in sparse mode (near-zero memory)

// --- Add elements ---
sketch.AddHash(hash_value);        // pre-computed 64-bit hash (preferred for DuckDB)
sketch.AddString("hello");         // hashes internally with MurmurHash3
sketch.AddInt64(42);
sketch.AddUInt64(100);
sketch.Add(raw_ptr, len);          // raw bytes

// --- Merge ---
sketch.Merge(other_sketch);        // handles sparse+sparse, sparse+dense, dense+dense

// --- Estimate ---
double cardinality = sketch.Estimate();

// --- Serialize / Deserialize ---
std::vector<uint8_t> blob = sketch.Serialize();     // variable size
auto restored = HyperLogLog::Deserialize(blob);

// --- Mode inspection ---
sketch.IsSparse();          // true if in sparse mode
sketch.IsDense();           // true if promoted to dense mode
sketch.SparseSize();        // number of non-zero register entries (sparse mode)
sketch.SerializedSize();    // exact byte count Serialize() will produce
sketch.MemoryUsage();       // approximate heap bytes used
sketch.Clear();             // reset to empty sparse state
```

### Sparse-Dense Behavior

| Cardinality | Mode   | Serialized Size | Memory   |
|-------------|--------|-----------------|----------|
| 10          | Sparse | 47 bytes        | ~120 B   |
| 100         | Sparse | 403 bytes       | ~568 B   |
| 1,000       | Sparse | 3,899 bytes     | ~4 KB    |
| 5,000+      | Dense  | 16,387 bytes    | ~16 KB   |

- Sketches start in **sparse mode** (only non-zero registers stored as sorted pairs)
- Auto-promotes to **dense mode** when sparse entries reach 4,096 (~5K–8K distinct elements)
- Promotion is **irreversible** (except via `Clear()`)
- **You don't need to manage modes** — it's fully automatic and transparent

---

## For Role B (Tian Xie) — DuckDB Extension Integration

### Step 1: Add HLL to the extension build

Copy or symlink the `hll/` directory into your extension tree, then in your `CMakeLists.txt`:

```cmake
add_subdirectory(hll)
target_link_libraries(${EXTENSION_NAME} hll)
```

Your code can then `#include "hll/hll.h"`.

### Step 2: Map HLL lifecycle to DuckDB aggregate callbacks

```
Initialize  →  auto *hll = new HyperLogLog();
Update      →  hll->AddHash(hash);        // one call per input row
Combine     →  hll_a->Merge(*hll_b);      // merge partial states
Finalize    →  hll->Estimate()            // for hll_estimate result
             or hll->Serialize()           // for BLOB result
```

### Step 3: Register SQL functions

| SQL Function | Type | Signature | Implementation |
|---|---|---|---|
| `hll_create_agg(column)` | Aggregate | `ANY → BLOB` | Update: `AddHash`, Combine: `Merge`, Finalize: `Serialize` |
| `hll_estimate(sketch)` | Scalar | `BLOB → BIGINT` | `Deserialize` → `Estimate` |
| `hll_merge(a, b)` | Scalar | `(BLOB, BLOB) → BLOB` | `Deserialize` both → `Merge` → `Serialize` |
| `hll_merge_agg(sketch)` | Aggregate | `BLOB → BLOB` | Update: `Deserialize` + `Merge`, Finalize: `Serialize` |
| `hll_serialize(sketch)` | Scalar | `BLOB → BLOB` | Pass-through (already serialized from `hll_create_agg`) |
| `hll_deserialize(blob)` | Scalar | `BLOB → BLOB` | `Deserialize` → validate → `Serialize` (round-trip check) |

### Integration notes

1. **`AddHash(uint64_t)`** is the fastest path. If DuckDB already hashes column values, pass that hash directly instead of re-hashing raw data.

2. **Serialized size is variable**, not fixed:
   - Sparse sketches: `7 + 4×N` bytes (N = non-zero registers), typically a few hundred bytes for low cardinalities
   - Dense sketches: 16,387 bytes
   - Use `SerializedSize()` if you need to know ahead of time

3. **Aggregate state**: use a pointer (`std::unique_ptr<HyperLogLog>`) in your aggregate state struct, not an embedded object. The object uses heap-allocated vectors internally.

   ```cpp
   struct HLLState {
       std::unique_ptr<hll::HyperLogLog> hll;
   };

   // In Initialize:
   state->hll = std::make_unique<hll::HyperLogLog>();
   ```

4. **Error handling**: `Deserialize` throws `std::invalid_argument` on bad input (wrong version, wrong precision, truncated buffer). Catch this and return a SQL error to the user.

5. **Wire format**: `Deserialize` accepts both **v1** (legacy dense-only, 16,386 bytes) and **v2** (sparse or dense with mode byte). Your extension only needs to call `Serialize()` / `Deserialize()` — the format is handled automatically.

6. **Copy/move semantics**: `HyperLogLog` supports copy and move. Copies are deep (independent sketches). This is safe for DuckDB's parallel combine step.

7. **Thread safety**: individual `HyperLogLog` objects are **not** thread-safe. DuckDB's aggregate framework handles this — each thread gets its own state, merged via `Combine`.

### Wire format reference

```
v2 sparse: [0x02] [0x0E] [0x00] [4-byte LE count] [N × 4-byte encoded pairs]
v2 dense:  [0x02] [0x0E] [0x01] [16,384 bytes of registers]
v1 dense:  [0x01] [0x0E] [16,384 bytes of registers]  (legacy, still accepted)
```

---

## For Role C (Congqi Xu) — Evaluation

### What you can do now (before Role B finishes)

1. **Generate TPC-H data** at SF 1, 10, 100 using DuckDB:
   ```sql
   INSTALL tpch; LOAD tpch;
   CALL dbgen(sf=1);
   ```

2. **Run baseline queries** — exact `COUNT(DISTINCT ...)` on high-cardinality columns:
   ```sql
   SELECT COUNT(DISTINCT l_orderkey) FROM lineitem;
   SELECT COUNT(DISTINCT o_custkey) FROM orders;
   SELECT COUNT(DISTINCT ps_suppkey) FROM partsupp;
   ```
   Record wall-clock times (use `.timer on` in DuckDB CLI).

### What you need from Role B

Once the extension is loadable, replace baseline queries with HLL equivalents:
```sql
LOAD 'hll_extension';

-- Build sketch and estimate in one query
SELECT hll_estimate(hll_create_agg(l_orderkey)) FROM lineitem;

-- Pre-aggregate by partition, then merge
SELECT hll_estimate(hll_merge_agg(sketch))
FROM (
    SELECT hll_create_agg(l_orderkey) AS sketch
    FROM lineitem
    GROUP BY l_shipdate
);
```

### Metrics to compute

- **Relative error**: `|hll_estimate - exact_count| / exact_count`
- **Speedup**: `exact_time / hll_time`
- Target: relative error < 2%, with significant speedup at SF 10+

### Memory savings data (from unit tests)

For the evaluation report — sparse mode provides ~40x memory reduction for low cardinalities:

| Cardinality | Mode   | Serialized Size | vs Dense (16,387 B) |
|-------------|--------|-----------------|---------------------|
| 10          | Sparse | 47 bytes        | 349x smaller        |
| 100         | Sparse | 403 bytes       | 41x smaller         |
| 1,000       | Sparse | 3,899 bytes     | 4.2x smaller        |
| 5,000+      | Dense  | 16,387 bytes    | 1x (baseline)       |

---

## References

- [HyperLogLog (Flajolet et al., 2007)](http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf)
- [HLL++ (Heule et al., 2013)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/40671.pdf)
- [LogLog-Beta (Qin et al., 2016)](https://arxiv.org/abs/1612.02284)
- [DuckDB Extension Template](https://github.com/duckdb/extension-template)
