#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "hll/hll.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using hll::HyperLogLog;

// Helper: relative error between estimate and true cardinality.
static double RelativeError(double estimate, double truth) {
    if (truth == 0.0)
        return (estimate == 0.0) ? 0.0 : 1.0;
    return std::abs(estimate - truth) / truth;
}

// ============================================================
// Accuracy tests
// ============================================================

TEST_CASE("Empty HLL estimates 0", "[accuracy]") {
    HyperLogLog hll;
    REQUIRE(hll.Estimate() == 0.0);
}

TEST_CASE("Single element", "[accuracy]") {
    HyperLogLog hll;
    hll.AddInt64(42);
    double est = hll.Estimate();
    REQUIRE(est > 0.0);
    REQUIRE(est < 5.0);
}

TEST_CASE("Duplicate insensitivity", "[accuracy]") {
    HyperLogLog hll;
    for (int i = 0; i < 1000; ++i) {
        hll.AddString("same_element");
    }
    double est = hll.Estimate();
    REQUIRE(est > 0.0);
    REQUIRE(est < 5.0);
}

TEST_CASE("Known cardinality — small (N=100)", "[accuracy]") {
    HyperLogLog hll;
    for (int64_t i = 0; i < 100; ++i) {
        hll.AddInt64(i);
    }
    REQUIRE(RelativeError(hll.Estimate(), 100.0) < 0.10);
}

TEST_CASE("Known cardinality — 1K", "[accuracy]") {
    HyperLogLog hll;
    for (int64_t i = 0; i < 1000; ++i) {
        hll.AddInt64(i);
    }
    REQUIRE(RelativeError(hll.Estimate(), 1000.0) < 0.05);
}

TEST_CASE("Known cardinality — 10K", "[accuracy]") {
    HyperLogLog hll;
    for (int64_t i = 0; i < 10000; ++i) {
        hll.AddInt64(i);
    }
    REQUIRE(RelativeError(hll.Estimate(), 10000.0) < 0.05);
}

TEST_CASE("Known cardinality — 100K", "[accuracy]") {
    HyperLogLog hll;
    for (int64_t i = 0; i < 100000; ++i) {
        hll.AddInt64(i);
    }
    REQUIRE(RelativeError(hll.Estimate(), 100000.0) < 0.05);
}

TEST_CASE("Known cardinality — 1M", "[accuracy]") {
    HyperLogLog hll;
    for (int64_t i = 0; i < 1000000; ++i) {
        hll.AddInt64(i);
    }
    REQUIRE(RelativeError(hll.Estimate(), 1000000.0) < 0.03);
}

TEST_CASE("Small cardinality — N=10", "[accuracy]") {
    HyperLogLog hll;
    for (int64_t i = 0; i < 10; ++i) {
        hll.AddInt64(i);
    }
    REQUIRE(RelativeError(hll.Estimate(), 10.0) < 0.20);
}

TEST_CASE("Small cardinality — N=50", "[accuracy]") {
    HyperLogLog hll;
    for (int64_t i = 0; i < 50; ++i) {
        hll.AddInt64(i);
    }
    REQUIRE(RelativeError(hll.Estimate(), 50.0) < 0.15);
}

// ============================================================
// Merge tests
// ============================================================

TEST_CASE("Merge equivalence — disjoint sets", "[merge]") {
    HyperLogLog a, b, combined;
    for (int64_t i = 0; i < 5000; ++i) {
        a.AddInt64(i);
        combined.AddInt64(i);
    }
    for (int64_t i = 5000; i < 10000; ++i) {
        b.AddInt64(i);
        combined.AddInt64(i);
    }
    a.Merge(b);
    // Merged estimate should be close to the single-pass estimate.
    REQUIRE(std::abs(a.Estimate() - combined.Estimate()) < 1.0);
}

TEST_CASE("Merge with overlap", "[merge]") {
    HyperLogLog a, b;
    for (int64_t i = 0; i < 7000; ++i) {
        a.AddInt64(i);
    }
    for (int64_t i = 3000; i < 10000; ++i) {
        b.AddInt64(i);
    }
    a.Merge(b);
    // True cardinality is 10000, not 14000.
    REQUIRE(RelativeError(a.Estimate(), 10000.0) < 0.05);
}

TEST_CASE("Merge commutativity", "[merge]") {
    HyperLogLog a1, b1, a2, b2;
    for (int64_t i = 0; i < 5000; ++i) {
        a1.AddInt64(i);
        a2.AddInt64(i);
    }
    for (int64_t i = 5000; i < 10000; ++i) {
        b1.AddInt64(i);
        b2.AddInt64(i);
    }
    a1.Merge(b1);
    b2.Merge(a2);
    REQUIRE(a1.Estimate() == b2.Estimate());
}

TEST_CASE("Merge with empty", "[merge]") {
    HyperLogLog a;
    for (int64_t i = 0; i < 1000; ++i) {
        a.AddInt64(i);
    }
    double before = a.Estimate();
    HyperLogLog empty;
    a.Merge(empty);
    REQUIRE(a.Estimate() == before);
}

TEST_CASE("Self merge", "[merge]") {
    HyperLogLog a;
    for (int64_t i = 0; i < 1000; ++i) {
        a.AddInt64(i);
    }
    double before = a.Estimate();
    a.Merge(a);
    REQUIRE(a.Estimate() == before);
}

// ============================================================
// Serialization tests
// ============================================================

TEST_CASE("Serialization size — sparse", "[serialization]") {
    HyperLogLog hll;
    auto blob = hll.Serialize();
    // Empty sparse: [0x02][0x0E][0x00][4B count=0] = 7 bytes.
    REQUIRE(blob.size() == 7);
    REQUIRE(blob.size() == hll.SerializedSize());
}

TEST_CASE("Serialization size — dense", "[serialization]") {
    HyperLogLog hll;
    // Force promotion by adding enough elements.
    for (int64_t i = 0; i < 100000; ++i) {
        hll.AddInt64(i);
    }
    REQUIRE(hll.IsDense());
    auto blob = hll.Serialize();
    // v2 dense: [0x02][precision][0x01][16384 registers] = 16387.
    REQUIRE(blob.size() == 3 + HyperLogLog::kNumRegisters);
    REQUIRE(blob.size() == hll.SerializedSize());
}

TEST_CASE("Round-trip identity", "[serialization]") {
    HyperLogLog hll;
    for (int64_t i = 0; i < 10000; ++i) {
        hll.AddInt64(i);
    }
    auto blob = hll.Serialize();
    auto restored = HyperLogLog::Deserialize(blob);

    // All registers must be identical.
    for (uint32_t i = 0; i < HyperLogLog::kNumRegisters; ++i) {
        REQUIRE(hll.GetRegister(i) == restored.GetRegister(i));
    }
}

TEST_CASE("Round-trip preserves estimate", "[serialization]") {
    HyperLogLog hll;
    for (int64_t i = 0; i < 50000; ++i) {
        hll.AddInt64(i);
    }
    double before = hll.Estimate();
    auto blob = hll.Serialize();
    auto restored = HyperLogLog::Deserialize(blob);
    REQUIRE(restored.Estimate() == before);
}

TEST_CASE("Deserialize rejects bad version", "[serialization]") {
    HyperLogLog hll;
    auto blob = hll.Serialize();
    blob[0] = 0xFF; // corrupt version
    REQUIRE_THROWS_AS(HyperLogLog::Deserialize(blob), std::invalid_argument);
}

TEST_CASE("Deserialize rejects bad precision", "[serialization]") {
    HyperLogLog hll;
    auto blob = hll.Serialize();
    blob[1] = 10; // wrong precision
    REQUIRE_THROWS_AS(HyperLogLog::Deserialize(blob), std::invalid_argument);
}

TEST_CASE("Deserialize rejects truncated buffer", "[serialization]") {
    std::vector<uint8_t> tiny = {0x01, 0x0E}; // v1 header only, no registers
    REQUIRE_THROWS_AS(HyperLogLog::Deserialize(tiny), std::invalid_argument);
}

// ============================================================
// Edge cases
// ============================================================

TEST_CASE("AddHash with hash=0", "[edge]") {
    HyperLogLog hll;
    hll.AddHash(0);
    // Should not crash. Register 0 should be set to max (51).
    REQUIRE(hll.GetRegister(0) == HyperLogLog::kMaxRegValue);
    REQUIRE(hll.Estimate() > 0.0);
}

TEST_CASE("AddHash with hash=UINT64_MAX", "[edge]") {
    HyperLogLog hll;
    hll.AddHash(UINT64_MAX);
    // Should not crash.
    REQUIRE(hll.Estimate() > 0.0);
}

TEST_CASE("Clear resets to empty", "[edge]") {
    HyperLogLog hll;
    for (int64_t i = 0; i < 1000; ++i) {
        hll.AddInt64(i);
    }
    REQUIRE(hll.Estimate() > 0.0);
    hll.Clear();
    REQUIRE(hll.Estimate() == 0.0);
    REQUIRE(hll.CountZeroRegisters() == HyperLogLog::kNumRegisters);
}

TEST_CASE("GetRegister out of range throws", "[edge]") {
    HyperLogLog hll;
    REQUIRE_THROWS_AS(hll.GetRegister(HyperLogLog::kNumRegisters),
                       std::out_of_range);
}

// ============================================================
// LogLog-Beta specific
// ============================================================

TEST_CASE("String elements accuracy", "[accuracy]") {
    HyperLogLog hll;
    for (int i = 0; i < 10000; ++i) {
        hll.AddString("item_" + std::to_string(i));
    }
    REQUIRE(RelativeError(hll.Estimate(), 10000.0) < 0.05);
}

TEST_CASE("AddUInt64 accuracy", "[accuracy]") {
    HyperLogLog hll;
    for (uint64_t i = 0; i < 10000; ++i) {
        hll.AddUInt64(i);
    }
    REQUIRE(RelativeError(hll.Estimate(), 10000.0) < 0.05);
}

// ============================================================
// Sparse mode tests (Phase A2)
// ============================================================

TEST_CASE("New HLL starts in sparse mode", "[sparse]") {
    HyperLogLog hll;
    REQUIRE(hll.IsSparse());
    REQUIRE(hll.SparseSize() == 0);
    REQUIRE(hll.MemoryUsage() > 0); // at least sizeof(HyperLogLog)
}

TEST_CASE("Sparse Add — single element stays sparse", "[sparse]") {
    HyperLogLog hll;
    hll.AddInt64(42);
    REQUIRE(hll.IsSparse());
    REQUIRE(hll.SparseSize() == 1);
}

TEST_CASE("Sparse Add — duplicates do not grow sparse list", "[sparse]") {
    HyperLogLog hll;
    for (int i = 0; i < 100; ++i) {
        hll.AddString("same_element");
    }
    REQUIRE(hll.IsSparse());
    REQUIRE(hll.SparseSize() == 1);
}

TEST_CASE("Sparse accuracy — N=10", "[sparse]") {
    HyperLogLog hll;
    for (int64_t i = 0; i < 10; ++i) {
        hll.AddInt64(i);
    }
    REQUIRE(hll.IsSparse());
    REQUIRE(RelativeError(hll.Estimate(), 10.0) < 0.20);
}

TEST_CASE("Sparse accuracy — N=100", "[sparse]") {
    HyperLogLog hll;
    for (int64_t i = 0; i < 100; ++i) {
        hll.AddInt64(i);
    }
    REQUIRE(hll.IsSparse());
    REQUIRE(RelativeError(hll.Estimate(), 100.0) < 0.10);
}

TEST_CASE("Sparse accuracy — N=1000", "[sparse]") {
    HyperLogLog hll;
    for (int64_t i = 0; i < 1000; ++i) {
        hll.AddInt64(i);
    }
    REQUIRE(hll.IsSparse());
    REQUIRE(RelativeError(hll.Estimate(), 1000.0) < 0.05);
}

TEST_CASE("Sparse GetRegister returns 0 for unstored index", "[sparse]") {
    HyperLogLog hll;
    hll.AddInt64(42);
    REQUIRE(hll.IsSparse());
    // Most registers should be 0.
    uint32_t zeros = 0;
    for (uint32_t i = 0; i < HyperLogLog::kNumRegisters; ++i) {
        if (hll.GetRegister(i) == 0) ++zeros;
    }
    REQUIRE(zeros == HyperLogLog::kNumRegisters - 1);
}

// ============================================================
// Promotion tests (Phase A3)
// ============================================================

TEST_CASE("Promotion occurs at threshold", "[promotion]") {
    HyperLogLog hll;
    // Add elements until we approach the threshold.
    // Each unique register index adds one sparse entry.
    // With 16384 possible indices, adding sequential int64s should
    // fill sparse entries relatively quickly.
    int64_t i = 0;
    while (hll.IsSparse() && i < 100000) {
        hll.AddInt64(i++);
    }
    REQUIRE(hll.IsDense());
    // Promotion should have happened around kSparseThreshold entries.
    // We added at most ~8K-12K elements to get 4096 unique register indices.
    REQUIRE(i < 20000);
}

TEST_CASE("Promotion is irreversible — Clear resets to sparse", "[promotion]") {
    HyperLogLog hll;
    for (int64_t i = 0; i < 100000; ++i) {
        hll.AddInt64(i);
    }
    REQUIRE(hll.IsDense());
    hll.Clear();
    REQUIRE(hll.IsSparse());
    REQUIRE(hll.Estimate() == 0.0);
}

TEST_CASE("Post-promotion estimate matches single-pass dense", "[promotion]") {
    // Build two HLLs with the same data: one that promotes naturally,
    // and one we verify is dense.
    HyperLogLog hll;
    for (int64_t i = 0; i < 100000; ++i) {
        hll.AddInt64(i);
    }
    REQUIRE(hll.IsDense());
    REQUIRE(RelativeError(hll.Estimate(), 100000.0) < 0.05);
}

// ============================================================
// Three merge strategy tests (Phase A3)
// ============================================================

TEST_CASE("Sparse + Sparse merge — stays sparse", "[merge][sparse]") {
    HyperLogLog a, b;
    for (int64_t i = 0; i < 50; ++i) {
        a.AddInt64(i);
    }
    for (int64_t i = 50; i < 100; ++i) {
        b.AddInt64(i);
    }
    REQUIRE(a.IsSparse());
    REQUIRE(b.IsSparse());
    a.Merge(b);
    REQUIRE(a.IsSparse());
    REQUIRE(RelativeError(a.Estimate(), 100.0) < 0.10);
}

TEST_CASE("Sparse + Sparse merge — promotes if exceeds threshold", "[merge][promotion]") {
    HyperLogLog a, b;
    // Fill each sketch close to half the threshold.
    for (int64_t i = 0; i < 5000; ++i) {
        a.AddInt64(i);
    }
    for (int64_t i = 5000; i < 10000; ++i) {
        b.AddInt64(i);
    }
    // Both may still be sparse or one may have promoted.
    // After merge, combined should exceed threshold and be dense.
    a.Merge(b);
    // With 10K distinct elements, we expect > 4096 unique register indices.
    REQUIRE(a.IsDense());
    REQUIRE(RelativeError(a.Estimate(), 10000.0) < 0.05);
}

TEST_CASE("Dense + Sparse merge — result is dense", "[merge][sparse]") {
    HyperLogLog dense_hll, sparse_hll;
    for (int64_t i = 0; i < 100000; ++i) {
        dense_hll.AddInt64(i);
    }
    for (int64_t i = 100000; i < 100050; ++i) {
        sparse_hll.AddInt64(i);
    }
    REQUIRE(dense_hll.IsDense());
    REQUIRE(sparse_hll.IsSparse());
    dense_hll.Merge(sparse_hll);
    REQUIRE(dense_hll.IsDense());
}

TEST_CASE("Sparse + Dense merge — promotes left, result is dense", "[merge][sparse]") {
    HyperLogLog sparse_hll, dense_hll;
    for (int64_t i = 0; i < 50; ++i) {
        sparse_hll.AddInt64(i);
    }
    for (int64_t i = 50; i < 100050; ++i) {
        dense_hll.AddInt64(i);
    }
    REQUIRE(sparse_hll.IsSparse());
    REQUIRE(dense_hll.IsDense());
    sparse_hll.Merge(dense_hll);
    REQUIRE(sparse_hll.IsDense());
}

TEST_CASE("Merge commutativity with sparse", "[merge][sparse]") {
    HyperLogLog a1, b1, a2, b2;
    for (int64_t i = 0; i < 200; ++i) {
        a1.AddInt64(i);
        a2.AddInt64(i);
    }
    for (int64_t i = 200; i < 400; ++i) {
        b1.AddInt64(i);
        b2.AddInt64(i);
    }
    REQUIRE(a1.IsSparse());
    REQUIRE(b1.IsSparse());
    a1.Merge(b1);
    b2.Merge(a2);
    REQUIRE(a1.Estimate() == b2.Estimate());
}

TEST_CASE("Self-merge in sparse mode", "[merge][sparse]") {
    HyperLogLog a;
    for (int64_t i = 0; i < 100; ++i) {
        a.AddInt64(i);
    }
    REQUIRE(a.IsSparse());
    double before = a.Estimate();
    a.Merge(a);
    REQUIRE(a.Estimate() == before);
    REQUIRE(a.IsSparse());
}

// ============================================================
// Dual-mode serialization tests (Phase A4)
// ============================================================

TEST_CASE("Sparse serialization round-trip", "[serialization][sparse]") {
    HyperLogLog hll;
    for (int64_t i = 0; i < 500; ++i) {
        hll.AddInt64(i);
    }
    REQUIRE(hll.IsSparse());
    double est_before = hll.Estimate();

    auto blob = hll.Serialize();
    auto restored = HyperLogLog::Deserialize(blob);

    REQUIRE(restored.IsSparse());
    REQUIRE(restored.SparseSize() == hll.SparseSize());
    REQUIRE(restored.Estimate() == est_before);
}

TEST_CASE("Sparse serialized size is much smaller than dense", "[serialization][sparse]") {
    HyperLogLog hll;
    for (int64_t i = 0; i < 100; ++i) {
        hll.AddInt64(i);
    }
    REQUIRE(hll.IsSparse());
    auto blob = hll.Serialize();
    // 100 elements -> ~100 sparse entries -> ~407 bytes.
    // Dense would be 16387 bytes.
    REQUIRE(blob.size() < 1000);
    REQUIRE(blob.size() < HyperLogLog::kNumRegisters);
}

TEST_CASE("Deserialize v1 format still works", "[serialization]") {
    // Manually construct a v1 dense blob.
    std::vector<uint8_t> v1_blob(2 + HyperLogLog::kNumRegisters, 0);
    v1_blob[0] = 0x01; // version
    v1_blob[1] = 0x0E; // precision = 14
    // Set register 0 to value 5.
    v1_blob[2] = 5;

    auto hll = HyperLogLog::Deserialize(v1_blob);
    REQUIRE(hll.IsDense());
    REQUIRE(hll.GetRegister(0) == 5);
    REQUIRE(hll.Estimate() > 0.0);
}

TEST_CASE("Deserialize rejects bad mode byte", "[serialization]") {
    // Construct a v2 header with invalid mode.
    std::vector<uint8_t> bad = {0x02, 0x0E, 0xFF};
    REQUIRE_THROWS_AS(HyperLogLog::Deserialize(bad), std::invalid_argument);
}

TEST_CASE("Deserialize rejects truncated sparse payload", "[serialization]") {
    // v2 sparse header says 100 entries but buffer is too short.
    std::vector<uint8_t> buf = {0x02, 0x0E, 0x00};
    uint32_t count = 100;
    buf.resize(7);
    std::memcpy(buf.data() + 3, &count, 4);
    // No payload bytes.
    REQUIRE_THROWS_AS(HyperLogLog::Deserialize(buf), std::invalid_argument);
}

// ============================================================
// Memory comparison (Phase A5 deliverable)
// ============================================================

TEST_CASE("Memory comparison — sparse vs dense sizes", "[memory]") {
    std::cout << "\n=== Memory Comparison: Sparse vs Dense ===\n";
    std::cout << "Cardinality | Mode   | Serialized Size | MemoryUsage\n";
    std::cout << "------------|--------|-----------------|------------\n";

    int cardinalities[] = {10, 100, 500, 1000, 2000, 5000, 10000, 100000};
    for (int n : cardinalities) {
        HyperLogLog hll;
        for (int64_t i = 0; i < n; ++i) {
            hll.AddInt64(i);
        }
        const char *mode = hll.IsSparse() ? "Sparse" : "Dense";
        auto blob = hll.Serialize();
        std::cout << std::setw(11) << n << " | "
                  << std::setw(6) << mode << " | "
                  << std::setw(15) << blob.size() << " | "
                  << std::setw(10) << hll.MemoryUsage() << "\n";
    }
    std::cout << "Dense baseline: " << (3 + HyperLogLog::kNumRegisters) << " bytes\n";
    std::cout << "==========================================\n";

    // Verify sparse is smaller for low cardinality.
    HyperLogLog small_hll;
    for (int64_t i = 0; i < 100; ++i) {
        small_hll.AddInt64(i);
    }
    REQUIRE(small_hll.IsSparse());
    REQUIRE(small_hll.MemoryUsage() < 2000);

    // Verify dense is ~16KB.
    HyperLogLog big_hll;
    for (int64_t i = 0; i < 100000; ++i) {
        big_hll.AddInt64(i);
    }
    REQUIRE(big_hll.IsDense());
    REQUIRE(big_hll.MemoryUsage() > 16000);
}
