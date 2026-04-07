#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hll {

class HyperLogLog {
public:
    // Representation mode.
    enum class Mode : uint8_t { kSparse = 0, kDense = 1 };

    // Constants
    static constexpr uint8_t kPrecision = 14;
    static constexpr uint32_t kNumRegisters = 1u << kPrecision; // 16384
    static constexpr uint8_t kMaxRegValue = 64 - kPrecision + 1; // 51

    // Promotion threshold: promote when sparse list reaches this size.
    // 4096 entries × 4 bytes = 16384 bytes = dense array size.
    static constexpr size_t kSparseThreshold = 4096;

    // Legacy v1 dense serialized size (for backward compatibility).
    static constexpr size_t kSerializedSize = 2 + kNumRegisters; // 16386

    // Construction — starts in sparse mode.
    HyperLogLog();
    ~HyperLogLog() = default;

    HyperLogLog(const HyperLogLog &) = default;
    HyperLogLog &operator=(const HyperLogLog &) = default;
    HyperLogLog(HyperLogLog &&) noexcept = default;
    HyperLogLog &operator=(HyperLogLog &&) noexcept = default;

    // --- Core operations ---

    // Add a raw data element (hashes internally with MurmurHash3).
    void Add(const void *data, size_t len);

    // Convenience overloads.
    void AddString(const std::string &value);
    void AddInt64(int64_t value);
    void AddUInt64(uint64_t value);

    // Add a pre-hashed 64-bit value (integration point for Role B / DuckDB).
    void AddHash(uint64_t hash);

    // Merge another HLL sketch into this one.
    // Three strategies: sparse+sparse, sparse+dense, dense+dense.
    void Merge(const HyperLogLog &other);

    // Estimate cardinality using LogLog-Beta.
    double Estimate() const;

    // --- Serialization ---

    // Serialize to binary (dual-mode: sparse or dense).
    // v2 dense:  [0x02][precision][0x01][16384 registers]
    // v2 sparse: [0x02][precision][0x00][4B count][N×4B pairs]
    std::vector<uint8_t> Serialize() const;

    // Actual serialized size for this sketch.
    size_t SerializedSize() const;

    // Deserialize from binary blob (accepts both v1 and v2 formats).
    static HyperLogLog Deserialize(const uint8_t *data, size_t len);
    static HyperLogLog Deserialize(const std::vector<uint8_t> &data);

    // --- Accessors ---
    uint8_t GetRegister(uint32_t index) const;
    uint32_t CountZeroRegisters() const;
    uint8_t GetPrecision() const { return kPrecision; }
    uint32_t GetNumRegisters() const { return kNumRegisters; }

    // Sparse-dense mode accessors.
    Mode GetMode() const { return mode_; }
    bool IsSparse() const { return mode_ == Mode::kSparse; }
    bool IsDense() const { return mode_ == Mode::kDense; }
    size_t SparseSize() const { return sparse_.size(); }
    size_t MemoryUsage() const;

    // Reset all registers to 0 (returns to sparse mode).
    void Clear();

private:
    Mode mode_;
    std::vector<uint32_t> sparse_;  // sorted encoded (index,rho) pairs
    std::vector<uint8_t> dense_;    // register array (size kNumRegisters when active)

    // Sparse pair encoding: (index << 6) | rho.
    // Index in high bits => natural sort order = sorted by index.
    static constexpr uint32_t EncodePair(uint32_t index, uint8_t rho) {
        return (index << 6) | static_cast<uint32_t>(rho);
    }
    static constexpr uint32_t PairIndex(uint32_t pair) { return pair >> 6; }
    static constexpr uint8_t PairRho(uint32_t pair) {
        return static_cast<uint8_t>(pair & 0x3F);
    }

    // Irreversible sparse-to-dense promotion.
    void PromoteToDense();

    // Merge two sparse sketches (may trigger promotion).
    void MergeSparseSparse(const HyperLogLog &other);

    // Extract register index and rho (leading-zero count + 1) from a 64-bit hash.
    static std::pair<uint32_t, uint8_t> RegisterIndexAndRho(uint64_t hash);

    // LogLog-Beta bias correction polynomial (Qin et al. 2016, p=14).
    static double Beta(double z);

    // Alpha constant for m = 16384.
    static double Alpha();
};

} // namespace hll
