#include "hll/hll.h"
#include "hll/murmur3.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace hll {

// ---------- Construction ----------

HyperLogLog::HyperLogLog() : mode_(Mode::kSparse) {}

// ---------- Internal helpers ----------

std::pair<uint32_t, uint8_t>
HyperLogLog::RegisterIndexAndRho(uint64_t hash) {
    // Bottom p bits -> register index.
    uint32_t index = static_cast<uint32_t>(hash & ((1u << kPrecision) - 1));
    // Remaining upper bits -> leading-zero count + 1.
    uint64_t w = hash >> kPrecision;
    uint8_t rho;
    if (w == 0) {
        rho = kMaxRegValue; // 64 - 14 + 1 = 51
    } else {
        // __builtin_clzll counts leading zeros of a 64-bit value.
        // w occupies at most (64 - p) = 50 bits, so subtract p to adjust.
        rho = static_cast<uint8_t>(__builtin_clzll(w) - kPrecision + 1);
    }
    return {index, rho};
}

double HyperLogLog::Alpha() {
    // For m >= 128: alpha_m = 0.7213 / (1 + 1.079 / m)
    constexpr double m = static_cast<double>(kNumRegisters);
    return 0.7213 / (1.0 + 1.079 / m);
}

double HyperLogLog::Beta(double z) {
    // LogLog-Beta polynomial for p=14 (Qin et al. 2016, appendix).
    if (z == 0.0)
        return 0.0;

    double zl = std::log(z + 1.0);
    constexpr double c[8] = {
        -3.7100976023069e-01,
         9.78811941207509e-03,
         1.85796293324165e-01,
         2.03015527328432e-01,
        -1.16710521803686e-01,
         4.31106699492820e-02,
        -5.99583405118316e-03,
         4.49704299509437e-04,
    };

    // Horner's method for c1*zl + c2*zl^2 + ... + c7*zl^7
    double poly = c[7];
    for (int i = 6; i >= 1; --i) {
        poly = poly * zl + c[i];
    }
    poly *= zl;

    return c[0] * z + poly;
}

// ---------- Sparse-to-Dense promotion ----------

void HyperLogLog::PromoteToDense() {
    dense_.resize(kNumRegisters, 0);
    for (uint32_t pair : sparse_) {
        uint32_t idx = PairIndex(pair);
        uint8_t rho = PairRho(pair);
        if (rho > dense_[idx]) {
            dense_[idx] = rho;
        }
    }
    sparse_.clear();
    sparse_.shrink_to_fit();
    mode_ = Mode::kDense;
}

// ---------- Core operations ----------

void HyperLogLog::AddHash(uint64_t hash) {
    auto [index, rho] = RegisterIndexAndRho(hash);

    if (mode_ == Mode::kDense) {
        if (rho > dense_[index]) {
            dense_[index] = rho;
        }
        return;
    }

    // Sparse mode: binary search for this index.
    // Since index is in the high bits, EncodePair(index, 0) is <= any
    // existing pair with the same index.
    uint32_t search_key = EncodePair(index, 0);
    auto it = std::lower_bound(sparse_.begin(), sparse_.end(), search_key);

    if (it != sparse_.end() && PairIndex(*it) == index) {
        // Entry exists — update if rho is larger.
        if (rho > PairRho(*it)) {
            *it = EncodePair(index, rho);
        }
    } else {
        // New index — insert to maintain sorted order.
        sparse_.insert(it, EncodePair(index, rho));
    }

    // Check promotion threshold.
    if (sparse_.size() >= kSparseThreshold) {
        PromoteToDense();
    }
}

void HyperLogLog::Add(const void *data, size_t len) {
    uint64_t out[2];
    murmur3::MurmurHash3_x64_128(data, static_cast<int>(len), 0, out);
    AddHash(out[0]);
}

void HyperLogLog::AddString(const std::string &value) {
    Add(value.data(), value.size());
}

void HyperLogLog::AddInt64(int64_t value) { Add(&value, sizeof(value)); }

void HyperLogLog::AddUInt64(uint64_t value) { Add(&value, sizeof(value)); }

// ---------- Merge ----------

void HyperLogLog::MergeSparseSparse(const HyperLogLog &other) {
    std::vector<uint32_t> merged;
    merged.reserve(sparse_.size() + other.sparse_.size());

    auto it_a = sparse_.begin();
    auto it_b = other.sparse_.begin();

    while (it_a != sparse_.end() && it_b != other.sparse_.end()) {
        uint32_t idx_a = PairIndex(*it_a);
        uint32_t idx_b = PairIndex(*it_b);
        if (idx_a < idx_b) {
            merged.push_back(*it_a++);
        } else if (idx_a > idx_b) {
            merged.push_back(*it_b++);
        } else {
            // Same index: take max rho.
            uint8_t rho = std::max(PairRho(*it_a), PairRho(*it_b));
            merged.push_back(EncodePair(idx_a, rho));
            ++it_a;
            ++it_b;
        }
    }
    while (it_a != sparse_.end())
        merged.push_back(*it_a++);
    while (it_b != other.sparse_.end())
        merged.push_back(*it_b++);

    sparse_ = std::move(merged);

    if (sparse_.size() >= kSparseThreshold) {
        PromoteToDense();
    }
}

void HyperLogLog::Merge(const HyperLogLog &other) {
    // Self-merge is always a no-op for register-wise max.
    if (this == &other)
        return;

    if (mode_ == Mode::kDense && other.mode_ == Mode::kDense) {
        // Dense + Dense -> Dense.
        for (uint32_t i = 0; i < kNumRegisters; ++i) {
            if (other.dense_[i] > dense_[i]) {
                dense_[i] = other.dense_[i];
            }
        }
    } else if (mode_ == Mode::kDense && other.mode_ == Mode::kSparse) {
        // Dense + Sparse -> Dense.
        for (uint32_t pair : other.sparse_) {
            uint32_t idx = PairIndex(pair);
            uint8_t rho = PairRho(pair);
            if (rho > dense_[idx]) {
                dense_[idx] = rho;
            }
        }
    } else if (mode_ == Mode::kSparse && other.mode_ == Mode::kDense) {
        // Sparse + Dense -> Dense (promote self first).
        PromoteToDense();
        for (uint32_t i = 0; i < kNumRegisters; ++i) {
            if (other.dense_[i] > dense_[i]) {
                dense_[i] = other.dense_[i];
            }
        }
    } else {
        // Sparse + Sparse -> Sparse (may promote if result exceeds threshold).
        MergeSparseSparse(other);
    }
}

// ---------- Estimate ----------

double HyperLogLog::Estimate() const {
    double sum = 0.0;
    uint32_t zeros = 0;

    if (mode_ == Mode::kDense) {
        for (uint32_t i = 0; i < kNumRegisters; ++i) {
            sum += 1.0 / static_cast<double>(1ULL << dense_[i]);
            if (dense_[i] == 0)
                ++zeros;
        }
    } else {
        // Sparse: unstored registers are implicitly zero.
        uint32_t k = static_cast<uint32_t>(sparse_.size());
        zeros = kNumRegisters - k;

        // All zero registers contribute 2^(-0) = 1.0 each.
        sum = static_cast<double>(zeros);

        // Add contributions from non-zero registers.
        for (uint32_t pair : sparse_) {
            uint8_t rho = PairRho(pair);
            sum += 1.0 / static_cast<double>(1ULL << rho);
        }
    }

    double m = static_cast<double>(kNumRegisters);
    double z = static_cast<double>(zeros);

    if (zeros == kNumRegisters) {
        return 0.0;
    }

    return Alpha() * m * (m - z) / (sum + Beta(z));
}

// ---------- Serialization ----------

size_t HyperLogLog::SerializedSize() const {
    if (mode_ == Mode::kDense) {
        return 3 + kNumRegisters; // [version][precision][mode][registers]
    }
    // Sparse: [version][precision][mode][4B count][N×4B pairs]
    return 7 + sparse_.size() * 4;
}

std::vector<uint8_t> HyperLogLog::Serialize() const {
    if (mode_ == Mode::kDense) {
        // v2 dense: [0x02][precision][0x01][16384 registers]
        std::vector<uint8_t> buf(3 + kNumRegisters);
        buf[0] = 0x02;
        buf[1] = kPrecision;
        buf[2] = static_cast<uint8_t>(Mode::kDense);
        std::memcpy(buf.data() + 3, dense_.data(), kNumRegisters);
        return buf;
    }

    // v2 sparse: [0x02][precision][0x00][4B count LE][N×4B pairs]
    uint32_t count = static_cast<uint32_t>(sparse_.size());
    std::vector<uint8_t> buf(7 + count * 4);
    buf[0] = 0x02;
    buf[1] = kPrecision;
    buf[2] = static_cast<uint8_t>(Mode::kSparse);
    std::memcpy(buf.data() + 3, &count, 4);
    if (count > 0) {
        std::memcpy(buf.data() + 7, sparse_.data(), count * 4);
    }
    return buf;
}

HyperLogLog HyperLogLog::Deserialize(const uint8_t *data, size_t len) {
    if (len < 2) {
        throw std::invalid_argument("Buffer too small for HLL header");
    }

    uint8_t version = data[0];

    if (version == 0x01) {
        // Legacy v1 dense format: [0x01][precision][16384B registers]
        if (len < 2 + kNumRegisters) {
            throw std::invalid_argument(
                "Buffer too small for HLL deserialization: need " +
                std::to_string(2 + kNumRegisters) + " bytes, got " +
                std::to_string(len));
        }
        if (data[1] != kPrecision) {
            throw std::invalid_argument(
                "Precision mismatch: expected " + std::to_string(kPrecision) +
                ", got " + std::to_string(data[1]));
        }
        HyperLogLog hll;
        hll.PromoteToDense();
        std::memcpy(hll.dense_.data(), data + 2, kNumRegisters);
        return hll;
    }

    if (version == 0x02) {
        if (len < 3) {
            throw std::invalid_argument("Buffer too small for v2 HLL header");
        }
        if (data[1] != kPrecision) {
            throw std::invalid_argument(
                "Precision mismatch: expected " + std::to_string(kPrecision) +
                ", got " + std::to_string(data[1]));
        }

        uint8_t mode_byte = data[2];

        if (mode_byte == static_cast<uint8_t>(Mode::kDense)) {
            if (len < 3 + kNumRegisters) {
                throw std::invalid_argument(
                    "Buffer too small for v2 dense HLL");
            }
            HyperLogLog hll;
            hll.PromoteToDense();
            std::memcpy(hll.dense_.data(), data + 3, kNumRegisters);
            return hll;
        }

        if (mode_byte == static_cast<uint8_t>(Mode::kSparse)) {
            if (len < 7) {
                throw std::invalid_argument(
                    "Buffer too small for v2 sparse HLL header");
            }
            uint32_t count;
            std::memcpy(&count, data + 3, 4);
            if (len < 7 + static_cast<size_t>(count) * 4) {
                throw std::invalid_argument(
                    "Buffer too small for v2 sparse HLL payload");
            }
            HyperLogLog hll;
            hll.sparse_.resize(count);
            if (count > 0) {
                std::memcpy(hll.sparse_.data(), data + 7, count * 4);
            }
            return hll;
        }

        throw std::invalid_argument("Unknown HLL mode: " +
                                    std::to_string(mode_byte));
    }

    throw std::invalid_argument("Unknown HLL version: " +
                                std::to_string(version));
}

HyperLogLog HyperLogLog::Deserialize(const std::vector<uint8_t> &data) {
    return Deserialize(data.data(), data.size());
}

// ---------- Accessors ----------

uint8_t HyperLogLog::GetRegister(uint32_t index) const {
    if (index >= kNumRegisters) {
        throw std::out_of_range("Register index out of range");
    }
    if (mode_ == Mode::kDense) {
        return dense_[index];
    }
    // Sparse: binary search for this index.
    uint32_t search_key = EncodePair(index, 0);
    auto it = std::lower_bound(sparse_.begin(), sparse_.end(), search_key);
    if (it != sparse_.end() && PairIndex(*it) == index) {
        return PairRho(*it);
    }
    return 0; // unstored = zero register
}

uint32_t HyperLogLog::CountZeroRegisters() const {
    if (mode_ == Mode::kDense) {
        uint32_t count = 0;
        for (uint32_t i = 0; i < kNumRegisters; ++i) {
            if (dense_[i] == 0)
                ++count;
        }
        return count;
    }
    // Sparse: all unstored registers are zero.
    return kNumRegisters - static_cast<uint32_t>(sparse_.size());
}

size_t HyperLogLog::MemoryUsage() const {
    size_t base = sizeof(HyperLogLog);
    if (mode_ == Mode::kDense) {
        return base + dense_.capacity() * sizeof(uint8_t);
    }
    return base + sparse_.capacity() * sizeof(uint32_t);
}

void HyperLogLog::Clear() {
    mode_ = Mode::kSparse;
    sparse_.clear();
    sparse_.shrink_to_fit();
    dense_.clear();
    dense_.shrink_to_fit();
}

} // namespace hll
