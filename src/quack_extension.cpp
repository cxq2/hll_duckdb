#define DUCKDB_EXTENSION_MAIN

#include "quack_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/vector_operations/binary_executor.hpp"
#include "duckdb/common/vector_operations/unary_executor.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/function/scalar_function.hpp"

#include "hll/hll.h"

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

#include <cstdint>
#include <exception>
#include <vector>

namespace duckdb {

namespace {

struct HLLState {
	hll::HyperLogLog *hll_obj;
};

static hll::HyperLogLog DeserializeSketch(const string_t &blob) {
	auto data_ptr = reinterpret_cast<const uint8_t *>(blob.GetData());
	auto len = blob.GetSize();
	try {
		return hll::HyperLogLog::Deserialize(data_ptr, len);
	} catch (const std::exception &ex) {
		throw InvalidInputException("Failed to deserialize HLL sketch: " + std::string(ex.what()));
	}
}

static string_t SerializeSketchToBlob(const hll::HyperLogLog &hll_obj, Vector &result) {
	auto serialized = hll_obj.Serialize();
	return StringVector::AddStringOrBlob(result, reinterpret_cast<const char *>(serialized.data()), serialized.size());
}

struct HLLCreateAggOperation {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.hll_obj = new hll::HyperLogLog();
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &) {
		state.hll_obj->Add(input.GetData(), input.GetSize());
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &input_data, idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			OP::template Operation<INPUT_TYPE, STATE, OP>(state, input, input_data);
		}
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &) {
		target.hll_obj->Merge(*source.hll_obj);
	}

	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		target = SerializeSketchToBlob(*state.hll_obj, finalize_data.result);
	}

	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
		delete state.hll_obj;
		state.hll_obj = nullptr;
	}

	static bool IgnoreNull() {
		return true;
	}
};

struct HLLMergeAggOperation {
	template <class STATE>
	static void Initialize(STATE &state) {
		state.hll_obj = new hll::HyperLogLog();
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &) {
		auto input_hll = DeserializeSketch(input);
		state.hll_obj->Merge(input_hll);
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &input_data, idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			OP::template Operation<INPUT_TYPE, STATE, OP>(state, input, input_data);
		}
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &) {
		target.hll_obj->Merge(*source.hll_obj);
	}

	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		target = SerializeSketchToBlob(*state.hll_obj, finalize_data.result);
	}

	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &) {
		delete state.hll_obj;
		state.hll_obj = nullptr;
	}

	static bool IgnoreNull() {
		return true;
	}
};

inline void QuackScalarFun(DataChunk &args, ExpressionState &, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "Quack " + name.GetString() + " 🐥");
	});
}

inline void QuackOpenSSLVersionScalarFun(DataChunk &args, ExpressionState &, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		return StringVector::AddString(result, "Quack " + name.GetString() + ", my linked OpenSSL version is " +
		                                           OPENSSL_VERSION_TEXT);
	});
}

inline void HLLEstimateScalarFun(DataChunk &args, ExpressionState &, Vector &result) {
	auto &blob_vector = args.data[0];
	UnaryExecutor::Execute<string_t, int64_t>(blob_vector, result, args.size(), [&](string_t blob) {
		auto hll_obj = DeserializeSketch(blob);
		return static_cast<int64_t>(hll_obj.Estimate());
	});
}

inline void HLLMergeScalarFun(DataChunk &args, ExpressionState &, Vector &result) {
	auto &left_vector = args.data[0];
	auto &right_vector = args.data[1];
	BinaryExecutor::Execute<string_t, string_t, string_t>(left_vector, right_vector, result, args.size(),
	                                                      [&](string_t left_blob, string_t right_blob) {
		                                                      auto left_hll = DeserializeSketch(left_blob);
		                                                      auto right_hll = DeserializeSketch(right_blob);
		                                                      left_hll.Merge(right_hll);
		                                                      return SerializeSketchToBlob(left_hll, result);
	                                                      });
}

inline void HLLSerializeScalarFun(DataChunk &args, ExpressionState &, Vector &result) {
	auto &blob_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(blob_vector, result, args.size(), [&](string_t blob) {
		auto hll_obj = DeserializeSketch(blob);
		return SerializeSketchToBlob(hll_obj, result);
	});
}

inline void HLLDeserializeScalarFun(DataChunk &args, ExpressionState &, Vector &result) {
	auto &blob_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(blob_vector, result, args.size(), [&](string_t blob) {
		auto hll_obj = DeserializeSketch(blob);
		return SerializeSketchToBlob(hll_obj, result);
	});
}

static void LoadInternal(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction("quack", {LogicalType::VARCHAR}, LogicalType::VARCHAR, QuackScalarFun));

	loader.RegisterFunction(ScalarFunction("quack_openssl_version", {LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                                     QuackOpenSSLVersionScalarFun));

	auto hll_create_agg =
	    AggregateFunction::UnaryAggregateDestructor<HLLState, string_t, string_t, HLLCreateAggOperation>(
	        LogicalType::VARCHAR, LogicalType::BLOB);
	hll_create_agg.name = "hll_create_agg";
	loader.RegisterFunction(hll_create_agg);

	auto hll_merge_agg =
	    AggregateFunction::UnaryAggregateDestructor<HLLState, string_t, string_t, HLLMergeAggOperation>(
	        LogicalType::BLOB, LogicalType::BLOB);
	hll_merge_agg.name = "hll_merge_agg";
	loader.RegisterFunction(hll_merge_agg);

	loader.RegisterFunction(ScalarFunction("hll_estimate", {LogicalType::BLOB}, LogicalType::BIGINT,
	                                     HLLEstimateScalarFun));

	loader.RegisterFunction(ScalarFunction("hll_merge", {LogicalType::BLOB, LogicalType::BLOB}, LogicalType::BLOB,
	                                     HLLMergeScalarFun));

	loader.RegisterFunction(ScalarFunction("hll_serialize", {LogicalType::BLOB}, LogicalType::BLOB,
	                                     HLLSerializeScalarFun));

	loader.RegisterFunction(ScalarFunction("hll_deserialize", {LogicalType::BLOB}, LogicalType::BLOB,
	                                     HLLDeserializeScalarFun));
}

} // namespace

void QuackExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string QuackExtension::Name() {
	return "quack";
}

std::string QuackExtension::Version() const {
#ifdef EXT_VERSION_QUACK
	return EXT_VERSION_QUACK;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(quack, loader) {
	duckdb::LoadInternal(loader);
}
}
