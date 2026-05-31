#include "runtime/core/allocator/vram_superallocator.h"
#include "runtime/core/scheduler/admission.h"
#include "runtime/kernels/decode/decode_kernels.h"
#include "runtime/loaders/iq_manifest.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kAlignment = 256;
constexpr std::uint64_t kIq2SBlockBytes = 82;
constexpr std::uint64_t kIq2SQsOffset = 2;
constexpr std::uint64_t kIq2SQhOffset = 66;
constexpr std::uint64_t kIq2SScalesOffset = 74;
constexpr std::uint64_t kIq3SBlockBytes = 110;
constexpr std::uint64_t kIq3SQsOffset = 2;
constexpr std::uint64_t kIq3SQhOffset = 66;
constexpr std::uint64_t kIq3SSignsOffset = 74;
constexpr std::uint64_t kIq3SScalesOffset = 106;
constexpr std::array<std::uint16_t, 1024> kIq2SKGrid = {{
#include "runtime/kernels/decode/iq2s_kgrid_values.inl"
}};
constexpr std::array<std::uint32_t, 512> kIq3SGrid = {{
#include "runtime/kernels/decode/iq3s_grid_values.inl"
}};

struct Options {
  std::filesystem::path manifest;
  std::string tensor;
  std::string label = "iq2s_matvec_probe";
  int steps = 8;
  std::uint32_t rows_limit = 64;
  std::size_t kv_mib = 1;
  std::uint32_t page_words = 256;
  std::uint32_t active_pages = 64;
  std::size_t wddm_guard_mib = 512;
};

struct TensorShape {
  std::uint32_t rows = 0;
  std::uint32_t cols = 0;
};

struct ReferenceResult {
  bool passed = false;
  float tolerance = 0.0025f;
  float max_abs_error = 0.0f;
  std::uint32_t max_error_row = 0;
  float first_expected = 0.0f;
  float first_observed = 0.0f;
  std::size_t mismatches = 0;
  float max_kv_abs_error = 0.0f;
  std::uint32_t max_kv_error_index = 0;
  std::size_t kv_mismatches = 0;
  std::uint32_t token_expected = 0;
  std::uint32_t token_observed = 0;
};

struct DecodeReference {
  std::vector<float> logits;
  std::vector<float> kv;
  std::uint32_t token = 0;
};

std::size_t mib(std::size_t value) {
  return value * 1024ull * 1024ull;
}

std::size_t align_up(std::size_t value) {
  const auto mask = kAlignment - 1;
  return (value + mask) & ~mask;
}

void check(cudaError_t status, const char* op) {
  if (status != cudaSuccess) {
    std::cerr << op << " failed: " << cudaGetErrorString(status) << "\n";
    std::exit(1);
  }
}

double percentile(std::vector<double> values, double pct) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(
      std::min<double>(values.size() - 1, std::ceil((pct / 100.0) * values.size()) - 1));
  return values[index];
}

std::size_t parse_size(const char* value, std::string_view name) {
  const unsigned long long parsed = std::stoull(value);
  if (parsed == 0) {
    throw std::invalid_argument(std::string(name) + " must be non-zero");
  }
  return static_cast<std::size_t>(parsed);
}

std::uint32_t parse_u32(const char* value, std::string_view name) {
  const unsigned long parsed = std::stoul(value);
  if (parsed == 0 || parsed > UINT32_MAX) {
    throw std::invalid_argument(std::string(name) + " must be in uint32 range and non-zero");
  }
  return static_cast<std::uint32_t>(parsed);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const auto require_value = [&](std::string_view name) -> const char* {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string(name) + " requires a value");
      }
      return argv[++i];
    };
    if (arg == "--iq-manifest") {
      options.manifest = require_value(arg);
    } else if (arg == "--tensor") {
      options.tensor = require_value(arg);
    } else if (arg == "--label") {
      options.label = require_value(arg);
    } else if (arg == "--steps") {
      options.steps = static_cast<int>(parse_size(require_value(arg), arg));
    } else if (arg == "--rows-limit") {
      options.rows_limit = parse_u32(require_value(arg), arg);
    } else if (arg == "--kv-mib") {
      options.kv_mib = parse_size(require_value(arg), arg);
    } else if (arg == "--page-words") {
      options.page_words = parse_u32(require_value(arg), arg);
    } else if (arg == "--active-pages") {
      options.active_pages = parse_u32(require_value(arg), arg);
    } else if (arg == "--wddm-guard-mib") {
      options.wddm_guard_mib = parse_size(require_value(arg), arg);
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "runtime_iq_matvec_probe --iq-manifest PATH --tensor NAME\n"
          << "                        [--rows-limit N] [--steps N] [--label LABEL]\n"
          << "                        [--kv-mib N] [--page-words N] [--active-pages N]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + std::string(arg));
    }
  }
  if (options.manifest.empty()) {
    throw std::invalid_argument("--iq-manifest is required");
  }
  return options;
}

std::string json_escape(std::string_view value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += ch;
        break;
    }
  }
  return out;
}

std::vector<std::uint8_t> read_binary_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open binary file: " + path.string());
  }
  in.seekg(0, std::ios::end);
  const auto size = static_cast<std::size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> data(size);
  if (size > 0) {
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!in) {
      throw std::runtime_error("failed to read binary file: " + path.string());
    }
  }
  return data;
}

std::uint64_t byte_checksum(const std::vector<std::uint8_t>& data) {
  return std::accumulate(
      data.begin(),
      data.end(),
      std::uint64_t{0},
      [](std::uint64_t acc, std::uint8_t value) {
        return acc + static_cast<std::uint64_t>(value);
      });
}

std::uint64_t block_bytes_for_type(const std::string& type) {
  if (type == "IQ2_S") {
    return kIq2SBlockBytes;
  }
  if (type == "IQ3_S") {
    return kIq3SBlockBytes;
  }
  throw std::runtime_error("unsupported IQ matvec tensor type: " + type);
}

float tolerance_for_type(const std::string& type) {
  return type == "IQ3_S" ? 0.01f : 0.0025f;
}

std::string matvec_path_for_type(const std::string& type) {
  return type == "IQ3_S" ? "native_iq3s_matvec_probe" : "native_iq2s_matvec_probe";
}

std::string runtime_kernel_for_type(const std::string& type) {
  return type == "IQ3_S" ? "iq3_s_dequant_matvec_decode_probe"
                         : "iq2_s_dequant_matvec_decode_probe";
}

float fp16_bits_to_float(std::uint16_t bits) {
  const float sign = (bits & 0x8000u) ? -1.0f : 1.0f;
  const int exp = static_cast<int>((bits >> 10u) & 0x1fu);
  const int mant = static_cast<int>(bits & 0x03ffu);
  if (exp == 0) {
    return mant == 0 ? sign * 0.0f : sign * std::ldexp(static_cast<float>(mant), -24);
  }
  if (exp == 31) {
    return mant == 0 ? sign * std::numeric_limits<float>::infinity()
                     : std::numeric_limits<float>::quiet_NaN();
  }
  return sign * std::ldexp(1.0f + static_cast<float>(mant) / 1024.0f, exp - 15);
}

TensorShape tensor_shape(const rtxllm::IQManifestTensor& tensor) {
  if (tensor.dimensions.empty()) {
    throw std::runtime_error("IQ tensor dimensions are missing: " + tensor.name);
  }
  const auto cols64 = tensor.dimensions.front();
  if (cols64 == 0 || cols64 > UINT32_MAX || (cols64 % 256u) != 0u) {
    throw std::runtime_error("IQ tensor first dimension must be a uint32 multiple of 256");
  }
  if ((tensor.element_count % cols64) != 0u) {
    throw std::runtime_error("IQ tensor element count does not divide by first dimension");
  }
  const auto rows64 = tensor.element_count / cols64;
  if (rows64 == 0 || rows64 > UINT32_MAX) {
    throw std::runtime_error("IQ tensor derived row count is out of range");
  }
  const auto expected_blocks = rows64 * (cols64 / 256u);
  if (expected_blocks != tensor.block_count) {
    throw std::runtime_error("IQ tensor block count does not match derived rows/cols");
  }
  return TensorShape{
      static_cast<std::uint32_t>(rows64),
      static_cast<std::uint32_t>(cols64),
  };
}

std::vector<float> make_activation(std::uint32_t cols) {
  std::vector<float> activation(cols);
  for (std::uint32_t col = 0; col < cols; ++col) {
    const float base = static_cast<float>((col * 131u + 17u) & 1023u) / 1024.0f;
    activation[col] = base - 0.5f;
  }
  return activation;
}

float iq2s_dequant_value(
    const std::uint8_t* block,
    std::uint32_t value_index) {
  const std::uint32_t ib32 = value_index >> 5u;
  const std::uint32_t within32 = value_index & 31u;
  const std::uint32_t group = within32 >> 3u;
  const std::uint32_t lane = within32 & 7u;
  const auto d = fp16_bits_to_float(static_cast<std::uint16_t>(
      block[0] | (static_cast<std::uint16_t>(block[1]) << 8u)));
  const auto* qs = block + kIq2SQsOffset;
  const auto* qh = block + kIq2SQhOffset;
  const auto* scales = block + kIq2SScalesOffset;
  const std::uint32_t group_offset = ib32 * 4u + group;
  const std::uint32_t grid_index =
      qs[group_offset] | ((static_cast<std::uint32_t>(qh[ib32]) << (8u - 2u * group)) & 0x300u);
  const std::uint32_t packed_grid = kIq2SKGrid[grid_index];
  const std::uint32_t grid = 2u * ((packed_grid >> (2u * lane)) & 0x3u) + 1u;
  const std::uint32_t scale_nibble =
      group < 2u ? (scales[ib32] & 0x0fu) : (scales[ib32] >> 4u);
  const float scale = d * (0.5f + static_cast<float>(scale_nibble)) * 0.25f;
  const bool negative = (qs[32u + group_offset] & (1u << lane)) != 0u;
  return scale * static_cast<float>(grid) * (negative ? -1.0f : 1.0f);
}

float iq3s_dequant_value(
    const std::uint8_t* block,
    std::uint32_t value_index) {
  const std::uint32_t ib32 = value_index >> 5u;
  const std::uint32_t within32 = value_index & 31u;
  const std::uint32_t group = within32 >> 3u;
  const std::uint32_t lane = within32 & 7u;
  const std::uint32_t pair = lane >> 2u;
  const std::uint32_t grid_lane = lane & 3u;
  const auto d = fp16_bits_to_float(static_cast<std::uint16_t>(
      block[0] | (static_cast<std::uint16_t>(block[1]) << 8u)));
  const auto* qs = block + kIq3SQsOffset;
  const auto* qh = block + kIq3SQhOffset;
  const auto* signs = block + kIq3SSignsOffset;
  const auto* scales = block + kIq3SScalesOffset;
  const std::uint32_t selector_index = ib32 * 8u + group * 2u + pair;
  const std::uint32_t high_bit = (qh[ib32] >> (2u * group + pair)) & 1u;
  const std::uint32_t grid_index = qs[selector_index] | (high_bit << 8u);
  const std::uint32_t packed_grid = kIq3SGrid[grid_index];
  const std::uint32_t grid = (packed_grid >> (8u * grid_lane)) & 0xffu;
  const std::uint32_t scale_nibble =
      (ib32 & 1u) == 0u ? (scales[ib32 >> 1u] & 0x0fu)
                        : (scales[ib32 >> 1u] >> 4u);
  const float scale = d * static_cast<float>(1u + 2u * scale_nibble);
  const bool negative = (signs[ib32 * 4u + group] & (1u << lane)) != 0u;
  return scale * static_cast<float>(grid) * (negative ? -1.0f : 1.0f);
}

float iq_dequant_value(
    const std::string& type,
    const std::uint8_t* block,
    std::uint32_t value_index) {
  if (type == "IQ3_S") {
    return iq3s_dequant_value(block, value_index);
  }
  return iq2s_dequant_value(block, value_index);
}

std::vector<float> reference_matvec(
    const std::vector<std::uint8_t>& payload,
    const std::vector<float>& activation,
    TensorShape shape,
    std::uint32_t rows_limit,
    const std::string& type) {
  const std::uint32_t checked_rows = std::min(shape.rows, rows_limit);
  const std::uint32_t blocks_per_row = shape.cols / 256u;
  const auto block_bytes = block_bytes_for_type(type);
  std::vector<float> logits(checked_rows, 0.0f);
  for (std::uint32_t row = 0; row < checked_rows; ++row) {
    float acc = 0.0f;
    for (std::uint32_t col = 0; col < shape.cols; ++col) {
      const auto block_index =
          static_cast<std::size_t>(row) * blocks_per_row + (col / 256u);
      const auto* block = payload.data() + block_index * block_bytes;
      acc += iq_dequant_value(type, block, col & 255u) * activation[col];
    }
    logits[row] = acc;
  }
  return logits;
}

std::uint32_t argmax_token(const std::vector<float>& logits) {
  float best_value = std::numeric_limits<float>::lowest();
  std::uint32_t best_index = 0;
  for (std::uint32_t row = 0; row < logits.size(); ++row) {
    const float value = logits[row];
    if (value > best_value || (value == best_value && row < best_index)) {
      best_value = value;
      best_index = row;
    }
  }
  return best_index;
}

DecodeReference reference_decode(
    const std::vector<std::uint8_t>& payload,
    const std::vector<float>& activation,
    TensorShape shape,
    std::uint32_t rows_limit,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    int steps,
    const std::string& type) {
  DecodeReference reference;
  reference.logits = reference_matvec(payload, activation, shape, rows_limit, type);
  reference.kv.assign(kv_words, 0.0f);
  const std::uint32_t best_index = argmax_token(reference.logits);
  const std::uint32_t active_rows =
      std::min<std::uint32_t>(static_cast<std::uint32_t>(reference.logits.size()), active_pages);
  for (int step = 0; step < steps; ++step) {
    if (kv_words > 0u && page_words > 0u) {
      for (std::uint32_t row = 0; row < active_rows; ++row) {
        const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
        const std::size_t offset = (reference.token + row) % page_words;
        const std::size_t pos = (page_base + offset) % kv_words;
        reference.kv[pos] = reference.logits[row];
      }
    }
    reference.token = (best_index + reference.token + 1u) % 32000u;
  }
  return reference;
}

ReferenceResult compare_decode(
    const DecodeReference& expected,
    const float* observed_logits,
    const float* observed_kv,
    std::uint32_t observed_token,
    float tolerance) {
  ReferenceResult result;
  result.tolerance = tolerance;
  if (!expected.logits.empty()) {
    result.first_expected = expected.logits.front();
    result.first_observed = observed_logits[0];
  }
  result.token_expected = expected.token;
  result.token_observed = observed_token;
  result.passed = true;
  for (std::size_t row = 0; row < expected.logits.size(); ++row) {
    const float error = std::fabs(expected.logits[row] - observed_logits[row]);
    if (error > result.max_abs_error) {
      result.max_abs_error = error;
      result.max_error_row = static_cast<std::uint32_t>(row);
    }
    if (error > result.tolerance) {
      result.passed = false;
      ++result.mismatches;
    }
  }
  for (std::size_t index = 0; index < expected.kv.size(); ++index) {
    const float error = std::fabs(expected.kv[index] - observed_kv[index]);
    if (error > result.max_kv_abs_error) {
      result.max_kv_abs_error = error;
      result.max_kv_error_index = static_cast<std::uint32_t>(index);
    }
    if (error > result.tolerance) {
      result.passed = false;
      ++result.kv_mismatches;
    }
  }
  if (result.token_expected != result.token_observed) {
    result.passed = false;
  }
  return result;
}

void print_rejection(
    const Options& options,
    const rtxllm::IQManifestTensor& tensor,
    const rtxllm::AdmissionDecision& decision) {
  std::cout << "{\n";
  std::cout << "  \"path\": \"" << matvec_path_for_type(tensor.type) << "\",\n";
  std::cout << "  \"label\": \"" << json_escape(options.label) << "\",\n";
  std::cout << "  \"admitted\": false,\n";
  std::cout << "  \"manifest\": \"" << json_escape(options.manifest.string()) << "\",\n";
  std::cout << "  \"tensor\": \"" << json_escape(tensor.name) << "\",\n";
  std::cout << "  \"type\": \"" << json_escape(tensor.type) << "\",\n";
  std::cout << "  \"reason\": \"" << json_escape(decision.reason) << "\",\n";
  std::cout << "  \"required_bytes\": " << decision.required_bytes << ",\n";
  std::cout << "  \"usable_bytes\": " << decision.usable_bytes << "\n";
  std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto manifest = rtxllm::load_iq_manifest(options.manifest);
    const auto tensor = rtxllm::select_iq_tensor(manifest, options.tensor);
    if (tensor.type != "IQ2_S" && tensor.type != "IQ3_S") {
      throw std::runtime_error("runtime_iq_matvec_probe supports IQ2_S and IQ3_S only");
    }
    const auto block_bytes = block_bytes_for_type(tensor.type);
    if (tensor.block_type_size != block_bytes) {
      throw std::runtime_error(tensor.type + " block type size mismatch");
    }
    const TensorShape shape = tensor_shape(tensor);
    const std::uint32_t checked_rows = std::min(shape.rows, options.rows_limit);

    const std::filesystem::path payload_path = tensor.payload_path.is_absolute()
        ? tensor.payload_path
        : options.manifest.parent_path() / tensor.payload_path;
    const auto payload = read_binary_file(payload_path);
    if (payload.size() != tensor.payload_bytes) {
      throw std::runtime_error("IQ payload file size does not match manifest: " + payload_path.string());
    }
    const auto host_weight_checksum = byte_checksum(payload);
    if (host_weight_checksum != tensor.payload_checksum ||
        host_weight_checksum != tensor.source_checksum) {
      throw std::runtime_error("IQ payload checksum does not match manifest: " + payload_path.string());
    }

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    const auto activation = make_activation(shape.cols);
    const std::size_t activation_bytes = activation.size() * sizeof(float);
    const std::size_t logits_bytes = static_cast<std::size_t>(checked_rows) * sizeof(float);
    const std::size_t kv_bytes = mib(options.kv_mib);
    if ((kv_bytes % sizeof(float)) != 0u) {
      throw std::runtime_error("KV arena byte count must be divisible by float size");
    }
    const auto kv_words_size = kv_bytes / sizeof(float);
    if (kv_words_size == 0 || kv_words_size > UINT32_MAX) {
      throw std::runtime_error("KV arena word count is out of uint32 range");
    }
    const auto kv_words = static_cast<std::uint32_t>(kv_words_size);
    constexpr std::size_t token_bytes = sizeof(std::uint32_t);
    const auto expected = reference_decode(
        payload,
        activation,
        shape,
        checked_rows,
        kv_words,
        options.page_words,
        options.active_pages,
        options.steps,
        tensor.type);
    const std::size_t resident_bytes = align_up(payload.size()) +
        align_up(activation_bytes) + align_up(logits_bytes) + align_up(kv_bytes) +
        align_up(token_bytes);
    const auto decision = rtxllm::decide_admission(
        rtxllm::ResidencyPolicy::Strict,
        rtxllm::MemoryBudget{free_bytes, mib(options.wddm_guard_mib)},
        rtxllm::RequestEstimate{
            resident_bytes,
            1,
            static_cast<std::size_t>(options.steps),
            0,
            mib(1)});
    if (!decision.admit) {
      print_rejection(options, tensor, decision);
      return 2;
    }

    rtxllm::VramSuperallocator pool;
    pool.initialize({
        {"weights", align_up(payload.size())},
        {"activation", align_up(activation_bytes)},
        {"logits", align_up(logits_bytes)},
        {"kv", align_up(kv_bytes)},
        {"token", align_up(token_bytes)},
        {"workspace", mib(1)},
    });

    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");

    auto* device_payload = static_cast<std::uint8_t*>(pool.arena_ptr("weights"));
    auto* device_activation = static_cast<float*>(pool.arena_ptr("activation"));
    auto* device_logits = static_cast<float*>(pool.arena_ptr("logits"));
    auto* device_kv = static_cast<float*>(pool.arena_ptr("kv"));
    auto* device_token = static_cast<std::uint32_t*>(pool.arena_ptr("token"));
    check(cudaMemcpyAsync(
              device_payload,
              payload.data(),
              payload.size(),
              cudaMemcpyHostToDevice,
              stream),
          "cudaMemcpyAsync(payload)");
    check(cudaMemcpyAsync(
              device_activation,
              activation.data(),
              activation_bytes,
              cudaMemcpyHostToDevice,
              stream),
          "cudaMemcpyAsync(activation)");
    check(cudaMemsetAsync(device_kv, 0, kv_bytes, stream), "cudaMemsetAsync(kv)");
    check(cudaMemsetAsync(device_token, 0, token_bytes, stream), "cudaMemsetAsync(token)");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(setup)");

    float* host_logits = nullptr;
    std::uint32_t* host_token = nullptr;
    check(cudaHostAlloc(
              reinterpret_cast<void**>(&host_logits),
              logits_bytes,
              cudaHostAllocDefault),
          "cudaHostAlloc(logits)");
    check(cudaHostAlloc(
              reinterpret_cast<void**>(&host_token),
              token_bytes,
              cudaHostAllocDefault),
          "cudaHostAlloc(token)");
    std::vector<float> host_kv(kv_words_size);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    const auto capture_start = std::chrono::steady_clock::now();
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
    if (tensor.type == "IQ3_S") {
      check(rtxllm_launch_iq3s_matvec_decode_probe(
                device_payload,
                device_activation,
                device_logits,
                device_kv,
                device_token,
                shape.rows,
                shape.cols,
                checked_rows,
                kv_words,
                options.page_words,
                options.active_pages,
                stream),
            "iq3s_matvec_decode_probe");
    } else {
      check(rtxllm_launch_iq2s_matvec_decode_probe(
                device_payload,
                device_activation,
                device_logits,
                device_kv,
                device_token,
                shape.rows,
                shape.cols,
                checked_rows,
                kv_words,
                options.page_words,
                options.active_pages,
                stream),
            "iq2s_matvec_decode_probe");
    }
    check(cudaMemcpyAsync(
              host_token,
              device_token,
              token_bytes,
              cudaMemcpyDeviceToHost,
              stream),
          "cudaMemcpyAsync(token)");
    check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    const auto capture_stop = std::chrono::steady_clock::now();
    check(cudaGraphInstantiate(&graph_exec, graph, 0), "cudaGraphInstantiate");
    const auto upload_start = std::chrono::steady_clock::now();
    check(cudaGraphUpload(graph_exec, stream), "cudaGraphUpload");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(upload)");
    const auto upload_stop = std::chrono::steady_clock::now();

    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<std::size_t>(options.steps));
    const auto wall_start = std::chrono::steady_clock::now();
    for (int step = 0; step < options.steps; ++step) {
      const auto start = std::chrono::steady_clock::now();
      check(cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch");
      check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(step)");
      const auto stop = std::chrono::steady_clock::now();
      latencies_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    const auto wall_stop = std::chrono::steady_clock::now();

    check(cudaMemcpyAsync(
              host_logits,
              device_logits,
              logits_bytes,
              cudaMemcpyDeviceToHost,
              stream),
          "cudaMemcpyAsync(logits-reference)");
    check(cudaMemcpyAsync(
              host_kv.data(),
              device_kv,
              kv_bytes,
              cudaMemcpyDeviceToHost,
              stream),
          "cudaMemcpyAsync(kv-reference)");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(reference)");

    const auto reference = compare_decode(
        expected,
        host_logits,
        host_kv.data(),
        *host_token,
        tolerance_for_type(tensor.type));
    const double wall_ms =
        std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
    const double mean_ms = latencies_ms.empty()
        ? 0.0
        : std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) /
              static_cast<double>(latencies_ms.size());
    const double capture_ms =
        std::chrono::duration<double, std::milli>(capture_stop - capture_start).count();
    const double upload_ms =
        std::chrono::duration<double, std::milli>(upload_stop - upload_start).count();

    std::cout << "{\n";
    std::cout << "  \"path\": \"" << matvec_path_for_type(tensor.type) << "\",\n";
    std::cout << "  \"label\": \"" << json_escape(options.label) << "\",\n";
    std::cout << "  \"admitted\": true,\n";
    std::cout << "  \"reference_passed\": " << (reference.passed ? "true" : "false") << ",\n";
    std::cout << "  \"manifest\": \"" << json_escape(options.manifest.string()) << "\",\n";
    std::cout << "  \"tensor\": \"" << json_escape(tensor.name) << "\",\n";
    std::cout << "  \"type\": \"" << json_escape(tensor.type) << "\",\n";
    std::cout << "  \"runtime_kernel\": \"" << runtime_kernel_for_type(tensor.type) << "\",\n";
    std::cout << "  \"rows\": " << shape.rows << ",\n";
    std::cout << "  \"cols\": " << shape.cols << ",\n";
    std::cout << "  \"rows_checked\": " << checked_rows << ",\n";
    std::cout << "  \"block_count\": " << tensor.block_count << ",\n";
    std::cout << "  \"payload_bytes\": " << payload.size() << ",\n";
    std::cout << "  \"resident_bytes\": " << tensor.resident_bytes << ",\n";
    std::cout << "  \"host_weight_checksum\": " << host_weight_checksum << ",\n";
    std::cout << "  \"logical_values_per_step\": "
              << (static_cast<std::uint64_t>(checked_rows) * shape.cols) << ",\n";
    std::cout << "  \"kv_mib\": " << options.kv_mib << ",\n";
    std::cout << "  \"kv_words\": " << kv_words << ",\n";
    std::cout << "  \"page_words\": " << options.page_words << ",\n";
    std::cout << "  \"active_pages\": " << options.active_pages << ",\n";
    std::cout << "  \"reference_tolerance\": " << reference.tolerance << ",\n";
    std::cout << "  \"max_logit_abs_error\": " << reference.max_abs_error << ",\n";
    std::cout << "  \"max_logit_error_row\": " << reference.max_error_row << ",\n";
    std::cout << "  \"first_logit_expected\": " << reference.first_expected << ",\n";
    std::cout << "  \"first_logit_observed\": " << reference.first_observed << ",\n";
    std::cout << "  \"logit_mismatches\": " << reference.mismatches << ",\n";
    std::cout << "  \"max_kv_abs_error\": " << reference.max_kv_abs_error << ",\n";
    std::cout << "  \"max_kv_error_index\": " << reference.max_kv_error_index << ",\n";
    std::cout << "  \"kv_mismatches\": " << reference.kv_mismatches << ",\n";
    std::cout << "  \"token_expected\": " << reference.token_expected << ",\n";
    std::cout << "  \"token_observed\": " << reference.token_observed << ",\n";
    std::cout << "  \"steps\": " << options.steps << ",\n";
    std::cout << "  \"wall_ms\": " << wall_ms << ",\n";
    std::cout << "  \"steps_per_second\": " << (options.steps * 1000.0 / wall_ms) << ",\n";
    std::cout << "  \"p50_ms\": " << percentile(latencies_ms, 50) << ",\n";
    std::cout << "  \"p95_ms\": " << percentile(latencies_ms, 95) << ",\n";
    std::cout << "  \"p99_ms\": " << percentile(latencies_ms, 99) << ",\n";
    std::cout << "  \"latency_mean_ms\": " << mean_ms << ",\n";
    std::cout << "  \"capture_ms\": " << capture_ms << ",\n";
    std::cout << "  \"graph_upload_ms\": " << upload_ms << ",\n";
    std::cout << "  \"graph_replay_rate\": 1.0,\n";
    std::cout << "  \"allocated_bytes\": " << pool.total_bytes() << ",\n";
    std::cout << "  \"setup_h2d_bytes\": " << (payload.size() + activation_bytes) << ",\n";
    std::cout << "  \"h2d_bytes\": 0,\n";
    std::cout << "  \"d2h_bytes\": " << (token_bytes * static_cast<std::size_t>(options.steps))
              << ",\n";
    std::cout << "  \"reference_d2h_bytes\": " << (logits_bytes + kv_bytes) << "\n";
    std::cout << "}\n";

    check(cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy");
    check(cudaGraphDestroy(graph), "cudaGraphDestroy");
    check(cudaFreeHost(host_logits), "cudaFreeHost(logits)");
    check(cudaFreeHost(host_token), "cudaFreeHost(token)");
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return reference.passed ? 0 : 3;
  } catch (const std::exception& exc) {
    std::cerr << "runtime_iq_matvec_probe failed: " << exc.what() << "\n";
    return 1;
  }
}
