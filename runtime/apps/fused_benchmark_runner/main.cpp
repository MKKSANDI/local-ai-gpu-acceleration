#include "runtime/core/allocator/vram_superallocator.h"
#include "runtime/core/scheduler/admission.h"
#include "runtime/kernels/decode/decode_kernels.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
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
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kAlignment = 256;

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

struct Options {
  int steps = 128;
  std::uint32_t rows = 2048;
  std::uint32_t cols = 4096;
  std::uint32_t page_words = 16u * 1024u;
  std::uint32_t active_pages = 256u;
  std::size_t kv_mib = 16;
  std::size_t wddm_guard_mib = 512;
  bool check_reference = false;
  bool check_only = false;
  std::string label = "default";
  std::string q4k_kernel = "direct";
  std::string packed_kernel = "block";
  std::filesystem::path weights_file;
  std::filesystem::path q4k_blocks_file;
  std::filesystem::path q4k_payload_file;
  std::filesystem::path q4k_meta_file;
  std::filesystem::path q4k_manifest_file;
  std::string q4k_manifest_tensor;
  std::uint64_t q4k_manifest_source_checksum = 0;
  std::uint64_t q4k_manifest_payload_checksum = 0;
  std::uint64_t q4k_manifest_metadata_checksum = 0;
};

struct Q4KManifestTensor {
  std::string name;
  std::uint32_t rows = 0;
  std::uint32_t cols = 0;
  std::string runtime_kernel;
  std::filesystem::path payload_path;
  std::filesystem::path metadata_path;
  std::filesystem::path blocks_path;
  std::uint64_t source_checksum = 0;
  std::uint64_t payload_checksum = 0;
  std::uint64_t metadata_checksum = 0;
};

struct ReferenceResult {
  bool checked = false;
  bool passed = false;
  float tolerance = 0.0f;
  float max_logit_abs_error = 0.0f;
  float max_kv_abs_error = 0.0f;
  float max_logit_expected = 0.0f;
  float max_logit_observed = 0.0f;
  float first_logit_expected = 0.0f;
  float first_logit_observed = 0.0f;
  std::uint32_t max_logit_error_row = 0;
  std::uint32_t max_kv_error_index = 0;
  std::uint32_t token_expected = 0;
  std::uint32_t token_observed = 0;
  std::size_t logit_mismatches = 0;
  std::size_t kv_mismatches = 0;
};

std::uint32_t parse_u32(const char* value, std::string_view name) {
  const unsigned long parsed = std::stoul(value);
  if (parsed == 0 || parsed > UINT32_MAX) {
    throw std::invalid_argument(std::string(name) + " must be in uint32 range and non-zero");
  }
  return static_cast<std::uint32_t>(parsed);
}

std::size_t parse_size(const char* value, std::string_view name) {
  const unsigned long long parsed = std::stoull(value);
  if (parsed == 0) {
    throw std::invalid_argument(std::string(name) + " must be non-zero");
  }
  return static_cast<std::size_t>(parsed);
}

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open manifest: " + path.string());
  }
  in.seekg(0, std::ios::end);
  const auto size = static_cast<std::size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  std::string text(size, '\0');
  if (size > 0) {
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!in) {
      throw std::runtime_error("failed to read manifest: " + path.string());
    }
  }
  return text;
}

std::size_t find_json_key(std::string_view text, std::string_view key) {
  const std::string needle = "\"" + std::string(key) + "\"";
  const auto pos = text.find(needle);
  if (pos == std::string_view::npos) {
    throw std::runtime_error("manifest field missing: " + std::string(key));
  }
  const auto colon = text.find(':', pos + needle.size());
  if (colon == std::string_view::npos) {
    throw std::runtime_error("manifest field has no value: " + std::string(key));
  }
  return colon + 1;
}

std::size_t skip_json_space(std::string_view text, std::size_t pos) {
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
    ++pos;
  }
  return pos;
}

std::string parse_json_string_at(std::string_view text, std::size_t pos) {
  pos = skip_json_space(text, pos);
  if (pos >= text.size() || text[pos] != '"') {
    throw std::runtime_error("manifest string value expected");
  }
  ++pos;
  std::string out;
  while (pos < text.size()) {
    const char ch = text[pos++];
    if (ch == '"') {
      return out;
    }
    if (ch != '\\') {
      out += ch;
      continue;
    }
    if (pos >= text.size()) {
      throw std::runtime_error("manifest string has trailing escape");
    }
    const char escaped = text[pos++];
    switch (escaped) {
      case '"':
      case '\\':
      case '/':
        out += escaped;
        break;
      case 'b':
        out += '\b';
        break;
      case 'f':
        out += '\f';
        break;
      case 'n':
        out += '\n';
        break;
      case 'r':
        out += '\r';
        break;
      case 't':
        out += '\t';
        break;
      default:
        throw std::runtime_error("manifest string contains unsupported escape");
    }
  }
  throw std::runtime_error("manifest string is unterminated");
}

std::string json_string_field(std::string_view object, std::string_view key) {
  return parse_json_string_at(object, find_json_key(object, key));
}

std::string json_optional_string_field(std::string_view object, std::string_view key) {
  const std::string needle = "\"" + std::string(key) + "\"";
  const auto pos = object.find(needle);
  if (pos == std::string_view::npos) {
    return {};
  }
  const auto colon = object.find(':', pos + needle.size());
  if (colon == std::string_view::npos) {
    return {};
  }
  return parse_json_string_at(object, colon + 1);
}

std::uint64_t json_u64_field(std::string_view object, std::string_view key) {
  std::size_t pos = skip_json_space(object, find_json_key(object, key));
  std::uint64_t value = 0;
  bool saw_digit = false;
  while (pos < object.size() && std::isdigit(static_cast<unsigned char>(object[pos])) != 0) {
    saw_digit = true;
    value = value * 10u + static_cast<std::uint64_t>(object[pos] - '0');
    ++pos;
  }
  if (!saw_digit) {
    throw std::runtime_error("manifest integer value expected: " + std::string(key));
  }
  return value;
}

std::vector<std::string_view> manifest_tensor_objects(std::string_view text) {
  const auto tensors_key = text.find("\"tensors\"");
  if (tensors_key == std::string_view::npos) {
    throw std::runtime_error("manifest does not contain tensors array");
  }
  const auto array_start = text.find('[', tensors_key);
  if (array_start == std::string_view::npos) {
    throw std::runtime_error("manifest tensors field is not an array");
  }
  std::vector<std::string_view> objects;
  bool in_string = false;
  bool escape = false;
  int depth = 0;
  std::size_t object_start = std::string_view::npos;
  for (std::size_t pos = array_start + 1; pos < text.size(); ++pos) {
    const char ch = text[pos];
    if (in_string) {
      if (escape) {
        escape = false;
      } else if (ch == '\\') {
        escape = true;
      } else if (ch == '"') {
        in_string = false;
      }
      continue;
    }
    if (ch == '"') {
      in_string = true;
      continue;
    }
    if (ch == '{') {
      if (depth == 0) {
        object_start = pos;
      }
      ++depth;
      continue;
    }
    if (ch == '}') {
      --depth;
      if (depth == 0 && object_start != std::string_view::npos) {
        objects.push_back(text.substr(object_start, pos - object_start + 1));
        object_start = std::string_view::npos;
      }
      continue;
    }
    if (ch == ']' && depth == 0) {
      break;
    }
  }
  return objects;
}

Q4KManifestTensor load_q4k_manifest_tensor(
    const std::filesystem::path& path,
    const std::string& selected_name) {
  const std::string text = read_text_file(path);
  const auto objects = manifest_tensor_objects(text);
  if (objects.empty()) {
    throw std::runtime_error("manifest contains no tensors");
  }
  if (selected_name.empty() && objects.size() != 1) {
    throw std::runtime_error("--q4k-tensor is required when a manifest contains multiple tensors");
  }

  for (const auto object : objects) {
    Q4KManifestTensor tensor;
    tensor.name = json_string_field(object, "name");
    if (!selected_name.empty() && tensor.name != selected_name) {
      continue;
    }
    tensor.rows = static_cast<std::uint32_t>(json_u64_field(object, "rows"));
    tensor.cols = static_cast<std::uint32_t>(json_u64_field(object, "cols"));
    tensor.runtime_kernel = json_optional_string_field(object, "runtime_kernel");
    if (tensor.runtime_kernel.empty()) {
      tensor.runtime_kernel = json_string_field(object, "policy");
    }
    tensor.payload_path = json_string_field(object, "payload_path");
    tensor.metadata_path = json_string_field(object, "metadata_path");
    tensor.blocks_path = json_optional_string_field(object, "blocks_path");
    tensor.source_checksum = json_u64_field(object, "source_checksum");
    tensor.payload_checksum = json_u64_field(object, "payload_checksum");
    tensor.metadata_checksum = json_u64_field(object, "metadata_checksum");
    if (tensor.rows == 0 || tensor.cols == 0) {
      throw std::runtime_error("manifest tensor has invalid rows/cols");
    }
    return tensor;
  }

  throw std::runtime_error("manifest tensor not found: " + selected_name);
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
    if (arg == "--steps") {
      options.steps = static_cast<int>(parse_size(require_value(arg), arg));
    } else if (arg == "--rows") {
      options.rows = parse_u32(require_value(arg), arg);
    } else if (arg == "--cols") {
      options.cols = parse_u32(require_value(arg), arg);
    } else if (arg == "--page-words") {
      options.page_words = parse_u32(require_value(arg), arg);
    } else if (arg == "--active-pages") {
      options.active_pages = parse_u32(require_value(arg), arg);
    } else if (arg == "--kv-mib") {
      options.kv_mib = parse_size(require_value(arg), arg);
    } else if (arg == "--wddm-guard-mib") {
      options.wddm_guard_mib = parse_size(require_value(arg), arg);
    } else if (arg == "--label") {
      options.label = require_value(arg);
    } else if (arg == "--q4k-kernel") {
      options.q4k_kernel = require_value(arg);
    } else if (arg == "--packed-kernel") {
      options.packed_kernel = require_value(arg);
    } else if (arg == "--weights-file") {
      options.weights_file = require_value(arg);
    } else if (arg == "--q4k-blocks-file") {
      options.q4k_blocks_file = require_value(arg);
    } else if (arg == "--q4k-payload-file") {
      options.q4k_payload_file = require_value(arg);
    } else if (arg == "--q4k-meta-file") {
      options.q4k_meta_file = require_value(arg);
    } else if (arg == "--q4k-manifest") {
      options.q4k_manifest_file = require_value(arg);
    } else if (arg == "--q4k-tensor") {
      options.q4k_manifest_tensor = require_value(arg);
    } else if (arg == "--check-reference") {
      options.check_reference = true;
    } else if (arg == "--check-only") {
      options.check_reference = true;
      options.check_only = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "runtime_fused_bench [--label NAME] [--steps N] [--rows N] [--cols N]\n"
          << "                    [--kv-mib N] [--page-words N] [--active-pages N]\n"
          << "                    [--wddm-guard-mib N] [--weights-file PATH]\n"
          << "                    [--q4k-blocks-file PATH]\n"
          << "                    [--q4k-payload-file PATH]\n"
          << "                    [--q4k-meta-file PATH]\n"
          << "                    [--q4k-manifest PATH --q4k-tensor NAME]\n"
          << "                    [--packed-kernel block|vec4x4]\n"
          << "                    [--q4k-kernel direct|shared|warp|broadcast|vec4|vec4x2|vec4x4|predecoded|split-predecoded|split-half|split-compact|split-native]\n"
          << "                    [--check-reference] [--check-only]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + std::string(arg));
    }
  }
  if (!options.q4k_manifest_file.empty()) {
    if (!options.weights_file.empty() || !options.q4k_blocks_file.empty() ||
        !options.q4k_payload_file.empty() || !options.q4k_meta_file.empty()) {
      throw std::invalid_argument(
          "--q4k-manifest cannot be combined with explicit weight/q4k file paths");
    }
    const Q4KManifestTensor tensor =
        load_q4k_manifest_tensor(options.q4k_manifest_file, options.q4k_manifest_tensor);
    options.rows = tensor.rows;
    options.cols = tensor.cols;
    options.q4k_kernel = tensor.runtime_kernel;
    options.q4k_payload_file = tensor.payload_path;
    options.q4k_meta_file = tensor.metadata_path;
    options.q4k_blocks_file = tensor.blocks_path;
    options.q4k_manifest_tensor = tensor.name;
    options.q4k_manifest_source_checksum = tensor.source_checksum;
    options.q4k_manifest_payload_checksum = tensor.payload_checksum;
    options.q4k_manifest_metadata_checksum = tensor.metadata_checksum;
  }
  if ((options.cols & 1u) != 0u) {
    throw std::invalid_argument("--cols must be even for packed Q4 layout");
  }
  if (!options.weights_file.empty() &&
      (!options.q4k_blocks_file.empty() || !options.q4k_payload_file.empty())) {
    throw std::invalid_argument("--weights-file and Q4_K layout files are mutually exclusive");
  }
  if (!options.q4k_blocks_file.empty() && options.cols % 256u != 0u) {
    throw std::invalid_argument("--cols must be a multiple of 256 for Q4_K block layout");
  }
  if (options.q4k_kernel != "direct" && options.q4k_kernel != "shared" &&
      options.q4k_kernel != "warp" && options.q4k_kernel != "broadcast" &&
      options.q4k_kernel != "vec4" && options.q4k_kernel != "vec4x2" &&
      options.q4k_kernel != "vec4x4" && options.q4k_kernel != "predecoded" &&
      options.q4k_kernel != "split-predecoded" && options.q4k_kernel != "split-half" &&
      options.q4k_kernel != "split-compact" && options.q4k_kernel != "split-native") {
    throw std::invalid_argument(
        "--q4k-kernel must be direct, shared, warp, broadcast, vec4, vec4x2, vec4x4, predecoded, split-predecoded, split-half, split-compact, or split-native");
  }
  const bool needs_q4k_payload =
      options.q4k_kernel == "split-predecoded" || options.q4k_kernel == "split-half" ||
      options.q4k_kernel == "split-compact" || options.q4k_kernel == "split-native";
  const bool has_q4k_layout_file =
      !options.q4k_blocks_file.empty() || !options.q4k_payload_file.empty();
  if (!options.q4k_payload_file.empty() && !needs_q4k_payload) {
    throw std::invalid_argument(
        "--q4k-payload-file is only valid with --q4k-kernel split-predecoded, split-half, split-compact, or split-native");
  }
  if (needs_q4k_payload && options.q4k_payload_file.empty()) {
    throw std::invalid_argument(
        "--q4k-kernel split-predecoded/split-half/split-compact/split-native requires --q4k-payload-file");
  }
  if (!options.q4k_meta_file.empty() && !needs_q4k_payload) {
    throw std::invalid_argument(
        "--q4k-meta-file is only valid with --q4k-kernel split-predecoded, split-half, split-compact, or split-native");
  }
  if (needs_q4k_payload && options.q4k_blocks_file.empty() && options.q4k_meta_file.empty()) {
    throw std::invalid_argument(
        "split Q4_K payload without --q4k-blocks-file requires --q4k-meta-file");
  }
  if (has_q4k_layout_file && options.cols % 256u != 0u) {
    throw std::invalid_argument("--cols must be a multiple of 256 for Q4_K layout");
  }
  if (!has_q4k_layout_file && options.q4k_kernel != "direct") {
    throw std::invalid_argument("--q4k-kernel is only valid with Q4_K block or payload files");
  }
  if (options.q4k_blocks_file.empty() && !needs_q4k_payload &&
      options.q4k_kernel != "direct") {
    throw std::invalid_argument("non-split Q4_K kernels require --q4k-blocks-file");
  }
  if (options.packed_kernel != "block" && options.packed_kernel != "vec4x4") {
    throw std::invalid_argument("--packed-kernel must be block or vec4x4");
  }
  if (has_q4k_layout_file && options.packed_kernel != "block") {
    throw std::invalid_argument("--packed-kernel is only valid without Q4_K layout files");
  }
  if (options.packed_kernel == "vec4x4" && options.cols % 8u != 0u) {
    throw std::invalid_argument("--cols must be a multiple of 8 for --packed-kernel vec4x4");
  }
  return options;
}

float dequant_q4_host(int nibble, float scale) {
  return static_cast<float>(nibble - 8) * scale;
}

float fp16_to_float_host(std::uint16_t bits) {
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

std::uint16_t float_to_fp16_bits_host(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  const std::uint32_t sign = (bits >> 16u) & 0x8000u;
  int exp = static_cast<int>((bits >> 23u) & 0xffu) - 127 + 15;
  std::uint32_t mant = bits & 0x007fffffu;

  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    mant |= 0x00800000u;
    const int shift = 14 - exp;
    const std::uint32_t rounded = (mant + (1u << (shift - 1))) >> shift;
    return static_cast<std::uint16_t>(sign | rounded);
  }
  if (exp >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }

  mant += 0x00001000u;
  if (mant & 0x00800000u) {
    mant = 0;
    ++exp;
    if (exp >= 31) {
      return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
  }
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10u) |
                                    (mant >> 13u));
}

void get_scale_min_k4_host(
    int index,
    const std::uint8_t* packed,
    std::uint8_t& scale,
    std::uint8_t& min_value) {
  if (index < 4) {
    scale = static_cast<std::uint8_t>(packed[index] & 63u);
    min_value = static_cast<std::uint8_t>(packed[index + 4] & 63u);
    return;
  }
  scale = static_cast<std::uint8_t>(
      (packed[index + 4] & 0x0fu) | ((packed[index - 4] >> 6u) << 4u));
  min_value = static_cast<std::uint8_t>(
      (packed[index + 4] >> 4u) | ((packed[index] >> 6u) << 4u));
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

std::uint64_t byte_checksum(const std::vector<std::uint8_t>& bytes) {
  std::uint64_t sum = 0;
  for (const auto byte : bytes) {
    sum += byte;
  }
  return sum;
}

template <typename T>
std::uint64_t byte_checksum_values(const std::vector<T>& values) {
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(values.data());
  std::uint64_t sum = 0;
  for (std::size_t index = 0; index < values.size() * sizeof(T); ++index) {
    sum += bytes[index];
  }
  return sum;
}

std::vector<std::uint8_t> load_binary_file(
    const std::filesystem::path& path,
    std::size_t expected,
    std::string_view label) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open " + std::string(label) + ": " + path.string());
  }
  in.seekg(0, std::ios::end);
  const auto size = static_cast<std::size_t>(in.tellg());
  if (size != expected) {
    throw std::runtime_error(
        std::string(label) + " size mismatch: expected " + std::to_string(expected) +
        " bytes, got " + std::to_string(size));
  }
  in.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> data(size);
  if (size > 0) {
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!in) {
      throw std::runtime_error("failed to read " + std::string(label) + ": " + path.string());
    }
  }
  return data;
}

std::vector<std::uint8_t> load_weights_file(
    const std::filesystem::path& path,
    std::size_t expected) {
  return load_binary_file(path, expected, "--weights-file");
}

template <typename T>
std::vector<T> load_typed_file(
    const std::filesystem::path& path,
    std::size_t expected_bytes,
    std::string_view label) {
  const auto bytes = load_binary_file(path, expected_bytes, label);
  if (bytes.size() % sizeof(T) != 0) {
    throw std::runtime_error(std::string(label) + " size is not a whole element count");
  }
  std::vector<T> values(bytes.size() / sizeof(T));
  if (!bytes.empty()) {
    std::memcpy(values.data(), bytes.data(), bytes.size());
  }
  return values;
}

std::vector<float> predecode_q4k_scale_min_pairs(
    const std::vector<std::uint8_t>& q4k_blocks,
    std::uint32_t rows,
    std::uint32_t cols) {
  constexpr std::size_t kQ4KValues = 256;
  constexpr std::size_t kQ4KBytes = 144;
  const std::size_t blocks_per_row = cols / kQ4KValues;
  std::vector<float> pairs(static_cast<std::size_t>(rows) * blocks_per_row * 8u * 2u);
  for (std::uint32_t row = 0; row < rows; ++row) {
    const std::size_t row_base = static_cast<std::size_t>(row) * blocks_per_row * kQ4KBytes;
    for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
      const auto* block = q4k_blocks.data() + row_base + block_index * kQ4KBytes;
      const auto d_bits = static_cast<std::uint16_t>(
          block[0] | (static_cast<std::uint16_t>(block[1]) << 8u));
      const auto dmin_bits = static_cast<std::uint16_t>(
          block[2] | (static_cast<std::uint16_t>(block[3]) << 8u));
      const float d = fp16_to_float_host(d_bits);
      const float dmin = fp16_to_float_host(dmin_bits);
      const auto* block_scales = block + 4;
      const std::size_t pair_base =
          (static_cast<std::size_t>(row) * blocks_per_row + block_index) * 8u * 2u;
      for (int pair = 0; pair < 8; ++pair) {
        std::uint8_t scale = 0;
        std::uint8_t min_value = 0;
        get_scale_min_k4_host(pair, block_scales, scale, min_value);
        pairs[pair_base + static_cast<std::size_t>(pair) * 2u] =
            d * static_cast<float>(scale);
        pairs[pair_base + static_cast<std::size_t>(pair) * 2u + 1u] =
            dmin * static_cast<float>(min_value);
      }
    }
  }
  return pairs;
}

std::vector<std::uint16_t> predecode_q4k_scale_min_half_pairs(
    const std::vector<std::uint8_t>& q4k_blocks,
    std::uint32_t rows,
    std::uint32_t cols) {
  const auto float_pairs = predecode_q4k_scale_min_pairs(q4k_blocks, rows, cols);
  std::vector<std::uint16_t> half_pairs(float_pairs.size());
  for (std::size_t index = 0; index < float_pairs.size(); ++index) {
    half_pairs[index] = float_to_fp16_bits_host(float_pairs[index]);
  }
  return half_pairs;
}

std::vector<std::uint8_t> predecode_q4k_compact_meta(
    const std::vector<std::uint8_t>& q4k_blocks,
    std::uint32_t rows,
    std::uint32_t cols) {
  constexpr std::size_t kQ4KValues = 256;
  constexpr std::size_t kQ4KBytes = 144;
  constexpr std::size_t kQ4KCompactMetaBytes = 20;
  const std::size_t blocks_per_row = cols / kQ4KValues;
  std::vector<std::uint8_t> meta(
      static_cast<std::size_t>(rows) * blocks_per_row * kQ4KCompactMetaBytes);
  for (std::uint32_t row = 0; row < rows; ++row) {
    const std::size_t row_base = static_cast<std::size_t>(row) * blocks_per_row * kQ4KBytes;
    for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
      const auto* block = q4k_blocks.data() + row_base + block_index * kQ4KBytes;
      auto* out = meta.data() +
          (static_cast<std::size_t>(row) * blocks_per_row + block_index) *
              kQ4KCompactMetaBytes;
      out[0] = block[0];
      out[1] = block[1];
      out[2] = block[2];
      out[3] = block[3];
      for (int pair = 0; pair < 8; ++pair) {
        std::uint8_t scale = 0;
        std::uint8_t min_value = 0;
        get_scale_min_k4_host(pair, block + 4, scale, min_value);
        out[4u + static_cast<std::size_t>(pair)] = scale;
        out[12u + static_cast<std::size_t>(pair)] = min_value;
      }
    }
  }
  return meta;
}

std::vector<std::uint8_t> extract_q4k_native_meta(
    const std::vector<std::uint8_t>& q4k_blocks,
    std::uint32_t rows,
    std::uint32_t cols) {
  constexpr std::size_t kQ4KValues = 256;
  constexpr std::size_t kQ4KBytes = 144;
  constexpr std::size_t kQ4KNativeMetaBytes = 16;
  const std::size_t blocks_per_row = cols / kQ4KValues;
  std::vector<std::uint8_t> meta(
      static_cast<std::size_t>(rows) * blocks_per_row * kQ4KNativeMetaBytes);
  for (std::uint32_t row = 0; row < rows; ++row) {
    const std::size_t row_base = static_cast<std::size_t>(row) * blocks_per_row * kQ4KBytes;
    for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
      const auto* block = q4k_blocks.data() + row_base + block_index * kQ4KBytes;
      auto* out = meta.data() +
          (static_cast<std::size_t>(row) * blocks_per_row + block_index) *
              kQ4KNativeMetaBytes;
      std::copy(block, block + kQ4KNativeMetaBytes, out);
    }
  }
  return meta;
}

ReferenceResult run_reference_check(
    const std::vector<std::uint8_t>& host_weights,
    const std::vector<float>& host_activation,
    const std::vector<float>& host_scales,
    bool use_q4k_blocks,
    bool use_q4k_shared_kernel,
    bool use_q4k_warp_kernel,
    bool use_q4k_broadcast_kernel,
    bool use_q4k_vec4_kernel,
    bool use_q4k_vec4x2_kernel,
    bool use_q4k_vec4x4_kernel,
    bool use_q4k_predecoded_kernel,
    bool use_q4k_split_predecoded_kernel,
    bool use_q4k_split_half_kernel,
    bool use_q4k_split_compact_kernel,
    bool use_q4k_split_native_kernel,
    bool use_packed_vec4x4_kernel,
    const std::uint8_t* packed_weights,
    const float* q4k_scale_min_pairs,
    const std::uint16_t* q4k_scale_min_half_pairs,
    const std::uint8_t* q4k_compact_meta,
    const std::uint8_t* q4k_native_meta,
    const float* scales,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::size_t packed_stride,
    std::size_t logits_bytes,
    std::size_t kv_bytes,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream) {
  const float tolerance = use_q4k_split_half_kernel ? 1.25e-2f
      : (use_q4k_blocks ? 5.0e-3f : 1.0e-4f);
  const std::uint32_t initial_token = 0;
  const std::uint32_t kv_words = static_cast<std::uint32_t>(kv_bytes / sizeof(float));

  check(cudaMemsetAsync(logits, 0, logits_bytes, stream), "reference logits memset");
  check(cudaMemsetAsync(kv, 0, kv_bytes, stream), "reference kv memset");
  check(cudaMemcpyAsync(token, &initial_token, sizeof(initial_token), cudaMemcpyHostToDevice, stream),
      "reference token reset");
  cudaError_t launch_status = cudaSuccess;
  if (use_q4k_blocks) {
    if (use_q4k_shared_kernel) {
      launch_status = rtxllm_launch_q4k_matvec_decode_shared(
          packed_weights, activation, logits, kv, token, rows, cols, kv_words, page_words,
          active_pages, stream);
    } else if (use_q4k_warp_kernel) {
      launch_status = rtxllm_launch_q4k_matvec_decode_warp(
          packed_weights, activation, logits, kv, token, rows, cols, kv_words, page_words,
          active_pages, stream);
    } else if (use_q4k_broadcast_kernel) {
      launch_status = rtxllm_launch_q4k_matvec_decode_warp_broadcast(
          packed_weights, activation, logits, kv, token, rows, cols, kv_words, page_words,
          active_pages, stream);
    } else if (use_q4k_vec4_kernel) {
      launch_status = rtxllm_launch_q4k_matvec_decode_warp_broadcast_vec4(
          packed_weights, activation, logits, kv, token, rows, cols, kv_words, page_words,
          active_pages, stream);
    } else if (use_q4k_vec4x2_kernel) {
      launch_status = rtxllm_launch_q4k_matvec_decode_warp_broadcast_vec4x2(
          packed_weights, activation, logits, kv, token, rows, cols, kv_words, page_words,
          active_pages, stream);
    } else if (use_q4k_vec4x4_kernel) {
      launch_status = rtxllm_launch_q4k_matvec_decode_warp_broadcast_vec4x4(
          packed_weights, activation, logits, kv, token, rows, cols, kv_words, page_words,
          active_pages, stream);
    } else if (use_q4k_predecoded_kernel) {
      launch_status = rtxllm_launch_q4k_matvec_decode_predecoded_vec4x4(
          packed_weights, q4k_scale_min_pairs, activation, logits, kv, token, rows, cols,
          kv_words, page_words, active_pages, stream);
    } else if (use_q4k_split_predecoded_kernel) {
      launch_status = rtxllm_launch_q4k_matvec_decode_split_predecoded_vec4x4(
          packed_weights, q4k_scale_min_pairs, activation, logits, kv, token, rows, cols,
          kv_words, page_words, active_pages, stream);
    } else if (use_q4k_split_half_kernel) {
      launch_status = rtxllm_launch_q4k_matvec_decode_split_half_vec4x4(
          packed_weights, q4k_scale_min_half_pairs, activation, logits, kv, token, rows, cols,
          kv_words, page_words, active_pages, stream);
    } else if (use_q4k_split_compact_kernel) {
      launch_status = rtxllm_launch_q4k_matvec_decode_split_compact_vec4x4(
          packed_weights, q4k_compact_meta, activation, logits, kv, token, rows, cols,
          kv_words, page_words, active_pages, stream);
    } else if (use_q4k_split_native_kernel) {
      launch_status = rtxllm_launch_q4k_matvec_decode_split_native_vec4x4(
          packed_weights, q4k_native_meta, activation, logits, kv, token, rows, cols,
          kv_words, page_words, active_pages, stream);
    } else {
      launch_status = rtxllm_launch_q4k_matvec_decode(
          packed_weights, activation, logits, kv, token, rows, cols, kv_words, page_words,
          active_pages, stream);
    }
  } else {
    launch_status = use_packed_vec4x4_kernel
        ? rtxllm_launch_q4_matvec_decode_vec4x4(
              packed_weights, scales, activation, logits, kv, token, rows, cols, kv_words,
              page_words, active_pages, stream)
        : rtxllm_launch_q4_matvec_decode(
              packed_weights, scales, activation, logits, kv, token, rows, cols, kv_words,
              page_words, active_pages, stream);
  }
  check(launch_status,
      "reference q4 matvec decode");

  std::vector<float> gpu_logits(rows);
  std::vector<float> gpu_kv(kv_words);
  std::uint32_t observed_token = 0;
  check(cudaMemcpyAsync(
            gpu_logits.data(), logits, logits_bytes, cudaMemcpyDeviceToHost, stream),
      "reference logits copy");
  check(cudaMemcpyAsync(gpu_kv.data(), kv, kv_bytes, cudaMemcpyDeviceToHost, stream),
      "reference kv copy");
  check(cudaMemcpyAsync(
            &observed_token, token, sizeof(observed_token), cudaMemcpyDeviceToHost, stream),
      "reference token copy");
  check(cudaStreamSynchronize(stream), "reference sync");

  std::vector<float> expected_logits(rows, 0.0f);
  std::vector<float> expected_kv(kv_words, 0.0f);
  float best_value = -3.4028234663852886e38f;
  std::uint32_t best_index = 0;
  constexpr std::size_t kQ4KValues = 256;
  constexpr std::size_t kQ4KBytes = 144;
  for (std::uint32_t row = 0; row < rows; ++row) {
    float sum = 0.0f;
    if (use_q4k_blocks) {
      const std::size_t blocks_per_row = cols / kQ4KValues;
      const std::size_t row_base = static_cast<std::size_t>(row) * blocks_per_row * kQ4KBytes;
      for (std::size_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const auto* block = host_weights.data() + row_base + block_index * kQ4KBytes;
        const auto d_bits = static_cast<std::uint16_t>(
            block[0] | (static_cast<std::uint16_t>(block[1]) << 8u));
        const auto dmin_bits = static_cast<std::uint16_t>(
            block[2] | (static_cast<std::uint16_t>(block[3]) << 8u));
        const float d = fp16_to_float_host(d_bits);
        const float dmin = fp16_to_float_host(dmin_bits);
        const auto* block_scales = block + 4;
        const auto* qs = block + 16;
        const std::size_t col_base = block_index * kQ4KValues;
        for (std::size_t group = 0; group < 4; ++group) {
          std::uint8_t scale_0 = 0;
          std::uint8_t min_0 = 0;
          std::uint8_t scale_1 = 0;
          std::uint8_t min_1 = 0;
          get_scale_min_k4_host(static_cast<int>(group * 2), block_scales, scale_0, min_0);
          get_scale_min_k4_host(static_cast<int>(group * 2 + 1), block_scales, scale_1, min_1);
          const auto* group_qs = qs + group * 32;
          for (std::size_t offset = 0; offset < 32; ++offset) {
            const std::uint8_t packed = group_qs[offset];
            const std::size_t col0 = col_base + group * 64 + offset;
            const std::size_t col1 = col0 + 32;
            const float value0 = d * static_cast<float>(scale_0) *
                    static_cast<float>(packed & 0x0f) -
                dmin * static_cast<float>(min_0);
            const float value1 = d * static_cast<float>(scale_1) *
                    static_cast<float>(packed >> 4) -
                dmin * static_cast<float>(min_1);
            sum += value0 * host_activation[col0];
            sum += value1 * host_activation[col1];
          }
        }
      }
    } else {
      const std::size_t row_base = static_cast<std::size_t>(row) * packed_stride;
      const float scale = host_scales[row];
      for (std::size_t packed_col = 0; packed_col < packed_stride; ++packed_col) {
        const std::uint8_t packed = host_weights[row_base + packed_col];
        const std::uint32_t col0 = static_cast<std::uint32_t>(packed_col * 2u);
        const std::uint32_t col1 = col0 + 1u;
        sum += dequant_q4_host(packed & 0x0f, scale) * host_activation[col0];
        if (col1 < cols) {
          sum += dequant_q4_host((packed >> 4) & 0x0f, scale) * host_activation[col1];
        }
      }
    }
    expected_logits[row] = sum;
    if (sum > best_value) {
      best_value = sum;
      best_index = row;
    }
    if (row < active_pages && kv_words > 0 && page_words > 0) {
      const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
      const std::size_t offset = (initial_token + row) % page_words;
      const std::size_t pos = (page_base + offset) % kv_words;
      expected_kv[pos] = sum;
    }
  }

  ReferenceResult result;
  result.checked = true;
  result.tolerance = tolerance;
  result.token_expected = (best_index + initial_token + 1u) % 32000u;
  result.token_observed = observed_token;
  if (rows > 0) {
    result.first_logit_expected = expected_logits[0];
    result.first_logit_observed = gpu_logits[0];
  }

  for (std::uint32_t row = 0; row < rows; ++row) {
    const float error = std::fabs(gpu_logits[row] - expected_logits[row]);
    if (error > result.max_logit_abs_error) {
      result.max_logit_abs_error = error;
      result.max_logit_error_row = row;
      result.max_logit_expected = expected_logits[row];
      result.max_logit_observed = gpu_logits[row];
    }
    if (error > tolerance) {
      ++result.logit_mismatches;
    }
  }
  for (std::uint32_t index = 0; index < kv_words; ++index) {
    const float error = std::fabs(gpu_kv[index] - expected_kv[index]);
    if (error > result.max_kv_abs_error) {
      result.max_kv_abs_error = error;
      result.max_kv_error_index = index;
    }
    if (error > tolerance) {
      ++result.kv_mismatches;
    }
  }
  result.passed = result.logit_mismatches == 0 && result.kv_mismatches == 0 &&
      result.token_expected == result.token_observed;
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

    const int steps = options.steps;
    const std::uint32_t rows = options.rows;
    const std::uint32_t cols = options.cols;
    const std::uint32_t page_words = options.page_words;
    const std::uint32_t active_pages = options.active_pages;
    const bool use_q4k_blocks = !options.q4k_blocks_file.empty();
    const bool use_q4k_layout = use_q4k_blocks || !options.q4k_payload_file.empty();
    const bool use_q4k_shared_kernel = use_q4k_blocks && options.q4k_kernel == "shared";
    const bool use_q4k_warp_kernel = use_q4k_blocks && options.q4k_kernel == "warp";
    const bool use_q4k_broadcast_kernel = use_q4k_blocks && options.q4k_kernel == "broadcast";
    const bool use_q4k_vec4_kernel = use_q4k_blocks && options.q4k_kernel == "vec4";
    const bool use_q4k_vec4x2_kernel = use_q4k_blocks && options.q4k_kernel == "vec4x2";
    const bool use_q4k_vec4x4_kernel = use_q4k_blocks && options.q4k_kernel == "vec4x4";
    const bool use_q4k_predecoded_kernel = use_q4k_blocks && options.q4k_kernel == "predecoded";
    const bool use_q4k_split_predecoded_kernel =
        use_q4k_layout && options.q4k_kernel == "split-predecoded";
    const bool use_q4k_split_half_kernel = use_q4k_layout && options.q4k_kernel == "split-half";
    const bool use_q4k_split_compact_kernel =
        use_q4k_layout && options.q4k_kernel == "split-compact";
    const bool use_q4k_split_native_kernel =
        use_q4k_layout && options.q4k_kernel == "split-native";
    const bool use_q4k_split_payload_kernel =
        use_q4k_split_predecoded_kernel || use_q4k_split_half_kernel ||
        use_q4k_split_compact_kernel || use_q4k_split_native_kernel;
    const bool use_q4k_predecoded_metadata =
        use_q4k_predecoded_kernel || use_q4k_split_predecoded_kernel ||
        use_q4k_split_half_kernel || use_q4k_split_compact_kernel ||
        use_q4k_split_native_kernel;
    const std::size_t q4k_meta_value_bytes = use_q4k_split_half_kernel
        ? sizeof(std::uint16_t)
        : sizeof(float);
    const bool use_packed_vec4x4_kernel = !use_q4k_layout && options.packed_kernel == "vec4x4";
    const std::size_t packed_stride = (cols + 1u) / 2u;
    const std::size_t packed_payload_bytes = packed_stride * rows;
    const std::size_t q4k_block_bytes =
        use_q4k_layout ? static_cast<std::size_t>(rows) * (cols / 256u) * 144u : 0u;
    const std::size_t q4k_block_count =
        use_q4k_layout ? static_cast<std::size_t>(rows) * (cols / 256u) : 0u;
    const std::size_t q4k_predecoded_meta_bytes = use_q4k_predecoded_metadata
        ? (use_q4k_split_compact_kernel ? q4k_block_count * 20u
           : (use_q4k_split_native_kernel ? q4k_block_count * 16u
                                           : q4k_block_count * 8u * 2u * q4k_meta_value_bytes))
        : 0u;
    const std::size_t weight_bytes =
        use_q4k_layout ? (use_q4k_split_payload_kernel ? packed_payload_bytes : q4k_block_bytes)
                       : packed_payload_bytes;
    const std::size_t scale_bytes = use_q4k_layout ? 0u : sizeof(float) * rows;
    const std::size_t activation_bytes = sizeof(float) * cols;
    const std::size_t logits_bytes = sizeof(float) * rows;
    const std::size_t logical_fp16_weight_bytes =
        static_cast<std::size_t>(rows) * cols * sizeof(std::uint16_t);
    const std::size_t workspace_bytes =
        align_up(activation_bytes) + align_up(q4k_predecoded_meta_bytes) +
        align_up(scale_bytes) + align_up(logits_bytes);
    const std::size_t kv_bytes = mib(options.kv_mib);
    const std::size_t dma_bytes = mib(1);
    const std::size_t setup_h2d_bytes =
        weight_bytes + activation_bytes + scale_bytes + q4k_predecoded_meta_bytes +
        sizeof(std::uint32_t);
    const std::string kernel_variant = [&]() -> std::string {
      if (!use_q4k_layout) {
        return use_packed_vec4x4_kernel ? "packed2_vec4x4" : "packed2";
      }
      if (use_q4k_shared_kernel) {
        return "q4k_blocks_shared";
      }
      if (use_q4k_warp_kernel) {
        return "q4k_blocks_warp";
      }
      if (use_q4k_broadcast_kernel) {
        return "q4k_blocks_warp_broadcast";
      }
      if (use_q4k_vec4_kernel) {
        return "q4k_blocks_warp_broadcast_vec4";
      }
      if (use_q4k_vec4x2_kernel) {
        return "q4k_blocks_warp_broadcast_vec4x2";
      }
      if (use_q4k_vec4x4_kernel) {
        return "q4k_blocks_warp_broadcast_vec4x4";
      }
      if (use_q4k_predecoded_kernel) {
        return "q4k_blocks_predecoded_vec4x4";
      }
      if (use_q4k_split_predecoded_kernel) {
        return "q4k_split_predecoded_vec4x4";
      }
      if (use_q4k_split_half_kernel) {
        return "q4k_split_half_vec4x4";
      }
      if (use_q4k_split_compact_kernel) {
        return "q4k_split_compact_vec4x4";
      }
      if (use_q4k_split_native_kernel) {
        return "q4k_split_native_vec4x4";
      }
      return "q4k_blocks";
    }();
    const std::string q4k_predecoded_meta_format = [&]() -> std::string {
      if (use_q4k_split_half_kernel) {
        return "fp16";
      }
      if (use_q4k_split_compact_kernel) {
        return "compact-u8";
      }
      if (use_q4k_split_native_kernel) {
        return "native16";
      }
      if (use_q4k_predecoded_metadata) {
        return "fp32";
      }
      return "none";
    }();

    const auto decision = rtxllm::decide_admission(
        rtxllm::ResidencyPolicy::Strict,
        rtxllm::MemoryBudget{free_bytes, mib(options.wddm_guard_mib)},
        rtxllm::RequestEstimate{
            weight_bytes,
            512,
            static_cast<std::size_t>(steps),
            1024,
            workspace_bytes + kv_bytes + dma_bytes});

    if (!decision.admit) {
      std::cout << "{\n";
      std::cout << "  \"path\": \"native_cuda_graph_q4_matvec_decode\",\n";
      std::cout << "  \"label\": \"" << options.label << "\",\n";
      std::cout << "  \"kernel_variant\": \"" << kernel_variant << "\",\n";
      std::cout << "  \"admitted\": false,\n";
      std::cout << "  \"reason\": \"" << decision.reason << "\",\n";
      std::cout << "  \"required_bytes\": " << decision.required_bytes << ",\n";
      std::cout << "  \"usable_bytes\": " << decision.usable_bytes << "\n";
      std::cout << "}\n";
      return 2;
    }

    rtxllm::VramSuperallocator pool;
    pool.initialize({
        {"weights", weight_bytes},
        {"kv_pages", kv_bytes},
        {"workspace", workspace_bytes},
        {"dma", dma_bytes},
    });

    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    pool.reset_async(stream);

    auto* packed_weights = static_cast<std::uint8_t*>(pool.arena_ptr("weights"));
    auto* kv = static_cast<float*>(pool.arena_ptr("kv_pages"));
    auto* workspace = static_cast<std::uint8_t*>(pool.arena_ptr("workspace"));
    auto* activation = reinterpret_cast<float*>(workspace);
    auto* q4k_scale_min_pairs =
        reinterpret_cast<float*>(workspace + align_up(activation_bytes));
    auto* q4k_scale_min_half_pairs =
        reinterpret_cast<std::uint16_t*>(workspace + align_up(activation_bytes));
    auto* q4k_compact_meta = workspace + align_up(activation_bytes);
    auto* q4k_native_meta = workspace + align_up(activation_bytes);
    auto* scales = reinterpret_cast<float*>(
        workspace + align_up(activation_bytes) + align_up(q4k_predecoded_meta_bytes));
    auto* logits = reinterpret_cast<float*>(
        workspace + align_up(activation_bytes) + align_up(q4k_predecoded_meta_bytes) +
        align_up(scale_bytes));
    auto* token = static_cast<std::uint32_t*>(pool.arena_ptr("dma"));

    std::vector<std::uint8_t> host_weights;
    std::vector<float> host_activation(cols);
    std::vector<float> host_scales(rows);
    std::vector<float> host_q4k_scale_min_pairs;
    std::vector<std::uint16_t> host_q4k_scale_min_half_pairs;
    std::vector<std::uint8_t> host_q4k_compact_meta;
    std::vector<std::uint8_t> host_q4k_native_meta;
    std::vector<std::uint8_t> host_upload_weights;
    std::string weight_source = "synthetic";
    std::filesystem::path effective_weights_file;
    std::uint64_t q4k_source_checksum = options.q4k_manifest_source_checksum;
    if (use_q4k_layout) {
      if (use_q4k_blocks) {
        std::vector<std::uint8_t> host_q4k_blocks =
            load_binary_file(options.q4k_blocks_file, q4k_block_bytes, "--q4k-blocks-file");
        host_weights = std::move(host_q4k_blocks);
        q4k_source_checksum = byte_checksum(host_weights);
      }
      effective_weights_file =
          use_q4k_split_payload_kernel
          ? options.q4k_payload_file
          : options.q4k_blocks_file;
      host_upload_weights = use_q4k_split_payload_kernel
          ? load_binary_file(options.q4k_payload_file, packed_payload_bytes, "--q4k-payload-file")
          : host_weights;
      weight_source = use_q4k_split_payload_kernel
          ? (options.q4k_manifest_file.empty() ? "q4k_split_payload_file"
                                               : "q4k_manifest_payload_file")
          : "q4k_blocks_file";
    } else if (!options.weights_file.empty()) {
      effective_weights_file = options.weights_file;
      host_weights = load_weights_file(options.weights_file, weight_bytes);
      host_upload_weights = host_weights;
      weight_source = "file";
    } else {
      host_weights.resize(weight_bytes);
      for (std::size_t i = 0; i < host_weights.size(); ++i) {
        host_weights[i] = static_cast<std::uint8_t>((i * 13u + i / 7u) & 0xffu);
      }
      host_upload_weights = host_weights;
    }
    const auto host_weight_checksum = byte_checksum(host_upload_weights);
    if (options.q4k_manifest_payload_checksum != 0 &&
        host_weight_checksum != options.q4k_manifest_payload_checksum) {
      throw std::runtime_error("manifest payload checksum mismatch");
    }
    std::uint64_t q4k_metadata_checksum = 0;
    if (use_q4k_split_half_kernel) {
      host_q4k_scale_min_half_pairs = !options.q4k_meta_file.empty()
          ? load_typed_file<std::uint16_t>(
                options.q4k_meta_file, q4k_predecoded_meta_bytes, "--q4k-meta-file")
          : predecode_q4k_scale_min_half_pairs(host_weights, rows, cols);
      q4k_metadata_checksum = byte_checksum_values(host_q4k_scale_min_half_pairs);
    } else if (use_q4k_split_compact_kernel) {
      host_q4k_compact_meta = !options.q4k_meta_file.empty()
          ? load_binary_file(options.q4k_meta_file, q4k_predecoded_meta_bytes, "--q4k-meta-file")
          : predecode_q4k_compact_meta(host_weights, rows, cols);
      q4k_metadata_checksum = byte_checksum(host_q4k_compact_meta);
    } else if (use_q4k_split_native_kernel) {
      host_q4k_native_meta = !options.q4k_meta_file.empty()
          ? load_binary_file(options.q4k_meta_file, q4k_predecoded_meta_bytes, "--q4k-meta-file")
          : extract_q4k_native_meta(host_weights, rows, cols);
      q4k_metadata_checksum = byte_checksum(host_q4k_native_meta);
    } else if (use_q4k_predecoded_metadata) {
      host_q4k_scale_min_pairs = !options.q4k_meta_file.empty()
          ? load_typed_file<float>(
                options.q4k_meta_file, q4k_predecoded_meta_bytes, "--q4k-meta-file")
          : predecode_q4k_scale_min_pairs(host_weights, rows, cols);
      q4k_metadata_checksum = byte_checksum_values(host_q4k_scale_min_pairs);
    }
    if (options.q4k_manifest_metadata_checksum != 0 &&
        q4k_metadata_checksum != options.q4k_manifest_metadata_checksum) {
      throw std::runtime_error("manifest metadata checksum mismatch");
    }
    for (std::uint32_t col = 0; col < cols; ++col) {
      host_activation[col] = (static_cast<int>(col % 127u) - 63) / 64.0f;
    }
    if (!use_q4k_layout) {
      for (std::uint32_t row = 0; row < rows; ++row) {
        host_scales[row] = 0.015625f + static_cast<float>(row % 17u) * 0.0005f;
      }
    }
    const std::uint32_t initial_token = 0;

    check(cudaMemcpyAsync(
              packed_weights,
              host_upload_weights.data(),
              host_upload_weights.size(),
              cudaMemcpyHostToDevice,
              stream),
        "weights upload");
    check(cudaMemcpyAsync(
              activation,
              host_activation.data(),
              activation_bytes,
              cudaMemcpyHostToDevice,
              stream),
        "activation upload");
    if (!use_q4k_layout) {
      check(cudaMemcpyAsync(scales, host_scales.data(), scale_bytes, cudaMemcpyHostToDevice, stream),
          "scales upload");
    }
    if (use_q4k_split_half_kernel) {
      check(cudaMemcpyAsync(
                q4k_scale_min_half_pairs,
                host_q4k_scale_min_half_pairs.data(),
                q4k_predecoded_meta_bytes,
                cudaMemcpyHostToDevice,
                stream),
          "q4k predecoded half metadata upload");
    } else if (use_q4k_split_compact_kernel) {
      check(cudaMemcpyAsync(
                q4k_compact_meta,
                host_q4k_compact_meta.data(),
                q4k_predecoded_meta_bytes,
                cudaMemcpyHostToDevice,
                stream),
          "q4k compact metadata upload");
    } else if (use_q4k_split_native_kernel) {
      check(cudaMemcpyAsync(
                q4k_native_meta,
                host_q4k_native_meta.data(),
                q4k_predecoded_meta_bytes,
                cudaMemcpyHostToDevice,
                stream),
          "q4k native metadata upload");
    } else if (use_q4k_predecoded_metadata) {
      check(cudaMemcpyAsync(
                q4k_scale_min_pairs,
                host_q4k_scale_min_pairs.data(),
                q4k_predecoded_meta_bytes,
                cudaMemcpyHostToDevice,
                stream),
          "q4k predecoded metadata upload");
    }
    check(cudaMemcpyAsync(token, &initial_token, sizeof(initial_token), cudaMemcpyHostToDevice, stream),
        "token upload");
    check(cudaStreamSynchronize(stream), "setup upload sync");

    const auto launch_decode = [&](cudaStream_t launch_stream) -> cudaError_t {
      if (use_q4k_layout) {
        if (use_q4k_shared_kernel) {
          return rtxllm_launch_q4k_matvec_decode_shared(
              packed_weights,
              activation,
              logits,
              kv,
              token,
              rows,
              cols,
              kv_bytes / sizeof(float),
              page_words,
              active_pages,
              launch_stream);
        }
        if (use_q4k_warp_kernel) {
          return rtxllm_launch_q4k_matvec_decode_warp(
              packed_weights,
              activation,
              logits,
              kv,
              token,
              rows,
              cols,
              kv_bytes / sizeof(float),
              page_words,
              active_pages,
              launch_stream);
        }
        if (use_q4k_broadcast_kernel) {
          return rtxllm_launch_q4k_matvec_decode_warp_broadcast(
              packed_weights,
              activation,
              logits,
              kv,
              token,
              rows,
              cols,
              kv_bytes / sizeof(float),
              page_words,
              active_pages,
              launch_stream);
        }
        if (use_q4k_vec4_kernel) {
          return rtxllm_launch_q4k_matvec_decode_warp_broadcast_vec4(
              packed_weights,
              activation,
              logits,
              kv,
              token,
              rows,
              cols,
              kv_bytes / sizeof(float),
              page_words,
              active_pages,
              launch_stream);
        }
        if (use_q4k_vec4x2_kernel) {
          return rtxllm_launch_q4k_matvec_decode_warp_broadcast_vec4x2(
              packed_weights,
              activation,
              logits,
              kv,
              token,
              rows,
              cols,
              kv_bytes / sizeof(float),
              page_words,
              active_pages,
              launch_stream);
        }
        if (use_q4k_vec4x4_kernel) {
          return rtxllm_launch_q4k_matvec_decode_warp_broadcast_vec4x4(
              packed_weights,
              activation,
              logits,
              kv,
              token,
              rows,
              cols,
              kv_bytes / sizeof(float),
              page_words,
              active_pages,
              launch_stream);
        }
        if (use_q4k_predecoded_kernel) {
          return rtxllm_launch_q4k_matvec_decode_predecoded_vec4x4(
              packed_weights,
              q4k_scale_min_pairs,
              activation,
              logits,
              kv,
              token,
              rows,
              cols,
              kv_bytes / sizeof(float),
              page_words,
              active_pages,
              launch_stream);
        }
        if (use_q4k_split_predecoded_kernel) {
          return rtxllm_launch_q4k_matvec_decode_split_predecoded_vec4x4(
              packed_weights,
              q4k_scale_min_pairs,
              activation,
              logits,
              kv,
              token,
              rows,
              cols,
              kv_bytes / sizeof(float),
              page_words,
              active_pages,
              launch_stream);
        }
        if (use_q4k_split_half_kernel) {
          return rtxllm_launch_q4k_matvec_decode_split_half_vec4x4(
              packed_weights,
              q4k_scale_min_half_pairs,
              activation,
              logits,
              kv,
              token,
              rows,
              cols,
              kv_bytes / sizeof(float),
              page_words,
              active_pages,
              launch_stream);
        }
        if (use_q4k_split_compact_kernel) {
          return rtxllm_launch_q4k_matvec_decode_split_compact_vec4x4(
              packed_weights,
              q4k_compact_meta,
              activation,
              logits,
              kv,
              token,
              rows,
              cols,
              kv_bytes / sizeof(float),
              page_words,
              active_pages,
              launch_stream);
        }
        if (use_q4k_split_native_kernel) {
          return rtxllm_launch_q4k_matvec_decode_split_native_vec4x4(
              packed_weights,
              q4k_native_meta,
              activation,
              logits,
              kv,
              token,
              rows,
              cols,
              kv_bytes / sizeof(float),
              page_words,
              active_pages,
              launch_stream);
        }
        return rtxllm_launch_q4k_matvec_decode(
            packed_weights,
            activation,
            logits,
            kv,
            token,
            rows,
            cols,
            kv_bytes / sizeof(float),
            page_words,
            active_pages,
            launch_stream);
      }
      return use_packed_vec4x4_kernel
          ? rtxllm_launch_q4_matvec_decode_vec4x4(
                packed_weights,
                scales,
                activation,
                logits,
                kv,
                token,
                rows,
                cols,
                kv_bytes / sizeof(float),
                page_words,
                active_pages,
                launch_stream)
          : rtxllm_launch_q4_matvec_decode(
                packed_weights,
                scales,
                activation,
                logits,
                kv,
                token,
                rows,
                cols,
                kv_bytes / sizeof(float),
                page_words,
                active_pages,
                launch_stream);
    };

    ReferenceResult reference;
    if (options.check_reference) {
      if (use_q4k_layout && !use_q4k_blocks) {
        throw std::runtime_error("Q4_K reference checks require --q4k-blocks-file");
      }
      reference = run_reference_check(
          host_weights,
          host_activation,
          host_scales,
          use_q4k_blocks,
          use_q4k_shared_kernel,
          use_q4k_warp_kernel,
          use_q4k_broadcast_kernel,
          use_q4k_vec4_kernel,
          use_q4k_vec4x2_kernel,
          use_q4k_vec4x4_kernel,
          use_q4k_predecoded_kernel,
          use_q4k_split_predecoded_kernel,
          use_q4k_split_half_kernel,
          use_q4k_split_compact_kernel,
          use_q4k_split_native_kernel,
          use_packed_vec4x4_kernel,
          packed_weights,
          q4k_scale_min_pairs,
          q4k_scale_min_half_pairs,
          q4k_compact_meta,
          q4k_native_meta,
          scales,
          activation,
          logits,
          kv,
          token,
          rows,
          cols,
          packed_stride,
          logits_bytes,
          kv_bytes,
          page_words,
          active_pages,
          stream);
      check(cudaMemsetAsync(logits, 0, logits_bytes, stream), "post-check logits reset");
      check(cudaMemsetAsync(kv, 0, kv_bytes, stream), "post-check kv reset");
      check(cudaMemcpyAsync(
                token, &initial_token, sizeof(initial_token), cudaMemcpyHostToDevice, stream),
          "post-check token reset");
      check(cudaStreamSynchronize(stream), "post-check sync");
    }

    if (options.check_only) {
      check(cudaStreamDestroy(stream), "cudaStreamDestroy");
      std::cout << "{\n";
      std::cout << "  \"path\": \"native_cuda_q4_matvec_reference_check\",\n";
      std::cout << "  \"label\": \"" << options.label << "\",\n";
      std::cout << "  \"kernel_variant\": \"" << kernel_variant << "\",\n";
      std::cout << "  \"reference_checked\": true,\n";
      std::cout << "  \"reference_passed\": " << (reference.passed ? "true" : "false") << ",\n";
      std::cout << "  \"rows\": " << rows << ",\n";
      std::cout << "  \"cols\": " << cols << ",\n";
      std::cout << "  \"q4_weight_bytes\": " << weight_bytes << ",\n";
      std::cout << "  \"weight_storage_bytes\": " << weight_bytes << ",\n";
      std::cout << "  \"weight_source\": \"" << weight_source << "\",\n";
      std::cout << "  \"weights_file\": \"" << json_escape(effective_weights_file.string()) << "\",\n";
      std::cout << "  \"host_weight_checksum\": " << host_weight_checksum << ",\n";
      std::cout << "  \"q4k_blocks_file\": \"" << json_escape(options.q4k_blocks_file.string()) << "\",\n";
      std::cout << "  \"q4k_payload_file\": \"" << json_escape(options.q4k_payload_file.string()) << "\",\n";
      std::cout << "  \"q4k_meta_file\": \"" << json_escape(options.q4k_meta_file.string()) << "\",\n";
      std::cout << "  \"q4k_manifest_file\": \""
                << json_escape(options.q4k_manifest_file.string()) << "\",\n";
      std::cout << "  \"q4k_manifest_tensor\": \""
                << json_escape(options.q4k_manifest_tensor) << "\",\n";
      std::cout << "  \"q4k_source_checksum\": " << q4k_source_checksum << ",\n";
      std::cout << "  \"q4k_metadata_checksum\": " << q4k_metadata_checksum << ",\n";
      std::cout << "  \"logical_fp16_weight_bytes\": " << logical_fp16_weight_bytes << ",\n";
      std::cout << "  \"weight_compression_ratio\": "
                << (static_cast<double>(logical_fp16_weight_bytes) / weight_bytes) << ",\n";
      std::cout << "  \"allocated_bytes\": " << pool.total_bytes() << ",\n";
      std::cout << "  \"setup_h2d_bytes\": " << setup_h2d_bytes << ",\n";
      std::cout << "  \"h2d_bytes\": 0,\n";
      std::cout << "  \"d2h_bytes\": " << (logits_bytes + kv_bytes + sizeof(std::uint32_t)) << ",\n";
      std::cout << "  \"q4_values_per_step\": " << (static_cast<std::size_t>(rows) * cols) << ",\n";
      std::cout << "  \"q4_packed_bytes_per_step\": "
                << packed_payload_bytes << ",\n";
      std::cout << "  \"q4k_block_bytes_per_step\": " << q4k_block_bytes << ",\n";
      std::cout << "  \"q4k_predecoded_meta_bytes\": " << q4k_predecoded_meta_bytes << ",\n";
      std::cout << "  \"q4k_predecoded_meta_format\": \""
                << q4k_predecoded_meta_format << "\",\n";
      std::cout << "  \"reference_tolerance\": " << reference.tolerance << ",\n";
      std::cout << "  \"max_logit_abs_error\": " << reference.max_logit_abs_error << ",\n";
      std::cout << "  \"max_logit_error_row\": " << reference.max_logit_error_row << ",\n";
      std::cout << "  \"max_logit_expected\": " << reference.max_logit_expected << ",\n";
      std::cout << "  \"max_logit_observed\": " << reference.max_logit_observed << ",\n";
      std::cout << "  \"first_logit_expected\": " << reference.first_logit_expected << ",\n";
      std::cout << "  \"first_logit_observed\": " << reference.first_logit_observed << ",\n";
      std::cout << "  \"max_kv_abs_error\": " << reference.max_kv_abs_error << ",\n";
      std::cout << "  \"max_kv_error_index\": " << reference.max_kv_error_index << ",\n";
      std::cout << "  \"logit_mismatches\": " << reference.logit_mismatches << ",\n";
      std::cout << "  \"kv_mismatches\": " << reference.kv_mismatches << ",\n";
      std::cout << "  \"token_expected\": " << reference.token_expected << ",\n";
      std::cout << "  \"token_observed\": " << reference.token_observed << "\n";
      std::cout << "}\n";
      return reference.passed ? 0 : 3;
    }

    std::uint32_t* host_token = nullptr;
    check(cudaHostAlloc(
              reinterpret_cast<void**>(&host_token),
              sizeof(std::uint32_t),
              cudaHostAllocDefault),
        "cudaHostAlloc");

    check(launch_decode(stream),
        "warmup q4 matvec decode");
    check(cudaMemcpyAsync(host_token, token, sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream),
        "warmup token memcpy");
    check(cudaStreamSynchronize(stream), "warmup sync");

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    const auto capture_start = std::chrono::steady_clock::now();
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
    check(launch_decode(stream),
        "captured q4 matvec decode");
    check(cudaMemcpyAsync(host_token, token, sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream),
        "captured token memcpy");
    check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    const auto capture_stop = std::chrono::steady_clock::now();

    check(cudaGraphInstantiate(&graph_exec, graph, 0), "cudaGraphInstantiate");
    const auto upload_start = std::chrono::steady_clock::now();
    check(cudaGraphUpload(graph_exec, stream), "cudaGraphUpload");
    check(cudaStreamSynchronize(stream), "graph upload sync");
    const auto upload_stop = std::chrono::steady_clock::now();

    std::vector<double> latencies_ms;
    latencies_ms.reserve(steps);
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
      const auto step_start = std::chrono::steady_clock::now();
      check(cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch");
      check(cudaStreamSynchronize(stream), "graph replay sync");
      const auto step_stop = std::chrono::steady_clock::now();
      latencies_ms.push_back(
          std::chrono::duration<double, std::milli>(step_stop - step_start).count());
    }
    const auto stop = std::chrono::steady_clock::now();

    const double wall_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    const double capture_ms =
        std::chrono::duration<double, std::milli>(capture_stop - capture_start).count();
    const double upload_ms =
        std::chrono::duration<double, std::milli>(upload_stop - upload_start).count();
    const double mean_ms =
        std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / latencies_ms.size();
    const std::uint32_t last_token_value = *host_token;

    check(cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy");
    check(cudaGraphDestroy(graph), "cudaGraphDestroy");
    check(cudaFreeHost(host_token), "cudaFreeHost");
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");

    std::cout << "{\n";
    std::cout << "  \"path\": \"native_cuda_graph_q4_matvec_decode\",\n";
    std::cout << "  \"label\": \"" << options.label << "\",\n";
    std::cout << "  \"kernel_variant\": \"" << kernel_variant << "\",\n";
    std::cout << "  \"admitted\": true,\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"rows\": " << rows << ",\n";
    std::cout << "  \"cols\": " << cols << ",\n";
    std::cout << "  \"q4_weight_bytes\": " << weight_bytes << ",\n";
    std::cout << "  \"weight_storage_bytes\": " << weight_bytes << ",\n";
    std::cout << "  \"weight_source\": \"" << weight_source << "\",\n";
    std::cout << "  \"weights_file\": \"" << json_escape(effective_weights_file.string()) << "\",\n";
    std::cout << "  \"host_weight_checksum\": " << host_weight_checksum << ",\n";
    std::cout << "  \"q4k_blocks_file\": \"" << json_escape(options.q4k_blocks_file.string()) << "\",\n";
    std::cout << "  \"q4k_payload_file\": \"" << json_escape(options.q4k_payload_file.string()) << "\",\n";
    std::cout << "  \"q4k_meta_file\": \"" << json_escape(options.q4k_meta_file.string()) << "\",\n";
    std::cout << "  \"q4k_manifest_file\": \""
              << json_escape(options.q4k_manifest_file.string()) << "\",\n";
    std::cout << "  \"q4k_manifest_tensor\": \""
              << json_escape(options.q4k_manifest_tensor) << "\",\n";
    std::cout << "  \"q4k_source_checksum\": " << q4k_source_checksum << ",\n";
    std::cout << "  \"q4k_metadata_checksum\": " << q4k_metadata_checksum << ",\n";
    std::cout << "  \"logical_fp16_weight_bytes\": " << logical_fp16_weight_bytes << ",\n";
    std::cout << "  \"weight_compression_ratio\": "
              << (static_cast<double>(logical_fp16_weight_bytes) / weight_bytes) << ",\n";
    std::cout << "  \"wall_ms\": " << wall_ms << ",\n";
    std::cout << "  \"steps_per_second\": " << (steps * 1000.0 / wall_ms) << ",\n";
    std::cout << "  \"p50_ms\": " << percentile(latencies_ms, 50) << ",\n";
    std::cout << "  \"p95_ms\": " << percentile(latencies_ms, 95) << ",\n";
    std::cout << "  \"p99_ms\": " << percentile(latencies_ms, 99) << ",\n";
    std::cout << "  \"latency_mean_ms\": " << mean_ms << ",\n";
    std::cout << "  \"capture_ms\": " << capture_ms << ",\n";
    std::cout << "  \"graph_upload_ms\": " << upload_ms << ",\n";
    std::cout << "  \"graph_replay_rate\": 1.0,\n";
    std::cout << "  \"allocated_bytes\": " << pool.total_bytes() << ",\n";
    std::cout << "  \"setup_h2d_bytes\": " << setup_h2d_bytes << ",\n";
    std::cout << "  \"h2d_bytes\": 0,\n";
    std::cout << "  \"d2h_bytes\": " << (sizeof(std::uint32_t) * steps) << ",\n";
    std::cout << "  \"q4_values_per_step\": " << (static_cast<std::size_t>(rows) * cols) << ",\n";
    std::cout << "  \"q4_packed_bytes_per_step\": " << packed_payload_bytes << ",\n";
    std::cout << "  \"q4k_block_bytes_per_step\": " << q4k_block_bytes << ",\n";
    std::cout << "  \"q4k_predecoded_meta_bytes\": " << q4k_predecoded_meta_bytes << ",\n";
    std::cout << "  \"q4k_predecoded_meta_format\": \""
              << q4k_predecoded_meta_format << "\",\n";
    std::cout << "  \"reference_checked\": "
              << (reference.checked ? "true" : "false") << ",\n";
    if (reference.checked) {
      std::cout << "  \"reference_passed\": " << (reference.passed ? "true" : "false") << ",\n";
      std::cout << "  \"reference_tolerance\": " << reference.tolerance << ",\n";
      std::cout << "  \"max_logit_abs_error\": " << reference.max_logit_abs_error << ",\n";
      std::cout << "  \"max_logit_error_row\": " << reference.max_logit_error_row << ",\n";
      std::cout << "  \"max_logit_expected\": " << reference.max_logit_expected << ",\n";
      std::cout << "  \"max_logit_observed\": " << reference.max_logit_observed << ",\n";
      std::cout << "  \"first_logit_expected\": " << reference.first_logit_expected << ",\n";
      std::cout << "  \"first_logit_observed\": " << reference.first_logit_observed << ",\n";
      std::cout << "  \"max_kv_abs_error\": " << reference.max_kv_abs_error << ",\n";
      std::cout << "  \"max_kv_error_index\": " << reference.max_kv_error_index << ",\n";
      std::cout << "  \"logit_mismatches\": " << reference.logit_mismatches << ",\n";
      std::cout << "  \"kv_mismatches\": " << reference.kv_mismatches << ",\n";
      std::cout << "  \"token_expected\": " << reference.token_expected << ",\n";
      std::cout << "  \"token_observed\": " << reference.token_observed << ",\n";
    }
    std::cout << "  \"last_token\": " << last_token_value << "\n";
    std::cout << "}\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }
}
