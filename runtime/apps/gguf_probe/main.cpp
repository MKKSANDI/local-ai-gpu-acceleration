#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

enum class ValueType : std::uint32_t {
  UInt8 = 0,
  Int8 = 1,
  UInt16 = 2,
  Int16 = 3,
  UInt32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  UInt64 = 10,
  Int64 = 11,
  Float64 = 12,
};

struct BlockFormat {
  std::string_view name;
  std::uint64_t block_size;
  std::uint64_t type_size;
  std::string_view family;
};

struct TensorInfo {
  std::string name;
  std::vector<std::uint64_t> dims;
  std::uint32_t type_id = 0;
  std::uint64_t offset = 0;
  std::uint64_t span_bytes = 0;
};

struct Options {
  std::filesystem::path model_path;
  std::filesystem::path dump_q4k_payload_path;
  std::filesystem::path dump_q4k_blocks_path;
  std::string q4k_tensor_name;
};

struct Q4KSmoke {
  bool present = false;
  bool valid = false;
  std::string tensor_name;
  std::vector<std::uint64_t> dims;
  std::uint64_t tensor_offset = 0;
  std::uint64_t absolute_offset = 0;
  std::uint64_t span_bytes = 0;
  std::uint64_t block_count = 0;
  std::uint64_t bytes_read = 0;
  std::uint64_t q4_payload_bytes = 0;
  std::uint64_t q4_values = 0;
  std::uint64_t scale_checksum = 0;
  std::uint64_t packed_q4_checksum = 0;
  bool full_payload_written = false;
  std::filesystem::path full_payload_path;
  std::uint64_t full_payload_bytes = 0;
  std::uint64_t full_payload_checksum = 0;
  bool full_blocks_written = false;
  std::filesystem::path full_blocks_path;
  std::uint64_t full_blocks_bytes = 0;
  std::uint64_t full_blocks_checksum = 0;
  std::uint16_t d_bits = 0;
  std::uint16_t dmin_bits = 0;
  float d = 0.0F;
  float dmin = 0.0F;
  std::uint8_t q_min = 0;
  std::uint8_t q_max = 0;
  bool dequant_valid = false;
  double dequant_sum = 0.0;
  float dequant_min = 0.0F;
  float dequant_max = 0.0F;
  float dequant_absmax = 0.0F;
  std::uint64_t decoded_scale_checksum = 0;
  std::uint64_t decoded_min_checksum = 0;
  std::vector<std::uint8_t> decoded_scales;
  std::vector<std::uint8_t> decoded_mins;
  std::vector<std::uint8_t> first_packed_bytes;
  std::vector<float> first_dequant_values;
  std::string reason;
};

struct ProbeResult {
  std::filesystem::path model_path;
  std::uint32_t version = 0;
  std::uint64_t tensor_count = 0;
  std::uint64_t metadata_count = 0;
  std::uint64_t tensor_info_end_offset = 0;
  std::uint64_t tensor_data_offset = 0;
  std::uint64_t tensor_data_alignment = 32;
  std::uint64_t supported_tensors = 0;
  std::uint64_t valid_tensors = 0;
  std::uint64_t invalid_tensors = 0;
  std::uint64_t expected_data_bytes = 0;
  std::uint64_t padding_bytes = 0;
  std::uint64_t max_padding_bytes = 0;
  std::string architecture;
  std::map<std::string, std::uint64_t> type_counts;
  std::map<std::string, std::uint64_t> family_counts;
  std::vector<std::string> invalid_reasons;
  Q4KSmoke q4k_smoke;
};

template <typename T>
T read_pod(std::istream& in) {
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!in) {
    throw std::runtime_error("unexpected end of GGUF file");
  }
  return value;
}

std::string read_string(std::istream& in) {
  const auto size = read_pod<std::uint64_t>(in);
  std::string value(size, '\0');
  if (size > 0) {
    in.read(value.data(), static_cast<std::streamsize>(size));
    if (!in) {
      throw std::runtime_error("unexpected end of GGUF string");
    }
  }
  return value;
}

void skip_bytes(std::istream& in, std::uint64_t count) {
  in.seekg(static_cast<std::streamoff>(count), std::ios::cur);
  if (!in) {
    throw std::runtime_error("unexpected end while skipping GGUF value");
  }
}

void skip_value(std::istream& in, std::uint32_t value_type);

void skip_array(std::istream& in) {
  const auto item_type = read_pod<std::uint32_t>(in);
  const auto length = read_pod<std::uint64_t>(in);
  for (std::uint64_t i = 0; i < length; ++i) {
    skip_value(in, item_type);
  }
}

void skip_value(std::istream& in, std::uint32_t value_type) {
  switch (static_cast<ValueType>(value_type)) {
    case ValueType::UInt8:
    case ValueType::Int8:
    case ValueType::Bool:
      skip_bytes(in, 1);
      break;
    case ValueType::UInt16:
    case ValueType::Int16:
      skip_bytes(in, 2);
      break;
    case ValueType::UInt32:
    case ValueType::Int32:
    case ValueType::Float32:
      skip_bytes(in, 4);
      break;
    case ValueType::UInt64:
    case ValueType::Int64:
    case ValueType::Float64:
      skip_bytes(in, 8);
      break;
    case ValueType::String:
      (void)read_string(in);
      break;
    case ValueType::Array:
      skip_array(in);
      break;
    default:
      throw std::runtime_error("unsupported GGUF metadata value type");
  }
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
  if (alignment == 0) {
    return value;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

const std::unordered_map<std::uint32_t, BlockFormat>& block_formats() {
  static const std::unordered_map<std::uint32_t, BlockFormat> formats = {
      {0, {"F32", 1, 4, "float"}},
      {1, {"F16", 1, 2, "float"}},
      {8, {"Q8_0", 32, 34, "legacy_quant"}},
      {12, {"Q4_K", 256, 144, "k_quant"}},
      {13, {"Q5_K", 256, 176, "k_quant"}},
      {21, {"IQ3_S", 256, 110, "iq_quant"}},
      {22, {"IQ2_S", 256, 82, "iq_quant"}},
  };
  return formats;
}

std::string type_name(std::uint32_t type_id) {
  const auto& formats = block_formats();
  const auto found = formats.find(type_id);
  if (found != formats.end()) {
    return std::string(found->second.name);
  }
  return "unknown:" + std::to_string(type_id);
}

std::uint64_t element_count(const std::vector<std::uint64_t>& dims) {
  std::uint64_t total = 1;
  for (const auto dim : dims) {
    total *= dim;
  }
  return total;
}

std::uint64_t gguf_file_size(const std::filesystem::path& path) {
  return static_cast<std::uint64_t>(std::filesystem::file_size(path));
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
      options.model_path = argv[++i];
    } else if (arg == "--dump-q4k-payload" && i + 1 < argc) {
      options.dump_q4k_payload_path = argv[++i];
    } else if (arg == "--dump-q4k-blocks" && i + 1 < argc) {
      options.dump_q4k_blocks_path = argv[++i];
    } else if (arg == "--q4k-tensor" && i + 1 < argc) {
      options.q4k_tensor_name = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "runtime_gguf_probe --model <path-to.gguf> [--q4k-tensor NAME]\n"
          << "                   [--dump-q4k-payload <path.bin>]\n"
          << "                   [--dump-q4k-blocks <path.bin>]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown or incomplete argument: " + std::string(arg));
    }
  }
  if (options.model_path.empty()) {
    throw std::invalid_argument("usage: runtime_gguf_probe --model <path-to.gguf>");
  }
  return options;
}

void validate_tensor(const TensorInfo& tensor, ProbeResult& result) {
  const auto type = type_name(tensor.type_id);
  result.type_counts[type] += 1;
  const auto& formats = block_formats();
  const auto found = formats.find(tensor.type_id);
  if (found == formats.end()) {
    result.invalid_tensors += 1;
    if (result.invalid_reasons.size() < 16) {
      result.invalid_reasons.push_back(tensor.name + ": unsupported type " + type);
    }
    return;
  }

  const auto& format = found->second;
  result.supported_tensors += 1;
  result.family_counts[std::string(format.family)] += 1;
  const auto elements = element_count(tensor.dims);
  if (elements % format.block_size != 0) {
    result.invalid_tensors += 1;
    if (result.invalid_reasons.size() < 16) {
      result.invalid_reasons.push_back(tensor.name + ": element count is not block aligned");
    }
    return;
  }
  const auto expected = (elements / format.block_size) * format.type_size;
  result.expected_data_bytes += expected;
  if (tensor.span_bytes < expected) {
    result.invalid_tensors += 1;
    if (result.invalid_reasons.size() < 16) {
      result.invalid_reasons.push_back(tensor.name + ": span is smaller than expected bytes");
    }
    return;
  }
  const auto padding = tensor.span_bytes - expected;
  result.padding_bytes += padding;
  result.max_padding_bytes = std::max(result.max_padding_bytes, padding);
  if (padding >= result.tensor_data_alignment) {
    result.invalid_tensors += 1;
    if (result.invalid_reasons.size() < 16) {
      result.invalid_reasons.push_back(tensor.name + ": padding exceeds tensor alignment");
    }
    return;
  }
  result.valid_tensors += 1;
}

float fp16_to_float(std::uint16_t bits) {
  const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
  const std::uint32_t exp_bits = (bits >> 10U) & 0x1FU;
  std::uint32_t mant = bits & 0x03FFU;

  std::uint32_t out = sign;
  if (exp_bits == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      int exp = -14;
      while ((mant & 0x0400U) == 0) {
        mant <<= 1U;
        --exp;
      }
      mant &= 0x03FFU;
      out |= (static_cast<std::uint32_t>(exp + 127) << 23U) | (mant << 13U);
    }
  } else if (exp_bits == 0x1FU) {
    out |= 0x7F800000U | (mant << 13U);
  } else {
    out |= ((exp_bits + (127U - 15U)) << 23U) | (mant << 13U);
  }

  float value = 0.0F;
  static_assert(sizeof(value) == sizeof(out));
  std::memcpy(&value, &out, sizeof(value));
  return value;
}

void get_scale_min_k4(
    int index,
    const std::uint8_t* packed,
    std::uint8_t& scale,
    std::uint8_t& min_value) {
  if (index < 4) {
    scale = static_cast<std::uint8_t>(packed[index] & 63U);
    min_value = static_cast<std::uint8_t>(packed[index + 4] & 63U);
    return;
  }
  scale = static_cast<std::uint8_t>(
      (packed[index + 4] & 0x0FU) | ((packed[index - 4] >> 6U) << 4U));
  min_value = static_cast<std::uint8_t>(
      (packed[index + 4] >> 4U) | ((packed[index] >> 6U) << 4U));
}

void decode_q4k_first_block(const std::vector<std::uint8_t>& block, Q4KSmoke& smoke) {
  if (block.size() < 144) {
    return;
  }
  smoke.d = fp16_to_float(smoke.d_bits);
  smoke.dmin = fp16_to_float(smoke.dmin_bits);
  if (!std::isfinite(smoke.d) || !std::isfinite(smoke.dmin)) {
    return;
  }

  const auto* scales = block.data() + 4;
  const auto* q = block.data() + 16;
  smoke.dequant_min = std::numeric_limits<float>::infinity();
  smoke.dequant_max = -std::numeric_limits<float>::infinity();
  smoke.dequant_absmax = 0.0F;

  for (int group = 0; group < 4; ++group) {
    const int scale_index = group * 2;
    std::uint8_t scale_0 = 0;
    std::uint8_t min_0 = 0;
    std::uint8_t scale_1 = 0;
    std::uint8_t min_1 = 0;
    get_scale_min_k4(scale_index, scales, scale_0, min_0);
    get_scale_min_k4(scale_index + 1, scales, scale_1, min_1);
    smoke.decoded_scales.push_back(scale_0);
    smoke.decoded_scales.push_back(scale_1);
    smoke.decoded_mins.push_back(min_0);
    smoke.decoded_mins.push_back(min_1);
    smoke.decoded_scale_checksum += scale_0 + scale_1;
    smoke.decoded_min_checksum += min_0 + min_1;

    const float d0 = smoke.d * static_cast<float>(scale_0);
    const float m0 = smoke.dmin * static_cast<float>(min_0);
    const float d1 = smoke.d * static_cast<float>(scale_1);
    const float m1 = smoke.dmin * static_cast<float>(min_1);
    const auto* group_q = q + static_cast<std::ptrdiff_t>(group) * 32;
    for (int i = 0; i < 32; ++i) {
      const float value = d0 * static_cast<float>(group_q[i] & 0x0FU) - m0;
      smoke.dequant_sum += value;
      smoke.dequant_min = std::min(smoke.dequant_min, value);
      smoke.dequant_max = std::max(smoke.dequant_max, value);
      smoke.dequant_absmax =
          std::max(smoke.dequant_absmax, static_cast<float>(std::fabs(value)));
      if (smoke.first_dequant_values.size() < 16) {
        smoke.first_dequant_values.push_back(value);
      }
    }
    for (int i = 0; i < 32; ++i) {
      const float value = d1 * static_cast<float>(group_q[i] >> 4U) - m1;
      smoke.dequant_sum += value;
      smoke.dequant_min = std::min(smoke.dequant_min, value);
      smoke.dequant_max = std::max(smoke.dequant_max, value);
      smoke.dequant_absmax =
          std::max(smoke.dequant_absmax, static_cast<float>(std::fabs(value)));
      if (smoke.first_dequant_values.size() < 16) {
        smoke.first_dequant_values.push_back(value);
      }
    }
  }
  smoke.dequant_valid = true;
}

Q4KSmoke read_q4k_smoke(
    std::ifstream& in,
    const std::filesystem::path& path,
    const TensorInfo& tensor,
    std::uint64_t tensor_data_offset,
    const std::filesystem::path& payload_dump_path,
    const std::filesystem::path& blocks_dump_path) {
  (void)path;
  Q4KSmoke smoke;
  smoke.present = true;
  smoke.tensor_name = tensor.name;
  smoke.dims = tensor.dims;
  smoke.tensor_offset = tensor.offset;
  smoke.absolute_offset = tensor_data_offset + tensor.offset;
  smoke.span_bytes = tensor.span_bytes;
  const auto& format = block_formats().at(12);
  smoke.block_count = tensor.span_bytes / format.type_size;
  if (tensor.span_bytes < format.type_size) {
    smoke.reason = "tensor span is smaller than one Q4_K block";
    return smoke;
  }

  std::vector<std::uint8_t> block(static_cast<std::size_t>(format.type_size));
  in.clear();
  in.seekg(static_cast<std::streamoff>(smoke.absolute_offset), std::ios::beg);
  in.read(reinterpret_cast<char*>(block.data()), static_cast<std::streamsize>(block.size()));
  if (!in) {
    smoke.reason = "failed to read first Q4_K block";
    return smoke;
  }

  smoke.bytes_read = block.size();
  smoke.d_bits = static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8));
  smoke.dmin_bits =
      static_cast<std::uint16_t>(block[2] | (static_cast<std::uint16_t>(block[3]) << 8));
  for (std::size_t i = 4; i < 16; ++i) {
    smoke.scale_checksum += block[i];
  }
  smoke.q4_payload_bytes = 128;
  smoke.q4_values = 256;
  smoke.q_min = 15;
  smoke.q_max = 0;
  for (std::size_t i = 16; i < block.size(); ++i) {
    const auto byte = block[i];
    smoke.packed_q4_checksum += byte;
    const auto lo = static_cast<std::uint8_t>(byte & 0x0F);
    const auto hi = static_cast<std::uint8_t>(byte >> 4);
    smoke.q_min = std::min(smoke.q_min, lo);
    smoke.q_min = std::min(smoke.q_min, hi);
    smoke.q_max = std::max(smoke.q_max, lo);
    smoke.q_max = std::max(smoke.q_max, hi);
    if (smoke.first_packed_bytes.size() < 16) {
      smoke.first_packed_bytes.push_back(byte);
    }
  }
  decode_q4k_first_block(block, smoke);
  if (!payload_dump_path.empty() || !blocks_dump_path.empty()) {
    std::ofstream payload_out;
    if (payload_dump_path.has_parent_path()) {
      std::filesystem::create_directories(payload_dump_path.parent_path());
    }
    if (!payload_dump_path.empty()) {
      payload_out.open(payload_dump_path, std::ios::binary);
      if (!payload_out) {
        smoke.reason = "failed to open Q4_K payload dump path";
        smoke.valid = false;
        return smoke;
      }
    }
    std::ofstream blocks_out;
    if (blocks_dump_path.has_parent_path()) {
      std::filesystem::create_directories(blocks_dump_path.parent_path());
    }
    if (!blocks_dump_path.empty()) {
      blocks_out.open(blocks_dump_path, std::ios::binary);
      if (!blocks_out) {
        smoke.reason = "failed to open Q4_K block dump path";
        smoke.valid = false;
        return smoke;
      }
    }
    in.clear();
    in.seekg(static_cast<std::streamoff>(smoke.absolute_offset), std::ios::beg);
    std::vector<std::uint8_t> full_block(static_cast<std::size_t>(format.type_size));
    for (std::uint64_t block_index = 0; block_index < smoke.block_count; ++block_index) {
      in.read(
          reinterpret_cast<char*>(full_block.data()),
          static_cast<std::streamsize>(full_block.size()));
      if (!in) {
        smoke.reason = "failed while reading full Q4_K payload";
        smoke.valid = false;
        return smoke;
      }
      if (payload_out) {
        payload_out.write(
            reinterpret_cast<const char*>(full_block.data() + 16),
            static_cast<std::streamsize>(smoke.q4_payload_bytes));
        if (!payload_out) {
          smoke.reason = "failed while writing Q4_K payload";
          smoke.valid = false;
          return smoke;
        }
      }
      if (blocks_out) {
        blocks_out.write(
            reinterpret_cast<const char*>(full_block.data()),
            static_cast<std::streamsize>(full_block.size()));
        if (!blocks_out) {
          smoke.reason = "failed while writing Q4_K blocks";
          smoke.valid = false;
          return smoke;
        }
      }
      for (std::size_t i = 16; i < full_block.size(); ++i) {
        smoke.full_payload_checksum += full_block[i];
      }
      for (const auto byte : full_block) {
        smoke.full_blocks_checksum += byte;
      }
      smoke.full_payload_bytes += smoke.q4_payload_bytes;
      smoke.full_blocks_bytes += full_block.size();
    }
    if (payload_out) {
      smoke.full_payload_written = true;
      smoke.full_payload_path = payload_dump_path;
    }
    if (blocks_out) {
      smoke.full_blocks_written = true;
      smoke.full_blocks_path = blocks_dump_path;
    }
  }
  smoke.valid = true;
  smoke.reason = "ok";
  return smoke;
}

ProbeResult probe_gguf(const Options& options) {
  const auto& path = options.model_path;
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open GGUF file");
  }

  char magic[4]{};
  in.read(magic, sizeof(magic));
  if (std::string_view(magic, sizeof(magic)) != "GGUF") {
    throw std::runtime_error("not a GGUF file");
  }

  ProbeResult result;
  result.model_path = path;
  result.version = read_pod<std::uint32_t>(in);
  result.tensor_count = read_pod<std::uint64_t>(in);
  result.metadata_count = read_pod<std::uint64_t>(in);

  for (std::uint64_t i = 0; i < result.metadata_count; ++i) {
    const auto key = read_string(in);
    const auto value_type = read_pod<std::uint32_t>(in);
    if (key == "general.alignment" && value_type == static_cast<std::uint32_t>(ValueType::UInt32)) {
      result.tensor_data_alignment = read_pod<std::uint32_t>(in);
    } else if (key == "general.architecture" &&
               value_type == static_cast<std::uint32_t>(ValueType::String)) {
      result.architecture = read_string(in);
    } else {
      skip_value(in, value_type);
    }
  }

  std::vector<TensorInfo> tensors;
  tensors.reserve(static_cast<std::size_t>(result.tensor_count));
  for (std::uint64_t i = 0; i < result.tensor_count; ++i) {
    TensorInfo tensor;
    tensor.name = read_string(in);
    const auto dim_count = read_pod<std::uint32_t>(in);
    tensor.dims.reserve(dim_count);
    for (std::uint32_t d = 0; d < dim_count; ++d) {
      tensor.dims.push_back(read_pod<std::uint64_t>(in));
    }
    tensor.type_id = read_pod<std::uint32_t>(in);
    tensor.offset = read_pod<std::uint64_t>(in);
    tensors.push_back(std::move(tensor));
  }

  result.tensor_info_end_offset = static_cast<std::uint64_t>(in.tellg());
  result.tensor_data_offset = align_up(result.tensor_info_end_offset, result.tensor_data_alignment);

  std::sort(tensors.begin(), tensors.end(), [](const TensorInfo& a, const TensorInfo& b) {
    return a.offset < b.offset;
  });
  const auto size = gguf_file_size(path);
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto next = (i + 1 < tensors.size())
                          ? tensors[i + 1].offset
                          : size - result.tensor_data_offset;
    tensors[i].span_bytes = next - tensors[i].offset;
    validate_tensor(tensors[i], result);
  }
  const auto q4k = std::find_if(tensors.begin(), tensors.end(), [&](const TensorInfo& tensor) {
    if (tensor.type_id != 12) {
      return false;
    }
    return options.q4k_tensor_name.empty() || tensor.name == options.q4k_tensor_name;
  });
  if (q4k != tensors.end()) {
    result.q4k_smoke = read_q4k_smoke(
        in,
        path,
        *q4k,
        result.tensor_data_offset,
        options.dump_q4k_payload_path,
        options.dump_q4k_blocks_path);
  }

  return result;
}

void print_json_map(const std::map<std::string, std::uint64_t>& values, int indent) {
  std::cout << "{";
  if (!values.empty()) {
    std::cout << "\n";
  }
  std::size_t index = 0;
  for (const auto& [key, value] : values) {
    std::cout << std::string(indent + 2, ' ') << "\"" << key << "\": " << value;
    if (++index < values.size()) {
      std::cout << ",";
    }
    std::cout << "\n";
  }
  if (!values.empty()) {
    std::cout << std::string(indent, ' ');
  }
  std::cout << "}";
}

void print_json_u8_array(const std::vector<std::uint8_t>& values, int indent) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    if (i == 0) {
      std::cout << "\n" << std::string(indent + 2, ' ');
    } else if (i % 16 == 0) {
      std::cout << "\n" << std::string(indent + 2, ' ');
    }
    std::cout << static_cast<unsigned>(values[i]);
  }
  if (!values.empty()) {
    std::cout << "\n" << std::string(indent, ' ');
  }
  std::cout << "]";
}

void print_json_u64_array(const std::vector<std::uint64_t>& values, int indent) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    if (i == 0) {
      std::cout << "\n" << std::string(indent + 2, ' ');
    }
    std::cout << values[i];
  }
  if (!values.empty()) {
    std::cout << "\n" << std::string(indent, ' ');
  }
  std::cout << "]";
}

void print_json_float_array(const std::vector<float>& values, int indent) {
  std::cout << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    if (i == 0) {
      std::cout << "\n" << std::string(indent + 2, ' ');
    } else if (i % 8 == 0) {
      std::cout << "\n" << std::string(indent + 2, ' ');
    }
    std::cout << values[i];
  }
  if (!values.empty()) {
    std::cout << "\n" << std::string(indent, ' ');
  }
  std::cout << "]";
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

void print_json(const ProbeResult& result) {
  std::cout << "{\n";
  std::cout << "  \"path\": \"native_gguf_probe\",\n";
  std::cout << "  \"model_path\": \"" << json_escape(result.model_path.string()) << "\",\n";
  std::cout << "  \"version\": " << result.version << ",\n";
  std::cout << "  \"architecture\": \"" << json_escape(result.architecture) << "\",\n";
  std::cout << "  \"tensor_count\": " << result.tensor_count << ",\n";
  std::cout << "  \"metadata_count\": " << result.metadata_count << ",\n";
  std::cout << "  \"tensor_info_end_offset\": " << result.tensor_info_end_offset << ",\n";
  std::cout << "  \"tensor_data_offset\": " << result.tensor_data_offset << ",\n";
  std::cout << "  \"tensor_data_alignment\": " << result.tensor_data_alignment << ",\n";
  std::cout << "  \"supported_tensors\": " << result.supported_tensors << ",\n";
  std::cout << "  \"valid_tensors\": " << result.valid_tensors << ",\n";
  std::cout << "  \"invalid_tensors\": " << result.invalid_tensors << ",\n";
  std::cout << "  \"expected_data_bytes\": " << result.expected_data_bytes << ",\n";
  std::cout << "  \"padding_bytes\": " << result.padding_bytes << ",\n";
  std::cout << "  \"max_padding_bytes\": " << result.max_padding_bytes << ",\n";
  std::cout << "  \"type_counts\": ";
  print_json_map(result.type_counts, 2);
  std::cout << ",\n";
  std::cout << "  \"family_counts\": ";
  print_json_map(result.family_counts, 2);
  std::cout << ",\n";
  std::cout << "  \"q4k_smoke\": {\n";
  std::cout << "    \"present\": " << (result.q4k_smoke.present ? "true" : "false") << ",\n";
  std::cout << "    \"valid\": " << (result.q4k_smoke.valid ? "true" : "false") << ",\n";
  std::cout << "    \"tensor_name\": \"" << json_escape(result.q4k_smoke.tensor_name) << "\",\n";
  std::cout << "    \"dimensions\": ";
  print_json_u64_array(result.q4k_smoke.dims, 4);
  std::cout << ",\n";
  std::cout << "    \"tensor_offset\": " << result.q4k_smoke.tensor_offset << ",\n";
  std::cout << "    \"absolute_offset\": " << result.q4k_smoke.absolute_offset << ",\n";
  std::cout << "    \"span_bytes\": " << result.q4k_smoke.span_bytes << ",\n";
  std::cout << "    \"block_count\": " << result.q4k_smoke.block_count << ",\n";
  std::cout << "    \"bytes_read\": " << result.q4k_smoke.bytes_read << ",\n";
  std::cout << "    \"q4_payload_bytes\": " << result.q4k_smoke.q4_payload_bytes << ",\n";
  std::cout << "    \"q4_values\": " << result.q4k_smoke.q4_values << ",\n";
  std::cout << "    \"scale_checksum\": " << result.q4k_smoke.scale_checksum << ",\n";
  std::cout << "    \"packed_q4_checksum\": " << result.q4k_smoke.packed_q4_checksum << ",\n";
  std::cout << "    \"full_payload_written\": "
            << (result.q4k_smoke.full_payload_written ? "true" : "false") << ",\n";
  std::cout << "    \"full_payload_path\": \""
            << json_escape(result.q4k_smoke.full_payload_path.string()) << "\",\n";
  std::cout << "    \"full_payload_bytes\": " << result.q4k_smoke.full_payload_bytes << ",\n";
  std::cout << "    \"full_payload_checksum\": " << result.q4k_smoke.full_payload_checksum << ",\n";
  std::cout << "    \"full_blocks_written\": "
            << (result.q4k_smoke.full_blocks_written ? "true" : "false") << ",\n";
  std::cout << "    \"full_blocks_path\": \""
            << json_escape(result.q4k_smoke.full_blocks_path.string()) << "\",\n";
  std::cout << "    \"full_blocks_bytes\": " << result.q4k_smoke.full_blocks_bytes << ",\n";
  std::cout << "    \"full_blocks_checksum\": " << result.q4k_smoke.full_blocks_checksum << ",\n";
  std::cout << "    \"d_bits\": " << result.q4k_smoke.d_bits << ",\n";
  std::cout << "    \"dmin_bits\": " << result.q4k_smoke.dmin_bits << ",\n";
  std::cout << "    \"d\": " << result.q4k_smoke.d << ",\n";
  std::cout << "    \"dmin\": " << result.q4k_smoke.dmin << ",\n";
  std::cout << "    \"q_min\": " << static_cast<unsigned>(result.q4k_smoke.q_min) << ",\n";
  std::cout << "    \"q_max\": " << static_cast<unsigned>(result.q4k_smoke.q_max) << ",\n";
  std::cout << "    \"dequant_valid\": "
            << (result.q4k_smoke.dequant_valid ? "true" : "false") << ",\n";
  std::cout << "    \"dequant_sum\": " << result.q4k_smoke.dequant_sum << ",\n";
  std::cout << "    \"dequant_min\": " << result.q4k_smoke.dequant_min << ",\n";
  std::cout << "    \"dequant_max\": " << result.q4k_smoke.dequant_max << ",\n";
  std::cout << "    \"dequant_absmax\": " << result.q4k_smoke.dequant_absmax << ",\n";
  std::cout << "    \"decoded_scale_checksum\": "
            << result.q4k_smoke.decoded_scale_checksum << ",\n";
  std::cout << "    \"decoded_min_checksum\": " << result.q4k_smoke.decoded_min_checksum << ",\n";
  std::cout << "    \"decoded_scales\": ";
  print_json_u8_array(result.q4k_smoke.decoded_scales, 4);
  std::cout << ",\n";
  std::cout << "    \"decoded_mins\": ";
  print_json_u8_array(result.q4k_smoke.decoded_mins, 4);
  std::cout << ",\n";
  std::cout << "    \"first_packed_bytes\": ";
  print_json_u8_array(result.q4k_smoke.first_packed_bytes, 4);
  std::cout << ",\n";
  std::cout << "    \"first_dequant_values\": ";
  print_json_float_array(result.q4k_smoke.first_dequant_values, 4);
  std::cout << ",\n";
  std::cout << "    \"reason\": \"" << json_escape(result.q4k_smoke.reason) << "\"\n";
  std::cout << "  },\n";
  std::cout << "  \"invalid_reasons\": [";
  if (!result.invalid_reasons.empty()) {
    std::cout << "\n";
  }
  for (std::size_t i = 0; i < result.invalid_reasons.size(); ++i) {
    std::cout << "    \"" << json_escape(result.invalid_reasons[i]) << "\"";
    if (i + 1 < result.invalid_reasons.size()) {
      std::cout << ",";
    }
    std::cout << "\n";
  }
  if (!result.invalid_reasons.empty()) {
    std::cout << "  ";
  }
  std::cout << "]\n";
  std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    const auto result = probe_gguf(options);
    print_json(result);
    return result.invalid_tensors == 0 ? 0 : 2;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }
}
