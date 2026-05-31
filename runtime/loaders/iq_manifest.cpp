#include "runtime/loaders/iq_manifest.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace rtxllm {
namespace {

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open IQ manifest: " + path.string());
  }
  in.seekg(0, std::ios::end);
  const auto size = static_cast<std::size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  std::string text(size, '\0');
  if (size > 0) {
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!in) {
      throw std::runtime_error("failed to read IQ manifest: " + path.string());
    }
  }
  return text;
}

std::size_t skip_space(std::string_view text, std::size_t pos) {
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
    ++pos;
  }
  return pos;
}

std::size_t find_key(std::string_view text, std::string_view key) {
  const std::string needle = "\"" + std::string(key) + "\"";
  const auto pos = text.find(needle);
  if (pos == std::string_view::npos) {
    throw std::runtime_error("IQ manifest field missing: " + std::string(key));
  }
  const auto colon = text.find(':', pos + needle.size());
  if (colon == std::string_view::npos) {
    throw std::runtime_error("IQ manifest field has no value: " + std::string(key));
  }
  return colon + 1;
}

std::string parse_string_at(std::string_view text, std::size_t pos) {
  pos = skip_space(text, pos);
  if (pos >= text.size() || text[pos] != '"') {
    throw std::runtime_error("IQ manifest string value expected");
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
      throw std::runtime_error("IQ manifest string has trailing escape");
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
        throw std::runtime_error("IQ manifest string contains unsupported escape");
    }
  }
  throw std::runtime_error("IQ manifest string is unterminated");
}

std::size_t end_of_string_at(std::string_view text, std::size_t pos) {
  pos = skip_space(text, pos);
  if (pos >= text.size() || text[pos] != '"') {
    throw std::runtime_error("IQ manifest string value expected");
  }
  ++pos;
  bool escape = false;
  while (pos < text.size()) {
    const char ch = text[pos++];
    if (escape) {
      escape = false;
      continue;
    }
    if (ch == '\\') {
      escape = true;
      continue;
    }
    if (ch == '"') {
      return pos;
    }
  }
  throw std::runtime_error("IQ manifest string is unterminated");
}

std::string string_field(std::string_view object, std::string_view key) {
  return parse_string_at(object, find_key(object, key));
}

std::string optional_string_field(std::string_view object, std::string_view key) {
  const std::string needle = "\"" + std::string(key) + "\"";
  const auto pos = object.find(needle);
  if (pos == std::string_view::npos) {
    return {};
  }
  const auto colon = object.find(':', pos + needle.size());
  if (colon == std::string_view::npos) {
    return {};
  }
  return parse_string_at(object, colon + 1);
}

std::uint64_t optional_u64_field(std::string_view object, std::string_view key) {
  const std::string needle = "\"" + std::string(key) + "\"";
  const auto pos = object.find(needle);
  if (pos == std::string_view::npos) {
    return 0;
  }
  const auto colon = object.find(':', pos + needle.size());
  if (colon == std::string_view::npos) {
    return 0;
  }
  std::size_t value_pos = skip_space(object, colon + 1);
  std::uint64_t value = 0;
  bool saw_digit = false;
  while (value_pos < object.size() &&
         std::isdigit(static_cast<unsigned char>(object[value_pos])) != 0) {
    saw_digit = true;
    value = value * 10u + static_cast<std::uint64_t>(object[value_pos] - '0');
    ++value_pos;
  }
  return saw_digit ? value : 0;
}

std::uint64_t u64_field(std::string_view object, std::string_view key) {
  const auto value = optional_u64_field(object, key);
  if (value == 0) {
    throw std::runtime_error("IQ manifest integer value expected: " + std::string(key));
  }
  return value;
}

std::vector<std::uint64_t> optional_u64_array_field(
    std::string_view object,
    std::string_view key) {
  const std::string needle = "\"" + std::string(key) + "\"";
  const auto pos = object.find(needle);
  if (pos == std::string_view::npos) {
    return {};
  }
  const auto colon = object.find(':', pos + needle.size());
  if (colon == std::string_view::npos) {
    return {};
  }
  std::size_t value_pos = skip_space(object, colon + 1);
  if (value_pos >= object.size() || object[value_pos] != '[') {
    throw std::runtime_error("IQ manifest array value expected: " + std::string(key));
  }
  ++value_pos;
  std::vector<std::uint64_t> values;
  while (value_pos < object.size()) {
    value_pos = skip_space(object, value_pos);
    if (value_pos < object.size() && object[value_pos] == ']') {
      return values;
    }
    std::uint64_t value = 0;
    bool saw_digit = false;
    while (value_pos < object.size() &&
           std::isdigit(static_cast<unsigned char>(object[value_pos])) != 0) {
      saw_digit = true;
      value = value * 10u + static_cast<std::uint64_t>(object[value_pos] - '0');
      ++value_pos;
    }
    if (!saw_digit) {
      throw std::runtime_error("IQ manifest array integer expected: " + std::string(key));
    }
    values.push_back(value);
    value_pos = skip_space(object, value_pos);
    if (value_pos < object.size() && object[value_pos] == ',') {
      ++value_pos;
      continue;
    }
    if (value_pos < object.size() && object[value_pos] == ']') {
      return values;
    }
    throw std::runtime_error("IQ manifest array delimiter expected: " + std::string(key));
  }
  throw std::runtime_error("IQ manifest array is unterminated: " + std::string(key));
}

std::vector<std::string> optional_string_array_field(
    std::string_view object,
    std::string_view key) {
  const std::string needle = "\"" + std::string(key) + "\"";
  const auto pos = object.find(needle);
  if (pos == std::string_view::npos) {
    return {};
  }
  const auto colon = object.find(':', pos + needle.size());
  if (colon == std::string_view::npos) {
    return {};
  }
  std::size_t value_pos = skip_space(object, colon + 1);
  if (value_pos >= object.size() || object[value_pos] != '[') {
    throw std::runtime_error("IQ manifest string array expected: " + std::string(key));
  }
  ++value_pos;
  std::vector<std::string> values;
  while (value_pos < object.size()) {
    value_pos = skip_space(object, value_pos);
    if (value_pos < object.size() && object[value_pos] == ']') {
      return values;
    }
    values.push_back(parse_string_at(object, value_pos));
    value_pos = end_of_string_at(object, value_pos);
    value_pos = skip_space(object, value_pos);
    if (value_pos < object.size() && object[value_pos] == ',') {
      ++value_pos;
      continue;
    }
    if (value_pos < object.size() && object[value_pos] == ']') {
      return values;
    }
    throw std::runtime_error("IQ manifest string array delimiter expected: " + std::string(key));
  }
  throw std::runtime_error("IQ manifest string array is unterminated: " + std::string(key));
}

std::vector<std::string_view> tensor_objects(std::string_view text) {
  const auto tensors_key = text.find("\"tensors\"");
  if (tensors_key == std::string_view::npos) {
    throw std::runtime_error("IQ manifest does not contain tensors array");
  }
  const auto array_start = text.find('[', tensors_key);
  if (array_start == std::string_view::npos) {
    throw std::runtime_error("IQ manifest tensors field is not an array");
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

struct TensorPlanEntry {
  std::string name;
  std::string type;
  std::vector<std::uint64_t> dimensions;
  std::uint64_t span_bytes = 0;
  std::uint64_t absolute_offset = 0;
};

std::string dimensions_to_string(const std::vector<std::uint64_t>& dimensions) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t index = 0; index < dimensions.size(); ++index) {
    if (index != 0) {
      oss << ", ";
    }
    oss << dimensions[index];
  }
  oss << "]";
  return oss.str();
}

std::map<std::string, TensorPlanEntry> load_tensor_plan_entries(
    const std::filesystem::path& path) {
  const std::string text = read_text_file(path);
  std::map<std::string, TensorPlanEntry> entries;
  for (const auto object : tensor_objects(text)) {
    TensorPlanEntry entry;
    entry.name = string_field(object, "name");
    entry.type = string_field(object, "type");
    entry.dimensions = optional_u64_array_field(object, "dimensions");
    entry.span_bytes = u64_field(object, "span_bytes");
    entry.absolute_offset = optional_u64_field(object, "absolute_offset");
    const auto name = entry.name;
    const auto [_, inserted] = entries.emplace(entry.name, std::move(entry));
    if (!inserted) {
      throw std::runtime_error("tensor plan contains duplicate tensor: " + name);
    }
  }
  if (entries.empty()) {
    throw std::runtime_error("tensor plan contains no tensor entries: " + path.string());
  }
  return entries;
}

}  // namespace

IQManifest load_iq_manifest(const std::filesystem::path& path) {
  const std::string text = read_text_file(path);
  IQManifest manifest;
  manifest.path = path;
  manifest.model = optional_string_field(text, "model");
  manifest.tensor_plan = optional_string_field(text, "tensor_plan");
  manifest.generated_at = optional_string_field(text, "generated_at");
  manifest.types = optional_string_array_field(text, "types");
  manifest.tensor_count = optional_u64_field(text, "tensor_count");
  manifest.source_bytes = optional_u64_field(text, "source_bytes");
  manifest.payload_bytes = optional_u64_field(text, "payload_bytes");
  manifest.resident_bytes = optional_u64_field(text, "resident_bytes");

  for (const auto object : tensor_objects(text)) {
    IQManifestTensor tensor;
    tensor.name = string_field(object, "name");
    tensor.type = string_field(object, "type");
    tensor.layer = static_cast<int>(optional_u64_field(object, "layer"));
    if (object.find("\"layer\": null") != std::string_view::npos) {
      tensor.layer = -1;
    }
    tensor.role = optional_string_field(object, "role");
    tensor.dimensions = optional_u64_array_field(object, "dimensions");
    tensor.element_count = u64_field(object, "element_count");
    tensor.block_size = u64_field(object, "block_size");
    tensor.block_type_size = u64_field(object, "block_type_size");
    tensor.block_count = u64_field(object, "block_count");
    tensor.source_offset = u64_field(object, "source_offset");
    tensor.source_bytes = u64_field(object, "source_bytes");
    tensor.source_checksum = u64_field(object, "source_checksum");
    tensor.payload_path = string_field(object, "payload_path");
    tensor.payload_bytes = u64_field(object, "payload_bytes");
    tensor.payload_checksum = u64_field(object, "payload_checksum");
    tensor.resident_bytes = u64_field(object, "resident_bytes");
    tensor.policy = optional_string_field(object, "policy");
    tensor.runtime_kernel = optional_string_field(object, "runtime_kernel");
    if (tensor.type != "IQ2_S" && tensor.type != "IQ3_S" && tensor.type != "Q5_K") {
      throw std::runtime_error("unsupported raw quant manifest tensor type: " + tensor.type);
    }
    if (tensor.block_count == 0 || tensor.block_size != 256 || tensor.payload_bytes == 0) {
      throw std::runtime_error("IQ manifest tensor has invalid block metadata: " + tensor.name);
    }
    if (tensor.source_bytes != tensor.payload_bytes) {
      throw std::runtime_error("IQ manifest tensor source/payload byte mismatch: " + tensor.name);
    }
    manifest.tensors.push_back(std::move(tensor));
  }

  if (manifest.tensors.empty()) {
    throw std::runtime_error("IQ manifest contains no tensors");
  }
  return manifest;
}

IQManifestTensor select_iq_tensor(
    const IQManifest& manifest,
    const std::string& name) {
  if (name.empty()) {
    return manifest.tensors.front();
  }
  const auto found = std::find_if(
      manifest.tensors.begin(), manifest.tensors.end(), [&](const auto& tensor) {
        return tensor.name == name;
      });
  if (found == manifest.tensors.end()) {
    throw std::runtime_error("IQ manifest tensor not found: " + name);
  }
  return *found;
}

IQTensorPlanValidation validate_iq_manifest_tensors_against_plan(
    const std::vector<IQManifestTensor>& selected,
    const std::filesystem::path& tensor_plan_path) {
  if (tensor_plan_path.empty()) {
    throw std::runtime_error("tensor plan path is empty");
  }
  const auto entries = load_tensor_plan_entries(tensor_plan_path);
  std::uint64_t checked = 0;
  for (const auto& tensor : selected) {
    const auto found = entries.find(tensor.name);
    if (found == entries.end()) {
      throw std::runtime_error("tensor plan is missing manifest tensor: " + tensor.name);
    }
    const auto& planned = found->second;
    if (planned.type != tensor.type) {
      throw std::runtime_error(
          "tensor plan type mismatch for " + tensor.name + ": manifest " + tensor.type +
          ", plan " + planned.type);
    }
    if (planned.dimensions != tensor.dimensions) {
      throw std::runtime_error(
          "tensor plan dimensions mismatch for " + tensor.name + ": manifest " +
          dimensions_to_string(tensor.dimensions) + ", plan " +
          dimensions_to_string(planned.dimensions));
    }
    if (planned.span_bytes != tensor.source_bytes) {
      throw std::runtime_error(
          "tensor plan source byte mismatch for " + tensor.name + ": manifest " +
          std::to_string(tensor.source_bytes) + ", plan " +
          std::to_string(planned.span_bytes));
    }
    if (tensor.source_offset != 0 && planned.absolute_offset != 0 &&
        planned.absolute_offset != tensor.source_offset) {
      throw std::runtime_error(
          "tensor plan source offset mismatch for " + tensor.name + ": manifest " +
          std::to_string(tensor.source_offset) + ", plan " +
          std::to_string(planned.absolute_offset));
    }
    ++checked;
  }
  return IQTensorPlanValidation{tensor_plan_path, checked};
}

}  // namespace rtxllm
