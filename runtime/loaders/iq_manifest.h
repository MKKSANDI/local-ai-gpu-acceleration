#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace rtxllm {

struct IQManifestTensor {
  std::string name;
  std::string type;
  int layer = -1;
  std::string role;
  std::vector<std::uint64_t> dimensions;
  std::uint64_t element_count = 0;
  std::uint64_t block_size = 0;
  std::uint64_t block_type_size = 0;
  std::uint64_t block_count = 0;
  std::uint64_t source_offset = 0;
  std::uint64_t source_bytes = 0;
  std::uint64_t source_checksum = 0;
  std::filesystem::path payload_path;
  std::uint64_t payload_bytes = 0;
  std::uint64_t payload_checksum = 0;
  std::uint64_t resident_bytes = 0;
  std::string policy;
  std::string runtime_kernel;
};

struct IQManifest {
  std::filesystem::path path;
  std::filesystem::path model;
  std::filesystem::path tensor_plan;
  std::string generated_at;
  std::vector<std::string> types;
  std::uint64_t tensor_count = 0;
  std::uint64_t source_bytes = 0;
  std::uint64_t payload_bytes = 0;
  std::uint64_t resident_bytes = 0;
  std::vector<IQManifestTensor> tensors;
};

struct IQTensorPlanValidation {
  std::filesystem::path path;
  std::uint64_t checked_tensors = 0;
};

IQManifest load_iq_manifest(const std::filesystem::path& path);

IQManifestTensor select_iq_tensor(
    const IQManifest& manifest,
    const std::string& name);

IQTensorPlanValidation validate_iq_manifest_tensors_against_plan(
    const std::vector<IQManifestTensor>& selected,
    const std::filesystem::path& tensor_plan_path);

}  // namespace rtxllm
