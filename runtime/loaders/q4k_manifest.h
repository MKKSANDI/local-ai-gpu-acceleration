#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace rtxllm {

struct Q4KManifestTensor {
  std::string name;
  int layer = -1;
  std::string role;
  std::vector<std::uint64_t> dimensions;
  std::uint32_t rows = 0;
  std::uint32_t cols = 0;
  std::uint64_t block_count = 0;
  std::uint64_t source_offset = 0;
  std::uint64_t source_bytes = 0;
  std::string policy;
  std::string metadata_format;
  std::string runtime_kernel;
  std::filesystem::path payload_path;
  std::filesystem::path metadata_path;
  std::filesystem::path blocks_path;
  std::uint64_t source_checksum = 0;
  std::uint64_t payload_checksum = 0;
  std::uint64_t metadata_checksum = 0;
  std::uint64_t payload_bytes = 0;
  std::uint64_t metadata_bytes = 0;
  std::uint64_t resident_bytes = 0;
};

struct Q4KManifest {
  std::filesystem::path path;
  std::filesystem::path model;
  std::filesystem::path tensor_plan;
  std::string generated_at;
  std::string policy;
  std::uint64_t tensor_count = 0;
  std::uint64_t payload_bytes = 0;
  std::uint64_t metadata_bytes = 0;
  std::uint64_t resident_bytes = 0;
  std::vector<Q4KManifestTensor> tensors;
};

struct Q4KTensorPlanValidation {
  std::filesystem::path path;
  std::uint64_t checked_tensors = 0;
};

Q4KManifest load_q4k_manifest(const std::filesystem::path& path);

std::vector<Q4KManifestTensor> select_q4k_tensors(
    const Q4KManifest& manifest,
    const std::vector<std::string>& names,
    std::size_t limit);

std::uint64_t byte_checksum(const std::vector<std::uint8_t>& bytes);

std::vector<std::uint8_t> load_binary_file(
    const std::filesystem::path& path,
    std::uint64_t expected_bytes,
    const char* label);

Q4KTensorPlanValidation validate_q4k_manifest_tensors_against_plan(
    const std::vector<Q4KManifestTensor>& selected,
    const std::filesystem::path& tensor_plan_path);

}  // namespace rtxllm
