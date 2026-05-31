#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rtxllm {

struct ArenaSpec {
  std::string name;
  std::size_t bytes = 0;
};

struct ArenaView {
  std::string name;
  std::size_t offset = 0;
  std::size_t bytes = 0;
};

class VramSuperallocator {
 public:
  static constexpr std::size_t kAlignment = 256;

  VramSuperallocator() = default;
  VramSuperallocator(const VramSuperallocator&) = delete;
  VramSuperallocator& operator=(const VramSuperallocator&) = delete;
  VramSuperallocator(VramSuperallocator&& other) noexcept;
  VramSuperallocator& operator=(VramSuperallocator&& other) noexcept;
  ~VramSuperallocator();

  void initialize(const std::vector<ArenaSpec>& specs);
  void reset();
  void reset_async(cudaStream_t stream);

  [[nodiscard]] void* base() const { return base_; }
  [[nodiscard]] std::size_t total_bytes() const { return total_bytes_; }
  [[nodiscard]] const std::vector<ArenaView>& arenas() const { return arenas_; }
  [[nodiscard]] void* arena_ptr(std::string_view name) const;
  [[nodiscard]] ArenaView arena(std::string_view name) const;

 private:
  static std::size_t align_up(std::size_t value);
  void release();

  void* base_ = nullptr;
  std::size_t total_bytes_ = 0;
  std::vector<ArenaView> arenas_;
};

}  // namespace rtxllm
