#include "runtime/core/allocator/vram_superallocator.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <sstream>

namespace rtxllm {

namespace {

void check_cuda(cudaError_t status, const char* op) {
  if (status == cudaSuccess) {
    return;
  }
  std::ostringstream oss;
  oss << op << " failed: " << cudaGetErrorString(status);
  throw std::runtime_error(oss.str());
}

}  // namespace

VramSuperallocator::VramSuperallocator(VramSuperallocator&& other) noexcept {
  *this = std::move(other);
}

VramSuperallocator& VramSuperallocator::operator=(VramSuperallocator&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  base_ = other.base_;
  total_bytes_ = other.total_bytes_;
  arenas_ = std::move(other.arenas_);
  other.base_ = nullptr;
  other.total_bytes_ = 0;
  return *this;
}

VramSuperallocator::~VramSuperallocator() {
  release();
}

void VramSuperallocator::initialize(const std::vector<ArenaSpec>& specs) {
  release();
  if (specs.empty()) {
    throw std::invalid_argument("VramSuperallocator requires at least one arena");
  }

  std::size_t offset = 0;
  arenas_.clear();
  arenas_.reserve(specs.size());
  for (const auto& spec : specs) {
    if (spec.name.empty() || spec.bytes == 0) {
      throw std::invalid_argument("arena name and byte size must be non-empty");
    }
    offset = align_up(offset);
    arenas_.push_back(ArenaView{spec.name, offset, spec.bytes});
    offset += spec.bytes;
  }
  total_bytes_ = align_up(offset);
  check_cuda(cudaMalloc(&base_, total_bytes_), "cudaMalloc(superpool)");
}

void VramSuperallocator::reset() {
  if (!base_) {
    return;
  }
  check_cuda(cudaMemset(base_, 0, total_bytes_), "cudaMemset(superpool)");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(superpool reset)");
}

void VramSuperallocator::reset_async(cudaStream_t stream) {
  if (!base_) {
    return;
  }
  check_cuda(cudaMemsetAsync(base_, 0, total_bytes_, stream), "cudaMemsetAsync(superpool)");
}

void* VramSuperallocator::arena_ptr(std::string_view name) const {
  const auto found = std::find_if(arenas_.begin(), arenas_.end(), [&](const ArenaView& view) {
    return view.name == name;
  });
  if (found == arenas_.end()) {
    throw std::out_of_range("unknown VRAM arena");
  }
  return static_cast<std::uint8_t*>(base_) + found->offset;
}

ArenaView VramSuperallocator::arena(std::string_view name) const {
  const auto found = std::find_if(arenas_.begin(), arenas_.end(), [&](const ArenaView& view) {
    return view.name == name;
  });
  if (found == arenas_.end()) {
    throw std::out_of_range("unknown VRAM arena");
  }
  return *found;
}

std::size_t VramSuperallocator::align_up(std::size_t value) {
  const auto mask = kAlignment - 1;
  return (value + mask) & ~mask;
}

void VramSuperallocator::release() {
  if (base_) {
    cudaFree(base_);
  }
  base_ = nullptr;
  total_bytes_ = 0;
  arenas_.clear();
}

}  // namespace rtxllm
