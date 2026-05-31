#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rtxllm {

struct CudaGraphBucketDescriptor {
  std::string name;
  std::uint32_t batch_bucket = 1;
  std::uint32_t query_bucket = 1;
  std::uint32_t phase_count = 0;
  std::uint32_t kernel_launches_per_replay = 0;
  std::size_t workspace_high_water_bytes = 0;
};

struct CudaGraphBucketAllocation {
  CudaGraphBucketDescriptor descriptor;
  std::size_t offset = 0;
  std::size_t metadata_bytes = 0;
  std::size_t workspace_guard_bytes = 0;
  std::size_t total_bytes = 0;
};

struct CudaGraphBucketPlan {
  std::vector<CudaGraphBucketAllocation> buckets;
  std::size_t total_bytes = 0;
  std::size_t max_bucket_bytes = 0;
};

inline std::size_t align_cuda_graph_bucket_bytes(
    std::size_t value,
    std::size_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument("CUDA graph bucket alignment must be a power of two");
  }
  const std::size_t mask = alignment - 1;
  if (value > std::numeric_limits<std::size_t>::max() - mask) {
    throw std::overflow_error("CUDA graph bucket byte count overflow");
  }
  return (value + mask) & ~mask;
}

inline std::size_t checked_cuda_graph_bucket_add(
    std::size_t lhs,
    std::size_t rhs) {
  if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
    throw std::overflow_error("CUDA graph bucket byte count overflow");
  }
  return lhs + rhs;
}

inline std::size_t checked_cuda_graph_bucket_mul(
    std::size_t lhs,
    std::size_t rhs) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error("CUDA graph bucket byte count overflow");
  }
  return lhs * rhs;
}

inline CudaGraphBucketAllocation estimate_cuda_graph_bucket_allocation(
    const CudaGraphBucketDescriptor& descriptor,
    std::size_t alignment = 256) {
  if (descriptor.name.empty()) {
    throw std::invalid_argument("CUDA graph bucket name must be non-empty");
  }
  if (descriptor.batch_bucket == 0 || descriptor.query_bucket == 0 ||
      descriptor.phase_count == 0 || descriptor.kernel_launches_per_replay == 0) {
    throw std::invalid_argument(
        "CUDA graph bucket dimensions and kernel count must be non-zero");
  }

  constexpr std::size_t kBaseGraphBytes = 64ull * 1024ull;
  constexpr std::size_t kKernelNodeBytes = 4ull * 1024ull;
  constexpr std::size_t kPhaseNodeBytes = 2ull * 1024ull;
  constexpr std::size_t kShapeBytes = 512ull;

  std::size_t metadata = kBaseGraphBytes;
  metadata = checked_cuda_graph_bucket_add(
      metadata,
      checked_cuda_graph_bucket_mul(
          descriptor.kernel_launches_per_replay,
          kKernelNodeBytes));
  metadata = checked_cuda_graph_bucket_add(
      metadata,
      checked_cuda_graph_bucket_mul(descriptor.phase_count, kPhaseNodeBytes));
  metadata = checked_cuda_graph_bucket_add(
      metadata,
      checked_cuda_graph_bucket_mul(
          checked_cuda_graph_bucket_mul(
              descriptor.batch_bucket,
              descriptor.query_bucket),
          kShapeBytes));
  metadata = align_cuda_graph_bucket_bytes(metadata, alignment);

  const std::size_t workspace_guard = align_cuda_graph_bucket_bytes(
      descriptor.workspace_high_water_bytes / 64u,
      alignment);
  const std::size_t total =
      checked_cuda_graph_bucket_add(metadata, workspace_guard);

  CudaGraphBucketAllocation allocation;
  allocation.descriptor = descriptor;
  allocation.metadata_bytes = metadata;
  allocation.workspace_guard_bytes = workspace_guard;
  allocation.total_bytes = total;
  return allocation;
}

inline CudaGraphBucketPlan build_cuda_graph_bucket_plan(
    std::span<const CudaGraphBucketDescriptor> descriptors,
    std::size_t alignment = 256) {
  if (descriptors.empty()) {
    throw std::invalid_argument("CUDA graph bucket plan requires at least one bucket");
  }

  CudaGraphBucketPlan plan;
  plan.buckets.reserve(descriptors.size());
  for (const auto& descriptor : descriptors) {
    auto allocation = estimate_cuda_graph_bucket_allocation(descriptor, alignment);
    allocation.offset = plan.total_bytes;
    plan.total_bytes =
        checked_cuda_graph_bucket_add(plan.total_bytes, allocation.total_bytes);
    if (allocation.total_bytes > plan.max_bucket_bytes) {
      plan.max_bucket_bytes = allocation.total_bytes;
    }
    plan.buckets.push_back(std::move(allocation));
  }
  return plan;
}

class CudaGraphBucket {
 public:
  CudaGraphBucket() = default;
  CudaGraphBucket(const CudaGraphBucket&) = delete;
  CudaGraphBucket& operator=(const CudaGraphBucket&) = delete;

  ~CudaGraphBucket() {
    if (exec_) {
      cudaGraphExecDestroy(exec_);
    }
    if (graph_) {
      cudaGraphDestroy(graph_);
    }
  }

  void instantiate(cudaGraph_t graph) {
    graph_ = graph;
    const auto status = cudaGraphInstantiate(&exec_, graph_, 0);
    if (status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(status));
    }
  }

  void replay(cudaStream_t stream) const {
    if (!exec_) {
      throw std::logic_error("CUDA graph bucket has not been instantiated");
    }
    const auto status = cudaGraphLaunch(exec_, stream);
    if (status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(status));
    }
  }

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
};

}  // namespace rtxllm
