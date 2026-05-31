#include "runtime/core/allocator/vram_superallocator.h"
#include "runtime/core/scheduler/admission.h"
#include "runtime/kernels/decode/decode_kernels.h"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::size_t mib(std::size_t value) {
  return value * 1024ull * 1024ull;
}

void check(cudaError_t status, const char* op) {
  if (status != cudaSuccess) {
    std::cerr << op << " failed: " << cudaGetErrorString(status) << "\n";
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  try {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

    const std::size_t weight_bytes = mib(256);
    const std::size_t kv_bytes = mib(64);
    const std::size_t workspace_bytes = mib(64);
    const std::size_t dma_bytes = mib(16);

    const auto decision = rtxllm::decide_admission(
        rtxllm::ResidencyPolicy::Strict,
        rtxllm::MemoryBudget{free_bytes, mib(512)},
        rtxllm::RequestEstimate{
            weight_bytes, 512, 128, 1024, workspace_bytes + dma_bytes});

    if (!decision.admit) {
      std::cerr << decision.reason << "\n";
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

    auto* weights = static_cast<std::uint8_t*>(pool.arena_ptr("weights"));
    auto* kv = static_cast<float*>(pool.arena_ptr("kv_pages"));
    auto* token = static_cast<std::uint32_t*>(pool.arena_ptr("dma"));

    constexpr int steps = 128;
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
      check(rtxllm_launch_synthetic_decode(
                weights,
                kv,
                token,
                weight_bytes,
                kv_bytes / sizeof(float),
                static_cast<std::uint32_t>(step),
                64,
                stream),
          "synthetic decode");
      std::uint32_t host_token = 0;
      check(cudaMemcpyAsync(&host_token, token, sizeof(host_token), cudaMemcpyDeviceToHost, stream),
          "token memcpy");
      check(cudaStreamSynchronize(stream), "stream sync");
    }
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();

    check(cudaStreamDestroy(stream), "cudaStreamDestroy");

    std::cout << "{\n";
    std::cout << "  \"path\": \"native_cuda_synthetic_decode\",\n";
    std::cout << "  \"steps\": " << steps << ",\n";
    std::cout << "  \"wall_ms\": " << elapsed_ms << ",\n";
    std::cout << "  \"steps_per_second\": " << (steps * 1000.0 / elapsed_ms) << ",\n";
    std::cout << "  \"allocated_bytes\": " << pool.total_bytes() << "\n";
    std::cout << "}\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }
}
