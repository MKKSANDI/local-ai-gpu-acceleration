#include "runtime/core/allocator/vram_superallocator.h"
#include "runtime/core/scheduler/admission.h"
#include "runtime/kernels/decode/decode_kernels.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
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

double percentile(std::vector<double> values, double pct) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(
      std::min<double>(values.size() - 1, std::ceil((pct / 100.0) * values.size()) - 1));
  return values[index];
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  try {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

    constexpr int steps = 256;
    const std::size_t weight_bytes = mib(256);
    const std::size_t kv_bytes = mib(16);
    const std::size_t workspace_bytes = mib(64);
    const std::size_t dma_bytes = mib(16);
    const std::uint32_t page_words = 16u * 1024u;
    const std::uint32_t active_pages = 256u;
    const int touch_stride = 64;

    const auto decision = rtxllm::decide_admission(
        rtxllm::ResidencyPolicy::Strict,
        rtxllm::MemoryBudget{free_bytes, mib(512)},
        rtxllm::RequestEstimate{
            weight_bytes, 512, 128, 1024, workspace_bytes + dma_bytes + kv_bytes});

    if (!decision.admit) {
      std::cout << "{\n";
      std::cout << "  \"path\": \"native_cuda_graph_synthetic_decode\",\n";
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

    auto* weights = static_cast<std::uint8_t*>(pool.arena_ptr("weights"));
    auto* kv = static_cast<float*>(pool.arena_ptr("kv_pages"));
    auto* token = static_cast<std::uint32_t*>(pool.arena_ptr("dma"));

    std::uint32_t* host_token = nullptr;
    check(cudaHostAlloc(
              reinterpret_cast<void**>(&host_token),
              sizeof(std::uint32_t),
              cudaHostAllocDefault),
        "cudaHostAlloc");

    check(rtxllm_launch_synthetic_decode_stateful(
              weights,
              kv,
              token,
              weight_bytes,
              kv_bytes / sizeof(float),
              page_words,
              active_pages,
              touch_stride,
              stream),
        "warmup synthetic decode");
    check(cudaMemcpyAsync(host_token, token, sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream),
        "warmup token memcpy");
    check(cudaStreamSynchronize(stream), "warmup sync");

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    const auto capture_start = std::chrono::steady_clock::now();
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
    check(rtxllm_launch_synthetic_decode_stateful(
              weights,
              kv,
              token,
              weight_bytes,
              kv_bytes / sizeof(float),
              page_words,
              active_pages,
              touch_stride,
              stream),
        "captured synthetic decode");
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
    std::cout << "  \"path\": \"native_cuda_graph_synthetic_decode\",\n";
    std::cout << "  \"admitted\": true,\n";
    std::cout << "  \"steps\": " << steps << ",\n";
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
    std::cout << "  \"h2d_bytes\": 0,\n";
    std::cout << "  \"d2h_bytes\": " << (sizeof(std::uint32_t) * steps) << ",\n";
    std::cout << "  \"last_token\": " << last_token_value << "\n";
    std::cout << "}\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }
}
