#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace rtxllm {

struct BenchmarkMetrics {
  std::string path;
  std::uint64_t steps = 0;
  double wall_ms = 0.0;
  double steps_per_second = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  double p99_ms = 0.0;
  std::size_t allocated_bytes = 0;
  std::size_t h2d_bytes = 0;
  std::size_t d2h_bytes = 0;
  double graph_replay_rate = 0.0;
};

}  // namespace rtxllm
