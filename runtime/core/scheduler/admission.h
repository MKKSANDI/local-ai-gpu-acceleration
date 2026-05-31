#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace rtxllm {

enum class ResidencyPolicy {
  Strict,
  Balanced,
  Overflow,
};

struct MemoryBudget {
  std::size_t vram_free_bytes = 0;
  std::size_t wddm_guard_bytes = 512ull * 1024ull * 1024ull;
};

struct RequestEstimate {
  std::size_t weight_bytes = 0;
  std::size_t prompt_tokens = 0;
  std::size_t max_new_tokens = 0;
  std::size_t kv_bytes_per_token = 0;
  std::size_t workspace_bytes = 0;
  std::size_t kv_cache_bytes = 0;
  std::size_t dma_bytes = 0;
  std::size_t graph_bytes = 0;
  std::size_t workspace_high_water_bytes = 0;
};

struct AdmissionBreakdown {
  std::size_t weight_bytes = 0;
  std::size_t prompt_tokens = 0;
  std::size_t max_new_tokens = 0;
  std::size_t token_count = 0;
  std::size_t kv_bytes_per_token = 0;
  std::size_t predicted_kv_bytes = 0;
  std::size_t kv_cache_bytes = 0;
  std::size_t total_kv_bytes = 0;
  std::size_t workspace_bytes = 0;
  std::size_t workspace_high_water_bytes = 0;
  std::size_t workspace_slack_bytes = 0;
  std::size_t dma_bytes = 0;
  std::size_t graph_bytes = 0;
  std::size_t required_bytes = 0;
  std::size_t vram_free_bytes = 0;
  std::size_t wddm_guard_bytes = 0;
  std::size_t usable_bytes = 0;
  std::size_t over_budget_bytes = 0;
};

struct AdmissionDecision {
  bool admit = false;
  bool host_spill_allowed = false;
  std::size_t required_bytes = 0;
  std::size_t usable_bytes = 0;
  AdmissionBreakdown breakdown;
  std::string reason;
};

AdmissionDecision decide_admission(
    ResidencyPolicy policy,
    const MemoryBudget& budget,
    const RequestEstimate& request);

}  // namespace rtxllm
