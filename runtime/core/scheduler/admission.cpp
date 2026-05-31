#include "runtime/core/scheduler/admission.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace rtxllm {

namespace {

std::size_t checked_add(std::size_t lhs, std::size_t rhs) {
  if (lhs > std::numeric_limits<std::size_t>::max() - rhs) {
    throw std::overflow_error("admission byte count overflow");
  }
  return lhs + rhs;
}

std::size_t checked_mul(std::size_t lhs, std::size_t rhs) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error("admission byte count overflow");
  }
  return lhs * rhs;
}

AdmissionBreakdown build_breakdown(
    const MemoryBudget& budget,
    const RequestEstimate& request) {
  AdmissionBreakdown breakdown;
  breakdown.weight_bytes = request.weight_bytes;
  breakdown.prompt_tokens = request.prompt_tokens;
  breakdown.max_new_tokens = request.max_new_tokens;
  breakdown.token_count = checked_add(request.prompt_tokens, request.max_new_tokens);
  breakdown.kv_bytes_per_token = request.kv_bytes_per_token;
  breakdown.predicted_kv_bytes =
      checked_mul(breakdown.token_count, request.kv_bytes_per_token);
  breakdown.kv_cache_bytes = request.kv_cache_bytes;
  breakdown.total_kv_bytes =
      checked_add(breakdown.predicted_kv_bytes, request.kv_cache_bytes);
  breakdown.workspace_bytes = request.workspace_bytes;
  breakdown.workspace_high_water_bytes = request.workspace_high_water_bytes;
  breakdown.workspace_slack_bytes =
      request.workspace_bytes > request.workspace_high_water_bytes
          ? request.workspace_bytes - request.workspace_high_water_bytes
          : 0;
  breakdown.dma_bytes = request.dma_bytes;
  breakdown.graph_bytes = request.graph_bytes;
  breakdown.vram_free_bytes = budget.vram_free_bytes;
  breakdown.wddm_guard_bytes = budget.wddm_guard_bytes;
  breakdown.usable_bytes = budget.vram_free_bytes > budget.wddm_guard_bytes
      ? budget.vram_free_bytes - budget.wddm_guard_bytes
      : 0;

  std::size_t required = breakdown.weight_bytes;
  required = checked_add(required, breakdown.total_kv_bytes);
  required = checked_add(required, breakdown.workspace_bytes);
  required = checked_add(required, breakdown.dma_bytes);
  required = checked_add(required, breakdown.graph_bytes);
  breakdown.required_bytes = required;
  breakdown.over_budget_bytes = required > breakdown.usable_bytes
      ? required - breakdown.usable_bytes
      : 0;
  return breakdown;
}

}  // namespace

AdmissionDecision decide_admission(
    ResidencyPolicy policy,
    const MemoryBudget& budget,
    const RequestEstimate& request) {
  const auto breakdown = build_breakdown(budget, request);

  AdmissionDecision decision;
  decision.required_bytes = breakdown.required_bytes;
  decision.usable_bytes = breakdown.usable_bytes;
  decision.breakdown = breakdown;

  if (breakdown.required_bytes <= breakdown.usable_bytes) {
    decision.admit = true;
    decision.reason = "fits strict VRAM budget";
    return decision;
  }

  if (policy == ResidencyPolicy::Overflow) {
    decision.admit = true;
    decision.host_spill_allowed = true;
    decision.reason = "requires explicit overflow policy";
    return decision;
  }

  std::ostringstream oss;
  oss << "rejected: required " << breakdown.required_bytes
      << " bytes, usable " << breakdown.usable_bytes
      << " bytes, over budget " << breakdown.over_budget_bytes << " bytes";
  decision.reason = oss.str();
  return decision;
}

}  // namespace rtxllm
