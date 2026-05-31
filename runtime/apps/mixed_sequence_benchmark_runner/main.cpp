#include "runtime/core/allocator/vram_superallocator.h"
#include "runtime/core/graphs/cuda_graph_bucket.h"
#include "runtime/core/scheduler/admission.h"
#include "runtime/core/scheduler/layer_executor.h"
#include "runtime/core/scheduler/layer_plan.h"
#include "runtime/kernels/decode/decode_kernels.h"
#include "runtime/loaders/iq_manifest.h"
#include "runtime/loaders/q4k_manifest.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <map>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kAlignment = 256;
constexpr std::uint32_t kQ4KValues = 256;
constexpr std::uint32_t kQ4KPayloadBytes = 128;
constexpr std::uint32_t kQ4KPredecodedMetaBytes = 64;
constexpr std::uint32_t kQ4KHalfMetaBytes = 32;
constexpr std::uint32_t kQ4KCompactMetaBytes = 20;
constexpr std::uint32_t kQ4KNativeMetaBytes = 16;
constexpr std::uint64_t kIq2SBlockBytes = 82;
constexpr std::uint64_t kIq2SQsOffset = 2;
constexpr std::uint64_t kIq2SQhOffset = 66;
constexpr std::uint64_t kIq2SScalesOffset = 74;
constexpr std::uint64_t kIq3SBlockBytes = 110;
constexpr std::uint64_t kIq3SQsOffset = 2;
constexpr std::uint64_t kIq3SQhOffset = 66;
constexpr std::uint64_t kIq3SSignsOffset = 74;
constexpr std::uint64_t kIq3SScalesOffset = 106;
constexpr std::uint64_t kQ5KBlockBytes = 176;
constexpr std::uint32_t kQ5KDOffset = 0;
constexpr std::uint32_t kQ5KDMinOffset = 2;
constexpr std::uint32_t kQ5KScalesOffset = 4;
constexpr std::uint32_t kQ5KQhOffset = 16;
constexpr std::uint32_t kQ5KQsOffset = 48;
constexpr float kSsmStateDecay = 0.875f;
constexpr float kSsmStateLogitScale = 0.03125f;
constexpr float kSsmStateActivationScale = 0.015625f;
constexpr float kSsmStateOutputScale = 0.125f;
constexpr const char* kSsmScanScratchName = "ssm_scan_scratch";
constexpr float kSsmScanScratchScale = 0.046875f;
constexpr float kSsmSelectiveGateLogitScale = 0.125f;
constexpr float kSsmSelectiveGateActivationScale = 0.0625f;
constexpr float kSsmSelectiveGateScratchScale = 0.03125f;
constexpr float kSsmSourceParamCacheLogitScale = 0.125f;
constexpr float kSsmSourceParamCacheActivationScale = 0.03125f;
constexpr float kSsmSourceParamAlphaScale = 0.0625f;
constexpr float kSsmSourceParamBetaScale = 0.046875f;
constexpr float kSsmSourceParamGateLogitScale = 0.125f;
constexpr float kSsmSourceParamGateActivationScale = 0.0625f;
constexpr std::uint32_t kOutputPhaseVocabSize = 32000;
constexpr std::uint32_t kOutputPhaseTokenOffset = 0;
constexpr const char* kAttentionQkvScratchName = "attention_qkv_state";
constexpr std::uint32_t kAttentionQkvComponents = 3;
constexpr float kAttentionQkvResidualScale = 0.03125f;
constexpr float kAttentionQkvReduceScale = 0.015625f;
constexpr std::uint32_t kAttentionQkvWindowSize = 4;
constexpr float kAttentionQkvWindowSoftmaxScale = 0.25f;
constexpr float kAttentionQkvWindowResidualScale = 0.01171875f;
constexpr std::uint32_t kAttentionQkvHeadWindowHeadCount = 8;
constexpr std::uint32_t kAttentionQkvHeadWindowSize = 4;
constexpr float kAttentionQkvHeadWindowSoftmaxScale = 0.25f;
constexpr float kAttentionQkvHeadWindowResidualScale = 0.0107421875f;
constexpr std::uint32_t kAttentionQkvHeadDimWindowHeadCount = 8;
constexpr std::uint32_t kAttentionQkvHeadDimWindowHeadDim = 8;
constexpr std::uint32_t kAttentionQkvHeadDimWindowSize = 4;
constexpr float kAttentionQkvHeadDimWindowSoftmaxScale = 0.125f;
constexpr float kAttentionQkvHeadDimWindowResidualScale = 0.009765625f;
constexpr std::uint32_t kAttentionQkvHeadTileWindowHeadCount = 8;
constexpr std::uint32_t kAttentionQkvHeadTileWindowHeadDim = 8;
constexpr std::uint32_t kAttentionQkvHeadTileWindowSize = 4;
constexpr float kAttentionQkvHeadTileWindowSoftmaxScale = 0.125f;
constexpr float kAttentionQkvHeadTileWindowResidualScale = 0.009765625f;
constexpr std::uint32_t kAttentionQkvHeadGroupWindowHeadCount = 8;
constexpr std::uint32_t kAttentionQkvHeadGroupWindowHeadDim = 8;
constexpr std::uint32_t kAttentionQkvHeadGroupWindowSize = 4;
constexpr std::uint32_t kAttentionQkvHeadGroupWindowContextsPerBlock = 4;
constexpr float kAttentionQkvHeadGroupWindowSoftmaxScale = 0.125f;
constexpr float kAttentionQkvHeadGroupWindowResidualScale = 0.009765625f;
constexpr std::uint32_t kAttentionQkvHeadGroupRopeWindowHeadCount = 8;
constexpr std::uint32_t kAttentionQkvHeadGroupRopeWindowHeadDim = 8;
constexpr std::uint32_t kAttentionQkvHeadGroupRopeWindowSize = 4;
constexpr std::uint32_t kAttentionQkvHeadGroupRopeWindowContextsPerBlock = 4;
constexpr float kAttentionQkvHeadGroupRopeWindowThetaScale = 0.03125f;
constexpr float kAttentionQkvHeadGroupRopeWindowSoftmaxScale = 0.125f;
constexpr float kAttentionQkvHeadGroupRopeWindowResidualScale = 0.009765625f;
constexpr std::uint32_t kAttentionQkvHeadGroupFusedHeadCount = 8;
constexpr std::uint32_t kAttentionQkvHeadGroupFusedHeadDim = 8;
constexpr std::uint32_t kAttentionQkvHeadGroupFusedSize = 4;
constexpr std::uint32_t kAttentionQkvHeadGroupFusedContextsPerBlock = 4;
constexpr float kAttentionQkvHeadGroupFusedSoftmaxScale = 0.125f;
constexpr float kAttentionQkvHeadGroupFusedResidualScale = 0.009765625f;
constexpr std::array<std::uint16_t, 1024> kIq2SKGrid = {{
#include "runtime/kernels/decode/iq2s_kgrid_values.inl"
}};
constexpr std::array<std::uint32_t, 512> kIq3SGrid = {{
#include "runtime/kernels/decode/iq3s_grid_values.inl"
}};

enum class StageKind {
  Q4K,
  IQ2S,
  IQ3S,
  Q5K,
};

enum class StageOrderPolicy {
  SourceOffset,
  RolePlan,
};

enum class FeedbackMode {
  Synthetic,
  RmsNorm,
  PhaseAwareFfnSilu,
  PhaseAwareFfnGatedSilu,
};

enum class FfnPhaseMode {
  Chained,
  GatedSilu,
};

enum class AttentionPhaseMode {
  Passthrough,
  QkvScratch,
  QkvReduce,
  QkvWindow,
  QkvHeadWindow,
  QkvHeadDimWindow,
  QkvHeadTileWindow,
  QkvHeadGroupWindow,
  QkvHeadGroupRopeWindow,
  QkvHeadGroupFused,
};

enum class SsmPhaseMode {
  Passthrough,
  RecurrentState,
  ScanScratch,
  SelectiveScan,
  SourceParameterized,
  SourceParameterizedFused,
};

enum class OutputPhaseMode {
  Passthrough,
  FinalToken,
};

struct Options {
  std::filesystem::path q4k_manifest;
  std::filesystem::path iq_manifest;
  std::vector<std::filesystem::path> extra_iq_manifests;
  std::filesystem::path q5k_manifest;
  std::filesystem::path tensor_plan;
  std::vector<std::string> q4k_tensors;
  std::vector<std::string> iq_tensors;
  std::string q5k_tensor = "output.weight";
  std::string label = "mixed_layer_sequence";
  int layer = 0;
  int steps = 8;
  std::uint32_t rows_limit = 64;
  std::size_t kv_mib = 8;
  std::size_t wddm_guard_mib = 512;
  std::uint32_t page_words = 256;
  std::uint32_t active_pages = 64;
  std::uint32_t graph_batch_bucket = 1;
  std::uint32_t graph_query_bucket = 1;
  std::uint32_t graph_bucket_count = 1;
  bool chain_activation = true;
  FeedbackMode feedback_mode = FeedbackMode::RmsNorm;
  FfnPhaseMode ffn_phase_mode = FfnPhaseMode::Chained;
  AttentionPhaseMode attention_phase_mode = AttentionPhaseMode::Passthrough;
  SsmPhaseMode ssm_phase_mode = SsmPhaseMode::Passthrough;
  OutputPhaseMode output_phase_mode = OutputPhaseMode::Passthrough;
  std::vector<rtxllm::LayerAuxWorkspaceDescriptor> phase_scratch_workspaces;
  bool source_plan_validation = true;
  bool check_reference = false;
  bool strict_layer_plan = false;
  float reference_tolerance = 0.05f;
  StageOrderPolicy stage_order = StageOrderPolicy::SourceOffset;
};

struct TensorShape {
  std::uint32_t rows = 0;
  std::uint32_t cols = 0;
};

struct Stage {
  StageKind kind = StageKind::Q4K;
  rtxllm::Q4KManifestTensor q4k;
  rtxllm::IQManifestTensor iq;
  std::filesystem::path payload_root;
  std::size_t payload_offset = 0;
  std::size_t metadata_offset = 0;
  std::uint32_t rows = 0;
  std::uint32_t cols = 0;
  int layer = -1;
  std::uint64_t source_offset = 0;
  std::string name;
  std::string role;
  std::string type;
};

struct IQStageInput {
  rtxllm::IQManifestTensor tensor;
  std::filesystem::path manifest_path;
};

struct ObservedStageSnapshot {
  std::vector<float> logits;
  std::uint32_t token = 0;
};

struct ReferenceStageResult {
  bool passed = false;
  float max_logit_abs_error = 0.0f;
  float first_logit_expected = 0.0f;
  float first_logit_observed = 0.0f;
  std::uint32_t max_logit_error_row = 0;
  std::uint32_t token_expected = 0;
  std::uint32_t token_observed = 0;
  std::uint32_t rows = 0;
  std::size_t logit_mismatches = 0;
  std::string name;
  std::string role;
  std::string type;
};

struct ReferenceResult {
  bool checked = false;
  bool passed = false;
  float tolerance = 0.0f;
  float max_logit_abs_error = 0.0f;
  float max_kv_abs_error = 0.0f;
  float first_logit_expected = 0.0f;
  float first_logit_observed = 0.0f;
  std::uint32_t max_logit_error_row = 0;
  std::uint32_t max_kv_error_index = 0;
  std::uint32_t token_expected = 0;
  std::uint32_t token_observed = 0;
  std::uint32_t final_rows = 0;
  std::size_t logit_mismatches = 0;
  std::size_t kv_mismatches = 0;
  std::size_t stage_mismatches = 0;
  std::string final_stage;
  std::vector<ReferenceStageResult> stages;
};

const char* stage_order_json_name(StageOrderPolicy policy) {
  switch (policy) {
    case StageOrderPolicy::SourceOffset:
      return "source_offset_with_global_tail";
    case StageOrderPolicy::RolePlan:
      return "role_plan_with_global_tail";
  }
  return "unknown";
}

const char* stage_order_cli_name(StageOrderPolicy policy) {
  switch (policy) {
    case StageOrderPolicy::SourceOffset:
      return "source-offset";
    case StageOrderPolicy::RolePlan:
      return "role-plan";
  }
  return "unknown";
}

StageOrderPolicy parse_stage_order_policy(std::string_view value) {
  if (value == "source-offset") {
    return StageOrderPolicy::SourceOffset;
  }
  if (value == "role-plan") {
    return StageOrderPolicy::RolePlan;
  }
  throw std::invalid_argument("--stage-order must be source-offset or role-plan");
}

const char* feedback_mode_name(FeedbackMode mode) {
  switch (mode) {
    case FeedbackMode::Synthetic:
      return "synthetic";
    case FeedbackMode::RmsNorm:
      return "rmsnorm";
    case FeedbackMode::PhaseAwareFfnSilu:
      return "phase_aware_ffn_silu";
    case FeedbackMode::PhaseAwareFfnGatedSilu:
      return "phase_aware_ffn_gated_silu";
  }
  return "unknown";
}

FeedbackMode parse_feedback_mode(std::string_view value) {
  if (value == "synthetic") {
    return FeedbackMode::Synthetic;
  }
  if (value == "rmsnorm") {
    return FeedbackMode::RmsNorm;
  }
  if (value == "phase-aware-ffn-silu" ||
      value == "phase_aware_ffn_silu") {
    return FeedbackMode::PhaseAwareFfnSilu;
  }
  if (value == "phase-aware-ffn-gated-silu" ||
      value == "phase_aware_ffn_gated_silu") {
    return FeedbackMode::PhaseAwareFfnGatedSilu;
  }
  throw std::invalid_argument(
      "--feedback-mode must be synthetic, rmsnorm, phase-aware-ffn-silu, "
      "or phase-aware-ffn-gated-silu");
}

const char* ffn_phase_mode_name(FfnPhaseMode mode) {
  switch (mode) {
    case FfnPhaseMode::Chained:
      return "chained";
    case FfnPhaseMode::GatedSilu:
      return "gated_silu";
  }
  return "unknown";
}

FfnPhaseMode parse_ffn_phase_mode(std::string_view value) {
  if (value == "chained") {
    return FfnPhaseMode::Chained;
  }
  if (value == "gated-silu" || value == "gated_silu") {
    return FfnPhaseMode::GatedSilu;
  }
  throw std::invalid_argument("--ffn-phase-mode must be chained or gated-silu");
}

const char* attention_phase_mode_name(AttentionPhaseMode mode) {
  switch (mode) {
    case AttentionPhaseMode::Passthrough:
      return "passthrough";
    case AttentionPhaseMode::QkvScratch:
      return "qkv_scratch";
    case AttentionPhaseMode::QkvReduce:
      return "qkv_reduce";
    case AttentionPhaseMode::QkvWindow:
      return "qkv_window";
    case AttentionPhaseMode::QkvHeadWindow:
      return "qkv_head_window";
    case AttentionPhaseMode::QkvHeadDimWindow:
      return "qkv_head_dim_window";
    case AttentionPhaseMode::QkvHeadTileWindow:
      return "qkv_head_tile_window";
    case AttentionPhaseMode::QkvHeadGroupWindow:
      return "qkv_head_group_window";
    case AttentionPhaseMode::QkvHeadGroupRopeWindow:
      return "qkv_head_group_rope_window";
    case AttentionPhaseMode::QkvHeadGroupFused:
      return "qkv_head_group_fused";
  }
  return "unknown";
}

AttentionPhaseMode parse_attention_phase_mode(std::string_view value) {
  if (value == "passthrough") {
    return AttentionPhaseMode::Passthrough;
  }
  if (value == "qkv-scratch" || value == "qkv_scratch") {
    return AttentionPhaseMode::QkvScratch;
  }
  if (value == "qkv-reduce" || value == "qkv_reduce") {
    return AttentionPhaseMode::QkvReduce;
  }
  if (value == "qkv-window" || value == "qkv_window") {
    return AttentionPhaseMode::QkvWindow;
  }
  if (value == "qkv-head-window" || value == "qkv_head_window") {
    return AttentionPhaseMode::QkvHeadWindow;
  }
  if (value == "qkv-head-dim-window" || value == "qkv_head_dim_window") {
    return AttentionPhaseMode::QkvHeadDimWindow;
  }
  if (value == "qkv-head-tile-window" || value == "qkv_head_tile_window") {
    return AttentionPhaseMode::QkvHeadTileWindow;
  }
  if (value == "qkv-head-group-window" || value == "qkv_head_group_window") {
    return AttentionPhaseMode::QkvHeadGroupWindow;
  }
  if (value == "qkv-head-group-rope-window" ||
      value == "qkv_head_group_rope_window") {
    return AttentionPhaseMode::QkvHeadGroupRopeWindow;
  }
  if (value == "qkv-head-group-fused" || value == "qkv_head_group_fused") {
    return AttentionPhaseMode::QkvHeadGroupFused;
  }
  throw std::invalid_argument(
      "--attention-phase-mode must be passthrough, qkv-scratch, qkv-reduce, "
      "qkv-window, qkv-head-window, qkv-head-dim-window, "
      "qkv-head-tile-window, qkv-head-group-window, "
      "qkv-head-group-rope-window, "
      "or qkv-head-group-fused");
}

bool attention_phase_mode_uses_qkv_scratch(AttentionPhaseMode mode) {
  return mode == AttentionPhaseMode::QkvScratch ||
      mode == AttentionPhaseMode::QkvReduce ||
      mode == AttentionPhaseMode::QkvWindow ||
      mode == AttentionPhaseMode::QkvHeadWindow ||
      mode == AttentionPhaseMode::QkvHeadDimWindow ||
      mode == AttentionPhaseMode::QkvHeadTileWindow ||
      mode == AttentionPhaseMode::QkvHeadGroupWindow ||
      mode == AttentionPhaseMode::QkvHeadGroupRopeWindow;
}

bool attention_phase_mode_uses_qkv_reduce(AttentionPhaseMode mode) {
  return mode == AttentionPhaseMode::QkvReduce;
}

bool attention_phase_mode_uses_qkv_window(AttentionPhaseMode mode) {
  return mode == AttentionPhaseMode::QkvWindow;
}

bool attention_phase_mode_uses_qkv_head_window(AttentionPhaseMode mode) {
  return mode == AttentionPhaseMode::QkvHeadWindow;
}

bool attention_phase_mode_uses_qkv_head_dim_window(AttentionPhaseMode mode) {
  return mode == AttentionPhaseMode::QkvHeadDimWindow;
}

bool attention_phase_mode_uses_qkv_head_tile_window(AttentionPhaseMode mode) {
  return mode == AttentionPhaseMode::QkvHeadTileWindow;
}

bool attention_phase_mode_uses_qkv_head_group_window(AttentionPhaseMode mode) {
  return mode == AttentionPhaseMode::QkvHeadGroupWindow;
}

bool attention_phase_mode_uses_qkv_head_group_rope_window(
    AttentionPhaseMode mode) {
  return mode == AttentionPhaseMode::QkvHeadGroupRopeWindow;
}

bool attention_phase_mode_uses_qkv_head_group_fused(AttentionPhaseMode mode) {
  return mode == AttentionPhaseMode::QkvHeadGroupFused;
}

std::size_t attention_phase_mode_kernel_count(AttentionPhaseMode mode) {
  if (attention_phase_mode_uses_qkv_head_group_fused(mode)) {
    return 1;
  }
  if (attention_phase_mode_uses_qkv_reduce(mode) ||
      attention_phase_mode_uses_qkv_window(mode) ||
      attention_phase_mode_uses_qkv_head_window(mode) ||
      attention_phase_mode_uses_qkv_head_dim_window(mode) ||
      attention_phase_mode_uses_qkv_head_tile_window(mode) ||
      attention_phase_mode_uses_qkv_head_group_window(mode) ||
      attention_phase_mode_uses_qkv_head_group_rope_window(mode)) {
    return 2;
  }
  return attention_phase_mode_uses_qkv_scratch(mode) ? 1 : 0;
}

const char* ssm_phase_mode_name(SsmPhaseMode mode) {
  switch (mode) {
    case SsmPhaseMode::Passthrough:
      return "passthrough";
    case SsmPhaseMode::RecurrentState:
      return "recurrent_state";
    case SsmPhaseMode::ScanScratch:
      return "scan_scratch";
    case SsmPhaseMode::SelectiveScan:
      return "selective_scan";
    case SsmPhaseMode::SourceParameterized:
      return "source_parameterized";
    case SsmPhaseMode::SourceParameterizedFused:
      return "source_parameterized_fused";
  }
  return "unknown";
}

SsmPhaseMode parse_ssm_phase_mode(std::string_view value) {
  if (value == "passthrough") {
    return SsmPhaseMode::Passthrough;
  }
  if (value == "recurrent-state" || value == "recurrent_state") {
    return SsmPhaseMode::RecurrentState;
  }
  if (value == "scan-scratch" || value == "scan_scratch") {
    return SsmPhaseMode::ScanScratch;
  }
  if (value == "selective-scan" || value == "selective_scan") {
    return SsmPhaseMode::SelectiveScan;
  }
  if (value == "source-parameterized" || value == "source_parameterized") {
    return SsmPhaseMode::SourceParameterized;
  }
  if (value == "source-parameterized-fused" ||
      value == "source_parameterized_fused") {
    return SsmPhaseMode::SourceParameterizedFused;
  }
  throw std::invalid_argument(
      "--ssm-phase-mode must be passthrough, recurrent-state, scan-scratch, "
      "selective-scan, source-parameterized, or "
      "source-parameterized-fused");
}

bool ssm_phase_mode_uses_recurrent_state(SsmPhaseMode mode) {
  return mode == SsmPhaseMode::RecurrentState ||
      mode == SsmPhaseMode::ScanScratch ||
      mode == SsmPhaseMode::SelectiveScan ||
      mode == SsmPhaseMode::SourceParameterized ||
      mode == SsmPhaseMode::SourceParameterizedFused;
}

bool ssm_phase_mode_uses_scan_scratch(SsmPhaseMode mode) {
  return mode == SsmPhaseMode::ScanScratch ||
      mode == SsmPhaseMode::SelectiveScan ||
      mode == SsmPhaseMode::SourceParameterized ||
      mode == SsmPhaseMode::SourceParameterizedFused;
}

bool ssm_phase_mode_uses_selective_scan(SsmPhaseMode mode) {
  return mode == SsmPhaseMode::SelectiveScan;
}

bool ssm_phase_mode_uses_source_parameterized(SsmPhaseMode mode) {
  return mode == SsmPhaseMode::SourceParameterized ||
      mode == SsmPhaseMode::SourceParameterizedFused;
}

bool ssm_phase_mode_uses_source_parameterized_fused(SsmPhaseMode mode) {
  return mode == SsmPhaseMode::SourceParameterizedFused;
}

std::size_t ssm_phase_mode_kernel_count(SsmPhaseMode mode) {
  if (ssm_phase_mode_uses_source_parameterized_fused(mode)) {
    return 1;
  }
  if (ssm_phase_mode_uses_source_parameterized(mode)) {
    return 3;
  }
  return ssm_phase_mode_uses_recurrent_state(mode) ? 1 : 0;
}

const char* output_phase_mode_name(OutputPhaseMode mode) {
  switch (mode) {
    case OutputPhaseMode::Passthrough:
      return "passthrough";
    case OutputPhaseMode::FinalToken:
      return "final_token";
  }
  return "unknown";
}

OutputPhaseMode parse_output_phase_mode(std::string_view value) {
  if (value == "passthrough") {
    return OutputPhaseMode::Passthrough;
  }
  if (value == "final-token" || value == "final_token") {
    return OutputPhaseMode::FinalToken;
  }
  throw std::invalid_argument(
      "--output-phase-mode must be passthrough or final-token");
}

bool output_phase_mode_uses_final_token(OutputPhaseMode mode) {
  return mode == OutputPhaseMode::FinalToken;
}

bool feedback_edge_uses_ffn_silu(
    FeedbackMode mode,
    const rtxllm::LayerActivationFeedbackEdge& edge) {
  return mode == FeedbackMode::PhaseAwareFfnSilu &&
      edge.source_phase == rtxllm::LayerPhase::Ffn &&
      edge.target_phase == rtxllm::LayerPhase::Ffn;
}

bool feedback_mode_uses_ffn_gate_cache(FeedbackMode mode) {
  return mode == FeedbackMode::PhaseAwareFfnGatedSilu;
}

bool ffn_phase_mode_uses_gated_silu(FfnPhaseMode mode) {
  return mode == FfnPhaseMode::GatedSilu;
}

bool is_ffn_gate_role(std::string_view role) {
  return role == "ffn_gate_exps.weight" || role == "ffn_gate_shexp.weight";
}

bool is_ffn_up_role(std::string_view role) {
  return role == "ffn_up_exps.weight" || role == "ffn_up_shexp.weight";
}

bool is_ffn_down_role(std::string_view role) {
  return role == "ffn_down_exps.weight" || role == "ffn_down_shexp.weight";
}

bool ffn_phase_edge_caches_gate(
    FfnPhaseMode mode,
    const rtxllm::LayerActivationFeedbackEdge& edge,
    const std::vector<Stage>& stages) {
  return mode == FfnPhaseMode::GatedSilu &&
      edge.source_phase == rtxllm::LayerPhase::Ffn &&
      edge.source_stage < stages.size() &&
      is_ffn_gate_role(stages[edge.source_stage].role);
}

bool ffn_phase_edge_caches_up(
    FfnPhaseMode mode,
    const rtxllm::LayerActivationFeedbackEdge& edge,
    const std::vector<Stage>& stages) {
  return mode == FfnPhaseMode::GatedSilu &&
      edge.source_phase == rtxllm::LayerPhase::Ffn &&
      edge.source_stage < stages.size() &&
      is_ffn_up_role(stages[edge.source_stage].role);
}

bool ffn_phase_edge_finalizes_gated_product(
    FfnPhaseMode mode,
    const rtxllm::LayerActivationFeedbackEdge& edge,
    const std::vector<Stage>& stages) {
  return mode == FfnPhaseMode::GatedSilu &&
      edge.source_phase == rtxllm::LayerPhase::Ffn &&
      edge.target_phase == rtxllm::LayerPhase::Ffn &&
      edge.source_stage < stages.size() &&
      stages[edge.source_stage].role == "ffn_up_shexp.weight";
}

bool feedback_edge_caches_ffn_gate(
    FeedbackMode mode,
    const rtxllm::LayerActivationFeedbackEdge& edge,
    const std::vector<Stage>& stages) {
  if (mode != FeedbackMode::PhaseAwareFfnGatedSilu ||
      edge.source_stage >= stages.size() ||
      edge.target_stage >= stages.size()) {
    return false;
  }
  return edge.source_phase == rtxllm::LayerPhase::Ffn &&
      edge.target_phase == rtxllm::LayerPhase::Ffn &&
      stages[edge.source_stage].role == "ffn_gate_shexp.weight" &&
      stages[edge.target_stage].role == "ffn_up_exps.weight";
}

bool feedback_edge_uses_ffn_gated_silu(
    FeedbackMode mode,
    const rtxllm::LayerActivationFeedbackEdge& edge,
    const std::vector<Stage>& stages) {
  if (mode != FeedbackMode::PhaseAwareFfnGatedSilu ||
      edge.source_stage >= stages.size() ||
      edge.target_stage >= stages.size()) {
    return false;
  }
  return edge.source_phase == rtxllm::LayerPhase::Ffn &&
      edge.target_phase == rtxllm::LayerPhase::Ffn &&
      stages[edge.source_stage].role == "ffn_up_shexp.weight" &&
      stages[edge.target_stage].role == "ffn_down_exps.weight";
}

bool source_offset_order_less(const Stage& lhs, const Stage& rhs) {
  const bool lhs_global = lhs.layer < 0;
  const bool rhs_global = rhs.layer < 0;
  if (lhs_global != rhs_global) {
    return !lhs_global;
  }
  if (lhs.layer != rhs.layer) {
    return lhs.layer < rhs.layer;
  }
  if (lhs.source_offset != rhs.source_offset) {
    return lhs.source_offset < rhs.source_offset;
  }
  return lhs.name < rhs.name;
}

bool role_plan_order_less(const Stage& lhs, const Stage& rhs) {
  const bool lhs_global = lhs.layer < 0;
  const bool rhs_global = rhs.layer < 0;
  if (lhs_global != rhs_global) {
    return !lhs_global;
  }
  if (lhs.layer != rhs.layer) {
    return lhs.layer < rhs.layer;
  }
  const int lhs_rank = rtxllm::layer_role_rank(lhs.role);
  const int rhs_rank = rtxllm::layer_role_rank(rhs.role);
  if (lhs_rank != rhs_rank) {
    return lhs_rank < rhs_rank;
  }
  if (lhs.source_offset != rhs.source_offset) {
    return lhs.source_offset < rhs.source_offset;
  }
  return lhs.name < rhs.name;
}

std::vector<rtxllm::LayerStageDescriptor> build_layer_stage_descriptors(
    const std::vector<Stage>& stages) {
  std::vector<rtxllm::LayerStageDescriptor> descriptors;
  descriptors.reserve(stages.size());
  for (const auto& stage : stages) {
    descriptors.push_back(rtxllm::LayerStageDescriptor{
        std::string_view(stage.role),
        stage.layer >= 0,
    });
  }
  return descriptors;
}

std::size_t mib(std::size_t value) {
  return value * 1024ull * 1024ull;
}

std::size_t align_up(std::size_t value) {
  const auto mask = kAlignment - 1;
  return (value + mask) & ~mask;
}

void check(cudaError_t status, const char* op) {
  if (status != cudaSuccess) {
    std::cerr << op << " failed: " << cudaGetErrorString(status) << "\n";
    std::exit(1);
  }
}

struct DeviceMemorySnapshot {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
};

DeviceMemorySnapshot device_memory_snapshot(const char* op) {
  DeviceMemorySnapshot snapshot;
  check(cudaMemGetInfo(&snapshot.free_bytes, &snapshot.total_bytes), op);
  return snapshot;
}

std::int64_t free_memory_delta(
    const DeviceMemorySnapshot& before,
    const DeviceMemorySnapshot& after) {
  if (before.free_bytes >= after.free_bytes) {
    return static_cast<std::int64_t>(before.free_bytes - after.free_bytes);
  }
  return -static_cast<std::int64_t>(after.free_bytes - before.free_bytes);
}

void print_admission_breakdown(const rtxllm::AdmissionBreakdown& breakdown) {
  std::cout << "  \"admission_vram_free_bytes\": "
            << breakdown.vram_free_bytes << ",\n";
  std::cout << "  \"admission_wddm_guard_bytes\": "
            << breakdown.wddm_guard_bytes << ",\n";
  std::cout << "  \"admission_usable_bytes\": "
            << breakdown.usable_bytes << ",\n";
  std::cout << "  \"admission_required_bytes\": "
            << breakdown.required_bytes << ",\n";
  std::cout << "  \"admission_over_budget_bytes\": "
            << breakdown.over_budget_bytes << ",\n";
  std::cout << "  \"admission_weight_bytes\": "
            << breakdown.weight_bytes << ",\n";
  std::cout << "  \"admission_predicted_kv_bytes\": "
            << breakdown.predicted_kv_bytes << ",\n";
  std::cout << "  \"admission_kv_cache_bytes\": "
            << breakdown.kv_cache_bytes << ",\n";
  std::cout << "  \"admission_total_kv_bytes\": "
            << breakdown.total_kv_bytes << ",\n";
  std::cout << "  \"admission_workspace_bytes\": "
            << breakdown.workspace_bytes << ",\n";
  std::cout << "  \"admission_workspace_high_water_bytes\": "
            << breakdown.workspace_high_water_bytes << ",\n";
  std::cout << "  \"admission_workspace_slack_bytes\": "
            << breakdown.workspace_slack_bytes << ",\n";
  std::cout << "  \"admission_dma_bytes\": "
            << breakdown.dma_bytes << ",\n";
  std::cout << "  \"admission_graph_bytes\": "
            << breakdown.graph_bytes << ",\n";
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

double mean_value(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  return std::accumulate(values.begin(), values.end(), 0.0) /
      static_cast<double>(values.size());
}

std::size_t parse_size(const char* value, std::string_view name) {
  const unsigned long long parsed = std::stoull(value);
  if (parsed == 0) {
    throw std::invalid_argument(std::string(name) + " must be non-zero");
  }
  return static_cast<std::size_t>(parsed);
}

int parse_int(const char* value, std::string_view name) {
  const long parsed = std::stol(value);
  if (parsed < 0 || parsed > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(name) + " must be a non-negative int");
  }
  return static_cast<int>(parsed);
}

std::uint32_t parse_u32(const char* value, std::string_view name) {
  const unsigned long parsed = std::stoul(value);
  if (parsed == 0 || parsed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(std::string(name) + " must be in uint32 range and non-zero");
  }
  return static_cast<std::uint32_t>(parsed);
}

float parse_float(const char* value, std::string_view name) {
  const float parsed = std::stof(value);
  if (!(parsed > 0.0f) || !std::isfinite(parsed)) {
    throw std::invalid_argument(std::string(name) + " must be a positive finite float");
  }
  return parsed;
}

rtxllm::LayerPhase parse_layer_phase_cli(std::string_view value) {
  if (value == "attention") {
    return rtxllm::LayerPhase::Attention;
  }
  if (value == "ffn") {
    return rtxllm::LayerPhase::Ffn;
  }
  if (value == "ssm") {
    return rtxllm::LayerPhase::Ssm;
  }
  if (value == "output") {
    return rtxllm::LayerPhase::Output;
  }
  throw std::invalid_argument(
      "layer phase must be attention, ffn, ssm, or output");
}

std::uint32_t layer_phase_scratch_id(rtxllm::LayerPhase phase) {
  switch (phase) {
    case rtxllm::LayerPhase::Attention:
      return 0u;
    case rtxllm::LayerPhase::Ffn:
      return 1u;
    case rtxllm::LayerPhase::Ssm:
      return 2u;
    case rtxllm::LayerPhase::Output:
      return 3u;
    case rtxllm::LayerPhase::Unknown:
      break;
  }
  throw std::runtime_error("scratch digest requires a known layer phase");
}

rtxllm::LayerAuxWorkspaceDescriptor parse_phase_scratch_workspace(
    const char* value) {
  const std::string text(value);
  const auto first = text.find(':');
  const auto second = first == std::string::npos
      ? std::string::npos
      : text.find(':', first + 1);
  if (first == std::string::npos || second == std::string::npos ||
      text.find(':', second + 1) != std::string::npos) {
    throw std::invalid_argument(
        "--phase-scratch must be NAME:PHASE:VALUES");
  }
  rtxllm::LayerAuxWorkspaceDescriptor descriptor;
  descriptor.name = text.substr(0, first);
  descriptor.phase = parse_layer_phase_cli(
      std::string_view(text).substr(first + 1, second - first - 1));
  const auto values = text.substr(second + 1);
  descriptor.values = parse_u32(values.c_str(), "--phase-scratch values");
  if (descriptor.name.empty()) {
    throw std::invalid_argument("--phase-scratch name must be non-empty");
  }
  return descriptor;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const auto require_value = [&](std::string_view name) -> const char* {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string(name) + " requires a value");
      }
      return argv[++i];
    };
    if (arg == "--q4k-manifest") {
      options.q4k_manifest = require_value(arg);
    } else if (arg == "--iq-manifest") {
      options.iq_manifest = require_value(arg);
    } else if (arg == "--extra-iq-manifest") {
      options.extra_iq_manifests.push_back(require_value(arg));
    } else if (arg == "--q5k-manifest") {
      options.q5k_manifest = require_value(arg);
    } else if (arg == "--tensor-plan") {
      options.tensor_plan = require_value(arg);
    } else if (arg == "--q4k-tensor") {
      options.q4k_tensors.push_back(require_value(arg));
    } else if (arg == "--iq-tensor") {
      options.iq_tensors.push_back(require_value(arg));
    } else if (arg == "--q5k-tensor") {
      options.q5k_tensor = require_value(arg);
    } else if (arg == "--label") {
      options.label = require_value(arg);
    } else if (arg == "--layer") {
      options.layer = parse_int(require_value(arg), arg);
    } else if (arg == "--steps") {
      options.steps = static_cast<int>(parse_size(require_value(arg), arg));
    } else if (arg == "--rows-limit") {
      options.rows_limit = parse_u32(require_value(arg), arg);
    } else if (arg == "--kv-mib") {
      options.kv_mib = parse_size(require_value(arg), arg);
    } else if (arg == "--wddm-guard-mib") {
      options.wddm_guard_mib = parse_size(require_value(arg), arg);
    } else if (arg == "--page-words") {
      options.page_words = parse_u32(require_value(arg), arg);
    } else if (arg == "--active-pages") {
      options.active_pages = parse_u32(require_value(arg), arg);
    } else if (arg == "--graph-batch-bucket") {
      options.graph_batch_bucket = parse_u32(require_value(arg), arg);
    } else if (arg == "--graph-query-bucket") {
      options.graph_query_bucket = parse_u32(require_value(arg), arg);
    } else if (arg == "--graph-bucket-count") {
      options.graph_bucket_count = parse_u32(require_value(arg), arg);
    } else if (arg == "--no-chain-activation") {
      options.chain_activation = false;
    } else if (arg == "--no-source-plan-validation") {
      options.source_plan_validation = false;
    } else if (arg == "--check-reference") {
      options.check_reference = true;
    } else if (arg == "--strict-layer-plan") {
      options.strict_layer_plan = true;
    } else if (arg == "--reference-tolerance") {
      options.reference_tolerance = parse_float(require_value(arg), arg);
    } else if (arg == "--feedback-mode") {
      options.feedback_mode = parse_feedback_mode(require_value(arg));
    } else if (arg == "--ffn-phase-mode") {
      options.ffn_phase_mode = parse_ffn_phase_mode(require_value(arg));
    } else if (arg == "--attention-phase-mode") {
      options.attention_phase_mode =
          parse_attention_phase_mode(require_value(arg));
    } else if (arg == "--ssm-phase-mode") {
      options.ssm_phase_mode = parse_ssm_phase_mode(require_value(arg));
    } else if (arg == "--output-phase-mode") {
      options.output_phase_mode = parse_output_phase_mode(require_value(arg));
    } else if (arg == "--phase-scratch") {
      options.phase_scratch_workspaces.push_back(
          parse_phase_scratch_workspace(require_value(arg)));
    } else if (arg == "--stage-order") {
      options.stage_order = parse_stage_order_policy(require_value(arg));
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "runtime_mixed_sequence_bench --q4k-manifest PATH --iq-manifest PATH\n"
          << "                             [--layer N] [--q4k-tensor NAME ...]\n"
          << "                             [--extra-iq-manifest PATH ...]\n"
          << "                             [--iq-tensor NAME ...] [--q5k-manifest PATH]\n"
          << "                             [--q5k-tensor NAME] [--steps N]\n"
          << "                             [--rows-limit N] [--kv-mib N]\n"
          << "                             [--graph-batch-bucket N]\n"
          << "                             [--graph-query-bucket N]\n"
          << "                             [--graph-bucket-count N]\n"
          << "                             [--no-chain-activation]\n"
          << "                             [--feedback-mode rmsnorm|synthetic|phase-aware-ffn-silu|phase-aware-ffn-gated-silu]\n"
          << "                             [--ffn-phase-mode chained|gated-silu]\n"
      << "                             [--attention-phase-mode passthrough|qkv-scratch|qkv-reduce|qkv-window|qkv-head-window|qkv-head-dim-window|qkv-head-tile-window|qkv-head-group-window|qkv-head-group-rope-window|qkv-head-group-fused]\n"
          << "                             [--ssm-phase-mode passthrough|recurrent-state|scan-scratch|selective-scan|source-parameterized|source-parameterized-fused]\n"
          << "                             [--output-phase-mode passthrough|final-token]\n"
          << "                             [--phase-scratch NAME:PHASE:VALUES]\n"
          << "                             [--check-reference]\n"
          << "                             [--stage-order source-offset|role-plan]\n"
          << "                             [--strict-layer-plan]\n"
          << "                             [--no-source-plan-validation]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + std::string(arg));
    }
  }
  if (options.q4k_manifest.empty()) {
    throw std::invalid_argument("--q4k-manifest is required");
  }
  if (options.iq_manifest.empty()) {
    throw std::invalid_argument("--iq-manifest is required");
  }
  return options;
}

std::string json_escape(std::string_view value) {
  std::string out;
  out.reserve(value.size() + 8);
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += ch;
        break;
    }
  }
  return out;
}

TensorShape raw_manifest_tensor_shape(
    const rtxllm::IQManifestTensor& tensor,
    std::string_view family) {
  if (tensor.dimensions.empty()) {
    throw std::runtime_error(std::string(family) + " tensor dimensions are missing: " + tensor.name);
  }
  const auto cols64 = tensor.dimensions.front();
  if (cols64 == 0 || cols64 > std::numeric_limits<std::uint32_t>::max() ||
      (cols64 % 256u) != 0u) {
    throw std::runtime_error(
        std::string(family) + " tensor first dimension must be a uint32 multiple of 256");
  }
  if ((tensor.element_count % cols64) != 0u) {
    throw std::runtime_error(
        std::string(family) + " tensor element count does not divide by first dimension");
  }
  const auto rows64 = tensor.element_count / cols64;
  if (rows64 == 0 || rows64 > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(std::string(family) + " tensor derived row count is out of range");
  }
  const auto expected_blocks = rows64 * (cols64 / 256u);
  if (expected_blocks != tensor.block_count) {
    throw std::runtime_error(
        std::string(family) + " tensor block count does not match derived rows/cols");
  }
  return TensorShape{
      static_cast<std::uint32_t>(rows64),
      static_cast<std::uint32_t>(cols64),
  };
}

std::filesystem::path resolve_payload_path(
    const std::filesystem::path& root,
    const std::filesystem::path& path) {
  return path.is_absolute() ? path : root / path;
}

std::uint16_t read_u16_le(const std::uint8_t* bytes) {
  return static_cast<std::uint16_t>(
      bytes[0] | (static_cast<std::uint16_t>(bytes[1]) << 8u));
}

float read_f32_le(const std::uint8_t* bytes) {
  float value = 0.0f;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

float fp16_bits_to_float(std::uint16_t bits) {
  const float sign = (bits & 0x8000u) ? -1.0f : 1.0f;
  const int exp = static_cast<int>((bits >> 10u) & 0x1fu);
  const int mant = static_cast<int>(bits & 0x03ffu);
  if (exp == 0) {
    return mant == 0 ? sign * 0.0f : sign * std::ldexp(static_cast<float>(mant), -24);
  }
  if (exp == 31) {
    return mant == 0 ? sign * std::numeric_limits<float>::infinity()
                     : std::numeric_limits<float>::quiet_NaN();
  }
  return sign * std::ldexp(1.0f + static_cast<float>(mant) / 1024.0f, exp - 15);
}

void get_scale_min_k4(
    int index,
    const std::uint8_t* packed,
    std::uint8_t& scale,
    std::uint8_t& min_value) {
  if (index < 4) {
    scale = static_cast<std::uint8_t>(packed[index] & 63u);
    min_value = static_cast<std::uint8_t>(packed[index + 4] & 63u);
    return;
  }
  scale = static_cast<std::uint8_t>(
      (packed[index + 4] & 0x0fu) | ((packed[index - 4] >> 6u) << 4u));
  min_value = static_cast<std::uint8_t>(
      (packed[index + 4] >> 4u) | ((packed[index] >> 6u) << 4u));
}

std::uint32_t q4k_meta_bytes_for_kernel(std::string_view runtime_kernel) {
  if (runtime_kernel == "split-predecoded") {
    return kQ4KPredecodedMetaBytes;
  }
  if (runtime_kernel == "split-half") {
    return kQ4KHalfMetaBytes;
  }
  if (runtime_kernel == "split-compact") {
    return kQ4KCompactMetaBytes;
  }
  if (runtime_kernel == "split-native") {
    return kQ4KNativeMetaBytes;
  }
  throw std::runtime_error("mixed reference does not support Q4_K runtime kernel: " +
                           std::string(runtime_kernel));
}

float q4k_dequant_value(
    const std::uint8_t* payload_block,
    const std::uint8_t* meta,
    std::string_view runtime_kernel,
    std::uint32_t value_index) {
  const std::uint32_t group64 = value_index >> 6u;
  const std::uint32_t within64 = value_index & 63u;
  const std::uint32_t lane = within64 & 31u;
  const bool high_half = within64 >= 32u;
  const std::uint32_t pair = group64 * 2u + (high_half ? 1u : 0u);
  const std::uint8_t packed = payload_block[group64 * 32u + lane];
  const float quant = static_cast<float>(high_half ? (packed >> 4u) : (packed & 0x0fu));

  if (runtime_kernel == "split-predecoded") {
    const auto* pair_meta = meta + pair * 2u * sizeof(float);
    return read_f32_le(pair_meta) * quant - read_f32_le(pair_meta + sizeof(float));
  }
  if (runtime_kernel == "split-half") {
    const auto* pair_meta = meta + pair * 2u * sizeof(std::uint16_t);
    return fp16_bits_to_float(read_u16_le(pair_meta)) * quant -
        fp16_bits_to_float(read_u16_le(pair_meta + sizeof(std::uint16_t)));
  }
  if (runtime_kernel == "split-compact") {
    const float d = fp16_bits_to_float(read_u16_le(meta));
    const float dmin = fp16_bits_to_float(read_u16_le(meta + 2u));
    return d * static_cast<float>(meta[4u + pair]) * quant -
        dmin * static_cast<float>(meta[12u + pair]);
  }
  if (runtime_kernel == "split-native") {
    const float d = fp16_bits_to_float(read_u16_le(meta));
    const float dmin = fp16_bits_to_float(read_u16_le(meta + 2u));
    std::uint8_t scale = 0;
    std::uint8_t min_value = 0;
    get_scale_min_k4(static_cast<int>(pair), meta + 4u, scale, min_value);
    return d * static_cast<float>(scale) * quant -
        dmin * static_cast<float>(min_value);
  }
  throw std::runtime_error("unsupported Q4_K runtime kernel in reference");
}

float iq2s_dequant_value(
    const std::uint8_t* block,
    std::uint32_t value_index) {
  const std::uint32_t ib32 = value_index >> 5u;
  const std::uint32_t within32 = value_index & 31u;
  const std::uint32_t group = within32 >> 3u;
  const std::uint32_t lane = within32 & 7u;
  const auto d = fp16_bits_to_float(read_u16_le(block));
  const auto* qs = block + kIq2SQsOffset;
  const auto* qh = block + kIq2SQhOffset;
  const auto* scales = block + kIq2SScalesOffset;
  const std::uint32_t group_offset = ib32 * 4u + group;
  const std::uint32_t grid_index =
      qs[group_offset] | ((static_cast<std::uint32_t>(qh[ib32]) << (8u - 2u * group)) & 0x300u);
  const std::uint32_t packed_grid = kIq2SKGrid[grid_index];
  const std::uint32_t grid = 2u * ((packed_grid >> (2u * lane)) & 0x3u) + 1u;
  const std::uint32_t scale_nibble =
      group < 2u ? (scales[ib32] & 0x0fu) : (scales[ib32] >> 4u);
  const float scale = d * (0.5f + static_cast<float>(scale_nibble)) * 0.25f;
  const bool negative = (qs[32u + group_offset] & (1u << lane)) != 0u;
  return scale * static_cast<float>(grid) * (negative ? -1.0f : 1.0f);
}

float iq3s_dequant_value(
    const std::uint8_t* block,
    std::uint32_t value_index) {
  const std::uint32_t ib32 = value_index >> 5u;
  const std::uint32_t within32 = value_index & 31u;
  const std::uint32_t group = within32 >> 3u;
  const std::uint32_t lane = within32 & 7u;
  const std::uint32_t pair = lane >> 2u;
  const std::uint32_t grid_lane = lane & 3u;
  const auto d = fp16_bits_to_float(read_u16_le(block));
  const auto* qs = block + kIq3SQsOffset;
  const auto* qh = block + kIq3SQhOffset;
  const auto* signs = block + kIq3SSignsOffset;
  const auto* scales = block + kIq3SScalesOffset;
  const std::uint32_t selector_index = ib32 * 8u + group * 2u + pair;
  const std::uint32_t high_bit = (qh[ib32] >> (2u * group + pair)) & 1u;
  const std::uint32_t grid_index = qs[selector_index] | (high_bit << 8u);
  const std::uint32_t packed_grid = kIq3SGrid[grid_index];
  const std::uint32_t grid = (packed_grid >> (8u * grid_lane)) & 0xffu;
  const std::uint32_t scale_nibble =
      (ib32 & 1u) == 0u ? (scales[ib32 >> 1u] & 0x0fu)
                        : (scales[ib32 >> 1u] >> 4u);
  const float scale = d * static_cast<float>(1u + 2u * scale_nibble);
  const bool negative = (signs[ib32 * 4u + group] & (1u << lane)) != 0u;
  return scale * static_cast<float>(grid) * (negative ? -1.0f : 1.0f);
}

float iq_dequant_value(
    const std::string& type,
    const std::uint8_t* block,
    std::uint32_t value_index) {
  return type == "IQ3_S" ? iq3s_dequant_value(block, value_index)
                         : iq2s_dequant_value(block, value_index);
}

float q5k_dequant_value(
    const std::uint8_t* block,
    std::uint32_t value_index) {
  const auto d = fp16_bits_to_float(read_u16_le(block + kQ5KDOffset));
  const auto dmin = fp16_bits_to_float(read_u16_le(block + kQ5KDMinOffset));
  const auto* scales = block + kQ5KScalesOffset;
  const auto* qh = block + kQ5KQhOffset;
  const auto* qs = block + kQ5KQsOffset;

  const std::uint32_t group64 = value_index >> 6u;
  const std::uint32_t within64 = value_index & 63u;
  const std::uint32_t lane = within64 & 31u;
  const bool high_half = within64 >= 32u;
  const std::uint32_t scale_index = group64 * 2u + (high_half ? 1u : 0u);
  std::uint8_t scale = 0;
  std::uint8_t min_value = 0;
  get_scale_min_k4(static_cast<int>(scale_index), scales, scale, min_value);

  const std::uint8_t packed = qs[group64 * 32u + lane];
  const std::uint8_t low_bits =
      high_half ? static_cast<std::uint8_t>(packed >> 4u)
                : static_cast<std::uint8_t>(packed & 0x0fu);
  const std::uint8_t high_mask =
      static_cast<std::uint8_t>((high_half ? 2u : 1u) << (group64 * 2u));
  const std::uint8_t quant =
      static_cast<std::uint8_t>(low_bits + ((qh[lane] & high_mask) ? 16u : 0u));
  return d * static_cast<float>(scale) * static_cast<float>(quant) -
      dmin * static_cast<float>(min_value);
}

std::vector<float> reference_stage_logits(
    const Stage& stage,
    const std::vector<std::uint8_t>& payload,
    const std::vector<std::uint8_t>& metadata,
    const std::vector<float>& activation) {
  std::vector<float> logits(stage.rows, 0.0f);
  if (stage.kind == StageKind::Q4K) {
    const std::uint32_t blocks_per_row = stage.cols / kQ4KValues;
    const auto meta_bytes = q4k_meta_bytes_for_kernel(stage.q4k.runtime_kernel);
    for (std::uint32_t row = 0; row < stage.rows; ++row) {
      float acc = 0.0f;
      for (std::uint32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
        const auto* payload_block = payload.data() +
            (static_cast<std::size_t>(row) * blocks_per_row + block_index) *
                kQ4KPayloadBytes;
        const auto* meta_block = metadata.data() +
            (static_cast<std::size_t>(row) * blocks_per_row + block_index) *
                meta_bytes;
        for (std::uint32_t value_index = 0; value_index < kQ4KValues; ++value_index) {
          const std::uint32_t col = block_index * kQ4KValues + value_index;
          acc += q4k_dequant_value(
                     payload_block, meta_block, stage.q4k.runtime_kernel, value_index) *
              activation[col];
        }
      }
      logits[row] = acc;
    }
    return logits;
  }

  const std::uint32_t blocks_per_row = stage.cols / 256u;
  const auto block_bytes = stage.kind == StageKind::Q5K
      ? kQ5KBlockBytes
      : (stage.kind == StageKind::IQ3S ? kIq3SBlockBytes : kIq2SBlockBytes);
  for (std::uint32_t row = 0; row < stage.rows; ++row) {
    float acc = 0.0f;
    for (std::uint32_t col = 0; col < stage.cols; ++col) {
      const auto block_index =
          static_cast<std::size_t>(row) * blocks_per_row + (col / 256u);
      const auto* block = payload.data() + block_index * block_bytes;
      const float value = stage.kind == StageKind::Q5K
          ? q5k_dequant_value(block, col & 255u)
          : iq_dequant_value(stage.type, block, col & 255u);
      acc += value * activation[col];
    }
    logits[row] = acc;
  }
  return logits;
}

std::uint32_t argmax_token(const std::vector<float>& logits) {
  float best_value = std::numeric_limits<float>::lowest();
  std::uint32_t best_index = 0;
  for (std::uint32_t row = 0; row < logits.size(); ++row) {
    const float value = logits[row];
    if (value > best_value || (value == best_value && row < best_index)) {
      best_value = value;
      best_index = row;
    }
  }
  return best_index;
}

void apply_reference_kv(
    const std::vector<float>& logits,
    std::vector<float>& kv,
    std::uint32_t token,
    std::uint32_t page_words,
    std::uint32_t active_pages) {
  if (kv.empty() || page_words == 0u) {
    return;
  }
  const auto active_rows = std::min<std::uint32_t>(
      static_cast<std::uint32_t>(logits.size()), active_pages);
  for (std::uint32_t row = 0; row < active_rows; ++row) {
    const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
    const std::size_t offset = (token + row) % page_words;
    const std::size_t pos = (page_base + offset) % kv.size();
    kv[pos] = logits[row];
  }
}

void apply_reference_feedback(
    const std::vector<float>& logits,
    std::vector<float>& activation,
    std::uint32_t token,
    std::uint32_t cols,
    std::uint32_t stage_index) {
  if (logits.empty()) {
    return;
  }
  for (std::uint32_t col = 0; col < cols; ++col) {
    const std::uint32_t src =
        (col * 131u + token * 17u + stage_index * 977u) %
        static_cast<std::uint32_t>(logits.size());
    const float value = logits[src];
    const float squashed = value / (1.0f + std::fabs(value));
    const float bias = static_cast<float>((col + stage_index) & 7u) * 0.0009765625f;
    activation[col] = squashed + bias;
  }
}

void apply_reference_rmsnorm_feedback(
    const std::vector<float>& logits,
    std::vector<float>& activation,
    std::uint32_t cols,
    float epsilon) {
  if (logits.empty()) {
    return;
  }
  float sum_sq = 0.0f;
  for (const float value : logits) {
    sum_sq += value * value;
  }
  const float mean_sq =
      sum_sq / static_cast<float>(static_cast<std::uint32_t>(logits.size()));
  const float inv_rms = 1.0f / std::sqrt(mean_sq + epsilon);
  const auto rows = static_cast<std::uint32_t>(logits.size());
  for (std::uint32_t col = 0; col < cols; ++col) {
    activation[col] = logits[col % rows] * inv_rms;
  }
}

void apply_reference_silu_rmsnorm_feedback(
    const std::vector<float>& logits,
    std::vector<float>& activation,
    std::uint32_t cols,
    float epsilon) {
  if (logits.empty()) {
    return;
  }
  float sum_sq = 0.0f;
  for (const float value : logits) {
    sum_sq += value * value;
  }
  const float mean_sq =
      sum_sq / static_cast<float>(static_cast<std::uint32_t>(logits.size()));
  const float inv_rms = 1.0f / std::sqrt(mean_sq + epsilon);
  const auto rows = static_cast<std::uint32_t>(logits.size());
  for (std::uint32_t col = 0; col < cols; ++col) {
    const float normalized = logits[col % rows] * inv_rms;
    const float sigmoid = 1.0f / (1.0f + std::exp(-normalized));
    activation[col] = normalized * sigmoid;
  }
}

void apply_reference_rmsnorm_feedback_with_silu_cache(
    const std::vector<float>& logits,
    std::vector<float>& activation,
    std::vector<float>& silu_cache,
    std::uint32_t cols,
    float epsilon) {
  if (logits.empty()) {
    return;
  }
  if (silu_cache.size() < cols) {
    silu_cache.resize(cols);
  }
  float sum_sq = 0.0f;
  for (const float value : logits) {
    sum_sq += value * value;
  }
  const float mean_sq =
      sum_sq / static_cast<float>(static_cast<std::uint32_t>(logits.size()));
  const float inv_rms = 1.0f / std::sqrt(mean_sq + epsilon);
  const auto rows = static_cast<std::uint32_t>(logits.size());
  for (std::uint32_t col = 0; col < cols; ++col) {
    const float normalized = logits[col % rows] * inv_rms;
    const float sigmoid = 1.0f / (1.0f + std::exp(-normalized));
    activation[col] = normalized;
    silu_cache[col] = normalized * sigmoid;
  }
}

void apply_reference_gated_silu_rmsnorm_feedback(
    const std::vector<float>& logits,
    const std::vector<float>& silu_cache,
    std::vector<float>& activation,
    std::uint32_t cols,
    float epsilon) {
  if (logits.empty() || silu_cache.empty()) {
    return;
  }
  float sum_sq = 0.0f;
  for (const float value : logits) {
    sum_sq += value * value;
  }
  const float mean_sq =
      sum_sq / static_cast<float>(static_cast<std::uint32_t>(logits.size()));
  const float inv_rms = 1.0f / std::sqrt(mean_sq + epsilon);
  const auto rows = static_cast<std::uint32_t>(logits.size());
  const auto cache_cols = static_cast<std::uint32_t>(silu_cache.size());
  for (std::uint32_t col = 0; col < cols; ++col) {
    const float normalized_up = logits[col % rows] * inv_rms;
    activation[col] = silu_cache[col % cache_cols] * normalized_up;
  }
}

void apply_reference_rmsnorm_cache(
    const std::vector<float>& logits,
    std::vector<float>& cache,
    std::uint32_t cache_cols,
    float epsilon,
    bool apply_silu,
    bool accumulate) {
  if (logits.empty()) {
    return;
  }
  if (cache.size() < cache_cols) {
    cache.resize(cache_cols, 0.0f);
  }
  float sum_sq = 0.0f;
  for (const float value : logits) {
    sum_sq += value * value;
  }
  const float mean_sq =
      sum_sq / static_cast<float>(static_cast<std::uint32_t>(logits.size()));
  const float inv_rms = 1.0f / std::sqrt(mean_sq + epsilon);
  const auto rows = static_cast<std::uint32_t>(logits.size());
  for (std::uint32_t col = 0; col < cache_cols; ++col) {
    float value = logits[col % rows] * inv_rms;
    if (apply_silu) {
      const float sigmoid = 1.0f / (1.0f + std::exp(-value));
      value *= sigmoid;
    }
    cache[col] = accumulate ? cache[col] + value : value;
  }
}

void apply_reference_gated_cache_product(
    const std::vector<float>& gate_cache,
    const std::vector<float>& up_cache,
    std::vector<float>& activation,
    std::uint32_t cols) {
  if (gate_cache.empty() || up_cache.empty()) {
    return;
  }
  const auto gate_cols = static_cast<std::uint32_t>(gate_cache.size());
  const auto up_cols = static_cast<std::uint32_t>(up_cache.size());
  for (std::uint32_t col = 0; col < cols; ++col) {
    activation[col] = gate_cache[col % gate_cols] * up_cache[col % up_cols];
  }
}

void apply_reference_ssm_recurrent_state(
    const std::vector<float>& logits,
    std::vector<float>& state,
    std::vector<float>& activation,
    std::uint32_t cols) {
  if (logits.empty() || cols == 0) {
    return;
  }
  if (activation.size() < cols) {
    activation.resize(cols, 0.0f);
  }
  if (state.size() < cols) {
    state.resize(cols, 0.0f);
  }
  const auto rows = static_cast<std::uint32_t>(logits.size());
  for (std::uint32_t col = 0; col < cols; ++col) {
    const float active = activation[col];
    const float next_state =
        state[col] * kSsmStateDecay +
        logits[col % rows] * kSsmStateLogitScale +
        active * kSsmStateActivationScale;
    state[col] = next_state;
    activation[col] = active + next_state * kSsmStateOutputScale;
  }
}

void apply_reference_ssm_scan_scratch(
    const std::vector<float>& logits,
    std::vector<float>& state,
    std::vector<float>& scratch,
    std::vector<float>& activation,
    std::uint32_t cols,
    std::uint32_t scratch_values) {
  if (logits.empty() || cols == 0) {
    return;
  }
  if (activation.size() < cols) {
    activation.resize(cols, 0.0f);
  }
  if (state.size() < cols) {
    state.resize(cols, 0.0f);
  }
  if (scratch_values == 0) {
    scratch_values = cols;
  }
  if (scratch.size() < scratch_values) {
    scratch.resize(scratch_values, 0.0f);
  }
  const auto rows = static_cast<std::uint32_t>(logits.size());
  for (std::uint32_t col = 0; col < cols; ++col) {
    const auto state_index = col % static_cast<std::uint32_t>(state.size());
    const auto scratch_index = col % scratch_values;
    const float active = activation[col];
    const float previous_state = state[state_index];
    const float logit = logits[col % rows];
    const float scan =
        scratch[scratch_index] * kSsmStateDecay +
        previous_state * kSsmScanScratchScale +
        logit * kSsmStateLogitScale +
        active * kSsmStateActivationScale;
    scratch[scratch_index] = scan;
    const float next_state =
        previous_state * kSsmStateDecay +
        scan * kSsmScanScratchScale +
        logit * kSsmStateLogitScale;
    state[state_index] = next_state;
    activation[col] =
        active + (next_state + scan * 0.25f) * kSsmStateOutputScale;
  }
}

void apply_reference_ssm_selective_scan(
    const std::vector<float>& logits,
    std::vector<float>& state,
    std::vector<float>& scratch,
    std::vector<float>& activation,
    std::uint32_t cols,
    std::uint32_t scratch_values) {
  if (logits.empty() || cols == 0) {
    return;
  }
  if (activation.size() < cols) {
    activation.resize(cols, 0.0f);
  }
  if (state.size() < cols) {
    state.resize(cols, 0.0f);
  }
  if (scratch_values == 0) {
    scratch_values = cols;
  }
  if (scratch.size() < scratch_values) {
    scratch.resize(scratch_values, 0.0f);
  }
  const auto rows = static_cast<std::uint32_t>(logits.size());
  for (std::uint32_t col = 0; col < cols; ++col) {
    const auto state_index = col % static_cast<std::uint32_t>(state.size());
    const auto scratch_index = col % scratch_values;
    const float active = activation[col];
    const float previous_state = state[state_index];
    const float previous_scan = scratch[scratch_index];
    const float logit = logits[col % rows];
    const float gate_input =
        logit * kSsmSelectiveGateLogitScale +
        active * kSsmSelectiveGateActivationScale +
        previous_scan * kSsmSelectiveGateScratchScale;
    const float gate = 1.0f / (1.0f + std::exp(-gate_input));
    const float candidate = std::tanh(
        logit * kSsmStateLogitScale +
        active * kSsmStateActivationScale +
        previous_state * kSsmScanScratchScale +
        previous_scan * kSsmScanScratchScale);
    const float scan =
        previous_scan * kSsmStateDecay +
        candidate * gate +
        previous_state * kSsmScanScratchScale;
    scratch[scratch_index] = scan;
    const float keep = kSsmStateDecay + (1.0f - kSsmStateDecay) * (1.0f - gate);
    const float next_state =
        previous_state * keep +
        candidate * gate +
        scan * kSsmScanScratchScale;
    state[state_index] = next_state;
    activation[col] =
        active + (next_state + scan * 0.25f + gate * 0.03125f) *
        kSsmStateOutputScale;
  }
}

void apply_reference_ssm_source_parameter_cache(
    const std::vector<float>& logits,
    const std::vector<float>& activation,
    std::vector<float>& scratch,
    std::uint32_t state_values,
    std::uint32_t parameter_slot) {
  if (logits.empty() || activation.empty() || state_values == 0 ||
      parameter_slot > 1u) {
    return;
  }
  const std::size_t required =
      static_cast<std::size_t>(state_values) * 2u;
  if (scratch.size() < required) {
    scratch.resize(required, 0.0f);
  }
  const auto rows = static_cast<std::uint32_t>(logits.size());
  const auto activation_cols = static_cast<std::uint32_t>(activation.size());
  const std::uint32_t offset = parameter_slot * state_values;
  for (std::uint32_t index = 0; index < state_values; ++index) {
    const float slot_bias =
        parameter_slot == 0u ? -0.03125f : 0.03125f;
    scratch[offset + index] = std::tanh(
        logits[index % rows] * kSsmSourceParamCacheLogitScale +
        activation[index % activation_cols] *
            kSsmSourceParamCacheActivationScale +
        slot_bias);
  }
}

void apply_reference_ssm_source_parameterized_scan(
    const std::vector<float>& logits,
    std::vector<float>& state,
    std::vector<float>& scratch,
    std::vector<float>& activation,
    std::uint32_t cols) {
  if (logits.empty() || cols == 0) {
    return;
  }
  if (activation.size() < cols) {
    activation.resize(cols, 0.0f);
  }
  if (state.size() < cols) {
    state.resize(cols, 0.0f);
  }
  const auto state_values = static_cast<std::uint32_t>(state.size());
  const std::size_t required =
      static_cast<std::size_t>(state_values) * 2u;
  if (scratch.size() < required) {
    scratch.resize(required, 0.0f);
  }
  const auto rows = static_cast<std::uint32_t>(logits.size());
  for (std::uint32_t col = 0; col < cols; ++col) {
    const auto state_index = col % state_values;
    const float alpha = scratch[state_index];
    const float beta = scratch[state_values + state_index];
    const float active = activation[col];
    const float previous_state = state[state_index];
    const float logit = logits[col % rows];
    const float gate_input =
        logit * kSsmSourceParamGateLogitScale +
        active * kSsmSourceParamGateActivationScale +
        alpha * kSsmSourceParamAlphaScale +
        beta * kSsmSourceParamBetaScale;
    const float gate = 1.0f / (1.0f + std::exp(-gate_input));
    const float candidate = std::tanh(
        logit * kSsmStateLogitScale +
        active * kSsmStateActivationScale +
        alpha * kSsmSourceParamAlphaScale -
        beta * kSsmSourceParamBetaScale +
        previous_state * kSsmScanScratchScale);
    const float scan =
        beta * kSsmStateDecay +
        alpha * kSsmScanScratchScale +
        candidate * gate;
    scratch[state_values + state_index] = scan;
    const float next_state =
        previous_state * kSsmStateDecay +
        candidate * gate +
        scan * kSsmScanScratchScale;
    state[state_index] = next_state;
    activation[col] =
        active + (next_state + scan * 0.25f + alpha * 0.03125f) *
        kSsmStateOutputScale;
  }
}

void apply_reference_output_final_token(
    const std::vector<float>& logits,
    std::uint32_t& token) {
  if (logits.empty()) {
    return;
  }
  token = (argmax_token(logits) + kOutputPhaseTokenOffset) % kOutputPhaseVocabSize;
}

float reference_attention_qkv_projection(
    float logit,
    float active,
    std::uint32_t component,
    std::uint32_t row) {
  const float row_bias =
      static_cast<float>((row & 0x1fu) + 1u) * 0.0001220703125f;
  if (component == 0u) {
    return logit * 0.125f + active * 0.03125f + row_bias;
  }
  if (component == 1u) {
    return logit * 0.0625f - active * 0.015625f - row_bias;
  }
  return logit * 0.03125f + active * 0.046875f + row_bias * 0.5f;
}

std::pair<float, float> reference_rope_rotate_pair(
    float in_even,
    float in_odd,
    std::uint32_t context,
    std::uint32_t pair,
    float theta_scale) {
  const float angle =
      static_cast<float>(context) * static_cast<float>(pair + 1u) *
      theta_scale;
  const float sin_value = std::sin(angle);
  const float cos_value = std::cos(angle);
  return {
      in_even * cos_value - in_odd * sin_value,
      in_even * sin_value + in_odd * cos_value,
  };
}

void apply_reference_attention_qkv_scratch(
    const std::vector<float>& logits,
    std::vector<float>& scratch,
    std::vector<float>& activation,
    std::uint32_t target_cols) {
  if (logits.empty() || target_cols == 0) {
    return;
  }
  if (activation.size() < target_cols) {
    activation.resize(target_cols, 0.0f);
  }
  const auto rows = static_cast<std::uint32_t>(logits.size());
  const std::uint64_t required_values =
      static_cast<std::uint64_t>(rows) * kAttentionQkvComponents;
  if (required_values > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("reference attention QKV scratch is too large");
  }
  if (scratch.size() < required_values) {
    scratch.resize(static_cast<std::size_t>(required_values), 0.0f);
  }
  for (std::uint32_t index = 0; index < required_values; ++index) {
    const std::uint32_t row = index % rows;
    const std::uint32_t component = index / rows;
    scratch[index] =
        scratch[index] * 0.25f +
        reference_attention_qkv_projection(logits[row], 0.0f, component, row);
  }
  for (std::uint32_t col = 0; col < target_cols; ++col) {
    const std::uint32_t row = col % rows;
    const float active = activation[col];
    const float q = reference_attention_qkv_projection(logits[row], active, 0u, row);
    const float k = reference_attention_qkv_projection(logits[row], active, 1u, row);
    const float v = reference_attention_qkv_projection(logits[row], active, 2u, row);
    activation[col] =
        active + (q * 0.0625f + k * 0.03125f + v * 0.015625f) *
        kAttentionQkvResidualScale;
  }
}

void apply_reference_attention_qkv_reduce(
    std::vector<float>& scratch,
    std::vector<float>& activation,
    std::uint32_t target_cols) {
  if (target_cols == 0 || scratch.size() < kAttentionQkvComponents) {
    return;
  }
  if (activation.size() < target_cols) {
    activation.resize(target_cols, 0.0f);
  }
  const auto qkv_rows =
      static_cast<std::uint32_t>(scratch.size() / kAttentionQkvComponents);
  if (qkv_rows == 0) {
    return;
  }
  for (std::uint32_t col = 0; col < target_cols; ++col) {
    const std::uint32_t row = col % qkv_rows;
    const std::uint32_t k_row =
        (row + ((col * 7u + 3u) % qkv_rows)) % qkv_rows;
    const std::uint32_t v_row = (row * 3u + col) % qkv_rows;
    const float q = scratch[row];
    const float k = scratch[qkv_rows + k_row];
    const float v = scratch[2u * qkv_rows + v_row];
    const float score = q * k;
    const float bounded_score = score / (1.0f + std::fabs(score));
    const float residual =
        v * 0.0625f +
        q * 0.015625f +
        k * 0.0078125f +
        bounded_score * 0.03125f;
    activation[col] += residual * kAttentionQkvReduceScale;
  }
}

void apply_reference_attention_qkv_window(
    std::vector<float>& scratch,
    std::vector<float>& activation,
    std::uint32_t target_cols) {
  if (target_cols == 0 || scratch.size() < kAttentionQkvComponents ||
      kAttentionQkvWindowSize == 0) {
    return;
  }
  if (activation.size() < target_cols) {
    activation.resize(target_cols, 0.0f);
  }
  const auto qkv_rows =
      static_cast<std::uint32_t>(scratch.size() / kAttentionQkvComponents);
  if (qkv_rows == 0) {
    return;
  }
  const std::uint32_t window =
      std::min<std::uint32_t>(kAttentionQkvWindowSize, qkv_rows);
  for (std::uint32_t col = 0; col < target_cols; ++col) {
    const std::uint32_t row = col % qkv_rows;
    const float q = scratch[row];
    float max_score = -std::numeric_limits<float>::infinity();
    for (std::uint32_t item = 0; item < window; ++item) {
      const std::uint32_t key_row =
          (row + ((col * 5u + item * 11u + 1u) % qkv_rows)) % qkv_rows;
      const float k = scratch[qkv_rows + key_row];
      const float score = q * k * kAttentionQkvWindowSoftmaxScale;
      max_score = std::max(max_score, score);
    }
    float denom = 0.0f;
    float weighted_v = 0.0f;
    float weighted_score = 0.0f;
    for (std::uint32_t item = 0; item < window; ++item) {
      const std::uint32_t key_row =
          (row + ((col * 5u + item * 11u + 1u) % qkv_rows)) % qkv_rows;
      const std::uint32_t value_row = (key_row * 3u + col + item) % qkv_rows;
      const float k = scratch[qkv_rows + key_row];
      const float score = q * k * kAttentionQkvWindowSoftmaxScale;
      const float weight = std::exp(score - max_score);
      denom += weight;
      weighted_v += weight * scratch[2u * qkv_rows + value_row];
      weighted_score += weight * score;
    }
    if (denom <= 0.0f) {
      continue;
    }
    const float inv_denom = 1.0f / denom;
    const float context = weighted_v * inv_denom;
    const float mean_score = weighted_score * inv_denom;
    const float bounded_score = mean_score / (1.0f + std::fabs(mean_score));
    const float residual =
        context * 0.078125f +
        q * 0.015625f +
        bounded_score * 0.015625f;
    activation[col] += residual * kAttentionQkvWindowResidualScale;
  }
}

void apply_reference_attention_qkv_head_window(
    std::vector<float>& scratch,
    std::vector<float>& activation,
    std::uint32_t target_cols) {
  if (target_cols == 0 || scratch.size() < kAttentionQkvComponents ||
      kAttentionQkvHeadWindowHeadCount == 0 ||
      kAttentionQkvHeadWindowSize == 0) {
    return;
  }
  if (activation.size() < target_cols) {
    activation.resize(target_cols, 0.0f);
  }
  const auto qkv_rows =
      static_cast<std::uint32_t>(scratch.size() / kAttentionQkvComponents);
  if (qkv_rows == 0) {
    return;
  }
  const std::uint32_t active_heads =
      std::min<std::uint32_t>(kAttentionQkvHeadWindowHeadCount, qkv_rows);
  const std::uint32_t rows_per_head = qkv_rows / active_heads;
  if (rows_per_head == 0) {
    return;
  }
  const std::uint32_t window =
      std::min<std::uint32_t>(kAttentionQkvHeadWindowSize, rows_per_head);
  for (std::uint32_t col = 0; col < target_cols; ++col) {
    const std::uint32_t head = col % active_heads;
    const std::uint32_t local_row = (col / active_heads) % rows_per_head;
    const std::uint32_t head_base = head * rows_per_head;
    const std::uint32_t query_row = head_base + local_row;
    const float q = scratch[query_row];
    float max_score = -std::numeric_limits<float>::infinity();
    for (std::uint32_t item = 0; item < window; ++item) {
      const std::uint32_t key_local =
          (local_row + rows_per_head - item) % rows_per_head;
      const std::uint32_t key_row = head_base + key_local;
      const float k = scratch[qkv_rows + key_row];
      const float score = q * k * kAttentionQkvHeadWindowSoftmaxScale;
      max_score = std::max(max_score, score);
    }
    float denom = 0.0f;
    float weighted_v = 0.0f;
    float weighted_score = 0.0f;
    for (std::uint32_t item = 0; item < window; ++item) {
      const std::uint32_t key_local =
          (local_row + rows_per_head - item) % rows_per_head;
      const std::uint32_t key_row = head_base + key_local;
      const float k = scratch[qkv_rows + key_row];
      const float score = q * k * kAttentionQkvHeadWindowSoftmaxScale;
      const float weight = std::exp(score - max_score);
      denom += weight;
      weighted_v += weight * scratch[2u * qkv_rows + key_row];
      weighted_score += weight * score;
    }
    if (denom <= 0.0f) {
      continue;
    }
    const float inv_denom = 1.0f / denom;
    const float context = weighted_v * inv_denom;
    const float mean_score = weighted_score * inv_denom;
    const float bounded_score = mean_score / (1.0f + std::fabs(mean_score));
    const float head_bias =
        static_cast<float>((head & 0x7u) + 1u) * 0.0009765625f;
    const float residual =
        context * 0.078125f +
        q * 0.015625f +
        bounded_score * 0.015625f +
        head_bias;
    activation[col] += residual * kAttentionQkvHeadWindowResidualScale;
  }
}

void apply_reference_attention_qkv_head_dim_window(
    std::vector<float>& scratch,
    std::vector<float>& activation,
    std::uint32_t target_cols) {
  if (target_cols == 0 || scratch.size() < kAttentionQkvComponents ||
      kAttentionQkvHeadDimWindowHeadCount == 0 ||
      kAttentionQkvHeadDimWindowHeadDim == 0 ||
      kAttentionQkvHeadDimWindowSize == 0) {
    return;
  }
  if (activation.size() < target_cols) {
    activation.resize(target_cols, 0.0f);
  }
  const auto qkv_rows =
      static_cast<std::uint32_t>(scratch.size() / kAttentionQkvComponents);
  if (qkv_rows == 0) {
    return;
  }
  const std::uint32_t possible_heads =
      qkv_rows / kAttentionQkvHeadDimWindowHeadDim;
  if (possible_heads == 0) {
    return;
  }
  const std::uint32_t active_heads =
      std::min<std::uint32_t>(kAttentionQkvHeadDimWindowHeadCount, possible_heads);
  const std::uint32_t rows_per_head = qkv_rows / active_heads;
  const std::uint32_t contexts_per_head =
      rows_per_head / kAttentionQkvHeadDimWindowHeadDim;
  if (contexts_per_head == 0) {
    return;
  }
  const std::uint32_t window =
      std::min<std::uint32_t>(
          kAttentionQkvHeadDimWindowSize,
          contexts_per_head);
  for (std::uint32_t col = 0; col < target_cols; ++col) {
    const std::uint32_t head = col % active_heads;
    const std::uint32_t dim =
        (col / active_heads) % kAttentionQkvHeadDimWindowHeadDim;
    const std::uint32_t query_context =
        (col / (active_heads * kAttentionQkvHeadDimWindowHeadDim)) %
        contexts_per_head;
    const std::uint32_t head_base = head * rows_per_head;
    const std::uint32_t query_base =
        head_base + query_context * kAttentionQkvHeadDimWindowHeadDim;
    const std::uint32_t query_row = query_base + dim;
    float max_score = -std::numeric_limits<float>::infinity();
    for (std::uint32_t item = 0; item < window; ++item) {
      const std::uint32_t key_context =
          (query_context + contexts_per_head - item) % contexts_per_head;
      const std::uint32_t key_base =
          head_base + key_context * kAttentionQkvHeadDimWindowHeadDim;
      float dot = 0.0f;
      for (std::uint32_t lane = 0;
           lane < kAttentionQkvHeadDimWindowHeadDim;
           ++lane) {
        dot += scratch[query_base + lane] *
            scratch[qkv_rows + key_base + lane];
      }
      const float score = dot * kAttentionQkvHeadDimWindowSoftmaxScale;
      max_score = std::max(max_score, score);
    }
    float denom = 0.0f;
    float weighted_v = 0.0f;
    float weighted_score = 0.0f;
    for (std::uint32_t item = 0; item < window; ++item) {
      const std::uint32_t key_context =
          (query_context + contexts_per_head - item) % contexts_per_head;
      const std::uint32_t key_base =
          head_base + key_context * kAttentionQkvHeadDimWindowHeadDim;
      float dot = 0.0f;
      for (std::uint32_t lane = 0;
           lane < kAttentionQkvHeadDimWindowHeadDim;
           ++lane) {
        dot += scratch[query_base + lane] *
            scratch[qkv_rows + key_base + lane];
      }
      const float score = dot * kAttentionQkvHeadDimWindowSoftmaxScale;
      const float weight = std::exp(score - max_score);
      denom += weight;
      weighted_v += weight * scratch[2u * qkv_rows + key_base + dim];
      weighted_score += weight * score;
    }
    if (denom <= 0.0f) {
      continue;
    }
    const float inv_denom = 1.0f / denom;
    const float context = weighted_v * inv_denom;
    const float mean_score = weighted_score * inv_denom;
    const float bounded_score = mean_score / (1.0f + std::fabs(mean_score));
    const float head_bias =
        static_cast<float>((head & 0x7u) + 1u) * 0.0009765625f;
    const float dim_bias =
        static_cast<float>((dim & 0xfu) + 1u) * 0.0001220703125f;
    const float residual =
        context * 0.078125f +
        scratch[query_row] * 0.015625f +
        bounded_score * 0.015625f +
        head_bias +
        dim_bias;
    activation[col] += residual * kAttentionQkvHeadDimWindowResidualScale;
  }
}

void apply_reference_attention_qkv_head_tile_window(
    std::vector<float>& scratch,
    std::vector<float>& activation,
    std::uint32_t target_cols) {
  if (target_cols == 0 || scratch.size() < kAttentionQkvComponents ||
      kAttentionQkvHeadTileWindowHeadCount == 0 ||
      kAttentionQkvHeadTileWindowHeadDim == 0 ||
      kAttentionQkvHeadTileWindowSize == 0) {
    return;
  }
  if (activation.size() < target_cols) {
    activation.resize(target_cols, 0.0f);
  }
  const auto qkv_rows =
      static_cast<std::uint32_t>(scratch.size() / kAttentionQkvComponents);
  const std::uint32_t possible_heads =
      qkv_rows / kAttentionQkvHeadTileWindowHeadDim;
  if (possible_heads == 0) {
    return;
  }
  const std::uint32_t active_heads =
      std::min<std::uint32_t>(kAttentionQkvHeadTileWindowHeadCount, possible_heads);
  const std::uint32_t rows_per_head = qkv_rows / active_heads;
  const std::uint32_t contexts_per_head =
      rows_per_head / kAttentionQkvHeadTileWindowHeadDim;
  if (contexts_per_head == 0) {
    return;
  }
  const std::uint32_t contexts_for_activation =
      (target_cols + active_heads * kAttentionQkvHeadTileWindowHeadDim - 1u) /
      (active_heads * kAttentionQkvHeadTileWindowHeadDim);
  const std::uint32_t active_contexts =
      std::min<std::uint32_t>(contexts_for_activation, contexts_per_head);
  const std::uint32_t window =
      std::min<std::uint32_t>(
          kAttentionQkvHeadTileWindowSize,
          contexts_per_head);
  for (std::uint32_t query_context = 0;
       query_context < active_contexts;
       ++query_context) {
    for (std::uint32_t head = 0; head < active_heads; ++head) {
      const std::uint32_t head_base = head * rows_per_head;
      const std::uint32_t query_base =
          head_base + query_context * kAttentionQkvHeadTileWindowHeadDim;
      std::array<float, kAttentionQkvHeadTileWindowSize> scores{};
      std::array<float, kAttentionQkvHeadTileWindowSize> weights{};
      float max_score = -std::numeric_limits<float>::infinity();
      for (std::uint32_t item = 0; item < window; ++item) {
        const std::uint32_t key_context =
            (query_context + contexts_per_head - item) % contexts_per_head;
        const std::uint32_t key_base =
            head_base + key_context * kAttentionQkvHeadTileWindowHeadDim;
        float dot = 0.0f;
        for (std::uint32_t lane = 0;
             lane < kAttentionQkvHeadTileWindowHeadDim;
             ++lane) {
          dot += scratch[query_base + lane] *
              scratch[qkv_rows + key_base + lane];
        }
        const float score = dot * kAttentionQkvHeadTileWindowSoftmaxScale;
        scores[item] = score;
        max_score = std::max(max_score, score);
      }
      float denom = 0.0f;
      float weighted_score = 0.0f;
      for (std::uint32_t item = 0; item < window; ++item) {
        const float weight = std::exp(scores[item] - max_score);
        weights[item] = weight;
        denom += weight;
        weighted_score += weight * scores[item];
      }
      if (denom <= 0.0f) {
        continue;
      }
      const float bounded_score =
          (weighted_score / denom) /
          (1.0f + std::fabs(weighted_score / denom));
      for (std::uint32_t dim = 0;
           dim < kAttentionQkvHeadTileWindowHeadDim;
           ++dim) {
        const std::uint32_t col =
            head + active_heads *
                (dim + kAttentionQkvHeadTileWindowHeadDim * query_context);
        if (col >= target_cols) {
          continue;
        }
        float weighted_v = 0.0f;
        for (std::uint32_t item = 0; item < window; ++item) {
          const std::uint32_t key_context =
              (query_context + contexts_per_head - item) % contexts_per_head;
          const std::uint32_t key_base =
              head_base + key_context * kAttentionQkvHeadTileWindowHeadDim;
          weighted_v += weights[item] * scratch[2u * qkv_rows + key_base + dim];
        }
        const float context = weighted_v / denom;
        const float head_bias =
            static_cast<float>((head & 0x7u) + 1u) * 0.0009765625f;
        const float dim_bias =
            static_cast<float>((dim & 0xfu) + 1u) * 0.0001220703125f;
        const float residual =
            context * 0.078125f +
            scratch[query_base + dim] * 0.015625f +
            bounded_score * 0.015625f +
            head_bias +
            dim_bias;
        activation[col] += residual * kAttentionQkvHeadTileWindowResidualScale;
      }
    }
  }
}

void apply_reference_attention_qkv_head_group_window(
    std::vector<float>& scratch,
    std::vector<float>& activation,
    std::uint32_t target_cols) {
  if (target_cols == 0 || scratch.size() < kAttentionQkvComponents ||
      kAttentionQkvHeadGroupWindowHeadCount == 0 ||
      kAttentionQkvHeadGroupWindowHeadDim == 0 ||
      kAttentionQkvHeadGroupWindowSize == 0 ||
      kAttentionQkvHeadGroupWindowContextsPerBlock == 0) {
    return;
  }
  if (activation.size() < target_cols) {
    activation.resize(target_cols, 0.0f);
  }
  const auto qkv_rows =
      static_cast<std::uint32_t>(scratch.size() / kAttentionQkvComponents);
  const std::uint32_t possible_heads =
      qkv_rows / kAttentionQkvHeadGroupWindowHeadDim;
  if (possible_heads == 0) {
    return;
  }
  const std::uint32_t active_heads =
      std::min<std::uint32_t>(
          kAttentionQkvHeadGroupWindowHeadCount,
          possible_heads);
  const std::uint32_t rows_per_head = qkv_rows / active_heads;
  const std::uint32_t contexts_per_head =
      rows_per_head / kAttentionQkvHeadGroupWindowHeadDim;
  if (contexts_per_head == 0) {
    return;
  }
  const std::uint32_t contexts_for_activation =
      (target_cols + active_heads * kAttentionQkvHeadGroupWindowHeadDim - 1u) /
      (active_heads * kAttentionQkvHeadGroupWindowHeadDim);
  const std::uint32_t active_contexts =
      std::min<std::uint32_t>(contexts_for_activation, contexts_per_head);
  const std::uint32_t window =
      std::min<std::uint32_t>(
          kAttentionQkvHeadGroupWindowSize,
          contexts_per_head);
  for (std::uint32_t context_group = 0;
       context_group < active_contexts;
       context_group += kAttentionQkvHeadGroupWindowContextsPerBlock) {
    const std::uint32_t contexts_in_group =
        std::min<std::uint32_t>(
            kAttentionQkvHeadGroupWindowContextsPerBlock,
            active_contexts - context_group);
    for (std::uint32_t head = 0; head < active_heads; ++head) {
      const std::uint32_t head_base = head * rows_per_head;
      for (std::uint32_t slot = 0; slot < contexts_in_group; ++slot) {
        const std::uint32_t query_context = context_group + slot;
        const std::uint32_t query_base =
            head_base + query_context * kAttentionQkvHeadGroupWindowHeadDim;
        std::array<float, kAttentionQkvHeadGroupWindowSize> scores{};
        std::array<float, kAttentionQkvHeadGroupWindowSize> weights{};
        float max_score = -std::numeric_limits<float>::infinity();
        for (std::uint32_t item = 0; item < window; ++item) {
          const std::uint32_t key_context =
              (query_context + contexts_per_head - item) % contexts_per_head;
          const std::uint32_t key_base =
              head_base + key_context * kAttentionQkvHeadGroupWindowHeadDim;
          float dot = 0.0f;
          for (std::uint32_t lane = 0;
               lane < kAttentionQkvHeadGroupWindowHeadDim;
               ++lane) {
            dot += scratch[query_base + lane] *
                scratch[qkv_rows + key_base + lane];
          }
          const float score = dot * kAttentionQkvHeadGroupWindowSoftmaxScale;
          scores[item] = score;
          max_score = std::max(max_score, score);
        }
        float denom = 0.0f;
        float weighted_score = 0.0f;
        for (std::uint32_t item = 0; item < window; ++item) {
          const float weight = std::exp(scores[item] - max_score);
          weights[item] = weight;
          denom += weight;
          weighted_score += weight * scores[item];
        }
        if (denom <= 0.0f) {
          continue;
        }
        const float mean_score = weighted_score / denom;
        const float bounded_score =
            mean_score / (1.0f + std::fabs(mean_score));
        for (std::uint32_t dim = 0;
             dim < kAttentionQkvHeadGroupWindowHeadDim;
             ++dim) {
          const std::uint32_t col =
              head + active_heads *
                  (dim + kAttentionQkvHeadGroupWindowHeadDim * query_context);
          if (col >= target_cols) {
            continue;
          }
          float weighted_v = 0.0f;
          for (std::uint32_t item = 0; item < window; ++item) {
            const std::uint32_t key_context =
                (query_context + contexts_per_head - item) % contexts_per_head;
            const std::uint32_t key_base =
                head_base + key_context * kAttentionQkvHeadGroupWindowHeadDim;
            weighted_v += weights[item] *
                scratch[2u * qkv_rows + key_base + dim];
          }
          const float context = weighted_v / denom;
          const float head_bias =
              static_cast<float>((head & 0x7u) + 1u) * 0.0009765625f;
          const float dim_bias =
              static_cast<float>((dim & 0xfu) + 1u) * 0.0001220703125f;
          const float residual =
              context * 0.078125f +
              scratch[query_base + dim] * 0.015625f +
              bounded_score * 0.015625f +
              head_bias +
              dim_bias;
          activation[col] += residual *
              kAttentionQkvHeadGroupWindowResidualScale;
        }
      }
    }
  }
}

void apply_reference_attention_qkv_head_group_rope_window(
    std::vector<float>& scratch,
    std::vector<float>& activation,
    std::uint32_t target_cols) {
  if (target_cols == 0 || scratch.size() < kAttentionQkvComponents ||
      kAttentionQkvHeadGroupRopeWindowHeadCount == 0 ||
      kAttentionQkvHeadGroupRopeWindowHeadDim == 0 ||
      (kAttentionQkvHeadGroupRopeWindowHeadDim & 1u) != 0 ||
      kAttentionQkvHeadGroupRopeWindowSize == 0 ||
      kAttentionQkvHeadGroupRopeWindowContextsPerBlock == 0) {
    return;
  }
  if (activation.size() < target_cols) {
    activation.resize(target_cols, 0.0f);
  }
  const auto qkv_rows =
      static_cast<std::uint32_t>(scratch.size() / kAttentionQkvComponents);
  const std::uint32_t possible_heads =
      qkv_rows / kAttentionQkvHeadGroupRopeWindowHeadDim;
  if (possible_heads == 0) {
    return;
  }
  const std::uint32_t active_heads =
      std::min<std::uint32_t>(
          kAttentionQkvHeadGroupRopeWindowHeadCount,
          possible_heads);
  const std::uint32_t rows_per_head = qkv_rows / active_heads;
  const std::uint32_t contexts_per_head =
      rows_per_head / kAttentionQkvHeadGroupRopeWindowHeadDim;
  if (contexts_per_head == 0) {
    return;
  }
  const std::uint32_t contexts_for_activation =
      (target_cols + active_heads * kAttentionQkvHeadGroupRopeWindowHeadDim -
       1u) /
      (active_heads * kAttentionQkvHeadGroupRopeWindowHeadDim);
  const std::uint32_t active_contexts =
      std::min<std::uint32_t>(contexts_for_activation, contexts_per_head);
  const std::uint32_t window =
      std::min<std::uint32_t>(
          kAttentionQkvHeadGroupRopeWindowSize,
          contexts_per_head);
  for (std::uint32_t context_group = 0;
       context_group < active_contexts;
       context_group += kAttentionQkvHeadGroupRopeWindowContextsPerBlock) {
    const std::uint32_t contexts_in_group =
        std::min<std::uint32_t>(
            kAttentionQkvHeadGroupRopeWindowContextsPerBlock,
            active_contexts - context_group);
    for (std::uint32_t head = 0; head < active_heads; ++head) {
      const std::uint32_t head_base = head * rows_per_head;
      for (std::uint32_t slot = 0; slot < contexts_in_group; ++slot) {
        const std::uint32_t query_context = context_group + slot;
        const std::uint32_t query_base =
            head_base + query_context *
                kAttentionQkvHeadGroupRopeWindowHeadDim;
        std::array<float, kAttentionQkvHeadGroupRopeWindowSize> scores{};
        std::array<float, kAttentionQkvHeadGroupRopeWindowSize> weights{};
        float max_score = -std::numeric_limits<float>::infinity();
        for (std::uint32_t item = 0; item < window; ++item) {
          const std::uint32_t key_context =
              (query_context + contexts_per_head - item) % contexts_per_head;
          const std::uint32_t key_base =
              head_base + key_context *
                  kAttentionQkvHeadGroupRopeWindowHeadDim;
          float dot = 0.0f;
          for (std::uint32_t lane = 0;
               lane < kAttentionQkvHeadGroupRopeWindowHeadDim;
               lane += 2u) {
            const std::uint32_t pair = lane >> 1u;
            const auto q = reference_rope_rotate_pair(
                scratch[query_base + lane],
                scratch[query_base + lane + 1u],
                query_context,
                pair,
                kAttentionQkvHeadGroupRopeWindowThetaScale);
            const auto k = reference_rope_rotate_pair(
                scratch[qkv_rows + key_base + lane],
                scratch[qkv_rows + key_base + lane + 1u],
                key_context,
                pair,
                kAttentionQkvHeadGroupRopeWindowThetaScale);
            dot += q.first * k.first + q.second * k.second;
          }
          const float score =
              dot * kAttentionQkvHeadGroupRopeWindowSoftmaxScale;
          scores[item] = score;
          max_score = std::max(max_score, score);
        }
        float denom = 0.0f;
        float weighted_score = 0.0f;
        for (std::uint32_t item = 0; item < window; ++item) {
          const float weight = std::exp(scores[item] - max_score);
          weights[item] = weight;
          denom += weight;
          weighted_score += weight * scores[item];
        }
        if (denom <= 0.0f) {
          continue;
        }
        const float mean_score = weighted_score / denom;
        const float bounded_score =
            mean_score / (1.0f + std::fabs(mean_score));
        for (std::uint32_t dim = 0;
             dim < kAttentionQkvHeadGroupRopeWindowHeadDim;
             ++dim) {
          const std::uint32_t col =
              head + active_heads *
                  (dim + kAttentionQkvHeadGroupRopeWindowHeadDim *
                      query_context);
          if (col >= target_cols) {
            continue;
          }
          float weighted_v = 0.0f;
          for (std::uint32_t item = 0; item < window; ++item) {
            const std::uint32_t key_context =
                (query_context + contexts_per_head - item) % contexts_per_head;
            const std::uint32_t key_base =
                head_base + key_context *
                    kAttentionQkvHeadGroupRopeWindowHeadDim;
            weighted_v += weights[item] *
                scratch[2u * qkv_rows + key_base + dim];
          }
          const std::uint32_t pair_base = dim & ~1u;
          const auto q_lane_pair = reference_rope_rotate_pair(
              scratch[query_base + pair_base],
              scratch[query_base + pair_base + 1u],
              query_context,
              pair_base >> 1u,
              kAttentionQkvHeadGroupRopeWindowThetaScale);
          const float q_lane =
              (dim & 1u) == 0u ? q_lane_pair.first : q_lane_pair.second;
          const float context = weighted_v / denom;
          const float head_bias =
              static_cast<float>((head & 0x7u) + 1u) * 0.0009765625f;
          const float dim_bias =
              static_cast<float>((dim & 0xfu) + 1u) * 0.0001220703125f;
          const float residual =
              context * 0.078125f +
              q_lane * 0.015625f +
              bounded_score * 0.015625f +
              head_bias +
              dim_bias;
          activation[col] += residual *
              kAttentionQkvHeadGroupRopeWindowResidualScale;
        }
      }
    }
  }
}

void apply_reference_attention_qkv_head_group_fused(
    const std::vector<float>& logits,
    std::vector<float>& activation,
    std::uint32_t target_cols) {
  if (logits.empty() || target_cols == 0 ||
      kAttentionQkvHeadGroupFusedHeadCount == 0 ||
      kAttentionQkvHeadGroupFusedHeadDim == 0 ||
      kAttentionQkvHeadGroupFusedSize == 0 ||
      kAttentionQkvHeadGroupFusedContextsPerBlock == 0) {
    return;
  }
  if (activation.size() < target_cols) {
    activation.resize(target_cols, 0.0f);
  }
  const auto rows = static_cast<std::uint32_t>(logits.size());
  const std::uint32_t possible_heads =
      rows / kAttentionQkvHeadGroupFusedHeadDim;
  if (possible_heads == 0) {
    return;
  }
  const std::uint32_t active_heads =
      std::min<std::uint32_t>(
          kAttentionQkvHeadGroupFusedHeadCount,
          possible_heads);
  const std::uint32_t rows_per_head = rows / active_heads;
  const std::uint32_t contexts_per_head =
      rows_per_head / kAttentionQkvHeadGroupFusedHeadDim;
  if (contexts_per_head == 0) {
    return;
  }
  const std::uint32_t contexts_for_activation =
      (target_cols + active_heads * kAttentionQkvHeadGroupFusedHeadDim - 1u) /
      (active_heads * kAttentionQkvHeadGroupFusedHeadDim);
  const std::uint32_t active_contexts =
      std::min<std::uint32_t>(contexts_for_activation, contexts_per_head);
  const std::uint32_t window =
      std::min<std::uint32_t>(
          kAttentionQkvHeadGroupFusedSize,
          contexts_per_head);
  for (std::uint32_t context_group = 0;
       context_group < active_contexts;
       context_group += kAttentionQkvHeadGroupFusedContextsPerBlock) {
    const std::uint32_t contexts_in_group =
        std::min<std::uint32_t>(
            kAttentionQkvHeadGroupFusedContextsPerBlock,
            active_contexts - context_group);
    for (std::uint32_t head = 0; head < active_heads; ++head) {
      const std::uint32_t head_base = head * rows_per_head;
      for (std::uint32_t slot = 0; slot < contexts_in_group; ++slot) {
        const std::uint32_t query_context = context_group + slot;
        const std::uint32_t query_base =
            head_base + query_context * kAttentionQkvHeadGroupFusedHeadDim;
        std::array<float, kAttentionQkvHeadGroupFusedSize> scores{};
        std::array<float, kAttentionQkvHeadGroupFusedSize> weights{};
        float max_score = -std::numeric_limits<float>::infinity();
        for (std::uint32_t item = 0; item < window; ++item) {
          const std::uint32_t key_context =
              (query_context + contexts_per_head - item) % contexts_per_head;
          const std::uint32_t key_base =
              head_base + key_context * kAttentionQkvHeadGroupFusedHeadDim;
          float dot = 0.0f;
          for (std::uint32_t lane = 0;
               lane < kAttentionQkvHeadGroupFusedHeadDim;
               ++lane) {
            const std::uint32_t query_row = query_base + lane;
            const std::uint32_t key_row = key_base + lane;
            const float q = reference_attention_qkv_projection(
                logits[query_row % rows], 0.0f, 0u, query_row);
            const float k = reference_attention_qkv_projection(
                logits[key_row % rows], 0.0f, 1u, key_row);
            dot += q * k;
          }
          const float score = dot * kAttentionQkvHeadGroupFusedSoftmaxScale;
          scores[item] = score;
          max_score = std::max(max_score, score);
        }
        float denom = 0.0f;
        float weighted_score = 0.0f;
        for (std::uint32_t item = 0; item < window; ++item) {
          const float weight = std::exp(scores[item] - max_score);
          weights[item] = weight;
          denom += weight;
          weighted_score += weight * scores[item];
        }
        if (denom <= 0.0f) {
          continue;
        }
        const float mean_score = weighted_score / denom;
        const float bounded_score =
            mean_score / (1.0f + std::fabs(mean_score));
        for (std::uint32_t dim = 0;
             dim < kAttentionQkvHeadGroupFusedHeadDim;
             ++dim) {
          const std::uint32_t col =
              head + active_heads *
                  (dim + kAttentionQkvHeadGroupFusedHeadDim * query_context);
          if (col >= target_cols) {
            continue;
          }
          const std::uint32_t activation_row = col % rows;
          const float active = activation[col];
          const float q_active = reference_attention_qkv_projection(
              logits[activation_row], active, 0u, activation_row);
          const float k_active = reference_attention_qkv_projection(
              logits[activation_row], active, 1u, activation_row);
          const float v_active = reference_attention_qkv_projection(
              logits[activation_row], active, 2u, activation_row);
          const float scratch_residual =
              (q_active * 0.0625f + k_active * 0.03125f +
               v_active * 0.015625f) *
              kAttentionQkvResidualScale;

          float weighted_v = 0.0f;
          for (std::uint32_t item = 0; item < window; ++item) {
            const std::uint32_t key_context =
                (query_context + contexts_per_head - item) % contexts_per_head;
            const std::uint32_t key_base =
                head_base + key_context * kAttentionQkvHeadGroupFusedHeadDim;
            const std::uint32_t value_row = key_base + dim;
            const float v = reference_attention_qkv_projection(
                logits[value_row % rows], 0.0f, 2u, value_row);
            weighted_v += weights[item] * v;
          }
          const float context = weighted_v / denom;
          const std::uint32_t query_row = query_base + dim;
          const float q_direct = reference_attention_qkv_projection(
              logits[query_row % rows], 0.0f, 0u, query_row);
          const float head_bias =
              static_cast<float>((head & 0x7u) + 1u) * 0.0009765625f;
          const float dim_bias =
              static_cast<float>((dim & 0xfu) + 1u) * 0.0001220703125f;
          const float residual =
              context * 0.078125f +
              q_direct * 0.015625f +
              bounded_score * 0.015625f +
              head_bias +
              dim_bias;
          activation[col] =
              active + scratch_residual +
              residual * kAttentionQkvHeadGroupFusedResidualScale;
        }
      }
    }
  }
}

void apply_reference_phase_feedback(
    FfnPhaseMode ffn_phase_mode,
    FeedbackMode feedback_mode,
    const std::vector<Stage>& stages,
    const std::vector<float>& logits,
    std::vector<float>& activation,
    std::vector<float>& ffn_gate_cache,
    std::vector<float>& ffn_up_cache,
    std::uint32_t token,
    const rtxllm::LayerActivationFeedbackEdge& edge) {
  if (ffn_phase_edge_caches_gate(ffn_phase_mode, edge, stages)) {
    const bool accumulate = stages[edge.source_stage].role != "ffn_gate_exps.weight";
    apply_reference_rmsnorm_cache(
        logits,
        ffn_gate_cache,
        edge.target_cols,
        1.0e-6f,
        true,
        accumulate);
    return;
  }
  if (ffn_phase_edge_caches_up(ffn_phase_mode, edge, stages)) {
    const bool accumulate = stages[edge.source_stage].role != "ffn_up_exps.weight";
    apply_reference_rmsnorm_cache(
        logits,
        ffn_up_cache,
        edge.target_cols,
        1.0e-6f,
        false,
        accumulate);
    if (ffn_phase_edge_finalizes_gated_product(ffn_phase_mode, edge, stages)) {
      apply_reference_gated_cache_product(
          ffn_gate_cache,
          ffn_up_cache,
          activation,
          edge.target_cols);
    }
    return;
  }

  switch (feedback_mode) {
    case FeedbackMode::Synthetic:
      apply_reference_feedback(
          logits,
          activation,
          token,
          edge.target_cols,
          edge.stage_id);
      return;
    case FeedbackMode::RmsNorm:
      apply_reference_rmsnorm_feedback(
          logits,
          activation,
          edge.target_cols,
          1.0e-6f);
      return;
    case FeedbackMode::PhaseAwareFfnSilu:
      if (feedback_edge_uses_ffn_silu(feedback_mode, edge)) {
        apply_reference_silu_rmsnorm_feedback(
            logits,
            activation,
            edge.target_cols,
            1.0e-6f);
        return;
      }
      apply_reference_rmsnorm_feedback(
          logits,
          activation,
          edge.target_cols,
          1.0e-6f);
      return;
    case FeedbackMode::PhaseAwareFfnGatedSilu:
      if (feedback_edge_caches_ffn_gate(feedback_mode, edge, stages)) {
        apply_reference_rmsnorm_feedback_with_silu_cache(
            logits,
            activation,
            ffn_gate_cache,
            edge.target_cols,
            1.0e-6f);
        return;
      }
      if (feedback_edge_uses_ffn_gated_silu(feedback_mode, edge, stages)) {
        apply_reference_gated_silu_rmsnorm_feedback(
            logits,
            ffn_gate_cache,
            activation,
            edge.target_cols,
            1.0e-6f);
        return;
      }
      apply_reference_rmsnorm_feedback(
          logits,
          activation,
          edge.target_cols,
          1.0e-6f);
      return;
  }
  throw std::runtime_error("unknown reference feedback mode");
}

ReferenceResult run_mixed_reference(
    const std::vector<Stage>& stages,
    const Options& options,
    const rtxllm::LayerExecutorDataflowPlan& dataflow_plan,
    const std::vector<float>& initial_activation,
    const std::vector<ObservedStageSnapshot>& observed_stages,
    const std::vector<float>& observed_logits,
    const std::vector<float>& observed_kv,
    std::uint32_t observed_token,
    std::uint32_t ssm_scan_scratch_values_for_reference) {
  ReferenceResult result;
  result.checked = true;
  result.tolerance = options.reference_tolerance;
  result.stages.reserve(stages.size());
  std::vector<float> activation = initial_activation;
  std::vector<float> ffn_gate_cache(initial_activation.size(), 0.0f);
  std::vector<float> ffn_up_cache(initial_activation.size(), 0.0f);
  std::vector<float> attention_qkv_scratch;
  std::vector<float> ssm_recurrent_state(initial_activation.size(), 0.0f);
  const auto ssm_scan_scratch_reference_values = std::max<std::size_t>(
      initial_activation.size(),
      static_cast<std::size_t>(ssm_scan_scratch_values_for_reference));
  std::vector<float> ssm_scan_scratch(
      ssm_scan_scratch_reference_values,
      0.0f);
  std::vector<float> expected_kv(observed_kv.size(), 0.0f);
  std::vector<float> final_logits;
  std::uint32_t token = 0;

  for (std::size_t index = 0; index < stages.size(); ++index) {
    const auto& stage = stages[index];
    const auto payload_path = resolve_payload_path(stage.payload_root, stage.kind == StageKind::Q4K
        ? stage.q4k.payload_path
        : stage.iq.payload_path);
    const auto payload = rtxllm::load_binary_file(
        payload_path,
        stage.kind == StageKind::Q4K ? stage.q4k.payload_bytes : stage.iq.payload_bytes,
        "reference payload");
    std::vector<std::uint8_t> metadata;
    if (stage.kind == StageKind::Q4K) {
      metadata = rtxllm::load_binary_file(
          resolve_payload_path(stage.payload_root, stage.q4k.metadata_path),
          stage.q4k.metadata_bytes,
          "reference q4k metadata");
    }

    final_logits = reference_stage_logits(stage, payload, metadata, activation);
    apply_reference_kv(final_logits, expected_kv, token, options.page_words, options.active_pages);
    const auto best_index = argmax_token(final_logits);
    token = (best_index + token + 1u) % 32000u;

    ReferenceStageResult stage_result;
    stage_result.name = stage.name;
    stage_result.role = stage.role;
    stage_result.type = stage.type;
    stage_result.rows = static_cast<std::uint32_t>(final_logits.size());
    stage_result.token_expected = token;
    if (index < observed_stages.size()) {
      const auto& observed = observed_stages[index];
      stage_result.token_observed = observed.token;
      if (!final_logits.empty() && !observed.logits.empty()) {
        stage_result.first_logit_expected = final_logits.front();
        stage_result.first_logit_observed = observed.logits.front();
      }
      const auto rows_to_compare = std::min(final_logits.size(), observed.logits.size());
      for (std::size_t row = 0; row < rows_to_compare; ++row) {
        const float error = std::fabs(final_logits[row] - observed.logits[row]);
        if (error > stage_result.max_logit_abs_error) {
          stage_result.max_logit_abs_error = error;
          stage_result.max_logit_error_row = static_cast<std::uint32_t>(row);
        }
        if (error > result.tolerance) {
          ++stage_result.logit_mismatches;
        }
      }
      if (observed.logits.size() != final_logits.size()) {
        stage_result.logit_mismatches +=
            observed.logits.size() > final_logits.size()
            ? observed.logits.size() - final_logits.size()
            : final_logits.size() - observed.logits.size();
      }
      stage_result.passed =
          stage_result.token_expected == stage_result.token_observed &&
          stage_result.logit_mismatches == 0;
    } else {
      stage_result.logit_mismatches = final_logits.size();
    }
    if (!stage_result.passed) {
      ++result.stage_mismatches;
    }
    result.stages.push_back(std::move(stage_result));

    if (options.chain_activation) {
      const auto& edge =
          rtxllm::find_layer_activation_feedback_edge(dataflow_plan, index);
      apply_reference_phase_feedback(
          options.ffn_phase_mode,
          options.feedback_mode,
          stages,
          final_logits,
          activation,
          ffn_gate_cache,
          ffn_up_cache,
          token,
          edge);
      if (ssm_phase_mode_uses_source_parameterized(options.ssm_phase_mode) &&
          edge.source_phase == rtxllm::LayerPhase::Ssm &&
          edge.target_phase == rtxllm::LayerPhase::Ssm) {
        if (stage.role == "ssm_alpha.weight") {
          apply_reference_ssm_source_parameter_cache(
              final_logits,
              activation,
              ssm_scan_scratch,
              static_cast<std::uint32_t>(ssm_recurrent_state.size()),
              0u);
        } else if (stage.role == "ssm_beta.weight") {
          apply_reference_ssm_source_parameter_cache(
              final_logits,
              activation,
              ssm_scan_scratch,
              static_cast<std::uint32_t>(ssm_recurrent_state.size()),
              1u);
        }
      }
      if (attention_phase_mode_uses_qkv_scratch(options.attention_phase_mode) &&
          edge.source_phase == rtxllm::LayerPhase::Attention &&
          edge.target_phase == rtxllm::LayerPhase::Ffn) {
        apply_reference_attention_qkv_scratch(
            final_logits,
            attention_qkv_scratch,
            activation,
            edge.target_cols);
        if (attention_phase_mode_uses_qkv_reduce(options.attention_phase_mode)) {
          apply_reference_attention_qkv_reduce(
              attention_qkv_scratch,
              activation,
              edge.target_cols);
        }
        if (attention_phase_mode_uses_qkv_window(options.attention_phase_mode)) {
          apply_reference_attention_qkv_window(
              attention_qkv_scratch,
              activation,
              edge.target_cols);
        }
        if (attention_phase_mode_uses_qkv_head_window(
                options.attention_phase_mode)) {
          apply_reference_attention_qkv_head_window(
              attention_qkv_scratch,
              activation,
              edge.target_cols);
        }
        if (attention_phase_mode_uses_qkv_head_dim_window(
                options.attention_phase_mode)) {
          apply_reference_attention_qkv_head_dim_window(
              attention_qkv_scratch,
              activation,
              edge.target_cols);
        }
        if (attention_phase_mode_uses_qkv_head_tile_window(
                options.attention_phase_mode)) {
          apply_reference_attention_qkv_head_tile_window(
              attention_qkv_scratch,
              activation,
              edge.target_cols);
        }
        if (attention_phase_mode_uses_qkv_head_group_window(
                options.attention_phase_mode)) {
          apply_reference_attention_qkv_head_group_window(
              attention_qkv_scratch,
              activation,
              edge.target_cols);
        }
        if (attention_phase_mode_uses_qkv_head_group_rope_window(
                options.attention_phase_mode)) {
          apply_reference_attention_qkv_head_group_rope_window(
              attention_qkv_scratch,
              activation,
              edge.target_cols);
        }
      }
      if (attention_phase_mode_uses_qkv_head_group_fused(
              options.attention_phase_mode) &&
          edge.source_phase == rtxllm::LayerPhase::Attention &&
          edge.target_phase == rtxllm::LayerPhase::Ffn) {
        apply_reference_attention_qkv_head_group_fused(
            final_logits,
            activation,
            edge.target_cols);
      }
      if (ssm_phase_mode_uses_recurrent_state(options.ssm_phase_mode) &&
          edge.source_phase == rtxllm::LayerPhase::Ssm &&
          edge.target_phase == rtxllm::LayerPhase::Output) {
        if (ssm_phase_mode_uses_source_parameterized(options.ssm_phase_mode)) {
          apply_reference_ssm_source_parameterized_scan(
              final_logits,
              ssm_recurrent_state,
              ssm_scan_scratch,
              activation,
              edge.target_cols);
        } else if (ssm_phase_mode_uses_selective_scan(options.ssm_phase_mode)) {
          apply_reference_ssm_selective_scan(
              final_logits,
              ssm_recurrent_state,
              ssm_scan_scratch,
              activation,
              edge.target_cols,
              ssm_scan_scratch_values_for_reference == 0
                  ? edge.target_cols
                  : ssm_scan_scratch_values_for_reference);
        } else if (ssm_phase_mode_uses_scan_scratch(options.ssm_phase_mode)) {
          apply_reference_ssm_scan_scratch(
              final_logits,
              ssm_recurrent_state,
              ssm_scan_scratch,
              activation,
              edge.target_cols,
              ssm_scan_scratch_values_for_reference == 0
                  ? edge.target_cols
                  : ssm_scan_scratch_values_for_reference);
        } else {
          apply_reference_ssm_recurrent_state(
              final_logits,
              ssm_recurrent_state,
              activation,
              edge.target_cols);
        }
      }
      if (output_phase_mode_uses_final_token(options.output_phase_mode) &&
          edge.source_phase == rtxllm::LayerPhase::Output) {
        apply_reference_output_final_token(final_logits, token);
      }
    } else if (output_phase_mode_uses_final_token(options.output_phase_mode) &&
               rtxllm::layer_phase_from_role(stage.role) ==
                   rtxllm::LayerPhase::Output) {
      apply_reference_output_final_token(final_logits, token);
    }
  }

  result.final_rows = static_cast<std::uint32_t>(final_logits.size());
  result.final_stage = stages.empty() ? "" : stages.back().name;
  result.token_expected = token;
  result.token_observed = observed_token;
  if (!final_logits.empty() && !observed_logits.empty()) {
    result.first_logit_expected = final_logits.front();
    result.first_logit_observed = observed_logits.front();
  }

  for (std::uint32_t row = 0; row < final_logits.size(); ++row) {
    const float error = std::fabs(final_logits[row] - observed_logits[row]);
    if (error > result.max_logit_abs_error) {
      result.max_logit_abs_error = error;
      result.max_logit_error_row = row;
    }
    if (error > result.tolerance) {
      ++result.logit_mismatches;
    }
  }
  for (std::uint32_t index = 0; index < expected_kv.size(); ++index) {
    const float error = std::fabs(expected_kv[index] - observed_kv[index]);
    if (error > result.max_kv_abs_error) {
      result.max_kv_abs_error = error;
      result.max_kv_error_index = index;
    }
    if (error > result.tolerance) {
      ++result.kv_mismatches;
    }
  }

  result.passed = result.token_expected == result.token_observed &&
      result.logit_mismatches == 0 && result.kv_mismatches == 0 &&
      result.stage_mismatches == 0;
  return result;
}

std::vector<rtxllm::Q4KManifestTensor> select_layer_q4k(
    const rtxllm::Q4KManifest& manifest,
    const Options& options) {
  if (!options.q4k_tensors.empty()) {
    return rtxllm::select_q4k_tensors(manifest, options.q4k_tensors, 0);
  }
  std::vector<rtxllm::Q4KManifestTensor> selected;
  for (const auto& tensor : manifest.tensors) {
    if (tensor.layer == options.layer) {
      selected.push_back(tensor);
    }
  }
  return selected;
}

std::vector<rtxllm::IQManifestTensor> select_layer_iq(
    const rtxllm::IQManifest& manifest,
    const Options& options) {
  std::vector<rtxllm::IQManifestTensor> selected;
  if (!options.iq_tensors.empty()) {
    selected.reserve(options.iq_tensors.size());
    for (const auto& name : options.iq_tensors) {
      selected.push_back(rtxllm::select_iq_tensor(manifest, name));
    }
    return selected;
  }
  for (const auto& tensor : manifest.tensors) {
    if (tensor.layer == options.layer) {
      selected.push_back(tensor);
    }
  }
  return selected;
}

std::vector<IQStageInput> select_layer_iq_sources(
    const std::vector<rtxllm::IQManifest>& manifests,
    const Options& options) {
  std::vector<IQStageInput> selected;
  if (!options.iq_tensors.empty()) {
    selected.reserve(options.iq_tensors.size());
    for (const auto& name : options.iq_tensors) {
      bool found_tensor = false;
      for (const auto& manifest : manifests) {
        const auto found = std::find_if(
            manifest.tensors.begin(), manifest.tensors.end(), [&](const auto& tensor) {
              return tensor.name == name;
            });
        if (found != manifest.tensors.end()) {
          selected.push_back(IQStageInput{*found, manifest.path});
          found_tensor = true;
          break;
        }
      }
      if (!found_tensor) {
        throw std::runtime_error("IQ manifest tensor not found: " + name);
      }
    }
    return selected;
  }
  for (const auto& manifest : manifests) {
    for (const auto& tensor : manifest.tensors) {
      if (tensor.layer == options.layer) {
        selected.push_back(IQStageInput{tensor, manifest.path});
      }
    }
  }
  return selected;
}

bool is_iq_stage(StageKind kind) {
  return kind == StageKind::IQ2S || kind == StageKind::IQ3S;
}

std::vector<rtxllm::IQManifestTensor> iq_stage_tensors(
    const std::vector<IQStageInput>& selected) {
  std::vector<rtxllm::IQManifestTensor> tensors;
  tensors.reserve(selected.size());
  for (const auto& input : selected) {
    tensors.push_back(input.tensor);
  }
  return tensors;
}

cudaError_t launch_q4k_stage(
    const Stage& stage,
    const std::uint8_t* payload_base,
    const std::uint8_t* metadata_base,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream) {
  const auto* payload = payload_base + stage.payload_offset;
  const auto* metadata = metadata_base + stage.metadata_offset;
  const auto& tensor = stage.q4k;
  if (tensor.runtime_kernel == "split-predecoded") {
    return rtxllm_launch_q4k_matvec_decode_split_predecoded_vec4x4(
        payload,
        reinterpret_cast<const float*>(metadata),
        activation,
        logits,
        kv,
        token,
        tensor.rows,
        tensor.cols,
        kv_words,
        page_words,
        active_pages,
        stream);
  }
  if (tensor.runtime_kernel == "split-half") {
    return rtxllm_launch_q4k_matvec_decode_split_half_vec4x4(
        payload,
        reinterpret_cast<const std::uint16_t*>(metadata),
        activation,
        logits,
        kv,
        token,
        tensor.rows,
        tensor.cols,
        kv_words,
        page_words,
        active_pages,
        stream);
  }
  if (tensor.runtime_kernel == "split-compact") {
    return rtxllm_launch_q4k_matvec_decode_split_compact_vec4x4(
        payload,
        metadata,
        activation,
        logits,
        kv,
        token,
        tensor.rows,
        tensor.cols,
        kv_words,
        page_words,
        active_pages,
        stream);
  }
  if (tensor.runtime_kernel == "split-native") {
    return rtxllm_launch_q4k_matvec_decode_split_native_vec4x4(
        payload,
        metadata,
        activation,
        logits,
        kv,
        token,
        tensor.rows,
        tensor.cols,
        kv_words,
        page_words,
        active_pages,
        stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto q4k_manifest = rtxllm::load_q4k_manifest(options.q4k_manifest);
    std::vector<rtxllm::IQManifest> iq_manifests;
    iq_manifests.reserve(1u + options.extra_iq_manifests.size());
    iq_manifests.push_back(rtxllm::load_iq_manifest(options.iq_manifest));
    for (const auto& manifest_path : options.extra_iq_manifests) {
      iq_manifests.push_back(rtxllm::load_iq_manifest(manifest_path));
    }
    const auto& iq_manifest = iq_manifests.front();
    const bool include_q5k = !options.q5k_manifest.empty();
    const auto q5k_manifest = include_q5k
        ? rtxllm::load_iq_manifest(options.q5k_manifest)
        : rtxllm::IQManifest{};
    const auto selected_q4k = select_layer_q4k(q4k_manifest, options);
    const auto selected_iq = select_layer_iq_sources(iq_manifests, options);
    std::vector<rtxllm::IQManifestTensor> selected_q5k;
    if (include_q5k) {
      selected_q5k.push_back(rtxllm::select_iq_tensor(q5k_manifest, options.q5k_tensor));
    }
    if (selected_q4k.empty()) {
      throw std::runtime_error("no Q4_K tensors selected for mixed sequence");
    }
    if (selected_iq.empty()) {
      throw std::runtime_error("no IQ tensors selected for mixed sequence");
    }
    const auto selected_iq_tensors = iq_stage_tensors(selected_iq);

    const auto q4k_tensor_plan =
        options.tensor_plan.empty() ? q4k_manifest.tensor_plan : options.tensor_plan;
    const auto iq_tensor_plan =
        options.tensor_plan.empty() ? iq_manifest.tensor_plan : options.tensor_plan;
    const auto q5k_tensor_plan =
        options.tensor_plan.empty() ? q5k_manifest.tensor_plan : options.tensor_plan;
    bool source_plan_validated = false;
    rtxllm::Q4KTensorPlanValidation q4k_plan_validation;
    rtxllm::IQTensorPlanValidation iq_plan_validation;
    rtxllm::IQTensorPlanValidation q5k_plan_validation;
    if (options.source_plan_validation) {
      q4k_plan_validation =
          rtxllm::validate_q4k_manifest_tensors_against_plan(selected_q4k, q4k_tensor_plan);
      iq_plan_validation =
          rtxllm::validate_iq_manifest_tensors_against_plan(selected_iq_tensors, iq_tensor_plan);
      if (include_q5k) {
        q5k_plan_validation =
            rtxllm::validate_iq_manifest_tensors_against_plan(selected_q5k, q5k_tensor_plan);
      }
      source_plan_validated = true;
    }

    std::vector<Stage> stages;
    stages.reserve(selected_q4k.size() + selected_iq.size() + selected_q5k.size());
    std::size_t q4k_payload_bytes = 0;
    std::size_t q4k_metadata_bytes = 0;
    std::size_t iq_payload_bytes = 0;
    std::size_t q5k_payload_bytes = 0;
    std::uint64_t logical_values_per_graph = 0;
    std::uint64_t host_weight_checksum = 0;
    std::uint32_t max_rows = 0;
    std::uint32_t max_cols = 0;
    std::map<std::string, std::size_t> role_counts;
    std::map<std::string, std::size_t> type_counts;
    std::set<int> observed_layers;

    for (const auto& tensor : selected_q4k) {
      Stage stage;
      stage.kind = StageKind::Q4K;
      stage.q4k = tensor;
      stage.payload_root = options.q4k_manifest.parent_path();
      stage.payload_offset = q4k_payload_bytes;
      stage.metadata_offset = q4k_metadata_bytes;
      stage.rows = tensor.rows;
      stage.cols = tensor.cols;
      stage.layer = tensor.layer;
      stage.source_offset = tensor.source_offset;
      stage.name = tensor.name;
      stage.role = tensor.role.empty() ? "unknown" : tensor.role;
      stage.type = "Q4_K";
      stages.push_back(stage);
      q4k_payload_bytes += align_up(static_cast<std::size_t>(tensor.payload_bytes));
      q4k_metadata_bytes += align_up(static_cast<std::size_t>(tensor.metadata_bytes));
      logical_values_per_graph += static_cast<std::uint64_t>(tensor.rows) * tensor.cols;
      host_weight_checksum += tensor.payload_checksum + tensor.metadata_checksum;
      max_rows = std::max(max_rows, tensor.rows);
      max_cols = std::max(max_cols, tensor.cols);
      role_counts["Q4_K:" + stage.role] += 1;
      type_counts[stage.type] += 1;
      if (tensor.layer >= 0) {
        observed_layers.insert(tensor.layer);
      }
    }

    for (const auto& input : selected_iq) {
      const auto& tensor = input.tensor;
      const bool is_iq2s = tensor.type == "IQ2_S";
      const bool is_iq3s = tensor.type == "IQ3_S";
      if (!is_iq2s && !is_iq3s) {
        throw std::runtime_error("mixed sequence supports IQ2_S and IQ3_S IQ stages only");
      }
      const auto expected_block_bytes = is_iq2s ? kIq2SBlockBytes : kIq3SBlockBytes;
      if (tensor.block_type_size != expected_block_bytes) {
        throw std::runtime_error(tensor.type + " block type size mismatch: " + tensor.name);
      }
      const TensorShape shape = raw_manifest_tensor_shape(tensor, tensor.type);
      Stage stage;
      stage.kind = is_iq2s ? StageKind::IQ2S : StageKind::IQ3S;
      stage.iq = tensor;
      stage.payload_root = input.manifest_path.parent_path();
      stage.payload_offset = iq_payload_bytes;
      stage.rows = std::min(shape.rows, options.rows_limit);
      stage.cols = shape.cols;
      stage.layer = tensor.layer;
      stage.source_offset = tensor.source_offset;
      stage.name = tensor.name;
      stage.role = tensor.role.empty() ? "unknown" : tensor.role;
      stage.type = tensor.type;
      stages.push_back(stage);
      iq_payload_bytes += align_up(static_cast<std::size_t>(tensor.payload_bytes));
      logical_values_per_graph += static_cast<std::uint64_t>(stage.rows) * stage.cols;
      host_weight_checksum += tensor.payload_checksum;
      max_rows = std::max(max_rows, stage.rows);
      max_cols = std::max(max_cols, stage.cols);
      role_counts[stage.type + ":" + stage.role] += 1;
      type_counts[stage.type] += 1;
      if (tensor.layer >= 0) {
        observed_layers.insert(tensor.layer);
      }
    }

    for (const auto& tensor : selected_q5k) {
      if (tensor.type != "Q5_K") {
        throw std::runtime_error("mixed sequence Q5_K stage requires Q5_K tensor");
      }
      if (tensor.block_type_size != kQ5KBlockBytes) {
        throw std::runtime_error("Q5_K block type size mismatch: " + tensor.name);
      }
      const TensorShape shape = raw_manifest_tensor_shape(tensor, "Q5_K");
      Stage stage;
      stage.kind = StageKind::Q5K;
      stage.iq = tensor;
      stage.payload_root = options.q5k_manifest.parent_path();
      stage.payload_offset = q5k_payload_bytes;
      stage.rows = std::min(shape.rows, options.rows_limit);
      stage.cols = shape.cols;
      stage.layer = tensor.layer;
      stage.source_offset = tensor.source_offset;
      stage.name = tensor.name;
      stage.role = tensor.role.empty() ? "output" : tensor.role;
      stage.type = tensor.type;
      stages.push_back(stage);
      q5k_payload_bytes += align_up(static_cast<std::size_t>(tensor.payload_bytes));
      logical_values_per_graph += static_cast<std::uint64_t>(stage.rows) * stage.cols;
      host_weight_checksum += tensor.payload_checksum;
      max_rows = std::max(max_rows, stage.rows);
      max_cols = std::max(max_cols, stage.cols);
      role_counts["Q5_K:" + stage.role] += 1;
      type_counts[stage.type] += 1;
    }

    if (options.stage_order == StageOrderPolicy::RolePlan) {
      std::stable_sort(stages.begin(), stages.end(), role_plan_order_less);
    } else {
      std::stable_sort(stages.begin(), stages.end(), source_offset_order_less);
    }
    if (stages.empty()) {
      throw std::runtime_error("mixed sequence requires at least one stage");
    }
    const auto layer_stages = build_layer_stage_descriptors(stages);
    const std::size_t stage_role_plan_unknowns =
        rtxllm::count_unknown_layer_roles(layer_stages);
    const std::size_t stage_phase_unknowns =
        rtxllm::count_unknown_layer_phases(layer_stages);
    const auto stage_phase_counts = rtxllm::build_layer_phase_counts(layer_stages);
    const auto stage_phase_segments = rtxllm::build_layer_phase_segments(layer_stages);
    const auto layer_plan =
        rtxllm::validate_layer_phase_plan(layer_stages, stage_phase_segments);
    const auto layer_execution_plan =
        rtxllm::build_layer_execution_plan(stage_phase_segments, options.chain_activation);
    std::vector<rtxllm::LayerStageWorkspaceDescriptor> layer_stage_workspaces;
    layer_stage_workspaces.reserve(stages.size());
    for (const auto& stage : stages) {
      layer_stage_workspaces.push_back(rtxllm::LayerStageWorkspaceDescriptor{
          stage.rows,
          stage.cols,
      });
    }
    const auto layer_workspace_plan =
        rtxllm::build_layer_executor_workspace_plan(
            layer_execution_plan,
            layer_stage_workspaces);
    if (layer_workspace_plan.max_rows != max_rows ||
        layer_workspace_plan.max_cols != max_cols ||
        layer_workspace_plan.logical_values_per_replay != logical_values_per_graph) {
      throw std::runtime_error("layer executor workspace plan does not match allocation envelope");
    }
    const bool strict_layer_plan_passed =
        options.stage_order == StageOrderPolicy::RolePlan && layer_plan.passed;
    if (options.strict_layer_plan && options.stage_order != StageOrderPolicy::RolePlan) {
      throw std::runtime_error("strict layer plan requires --stage-order role-plan");
    }
    if (options.strict_layer_plan && !layer_plan.passed) {
      throw std::runtime_error(layer_plan.message);
    }
    if (ffn_phase_mode_uses_gated_silu(options.ffn_phase_mode)) {
      if (!options.chain_activation) {
        throw std::runtime_error("gated FFN phase mode requires chained activation");
      }
      if (!strict_layer_plan_passed) {
        throw std::runtime_error(
            "gated FFN phase mode requires a validated role-plan stage order");
      }
      const auto gate_roles = static_cast<std::size_t>(std::count_if(
          stages.begin(),
          stages.end(),
          [](const Stage& stage) { return is_ffn_gate_role(stage.role); }));
      const auto up_roles = static_cast<std::size_t>(std::count_if(
          stages.begin(),
          stages.end(),
          [](const Stage& stage) { return is_ffn_up_role(stage.role); }));
      const auto down_roles = static_cast<std::size_t>(std::count_if(
          stages.begin(),
          stages.end(),
          [](const Stage& stage) { return is_ffn_down_role(stage.role); }));
      if (gate_roles == 0 || up_roles == 0 || down_roles == 0) {
        throw std::runtime_error(
            "gated FFN phase mode requires gate, up, and down FFN roles");
      }
    }
    if (attention_phase_mode_uses_qkv_scratch(options.attention_phase_mode)) {
      if (!options.chain_activation) {
        throw std::runtime_error("attention QKV scratch phase mode requires chained activation");
      }
      if (!strict_layer_plan_passed) {
        throw std::runtime_error(
            "attention QKV scratch phase mode requires a validated role-plan stage order");
      }
      const auto attention_roles = static_cast<std::size_t>(std::count_if(
          stages.begin(),
          stages.end(),
          [](const Stage& stage) {
            return rtxllm::layer_phase_from_role(stage.role) ==
                rtxllm::LayerPhase::Attention;
          }));
      const auto ffn_roles = static_cast<std::size_t>(std::count_if(
          stages.begin(),
          stages.end(),
          [](const Stage& stage) {
            return rtxllm::layer_phase_from_role(stage.role) ==
                rtxllm::LayerPhase::Ffn;
          }));
      if (attention_roles == 0 || ffn_roles == 0) {
        throw std::runtime_error(
            "attention QKV scratch phase mode requires attention and FFN roles");
      }
      const bool collides_with_user_scratch = std::any_of(
          options.phase_scratch_workspaces.begin(),
          options.phase_scratch_workspaces.end(),
          [](const rtxllm::LayerAuxWorkspaceDescriptor& descriptor) {
            return descriptor.name == kAttentionQkvScratchName;
          });
      if (collides_with_user_scratch) {
        throw std::runtime_error(
            "attention QKV scratch phase mode reserves attention_qkv_state");
      }
    }
    if (ssm_phase_mode_uses_recurrent_state(options.ssm_phase_mode)) {
      if (!options.chain_activation) {
        throw std::runtime_error("SSM recurrent phase mode requires chained activation");
      }
      if (!strict_layer_plan_passed) {
        throw std::runtime_error(
            "SSM recurrent phase mode requires a validated role-plan stage order");
      }
      const auto ssm_roles = static_cast<std::size_t>(std::count_if(
          stages.begin(),
          stages.end(),
          [](const Stage& stage) {
            return rtxllm::layer_phase_from_role(stage.role) ==
                rtxllm::LayerPhase::Ssm;
          }));
      if (ssm_roles == 0) {
        throw std::runtime_error("SSM recurrent phase mode requires SSM roles");
      }
      if (ssm_phase_mode_uses_source_parameterized(options.ssm_phase_mode)) {
        if (ssm_phase_mode_uses_source_parameterized_fused(
                options.ssm_phase_mode) &&
            options.feedback_mode != FeedbackMode::RmsNorm) {
          throw std::runtime_error(
              "SSM source-parameterized-fused phase mode requires rmsnorm feedback");
        }
        const bool has_alpha = std::any_of(
            stages.begin(),
            stages.end(),
            [](const Stage& stage) {
              return stage.role == "ssm_alpha.weight";
            });
        const bool has_beta = std::any_of(
            stages.begin(),
            stages.end(),
            [](const Stage& stage) {
              return stage.role == "ssm_beta.weight";
            });
        const bool has_out = std::any_of(
            stages.begin(),
            stages.end(),
            [](const Stage& stage) {
              return stage.role == "ssm_out.weight";
            });
        if (!has_alpha || !has_beta || !has_out) {
          throw std::runtime_error(
              "SSM source-parameterized phase mode requires alpha, beta, and out roles");
        }
      }
      if (ssm_phase_mode_uses_scan_scratch(options.ssm_phase_mode)) {
        const bool collides_with_user_scratch = std::any_of(
            options.phase_scratch_workspaces.begin(),
            options.phase_scratch_workspaces.end(),
            [](const rtxllm::LayerAuxWorkspaceDescriptor& descriptor) {
              return descriptor.name == kSsmScanScratchName;
            });
        if (collides_with_user_scratch) {
          throw std::runtime_error(
              "SSM scan scratch phase mode reserves ssm_scan_scratch");
        }
      }
    }
    if (output_phase_mode_uses_final_token(options.output_phase_mode)) {
      if (!options.chain_activation) {
        throw std::runtime_error("output token phase mode requires chained activation");
      }
      if (!strict_layer_plan_passed) {
        throw std::runtime_error(
            "output token phase mode requires a validated role-plan stage order");
      }
      const auto output_roles = static_cast<std::size_t>(std::count_if(
          stages.begin(),
          stages.end(),
          [](const Stage& stage) {
            return rtxllm::layer_phase_from_role(stage.role) ==
                rtxllm::LayerPhase::Output;
          }));
      if (output_roles == 0) {
        throw std::runtime_error("output token phase mode requires output roles");
      }
    }

    const bool use_ffn_gate_cache =
        (feedback_mode_uses_ffn_gate_cache(options.feedback_mode) ||
         ffn_phase_mode_uses_gated_silu(options.ffn_phase_mode)) &&
        options.chain_activation;
    const bool use_ffn_up_cache =
        ffn_phase_mode_uses_gated_silu(options.ffn_phase_mode) &&
        options.chain_activation;
    const bool use_ssm_recurrent_state =
        ssm_phase_mode_uses_recurrent_state(options.ssm_phase_mode) &&
        options.chain_activation;
    const bool use_ssm_scan_scratch =
        ssm_phase_mode_uses_scan_scratch(options.ssm_phase_mode) &&
        options.chain_activation;
    const bool use_ssm_selective_scan =
        ssm_phase_mode_uses_selective_scan(options.ssm_phase_mode) &&
        options.chain_activation;
    const bool use_ssm_source_parameterized =
        ssm_phase_mode_uses_source_parameterized(options.ssm_phase_mode) &&
        options.chain_activation;
    const bool use_ssm_source_parameterized_fused =
        ssm_phase_mode_uses_source_parameterized_fused(
            options.ssm_phase_mode) &&
        options.chain_activation;
    const bool use_output_final_token =
        output_phase_mode_uses_final_token(options.output_phase_mode) &&
        options.chain_activation;
    const bool use_attention_qkv_scratch =
        attention_phase_mode_uses_qkv_scratch(options.attention_phase_mode) &&
        options.chain_activation;
    const bool use_attention_qkv_reduce =
        attention_phase_mode_uses_qkv_reduce(options.attention_phase_mode) &&
        options.chain_activation;
    const bool use_attention_qkv_window =
        attention_phase_mode_uses_qkv_window(options.attention_phase_mode) &&
        options.chain_activation;
    const bool use_attention_qkv_head_window =
        attention_phase_mode_uses_qkv_head_window(options.attention_phase_mode) &&
        options.chain_activation;
    const bool use_attention_qkv_head_dim_window =
        attention_phase_mode_uses_qkv_head_dim_window(
            options.attention_phase_mode) &&
        options.chain_activation;
    const bool use_attention_qkv_head_tile_window =
        attention_phase_mode_uses_qkv_head_tile_window(
            options.attention_phase_mode) &&
        options.chain_activation;
    const bool use_attention_qkv_head_group_window =
        attention_phase_mode_uses_qkv_head_group_window(
            options.attention_phase_mode) &&
        options.chain_activation;
    const bool use_attention_qkv_head_group_rope_window =
        attention_phase_mode_uses_qkv_head_group_rope_window(
            options.attention_phase_mode) &&
        options.chain_activation;
    const bool use_attention_qkv_head_group_fused =
        attention_phase_mode_uses_qkv_head_group_fused(
            options.attention_phase_mode) &&
        options.chain_activation;
    const bool use_attention_phase_kernel =
        use_attention_qkv_scratch || use_attention_qkv_head_group_fused;
    auto layer_aux_workspaces =
        rtxllm::build_gated_ffn_aux_workspace_descriptors(
            layer_workspace_plan,
            use_ffn_gate_cache,
            use_ffn_up_cache);
    std::vector<rtxllm::LayerAuxWorkspaceDescriptor> attention_aux_workspaces;
    if (use_attention_qkv_scratch) {
      const auto& attention_workspace =
          rtxllm::find_layer_workspace_phase_descriptor(
              layer_workspace_plan,
              rtxllm::LayerPhase::Attention);
      const std::uint64_t scratch_values =
          static_cast<std::uint64_t>(attention_workspace.max_rows) *
          kAttentionQkvComponents;
      if (scratch_values == 0 ||
          scratch_values > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("attention QKV scratch workspace size is invalid");
      }
      attention_aux_workspaces.push_back(rtxllm::LayerAuxWorkspaceDescriptor{
          kAttentionQkvScratchName,
          rtxllm::LayerPhase::Attention,
          static_cast<std::uint32_t>(scratch_values),
      });
    }
    layer_aux_workspaces.insert(
        layer_aux_workspaces.end(),
        attention_aux_workspaces.begin(),
        attention_aux_workspaces.end());
    const auto ssm_aux_workspaces =
        rtxllm::build_ssm_recurrent_aux_workspace_descriptors(
            layer_workspace_plan,
            use_ssm_recurrent_state,
            use_ssm_scan_scratch);
    layer_aux_workspaces.insert(
        layer_aux_workspaces.end(),
        ssm_aux_workspaces.begin(),
        ssm_aux_workspaces.end());
    const auto phase_scratch_aux_workspaces =
        rtxllm::build_layer_phase_scratch_workspace_descriptors(
            layer_workspace_plan,
            options.phase_scratch_workspaces);
    layer_aux_workspaces.insert(
        layer_aux_workspaces.end(),
        phase_scratch_aux_workspaces.begin(),
        phase_scratch_aux_workspaces.end());
    const auto layer_workspace_allocation_plan =
        rtxllm::build_layer_executor_workspace_allocation_plan(
            layer_workspace_plan,
            layer_aux_workspaces,
            sizeof(float),
            kAlignment);
    const auto layer_workspace_usage =
        rtxllm::measure_layer_executor_workspace_usage(
            layer_workspace_plan,
            layer_workspace_allocation_plan,
            sizeof(float));
    const std::size_t workspace_bytes =
        layer_workspace_allocation_plan.total_bytes;
    std::uint32_t ffn_gate_cache_cols = 0;
    std::uint32_t ffn_up_cache_cols = 0;
    std::uint32_t attention_qkv_scratch_values = 0;
    std::uint32_t ssm_recurrent_state_values = 0;
    std::uint32_t ssm_scan_scratch_values = 0;
    std::size_t ffn_gate_cache_workspace_bytes = 0;
    std::size_t ffn_up_cache_workspace_bytes = 0;
    std::size_t attention_qkv_scratch_workspace_bytes = 0;
    std::size_t ssm_recurrent_state_workspace_bytes = 0;
    std::size_t ssm_scan_scratch_workspace_bytes = 0;
    std::size_t phase_scratch_workspace_bytes = 0;
    const std::size_t attention_phase_kernel_launches_per_graph =
        use_attention_phase_kernel
            ? attention_phase_mode_kernel_count(options.attention_phase_mode) *
                  static_cast<std::size_t>(std::count_if(
                  layer_execution_plan.phases.begin(),
                  layer_execution_plan.phases.end(),
                  [](const rtxllm::LayerPhaseExecution& phase) {
                    return phase.phase == rtxllm::LayerPhase::Attention;
                  }))
            : 0;
    const std::size_t ssm_phase_kernel_launches_per_graph =
        use_ssm_recurrent_state
            ? ssm_phase_mode_kernel_count(options.ssm_phase_mode) *
                static_cast<std::size_t>(std::count_if(
                  layer_execution_plan.phases.begin(),
                  layer_execution_plan.phases.end(),
                  [](const rtxllm::LayerPhaseExecution& phase) {
                    return phase.phase == rtxllm::LayerPhase::Ssm;
                  }))
            : 0;
    const std::size_t output_phase_kernel_launches_per_graph =
        use_output_final_token
            ? static_cast<std::size_t>(std::count_if(
                  layer_execution_plan.phases.begin(),
                  layer_execution_plan.phases.end(),
                  [](const rtxllm::LayerPhaseExecution& phase) {
                    return phase.phase == rtxllm::LayerPhase::Output;
                  }))
            : 0;
    const std::size_t phase_scratch_kernel_launches_per_graph =
        phase_scratch_aux_workspaces.size();
    const std::size_t actual_kernel_launches_per_graph =
        layer_execution_plan.total_kernels_per_replay +
        attention_phase_kernel_launches_per_graph +
        ssm_phase_kernel_launches_per_graph +
        output_phase_kernel_launches_per_graph +
        phase_scratch_kernel_launches_per_graph;
    if (actual_kernel_launches_per_graph >
        std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("graph bucket kernel count exceeds uint32 range");
    }
    if (layer_execution_plan.phase_count >
        std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("graph bucket phase count exceeds uint32 range");
    }
    std::vector<rtxllm::CudaGraphBucketDescriptor> graph_bucket_descriptors;
    graph_bucket_descriptors.reserve(options.graph_bucket_count);
    for (std::uint32_t bucket_index = 0;
         bucket_index < options.graph_bucket_count;
         ++bucket_index) {
      const std::uint64_t query_bucket =
          static_cast<std::uint64_t>(options.graph_query_bucket) *
          static_cast<std::uint64_t>(bucket_index + 1u);
      if (query_bucket > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("graph query bucket exceeds uint32 range");
      }
      graph_bucket_descriptors.push_back(rtxllm::CudaGraphBucketDescriptor{
          "decode_b" + std::to_string(options.graph_batch_bucket) + "_q" +
              std::to_string(query_bucket),
          options.graph_batch_bucket,
          static_cast<std::uint32_t>(query_bucket),
          static_cast<std::uint32_t>(layer_execution_plan.phase_count),
          static_cast<std::uint32_t>(actual_kernel_launches_per_graph),
          layer_workspace_usage.total_high_water_bytes,
      });
    }
    const auto graph_bucket_plan =
        rtxllm::build_cuda_graph_bucket_plan(graph_bucket_descriptors, kAlignment);
    for (const auto& auxiliary : layer_workspace_allocation_plan.auxiliaries) {
      if (auxiliary.name == "ffn_gate_cache") {
        ffn_gate_cache_cols = auxiliary.values;
        ffn_gate_cache_workspace_bytes = auxiliary.bytes;
      } else if (auxiliary.name == "ffn_up_cache") {
        ffn_up_cache_cols = auxiliary.values;
        ffn_up_cache_workspace_bytes = auxiliary.bytes;
      } else if (auxiliary.name == kAttentionQkvScratchName) {
        attention_qkv_scratch_values = auxiliary.values;
        attention_qkv_scratch_workspace_bytes = auxiliary.bytes;
      } else if (auxiliary.name == "ssm_recurrent_state") {
        ssm_recurrent_state_values = auxiliary.values;
        ssm_recurrent_state_workspace_bytes = auxiliary.bytes;
      } else if (auxiliary.name == kSsmScanScratchName) {
        ssm_scan_scratch_values = auxiliary.values;
        ssm_scan_scratch_workspace_bytes = auxiliary.bytes;
      } else if (std::any_of(
                     phase_scratch_aux_workspaces.begin(),
                     phase_scratch_aux_workspaces.end(),
                     [&](const rtxllm::LayerAuxWorkspaceDescriptor& descriptor) {
                       return descriptor.name == auxiliary.name;
                     })) {
        phase_scratch_workspace_bytes += auxiliary.bytes;
      }
    }
    std::size_t initial_activation_h2d_bytes = 0;
    for (const auto& phase : layer_workspace_plan.phases) {
      initial_activation_h2d_bytes +=
          static_cast<std::size_t>(phase.max_cols) * sizeof(float);
    }
    const std::size_t kv_bytes = mib(options.kv_mib);
    const auto kv_words_size = kv_bytes / sizeof(float);
    if (kv_words_size == 0 ||
        kv_words_size > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("KV arena word count is out of uint32 range");
    }
    const auto kv_words = static_cast<std::uint32_t>(kv_words_size);
    const std::size_t dma_bytes = mib(1);
    const std::size_t resident_bytes =
        q4k_payload_bytes + q4k_metadata_bytes + iq_payload_bytes + q5k_payload_bytes;
    const std::size_t setup_h2d_bytes =
        resident_bytes + initial_activation_h2d_bytes + sizeof(std::uint32_t);

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

    rtxllm::RequestEstimate request_estimate;
    request_estimate.weight_bytes = resident_bytes;
    request_estimate.workspace_bytes = workspace_bytes;
    request_estimate.kv_cache_bytes = kv_bytes;
    request_estimate.dma_bytes = dma_bytes;
    request_estimate.graph_bytes = graph_bucket_plan.total_bytes;
    request_estimate.workspace_high_water_bytes =
        layer_workspace_usage.total_high_water_bytes;

    const auto decision = rtxllm::decide_admission(
        rtxllm::ResidencyPolicy::Strict,
        rtxllm::MemoryBudget{free_bytes, mib(options.wddm_guard_mib)},
        request_estimate);

    if (!decision.admit) {
      std::cout << "{\n";
      std::cout << "  \"path\": \""
                << (include_q5k ? "native_mixed_q4k_iq_q5k_sequence_graph"
                                : "native_mixed_q4k_iq_sequence_graph")
                << "\",\n";
      std::cout << "  \"label\": \"" << json_escape(options.label) << "\",\n";
      std::cout << "  \"admitted\": false,\n";
      std::cout << "  \"reason\": \"" << json_escape(decision.reason) << "\",\n";
      std::cout << "  \"required_bytes\": " << decision.required_bytes << ",\n";
      std::cout << "  \"usable_bytes\": " << decision.usable_bytes << ",\n";
      print_admission_breakdown(decision.breakdown);
      std::cout << "  \"graph_bucket_count\": "
                << graph_bucket_plan.buckets.size() << ",\n";
      std::cout << "  \"graph_bucket_estimated_bytes\": "
                << graph_bucket_plan.total_bytes << ",\n";
      std::cout << "  \"graph_bucket_max_estimated_bytes\": "
                << graph_bucket_plan.max_bucket_bytes << ",\n";
      std::cout << "  \"admission_policy\": \"strict\"\n";
      std::cout << "}\n";
      return 2;
    }

    std::vector<rtxllm::ArenaSpec> arena_specs = {
        {"q4k_payload", q4k_payload_bytes},
        {"q4k_metadata", q4k_metadata_bytes},
        {"iq_payload", iq_payload_bytes},
    };
    if (q5k_payload_bytes > 0) {
      arena_specs.push_back({"q5k_payload", q5k_payload_bytes});
    }
    arena_specs.push_back({"kv_pages", kv_bytes});
    arena_specs.push_back({"workspace", workspace_bytes});
    arena_specs.push_back({"dma", dma_bytes});

    rtxllm::VramSuperallocator pool;
    pool.initialize(arena_specs);

    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    pool.reset_async(stream);

    auto* q4k_payload_base = static_cast<std::uint8_t*>(pool.arena_ptr("q4k_payload"));
    auto* q4k_metadata_base = static_cast<std::uint8_t*>(pool.arena_ptr("q4k_metadata"));
    auto* iq_payload_base = static_cast<std::uint8_t*>(pool.arena_ptr("iq_payload"));
    auto* q5k_payload_base = q5k_payload_bytes > 0
        ? static_cast<std::uint8_t*>(pool.arena_ptr("q5k_payload"))
        : nullptr;
    auto* workspace = static_cast<std::uint8_t*>(pool.arena_ptr("workspace"));
    auto* kv = static_cast<float*>(pool.arena_ptr("kv_pages"));
    auto* token = static_cast<std::uint32_t*>(pool.arena_ptr("dma"));
    const auto layer_workspace_runtime =
        rtxllm::bind_layer_executor_workspace_runtime(
            layer_workspace_plan,
            layer_workspace_allocation_plan,
            workspace);
    float* ffn_gate_cache = nullptr;
    float* ffn_up_cache = nullptr;
    float* attention_qkv_scratch = nullptr;
    float* ssm_recurrent_state = nullptr;
    float* ssm_scan_scratch = nullptr;
    if (use_ffn_gate_cache) {
      const auto& auxiliary =
          rtxllm::find_layer_aux_workspace_runtime(
              layer_workspace_runtime,
              "ffn_gate_cache");
      ffn_gate_cache = auxiliary.data;
      if (ffn_gate_cache_cols != auxiliary.values ||
          ffn_gate_cache_workspace_bytes != auxiliary.bytes) {
        throw std::runtime_error("FFN gate cache allocation mismatch");
      }
    }
    if (use_ffn_up_cache) {
      const auto& auxiliary =
          rtxllm::find_layer_aux_workspace_runtime(
              layer_workspace_runtime,
              "ffn_up_cache");
      ffn_up_cache = auxiliary.data;
      if (ffn_up_cache_cols != auxiliary.values ||
          ffn_up_cache_workspace_bytes != auxiliary.bytes) {
        throw std::runtime_error("FFN up cache allocation mismatch");
      }
    }
    if (use_attention_qkv_scratch) {
      const auto& auxiliary =
          rtxllm::find_layer_aux_workspace_runtime(
              layer_workspace_runtime,
              kAttentionQkvScratchName);
      attention_qkv_scratch = auxiliary.data;
      if (auxiliary.phase != rtxllm::LayerPhase::Attention ||
          attention_qkv_scratch_values != auxiliary.values ||
          attention_qkv_scratch_workspace_bytes != auxiliary.bytes) {
        throw std::runtime_error("attention QKV scratch allocation mismatch");
      }
    }
    if (use_ssm_recurrent_state) {
      const auto& auxiliary =
          rtxllm::find_layer_aux_workspace_runtime(
              layer_workspace_runtime,
              "ssm_recurrent_state");
      ssm_recurrent_state = auxiliary.data;
      if (ssm_recurrent_state_values != auxiliary.values ||
          ssm_recurrent_state_workspace_bytes != auxiliary.bytes) {
        throw std::runtime_error("SSM recurrent state allocation mismatch");
      }
    }
    if (use_ssm_scan_scratch) {
      const auto& auxiliary =
          rtxllm::find_layer_aux_workspace_runtime(
              layer_workspace_runtime,
              kSsmScanScratchName);
      ssm_scan_scratch = auxiliary.data;
      if (auxiliary.phase != rtxllm::LayerPhase::Ssm ||
          ssm_scan_scratch_values != auxiliary.values ||
          ssm_scan_scratch_workspace_bytes != auxiliary.bytes) {
        throw std::runtime_error("SSM scan scratch allocation mismatch");
      }
    }
    std::vector<const rtxllm::LayerAuxWorkspaceRuntime*> phase_scratch_runtimes;
    phase_scratch_runtimes.reserve(phase_scratch_aux_workspaces.size());
    for (const auto& descriptor : phase_scratch_aux_workspaces) {
      const auto& auxiliary =
          rtxllm::find_layer_aux_workspace_runtime(
              layer_workspace_runtime,
              descriptor.name);
      if (auxiliary.phase != descriptor.phase ||
          auxiliary.values != descriptor.values) {
        throw std::runtime_error("phase scratch workspace allocation mismatch");
      }
      phase_scratch_runtimes.push_back(&auxiliary);
    }
    const auto runtime_workspace_usage =
        rtxllm::measure_layer_executor_workspace_usage(
            layer_workspace_runtime,
            sizeof(float));
    if (runtime_workspace_usage.total_high_water_bytes !=
            layer_workspace_usage.total_high_water_bytes ||
        runtime_workspace_usage.total_slack_bytes !=
            layer_workspace_usage.total_slack_bytes) {
      throw std::runtime_error("runtime workspace usage does not match admission estimate");
    }
    std::vector<std::size_t> stage_workspace_phase_indices;
    stage_workspace_phase_indices.reserve(stages.size());
    for (std::size_t index = 0; index < stages.size(); ++index) {
      stage_workspace_phase_indices.push_back(
          rtxllm::find_layer_workspace_runtime_phase_index_for_stage(
              layer_workspace_runtime,
              index));
    }
    const auto layer_dataflow_plan =
        rtxllm::build_layer_executor_dataflow_plan(
            layer_execution_plan,
            layer_stage_workspaces,
            layer_workspace_runtime);
    const rtxllm::LayerActivationFeedbackEdge* attention_ffn_edge = nullptr;
    const rtxllm::LayerActivationFeedbackEdge* ssm_output_edge = nullptr;
    for (const auto& edge : layer_dataflow_plan.activation_feedback_edges) {
      if (edge.source_phase == rtxllm::LayerPhase::Attention &&
          edge.target_phase == rtxllm::LayerPhase::Ffn) {
        attention_ffn_edge = &edge;
      }
      if (edge.source_phase == rtxllm::LayerPhase::Ssm &&
          edge.target_phase == rtxllm::LayerPhase::Output) {
        ssm_output_edge = &edge;
      }
    }
    if (use_attention_phase_kernel && attention_ffn_edge == nullptr) {
      throw std::runtime_error(
          "attention QKV phase mode requires an attention to FFN feedback edge");
    }
    if (use_ssm_recurrent_state && ssm_output_edge == nullptr) {
      throw std::runtime_error(
          "SSM recurrent phase mode requires an SSM to output feedback edge");
    }

    for (const auto& stage : stages) {
      if (stage.kind == StageKind::Q4K) {
        const auto payload = rtxllm::load_binary_file(
            stage.q4k.payload_path, stage.q4k.payload_bytes, "q4k payload");
        if (rtxllm::byte_checksum(payload) != stage.q4k.payload_checksum) {
          throw std::runtime_error("Q4_K payload checksum mismatch: " + stage.name);
        }
        check(cudaMemcpyAsync(
                  q4k_payload_base + stage.payload_offset,
                  payload.data(),
                  payload.size(),
                  cudaMemcpyHostToDevice,
                  stream),
            "q4k payload upload");
        const auto metadata = rtxllm::load_binary_file(
            stage.q4k.metadata_path, stage.q4k.metadata_bytes, "q4k metadata");
        if (rtxllm::byte_checksum(metadata) != stage.q4k.metadata_checksum) {
          throw std::runtime_error("Q4_K metadata checksum mismatch: " + stage.name);
        }
        check(cudaMemcpyAsync(
                  q4k_metadata_base + stage.metadata_offset,
                  metadata.data(),
                  metadata.size(),
                  cudaMemcpyHostToDevice,
                  stream),
            "q4k metadata upload");
      } else if (is_iq_stage(stage.kind)) {
        const std::filesystem::path payload_path = stage.iq.payload_path.is_absolute()
            ? stage.iq.payload_path
            : stage.payload_root / stage.iq.payload_path;
        const auto payload = rtxllm::load_binary_file(
            payload_path, stage.iq.payload_bytes, "iq payload");
        if (rtxllm::byte_checksum(payload) != stage.iq.payload_checksum) {
          throw std::runtime_error("IQ payload checksum mismatch: " + stage.name);
        }
        check(cudaMemcpyAsync(
                  iq_payload_base + stage.payload_offset,
                  payload.data(),
                  payload.size(),
                  cudaMemcpyHostToDevice,
                  stream),
            "iq payload upload");
      } else if (stage.kind == StageKind::Q5K) {
        const std::filesystem::path payload_path = stage.iq.payload_path.is_absolute()
            ? stage.iq.payload_path
            : options.q5k_manifest.parent_path() / stage.iq.payload_path;
        const auto payload = rtxllm::load_binary_file(
            payload_path, stage.iq.payload_bytes, "q5k payload");
        if (rtxllm::byte_checksum(payload) != stage.iq.payload_checksum) {
          throw std::runtime_error("Q5_K payload checksum mismatch: " + stage.name);
        }
        check(cudaMemcpyAsync(
                  q5k_payload_base + stage.payload_offset,
                  payload.data(),
                  payload.size(),
                  cudaMemcpyHostToDevice,
                  stream),
            "q5k payload upload");
      } else {
        throw std::runtime_error("unknown mixed sequence stage kind");
      }
    }

    std::vector<float> host_activation(max_cols);
    for (std::uint32_t col = 0; col < max_cols; ++col) {
      const float base = static_cast<float>((col * 131u + 17u) & 1023u) / 1024.0f;
      host_activation[col] = base - 0.5f;
    }
    const auto upload_initial_activations = [&](const char* op) {
      for (std::size_t index = 0; index < layer_workspace_runtime.phases.size(); ++index) {
        const auto& phase = layer_workspace_runtime.phases[index];
        check(cudaMemcpyAsync(
                  phase.activation,
                  host_activation.data(),
                  static_cast<std::size_t>(phase.max_cols) * sizeof(float),
                  cudaMemcpyHostToDevice,
                  stream),
            op);
      }
    };
    const std::uint32_t initial_token = 0;
    upload_initial_activations("activation upload");
    check(cudaMemcpyAsync(token, &initial_token, sizeof(initial_token), cudaMemcpyHostToDevice, stream),
        "token upload");
    check(cudaStreamSynchronize(stream), "setup sync");

    const auto launch_stage = [&](std::size_t index, cudaStream_t launch_stream) {
      const auto& stage = stages[index];
      const auto& phase_workspace =
          layer_workspace_runtime.phases[stage_workspace_phase_indices[index]];
      if (stage.kind == StageKind::Q4K) {
        check(launch_q4k_stage(
                  stage,
                  q4k_payload_base,
                  q4k_metadata_base,
                  phase_workspace.activation,
                  phase_workspace.logits,
                  kv,
                  token,
                  kv_words,
                  options.page_words,
                  options.active_pages,
                  launch_stream),
            "mixed q4k launch");
      } else if (stage.kind == StageKind::IQ2S) {
        const TensorShape shape = raw_manifest_tensor_shape(stage.iq, "IQ2_S");
        check(rtxllm_launch_iq2s_matvec_decode_probe(
                  iq_payload_base + stage.payload_offset,
                  phase_workspace.activation,
                  phase_workspace.logits,
                  kv,
                  token,
                  shape.rows,
                  shape.cols,
                  stage.rows,
                  kv_words,
                  options.page_words,
                  options.active_pages,
                  launch_stream),
            "mixed iq2s launch");
      } else if (stage.kind == StageKind::IQ3S) {
        const TensorShape shape = raw_manifest_tensor_shape(stage.iq, "IQ3_S");
        check(rtxllm_launch_iq3s_matvec_decode_probe(
                  iq_payload_base + stage.payload_offset,
                  phase_workspace.activation,
                  phase_workspace.logits,
                  kv,
                  token,
                  shape.rows,
                  shape.cols,
                  stage.rows,
                  kv_words,
                  options.page_words,
                  options.active_pages,
                  launch_stream),
            "mixed iq3s launch");
      } else if (stage.kind == StageKind::Q5K) {
        const TensorShape shape = raw_manifest_tensor_shape(stage.iq, "Q5_K");
        check(rtxllm_launch_q5k_matvec_decode_probe(
                  q5k_payload_base + stage.payload_offset,
                  phase_workspace.activation,
                  phase_workspace.logits,
                  kv,
                  token,
                  shape.rows,
                  shape.cols,
                  stage.rows,
                  kv_words,
                  options.page_words,
                  options.active_pages,
                  launch_stream),
            "mixed q5k launch");
      } else {
        throw std::runtime_error("unknown mixed sequence stage kind");
      }
    };

    const auto launch_feedback = [&](std::size_t index, cudaStream_t launch_stream) {
      if (!options.chain_activation) {
        return;
      }
      const auto& edge =
          rtxllm::find_layer_activation_feedback_edge(layer_dataflow_plan, index);
      const auto& source_workspace =
          layer_workspace_runtime.phases[edge.source_phase_index];
      const auto& target_workspace =
          layer_workspace_runtime.phases[edge.target_phase_index];
      if (ffn_phase_edge_caches_gate(options.ffn_phase_mode, edge, stages)) {
        if (ffn_gate_cache == nullptr) {
          throw std::runtime_error("FFN gate cache is not allocated");
        }
        const std::uint32_t accumulate =
            stages[edge.source_stage].role == "ffn_gate_exps.weight" ? 0u : 1u;
        check(rtxllm_launch_rmsnorm_silu_cache(
                  source_workspace.logits,
                  ffn_gate_cache,
                  edge.source_rows,
                  ffn_gate_cache_cols,
                  1.0e-6f,
                  accumulate,
                  launch_stream),
            "gated FFN phase gate cache");
        return;
      }
      if (ffn_phase_edge_caches_up(options.ffn_phase_mode, edge, stages)) {
        if (ffn_up_cache == nullptr) {
          throw std::runtime_error("FFN up cache is not allocated");
        }
        const std::uint32_t accumulate =
            stages[edge.source_stage].role == "ffn_up_exps.weight" ? 0u : 1u;
        if (ffn_phase_edge_finalizes_gated_product(
                options.ffn_phase_mode,
                edge,
                stages)) {
          if (ffn_gate_cache == nullptr) {
            throw std::runtime_error("FFN gate cache is not allocated");
          }
          check(rtxllm_launch_rmsnorm_cache_and_gated_product(
                    source_workspace.logits,
                    ffn_gate_cache,
                    ffn_up_cache,
                    target_workspace.activation,
                    edge.source_rows,
                    ffn_up_cache_cols,
                    edge.target_cols,
                    1.0e-6f,
                    accumulate,
                    launch_stream),
              "gated FFN phase up cache product");
        } else {
          check(rtxllm_launch_rmsnorm_cache(
                    source_workspace.logits,
                    ffn_up_cache,
                    edge.source_rows,
                    ffn_up_cache_cols,
                    1.0e-6f,
                    accumulate,
                    launch_stream),
              "gated FFN phase up cache");
        }
        return;
      }
      if (use_ssm_source_parameterized_fused &&
          edge.source_phase == rtxllm::LayerPhase::Ssm &&
          edge.target_phase == rtxllm::LayerPhase::Ssm) {
        if (ssm_scan_scratch == nullptr || ssm_scan_scratch_values == 0 ||
            ssm_recurrent_state_values == 0) {
          throw std::runtime_error("SSM source parameter scratch is not allocated");
        }
        const auto& role = stages[edge.source_stage].role;
        if (role == "ssm_alpha.weight" || role == "ssm_beta.weight") {
          const std::uint32_t parameter_slot =
              role == "ssm_alpha.weight" ? 0u : 1u;
          check(rtxllm_launch_ssm_rmsnorm_feedback_parameter_cache(
                    source_workspace.logits,
                    target_workspace.activation,
                    ssm_scan_scratch,
                    edge.source_rows,
                    edge.target_cols,
                    ssm_scan_scratch_values,
                    ssm_recurrent_state_values,
                    parameter_slot,
                    1.0e-6f,
                    kSsmSourceParamCacheLogitScale,
                    kSsmSourceParamCacheActivationScale,
                    launch_stream),
              "SSM fused rmsnorm source parameter cache feedback");
          return;
        }
      }
      switch (options.feedback_mode) {
        case FeedbackMode::Synthetic:
          check(rtxllm_launch_activation_feedback(
                    source_workspace.logits,
                    target_workspace.activation,
                    token,
                    edge.source_rows,
                    edge.target_cols,
                    edge.stage_id,
                    launch_stream),
              "activation feedback");
          break;
        case FeedbackMode::RmsNorm:
          check(rtxllm_launch_rmsnorm_feedback(
                    source_workspace.logits,
                    target_workspace.activation,
                    edge.source_rows,
                    edge.target_cols,
                    1.0e-6f,
                    launch_stream),
              "rmsnorm feedback");
          break;
        case FeedbackMode::PhaseAwareFfnSilu:
          if (feedback_edge_uses_ffn_silu(options.feedback_mode, edge)) {
            check(rtxllm_launch_silu_rmsnorm_feedback(
                      source_workspace.logits,
                      target_workspace.activation,
                      edge.source_rows,
                      edge.target_cols,
                      1.0e-6f,
                      launch_stream),
                "phase-aware ffn silu feedback");
          } else {
            check(rtxllm_launch_rmsnorm_feedback(
                      source_workspace.logits,
                      target_workspace.activation,
                      edge.source_rows,
                      edge.target_cols,
                      1.0e-6f,
                      launch_stream),
                "phase-aware rmsnorm feedback");
          }
          break;
        case FeedbackMode::PhaseAwareFfnGatedSilu:
          if (feedback_edge_caches_ffn_gate(options.feedback_mode, edge, stages)) {
            if (ffn_gate_cache == nullptr) {
              throw std::runtime_error("FFN gate cache is not allocated");
            }
            check(rtxllm_launch_rmsnorm_feedback_with_silu_cache(
                      source_workspace.logits,
                      target_workspace.activation,
                      ffn_gate_cache,
                      edge.source_rows,
                      edge.target_cols,
                      1.0e-6f,
                      launch_stream),
                "phase-aware ffn gate cache feedback");
          } else if (feedback_edge_uses_ffn_gated_silu(
                         options.feedback_mode,
                         edge,
                         stages)) {
            if (ffn_gate_cache == nullptr) {
              throw std::runtime_error("FFN gate cache is not allocated");
            }
            check(rtxllm_launch_gated_silu_rmsnorm_feedback(
                      source_workspace.logits,
                      ffn_gate_cache,
                      target_workspace.activation,
                      edge.source_rows,
                      ffn_gate_cache_cols,
                      edge.target_cols,
                      1.0e-6f,
                      launch_stream),
                "phase-aware ffn gated silu feedback");
          } else {
            check(rtxllm_launch_rmsnorm_feedback(
                      source_workspace.logits,
                      target_workspace.activation,
                      edge.source_rows,
                      edge.target_cols,
                      1.0e-6f,
                      launch_stream),
                "phase-aware gated rmsnorm feedback");
          }
          break;
      }
      if (use_ssm_source_parameterized &&
          !use_ssm_source_parameterized_fused &&
          edge.source_phase == rtxllm::LayerPhase::Ssm &&
          edge.target_phase == rtxllm::LayerPhase::Ssm) {
        if (ssm_scan_scratch == nullptr || ssm_scan_scratch_values == 0 ||
            ssm_recurrent_state_values == 0) {
          throw std::runtime_error("SSM source parameter scratch is not allocated");
        }
        const auto& role = stages[edge.source_stage].role;
        if (role == "ssm_alpha.weight" || role == "ssm_beta.weight") {
          const std::uint32_t parameter_slot =
              role == "ssm_alpha.weight" ? 0u : 1u;
          check(rtxllm_launch_ssm_source_parameter_cache(
                    source_workspace.logits,
                    target_workspace.activation,
                    ssm_scan_scratch,
                    edge.source_rows,
                    ssm_scan_scratch_values,
                    ssm_recurrent_state_values,
                    edge.target_cols,
                    parameter_slot,
                    kSsmSourceParamCacheLogitScale,
                    kSsmSourceParamCacheActivationScale,
                    launch_stream),
              "SSM source parameter cache update");
        }
      }
    };

    const auto launch_attention_qkv_scratch_update =
        [&](const rtxllm::LayerActivationFeedbackEdge& edge,
            cudaStream_t launch_stream) {
          if (!use_attention_qkv_scratch) {
            return;
          }
          if (attention_qkv_scratch == nullptr ||
              attention_qkv_scratch_values == 0) {
            throw std::runtime_error("attention QKV scratch is not allocated");
          }
          const auto& source_workspace =
              layer_workspace_runtime.phases[edge.source_phase_index];
          const auto& target_workspace =
              layer_workspace_runtime.phases[edge.target_phase_index];
          check(rtxllm_launch_attention_qkv_scratch(
                    source_workspace.logits,
                    attention_qkv_scratch,
                    target_workspace.activation,
                    edge.source_rows,
                    attention_qkv_scratch_values,
                    edge.target_cols,
                    kAttentionQkvResidualScale,
                    launch_stream),
              "attention QKV scratch phase update");
          if (use_attention_qkv_reduce) {
            check(rtxllm_launch_attention_qkv_reduce(
                      attention_qkv_scratch,
                      target_workspace.activation,
                      edge.source_rows,
                      attention_qkv_scratch_values,
                      edge.target_cols,
                      kAttentionQkvReduceScale,
                      launch_stream),
                "attention QKV reduce phase update");
          }
          if (use_attention_qkv_window) {
            check(rtxllm_launch_attention_qkv_window(
                      attention_qkv_scratch,
                      target_workspace.activation,
                      edge.source_rows,
                      attention_qkv_scratch_values,
                      edge.target_cols,
                      kAttentionQkvWindowSize,
                      kAttentionQkvWindowSoftmaxScale,
                      kAttentionQkvWindowResidualScale,
                      launch_stream),
                "attention QKV window phase update");
          }
          if (use_attention_qkv_head_window) {
            check(rtxllm_launch_attention_qkv_head_window(
                      attention_qkv_scratch,
                      target_workspace.activation,
                      edge.source_rows,
                      attention_qkv_scratch_values,
                      edge.target_cols,
                      kAttentionQkvHeadWindowHeadCount,
                      kAttentionQkvHeadWindowSize,
                      kAttentionQkvHeadWindowSoftmaxScale,
                      kAttentionQkvHeadWindowResidualScale,
                      launch_stream),
                "attention QKV head-window phase update");
          }
          if (use_attention_qkv_head_dim_window) {
            check(rtxllm_launch_attention_qkv_head_dim_window(
                      attention_qkv_scratch,
                      target_workspace.activation,
                      edge.source_rows,
                      attention_qkv_scratch_values,
                      edge.target_cols,
                      kAttentionQkvHeadDimWindowHeadCount,
                      kAttentionQkvHeadDimWindowHeadDim,
                      kAttentionQkvHeadDimWindowSize,
                      kAttentionQkvHeadDimWindowSoftmaxScale,
                      kAttentionQkvHeadDimWindowResidualScale,
                      launch_stream),
                "attention QKV head-dim-window phase update");
          }
          if (use_attention_qkv_head_tile_window) {
            check(rtxllm_launch_attention_qkv_head_tile_window(
                      attention_qkv_scratch,
                      target_workspace.activation,
                      edge.source_rows,
                      attention_qkv_scratch_values,
                      edge.target_cols,
                      kAttentionQkvHeadTileWindowHeadCount,
                      kAttentionQkvHeadTileWindowHeadDim,
                      kAttentionQkvHeadTileWindowSize,
                      kAttentionQkvHeadTileWindowSoftmaxScale,
                      kAttentionQkvHeadTileWindowResidualScale,
                      launch_stream),
                "attention QKV head-tile-window phase update");
          }
          if (use_attention_qkv_head_group_window) {
            check(rtxllm_launch_attention_qkv_head_group_window(
                      attention_qkv_scratch,
                      target_workspace.activation,
                      edge.source_rows,
                      attention_qkv_scratch_values,
                      edge.target_cols,
                      kAttentionQkvHeadGroupWindowHeadCount,
                      kAttentionQkvHeadGroupWindowHeadDim,
                      kAttentionQkvHeadGroupWindowSize,
                      kAttentionQkvHeadGroupWindowContextsPerBlock,
                      kAttentionQkvHeadGroupWindowSoftmaxScale,
                      kAttentionQkvHeadGroupWindowResidualScale,
                      launch_stream),
                "attention QKV head-group-window phase update");
          }
          if (use_attention_qkv_head_group_rope_window) {
            check(rtxllm_launch_attention_qkv_head_group_rope_window(
                      attention_qkv_scratch,
                      target_workspace.activation,
                      edge.source_rows,
                      attention_qkv_scratch_values,
                      edge.target_cols,
                      kAttentionQkvHeadGroupRopeWindowHeadCount,
                      kAttentionQkvHeadGroupRopeWindowHeadDim,
                      kAttentionQkvHeadGroupRopeWindowSize,
                      kAttentionQkvHeadGroupRopeWindowContextsPerBlock,
                      kAttentionQkvHeadGroupRopeWindowThetaScale,
                      kAttentionQkvHeadGroupRopeWindowSoftmaxScale,
                      kAttentionQkvHeadGroupRopeWindowResidualScale,
                      launch_stream),
                "attention QKV head-group-rope-window phase update");
          }
        };

    const auto launch_attention_qkv_head_group_fused_update =
        [&](const rtxllm::LayerActivationFeedbackEdge& edge,
            cudaStream_t launch_stream) {
          if (!use_attention_qkv_head_group_fused) {
            return;
          }
          const auto& source_workspace =
              layer_workspace_runtime.phases[edge.source_phase_index];
          const auto& target_workspace =
              layer_workspace_runtime.phases[edge.target_phase_index];
          check(rtxllm_launch_attention_qkv_head_group_fused(
                    source_workspace.logits,
                    target_workspace.activation,
                    edge.source_rows,
                    edge.target_cols,
                    kAttentionQkvHeadGroupFusedHeadCount,
                    kAttentionQkvHeadGroupFusedHeadDim,
                    kAttentionQkvHeadGroupFusedSize,
                    kAttentionQkvHeadGroupFusedContextsPerBlock,
                    kAttentionQkvResidualScale,
                    kAttentionQkvHeadGroupFusedSoftmaxScale,
                    kAttentionQkvHeadGroupFusedResidualScale,
                    launch_stream),
              "attention QKV head-group-fused phase update");
        };

    const auto launch_ssm_recurrent_state_update =
        [&](const rtxllm::LayerActivationFeedbackEdge& edge,
            cudaStream_t launch_stream) {
          if (!use_ssm_recurrent_state) {
            return;
          }
          if (ssm_recurrent_state == nullptr || ssm_recurrent_state_values == 0) {
            throw std::runtime_error("SSM recurrent state is not allocated");
          }
          const auto& source_workspace =
              layer_workspace_runtime.phases[edge.source_phase_index];
          const auto& target_workspace =
              layer_workspace_runtime.phases[edge.target_phase_index];
          if (use_ssm_source_parameterized ||
              use_ssm_selective_scan ||
              use_ssm_scan_scratch) {
            if (ssm_scan_scratch == nullptr || ssm_scan_scratch_values == 0) {
              throw std::runtime_error("SSM scan scratch is not allocated");
            }
          }
          if (use_ssm_source_parameterized) {
            check(rtxllm_launch_ssm_source_parameterized_scan(
                      source_workspace.logits,
                      ssm_recurrent_state,
                      ssm_scan_scratch,
                      target_workspace.activation,
                      edge.source_rows,
                      ssm_recurrent_state_values,
                      ssm_scan_scratch_values,
                      edge.target_cols,
                      kSsmStateDecay,
                      kSsmStateLogitScale,
                      kSsmScanScratchScale,
                      kSsmStateActivationScale,
                      kSsmSourceParamAlphaScale,
                      kSsmSourceParamBetaScale,
                      kSsmSourceParamGateLogitScale,
                      kSsmSourceParamGateActivationScale,
                      kSsmStateOutputScale,
                      launch_stream),
                "SSM source-parameterized phase update");
            return;
          }
          if (use_ssm_selective_scan) {
            check(rtxllm_launch_ssm_selective_scan(
                      source_workspace.logits,
                      ssm_recurrent_state,
                      ssm_scan_scratch,
                      target_workspace.activation,
                      edge.source_rows,
                      ssm_recurrent_state_values,
                      ssm_scan_scratch_values,
                      edge.target_cols,
                      kSsmStateDecay,
                      kSsmStateLogitScale,
                      kSsmScanScratchScale,
                      kSsmStateActivationScale,
                      kSsmSelectiveGateLogitScale,
                      kSsmSelectiveGateActivationScale,
                      kSsmSelectiveGateScratchScale,
                      kSsmStateOutputScale,
                      launch_stream),
                "SSM selective scan phase update");
            return;
          }
          if (use_ssm_scan_scratch) {
            check(rtxllm_launch_ssm_scan_scratch(
                      source_workspace.logits,
                      ssm_recurrent_state,
                      ssm_scan_scratch,
                      target_workspace.activation,
                      edge.source_rows,
                      ssm_recurrent_state_values,
                      ssm_scan_scratch_values,
                      edge.target_cols,
                      kSsmStateDecay,
                      kSsmStateLogitScale,
                      kSsmScanScratchScale,
                      kSsmStateActivationScale,
                      kSsmStateOutputScale,
                      launch_stream),
                "SSM scan scratch phase update");
            return;
          }
          check(rtxllm_launch_ssm_recurrent_state(
                    source_workspace.logits,
                    ssm_recurrent_state,
                    target_workspace.activation,
                    edge.source_rows,
                    ssm_recurrent_state_values,
                    edge.target_cols,
                    kSsmStateDecay,
                    kSsmStateLogitScale,
                    kSsmStateActivationScale,
                    kSsmStateOutputScale,
                    launch_stream),
              "SSM recurrent state phase update");
        };

    const auto launch_output_token_sample =
        [&](const rtxllm::LayerPhaseExecution& phase,
            cudaStream_t launch_stream) {
          if (!use_output_final_token) {
            return;
          }
          if (phase.count == 0u || phase.begin >= stages.size()) {
            throw std::runtime_error("output token phase has no output stage");
          }
          const std::size_t output_stage_index = phase.begin + phase.count - 1u;
          if (output_stage_index >= stages.size()) {
            throw std::runtime_error("output token phase exceeds stage count");
          }
          const auto& output_workspace =
              layer_workspace_runtime.phases[
                  stage_workspace_phase_indices[output_stage_index]];
          check(rtxllm_launch_output_token_sample(
                    output_workspace.logits,
                    token,
                    stages[output_stage_index].rows,
                    kOutputPhaseVocabSize,
                    kOutputPhaseTokenOffset,
                    launch_stream),
              "output phase token sample");
        };

    const auto launch_phase = [&](const rtxllm::LayerPhaseExecution& phase,
                                  cudaStream_t launch_stream) {
      auto counters = rtxllm::replay_layer_phase(
          phase,
          stages.size(),
          [&](std::size_t index) {
            launch_stage(index, launch_stream);
          },
          [&](std::size_t index) {
            launch_feedback(index, launch_stream);
          });
      if (phase.phase == rtxllm::LayerPhase::Attention &&
          attention_ffn_edge != nullptr) {
        launch_attention_qkv_scratch_update(*attention_ffn_edge, launch_stream);
        launch_attention_qkv_head_group_fused_update(
            *attention_ffn_edge,
            launch_stream);
      }
      if (phase.phase == rtxllm::LayerPhase::Ssm && ssm_output_edge != nullptr) {
        launch_ssm_recurrent_state_update(*ssm_output_edge, launch_stream);
      }
      if (phase.phase == rtxllm::LayerPhase::Output) {
        launch_output_token_sample(phase, launch_stream);
      }
      if (!phase_scratch_runtimes.empty()) {
        const auto& phase_workspace =
            layer_workspace_runtime.phases[stage_workspace_phase_indices[phase.begin]];
        for (const auto* scratch : phase_scratch_runtimes) {
          if (scratch->phase != phase.phase) {
            continue;
          }
          check(rtxllm_launch_phase_scratch_digest(
                    phase_workspace.activation,
                    phase_workspace.logits,
                    scratch->data,
                    phase_workspace.max_rows,
                    phase_workspace.max_cols,
                    scratch->values,
                    layer_phase_scratch_id(phase.phase),
                    launch_stream),
              "phase scratch digest");
        }
      }
      return counters;
    };

    const auto launch_attention_phase = [&](const rtxllm::LayerPhaseExecution& phase,
                                            cudaStream_t launch_stream) {
      return launch_phase(phase, launch_stream);
    };
    const auto launch_ffn_phase = [&](const rtxllm::LayerPhaseExecution& phase,
                                      cudaStream_t launch_stream) {
      return launch_phase(phase, launch_stream);
    };
    const auto launch_ssm_phase = [&](const rtxllm::LayerPhaseExecution& phase,
                                      cudaStream_t launch_stream) {
      return launch_phase(phase, launch_stream);
    };
    const auto launch_output_phase = [&](const rtxllm::LayerPhaseExecution& phase,
                                         cudaStream_t launch_stream) {
      return launch_phase(phase, launch_stream);
    };

    const auto launch_sequence = [&](cudaStream_t launch_stream) {
      return rtxllm::replay_layer_execution_plan_by_phase(
          layer_execution_plan,
          stages.size(),
          [&](const rtxllm::LayerPhaseExecution& phase) {
            return launch_attention_phase(phase, launch_stream);
          },
          [&](const rtxllm::LayerPhaseExecution& phase) {
            return launch_ffn_phase(phase, launch_stream);
          },
          [&](const rtxllm::LayerPhaseExecution& phase) {
            return launch_ssm_phase(phase, launch_stream);
          },
          [&](const rtxllm::LayerPhaseExecution& phase) {
            return launch_output_phase(phase, launch_stream);
          });
    };

    const auto reset_execution_state = [&](const char* workspace_op,
                                           const char* kv_op,
                                           const char* activation_op,
                                           const char* token_op) {
      const std::uint32_t zero_token = 0;
      check(cudaMemsetAsync(workspace, 0, workspace_bytes, stream), workspace_op);
      check(cudaMemsetAsync(kv, 0, kv_bytes, stream), kv_op);
      upload_initial_activations(activation_op);
      check(cudaMemcpyAsync(
                token,
                &zero_token,
                sizeof(zero_token),
                cudaMemcpyHostToDevice,
                stream),
          token_op);
    };

    std::vector<cudaEvent_t> phase_start_events(layer_execution_plan.phases.size(), nullptr);
    std::vector<cudaEvent_t> phase_stop_events(layer_execution_plan.phases.size(), nullptr);
    for (std::size_t index = 0; index < layer_execution_plan.phases.size(); ++index) {
      check(cudaEventCreate(&phase_start_events[index]), "cudaEventCreate phase start");
      check(cudaEventCreate(&phase_stop_events[index]), "cudaEventCreate phase stop");
    }

    ReferenceResult reference;
    if (options.check_reference) {
      reset_execution_state(
          "reference workspace reset",
          "reference kv reset",
          "reference activation reset",
          "reference token reset");

      std::vector<ObservedStageSnapshot> observed_stages;
      observed_stages.reserve(stages.size());
      for (std::size_t index = 0; index < stages.size(); ++index) {
        const auto& stage = stages[index];
        launch_stage(index, stream);
        const auto& phase_workspace =
            layer_workspace_runtime.phases[stage_workspace_phase_indices[index]];

        ObservedStageSnapshot snapshot;
        snapshot.logits.resize(stage.rows, 0.0f);
        check(cudaMemcpyAsync(
                  snapshot.logits.data(),
                  phase_workspace.logits,
                  sizeof(float) * stage.rows,
                  cudaMemcpyDeviceToHost,
                  stream),
            "reference stage logits copy");
        check(cudaMemcpyAsync(
                  &snapshot.token,
                  token,
                  sizeof(snapshot.token),
                  cudaMemcpyDeviceToHost,
                  stream),
            "reference stage token copy");
        check(cudaStreamSynchronize(stream), "reference stage sync");
        observed_stages.push_back(std::move(snapshot));

        launch_feedback(index, stream);
        if (use_attention_phase_kernel ||
            use_ssm_recurrent_state ||
            use_output_final_token) {
          const auto& edge =
              rtxllm::find_layer_activation_feedback_edge(layer_dataflow_plan, index);
          if (use_attention_qkv_scratch &&
              edge.source_phase == rtxllm::LayerPhase::Attention &&
              edge.target_phase == rtxllm::LayerPhase::Ffn) {
            launch_attention_qkv_scratch_update(edge, stream);
          }
          if (use_attention_qkv_head_group_fused &&
              edge.source_phase == rtxllm::LayerPhase::Attention &&
              edge.target_phase == rtxllm::LayerPhase::Ffn) {
            launch_attention_qkv_head_group_fused_update(edge, stream);
          }
          if (use_ssm_recurrent_state &&
              edge.source_phase == rtxllm::LayerPhase::Ssm &&
              edge.target_phase == rtxllm::LayerPhase::Output) {
            launch_ssm_recurrent_state_update(edge, stream);
          }
          if (use_output_final_token &&
              edge.source_phase == rtxllm::LayerPhase::Output) {
            const auto phase_it = std::find_if(
                layer_execution_plan.phases.begin(),
                layer_execution_plan.phases.end(),
                [&](const rtxllm::LayerPhaseExecution& phase) {
                  return phase.phase == rtxllm::LayerPhase::Output &&
                      edge.source_stage >= phase.begin &&
                      edge.source_stage < phase.begin + phase.count;
                });
            if (phase_it == layer_execution_plan.phases.end()) {
              throw std::runtime_error("output token phase is missing from execution plan");
            }
            if (edge.source_stage + 1u == phase_it->begin + phase_it->count) {
              launch_output_token_sample(*phase_it, stream);
            }
          }
        }
      }

      std::vector<float> observed_logits(max_rows, 0.0f);
      std::vector<float> observed_kv(kv_words, 0.0f);
      std::uint32_t observed_token = 0;
      const auto final_stage_index = stages.empty() ? 0 : stages.size() - 1;
      const auto final_rows = stages.empty() ? 0 : stages.back().rows;
      const auto& final_workspace =
          layer_workspace_runtime.phases[stage_workspace_phase_indices[final_stage_index]];
      check(cudaMemcpyAsync(
                observed_logits.data(),
                final_workspace.logits,
                sizeof(float) * final_rows,
                cudaMemcpyDeviceToHost,
                stream),
          "reference logits copy");
      check(cudaMemcpyAsync(
                observed_kv.data(),
                kv,
                sizeof(float) * kv_words,
                cudaMemcpyDeviceToHost,
                stream),
          "reference kv copy");
      check(cudaMemcpyAsync(
                &observed_token,
                token,
                sizeof(observed_token),
                cudaMemcpyDeviceToHost,
                stream),
          "reference token copy");
      check(cudaStreamSynchronize(stream), "reference sync");

      reference = run_mixed_reference(
          stages,
          options,
          layer_dataflow_plan,
          host_activation,
          observed_stages,
          observed_logits,
          observed_kv,
          observed_token,
          ssm_scan_scratch_values);

      reset_execution_state(
          "post-reference workspace reset",
          "post-reference kv reset",
          "post-reference activation reset",
          "post-reference token reset");
      check(cudaStreamSynchronize(stream), "post-reference sync");
    }

    std::uint32_t* host_token = nullptr;
    check(cudaHostAlloc(
              reinterpret_cast<void**>(&host_token),
              sizeof(std::uint32_t),
              cudaHostAllocDefault),
        "cudaHostAlloc");

    const auto warmup_replay = launch_sequence(stream);
    check(cudaMemcpyAsync(host_token, token, sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream),
        "warmup token memcpy");
    check(cudaStreamSynchronize(stream), "warmup sync");

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    const auto graph_memory_before_capture =
        device_memory_snapshot("cudaMemGetInfo before graph capture");
    const auto capture_start = std::chrono::steady_clock::now();
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
    const auto graph_capture_replay = launch_sequence(stream);
    check(cudaMemcpyAsync(host_token, token, sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream),
        "captured token memcpy");
    check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    const auto capture_stop = std::chrono::steady_clock::now();
    const auto graph_memory_after_capture =
        device_memory_snapshot("cudaMemGetInfo after graph capture");

    check(cudaGraphInstantiate(&graph_exec, graph, 0), "cudaGraphInstantiate");
    const auto graph_memory_after_instantiate =
        device_memory_snapshot("cudaMemGetInfo after graph instantiate");
    const auto upload_start = std::chrono::steady_clock::now();
    check(cudaGraphUpload(graph_exec, stream), "cudaGraphUpload");
    check(cudaStreamSynchronize(stream), "graph upload sync");
    const auto upload_stop = std::chrono::steady_clock::now();
    const auto graph_memory_after_upload =
        device_memory_snapshot("cudaMemGetInfo after graph upload");

    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<std::size_t>(options.steps));
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < options.steps; ++step) {
      const auto step_start = std::chrono::steady_clock::now();
      check(cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch");
      check(cudaStreamSynchronize(stream), "graph replay sync");
      const auto step_stop = std::chrono::steady_clock::now();
      latencies_ms.push_back(
          std::chrono::duration<double, std::milli>(step_stop - step_start).count());
    }
    const auto stop = std::chrono::steady_clock::now();

    std::vector<std::vector<double>> phase_latencies_ms(layer_execution_plan.phases.size());
    for (auto& phase_latencies : phase_latencies_ms) {
      phase_latencies.reserve(static_cast<std::size_t>(options.steps));
    }
    if (!layer_execution_plan.phases.empty()) {
      reset_execution_state(
          "phase timing workspace reset",
          "phase timing kv reset",
          "phase timing activation reset",
          "phase timing token reset");
      check(cudaStreamSynchronize(stream), "phase timing reset sync");

      for (int step = 0; step < options.steps; ++step) {
        for (std::size_t index = 0; index < layer_execution_plan.phases.size(); ++index) {
          check(cudaEventRecord(phase_start_events[index], stream),
              "cudaEventRecord phase start");
          (void)launch_phase(layer_execution_plan.phases[index], stream);
          check(cudaEventRecord(phase_stop_events[index], stream),
              "cudaEventRecord phase stop");
        }
        check(cudaEventSynchronize(phase_stop_events.back()), "phase timing sync");
        for (std::size_t index = 0; index < layer_execution_plan.phases.size(); ++index) {
          float elapsed_ms = 0.0f;
          check(cudaEventElapsedTime(
                    &elapsed_ms,
                    phase_start_events[index],
                    phase_stop_events[index]),
              "cudaEventElapsedTime phase");
          phase_latencies_ms[index].push_back(static_cast<double>(elapsed_ms));
        }
      }
    }

    const double wall_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    const double capture_ms =
        std::chrono::duration<double, std::milli>(capture_stop - capture_start).count();
    const double upload_ms =
        std::chrono::duration<double, std::milli>(upload_stop - upload_start).count();
    const double mean_ms =
        std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / latencies_ms.size();
    std::array<double, 4> phase_timing_mean_ms = {};
    std::array<double, 4> phase_timing_p50_ms = {};
    std::array<double, 4> phase_timing_p95_ms = {};
    std::array<double, 4> phase_timing_p99_ms = {};
    std::array<std::size_t, 4> phase_timing_samples = {};
    const auto phase_timing_slot =
        [](rtxllm::LayerPhase phase) -> std::size_t {
      switch (phase) {
        case rtxllm::LayerPhase::Attention:
          return 0;
        case rtxllm::LayerPhase::Ffn:
          return 1;
        case rtxllm::LayerPhase::Ssm:
          return 2;
        case rtxllm::LayerPhase::Output:
          return 3;
        case rtxllm::LayerPhase::Unknown:
          break;
      }
      return 0;
    };
    for (std::size_t index = 0; index < layer_execution_plan.phases.size(); ++index) {
      const auto slot =
          phase_timing_slot(layer_execution_plan.phases[index].phase);
      phase_timing_samples[slot] = phase_latencies_ms[index].size();
      phase_timing_mean_ms[slot] = mean_value(phase_latencies_ms[index]);
      phase_timing_p50_ms[slot] = percentile(phase_latencies_ms[index], 50);
      phase_timing_p95_ms[slot] = percentile(phase_latencies_ms[index], 95);
      phase_timing_p99_ms[slot] = percentile(phase_latencies_ms[index], 99);
    }
    const double phase_timing_sum_mean_ms =
        std::accumulate(
            phase_timing_mean_ms.begin(),
            phase_timing_mean_ms.end(),
            0.0);
    const double phase_timing_sum_p50_ms =
        std::accumulate(
            phase_timing_p50_ms.begin(),
            phase_timing_p50_ms.end(),
            0.0);
    const double phase_timing_sum_p95_ms =
        std::accumulate(
            phase_timing_p95_ms.begin(),
            phase_timing_p95_ms.end(),
            0.0);
    const double phase_timing_sum_p99_ms =
        std::accumulate(
            phase_timing_p99_ms.begin(),
            phase_timing_p99_ms.end(),
            0.0);
    const auto slowest_phase_p95_slot = static_cast<std::size_t>(
        std::distance(
            phase_timing_p95_ms.begin(),
            std::max_element(
                phase_timing_p95_ms.begin(),
                phase_timing_p95_ms.end())));
    const char* phase_timing_names[] = {"attention", "ffn", "ssm", "output"};
    const auto last_token = *host_token;
    const auto phase_aware_ffn_silu_feedback_edges =
        static_cast<std::size_t>(std::count_if(
            layer_dataflow_plan.activation_feedback_edges.begin(),
            layer_dataflow_plan.activation_feedback_edges.end(),
            [&](const rtxllm::LayerActivationFeedbackEdge& edge) {
              return feedback_edge_uses_ffn_silu(options.feedback_mode, edge);
            }));
    const auto phase_aware_ffn_gate_cache_edges =
        static_cast<std::size_t>(std::count_if(
            layer_dataflow_plan.activation_feedback_edges.begin(),
            layer_dataflow_plan.activation_feedback_edges.end(),
            [&](const rtxllm::LayerActivationFeedbackEdge& edge) {
              return feedback_edge_caches_ffn_gate(options.feedback_mode, edge, stages);
            }));
    const auto phase_aware_ffn_gated_silu_feedback_edges =
        static_cast<std::size_t>(std::count_if(
            layer_dataflow_plan.activation_feedback_edges.begin(),
            layer_dataflow_plan.activation_feedback_edges.end(),
            [&](const rtxllm::LayerActivationFeedbackEdge& edge) {
              return feedback_edge_uses_ffn_gated_silu(
                  options.feedback_mode,
                  edge,
                  stages);
            }));
    const auto ffn_phase_gate_cache_edges =
        static_cast<std::size_t>(std::count_if(
            layer_dataflow_plan.activation_feedback_edges.begin(),
            layer_dataflow_plan.activation_feedback_edges.end(),
            [&](const rtxllm::LayerActivationFeedbackEdge& edge) {
              return ffn_phase_edge_caches_gate(
                  options.ffn_phase_mode,
                  edge,
                  stages);
            }));
    const auto ffn_phase_up_cache_edges =
        static_cast<std::size_t>(std::count_if(
            layer_dataflow_plan.activation_feedback_edges.begin(),
            layer_dataflow_plan.activation_feedback_edges.end(),
            [&](const rtxllm::LayerActivationFeedbackEdge& edge) {
              return ffn_phase_edge_caches_up(
                  options.ffn_phase_mode,
                  edge,
                  stages);
            }));
    const auto ffn_phase_gated_product_edges =
        static_cast<std::size_t>(std::count_if(
            layer_dataflow_plan.activation_feedback_edges.begin(),
            layer_dataflow_plan.activation_feedback_edges.end(),
            [&](const rtxllm::LayerActivationFeedbackEdge& edge) {
              return ffn_phase_edge_finalizes_gated_product(
                  options.ffn_phase_mode,
                  edge,
                  stages);
            }));
    const auto phase_scratch_kernel_count =
        [&](rtxllm::LayerPhase phase) -> std::size_t {
      return static_cast<std::size_t>(std::count_if(
          phase_scratch_aux_workspaces.begin(),
          phase_scratch_aux_workspaces.end(),
          [&](const rtxllm::LayerAuxWorkspaceDescriptor& descriptor) {
            return descriptor.phase == phase;
          }));
    };
    const auto attention_phase_kernel_count =
        [&](rtxllm::LayerPhase phase) -> std::size_t {
      return use_attention_phase_kernel && phase == rtxllm::LayerPhase::Attention
          ? attention_phase_mode_kernel_count(options.attention_phase_mode)
          : 0u;
    };
    const auto ssm_phase_kernel_count =
        [&](rtxllm::LayerPhase phase) -> std::size_t {
      return use_ssm_recurrent_state && phase == rtxllm::LayerPhase::Ssm
          ? ssm_phase_mode_kernel_count(options.ssm_phase_mode)
          : 0u;
    };
    const auto output_phase_kernel_count =
        [&](rtxllm::LayerPhase phase) -> std::size_t {
      return use_output_final_token && phase == rtxllm::LayerPhase::Output
          ? 1u
          : 0u;
    };

    const auto graph_memory_before_destroy =
        device_memory_snapshot("cudaMemGetInfo before graph destroy");
    check(cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy");
    check(cudaGraphDestroy(graph), "cudaGraphDestroy");
    const auto graph_memory_after_destroy =
        device_memory_snapshot("cudaMemGetInfo after graph destroy");
    for (std::size_t index = 0; index < phase_start_events.size(); ++index) {
      check(cudaEventDestroy(phase_start_events[index]), "cudaEventDestroy phase start");
      check(cudaEventDestroy(phase_stop_events[index]), "cudaEventDestroy phase stop");
    }
    check(cudaFreeHost(host_token), "cudaFreeHost");
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");

    std::cout << "{\n";
    std::cout << "  \"path\": \""
              << (include_q5k ? "native_mixed_q4k_iq_q5k_sequence_graph"
                              : "native_mixed_q4k_iq_sequence_graph")
              << "\",\n";
    std::cout << "  \"label\": \"" << json_escape(options.label) << "\",\n";
    std::cout << "  \"admitted\": true,\n";
    std::cout << "  \"admission_policy\": \"strict\",\n";
    print_admission_breakdown(decision.breakdown);
    std::cout << "  \"q4k_manifest\": \"" << json_escape(options.q4k_manifest.string()) << "\",\n";
    std::cout << "  \"iq_manifest\": \"" << json_escape(options.iq_manifest.string()) << "\",\n";
    std::cout << "  \"iq_manifest_count\": " << iq_manifests.size() << ",\n";
    if (!options.extra_iq_manifests.empty()) {
      std::cout << "  \"extra_iq_manifests\": [";
      for (std::size_t index = 0; index < options.extra_iq_manifests.size(); ++index) {
        if (index != 0) {
          std::cout << ", ";
        }
        std::cout << "\"" << json_escape(options.extra_iq_manifests[index].string()) << "\"";
      }
      std::cout << "],\n";
    }
    if (include_q5k) {
      std::cout << "  \"q5k_manifest\": \"" << json_escape(options.q5k_manifest.string()) << "\",\n";
    }
    std::cout << "  \"layer\": " << options.layer << ",\n";
    std::cout << "  \"tensor_count\": " << stages.size() << ",\n";
    std::cout << "  \"q4k_tensor_count\": " << selected_q4k.size() << ",\n";
    std::cout << "  \"iq_tensor_count\": " << selected_iq.size() << ",\n";
    std::cout << "  \"q5k_tensor_count\": " << selected_q5k.size() << ",\n";
    std::cout << "  \"source_plan_validated\": "
              << (source_plan_validated ? "true" : "false") << ",\n";
    if (source_plan_validated) {
      std::cout << "  \"q4k_tensor_plan\": \"" << json_escape(q4k_plan_validation.path.string())
                << "\",\n";
      std::cout << "  \"iq_tensor_plan\": \"" << json_escape(iq_plan_validation.path.string())
                << "\",\n";
      if (include_q5k) {
        std::cout << "  \"q5k_tensor_plan\": \""
                  << json_escape(q5k_plan_validation.path.string()) << "\",\n";
      }
      std::cout << "  \"source_plan_tensors_checked\": "
                << (q4k_plan_validation.checked_tensors + iq_plan_validation.checked_tensors +
                    q5k_plan_validation.checked_tensors)
                << ",\n";
    }
    std::cout << "  \"steps\": " << options.steps << ",\n";
    std::cout << "  \"rows_limit\": " << options.rows_limit << ",\n";
    std::cout << "  \"stage_order_policy\": \""
              << stage_order_json_name(options.stage_order) << "\",\n";
    std::cout << "  \"stage_order\": \"" << stage_order_cli_name(options.stage_order) << "\",\n";
    std::cout << "  \"stage_role_plan_unknown_roles\": "
              << stage_role_plan_unknowns << ",\n";
    std::cout << "  \"stage_phase_unknown_roles\": "
              << stage_phase_unknowns << ",\n";
    std::cout << "  \"strict_layer_plan\": "
              << (options.strict_layer_plan ? "true" : "false") << ",\n";
    std::cout << "  \"strict_layer_plan_passed\": "
              << (strict_layer_plan_passed ? "true" : "false") << ",\n";
    std::cout << "  \"layer_plan_checked\": true,\n";
    std::cout << "  \"layer_plan_passed\": "
              << (layer_plan.passed ? "true" : "false") << ",\n";
    std::cout << "  \"layer_plan_message\": \""
              << json_escape(layer_plan.message) << "\",\n";
    std::cout << "  \"chain_activation\": " << (options.chain_activation ? "true" : "false") << ",\n";
    std::cout << "  \"feedback_mode\": \""
              << feedback_mode_name(options.feedback_mode) << "\",\n";
    std::cout << "  \"ffn_phase_mode\": \""
              << ffn_phase_mode_name(options.ffn_phase_mode) << "\",\n";
    std::cout << "  \"attention_phase_mode\": \""
              << attention_phase_mode_name(options.attention_phase_mode)
              << "\",\n";
    std::cout << "  \"ssm_phase_mode\": \""
              << ssm_phase_mode_name(options.ssm_phase_mode) << "\",\n";
    std::cout << "  \"output_phase_mode\": \""
              << output_phase_mode_name(options.output_phase_mode) << "\",\n";
    std::cout << "  \"ffn_phase_primitive_enabled\": "
              << (ffn_phase_mode_uses_gated_silu(options.ffn_phase_mode)
                  ? "true"
                  : "false") << ",\n";
    std::cout << "  \"attention_phase_qkv_scratch_enabled\": "
              << (use_attention_qkv_scratch ? "true" : "false") << ",\n";
    std::cout << "  \"attention_phase_qkv_reduce_enabled\": "
              << (use_attention_qkv_reduce ? "true" : "false") << ",\n";
    std::cout << "  \"attention_phase_qkv_window_enabled\": "
              << (use_attention_qkv_window ? "true" : "false") << ",\n";
    std::cout << "  \"attention_phase_qkv_head_window_enabled\": "
              << (use_attention_qkv_head_window ? "true" : "false") << ",\n";
    std::cout << "  \"attention_phase_qkv_head_dim_window_enabled\": "
              << (use_attention_qkv_head_dim_window ? "true" : "false")
              << ",\n";
    std::cout << "  \"attention_phase_qkv_head_tile_window_enabled\": "
              << (use_attention_qkv_head_tile_window ? "true" : "false")
              << ",\n";
    std::cout << "  \"attention_phase_qkv_head_group_window_enabled\": "
              << (use_attention_qkv_head_group_window ? "true" : "false")
              << ",\n";
    std::cout << "  \"attention_phase_qkv_head_group_rope_window_enabled\": "
              << (use_attention_qkv_head_group_rope_window ? "true" : "false")
              << ",\n";
    std::cout << "  \"attention_phase_qkv_head_group_fused_enabled\": "
              << (use_attention_qkv_head_group_fused ? "true" : "false")
              << ",\n";
    std::cout << "  \"ssm_phase_recurrent_state_enabled\": "
              << (use_ssm_recurrent_state ? "true" : "false") << ",\n";
    std::cout << "  \"ssm_phase_scan_scratch_enabled\": "
              << (use_ssm_scan_scratch ? "true" : "false") << ",\n";
    std::cout << "  \"ssm_phase_selective_scan_enabled\": "
              << (use_ssm_selective_scan ? "true" : "false") << ",\n";
    std::cout << "  \"ssm_phase_source_parameterized_enabled\": "
              << (use_ssm_source_parameterized ? "true" : "false") << ",\n";
    std::cout << "  \"ssm_phase_source_parameterized_fused_enabled\": "
              << (use_ssm_source_parameterized_fused ? "true" : "false")
              << ",\n";
    std::cout << "  \"output_phase_final_token_enabled\": "
              << (use_output_final_token ? "true" : "false") << ",\n";
    std::cout << "  \"output_phase_vocab_size\": "
              << kOutputPhaseVocabSize << ",\n";
    std::cout << "  \"output_phase_token_offset\": "
              << kOutputPhaseTokenOffset << ",\n";
    std::cout << "  \"ffn_phase_gate_cache_edges\": "
              << ffn_phase_gate_cache_edges << ",\n";
    std::cout << "  \"ffn_phase_up_cache_edges\": "
              << ffn_phase_up_cache_edges << ",\n";
    std::cout << "  \"ffn_phase_gated_product_edges\": "
              << ffn_phase_gated_product_edges << ",\n";
    std::cout << "  \"phase_aware_ffn_silu_feedback_edges\": "
              << phase_aware_ffn_silu_feedback_edges << ",\n";
    std::cout << "  \"phase_aware_ffn_gate_cache_edges\": "
              << phase_aware_ffn_gate_cache_edges << ",\n";
    std::cout << "  \"phase_aware_ffn_gated_silu_feedback_edges\": "
              << phase_aware_ffn_gated_silu_feedback_edges << ",\n";
    std::cout << "  \"activation_feedbacks_per_graph\": "
              << layer_execution_plan.activation_feedbacks_per_replay << ",\n";
    std::cout << "  \"kernel_launches_per_graph\": "
              << layer_execution_plan.total_kernels_per_replay << ",\n";
    std::cout << "  \"attention_phase_kernel_launches_per_graph\": "
              << attention_phase_kernel_launches_per_graph << ",\n";
    std::cout << "  \"ssm_phase_kernel_launches_per_graph\": "
              << ssm_phase_kernel_launches_per_graph << ",\n";
    std::cout << "  \"output_phase_kernel_launches_per_graph\": "
              << output_phase_kernel_launches_per_graph << ",\n";
    std::cout << "  \"phase_scratch_kernel_launches_per_graph\": "
              << phase_scratch_kernel_launches_per_graph << ",\n";
    std::cout << "  \"actual_kernel_launches_per_graph\": "
              << actual_kernel_launches_per_graph << ",\n";
    std::cout << "  \"layer_execution_phase_count\": "
              << layer_execution_plan.phase_count << ",\n";
    std::cout << "  \"layer_execution_stage_count\": "
              << layer_execution_plan.stage_count << ",\n";
    std::cout << "  \"layer_execution_matvec_kernels_per_graph\": "
              << layer_execution_plan.matvec_kernels_per_replay << ",\n";
    std::cout << "  \"layer_execution_activation_feedbacks_per_graph\": "
              << layer_execution_plan.activation_feedbacks_per_replay << ",\n";
    std::cout << "  \"layer_execution_kernel_launches_per_graph\": "
              << layer_execution_plan.total_kernels_per_replay << ",\n";
    std::cout << "  \"layer_execution_actual_kernel_launches_per_graph\": "
              << actual_kernel_launches_per_graph << ",\n";
    std::cout << "  \"layer_execution_timing_measured\": true,\n";
    std::cout << "  \"layer_execution_timing_method\": \"cuda_events_phase_replay\",\n";
    std::cout << "  \"layer_executor_contract\": \"core_scheduler_layer_executor\",\n";
    std::cout << "  \"layer_executor_dataflow_plan\": "
              << "\"phase_local_activation_feedback_edges\",\n";
    std::cout << "  \"layer_executor_activation_feedback_edges\": "
              << layer_dataflow_plan.activation_feedback_edges.size() << ",\n";
    std::cout << "  \"layer_executor_phase_crossing_feedback_edges\": "
              << layer_dataflow_plan.phase_crossing_feedbacks << ",\n";
    std::cout << "  \"layer_executor_wrap_feedback_edges\": "
              << layer_dataflow_plan.wrap_feedbacks << ",\n";
    std::cout << "  \"layer_executor_workspace_plan\": \"phase_local_workspace_descriptors\",\n";
    std::cout << "  \"layer_executor_workspace_phase_count\": "
              << layer_workspace_plan.phases.size() << ",\n";
    std::cout << "  \"layer_executor_workspace_max_rows\": "
              << layer_workspace_plan.max_rows << ",\n";
    std::cout << "  \"layer_executor_workspace_max_cols\": "
              << layer_workspace_plan.max_cols << ",\n";
    std::cout << "  \"layer_executor_workspace_logical_values_per_graph\": "
              << layer_workspace_plan.logical_values_per_replay << ",\n";
    std::cout << "  \"layer_executor_workspace_allocation_plan\": "
              << "\"phase_local_activation_logits_auxiliary\",\n";
    std::cout << "  \"layer_executor_workspace_bytes\": "
              << layer_workspace_allocation_plan.total_bytes << ",\n";
    std::cout << "  \"layer_executor_workspace_phase_bytes\": "
              << layer_workspace_allocation_plan.phase_bytes << ",\n";
    std::cout << "  \"layer_executor_aux_workspace_count\": "
              << layer_workspace_allocation_plan.auxiliaries.size() << ",\n";
    std::cout << "  \"layer_executor_aux_workspace_bytes\": "
              << layer_workspace_allocation_plan.auxiliary_bytes << ",\n";
    std::cout << "  \"phase_scratch_workspace_count\": "
              << phase_scratch_aux_workspaces.size() << ",\n";
    std::cout << "  \"phase_scratch_workspace_bytes\": "
              << phase_scratch_workspace_bytes << ",\n";
    std::cout << "  \"phase_scratch_digest_enabled\": "
              << (phase_scratch_kernel_launches_per_graph == 0 ? "false" : "true")
              << ",\n";
    std::cout << "  \"layer_executor_workspace_total_allocated_bytes\": "
              << workspace_bytes << ",\n";
    std::cout << "  \"ffn_gate_cache_workspace_bytes\": "
              << ffn_gate_cache_workspace_bytes << ",\n";
    std::cout << "  \"ffn_gate_cache_cols\": "
              << ffn_gate_cache_cols << ",\n";
    std::cout << "  \"ffn_up_cache_workspace_bytes\": "
              << ffn_up_cache_workspace_bytes << ",\n";
    std::cout << "  \"ffn_up_cache_cols\": "
              << ffn_up_cache_cols << ",\n";
    std::cout << "  \"attention_qkv_scratch_workspace_bytes\": "
              << attention_qkv_scratch_workspace_bytes << ",\n";
    std::cout << "  \"attention_qkv_scratch_values\": "
              << attention_qkv_scratch_values << ",\n";
    std::cout << "  \"attention_qkv_residual_scale\": "
              << kAttentionQkvResidualScale << ",\n";
    std::cout << "  \"attention_qkv_reduce_scale\": "
              << kAttentionQkvReduceScale << ",\n";
    std::cout << "  \"attention_qkv_window_size\": "
              << kAttentionQkvWindowSize << ",\n";
    std::cout << "  \"attention_qkv_window_softmax_scale\": "
              << kAttentionQkvWindowSoftmaxScale << ",\n";
    std::cout << "  \"attention_qkv_window_residual_scale\": "
              << kAttentionQkvWindowResidualScale << ",\n";
    std::cout << "  \"attention_qkv_head_window_head_count\": "
              << kAttentionQkvHeadWindowHeadCount << ",\n";
    std::cout << "  \"attention_qkv_head_window_size\": "
              << kAttentionQkvHeadWindowSize << ",\n";
    std::cout << "  \"attention_qkv_head_window_softmax_scale\": "
              << kAttentionQkvHeadWindowSoftmaxScale << ",\n";
    std::cout << "  \"attention_qkv_head_window_residual_scale\": "
              << kAttentionQkvHeadWindowResidualScale << ",\n";
    std::cout << "  \"attention_qkv_head_dim_window_head_count\": "
              << kAttentionQkvHeadDimWindowHeadCount << ",\n";
    std::cout << "  \"attention_qkv_head_dim_window_head_dim\": "
              << kAttentionQkvHeadDimWindowHeadDim << ",\n";
    std::cout << "  \"attention_qkv_head_dim_window_size\": "
              << kAttentionQkvHeadDimWindowSize << ",\n";
    std::cout << "  \"attention_qkv_head_dim_window_softmax_scale\": "
              << kAttentionQkvHeadDimWindowSoftmaxScale << ",\n";
    std::cout << "  \"attention_qkv_head_dim_window_residual_scale\": "
              << kAttentionQkvHeadDimWindowResidualScale << ",\n";
    std::cout << "  \"attention_qkv_head_tile_window_head_count\": "
              << kAttentionQkvHeadTileWindowHeadCount << ",\n";
    std::cout << "  \"attention_qkv_head_tile_window_head_dim\": "
              << kAttentionQkvHeadTileWindowHeadDim << ",\n";
    std::cout << "  \"attention_qkv_head_tile_window_size\": "
              << kAttentionQkvHeadTileWindowSize << ",\n";
    std::cout << "  \"attention_qkv_head_tile_window_softmax_scale\": "
              << kAttentionQkvHeadTileWindowSoftmaxScale << ",\n";
    std::cout << "  \"attention_qkv_head_tile_window_residual_scale\": "
              << kAttentionQkvHeadTileWindowResidualScale << ",\n";
    std::cout << "  \"attention_qkv_head_group_window_head_count\": "
              << kAttentionQkvHeadGroupWindowHeadCount << ",\n";
    std::cout << "  \"attention_qkv_head_group_window_head_dim\": "
              << kAttentionQkvHeadGroupWindowHeadDim << ",\n";
    std::cout << "  \"attention_qkv_head_group_window_size\": "
              << kAttentionQkvHeadGroupWindowSize << ",\n";
    std::cout << "  \"attention_qkv_head_group_window_contexts_per_block\": "
              << kAttentionQkvHeadGroupWindowContextsPerBlock << ",\n";
    std::cout << "  \"attention_qkv_head_group_window_softmax_scale\": "
              << kAttentionQkvHeadGroupWindowSoftmaxScale << ",\n";
    std::cout << "  \"attention_qkv_head_group_window_residual_scale\": "
              << kAttentionQkvHeadGroupWindowResidualScale << ",\n";
    std::cout << "  \"attention_qkv_head_group_rope_window_head_count\": "
              << kAttentionQkvHeadGroupRopeWindowHeadCount << ",\n";
    std::cout << "  \"attention_qkv_head_group_rope_window_head_dim\": "
              << kAttentionQkvHeadGroupRopeWindowHeadDim << ",\n";
    std::cout << "  \"attention_qkv_head_group_rope_window_size\": "
              << kAttentionQkvHeadGroupRopeWindowSize << ",\n";
    std::cout << "  \"attention_qkv_head_group_rope_window_contexts_per_block\": "
              << kAttentionQkvHeadGroupRopeWindowContextsPerBlock << ",\n";
    std::cout << "  \"attention_qkv_head_group_rope_window_theta_scale\": "
              << kAttentionQkvHeadGroupRopeWindowThetaScale << ",\n";
    std::cout << "  \"attention_qkv_head_group_rope_window_softmax_scale\": "
              << kAttentionQkvHeadGroupRopeWindowSoftmaxScale << ",\n";
    std::cout << "  \"attention_qkv_head_group_rope_window_residual_scale\": "
              << kAttentionQkvHeadGroupRopeWindowResidualScale << ",\n";
    std::cout << "  \"attention_qkv_head_group_fused_head_count\": "
              << kAttentionQkvHeadGroupFusedHeadCount << ",\n";
    std::cout << "  \"attention_qkv_head_group_fused_head_dim\": "
              << kAttentionQkvHeadGroupFusedHeadDim << ",\n";
    std::cout << "  \"attention_qkv_head_group_fused_size\": "
              << kAttentionQkvHeadGroupFusedSize << ",\n";
    std::cout << "  \"attention_qkv_head_group_fused_contexts_per_block\": "
              << kAttentionQkvHeadGroupFusedContextsPerBlock << ",\n";
    std::cout << "  \"attention_qkv_head_group_fused_softmax_scale\": "
              << kAttentionQkvHeadGroupFusedSoftmaxScale << ",\n";
    std::cout << "  \"attention_qkv_head_group_fused_residual_scale\": "
              << kAttentionQkvHeadGroupFusedResidualScale << ",\n";
    std::cout << "  \"ssm_recurrent_state_workspace_bytes\": "
              << ssm_recurrent_state_workspace_bytes << ",\n";
    std::cout << "  \"ssm_recurrent_state_values\": "
              << ssm_recurrent_state_values << ",\n";
    std::cout << "  \"ssm_scan_scratch_workspace_bytes\": "
              << ssm_scan_scratch_workspace_bytes << ",\n";
    std::cout << "  \"ssm_scan_scratch_values\": "
              << ssm_scan_scratch_values << ",\n";
    std::cout << "  \"ssm_scan_scratch_scale\": "
              << kSsmScanScratchScale << ",\n";
    std::cout << "  \"ssm_selective_gate_logit_scale\": "
              << kSsmSelectiveGateLogitScale << ",\n";
    std::cout << "  \"ssm_selective_gate_activation_scale\": "
              << kSsmSelectiveGateActivationScale << ",\n";
    std::cout << "  \"ssm_selective_gate_scratch_scale\": "
              << kSsmSelectiveGateScratchScale << ",\n";
    std::cout << "  \"ssm_source_param_cache_logit_scale\": "
              << kSsmSourceParamCacheLogitScale << ",\n";
    std::cout << "  \"ssm_source_param_cache_activation_scale\": "
              << kSsmSourceParamCacheActivationScale << ",\n";
    std::cout << "  \"ssm_source_param_alpha_scale\": "
              << kSsmSourceParamAlphaScale << ",\n";
    std::cout << "  \"ssm_source_param_beta_scale\": "
              << kSsmSourceParamBetaScale << ",\n";
    std::cout << "  \"ssm_source_param_gate_logit_scale\": "
              << kSsmSourceParamGateLogitScale << ",\n";
    std::cout << "  \"ssm_source_param_gate_activation_scale\": "
              << kSsmSourceParamGateActivationScale << ",\n";
    std::cout << "  \"layer_executor_workspace_activation_bytes\": "
              << layer_workspace_allocation_plan.activation_bytes << ",\n";
    std::cout << "  \"layer_executor_workspace_logits_bytes\": "
              << layer_workspace_allocation_plan.logits_bytes << ",\n";
    std::cout << "  \"layer_executor_workspace_usage_plan\": "
              << "\"phase_local_logical_high_water\",\n";
    std::cout << "  \"layer_executor_workspace_high_water_bytes\": "
              << layer_workspace_usage.total_high_water_bytes << ",\n";
    std::cout << "  \"layer_executor_workspace_activation_high_water_bytes\": "
              << layer_workspace_usage.activation_high_water_bytes << ",\n";
    std::cout << "  \"layer_executor_workspace_logits_high_water_bytes\": "
              << layer_workspace_usage.logits_high_water_bytes << ",\n";
    std::cout << "  \"layer_executor_workspace_auxiliary_high_water_bytes\": "
              << layer_workspace_usage.auxiliary_high_water_bytes << ",\n";
    std::cout << "  \"layer_executor_workspace_slack_bytes\": "
              << layer_workspace_usage.total_slack_bytes << ",\n";
    std::cout << "  \"layer_executor_workspace_activation_slack_bytes\": "
              << layer_workspace_usage.activation_slack_bytes << ",\n";
    std::cout << "  \"layer_executor_workspace_logits_slack_bytes\": "
              << layer_workspace_usage.logits_slack_bytes << ",\n";
    std::cout << "  \"layer_executor_workspace_auxiliary_slack_bytes\": "
              << layer_workspace_usage.auxiliary_slack_bytes << ",\n";
    std::cout << "  \"layer_executor_warmup_phase_callbacks\": "
              << warmup_replay.phase_callbacks << ",\n";
    std::cout << "  \"layer_executor_warmup_attention_phase_callbacks\": "
              << warmup_replay.attention_phase_callbacks << ",\n";
    std::cout << "  \"layer_executor_warmup_ffn_phase_callbacks\": "
              << warmup_replay.ffn_phase_callbacks << ",\n";
    std::cout << "  \"layer_executor_warmup_ssm_phase_callbacks\": "
              << warmup_replay.ssm_phase_callbacks << ",\n";
    std::cout << "  \"layer_executor_warmup_output_phase_callbacks\": "
              << warmup_replay.output_phase_callbacks << ",\n";
    std::cout << "  \"layer_executor_warmup_stage_callbacks\": "
              << warmup_replay.stage_callbacks << ",\n";
    std::cout << "  \"layer_executor_warmup_feedback_callbacks\": "
              << warmup_replay.activation_feedback_callbacks << ",\n";
    std::cout << "  \"layer_executor_warmup_total_callbacks\": "
              << warmup_replay.total_callbacks << ",\n";
    std::cout << "  \"layer_executor_capture_phase_callbacks\": "
              << graph_capture_replay.phase_callbacks << ",\n";
    std::cout << "  \"layer_executor_capture_attention_phase_callbacks\": "
              << graph_capture_replay.attention_phase_callbacks << ",\n";
    std::cout << "  \"layer_executor_capture_ffn_phase_callbacks\": "
              << graph_capture_replay.ffn_phase_callbacks << ",\n";
    std::cout << "  \"layer_executor_capture_ssm_phase_callbacks\": "
              << graph_capture_replay.ssm_phase_callbacks << ",\n";
    std::cout << "  \"layer_executor_capture_output_phase_callbacks\": "
              << graph_capture_replay.output_phase_callbacks << ",\n";
    std::cout << "  \"layer_executor_capture_stage_callbacks\": "
              << graph_capture_replay.stage_callbacks << ",\n";
    std::cout << "  \"layer_executor_capture_feedback_callbacks\": "
              << graph_capture_replay.activation_feedback_callbacks << ",\n";
    std::cout << "  \"layer_executor_capture_total_callbacks\": "
              << graph_capture_replay.total_callbacks << ",\n";
    std::cout << "  \"graph_replay_rate\": 1.0,\n";
    std::cout << "  \"graph_bucket_plan\": \"decode_bucket_admission_estimate\",\n";
    std::cout << "  \"graph_bucket_count\": "
              << graph_bucket_plan.buckets.size() << ",\n";
    std::cout << "  \"graph_bucket_batch_bucket\": "
              << options.graph_batch_bucket << ",\n";
    std::cout << "  \"graph_bucket_query_bucket\": "
              << options.graph_query_bucket << ",\n";
    std::cout << "  \"graph_bucket_estimated_bytes\": "
              << graph_bucket_plan.total_bytes << ",\n";
    std::cout << "  \"graph_bucket_max_estimated_bytes\": "
              << graph_bucket_plan.max_bucket_bytes << ",\n";
    std::cout << "  \"graph_buckets\": [";
    for (std::size_t index = 0; index < graph_bucket_plan.buckets.size(); ++index) {
      const auto& bucket = graph_bucket_plan.buckets[index];
      if (index != 0) {
        std::cout << ", ";
      }
      std::cout << "{\"name\": \""
                << json_escape(bucket.descriptor.name)
                << "\", \"batch_bucket\": " << bucket.descriptor.batch_bucket
                << ", \"query_bucket\": " << bucket.descriptor.query_bucket
                << ", \"phase_count\": " << bucket.descriptor.phase_count
                << ", \"kernel_launches_per_replay\": "
                << bucket.descriptor.kernel_launches_per_replay
                << ", \"workspace_high_water_bytes\": "
                << bucket.descriptor.workspace_high_water_bytes
                << ", \"offset\": " << bucket.offset
                << ", \"metadata_bytes\": " << bucket.metadata_bytes
                << ", \"workspace_guard_bytes\": "
                << bucket.workspace_guard_bytes
                << ", \"total_bytes\": " << bucket.total_bytes
                << "}";
    }
    std::cout << "],\n";
    std::cout << "  \"reference_checked\": " << (reference.checked ? "true" : "false") << ",\n";
    if (reference.checked) {
      std::cout << "  \"reference_passed\": " << (reference.passed ? "true" : "false") << ",\n";
      std::cout << "  \"reference_tolerance\": " << reference.tolerance << ",\n";
      std::cout << "  \"reference_final_stage\": \""
                << json_escape(reference.final_stage) << "\",\n";
      std::cout << "  \"reference_final_rows\": " << reference.final_rows << ",\n";
      std::cout << "  \"max_logit_abs_error\": " << reference.max_logit_abs_error << ",\n";
      std::cout << "  \"max_logit_error_row\": " << reference.max_logit_error_row << ",\n";
      std::cout << "  \"max_kv_abs_error\": " << reference.max_kv_abs_error << ",\n";
      std::cout << "  \"max_kv_error_index\": " << reference.max_kv_error_index << ",\n";
      std::cout << "  \"logit_mismatches\": " << reference.logit_mismatches << ",\n";
      std::cout << "  \"kv_mismatches\": " << reference.kv_mismatches << ",\n";
      std::cout << "  \"reference_stage_count\": " << reference.stages.size() << ",\n";
      std::cout << "  \"reference_stage_mismatches\": "
                << reference.stage_mismatches << ",\n";
      std::cout << "  \"token_expected\": " << reference.token_expected << ",\n";
      std::cout << "  \"token_observed\": " << reference.token_observed << ",\n";
      std::cout << "  \"first_logit_expected\": " << reference.first_logit_expected << ",\n";
      std::cout << "  \"first_logit_observed\": " << reference.first_logit_observed << ",\n";
      std::cout << "  \"reference_stage_results\": [";
      for (std::size_t index = 0; index < reference.stages.size(); ++index) {
        const auto& stage_result = reference.stages[index];
        if (index != 0) {
          std::cout << ", ";
        }
        std::cout << "{"
                  << "\"index\": " << index
                  << ", \"name\": \"" << json_escape(stage_result.name) << "\""
                  << ", \"type\": \"" << json_escape(stage_result.type) << "\""
                  << ", \"role\": \"" << json_escape(stage_result.role) << "\""
                  << ", \"rows\": " << stage_result.rows
                  << ", \"passed\": " << (stage_result.passed ? "true" : "false")
                  << ", \"token_expected\": " << stage_result.token_expected
                  << ", \"token_observed\": " << stage_result.token_observed
                  << ", \"max_logit_abs_error\": " << stage_result.max_logit_abs_error
                  << ", \"max_logit_error_row\": " << stage_result.max_logit_error_row
                  << ", \"logit_mismatches\": " << stage_result.logit_mismatches
                  << ", \"first_logit_expected\": "
                  << stage_result.first_logit_expected
                  << ", \"first_logit_observed\": "
                  << stage_result.first_logit_observed
                  << "}";
      }
      std::cout << "],\n";
    }
    std::cout << "  \"wall_ms\": " << wall_ms << ",\n";
    std::cout << "  \"steps_per_second\": " << (options.steps * 1000.0 / wall_ms) << ",\n";
    std::cout << "  \"tensor_launches_per_second\": "
              << (layer_execution_plan.matvec_kernels_per_replay * options.steps * 1000.0 /
                  wall_ms) << ",\n";
    std::cout << "  \"graph_kernel_launches_per_second\": "
              << (layer_execution_plan.total_kernels_per_replay * options.steps * 1000.0 /
                  wall_ms) << ",\n";
    std::cout << "  \"actual_graph_kernel_launches_per_second\": "
              << (actual_kernel_launches_per_graph * options.steps * 1000.0 /
                  wall_ms) << ",\n";
    std::cout << "  \"p50_ms\": " << percentile(latencies_ms, 50) << ",\n";
    std::cout << "  \"p95_ms\": " << percentile(latencies_ms, 95) << ",\n";
    std::cout << "  \"p99_ms\": " << percentile(latencies_ms, 99) << ",\n";
    std::cout << "  \"latency_mean_ms\": " << mean_ms << ",\n";
    std::cout << "  \"phase_timing_attention_p50_ms\": "
              << phase_timing_p50_ms[0] << ",\n";
    std::cout << "  \"phase_timing_attention_p95_ms\": "
              << phase_timing_p95_ms[0] << ",\n";
    std::cout << "  \"phase_timing_attention_p99_ms\": "
              << phase_timing_p99_ms[0] << ",\n";
    std::cout << "  \"phase_timing_attention_samples\": "
              << phase_timing_samples[0] << ",\n";
    std::cout << "  \"phase_timing_ffn_p50_ms\": "
              << phase_timing_p50_ms[1] << ",\n";
    std::cout << "  \"phase_timing_ffn_p95_ms\": "
              << phase_timing_p95_ms[1] << ",\n";
    std::cout << "  \"phase_timing_ffn_p99_ms\": "
              << phase_timing_p99_ms[1] << ",\n";
    std::cout << "  \"phase_timing_ffn_samples\": "
              << phase_timing_samples[1] << ",\n";
    std::cout << "  \"phase_timing_ssm_p50_ms\": "
              << phase_timing_p50_ms[2] << ",\n";
    std::cout << "  \"phase_timing_ssm_p95_ms\": "
              << phase_timing_p95_ms[2] << ",\n";
    std::cout << "  \"phase_timing_ssm_p99_ms\": "
              << phase_timing_p99_ms[2] << ",\n";
    std::cout << "  \"phase_timing_ssm_samples\": "
              << phase_timing_samples[2] << ",\n";
    std::cout << "  \"phase_timing_output_p50_ms\": "
              << phase_timing_p50_ms[3] << ",\n";
    std::cout << "  \"phase_timing_output_p95_ms\": "
              << phase_timing_p95_ms[3] << ",\n";
    std::cout << "  \"phase_timing_output_p99_ms\": "
              << phase_timing_p99_ms[3] << ",\n";
    std::cout << "  \"phase_timing_output_samples\": "
              << phase_timing_samples[3] << ",\n";
    std::cout << "  \"phase_timing_sum_mean_ms\": "
              << phase_timing_sum_mean_ms << ",\n";
    std::cout << "  \"phase_timing_sum_p50_ms\": "
              << phase_timing_sum_p50_ms << ",\n";
    std::cout << "  \"phase_timing_sum_p95_ms\": "
              << phase_timing_sum_p95_ms << ",\n";
    std::cout << "  \"phase_timing_sum_p99_ms\": "
              << phase_timing_sum_p99_ms << ",\n";
    std::cout << "  \"graph_minus_phase_timing_sum_p50_ms\": "
              << (percentile(latencies_ms, 50) - phase_timing_sum_p50_ms)
              << ",\n";
    std::cout << "  \"graph_minus_phase_timing_sum_p95_ms\": "
              << (percentile(latencies_ms, 95) - phase_timing_sum_p95_ms)
              << ",\n";
    std::cout << "  \"graph_minus_phase_timing_sum_p99_ms\": "
              << (percentile(latencies_ms, 99) - phase_timing_sum_p99_ms)
              << ",\n";
    std::cout << "  \"phase_timing_slowest_p95_phase\": \""
              << phase_timing_names[slowest_phase_p95_slot] << "\",\n";
    std::cout << "  \"phase_timing_slowest_p95_ms\": "
              << phase_timing_p95_ms[slowest_phase_p95_slot] << ",\n";
    std::cout << "  \"capture_ms\": " << capture_ms << ",\n";
    std::cout << "  \"graph_upload_ms\": " << upload_ms << ",\n";
    std::cout << "  \"graph_capture_free_before_bytes\": "
              << graph_memory_before_capture.free_bytes << ",\n";
    std::cout << "  \"graph_capture_free_after_bytes\": "
              << graph_memory_after_capture.free_bytes << ",\n";
    std::cout << "  \"graph_capture_device_pressure_bytes\": "
              << free_memory_delta(
                     graph_memory_before_capture,
                     graph_memory_after_capture)
              << ",\n";
    std::cout << "  \"graph_instantiate_device_pressure_bytes\": "
              << free_memory_delta(
                     graph_memory_after_capture,
                     graph_memory_after_instantiate)
              << ",\n";
    std::cout << "  \"graph_upload_device_pressure_bytes\": "
              << free_memory_delta(
                     graph_memory_after_instantiate,
                     graph_memory_after_upload)
              << ",\n";
    std::cout << "  \"graph_live_device_pressure_bytes\": "
              << free_memory_delta(
                     graph_memory_before_capture,
                     graph_memory_after_upload)
              << ",\n";
    std::cout << "  \"graph_destroy_free_before_bytes\": "
              << graph_memory_before_destroy.free_bytes << ",\n";
    std::cout << "  \"graph_destroy_free_after_bytes\": "
              << graph_memory_after_destroy.free_bytes << ",\n";
    std::cout << "  \"graph_destroy_recovered_bytes\": "
              << -free_memory_delta(
                     graph_memory_before_destroy,
                     graph_memory_after_destroy)
              << ",\n";
    std::cout << "  \"q4k_payload_bytes\": " << q4k_payload_bytes << ",\n";
    std::cout << "  \"q4k_metadata_bytes\": " << q4k_metadata_bytes << ",\n";
    std::cout << "  \"iq_payload_bytes\": " << iq_payload_bytes << ",\n";
    std::cout << "  \"q5k_payload_bytes\": " << q5k_payload_bytes << ",\n";
    std::cout << "  \"resident_bytes\": " << resident_bytes << ",\n";
    std::cout << "  \"allocated_bytes\": " << pool.total_bytes() << ",\n";
    std::cout << "  \"setup_h2d_bytes\": " << setup_h2d_bytes << ",\n";
    std::cout << "  \"h2d_bytes\": 0,\n";
    std::cout << "  \"d2h_bytes\": " << (sizeof(std::uint32_t) * options.steps) << ",\n";
    std::cout << "  \"kv_mib\": " << options.kv_mib << ",\n";
    std::cout << "  \"kv_words\": " << kv_words << ",\n";
    std::cout << "  \"page_words\": " << options.page_words << ",\n";
    std::cout << "  \"active_pages\": " << options.active_pages << ",\n";
    std::cout << "  \"max_rows\": " << max_rows << ",\n";
    std::cout << "  \"max_cols\": " << max_cols << ",\n";
    std::cout << "  \"logical_values_per_graph\": " << logical_values_per_graph << ",\n";
    std::cout << "  \"layer_count_observed\": " << observed_layers.size() << ",\n";
    std::cout << "  \"host_weight_checksum\": " << host_weight_checksum << ",\n";
    std::cout << "  \"weight_source\": \""
              << (include_q5k ? "mixed_q4k_iq_q5k_manifest_payload_file"
                              : "mixed_q4k_iq_manifest_payload_file")
              << "\",\n";
    std::cout << "  \"last_token\": " << last_token << ",\n";
    std::cout << "  \"type_counts\": {";
    std::size_t type_index = 0;
    for (const auto& [type, count] : type_counts) {
      if (type_index++ != 0) {
        std::cout << ", ";
      }
      std::cout << "\"" << json_escape(type) << "\": " << count;
    }
    std::cout << "},\n";
    std::cout << "  \"stage_phase_counts\": {";
    std::size_t phase_index = 0;
    for (const auto& [phase, count] : stage_phase_counts) {
      if (phase_index++ != 0) {
        std::cout << ", ";
      }
      std::cout << "\"" << json_escape(phase) << "\": " << count;
    }
    std::cout << "},\n";
    std::cout << "  \"role_counts\": {";
    std::size_t role_index = 0;
    for (const auto& [role, count] : role_counts) {
      if (role_index++ != 0) {
        std::cout << ", ";
      }
      std::cout << "\"" << json_escape(role) << "\": " << count;
    }
    std::cout << "},\n";
    std::cout << "  \"tensors\": [";
    for (std::size_t index = 0; index < stages.size(); ++index) {
      if (index != 0) {
        std::cout << ", ";
      }
      std::cout << "\"" << json_escape(stages[index].name) << "\"";
    }
    std::cout << "],\n";
    std::cout << "  \"stage_roles\": [";
    for (std::size_t index = 0; index < stages.size(); ++index) {
      if (index != 0) {
        std::cout << ", ";
      }
      std::cout << "\"" << json_escape(stages[index].role) << "\"";
    }
    std::cout << "],\n";
    std::cout << "  \"stage_phases\": [";
    for (std::size_t index = 0; index < stages.size(); ++index) {
      if (index != 0) {
        std::cout << ", ";
      }
      std::cout << "\""
                << rtxllm::layer_phase_name(
                       rtxllm::layer_phase_from_role(stages[index].role)) << "\"";
    }
    std::cout << "],\n";
    std::cout << "  \"stage_phase_segments\": [";
    for (std::size_t index = 0; index < stage_phase_segments.size(); ++index) {
      const auto& segment = stage_phase_segments[index];
      if (index != 0) {
        std::cout << ", ";
      }
      std::cout << "{\"phase\": \""
                << rtxllm::layer_phase_name(segment.phase)
                << "\", \"begin\": " << segment.begin
                << ", \"count\": " << segment.count << "}";
    }
    std::cout << "],\n";
    std::cout << "  \"layer_executor_phase_workspaces\": [";
    for (std::size_t index = 0; index < layer_workspace_plan.phases.size(); ++index) {
      const auto& phase = layer_workspace_plan.phases[index];
      const auto& allocation = layer_workspace_allocation_plan.phases[index];
      const auto& usage = layer_workspace_usage.phases[index];
      if (index != 0) {
        std::cout << ", ";
      }
      std::cout << "{\"phase\": \""
                << rtxllm::layer_phase_name(phase.phase)
                << "\", \"begin\": " << phase.begin
                << ", \"count\": " << phase.count
                << ", \"max_rows\": " << phase.max_rows
                << ", \"max_cols\": " << phase.max_cols
                << ", \"logical_values_per_graph\": "
                << phase.logical_values_per_replay
                << ", \"activation_offset\": " << allocation.activation_offset
                << ", \"activation_bytes\": " << allocation.activation_bytes
                << ", \"logits_offset\": " << allocation.logits_offset
                << ", \"logits_bytes\": " << allocation.logits_bytes
                << ", \"total_bytes\": " << allocation.total_bytes
                << ", \"activation_high_water_bytes\": "
                << usage.activation_high_water_bytes
                << ", \"logits_high_water_bytes\": "
                << usage.logits_high_water_bytes
                << ", \"total_high_water_bytes\": "
                << usage.total_high_water_bytes
                << ", \"activation_slack_bytes\": "
                << usage.activation_slack_bytes
                << ", \"logits_slack_bytes\": "
                << usage.logits_slack_bytes
                << ", \"total_slack_bytes\": "
                << usage.total_slack_bytes << "}";
    }
    std::cout << "],\n";
    std::cout << "  \"layer_executor_aux_workspaces\": [";
    for (std::size_t index = 0;
         index < layer_workspace_allocation_plan.auxiliaries.size();
         ++index) {
      const auto& auxiliary = layer_workspace_allocation_plan.auxiliaries[index];
      const auto& usage = layer_workspace_usage.auxiliaries[index];
      if (index != 0) {
        std::cout << ", ";
      }
      std::cout << "{\"name\": \"" << json_escape(auxiliary.name)
                << "\", \"phase\": \""
                << rtxllm::layer_phase_name(auxiliary.phase)
                << "\", \"values\": " << auxiliary.values
                << ", \"offset\": " << auxiliary.offset
                << ", \"bytes\": " << auxiliary.bytes
                << ", \"high_water_bytes\": " << usage.high_water_bytes
                << ", \"slack_bytes\": " << usage.slack_bytes << "}";
    }
    std::cout << "],\n";
    std::cout << "  \"layer_executor_activation_feedback_plan\": [";
    for (std::size_t index = 0;
         index < layer_dataflow_plan.activation_feedback_edges.size();
         ++index) {
      const auto& edge = layer_dataflow_plan.activation_feedback_edges[index];
      if (index != 0) {
        std::cout << ", ";
      }
      std::cout << "{\"source_stage\": " << edge.source_stage
                << ", \"target_stage\": " << edge.target_stage
                << ", \"source_phase\": \""
                << rtxllm::layer_phase_name(edge.source_phase)
                << "\", \"target_phase\": \""
                << rtxllm::layer_phase_name(edge.target_phase)
                << "\", \"source_phase_index\": " << edge.source_phase_index
                << ", \"target_phase_index\": " << edge.target_phase_index
                << ", \"source_rows\": " << edge.source_rows
                << ", \"target_cols\": " << edge.target_cols
                << ", \"stage_id\": " << edge.stage_id
                << ", \"crosses_phase\": "
                << (edge.crosses_phase ? "true" : "false")
                << ", \"wraps_to_first_stage\": "
                << (edge.wraps_to_first_stage ? "true" : "false")
                << "}";
    }
    std::cout << "],\n";
    std::cout << "  \"layer_execution_phases\": [";
    for (std::size_t index = 0; index < layer_execution_plan.phases.size(); ++index) {
      const auto& phase = layer_execution_plan.phases[index];
      if (index != 0) {
        std::cout << ", ";
      }
      std::cout << "{\"phase\": \""
                << rtxllm::layer_phase_name(phase.phase)
                << "\", \"begin\": " << phase.begin
                << ", \"count\": " << phase.count
                << ", \"matvec_kernels_per_graph\": "
                << phase.matvec_kernels_per_replay
                << ", \"activation_feedbacks_per_graph\": "
                << phase.activation_feedbacks_per_replay
                << ", \"kernel_launches_per_graph\": "
                << phase.total_kernels_per_replay
                << ", \"attention_phase_kernel_launches_per_graph\": "
                << attention_phase_kernel_count(phase.phase)
                << ", \"ssm_phase_kernel_launches_per_graph\": "
                << ssm_phase_kernel_count(phase.phase)
                << ", \"output_phase_kernel_launches_per_graph\": "
                << output_phase_kernel_count(phase.phase)
                << ", \"phase_scratch_kernel_launches_per_graph\": "
                << phase_scratch_kernel_count(phase.phase)
                << ", \"actual_kernel_launches_per_graph\": "
                << (phase.total_kernels_per_replay +
                    attention_phase_kernel_count(phase.phase) +
                    ssm_phase_kernel_count(phase.phase) +
                    output_phase_kernel_count(phase.phase) +
                    phase_scratch_kernel_count(phase.phase))
                << ", \"timing_samples\": " << phase_latencies_ms[index].size()
                << ", \"timing_mean_ms\": " << mean_value(phase_latencies_ms[index])
                << ", \"timing_p50_ms\": " << percentile(phase_latencies_ms[index], 50)
                << ", \"timing_p95_ms\": " << percentile(phase_latencies_ms[index], 95)
                << ", \"timing_p99_ms\": " << percentile(phase_latencies_ms[index], 99)
                << "}";
    }
    std::cout << "],\n";
    std::cout << "  \"layer_plan_expected_phases\": [";
    for (std::size_t index = 0; index < layer_plan.expected_phases.size(); ++index) {
      if (index != 0) {
        std::cout << ", ";
      }
      std::cout << "\"" << rtxllm::layer_phase_name(layer_plan.expected_phases[index]) << "\"";
    }
    std::cout << "],\n";
    std::cout << "  \"layer_plan_observed_phases\": [";
    for (std::size_t index = 0; index < layer_plan.observed_phases.size(); ++index) {
      if (index != 0) {
        std::cout << ", ";
      }
      std::cout << "\"" << rtxllm::layer_phase_name(layer_plan.observed_phases[index]) << "\"";
    }
    std::cout << "],\n";
    std::cout << "  \"stage_source_offsets\": [";
    for (std::size_t index = 0; index < stages.size(); ++index) {
      if (index != 0) {
        std::cout << ", ";
      }
      std::cout << stages[index].source_offset;
    }
    std::cout << "]\n";
    std::cout << "}\n";
    return reference.checked && !reference.passed ? 3 : 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }
}
