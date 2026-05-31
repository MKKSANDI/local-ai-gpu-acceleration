#pragma once

#include <cstddef>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rtxllm {

enum class LayerPhase {
  Attention,
  Ffn,
  Ssm,
  Output,
  Unknown,
};

struct LayerStageDescriptor {
  std::string_view role;
  bool layer_local = true;
};

struct LayerPhaseSegment {
  LayerPhase phase = LayerPhase::Unknown;
  std::size_t begin = 0;
  std::size_t count = 0;
};

struct LayerPlanValidation {
  bool passed = false;
  std::string message;
  std::vector<LayerPhase> expected_phases;
  std::vector<LayerPhase> observed_phases;
};

struct LayerPhaseExecution {
  LayerPhase phase = LayerPhase::Unknown;
  std::size_t begin = 0;
  std::size_t count = 0;
  std::size_t matvec_kernels_per_replay = 0;
  std::size_t activation_feedbacks_per_replay = 0;
  std::size_t total_kernels_per_replay = 0;
};

struct LayerExecutionPlan {
  std::vector<LayerPhaseExecution> phases;
  std::size_t stage_count = 0;
  std::size_t phase_count = 0;
  std::size_t matvec_kernels_per_replay = 0;
  std::size_t activation_feedbacks_per_replay = 0;
  std::size_t total_kernels_per_replay = 0;
};

const char* layer_phase_name(LayerPhase phase);

int layer_role_rank(std::string_view role);

LayerPhase layer_phase_from_role(std::string_view role);

std::size_t count_unknown_layer_roles(std::span<const LayerStageDescriptor> stages);

std::size_t count_unknown_layer_phases(std::span<const LayerStageDescriptor> stages);

std::map<std::string, std::size_t> build_layer_phase_counts(
    std::span<const LayerStageDescriptor> stages);

std::vector<LayerPhaseSegment> build_layer_phase_segments(
    std::span<const LayerStageDescriptor> stages);

LayerPlanValidation validate_layer_phase_plan(
    std::span<const LayerStageDescriptor> stages,
    std::span<const LayerPhaseSegment> segments);

LayerExecutionPlan build_layer_execution_plan(
    std::span<const LayerPhaseSegment> segments,
    bool chain_activation);

}  // namespace rtxllm
