#include "runtime/core/scheduler/layer_plan.h"

namespace rtxllm {

namespace {

constexpr int kUnknownRoleRank = 5000;

}  // namespace

const char* layer_phase_name(LayerPhase phase) {
  switch (phase) {
    case LayerPhase::Attention:
      return "attention";
    case LayerPhase::Ffn:
      return "ffn";
    case LayerPhase::Ssm:
      return "ssm";
    case LayerPhase::Output:
      return "output";
    case LayerPhase::Unknown:
      return "unknown";
  }
  return "unknown";
}

int layer_role_rank(std::string_view role) {
  if (role == "attn_gate.weight") {
    return 10;
  }
  if (role == "attn_qkv.weight") {
    return 20;
  }
  if (role == "ffn_gate_exps.weight") {
    return 30;
  }
  if (role == "ffn_gate_shexp.weight") {
    return 40;
  }
  if (role == "ffn_up_exps.weight") {
    return 50;
  }
  if (role == "ffn_up_shexp.weight") {
    return 60;
  }
  if (role == "ffn_down_exps.weight") {
    return 70;
  }
  if (role == "ffn_down_shexp.weight") {
    return 80;
  }
  if (role == "ssm_alpha.weight") {
    return 90;
  }
  if (role == "ssm_beta.weight") {
    return 100;
  }
  if (role == "ssm_out.weight") {
    return 110;
  }
  if (role == "output" || role == "output.weight") {
    return 10000;
  }
  return kUnknownRoleRank;
}

LayerPhase layer_phase_from_role(std::string_view role) {
  if (role == "output" || role == "output.weight") {
    return LayerPhase::Output;
  }
  if (role.rfind("attn_", 0) == 0) {
    return LayerPhase::Attention;
  }
  if (role.rfind("ffn_", 0) == 0) {
    return LayerPhase::Ffn;
  }
  if (role.rfind("ssm_", 0) == 0) {
    return LayerPhase::Ssm;
  }
  return LayerPhase::Unknown;
}

std::size_t count_unknown_layer_roles(std::span<const LayerStageDescriptor> stages) {
  std::size_t unknowns = 0;
  for (const auto& stage : stages) {
    if (stage.layer_local && layer_role_rank(stage.role) == kUnknownRoleRank) {
      ++unknowns;
    }
  }
  return unknowns;
}

std::size_t count_unknown_layer_phases(std::span<const LayerStageDescriptor> stages) {
  std::size_t unknowns = 0;
  for (const auto& stage : stages) {
    if (layer_phase_from_role(stage.role) == LayerPhase::Unknown) {
      ++unknowns;
    }
  }
  return unknowns;
}

std::map<std::string, std::size_t> build_layer_phase_counts(
    std::span<const LayerStageDescriptor> stages) {
  std::map<std::string, std::size_t> counts;
  for (const auto& stage : stages) {
    counts[layer_phase_name(layer_phase_from_role(stage.role))] += 1;
  }
  return counts;
}

std::vector<LayerPhaseSegment> build_layer_phase_segments(
    std::span<const LayerStageDescriptor> stages) {
  std::vector<LayerPhaseSegment> segments;
  for (std::size_t index = 0; index < stages.size(); ++index) {
    const LayerPhase phase = layer_phase_from_role(stages[index].role);
    if (segments.empty() || segments.back().phase != phase) {
      segments.push_back(LayerPhaseSegment{phase, index, 1});
    } else {
      segments.back().count += 1;
    }
  }
  return segments;
}

LayerPlanValidation validate_layer_phase_plan(
    std::span<const LayerStageDescriptor> stages,
    std::span<const LayerPhaseSegment> segments) {
  LayerPlanValidation result;
  result.expected_phases = {
      LayerPhase::Attention,
      LayerPhase::Ffn,
      LayerPhase::Ssm,
  };
  if (!segments.empty() && segments.back().phase == LayerPhase::Output) {
    result.expected_phases.push_back(LayerPhase::Output);
  }
  for (const auto& segment : segments) {
    result.observed_phases.push_back(segment.phase);
  }

  if (stages.empty()) {
    result.message = "layer phase plan requires at least one stage";
    return result;
  }
  if (count_unknown_layer_roles(stages) != 0) {
    result.message = "layer phase plan contains unknown role-plan roles";
    return result;
  }
  if (count_unknown_layer_phases(stages) != 0) {
    result.message = "layer phase plan contains unknown execution phases";
    return result;
  }
  if (segments.size() != result.expected_phases.size()) {
    result.message = "layer phase plan segment count mismatch";
    return result;
  }
  for (std::size_t index = 0; index < segments.size(); ++index) {
    if (segments[index].phase != result.expected_phases[index]) {
      result.message = "layer phase plan phase order mismatch";
      return result;
    }
    if (segments[index].count == 0) {
      result.message = "layer phase plan contains an empty phase segment";
      return result;
    }
  }
  result.passed = true;
  result.message = "layer phase plan validated";
  return result;
}

LayerExecutionPlan build_layer_execution_plan(
    std::span<const LayerPhaseSegment> segments,
    bool chain_activation) {
  LayerExecutionPlan plan;
  plan.phase_count = segments.size();
  plan.phases.reserve(segments.size());

  for (const auto& segment : segments) {
    LayerPhaseExecution phase;
    phase.phase = segment.phase;
    phase.begin = segment.begin;
    phase.count = segment.count;
    phase.matvec_kernels_per_replay = segment.count;
    phase.activation_feedbacks_per_replay = chain_activation ? segment.count : 0;
    phase.total_kernels_per_replay =
        phase.matvec_kernels_per_replay + phase.activation_feedbacks_per_replay;

    plan.stage_count += segment.count;
    plan.matvec_kernels_per_replay += phase.matvec_kernels_per_replay;
    plan.activation_feedbacks_per_replay += phase.activation_feedbacks_per_replay;
    plan.total_kernels_per_replay += phase.total_kernels_per_replay;
    plan.phases.push_back(phase);
  }

  return plan;
}

}  // namespace rtxllm
