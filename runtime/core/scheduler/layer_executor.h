#pragma once

#include "runtime/core/scheduler/layer_plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rtxllm {

struct LayerStageWorkspaceDescriptor {
  std::uint32_t rows = 0;
  std::uint32_t cols = 0;
};

struct LayerPhaseWorkspaceDescriptor {
  LayerPhase phase = LayerPhase::Unknown;
  std::size_t begin = 0;
  std::size_t count = 0;
  std::uint32_t max_rows = 0;
  std::uint32_t max_cols = 0;
  std::uint64_t logical_values_per_replay = 0;
};

struct LayerAuxWorkspaceDescriptor {
  std::string name;
  LayerPhase phase = LayerPhase::Unknown;
  std::uint32_t values = 0;
};

struct LayerExecutorWorkspacePlan {
  std::vector<LayerPhaseWorkspaceDescriptor> phases;
  std::uint32_t max_rows = 0;
  std::uint32_t max_cols = 0;
  std::uint64_t logical_values_per_replay = 0;
};

struct LayerPhaseWorkspaceAllocation {
  LayerPhase phase = LayerPhase::Unknown;
  std::size_t begin = 0;
  std::size_t count = 0;
  std::size_t activation_offset = 0;
  std::size_t activation_bytes = 0;
  std::size_t logits_offset = 0;
  std::size_t logits_bytes = 0;
  std::size_t total_bytes = 0;
};

struct LayerAuxWorkspaceAllocation {
  std::string name;
  LayerPhase phase = LayerPhase::Unknown;
  std::uint32_t values = 0;
  std::size_t offset = 0;
  std::size_t bytes = 0;
};

struct LayerExecutorWorkspaceAllocationPlan {
  std::vector<LayerPhaseWorkspaceAllocation> phases;
  std::vector<LayerAuxWorkspaceAllocation> auxiliaries;
  std::size_t activation_bytes = 0;
  std::size_t logits_bytes = 0;
  std::size_t phase_bytes = 0;
  std::size_t auxiliary_bytes = 0;
  std::size_t total_bytes = 0;
};

struct LayerPhaseWorkspaceRuntime {
  LayerPhase phase = LayerPhase::Unknown;
  std::size_t begin = 0;
  std::size_t count = 0;
  float* activation = nullptr;
  float* logits = nullptr;
  std::uint32_t max_rows = 0;
  std::uint32_t max_cols = 0;
  std::size_t activation_offset = 0;
  std::size_t activation_bytes = 0;
  std::size_t logits_offset = 0;
  std::size_t logits_bytes = 0;
  std::size_t total_bytes = 0;
};

struct LayerAuxWorkspaceRuntime {
  std::string name;
  LayerPhase phase = LayerPhase::Unknown;
  float* data = nullptr;
  std::uint32_t values = 0;
  std::size_t offset = 0;
  std::size_t bytes = 0;
};

struct LayerExecutorWorkspaceRuntime {
  std::vector<LayerPhaseWorkspaceRuntime> phases;
  std::vector<LayerAuxWorkspaceRuntime> auxiliaries;
  std::size_t activation_bytes = 0;
  std::size_t logits_bytes = 0;
  std::size_t phase_bytes = 0;
  std::size_t auxiliary_bytes = 0;
  std::size_t total_bytes = 0;
};

struct LayerPhaseWorkspaceUsage {
  LayerPhase phase = LayerPhase::Unknown;
  std::size_t begin = 0;
  std::size_t count = 0;
  std::size_t activation_high_water_bytes = 0;
  std::size_t logits_high_water_bytes = 0;
  std::size_t total_high_water_bytes = 0;
  std::size_t activation_slack_bytes = 0;
  std::size_t logits_slack_bytes = 0;
  std::size_t total_slack_bytes = 0;
};

struct LayerAuxWorkspaceUsage {
  std::string name;
  LayerPhase phase = LayerPhase::Unknown;
  std::size_t high_water_bytes = 0;
  std::size_t slack_bytes = 0;
};

struct LayerExecutorWorkspaceUsage {
  std::vector<LayerPhaseWorkspaceUsage> phases;
  std::vector<LayerAuxWorkspaceUsage> auxiliaries;
  std::size_t activation_high_water_bytes = 0;
  std::size_t logits_high_water_bytes = 0;
  std::size_t auxiliary_high_water_bytes = 0;
  std::size_t total_high_water_bytes = 0;
  std::size_t activation_slack_bytes = 0;
  std::size_t logits_slack_bytes = 0;
  std::size_t auxiliary_slack_bytes = 0;
  std::size_t total_slack_bytes = 0;
};

struct LayerActivationFeedbackEdge {
  std::size_t source_stage = 0;
  std::size_t target_stage = 0;
  std::size_t source_phase_index = 0;
  std::size_t target_phase_index = 0;
  LayerPhase source_phase = LayerPhase::Unknown;
  LayerPhase target_phase = LayerPhase::Unknown;
  std::uint32_t source_rows = 0;
  std::uint32_t target_cols = 0;
  std::uint32_t stage_id = 0;
  bool crosses_phase = false;
  bool wraps_to_first_stage = false;
};

struct LayerExecutorDataflowPlan {
  std::vector<LayerActivationFeedbackEdge> activation_feedback_edges;
  std::size_t phase_crossing_feedbacks = 0;
  std::size_t wrap_feedbacks = 0;
};

struct LayerReplayCounters {
  std::size_t phase_callbacks = 0;
  std::size_t attention_phase_callbacks = 0;
  std::size_t ffn_phase_callbacks = 0;
  std::size_t ssm_phase_callbacks = 0;
  std::size_t output_phase_callbacks = 0;
  std::size_t stage_callbacks = 0;
  std::size_t activation_feedback_callbacks = 0;
  std::size_t total_callbacks = 0;
};

inline void note_layer_phase_callback(
    LayerReplayCounters& counters,
    LayerPhase phase) {
  switch (phase) {
    case LayerPhase::Attention:
      ++counters.attention_phase_callbacks;
      break;
    case LayerPhase::Ffn:
      ++counters.ffn_phase_callbacks;
      break;
    case LayerPhase::Ssm:
      ++counters.ssm_phase_callbacks;
      break;
    case LayerPhase::Output:
      ++counters.output_phase_callbacks;
      break;
    case LayerPhase::Unknown:
      throw std::runtime_error("layer executor cannot replay an unknown phase");
  }
}

inline void accumulate_layer_replay_counters(
    LayerReplayCounters& target,
    const LayerReplayCounters& source) {
  target.phase_callbacks += source.phase_callbacks;
  target.attention_phase_callbacks += source.attention_phase_callbacks;
  target.ffn_phase_callbacks += source.ffn_phase_callbacks;
  target.ssm_phase_callbacks += source.ssm_phase_callbacks;
  target.output_phase_callbacks += source.output_phase_callbacks;
  target.stage_callbacks += source.stage_callbacks;
  target.activation_feedback_callbacks += source.activation_feedback_callbacks;
  target.total_callbacks += source.total_callbacks;
}

inline void validate_layer_phase_execution_bounds(
    const LayerPhaseExecution& phase,
    std::size_t stage_count) {
  if (phase.count == 0) {
    throw std::runtime_error("layer executor phase contains no stages");
  }
  if (phase.begin > stage_count || phase.count > stage_count - phase.begin) {
    throw std::runtime_error("layer executor phase exceeds stage count");
  }
  if (phase.matvec_kernels_per_replay != phase.count) {
    throw std::runtime_error("layer executor phase matvec count mismatch");
  }
  if (phase.activation_feedbacks_per_replay != 0 &&
      phase.activation_feedbacks_per_replay != phase.count) {
    throw std::runtime_error("layer executor phase feedback count mismatch");
  }
  if (phase.total_kernels_per_replay !=
      phase.matvec_kernels_per_replay + phase.activation_feedbacks_per_replay) {
    throw std::runtime_error("layer executor phase total count mismatch");
  }
}

inline void validate_layer_execution_plan_bounds(
    const LayerExecutionPlan& plan,
    std::size_t stage_count) {
  if (plan.stage_count != stage_count) {
    throw std::runtime_error("layer executor plan stage count mismatch");
  }
  if (plan.phase_count != plan.phases.size()) {
    throw std::runtime_error("layer executor plan phase count mismatch");
  }
  std::size_t matvec_count = 0;
  std::size_t feedback_count = 0;
  std::size_t total_count = 0;
  for (const auto& phase : plan.phases) {
    validate_layer_phase_execution_bounds(phase, stage_count);
    matvec_count += phase.matvec_kernels_per_replay;
    feedback_count += phase.activation_feedbacks_per_replay;
    total_count += phase.total_kernels_per_replay;
  }
  if (matvec_count != plan.matvec_kernels_per_replay ||
      feedback_count != plan.activation_feedbacks_per_replay ||
      total_count != plan.total_kernels_per_replay) {
    throw std::runtime_error("layer executor plan kernel count mismatch");
  }
}

inline LayerExecutorWorkspacePlan build_layer_executor_workspace_plan(
    const LayerExecutionPlan& plan,
    std::span<const LayerStageWorkspaceDescriptor> stages) {
  validate_layer_execution_plan_bounds(plan, stages.size());

  LayerExecutorWorkspacePlan workspace;
  workspace.phases.reserve(plan.phases.size());
  for (const auto& phase : plan.phases) {
    LayerPhaseWorkspaceDescriptor phase_workspace;
    phase_workspace.phase = phase.phase;
    phase_workspace.begin = phase.begin;
    phase_workspace.count = phase.count;
    const std::size_t end = phase.begin + phase.count;
    for (std::size_t index = phase.begin; index < end; ++index) {
      const auto& stage = stages[index];
      if (stage.rows == 0 || stage.cols == 0) {
        throw std::runtime_error("layer executor workspace contains an empty stage");
      }
      phase_workspace.max_rows = std::max(phase_workspace.max_rows, stage.rows);
      phase_workspace.max_cols = std::max(phase_workspace.max_cols, stage.cols);
      phase_workspace.logical_values_per_replay +=
          static_cast<std::uint64_t>(stage.rows) * stage.cols;
    }
    workspace.max_rows = std::max(workspace.max_rows, phase_workspace.max_rows);
    workspace.max_cols = std::max(workspace.max_cols, phase_workspace.max_cols);
    workspace.logical_values_per_replay +=
        phase_workspace.logical_values_per_replay;
    workspace.phases.push_back(phase_workspace);
  }
  return workspace;
}

inline const LayerPhaseWorkspaceDescriptor& find_layer_workspace_phase_descriptor(
    const LayerExecutorWorkspacePlan& workspace,
    LayerPhase phase) {
  if (phase == LayerPhase::Unknown) {
    throw std::runtime_error("layer executor workspace lookup requires a known phase");
  }
  for (const auto& descriptor : workspace.phases) {
    if (descriptor.phase == phase) {
      return descriptor;
    }
  }
  throw std::runtime_error("layer executor workspace phase is not present");
}

inline std::vector<LayerAuxWorkspaceDescriptor>
build_gated_ffn_aux_workspace_descriptors(
    const LayerExecutorWorkspacePlan& workspace,
    bool gate_cache,
    bool up_cache) {
  std::vector<LayerAuxWorkspaceDescriptor> auxiliaries;
  if (!gate_cache && !up_cache) {
    return auxiliaries;
  }

  const auto& ffn = find_layer_workspace_phase_descriptor(workspace, LayerPhase::Ffn);
  if (ffn.max_cols == 0) {
    throw std::runtime_error("gated FFN auxiliary workspace requires FFN columns");
  }
  if (gate_cache) {
    auxiliaries.push_back(LayerAuxWorkspaceDescriptor{
        "ffn_gate_cache",
        LayerPhase::Ffn,
        ffn.max_cols,
    });
  }
  if (up_cache) {
    auxiliaries.push_back(LayerAuxWorkspaceDescriptor{
        "ffn_up_cache",
        LayerPhase::Ffn,
        ffn.max_cols,
    });
  }
  return auxiliaries;
}

inline std::vector<LayerAuxWorkspaceDescriptor>
build_ssm_recurrent_aux_workspace_descriptors(
    const LayerExecutorWorkspacePlan& workspace,
    bool recurrent_state,
    bool scan_scratch = false) {
  std::vector<LayerAuxWorkspaceDescriptor> auxiliaries;
  if (!recurrent_state && !scan_scratch) {
    return auxiliaries;
  }

  const auto& ssm =
      find_layer_workspace_phase_descriptor(workspace, LayerPhase::Ssm);
  const auto& output =
      find_layer_workspace_phase_descriptor(workspace, LayerPhase::Output);
  if (recurrent_state && output.max_cols == 0) {
    throw std::runtime_error("SSM recurrent workspace requires output columns");
  }
  if (scan_scratch && ssm.max_cols == 0) {
    throw std::runtime_error("SSM scan scratch workspace requires SSM columns");
  }
  if (recurrent_state) {
    auxiliaries.push_back(LayerAuxWorkspaceDescriptor{
        "ssm_recurrent_state",
        LayerPhase::Ssm,
        output.max_cols,
    });
  }
  if (scan_scratch) {
    auxiliaries.push_back(LayerAuxWorkspaceDescriptor{
        "ssm_scan_scratch",
        LayerPhase::Ssm,
        ssm.max_cols,
    });
  }
  return auxiliaries;
}

inline std::vector<LayerAuxWorkspaceDescriptor>
build_layer_phase_scratch_workspace_descriptors(
    const LayerExecutorWorkspacePlan& workspace,
    std::span<const LayerAuxWorkspaceDescriptor> scratch_requests) {
  std::vector<LayerAuxWorkspaceDescriptor> auxiliaries;
  auxiliaries.reserve(scratch_requests.size());
  for (const auto& request : scratch_requests) {
    if (request.name.empty()) {
      throw std::runtime_error("layer phase scratch workspace requires a name");
    }
    if (request.phase == LayerPhase::Unknown) {
      throw std::runtime_error("layer phase scratch workspace requires a known phase");
    }
    if (request.values == 0) {
      throw std::runtime_error("layer phase scratch workspace requires values");
    }
    (void)find_layer_workspace_phase_descriptor(workspace, request.phase);
    auxiliaries.push_back(request);
  }
  return auxiliaries;
}

inline std::size_t align_layer_workspace_bytes(
    std::size_t value,
    std::size_t alignment) {
  if (alignment == 0) {
    throw std::runtime_error("layer executor workspace alignment must be non-zero");
  }
  const auto remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  return value + alignment - remainder;
}

inline std::size_t layer_workspace_logical_bytes(
    std::uint32_t values,
    std::size_t element_bytes) {
  if (element_bytes == 0) {
    throw std::runtime_error("layer executor workspace element size must be non-zero");
  }
  if (values > std::numeric_limits<std::size_t>::max() / element_bytes) {
    throw std::runtime_error("layer executor workspace byte count overflow");
  }
  return static_cast<std::size_t>(values) * element_bytes;
}

inline LayerExecutorWorkspaceAllocationPlan
build_layer_executor_workspace_allocation_plan(
    const LayerExecutorWorkspacePlan& workspace,
    std::span<const LayerAuxWorkspaceDescriptor> auxiliaries,
    std::size_t element_bytes,
    std::size_t alignment) {
  if (workspace.phases.empty()) {
    throw std::runtime_error("layer executor workspace allocation requires phases");
  }
  if (element_bytes == 0) {
    throw std::runtime_error("layer executor workspace element size must be non-zero");
  }

  LayerExecutorWorkspaceAllocationPlan allocation;
  allocation.phases.reserve(workspace.phases.size());
  std::size_t offset = 0;
  for (const auto& phase : workspace.phases) {
    if (phase.phase == LayerPhase::Unknown) {
      throw std::runtime_error("layer executor workspace allocation has unknown phase");
    }
    if (phase.count == 0 || phase.max_rows == 0 || phase.max_cols == 0) {
      throw std::runtime_error("layer executor workspace allocation has empty phase");
    }

    LayerPhaseWorkspaceAllocation phase_allocation;
    phase_allocation.phase = phase.phase;
    phase_allocation.begin = phase.begin;
    phase_allocation.count = phase.count;

    offset = align_layer_workspace_bytes(offset, alignment);
    phase_allocation.activation_offset = offset;
    phase_allocation.activation_bytes =
        align_layer_workspace_bytes(
            static_cast<std::size_t>(phase.max_cols) * element_bytes,
            alignment);
    offset += phase_allocation.activation_bytes;

    offset = align_layer_workspace_bytes(offset, alignment);
    phase_allocation.logits_offset = offset;
    phase_allocation.logits_bytes =
        align_layer_workspace_bytes(
            static_cast<std::size_t>(phase.max_rows) * element_bytes,
            alignment);
    offset += phase_allocation.logits_bytes;

    phase_allocation.total_bytes =
        phase_allocation.activation_bytes + phase_allocation.logits_bytes;
    allocation.activation_bytes += phase_allocation.activation_bytes;
    allocation.logits_bytes += phase_allocation.logits_bytes;
    allocation.phases.push_back(phase_allocation);
  }

  allocation.phase_bytes = align_layer_workspace_bytes(offset, alignment);
  offset = allocation.phase_bytes;

  allocation.auxiliaries.reserve(auxiliaries.size());
  for (const auto& auxiliary : auxiliaries) {
    if (auxiliary.name.empty()) {
      throw std::runtime_error("layer executor auxiliary workspace requires a name");
    }
    if (auxiliary.phase == LayerPhase::Unknown) {
      throw std::runtime_error("layer executor auxiliary workspace has unknown phase");
    }
    if (auxiliary.values == 0) {
      throw std::runtime_error("layer executor auxiliary workspace is empty");
    }
    const auto duplicate = std::find_if(
        allocation.auxiliaries.begin(),
        allocation.auxiliaries.end(),
        [&](const LayerAuxWorkspaceAllocation& existing) {
          return existing.name == auxiliary.name;
        });
    if (duplicate != allocation.auxiliaries.end()) {
      throw std::runtime_error("layer executor auxiliary workspace name is duplicated");
    }

    offset = align_layer_workspace_bytes(offset, alignment);
    LayerAuxWorkspaceAllocation aux_allocation;
    aux_allocation.name = auxiliary.name;
    aux_allocation.phase = auxiliary.phase;
    aux_allocation.values = auxiliary.values;
    aux_allocation.offset = offset;
    aux_allocation.bytes = align_layer_workspace_bytes(
        static_cast<std::size_t>(auxiliary.values) * element_bytes,
        alignment);
    offset += aux_allocation.bytes;
    allocation.auxiliary_bytes += aux_allocation.bytes;
    allocation.auxiliaries.push_back(std::move(aux_allocation));
  }
  allocation.total_bytes = align_layer_workspace_bytes(offset, alignment);
  return allocation;
}

inline LayerExecutorWorkspaceAllocationPlan
build_layer_executor_workspace_allocation_plan(
    const LayerExecutorWorkspacePlan& workspace,
    std::size_t element_bytes,
    std::size_t alignment) {
  return build_layer_executor_workspace_allocation_plan(
      workspace,
      std::span<const LayerAuxWorkspaceDescriptor>{},
      element_bytes,
      alignment);
}

inline std::size_t find_layer_workspace_phase_index_for_stage(
    const LayerExecutorWorkspaceAllocationPlan& allocation,
    std::size_t stage_index) {
  for (std::size_t index = 0; index < allocation.phases.size(); ++index) {
    const auto& phase = allocation.phases[index];
    if (stage_index >= phase.begin && stage_index < phase.begin + phase.count) {
      return index;
    }
  }
  throw std::runtime_error("layer executor workspace does not cover stage");
}

inline LayerExecutorWorkspaceRuntime bind_layer_executor_workspace_runtime(
    const LayerExecutorWorkspacePlan& workspace,
    const LayerExecutorWorkspaceAllocationPlan& allocation,
    void* workspace_base) {
  if (workspace.phases.size() != allocation.phases.size()) {
    throw std::runtime_error("layer executor workspace runtime phase count mismatch");
  }
  if (allocation.total_bytes == 0) {
    throw std::runtime_error("layer executor workspace runtime requires bytes");
  }
  if (workspace_base == nullptr) {
    throw std::runtime_error("layer executor workspace runtime base is null");
  }

  auto* base = static_cast<std::uint8_t*>(workspace_base);
  LayerExecutorWorkspaceRuntime runtime;
  runtime.activation_bytes = allocation.activation_bytes;
  runtime.logits_bytes = allocation.logits_bytes;
  runtime.phase_bytes = allocation.phase_bytes;
  runtime.auxiliary_bytes = allocation.auxiliary_bytes;
  runtime.total_bytes = allocation.total_bytes;
  runtime.phases.reserve(allocation.phases.size());
  for (std::size_t index = 0; index < allocation.phases.size(); ++index) {
    const auto& descriptor = workspace.phases[index];
    const auto& phase_allocation = allocation.phases[index];
    if (descriptor.phase != phase_allocation.phase ||
        descriptor.begin != phase_allocation.begin ||
        descriptor.count != phase_allocation.count) {
      throw std::runtime_error("layer executor workspace runtime phase mismatch");
    }
    if (phase_allocation.activation_bytes == 0 ||
        phase_allocation.logits_bytes == 0) {
      throw std::runtime_error("layer executor workspace runtime has empty allocation");
    }
    if (phase_allocation.activation_offset + phase_allocation.activation_bytes >
            allocation.total_bytes ||
        phase_allocation.logits_offset + phase_allocation.logits_bytes >
            allocation.total_bytes) {
      throw std::runtime_error("layer executor workspace runtime allocation exceeds arena");
    }
    runtime.phases.push_back(LayerPhaseWorkspaceRuntime{
        descriptor.phase,
        descriptor.begin,
        descriptor.count,
        reinterpret_cast<float*>(base + phase_allocation.activation_offset),
        reinterpret_cast<float*>(base + phase_allocation.logits_offset),
        descriptor.max_rows,
        descriptor.max_cols,
        phase_allocation.activation_offset,
        phase_allocation.activation_bytes,
        phase_allocation.logits_offset,
        phase_allocation.logits_bytes,
        phase_allocation.total_bytes,
    });
  }
  runtime.auxiliaries.reserve(allocation.auxiliaries.size());
  for (const auto& auxiliary : allocation.auxiliaries) {
    if (auxiliary.bytes == 0 || auxiliary.values == 0) {
      throw std::runtime_error("layer executor auxiliary runtime has empty allocation");
    }
    if (auxiliary.offset + auxiliary.bytes > allocation.total_bytes) {
      throw std::runtime_error("layer executor auxiliary runtime allocation exceeds arena");
    }
    runtime.auxiliaries.push_back(LayerAuxWorkspaceRuntime{
        auxiliary.name,
        auxiliary.phase,
        reinterpret_cast<float*>(base + auxiliary.offset),
        auxiliary.values,
        auxiliary.offset,
        auxiliary.bytes,
    });
  }
  return runtime;
}

inline LayerAuxWorkspaceRuntime& find_layer_aux_workspace_runtime(
    LayerExecutorWorkspaceRuntime& runtime,
    std::string_view name) {
  for (auto& auxiliary : runtime.auxiliaries) {
    if (auxiliary.name == name) {
      return auxiliary;
    }
  }
  throw std::runtime_error("layer executor auxiliary workspace is not present");
}

inline const LayerAuxWorkspaceRuntime& find_layer_aux_workspace_runtime(
    const LayerExecutorWorkspaceRuntime& runtime,
    std::string_view name) {
  for (const auto& auxiliary : runtime.auxiliaries) {
    if (auxiliary.name == name) {
      return auxiliary;
    }
  }
  throw std::runtime_error("layer executor auxiliary workspace is not present");
}

inline std::size_t find_layer_workspace_runtime_phase_index_for_stage(
    const LayerExecutorWorkspaceRuntime& runtime,
    std::size_t stage_index) {
  for (std::size_t index = 0; index < runtime.phases.size(); ++index) {
    const auto& phase = runtime.phases[index];
    if (stage_index >= phase.begin && stage_index < phase.begin + phase.count) {
      return index;
    }
  }
  throw std::runtime_error("layer executor workspace runtime does not cover stage");
}

inline LayerExecutorDataflowPlan build_layer_executor_dataflow_plan(
    const LayerExecutionPlan& plan,
    std::span<const LayerStageWorkspaceDescriptor> stages,
    const LayerExecutorWorkspaceRuntime& runtime) {
  validate_layer_execution_plan_bounds(plan, stages.size());

  LayerExecutorDataflowPlan dataflow;
  if (plan.activation_feedbacks_per_replay == 0) {
    return dataflow;
  }
  if (runtime.phases.empty()) {
    throw std::runtime_error("layer executor dataflow requires runtime phases");
  }

  dataflow.activation_feedback_edges.reserve(plan.activation_feedbacks_per_replay);
  for (const auto& phase : plan.phases) {
    if (phase.activation_feedbacks_per_replay == 0) {
      continue;
    }
    if (phase.activation_feedbacks_per_replay != phase.count) {
      throw std::runtime_error("layer executor dataflow feedback count mismatch");
    }
    const auto end = phase.begin + phase.count;
    for (std::size_t source_stage = phase.begin; source_stage < end; ++source_stage) {
      const auto target_stage = (source_stage + 1u) % stages.size();
      const auto& source = stages[source_stage];
      const auto& target = stages[target_stage];
      if (source.rows == 0 || target.cols == 0) {
        throw std::runtime_error("layer executor dataflow contains an empty edge");
      }

      LayerActivationFeedbackEdge edge;
      edge.source_stage = source_stage;
      edge.target_stage = target_stage;
      edge.source_phase_index =
          find_layer_workspace_runtime_phase_index_for_stage(runtime, source_stage);
      edge.target_phase_index =
          find_layer_workspace_runtime_phase_index_for_stage(runtime, target_stage);
      edge.source_phase = runtime.phases[edge.source_phase_index].phase;
      edge.target_phase = runtime.phases[edge.target_phase_index].phase;
      edge.source_rows = source.rows;
      edge.target_cols = target.cols;
      edge.stage_id = static_cast<std::uint32_t>(source_stage);
      edge.crosses_phase = edge.source_phase_index != edge.target_phase_index;
      edge.wraps_to_first_stage = target_stage == 0;
      if (edge.crosses_phase) {
        ++dataflow.phase_crossing_feedbacks;
      }
      if (edge.wraps_to_first_stage) {
        ++dataflow.wrap_feedbacks;
      }
      dataflow.activation_feedback_edges.push_back(edge);
    }
  }
  if (dataflow.activation_feedback_edges.size() !=
      plan.activation_feedbacks_per_replay) {
    throw std::runtime_error("layer executor dataflow edge count mismatch");
  }
  return dataflow;
}

inline const LayerActivationFeedbackEdge& find_layer_activation_feedback_edge(
    const LayerExecutorDataflowPlan& dataflow,
    std::size_t source_stage) {
  for (const auto& edge : dataflow.activation_feedback_edges) {
    if (edge.source_stage == source_stage) {
      return edge;
    }
  }
  throw std::runtime_error("layer executor dataflow does not cover source stage");
}

inline LayerExecutorWorkspaceUsage measure_layer_executor_workspace_usage(
    const LayerExecutorWorkspacePlan& workspace,
    const LayerExecutorWorkspaceAllocationPlan& allocation,
    std::size_t element_bytes) {
  if (workspace.phases.size() != allocation.phases.size()) {
    throw std::runtime_error("layer executor workspace usage phase count mismatch");
  }
  if (allocation.phases.empty()) {
    throw std::runtime_error("layer executor workspace usage requires phases");
  }

  LayerExecutorWorkspaceUsage usage;
  usage.phases.reserve(allocation.phases.size());
  for (std::size_t index = 0; index < allocation.phases.size(); ++index) {
    const auto& descriptor = workspace.phases[index];
    const auto& phase_allocation = allocation.phases[index];
    if (descriptor.phase != phase_allocation.phase ||
        descriptor.begin != phase_allocation.begin ||
        descriptor.count != phase_allocation.count) {
      throw std::runtime_error("layer executor workspace usage phase mismatch");
    }
    if (phase_allocation.activation_bytes == 0 ||
        phase_allocation.logits_bytes == 0) {
      throw std::runtime_error("layer executor workspace usage has empty allocation");
    }
    const auto activation_high_water =
        layer_workspace_logical_bytes(descriptor.max_cols, element_bytes);
    const auto logits_high_water =
        layer_workspace_logical_bytes(descriptor.max_rows, element_bytes);
    if (activation_high_water > phase_allocation.activation_bytes ||
        logits_high_water > phase_allocation.logits_bytes) {
      throw std::runtime_error("layer executor workspace usage exceeds allocation");
    }
    const auto total_high_water = activation_high_water + logits_high_water;
    if (total_high_water < activation_high_water ||
        total_high_water > phase_allocation.total_bytes) {
      throw std::runtime_error("layer executor workspace usage total exceeds allocation");
    }

    LayerPhaseWorkspaceUsage phase_usage;
    phase_usage.phase = descriptor.phase;
    phase_usage.begin = descriptor.begin;
    phase_usage.count = descriptor.count;
    phase_usage.activation_high_water_bytes = activation_high_water;
    phase_usage.logits_high_water_bytes = logits_high_water;
    phase_usage.total_high_water_bytes = total_high_water;
    phase_usage.activation_slack_bytes =
        phase_allocation.activation_bytes - activation_high_water;
    phase_usage.logits_slack_bytes =
        phase_allocation.logits_bytes - logits_high_water;
    phase_usage.total_slack_bytes =
        phase_allocation.total_bytes - total_high_water;
    usage.activation_high_water_bytes += phase_usage.activation_high_water_bytes;
    usage.logits_high_water_bytes += phase_usage.logits_high_water_bytes;
    usage.total_high_water_bytes += phase_usage.total_high_water_bytes;
    usage.activation_slack_bytes += phase_usage.activation_slack_bytes;
    usage.logits_slack_bytes += phase_usage.logits_slack_bytes;
    usage.phases.push_back(phase_usage);
  }
  usage.auxiliaries.reserve(allocation.auxiliaries.size());
  for (const auto& auxiliary : allocation.auxiliaries) {
    if (auxiliary.phase == LayerPhase::Unknown) {
      throw std::runtime_error("layer executor auxiliary workspace usage has unknown phase");
    }
    if (auxiliary.bytes == 0 || auxiliary.values == 0) {
      throw std::runtime_error("layer executor auxiliary workspace usage has empty allocation");
    }
    const auto high_water =
        layer_workspace_logical_bytes(auxiliary.values, element_bytes);
    if (high_water > auxiliary.bytes) {
      throw std::runtime_error("layer executor auxiliary workspace usage exceeds allocation");
    }
    LayerAuxWorkspaceUsage auxiliary_usage;
    auxiliary_usage.name = auxiliary.name;
    auxiliary_usage.phase = auxiliary.phase;
    auxiliary_usage.high_water_bytes = high_water;
    auxiliary_usage.slack_bytes = auxiliary.bytes - high_water;
    usage.auxiliary_high_water_bytes += auxiliary_usage.high_water_bytes;
    usage.auxiliary_slack_bytes += auxiliary_usage.slack_bytes;
    usage.total_high_water_bytes += auxiliary_usage.high_water_bytes;
    usage.auxiliaries.push_back(std::move(auxiliary_usage));
  }
  if (usage.total_high_water_bytes > allocation.total_bytes) {
    throw std::runtime_error("layer executor workspace usage exceeds allocation plan");
  }
  usage.total_slack_bytes = allocation.total_bytes - usage.total_high_water_bytes;
  return usage;
}

inline LayerExecutorWorkspaceUsage measure_layer_executor_workspace_usage(
    const LayerExecutorWorkspaceRuntime& runtime,
    std::size_t element_bytes) {
  if (runtime.phases.empty()) {
    throw std::runtime_error("layer executor workspace usage requires phases");
  }

  LayerExecutorWorkspaceUsage usage;
  usage.phases.reserve(runtime.phases.size());
  for (const auto& phase : runtime.phases) {
    if (phase.phase == LayerPhase::Unknown) {
      throw std::runtime_error("layer executor workspace usage has unknown phase");
    }
    if (phase.activation == nullptr || phase.logits == nullptr) {
      throw std::runtime_error("layer executor workspace usage has null runtime pointer");
    }
    const auto activation_high_water =
        layer_workspace_logical_bytes(phase.max_cols, element_bytes);
    const auto logits_high_water =
        layer_workspace_logical_bytes(phase.max_rows, element_bytes);
    if (activation_high_water > phase.activation_bytes ||
        logits_high_water > phase.logits_bytes) {
      throw std::runtime_error("layer executor workspace usage exceeds allocation");
    }
    const auto total_high_water = activation_high_water + logits_high_water;
    if (total_high_water < activation_high_water ||
        total_high_water > phase.total_bytes) {
      throw std::runtime_error("layer executor workspace usage total exceeds allocation");
    }

    LayerPhaseWorkspaceUsage phase_usage;
    phase_usage.phase = phase.phase;
    phase_usage.begin = phase.begin;
    phase_usage.count = phase.count;
    phase_usage.activation_high_water_bytes = activation_high_water;
    phase_usage.logits_high_water_bytes = logits_high_water;
    phase_usage.total_high_water_bytes = total_high_water;
    phase_usage.activation_slack_bytes =
        phase.activation_bytes - activation_high_water;
    phase_usage.logits_slack_bytes = phase.logits_bytes - logits_high_water;
    phase_usage.total_slack_bytes = phase.total_bytes - total_high_water;
    usage.activation_high_water_bytes += phase_usage.activation_high_water_bytes;
    usage.logits_high_water_bytes += phase_usage.logits_high_water_bytes;
    usage.total_high_water_bytes += phase_usage.total_high_water_bytes;
    usage.activation_slack_bytes += phase_usage.activation_slack_bytes;
    usage.logits_slack_bytes += phase_usage.logits_slack_bytes;
    usage.phases.push_back(phase_usage);
  }
  usage.auxiliaries.reserve(runtime.auxiliaries.size());
  for (const auto& auxiliary : runtime.auxiliaries) {
    if (auxiliary.phase == LayerPhase::Unknown) {
      throw std::runtime_error("layer executor auxiliary workspace usage has unknown phase");
    }
    if (auxiliary.data == nullptr) {
      throw std::runtime_error("layer executor auxiliary workspace usage has null runtime pointer");
    }
    if (auxiliary.bytes == 0 || auxiliary.values == 0) {
      throw std::runtime_error("layer executor auxiliary workspace usage has empty allocation");
    }
    const auto high_water =
        layer_workspace_logical_bytes(auxiliary.values, element_bytes);
    if (high_water > auxiliary.bytes) {
      throw std::runtime_error("layer executor auxiliary workspace usage exceeds allocation");
    }
    LayerAuxWorkspaceUsage auxiliary_usage;
    auxiliary_usage.name = auxiliary.name;
    auxiliary_usage.phase = auxiliary.phase;
    auxiliary_usage.high_water_bytes = high_water;
    auxiliary_usage.slack_bytes = auxiliary.bytes - high_water;
    usage.auxiliary_high_water_bytes += auxiliary_usage.high_water_bytes;
    usage.auxiliary_slack_bytes += auxiliary_usage.slack_bytes;
    usage.total_high_water_bytes += auxiliary_usage.high_water_bytes;
    usage.auxiliaries.push_back(std::move(auxiliary_usage));
  }
  if (usage.total_high_water_bytes > runtime.total_bytes) {
    throw std::runtime_error("layer executor workspace usage exceeds runtime arena");
  }
  usage.total_slack_bytes = runtime.total_bytes - usage.total_high_water_bytes;
  return usage;
}

template <typename LaunchStage, typename LaunchFeedback>
LayerReplayCounters replay_layer_phase(
    const LayerPhaseExecution& phase,
    std::size_t stage_count,
    LaunchStage&& launch_stage,
    LaunchFeedback&& launch_feedback) {
  validate_layer_phase_execution_bounds(phase, stage_count);

  LayerReplayCounters counters;
  counters.phase_callbacks = 1;
  note_layer_phase_callback(counters, phase.phase);
  const std::size_t end = phase.begin + phase.count;
  for (std::size_t index = phase.begin; index < end; ++index) {
    launch_stage(index);
    ++counters.stage_callbacks;
    if (phase.activation_feedbacks_per_replay != 0) {
      launch_feedback(index);
      ++counters.activation_feedback_callbacks;
    }
  }
  counters.total_callbacks =
      counters.stage_callbacks + counters.activation_feedback_callbacks;
  return counters;
}

inline void validate_layer_phase_replay_counters(
    const LayerPhaseExecution& phase,
    const LayerReplayCounters& counters) {
  if (counters.phase_callbacks != 1) {
    throw std::runtime_error("layer executor phase callback count mismatch");
  }
  if (counters.stage_callbacks != phase.count) {
    throw std::runtime_error("layer executor stage callback count mismatch");
  }
  if (counters.activation_feedback_callbacks !=
      phase.activation_feedbacks_per_replay) {
    throw std::runtime_error("layer executor feedback callback count mismatch");
  }
  if (counters.total_callbacks != phase.total_kernels_per_replay) {
    throw std::runtime_error("layer executor total callback count mismatch");
  }
  switch (phase.phase) {
    case LayerPhase::Attention:
      if (counters.attention_phase_callbacks != 1) {
        throw std::runtime_error("layer executor attention phase callback mismatch");
      }
      break;
    case LayerPhase::Ffn:
      if (counters.ffn_phase_callbacks != 1) {
        throw std::runtime_error("layer executor ffn phase callback mismatch");
      }
      break;
    case LayerPhase::Ssm:
      if (counters.ssm_phase_callbacks != 1) {
        throw std::runtime_error("layer executor ssm phase callback mismatch");
      }
      break;
    case LayerPhase::Output:
      if (counters.output_phase_callbacks != 1) {
        throw std::runtime_error("layer executor output phase callback mismatch");
      }
      break;
    case LayerPhase::Unknown:
      throw std::runtime_error("layer executor cannot validate an unknown phase");
  }
}

template <typename LaunchStage, typename LaunchFeedback>
LayerReplayCounters replay_layer_execution_plan(
    const LayerExecutionPlan& plan,
    std::size_t stage_count,
    LaunchStage&& launch_stage,
    LaunchFeedback&& launch_feedback) {
  validate_layer_execution_plan_bounds(plan, stage_count);

  LayerReplayCounters counters;
  for (const auto& phase : plan.phases) {
    const auto phase_counters = replay_layer_phase(
        phase,
        stage_count,
        launch_stage,
        launch_feedback);
    accumulate_layer_replay_counters(counters, phase_counters);
  }
  return counters;
}

template <
    typename LaunchAttentionPhase,
    typename LaunchFfnPhase,
    typename LaunchSsmPhase,
    typename LaunchOutputPhase>
LayerReplayCounters replay_layer_execution_plan_by_phase(
    const LayerExecutionPlan& plan,
    std::size_t stage_count,
    LaunchAttentionPhase&& launch_attention_phase,
    LaunchFfnPhase&& launch_ffn_phase,
    LaunchSsmPhase&& launch_ssm_phase,
    LaunchOutputPhase&& launch_output_phase) {
  validate_layer_execution_plan_bounds(plan, stage_count);

  LayerReplayCounters counters;
  for (const auto& phase : plan.phases) {
    LayerReplayCounters phase_counters;
    switch (phase.phase) {
      case LayerPhase::Attention:
        phase_counters = launch_attention_phase(phase);
        break;
      case LayerPhase::Ffn:
        phase_counters = launch_ffn_phase(phase);
        break;
      case LayerPhase::Ssm:
        phase_counters = launch_ssm_phase(phase);
        break;
      case LayerPhase::Output:
        phase_counters = launch_output_phase(phase);
        break;
      case LayerPhase::Unknown:
        throw std::runtime_error("layer executor cannot dispatch an unknown phase");
    }
    validate_layer_phase_replay_counters(phase, phase_counters);
    accumulate_layer_replay_counters(counters, phase_counters);
  }
  return counters;
}

}  // namespace rtxllm
