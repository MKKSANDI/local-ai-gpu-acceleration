#include "runtime/core/graphs/cuda_graph_bucket.h"
#include "runtime/core/scheduler/admission.h"
#include "runtime/core/scheduler/layer_executor.h"
#include "runtime/core/scheduler/layer_plan.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct CaseResult {
  std::string name;
  bool passed = false;
  std::string detail;
};

const char* json_bool(bool value) {
  return value ? "true" : "false";
}

bool phases_equal(const std::vector<rtxllm::LayerPhase>& actual,
                  const std::vector<rtxllm::LayerPhase>& expected) {
  return actual.size() == expected.size() &&
         std::equal(actual.begin(), actual.end(), expected.begin());
}

std::vector<rtxllm::LayerStageDescriptor> valid_role_plan() {
  return {
      {"attn_gate.weight", true},
      {"attn_qkv.weight", true},
      {"ffn_gate_exps.weight", true},
      {"ffn_gate_shexp.weight", true},
      {"ffn_up_exps.weight", true},
      {"ffn_up_shexp.weight", true},
      {"ffn_down_exps.weight", true},
      {"ffn_down_shexp.weight", true},
      {"ssm_alpha.weight", true},
      {"ssm_beta.weight", true},
      {"ssm_out.weight", true},
      {"output.weight", true},
  };
}

bool rejected_with(std::vector<rtxllm::LayerStageDescriptor> stages,
                   std::string expected_message,
                   std::string* detail) {
  const auto segments = rtxllm::build_layer_phase_segments(stages);
  const auto validation = rtxllm::validate_layer_phase_plan(stages, segments);
  if (validation.passed) {
    *detail = "plan unexpectedly passed";
    return false;
  }
  if (validation.message != expected_message) {
    *detail = "expected '" + expected_message + "', got '" + validation.message + "'";
    return false;
  }
  *detail = validation.message;
  return true;
}

}  // namespace

int main() {
  using rtxllm::LayerPhase;
  using rtxllm::LayerPhaseSegment;
  using rtxllm::LayerStageDescriptor;

  std::vector<CaseResult> cases;

  const auto valid_stages = valid_role_plan();
  const auto valid_segments = rtxllm::build_layer_phase_segments(valid_stages);
  const auto validation = rtxllm::validate_layer_phase_plan(valid_stages, valid_segments);
  const auto counts = rtxllm::build_layer_phase_counts(valid_stages);

  const bool valid_plan_passed =
      validation.passed &&
      validation.message == "layer phase plan validated" &&
      phases_equal(validation.observed_phases,
                   {LayerPhase::Attention, LayerPhase::Ffn, LayerPhase::Ssm,
                    LayerPhase::Output}) &&
      valid_segments.size() == 4 &&
      valid_segments[0].begin == 0 && valid_segments[0].count == 2 &&
      valid_segments[1].begin == 2 && valid_segments[1].count == 6 &&
      valid_segments[2].begin == 8 && valid_segments[2].count == 3 &&
      valid_segments[3].begin == 11 && valid_segments[3].count == 1 &&
      counts.at("attention") == 2 && counts.at("ffn") == 6 &&
      counts.at("ssm") == 3 && counts.at("output") == 1;
  cases.push_back({"valid_plan_passed", valid_plan_passed, validation.message});

  const auto chained_execution = rtxllm::build_layer_execution_plan(valid_segments, true);
  const bool execution_plan_passed =
      chained_execution.phase_count == 4 &&
      chained_execution.stage_count == 12 &&
      chained_execution.matvec_kernels_per_replay == 12 &&
      chained_execution.activation_feedbacks_per_replay == 12 &&
      chained_execution.total_kernels_per_replay == 24 &&
      chained_execution.phases.size() == 4 &&
      chained_execution.phases[0].total_kernels_per_replay == 4 &&
      chained_execution.phases[1].total_kernels_per_replay == 12 &&
      chained_execution.phases[2].total_kernels_per_replay == 6 &&
      chained_execution.phases[3].total_kernels_per_replay == 2;
  cases.push_back(
      {"execution_plan_passed", execution_plan_passed, "chained execution counts"});

  const auto unchained_execution = rtxllm::build_layer_execution_plan(valid_segments, false);
  const bool unchained_execution_passed =
      unchained_execution.phase_count == 4 &&
      unchained_execution.stage_count == 12 &&
      unchained_execution.matvec_kernels_per_replay == 12 &&
      unchained_execution.activation_feedbacks_per_replay == 0 &&
      unchained_execution.total_kernels_per_replay == 12;
  cases.push_back({"unchained_execution_passed", unchained_execution_passed,
                   "unchained execution counts"});

  const std::vector<rtxllm::LayerStageWorkspaceDescriptor> valid_stage_workspaces = {
      {64, 2048},
      {8192, 4096},
      {64, 2048},
      {64, 2048},
      {64, 2048},
      {64, 2048},
      {64, 512},
      {64, 512},
      {32, 2048},
      {32, 2048},
      {64, 2048},
      {64, 248320},
  };
  const auto workspace_plan = rtxllm::build_layer_executor_workspace_plan(
      chained_execution,
      valid_stage_workspaces);
  const bool workspace_plan_passed =
      workspace_plan.phases.size() == 4 &&
      workspace_plan.max_rows == 8192 &&
      workspace_plan.max_cols == 248320 &&
      workspace_plan.logical_values_per_replay == 50429952ull &&
      workspace_plan.phases[0].phase == LayerPhase::Attention &&
      workspace_plan.phases[0].max_rows == 8192 &&
      workspace_plan.phases[0].max_cols == 4096 &&
      workspace_plan.phases[0].logical_values_per_replay == 33685504ull &&
      workspace_plan.phases[1].phase == LayerPhase::Ffn &&
      workspace_plan.phases[1].max_rows == 64 &&
      workspace_plan.phases[1].max_cols == 2048 &&
      workspace_plan.phases[1].logical_values_per_replay == 589824ull &&
      workspace_plan.phases[2].phase == LayerPhase::Ssm &&
      workspace_plan.phases[2].max_rows == 64 &&
      workspace_plan.phases[2].max_cols == 2048 &&
      workspace_plan.phases[2].logical_values_per_replay == 262144ull &&
      workspace_plan.phases[3].phase == LayerPhase::Output &&
      workspace_plan.phases[3].max_rows == 64 &&
      workspace_plan.phases[3].max_cols == 248320 &&
      workspace_plan.phases[3].logical_values_per_replay == 15892480ull;
  cases.push_back({"workspace_plan_passed", workspace_plan_passed,
                   "phase-local workspace descriptors"});

  const auto workspace_allocation_plan =
      rtxllm::build_layer_executor_workspace_allocation_plan(
          workspace_plan,
          sizeof(float),
          256);
  const bool workspace_allocation_plan_passed =
      workspace_allocation_plan.phases.size() == 4 &&
      workspace_allocation_plan.activation_bytes == 1026048ull &&
      workspace_allocation_plan.logits_bytes == 33536ull &&
      workspace_allocation_plan.total_bytes == 1059584ull &&
      workspace_allocation_plan.phases[0].activation_offset == 0 &&
      workspace_allocation_plan.phases[0].activation_bytes == 16384ull &&
      workspace_allocation_plan.phases[0].logits_offset == 16384ull &&
      workspace_allocation_plan.phases[0].logits_bytes == 32768ull &&
      workspace_allocation_plan.phases[1].activation_offset == 49152ull &&
      workspace_allocation_plan.phases[1].activation_bytes == 8192ull &&
      workspace_allocation_plan.phases[1].logits_offset == 57344ull &&
      workspace_allocation_plan.phases[1].logits_bytes == 256ull &&
      workspace_allocation_plan.phases[2].activation_offset == 57600ull &&
      workspace_allocation_plan.phases[2].activation_bytes == 8192ull &&
      workspace_allocation_plan.phases[2].logits_offset == 65792ull &&
      workspace_allocation_plan.phases[2].logits_bytes == 256ull &&
      workspace_allocation_plan.phases[3].activation_offset == 66048ull &&
      workspace_allocation_plan.phases[3].activation_bytes == 993280ull &&
      workspace_allocation_plan.phases[3].logits_offset == 1059328ull &&
      workspace_allocation_plan.phases[3].logits_bytes == 256ull &&
      rtxllm::find_layer_workspace_phase_index_for_stage(
          workspace_allocation_plan,
          0) == 0 &&
      rtxllm::find_layer_workspace_phase_index_for_stage(
          workspace_allocation_plan,
          7) == 1 &&
      rtxllm::find_layer_workspace_phase_index_for_stage(
          workspace_allocation_plan,
          10) == 2 &&
      rtxllm::find_layer_workspace_phase_index_for_stage(
          workspace_allocation_plan,
          11) == 3;
  cases.push_back({"workspace_allocation_plan_passed",
                   workspace_allocation_plan_passed,
                   "phase-local activation/logits allocation"});

  std::vector<std::uint8_t> runtime_workspace(
      workspace_allocation_plan.total_bytes,
      0);
  const auto workspace_runtime =
      rtxllm::bind_layer_executor_workspace_runtime(
          workspace_plan,
          workspace_allocation_plan,
          runtime_workspace.data());
  const bool workspace_runtime_binding_passed =
      workspace_runtime.phases.size() == 4 &&
      workspace_runtime.activation_bytes == workspace_allocation_plan.activation_bytes &&
      workspace_runtime.logits_bytes == workspace_allocation_plan.logits_bytes &&
      workspace_runtime.total_bytes == workspace_allocation_plan.total_bytes &&
      workspace_runtime.phases[0].phase == LayerPhase::Attention &&
      workspace_runtime.phases[0].activation ==
          reinterpret_cast<float*>(runtime_workspace.data()) &&
      workspace_runtime.phases[0].logits ==
          reinterpret_cast<float*>(runtime_workspace.data() + 16384ull) &&
      workspace_runtime.phases[3].phase == LayerPhase::Output &&
      workspace_runtime.phases[3].activation ==
          reinterpret_cast<float*>(runtime_workspace.data() + 66048ull) &&
      workspace_runtime.phases[3].logits ==
          reinterpret_cast<float*>(runtime_workspace.data() + 1059328ull) &&
      workspace_runtime.phases[3].max_cols == 248320 &&
      rtxllm::find_layer_workspace_runtime_phase_index_for_stage(
          workspace_runtime,
          0) == 0 &&
      rtxllm::find_layer_workspace_runtime_phase_index_for_stage(
          workspace_runtime,
          7) == 1 &&
      rtxllm::find_layer_workspace_runtime_phase_index_for_stage(
          workspace_runtime,
          10) == 2 &&
      rtxllm::find_layer_workspace_runtime_phase_index_for_stage(
          workspace_runtime,
          11) == 3;
  cases.push_back({"workspace_runtime_binding_passed",
                   workspace_runtime_binding_passed,
                   "phase-local runtime binding"});

  const auto dataflow_plan = rtxllm::build_layer_executor_dataflow_plan(
      chained_execution,
      valid_stage_workspaces,
      workspace_runtime);
  const bool dataflow_plan_passed =
      dataflow_plan.activation_feedback_edges.size() == 12 &&
      dataflow_plan.phase_crossing_feedbacks == 4 &&
      dataflow_plan.wrap_feedbacks == 1 &&
      dataflow_plan.activation_feedback_edges[0].source_stage == 0 &&
      dataflow_plan.activation_feedback_edges[0].target_stage == 1 &&
      dataflow_plan.activation_feedback_edges[0].source_phase == LayerPhase::Attention &&
      dataflow_plan.activation_feedback_edges[0].target_phase == LayerPhase::Attention &&
      !dataflow_plan.activation_feedback_edges[0].crosses_phase &&
      dataflow_plan.activation_feedback_edges[0].source_rows == 64 &&
      dataflow_plan.activation_feedback_edges[0].target_cols == 4096 &&
      dataflow_plan.activation_feedback_edges[1].source_stage == 1 &&
      dataflow_plan.activation_feedback_edges[1].target_stage == 2 &&
      dataflow_plan.activation_feedback_edges[1].crosses_phase &&
      dataflow_plan.activation_feedback_edges[1].source_phase == LayerPhase::Attention &&
      dataflow_plan.activation_feedback_edges[1].target_phase == LayerPhase::Ffn &&
      dataflow_plan.activation_feedback_edges[11].source_stage == 11 &&
      dataflow_plan.activation_feedback_edges[11].target_stage == 0 &&
      dataflow_plan.activation_feedback_edges[11].wraps_to_first_stage &&
      dataflow_plan.activation_feedback_edges[11].crosses_phase &&
      dataflow_plan.activation_feedback_edges[11].source_phase == LayerPhase::Output &&
      dataflow_plan.activation_feedback_edges[11].target_phase == LayerPhase::Attention &&
      dataflow_plan.activation_feedback_edges[11].source_rows == 64 &&
      dataflow_plan.activation_feedback_edges[11].target_cols == 2048 &&
      rtxllm::find_layer_activation_feedback_edge(dataflow_plan, 7).target_stage == 8;
  cases.push_back({"dataflow_plan_passed",
                   dataflow_plan_passed,
                   "phase-local activation feedback dataflow"});

  const auto unchained_dataflow_plan = rtxllm::build_layer_executor_dataflow_plan(
      unchained_execution,
      valid_stage_workspaces,
      workspace_runtime);
  const bool unchained_dataflow_passed =
      unchained_dataflow_plan.activation_feedback_edges.empty() &&
      unchained_dataflow_plan.phase_crossing_feedbacks == 0 &&
      unchained_dataflow_plan.wrap_feedbacks == 0;
  cases.push_back({"unchained_dataflow_passed",
                   unchained_dataflow_passed,
                   "unchained execution has no feedback edges"});

  const auto workspace_usage =
      rtxllm::measure_layer_executor_workspace_usage(
          workspace_runtime,
          sizeof(float));
  const bool workspace_usage_passed =
      workspace_usage.phases.size() == 4 &&
      workspace_usage.activation_high_water_bytes == 1026048ull &&
      workspace_usage.logits_high_water_bytes == 33536ull &&
      workspace_usage.total_high_water_bytes == 1059584ull &&
      workspace_usage.activation_slack_bytes == 0 &&
      workspace_usage.logits_slack_bytes == 0 &&
      workspace_usage.total_slack_bytes == 0 &&
      workspace_usage.phases[0].phase == LayerPhase::Attention &&
      workspace_usage.phases[0].activation_high_water_bytes == 16384ull &&
      workspace_usage.phases[0].logits_high_water_bytes == 32768ull &&
      workspace_usage.phases[3].phase == LayerPhase::Output &&
      workspace_usage.phases[3].activation_high_water_bytes == 993280ull &&
      workspace_usage.phases[3].logits_high_water_bytes == 256ull;
  cases.push_back({"workspace_usage_passed",
                   workspace_usage_passed,
                   "phase-local workspace high-water usage"});

  const auto ffn_aux_workspaces =
      rtxllm::build_gated_ffn_aux_workspace_descriptors(
          workspace_plan,
          true,
          true);
  const auto aux_workspace_allocation_plan =
      rtxllm::build_layer_executor_workspace_allocation_plan(
          workspace_plan,
          ffn_aux_workspaces,
          sizeof(float),
          256);
  std::vector<std::uint8_t> aux_runtime_workspace(
      aux_workspace_allocation_plan.total_bytes,
      0);
  const auto aux_workspace_runtime =
      rtxllm::bind_layer_executor_workspace_runtime(
          workspace_plan,
          aux_workspace_allocation_plan,
          aux_runtime_workspace.data());
  const auto aux_workspace_usage =
      rtxllm::measure_layer_executor_workspace_usage(
          aux_workspace_runtime,
          sizeof(float));
  const auto& gate_aux =
      rtxllm::find_layer_aux_workspace_runtime(aux_workspace_runtime, "ffn_gate_cache");
  const auto& up_aux =
      rtxllm::find_layer_aux_workspace_runtime(aux_workspace_runtime, "ffn_up_cache");
  const bool aux_workspace_allocation_passed =
      ffn_aux_workspaces.size() == 2 &&
      aux_workspace_allocation_plan.phases.size() == 4 &&
      aux_workspace_allocation_plan.auxiliaries.size() == 2 &&
      aux_workspace_allocation_plan.phase_bytes == 1059584ull &&
      aux_workspace_allocation_plan.auxiliary_bytes == 16384ull &&
      aux_workspace_allocation_plan.total_bytes == 1075968ull &&
      aux_workspace_allocation_plan.auxiliaries[0].name == "ffn_gate_cache" &&
      aux_workspace_allocation_plan.auxiliaries[0].phase == LayerPhase::Ffn &&
      aux_workspace_allocation_plan.auxiliaries[0].values == 2048 &&
      aux_workspace_allocation_plan.auxiliaries[0].offset == 1059584ull &&
      aux_workspace_allocation_plan.auxiliaries[0].bytes == 8192ull &&
      aux_workspace_allocation_plan.auxiliaries[1].name == "ffn_up_cache" &&
      aux_workspace_allocation_plan.auxiliaries[1].phase == LayerPhase::Ffn &&
      aux_workspace_allocation_plan.auxiliaries[1].values == 2048 &&
      aux_workspace_allocation_plan.auxiliaries[1].offset == 1067776ull &&
      aux_workspace_allocation_plan.auxiliaries[1].bytes == 8192ull &&
      aux_workspace_runtime.phase_bytes == aux_workspace_allocation_plan.phase_bytes &&
      aux_workspace_runtime.auxiliary_bytes == 16384ull &&
      gate_aux.data == reinterpret_cast<float*>(
          aux_runtime_workspace.data() + 1059584ull) &&
      up_aux.data == reinterpret_cast<float*>(
          aux_runtime_workspace.data() + 1067776ull) &&
      aux_workspace_usage.auxiliaries.size() == 2 &&
      aux_workspace_usage.auxiliary_high_water_bytes == 16384ull &&
      aux_workspace_usage.total_high_water_bytes == 1075968ull &&
      aux_workspace_usage.total_slack_bytes == 0;
  cases.push_back({"aux_workspace_allocation_passed",
                   aux_workspace_allocation_passed,
                   "core executor auxiliary FFN cache workspace"});

  const std::vector<rtxllm::LayerAuxWorkspaceDescriptor> scratch_requests = {
      {"attention_qkv_scratch", LayerPhase::Attention, 8192},
      {"ssm_state_scratch", LayerPhase::Ssm, 2048},
  };
  const auto scratch_workspaces =
      rtxllm::build_layer_phase_scratch_workspace_descriptors(
          workspace_plan,
          scratch_requests);
  const auto scratch_workspace_allocation_plan =
      rtxllm::build_layer_executor_workspace_allocation_plan(
          workspace_plan,
          scratch_workspaces,
          sizeof(float),
          256);
  std::vector<std::uint8_t> scratch_runtime_workspace(
      scratch_workspace_allocation_plan.total_bytes,
      0);
  const auto scratch_workspace_runtime =
      rtxllm::bind_layer_executor_workspace_runtime(
          workspace_plan,
          scratch_workspace_allocation_plan,
          scratch_runtime_workspace.data());
  const auto scratch_workspace_usage =
      rtxllm::measure_layer_executor_workspace_usage(
          scratch_workspace_runtime,
          sizeof(float));
  const auto& attention_scratch =
      rtxllm::find_layer_aux_workspace_runtime(
          scratch_workspace_runtime,
          "attention_qkv_scratch");
  const auto& ssm_scratch =
      rtxllm::find_layer_aux_workspace_runtime(
          scratch_workspace_runtime,
          "ssm_state_scratch");
  const bool phase_scratch_workspace_passed =
      scratch_workspaces.size() == 2 &&
      scratch_workspace_allocation_plan.auxiliaries.size() == 2 &&
      scratch_workspace_allocation_plan.phase_bytes == 1059584ull &&
      scratch_workspace_allocation_plan.auxiliary_bytes == 40960ull &&
      scratch_workspace_allocation_plan.total_bytes == 1100544ull &&
      scratch_workspace_allocation_plan.auxiliaries[0].name == "attention_qkv_scratch" &&
      scratch_workspace_allocation_plan.auxiliaries[0].phase == LayerPhase::Attention &&
      scratch_workspace_allocation_plan.auxiliaries[0].values == 8192 &&
      scratch_workspace_allocation_plan.auxiliaries[0].offset == 1059584ull &&
      scratch_workspace_allocation_plan.auxiliaries[0].bytes == 32768ull &&
      scratch_workspace_allocation_plan.auxiliaries[1].name == "ssm_state_scratch" &&
      scratch_workspace_allocation_plan.auxiliaries[1].phase == LayerPhase::Ssm &&
      scratch_workspace_allocation_plan.auxiliaries[1].values == 2048 &&
      scratch_workspace_allocation_plan.auxiliaries[1].offset == 1092352ull &&
      scratch_workspace_allocation_plan.auxiliaries[1].bytes == 8192ull &&
      attention_scratch.data == reinterpret_cast<float*>(
          scratch_runtime_workspace.data() + 1059584ull) &&
      ssm_scratch.data == reinterpret_cast<float*>(
          scratch_runtime_workspace.data() + 1092352ull) &&
      scratch_workspace_usage.auxiliaries.size() == 2 &&
      scratch_workspace_usage.auxiliary_high_water_bytes == 40960ull &&
      scratch_workspace_usage.total_high_water_bytes == 1100544ull &&
      scratch_workspace_usage.total_slack_bytes == 0;
  cases.push_back({"phase_scratch_workspace_passed",
                   phase_scratch_workspace_passed,
                   "core executor generic phase scratch workspace"});

  const auto ssm_aux_workspaces =
      rtxllm::build_ssm_recurrent_aux_workspace_descriptors(
          workspace_plan,
          true);
  const auto ssm_workspace_allocation_plan =
      rtxllm::build_layer_executor_workspace_allocation_plan(
          workspace_plan,
          ssm_aux_workspaces,
          sizeof(float),
          256);
  std::vector<std::uint8_t> ssm_runtime_workspace(
      ssm_workspace_allocation_plan.total_bytes,
      0);
  const auto ssm_workspace_runtime =
      rtxllm::bind_layer_executor_workspace_runtime(
          workspace_plan,
          ssm_workspace_allocation_plan,
          ssm_runtime_workspace.data());
  const auto& ssm_state =
      rtxllm::find_layer_aux_workspace_runtime(
          ssm_workspace_runtime,
          "ssm_recurrent_state");
  const bool ssm_recurrent_workspace_passed =
      ssm_aux_workspaces.size() == 1 &&
      ssm_workspace_allocation_plan.auxiliaries.size() == 1 &&
      ssm_workspace_allocation_plan.phase_bytes == 1059584ull &&
      ssm_workspace_allocation_plan.auxiliary_bytes == 993280ull &&
      ssm_workspace_allocation_plan.total_bytes == 2052864ull &&
      ssm_workspace_allocation_plan.auxiliaries[0].name == "ssm_recurrent_state" &&
      ssm_workspace_allocation_plan.auxiliaries[0].phase == LayerPhase::Ssm &&
      ssm_workspace_allocation_plan.auxiliaries[0].values == 248320 &&
      ssm_workspace_allocation_plan.auxiliaries[0].offset == 1059584ull &&
      ssm_workspace_allocation_plan.auxiliaries[0].bytes == 993280ull &&
      ssm_state.data == reinterpret_cast<float*>(
          ssm_runtime_workspace.data() + 1059584ull) &&
      ssm_state.values == 248320 &&
      ssm_state.bytes == 993280ull;
  cases.push_back({"ssm_recurrent_workspace_passed",
                   ssm_recurrent_workspace_passed,
                   "core executor SSM recurrent state workspace"});

  const auto ssm_scan_aux_workspaces =
      rtxllm::build_ssm_recurrent_aux_workspace_descriptors(
          workspace_plan,
          true,
          true);
  const auto ssm_scan_workspace_allocation_plan =
      rtxllm::build_layer_executor_workspace_allocation_plan(
          workspace_plan,
          ssm_scan_aux_workspaces,
          sizeof(float),
          256);
  std::vector<std::uint8_t> ssm_scan_runtime_workspace(
      ssm_scan_workspace_allocation_plan.total_bytes,
      0);
  const auto ssm_scan_workspace_runtime =
      rtxllm::bind_layer_executor_workspace_runtime(
          workspace_plan,
          ssm_scan_workspace_allocation_plan,
          ssm_scan_runtime_workspace.data());
  const auto& ssm_scan_state =
      rtxllm::find_layer_aux_workspace_runtime(
          ssm_scan_workspace_runtime,
          "ssm_recurrent_state");
  const auto& ssm_scan_scratch =
      rtxllm::find_layer_aux_workspace_runtime(
          ssm_scan_workspace_runtime,
          "ssm_scan_scratch");
  const bool ssm_scan_workspace_passed =
      ssm_scan_aux_workspaces.size() == 2 &&
      ssm_scan_workspace_allocation_plan.auxiliaries.size() == 2 &&
      ssm_scan_workspace_allocation_plan.phase_bytes == 1059584ull &&
      ssm_scan_workspace_allocation_plan.auxiliary_bytes == 1001472ull &&
      ssm_scan_workspace_allocation_plan.total_bytes == 2061056ull &&
      ssm_scan_workspace_allocation_plan.auxiliaries[0].name ==
          "ssm_recurrent_state" &&
      ssm_scan_workspace_allocation_plan.auxiliaries[0].phase ==
          LayerPhase::Ssm &&
      ssm_scan_workspace_allocation_plan.auxiliaries[0].values == 248320 &&
      ssm_scan_workspace_allocation_plan.auxiliaries[0].offset == 1059584ull &&
      ssm_scan_workspace_allocation_plan.auxiliaries[0].bytes == 993280ull &&
      ssm_scan_workspace_allocation_plan.auxiliaries[1].name ==
          "ssm_scan_scratch" &&
      ssm_scan_workspace_allocation_plan.auxiliaries[1].phase ==
          LayerPhase::Ssm &&
      ssm_scan_workspace_allocation_plan.auxiliaries[1].values == 2048 &&
      ssm_scan_workspace_allocation_plan.auxiliaries[1].offset == 2052864ull &&
      ssm_scan_workspace_allocation_plan.auxiliaries[1].bytes == 8192ull &&
      ssm_scan_state.data == reinterpret_cast<float*>(
          ssm_scan_runtime_workspace.data() + 1059584ull) &&
      ssm_scan_scratch.data == reinterpret_cast<float*>(
          ssm_scan_runtime_workspace.data() + 2052864ull) &&
      ssm_scan_scratch.values == 2048 &&
      ssm_scan_scratch.bytes == 8192ull;
  cases.push_back({"ssm_scan_workspace_passed",
                   ssm_scan_workspace_passed,
                   "core executor SSM scan scratch workspace"});

  const std::vector<rtxllm::CudaGraphBucketDescriptor> graph_bucket_descriptors = {
      {"decode_b1_q1",
       1,
       1,
       static_cast<std::uint32_t>(chained_execution.phase_count),
       static_cast<std::uint32_t>(chained_execution.total_kernels_per_replay),
       workspace_usage.total_high_water_bytes},
      {"prefill_b1_q4",
       1,
       4,
       static_cast<std::uint32_t>(chained_execution.phase_count),
       static_cast<std::uint32_t>(chained_execution.total_kernels_per_replay + 6),
       scratch_workspace_usage.total_high_water_bytes},
  };
  const auto graph_bucket_plan =
      rtxllm::build_cuda_graph_bucket_plan(graph_bucket_descriptors, 256);
  const bool graph_bucket_plan_passed =
      graph_bucket_plan.buckets.size() == 2 &&
      graph_bucket_plan.buckets[0].descriptor.name == "decode_b1_q1" &&
      graph_bucket_plan.buckets[0].offset == 0 &&
      graph_bucket_plan.buckets[0].metadata_bytes == 172544ull &&
      graph_bucket_plan.buckets[0].workspace_guard_bytes == 16640ull &&
      graph_bucket_plan.buckets[0].total_bytes == 189184ull &&
      graph_bucket_plan.buckets[1].descriptor.name == "prefill_b1_q4" &&
      graph_bucket_plan.buckets[1].offset == 189184ull &&
      graph_bucket_plan.buckets[1].metadata_bytes == 198656ull &&
      graph_bucket_plan.buckets[1].workspace_guard_bytes == 17408ull &&
      graph_bucket_plan.buckets[1].total_bytes == 216064ull &&
      graph_bucket_plan.total_bytes == 405248ull &&
      graph_bucket_plan.max_bucket_bytes == 216064ull;
  cases.push_back({"graph_bucket_plan_passed",
                   graph_bucket_plan_passed,
                   "core CUDA graph bucket admission estimate"});

  bool workspace_empty_stage_rejected = false;
  try {
    auto bad_workspaces = valid_stage_workspaces;
    bad_workspaces[0].rows = 0;
    (void)rtxllm::build_layer_executor_workspace_plan(
        chained_execution,
        bad_workspaces);
  } catch (const std::runtime_error&) {
    workspace_empty_stage_rejected = true;
  }
  cases.push_back({"workspace_empty_stage_rejected", workspace_empty_stage_rejected,
                   "phase-local workspace rejected empty stage"});

  rtxllm::RequestEstimate admitted_request;
  admitted_request.weight_bytes = 1024;
  admitted_request.workspace_bytes = 256;
  admitted_request.kv_cache_bytes = 512;
  admitted_request.dma_bytes = 128;
  admitted_request.graph_bytes = 64;
  admitted_request.workspace_high_water_bytes = 192;
  const auto admitted_decision = rtxllm::decide_admission(
      rtxllm::ResidencyPolicy::Strict,
      rtxllm::MemoryBudget{4096, 512},
      admitted_request);
  const bool admission_breakdown_passed =
      admitted_decision.admit &&
      !admitted_decision.host_spill_allowed &&
      admitted_decision.required_bytes == 1984 &&
      admitted_decision.usable_bytes == 3584 &&
      admitted_decision.breakdown.weight_bytes == 1024 &&
      admitted_decision.breakdown.total_kv_bytes == 512 &&
      admitted_decision.breakdown.workspace_bytes == 256 &&
      admitted_decision.breakdown.workspace_high_water_bytes == 192 &&
      admitted_decision.breakdown.workspace_slack_bytes == 64 &&
      admitted_decision.breakdown.dma_bytes == 128 &&
      admitted_decision.breakdown.graph_bytes == 64 &&
      admitted_decision.breakdown.over_budget_bytes == 0;
  cases.push_back({"admission_breakdown_passed",
                   admission_breakdown_passed,
                   "strict admission memory breakdown"});

  const auto rejected_decision = rtxllm::decide_admission(
      rtxllm::ResidencyPolicy::Strict,
      rtxllm::MemoryBudget{1500, 0},
      admitted_request);
  const bool admission_over_budget_passed =
      !rejected_decision.admit &&
      rejected_decision.required_bytes == 1984 &&
      rejected_decision.usable_bytes == 1500 &&
      rejected_decision.breakdown.over_budget_bytes == 484;
  cases.push_back({"admission_over_budget_passed",
                   admission_over_budget_passed,
                   "strict admission over-budget reporting"});

  std::vector<std::size_t> replayed_stages;
  std::vector<std::size_t> replayed_feedbacks;
  const auto replay_counters = rtxllm::replay_layer_execution_plan(
      chained_execution,
      valid_stages.size(),
      [&](std::size_t index) {
        replayed_stages.push_back(index);
      },
      [&](std::size_t index) {
        replayed_feedbacks.push_back(index);
      });
  std::vector<std::size_t> expected_indices;
  for (std::size_t index = 0; index < valid_stages.size(); ++index) {
    expected_indices.push_back(index);
  }
  const bool executor_replay_passed =
      replay_counters.phase_callbacks == 4 &&
      replay_counters.attention_phase_callbacks == 1 &&
      replay_counters.ffn_phase_callbacks == 1 &&
      replay_counters.ssm_phase_callbacks == 1 &&
      replay_counters.output_phase_callbacks == 1 &&
      replay_counters.stage_callbacks == 12 &&
      replay_counters.activation_feedback_callbacks == 12 &&
      replay_counters.total_callbacks == 24 &&
      replayed_stages == expected_indices &&
      replayed_feedbacks == expected_indices;
  cases.push_back(
      {"executor_replay_passed", executor_replay_passed, "core executor full replay"});

  std::vector<std::size_t> ffn_stage_indices;
  std::vector<std::size_t> ffn_feedback_indices;
  const auto ffn_replay_counters = rtxllm::replay_layer_phase(
      chained_execution.phases[1],
      valid_stages.size(),
      [&](std::size_t index) {
        ffn_stage_indices.push_back(index);
      },
      [&](std::size_t index) {
        ffn_feedback_indices.push_back(index);
      });
  const std::vector<std::size_t> expected_ffn_indices = {2, 3, 4, 5, 6, 7};
  const bool executor_phase_replay_passed =
      ffn_replay_counters.phase_callbacks == 1 &&
      ffn_replay_counters.attention_phase_callbacks == 0 &&
      ffn_replay_counters.ffn_phase_callbacks == 1 &&
      ffn_replay_counters.ssm_phase_callbacks == 0 &&
      ffn_replay_counters.output_phase_callbacks == 0 &&
      ffn_replay_counters.stage_callbacks == 6 &&
      ffn_replay_counters.activation_feedback_callbacks == 6 &&
      ffn_replay_counters.total_callbacks == 12 &&
      ffn_stage_indices == expected_ffn_indices &&
      ffn_feedback_indices == expected_ffn_indices;
  cases.push_back({"executor_phase_replay_passed", executor_phase_replay_passed,
                   "core executor phase replay"});

  bool executor_bounds_rejected = false;
  try {
    rtxllm::LayerPhaseExecution bad_phase;
    bad_phase.phase = LayerPhase::Output;
    bad_phase.begin = valid_stages.size() - 1;
    bad_phase.count = 2;
    bad_phase.matvec_kernels_per_replay = 2;
    bad_phase.activation_feedbacks_per_replay = 2;
    bad_phase.total_kernels_per_replay = 4;
    (void)rtxllm::replay_layer_phase(
        bad_phase,
        valid_stages.size(),
        [](std::size_t) {},
        [](std::size_t) {});
  } catch (const std::runtime_error&) {
    executor_bounds_rejected = true;
  }
  cases.push_back({"executor_bounds_rejected", executor_bounds_rejected,
                   "core executor rejected out-of-range phase"});

  std::vector<std::string> named_phase_order;
  const auto replay_named_phase =
      [&](const rtxllm::LayerPhaseExecution& phase, std::string name) {
        named_phase_order.push_back(std::move(name));
        return rtxllm::replay_layer_phase(
            phase,
            valid_stages.size(),
            [](std::size_t) {},
            [](std::size_t) {});
      };
  const auto named_replay_counters = rtxllm::replay_layer_execution_plan_by_phase(
      chained_execution,
      valid_stages.size(),
      [&](const rtxllm::LayerPhaseExecution& phase) {
        return replay_named_phase(phase, "attention");
      },
      [&](const rtxllm::LayerPhaseExecution& phase) {
        return replay_named_phase(phase, "ffn");
      },
      [&](const rtxllm::LayerPhaseExecution& phase) {
        return replay_named_phase(phase, "ssm");
      },
      [&](const rtxllm::LayerPhaseExecution& phase) {
        return replay_named_phase(phase, "output");
      });
  const std::vector<std::string> expected_phase_order = {
      "attention",
      "ffn",
      "ssm",
      "output",
  };
  const bool executor_named_phase_dispatch_passed =
      named_replay_counters.phase_callbacks == 4 &&
      named_replay_counters.attention_phase_callbacks == 1 &&
      named_replay_counters.ffn_phase_callbacks == 1 &&
      named_replay_counters.ssm_phase_callbacks == 1 &&
      named_replay_counters.output_phase_callbacks == 1 &&
      named_replay_counters.stage_callbacks == 12 &&
      named_replay_counters.activation_feedback_callbacks == 12 &&
      named_replay_counters.total_callbacks == 24 &&
      named_phase_order == expected_phase_order;
  cases.push_back({"executor_named_phase_dispatch_passed",
                   executor_named_phase_dispatch_passed,
                   "core executor named phase dispatch"});

  std::string detail;
  const bool empty_rejected =
      rejected_with({}, "layer phase plan requires at least one stage", &detail);
  cases.push_back({"empty_rejected", empty_rejected, detail});

  const bool unknown_role_rejected = rejected_with(
      {LayerStageDescriptor{"mystery.weight", true}},
      "layer phase plan contains unknown role-plan roles", &detail);
  cases.push_back({"unknown_role_rejected", unknown_role_rejected, detail});

  const bool unknown_phase_rejected = rejected_with(
      {LayerStageDescriptor{"global_mystery.weight", false}},
      "layer phase plan contains unknown execution phases", &detail);
  cases.push_back({"unknown_phase_rejected", unknown_phase_rejected, detail});

  const bool phase_order_rejected =
      rejected_with({LayerStageDescriptor{"ffn_down_exps.weight", true},
                     LayerStageDescriptor{"attn_qkv.weight", true},
                     LayerStageDescriptor{"ssm_alpha.weight", true}},
                    "layer phase plan phase order mismatch", &detail);
  cases.push_back({"phase_order_rejected", phase_order_rejected, detail});

  const bool segment_count_rejected =
      rejected_with({LayerStageDescriptor{"attn_qkv.weight", true},
                     LayerStageDescriptor{"ffn_down_exps.weight", true},
                     LayerStageDescriptor{"ssm_alpha.weight", true},
                     LayerStageDescriptor{"attn_gate.weight", true}},
                    "layer phase plan segment count mismatch", &detail);
  cases.push_back({"segment_count_rejected", segment_count_rejected, detail});

  std::vector<LayerPhaseSegment> empty_segment = valid_segments;
  empty_segment[0].count = 0;
  const auto empty_segment_validation =
      rtxllm::validate_layer_phase_plan(valid_stages, empty_segment);
  const bool empty_segment_rejected =
      !empty_segment_validation.passed &&
      empty_segment_validation.message == "layer phase plan contains an empty phase segment";
  cases.push_back(
      {"empty_segment_rejected", empty_segment_rejected, empty_segment_validation.message});

  const auto passed_count =
      std::count_if(cases.begin(), cases.end(), [](const CaseResult& item) {
        return item.passed;
      });
  const bool reference_passed = passed_count == static_cast<int>(cases.size());

  std::cout << "{\n";
  std::cout << "  \"path\": \"native_layer_plan_selftest\",\n";
  std::cout << "  \"label\": \"layer_plan_contract\",\n";
  std::cout << "  \"reference_passed\": " << json_bool(reference_passed) << ",\n";
  std::cout << "  \"cases_total\": " << cases.size() << ",\n";
  std::cout << "  \"cases_passed\": " << passed_count << ",\n";
  for (const auto& item : cases) {
    std::cout << "  \"" << item.name << "\": " << json_bool(item.passed) << ",\n";
  }
  std::cout << "  \"cases\": [\n";
  for (std::size_t index = 0; index < cases.size(); ++index) {
    const auto& item = cases[index];
    std::cout << "    {\"name\": \"" << item.name << "\", \"passed\": "
              << json_bool(item.passed) << ", \"detail\": \"" << item.detail << "\"}";
    if (index + 1 != cases.size()) {
      std::cout << ",";
    }
    std::cout << "\n";
  }
  std::cout << "  ]\n";
  std::cout << "}\n";

  return reference_passed ? 0 : 1;
}
