#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

extern "C" cudaError_t rtxllm_launch_synthetic_decode(
    const std::uint8_t* weights,
    float* kv,
    std::uint32_t* token,
    std::size_t weight_bytes,
    std::size_t kv_words,
    std::uint32_t step,
    int touch_stride,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_synthetic_decode_stateful(
    const std::uint8_t* weights,
    float* kv,
    std::uint32_t* token,
    std::size_t weight_bytes,
    std::size_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    int touch_stride,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_activation_feedback(
    const float* logits,
    float* activation,
    const std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t stage,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_rmsnorm_feedback(
    const float* logits,
    float* activation,
    std::uint32_t rows,
    std::uint32_t cols,
    float epsilon,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_silu_rmsnorm_feedback(
    const float* logits,
    float* activation,
    std::uint32_t rows,
    std::uint32_t cols,
    float epsilon,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_rmsnorm_feedback_with_silu_cache(
    const float* logits,
    float* activation,
    float* silu_cache,
    std::uint32_t rows,
    std::uint32_t cols,
    float epsilon,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_gated_silu_rmsnorm_feedback(
    const float* logits,
    const float* silu_cache,
    float* activation,
    std::uint32_t rows,
    std::uint32_t cache_cols,
    std::uint32_t cols,
    float epsilon,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_rmsnorm_silu_cache(
    const float* logits,
    float* silu_cache,
    std::uint32_t rows,
    std::uint32_t cache_cols,
    float epsilon,
    std::uint32_t accumulate,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_rmsnorm_cache(
    const float* logits,
    float* cache,
    std::uint32_t rows,
    std::uint32_t cache_cols,
    float epsilon,
    std::uint32_t accumulate,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_rmsnorm_cache_and_gated_product(
    const float* logits,
    const float* gate_cache,
    float* up_cache,
    float* activation,
    std::uint32_t rows,
    std::uint32_t cache_cols,
    std::uint32_t output_cols,
    float epsilon,
    std::uint32_t accumulate,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_phase_scratch_digest(
    const float* activation,
    const float* logits,
    float* scratch,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t scratch_values,
    std::uint32_t phase_id,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_attention_qkv_scratch(
    const float* logits,
    float* scratch,
    float* activation,
    std::uint32_t rows,
    std::uint32_t scratch_values,
    std::uint32_t activation_cols,
    float residual_scale,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_attention_qkv_reduce(
    float* scratch,
    float* activation,
    std::uint32_t rows,
    std::uint32_t scratch_values,
    std::uint32_t activation_cols,
    float reduce_scale,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_attention_qkv_window(
    float* scratch,
    float* activation,
    std::uint32_t rows,
    std::uint32_t scratch_values,
    std::uint32_t activation_cols,
    std::uint32_t window_size,
    float softmax_scale,
    float residual_scale,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_attention_qkv_head_window(
    float* scratch,
    float* activation,
    std::uint32_t rows,
    std::uint32_t scratch_values,
    std::uint32_t activation_cols,
    std::uint32_t head_count,
    std::uint32_t window_size,
    float softmax_scale,
    float residual_scale,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_attention_qkv_head_dim_window(
    float* scratch,
    float* activation,
    std::uint32_t rows,
    std::uint32_t scratch_values,
    std::uint32_t activation_cols,
    std::uint32_t head_count,
    std::uint32_t head_dim,
    std::uint32_t window_size,
    float softmax_scale,
    float residual_scale,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_attention_qkv_head_tile_window(
    float* scratch,
    float* activation,
    std::uint32_t rows,
    std::uint32_t scratch_values,
    std::uint32_t activation_cols,
    std::uint32_t head_count,
    std::uint32_t head_dim,
    std::uint32_t window_size,
    float softmax_scale,
    float residual_scale,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_attention_qkv_head_group_window(
    float* scratch,
    float* activation,
    std::uint32_t rows,
    std::uint32_t scratch_values,
    std::uint32_t activation_cols,
    std::uint32_t head_count,
    std::uint32_t head_dim,
    std::uint32_t window_size,
    std::uint32_t contexts_per_block,
    float softmax_scale,
    float residual_scale,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_attention_qkv_head_group_rope_window(
    float* scratch,
    float* activation,
    std::uint32_t rows,
    std::uint32_t scratch_values,
    std::uint32_t activation_cols,
    std::uint32_t head_count,
    std::uint32_t head_dim,
    std::uint32_t window_size,
    std::uint32_t contexts_per_block,
    float rope_theta_scale,
    float softmax_scale,
    float residual_scale,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_attention_qkv_head_group_fused(
    const float* logits,
    float* activation,
    std::uint32_t rows,
    std::uint32_t activation_cols,
    std::uint32_t head_count,
    std::uint32_t head_dim,
    std::uint32_t window_size,
    std::uint32_t contexts_per_block,
    float scratch_residual_scale,
    float softmax_scale,
    float residual_scale,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_ssm_recurrent_state(
    const float* logits,
    float* state,
    float* activation,
    std::uint32_t rows,
    std::uint32_t state_values,
    std::uint32_t activation_cols,
    float decay,
    float logit_scale,
    float activation_scale,
    float output_scale,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_ssm_scan_scratch(
    const float* logits,
    float* state,
    float* scratch,
    float* activation,
    std::uint32_t rows,
    std::uint32_t state_values,
    std::uint32_t scratch_values,
    std::uint32_t activation_cols,
    float decay,
    float logit_scale,
    float scratch_scale,
    float activation_scale,
    float output_scale,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_ssm_selective_scan(
    const float* logits,
    float* state,
    float* scratch,
    float* activation,
    std::uint32_t rows,
    std::uint32_t state_values,
    std::uint32_t scratch_values,
    std::uint32_t activation_cols,
    float decay,
    float logit_scale,
    float scratch_scale,
    float activation_scale,
    float gate_logit_scale,
    float gate_activation_scale,
    float gate_scratch_scale,
    float output_scale,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_ssm_source_parameter_cache(
    const float* logits,
    const float* activation,
    float* scratch,
    std::uint32_t rows,
    std::uint32_t scratch_values,
    std::uint32_t state_values,
    std::uint32_t activation_cols,
    std::uint32_t parameter_slot,
    float logit_scale,
    float activation_scale,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_ssm_rmsnorm_feedback_parameter_cache(
    const float* logits,
    float* activation,
    float* scratch,
    std::uint32_t rows,
    std::uint32_t activation_cols,
    std::uint32_t scratch_values,
    std::uint32_t state_values,
    std::uint32_t parameter_slot,
    float epsilon,
    float logit_scale,
    float activation_scale,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_ssm_source_parameterized_scan(
    const float* logits,
    float* state,
    float* scratch,
    float* activation,
    std::uint32_t rows,
    std::uint32_t state_values,
    std::uint32_t scratch_values,
    std::uint32_t activation_cols,
    float decay,
    float logit_scale,
    float scratch_scale,
    float activation_scale,
    float alpha_scale,
    float beta_scale,
    float gate_logit_scale,
    float gate_activation_scale,
    float output_scale,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_output_token_sample(
    const float* logits,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t vocab_size,
    std::uint32_t token_offset,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_iq2s_probe(
    const std::uint8_t* iq2s_blocks,
    std::uint64_t block_count,
    std::uint64_t* output,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_iq3s_probe(
    const std::uint8_t* iq3s_blocks,
    std::uint64_t block_count,
    std::uint64_t* output,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_q5k_probe(
    const std::uint8_t* q5k_blocks,
    std::uint64_t block_count,
    std::uint64_t* output,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_q5k_matvec_decode_probe(
    const std::uint8_t* q5k_blocks,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t rows_limit,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_iq2s_matvec_probe(
    const std::uint8_t* iq2s_blocks,
    const float* activation,
    float* logits,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t rows_limit,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_iq2s_matvec_decode_probe(
    const std::uint8_t* iq2s_blocks,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t rows_limit,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_iq3s_matvec_decode_probe(
    const std::uint8_t* iq3s_blocks,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t rows_limit,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_q4_matvec_decode(
    const std::uint8_t* packed_weights,
    const float* scales,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_q4_matvec_decode_vec4x4(
    const std::uint8_t* packed_weights,
    const float* scales,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_q4k_matvec_decode(
    const std::uint8_t* q4k_blocks,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_q4k_matvec_decode_shared(
    const std::uint8_t* q4k_blocks,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_q4k_matvec_decode_warp(
    const std::uint8_t* q4k_blocks,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_q4k_matvec_decode_warp_broadcast(
    const std::uint8_t* q4k_blocks,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_q4k_matvec_decode_warp_broadcast_vec4(
    const std::uint8_t* q4k_blocks,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_q4k_matvec_decode_warp_broadcast_vec4x2(
    const std::uint8_t* q4k_blocks,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_q4k_matvec_decode_warp_broadcast_vec4x4(
    const std::uint8_t* q4k_blocks,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_q4k_matvec_decode_predecoded_vec4x4(
    const std::uint8_t* q4k_blocks,
    const float* q4k_scale_min_pairs,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_q4k_matvec_decode_split_predecoded_vec4x4(
    const std::uint8_t* q4k_payload,
    const float* q4k_scale_min_pairs,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_q4k_matvec_decode_split_half_vec4x4(
    const std::uint8_t* q4k_payload,
    const std::uint16_t* q4k_scale_min_half_pairs,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_q4k_matvec_decode_split_compact_vec4x4(
    const std::uint8_t* q4k_payload,
    const std::uint8_t* q4k_compact_meta,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream);

extern "C" cudaError_t rtxllm_launch_q4k_matvec_decode_split_native_vec4x4(
    const std::uint8_t* q4k_payload,
    const std::uint8_t* q4k_native_meta,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    cudaStream_t stream);
