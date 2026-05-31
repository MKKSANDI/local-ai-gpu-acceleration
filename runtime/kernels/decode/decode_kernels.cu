#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr std::uint32_t kIq2SBlockBytes = 82;
constexpr std::uint32_t kIq2SScaleOffset = 0;
constexpr std::uint32_t kIq2SQsOffset = 2;
constexpr std::uint32_t kIq2SQsBytes = 64;
constexpr std::uint32_t kIq2SQhOffset = kIq2SQsOffset + kIq2SQsBytes;
constexpr std::uint32_t kIq2SQhBytes = 8;
constexpr std::uint32_t kIq2SScalesOffset = kIq2SQhOffset + kIq2SQhBytes;
constexpr std::uint32_t kIq2SScalesBytes = 8;
constexpr std::uint32_t kIq3SBlockBytes = 110;
constexpr std::uint32_t kIq3SScaleOffset = 0;
constexpr std::uint32_t kIq3SQsOffset = 2;
constexpr std::uint32_t kIq3SQsBytes = 64;
constexpr std::uint32_t kIq3SQhOffset = kIq3SQsOffset + kIq3SQsBytes;
constexpr std::uint32_t kIq3SQhBytes = 8;
constexpr std::uint32_t kIq3SSignsOffset = kIq3SQhOffset + kIq3SQhBytes;
constexpr std::uint32_t kIq3SSignsBytes = 32;
constexpr std::uint32_t kIq3SScalesOffset = kIq3SSignsOffset + kIq3SSignsBytes;
constexpr std::uint32_t kIq3SScalesBytes = 4;
constexpr std::uint32_t kQ5KBlockBytes = 176;
constexpr std::uint32_t kQ5KDOffset = 0;
constexpr std::uint32_t kQ5KDMinOffset = 2;
constexpr std::uint32_t kQ5KScalesOffset = 4;
constexpr std::uint32_t kQ5KScalesBytes = 12;
constexpr std::uint32_t kQ5KQhOffset = kQ5KScalesOffset + kQ5KScalesBytes;
constexpr std::uint32_t kQ5KQhBytes = 32;
constexpr std::uint32_t kQ5KQsOffset = kQ5KQhOffset + kQ5KQhBytes;
constexpr std::uint32_t kQ5KQsBytes = 128;

__device__ __constant__ std::uint16_t kIq2SKGridDevice[1024] = {
#include "runtime/kernels/decode/iq2s_kgrid_values.inl"
};

__device__ __constant__ std::uint32_t kIq3SGridDevice[512] = {
#include "runtime/kernels/decode/iq3s_grid_values.inl"
};

__device__ float fp16_bits_to_float(std::uint16_t bits);

__global__ void synthetic_decode_kernel(
    const std::uint8_t* weights,
    float* kv,
    std::uint32_t* token,
    std::size_t weight_bytes,
    std::size_t kv_words,
    std::uint32_t step,
    int touch_stride) {
  const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = gridDim.x * blockDim.x * static_cast<std::size_t>(touch_stride);
  std::uint32_t acc = 0;
  for (std::size_t i = tid; i < weight_bytes; i += stride) {
    acc += weights[i];
  }

  if (tid < kv_words) {
    const std::size_t pos = (tid + step * 1315423911u) % kv_words;
    kv[pos] = kv[pos] + static_cast<float>(acc & 0xffu) * 0.000001f;
  }

  if (tid == 0) {
    token[0] = (step + acc) % 32000u;
  }
}

__global__ void synthetic_decode_stateful_kernel(
    const std::uint8_t* weights,
    float* kv,
    std::uint32_t* token,
    std::size_t weight_bytes,
    std::size_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    int touch_stride) {
  const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t stride = gridDim.x * blockDim.x * static_cast<std::size_t>(touch_stride);
  std::uint32_t acc = 0;
  for (std::size_t i = tid; i < weight_bytes; i += stride) {
    acc += weights[i];
  }

  if (tid < active_pages && kv_words > 0) {
    const std::size_t page_base = tid * static_cast<std::size_t>(page_words);
    const std::size_t offset = (token[0] + static_cast<std::uint32_t>(tid)) % page_words;
    const std::size_t pos = (page_base + offset) % kv_words;
    kv[pos] = kv[pos] + static_cast<float>(acc & 0xffu) * 0.000001f;
  }

  if (tid == 0) {
    token[0] = (token[0] + 1u + acc) % 32000u;
  }
}

__global__ void activation_feedback_kernel(
    const float* logits,
    float* activation,
    const std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t stage) {
  const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols || rows == 0u) {
    return;
  }
  const std::uint32_t src =
      (col * 131u + token[0] * 17u + stage * 977u) % rows;
  const float value = logits[src];
  const float squashed = value / (1.0f + fabsf(value));
  const float bias = static_cast<float>((col + stage) & 7u) * 0.0009765625f;
  activation[col] = squashed + bias;
}

__global__ void rmsnorm_feedback_kernel(
    const float* logits,
    float* activation,
    std::uint32_t rows,
    std::uint32_t cols,
    float epsilon) {
  const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols || rows == 0u) {
    return;
  }

  float sum_sq = 0.0f;
  for (std::uint32_t row = 0; row < rows; ++row) {
    const float value = logits[row];
    sum_sq += value * value;
  }
  const float mean_sq = sum_sq / static_cast<float>(rows);
  const float inv_rms = rsqrtf(mean_sq + epsilon);
  activation[col] = logits[col % rows] * inv_rms;
}

__global__ void silu_rmsnorm_feedback_kernel(
    const float* logits,
    float* activation,
    std::uint32_t rows,
    std::uint32_t cols,
    float epsilon) {
  const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols || rows == 0u) {
    return;
  }

  float sum_sq = 0.0f;
  for (std::uint32_t row = 0; row < rows; ++row) {
    const float value = logits[row];
    sum_sq += value * value;
  }
  const float mean_sq = sum_sq / static_cast<float>(rows);
  const float inv_rms = rsqrtf(mean_sq + epsilon);
  const float normalized = logits[col % rows] * inv_rms;
  const float sigmoid = 1.0f / (1.0f + expf(-normalized));
  activation[col] = normalized * sigmoid;
}

__global__ void rmsnorm_feedback_with_silu_cache_kernel(
    const float* logits,
    float* activation,
    float* silu_cache,
    std::uint32_t rows,
    std::uint32_t cols,
    float epsilon) {
  const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols || rows == 0u) {
    return;
  }

  float sum_sq = 0.0f;
  for (std::uint32_t row = 0; row < rows; ++row) {
    const float value = logits[row];
    sum_sq += value * value;
  }
  const float mean_sq = sum_sq / static_cast<float>(rows);
  const float inv_rms = rsqrtf(mean_sq + epsilon);
  const float normalized = logits[col % rows] * inv_rms;
  const float sigmoid = 1.0f / (1.0f + expf(-normalized));
  activation[col] = normalized;
  silu_cache[col] = normalized * sigmoid;
}

__global__ void gated_silu_rmsnorm_feedback_kernel(
    const float* logits,
    const float* silu_cache,
    float* activation,
    std::uint32_t rows,
    std::uint32_t cache_cols,
    std::uint32_t cols,
    float epsilon) {
  const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols || rows == 0u || cache_cols == 0u) {
    return;
  }

  float sum_sq = 0.0f;
  for (std::uint32_t row = 0; row < rows; ++row) {
    const float value = logits[row];
    sum_sq += value * value;
  }
  const float mean_sq = sum_sq / static_cast<float>(rows);
  const float inv_rms = rsqrtf(mean_sq + epsilon);
  const float normalized_up = logits[col % rows] * inv_rms;
  activation[col] = silu_cache[col % cache_cols] * normalized_up;
}

__global__ void rmsnorm_silu_cache_kernel(
    const float* logits,
    float* silu_cache,
    std::uint32_t rows,
    std::uint32_t cache_cols,
    float epsilon,
    std::uint32_t accumulate) {
  const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cache_cols || rows == 0u) {
    return;
  }

  float sum_sq = 0.0f;
  for (std::uint32_t row = 0; row < rows; ++row) {
    const float value = logits[row];
    sum_sq += value * value;
  }
  const float mean_sq = sum_sq / static_cast<float>(rows);
  const float inv_rms = rsqrtf(mean_sq + epsilon);
  const float normalized = logits[col % rows] * inv_rms;
  const float sigmoid = 1.0f / (1.0f + expf(-normalized));
  const float value = normalized * sigmoid;
  silu_cache[col] = accumulate != 0u ? silu_cache[col] + value : value;
}

__global__ void rmsnorm_cache_kernel(
    const float* logits,
    float* cache,
    std::uint32_t rows,
    std::uint32_t cache_cols,
    float epsilon,
    std::uint32_t accumulate) {
  const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cache_cols || rows == 0u) {
    return;
  }

  float sum_sq = 0.0f;
  for (std::uint32_t row = 0; row < rows; ++row) {
    const float value = logits[row];
    sum_sq += value * value;
  }
  const float mean_sq = sum_sq / static_cast<float>(rows);
  const float inv_rms = rsqrtf(mean_sq + epsilon);
  const float value = logits[col % rows] * inv_rms;
  cache[col] = accumulate != 0u ? cache[col] + value : value;
}

__global__ void rmsnorm_cache_and_gated_product_kernel(
    const float* logits,
    const float* gate_cache,
    float* up_cache,
    float* activation,
    std::uint32_t rows,
    std::uint32_t cache_cols,
    std::uint32_t output_cols,
    float epsilon,
    std::uint32_t accumulate) {
  const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= output_cols || rows == 0u || cache_cols == 0u) {
    return;
  }

  float sum_sq = 0.0f;
  for (std::uint32_t row = 0; row < rows; ++row) {
    const float value = logits[row];
    sum_sq += value * value;
  }
  const float mean_sq = sum_sq / static_cast<float>(rows);
  const float inv_rms = rsqrtf(mean_sq + epsilon);
  const float value = logits[col % rows] * inv_rms;
  const float up_value = accumulate != 0u ? up_cache[col] + value : value;
  up_cache[col] = up_value;
  activation[col] = gate_cache[col] * up_value;
}

__global__ void phase_scratch_digest_kernel(
    const float* activation,
    const float* logits,
    float* scratch,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t scratch_values,
    std::uint32_t phase_id) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= scratch_values) {
    return;
  }
  const float logit =
      rows == 0u ? 0.0f : logits[(index * 131u + phase_id * 17u) % rows];
  const float active =
      cols == 0u ? 0.0f : activation[(index * 17u + phase_id * 131u) % cols];
  const float phase_bias = static_cast<float>(phase_id + 1u) * 0.000244140625f;
  scratch[index] = scratch[index] * 0.5f + logit * 0.03125f + active * 0.015625f + phase_bias;
}

__device__ float attention_qkv_projection(
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

__device__ void attention_rope_rotate_pair(
    float in_even,
    float in_odd,
    std::uint32_t context,
    std::uint32_t pair,
    float theta_scale,
    float* out_even,
    float* out_odd) {
  const float angle =
      static_cast<float>(context) * static_cast<float>(pair + 1u) *
      theta_scale;
  float sin_value = 0.0f;
  float cos_value = 1.0f;
  sincosf(angle, &sin_value, &cos_value);
  *out_even = in_even * cos_value - in_odd * sin_value;
  *out_odd = in_even * sin_value + in_odd * cos_value;
}

__global__ void attention_qkv_scratch_kernel(
    const float* logits,
    float* scratch,
    float* activation,
    std::uint32_t rows,
    std::uint32_t qkv_rows,
    std::uint32_t qkv_values,
    std::uint32_t activation_cols,
    float residual_scale) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < qkv_values) {
    const std::uint32_t row = index % qkv_rows;
    const std::uint32_t component = index / qkv_rows;
    const float logit = logits[row % rows];
    scratch[index] =
        scratch[index] * 0.25f +
        attention_qkv_projection(logit, 0.0f, component, row);
  }
  if (index < activation_cols) {
    const std::uint32_t row = index % qkv_rows;
    const float logit = logits[row % rows];
    const float active = activation[index];
    const float q = attention_qkv_projection(logit, active, 0u, row);
    const float k = attention_qkv_projection(logit, active, 1u, row);
    const float v = attention_qkv_projection(logit, active, 2u, row);
    activation[index] = active + (q * 0.0625f + k * 0.03125f + v * 0.015625f) *
        residual_scale;
  }
}

__global__ void attention_qkv_reduce_kernel(
    float* scratch,
    float* activation,
    std::uint32_t qkv_rows,
    std::uint32_t activation_cols,
    float reduce_scale) {
  const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= activation_cols || qkv_rows == 0u) {
    return;
  }
  const std::uint32_t row = col % qkv_rows;
  const std::uint32_t k_row = (row + ((col * 7u + 3u) % qkv_rows)) % qkv_rows;
  const std::uint32_t v_row = (row * 3u + col) % qkv_rows;
  const float q = scratch[row];
  const float k = scratch[qkv_rows + k_row];
  const float v = scratch[2u * qkv_rows + v_row];
  const float score = q * k;
  const float bounded_score = score / (1.0f + fabsf(score));
  const float residual =
      v * 0.0625f +
      q * 0.015625f +
      k * 0.0078125f +
      bounded_score * 0.03125f;
  activation[col] = activation[col] + residual * reduce_scale;
}

__global__ void attention_qkv_window_kernel(
    float* scratch,
    float* activation,
    std::uint32_t qkv_rows,
    std::uint32_t activation_cols,
    std::uint32_t window_size,
    float softmax_scale,
    float residual_scale) {
  const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= activation_cols || qkv_rows == 0u || window_size == 0u) {
    return;
  }
  const std::uint32_t row = col % qkv_rows;
  const std::uint32_t window =
      window_size < qkv_rows ? window_size : qkv_rows;
  const float q = scratch[row];
  float max_score = -CUDART_INF_F;
  for (std::uint32_t item = 0; item < window; ++item) {
    const std::uint32_t key_row =
        (row + ((col * 5u + item * 11u + 1u) % qkv_rows)) % qkv_rows;
    const float k = scratch[qkv_rows + key_row];
    const float score = q * k * softmax_scale;
    max_score = fmaxf(max_score, score);
  }
  float denom = 0.0f;
  float weighted_v = 0.0f;
  float weighted_score = 0.0f;
  for (std::uint32_t item = 0; item < window; ++item) {
    const std::uint32_t key_row =
        (row + ((col * 5u + item * 11u + 1u) % qkv_rows)) % qkv_rows;
    const std::uint32_t value_row = (key_row * 3u + col + item) % qkv_rows;
    const float k = scratch[qkv_rows + key_row];
    const float score = q * k * softmax_scale;
    const float weight = expf(score - max_score);
    denom += weight;
    weighted_v += weight * scratch[2u * qkv_rows + value_row];
    weighted_score += weight * score;
  }
  if (denom <= 0.0f) {
    return;
  }
  const float inv_denom = 1.0f / denom;
  const float context = weighted_v * inv_denom;
  const float mean_score = weighted_score * inv_denom;
  const float bounded_score = mean_score / (1.0f + fabsf(mean_score));
  const float residual =
      context * 0.078125f +
      q * 0.015625f +
      bounded_score * 0.015625f;
  activation[col] = activation[col] + residual * residual_scale;
}

__global__ void attention_qkv_head_window_kernel(
    float* scratch,
    float* activation,
    std::uint32_t qkv_rows,
    std::uint32_t activation_cols,
    std::uint32_t head_count,
    std::uint32_t window_size,
    float softmax_scale,
    float residual_scale) {
  const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= activation_cols || qkv_rows == 0u || head_count == 0u ||
      window_size == 0u) {
    return;
  }
  const std::uint32_t active_heads =
      head_count < qkv_rows ? head_count : qkv_rows;
  const std::uint32_t rows_per_head = qkv_rows / active_heads;
  if (rows_per_head == 0u) {
    return;
  }
  const std::uint32_t head = col % active_heads;
  const std::uint32_t local_row = (col / active_heads) % rows_per_head;
  const std::uint32_t head_base = head * rows_per_head;
  const std::uint32_t query_row = head_base + local_row;
  const std::uint32_t window =
      window_size < rows_per_head ? window_size : rows_per_head;
  const float q = scratch[query_row];
  float max_score = -CUDART_INF_F;
  for (std::uint32_t item = 0; item < window; ++item) {
    const std::uint32_t key_local =
        (local_row + rows_per_head - item) % rows_per_head;
    const std::uint32_t key_row = head_base + key_local;
    const float k = scratch[qkv_rows + key_row];
    const float score = q * k * softmax_scale;
    max_score = fmaxf(max_score, score);
  }
  float denom = 0.0f;
  float weighted_v = 0.0f;
  float weighted_score = 0.0f;
  for (std::uint32_t item = 0; item < window; ++item) {
    const std::uint32_t key_local =
        (local_row + rows_per_head - item) % rows_per_head;
    const std::uint32_t key_row = head_base + key_local;
    const float k = scratch[qkv_rows + key_row];
    const float score = q * k * softmax_scale;
    const float weight = expf(score - max_score);
    denom += weight;
    weighted_v += weight * scratch[2u * qkv_rows + key_row];
    weighted_score += weight * score;
  }
  if (denom <= 0.0f) {
    return;
  }
  const float inv_denom = 1.0f / denom;
  const float context = weighted_v * inv_denom;
  const float mean_score = weighted_score * inv_denom;
  const float bounded_score = mean_score / (1.0f + fabsf(mean_score));
  const float head_bias =
      static_cast<float>((head & 0x7u) + 1u) * 0.0009765625f;
  const float residual =
      context * 0.078125f +
      q * 0.015625f +
      bounded_score * 0.015625f +
      head_bias;
  activation[col] = activation[col] + residual * residual_scale;
}

__global__ void attention_qkv_head_dim_window_kernel(
    float* scratch,
    float* activation,
    std::uint32_t qkv_rows,
    std::uint32_t activation_cols,
    std::uint32_t head_count,
    std::uint32_t head_dim,
    std::uint32_t window_size,
    float softmax_scale,
    float residual_scale) {
  const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= activation_cols || qkv_rows == 0u || head_count == 0u ||
      head_dim == 0u || window_size == 0u) {
    return;
  }
  const std::uint32_t possible_heads = qkv_rows / head_dim;
  if (possible_heads == 0u) {
    return;
  }
  const std::uint32_t active_heads =
      head_count < possible_heads ? head_count : possible_heads;
  const std::uint32_t rows_per_head = qkv_rows / active_heads;
  const std::uint32_t contexts_per_head = rows_per_head / head_dim;
  if (contexts_per_head == 0u) {
    return;
  }
  const std::uint32_t head = col % active_heads;
  const std::uint32_t dim = (col / active_heads) % head_dim;
  const std::uint32_t query_context =
      (col / (active_heads * head_dim)) % contexts_per_head;
  const std::uint32_t head_base = head * rows_per_head;
  const std::uint32_t query_base = head_base + query_context * head_dim;
  const std::uint32_t query_row = query_base + dim;
  const std::uint32_t window =
      window_size < contexts_per_head ? window_size : contexts_per_head;
  float max_score = -CUDART_INF_F;
  for (std::uint32_t item = 0; item < window; ++item) {
    const std::uint32_t key_context =
        (query_context + contexts_per_head - item) % contexts_per_head;
    const std::uint32_t key_base = head_base + key_context * head_dim;
    float dot = 0.0f;
    for (std::uint32_t lane = 0; lane < head_dim; ++lane) {
      dot += scratch[query_base + lane] *
          scratch[qkv_rows + key_base + lane];
    }
    max_score = fmaxf(max_score, dot * softmax_scale);
  }
  float denom = 0.0f;
  float weighted_v = 0.0f;
  float weighted_score = 0.0f;
  for (std::uint32_t item = 0; item < window; ++item) {
    const std::uint32_t key_context =
        (query_context + contexts_per_head - item) % contexts_per_head;
    const std::uint32_t key_base = head_base + key_context * head_dim;
    float dot = 0.0f;
    for (std::uint32_t lane = 0; lane < head_dim; ++lane) {
      dot += scratch[query_base + lane] *
          scratch[qkv_rows + key_base + lane];
    }
    const float score = dot * softmax_scale;
    const float weight = expf(score - max_score);
    denom += weight;
    weighted_v += weight * scratch[2u * qkv_rows + key_base + dim];
    weighted_score += weight * score;
  }
  if (denom <= 0.0f) {
    return;
  }
  const float inv_denom = 1.0f / denom;
  const float context = weighted_v * inv_denom;
  const float mean_score = weighted_score * inv_denom;
  const float bounded_score = mean_score / (1.0f + fabsf(mean_score));
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
  activation[col] = activation[col] + residual * residual_scale;
}

__global__ void attention_qkv_head_tile_window_kernel(
    float* scratch,
    float* activation,
    std::uint32_t qkv_rows,
    std::uint32_t activation_cols,
    std::uint32_t head_count,
    std::uint32_t head_dim,
    std::uint32_t window_size,
    float softmax_scale,
    float residual_scale) {
  __shared__ float scores[16];
  __shared__ float weights[16];
  __shared__ float denom_shared;
  __shared__ float bounded_score_shared;

  if (qkv_rows == 0u || head_count == 0u || head_dim == 0u ||
      window_size == 0u || head_dim > 32u || window_size > 16u) {
    return;
  }
  const std::uint32_t possible_heads = qkv_rows / head_dim;
  if (possible_heads == 0u) {
    return;
  }
  const std::uint32_t active_heads =
      head_count < possible_heads ? head_count : possible_heads;
  const std::uint32_t rows_per_head = qkv_rows / active_heads;
  const std::uint32_t contexts_per_head = rows_per_head / head_dim;
  if (contexts_per_head == 0u) {
    return;
  }
  const std::uint32_t contexts_for_activation =
      (activation_cols + active_heads * head_dim - 1u) /
      (active_heads * head_dim);
  if (contexts_for_activation == 0u) {
    return;
  }
  const std::uint32_t active_contexts =
      contexts_for_activation < contexts_per_head
          ? contexts_for_activation
          : contexts_per_head;
  const std::uint32_t tile = blockIdx.x;
  if (tile >= active_heads * active_contexts) {
    return;
  }
  const std::uint32_t head = tile % active_heads;
  const std::uint32_t query_context = tile / active_heads;
  const std::uint32_t head_base = head * rows_per_head;
  const std::uint32_t query_base = head_base + query_context * head_dim;
  const std::uint32_t window =
      window_size < contexts_per_head ? window_size : contexts_per_head;

  if (threadIdx.x == 0u) {
    float max_score = -CUDART_INF_F;
    for (std::uint32_t item = 0; item < window; ++item) {
      const std::uint32_t key_context =
          (query_context + contexts_per_head - item) % contexts_per_head;
      const std::uint32_t key_base = head_base + key_context * head_dim;
      float dot = 0.0f;
      for (std::uint32_t lane = 0; lane < head_dim; ++lane) {
        dot += scratch[query_base + lane] *
            scratch[qkv_rows + key_base + lane];
      }
      const float score = dot * softmax_scale;
      scores[item] = score;
      max_score = fmaxf(max_score, score);
    }
    float denom = 0.0f;
    float weighted_score = 0.0f;
    for (std::uint32_t item = 0; item < window; ++item) {
      const float weight = expf(scores[item] - max_score);
      weights[item] = weight;
      denom += weight;
      weighted_score += weight * scores[item];
    }
    denom_shared = denom;
    const float mean_score = denom > 0.0f ? weighted_score / denom : 0.0f;
    bounded_score_shared = mean_score / (1.0f + fabsf(mean_score));
  }
  __syncthreads();

  const std::uint32_t dim = threadIdx.x;
  if (dim >= head_dim || denom_shared <= 0.0f) {
    return;
  }
  const std::uint32_t col =
      head + active_heads * (dim + head_dim * query_context);
  if (col >= activation_cols) {
    return;
  }
  float weighted_v = 0.0f;
  for (std::uint32_t item = 0; item < window; ++item) {
    const std::uint32_t key_context =
        (query_context + contexts_per_head - item) % contexts_per_head;
    const std::uint32_t key_base = head_base + key_context * head_dim;
    weighted_v += weights[item] * scratch[2u * qkv_rows + key_base + dim];
  }
  const float context = weighted_v / denom_shared;
  const float head_bias =
      static_cast<float>((head & 0x7u) + 1u) * 0.0009765625f;
  const float dim_bias =
      static_cast<float>((dim & 0xfu) + 1u) * 0.0001220703125f;
  const float residual =
      context * 0.078125f +
      scratch[query_base + dim] * 0.015625f +
      bounded_score_shared * 0.015625f +
      head_bias +
      dim_bias;
  activation[col] = activation[col] + residual * residual_scale;
}

__global__ void attention_qkv_head_group_window_kernel(
    float* scratch,
    float* activation,
    std::uint32_t qkv_rows,
    std::uint32_t activation_cols,
    std::uint32_t head_count,
    std::uint32_t head_dim,
    std::uint32_t window_size,
    std::uint32_t contexts_per_block,
    float softmax_scale,
    float residual_scale) {
  __shared__ float scores[4][16];
  __shared__ float weights[4][16];
  __shared__ float denom_shared[4];
  __shared__ float bounded_score_shared[4];

  if (qkv_rows == 0u || head_count == 0u || head_dim == 0u ||
      window_size == 0u || contexts_per_block == 0u || head_dim > 32u ||
      window_size > 16u || contexts_per_block > 4u) {
    return;
  }
  const std::uint32_t possible_heads = qkv_rows / head_dim;
  if (possible_heads == 0u) {
    return;
  }
  const std::uint32_t active_heads =
      head_count < possible_heads ? head_count : possible_heads;
  const std::uint32_t rows_per_head = qkv_rows / active_heads;
  const std::uint32_t contexts_per_head = rows_per_head / head_dim;
  if (contexts_per_head == 0u) {
    return;
  }
  const std::uint32_t contexts_for_activation =
      (activation_cols + active_heads * head_dim - 1u) /
      (active_heads * head_dim);
  if (contexts_for_activation == 0u) {
    return;
  }
  const std::uint32_t active_contexts =
      contexts_for_activation < contexts_per_head
          ? contexts_for_activation
          : contexts_per_head;
  const std::uint32_t group = blockIdx.x;
  const std::uint32_t head = group % active_heads;
  const std::uint32_t context_group = group / active_heads;
  const std::uint32_t context_slot = threadIdx.x / head_dim;
  const std::uint32_t dim = threadIdx.x - context_slot * head_dim;
  if (context_slot >= contexts_per_block || dim >= head_dim) {
    return;
  }
  const std::uint32_t query_context =
      context_group * contexts_per_block + context_slot;
  const std::uint32_t head_base = head * rows_per_head;
  const std::uint32_t query_base = head_base + query_context * head_dim;
  const std::uint32_t window =
      window_size < contexts_per_head ? window_size : contexts_per_head;

  if (dim == 0u) {
    denom_shared[context_slot] = 0.0f;
    bounded_score_shared[context_slot] = 0.0f;
    if (query_context < active_contexts) {
      float max_score = -CUDART_INF_F;
      for (std::uint32_t item = 0; item < window; ++item) {
        const std::uint32_t key_context =
            (query_context + contexts_per_head - item) % contexts_per_head;
        const std::uint32_t key_base = head_base + key_context * head_dim;
        float dot = 0.0f;
        for (std::uint32_t lane = 0; lane < head_dim; ++lane) {
          dot += scratch[query_base + lane] *
              scratch[qkv_rows + key_base + lane];
        }
        const float score = dot * softmax_scale;
        scores[context_slot][item] = score;
        max_score = fmaxf(max_score, score);
      }
      float denom = 0.0f;
      float weighted_score = 0.0f;
      for (std::uint32_t item = 0; item < window; ++item) {
        const float weight = expf(scores[context_slot][item] - max_score);
        weights[context_slot][item] = weight;
        denom += weight;
        weighted_score += weight * scores[context_slot][item];
      }
      denom_shared[context_slot] = denom;
      const float mean_score = denom > 0.0f ? weighted_score / denom : 0.0f;
      bounded_score_shared[context_slot] =
          mean_score / (1.0f + fabsf(mean_score));
    }
  }
  __syncthreads();

  if (query_context >= active_contexts || denom_shared[context_slot] <= 0.0f) {
    return;
  }
  const std::uint32_t col =
      head + active_heads * (dim + head_dim * query_context);
  if (col >= activation_cols) {
    return;
  }
  float weighted_v = 0.0f;
  for (std::uint32_t item = 0; item < window; ++item) {
    const std::uint32_t key_context =
        (query_context + contexts_per_head - item) % contexts_per_head;
    const std::uint32_t key_base = head_base + key_context * head_dim;
    weighted_v += weights[context_slot][item] *
        scratch[2u * qkv_rows + key_base + dim];
  }
  const float context = weighted_v / denom_shared[context_slot];
  const float head_bias =
      static_cast<float>((head & 0x7u) + 1u) * 0.0009765625f;
  const float dim_bias =
      static_cast<float>((dim & 0xfu) + 1u) * 0.0001220703125f;
  const float residual =
      context * 0.078125f +
      scratch[query_base + dim] * 0.015625f +
      bounded_score_shared[context_slot] * 0.015625f +
      head_bias +
      dim_bias;
  activation[col] = activation[col] + residual * residual_scale;
}

__global__ void attention_qkv_head_group_rope_window_kernel(
    float* scratch,
    float* activation,
    std::uint32_t qkv_rows,
    std::uint32_t activation_cols,
    std::uint32_t head_count,
    std::uint32_t head_dim,
    std::uint32_t window_size,
    std::uint32_t contexts_per_block,
    float rope_theta_scale,
    float softmax_scale,
    float residual_scale) {
  __shared__ float scores[4][16];
  __shared__ float weights[4][16];
  __shared__ float denom_shared[4];
  __shared__ float bounded_score_shared[4];

  if (qkv_rows == 0u || head_count == 0u || head_dim == 0u ||
      window_size == 0u || contexts_per_block == 0u || head_dim > 32u ||
      (head_dim & 1u) != 0u || window_size > 16u ||
      contexts_per_block > 4u) {
    return;
  }
  const std::uint32_t possible_heads = qkv_rows / head_dim;
  if (possible_heads == 0u) {
    return;
  }
  const std::uint32_t active_heads =
      head_count < possible_heads ? head_count : possible_heads;
  const std::uint32_t rows_per_head = qkv_rows / active_heads;
  const std::uint32_t contexts_per_head = rows_per_head / head_dim;
  if (contexts_per_head == 0u) {
    return;
  }
  const std::uint32_t contexts_for_activation =
      (activation_cols + active_heads * head_dim - 1u) /
      (active_heads * head_dim);
  if (contexts_for_activation == 0u) {
    return;
  }
  const std::uint32_t active_contexts =
      contexts_for_activation < contexts_per_head
          ? contexts_for_activation
          : contexts_per_head;
  const std::uint32_t group = blockIdx.x;
  const std::uint32_t head = group % active_heads;
  const std::uint32_t context_group = group / active_heads;
  const std::uint32_t context_slot = threadIdx.x / head_dim;
  const std::uint32_t dim = threadIdx.x - context_slot * head_dim;
  if (context_slot >= contexts_per_block || dim >= head_dim) {
    return;
  }
  const std::uint32_t query_context =
      context_group * contexts_per_block + context_slot;
  const std::uint32_t head_base = head * rows_per_head;
  const std::uint32_t query_base = head_base + query_context * head_dim;
  const std::uint32_t window =
      window_size < contexts_per_head ? window_size : contexts_per_head;

  if (dim == 0u) {
    denom_shared[context_slot] = 0.0f;
    bounded_score_shared[context_slot] = 0.0f;
    if (query_context < active_contexts) {
      float max_score = -CUDART_INF_F;
      for (std::uint32_t item = 0; item < window; ++item) {
        const std::uint32_t key_context =
            (query_context + contexts_per_head - item) % contexts_per_head;
        const std::uint32_t key_base = head_base + key_context * head_dim;
        float dot = 0.0f;
        for (std::uint32_t lane = 0; lane < head_dim; lane += 2u) {
          const std::uint32_t pair = lane >> 1u;
          float q_even = 0.0f;
          float q_odd = 0.0f;
          float k_even = 0.0f;
          float k_odd = 0.0f;
          attention_rope_rotate_pair(
              scratch[query_base + lane],
              scratch[query_base + lane + 1u],
              query_context,
              pair,
              rope_theta_scale,
              &q_even,
              &q_odd);
          attention_rope_rotate_pair(
              scratch[qkv_rows + key_base + lane],
              scratch[qkv_rows + key_base + lane + 1u],
              key_context,
              pair,
              rope_theta_scale,
              &k_even,
              &k_odd);
          dot += q_even * k_even + q_odd * k_odd;
        }
        const float score = dot * softmax_scale;
        scores[context_slot][item] = score;
        max_score = fmaxf(max_score, score);
      }
      float denom = 0.0f;
      float weighted_score = 0.0f;
      for (std::uint32_t item = 0; item < window; ++item) {
        const float weight = expf(scores[context_slot][item] - max_score);
        weights[context_slot][item] = weight;
        denom += weight;
        weighted_score += weight * scores[context_slot][item];
      }
      denom_shared[context_slot] = denom;
      const float mean_score = denom > 0.0f ? weighted_score / denom : 0.0f;
      bounded_score_shared[context_slot] =
          mean_score / (1.0f + fabsf(mean_score));
    }
  }
  __syncthreads();

  if (query_context >= active_contexts || denom_shared[context_slot] <= 0.0f) {
    return;
  }
  const std::uint32_t col =
      head + active_heads * (dim + head_dim * query_context);
  if (col >= activation_cols) {
    return;
  }
  float weighted_v = 0.0f;
  for (std::uint32_t item = 0; item < window; ++item) {
    const std::uint32_t key_context =
        (query_context + contexts_per_head - item) % contexts_per_head;
    const std::uint32_t key_base = head_base + key_context * head_dim;
    weighted_v += weights[context_slot][item] *
        scratch[2u * qkv_rows + key_base + dim];
  }
  const std::uint32_t pair_base = dim & ~1u;
  const std::uint32_t pair = pair_base >> 1u;
  float q_even = 0.0f;
  float q_odd = 0.0f;
  attention_rope_rotate_pair(
      scratch[query_base + pair_base],
      scratch[query_base + pair_base + 1u],
      query_context,
      pair,
      rope_theta_scale,
      &q_even,
      &q_odd);
  const float q_lane = (dim & 1u) == 0u ? q_even : q_odd;
  const float context = weighted_v / denom_shared[context_slot];
  const float head_bias =
      static_cast<float>((head & 0x7u) + 1u) * 0.0009765625f;
  const float dim_bias =
      static_cast<float>((dim & 0xfu) + 1u) * 0.0001220703125f;
  const float residual =
      context * 0.078125f +
      q_lane * 0.015625f +
      bounded_score_shared[context_slot] * 0.015625f +
      head_bias +
      dim_bias;
  activation[col] = activation[col] + residual * residual_scale;
}

__global__ void attention_qkv_head_group_fused_kernel(
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
    float residual_scale) {
  __shared__ float scores[4][16];
  __shared__ float weights[4][16];
  __shared__ float denom_shared[4];
  __shared__ float bounded_score_shared[4];

  if (logits == nullptr || activation == nullptr || rows == 0u ||
      activation_cols == 0u || head_count == 0u || head_dim == 0u ||
      window_size == 0u || contexts_per_block == 0u || head_dim > 32u ||
      window_size > 16u || contexts_per_block > 4u) {
    return;
  }
  const std::uint32_t possible_heads = rows / head_dim;
  if (possible_heads == 0u) {
    return;
  }
  const std::uint32_t active_heads =
      head_count < possible_heads ? head_count : possible_heads;
  const std::uint32_t rows_per_head = rows / active_heads;
  const std::uint32_t contexts_per_head = rows_per_head / head_dim;
  if (contexts_per_head == 0u) {
    return;
  }
  const std::uint32_t contexts_for_activation =
      (activation_cols + active_heads * head_dim - 1u) /
      (active_heads * head_dim);
  if (contexts_for_activation == 0u) {
    return;
  }
  const std::uint32_t active_contexts =
      contexts_for_activation < contexts_per_head
          ? contexts_for_activation
          : contexts_per_head;
  const std::uint32_t group = blockIdx.x;
  const std::uint32_t head = group % active_heads;
  const std::uint32_t context_group = group / active_heads;
  const std::uint32_t context_slot = threadIdx.x / head_dim;
  const std::uint32_t dim = threadIdx.x - context_slot * head_dim;
  if (context_slot >= contexts_per_block || dim >= head_dim) {
    return;
  }
  const std::uint32_t query_context =
      context_group * contexts_per_block + context_slot;
  const std::uint32_t head_base = head * rows_per_head;
  const std::uint32_t query_base = head_base + query_context * head_dim;
  const std::uint32_t window =
      window_size < contexts_per_head ? window_size : contexts_per_head;

  if (dim == 0u) {
    denom_shared[context_slot] = 0.0f;
    bounded_score_shared[context_slot] = 0.0f;
    if (query_context < active_contexts) {
      float max_score = -CUDART_INF_F;
      for (std::uint32_t item = 0; item < window; ++item) {
        const std::uint32_t key_context =
            (query_context + contexts_per_head - item) % contexts_per_head;
        const std::uint32_t key_base = head_base + key_context * head_dim;
        float dot = 0.0f;
        for (std::uint32_t lane = 0; lane < head_dim; ++lane) {
          const std::uint32_t query_row = query_base + lane;
          const std::uint32_t key_row = key_base + lane;
          const float q = attention_qkv_projection(
              logits[query_row % rows], 0.0f, 0u, query_row);
          const float k = attention_qkv_projection(
              logits[key_row % rows], 0.0f, 1u, key_row);
          dot += q * k;
        }
        const float score = dot * softmax_scale;
        scores[context_slot][item] = score;
        max_score = fmaxf(max_score, score);
      }
      float denom = 0.0f;
      float weighted_score = 0.0f;
      for (std::uint32_t item = 0; item < window; ++item) {
        const float weight = expf(scores[context_slot][item] - max_score);
        weights[context_slot][item] = weight;
        denom += weight;
        weighted_score += weight * scores[context_slot][item];
      }
      denom_shared[context_slot] = denom;
      const float mean_score = denom > 0.0f ? weighted_score / denom : 0.0f;
      bounded_score_shared[context_slot] =
          mean_score / (1.0f + fabsf(mean_score));
    }
  }
  __syncthreads();

  if (query_context >= active_contexts || denom_shared[context_slot] <= 0.0f) {
    return;
  }
  const std::uint32_t col =
      head + active_heads * (dim + head_dim * query_context);
  if (col >= activation_cols) {
    return;
  }
  const std::uint32_t activation_row = col % rows;
  const float active = activation[col];
  const float q_active = attention_qkv_projection(
      logits[activation_row], active, 0u, activation_row);
  const float k_active = attention_qkv_projection(
      logits[activation_row], active, 1u, activation_row);
  const float v_active = attention_qkv_projection(
      logits[activation_row], active, 2u, activation_row);
  const float scratch_residual =
      (q_active * 0.0625f + k_active * 0.03125f +
       v_active * 0.015625f) *
      scratch_residual_scale;

  float weighted_v = 0.0f;
  for (std::uint32_t item = 0; item < window; ++item) {
    const std::uint32_t key_context =
        (query_context + contexts_per_head - item) % contexts_per_head;
    const std::uint32_t key_base = head_base + key_context * head_dim;
    const std::uint32_t value_row = key_base + dim;
    const float v = attention_qkv_projection(
        logits[value_row % rows], 0.0f, 2u, value_row);
    weighted_v += weights[context_slot][item] * v;
  }
  const float context = weighted_v / denom_shared[context_slot];
  const std::uint32_t query_row = query_base + dim;
  const float q_direct = attention_qkv_projection(
      logits[query_row % rows], 0.0f, 0u, query_row);
  const float head_bias =
      static_cast<float>((head & 0x7u) + 1u) * 0.0009765625f;
  const float dim_bias =
      static_cast<float>((dim & 0xfu) + 1u) * 0.0001220703125f;
  const float residual =
      context * 0.078125f +
      q_direct * 0.015625f +
      bounded_score_shared[context_slot] * 0.015625f +
      head_bias +
      dim_bias;
  activation[col] = active + scratch_residual + residual * residual_scale;
}

__global__ void ssm_recurrent_state_kernel(
    const float* logits,
    float* state,
    float* activation,
    std::uint32_t rows,
    std::uint32_t state_values,
    std::uint32_t activation_cols,
    float decay,
    float logit_scale,
    float activation_scale,
    float output_scale) {
  const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= activation_cols || rows == 0u || state_values == 0u) {
    return;
  }
  const std::uint32_t state_index = col % state_values;
  const float logit = logits[col % rows];
  const float active = activation[col];
  const float next_state =
      state[state_index] * decay + logit * logit_scale + active * activation_scale;
  state[state_index] = next_state;
  activation[col] = active + next_state * output_scale;
}

__global__ void ssm_scan_scratch_kernel(
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
    float output_scale) {
  const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= activation_cols || rows == 0u || state_values == 0u ||
      scratch_values == 0u) {
    return;
  }
  const std::uint32_t state_index = col % state_values;
  const std::uint32_t scratch_index = col % scratch_values;
  const float logit = logits[col % rows];
  const float active = activation[col];
  const float previous_state = state[state_index];
  const float scan =
      scratch[scratch_index] * decay +
      previous_state * scratch_scale +
      logit * logit_scale +
      active * activation_scale;
  scratch[scratch_index] = scan;
  const float next_state =
      previous_state * decay + scan * scratch_scale + logit * logit_scale;
  state[state_index] = next_state;
  activation[col] = active + (next_state + scan * 0.25f) * output_scale;
}

__global__ void ssm_selective_scan_kernel(
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
    float output_scale) {
  const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= activation_cols || rows == 0u || state_values == 0u ||
      scratch_values == 0u) {
    return;
  }
  const std::uint32_t state_index = col % state_values;
  const std::uint32_t scratch_index = col % scratch_values;
  const float logit = logits[col % rows];
  const float active = activation[col];
  const float previous_state = state[state_index];
  const float previous_scan = scratch[scratch_index];
  const float gate_input =
      logit * gate_logit_scale +
      active * gate_activation_scale +
      previous_scan * gate_scratch_scale;
  const float gate = 1.0f / (1.0f + expf(-gate_input));
  const float candidate = tanhf(
      logit * logit_scale +
      active * activation_scale +
      previous_state * scratch_scale +
      previous_scan * scratch_scale);
  const float scan =
      previous_scan * decay +
      candidate * gate +
      previous_state * scratch_scale;
  scratch[scratch_index] = scan;
  const float keep = decay + (1.0f - decay) * (1.0f - gate);
  const float next_state =
      previous_state * keep +
      candidate * gate +
      scan * scratch_scale;
  state[state_index] = next_state;
  activation[col] =
      active + (next_state + scan * 0.25f + gate * 0.03125f) * output_scale;
}

__global__ void ssm_source_parameter_cache_kernel(
    const float* logits,
    const float* activation,
    float* scratch,
    std::uint32_t rows,
    std::uint32_t scratch_values,
    std::uint32_t state_values,
    std::uint32_t activation_cols,
    std::uint32_t parameter_slot,
    float logit_scale,
    float activation_scale) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= state_values || rows == 0u || scratch_values == 0u ||
      activation_cols == 0u || parameter_slot > 1u) {
    return;
  }
  const std::uint32_t scratch_index = parameter_slot * state_values + index;
  if (scratch_index >= scratch_values) {
    return;
  }
  const float logit = logits[index % rows];
  const float active = activation[index % activation_cols];
  const float slot_bias =
      parameter_slot == 0u ? -0.03125f : 0.03125f;
  scratch[scratch_index] =
      tanhf(logit * logit_scale + active * activation_scale + slot_bias);
}

__global__ void ssm_rmsnorm_feedback_parameter_cache_kernel(
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
    float activation_scale) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (rows == 0u || activation_cols == 0u || state_values == 0u ||
      scratch_values == 0u || parameter_slot > 1u) {
    return;
  }

  float sum_sq = 0.0f;
  for (std::uint32_t row = 0; row < rows; ++row) {
    const float value = logits[row];
    sum_sq += value * value;
  }
  const float mean_sq = sum_sq / static_cast<float>(rows);
  const float inv_rms = rsqrtf(mean_sq + epsilon);

  if (index < activation_cols) {
    activation[index] = logits[index % rows] * inv_rms;
  }

  if (index >= state_values) {
    return;
  }
  const std::uint32_t scratch_index = parameter_slot * state_values + index;
  if (scratch_index >= scratch_values) {
    return;
  }
  const float normalized = logits[(index % activation_cols) % rows] * inv_rms;
  const float slot_bias =
      parameter_slot == 0u ? -0.03125f : 0.03125f;
  scratch[scratch_index] =
      tanhf(logits[index % rows] * logit_scale +
            normalized * activation_scale +
            slot_bias);
}

__global__ void ssm_source_parameterized_scan_kernel(
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
    float output_scale) {
  const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= activation_cols || rows == 0u || state_values == 0u ||
      scratch_values < state_values * 2u) {
    return;
  }
  const std::uint32_t state_index = col % state_values;
  const std::uint32_t alpha_index = state_index;
  const std::uint32_t beta_index = state_values + state_index;
  const float alpha = scratch[alpha_index];
  const float beta = scratch[beta_index];
  const float logit = logits[col % rows];
  const float active = activation[col];
  const float previous_state = state[state_index];
  const float gate_input =
      logit * gate_logit_scale +
      active * gate_activation_scale +
      alpha * alpha_scale +
      beta * beta_scale;
  const float gate = 1.0f / (1.0f + expf(-gate_input));
  const float candidate = tanhf(
      logit * logit_scale +
      active * activation_scale +
      alpha * alpha_scale -
      beta * beta_scale +
      previous_state * scratch_scale);
  const float scan =
      beta * decay +
      alpha * scratch_scale +
      candidate * gate;
  scratch[beta_index] = scan;
  const float next_state =
      previous_state * decay +
      candidate * gate +
      scan * scratch_scale;
  state[state_index] = next_state;
  activation[col] =
      active + (next_state + scan * 0.25f + alpha * 0.03125f) * output_scale;
}

__global__ void iq2s_probe_kernel(
    const std::uint8_t* iq2s_blocks,
    std::uint64_t block_count,
    unsigned long long* output) {
  const std::uint64_t payload_bytes = block_count * kIq2SBlockBytes;
  const std::uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint64_t stride = gridDim.x * blockDim.x;
  unsigned long long local_sum = 0;
  for (std::uint64_t index = tid; index < payload_bytes; index += stride) {
    local_sum += static_cast<unsigned long long>(iq2s_blocks[index]);
  }
  if (local_sum != 0ull) {
    atomicAdd(output, local_sum);
  }

  if (tid == 0 && block_count > 0) {
    output[1] = static_cast<unsigned long long>(
        iq2s_blocks[kIq2SScaleOffset] |
        (static_cast<std::uint16_t>(iq2s_blocks[kIq2SScaleOffset + 1]) << 8u));
    unsigned long long qs_sum = 0;
    for (std::uint32_t index = 0; index < kIq2SQsBytes; ++index) {
      qs_sum += iq2s_blocks[kIq2SQsOffset + index];
    }
    unsigned long long qh_sum = 0;
    for (std::uint32_t index = 0; index < kIq2SQhBytes; ++index) {
      qh_sum += iq2s_blocks[kIq2SQhOffset + index];
    }
    unsigned long long scales_sum = 0;
    for (std::uint32_t index = 0; index < kIq2SScalesBytes; ++index) {
      scales_sum += iq2s_blocks[kIq2SScalesOffset + index];
    }
    output[2] = qs_sum;
    output[3] = qh_sum;
    output[4] = scales_sum;
    output[5] = block_count;
  }
}

__global__ void iq3s_probe_kernel(
    const std::uint8_t* iq3s_blocks,
    std::uint64_t block_count,
    unsigned long long* output) {
  const std::uint64_t payload_bytes = block_count * kIq3SBlockBytes;
  const std::uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint64_t stride = gridDim.x * blockDim.x;
  unsigned long long local_sum = 0;
  for (std::uint64_t index = tid; index < payload_bytes; index += stride) {
    local_sum += static_cast<unsigned long long>(iq3s_blocks[index]);
  }
  if (local_sum != 0ull) {
    atomicAdd(output, local_sum);
  }

  if (tid == 0 && block_count > 0) {
    output[1] = static_cast<unsigned long long>(
        iq3s_blocks[kIq3SScaleOffset] |
        (static_cast<std::uint16_t>(iq3s_blocks[kIq3SScaleOffset + 1]) << 8u));
    unsigned long long qs_sum = 0;
    for (std::uint32_t index = 0; index < kIq3SQsBytes; ++index) {
      qs_sum += iq3s_blocks[kIq3SQsOffset + index];
    }
    unsigned long long qh_sum = 0;
    for (std::uint32_t index = 0; index < kIq3SQhBytes; ++index) {
      qh_sum += iq3s_blocks[kIq3SQhOffset + index];
    }
    unsigned long long signs_sum = 0;
    for (std::uint32_t index = 0; index < kIq3SSignsBytes; ++index) {
      signs_sum += iq3s_blocks[kIq3SSignsOffset + index];
    }
    unsigned long long scales_sum = 0;
    for (std::uint32_t index = 0; index < kIq3SScalesBytes; ++index) {
      scales_sum += iq3s_blocks[kIq3SScalesOffset + index];
    }
    output[2] = qs_sum;
    output[3] = qh_sum;
    output[4] = signs_sum;
    output[5] = scales_sum;
    output[6] = block_count;
  }
}

__global__ void q5k_probe_kernel(
    const std::uint8_t* q5k_blocks,
    std::uint64_t block_count,
    unsigned long long* output) {
  const std::uint64_t payload_bytes = block_count * kQ5KBlockBytes;
  const std::uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint64_t stride = gridDim.x * blockDim.x;
  unsigned long long local_sum = 0;
  for (std::uint64_t index = tid; index < payload_bytes; index += stride) {
    local_sum += static_cast<unsigned long long>(q5k_blocks[index]);
  }
  if (local_sum != 0ull) {
    atomicAdd(output, local_sum);
  }

  if (tid == 0 && block_count > 0) {
    output[1] = static_cast<unsigned long long>(
        q5k_blocks[kQ5KDOffset] |
        (static_cast<std::uint16_t>(q5k_blocks[kQ5KDOffset + 1]) << 8u));
    output[2] = static_cast<unsigned long long>(
        q5k_blocks[kQ5KDMinOffset] |
        (static_cast<std::uint16_t>(q5k_blocks[kQ5KDMinOffset + 1]) << 8u));
    unsigned long long scales_sum = 0;
    for (std::uint32_t index = 0; index < kQ5KScalesBytes; ++index) {
      scales_sum += q5k_blocks[kQ5KScalesOffset + index];
    }
    unsigned long long qh_sum = 0;
    for (std::uint32_t index = 0; index < kQ5KQhBytes; ++index) {
      qh_sum += q5k_blocks[kQ5KQhOffset + index];
    }
    unsigned long long qs_sum = 0;
    for (std::uint32_t index = 0; index < kQ5KQsBytes; ++index) {
      qs_sum += q5k_blocks[kQ5KQsOffset + index];
    }
    output[3] = scales_sum;
    output[4] = qh_sum;
    output[5] = qs_sum;
    output[6] = block_count;
  }
}

__device__ float iq2s_dequant_value(
    const std::uint8_t* block,
    std::uint32_t value_index) {
  const std::uint32_t ib32 = value_index >> 5u;
  const std::uint32_t within32 = value_index & 31u;
  const std::uint32_t group = within32 >> 3u;
  const std::uint32_t lane = within32 & 7u;
  const auto d = fp16_bits_to_float(static_cast<std::uint16_t>(
      block[kIq2SScaleOffset] |
      (static_cast<std::uint16_t>(block[kIq2SScaleOffset + 1]) << 8u)));
  const auto* qs = block + kIq2SQsOffset;
  const auto* qh = block + kIq2SQhOffset;
  const auto* scales = block + kIq2SScalesOffset;
  const std::uint32_t group_offset = ib32 * 4u + group;
  const std::uint32_t grid_index =
      qs[group_offset] | ((static_cast<std::uint32_t>(qh[ib32]) << (8u - 2u * group)) & 0x300u);
  const std::uint32_t packed_grid = kIq2SKGridDevice[grid_index];
  const std::uint32_t grid = 2u * ((packed_grid >> (2u * lane)) & 0x3u) + 1u;
  const std::uint32_t scale_nibble =
      group < 2u ? (scales[ib32] & 0x0fu) : (scales[ib32] >> 4u);
  const float scale = d * (0.5f + static_cast<float>(scale_nibble)) * 0.25f;
  const bool negative = (qs[32u + group_offset] & (1u << lane)) != 0u;
  return scale * static_cast<float>(grid) * (negative ? -1.0f : 1.0f);
}

__device__ float iq3s_dequant_value(
    const std::uint8_t* block,
    std::uint32_t value_index) {
  const std::uint32_t ib32 = value_index >> 5u;
  const std::uint32_t within32 = value_index & 31u;
  const std::uint32_t group = within32 >> 3u;
  const std::uint32_t lane = within32 & 7u;
  const std::uint32_t pair = lane >> 2u;
  const std::uint32_t grid_lane = lane & 3u;
  const auto d = fp16_bits_to_float(static_cast<std::uint16_t>(
      block[kIq3SScaleOffset] |
      (static_cast<std::uint16_t>(block[kIq3SScaleOffset + 1]) << 8u)));
  const auto* qs = block + kIq3SQsOffset;
  const auto* qh = block + kIq3SQhOffset;
  const auto* signs = block + kIq3SSignsOffset;
  const auto* scales = block + kIq3SScalesOffset;

  const std::uint32_t selector_index = ib32 * 8u + group * 2u + pair;
  const std::uint32_t high_bit = (qh[ib32] >> (2u * group + pair)) & 1u;
  const std::uint32_t grid_index = qs[selector_index] | (high_bit << 8u);
  const std::uint32_t packed_grid = kIq3SGridDevice[grid_index];
  const std::uint32_t grid = (packed_grid >> (8u * grid_lane)) & 0xffu;
  const std::uint32_t scale_nibble =
      (ib32 & 1u) == 0u ? (scales[ib32 >> 1u] & 0x0fu)
                        : (scales[ib32 >> 1u] >> 4u);
  const float scale = d * static_cast<float>(1u + 2u * scale_nibble);
  const bool negative = (signs[ib32 * 4u + group] & (1u << lane)) != 0u;
  return scale * static_cast<float>(grid) * (negative ? -1.0f : 1.0f);
}

__global__ void iq2s_matvec_probe_kernel(
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
    std::uint32_t active_pages) {
  extern __shared__ float partial[];
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  if (row >= rows || row >= rows_limit) {
    return;
  }

  const std::uint32_t blocks_per_row = cols / 256u;
  float acc = 0.0f;
  for (std::uint32_t col = tid; col < cols; col += blockDim.x) {
    const std::uint32_t block_in_row = col / 256u;
    const std::uint32_t value_in_block = col & 255u;
    const auto* block = iq2s_blocks +
        (static_cast<std::size_t>(row) * blocks_per_row + block_in_row) *
            static_cast<std::size_t>(kIq2SBlockBytes);
    acc += iq2s_dequant_value(block, value_in_block) * activation[col];
  }

  partial[tid] = acc;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const float value = partial[0];
    logits[row] = value;
    if (kv != nullptr && token != nullptr && row < active_pages && kv_words > 0u &&
        page_words > 0u) {
      const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
      const std::size_t offset = (token[0] + row) % page_words;
      const std::size_t pos = (page_base + offset) % kv_words;
      kv[pos] = value;
    }
  }
}

__global__ void iq3s_matvec_probe_kernel(
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
    std::uint32_t active_pages) {
  extern __shared__ float partial[];
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  if (row >= rows || row >= rows_limit) {
    return;
  }

  const std::uint32_t blocks_per_row = cols / 256u;
  float acc = 0.0f;
  for (std::uint32_t col = tid; col < cols; col += blockDim.x) {
    const std::uint32_t block_in_row = col / 256u;
    const std::uint32_t value_in_block = col & 255u;
    const auto* block = iq3s_blocks +
        (static_cast<std::size_t>(row) * blocks_per_row + block_in_row) *
            static_cast<std::size_t>(kIq3SBlockBytes);
    acc += iq3s_dequant_value(block, value_in_block) * activation[col];
  }

  partial[tid] = acc;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const float value = partial[0];
    logits[row] = value;
    if (kv != nullptr && token != nullptr && row < active_pages && kv_words > 0u &&
        page_words > 0u) {
      const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
      const std::size_t offset = (token[0] + row) % page_words;
      const std::size_t pos = (page_base + offset) % kv_words;
      kv[pos] = value;
    }
  }
}

__device__ float dequant_q4_nibble(int nibble, float scale) {
  return static_cast<float>(nibble - 8) * scale;
}

__device__ float fp16_bits_to_float(std::uint16_t bits) {
  const float sign = (bits & 0x8000u) ? -1.0f : 1.0f;
  const int exp = static_cast<int>((bits >> 10u) & 0x1fu);
  const int mant = static_cast<int>(bits & 0x03ffu);
  if (exp == 0) {
    return mant == 0 ? sign * 0.0f : sign * ldexpf(static_cast<float>(mant), -24);
  }
  if (exp == 31) {
    return mant == 0 ? sign * CUDART_INF_F : CUDART_NAN_F;
  }
  return sign * ldexpf(1.0f + static_cast<float>(mant) / 1024.0f, exp - 15);
}

__device__ void get_scale_min_k4(
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

__device__ float q5k_dequant_value(
    const std::uint8_t* block,
    std::uint32_t value_index) {
  const std::uint16_t d_bits = static_cast<std::uint16_t>(
      block[kQ5KDOffset] |
      (static_cast<std::uint16_t>(block[kQ5KDOffset + 1]) << 8u));
  const std::uint16_t dmin_bits = static_cast<std::uint16_t>(
      block[kQ5KDMinOffset] |
      (static_cast<std::uint16_t>(block[kQ5KDMinOffset + 1]) << 8u));
  const float d = fp16_bits_to_float(d_bits);
  const float dmin = fp16_bits_to_float(dmin_bits);
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
  const std::uint8_t q =
      static_cast<std::uint8_t>(low_bits + ((qh[lane] & high_mask) ? 16u : 0u));
  return d * static_cast<float>(scale) * static_cast<float>(q) -
      dmin * static_cast<float>(min_value);
}

__global__ void q5k_matvec_decode_kernel(
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
    std::uint32_t active_pages) {
  extern __shared__ float partial[];
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  if (row >= rows || row >= rows_limit) {
    return;
  }

  const std::uint32_t blocks_per_row = cols / 256u;
  float acc = 0.0f;
  for (std::uint32_t col = tid; col < cols; col += blockDim.x) {
    const std::uint32_t block_in_row = col / 256u;
    const std::uint32_t value_in_block = col & 255u;
    const auto* block = q5k_blocks +
        (static_cast<std::size_t>(row) * blocks_per_row + block_in_row) *
            static_cast<std::size_t>(kQ5KBlockBytes);
    acc += q5k_dequant_value(block, value_in_block) * activation[col];
  }

  partial[tid] = acc;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float value = partial[0];
    logits[row] = value;
    if (kv != nullptr && token != nullptr && row < active_pages && kv_words > 0u &&
        page_words > 0u) {
      const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
      const std::size_t offset = (token[0] + row) % page_words;
      const std::size_t pos = (page_base + offset) % kv_words;
      kv[pos] = value;
    }
  }
}

__global__ void q4_matvec_decode_kernel(
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
    std::uint32_t active_pages) {
  extern __shared__ float partial[];

  const std::uint32_t row = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t row_packed_stride = (cols + 1u) / 2u;
  const std::size_t row_base = static_cast<std::size_t>(row) * row_packed_stride;
  const float scale = scales[row];

  float sum = 0.0f;
  for (std::uint32_t packed_col = tid; packed_col < row_packed_stride;
       packed_col += blockDim.x) {
    const std::uint8_t packed = packed_weights[row_base + packed_col];
    const std::uint32_t col0 = packed_col * 2u;
    const std::uint32_t col1 = col0 + 1u;
    sum += dequant_q4_nibble(packed & 0x0f, scale) * activation[col0];
    if (col1 < cols) {
      sum += dequant_q4_nibble((packed >> 4) & 0x0f, scale) * activation[col1];
    }
  }

  partial[tid] = sum;
  __syncthreads();

  for (std::uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0u) {
    const float value = partial[0];
    logits[row] = value;
    if (row < active_pages && kv_words > 0u && page_words > 0u) {
      const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
      const std::size_t offset = (token[0] + row) % page_words;
      const std::size_t pos = (page_base + offset) % kv_words;
      kv[pos] = value;
    }
  }
}

__global__ void q4_matvec_decode_vec4x4_kernel(
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
    std::uint32_t active_pages) {
  __shared__ float warp_sums[4];

  constexpr unsigned kFullWarpMask = 0xffffffffu;

  const std::uint32_t row = blockIdx.x;
  const std::uint32_t warp_id = threadIdx.x >> 5u;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t row_packed_stride = (cols + 1u) / 2u;
  const std::size_t row_base = static_cast<std::size_t>(row) * row_packed_stride;
  const std::uint32_t words_per_row = row_packed_stride / 4u;
  const float scale = scales[row];

  float sum = 0.0f;
  const auto* packed_words =
      reinterpret_cast<const std::uint32_t*>(packed_weights + row_base);
  for (std::uint32_t word_index = warp_id * 32u + lane; word_index < words_per_row;
       word_index += 128u) {
    const std::uint32_t packed4 = packed_words[word_index];
    const std::uint32_t col_base = word_index * 8u;
    const auto act0 = *reinterpret_cast<const float4*>(activation + col_base);
    const auto act1 = *reinterpret_cast<const float4*>(activation + col_base + 4u);

    const std::uint8_t packed0 = static_cast<std::uint8_t>(packed4 & 0xffu);
    const std::uint8_t packed1 = static_cast<std::uint8_t>((packed4 >> 8u) & 0xffu);
    const std::uint8_t packed2 = static_cast<std::uint8_t>((packed4 >> 16u) & 0xffu);
    const std::uint8_t packed3 = static_cast<std::uint8_t>((packed4 >> 24u) & 0xffu);

    sum += dequant_q4_nibble(packed0 & 0x0f, scale) * act0.x;
    sum += dequant_q4_nibble((packed0 >> 4u) & 0x0f, scale) * act0.y;
    sum += dequant_q4_nibble(packed1 & 0x0f, scale) * act0.z;
    sum += dequant_q4_nibble((packed1 >> 4u) & 0x0f, scale) * act0.w;
    sum += dequant_q4_nibble(packed2 & 0x0f, scale) * act1.x;
    sum += dequant_q4_nibble((packed2 >> 4u) & 0x0f, scale) * act1.y;
    sum += dequant_q4_nibble(packed3 & 0x0f, scale) * act1.z;
    sum += dequant_q4_nibble((packed3 >> 4u) & 0x0f, scale) * act1.w;
  }

  const std::uint32_t tail_start = words_per_row * 4u;
  if (warp_id == 0u) {
    for (std::uint32_t packed_col = tail_start + lane; packed_col < row_packed_stride;
         packed_col += 32u) {
      const std::uint8_t packed = packed_weights[row_base + packed_col];
      const std::uint32_t col0 = packed_col * 2u;
      const std::uint32_t col1 = col0 + 1u;
      sum += dequant_q4_nibble(packed & 0x0f, scale) * activation[col0];
      if (col1 < cols) {
        sum += dequant_q4_nibble((packed >> 4u) & 0x0f, scale) * activation[col1];
      }
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(kFullWarpMask, sum, offset);
  }

  if (lane == 0u) {
    warp_sums[warp_id] = sum;
  }
  __syncthreads();

  if (threadIdx.x == 0u) {
    const float value = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
    logits[row] = value;
    if (row < active_pages && kv_words > 0u && page_words > 0u) {
      const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
      const std::size_t offset = (token[0] + row) % page_words;
      const std::size_t pos = (page_base + offset) % kv_words;
      kv[pos] = value;
    }
  }
}

__global__ void q4k_matvec_decode_kernel(
    const std::uint8_t* q4k_blocks,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages) {
  extern __shared__ float partial[];

  constexpr std::uint32_t kQ4KValues = 256;
  constexpr std::uint32_t kQ4KBytes = 144;
  constexpr std::uint32_t kQ4KPayloadOffset = 16;

  const std::uint32_t row = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t blocks_per_row = cols / kQ4KValues;
  const std::size_t row_base =
      static_cast<std::size_t>(row) * blocks_per_row * kQ4KBytes;

  float sum = 0.0f;
  for (std::uint32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
    const auto* block = q4k_blocks + row_base + static_cast<std::size_t>(block_index) * kQ4KBytes;
    const std::uint16_t d_bits =
        static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8u));
    const std::uint16_t dmin_bits =
        static_cast<std::uint16_t>(block[2] | (static_cast<std::uint16_t>(block[3]) << 8u));
    const float d = fp16_bits_to_float(d_bits);
    const float dmin = fp16_bits_to_float(dmin_bits);
    const auto* scales = block + 4;
    const auto* qs = block + kQ4KPayloadOffset;
    const std::uint32_t col_base = block_index * kQ4KValues;

    for (std::uint32_t packed_index = tid; packed_index < 128u; packed_index += blockDim.x) {
      const std::uint32_t group = packed_index / 32u;
      const std::uint32_t group_offset = packed_index - group * 32u;
      const int scale_index = static_cast<int>(group * 2u);
      std::uint8_t scale_0 = 0;
      std::uint8_t min_0 = 0;
      std::uint8_t scale_1 = 0;
      std::uint8_t min_1 = 0;
      get_scale_min_k4(scale_index, scales, scale_0, min_0);
      get_scale_min_k4(scale_index + 1, scales, scale_1, min_1);

      const std::uint8_t packed = qs[packed_index];
      const std::uint32_t col0 = col_base + group * 64u + group_offset;
      const std::uint32_t col1 = col0 + 32u;
      const float value0 =
          d * static_cast<float>(scale_0) * static_cast<float>(packed & 0x0fu) -
          dmin * static_cast<float>(min_0);
      const float value1 =
          d * static_cast<float>(scale_1) * static_cast<float>(packed >> 4u) -
          dmin * static_cast<float>(min_1);
      sum += value0 * activation[col0];
      sum += value1 * activation[col1];
    }
  }

  partial[tid] = sum;
  __syncthreads();

  for (std::uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0u) {
    const float value = partial[0];
    logits[row] = value;
    if (row < active_pages && kv_words > 0u && page_words > 0u) {
      const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
      const std::size_t offset = (token[0] + row) % page_words;
      const std::size_t pos = (page_base + offset) % kv_words;
      kv[pos] = value;
    }
  }
}

__global__ void q4k_matvec_decode_shared_kernel(
    const std::uint8_t* q4k_blocks,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages) {
  extern __shared__ float partial[];
  __shared__ float qscale[8];
  __shared__ float qmin[8];

  constexpr std::uint32_t kQ4KValues = 256;
  constexpr std::uint32_t kQ4KBytes = 144;
  constexpr std::uint32_t kQ4KPayloadOffset = 16;

  const std::uint32_t row = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t blocks_per_row = cols / kQ4KValues;
  const std::size_t row_base =
      static_cast<std::size_t>(row) * blocks_per_row * kQ4KBytes;

  float sum = 0.0f;
  for (std::uint32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
    const auto* block = q4k_blocks + row_base + static_cast<std::size_t>(block_index) * kQ4KBytes;
    if (tid < 8u) {
      const std::uint16_t d_bits =
          static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8u));
      const std::uint16_t dmin_bits =
          static_cast<std::uint16_t>(block[2] | (static_cast<std::uint16_t>(block[3]) << 8u));
      const float d = fp16_bits_to_float(d_bits);
      const float dmin = fp16_bits_to_float(dmin_bits);
      std::uint8_t scale = 0;
      std::uint8_t min_value = 0;
      get_scale_min_k4(static_cast<int>(tid), block + 4, scale, min_value);
      qscale[tid] = d * static_cast<float>(scale);
      qmin[tid] = dmin * static_cast<float>(min_value);
    }
    __syncthreads();

    const auto* qs = block + kQ4KPayloadOffset;
    const std::uint32_t col_base = block_index * kQ4KValues;
    for (std::uint32_t packed_index = tid; packed_index < 128u; packed_index += blockDim.x) {
      const std::uint32_t group = packed_index / 32u;
      const std::uint32_t group_offset = packed_index - group * 32u;
      const std::uint8_t packed = qs[packed_index];
      const std::uint32_t col0 = col_base + group * 64u + group_offset;
      const std::uint32_t col1 = col0 + 32u;
      const std::uint32_t scale_index = group * 2u;
      const float value0 = qscale[scale_index] * static_cast<float>(packed & 0x0fu) -
          qmin[scale_index];
      const float value1 = qscale[scale_index + 1u] * static_cast<float>(packed >> 4u) -
          qmin[scale_index + 1u];
      sum += value0 * activation[col0];
      sum += value1 * activation[col1];
    }
    __syncthreads();
  }

  partial[tid] = sum;
  __syncthreads();

  for (std::uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0u) {
    const float value = partial[0];
    logits[row] = value;
    if (row < active_pages && kv_words > 0u && page_words > 0u) {
      const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
      const std::size_t offset = (token[0] + row) % page_words;
      const std::size_t pos = (page_base + offset) % kv_words;
      kv[pos] = value;
    }
  }
}

__global__ void q4k_matvec_decode_warp_kernel(
    const std::uint8_t* q4k_blocks,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages) {
  constexpr std::uint32_t kQ4KValues = 256;
  constexpr std::uint32_t kQ4KBytes = 144;
  constexpr std::uint32_t kQ4KPayloadOffset = 16;
  constexpr unsigned kFullWarpMask = 0xffffffffu;

  const std::uint32_t row = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t blocks_per_row = cols / kQ4KValues;
  const std::size_t row_base =
      static_cast<std::size_t>(row) * blocks_per_row * kQ4KBytes;

  float sum = 0.0f;
  for (std::uint32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
    const auto* block = q4k_blocks + row_base + static_cast<std::size_t>(block_index) * kQ4KBytes;
    const std::uint16_t d_bits =
        static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8u));
    const std::uint16_t dmin_bits =
        static_cast<std::uint16_t>(block[2] | (static_cast<std::uint16_t>(block[3]) << 8u));
    const float d = fp16_bits_to_float(d_bits);
    const float dmin = fp16_bits_to_float(dmin_bits);
    const auto* scales = block + 4;
    const auto* qs = block + kQ4KPayloadOffset;
    const std::uint32_t col_base = block_index * kQ4KValues;

    for (std::uint32_t group = 0; group < 4u; ++group) {
      std::uint8_t scale_0 = 0;
      std::uint8_t min_0 = 0;
      std::uint8_t scale_1 = 0;
      std::uint8_t min_1 = 0;
      get_scale_min_k4(static_cast<int>(group * 2u), scales, scale_0, min_0);
      get_scale_min_k4(static_cast<int>(group * 2u + 1u), scales, scale_1, min_1);

      const std::uint8_t packed = qs[group * 32u + lane];
      const std::uint32_t col0 = col_base + group * 64u + lane;
      const std::uint32_t col1 = col0 + 32u;
      const float value0 =
          d * static_cast<float>(scale_0) * static_cast<float>(packed & 0x0fu) -
          dmin * static_cast<float>(min_0);
      const float value1 =
          d * static_cast<float>(scale_1) * static_cast<float>(packed >> 4u) -
          dmin * static_cast<float>(min_1);
      sum += value0 * activation[col0];
      sum += value1 * activation[col1];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(kFullWarpMask, sum, offset);
  }

  if (lane == 0u) {
    logits[row] = sum;
    if (row < active_pages && kv_words > 0u && page_words > 0u) {
      const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
      const std::size_t offset = (token[0] + row) % page_words;
      const std::size_t pos = (page_base + offset) % kv_words;
      kv[pos] = sum;
    }
  }
}

__global__ void q4k_matvec_decode_warp_broadcast_kernel(
    const std::uint8_t* q4k_blocks,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages) {
  constexpr std::uint32_t kQ4KValues = 256;
  constexpr std::uint32_t kQ4KBytes = 144;
  constexpr std::uint32_t kQ4KPayloadOffset = 16;
  constexpr unsigned kFullWarpMask = 0xffffffffu;

  const std::uint32_t row = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t blocks_per_row = cols / kQ4KValues;
  const std::size_t row_base =
      static_cast<std::size_t>(row) * blocks_per_row * kQ4KBytes;

  float sum = 0.0f;
  for (std::uint32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
    const auto* block = q4k_blocks + row_base + static_cast<std::size_t>(block_index) * kQ4KBytes;
    float local_scale = 0.0f;
    float local_min = 0.0f;
    if (lane < 8u) {
      const std::uint16_t d_bits =
          static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8u));
      const std::uint16_t dmin_bits =
          static_cast<std::uint16_t>(block[2] | (static_cast<std::uint16_t>(block[3]) << 8u));
      const float d = fp16_bits_to_float(d_bits);
      const float dmin = fp16_bits_to_float(dmin_bits);
      std::uint8_t scale = 0;
      std::uint8_t min_value = 0;
      get_scale_min_k4(static_cast<int>(lane), block + 4, scale, min_value);
      local_scale = d * static_cast<float>(scale);
      local_min = dmin * static_cast<float>(min_value);
    }

    const auto* qs = block + kQ4KPayloadOffset;
    const std::uint32_t col_base = block_index * kQ4KValues;
    for (std::uint32_t group = 0; group < 4u; ++group) {
      const std::uint32_t scale_index = group * 2u;
      const float scale_0 = __shfl_sync(kFullWarpMask, local_scale, scale_index);
      const float min_0 = __shfl_sync(kFullWarpMask, local_min, scale_index);
      const float scale_1 = __shfl_sync(kFullWarpMask, local_scale, scale_index + 1u);
      const float min_1 = __shfl_sync(kFullWarpMask, local_min, scale_index + 1u);

      const std::uint8_t packed = qs[group * 32u + lane];
      const std::uint32_t col0 = col_base + group * 64u + lane;
      const std::uint32_t col1 = col0 + 32u;
      const float value0 = scale_0 * static_cast<float>(packed & 0x0fu) - min_0;
      const float value1 = scale_1 * static_cast<float>(packed >> 4u) - min_1;
      sum += value0 * activation[col0];
      sum += value1 * activation[col1];
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(kFullWarpMask, sum, offset);
  }

  if (lane == 0u) {
    logits[row] = sum;
    if (row < active_pages && kv_words > 0u && page_words > 0u) {
      const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
      const std::size_t offset = (token[0] + row) % page_words;
      const std::size_t pos = (page_base + offset) % kv_words;
      kv[pos] = sum;
    }
  }
}

__global__ void q4k_matvec_decode_warp_broadcast_vec4_kernel(
    const std::uint8_t* q4k_blocks,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages) {
  constexpr std::uint32_t kQ4KValues = 256;
  constexpr std::uint32_t kQ4KBytes = 144;
  constexpr std::uint32_t kQ4KPayloadOffset = 16;
  constexpr unsigned kFullWarpMask = 0xffffffffu;

  const std::uint32_t row = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t group = lane >> 3u;
  const std::uint32_t group_lane = lane & 7u;
  const std::uint32_t blocks_per_row = cols / kQ4KValues;
  const std::size_t row_base =
      static_cast<std::size_t>(row) * blocks_per_row * kQ4KBytes;

  float sum = 0.0f;
  for (std::uint32_t block_index = 0; block_index < blocks_per_row; ++block_index) {
    const auto* block = q4k_blocks + row_base + static_cast<std::size_t>(block_index) * kQ4KBytes;
    float local_scale = 0.0f;
    float local_min = 0.0f;
    if (lane < 8u) {
      const std::uint16_t d_bits =
          static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8u));
      const std::uint16_t dmin_bits =
          static_cast<std::uint16_t>(block[2] | (static_cast<std::uint16_t>(block[3]) << 8u));
      const float d = fp16_bits_to_float(d_bits);
      const float dmin = fp16_bits_to_float(dmin_bits);
      std::uint8_t scale = 0;
      std::uint8_t min_value = 0;
      get_scale_min_k4(static_cast<int>(lane), block + 4, scale, min_value);
      local_scale = d * static_cast<float>(scale);
      local_min = dmin * static_cast<float>(min_value);
    }

    const std::uint32_t scale_index = group * 2u;
    const float scale_0 = __shfl_sync(kFullWarpMask, local_scale, scale_index);
    const float min_0 = __shfl_sync(kFullWarpMask, local_min, scale_index);
    const float scale_1 = __shfl_sync(kFullWarpMask, local_scale, scale_index + 1u);
    const float min_1 = __shfl_sync(kFullWarpMask, local_min, scale_index + 1u);

    const auto* qs = block + kQ4KPayloadOffset;
    const auto* packed_words =
        reinterpret_cast<const std::uint32_t*>(qs + group * 32u);
    const std::uint32_t packed4 = packed_words[group_lane];
    const std::uint32_t col_base =
        block_index * kQ4KValues + group * 64u + group_lane * 4u;
    const auto act0 = *reinterpret_cast<const float4*>(activation + col_base);
    const auto act1 = *reinterpret_cast<const float4*>(activation + col_base + 32u);

    const std::uint8_t packed0 = static_cast<std::uint8_t>(packed4 & 0xffu);
    const std::uint8_t packed1 = static_cast<std::uint8_t>((packed4 >> 8u) & 0xffu);
    const std::uint8_t packed2 = static_cast<std::uint8_t>((packed4 >> 16u) & 0xffu);
    const std::uint8_t packed3 = static_cast<std::uint8_t>((packed4 >> 24u) & 0xffu);

    sum += (scale_0 * static_cast<float>(packed0 & 0x0fu) - min_0) * act0.x;
    sum += (scale_1 * static_cast<float>(packed0 >> 4u) - min_1) * act1.x;
    sum += (scale_0 * static_cast<float>(packed1 & 0x0fu) - min_0) * act0.y;
    sum += (scale_1 * static_cast<float>(packed1 >> 4u) - min_1) * act1.y;
    sum += (scale_0 * static_cast<float>(packed2 & 0x0fu) - min_0) * act0.z;
    sum += (scale_1 * static_cast<float>(packed2 >> 4u) - min_1) * act1.z;
    sum += (scale_0 * static_cast<float>(packed3 & 0x0fu) - min_0) * act0.w;
    sum += (scale_1 * static_cast<float>(packed3 >> 4u) - min_1) * act1.w;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(kFullWarpMask, sum, offset);
  }

  if (lane == 0u) {
    logits[row] = sum;
    if (row < active_pages && kv_words > 0u && page_words > 0u) {
      const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
      const std::size_t offset = (token[0] + row) % page_words;
      const std::size_t pos = (page_base + offset) % kv_words;
      kv[pos] = sum;
    }
  }
}

__global__ void q4k_matvec_decode_warp_broadcast_vec4x2_kernel(
    const std::uint8_t* q4k_blocks,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages) {
  __shared__ float warp_sums[2];

  constexpr std::uint32_t kQ4KValues = 256;
  constexpr std::uint32_t kQ4KBytes = 144;
  constexpr std::uint32_t kQ4KPayloadOffset = 16;
  constexpr unsigned kFullWarpMask = 0xffffffffu;

  const std::uint32_t row = blockIdx.x;
  const std::uint32_t warp_id = threadIdx.x >> 5u;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t group = lane >> 3u;
  const std::uint32_t group_lane = lane & 7u;
  const std::uint32_t blocks_per_row = cols / kQ4KValues;
  const std::size_t row_base =
      static_cast<std::size_t>(row) * blocks_per_row * kQ4KBytes;

  float sum = 0.0f;
  for (std::uint32_t block_index = warp_id; block_index < blocks_per_row; block_index += 2u) {
    const auto* block = q4k_blocks + row_base + static_cast<std::size_t>(block_index) * kQ4KBytes;
    float local_scale = 0.0f;
    float local_min = 0.0f;
    if (lane < 8u) {
      const std::uint16_t d_bits =
          static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8u));
      const std::uint16_t dmin_bits =
          static_cast<std::uint16_t>(block[2] | (static_cast<std::uint16_t>(block[3]) << 8u));
      const float d = fp16_bits_to_float(d_bits);
      const float dmin = fp16_bits_to_float(dmin_bits);
      std::uint8_t scale = 0;
      std::uint8_t min_value = 0;
      get_scale_min_k4(static_cast<int>(lane), block + 4, scale, min_value);
      local_scale = d * static_cast<float>(scale);
      local_min = dmin * static_cast<float>(min_value);
    }

    const std::uint32_t scale_index = group * 2u;
    const float scale_0 = __shfl_sync(kFullWarpMask, local_scale, scale_index);
    const float min_0 = __shfl_sync(kFullWarpMask, local_min, scale_index);
    const float scale_1 = __shfl_sync(kFullWarpMask, local_scale, scale_index + 1u);
    const float min_1 = __shfl_sync(kFullWarpMask, local_min, scale_index + 1u);

    const auto* qs = block + kQ4KPayloadOffset;
    const auto* packed_words =
        reinterpret_cast<const std::uint32_t*>(qs + group * 32u);
    const std::uint32_t packed4 = packed_words[group_lane];
    const std::uint32_t col_base =
        block_index * kQ4KValues + group * 64u + group_lane * 4u;
    const auto act0 = *reinterpret_cast<const float4*>(activation + col_base);
    const auto act1 = *reinterpret_cast<const float4*>(activation + col_base + 32u);

    const std::uint8_t packed0 = static_cast<std::uint8_t>(packed4 & 0xffu);
    const std::uint8_t packed1 = static_cast<std::uint8_t>((packed4 >> 8u) & 0xffu);
    const std::uint8_t packed2 = static_cast<std::uint8_t>((packed4 >> 16u) & 0xffu);
    const std::uint8_t packed3 = static_cast<std::uint8_t>((packed4 >> 24u) & 0xffu);

    sum += (scale_0 * static_cast<float>(packed0 & 0x0fu) - min_0) * act0.x;
    sum += (scale_1 * static_cast<float>(packed0 >> 4u) - min_1) * act1.x;
    sum += (scale_0 * static_cast<float>(packed1 & 0x0fu) - min_0) * act0.y;
    sum += (scale_1 * static_cast<float>(packed1 >> 4u) - min_1) * act1.y;
    sum += (scale_0 * static_cast<float>(packed2 & 0x0fu) - min_0) * act0.z;
    sum += (scale_1 * static_cast<float>(packed2 >> 4u) - min_1) * act1.z;
    sum += (scale_0 * static_cast<float>(packed3 & 0x0fu) - min_0) * act0.w;
    sum += (scale_1 * static_cast<float>(packed3 >> 4u) - min_1) * act1.w;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(kFullWarpMask, sum, offset);
  }

  if (lane == 0u) {
    warp_sums[warp_id] = sum;
  }
  __syncthreads();

  if (threadIdx.x == 0u) {
    const float value = warp_sums[0] + warp_sums[1];
    logits[row] = value;
    if (row < active_pages && kv_words > 0u && page_words > 0u) {
      const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
      const std::size_t offset = (token[0] + row) % page_words;
      const std::size_t pos = (page_base + offset) % kv_words;
      kv[pos] = value;
    }
  }
}

__global__ void q4k_matvec_decode_warp_broadcast_vec4x4_kernel(
    const std::uint8_t* q4k_blocks,
    const float* activation,
    float* logits,
    float* kv,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages) {
  __shared__ float warp_sums[4];

  constexpr std::uint32_t kQ4KValues = 256;
  constexpr std::uint32_t kQ4KBytes = 144;
  constexpr std::uint32_t kQ4KPayloadOffset = 16;
  constexpr unsigned kFullWarpMask = 0xffffffffu;

  const std::uint32_t row = blockIdx.x;
  const std::uint32_t warp_id = threadIdx.x >> 5u;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t group = lane >> 3u;
  const std::uint32_t group_lane = lane & 7u;
  const std::uint32_t blocks_per_row = cols / kQ4KValues;
  const std::size_t row_base =
      static_cast<std::size_t>(row) * blocks_per_row * kQ4KBytes;

  float sum = 0.0f;
  for (std::uint32_t block_index = warp_id; block_index < blocks_per_row; block_index += 4u) {
    const auto* block = q4k_blocks + row_base + static_cast<std::size_t>(block_index) * kQ4KBytes;
    float local_scale = 0.0f;
    float local_min = 0.0f;
    if (lane < 8u) {
      const std::uint16_t d_bits =
          static_cast<std::uint16_t>(block[0] | (static_cast<std::uint16_t>(block[1]) << 8u));
      const std::uint16_t dmin_bits =
          static_cast<std::uint16_t>(block[2] | (static_cast<std::uint16_t>(block[3]) << 8u));
      const float d = fp16_bits_to_float(d_bits);
      const float dmin = fp16_bits_to_float(dmin_bits);
      std::uint8_t scale = 0;
      std::uint8_t min_value = 0;
      get_scale_min_k4(static_cast<int>(lane), block + 4, scale, min_value);
      local_scale = d * static_cast<float>(scale);
      local_min = dmin * static_cast<float>(min_value);
    }

    const std::uint32_t scale_index = group * 2u;
    const float scale_0 = __shfl_sync(kFullWarpMask, local_scale, scale_index);
    const float min_0 = __shfl_sync(kFullWarpMask, local_min, scale_index);
    const float scale_1 = __shfl_sync(kFullWarpMask, local_scale, scale_index + 1u);
    const float min_1 = __shfl_sync(kFullWarpMask, local_min, scale_index + 1u);

    const auto* qs = block + kQ4KPayloadOffset;
    const auto* packed_words =
        reinterpret_cast<const std::uint32_t*>(qs + group * 32u);
    const std::uint32_t packed4 = packed_words[group_lane];
    const std::uint32_t col_base =
        block_index * kQ4KValues + group * 64u + group_lane * 4u;
    const auto act0 = *reinterpret_cast<const float4*>(activation + col_base);
    const auto act1 = *reinterpret_cast<const float4*>(activation + col_base + 32u);

    const std::uint8_t packed0 = static_cast<std::uint8_t>(packed4 & 0xffu);
    const std::uint8_t packed1 = static_cast<std::uint8_t>((packed4 >> 8u) & 0xffu);
    const std::uint8_t packed2 = static_cast<std::uint8_t>((packed4 >> 16u) & 0xffu);
    const std::uint8_t packed3 = static_cast<std::uint8_t>((packed4 >> 24u) & 0xffu);

    sum += (scale_0 * static_cast<float>(packed0 & 0x0fu) - min_0) * act0.x;
    sum += (scale_1 * static_cast<float>(packed0 >> 4u) - min_1) * act1.x;
    sum += (scale_0 * static_cast<float>(packed1 & 0x0fu) - min_0) * act0.y;
    sum += (scale_1 * static_cast<float>(packed1 >> 4u) - min_1) * act1.y;
    sum += (scale_0 * static_cast<float>(packed2 & 0x0fu) - min_0) * act0.z;
    sum += (scale_1 * static_cast<float>(packed2 >> 4u) - min_1) * act1.z;
    sum += (scale_0 * static_cast<float>(packed3 & 0x0fu) - min_0) * act0.w;
    sum += (scale_1 * static_cast<float>(packed3 >> 4u) - min_1) * act1.w;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(kFullWarpMask, sum, offset);
  }

  if (lane == 0u) {
    warp_sums[warp_id] = sum;
  }
  __syncthreads();

  if (threadIdx.x == 0u) {
    const float value = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
    logits[row] = value;
    if (row < active_pages && kv_words > 0u && page_words > 0u) {
      const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
      const std::size_t offset = (token[0] + row) % page_words;
      const std::size_t pos = (page_base + offset) % kv_words;
      kv[pos] = value;
    }
  }
}

__global__ void q4k_matvec_decode_predecoded_vec4x4_kernel(
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
    std::uint32_t active_pages) {
  __shared__ float warp_sums[4];

  constexpr std::uint32_t kQ4KValues = 256;
  constexpr std::uint32_t kQ4KBytes = 144;
  constexpr std::uint32_t kQ4KPayloadOffset = 16;
  constexpr unsigned kFullWarpMask = 0xffffffffu;

  const std::uint32_t row = blockIdx.x;
  const std::uint32_t warp_id = threadIdx.x >> 5u;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t group = lane >> 3u;
  const std::uint32_t group_lane = lane & 7u;
  const std::uint32_t blocks_per_row = cols / kQ4KValues;
  const std::size_t row_base =
      static_cast<std::size_t>(row) * blocks_per_row * kQ4KBytes;

  float sum = 0.0f;
  for (std::uint32_t block_index = warp_id; block_index < blocks_per_row; block_index += 4u) {
    const auto* block = q4k_blocks + row_base + static_cast<std::size_t>(block_index) * kQ4KBytes;
    float local_scale = 0.0f;
    float local_min = 0.0f;
    if (lane < 8u) {
      const std::size_t pair_base =
          ((static_cast<std::size_t>(row) * blocks_per_row + block_index) * 8u + lane) * 2u;
      local_scale = q4k_scale_min_pairs[pair_base];
      local_min = q4k_scale_min_pairs[pair_base + 1u];
    }

    const std::uint32_t scale_index = group * 2u;
    const float scale_0 = __shfl_sync(kFullWarpMask, local_scale, scale_index);
    const float min_0 = __shfl_sync(kFullWarpMask, local_min, scale_index);
    const float scale_1 = __shfl_sync(kFullWarpMask, local_scale, scale_index + 1u);
    const float min_1 = __shfl_sync(kFullWarpMask, local_min, scale_index + 1u);

    const auto* qs = block + kQ4KPayloadOffset;
    const auto* packed_words =
        reinterpret_cast<const std::uint32_t*>(qs + group * 32u);
    const std::uint32_t packed4 = packed_words[group_lane];
    const std::uint32_t col_base =
        block_index * kQ4KValues + group * 64u + group_lane * 4u;
    const auto act0 = *reinterpret_cast<const float4*>(activation + col_base);
    const auto act1 = *reinterpret_cast<const float4*>(activation + col_base + 32u);

    const std::uint8_t packed0 = static_cast<std::uint8_t>(packed4 & 0xffu);
    const std::uint8_t packed1 = static_cast<std::uint8_t>((packed4 >> 8u) & 0xffu);
    const std::uint8_t packed2 = static_cast<std::uint8_t>((packed4 >> 16u) & 0xffu);
    const std::uint8_t packed3 = static_cast<std::uint8_t>((packed4 >> 24u) & 0xffu);

    sum += (scale_0 * static_cast<float>(packed0 & 0x0fu) - min_0) * act0.x;
    sum += (scale_1 * static_cast<float>(packed0 >> 4u) - min_1) * act1.x;
    sum += (scale_0 * static_cast<float>(packed1 & 0x0fu) - min_0) * act0.y;
    sum += (scale_1 * static_cast<float>(packed1 >> 4u) - min_1) * act1.y;
    sum += (scale_0 * static_cast<float>(packed2 & 0x0fu) - min_0) * act0.z;
    sum += (scale_1 * static_cast<float>(packed2 >> 4u) - min_1) * act1.z;
    sum += (scale_0 * static_cast<float>(packed3 & 0x0fu) - min_0) * act0.w;
    sum += (scale_1 * static_cast<float>(packed3 >> 4u) - min_1) * act1.w;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(kFullWarpMask, sum, offset);
  }

  if (lane == 0u) {
    warp_sums[warp_id] = sum;
  }
  __syncthreads();

  if (threadIdx.x == 0u) {
    const float value = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
    logits[row] = value;
    if (row < active_pages && kv_words > 0u && page_words > 0u) {
      const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
      const std::size_t offset = (token[0] + row) % page_words;
      const std::size_t pos = (page_base + offset) % kv_words;
      kv[pos] = value;
    }
  }
}

__global__ void q4k_matvec_decode_split_predecoded_vec4x4_kernel(
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
    std::uint32_t active_pages) {
  __shared__ float warp_sums[4];

  constexpr std::uint32_t kQ4KValues = 256;
  constexpr std::uint32_t kQ4KPayloadBytes = 128;
  constexpr unsigned kFullWarpMask = 0xffffffffu;

  const std::uint32_t row = blockIdx.x;
  const std::uint32_t warp_id = threadIdx.x >> 5u;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t group = lane >> 3u;
  const std::uint32_t group_lane = lane & 7u;
  const std::uint32_t blocks_per_row = cols / kQ4KValues;
  const std::size_t row_base =
      static_cast<std::size_t>(row) * blocks_per_row * kQ4KPayloadBytes;

  float sum = 0.0f;
  for (std::uint32_t block_index = warp_id; block_index < blocks_per_row; block_index += 4u) {
    float local_scale = 0.0f;
    float local_min = 0.0f;
    if (lane < 8u) {
      const std::size_t pair_base =
          ((static_cast<std::size_t>(row) * blocks_per_row + block_index) * 8u + lane) * 2u;
      local_scale = q4k_scale_min_pairs[pair_base];
      local_min = q4k_scale_min_pairs[pair_base + 1u];
    }

    const std::uint32_t scale_index = group * 2u;
    const float scale_0 = __shfl_sync(kFullWarpMask, local_scale, scale_index);
    const float min_0 = __shfl_sync(kFullWarpMask, local_min, scale_index);
    const float scale_1 = __shfl_sync(kFullWarpMask, local_scale, scale_index + 1u);
    const float min_1 = __shfl_sync(kFullWarpMask, local_min, scale_index + 1u);

    const auto* payload_block =
        q4k_payload + row_base + static_cast<std::size_t>(block_index) * kQ4KPayloadBytes;
    const auto* packed_words =
        reinterpret_cast<const std::uint32_t*>(payload_block + group * 32u);
    const std::uint32_t packed4 = packed_words[group_lane];
    const std::uint32_t col_base =
        block_index * kQ4KValues + group * 64u + group_lane * 4u;
    const auto act0 = *reinterpret_cast<const float4*>(activation + col_base);
    const auto act1 = *reinterpret_cast<const float4*>(activation + col_base + 32u);

    const std::uint8_t packed0 = static_cast<std::uint8_t>(packed4 & 0xffu);
    const std::uint8_t packed1 = static_cast<std::uint8_t>((packed4 >> 8u) & 0xffu);
    const std::uint8_t packed2 = static_cast<std::uint8_t>((packed4 >> 16u) & 0xffu);
    const std::uint8_t packed3 = static_cast<std::uint8_t>((packed4 >> 24u) & 0xffu);

    sum += (scale_0 * static_cast<float>(packed0 & 0x0fu) - min_0) * act0.x;
    sum += (scale_1 * static_cast<float>(packed0 >> 4u) - min_1) * act1.x;
    sum += (scale_0 * static_cast<float>(packed1 & 0x0fu) - min_0) * act0.y;
    sum += (scale_1 * static_cast<float>(packed1 >> 4u) - min_1) * act1.y;
    sum += (scale_0 * static_cast<float>(packed2 & 0x0fu) - min_0) * act0.z;
    sum += (scale_1 * static_cast<float>(packed2 >> 4u) - min_1) * act1.z;
    sum += (scale_0 * static_cast<float>(packed3 & 0x0fu) - min_0) * act0.w;
    sum += (scale_1 * static_cast<float>(packed3 >> 4u) - min_1) * act1.w;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(kFullWarpMask, sum, offset);
  }

  if (lane == 0u) {
    warp_sums[warp_id] = sum;
  }
  __syncthreads();

  if (threadIdx.x == 0u) {
    const float value = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
    logits[row] = value;
    if (row < active_pages && kv_words > 0u && page_words > 0u) {
      const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
      const std::size_t offset = (token[0] + row) % page_words;
      const std::size_t pos = (page_base + offset) % kv_words;
      kv[pos] = value;
    }
  }
}

__global__ void q4k_matvec_decode_split_half_vec4x4_kernel(
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
    std::uint32_t active_pages) {
  __shared__ float warp_sums[4];

  constexpr std::uint32_t kQ4KValues = 256;
  constexpr std::uint32_t kQ4KPayloadBytes = 128;
  constexpr unsigned kFullWarpMask = 0xffffffffu;

  const std::uint32_t row = blockIdx.x;
  const std::uint32_t warp_id = threadIdx.x >> 5u;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t group = lane >> 3u;
  const std::uint32_t group_lane = lane & 7u;
  const std::uint32_t blocks_per_row = cols / kQ4KValues;
  const std::size_t row_base =
      static_cast<std::size_t>(row) * blocks_per_row * kQ4KPayloadBytes;

  float sum = 0.0f;
  for (std::uint32_t block_index = warp_id; block_index < blocks_per_row; block_index += 4u) {
    float local_scale = 0.0f;
    float local_min = 0.0f;
    if (lane < 8u) {
      const std::size_t pair_base =
          ((static_cast<std::size_t>(row) * blocks_per_row + block_index) * 8u + lane) * 2u;
      local_scale =
          __half2float(*reinterpret_cast<const __half*>(q4k_scale_min_half_pairs + pair_base));
      local_min =
          __half2float(*reinterpret_cast<const __half*>(q4k_scale_min_half_pairs + pair_base + 1u));
    }

    const std::uint32_t scale_index = group * 2u;
    const float scale_0 = __shfl_sync(kFullWarpMask, local_scale, scale_index);
    const float min_0 = __shfl_sync(kFullWarpMask, local_min, scale_index);
    const float scale_1 = __shfl_sync(kFullWarpMask, local_scale, scale_index + 1u);
    const float min_1 = __shfl_sync(kFullWarpMask, local_min, scale_index + 1u);

    const auto* payload_block =
        q4k_payload + row_base + static_cast<std::size_t>(block_index) * kQ4KPayloadBytes;
    const auto* packed_words =
        reinterpret_cast<const std::uint32_t*>(payload_block + group * 32u);
    const std::uint32_t packed4 = packed_words[group_lane];
    const std::uint32_t col_base =
        block_index * kQ4KValues + group * 64u + group_lane * 4u;
    const auto act0 = *reinterpret_cast<const float4*>(activation + col_base);
    const auto act1 = *reinterpret_cast<const float4*>(activation + col_base + 32u);

    const std::uint8_t packed0 = static_cast<std::uint8_t>(packed4 & 0xffu);
    const std::uint8_t packed1 = static_cast<std::uint8_t>((packed4 >> 8u) & 0xffu);
    const std::uint8_t packed2 = static_cast<std::uint8_t>((packed4 >> 16u) & 0xffu);
    const std::uint8_t packed3 = static_cast<std::uint8_t>((packed4 >> 24u) & 0xffu);

    sum += (scale_0 * static_cast<float>(packed0 & 0x0fu) - min_0) * act0.x;
    sum += (scale_1 * static_cast<float>(packed0 >> 4u) - min_1) * act1.x;
    sum += (scale_0 * static_cast<float>(packed1 & 0x0fu) - min_0) * act0.y;
    sum += (scale_1 * static_cast<float>(packed1 >> 4u) - min_1) * act1.y;
    sum += (scale_0 * static_cast<float>(packed2 & 0x0fu) - min_0) * act0.z;
    sum += (scale_1 * static_cast<float>(packed2 >> 4u) - min_1) * act1.z;
    sum += (scale_0 * static_cast<float>(packed3 & 0x0fu) - min_0) * act0.w;
    sum += (scale_1 * static_cast<float>(packed3 >> 4u) - min_1) * act1.w;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(kFullWarpMask, sum, offset);
  }

  if (lane == 0u) {
    warp_sums[warp_id] = sum;
  }
  __syncthreads();

  if (threadIdx.x == 0u) {
    const float value = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
    logits[row] = value;
    if (row < active_pages && kv_words > 0u && page_words > 0u) {
      const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
      const std::size_t offset = (token[0] + row) % page_words;
      const std::size_t pos = (page_base + offset) % kv_words;
      kv[pos] = value;
    }
  }
}

__global__ void q4k_matvec_decode_split_compact_vec4x4_kernel(
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
    std::uint32_t active_pages) {
  __shared__ float warp_sums[4];

  constexpr std::uint32_t kQ4KValues = 256;
  constexpr std::uint32_t kQ4KPayloadBytes = 128;
  constexpr std::uint32_t kQ4KCompactMetaBytes = 20;
  constexpr unsigned kFullWarpMask = 0xffffffffu;

  const std::uint32_t row = blockIdx.x;
  const std::uint32_t warp_id = threadIdx.x >> 5u;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t group = lane >> 3u;
  const std::uint32_t group_lane = lane & 7u;
  const std::uint32_t blocks_per_row = cols / kQ4KValues;
  const std::size_t row_base =
      static_cast<std::size_t>(row) * blocks_per_row * kQ4KPayloadBytes;

  float sum = 0.0f;
  for (std::uint32_t block_index = warp_id; block_index < blocks_per_row; block_index += 4u) {
    float local_scale = 0.0f;
    float local_min = 0.0f;
    const auto* meta = q4k_compact_meta +
        (static_cast<std::size_t>(row) * blocks_per_row + block_index) *
            kQ4KCompactMetaBytes;
    float local_d = 0.0f;
    float local_dmin = 0.0f;
    if (lane == 0u) {
      const std::uint16_t d_bits =
          static_cast<std::uint16_t>(meta[0] | (static_cast<std::uint16_t>(meta[1]) << 8u));
      const std::uint16_t dmin_bits =
          static_cast<std::uint16_t>(meta[2] | (static_cast<std::uint16_t>(meta[3]) << 8u));
      local_d = fp16_bits_to_float(d_bits);
      local_dmin = fp16_bits_to_float(dmin_bits);
    }
    const float d = __shfl_sync(kFullWarpMask, local_d, 0u);
    const float dmin = __shfl_sync(kFullWarpMask, local_dmin, 0u);
    if (lane < 8u) {
      local_scale = d * static_cast<float>(meta[4u + lane]);
      local_min = dmin * static_cast<float>(meta[12u + lane]);
    }

    const std::uint32_t scale_index = group * 2u;
    const float scale_0 = __shfl_sync(kFullWarpMask, local_scale, scale_index);
    const float min_0 = __shfl_sync(kFullWarpMask, local_min, scale_index);
    const float scale_1 = __shfl_sync(kFullWarpMask, local_scale, scale_index + 1u);
    const float min_1 = __shfl_sync(kFullWarpMask, local_min, scale_index + 1u);

    const auto* payload_block =
        q4k_payload + row_base + static_cast<std::size_t>(block_index) * kQ4KPayloadBytes;
    const auto* packed_words =
        reinterpret_cast<const std::uint32_t*>(payload_block + group * 32u);
    const std::uint32_t packed4 = packed_words[group_lane];
    const std::uint32_t col_base =
        block_index * kQ4KValues + group * 64u + group_lane * 4u;
    const auto act0 = *reinterpret_cast<const float4*>(activation + col_base);
    const auto act1 = *reinterpret_cast<const float4*>(activation + col_base + 32u);

    const std::uint8_t packed0 = static_cast<std::uint8_t>(packed4 & 0xffu);
    const std::uint8_t packed1 = static_cast<std::uint8_t>((packed4 >> 8u) & 0xffu);
    const std::uint8_t packed2 = static_cast<std::uint8_t>((packed4 >> 16u) & 0xffu);
    const std::uint8_t packed3 = static_cast<std::uint8_t>((packed4 >> 24u) & 0xffu);

    sum += (scale_0 * static_cast<float>(packed0 & 0x0fu) - min_0) * act0.x;
    sum += (scale_1 * static_cast<float>(packed0 >> 4u) - min_1) * act1.x;
    sum += (scale_0 * static_cast<float>(packed1 & 0x0fu) - min_0) * act0.y;
    sum += (scale_1 * static_cast<float>(packed1 >> 4u) - min_1) * act1.y;
    sum += (scale_0 * static_cast<float>(packed2 & 0x0fu) - min_0) * act0.z;
    sum += (scale_1 * static_cast<float>(packed2 >> 4u) - min_1) * act1.z;
    sum += (scale_0 * static_cast<float>(packed3 & 0x0fu) - min_0) * act0.w;
    sum += (scale_1 * static_cast<float>(packed3 >> 4u) - min_1) * act1.w;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(kFullWarpMask, sum, offset);
  }

  if (lane == 0u) {
    warp_sums[warp_id] = sum;
  }
  __syncthreads();

  if (threadIdx.x == 0u) {
    const float value = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
    logits[row] = value;
    if (row < active_pages && kv_words > 0u && page_words > 0u) {
      const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
      const std::size_t offset = (token[0] + row) % page_words;
      const std::size_t pos = (page_base + offset) % kv_words;
      kv[pos] = value;
    }
  }
}

__global__ void q4k_matvec_decode_split_native_vec4x4_kernel(
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
    std::uint32_t active_pages) {
  __shared__ float warp_sums[4];

  constexpr std::uint32_t kQ4KValues = 256;
  constexpr std::uint32_t kQ4KPayloadBytes = 128;
  constexpr std::uint32_t kQ4KNativeMetaBytes = 16;
  constexpr unsigned kFullWarpMask = 0xffffffffu;

  const std::uint32_t row = blockIdx.x;
  const std::uint32_t warp_id = threadIdx.x >> 5u;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t group = lane >> 3u;
  const std::uint32_t group_lane = lane & 7u;
  const std::uint32_t blocks_per_row = cols / kQ4KValues;
  const std::size_t row_base =
      static_cast<std::size_t>(row) * blocks_per_row * kQ4KPayloadBytes;

  float sum = 0.0f;
  for (std::uint32_t block_index = warp_id; block_index < blocks_per_row; block_index += 4u) {
    const auto* meta = q4k_native_meta +
        (static_cast<std::size_t>(row) * blocks_per_row + block_index) * kQ4KNativeMetaBytes;
    float local_scale = 0.0f;
    float local_min = 0.0f;
    if (lane < 8u) {
      const std::uint16_t d_bits =
          static_cast<std::uint16_t>(meta[0] | (static_cast<std::uint16_t>(meta[1]) << 8u));
      const std::uint16_t dmin_bits =
          static_cast<std::uint16_t>(meta[2] | (static_cast<std::uint16_t>(meta[3]) << 8u));
      const float d = fp16_bits_to_float(d_bits);
      const float dmin = fp16_bits_to_float(dmin_bits);
      std::uint8_t scale = 0;
      std::uint8_t min_value = 0;
      get_scale_min_k4(static_cast<int>(lane), meta + 4, scale, min_value);
      local_scale = d * static_cast<float>(scale);
      local_min = dmin * static_cast<float>(min_value);
    }

    const std::uint32_t scale_index = group * 2u;
    const float scale_0 = __shfl_sync(kFullWarpMask, local_scale, scale_index);
    const float min_0 = __shfl_sync(kFullWarpMask, local_min, scale_index);
    const float scale_1 = __shfl_sync(kFullWarpMask, local_scale, scale_index + 1u);
    const float min_1 = __shfl_sync(kFullWarpMask, local_min, scale_index + 1u);

    const auto* payload_block =
        q4k_payload + row_base + static_cast<std::size_t>(block_index) * kQ4KPayloadBytes;
    const auto* packed_words =
        reinterpret_cast<const std::uint32_t*>(payload_block + group * 32u);
    const std::uint32_t packed4 = packed_words[group_lane];
    const std::uint32_t col_base =
        block_index * kQ4KValues + group * 64u + group_lane * 4u;
    const auto act0 = *reinterpret_cast<const float4*>(activation + col_base);
    const auto act1 = *reinterpret_cast<const float4*>(activation + col_base + 32u);

    const std::uint8_t packed0 = static_cast<std::uint8_t>(packed4 & 0xffu);
    const std::uint8_t packed1 = static_cast<std::uint8_t>((packed4 >> 8u) & 0xffu);
    const std::uint8_t packed2 = static_cast<std::uint8_t>((packed4 >> 16u) & 0xffu);
    const std::uint8_t packed3 = static_cast<std::uint8_t>((packed4 >> 24u) & 0xffu);

    sum += (scale_0 * static_cast<float>(packed0 & 0x0fu) - min_0) * act0.x;
    sum += (scale_1 * static_cast<float>(packed0 >> 4u) - min_1) * act1.x;
    sum += (scale_0 * static_cast<float>(packed1 & 0x0fu) - min_0) * act0.y;
    sum += (scale_1 * static_cast<float>(packed1 >> 4u) - min_1) * act1.y;
    sum += (scale_0 * static_cast<float>(packed2 & 0x0fu) - min_0) * act0.z;
    sum += (scale_1 * static_cast<float>(packed2 >> 4u) - min_1) * act1.z;
    sum += (scale_0 * static_cast<float>(packed3 & 0x0fu) - min_0) * act0.w;
    sum += (scale_1 * static_cast<float>(packed3 >> 4u) - min_1) * act1.w;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(kFullWarpMask, sum, offset);
  }

  if (lane == 0u) {
    warp_sums[warp_id] = sum;
  }
  __syncthreads();

  if (threadIdx.x == 0u) {
    const float value = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
    logits[row] = value;
    if (row < active_pages && kv_words > 0u && page_words > 0u) {
      const std::size_t page_base = static_cast<std::size_t>(row) * page_words;
      const std::size_t offset = (token[0] + row) % page_words;
      const std::size_t pos = (page_base + offset) % kv_words;
      kv[pos] = value;
    }
  }
}

__global__ void argmax_token_kernel(
    const float* logits,
    std::uint32_t* token,
    std::uint32_t rows) {
  __shared__ float values[256];
  __shared__ std::uint32_t indices[256];

  const std::uint32_t tid = threadIdx.x;
  float best_value = -3.4028234663852886e38f;
  std::uint32_t best_index = 0u;
  for (std::uint32_t row = tid; row < rows; row += blockDim.x) {
    const float value = logits[row];
    if (value > best_value || (value == best_value && row < best_index)) {
      best_value = value;
      best_index = row;
    }
  }

  values[tid] = best_value;
  indices[tid] = best_index;
  __syncthreads();

  for (std::uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      const float other_value = values[tid + stride];
      const std::uint32_t other_index = indices[tid + stride];
      if (other_value > values[tid] ||
          (other_value == values[tid] && other_index < indices[tid])) {
        values[tid] = other_value;
        indices[tid] = other_index;
      }
    }
    __syncthreads();
  }

  if (tid == 0u) {
    token[0] = (indices[0] + token[0] + 1u) % 32000u;
  }
}

__global__ void output_token_sample_kernel(
    const float* logits,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t vocab_size,
    std::uint32_t token_offset) {
  __shared__ float values[256];
  __shared__ std::uint32_t indices[256];

  const std::uint32_t tid = threadIdx.x;
  float best_value = -3.4028234663852886e38f;
  std::uint32_t best_index = 0u;
  for (std::uint32_t row = tid; row < rows; row += blockDim.x) {
    const float value = logits[row];
    if (value > best_value || (value == best_value && row < best_index)) {
      best_value = value;
      best_index = row;
    }
  }

  values[tid] = best_value;
  indices[tid] = best_index;
  __syncthreads();

  for (std::uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (tid < stride) {
      const float other_value = values[tid + stride];
      const std::uint32_t other_index = indices[tid + stride];
      if (other_value > values[tid] ||
          (other_value == values[tid] && other_index < indices[tid])) {
        values[tid] = other_value;
        indices[tid] = other_index;
      }
    }
    __syncthreads();
  }

  if (tid == 0u) {
    token[0] = (indices[0] + token_offset) % vocab_size;
  }
}

}  // namespace

extern "C" cudaError_t rtxllm_launch_synthetic_decode(
    const std::uint8_t* weights,
    float* kv,
    std::uint32_t* token,
    std::size_t weight_bytes,
    std::size_t kv_words,
    std::uint32_t step,
    int touch_stride,
    cudaStream_t stream) {
  synthetic_decode_kernel<<<256, 128, 0, stream>>>(
      weights, kv, token, weight_bytes, kv_words, step, touch_stride);
  return cudaGetLastError();
}

extern "C" cudaError_t rtxllm_launch_synthetic_decode_stateful(
    const std::uint8_t* weights,
    float* kv,
    std::uint32_t* token,
    std::size_t weight_bytes,
    std::size_t kv_words,
    std::uint32_t page_words,
    std::uint32_t active_pages,
    int touch_stride,
    cudaStream_t stream) {
  synthetic_decode_stateful_kernel<<<256, 128, 0, stream>>>(
      weights, kv, token, weight_bytes, kv_words, page_words, active_pages, touch_stride);
  return cudaGetLastError();
}

extern "C" cudaError_t rtxllm_launch_activation_feedback(
    const float* logits,
    float* activation,
    const std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t stage,
    cudaStream_t stream) {
  constexpr int kBlockThreads = 256;
  const std::uint32_t blocks = (cols + kBlockThreads - 1u) / kBlockThreads;
  activation_feedback_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      logits, activation, token, rows, cols, stage);
  return cudaGetLastError();
}

extern "C" cudaError_t rtxllm_launch_rmsnorm_feedback(
    const float* logits,
    float* activation,
    std::uint32_t rows,
    std::uint32_t cols,
    float epsilon,
    cudaStream_t stream) {
  constexpr int kBlockThreads = 256;
  const std::uint32_t blocks = (cols + kBlockThreads - 1u) / kBlockThreads;
  rmsnorm_feedback_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      logits, activation, rows, cols, epsilon);
  return cudaGetLastError();
}

extern "C" cudaError_t rtxllm_launch_silu_rmsnorm_feedback(
    const float* logits,
    float* activation,
    std::uint32_t rows,
    std::uint32_t cols,
    float epsilon,
    cudaStream_t stream) {
  constexpr int kBlockThreads = 256;
  const std::uint32_t blocks = (cols + kBlockThreads - 1u) / kBlockThreads;
  silu_rmsnorm_feedback_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      logits, activation, rows, cols, epsilon);
  return cudaGetLastError();
}

extern "C" cudaError_t rtxllm_launch_rmsnorm_feedback_with_silu_cache(
    const float* logits,
    float* activation,
    float* silu_cache,
    std::uint32_t rows,
    std::uint32_t cols,
    float epsilon,
    cudaStream_t stream) {
  constexpr int kBlockThreads = 256;
  const std::uint32_t blocks = (cols + kBlockThreads - 1u) / kBlockThreads;
  rmsnorm_feedback_with_silu_cache_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      logits, activation, silu_cache, rows, cols, epsilon);
  return cudaGetLastError();
}

extern "C" cudaError_t rtxllm_launch_gated_silu_rmsnorm_feedback(
    const float* logits,
    const float* silu_cache,
    float* activation,
    std::uint32_t rows,
    std::uint32_t cache_cols,
    std::uint32_t cols,
    float epsilon,
    cudaStream_t stream) {
  constexpr int kBlockThreads = 256;
  const std::uint32_t blocks = (cols + kBlockThreads - 1u) / kBlockThreads;
  gated_silu_rmsnorm_feedback_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      logits, silu_cache, activation, rows, cache_cols, cols, epsilon);
  return cudaGetLastError();
}

extern "C" cudaError_t rtxllm_launch_rmsnorm_silu_cache(
    const float* logits,
    float* silu_cache,
    std::uint32_t rows,
    std::uint32_t cache_cols,
    float epsilon,
    std::uint32_t accumulate,
    cudaStream_t stream) {
  if (cache_cols == 0u) {
    return cudaErrorInvalidValue;
  }
  constexpr int kBlockThreads = 256;
  const std::uint32_t blocks = (cache_cols + kBlockThreads - 1u) / kBlockThreads;
  rmsnorm_silu_cache_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      logits, silu_cache, rows, cache_cols, epsilon, accumulate);
  return cudaGetLastError();
}

extern "C" cudaError_t rtxllm_launch_rmsnorm_cache(
    const float* logits,
    float* cache,
    std::uint32_t rows,
    std::uint32_t cache_cols,
    float epsilon,
    std::uint32_t accumulate,
    cudaStream_t stream) {
  if (cache_cols == 0u) {
    return cudaErrorInvalidValue;
  }
  constexpr int kBlockThreads = 256;
  const std::uint32_t blocks = (cache_cols + kBlockThreads - 1u) / kBlockThreads;
  rmsnorm_cache_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      logits, cache, rows, cache_cols, epsilon, accumulate);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  if (cache_cols == 0u || output_cols == 0u) {
    return cudaErrorInvalidValue;
  }
  constexpr int kBlockThreads = 256;
  const std::uint32_t blocks = (output_cols + kBlockThreads - 1u) / kBlockThreads;
  rmsnorm_cache_and_gated_product_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      logits,
      gate_cache,
      up_cache,
      activation,
      rows,
      cache_cols,
      output_cols,
      epsilon,
      accumulate);
  return cudaGetLastError();
}

extern "C" cudaError_t rtxllm_launch_phase_scratch_digest(
    const float* activation,
    const float* logits,
    float* scratch,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t scratch_values,
    std::uint32_t phase_id,
    cudaStream_t stream) {
  if (activation == nullptr || logits == nullptr || scratch == nullptr ||
      scratch_values == 0u) {
    return cudaErrorInvalidValue;
  }
  constexpr int kBlockThreads = 256;
  const std::uint32_t blocks =
      (scratch_values + kBlockThreads - 1u) / kBlockThreads;
  phase_scratch_digest_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      activation,
      logits,
      scratch,
      rows,
      cols,
      scratch_values,
      phase_id);
  return cudaGetLastError();
}

extern "C" cudaError_t rtxllm_launch_attention_qkv_scratch(
    const float* logits,
    float* scratch,
    float* activation,
    std::uint32_t rows,
    std::uint32_t scratch_values,
    std::uint32_t activation_cols,
    float residual_scale,
    cudaStream_t stream) {
  if (rows == 0u || scratch_values < 3u || activation_cols == 0u) {
    return cudaSuccess;
  }
  const std::uint32_t qkv_rows =
      std::min(rows, static_cast<std::uint32_t>(scratch_values / 3u));
  if (qkv_rows == 0u) {
    return cudaSuccess;
  }
  const std::uint64_t qkv_value_count =
      static_cast<std::uint64_t>(qkv_rows) * 3ull;
  const std::uint32_t qkv_values =
      qkv_value_count > std::numeric_limits<std::uint32_t>::max()
      ? std::numeric_limits<std::uint32_t>::max()
      : static_cast<std::uint32_t>(qkv_value_count);
  constexpr int kBlockThreads = 256;
  const std::uint32_t total_values = std::max(qkv_values, activation_cols);
  const std::uint32_t blocks =
      (total_values + kBlockThreads - 1u) / kBlockThreads;
  attention_qkv_scratch_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      logits,
      scratch,
      activation,
      rows,
      qkv_rows,
      qkv_values,
      activation_cols,
      residual_scale);
  return cudaGetLastError();
}

extern "C" cudaError_t rtxllm_launch_attention_qkv_reduce(
    float* scratch,
    float* activation,
    std::uint32_t rows,
    std::uint32_t scratch_values,
    std::uint32_t activation_cols,
    float reduce_scale,
    cudaStream_t stream) {
  if (rows == 0u || scratch_values < 3u || activation_cols == 0u) {
    return cudaSuccess;
  }
  if (scratch == nullptr || activation == nullptr) {
    return cudaErrorInvalidValue;
  }
  const std::uint32_t qkv_rows =
      std::min(rows, static_cast<std::uint32_t>(scratch_values / 3u));
  if (qkv_rows == 0u) {
    return cudaSuccess;
  }
  constexpr int kBlockThreads = 256;
  const std::uint32_t blocks =
      (activation_cols + kBlockThreads - 1u) / kBlockThreads;
  attention_qkv_reduce_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      scratch,
      activation,
      qkv_rows,
      activation_cols,
      reduce_scale);
  return cudaGetLastError();
}

extern "C" cudaError_t rtxllm_launch_attention_qkv_window(
    float* scratch,
    float* activation,
    std::uint32_t rows,
    std::uint32_t scratch_values,
    std::uint32_t activation_cols,
    std::uint32_t window_size,
    float softmax_scale,
    float residual_scale,
    cudaStream_t stream) {
  if (rows == 0u || scratch_values < 3u || activation_cols == 0u ||
      window_size == 0u) {
    return cudaSuccess;
  }
  if (scratch == nullptr || activation == nullptr) {
    return cudaErrorInvalidValue;
  }
  const std::uint32_t qkv_rows =
      std::min(rows, static_cast<std::uint32_t>(scratch_values / 3u));
  if (qkv_rows == 0u) {
    return cudaSuccess;
  }
  constexpr int kBlockThreads = 256;
  const std::uint32_t blocks =
      (activation_cols + kBlockThreads - 1u) / kBlockThreads;
  attention_qkv_window_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      scratch,
      activation,
      qkv_rows,
      activation_cols,
      window_size,
      softmax_scale,
      residual_scale);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  if (rows == 0u || scratch_values < 3u || activation_cols == 0u ||
      head_count == 0u || window_size == 0u) {
    return cudaSuccess;
  }
  if (scratch == nullptr || activation == nullptr) {
    return cudaErrorInvalidValue;
  }
  const std::uint32_t qkv_rows =
      std::min(rows, static_cast<std::uint32_t>(scratch_values / 3u));
  if (qkv_rows == 0u) {
    return cudaSuccess;
  }
  constexpr int kBlockThreads = 256;
  const std::uint32_t blocks =
      (activation_cols + kBlockThreads - 1u) / kBlockThreads;
  attention_qkv_head_window_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      scratch,
      activation,
      qkv_rows,
      activation_cols,
      head_count,
      window_size,
      softmax_scale,
      residual_scale);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  if (rows == 0u || scratch_values < 3u || activation_cols == 0u ||
      head_count == 0u || head_dim == 0u || window_size == 0u) {
    return cudaSuccess;
  }
  if (scratch == nullptr || activation == nullptr) {
    return cudaErrorInvalidValue;
  }
  const std::uint32_t qkv_rows =
      std::min(rows, static_cast<std::uint32_t>(scratch_values / 3u));
  if (qkv_rows == 0u) {
    return cudaSuccess;
  }
  constexpr int kBlockThreads = 256;
  const std::uint32_t blocks =
      (activation_cols + kBlockThreads - 1u) / kBlockThreads;
  attention_qkv_head_dim_window_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      scratch,
      activation,
      qkv_rows,
      activation_cols,
      head_count,
      head_dim,
      window_size,
      softmax_scale,
      residual_scale);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  if (rows == 0u || scratch_values < 3u || activation_cols == 0u ||
      head_count == 0u || head_dim == 0u || window_size == 0u) {
    return cudaSuccess;
  }
  if (scratch == nullptr || activation == nullptr || head_dim > 32u ||
      window_size > 16u) {
    return cudaErrorInvalidValue;
  }
  const std::uint32_t qkv_rows =
      std::min(rows, static_cast<std::uint32_t>(scratch_values / 3u));
  const std::uint32_t possible_heads = qkv_rows / head_dim;
  if (possible_heads == 0u) {
    return cudaSuccess;
  }
  const std::uint32_t active_heads =
      std::min(head_count, possible_heads);
  const std::uint32_t rows_per_head = qkv_rows / active_heads;
  const std::uint32_t contexts_per_head = rows_per_head / head_dim;
  if (contexts_per_head == 0u) {
    return cudaSuccess;
  }
  const std::uint32_t contexts_for_activation =
      (activation_cols + active_heads * head_dim - 1u) /
      (active_heads * head_dim);
  const std::uint32_t active_contexts =
      std::min(contexts_for_activation, contexts_per_head);
  if (active_contexts == 0u) {
    return cudaSuccess;
  }
  constexpr int kBlockThreads = 32;
  const std::uint32_t blocks = active_heads * active_contexts;
  attention_qkv_head_tile_window_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      scratch,
      activation,
      qkv_rows,
      activation_cols,
      head_count,
      head_dim,
      window_size,
      softmax_scale,
      residual_scale);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  if (rows == 0u || scratch_values < 3u || activation_cols == 0u ||
      head_count == 0u || head_dim == 0u || window_size == 0u ||
      contexts_per_block == 0u) {
    return cudaSuccess;
  }
  if (scratch == nullptr || activation == nullptr || head_dim > 32u ||
      window_size > 16u || contexts_per_block > 4u ||
      head_dim * contexts_per_block > 256u) {
    return cudaErrorInvalidValue;
  }
  const std::uint32_t qkv_rows =
      std::min(rows, static_cast<std::uint32_t>(scratch_values / 3u));
  const std::uint32_t possible_heads = qkv_rows / head_dim;
  if (possible_heads == 0u) {
    return cudaSuccess;
  }
  const std::uint32_t active_heads =
      std::min(head_count, possible_heads);
  const std::uint32_t rows_per_head = qkv_rows / active_heads;
  const std::uint32_t contexts_per_head = rows_per_head / head_dim;
  if (contexts_per_head == 0u) {
    return cudaSuccess;
  }
  const std::uint32_t contexts_for_activation =
      (activation_cols + active_heads * head_dim - 1u) /
      (active_heads * head_dim);
  const std::uint32_t active_contexts =
      std::min(contexts_for_activation, contexts_per_head);
  if (active_contexts == 0u) {
    return cudaSuccess;
  }
  const std::uint32_t context_groups =
      (active_contexts + contexts_per_block - 1u) / contexts_per_block;
  const std::uint32_t blocks = active_heads * context_groups;
  const std::uint32_t threads = head_dim * contexts_per_block;
  attention_qkv_head_group_window_kernel<<<blocks, threads, 0, stream>>>(
      scratch,
      activation,
      qkv_rows,
      activation_cols,
      head_count,
      head_dim,
      window_size,
      contexts_per_block,
      softmax_scale,
      residual_scale);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  if (rows == 0u || scratch_values < 3u || activation_cols == 0u ||
      head_count == 0u || head_dim == 0u || window_size == 0u ||
      contexts_per_block == 0u) {
    return cudaSuccess;
  }
  if (scratch == nullptr || activation == nullptr || head_dim > 32u ||
      (head_dim & 1u) != 0u || window_size > 16u ||
      contexts_per_block > 4u || head_dim * contexts_per_block > 256u) {
    return cudaErrorInvalidValue;
  }
  const std::uint32_t qkv_rows =
      std::min(rows, static_cast<std::uint32_t>(scratch_values / 3u));
  const std::uint32_t possible_heads = qkv_rows / head_dim;
  if (possible_heads == 0u) {
    return cudaSuccess;
  }
  const std::uint32_t active_heads =
      std::min(head_count, possible_heads);
  const std::uint32_t rows_per_head = qkv_rows / active_heads;
  const std::uint32_t contexts_per_head = rows_per_head / head_dim;
  if (contexts_per_head == 0u) {
    return cudaSuccess;
  }
  const std::uint32_t contexts_for_activation =
      (activation_cols + active_heads * head_dim - 1u) /
      (active_heads * head_dim);
  const std::uint32_t active_contexts =
      std::min(contexts_for_activation, contexts_per_head);
  if (active_contexts == 0u) {
    return cudaSuccess;
  }
  const std::uint32_t context_groups =
      (active_contexts + contexts_per_block - 1u) / contexts_per_block;
  const std::uint32_t blocks = active_heads * context_groups;
  const std::uint32_t threads = head_dim * contexts_per_block;
  attention_qkv_head_group_rope_window_kernel<<<blocks, threads, 0, stream>>>(
      scratch,
      activation,
      qkv_rows,
      activation_cols,
      head_count,
      head_dim,
      window_size,
      contexts_per_block,
      rope_theta_scale,
      softmax_scale,
      residual_scale);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  if (rows == 0u || activation_cols == 0u || head_count == 0u ||
      head_dim == 0u || window_size == 0u || contexts_per_block == 0u) {
    return cudaSuccess;
  }
  if (logits == nullptr || activation == nullptr || head_dim > 32u ||
      window_size > 16u || contexts_per_block > 4u ||
      head_dim * contexts_per_block > 256u) {
    return cudaErrorInvalidValue;
  }
  const std::uint32_t possible_heads = rows / head_dim;
  if (possible_heads == 0u) {
    return cudaSuccess;
  }
  const std::uint32_t active_heads =
      std::min(head_count, possible_heads);
  const std::uint32_t rows_per_head = rows / active_heads;
  const std::uint32_t contexts_per_head = rows_per_head / head_dim;
  if (contexts_per_head == 0u) {
    return cudaSuccess;
  }
  const std::uint32_t contexts_for_activation =
      (activation_cols + active_heads * head_dim - 1u) /
      (active_heads * head_dim);
  const std::uint32_t active_contexts =
      std::min(contexts_for_activation, contexts_per_head);
  if (active_contexts == 0u) {
    return cudaSuccess;
  }
  const std::uint32_t context_groups =
      (active_contexts + contexts_per_block - 1u) / contexts_per_block;
  const std::uint32_t blocks = active_heads * context_groups;
  const std::uint32_t threads = head_dim * contexts_per_block;
  attention_qkv_head_group_fused_kernel<<<blocks, threads, 0, stream>>>(
      logits,
      activation,
      rows,
      activation_cols,
      head_count,
      head_dim,
      window_size,
      contexts_per_block,
      scratch_residual_scale,
      softmax_scale,
      residual_scale);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  if (logits == nullptr || state == nullptr || activation == nullptr ||
      rows == 0u || state_values == 0u || activation_cols == 0u) {
    return cudaErrorInvalidValue;
  }
  constexpr int kBlockThreads = 256;
  const std::uint32_t blocks =
      (activation_cols + kBlockThreads - 1u) / kBlockThreads;
  ssm_recurrent_state_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      logits,
      state,
      activation,
      rows,
      state_values,
      activation_cols,
      decay,
      logit_scale,
      activation_scale,
      output_scale);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  if (logits == nullptr || state == nullptr || scratch == nullptr ||
      activation == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0u || state_values == 0u || scratch_values == 0u ||
      activation_cols == 0u) {
    return cudaSuccess;
  }
  constexpr int kBlockThreads = 256;
  const std::uint32_t blocks =
      (activation_cols + kBlockThreads - 1u) / kBlockThreads;
  ssm_scan_scratch_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      logits,
      state,
      scratch,
      activation,
      rows,
      state_values,
      scratch_values,
      activation_cols,
      decay,
      logit_scale,
      scratch_scale,
      activation_scale,
      output_scale);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  if (logits == nullptr || state == nullptr || scratch == nullptr ||
      activation == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0u || state_values == 0u || scratch_values == 0u ||
      activation_cols == 0u) {
    return cudaSuccess;
  }
  constexpr int kBlockThreads = 256;
  const std::uint32_t blocks =
      (activation_cols + kBlockThreads - 1u) / kBlockThreads;
  ssm_selective_scan_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      logits,
      state,
      scratch,
      activation,
      rows,
      state_values,
      scratch_values,
      activation_cols,
      decay,
      logit_scale,
      scratch_scale,
      activation_scale,
      gate_logit_scale,
      gate_activation_scale,
      gate_scratch_scale,
      output_scale);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  if (logits == nullptr || activation == nullptr || scratch == nullptr ||
      parameter_slot > 1u) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0u || scratch_values == 0u || state_values == 0u ||
      activation_cols == 0u) {
    return cudaSuccess;
  }
  constexpr int kBlockThreads = 256;
  const std::uint32_t blocks =
      (state_values + kBlockThreads - 1u) / kBlockThreads;
  ssm_source_parameter_cache_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      logits,
      activation,
      scratch,
      rows,
      scratch_values,
      state_values,
      activation_cols,
      parameter_slot,
      logit_scale,
      activation_scale);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  if (logits == nullptr || activation == nullptr || scratch == nullptr ||
      parameter_slot > 1u) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0u || activation_cols == 0u || scratch_values == 0u ||
      state_values == 0u) {
    return cudaSuccess;
  }
  constexpr int kBlockThreads = 256;
  const std::uint32_t work_items =
      activation_cols > state_values ? activation_cols : state_values;
  const std::uint32_t blocks =
      (work_items + kBlockThreads - 1u) / kBlockThreads;
  ssm_rmsnorm_feedback_parameter_cache_kernel<<<
      blocks,
      kBlockThreads,
      0,
      stream>>>(
          logits,
          activation,
          scratch,
          rows,
          activation_cols,
          scratch_values,
          state_values,
          parameter_slot,
          epsilon,
          logit_scale,
          activation_scale);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  if (logits == nullptr || state == nullptr || scratch == nullptr ||
      activation == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0u || state_values == 0u || scratch_values == 0u ||
      activation_cols == 0u) {
    return cudaSuccess;
  }
  constexpr int kBlockThreads = 256;
  const std::uint32_t blocks =
      (activation_cols + kBlockThreads - 1u) / kBlockThreads;
  ssm_source_parameterized_scan_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      logits,
      state,
      scratch,
      activation,
      rows,
      state_values,
      scratch_values,
      activation_cols,
      decay,
      logit_scale,
      scratch_scale,
      activation_scale,
      alpha_scale,
      beta_scale,
      gate_logit_scale,
      gate_activation_scale,
      output_scale);
  return cudaGetLastError();
}

extern "C" cudaError_t rtxllm_launch_output_token_sample(
    const float* logits,
    std::uint32_t* token,
    std::uint32_t rows,
    std::uint32_t vocab_size,
    std::uint32_t token_offset,
    cudaStream_t stream) {
  if (logits == nullptr || token == nullptr || rows == 0u || vocab_size == 0u) {
    return cudaErrorInvalidValue;
  }
  output_token_sample_kernel<<<1, 256, 0, stream>>>(
      logits,
      token,
      rows,
      vocab_size,
      token_offset);
  return cudaGetLastError();
}

extern "C" cudaError_t rtxllm_launch_iq2s_probe(
    const std::uint8_t* iq2s_blocks,
    std::uint64_t block_count,
    std::uint64_t* output,
    cudaStream_t stream) {
  constexpr int kBlockThreads = 256;
  if (block_count == 0) {
    return cudaErrorInvalidValue;
  }
  const std::uint64_t payload_bytes = block_count * kIq2SBlockBytes;
  const std::uint64_t required_blocks = (payload_bytes + kBlockThreads - 1ull) / kBlockThreads;
  const auto blocks = static_cast<int>(required_blocks > 4096ull ? 4096ull : required_blocks);
  iq2s_probe_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      iq2s_blocks,
      block_count,
      reinterpret_cast<unsigned long long*>(output));
  return cudaGetLastError();
}

extern "C" cudaError_t rtxllm_launch_iq3s_probe(
    const std::uint8_t* iq3s_blocks,
    std::uint64_t block_count,
    std::uint64_t* output,
    cudaStream_t stream) {
  constexpr int kBlockThreads = 256;
  if (block_count == 0) {
    return cudaErrorInvalidValue;
  }
  const std::uint64_t payload_bytes = block_count * kIq3SBlockBytes;
  const std::uint64_t required_blocks = (payload_bytes + kBlockThreads - 1ull) / kBlockThreads;
  const auto blocks = static_cast<int>(required_blocks > 4096ull ? 4096ull : required_blocks);
  iq3s_probe_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      iq3s_blocks,
      block_count,
      reinterpret_cast<unsigned long long*>(output));
  return cudaGetLastError();
}

extern "C" cudaError_t rtxllm_launch_q5k_probe(
    const std::uint8_t* q5k_blocks,
    std::uint64_t block_count,
    std::uint64_t* output,
    cudaStream_t stream) {
  constexpr int kBlockThreads = 256;
  if (block_count == 0) {
    return cudaErrorInvalidValue;
  }
  const std::uint64_t payload_bytes = block_count * kQ5KBlockBytes;
  const std::uint64_t required_blocks = (payload_bytes + kBlockThreads - 1ull) / kBlockThreads;
  const auto blocks = static_cast<int>(required_blocks > 4096ull ? 4096ull : required_blocks);
  q5k_probe_kernel<<<blocks, kBlockThreads, 0, stream>>>(
      q5k_blocks,
      block_count,
      reinterpret_cast<unsigned long long*>(output));
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  constexpr int kBlockThreads = 256;
  if (rows == 0 || cols == 0 || rows_limit == 0 || (cols % 256u) != 0u) {
    return cudaErrorInvalidValue;
  }
  const std::uint32_t launched_rows = rows_limit < rows ? rows_limit : rows;
  q5k_matvec_decode_kernel<<<
      launched_rows,
      kBlockThreads,
      kBlockThreads * sizeof(float),
      stream>>>(
      q5k_blocks,
      activation,
      logits,
      kv,
      token,
      rows,
      cols,
      launched_rows,
      kv_words,
      page_words,
      active_pages);
  const auto status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  argmax_token_kernel<<<1, 256, 0, stream>>>(logits, token, launched_rows);
  return cudaGetLastError();
}

extern "C" cudaError_t rtxllm_launch_iq2s_matvec_probe(
    const std::uint8_t* iq2s_blocks,
    const float* activation,
    float* logits,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t rows_limit,
    cudaStream_t stream) {
  constexpr int kBlockThreads = 256;
  if (rows == 0 || cols == 0 || rows_limit == 0 || (cols % 256u) != 0u) {
    return cudaErrorInvalidValue;
  }
  const std::uint32_t launched_rows = rows_limit < rows ? rows_limit : rows;
  iq2s_matvec_probe_kernel<<<
      launched_rows,
      kBlockThreads,
      kBlockThreads * sizeof(float),
      stream>>>(iq2s_blocks, activation, logits, nullptr, nullptr, rows, cols, rows_limit, 0, 0, 0);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  constexpr int kBlockThreads = 256;
  if (rows == 0 || cols == 0 || rows_limit == 0 || (cols % 256u) != 0u) {
    return cudaErrorInvalidValue;
  }
  const std::uint32_t launched_rows = rows_limit < rows ? rows_limit : rows;
  iq2s_matvec_probe_kernel<<<
      launched_rows,
      kBlockThreads,
      kBlockThreads * sizeof(float),
      stream>>>(
      iq2s_blocks,
      activation,
      logits,
      kv,
      token,
      rows,
      cols,
      launched_rows,
      kv_words,
      page_words,
      active_pages);
  const auto status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  argmax_token_kernel<<<1, 256, 0, stream>>>(logits, token, launched_rows);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  constexpr int kBlockThreads = 256;
  if (rows == 0 || cols == 0 || rows_limit == 0 || (cols % 256u) != 0u) {
    return cudaErrorInvalidValue;
  }
  const std::uint32_t launched_rows = rows_limit < rows ? rows_limit : rows;
  iq3s_matvec_probe_kernel<<<
      launched_rows,
      kBlockThreads,
      kBlockThreads * sizeof(float),
      stream>>>(
      iq3s_blocks,
      activation,
      logits,
      kv,
      token,
      rows,
      cols,
      launched_rows,
      kv_words,
      page_words,
      active_pages);
  const auto status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  argmax_token_kernel<<<1, 256, 0, stream>>>(logits, token, launched_rows);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  constexpr int kBlockThreads = 128;
  q4_matvec_decode_kernel<<<rows, kBlockThreads, kBlockThreads * sizeof(float), stream>>>(
      packed_weights,
      scales,
      activation,
      logits,
      kv,
      token,
      rows,
      cols,
      kv_words,
      page_words,
      active_pages);
  const cudaError_t matvec_status = cudaGetLastError();
  if (matvec_status != cudaSuccess) {
    return matvec_status;
  }
  argmax_token_kernel<<<1, 256, 0, stream>>>(logits, token, rows);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  constexpr int kBlockThreads = 128;
  q4_matvec_decode_vec4x4_kernel<<<rows, kBlockThreads, 0, stream>>>(
      packed_weights,
      scales,
      activation,
      logits,
      kv,
      token,
      rows,
      cols,
      kv_words,
      page_words,
      active_pages);
  const cudaError_t matvec_status = cudaGetLastError();
  if (matvec_status != cudaSuccess) {
    return matvec_status;
  }
  argmax_token_kernel<<<1, 256, 0, stream>>>(logits, token, rows);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  constexpr int kBlockThreads = 128;
  q4k_matvec_decode_shared_kernel<<<rows, kBlockThreads, kBlockThreads * sizeof(float), stream>>>(
      q4k_blocks,
      activation,
      logits,
      kv,
      token,
      rows,
      cols,
      kv_words,
      page_words,
      active_pages);
  const cudaError_t matvec_status = cudaGetLastError();
  if (matvec_status != cudaSuccess) {
    return matvec_status;
  }
  argmax_token_kernel<<<1, 256, 0, stream>>>(logits, token, rows);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  constexpr int kBlockThreads = 32;
  q4k_matvec_decode_warp_kernel<<<rows, kBlockThreads, 0, stream>>>(
      q4k_blocks,
      activation,
      logits,
      kv,
      token,
      rows,
      cols,
      kv_words,
      page_words,
      active_pages);
  const cudaError_t matvec_status = cudaGetLastError();
  if (matvec_status != cudaSuccess) {
    return matvec_status;
  }
  argmax_token_kernel<<<1, 256, 0, stream>>>(logits, token, rows);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  constexpr int kBlockThreads = 32;
  q4k_matvec_decode_warp_broadcast_kernel<<<rows, kBlockThreads, 0, stream>>>(
      q4k_blocks,
      activation,
      logits,
      kv,
      token,
      rows,
      cols,
      kv_words,
      page_words,
      active_pages);
  const cudaError_t matvec_status = cudaGetLastError();
  if (matvec_status != cudaSuccess) {
    return matvec_status;
  }
  argmax_token_kernel<<<1, 256, 0, stream>>>(logits, token, rows);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  constexpr int kBlockThreads = 32;
  q4k_matvec_decode_warp_broadcast_vec4_kernel<<<rows, kBlockThreads, 0, stream>>>(
      q4k_blocks,
      activation,
      logits,
      kv,
      token,
      rows,
      cols,
      kv_words,
      page_words,
      active_pages);
  const cudaError_t matvec_status = cudaGetLastError();
  if (matvec_status != cudaSuccess) {
    return matvec_status;
  }
  argmax_token_kernel<<<1, 256, 0, stream>>>(logits, token, rows);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  constexpr int kBlockThreads = 64;
  q4k_matvec_decode_warp_broadcast_vec4x2_kernel<<<rows, kBlockThreads, 0, stream>>>(
      q4k_blocks,
      activation,
      logits,
      kv,
      token,
      rows,
      cols,
      kv_words,
      page_words,
      active_pages);
  const cudaError_t matvec_status = cudaGetLastError();
  if (matvec_status != cudaSuccess) {
    return matvec_status;
  }
  argmax_token_kernel<<<1, 256, 0, stream>>>(logits, token, rows);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  constexpr int kBlockThreads = 128;
  q4k_matvec_decode_warp_broadcast_vec4x4_kernel<<<rows, kBlockThreads, 0, stream>>>(
      q4k_blocks,
      activation,
      logits,
      kv,
      token,
      rows,
      cols,
      kv_words,
      page_words,
      active_pages);
  const cudaError_t matvec_status = cudaGetLastError();
  if (matvec_status != cudaSuccess) {
    return matvec_status;
  }
  argmax_token_kernel<<<1, 256, 0, stream>>>(logits, token, rows);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  constexpr int kBlockThreads = 128;
  q4k_matvec_decode_predecoded_vec4x4_kernel<<<rows, kBlockThreads, 0, stream>>>(
      q4k_blocks,
      q4k_scale_min_pairs,
      activation,
      logits,
      kv,
      token,
      rows,
      cols,
      kv_words,
      page_words,
      active_pages);
  const cudaError_t matvec_status = cudaGetLastError();
  if (matvec_status != cudaSuccess) {
    return matvec_status;
  }
  argmax_token_kernel<<<1, 256, 0, stream>>>(logits, token, rows);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  constexpr int kBlockThreads = 128;
  q4k_matvec_decode_split_predecoded_vec4x4_kernel<<<rows, kBlockThreads, 0, stream>>>(
      q4k_payload,
      q4k_scale_min_pairs,
      activation,
      logits,
      kv,
      token,
      rows,
      cols,
      kv_words,
      page_words,
      active_pages);
  const cudaError_t matvec_status = cudaGetLastError();
  if (matvec_status != cudaSuccess) {
    return matvec_status;
  }
  argmax_token_kernel<<<1, 256, 0, stream>>>(logits, token, rows);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  constexpr int kBlockThreads = 128;
  q4k_matvec_decode_split_half_vec4x4_kernel<<<rows, kBlockThreads, 0, stream>>>(
      q4k_payload,
      q4k_scale_min_half_pairs,
      activation,
      logits,
      kv,
      token,
      rows,
      cols,
      kv_words,
      page_words,
      active_pages);
  const cudaError_t matvec_status = cudaGetLastError();
  if (matvec_status != cudaSuccess) {
    return matvec_status;
  }
  argmax_token_kernel<<<1, 256, 0, stream>>>(logits, token, rows);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  constexpr int kBlockThreads = 128;
  q4k_matvec_decode_split_compact_vec4x4_kernel<<<rows, kBlockThreads, 0, stream>>>(
      q4k_payload,
      q4k_compact_meta,
      activation,
      logits,
      kv,
      token,
      rows,
      cols,
      kv_words,
      page_words,
      active_pages);
  const cudaError_t matvec_status = cudaGetLastError();
  if (matvec_status != cudaSuccess) {
    return matvec_status;
  }
  argmax_token_kernel<<<1, 256, 0, stream>>>(logits, token, rows);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  constexpr int kBlockThreads = 128;
  q4k_matvec_decode_split_native_vec4x4_kernel<<<rows, kBlockThreads, 0, stream>>>(
      q4k_payload,
      q4k_native_meta,
      activation,
      logits,
      kv,
      token,
      rows,
      cols,
      kv_words,
      page_words,
      active_pages);
  const cudaError_t matvec_status = cudaGetLastError();
  if (matvec_status != cudaSuccess) {
    return matvec_status;
  }
  argmax_token_kernel<<<1, 256, 0, stream>>>(logits, token, rows);
  return cudaGetLastError();
}

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
    cudaStream_t stream) {
  constexpr int kBlockThreads = 128;
  q4k_matvec_decode_kernel<<<rows, kBlockThreads, kBlockThreads * sizeof(float), stream>>>(
      q4k_blocks,
      activation,
      logits,
      kv,
      token,
      rows,
      cols,
      kv_words,
      page_words,
      active_pages);
  const cudaError_t matvec_status = cudaGetLastError();
  if (matvec_status != cudaSuccess) {
    return matvec_status;
  }
  argmax_token_kernel<<<1, 256, 0, stream>>>(logits, token, rows);
  return cudaGetLastError();
}
