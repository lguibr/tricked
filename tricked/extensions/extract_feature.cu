#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

__constant__ int HEX_MAP_ROW[96] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7};
__constant__ int HEX_MAP_COL[96] = {
    4,  5,  6,  7,  8,  9,  10, 11, 12, 3,  4,  5,  6,  7,  8,  9,
    10, 11, 12, 13, 2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
    14, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 2,
    3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 3,  4,  5,  6,
    7,  8,  9,  10, 11, 12, 13, 4,  5,  6,  7,  8,  9,  10, 11, 12};

__global__ void extract_features_kernel(
    const int64_t *__restrict__ boards, const int32_t *__restrict__ avail,
    const int64_t *__restrict__ hist, const int32_t *__restrict__ acts,
    const int32_t *__restrict__ diff, float *__restrict__ out,
    const int32_t *__restrict__ canonical, const int64_t *__restrict__ compact,
    const int64_t *__restrict__ standard_pieces, int batch_size,
    int num_standard_pieces, int out_channels, int spatial_rows, int spatial_cols) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch_size)
    return;

  int spatial_size = spatial_rows * spatial_cols;
  float *batch_out = out + b * (out_channels * spatial_size);

  uint64_t b0 = *(uint64_t *)&boards[b * 2 + 0];
  uint64_t b1 = *(uint64_t *)&boards[b * 2 + 1];

  auto fill_channel = [&](int c_idx, uint64_t m0, uint64_t m1) {
    int offset = c_idx * spatial_size;
    uint64_t t0 = m0;
    while (t0) {
      int bit = __ffsll(t0) - 1;
      int spatial = HEX_MAP_ROW[bit] * 16 + HEX_MAP_COL[bit];
      batch_out[offset + spatial] = 1.0f;
      t0 &= t0 - 1;
    }
    uint64_t t1 = m1 & 0xFFFFFFFF; // Only 32 bits needed for 96
    while (t1) {
      int bit = __ffsll(t1) - 1 + 64;
      int spatial = HEX_MAP_ROW[bit] * 16 + HEX_MAP_COL[bit];
      batch_out[offset + spatial] = 1.0f;
      t1 &= t1 - 1;
    }
  };

  // Channel 0: board
  fill_channel(0, b0, b1);

  // Channel 1-7: history
  for (int i = 0; i < 7; ++i) {
    uint64_t h0 = *(uint64_t *)&hist[b * 14 + i * 2 + 0];
    uint64_t h1 = *(uint64_t *)&hist[b * 14 + i * 2 + 1];
    fill_channel(i + 1, h0, h1);
  }

  // Channel 8-10: actions
  for (int i = 0; i < 3; ++i) {
    int action = acts[b * 3 + i];
    if (action != -1) {
      int slot = action / 96;
      int pos = action % 96;
      int spatial = HEX_MAP_ROW[pos] * 16 + HEX_MAP_COL[pos];
      batch_out[(8 + i) * spatial_size + spatial] = (float)(slot + 1) * 0.33f;
    }
  }

  // Channel 11-16: pieces canonical footprint & validity
  for (int slot = 0; slot < 3; ++slot) {
    int p_id = avail[b * 3 + slot];
    if (p_id == -1)
      continue;

    for (int k = 0; k < spatial_size; ++k) {
      int spatial = canonical[p_id * spatial_size + k];
      if (spatial == -1)
        break;
      batch_out[(11 + slot * 2) * spatial_size + spatial] = 1.0f;
    }

    uint64_t val0 = 0, val1 = 0;
    for (int k = 0; k < 64; ++k) {
      uint64_t m0 = *(uint64_t *)&compact[p_id * spatial_size + k * 2 + 0];
      uint64_t m1 = *(uint64_t *)&compact[p_id * spatial_size + k * 2 + 1];
      if (m0 == 0 && m1 == 0)
        break;
      if ((b0 & m0) == 0 && (b1 & m1) == 0) {
        val0 |= m0;
        val1 |= m1;
      }
    }
    fill_channel(12 + slot * 2, val0, val1);
  }

  // Channel 17-18: Constants
  float norm_diff = (float)diff[b] / 6.0f;
  for (int bit = 0; bit < 96; ++bit) {
    int spatial = HEX_MAP_ROW[bit] * 16 + HEX_MAP_COL[bit];
    batch_out[17 * spatial_size + spatial] = 1.0f / 22.0f;
    batch_out[18 * spatial_size + spatial] = norm_diff;
  }

  // Channel 19: Dead zone
  uint64_t glob0 = 0, glob1 = 0;
  for (int i = 0; i < num_standard_pieces; ++i) {
    uint64_t m0 = *(uint64_t *)&standard_pieces[i * 2 + 0];
    uint64_t m1 = *(uint64_t *)&standard_pieces[i * 2 + 1];
    if (m0 == 0 && m1 == 0)
      continue;
    if ((b0 & m0) == 0 && (b1 & m1) == 0) {
      glob0 |= m0;
      glob1 |= m1;
    }
  }

  // all_hexes 96 bit mask = 64 ones, 32 ones
  uint64_t dead0 = (~b0) & (~glob0);
  uint64_t dead1 = (~b1) & (~glob1) & 0xFFFFFFFFULL;

  fill_channel(19, dead0, dead1);
}

__global__ void extract_unrolled_features_kernel(
    const int64_t *__restrict__ boards, const int64_t *__restrict__ hist,
    float *__restrict__ out, int batch_size, int unroll_steps,
    int out_channels, int spatial_rows, int spatial_cols) {
  int b = blockIdx.x;
  int u = threadIdx.x;

  if (b >= batch_size || u >= unroll_steps)
    return;

  int spatial_size = spatial_rows * spatial_cols;
  float *batch_out = out + (b * unroll_steps + u) * (out_channels * spatial_size);

  for (int i = 0; i < (out_channels * spatial_size); ++i) {
    batch_out[i] = 0.0f;
  }

  uint64_t b0 = *(uint64_t *)&boards[(b * unroll_steps + u) * 2 + 0];
  uint64_t b1 = *(uint64_t *)&boards[(b * unroll_steps + u) * 2 + 1];

  auto fill_channel = [&](int c_idx, uint64_t m0, uint64_t m1) {
    int offset = c_idx * spatial_size;
    uint64_t t0 = m0;
    while (t0) {
      int bit = __ffsll(t0) - 1;
      int spatial = HEX_MAP_ROW[bit] * 16 + HEX_MAP_COL[bit];
      batch_out[offset + spatial] = 1.0f;
      t0 &= t0 - 1;
    }
    uint64_t t1 = m1 & 0xFFFFFFFF; // Only 32 bits needed for 96
    while (t1) {
      int bit = __ffsll(t1) - 1 + 64;
      int spatial = HEX_MAP_ROW[bit] * 16 + HEX_MAP_COL[bit];
      batch_out[offset + spatial] = 1.0f;
      t1 &= t1 - 1;
    }
  };

  fill_channel(0, b0, b1);

  for (int i = 0; i < 7; ++i) {
    uint64_t h0 = *(uint64_t *)&hist[(b * unroll_steps + u) * 14 + i * 2 + 0];
    uint64_t h1 = *(uint64_t *)&hist[(b * unroll_steps + u) * 14 + i * 2 + 1];
    fill_channel(i + 1, h0, h1);
  }
}

__global__ void support_to_scalar_kernel(const float* __restrict__ logits, float* __restrict__ out, int batch_size, int support_size, float epsilon) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch_size)
    return;

  int S = 2 * support_size + 1;
  const float *row_logits = logits + b * S;

  float max_logit = -1e9f;
  for (int i = 0; i < S; ++i) {
    if (row_logits[i] > max_logit)
      max_logit = row_logits[i];
  }

  float sum_exp = 0.0f;
  for (int i = 0; i < S; ++i) {
    sum_exp += expf(row_logits[i] - max_logit);
  }

  float expected_value = 0.0f;
  for (int i = 0; i < S; ++i) {
    float prob = (sum_exp > 0.0f) ? expf(row_logits[i] - max_logit) / sum_exp : 0.0f;
    float support_val = (float)(i - support_size);
    expected_value += prob * support_val;
  }

  float sgn = (expected_value > 0.0f) ? 1.0f : ((expected_value < 0.0f) ? -1.0f : 0.0f);
  float abs_x = fabsf(expected_value);

  float term1 = sqrtf(1.0f + 4.0f * epsilon * (abs_x + 1.0f + epsilon)) - 1.0f;
  float term2 = term1 / (2.0f * epsilon);
  float inv = sgn * (term2 * term2 - 1.0f);

  out[b] = inv;
}

__global__ void scalar_to_support_kernel(const float* __restrict__ scalar, float* __restrict__ out_probs, int batch_size, int support_size, float epsilon) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch_size)
    return;

  int S = 2 * support_size + 1;
  float *row_probs = out_probs + b * S;

  for (int i = 0; i < S; ++i)
    row_probs[i] = 0.0f;

  float x = scalar[b];
  if (isnan(x) || isinf(x))
    x = 0.0f;

  float sgn = (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
  float abs_x = fabsf(x);

  float transformed = sgn * (sqrtf(abs_x + 1.0f) - 1.0f) + epsilon * x;

  float f_support = (float)support_size;
  float clamped = f_support * tanhf(transformed / f_support);

  float shifted = clamped + f_support;
  float floor_val = floorf(shifted);
  float ceil_val = ceilf(shifted);

  float upper_prob = shifted - floor_val;
  float lower_prob = 1.0f - upper_prob;

  int lower_idx = (int)floor_val;
  int upper_idx = (int)ceil_val;

  if (lower_idx >= 0 && lower_idx < S)
    row_probs[lower_idx] += lower_prob;
  if (upper_idx >= 0 && upper_idx < S)
    row_probs[upper_idx] += upper_prob;
}

extern "C" {
void launch_extract_features(const int64_t *boards, const int32_t *avail,
                             const int64_t *hist, const int32_t *acts,
                             const int32_t *diff, float *out,
                             const int32_t *canonical, const int64_t *compact,
                             const int64_t *standard_pieces, int batch_size,
                             int num_standard_pieces, int out_channels,
                             int spatial_rows, int spatial_cols) {
  int blocks = (batch_size + 255) / 256;
  extract_features_kernel<<<blocks, 256>>>(
      boards, avail, hist, acts, diff, out, canonical, compact, standard_pieces,
      batch_size, num_standard_pieces, out_channels, spatial_rows, spatial_cols);
  cudaDeviceSynchronize();
}

void launch_extract_unrolled_features(const int64_t *boards,
                                      const int64_t *hist, float *out,
                                      int batch_size, int unroll_steps,
                                      int out_channels, int spatial_rows, int spatial_cols) {
  extract_unrolled_features_kernel<<<batch_size, 32>>>(
      boards, hist, out, batch_size, unroll_steps, out_channels, spatial_rows, spatial_cols);
  cudaDeviceSynchronize();
}

void launch_support_to_scalar(const float *logits, float *out, int batch_size, int support_size, float epsilon) {
  int blocks = (batch_size + 255) / 256;
  support_to_scalar_kernel<<<blocks, 256>>>(logits, out, batch_size, support_size, epsilon);
  cudaDeviceSynchronize();
}

void launch_scalar_to_support(const float *scalar, float *out_probs, int batch_size, int support_size, float epsilon) {
  int blocks = (batch_size + 255) / 256;
  scalar_to_support_kernel<<<blocks, 256>>>(scalar, out_probs, batch_size, support_size, epsilon);
  cudaDeviceSynchronize();
}
}

