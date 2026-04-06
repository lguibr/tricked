#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

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
    int num_standard_pieces) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch_size)
    return;

  float *batch_out = out + b * 2560;

  uint64_t b0 = *(uint64_t *)&boards[b * 2 + 0];
  uint64_t b1 = *(uint64_t *)&boards[b * 2 + 1];

  auto fill_channel = [&](int c_idx, uint64_t m0, uint64_t m1) {
    int offset = c_idx * 128;
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
      batch_out[(8 + i) * 128 + spatial] = (float)(slot + 1) * 0.33f;
    }
  }

  // Channel 11-16: pieces canonical footprint & validity
  for (int slot = 0; slot < 3; ++slot) {
    int p_id = avail[b * 3 + slot];
    if (p_id == -1)
      continue;

    for (int k = 0; k < 128; ++k) {
      int spatial = canonical[p_id * 128 + k];
      if (spatial == -1)
        break;
      batch_out[(11 + slot * 2) * 128 + spatial] = 1.0f;
    }

    uint64_t val0 = 0, val1 = 0;
    for (int k = 0; k < 64; ++k) {
      uint64_t m0 = *(uint64_t *)&compact[p_id * 128 + k * 2 + 0];
      uint64_t m1 = *(uint64_t *)&compact[p_id * 128 + k * 2 + 1];
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
    batch_out[17 * 128 + spatial] = 1.0f / 22.0f;
    batch_out[18 * 128 + spatial] = norm_diff;
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

torch::Tensor extract_feature_cuda(torch::Tensor boards, torch::Tensor avail,
                                   torch::Tensor hist, torch::Tensor acts,
                                   torch::Tensor diff, torch::Tensor canonical,
                                   torch::Tensor compact,
                                   torch::Tensor standard) {
  int batch_size = boards.size(0);
  auto out = torch::zeros({batch_size, 20, 8, 16},
                          boards.options().dtype(torch::kFloat32));
  int num_standard = standard.size(0);

  int threads = 256;
  int blocks = (batch_size + threads - 1) / threads;

  extract_features_kernel<<<blocks, threads>>>(
      boards.data_ptr<int64_t>(), avail.data_ptr<int32_t>(),
      hist.data_ptr<int64_t>(), acts.data_ptr<int32_t>(),
      diff.data_ptr<int32_t>(), out.data_ptr<float>(),
      canonical.data_ptr<int32_t>(), compact.data_ptr<int64_t>(),
      standard.data_ptr<int64_t>(), batch_size, num_standard);

  return out;
}

__global__ void extract_unrolled_features_kernel(
    const int64_t *__restrict__ boards, const int64_t *__restrict__ hist,
    float *__restrict__ out, int batch_size, int unroll_steps) {
  int b = blockIdx.x;
  int u = threadIdx.x;

  if (b >= batch_size || u >= unroll_steps)
    return;

  float *batch_out = out + (b * unroll_steps + u) * 2560;

  for (int i = 0; i < 2560; ++i) {
    batch_out[i] = 0.0f;
  }

  uint64_t b0 = *(uint64_t *)&boards[(b * unroll_steps + u) * 2 + 0];
  uint64_t b1 = *(uint64_t *)&boards[(b * unroll_steps + u) * 2 + 1];

  auto fill_channel = [&](int c_idx, uint64_t m0, uint64_t m1) {
    int offset = c_idx * 128;
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

torch::Tensor extract_unrolled_features_cuda(torch::Tensor boards,
                                             torch::Tensor hist) {
  int batch_size = boards.size(0);
  int unroll_steps = boards.size(1);
  auto out = torch::zeros({batch_size, unroll_steps, 20, 8, 16},
                          boards.options().dtype(torch::kFloat32));

  extract_unrolled_features_kernel<<<batch_size, 32>>>(
      boards.data_ptr<int64_t>(), hist.data_ptr<int64_t>(),
      out.data_ptr<float>(), batch_size, unroll_steps);

  return out;
}
