// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#ifdef __powerpc__

#include <altivec.h>
#include <algorithm>
#include <cstdint>
#include <torch/all.h>

#include "cpu_types.hpp"

namespace mamba_cpu {
namespace vsx {

// PowerPC Power10 VSX specialisation for c10::BFloat16 causal_conv1d_update:
// - Vectorized depthwise 1D conv processing 8 channels per cycle
// - On-the-fly in-register weight transpose via vec_perm (no weight pre-packing)
// - In-place state buffer update (zero memmove calls)
// - Dual-issue hardware FMAs and fused SiLU using vec_op
// - Adaptive 1D/2D thread scheduling for optimal SMT scaling
inline void causal_conv1d_update_bf16(
    const c10::BFloat16* __restrict__ x_ptr,
    c10::BFloat16* __restrict__ state_ptr,
    int64_t stride_s_slot, int64_t stride_s_dim, int64_t stride_s_state,
    const c10::BFloat16* __restrict__ weight_ptr,
    const float* __restrict__ bias_ptr,
    c10::BFloat16* __restrict__ out_ptr,
    const int32_t* __restrict__ cache_idxs,
    int32_t pad_slot_id, int64_t batch, int64_t dim, int64_t seqlen,
    int64_t width, int64_t state_len, bool do_silu) {

  constexpr int64_t BLOCK_DIM = 256;
  const int64_t num_dim_blocks = (dim + BLOCK_DIM - 1) / BLOCK_DIM;

  // Permutation masks to transpose (8, 4) weights in VSX registers
  const __vector unsigned char m_t0 = {
      0, 1, 8, 9, 16, 17, 24, 25, 0, 0, 0, 0, 0, 0, 0, 0};
  const __vector unsigned char m_t1 = {
      2, 3, 10, 11, 18, 19, 26, 27, 0, 0, 0, 0, 0, 0, 0, 0};
  const __vector unsigned char m_t2 = {
      4, 5, 12, 13, 20, 21, 28, 29, 0, 0, 0, 0, 0, 0, 0, 0};
  const __vector unsigned char m_t3 = {
      6, 7, 14, 15, 22, 23, 30, 31, 0, 0, 0, 0, 0, 0, 0, 0};
  const __vector unsigned char m_combine = {
      0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};

  auto compute_block = [&](int64_t b, int64_t d_start, int64_t d_end) {
    int64_t cache_idx = (cache_idxs != nullptr) ? cache_idxs[b] : b;
    if (cache_idx == pad_slot_id) return;

    c10::BFloat16* s_base = state_ptr + cache_idx * stride_s_slot;
    const c10::BFloat16* x_b = x_ptr + b * dim;
    c10::BFloat16* out_b = out_ptr + b * dim;

    c10::BFloat16* s0_ptr = s_base + 0 * stride_s_state;
    c10::BFloat16* s1_ptr = s_base + 1 * stride_s_state;
    c10::BFloat16* s2_ptr = s_base + 2 * stride_s_state;

    for (int64_t d = d_start; d < d_end; d += 8) {
      // 1. Load weights for 8 channels (32 BF16 elements = 64 bytes)
      const char* wb = reinterpret_cast<const char*>(weight_ptr + d * 4);
      vec_op::BF16Vec8 w_c01(wb);
      vec_op::BF16Vec8 w_c23(wb + 16);
      vec_op::BF16Vec8 w_c45(wb + 32);
      vec_op::BF16Vec8 w_c67(wb + 48);

      // In-register transpose of (8, 4) weights into 4x 8-channel tap vectors
      __vector signed short t0_lo = vec_perm(w_c01.reg, w_c23.reg, m_t0);
      __vector signed short t1_lo = vec_perm(w_c01.reg, w_c23.reg, m_t1);
      __vector signed short t2_lo = vec_perm(w_c01.reg, w_c23.reg, m_t2);
      __vector signed short t3_lo = vec_perm(w_c01.reg, w_c23.reg, m_t3);

      __vector signed short t0_hi = vec_perm(w_c45.reg, w_c67.reg, m_t0);
      __vector signed short t1_hi = vec_perm(w_c45.reg, w_c67.reg, m_t1);
      __vector signed short t2_hi = vec_perm(w_c45.reg, w_c67.reg, m_t2);
      __vector signed short t3_hi = vec_perm(w_c45.reg, w_c67.reg, m_t3);

      vec_op::FP32Vec8 w0(vec_op::BF16Vec8(vec_perm(t0_lo, t0_hi, m_combine)));
      vec_op::FP32Vec8 w1(vec_op::BF16Vec8(vec_perm(t1_lo, t1_hi, m_combine)));
      vec_op::FP32Vec8 w2(vec_op::BF16Vec8(vec_perm(t2_lo, t2_hi, m_combine)));
      vec_op::FP32Vec8 w3(vec_op::BF16Vec8(vec_perm(t3_lo, t3_hi, m_combine)));

      // 2. Load 8 BF16 elements for x and state taps 0..2
      vec_op::BF16Vec8 x_bf(x_b + d);
      vec_op::BF16Vec8 s0_bf(s0_ptr + d);
      vec_op::BF16Vec8 s1_bf(s1_ptr + d);
      vec_op::BF16Vec8 s2_bf(s2_ptr + d);

      vec_op::FP32Vec8 x_vec(x_bf);
      vec_op::FP32Vec8 s0_vec(s0_bf);
      vec_op::FP32Vec8 s1_vec(s1_bf);
      vec_op::FP32Vec8 s2_vec(s2_bf);

      // 3. Initialize accumulator with bias (or zero)
      vec_op::FP32Vec8 acc = (bias_ptr != nullptr) ? vec_op::FP32Vec8(bias_ptr + d)
                                                   : vec_op::FP32Vec8(0.0f);

      // 4. Dual-issue VSX FMAs: acc = bias + w0*s0 + w1*s1 + w2*s2 + w3*x
      vec_op::fma(acc, w0, s0_vec);
      vec_op::fma(acc, w1, s1_vec);
      vec_op::fma(acc, w2, s2_vec);
      vec_op::fma(acc, w3, x_vec);

      // 5. In-place state roll: s0 <= s1, s1 <= s2, s2 <= x (zero memmove calls)
      s1_bf.save(s0_ptr + d);
      s2_bf.save(s1_ptr + d);
      x_bf.save(s2_ptr + d);

      // 6. Vectorized SiLU
      if (do_silu) {
        acc = acc.silu();
      }

      // 7. Store BF16 output
      vec_op::BF16Vec8(acc).save(out_b + d);
    }
  };

  // Adaptive thread scheduling: 1D for batch >= 8, 2D (batch, dim) for batch < 8
  if (batch >= 8) {
#pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < batch; ++b) {
      compute_block(b, 0, dim);
    }
  } else {
#pragma omp parallel for collapse(2) schedule(static)
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t db = 0; db < num_dim_blocks; ++db) {
        int64_t d_start = db * BLOCK_DIM;
        int64_t d_end = std::min(d_start + BLOCK_DIM, dim);
        compute_block(b, d_start, d_end);
      }
    }
  }
}

template <typename scalar_t>
inline void causal_conv1d_update(
    const scalar_t* __restrict__ x_ptr,
    scalar_t* __restrict__ state_ptr,
    int64_t stride_s_slot, int64_t stride_s_dim, int64_t stride_s_state,
    const scalar_t* __restrict__ weight_ptr,
    const float* __restrict__ bias_ptr,
    scalar_t* __restrict__ out_ptr,
    const int32_t* __restrict__ cache_idxs,
    int32_t pad_slot_id, int64_t batch, int64_t dim, int64_t seqlen,
    int64_t width, int64_t state_len, bool do_silu) {

  if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
    causal_conv1d_update_bf16(
        x_ptr, state_ptr, stride_s_slot, stride_s_dim, stride_s_state,
        weight_ptr, bias_ptr, out_ptr, cache_idxs, pad_slot_id,
        batch, dim, seqlen, width, state_len, do_silu);
  } else {
    causal_conv1d_update_kernel<scalar_t>(
        x_ptr, state_ptr, stride_s_slot, stride_s_dim, stride_s_state,
        weight_ptr, bias_ptr, out_ptr, cache_idxs, pad_slot_id,
        batch, dim, seqlen, width, state_len, do_silu);
  }
}

}  // namespace vsx
}  // namespace mamba_cpu

#endif  // __powerpc__
