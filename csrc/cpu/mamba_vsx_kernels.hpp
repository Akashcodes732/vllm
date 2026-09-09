// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#ifdef __powerpc__

#include <altivec.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <torch/all.h>

#include "cpu_types.hpp"
#include "cpu/mamba_kernels.hpp"

namespace mamba_cpu {
namespace vsx {

// ===========================================================================
// 1. causal_conv1d_update (PowerPC Power10 VSX)
// ===========================================================================
// Specialised for depthwise 1-D conv in Mamba/Granite-4 decode steps:
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

// ===========================================================================
// 2. mamba_chunk_scan_fwd (PowerPC Power10 VSX)
// ===========================================================================
// Prefill SSM recurrence for Mamba2 / SSD models:
// - 8-way row-tiled recurrence over headdim (B and C vectors loaded once per 8 rows, cutting 87.5% bandwidth)
// - Precomputed scalar factors (x * dt) outside inner dstate loop
// - Dual-issue hardware FMAs (vec_madd) for state update and readout accumulation
// - Vectorized SiLU gating using FP32Vec8::silu()
// - Vectorized 16-byte aligned BF16 stores (8 outputs per cycle)
inline void mamba_chunk_scan_fwd_bf16(
    float* __restrict__ states_ptr,
    const c10::BFloat16* __restrict__ x_ptr,
    const float* __restrict__ dt_ptr,
    const float* __restrict__ A_ptr,
    const c10::BFloat16* __restrict__ B_ptr,
    const c10::BFloat16* __restrict__ C_ptr,
    const float* __restrict__ D_ptr,
    const c10::BFloat16* __restrict__ z_ptr,
    c10::BFloat16* __restrict__ out_ptr,
    const int32_t* __restrict__ cu_seqlens,
    int64_t batch, int64_t nheads, int64_t ngroups, int64_t headdim,
    int64_t dstate) {

  const int64_t nheads_per_group = nheads / ngroups;
  const int64_t stride_s_b = nheads * headdim * dstate;
  const int64_t stride_s_h = headdim * dstate;

#pragma omp parallel for collapse(2) schedule(static)
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t h = 0; h < nheads; ++h) {
      const int64_t seq_start = cu_seqlens[b];
      const int64_t seq_end = cu_seqlens[b + 1];
      const int64_t g = h / nheads_per_group;

      const float A_val = A_ptr[h];
      const float D_val = (D_ptr != nullptr) ? D_ptr[h] : 0.0f;
      const __vector float v_D = vec_splats(D_val);

      float* s_bh = states_ptr + b * stride_s_b + h * stride_s_h;

      for (int64_t t = seq_start; t < seq_end; ++t) {
        const c10::BFloat16* x_h = x_ptr + t * nheads * headdim + h * headdim;
        const float* dt_h = dt_ptr + t * nheads + h;
        const c10::BFloat16* B_g = B_ptr + t * ngroups * dstate + g * dstate;
        const c10::BFloat16* C_g = C_ptr + t * ngroups * dstate + g * dstate;
        const c10::BFloat16* z_h = (z_ptr != nullptr)
                                       ? z_ptr + t * nheads * headdim + h * headdim
                                       : nullptr;
        c10::BFloat16* out_h = out_ptr + t * nheads * headdim + h * headdim;

        const float dt_val = *dt_h;
        const float dA_val = std::exp(A_val * dt_val);
        const __vector float v_dA = vec_splats(dA_val);

        int64_t d = 0;
        // Tile 8 rows of headdim in parallel: B and C vectors are reused 8 times
        for (; d <= headdim - 8; d += 8) {
          vec_op::BF16Vec8 x_bf(x_h + d);
          vec_op::FP32Vec8 x_v(x_bf);

          const __vector float v_xdt0 = vec_splats(vec_extract(x_v.reg.val[0], 0) * dt_val);
          const __vector float v_xdt1 = vec_splats(vec_extract(x_v.reg.val[0], 1) * dt_val);
          const __vector float v_xdt2 = vec_splats(vec_extract(x_v.reg.val[0], 2) * dt_val);
          const __vector float v_xdt3 = vec_splats(vec_extract(x_v.reg.val[0], 3) * dt_val);

          const __vector float v_xdt4 = vec_splats(vec_extract(x_v.reg.val[1], 0) * dt_val);
          const __vector float v_xdt5 = vec_splats(vec_extract(x_v.reg.val[1], 1) * dt_val);
          const __vector float v_xdt6 = vec_splats(vec_extract(x_v.reg.val[1], 2) * dt_val);
          const __vector float v_xdt7 = vec_splats(vec_extract(x_v.reg.val[1], 3) * dt_val);

          float* s0 = s_bh + (d + 0) * dstate;
          float* s1 = s_bh + (d + 1) * dstate;
          float* s2 = s_bh + (d + 2) * dstate;
          float* s3 = s_bh + (d + 3) * dstate;
          float* s4 = s_bh + (d + 4) * dstate;
          float* s5 = s_bh + (d + 5) * dstate;
          float* s6 = s_bh + (d + 6) * dstate;
          float* s7 = s_bh + (d + 7) * dstate;

          vec_op::FP32Vec8 y_acc0(0.0f);
          vec_op::FP32Vec8 y_acc1(0.0f);
          vec_op::FP32Vec8 y_acc2(0.0f);
          vec_op::FP32Vec8 y_acc3(0.0f);
          vec_op::FP32Vec8 y_acc4(0.0f);
          vec_op::FP32Vec8 y_acc5(0.0f);
          vec_op::FP32Vec8 y_acc6(0.0f);
          vec_op::FP32Vec8 y_acc7(0.0f);

          int64_t n = 0;
          for (; n <= dstate - 8; n += 8) {
            vec_op::BF16Vec8 b_bf(B_g + n);
            vec_op::BF16Vec8 c_bf(C_g + n);
            vec_op::FP32Vec8 B_v(b_bf);
            vec_op::FP32Vec8 C_v(c_bf);

            const __vector float B_hi = B_v.reg.val[0];
            const __vector float B_lo = B_v.reg.val[1];
            const __vector float C_hi = C_v.reg.val[0];
            const __vector float C_lo = C_v.reg.val[1];

            // Row 0
            __vector float sn0_hi = vec_madd(v_xdt0, B_hi, vec_mul(vec_xl(0,  s0 + n), v_dA));
            __vector float sn0_lo = vec_madd(v_xdt0, B_lo, vec_mul(vec_xl(16, s0 + n), v_dA));
            vec_xst(sn0_hi, 0,  s0 + n);
            vec_xst(sn0_lo, 16, s0 + n);
            y_acc0.reg.val[0] = vec_madd(sn0_hi, C_hi, y_acc0.reg.val[0]);
            y_acc0.reg.val[1] = vec_madd(sn0_lo, C_lo, y_acc0.reg.val[1]);

            // Row 1
            __vector float sn1_hi = vec_madd(v_xdt1, B_hi, vec_mul(vec_xl(0,  s1 + n), v_dA));
            __vector float sn1_lo = vec_madd(v_xdt1, B_lo, vec_mul(vec_xl(16, s1 + n), v_dA));
            vec_xst(sn1_hi, 0,  s1 + n);
            vec_xst(sn1_lo, 16, s1 + n);
            y_acc1.reg.val[0] = vec_madd(sn1_hi, C_hi, y_acc1.reg.val[0]);
            y_acc1.reg.val[1] = vec_madd(sn1_lo, C_lo, y_acc1.reg.val[1]);

            // Row 2
            __vector float sn2_hi = vec_madd(v_xdt2, B_hi, vec_mul(vec_xl(0,  s2 + n), v_dA));
            __vector float sn2_lo = vec_madd(v_xdt2, B_lo, vec_mul(vec_xl(16, s2 + n), v_dA));
            vec_xst(sn2_hi, 0,  s2 + n);
            vec_xst(sn2_lo, 16, s2 + n);
            y_acc2.reg.val[0] = vec_madd(sn2_hi, C_hi, y_acc2.reg.val[0]);
            y_acc2.reg.val[1] = vec_madd(sn2_lo, C_lo, y_acc2.reg.val[1]);

            // Row 3
            __vector float sn3_hi = vec_madd(v_xdt3, B_hi, vec_mul(vec_xl(0,  s3 + n), v_dA));
            __vector float sn3_lo = vec_madd(v_xdt3, B_lo, vec_mul(vec_xl(16, s3 + n), v_dA));
            vec_xst(sn3_hi, 0,  s3 + n);
            vec_xst(sn3_lo, 16, s3 + n);
            y_acc3.reg.val[0] = vec_madd(sn3_hi, C_hi, y_acc3.reg.val[0]);
            y_acc3.reg.val[1] = vec_madd(sn3_lo, C_lo, y_acc3.reg.val[1]);

            // Row 4
            __vector float sn4_hi = vec_madd(v_xdt4, B_hi, vec_mul(vec_xl(0,  s4 + n), v_dA));
            __vector float sn4_lo = vec_madd(v_xdt4, B_lo, vec_mul(vec_xl(16, s4 + n), v_dA));
            vec_xst(sn4_hi, 0,  s4 + n);
            vec_xst(sn4_lo, 16, s4 + n);
            y_acc4.reg.val[0] = vec_madd(sn4_hi, C_hi, y_acc4.reg.val[0]);
            y_acc4.reg.val[1] = vec_madd(sn4_lo, C_lo, y_acc4.reg.val[1]);

            // Row 5
            __vector float sn5_hi = vec_madd(v_xdt5, B_hi, vec_mul(vec_xl(0,  s5 + n), v_dA));
            __vector float sn5_lo = vec_madd(v_xdt5, B_lo, vec_mul(vec_xl(16, s5 + n), v_dA));
            vec_xst(sn5_hi, 0,  s5 + n);
            vec_xst(sn5_lo, 16, s5 + n);
            y_acc5.reg.val[0] = vec_madd(sn5_hi, C_hi, y_acc5.reg.val[0]);
            y_acc5.reg.val[1] = vec_madd(sn5_lo, C_lo, y_acc5.reg.val[1]);

            // Row 6
            __vector float sn6_hi = vec_madd(v_xdt6, B_hi, vec_mul(vec_xl(0,  s6 + n), v_dA));
            __vector float sn6_lo = vec_madd(v_xdt6, B_lo, vec_mul(vec_xl(16, s6 + n), v_dA));
            vec_xst(sn6_hi, 0,  s6 + n);
            vec_xst(sn6_lo, 16, s6 + n);
            y_acc6.reg.val[0] = vec_madd(sn6_hi, C_hi, y_acc6.reg.val[0]);
            y_acc6.reg.val[1] = vec_madd(sn6_lo, C_lo, y_acc6.reg.val[1]);

            // Row 7
            __vector float sn7_hi = vec_madd(v_xdt7, B_hi, vec_mul(vec_xl(0,  s7 + n), v_dA));
            __vector float sn7_lo = vec_madd(v_xdt7, B_lo, vec_mul(vec_xl(16, s7 + n), v_dA));
            vec_xst(sn7_hi, 0,  s7 + n);
            vec_xst(sn7_lo, 16, s7 + n);
            y_acc7.reg.val[0] = vec_madd(sn7_hi, C_hi, y_acc7.reg.val[0]);
            y_acc7.reg.val[1] = vec_madd(sn7_lo, C_lo, y_acc7.reg.val[1]);
          }

          __vector float y_hi = {
              y_acc0.reduce_sum(),
              y_acc1.reduce_sum(),
              y_acc2.reduce_sum(),
              y_acc3.reduce_sum()};

          __vector float y_lo = {
              y_acc4.reduce_sum(),
              y_acc5.reduce_sum(),
              y_acc6.reduce_sum(),
              y_acc7.reduce_sum()};

          // D skip connection
          if (D_ptr != nullptr) {
            y_hi = vec_madd(x_v.reg.val[0], v_D, y_hi);
            y_lo = vec_madd(x_v.reg.val[1], v_D, y_lo);
          }

          // Vectorized SiLU Gating
          if (z_h != nullptr) {
            vec_op::BF16Vec8 z_bf(z_h + d);
            vec_op::FP32Vec8 z_silu = vec_op::FP32Vec8(z_bf).silu();

            y_hi = vec_mul(y_hi, z_silu.reg.val[0]);
            y_lo = vec_mul(y_lo, z_silu.reg.val[1]);
          }

          // Convert and store all 8 BF16 elements in a single 16-byte write
          vec_op::FP32Vec8 y_out;
          y_out.reg.val[0] = y_hi;
          y_out.reg.val[1] = y_lo;
          vec_op::BF16Vec8(y_out).save(out_h + d);
        }

        // Remainder loop for any headdim tail not multiple of 8
        for (; d < headdim; ++d) {
          const float x_val = static_cast<float>(x_h[d]);
          float* s_bhd = s_bh + d * dstate;
          float y_val = 0.0f;
          for (int64_t n = 0; n < dstate; ++n) {
            const float Bn = static_cast<float>(B_g[n]);
            const float Cn = static_cast<float>(C_g[n]);
            const float s_new = s_bhd[n] * dA_val + x_val * dt_val * Bn;
            s_bhd[n] = s_new;
            y_val += s_new * Cn;
          }
          if (D_ptr != nullptr) y_val += x_val * D_val;
          if (z_h != nullptr) {
            const float z_val = static_cast<float>(z_h[d]);
            const float sig = (z_val >= 0.0f) ? 1.0f / (1.0f + std::exp(-z_val))
                                              : std::exp(z_val) / (1.0f + std::exp(z_val));
            y_val *= z_val * sig;
          }
          out_h[d] = static_cast<c10::BFloat16>(y_val);
        }
      }
    }
  }
}

template <typename scalar_t>
inline void mamba_chunk_scan_fwd(
    float* __restrict__ states_ptr,
    const scalar_t* __restrict__ x_ptr,
    const float* __restrict__ dt_ptr,
    const float* __restrict__ A_ptr,
    const scalar_t* __restrict__ B_ptr,
    const scalar_t* __restrict__ C_ptr,
    const float* __restrict__ D_ptr,
    const scalar_t* __restrict__ z_ptr,
    scalar_t* __restrict__ out_ptr,
    const int32_t* __restrict__ cu_seqlens,
    int64_t batch, int64_t nheads, int64_t ngroups, int64_t headdim,
    int64_t dstate) {

  if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
    mamba_chunk_scan_fwd_bf16(
        states_ptr, x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr,
        out_ptr, cu_seqlens, batch, nheads, ngroups, headdim, dstate);
  } else {
    mamba_chunk_scan_fwd_kernel<scalar_t>(
        states_ptr, x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr,
        out_ptr, cu_seqlens, batch, nheads, ngroups, headdim, dstate);
  }
}

}  // namespace vsx
}  // namespace mamba_cpu

#endif  // __powerpc__
