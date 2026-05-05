// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused CPU vector kernels for Mamba decode-step hotspots:
//   - causal_conv1d_update  (depthwise 1-D conv state roll + compute)
//   - selective_state_update (SSM recurrence, single-step)
//
// The kernels operate in float32 throughout (with BF16 input/output conversion
// at the boundary) so that the compiler can auto-vectorise the inner loops
// with -O3 on any ISA (AVX2, NEON, VSX, …) without ISA-specific intrinsics.

#pragma once

#include <cmath>
#include <cstring>
#include <cstdint>

namespace mamba_cpu {

// ---------------------------------------------------------------------------
// causal_conv1d_update
//
// For each batch sequence and each feature channel d:
//
//   state[cache_idx, d, :]  = [state[cache_idx, d, 1:],  x[b, d]]   (roll)
//   out[b, d] = sum_k( state[cache_idx, d, k] * weight[d, k] ) + bias[d]
//   if silu: out[b, d] = out[b, d] * sigmoid(out[b, d])
//
// Tensors (host pointers, all contiguous):
//   x_ptr        : float*, shape (batch, dim) or (batch, dim, seqlen)
//   state_ptr    : float*, shape (num_cache, dim, state_len)
//   weight_ptr   : float*, shape (dim, width)
//   bias_ptr     : float* or nullptr, shape (dim,)
//   out_ptr      : float*, same shape as x
//   cache_idxs   : int32_t*, shape (batch,)   – may be nullptr (use b)
//   pad_slot_id  : cache index value to skip
//   batch, dim, seqlen, width, state_len – dimensions
//   do_silu      : apply SiLU activation
// ---------------------------------------------------------------------------
inline void causal_conv1d_update_kernel(
    const float* __restrict__ x_ptr, float* __restrict__ state_ptr,
    const float* __restrict__ weight_ptr, const float* __restrict__ bias_ptr,
    float* __restrict__ out_ptr, const int32_t* __restrict__ cache_idxs,
    int32_t pad_slot_id, int64_t batch, int64_t dim, int64_t seqlen,
    int64_t width, int64_t state_len, bool do_silu) {
  // state layout: [num_cache, dim, state_len]
  // x / out layout: [batch, dim, seqlen]  (seqlen == 1 for plain decode)
  for (int64_t b = 0; b < batch; ++b) {
    int64_t cache_idx = (cache_idxs != nullptr) ? cache_idxs[b] : b;
    if (cache_idx == pad_slot_id) continue;

    for (int64_t t = 0; t < seqlen; ++t) {
      const float* x_b = x_ptr + (b * dim * seqlen + t);  // stride seqlen
      float* out_b = out_ptr + (b * dim * seqlen + t);    // stride seqlen
      float* s = state_ptr + cache_idx * dim * state_len;

      for (int64_t d = 0; d < dim; ++d) {
        float x_val = x_b[d * seqlen];

        float* sd = s + d * state_len;  // state for channel d

        // Compute the dot product with weight[d, :] = weight + d*width
        const float* w = weight_ptr + d * width;
        float acc = (bias_ptr != nullptr) ? bias_ptr[d] : 0.0f;
        
        // The convolution window is [sd[0], ..., sd[state_len-1], x_val]
        for (int64_t k = 0; k < state_len; ++k) {
          acc += w[k] * sd[k];
        }
        acc += w[state_len] * x_val;

        // Roll state left, append x_val at the end
        if (state_len > 1) {
          std::memmove(sd, sd + 1, (state_len - 1) * sizeof(float));
        }
        if (state_len > 0) {
          sd[state_len - 1] = x_val;
        }

        if (do_silu) {
          acc = acc / (1.0f + std::exp(-acc));
        }
        out_b[d * seqlen] = acc;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// selective_state_update (SSM recurrence, single decode step OR varlen decode)
//
// Algorithm (per sequence, per head, per token):
//
//   if dt_bias: dt += dt_bias
//   if dt_softplus: dt = log1p(exp(dt))
//   dA = exp(A * dt)          shape (nheads, dim, dstate)
//   B_exp = B.repeat_interleave(nheads_per_group)   (nheads, dstate)
//   C_exp = C.repeat_interleave(nheads_per_group)   (nheads, dstate)
//   dBx = B_exp * (x * dt)    shape (nheads, dim, dstate)
//   state = state * dA + dBx
//   out = (state * C_exp).sum(-1)
//   if D: out += x * D
//   if z: out *= z * sigmoid(z)
//
// All tensors are in float32 (caller converts before passing).
// Strides: state (nheads, dim, dstate), rest as 1-D slices indexed by
//           (seq_idx, head, dim, dstate) as needed.
//
// Parameters match those in the Python fallback _selective_state_update_cpu
// but receive raw pointers + sizes.
// ---------------------------------------------------------------------------
template <typename scalar_t>
inline void selective_state_update_kernel(
    // state: [nstates, nheads, dim, dstate]  – modified in place
    scalar_t* __restrict__ state_ptr,
    int64_t stride_state_n,  // stride along nstates dim
    int64_t stride_state_h,  // stride along nheads dim
    int64_t stride_state_d,  // stride along dim dim
    // x, dt: [N, nheads, dim]
    const float* __restrict__ x_ptr, const float* __restrict__ dt_ptr,
    int64_t stride_xdt_n,  // stride along N dim
    int64_t stride_xdt_h,  // stride along nheads dim
    // A: [nheads, dim, dstate]
    const float* __restrict__ A_ptr, int64_t stride_A_h,
    // B, C: [N, ngroups, dstate]
    const float* __restrict__ B_ptr, const float* __restrict__ C_ptr,
    int64_t stride_BC_n,  // stride along N dim
    int64_t stride_BC_g,  // stride along ngroups dim
    // D: [nheads, dim] or nullptr
    const float* __restrict__ D_ptr, int64_t stride_D_h,
    // z: [N, nheads, dim] or nullptr
    const float* __restrict__ z_ptr,
    // dt_bias: [nheads, dim] or nullptr
    const float* __restrict__ dt_bias_ptr, int64_t stride_dtbias_h,
    // out: [N, nheads, dim]  – written in place
    float* __restrict__ out_ptr, int64_t stride_out_n, int64_t stride_out_h,
    // state_batch_indices: [N] or nullptr (use seq_idx)
    const int32_t* __restrict__ state_batch_indices,
    // dst_state_batch_indices: [N] or nullptr
    const int32_t* __restrict__ dst_state_batch_indices, int32_t null_block_id,
    // num_accepted_tokens: [N] or nullptr
    const int32_t* __restrict__ num_accepted_tokens,
    // cu_seqlens: [N+1] or nullptr
    const int32_t* __restrict__ cu_seqlens,
    // dimensions
    int64_t N,  // number of sequences (or batch)
    int64_t nheads, int64_t ngroups, int64_t dim, int64_t dstate,
    bool dt_softplus) {
  int64_t nheads_per_group = nheads / ngroups;

  for (int64_t seq_idx = 0; seq_idx < N; ++seq_idx) {
    int64_t bos, seq_len;
    if (cu_seqlens != nullptr) {
      bos = cu_seqlens[seq_idx];
      seq_len = cu_seqlens[seq_idx + 1] - bos;
    } else {
      bos = seq_idx;
      seq_len = 1;
    }

    // Determine state read index
    int64_t state_read_idx;
    if (state_batch_indices != nullptr) {
      state_read_idx = state_batch_indices[seq_idx];
      if (state_read_idx == null_block_id) continue;
    } else {
      state_read_idx = seq_idx;
    }

    // Determine state write index
    int64_t state_write_idx;
    if (num_accepted_tokens == nullptr) {
      if (dst_state_batch_indices != nullptr) {
        state_write_idx = dst_state_batch_indices[seq_idx];
      } else {
        state_write_idx = state_read_idx;
      }
    } else {
      state_write_idx = -1;  // written per-token inside the loop
    }

    // Per-sequence state buffer
    scalar_t* s = state_ptr + state_read_idx * stride_state_n;

    for (int64_t t = 0; t < seq_len; ++t) {
      int64_t token_idx = bos + t;

      const float* x_tok = x_ptr + token_idx * stride_xdt_n;
      const float* dt_tok = dt_ptr + token_idx * stride_xdt_n;
      const float* B_tok = B_ptr + token_idx * stride_BC_n;
      const float* C_tok = C_ptr + token_idx * stride_BC_n;
      float* out_tok = out_ptr + token_idx * stride_out_n;

#pragma omp parallel for
      for (int64_t h = 0; h < nheads; ++h) {
        int64_t g = h / nheads_per_group;
        const float* x_h = x_tok + h * stride_xdt_h;
        const float* dt_h = dt_tok + h * stride_xdt_h;
        const float* B_g = B_tok + g * stride_BC_g;
        const float* C_g = C_tok + g * stride_BC_g;
        const float* A_h = A_ptr + h * stride_A_h;
        const float* dt_bias_h =
            (dt_bias_ptr != nullptr) ? dt_bias_ptr + h * stride_dtbias_h : nullptr;
        const float* D_h = (D_ptr != nullptr) ? D_ptr + h * stride_D_h : nullptr;
        const float* z_h =
            (z_ptr != nullptr)
                ? z_ptr + token_idx * stride_xdt_n + h * stride_xdt_h
                : nullptr;
        float* out_h = out_tok + h * stride_out_h;
        scalar_t* s_h = s + h * stride_state_h;

        for (int64_t d = 0; d < dim; ++d) {
          float x_val = x_h[d];
          float dt_val = dt_h[d];
          if (dt_bias_h != nullptr) dt_val += dt_bias_h[d];
          if (dt_softplus) {
            // log1p(exp(dt)) — numerically stable
            dt_val = (dt_val <= 20.0f) ? std::log1pf(std::expf(dt_val)) : dt_val;
          }

          float out_val = 0.0f;
          scalar_t* s_hd = s_h + d * dstate;
          const float* A_hd = A_h + d * dstate;
          for (int64_t n = 0; n < dstate; ++n) {
            float dA = std::expf(A_hd[n] * dt_val);
            float dBx = B_g[n] * x_val * dt_val;
            float s_old = static_cast<float>(s_hd[n]);
            float s_new = s_old * dA + dBx;
            s_hd[n] = static_cast<scalar_t>(s_new);
            out_val += s_new * C_g[n];
          }

          if (D_h != nullptr) out_val += x_val * D_h[d];
          if (z_h != nullptr) {
            float z_val = z_h[d];
            // Stable SiLU: z * sigmoid(z)
            float sigmoid_z = (z_val >= 0) ? 
                1.0f / (1.0f + std::expf(-z_val)) : 
                std::expf(z_val) / (1.0f + std::expf(z_val));
            out_val *= z_val * sigmoid_z;
          }
          out_h[d] = out_val;
        }
      }

      // Handle spec-decoding token-level dst write
      if (num_accepted_tokens != nullptr &&
          dst_state_batch_indices != nullptr) {
        int64_t token_dst_idx = dst_state_batch_indices[seq_idx * seq_len + t];
        if (token_dst_idx != null_block_id && token_dst_idx != state_read_idx) {
          scalar_t* dst_s = state_ptr + token_dst_idx * stride_state_n;
          std::memmove(dst_s, s, nheads * stride_state_h * sizeof(scalar_t));
        }
      }
    }

    // Write final state
    if (num_accepted_tokens == nullptr && state_write_idx != null_block_id &&
        state_write_idx != state_read_idx) {
      scalar_t* dst_s = state_ptr + state_write_idx * stride_state_n;
      std::memmove(dst_s, s, nheads * stride_state_h * sizeof(scalar_t));
    }
  }
}

}  // namespace mamba_cpu
