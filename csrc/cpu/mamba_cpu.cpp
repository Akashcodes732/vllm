// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// CPU at::Tensor wrappers for Mamba decode-step kernels defined in
// mamba_kernels.hpp.

#include "cpu/mamba_kernels.hpp"

#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/util/Optional.h>

// ---------------------------------------------------------------------------
// causal_conv1d_update
//
// x            : (batch, dim) [decode] or (batch, dim, seqlen) or
// (total_tokens, dim) [varlen] conv_state   : (num_cache, dim, state_len)  –
// updated in-place weight       : (dim, width) bias         : (dim,) optional
// activation   : "silu" / "swish" / None / True / False
// conv_state_indices : (batch,) int32 optional
// query_start_loc    : (batch+1,) int32 optional  [varlen mode]
// pad_slot_id  : int
//
// Returns x (overwritten with output) cast back to original dtype.
// ---------------------------------------------------------------------------
at::Tensor causal_conv1d_update_cpu_impl(
    at::Tensor& x,  // modified in-place (re-typed to float32)
    at::Tensor& conv_state, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<std::string>& activation,
    const c10::optional<at::Tensor>& conv_state_indices,
    const c10::optional<at::Tensor>& query_start_loc, int64_t pad_slot_id) {
  // ------------------------------------------------------------------
  // Resolve activation
  // ------------------------------------------------------------------
  bool do_silu = false;
  if (activation.has_value()) {
    const std::string& act = activation.value();
    do_silu = (act == "silu" || act == "swish");
  }

  // ------------------------------------------------------------------
  // Keep original dtype; work in float32
  // ------------------------------------------------------------------
  at::ScalarType orig_dtype = x.scalar_type();
  at::Tensor x_f32 = x.to(at::kFloat);
  at::Tensor state_f32 = conv_state.to(at::kFloat);
  at::Tensor w_f32 = weight.to(at::kFloat);
  at::Tensor bias_f32;
  bool has_bias = bias.has_value() && bias.value().defined();
  if (has_bias) bias_f32 = bias.value().to(at::kFloat);

  // ------------------------------------------------------------------
  // Dimensions
  // ------------------------------------------------------------------
  bool is_2d = (x_f32.dim() == 2);
  bool is_varlen = query_start_loc.has_value();
  at::Tensor x_3d;

  if (is_varlen) {
    // x: (total_tokens, dim) – treat as (dim, total_tokens) then wrap
    // We reshape to (1, dim, total_tokens) and iterate via query_start_loc
    // For simplicity, call the PyTorch fallback for varlen path.
    // (Varlen is only used during prefill-style chunked decode, not the
    //  critical single-token decode path.)
    TORCH_CHECK(false,
                "causal_conv1d_update_cpu_impl: varlen mode not yet supported "
                "in C++ kernel; use the PyTorch fallback.");
  }

  if (is_2d) {
    x_3d = x_f32.unsqueeze(-1);  // (batch, dim, 1)
  } else {
    x_3d = x_f32;
  }

  int64_t batch = x_3d.size(0);
  int64_t dim = x_3d.size(1);
  int64_t seqlen = x_3d.size(2);
  int64_t width = w_f32.size(1);
  int64_t state_len = state_f32.size(2);

  // Ensure contiguous
  x_3d = x_3d.contiguous();
  state_f32 = state_f32.contiguous();
  at::Tensor w_cont = w_f32.contiguous();

  // Output: same shape as x_3d (written in place)
  at::Tensor out = x_3d.clone();

  const int32_t* cache_idx_ptr = nullptr;
  at::Tensor cache_idx_int;
  if (conv_state_indices.has_value()) {
    cache_idx_int = conv_state_indices.value().to(at::kInt).contiguous();
    cache_idx_ptr = cache_idx_int.data_ptr<int32_t>();
  }

  mamba_cpu::causal_conv1d_update_kernel(
      x_3d.data_ptr<float>(), state_f32.data_ptr<float>(),
      w_cont.data_ptr<float>(),
      has_bias ? bias_f32.contiguous().data_ptr<float>() : nullptr,
      out.data_ptr<float>(), cache_idx_ptr, static_cast<int32_t>(pad_slot_id),
      batch, dim, seqlen, width, state_len, do_silu);

  // Write back updated state
  conv_state.copy_(state_f32.to(conv_state.scalar_type()));

  at::Tensor out_final = is_2d ? out.squeeze(-1) : out;
  return out_final.to(orig_dtype);
}

// ---------------------------------------------------------------------------
// selective_state_update
//
// Decode-step (single token or short varlen sequence) SSM recurrence.
//
// Tensor shapes follow _selective_state_update_cuda convention but all tensors
// are on CPU.
// ---------------------------------------------------------------------------
void selective_state_update_cpu_impl(
    at::Tensor& state,    // (batch_or_nstates, nheads, dim, dstate) – in-place
    const at::Tensor& x,  // (N, nheads, dim)  where N == batch
                          // (cu_seqlens==None) or total tokens
    const at::Tensor& dt,
    const at::Tensor& A,                       // (nheads, dim, dstate)
    const at::Tensor& B,                       // (N, ngroups, dstate)
    const at::Tensor& C,                       // (N, ngroups, dstate)
    const c10::optional<at::Tensor>& D,        // (nheads, dim)
    const c10::optional<at::Tensor>& z,        // (N, nheads, dim)
    const c10::optional<at::Tensor>& dt_bias,  // (nheads, dim)
    bool dt_softplus,
    const c10::optional<at::Tensor>& state_batch_indices,      // (N,) int32
    const c10::optional<at::Tensor>& dst_state_batch_indices,  // (N,) int32
    int64_t null_block_id,
    at::Tensor& out,  // (N, nheads, dim) – written
    const c10::optional<at::Tensor>& num_accepted_tokens,  // (N,) int32
    const c10::optional<at::Tensor>& cu_seqlens            // (N+1,) int32
) {
  // ------------------------------------------------------------------
  // Work in float32
  // ------------------------------------------------------------------
  at::Tensor state_f32 = state.to(at::kFloat);
  at::Tensor x_f32 = x.to(at::kFloat).contiguous();
  at::Tensor dt_f32 = dt.to(at::kFloat).contiguous();
  at::Tensor A_f32 = A.to(at::kFloat).contiguous();
  at::Tensor B_f32 = B.to(at::kFloat).contiguous();
  at::Tensor C_f32 = C.to(at::kFloat).contiguous();
  at::Tensor out_f32 = at::zeros_like(x_f32);  // (N, nheads, dim)
  state_f32 = state_f32.contiguous();

  at::Tensor D_f32, z_f32, dtbias_f32;
  bool has_D = D.has_value() && D.value().defined();
  bool has_z = z.has_value() && z.value().defined();
  bool has_dt_bias = dt_bias.has_value() && dt_bias.value().defined();
  if (has_D) D_f32 = D.value().to(at::kFloat).contiguous();
  if (has_z) z_f32 = z.value().to(at::kFloat).contiguous();
  if (has_dt_bias) dtbias_f32 = dt_bias.value().to(at::kFloat).contiguous();

  // ------------------------------------------------------------------
  // Dimensions
  // ------------------------------------------------------------------
  //  state_f32: (nstates, nheads, dim, dstate)
  int64_t nstates = state_f32.size(0);
  int64_t nheads = state_f32.size(1);
  int64_t dim = state_f32.size(2);
  int64_t dstate = state_f32.size(3);
  int64_t ngroups = B_f32.size(1);

  int64_t N;
  if (cu_seqlens.has_value()) {
    N = cu_seqlens.value().size(0) - 1;
  } else {
    N = x_f32.size(0);
  }

  // Strides (all contiguous after .contiguous())
  int64_t stride_state_n = state_f32.stride(0);
  int64_t stride_state_h = state_f32.stride(1);
  int64_t stride_state_d = state_f32.stride(2);
  int64_t stride_xdt_n = x_f32.stride(0);
  int64_t stride_xdt_h = x_f32.stride(1);
  int64_t stride_BC_n = B_f32.stride(0);
  int64_t stride_BC_g = B_f32.stride(1);
  int64_t stride_out_n = out_f32.stride(0);
  int64_t stride_out_h = out_f32.stride(1);

  // Optional pointer helpers
  const int32_t* sbi_ptr = nullptr;
  const int32_t* dsbi_ptr = nullptr;
  const int32_t* nat_ptr = nullptr;
  const int32_t* csl_ptr = nullptr;

  at::Tensor sbi_int, dsbi_int, nat_int, csl_int;
  if (state_batch_indices.has_value()) {
    sbi_int = state_batch_indices.value().to(at::kInt).contiguous();
    sbi_ptr = sbi_int.data_ptr<int32_t>();
  }
  if (dst_state_batch_indices.has_value()) {
    dsbi_int = dst_state_batch_indices.value().to(at::kInt).contiguous();
    dsbi_ptr = dsbi_int.data_ptr<int32_t>();
  }
  if (num_accepted_tokens.has_value()) {
    nat_int = num_accepted_tokens.value().to(at::kInt).contiguous();
    nat_ptr = nat_int.data_ptr<int32_t>();
  }
  if (cu_seqlens.has_value()) {
    csl_int = cu_seqlens.value().to(at::kInt).contiguous();
    csl_ptr = csl_int.data_ptr<int32_t>();
  }

  mamba_cpu::selective_state_update_kernel(
      state_f32.data_ptr<float>(), stride_state_n, stride_state_h,
      stride_state_d, x_f32.data_ptr<float>(), dt_f32.data_ptr<float>(),
      stride_xdt_n, stride_xdt_h, A_f32.data_ptr<float>(),
      B_f32.data_ptr<float>(), C_f32.data_ptr<float>(), stride_BC_n,
      stride_BC_g, has_D ? D_f32.data_ptr<float>() : nullptr,
      has_z ? z_f32.data_ptr<float>() : nullptr,
      has_dt_bias ? dtbias_f32.data_ptr<float>() : nullptr,
      out_f32.data_ptr<float>(), stride_out_n, stride_out_h, sbi_ptr, dsbi_ptr,
      static_cast<int32_t>(null_block_id), nat_ptr, csl_ptr, N, nheads, ngroups,
      dim, dstate, dt_softplus);

  // Write back
  state.copy_(state_f32.to(state.scalar_type()));
  out.copy_(out_f32.to(out.scalar_type()));
}
