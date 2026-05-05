// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// CPU at::Tensor wrappers for Mamba decode-step kernels defined in
// mamba_kernels.hpp.

#include "cpu/mamba_kernels.hpp"

#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/util/Optional.h>

#include "cpu_types.hpp"

// ---------------------------------------------------------------------------
// causal_conv1d_update
// ---------------------------------------------------------------------------
at::Tensor causal_conv1d_update_cpu_impl(
    at::Tensor& x,  // modified in-place (re-typed to float32)
    at::Tensor& conv_state, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<std::string>& activation,
    const c10::optional<at::Tensor>& conv_state_indices,
    const c10::optional<at::Tensor>& query_start_loc, int64_t pad_slot_id) {
  
  bool do_silu = false;
  if (activation.has_value()) {
    const std::string& act = activation.value();
    do_silu = (act == "silu" || act == "swish");
  }

  // Causal conv still works in float32 for now (minimal overhead compared to SSM)
  at::Tensor x_f32 = x.to(at::kFloat).contiguous();
  at::Tensor state_f32 = conv_state.to(at::kFloat).contiguous();
  at::Tensor w_f32 = weight.to(at::kFloat).contiguous();
  at::Tensor bias_f32;
  if (bias.has_value() && bias.value().defined()) bias_f32 = bias.value().to(at::kFloat).contiguous();

  int64_t batch = x_f32.size(0);
  int64_t dim = x_f32.size(1);
  int64_t seqlen = (x_f32.dim() == 3) ? x_f32.size(2) : 1;
  int64_t width = w_f32.size(1);
  int64_t state_len = state_f32.size(2);

  at::Tensor out_f32 = at::empty_like(x_f32);

  const int32_t* cache_idx_ptr = nullptr;
  at::Tensor cache_idx_int;
  if (conv_state_indices.has_value()) {
    cache_idx_int = conv_state_indices.value().to(at::kInt).contiguous();
    cache_idx_ptr = cache_idx_int.data_ptr<int32_t>();
  }

  mamba_cpu::causal_conv1d_update_kernel(
      x_f32.data_ptr<float>(), state_f32.data_ptr<float>(),
      w_f32.data_ptr<float>(), bias_f32.defined() ? bias_f32.data_ptr<float>() : nullptr,
      out_f32.data_ptr<float>(), cache_idx_ptr, static_cast<int32_t>(pad_slot_id),
      batch, dim, seqlen, width, state_len, do_silu);

  conv_state.copy_(state_f32.to(conv_state.scalar_type()));
  return out_f32.to(x.scalar_type());
}

// ---------------------------------------------------------------------------
// selective_state_update
// ---------------------------------------------------------------------------
void selective_state_update_cpu_impl(
    at::Tensor& state,    // (nstates, nheads, dim, dstate)
    const at::Tensor& x,  // (N, nheads, dim)
    const at::Tensor& dt,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const c10::optional<at::Tensor>& D,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias,
    bool dt_softplus,
    const c10::optional<at::Tensor>& state_batch_indices,
    const c10::optional<at::Tensor>& dst_state_batch_indices,
    int64_t null_block_id,
    at::Tensor& out,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    const c10::optional<at::Tensor>& cu_seqlens
) {
  // Optimization: No full-tensor conversions to float32 here.
  // We pass original dtypes and let the kernel handle it via vec_op.
  
  int64_t nheads = state.size(1);
  int64_t dim = state.size(2);
  int64_t dstate = state.size(3);
  int64_t N = x.size(0);
  int64_t ngroups = B.size(1);

  // Strides (crucial for correctness)
  int64_t stride_state_n = state.stride(0);
  int64_t stride_state_h = state.stride(1);
  int64_t stride_state_d = state.stride(2);
  int64_t stride_xdt_n = x.stride(0);
  int64_t stride_xdt_h = x.stride(1);
  int64_t stride_A_h = A.stride(0);
  int64_t stride_BC_n = B.stride(0);
  int64_t stride_BC_g = B.stride(1);
  int64_t stride_out_n = out.stride(0);
  int64_t stride_out_h = out.stride(1);
  int64_t stride_D_h = D.has_value() ? D.value().stride(0) : 0;
  int64_t stride_dtbias_h = dt_bias.has_value() ? dt_bias.value().stride(0) : 0;

  // Optional pointers
  const int32_t* sbi_ptr = state_batch_indices.has_value() ? state_batch_indices.value().data_ptr<int32_t>() : nullptr;
  const int32_t* dsbi_ptr = dst_state_batch_indices.has_value() ? dst_state_batch_indices.value().data_ptr<int32_t>() : nullptr;
  const int32_t* nat_ptr = num_accepted_tokens.has_value() ? num_accepted_tokens.value().data_ptr<int32_t>() : nullptr;
  const int32_t* csl_ptr = cu_seqlens.has_value() ? cu_seqlens.value().data_ptr<int32_t>() : nullptr;

  // out is often bfloat16, but kernel math is float32.
  // We use a temporary float32 buffer per token if needed, or just convert at the end.
  // For simplicity and correctness, we use a float32 out buffer and copy back.
  at::Tensor out_f32 = at::empty_like(out, at::kFloat);

  VLLM_DISPATCH_FLOATING_TYPES(state.scalar_type(), "ssu_state", [&] {
    using state_t = scalar_t;
    VLLM_DISPATCH_FLOATING_TYPES(x.scalar_type(), "ssu_input", [&] {
      using input_t = scalar_t;
      mamba_cpu::selective_state_update_kernel<state_t, input_t>(
          state.data_ptr<state_t>(), stride_state_n, stride_state_h, stride_state_d,
          x.data_ptr<input_t>(), dt.data_ptr<input_t>(), stride_xdt_n, stride_xdt_h,
          A.data_ptr<input_t>(), stride_A_h,
          B.data_ptr<input_t>(), C.data_ptr<input_t>(), stride_BC_n, stride_BC_g,
          D.has_value() ? D.value().data_ptr<input_t>() : nullptr, stride_D_h,
          z.has_value() ? z.value().data_ptr<input_t>() : nullptr,
          dt_bias.has_value() ? dt_bias.value().data_ptr<input_t>() : nullptr, stride_dtbias_h,
          out_f32.data_ptr<float>(), stride_out_n, stride_out_h,
          sbi_ptr, dsbi_ptr, static_cast<int32_t>(null_block_id),
          nat_ptr, csl_ptr, N, nheads, ngroups, dim, dstate, dt_softplus);
    });
  });

  out.copy_(out_f32);
}
