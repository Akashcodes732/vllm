// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Translation unit for Power10 VSX dense BF16 linear ops.
// Provides the C-linkage function bodies that torch_bindings.cpp declares
// as extern and registers as Torch custom ops.

#if defined(__powerpc__) || defined(__powerpc64__) || defined(__PPC64__)

#include "cpu/cpu_linear_vsx.hpp"

at::Tensor vsx_pack_weight(at::Tensor& weight) {
  return cpu_linear_vsx::pack_weight(weight);
}

at::Tensor vsx_bf16_mm(const at::Tensor& A, const at::Tensor& packed_B,
                        const std::optional<at::Tensor>& bias) {
  return cpu_linear_vsx::bf16_mm(A, packed_B, bias);
}

#endif  // defined(__powerpc__) || defined(__powerpc64__) || defined(__PPC64__)
