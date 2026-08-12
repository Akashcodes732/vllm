// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Power10 VSX dense BF16 linear kernel.
//
// Provides two operations:
//   vsx_pack_weight()  - Packs a [N, K] BF16 weight into MMA-friendly layout.
//   vsx_bf16_mm()      - Dense BF16 GEMM using xvbf16ger2pp MMA instructions.
//
// The packed layout is identical to the one used by the MoE backend
// (cpu_micro_gemm_vsx.hpp: MicroGemm<VSX>::pack_weight). Re-using the same
// kernel avoids duplicating intrinsic code.
//
// Alignment requirements (enforced at pack time):
//   N % 16 == 0  (NSize tile size for MMA)
//   K % 2  == 0  (two BF16 elements per MMA pair)

#pragma once
#ifdef __powerpc64__

#include <torch/all.h>
#include <algorithm>
#include <optional>

#include "cpu/micro_gemm/cpu_micro_gemm_impl.hpp"
#include "cpu/micro_gemm/cpu_micro_gemm_vsx.hpp"
#include "cpu/utils.hpp"

namespace cpu_linear_vsx {

using GemmT = cpu_micro_gemm::MicroGemm<cpu_utils::ISA::VSX, at::BFloat16>;

// Pack weight from row-major [N, K] into MMA-friendly interleaved layout.
// Output tensor has the same number of elements as input but in a different
// order. Calls MicroGemm<VSX>::pack_weight which is proven by the MoE backend.
//
// Args:
//   weight: contiguous BF16 2D tensor [N, K]
//
// Returns:
//   Packed BF16 tensor, same shape [N, K], MMA-interleaved.
inline at::Tensor pack_weight(at::Tensor& weight) {
  TORCH_CHECK(weight.dtype() == at::kBFloat16,
              "vsx_pack_weight: weight must be BFloat16, got ", weight.dtype());
  TORCH_CHECK(weight.dim() == 2,
              "vsx_pack_weight: weight must be 2D, got ", weight.dim(), "D");
  TORCH_CHECK(weight.is_contiguous(),
              "vsx_pack_weight: weight must be contiguous");

  const int32_t N = static_cast<int32_t>(weight.size(0));
  const int32_t K = static_cast<int32_t>(weight.size(1));

  TORCH_CHECK(N % GemmT::NSize == 0,
              "vsx_pack_weight: N (", N, ") must be a multiple of ",
              GemmT::NSize, " (NSize)");
  TORCH_CHECK(K % 2 == 0, "vsx_pack_weight: K (", K, ") must be even");

  at::Tensor packed = at::empty_like(weight);
  GemmT::pack_weight(weight.data_ptr<at::BFloat16>(),
                     packed.data_ptr<at::BFloat16>(), N, K);
  return packed;
}

// Dense BF16 GEMM: out = A @ packed_B^T  (+ optional bias)
//
// A:        BF16 tensor [..., K]
// packed_B: BF16 tensor [N, K] in MMA-packed layout (output of pack_weight)
// bias:     optional BF16 tensor [N]
//
// Accumulates in FP32 then converts back to BF16.
// Uses xvbf16ger2pp MMA for M > 4; scalar VSX FMADD for M <= 4.
//
// Returns:
//   BF16 tensor [..., N]
inline at::Tensor bf16_mm(const at::Tensor& A, const at::Tensor& packed_B,
                           const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(A.dtype() == at::kBFloat16, "vsx_bf16_mm: A must be BFloat16");
  TORCH_CHECK(packed_B.dtype() == at::kBFloat16,
              "vsx_bf16_mm: packed_B must be BFloat16");
  TORCH_CHECK(packed_B.dim() == 2,
              "vsx_bf16_mm: packed_B must be 2D, got ", packed_B.dim(), "D");
  TORCH_CHECK(packed_B.is_contiguous(),
              "vsx_bf16_mm: packed_B must be contiguous");

  const int64_t K = A.size(-1);
  const int64_t N = packed_B.size(0);
  const int64_t K_b = packed_B.size(1);
  const int64_t M = A.numel() / K;

  TORCH_CHECK(K == K_b, "vsx_bf16_mm: K mismatch: A has K=", K,
              " but packed_B has K=", K_b);

  if (bias.has_value()) {
    TORCH_CHECK(bias->dtype() == at::kBFloat16,
                "vsx_bf16_mm: bias must be BFloat16");
    TORCH_CHECK(bias->numel() == N,
                "vsx_bf16_mm: bias size (", bias->numel(),
                ") must equal N (", N, ")");
  }

  // Output shape: A.shape[:-1] + [N]
  auto out_shape = A.sizes().vec();
  out_shape.back() = N;
  at::Tensor out_bf16 = at::empty(out_shape, A.options());

  // FP32 accumulation buffer [M, N]
  at::Tensor c_fp32 =
      at::empty({M, N}, at::TensorOptions().dtype(at::kFloat));

  // Ensure A is contiguous for sequential row access
  const at::Tensor A_contig = A.contiguous();
  const at::BFloat16* a_ptr = A_contig.data_ptr<at::BFloat16>();
  const at::BFloat16* b_ptr = packed_B.data_ptr<at::BFloat16>();
  float* c_ptr = c_fp32.data_ptr<float>();
  at::BFloat16* out_ptr = out_bf16.data_ptr<at::BFloat16>();

  GemmT gemm;
  constexpr int32_t MaxM = GemmT::MaxMSize;   // 8
  constexpr int32_t NTile = GemmT::NSize;      // 16

  // b_n_group_stride: stride in elements between consecutive WeightOCGroupSize
  // (16) N-groups in the packed B tensor. Verified against MoE call pattern:
  //   gemm(A, B_tile, C, m, k, lda=K, b_n_group_stride=K, ldc, accum_c)
  const int64_t b_n_group_stride = K;
  const int64_t lda = K;
  const int64_t ldc = N;

  // Tile over N (output columns) then M (input rows).
  // Each N-tile is NTile=16 columns wide.
  // In the packed layout, consecutive N-tiles are stored contiguously,
  // each occupying K*NTile elements, so the pointer offset is n_off * K.
  for (int64_t n_off = 0; n_off < N; n_off += NTile) {
    const at::BFloat16* b_tile = b_ptr + n_off * K;

    for (int64_t m_off = 0; m_off < M; m_off += MaxM) {
      const int32_t m_actual = static_cast<int32_t>(
          std::min(static_cast<int64_t>(MaxM), M - m_off));

      const at::BFloat16* a_tile = a_ptr + m_off * K;
      float* c_tile = c_ptr + m_off * N + n_off;

      gemm.gemm(const_cast<at::BFloat16*>(a_tile),
                const_cast<at::BFloat16*>(b_tile), c_tile, m_actual,
                static_cast<int32_t>(K), lda, b_n_group_stride, ldc,
                /*accum_c=*/false);
    }
  }

  // Epilogue: FP32 -> BF16, with optional bias.
  // bias_epilogue / default_epilogue are templated on compile-time n_size=16.
  // Loop over N-tiles applying the epilogue per tile.
  const int32_t n_tiles = static_cast<int32_t>(N / NTile);
  for (int32_t n_t = 0; n_t < n_tiles; ++n_t) {
    float* c_tile = c_ptr + n_t * NTile;
    at::BFloat16* out_tile = out_ptr + n_t * NTile;

    if (bias.has_value()) {
      at::BFloat16* bias_tile =
          const_cast<at::BFloat16*>(bias->data_ptr<at::BFloat16>()) +
          n_t * NTile;
      cpu_micro_gemm::bias_epilogue<NTile, at::BFloat16>(
          c_tile, out_tile, bias_tile, static_cast<int32_t>(M), ldc, ldc);
    } else {
      cpu_micro_gemm::default_epilogue<NTile, at::BFloat16>(
          c_tile, out_tile, static_cast<int32_t>(M), ldc, ldc);
    }
  }

  return out_bf16;
}

}  // namespace cpu_linear_vsx

#endif  // __powerpc64__
