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
#include <ATen/Parallel.h>
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

  const at::Tensor A_contig = A.contiguous();
  const at::BFloat16* a_ptr = A_contig.data_ptr<at::BFloat16>();
  const at::BFloat16* b_ptr = packed_B.data_ptr<at::BFloat16>();
  at::BFloat16* out_ptr = out_bf16.data_ptr<at::BFloat16>();
  const at::BFloat16* bias_ptr =
      bias.has_value() ? bias->data_ptr<at::BFloat16>() : nullptr;

  constexpr int32_t MaxM = GemmT::MaxMSize;  // 8
  constexpr int32_t NTile = GemmT::NSize;    // 16
  
  // Power10 L1d is 32 KB per core. 
  // A chunk of A: 8 * 256 * 2 bytes = 4 KB.
  // A chunk of packed B: 16 * 256 * 2 bytes = 8 KB.
  // Total working set = 12 KB (fits comfortably in 32 KB L1d).
  constexpr int64_t K_BLOCK = 256;

  const int32_t num_n_tiles = static_cast<int32_t>(N / NTile);
  const int32_t num_m_tiles = static_cast<int32_t>((M + MaxM - 1) / MaxM);

  // PyTorch's native thread pool, parallelizing over 2D tile blocks.
  // This avoids OpenMP oversubscription when vLLM is already using threads.
  at::parallel_for_2d(
      0, num_m_tiles, 0, num_n_tiles, 1,
      [&](int64_t m_start_t, int64_t m_end_t, int64_t n_start_t, int64_t n_end_t) {
        
        // Per-thread, per-tile FP32 accumulation buffer: 8*16*4 = 512 bytes.
        alignas(64) float c_tile_buf[MaxM * NTile];
        GemmT gemm;

        for (int64_t m_t = m_start_t; m_t < m_end_t; ++m_t) {
          int64_t m_off = m_t * MaxM;
          int32_t m_actual = static_cast<int32_t>(
              std::min(static_cast<int64_t>(MaxM), M - m_off));
              
          for (int64_t n_t = n_start_t; n_t < n_end_t; ++n_t) {
            const at::BFloat16* b_tile =
                b_ptr + static_cast<int64_t>(n_t) * NTile * K;
            at::BFloat16* out_col = out_ptr + n_t * NTile;
            const at::BFloat16* bias_col =
                bias_ptr ? bias_ptr + n_t * NTile : nullptr;

            // K-Blocking loop: Keep working set inside L1 cache
            for (int64_t k_off = 0; k_off < K; k_off += K_BLOCK) {
              int32_t k_actual = static_cast<int32_t>(
                  std::min(K_BLOCK, K - k_off));
              
              // accumulate into c_tile_buf after the first K-block
              bool accum = (k_off != 0);

              gemm.gemm(const_cast<at::BFloat16*>(a_ptr + m_off * K + k_off),
                        const_cast<at::BFloat16*>(b_tile + k_off * NTile),
                        c_tile_buf,
                        m_actual,
                        k_actual,
                        static_cast<int32_t>(K),    // lda
                        static_cast<int32_t>(K),    // b_n_group_stride
                        NTile,                      // ldc = tile width
                        accum);
            }

            // Epilogue: FP32 tile -> BF16, writing into the correct output column.
            at::BFloat16* out_tile_row = out_col + m_off * N;
            if (bias_col) {
              cpu_micro_gemm::bias_epilogue<NTile, at::BFloat16>(
                  c_tile_buf, out_tile_row,
                  const_cast<at::BFloat16*>(bias_col),
                  m_actual, NTile, static_cast<int32_t>(N));
            } else {
              cpu_micro_gemm::default_epilogue<NTile, at::BFloat16>(
                  c_tile_buf, out_tile_row, m_actual, NTile,
                  static_cast<int32_t>(N));
            }
          }
        }
      });

  return out_bf16;
}

}  // namespace cpu_linear_vsx

#endif  // __powerpc64__
