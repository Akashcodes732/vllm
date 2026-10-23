// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef CPU_ATTN_VSX_HPP
#define CPU_ATTN_VSX_HPP

#include "cpu_attn_impl.hpp"
#include <altivec.h>
#include <type_traits>

namespace cpu_attention {

namespace {

// ppc64le Vector = 16 bytes (128 bits)
#define BLOCK_SIZE_ALIGNMENT 32
#define HEAD_SIZE_ALIGNMENT 32
#define MAX_Q_HEAD_NUM_PER_ITER 16

template <typename kv_cache_t>
FORCE_INLINE void load_row8_B_as_f32(const kv_cache_t* p, __vector float& b0,
                                     __vector float& b1);

// [1] Float Specialization
template <>
FORCE_INLINE void load_row8_B_as_f32<float>(const float* p, __vector float& b0,
                                            __vector float& b1) {
  b0 = vec_xl(0, const_cast<float*>(p));
  b1 = vec_xl(0, const_cast<float*>(p + 4));
}

// [2] BFloat16 Specialization (Little Endian ppc64le)
// On ppc64le (LE): BF16 bits should land in the HIGH 16 bits of each float32.
// Byte layout of float32 on LE: [byte0(LSB), byte1, byte2, byte3(MSB)]
// We need BF16 in bytes2-3 (high half) with bytes0-1 zeroed.
// vec_mergeh on LE interleaves elements 0..3: result_i = {a[i], b[i]}
// So vec_mergeh(zeros_u16, raw_u16) gives for each uint16 pair:
//   uint16[2i]   = zeros[i]  -> low 16 bits of uint32  -> zeroed mantissa LSBs
//   uint16[2i+1] = raw[i]    -> high 16 bits of uint32 -> BF16 bits
// Cast to float32 gives exactly (bf16_bits << 16) per element.
template <>
FORCE_INLINE void load_row8_B_as_f32<c10::BFloat16>(const c10::BFloat16* p,
                                                    __vector float& b0,
                                                    __vector float& b1) {
  __vector unsigned short raw = vec_xl(
      0, reinterpret_cast<unsigned short*>(const_cast<c10::BFloat16*>(p)));
  __vector unsigned short zeros = vec_splat_u16(0);

  // LE: zeros in low 16 bits, raw in high 16 bits → bf16 << 16 == float32
  b0 = (__vector float)vec_mergeh(zeros, raw);
  b1 = (__vector float)vec_mergel(zeros, raw);
}

// [3] Half (FP16) Specialization
template <>
FORCE_INLINE void load_row8_B_as_f32<c10::Half>(const c10::Half* p,
                                                __vector float& b0,
                                                __vector float& b1) {
  vec_op::FP16Vec8 fp16_vec(p);
  vec_op::FP32Vec8 fp32_vec(fp16_vec);
  b0 = fp32_vec.reg.val[0];
  b1 = fp32_vec.reg.val[1];
}

template <typename a_t>
FORCE_INLINE void load_row4_A_as_f32(const a_t* p, __vector float& a) {
  if constexpr (std::is_same_v<a_t, float>) {
    a = vec_xl(0, const_cast<float*>(p));
  } else if constexpr (std::is_same_v<a_t, c10::BFloat16>) {
    uint64_t raw64;
    std::memcpy(&raw64, p, sizeof(raw64));
    __vector unsigned long long v64 = {raw64, 0};
    __vector unsigned short raw = (__vector unsigned short)v64;
    __vector unsigned short zeros = vec_splat_u16(0);
    a = (__vector float)vec_mergeh(zeros, raw);
  } else {
    static_assert(sizeof(a_t) == 0,
                  "Unsupported a_t type for load_row4_A_as_f32");
  }
}

template <int32_t M, typename a_t, typename kv_cache_t>
FORCE_INLINE void gemm_micro_ppc64le_Mx8_Ku4(
    const a_t* __restrict A,         // [M x K]
    const kv_cache_t* __restrict B,  // [K x 8]
    float* __restrict C,             // [M x 8]
    int64_t lda, int64_t ldb, int64_t ldc, int32_t K, bool accumulate) {
  static_assert(1 <= M && M <= 8, "M must be in [1,8]");

#define ROWS_APPLY(OP) OP(0) OP(1) OP(2) OP(3) OP(4) OP(5) OP(6) OP(7)
#define IF_M(i) if constexpr (M > (i))

  // 1. Define A pointers
#define DECL_A(i) const a_t* a##i = A + (i) * lda;
  ROWS_APPLY(DECL_A)
#undef DECL_A

  // 2. Define Accumulators (2 vectors covers 8 columns)
#define DECL_ACC(i) __vector float acc##i##_0, acc##i##_1;
  ROWS_APPLY(DECL_ACC)
#undef DECL_ACC

  // 3. Initialize Accumulators (Load C or Zero)
#define INIT_ACC(i)                                                  \
  IF_M(i) {                                                          \
    if (accumulate) {                                                \
      acc##i##_0 = vec_xl(0, const_cast<float*>(C + (i) * ldc + 0)); \
      acc##i##_1 = vec_xl(0, const_cast<float*>(C + (i) * ldc + 4)); \
    } else {                                                         \
      acc##i##_0 = vec_splats(0.0f);                                 \
      acc##i##_1 = vec_splats(0.0f);                                 \
    }                                                                \
  }
  ROWS_APPLY(INIT_ACC)
#undef INIT_ACC

  int32_t k = 0;

  for (; k + 3 < K; k += 4) {
    // Load 4 values of A for each Row M: A[k...k+3]
#define LOAD_A4(i)        \
  __vector float a##i##v; \
  IF_M(i) load_row4_A_as_f32<a_t>(a##i + k, a##i##v);
    ROWS_APPLY(LOAD_A4)
#undef LOAD_A4

    // FMA for specific lane L of A
    // ppc64le: vec_madd(b, vec_splat(a, lane), acc)
#define FMAS_LANE(i, aiv, L)                        \
  IF_M(i) {                                         \
    __vector float a_broad = vec_splat(aiv, L);     \
    acc##i##_0 = vec_madd(b0, a_broad, acc##i##_0); \
    acc##i##_1 = vec_madd(b1, a_broad, acc##i##_1); \
  }

    // Unroll K=0..3
    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(B + (int64_t)(k + 0) * ldb, b0, b1);
#define STEP_K0(i) FMAS_LANE(i, a##i##v, 0)
      ROWS_APPLY(STEP_K0)
#undef STEP_K0
    }
    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(B + (int64_t)(k + 1) * ldb, b0, b1);
#define STEP_K1(i) FMAS_LANE(i, a##i##v, 1)
      ROWS_APPLY(STEP_K1)
#undef STEP_K1
    }
    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(B + (int64_t)(k + 2) * ldb, b0, b1);
#define STEP_K2(i) FMAS_LANE(i, a##i##v, 2)
      ROWS_APPLY(STEP_K2)
#undef STEP_K2
    }
    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(B + (int64_t)(k + 3) * ldb, b0, b1);
#define STEP_K3(i) FMAS_LANE(i, a##i##v, 3)
      ROWS_APPLY(STEP_K3)
#undef STEP_K3
    }
#undef FMAS_LANE
  }

  for (; k < K; ++k) {
    __vector float b0, b1;
    load_row8_B_as_f32<kv_cache_t>(B + (int64_t)k * ldb, b0, b1);
#define TAIL_ROW(i)                                                     \
  IF_M(i) {                                                             \
    __vector float ai = vec_splats(static_cast<float>(*(a##i + k)));   \
    acc##i##_0 = vec_madd(b0, ai, acc##i##_0);                          \
    acc##i##_1 = vec_madd(b1, ai, acc##i##_1);                          \
  }
    ROWS_APPLY(TAIL_ROW)
#undef TAIL_ROW
  }

#define STORE_ROW(i)                           \
  IF_M(i) {                                    \
    vec_xst(acc##i##_0, 0, C + (i) * ldc + 0); \
    vec_xst(acc##i##_1, 0, C + (i) * ldc + 4); \
  }
  ROWS_APPLY(STORE_ROW)
#undef STORE_ROW

#undef ROWS_APPLY
#undef IF_M
}

template <int32_t N, typename a_t, typename kv_cache_t>
FORCE_INLINE void gemm_macro_ppc64le_Mx8_Ku4(const a_t* __restrict A,
                                             const kv_cache_t* __restrict B,
                                             float* __restrict C, int32_t M,
                                             int32_t K, int64_t lda,
                                             int64_t ldb, int64_t ldc,
                                             bool accumulate) {
  static_assert(N % 8 == 0, "N must be a multiple of 8");
  for (int32_t m = 0; m < M;) {
    int32_t mb = (M - m >= 8) ? 8 : (M - m >= 4) ? 4 : (M - m >= 2) ? 2 : 1;
    const a_t* Ab = A + m * lda;
    float* Cb = C + m * ldc;

    for (int32_t n = 0; n < N; n += 8) {
      const kv_cache_t* Bn = B + n;
      float* Cn = Cb + n;
      switch (mb) {
        case 8:
          gemm_micro_ppc64le_Mx8_Ku4<8, a_t, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                         K, accumulate);
          break;
        case 4:
          gemm_micro_ppc64le_Mx8_Ku4<4, a_t, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                         K, accumulate);
          break;
        case 2:
          gemm_micro_ppc64le_Mx8_Ku4<2, a_t, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                         K, accumulate);
          break;
        default:
          gemm_micro_ppc64le_Mx8_Ku4<1, a_t, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                         K, accumulate);
          break;
      }
    }
    m += mb;
  }
}

template <int32_t M>
__attribute__((target("cpu=power10"))) FORCE_INLINE void
gemm_micro_mma_Mx32_Ku2(const c10::BFloat16* __restrict A,
                        const c10::BFloat16* __restrict B,
                        float* __restrict C, int64_t lda, int64_t ldb,
                        int64_t ldc, int32_t K, bool accumulate) {
  static_assert(1 <= M && M <= 4, "M must be in [1, 4]");

  if (K <= 0) {
    if (!accumulate) {
      for (int r = 0; r < M; ++r) {
        std::memset(C + r * ldc, 0, sizeof(float) * 32);
      }
    }
    return;
  }

  __vector_quad acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
  __builtin_mma_xxsetaccz(&acc0);
  __builtin_mma_xxsetaccz(&acc1);
  __builtin_mma_xxsetaccz(&acc2);
  __builtin_mma_xxsetaccz(&acc3);
  __builtin_mma_xxsetaccz(&acc4);
  __builtin_mma_xxsetaccz(&acc5);
  __builtin_mma_xxsetaccz(&acc6);
  __builtin_mma_xxsetaccz(&acc7);

  int32_t k = 0;
  for (; k + 1 < K; k += 2) {
    __vector unsigned int va;
    if constexpr (M == 1) {
      uint32_t a0;
      std::memcpy(&a0, A + k, 4);
      va = (__vector unsigned int){a0, 0, 0, 0};
    } else if constexpr (M == 2) {
      uint32_t a0, a1;
      std::memcpy(&a0, A + 0 * lda + k, 4);
      std::memcpy(&a1, A + 1 * lda + k, 4);
      va = (__vector unsigned int){a0, a1, 0, 0};
    } else if constexpr (M == 3) {
      uint32_t a0, a1, a2;
      std::memcpy(&a0, A + 0 * lda + k, 4);
      std::memcpy(&a1, A + 1 * lda + k, 4);
      std::memcpy(&a2, A + 2 * lda + k, 4);
      va = (__vector unsigned int){a0, a1, a2, 0};
    } else if constexpr (M == 4) {
      uint32_t a0, a1, a2, a3;
      std::memcpy(&a0, A + 0 * lda + k, 4);
      std::memcpy(&a1, A + 1 * lda + k, 4);
      std::memcpy(&a2, A + 2 * lda + k, 4);
      std::memcpy(&a3, A + 3 * lda + k, 4);
      va = (__vector unsigned int){a0, a1, a2, a3};
    }
    const __vector unsigned char avec = (__vector unsigned char)va;

    const auto* b0 =
        reinterpret_cast<const unsigned short*>(B + (int64_t)k * ldb);
    const auto* b1 =
        reinterpret_cast<const unsigned short*>(B + (int64_t)(k + 1) * ldb);

    const __vector unsigned short b0_0 = vec_xl(0, b0 + 0);
    const __vector unsigned short b1_0 = vec_xl(0, b1 + 0);
    __builtin_mma_xvbf16ger2pp(
        &acc0, avec, (__vector unsigned char)vec_mergeh(b0_0, b1_0));
    __builtin_mma_xvbf16ger2pp(
        &acc1, avec, (__vector unsigned char)vec_mergel(b0_0, b1_0));

    const __vector unsigned short b0_1 = vec_xl(0, b0 + 8);
    const __vector unsigned short b1_1 = vec_xl(0, b1 + 8);
    __builtin_mma_xvbf16ger2pp(
        &acc2, avec, (__vector unsigned char)vec_mergeh(b0_1, b1_1));
    __builtin_mma_xvbf16ger2pp(
        &acc3, avec, (__vector unsigned char)vec_mergel(b0_1, b1_1));

    const __vector unsigned short b0_2 = vec_xl(0, b0 + 16);
    const __vector unsigned short b1_2 = vec_xl(0, b1 + 16);
    __builtin_mma_xvbf16ger2pp(
        &acc4, avec, (__vector unsigned char)vec_mergeh(b0_2, b1_2));
    __builtin_mma_xvbf16ger2pp(
        &acc5, avec, (__vector unsigned char)vec_mergel(b0_2, b1_2));

    const __vector unsigned short b0_3 = vec_xl(0, b0 + 24);
    const __vector unsigned short b1_3 = vec_xl(0, b1 + 24);
    __builtin_mma_xvbf16ger2pp(
        &acc6, avec, (__vector unsigned char)vec_mergeh(b0_3, b1_3));
    __builtin_mma_xvbf16ger2pp(
        &acc7, avec, (__vector unsigned char)vec_mergel(b0_3, b1_3));
  }

  if (k < K) {
    __vector unsigned int va;
    if constexpr (M == 1) {
      uint16_t a0;
      std::memcpy(&a0, A + k, 2);
      va = (__vector unsigned int){static_cast<uint32_t>(a0), 0, 0, 0};
    } else if constexpr (M == 2) {
      uint16_t a0, a1;
      std::memcpy(&a0, A + 0 * lda + k, 2);
      std::memcpy(&a1, A + 1 * lda + k, 2);
      va = (__vector unsigned int){static_cast<uint32_t>(a0),
                                   static_cast<uint32_t>(a1), 0, 0};
    } else if constexpr (M == 3) {
      uint16_t a0, a1, a2;
      std::memcpy(&a0, A + 0 * lda + k, 2);
      std::memcpy(&a1, A + 1 * lda + k, 2);
      std::memcpy(&a2, A + 2 * lda + k, 2);
      va = (__vector unsigned int){static_cast<uint32_t>(a0),
                                   static_cast<uint32_t>(a1),
                                   static_cast<uint32_t>(a2), 0};
    } else if constexpr (M == 4) {
      uint16_t a0, a1, a2, a3;
      std::memcpy(&a0, A + 0 * lda + k, 2);
      std::memcpy(&a1, A + 1 * lda + k, 2);
      std::memcpy(&a2, A + 2 * lda + k, 2);
      std::memcpy(&a3, A + 3 * lda + k, 2);
      va = (__vector unsigned int){static_cast<uint32_t>(a0),
                                   static_cast<uint32_t>(a1),
                                   static_cast<uint32_t>(a2),
                                   static_cast<uint32_t>(a3)};
    }
    const __vector unsigned char avec = (__vector unsigned char)va;

    const auto* b0 =
        reinterpret_cast<const unsigned short*>(B + (int64_t)k * ldb);
    const __vector unsigned short b1_zero = vec_splat_u16(0);

    const __vector unsigned short b0_0 = vec_xl(0, b0 + 0);
    __builtin_mma_xvbf16ger2pp(
        &acc0, avec, (__vector unsigned char)vec_mergeh(b0_0, b1_zero));
    __builtin_mma_xvbf16ger2pp(
        &acc1, avec, (__vector unsigned char)vec_mergel(b0_0, b1_zero));

    const __vector unsigned short b0_1 = vec_xl(0, b0 + 8);
    __builtin_mma_xvbf16ger2pp(
        &acc2, avec, (__vector unsigned char)vec_mergeh(b0_1, b1_zero));
    __builtin_mma_xvbf16ger2pp(
        &acc3, avec, (__vector unsigned char)vec_mergel(b0_1, b1_zero));

    const __vector unsigned short b0_2 = vec_xl(0, b0 + 16);
    __builtin_mma_xvbf16ger2pp(
        &acc4, avec, (__vector unsigned char)vec_mergeh(b0_2, b1_zero));
    __builtin_mma_xvbf16ger2pp(
        &acc5, avec, (__vector unsigned char)vec_mergel(b0_2, b1_zero));

    const __vector unsigned short b0_3 = vec_xl(0, b0 + 24);
    __builtin_mma_xvbf16ger2pp(
        &acc6, avec, (__vector unsigned char)vec_mergeh(b0_3, b1_zero));
    __builtin_mma_xvbf16ger2pp(
        &acc7, avec, (__vector unsigned char)vec_mergel(b0_3, b1_zero));
  }

  typedef float v4sf __attribute__((vector_size(16)));
  auto store_acc = [&](__vector_quad& acc, int32_t col_start) {
    v4sf tmp[4];
    __builtin_mma_disassemble_acc(tmp, &acc);
    for (int r = 0; r < M; ++r) {
      float* dst = C + r * ldc + col_start;
      __vector float val = (__vector float)tmp[r];
      if (accumulate) {
        val = vec_add(val, vec_xl(0, dst));
      }
      vec_xst(val, 0, dst);
    }
  };

  store_acc(acc0, 0);
  store_acc(acc1, 4);
  store_acc(acc2, 8);
  store_acc(acc3, 12);
  store_acc(acc4, 16);
  store_acc(acc5, 20);
  store_acc(acc6, 24);
  store_acc(acc7, 28);
}

template <int32_t N>
FORCE_INLINE void gemm_macro_mma_Nx32(
    const c10::BFloat16* __restrict A, const c10::BFloat16* __restrict B,
    float* __restrict C, int32_t M, int32_t K, int64_t lda, int64_t ldb,
    int64_t ldc, bool accumulate) {
  static_assert(N % 32 == 0, "N must be a multiple of 32");
  for (int32_t m = 0; m < M;) {
    int32_t mb = (M - m >= 4) ? 4 : (M - m >= 2) ? 2 : 1;
    const c10::BFloat16* Ab = A + m * lda;
    float* Cb = C + m * ldc;

    for (int32_t n = 0; n < N; n += 32) {
      const c10::BFloat16* Bn = B + n;
      float* Cn = Cb + n;
      switch (mb) {
        case 4:
          gemm_micro_mma_Mx32_Ku2<4>(Ab, Bn, Cn, lda, ldb, ldc, K, accumulate);
          break;
        case 2:
          gemm_micro_mma_Mx32_Ku2<2>(Ab, Bn, Cn, lda, ldb, ldc, K, accumulate);
          break;
        default:
          gemm_micro_mma_Mx32_Ku2<1>(Ab, Bn, Cn, lda, ldb, ldc, K, accumulate);
          break;
      }
    }
    m += mb;
  }
}

template <typename kv_cache_t, int64_t head_dim = -1>
class TileGemmPPC64 {
 public:
  using prob_t = std::conditional_t<std::is_same_v<kv_cache_t, float>, float,
                                    c10::BFloat16>;

  template <AttentionGemmPhase phase, int32_t k_size>
  FORCE_INLINE static void gemm(const int32_t m_size,
                                void* __restrict__ a_tile,
                                kv_cache_t* __restrict__ b_tile,
                                float* __restrict__ c_tile, const int64_t lda,
                                const int64_t ldb, const int64_t ldc,
                                const int32_t block_size,
                                const int32_t dynamic_k_size,
                                const bool accum_c) {
    if constexpr (phase == AttentionGemmPhase::QK) {
      gemm_macro_ppc64le_Mx8_Ku4<BLOCK_SIZE_ALIGNMENT, float, kv_cache_t>(
          reinterpret_cast<const float*>(a_tile), b_tile, c_tile, m_size,
          k_size, lda, ldb, ldc, accum_c);
    } else {
      if constexpr (std::is_same_v<kv_cache_t, c10::BFloat16>) {
        if constexpr (head_dim >= 256 || head_dim == -1) {
          if ((head_dim >= 256 || ldc >= 256) && m_size >= 4) {
            gemm_macro_mma_Nx32<HEAD_SIZE_ALIGNMENT>(
                reinterpret_cast<const c10::BFloat16*>(a_tile),
                reinterpret_cast<const c10::BFloat16*>(b_tile), c_tile, m_size,
                dynamic_k_size, lda, ldb, ldc, accum_c);
            return;
          }
        }
        gemm_macro_ppc64le_Mx8_Ku4<HEAD_SIZE_ALIGNMENT, prob_t, kv_cache_t>(
            reinterpret_cast<const prob_t*>(a_tile), b_tile, c_tile, m_size,
            dynamic_k_size, lda, ldb, ldc, accum_c);
      } else {
        gemm_macro_ppc64le_Mx8_Ku4<HEAD_SIZE_ALIGNMENT, prob_t, kv_cache_t>(
            reinterpret_cast<const prob_t*>(a_tile), b_tile, c_tile, m_size,
            dynamic_k_size, lda, ldb, ldc, accum_c);
      }
    }
  }
};

}  // namespace

template <typename scalar_t, int64_t head_dim>
class AttentionImpl<ISA::VSX, scalar_t, head_dim> {
 public:
  using query_t = scalar_t;
  using q_buffer_t = float;
  using kv_cache_t = scalar_t;
  using logits_buffer_t = float;
  using partial_output_buffer_t = float;
  using prob_buffer_t = std::conditional_t<std::is_same_v<scalar_t, float>,
                                           float, c10::BFloat16>;

  constexpr static int64_t BlockSizeAlignment = BLOCK_SIZE_ALIGNMENT;
  constexpr static int64_t HeadDimAlignment = HEAD_SIZE_ALIGNMENT;
  constexpr static int64_t MaxQHeadNumPerIteration = MAX_Q_HEAD_NUM_PER_ITER;
  constexpr static int64_t HeadDim = head_dim;
  constexpr static ISA ISAType = ISA::VSX;
  constexpr static bool scale_on_logits =
      false;  // Scale is applied to Q during copy

 public:
  AttentionImpl() {}

  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    attention<TileGemmPPC64<kv_cache_t, head_dim>> attention_iteration;
    attention_iteration(CPU_ATTENTION_PARAMS);
  }

  // Strides for Memory Layout
  constexpr static int64_t k_cache_token_group_stride(
      const int32_t block_size) {
    return BlockSizeAlignment;  // [head_dim, block_size] layout
  }

  constexpr static int64_t v_cache_token_group_stride(
      const int32_t block_size) {
    return head_dim * BlockSizeAlignment;
  }

  constexpr static int64_t v_cache_head_group_stride(const int32_t block_size) {
    return HeadDimAlignment;
  }

  static void copy_q_heads_tile(scalar_t* __restrict__ src,
                                float* __restrict__ q_buffer,
                                const int32_t q_num,
                                const int32_t q_heads_per_kv,
                                const int64_t q_num_stride,
                                const int64_t q_head_stride, float scale) {
    __vector float scale_vec = vec_splats(scale);
    constexpr bool is_bf16 = std::is_same<scalar_t, c10::BFloat16>::value;

    for (int32_t i = 0; i < q_num; ++i) {
      for (int32_t h = 0; h < q_heads_per_kv; ++h) {
        scalar_t* curr_src = src + i * q_num_stride + h * q_head_stride;
        float* curr_dst =
            q_buffer + i * q_heads_per_kv * head_dim + h * head_dim;

        int32_t d = 0;
        for (; d <= head_dim - 8; d += 8) {
          __vector float v0, v1;
          load_row8_B_as_f32<scalar_t>(curr_src + d, v0, v1);

          v0 = vec_mul(v0, scale_vec);
          v1 = vec_mul(v1, scale_vec);

          vec_xst(v0, 0, curr_dst + d);
          vec_xst(v1, 0, curr_dst + d + 4);
        }

        for (; d < head_dim; ++d) {
          float val = static_cast<float>(curr_src[d]);
          curr_dst[d] = val * scale;
        }
      }
    }
  }

  static void reshape_and_cache(
      const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
      scalar_t* __restrict__ key_cache, scalar_t* __restrict__ value_cache,
      const int64_t* __restrict__ slot_mapping, const int64_t token_num,
      const int64_t key_token_num_stride, const int64_t value_token_num_stride,
      const int64_t head_num, const int64_t key_head_num_stride,
      const int64_t value_head_num_stride, const int64_t num_blocks,
      const int64_t num_blocks_stride, const int64_t cache_head_num_stride,
      const int64_t block_size, const int64_t block_size_stride,
      const float k_inv = 0.0f, const float v_inv = 0.0f) {
#pragma omp parallel for collapse(2)
    for (int64_t token_idx = 0; token_idx < token_num; ++token_idx) {
      for (int64_t head_idx = 0; head_idx < head_num; ++head_idx) {
        const int64_t pos = slot_mapping[token_idx];
        if (pos < 0) continue;

        const int64_t block_idx = pos / block_size;
        const int64_t block_offset = pos % block_size;

        {
          const scalar_t* key_src = key + token_idx * key_token_num_stride +
                                    head_idx * key_head_num_stride;
          scalar_t* key_dst = key_cache + block_idx * num_blocks_stride +
                              head_idx * cache_head_num_stride + block_offset;

          for (int64_t i = 0, j = 0; i < head_dim; ++i, j += block_size) {
            key_dst[j] = key_src[i];
          }
        }

        {
          const scalar_t* val_src = value + token_idx * value_token_num_stride +
                                    head_idx * value_head_num_stride;
          scalar_t* val_dst = value_cache + block_idx * num_blocks_stride +
                              head_idx * cache_head_num_stride +
                              block_offset * head_dim;

          std::memcpy(val_dst, val_src, sizeof(scalar_t) * head_dim);
        }
      }
    }
  }
};

}  // namespace cpu_attention

#undef BLOCK_SIZE_ALIGNMENT
#undef HEAD_SIZE_ALIGNMENT
#undef MAX_Q_HEAD_NUM_PER_ITER

#endif  // CPU_ATTN_VSX_HPP
