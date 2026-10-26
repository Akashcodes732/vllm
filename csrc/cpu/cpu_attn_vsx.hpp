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
#define BLOCK_SIZE_ALIGNMENT 16
#define HEAD_SIZE_ALIGNMENT 32
#define MAX_Q_HEAD_NUM_PER_ITER 16

typedef uint64_t __attribute__((aligned(1))) unaligned_u64;
typedef uint32_t __attribute__((aligned(1))) unaligned_u32;
typedef uint16_t __attribute__((aligned(1))) unaligned_u16;

template <typename kv_cache_t>
FORCE_INLINE void load_row8_B_as_f32(const kv_cache_t* p, __vector float& b0,
                                     __vector float& b1);

// [1] Float Specialization for 8 elements of B
template <>
FORCE_INLINE void load_row8_B_as_f32<float>(const float* p, __vector float& b0,
                                            __vector float& b1) {
  b0 = vec_xl(0, const_cast<float*>(p));
  b1 = vec_xl(0, const_cast<float*>(p + 4));
}

// [2] BFloat16 Specialization for 8 elements of B (Little Endian ppc64le)
template <>
FORCE_INLINE void load_row8_B_as_f32<c10::BFloat16>(const c10::BFloat16* p,
                                                    __vector float& b0,
                                                    __vector float& b1) {
  __vector unsigned short raw = vec_xl(
      0, reinterpret_cast<unsigned short*>(const_cast<c10::BFloat16*>(p)));
  __vector unsigned short zeros = vec_splat_u16(0);

  // LE: zeros in low 16 bits, raw in high 16 bits -> bf16 << 16 == float32
  b0 = (__vector float)vec_mergeh(zeros, raw);
  b1 = (__vector float)vec_mergel(zeros, raw);
}

// [3] Half (FP16) Specialization for 8 elements of B
template <>
FORCE_INLINE void load_row8_B_as_f32<c10::Half>(const c10::Half* p,
                                                __vector float& b0,
                                                __vector float& b1) {
  vec_op::FP16Vec8 fp16_vec(p);
  vec_op::FP32Vec8 fp32_vec(fp16_vec);
  b0 = fp32_vec.reg.val[0];
  b1 = fp32_vec.reg.val[1];
}

template <typename kv_cache_t>
FORCE_INLINE void load_row16_B_as_f32(const kv_cache_t* p,
                                      __vector float& b0,
                                      __vector float& b1,
                                      __vector float& b2,
                                      __vector float& b3);

// [1] Float Specialization for 16 elements of B
template <>
FORCE_INLINE void load_row16_B_as_f32<float>(const float* p,
                                             __vector float& b0,
                                             __vector float& b1,
                                             __vector float& b2,
                                             __vector float& b3) {
  b0 = vec_xl(0, const_cast<float*>(p + 0));
  b1 = vec_xl(0, const_cast<float*>(p + 4));
  b2 = vec_xl(0, const_cast<float*>(p + 8));
  b3 = vec_xl(0, const_cast<float*>(p + 12));
}

// [2] BFloat16 Specialization for 16 elements of B
template <>
FORCE_INLINE void load_row16_B_as_f32<c10::BFloat16>(const c10::BFloat16* p,
                                                     __vector float& b0,
                                                     __vector float& b1,
                                                     __vector float& b2,
                                                     __vector float& b3) {
  const auto* u = reinterpret_cast<const unsigned short*>(p);
  __vector unsigned short raw0 = vec_xl(0, const_cast<unsigned short*>(u + 0));
  __vector unsigned short raw1 = vec_xl(0, const_cast<unsigned short*>(u + 8));
  __vector unsigned short zeros = vec_splat_u16(0);

  b0 = (__vector float)vec_mergeh(zeros, raw0);
  b1 = (__vector float)vec_mergel(zeros, raw0);
  b2 = (__vector float)vec_mergeh(zeros, raw1);
  b3 = (__vector float)vec_mergel(zeros, raw1);
}

// [3] Half (FP16) Specialization for 16 elements of B
template <>
FORCE_INLINE void load_row16_B_as_f32<c10::Half>(const c10::Half* p,
                                                 __vector float& b0,
                                                 __vector float& b1,
                                                 __vector float& b2,
                                                 __vector float& b3) {
  load_row8_B_as_f32<c10::Half>(p + 0, b0, b1);
  load_row8_B_as_f32<c10::Half>(p + 8, b2, b3);
}

// Load 8 values of A (FP32)
FORCE_INLINE void load_row8_A_f32(const float* p, __vector float& a0,
                                  __vector float& a1) {
  a0 = vec_xl(0, const_cast<float*>(p + 0));
  a1 = vec_xl(0, const_cast<float*>(p + 4));
}

// Load 4 values of A (FP32)
FORCE_INLINE void load_row4_A_f32(const float* p, __vector float& a) {
  a = vec_xl(0, const_cast<float*>(p));
}

// High-performance 16-column VSX microkernel with K unroll = 8
template <int32_t M, typename kv_cache_t>
FORCE_INLINE void gemm_micro_ppc64le_Mx16_Ku8(
    const float* __restrict A,       // [M x K]
    const kv_cache_t* __restrict B,  // [K x 16]
    float* __restrict C,             // [M x 16]
    int64_t lda, int64_t ldb, int64_t ldc, int32_t K, bool accumulate) {
  static_assert(1 <= M && M <= 8, "M must be in [1,8]");

#define ROWS_APPLY(OP) OP(0) OP(1) OP(2) OP(3) OP(4) OP(5) OP(6) OP(7)
#define IF_M(i) if constexpr (M > (i))

  // 1. Define A pointers
#define DECL_A(i) const float* a##i = A + (i) * lda;
  ROWS_APPLY(DECL_A)
#undef DECL_A

  // 2. Define Accumulators (4 vectors cover 16 columns)
#define DECL_ACC(i) __vector float acc##i##_0, acc##i##_1, acc##i##_2, acc##i##_3;
  ROWS_APPLY(DECL_ACC)
#undef DECL_ACC

  // 3. Initialize Accumulators (Load C or Zero)
#define INIT_ACC(i)                                                    \
  IF_M(i) {                                                            \
    if (accumulate) {                                                  \
      acc##i##_0 = vec_xl(0, const_cast<float*>(C + (i) * ldc + 0));   \
      acc##i##_1 = vec_xl(0, const_cast<float*>(C + (i) * ldc + 4));   \
      acc##i##_2 = vec_xl(0, const_cast<float*>(C + (i) * ldc + 8));   \
      acc##i##_3 = vec_xl(0, const_cast<float*>(C + (i) * ldc + 12));  \
    } else {                                                           \
      acc##i##_0 = vec_splats(0.0f);                                   \
      acc##i##_1 = vec_splats(0.0f);                                   \
      acc##i##_2 = vec_splats(0.0f);                                   \
      acc##i##_3 = vec_splats(0.0f);                                   \
    }                                                                  \
  }
  ROWS_APPLY(INIT_ACC)
#undef INIT_ACC

  int32_t k = 0;

  for (; k + 7 < K; k += 8) {
#define LOAD_A8(i)                                \
  __vector float a##i##_03, a##i##_47;            \
  IF_M(i) load_row8_A_f32(a##i + k, a##i##_03, a##i##_47);
    ROWS_APPLY(LOAD_A8)
#undef LOAD_A8

#define FMAS_LANE(i, aiv, L)                        \
  IF_M(i) {                                         \
    __vector float a_broad = vec_splat(aiv, L);     \
    acc##i##_0 = vec_madd(b0, a_broad, acc##i##_0); \
    acc##i##_1 = vec_madd(b1, a_broad, acc##i##_1); \
    acc##i##_2 = vec_madd(b2, a_broad, acc##i##_2); \
    acc##i##_3 = vec_madd(b3, a_broad, acc##i##_3); \
  }

    const kv_cache_t* b_k = B + (int64_t)k * ldb;
    {
      __vector float b0, b1, b2, b3;
      load_row16_B_as_f32<kv_cache_t>(b_k, b0, b1, b2, b3);
#define STEP0(i) FMAS_LANE(i, a##i##_03, 0)
      ROWS_APPLY(STEP0)
#undef STEP0
    }
    b_k += ldb;
    {
      __vector float b0, b1, b2, b3;
      load_row16_B_as_f32<kv_cache_t>(b_k, b0, b1, b2, b3);
#define STEP1(i) FMAS_LANE(i, a##i##_03, 1)
      ROWS_APPLY(STEP1)
#undef STEP1
    }
    b_k += ldb;
    {
      __vector float b0, b1, b2, b3;
      load_row16_B_as_f32<kv_cache_t>(b_k, b0, b1, b2, b3);
#define STEP2(i) FMAS_LANE(i, a##i##_03, 2)
      ROWS_APPLY(STEP2)
#undef STEP2
    }
    b_k += ldb;
    {
      __vector float b0, b1, b2, b3;
      load_row16_B_as_f32<kv_cache_t>(b_k, b0, b1, b2, b3);
#define STEP3(i) FMAS_LANE(i, a##i##_03, 3)
      ROWS_APPLY(STEP3)
#undef STEP3
    }
    b_k += ldb;
    {
      __vector float b0, b1, b2, b3;
      load_row16_B_as_f32<kv_cache_t>(b_k, b0, b1, b2, b3);
#define STEP4(i) FMAS_LANE(i, a##i##_47, 0)
      ROWS_APPLY(STEP4)
#undef STEP4
    }
    b_k += ldb;
    {
      __vector float b0, b1, b2, b3;
      load_row16_B_as_f32<kv_cache_t>(b_k, b0, b1, b2, b3);
#define STEP5(i) FMAS_LANE(i, a##i##_47, 1)
      ROWS_APPLY(STEP5)
#undef STEP5
    }
    b_k += ldb;
    {
      __vector float b0, b1, b2, b3;
      load_row16_B_as_f32<kv_cache_t>(b_k, b0, b1, b2, b3);
#define STEP6(i) FMAS_LANE(i, a##i##_47, 2)
      ROWS_APPLY(STEP6)
#undef STEP6
    }
    b_k += ldb;
    {
      __vector float b0, b1, b2, b3;
      load_row16_B_as_f32<kv_cache_t>(b_k, b0, b1, b2, b3);
#define STEP7(i) FMAS_LANE(i, a##i##_47, 3)
      ROWS_APPLY(STEP7)
#undef STEP7
    }
#undef FMAS_LANE
  }

  // Remainder Ku4
  if (k + 3 < K) {
#define LOAD_A4(i)                           \
  __vector float a##i##_03;                  \
  IF_M(i) load_row4_A_f32(a##i + k, a##i##_03);
    ROWS_APPLY(LOAD_A4)
#undef LOAD_A4

#define FMAS_LANE(i, aiv, L)                        \
  IF_M(i) {                                         \
    __vector float a_broad = vec_splat(aiv, L);     \
    acc##i##_0 = vec_madd(b0, a_broad, acc##i##_0); \
    acc##i##_1 = vec_madd(b1, a_broad, acc##i##_1); \
    acc##i##_2 = vec_madd(b2, a_broad, acc##i##_2); \
    acc##i##_3 = vec_madd(b3, a_broad, acc##i##_3); \
  }
    const kv_cache_t* b_k = B + (int64_t)k * ldb;
    {
      __vector float b0, b1, b2, b3;
      load_row16_B_as_f32<kv_cache_t>(b_k, b0, b1, b2, b3);
#define STEP0(i) FMAS_LANE(i, a##i##_03, 0)
      ROWS_APPLY(STEP0)
#undef STEP0
    }
    b_k += ldb;
    {
      __vector float b0, b1, b2, b3;
      load_row16_B_as_f32<kv_cache_t>(b_k, b0, b1, b2, b3);
#define STEP1(i) FMAS_LANE(i, a##i##_03, 1)
      ROWS_APPLY(STEP1)
#undef STEP1
    }
    b_k += ldb;
    {
      __vector float b0, b1, b2, b3;
      load_row16_B_as_f32<kv_cache_t>(b_k, b0, b1, b2, b3);
#define STEP2(i) FMAS_LANE(i, a##i##_03, 2)
      ROWS_APPLY(STEP2)
#undef STEP2
    }
    b_k += ldb;
    {
      __vector float b0, b1, b2, b3;
      load_row16_B_as_f32<kv_cache_t>(b_k, b0, b1, b2, b3);
#define STEP3(i) FMAS_LANE(i, a##i##_03, 3)
      ROWS_APPLY(STEP3)
#undef STEP3
    }
#undef FMAS_LANE
    k += 4;
  }

  // Scalar tail for remaining 1..3 elements
  for (; k < K; ++k) {
    __vector float b0, b1, b2, b3;
    load_row16_B_as_f32<kv_cache_t>(B + (int64_t)k * ldb, b0, b1, b2, b3);
#define TAIL_ROW(i)                                                     \
  IF_M(i) {                                                             \
    __vector float ai = vec_splats(*(a##i + k));                        \
    acc##i##_0 = vec_madd(b0, ai, acc##i##_0);                          \
    acc##i##_1 = vec_madd(b1, ai, acc##i##_1);                          \
    acc##i##_2 = vec_madd(b2, ai, acc##i##_2);                          \
    acc##i##_3 = vec_madd(b3, ai, acc##i##_3);                          \
  }
    ROWS_APPLY(TAIL_ROW)
#undef TAIL_ROW
  }

#define STORE_ROW(i)                             \
  IF_M(i) {                                      \
    vec_xst(acc##i##_0, 0, C + (i) * ldc + 0);   \
    vec_xst(acc##i##_1, 0, C + (i) * ldc + 4);   \
    vec_xst(acc##i##_2, 0, C + (i) * ldc + 8);   \
    vec_xst(acc##i##_3, 0, C + (i) * ldc + 12);  \
  }
  ROWS_APPLY(STORE_ROW)
#undef STORE_ROW

#undef ROWS_APPLY
#undef IF_M
}

// 8-column VSX microkernel with K unroll = 8
template <int32_t M, typename kv_cache_t>
FORCE_INLINE void gemm_micro_ppc64le_Mx8_Ku8(
    const float* __restrict A,       // [M x K]
    const kv_cache_t* __restrict B,  // [K x 8]
    float* __restrict C,             // [M x 8]
    int64_t lda, int64_t ldb, int64_t ldc, int32_t K, bool accumulate) {
  static_assert(1 <= M && M <= 8, "M must be in [1,8]");

#define ROWS_APPLY(OP) OP(0) OP(1) OP(2) OP(3) OP(4) OP(5) OP(6) OP(7)
#define IF_M(i) if constexpr (M > (i))

  // 1. Define A pointers
#define DECL_A(i) const float* a##i = A + (i) * lda;
  ROWS_APPLY(DECL_A)
#undef DECL_A

  // 2. Define Accumulators (2 vectors cover 8 columns)
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

  for (; k + 7 < K; k += 8) {
#define LOAD_A8(i)                                \
  __vector float a##i##_03, a##i##_47;            \
  IF_M(i) load_row8_A_f32(a##i + k, a##i##_03, a##i##_47);
    ROWS_APPLY(LOAD_A8)
#undef LOAD_A8

#define FMAS_LANE(i, aiv, L)                        \
  IF_M(i) {                                         \
    __vector float a_broad = vec_splat(aiv, L);     \
    acc##i##_0 = vec_madd(b0, a_broad, acc##i##_0); \
    acc##i##_1 = vec_madd(b1, a_broad, acc##i##_1); \
  }

    const kv_cache_t* b_k = B + (int64_t)k * ldb;
    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(b_k, b0, b1);
#define STEP0(i) FMAS_LANE(i, a##i##_03, 0)
      ROWS_APPLY(STEP0)
#undef STEP0
    }
    b_k += ldb;
    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(b_k, b0, b1);
#define STEP1(i) FMAS_LANE(i, a##i##_03, 1)
      ROWS_APPLY(STEP1)
#undef STEP1
    }
    b_k += ldb;
    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(b_k, b0, b1);
#define STEP2(i) FMAS_LANE(i, a##i##_03, 2)
      ROWS_APPLY(STEP2)
#undef STEP2
    }
    b_k += ldb;
    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(b_k, b0, b1);
#define STEP3(i) FMAS_LANE(i, a##i##_03, 3)
      ROWS_APPLY(STEP3)
#undef STEP3
    }
    b_k += ldb;
    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(b_k, b0, b1);
#define STEP4(i) FMAS_LANE(i, a##i##_47, 0)
      ROWS_APPLY(STEP4)
#undef STEP4
    }
    b_k += ldb;
    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(b_k, b0, b1);
#define STEP5(i) FMAS_LANE(i, a##i##_47, 1)
      ROWS_APPLY(STEP5)
#undef STEP5
    }
    b_k += ldb;
    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(b_k, b0, b1);
#define STEP6(i) FMAS_LANE(i, a##i##_47, 2)
      ROWS_APPLY(STEP6)
#undef STEP6
    }
    b_k += ldb;
    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(b_k, b0, b1);
#define STEP7(i) FMAS_LANE(i, a##i##_47, 3)
      ROWS_APPLY(STEP7)
#undef STEP7
    }
#undef FMAS_LANE
  }

  if (k + 3 < K) {
#define LOAD_A4(i)                           \
  __vector float a##i##_03;                  \
  IF_M(i) load_row4_A_f32(a##i + k, a##i##_03);
    ROWS_APPLY(LOAD_A4)
#undef LOAD_A4

#define FMAS_LANE(i, aiv, L)                        \
  IF_M(i) {                                         \
    __vector float a_broad = vec_splat(aiv, L);     \
    acc##i##_0 = vec_madd(b0, a_broad, acc##i##_0); \
    acc##i##_1 = vec_madd(b1, a_broad, acc##i##_1); \
  }
    const kv_cache_t* b_k = B + (int64_t)k * ldb;
    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(b_k, b0, b1);
#define STEP0(i) FMAS_LANE(i, a##i##_03, 0)
      ROWS_APPLY(STEP0)
#undef STEP0
    }
    b_k += ldb;
    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(b_k, b0, b1);
#define STEP1(i) FMAS_LANE(i, a##i##_03, 1)
      ROWS_APPLY(STEP1)
#undef STEP1
    }
    b_k += ldb;
    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(b_k, b0, b1);
#define STEP2(i) FMAS_LANE(i, a##i##_03, 2)
      ROWS_APPLY(STEP2)
#undef STEP2
    }
    b_k += ldb;
    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(b_k, b0, b1);
#define STEP3(i) FMAS_LANE(i, a##i##_03, 3)
      ROWS_APPLY(STEP3)
#undef STEP3
    }
#undef FMAS_LANE
    k += 4;
  }

  for (; k < K; ++k) {
    __vector float b0, b1;
    load_row8_B_as_f32<kv_cache_t>(B + (int64_t)k * ldb, b0, b1);
#define TAIL_ROW(i)                                                     \
  IF_M(i) {                                                             \
    __vector float ai = vec_splats(*(a##i + k));                        \
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

// Backward compatibility wrapper
template <int32_t M, typename kv_cache_t>
FORCE_INLINE void gemm_micro_ppc64le_Mx8_Ku4(
    const float* __restrict A, const kv_cache_t* __restrict B,
    float* __restrict C, int64_t lda, int64_t ldb, int64_t ldc, int32_t K,
    bool accumulate) {
  gemm_micro_ppc64le_Mx8_Ku8<M, kv_cache_t>(A, B, C, lda, ldb, ldc, K,
                                            accumulate);
}

// Macro GEMM using 16-column tiles with 8-column remainder
template <int32_t N, typename kv_cache_t>
FORCE_INLINE void gemm_macro_ppc64le(const float* __restrict A,
                                     const kv_cache_t* __restrict B,
                                     float* __restrict C, int32_t M,
                                     int32_t K, int64_t lda,
                                     int64_t ldb, int64_t ldc,
                                     bool accumulate) {
  static_assert(N % 8 == 0, "N must be a multiple of 8");
  for (int32_t m = 0; m < M;) {
    int32_t mb = (M - m >= 8) ? 8 : (M - m >= 6) ? 6 : (M - m >= 4) ? 4 : (M - m >= 2) ? 2 : 1;
    const float* Ab = A + m * lda;
    float* Cb = C + m * ldc;

    int32_t n = 0;
    for (; n + 15 < N; n += 16) {
      const kv_cache_t* Bn = B + n;
      float* Cn = Cb + n;
      switch (mb) {
        case 8:
          gemm_micro_ppc64le_Mx16_Ku8<8, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                           K, accumulate);
          break;
        case 6:
          gemm_micro_ppc64le_Mx16_Ku8<6, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                           K, accumulate);
          break;
        case 4:
          gemm_micro_ppc64le_Mx16_Ku8<4, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                           K, accumulate);
          break;
        case 2:
          gemm_micro_ppc64le_Mx16_Ku8<2, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                           K, accumulate);
          break;
        default:
          gemm_micro_ppc64le_Mx16_Ku8<1, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                           K, accumulate);
          break;
      }
    }

    for (; n + 7 < N; n += 8) {
      const kv_cache_t* Bn = B + n;
      float* Cn = Cb + n;
      switch (mb) {
        case 8:
          gemm_micro_ppc64le_Mx8_Ku8<8, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                          K, accumulate);
          break;
        case 6:
          gemm_micro_ppc64le_Mx8_Ku8<6, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                          K, accumulate);
          break;
        case 4:
          gemm_micro_ppc64le_Mx8_Ku8<4, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                          K, accumulate);
          break;
        case 2:
          gemm_micro_ppc64le_Mx8_Ku8<2, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                          K, accumulate);
          break;
        default:
          gemm_micro_ppc64le_Mx8_Ku8<1, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                          K, accumulate);
          break;
      }
    }
    m += mb;
  }
}

// Backward compatibility alias
template <int32_t N, typename kv_cache_t>
FORCE_INLINE void gemm_macro_ppc64le_Mx8_Ku4(const float* __restrict A,
                                             const kv_cache_t* __restrict B,
                                             float* __restrict C, int32_t M,
                                             int32_t K, int64_t lda,
                                             int64_t ldb, int64_t ldc,
                                             bool accumulate) {
  gemm_macro_ppc64le<N, kv_cache_t>(A, B, C, M, K, lda, ldb, ldc, accumulate);
}

template <typename kv_cache_t>
class TileGemmPPC64 {
 public:
  template <AttentionGemmPhase phase, int32_t k_size>
  FORCE_INLINE static void gemm(const int32_t m_size,
                                float* __restrict__ a_tile,
                                kv_cache_t* __restrict__ b_tile,
                                float* __restrict__ c_tile, const int64_t lda,
                                const int64_t ldb, const int64_t ldc,
                                const int32_t block_size,
                                const int32_t dynamic_k_size,
                                const bool accum_c) {
    if constexpr (phase == AttentionGemmPhase::QK) {
      gemm_macro_ppc64le<BLOCK_SIZE_ALIGNMENT, kv_cache_t>(
          a_tile, b_tile, c_tile, m_size, k_size, lda, ldb, ldc, accum_c);
    } else {
      gemm_macro_ppc64le<HEAD_SIZE_ALIGNMENT, kv_cache_t>(
          a_tile, b_tile, c_tile, m_size, dynamic_k_size, lda, ldb, ldc,
          accum_c);
    }
  }
};

}  // namespace

template <typename scalar_t, int64_t head_dim, typename kv_cache_scalar_t = scalar_t>
class AttentionImpl<ISA::VSX, scalar_t, head_dim, kv_cache_scalar_t> {
 public:
  using query_t = scalar_t;
  using q_buffer_t = float;
  using kv_cache_t = scalar_t;
  using logits_buffer_t = float;
  using partial_output_buffer_t = float;
  using prob_buffer_t = float;

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
    attention<TileGemmPPC64<kv_cache_t>> attention_iteration;
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

    for (int32_t i = 0; i < q_num; ++i) {
      for (int32_t h = 0; h < q_heads_per_kv; ++h) {
        scalar_t* curr_src = src + i * q_num_stride + h * q_head_stride;
        float* curr_dst =
            q_buffer + i * q_heads_per_kv * head_dim + h * head_dim;

        int32_t d = 0;
        for (; d <= head_dim - 16; d += 16) {
          __vector float v0, v1, v2, v3;
          load_row16_B_as_f32<scalar_t>(curr_src + d, v0, v1, v2, v3);

          v0 = vec_mul(v0, scale_vec);
          v1 = vec_mul(v1, scale_vec);
          v2 = vec_mul(v2, scale_vec);
          v3 = vec_mul(v3, scale_vec);

          vec_xst(v0, 0, curr_dst + d + 0);
          vec_xst(v1, 0, curr_dst + d + 4);
          vec_xst(v2, 0, curr_dst + d + 8);
          vec_xst(v3, 0, curr_dst + d + 12);
        }

        for (; d <= head_dim - 8; d += 8) {
          __vector float v0, v1;
          load_row8_B_as_f32<scalar_t>(curr_src + d, v0, v1);

          v0 = vec_mul(v0, scale_vec);
          v1 = vec_mul(v1, scale_vec);

          vec_xst(v0, 0, curr_dst + d + 0);
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
