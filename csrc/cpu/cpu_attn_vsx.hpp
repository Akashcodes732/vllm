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

  // LE: zeros in low 16 bits, raw in high 16 bits -> bf16 << 16 == float32
  b0 = (__vector float)vec_mergeh(zeros, raw);
  b1 = (__vector float)vec_mergel(zeros, raw);
}

// Note: c10::Half (FP16) is not supported on PowerPC architecture

// Prefetch distance for B rows in the GEMM hot loops (VSX and MMA).
// Tunable at compile time: -DVSX_ATTN_PREFETCH_DIST=N (0 disables prefetch).
#ifndef VSX_ATTN_PREFETCH_DIST
#  define VSX_ATTN_PREFETCH_DIST 16
#endif

// ---------------------------------------------------------------------------
// VSX micro-kernel: gemm_micro_ppc64le_Mx8_Ku4
// Computes C += A[M×K] × B[K×8] using VSX vec_madd with K-unroll-4.
// Used as the fallback for M=1,2 and on non-Power10 targets for all M.
// ---------------------------------------------------------------------------
template <int32_t M, typename kv_cache_t>
FORCE_INLINE void gemm_micro_ppc64le_Mx8_Ku4(
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

  // dcbt prefetch for B rows in the GEMM hot loop.
#if VSX_ATTN_PREFETCH_DIST > 0
#  define PREFETCH_B(k_off) \
     __builtin_prefetch(B + (int64_t)((k_off)) * ldb, 0, 3);
#else
#  define PREFETCH_B(k_off)  // prefetch disabled: VSX_ATTN_PREFETCH_DIST=0
#endif

  for (; k + 3 < K; k += 4) {
    PREFETCH_B(k + VSX_ATTN_PREFETCH_DIST)

    // Load 4 values of A for each Row M: A[k...k+3]
#define LOAD_A4(i)        \
  __vector float a##i##v; \
  IF_M(i) a##i##v = vec_xl(0, const_cast<float*>(a##i + k));
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
#undef PREFETCH_B

  for (; k < K; ++k) {
    __vector float b0, b1;
    load_row8_B_as_f32<kv_cache_t>(B + (int64_t)k * ldb, b0, b1);
#define TAIL_ROW(i)                              \
  IF_M(i) {                                      \
    __vector float ai = vec_splats(*(a##i + k)); \
    acc##i##_0 = vec_madd(b0, ai, acc##i##_0);   \
    acc##i##_1 = vec_madd(b1, ai, acc##i##_1);   \
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

// ---------------------------------------------------------------------------
// Power10 MMA (Matrix-Math Assist) micro-kernel
//
// Computes C[M×8] += A[M×K] × B[K×8] using xvf32gerpp outer-product
// instructions. Each xvf32gerpp accumulates a 4×4 FP32 outer product.
//
// Accumulator layout (M=16, N=8, 8 x __vector_quad — saturates all 8 acc regs):
//   acc0_lo: rows  0-3  x cols 0-3    acc0_hi: rows  0-3  x cols 4-7
//   acc1_lo: rows  4-7  x cols 0-3    acc1_hi: rows  4-7  x cols 4-7
//   acc2_lo: rows  8-11 x cols 0-3    acc2_hi: rows  8-11 x cols 4-7
//   acc3_lo: rows 12-15 x cols 0-3    acc3_hi: rows 12-15 x cols 4-7
//
// K-unroll=8: each iteration loads an 8×4 A-block per row-quadrant and
// transposes it in registers, halving loop overhead for large K.
// ---------------------------------------------------------------------------
#ifdef _ARCH_PWR10
template <int32_t M, typename kv_cache_t>
FORCE_INLINE void gemm_micro_MMA_Mx8(
    const float* __restrict A,       // [M x K], row-major, stride lda (FP32)
    const kv_cache_t* __restrict B,  // [K x 8], stride ldb (BF16 or FP32)
    float* __restrict C,             // [M x 8], row-major, stride ldc (FP32)
    int64_t lda, int64_t ldb, int64_t ldc,
    int32_t K, bool accumulate) {
  static_assert(M == 4 || M == 8 || M == 16,
                "MMA micro-kernel: M must be 4, 8, or 16");

  __vector_quad acc0_lo, acc0_hi;  // rows  0- 3
  __builtin_mma_xxsetaccz(&acc0_lo);
  __builtin_mma_xxsetaccz(&acc0_hi);

  [[maybe_unused]] __vector_quad acc1_lo, acc1_hi;  // rows  4- 7  (M >= 8)
  if constexpr (M >= 8) {
    __builtin_mma_xxsetaccz(&acc1_lo);
    __builtin_mma_xxsetaccz(&acc1_hi);
  }

  [[maybe_unused]] __vector_quad acc2_lo, acc2_hi;  // rows  8-11  (M == 16)
  [[maybe_unused]] __vector_quad acc3_lo, acc3_hi;  // rows 12-15  (M == 16)
  if constexpr (M == 16) {
    __builtin_mma_xxsetaccz(&acc2_lo);
    __builtin_mma_xxsetaccz(&acc2_hi);
    __builtin_mma_xxsetaccz(&acc3_lo);
    __builtin_mma_xxsetaccz(&acc3_hi);
  }

  int32_t k = 0;

  // -----------------------------------------------------------------------
  // Main loop: K-unroll = 8
  // Loads a 4-row × 8-col block of A per quadrant, transposes in registers
  // to produce 8 column-vectors, then fires 16 xvf32gerpp per quadrant.
  // -----------------------------------------------------------------------

  // Macro: load 4 rows × 8 K-cols of A at (base, k), produce 8 col-vectors.
  // suf_a covers cols k..k+3; suf_b covers cols k+4..k+7.
#define MMA_LOAD_T8K(base, sa, sb)                                          \
  __vector float _r0##sa = vec_xl(0, A + (base+0)*lda + k);                \
  __vector float _r1##sa = vec_xl(0, A + (base+1)*lda + k);                \
  __vector float _r2##sa = vec_xl(0, A + (base+2)*lda + k);                \
  __vector float _r3##sa = vec_xl(0, A + (base+3)*lda + k);                \
  __vector float _r0##sb = vec_xl(0, A + (base+0)*lda + k + 4);            \
  __vector float _r1##sb = vec_xl(0, A + (base+1)*lda + k + 4);            \
  __vector float _r2##sb = vec_xl(0, A + (base+2)*lda + k + 4);            \
  __vector float _r3##sb = vec_xl(0, A + (base+3)*lda + k + 4);            \
  __vector float _ta0##sa = vec_mergeh(_r0##sa, _r2##sa);                   \
  __vector float _ta1##sa = vec_mergel(_r0##sa, _r2##sa);                   \
  __vector float _ta2##sa = vec_mergeh(_r1##sa, _r3##sa);                   \
  __vector float _ta3##sa = vec_mergel(_r1##sa, _r3##sa);                   \
  __vector float _ta0##sb = vec_mergeh(_r0##sb, _r2##sb);                   \
  __vector float _ta1##sb = vec_mergel(_r0##sb, _r2##sb);                   \
  __vector float _ta2##sb = vec_mergeh(_r1##sb, _r3##sb);                   \
  __vector float _ta3##sb = vec_mergel(_r1##sb, _r3##sb);                   \
  __vector unsigned char sa##c0 =                                           \
      (__vector unsigned char)vec_mergeh(_ta0##sa, _ta2##sa);               \
  __vector unsigned char sa##c1 =                                           \
      (__vector unsigned char)vec_mergel(_ta0##sa, _ta2##sa);               \
  __vector unsigned char sa##c2 =                                           \
      (__vector unsigned char)vec_mergeh(_ta1##sa, _ta3##sa);               \
  __vector unsigned char sa##c3 =                                           \
      (__vector unsigned char)vec_mergel(_ta1##sa, _ta3##sa);               \
  __vector unsigned char sb##c4 =                                           \
      (__vector unsigned char)vec_mergeh(_ta0##sb, _ta2##sb);               \
  __vector unsigned char sb##c5 =                                           \
      (__vector unsigned char)vec_mergel(_ta0##sb, _ta2##sb);               \
  __vector unsigned char sb##c6 =                                           \
      (__vector unsigned char)vec_mergeh(_ta1##sb, _ta3##sb);               \
  __vector unsigned char sb##c7 =                                           \
      (__vector unsigned char)vec_mergel(_ta1##sb, _ta3##sb);

  // Macro: issue 16 gerpp for one row-quadrant using 8 A-cols and 8 B-pairs.
#define MMA_GERPP8(al, ah, a0,a1,a2,a3,a4,a5,a6,a7)                        \
  __builtin_mma_xvf32gerpp(&al, a0, (__vector unsigned char)b_lo_0);        \
  __builtin_mma_xvf32gerpp(&ah, a0, (__vector unsigned char)b_hi_0);        \
  __builtin_mma_xvf32gerpp(&al, a1, (__vector unsigned char)b_lo_1);        \
  __builtin_mma_xvf32gerpp(&ah, a1, (__vector unsigned char)b_hi_1);        \
  __builtin_mma_xvf32gerpp(&al, a2, (__vector unsigned char)b_lo_2);        \
  __builtin_mma_xvf32gerpp(&ah, a2, (__vector unsigned char)b_hi_2);        \
  __builtin_mma_xvf32gerpp(&al, a3, (__vector unsigned char)b_lo_3);        \
  __builtin_mma_xvf32gerpp(&ah, a3, (__vector unsigned char)b_hi_3);        \
  __builtin_mma_xvf32gerpp(&al, a4, (__vector unsigned char)b_lo_4);        \
  __builtin_mma_xvf32gerpp(&ah, a4, (__vector unsigned char)b_hi_4);        \
  __builtin_mma_xvf32gerpp(&al, a5, (__vector unsigned char)b_lo_5);        \
  __builtin_mma_xvf32gerpp(&ah, a5, (__vector unsigned char)b_hi_5);        \
  __builtin_mma_xvf32gerpp(&al, a6, (__vector unsigned char)b_lo_6);        \
  __builtin_mma_xvf32gerpp(&ah, a6, (__vector unsigned char)b_hi_6);        \
  __builtin_mma_xvf32gerpp(&al, a7, (__vector unsigned char)b_lo_7);        \
  __builtin_mma_xvf32gerpp(&ah, a7, (__vector unsigned char)b_hi_7);

  for (; k + 7 < K; k += 8) {
#if VSX_ATTN_PREFETCH_DIST > 0
    __builtin_prefetch(B + (int64_t)(k + VSX_ATTN_PREFETCH_DIST) * ldb, 0, 3);
#endif

    __vector float b_lo_0, b_hi_0, b_lo_1, b_hi_1;
    __vector float b_lo_2, b_hi_2, b_lo_3, b_hi_3;
    __vector float b_lo_4, b_hi_4, b_lo_5, b_hi_5;
    __vector float b_lo_6, b_hi_6, b_lo_7, b_hi_7;
    load_row8_B_as_f32<kv_cache_t>(B+(int64_t)(k+0)*ldb, b_lo_0, b_hi_0);
    load_row8_B_as_f32<kv_cache_t>(B+(int64_t)(k+1)*ldb, b_lo_1, b_hi_1);
    load_row8_B_as_f32<kv_cache_t>(B+(int64_t)(k+2)*ldb, b_lo_2, b_hi_2);
    load_row8_B_as_f32<kv_cache_t>(B+(int64_t)(k+3)*ldb, b_lo_3, b_hi_3);
    load_row8_B_as_f32<kv_cache_t>(B+(int64_t)(k+4)*ldb, b_lo_4, b_hi_4);
    load_row8_B_as_f32<kv_cache_t>(B+(int64_t)(k+5)*ldb, b_lo_5, b_hi_5);
    load_row8_B_as_f32<kv_cache_t>(B+(int64_t)(k+6)*ldb, b_lo_6, b_hi_6);
    load_row8_B_as_f32<kv_cache_t>(B+(int64_t)(k+7)*ldb, b_lo_7, b_hi_7);

    MMA_LOAD_T8K(0, q0a, q0b)
    MMA_GERPP8(acc0_lo, acc0_hi,
               q0ac0, q0ac1, q0ac2, q0ac3, q0bc4, q0bc5, q0bc6, q0bc7)

    if constexpr (M >= 8) {
      MMA_LOAD_T8K(4, q1a, q1b)
      MMA_GERPP8(acc1_lo, acc1_hi,
                 q1ac0, q1ac1, q1ac2, q1ac3, q1bc4, q1bc5, q1bc6, q1bc7)
    }

    if constexpr (M == 16) {
      MMA_LOAD_T8K(8, q2a, q2b)
      MMA_GERPP8(acc2_lo, acc2_hi,
                 q2ac0, q2ac1, q2ac2, q2ac3, q2bc4, q2bc5, q2bc6, q2bc7)

      MMA_LOAD_T8K(12, q3a, q3b)
      MMA_GERPP8(acc3_lo, acc3_hi,
                 q3ac0, q3ac1, q3ac2, q3ac3, q3bc4, q3bc5, q3bc6, q3bc7)
    }
  }
#undef MMA_LOAD_T8K
#undef MMA_GERPP8

  // -----------------------------------------------------------------------
  // Residual loop: K-unroll = 4 (handles K % 8 in {4,5,6,7})
  // -----------------------------------------------------------------------
  for (; k + 3 < K; k += 4) {
    __vector float b_lo_0, b_hi_0, b_lo_1, b_hi_1;
    __vector float b_lo_2, b_hi_2, b_lo_3, b_hi_3;
    load_row8_B_as_f32<kv_cache_t>(B+(int64_t)(k+0)*ldb, b_lo_0, b_hi_0);
    load_row8_B_as_f32<kv_cache_t>(B+(int64_t)(k+1)*ldb, b_lo_1, b_hi_1);
    load_row8_B_as_f32<kv_cache_t>(B+(int64_t)(k+2)*ldb, b_lo_2, b_hi_2);
    load_row8_B_as_f32<kv_cache_t>(B+(int64_t)(k+3)*ldb, b_lo_3, b_hi_3);

    auto do_quad4 = [&](int base,
                        __vector_quad& al, __vector_quad& ah) {
      __vector float r0 = vec_xl(0, A + (base+0)*lda + k);
      __vector float r1 = vec_xl(0, A + (base+1)*lda + k);
      __vector float r2 = vec_xl(0, A + (base+2)*lda + k);
      __vector float r3 = vec_xl(0, A + (base+3)*lda + k);
      __vector float t0 = vec_mergeh(r0, r2);
      __vector float t1 = vec_mergel(r0, r2);
      __vector float t2 = vec_mergeh(r1, r3);
      __vector float t3 = vec_mergel(r1, r3);
      __vector unsigned char c0 = (__vector unsigned char)vec_mergeh(t0, t2);
      __vector unsigned char c1 = (__vector unsigned char)vec_mergel(t0, t2);
      __vector unsigned char c2 = (__vector unsigned char)vec_mergeh(t1, t3);
      __vector unsigned char c3 = (__vector unsigned char)vec_mergel(t1, t3);
      __builtin_mma_xvf32gerpp(&al, c0, (__vector unsigned char)b_lo_0);
      __builtin_mma_xvf32gerpp(&ah, c0, (__vector unsigned char)b_hi_0);
      __builtin_mma_xvf32gerpp(&al, c1, (__vector unsigned char)b_lo_1);
      __builtin_mma_xvf32gerpp(&ah, c1, (__vector unsigned char)b_hi_1);
      __builtin_mma_xvf32gerpp(&al, c2, (__vector unsigned char)b_lo_2);
      __builtin_mma_xvf32gerpp(&ah, c2, (__vector unsigned char)b_hi_2);
      __builtin_mma_xvf32gerpp(&al, c3, (__vector unsigned char)b_lo_3);
      __builtin_mma_xvf32gerpp(&ah, c3, (__vector unsigned char)b_hi_3);
    };

    do_quad4(0, acc0_lo, acc0_hi);
    if constexpr (M >= 8)  do_quad4(4,  acc1_lo, acc1_hi);
    if constexpr (M == 16) do_quad4(8,  acc2_lo, acc2_hi);
    if constexpr (M == 16) do_quad4(12, acc3_lo, acc3_hi);
  }

  // -----------------------------------------------------------------------
  // Scalar tail: K % 4 != 0
  // -----------------------------------------------------------------------
  for (; k < K; ++k) {
    __vector float b_lo_f, b_hi_f;
    load_row8_B_as_f32<kv_cache_t>(B + (int64_t)k * ldb, b_lo_f, b_hi_f);
    __vector unsigned char b_lo = (__vector unsigned char)b_lo_f;
    __vector unsigned char b_hi = (__vector unsigned char)b_hi_f;

    auto tail_quad = [&](int base,
                         __vector_quad& al, __vector_quad& ah) {
      __vector float a_col_f = (__vector float){
          A[(int64_t)(base+0)*lda + k], A[(int64_t)(base+1)*lda + k],
          A[(int64_t)(base+2)*lda + k], A[(int64_t)(base+3)*lda + k]};
      __vector unsigned char a_col = (__vector unsigned char)a_col_f;
      __builtin_mma_xvf32gerpp(&al, a_col, b_lo);
      __builtin_mma_xvf32gerpp(&ah, a_col, b_hi);
    };

    tail_quad(0, acc0_lo, acc0_hi);
    if constexpr (M >= 8)  tail_quad(4,  acc1_lo, acc1_hi);
    if constexpr (M == 16) tail_quad(8,  acc2_lo, acc2_hi);
    if constexpr (M == 16) tail_quad(12, acc3_lo, acc3_hi);
  }

  // -----------------------------------------------------------------------
  // Disassemble accumulators and store to C.
  // -----------------------------------------------------------------------
  auto store_quad = [&](int base,
                        __vector_quad& al, __vector_quad& ah) {
    __vector float res_lo[4], res_hi[4];
    __builtin_mma_disassemble_acc(res_lo, &al);
    __builtin_mma_disassemble_acc(res_hi, &ah);
    for (int32_t i = 0; i < 4; ++i) {
      float* row_c = C + (int64_t)(base + i) * ldc;
      if (accumulate) {
        res_lo[i] = vec_add(res_lo[i], vec_xl(0, row_c));
        res_hi[i] = vec_add(res_hi[i], vec_xl(0, row_c + 4));
      }
      vec_xst(res_lo[i], 0, row_c);
      vec_xst(res_hi[i], 0, row_c + 4);
    }
  };

  store_quad(0, acc0_lo, acc0_hi);
  if constexpr (M >= 8)  store_quad(4,  acc1_lo, acc1_hi);
  if constexpr (M == 16) store_quad(8,  acc2_lo, acc2_hi);
  if constexpr (M == 16) store_quad(12, acc3_lo, acc3_hi);
}
#endif  // _ARCH_PWR10

// ---------------------------------------------------------------------------
// Macro dispatcher: routes M-block tiles to MMA (Power10) or VSX (fallback).
// M=16 preferred (saturates all 8 acc regs); M=8 next; M=4 minimum MMA tile.
// M=2 and M=1 always use the VSX kernel (MMA overhead not amortized).
// To disable MMA and force VSX path, compile with -DVSX_DISABLE_MMA=1
// ---------------------------------------------------------------------------
template <int32_t N, typename kv_cache_t>
FORCE_INLINE void gemm_macro_ppc64le_Mx8_Ku4(const float* __restrict A,
                                             const kv_cache_t* __restrict B,
                                             float* __restrict C, int32_t M,
                                             int32_t K, int64_t lda,
                                             int64_t ldb, int64_t ldc,
                                             bool accumulate) {
  static_assert(N % 8 == 0, "N must be a multiple of 8");
  for (int32_t m = 0; m < M;) {
    int32_t mb = (M - m >= 16) ? 16
               : (M - m >=  8) ?  8
               : (M - m >=  4) ?  4
               : (M - m >=  2) ?  2 : 1;
    const float* Ab = A + m * lda;
    float* Cb = C + m * ldc;

    for (int32_t n = 0; n < N; n += 8) {
      const kv_cache_t* Bn = B + n;
      float* Cn = Cb + n;
      switch (mb) {
        case 16:
#if defined(_ARCH_PWR10) && !defined(VSX_DISABLE_MMA)
          gemm_micro_MMA_Mx8<16, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                             K, accumulate);
#else
          // VSX fallback: split into two M=8 calls
          gemm_micro_ppc64le_Mx8_Ku4<8, kv_cache_t>(Ab,            Bn, Cn,
                                                    lda, ldb, ldc, K, accumulate);
          gemm_micro_ppc64le_Mx8_Ku4<8, kv_cache_t>(Ab + 8 * lda,  Bn,
                                                    Cn + 8 * ldc,
                                                    lda, ldb, ldc, K, accumulate);
#endif
          break;
        case 8:
#if defined(_ARCH_PWR10) && !defined(VSX_DISABLE_MMA)
          gemm_micro_MMA_Mx8<8, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                            K, accumulate);
#else
          gemm_micro_ppc64le_Mx8_Ku4<8, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                    K, accumulate);
#endif
          break;
        case 4:
#if defined(_ARCH_PWR10) && !defined(VSX_DISABLE_MMA)
          gemm_micro_MMA_Mx8<4, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                            K, accumulate);
#else
          gemm_micro_ppc64le_Mx8_Ku4<4, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                    K, accumulate);
#endif
          break;
        case 2:
          gemm_micro_ppc64le_Mx8_Ku4<2, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                    K, accumulate);
          break;
        default:
          gemm_micro_ppc64le_Mx8_Ku4<1, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                    K, accumulate);
          break;
      }
    }
    m += mb;
  }
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
      gemm_macro_ppc64le_Mx8_Ku4<BLOCK_SIZE_ALIGNMENT, kv_cache_t>(
          a_tile, b_tile, c_tile, m_size, k_size, lda, ldb, ldc, accum_c);
    } else {
      gemm_macro_ppc64le_Mx8_Ku4<HEAD_SIZE_ALIGNMENT, kv_cache_t>(
          a_tile, b_tile, c_tile, m_size, dynamic_k_size, lda, ldb, ldc,
          accum_c);
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

  // Vectorized copy_q_heads_tile
  // Converts BF16 to FP32 and scales, with vectorized tail handling.
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

        // Vectorized tail: handle remaining 1-7 elements without scalar
        // fallback. vec_xl_len(ptr, bytes) RETURNS the loaded vector
        // (same pattern as vec_xl); vec_xst_len(vec, ptr, bytes) stores it.
        if (d < head_dim) {
          const int32_t tail = head_dim - d;
          if constexpr (is_bf16) {
            // Partial load of `tail` BF16 elements (2 bytes each) -> expand to
            // FP32 via the same mergeh/mergel trick as load_row8_B_as_f32.
            // vec_xl_len returns __vector signed char; cast to unsigned short.
            __vector unsigned short raw =
                (__vector unsigned short)vec_xl_len(
                    reinterpret_cast<const signed char*>(curr_src + d),
                    static_cast<size_t>(tail) * sizeof(uint16_t));
            __vector unsigned short zeros = vec_splat_u16(0);
            // Only the low half (elements 0..3) is used for tail <= 4;
            // for tail 5..7 the high half carries elements 4..6.
            __vector float v0 = (__vector float)vec_mergeh(zeros, raw);
            v0 = vec_mul(v0, scale_vec);
            // Store only the bytes that correspond to valid elements.
            const size_t lo_bytes =
                static_cast<size_t>(tail > 4 ? 4 : tail) * sizeof(float);
            vec_xst_len(v0, curr_dst + d, lo_bytes);
            if (tail > 4) {
              __vector float v1 = (__vector float)vec_mergel(zeros, raw);
              v1 = vec_mul(v1, scale_vec);
              const size_t hi_bytes =
                  static_cast<size_t>(tail - 4) * sizeof(float);
              vec_xst_len(v1, curr_dst + d + 4, hi_bytes);
            }
          } else {
            // FP32 tail: partial load of `tail` floats, scale, partial store.
            // vec_xl_len returns __vector signed char; cast to float.
            __vector float v0 =
                (__vector float)vec_xl_len(
                    reinterpret_cast<const signed char*>(curr_src + d),
                    static_cast<size_t>(tail) * sizeof(float));
            v0 = vec_mul(v0, scale_vec);
            vec_xst_len(v0, curr_dst + d,
                        static_cast<size_t>(tail) * sizeof(float));
          }
        }
      }
    }
  }

  // Cache KV sequence
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
    // k_inv and v_inv are unused on VSX: FP8 KV cache is not supported on
    // PowerPC. The parameters are present to match the common interface.
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

          // Unroll hint: each iteration touches an independent cache line
          // (stride = block_size elements). The compiler can pipeline
          // store-address computation across unrolled iterations.
#pragma GCC unroll 8
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

          // Vectorized value copy: type-agnostic 16-byte VSX loads/stores.
          // head_dim is a compile-time template constant so the compiler
          // fully unrolls this loop; vec_xl_len/vec_xst_len handle any tail.
          constexpr int64_t bytes =
              static_cast<int64_t>(sizeof(scalar_t)) * head_dim;
          const signed char* rs =
              reinterpret_cast<const signed char*>(val_src);
          signed char* rd = reinterpret_cast<signed char*>(val_dst);
          int64_t b = 0;
          for (; b + 15 < bytes; b += 16) {
            __vector signed char chunk = vec_xl(0, rs + b);
            vec_xst(chunk, 0, rd + b);
          }
          if (b < bytes) {
            __vector signed char chunk =
                vec_xl_len(rs + b, static_cast<size_t>(bytes - b));
            vec_xst_len(chunk, rd + b, static_cast<size_t>(bytes - b));
          }
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
