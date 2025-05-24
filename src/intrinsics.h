#ifndef __INTRINSICS_H
#define __INTRINSICS_H

#include <stdint.h>

#if defined(__ARM_NEON__)
#include <arm_neon.h>

#define INTRINSIC static inline __attribute__((always_inline))

typedef int64x2_t v128;

INTRINSIC v128 v128_load_aligned(const void *p) { return vreinterpretq_s64_u8(vld1q_u8((const uint8_t *)p)); }

INTRINSIC void v128_store_aligned(void *p, v128 r) { vst1q_u8((uint8_t *)p, vreinterpretq_u8_s64(r)); }

INTRINSIC int v128_test(v128 x, v128 y) {
  int8x16_t d = vsubq_u8(x, y);
  return !!(vget_high_s64(d) | vget_low_s64(d));
}

INTRINSIC v128 v128_max_u8(v128 x, v128 y) { return vreinterpretq_s64_u8(vmaxq_u8(vreinterpretq_u8_s64(x), vreinterpretq_u8_s64(y))); }

INTRINSIC v128 v128_min_u8(v128 x, v128 y) { return vreinterpretq_s64_u8(vminq_u8(vreinterpretq_u8_s64(x), vreinterpretq_u8_s64(y))); }

INTRINSIC v128 v128_sub_u8(v128 x, v128 y) { return vreinterpretq_s64_u8(vsubq_u8(vreinterpretq_u8_s64(x), vreinterpretq_u8_s64(y))); }

INTRINSIC v128 v128_add_u8(v128 x, v128 y) { return vreinterpretq_s64_u8(vaddq_u8(vreinterpretq_u8_s64(x), vreinterpretq_u8_s64(y))); }

INTRINSIC v128 v128_dup_u8(uint32_t x) { return vreinterpretq_s64_u8(vdupq_n_u8(x)); }

INTRINSIC v128 v128_cmpgt_s8(v128 x, v128 y) { return vreinterpretq_s64_u8(vcgtq_s8(vreinterpretq_s8_s64(x), vreinterpretq_s8_s64(y))); }

INTRINSIC v128 v128_and(v128 x, v128 y) { return vandq_s64(x, y); }

INTRINSIC v128 v128_lsr_u8(v128 a, int32_t c) { return vreinterpretq_s64_u8(vshlq_u8(vreinterpretq_u8_s64(a), vdupq_n_s8(-c))); }

#elif defined(__SSE2__)
#include "emmintrin.h"

#define INTRINSIC static inline __attribute__((always_inline))

typedef __m128i v128;

INTRINSIC v128 v128_load_aligned(const void *p) { return _mm_load_si128((__m128i *)p); }

INTRINSIC void v128_store_aligned(void *p, v128 r) { _mm_store_si128((__m128i *)p, r); }

INTRINSIC int v128_test(v128 x, v128 y) { return _mm_movemask_epi8(_mm_cmpeq_epi8(x, y)) != 0xffff; }

INTRINSIC v128 v128_max_u8(v128 a, v128 b) { return _mm_max_epu8(a, b); }

INTRINSIC v128 v128_min_u8(v128 a, v128 b) { return _mm_min_epu8(a, b); }

INTRINSIC v128 v128_sub_u8(v128 a, v128 b) { return _mm_sub_epu8(a, b); }

INTRINSIC v128 v128_add_u8(v128 a, v128 b) { return _mm_add_epu8(a, b); }

INTRINSIC v128 v128_dup_u8(uint32_t x) { return _mm_set1_epi8(x); }

INTRINSIC v128 v128_cmpgt_s8(v128 a, v128 b) { return _mm_cmpgt_epi8(a, b); }

INTRINSIC v128 v128_and(v128 a, v128 b) { return _mm_and_si128(a, b); }

INTRINSIC v128 v128_lsr_u8(v128 a, int32_t c) {
  __m128i x = _mm_cvtsi32_si128(c + 8);
  return _mm_packus_epi16(_mm_srl_epi16(_mm_unpacklo_epi8(_mm_setzero_si128(), a), x),
                          _mm_srl_epi16(_mm_unpackhi_epi8(_mm_setzero_si128(), a), x));
}

#endif

#endif
