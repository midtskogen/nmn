#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <fstream>
#include <jpeglib.h>
#include <math.h>
#include <pthread.h>
#include <setjmp.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/sendfile.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/mem.h>
}

#include "PanoToolsInterface.h"
#include "Panorama.h"

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(55, 28, 1)
#define av_frame_alloc avcodec_alloc_frame
#define av_frame_free avcodec_free_frame
#endif

// For a brief description of the algorithm, see the comments in the main()
// function towards the end of this file.  All work is done from there.

// Example compile options and usage:

// g++ -o bin/metdetect src/metdetect.c -msse4.2 -Wno-deprecated-declarations
// -Isrc/hugin-2023.0.0/build/src -Isrc/hugin-2023.0.0/src/hugin_base/panotools
// -Isrc/hugin-2023.0.0/src/hugin_base
// -Isrc/hugin-2023.0.0/src/hugin_base/panodata -Isrc/hugin-2023.0.0/src -O6
// -ljpeg -lm -lavutil -lavcodec -pthread -Wl,-rpath=/usr/lib/hugin
// /usr/lib/hugin/libhuginbase.so.0.0

// g++ -o bin/metdetect src/metdetect.c -march=native -mfpu=neon
// -Wno-deprecated-declarations -Isrc/hugin-2022.0.0/build/src
// -Isrc/hugin-2022.0.0/src/hugin_base/panotools
// -Isrc/hugin-2022.0.0/src/hugin_base
// -Isrc/hugin-2022.0.0/src/hugin_base/panodata -Isrc/hugin-2022.0.0/src -O6
// -ljpeg -lm -lavutil -lavcodec -pthread -Wl,-rpath=/usr/local/lib/hugin
// /usr/lib/hugin/libhuginbase.so.0.0

// g++ -o metdetect metdetect.c -Isrc/hugin -O6 -ljpeg -lm -lavutil -lavcodec
// -pthread -Wl,-rpath=/usr/lib/hugin /usr/lib/hugin/libhuginbase.so.0.0

// g++ -o bin/metdetect src/metdetect.c -Isrc/hugin -O6 -mfpu=neon
// -fomit-frame-pointer -ljpeg -lm -lavutil -lavcodec -pthread
// -Wl,-rpath=/usr/lib/hugin /usr/lib/hugin/libhuginbase.so.0.0

// ffmpeg -loglevel quiet -vsync 0 -nostdin -i
// /meteor/cam1/20141202/21/mini_13.mp4 -f rawvideo -pix_fmt gray - | time
// ./metdetect -w800 -h600 -m/meteor/cam1/mask-mini.jpg -j 0 -; eog frame-00*

// ffmpeg -loglevel quiet -vsync 0 -nostdin -i
// /meteor/cam3/20150428/00/full_32.mp4 -f rawvideo -codec:v copy -codec:a none
// -bsf:v h264_mp4toannexb - | ./metdetect -m/meteor/cam3/mask-mini.jpg -e -

static FILE *inputfile = 0;
static int mfc_open = 0;
static int ffmpeg_open = 0;

typedef enum { ALL, IP8172, IP816A, IP9171, IP8151, IMX291, IMX291SD, IMX291HD, IMX307, IMX307SD, IMX307HD } camid;

camid camera = ALL;

unsigned char *readh264frame(unsigned int *l);

// Exynox MFC hardware decoder.
int init_mfcdec(unsigned char *h264, unsigned int s);

unsigned char *decode_mfc(unsigned int *width, unsigned int *height, unsigned int *stride, unsigned char *h264, unsigned int size,
                          unsigned char **u, unsigned char **v);
int close_mfcdec();

// Fallback ffmpeg software decoder
int init_ffmpegdec(int single_thread);
unsigned char *decode_ffmpeg(unsigned int *width, unsigned int *height, unsigned int *stride, unsigned char *h264, unsigned int size,
                             unsigned char **u, unsigned char **v);
int close_ffmpegdec();

// H.264 decode functions.  Try hardware decoding with software fallback.
static unsigned char *decode(unsigned int *width, unsigned int *height, unsigned int *stride, unsigned char *h264, unsigned int size,
                             unsigned char **u, unsigned char **v) {
  unsigned char *p = decode_mfc(width, height, stride, h264, size, u, v);
  return p ? p : decode_ffmpeg(width, height, stride, h264, size, u, v);
}

// Initialise, decode first frame and return geometry and Y frame
static unsigned char *init_dec(FILE *f, unsigned int *width, unsigned int *height, unsigned int *stride, int sw, unsigned char **u,
                               unsigned char **v) {
  inputfile = f;
  unsigned int size;
  unsigned char *h264 = readh264frame(&size);
  if (sw || !init_mfcdec(h264, size)) init_ffmpegdec(sw == 2);
  unsigned char *ret = decode(width, height, stride, h264, size, u, v);
  return ret;
}

static void close_dec() {
  if (!close_mfcdec()) close_ffmpegdec();
}

static void *aligned_malloc(size_t size, uintptr_t align) {
  void *m = malloc(size + sizeof(void *) + align);
  if (!m) return m;
  void **r = (void **)((((uintptr_t)m) + sizeof(void *) + align - 1) & ~(align - 1));
  r[-1] = m;
  return r;
}

static void aligned_free(void *p) { free(((void **)p)[-1]); }

#ifndef SIMD_INLINE
#ifdef __GNUC__
#define SIMD_INLINE static inline __attribute__((always_inline))
#elif __STDC_VERSION__ >= 199901L
#define SIMD_INLINE static inline
#else
#define SIMD_INLINE static
#endif
#endif

SIMD_INLINE unsigned int log2i(uint32_t x) { return 31 - __builtin_clz(x); }

#if defined(__ARM_NEON__) || defined(__aarch64__)
#include "arm_neon.h"

typedef int64x2_t v128;
typedef int64x1_t v64;

SIMD_INLINE v128 v128_load_aligned(const void *p) { return vreinterpretq_s64_u8(vld1q_u8((const uint8_t *)p)); }

SIMD_INLINE v128 v128_load_unaligned(const void *p) { return v128_load_aligned(p); }

SIMD_INLINE v64 v64_load_aligned(const void *p) { return vreinterpret_s64_u8(vld1_u8((const uint8_t *)p)); }

SIMD_INLINE void v128_store_aligned(void *p, v128 r) { vst1q_u8((uint8_t *)p, vreinterpretq_u8_s64(r)); }

SIMD_INLINE void v128_store_unaligned(void *p, v128 r) { vst1q_u8((uint8_t *)p, vreinterpretq_u8_s64(r)); }

SIMD_INLINE v128 v128_zero() { return vreinterpretq_s64_u8(vdupq_n_u8(0)); }

SIMD_INLINE v128 v128_or(v128 x, v128 y) { return vorrq_s64(x, y); }

SIMD_INLINE v128 v128_and(v128 x, v128 y) { return vandq_s64(x, y); }

SIMD_INLINE v128 v128_add_8(v128 x, v128 y) { return vreinterpretq_s64_u8(vaddq_u8(vreinterpretq_u8_s64(x), vreinterpretq_u8_s64(y))); }

SIMD_INLINE v128 v128_add_16(v128 x, v128 y) {
  return vreinterpretq_s64_u16(vaddq_u16(vreinterpretq_u16_s64(x), vreinterpretq_u16_s64(y)));
}

SIMD_INLINE uint64_t v128_hadd_u8(v128 x) {
  uint64x2_t t = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(vreinterpretq_u8_s64(x))));
  return vget_lane_s32(vreinterpret_s32_u64(vadd_u64(vget_high_u64(t), vget_low_u64(t))), 0);
}

SIMD_INLINE v128 v128_sub_8(v128 x, v128 y) { return vreinterpretq_s64_u8(vsubq_u8(vreinterpretq_u8_s64(x), vreinterpretq_u8_s64(y))); }

SIMD_INLINE v128 v128_sub_16(v128 x, v128 y) {
  return vreinterpretq_s64_u16(vsubq_u16(vreinterpretq_u16_s64(x), vreinterpretq_u16_s64(y)));
}

SIMD_INLINE v128 v128_ssub_u8(v128 x, v128 y) { return vreinterpretq_s64_u8(vqsubq_u8(vreinterpretq_u8_s64(x), vreinterpretq_u8_s64(y))); }

SIMD_INLINE v128 v128_shr_u8(v128 a, unsigned int c) { return vreinterpretq_s64_u8(vshlq_u8(vreinterpretq_u8_s64(a), vdupq_n_s8(-c))); }

SIMD_INLINE v128 v128_shr_u16(v128 a, unsigned int c) {
  return vreinterpretq_s64_u16(vshlq_u16(vreinterpretq_u16_s64(a), vdupq_n_s16(-c)));
}

SIMD_INLINE v128 v128_shl_16(v128 a, unsigned int c) { return vreinterpretq_s64_u16(vshlq_u16(vreinterpretq_u16_s64(a), vdupq_n_s16(c))); }

SIMD_INLINE v128 v128_unziplo_8(v128 x, v128 y) {
  uint8x16x2_t r = vuzpq_u8(vreinterpretq_u8_s64(y), vreinterpretq_u8_s64(x));
  return vreinterpretq_s64_u8(r.val[0]);
}

SIMD_INLINE v128 v128_padd_u8(v128 a) { return vreinterpretq_s64_u16(vpaddlq_u8(vreinterpretq_u8_s64(a))); }

SIMD_INLINE v128 v128_cmple_u8(v128 x, v128 y) { return vreinterpretq_s64_u8(vcleq_u8(vreinterpretq_u8_s64(x), vreinterpretq_u8_s64(y))); }

SIMD_INLINE v64 v64_cmple_u8(v64 x, v64 y) { return vreinterpret_s64_u8(vcle_u8(vreinterpret_u8_s64(x), vreinterpret_u8_s64(y))); }

SIMD_INLINE v128 v128_max_u8(v128 x, v128 y) { return vreinterpretq_s64_u8(vmaxq_u8(vreinterpretq_u8_s64(x), vreinterpretq_u8_s64(y))); }

SIMD_INLINE v64 v64_max_u8(v64 x, v64 y) { return vreinterpret_s64_u8(vmax_u8(vreinterpret_u8_s64(x), vreinterpret_u8_s64(y))); }

SIMD_INLINE int v128_hasgt_u8(v128 a, v128 b) {
  v128 t = vreinterpretq_s64_u8(vcgtq_u8(vreinterpretq_u8_s64(a), vreinterpretq_u8_s64(b)));
  return !!((uint64_t)vget_low_s64(t) | (uint64_t)vget_high_s64(t));
}

SIMD_INLINE int v64_hasgt_u8(v64 a, v64 b) {
  v64 t = vreinterpret_s64_u8(vcgt_u8(vreinterpret_u8_s64(a), vreinterpret_u8_s64(b)));
  return !!(uint64_t)t;
}

SIMD_INLINE int v128_haszero_u8(v128 a) {
  v128 t = vreinterpretq_s64_u8(vceqq_u8(vreinterpretq_u8_s64(a), vdupq_n_u8(0)));
  return !!((uint64_t)vget_low_s64(t) | (uint64_t)vget_high_s64(t));
}

SIMD_INLINE v64 v64_dup_16(uint16_t x) { return vreinterpret_s64_u16(vdup_n_u16(x)); }

SIMD_INLINE v128 v128_dup_16(uint16_t x) { return vreinterpretq_s64_u16(vdupq_n_u16(x)); }

SIMD_INLINE v128 v128_dup_8(uint8_t x) { return vreinterpretq_s64_u8(vdupq_n_u8(x)); }

SIMD_INLINE v128 v128_from_64(uint64_t x, uint64_t y) { return vcombine_s64((int64x1_t)y, (int64x1_t)x); }

SIMD_INLINE v64 v64_from_64(uint64_t x) { return vcreate_s64(x); }

SIMD_INLINE int64_t v64_dotp_s16(v64 x, v64 y) {
  int64x2_t r = vpaddlq_s32(vmull_s16(vreinterpret_s16_s64(x), vreinterpret_s16_s64(y)));
  return (int64_t)vget_high_s64(r) + (int64_t)vget_low_s64(r);
}

SIMD_INLINE v64 v64_from_16(uint16_t a, uint16_t b, uint16_t c, uint16_t d) {
  return vcreate_s64((uint64_t)a << 48 | (uint64_t)b << 32 | (uint64_t)c << 16 | d);
}

SIMD_INLINE uint64_t v64_hadd_u8(v64 x) { return (uint64_t)vpaddl_u32(vpaddl_u16(vpaddl_u8(vreinterpret_u8_s64(x)))); }

SIMD_INLINE v64 v64_cmpgt_s16(v64 x, v64 y) { return vreinterpret_s64_u16(vcgt_s16(vreinterpret_s16_s64(x), vreinterpret_s16_s64(y))); }

SIMD_INLINE v64 v64_abs_s16(v64 x) { return vreinterpret_s64_s16(vabs_s16(vreinterpret_s16_s64(x))); }

SIMD_INLINE v64 v64_unpacklo_u8_s16(v64 a) { return vreinterpret_s64_u16(vget_low_u16(vmovl_u8(vreinterpret_u8_s64(a)))); }

SIMD_INLINE v64 v64_load_unaligned(const void *p) { return vreinterpret_s64_u8(vld1_u8((const uint8_t *)p)); }

SIMD_INLINE v64 v64_add_16(v64 x, v64 y) { return vreinterpret_s64_s16(vadd_s16(vreinterpret_s16_s64(x), vreinterpret_s16_s64(y))); }

SIMD_INLINE v64 v64_padd_u8(v64 a) { return vreinterpret_s64_u16(vpaddl_u8(vreinterpret_u8_s64(a))); }

#elif __SSE2__
#include <emmintrin.h>
#if defined(__SSE4_1__)
#include <smmintrin.h>
#endif

typedef __m128i v128;
typedef __m128i v64;

SIMD_INLINE v128 v128_load_aligned(const void *p) { return _mm_load_si128((__m128i *)p); }

SIMD_INLINE v128 v128_load_unaligned(const void *p) {
#if defined(__SSSE3__)
  return (__m128i)_mm_lddqu_si128((__m128i *)p);
#else
  return _mm_loadu_si128((__m128i *)p);
#endif
}

SIMD_INLINE v64 v64_load_aligned(const void *p) { return _mm_loadl_epi64((__m128i *)p); }

SIMD_INLINE void v128_store_aligned(void *p, v128 a) { _mm_store_si128((__m128i *)p, a); }

SIMD_INLINE void v128_store_unaligned(void *p, v128 a) { _mm_storeu_si128((__m128i *)p, a); }

SIMD_INLINE v128 v128_zero() { return _mm_setzero_si128(); }

SIMD_INLINE v128 v128_or(v128 a, v128 b) { return _mm_or_si128(a, b); }

SIMD_INLINE v128 v128_and(v128 a, v128 b) { return _mm_and_si128(a, b); }

SIMD_INLINE v128 v128_add_8(v128 a, v128 b) { return _mm_add_epi8(a, b); }

SIMD_INLINE v128 v128_add_16(v128 a, v128 b) { return _mm_add_epi16(a, b); }

SIMD_INLINE v128 v128_sub_8(v128 a, v128 b) { return _mm_sub_epi8(a, b); }

SIMD_INLINE v128 v128_sub_16(v128 a, v128 b) { return _mm_sub_epi16(a, b); }

SIMD_INLINE v128 v128_ssub_u8(v128 a, v128 b) { return _mm_subs_epu8(a, b); }

SIMD_INLINE v128 v128_shr_u8(v128 a, unsigned int c) {
  __m128i x = _mm_cvtsi32_si128(c + 8);
  return _mm_packus_epi16(_mm_srl_epi16(_mm_unpacklo_epi8(_mm_setzero_si128(), a), x),
                          _mm_srl_epi16(_mm_unpackhi_epi8(_mm_setzero_si128(), a), x));
}

SIMD_INLINE v128 v128_shr_u16(v128 a, unsigned int c) { return _mm_srli_epi16(a, c); }

SIMD_INLINE v128 v128_shl_16(v128 a, unsigned int c) { return _mm_slli_epi16(a, c); }

SIMD_INLINE uint64_t v128_hadd_u8(v128 a) {
  v128 t = _mm_sad_epu8(a, _mm_setzero_si128());
  return _mm_cvtsi128_si32(_mm_unpacklo_epi64(t, _mm_setzero_si128())) + (uint32_t)_mm_cvtsi128_si32(_mm_srli_si128(t, 8));
}

SIMD_INLINE v128 v128_padd_u8(v128 a) { return _mm_add_epi16(_mm_srli_epi16(a, 8), _mm_srli_epi16(_mm_slli_epi16(a, 8), 8)); }

SIMD_INLINE v128 v128_unziphi_8(v128 a, v128 b) { return _mm_packs_epi16(_mm_srai_epi16(b, 8), _mm_srai_epi16(a, 8)); }

SIMD_INLINE v128 v128_unziplo_8(v128 a, v128 b) {
#if defined(__SSSE3__)
  v128 order = _mm_cvtsi64_si128(0x0e0c0a0806040200LL);
  return _mm_unpacklo_epi64(_mm_shuffle_epi8(b, order), _mm_shuffle_epi8(a, order));
#else
  return v128_unziphi_8(_mm_slli_si128(a, 1), _mm_slli_si128(b, 1));
#endif
}

SIMD_INLINE v128 v128_max_u8(v128 a, v128 b) { return _mm_max_epu8(a, b); }

SIMD_INLINE v128 v128_cmple_u8(v128 a, v128 b) { return _mm_cmpeq_epi8(_mm_min_epu8(a, b), a); }

SIMD_INLINE v64 v64_cmple_u8(v64 a, v64 b) { return _mm_cmpeq_epi8(_mm_min_epu8(a, b), a); }

SIMD_INLINE int v128_hasgt_u8(v128 a, v128 b) { return _mm_movemask_epi8(v128_cmple_u8(a, b)) != 0xffff; }

SIMD_INLINE int v64_hasgt_u8(v64 a, v64 b) {
  return _mm_movemask_epi8(v64_cmple_u8(_mm_unpacklo_epi64(a, a), _mm_unpacklo_epi64(b, b))) != 0xffff;
}

SIMD_INLINE int v128_haszero_u8(v128 a) { return _mm_movemask_epi8(_mm_cmpeq_epi8(a, _mm_setzero_si128())); }

SIMD_INLINE v64 v64_dup_16(uint16_t x) { return _mm_set1_epi16(x); }

SIMD_INLINE v128 v128_dup_16(uint16_t x) { return _mm_set1_epi16(x); }

SIMD_INLINE v128 v128_dup_8(uint8_t x) { return _mm_set1_epi8(x); }

SIMD_INLINE v128 v128_from_64(uint64_t x, uint64_t y) { return _mm_set_epi64(_mm_cvtsi64_m64(x), _mm_cvtsi64_m64(y)); }

SIMD_INLINE v64 v64_from_64(uint64_t x) { return _mm_set_epi32(0, 0, x >> 32, (uint32_t)x); }

SIMD_INLINE int64_t v64_dotp_s16(v64 a, v64 b) {
  __m128i r = _mm_madd_epi16(a, b);
#if defined(__SSE4_1__)
  __m128i x = _mm_cvtepi32_epi64(r);
  return _mm_cvtsi128_si64(_mm_add_epi64(x, _mm_srli_si128(x, 8)));
#else
  return (int64_t)_mm_cvtsi128_si32(_mm_srli_si128(r, 4)) + (int64_t)_mm_cvtsi128_si32(r);
#endif
}

SIMD_INLINE v64 v64_from_16(uint16_t a, uint16_t b, uint16_t c, uint16_t d) {
  return _mm_packs_epi32(_mm_set_epi32((int16_t)a, (int16_t)b, (int16_t)c, (int16_t)d), _mm_setzero_si128());
}

SIMD_INLINE uint64_t v64_hadd_u8(v64 a) { return (uint32_t)_mm_cvtsi128_si32(_mm_sad_epu8(a, _mm_setzero_si128())); }

SIMD_INLINE v64 v64_cmpgt_s16(v64 a, v64 b) { return _mm_cmpgt_epi16(a, b); }

SIMD_INLINE v64 v64_abs_s16(v64 a) {
#if defined(__SSSE3__)
  return _mm_abs_epi16(a);
#else
  return _mm_max_epi16(a, _mm_sub_epi16(_mm_setzero_si128(), a));
#endif
}

SIMD_INLINE v64 v64_unpacklo_u8_s16(v64 a) { return _mm_unpacklo_epi8(a, _mm_setzero_si128()); }

SIMD_INLINE v64 v64_load_unaligned(const void *p) { return _mm_loadl_epi64((__m128i *)p); }

SIMD_INLINE v64 v64_add_16(v64 a, v64 b) { return _mm_add_epi16(a, b); }

SIMD_INLINE v64 v64_padd_u8(v64 a) { return _mm_add_epi16(_mm_srli_epi16(a, 8), _mm_srli_epi16(_mm_slli_epi16(a, 8), 8)); }

#else

#warning Compiling without SIMD optimisations

typedef union {
  uint8_t u8[16];
  uint16_t u16[8];
  uint32_t u32[4];
  uint64_t u64[2];
  int8_t s8[16];
  int16_t s16[8];
  int32_t s32[4];
  int64_t s64[2];
} v128;

typedef union {
  uint8_t u8[8];
  uint16_t u16[4];
  uint32_t u32[2];
  uint64_t u64;
  int8_t s8[8];
  int16_t s16[4];
  int32_t s32[2];
  int64_t s64;
} v64;

SIMD_INLINE v128 v128_load_unaligned(const void *p) {
  v128 t;
  uint8_t *pp = (uint8_t *)p;
  uint8_t *q = (uint8_t *)&t;
  int c;
  for (c = 0; c < 16; c++) q[c] = pp[c];
  return t;
}

SIMD_INLINE v128 v128_load_aligned(const void *p) {
  if ((uintptr_t)p & 15) {
    fprintf(stderr, "Warning: unaligned v128 load at %p\n", p);
    abort();
  }
  return v128_load_unaligned(p);
}

SIMD_INLINE v64 v64_load_unaligned(const void *p) {
  v64 t;
  uint8_t *pp = (uint8_t *)p;
  uint8_t *q = (uint8_t *)&t;
  int c;
  for (c = 0; c < 8; c++) q[c] = pp[c];
  return t;
}

SIMD_INLINE v64 v64_load_aligned(const void *p) {
  if ((uintptr_t)p & 7) {
    fprintf(stderr, "Warning: unaligned v64 load at %p\n", p);
    abort();
  }
  return v64_load_unaligned(p);
}

SIMD_INLINE void v128_store_unaligned(void *p, v128 a) {
  uint8_t *pp = (uint8_t *)p;
  uint8_t *q = (uint8_t *)&a;
  int c;
  for (c = 0; c < 16; c++) pp[c] = q[c];
}

SIMD_INLINE void v128_store_aligned(void *p, v128 a) {
  if ((uintptr_t)p & 15) {
    fprintf(stderr, "Warning: unaligned v128 store at %p\n", p);
    abort();
  }
  v128_store_unaligned(p, a);
}

SIMD_INLINE v128 v128_dup_8(uint8_t x) {
  v128 t;
  for (int c = 0; c < 16; c++) t.u8[c] = x;
  return t;
}

SIMD_INLINE v128 v128_dup_16(uint16_t x) {
  v128 t;
  for (int c = 0; c < 8; c++) t.u16[c] = x;
  return t;
}

SIMD_INLINE v128 v128_zero() {
  v128 t;
  t.u64[1] = t.u64[0] = 0;
  return t;
}

SIMD_INLINE v128 v128_or(v128 a, v128 b) {
  v128 t;
  t.u64[0] = a.u64[0] | b.u64[0];
  t.u64[1] = a.u64[1] | b.u64[1];
  return t;
}

SIMD_INLINE v128 v128_and(v128 a, v128 b) {
  v128 t;
  t.u64[0] = a.u64[0] & b.u64[0];
  t.u64[1] = a.u64[1] & b.u64[1];
  return t;
}

SIMD_INLINE v128 v128_add_8(v128 a, v128 b) {
  v128 t;
  int c;
  for (c = 0; c < 16; c++) t.u8[c] = a.u8[c] + b.u8[c];
  return t;
}

SIMD_INLINE v128 v128_add_16(v128 a, v128 b) {
  v128 t;
  int c;
  for (c = 0; c < 8; c++) t.u16[c] = a.u16[c] + b.u16[c];
  return t;
}

SIMD_INLINE v128 v128_sub_8(v128 a, v128 b) {
  v128 t;
  int c;
  for (c = 0; c < 16; c++) t.u8[c] = a.u8[c] - b.u8[c];
  return t;
}

SIMD_INLINE v128 v128_ssub_u8(v128 a, v128 b) {
  v128 t;
  int c;
  for (c = 0; c < 16; c++) t.u8[c] = (int32_t)((uint32_t)a.u8[c] - (uint32_t)b.u8[c]) < 0 ? 0 : a.u8[c] - b.u8[c];
  return t;
}

SIMD_INLINE v128 v128_sub_16(v128 a, v128 b) {
  v128 t;
  int c;
  for (c = 0; c < 8; c++) t.u16[c] = a.u16[c] - b.u16[c];
  return t;
}

SIMD_INLINE v128 v128_shr_u8(v128 a, unsigned int n) {
  v128 t;
  int c;
  if (n > 7) {
    fprintf(stderr, "Undefined u8 shift right %d\n", n);
    abort();
  }
  for (c = 0; c < 16; c++) t.u8[c] = a.u8[c] >> n;
  return t;
}

SIMD_INLINE v128 v128_shr_u16(v128 a, unsigned int n) {
  v128 t;
  int c;
  if (n > 15) {
    fprintf(stderr, "Undefined u16 shift right %d\n", n);
    abort();
  }
  for (c = 0; c < 8; c++) t.u16[c] = a.u16[c] >> n;
  return t;
}

SIMD_INLINE v128 v128_shl_16(v128 a, unsigned int n) {
  v128 t;
  int c;
  if (n > 15) {
    fprintf(stderr, "Undefined u16 shift left %d\n", n);
    abort();
  }
  for (c = 0; c < 8; c++) t.u16[c] = a.u16[c] << n;
  return t;
}

SIMD_INLINE uint64_t v128_hadd_u8(v128 a) {
  return a.u8[15] + a.u8[14] + a.u8[13] + a.u8[12] + a.u8[11] + a.u8[10] + a.u8[9] + a.u8[8] + a.u8[7] + a.u8[6] + a.u8[5] + a.u8[4] +
         a.u8[3] + a.u8[2] + a.u8[1] + a.u8[0];
}

SIMD_INLINE int big_endian() {
  const uint16_t t = 0x100;
  return *(const uint8_t *)&t;
}

SIMD_INLINE v128 v128_unziplo_8(v128 a, v128 b) {
  v128 t;
  if (big_endian()) {
    t.u8[15] = b.u8[15];
    t.u8[14] = b.u8[13];
    t.u8[13] = b.u8[11];
    t.u8[12] = b.u8[9];
    t.u8[11] = b.u8[7];
    t.u8[10] = b.u8[5];
    t.u8[9] = b.u8[3];
    t.u8[8] = b.u8[1];
    t.u8[7] = a.u8[15];
    t.u8[6] = a.u8[13];
    t.u8[5] = a.u8[11];
    t.u8[4] = a.u8[9];
    t.u8[3] = a.u8[7];
    t.u8[2] = a.u8[5];
    t.u8[1] = a.u8[3];
    t.u8[0] = a.u8[1];
  } else {
    t.u8[15] = a.u8[14];
    t.u8[14] = a.u8[12];
    t.u8[13] = a.u8[10];
    t.u8[12] = a.u8[8];
    t.u8[11] = a.u8[6];
    t.u8[10] = a.u8[4];
    t.u8[9] = a.u8[2];
    t.u8[8] = a.u8[0];
    t.u8[7] = b.u8[14];
    t.u8[6] = b.u8[12];
    t.u8[5] = b.u8[10];
    t.u8[4] = b.u8[8];
    t.u8[3] = b.u8[6];
    t.u8[2] = b.u8[4];
    t.u8[1] = b.u8[2];
    t.u8[0] = b.u8[0];
  }
  return t;
}

SIMD_INLINE v128 v128_padd_u8(v128 a) {
  v128 t;
  t.u16[0] = (uint16_t)a.u8[0] + (uint16_t)a.u8[1];
  t.u16[1] = (uint16_t)a.u8[2] + (uint16_t)a.u8[3];
  t.u16[2] = (uint16_t)a.u8[4] + (uint16_t)a.u8[5];
  t.u16[3] = (uint16_t)a.u8[6] + (uint16_t)a.u8[7];
  t.u16[4] = (uint16_t)a.u8[8] + (uint16_t)a.u8[9];
  t.u16[5] = (uint16_t)a.u8[10] + (uint16_t)a.u8[11];
  t.u16[6] = (uint16_t)a.u8[12] + (uint16_t)a.u8[13];
  t.u16[7] = (uint16_t)a.u8[14] + (uint16_t)a.u8[15];
  return t;
}

SIMD_INLINE v128 v128_max_u8(v128 a, v128 b) {
  v128 t;
  int c;
  for (c = 0; c < 16; c++) t.u8[c] = a.u8[c] > b.u8[c] ? a.u8[c] : b.u8[c];
  return t;
}

SIMD_INLINE v128 v128_cmple_u8(v128 a, v128 b) {
  v128 t;
  int c;
  for (c = 0; c < 16; c++) t.u8[c] = -(a.u8[c] <= b.u8[c]);
  return t;
}

SIMD_INLINE int v128_hasgt_u8(v128 a, v128 b) {
  int c, r = 0;
  for (c = 0; c < 16; c++) r |= a.u8[c] > b.u8[c];

  return r;
}

SIMD_INLINE int v64_hasgt_u8(v64 a, v64 b) {
  int c, r = 0;
  for (c = 0; c < 8; c++) r |= a.u8[c] > b.u8[c];

  return r;
}

SIMD_INLINE int v128_haszero_u8(v128 a) {
  int c, r = 0;
  for (c = 0; c < 16; c++) r |= a.u8[c] == 0;
  return r;
}

SIMD_INLINE v64 v64_dup_16(uint16_t x) {
  v64 t;
  t.u16[0] = t.u16[1] = t.u16[2] = t.u16[3] = x;
  return t;
}

SIMD_INLINE v128 v128_from_64(uint64_t x, uint64_t y) {
  v128 t;
  t.u64[0] = t.u64[1] = x;
  return t;
}

SIMD_INLINE v64 v64_from_64(uint64_t x) {
  v64 t;
  t.u64 = x;
  return t;
}

SIMD_INLINE int64_t v64_dotp_s16(v64 a, v64 b) {
  return (int64_t)(a.s16[3] * b.s16[3] + a.s16[2] * b.s16[2]) + (int64_t)(a.s16[1] * b.s16[1] + a.s16[0] * b.s16[0]);
}

SIMD_INLINE v64 v64_from_16(uint16_t a, uint16_t b, uint16_t c, uint16_t d) {
  v64 t;
  if (big_endian()) {
    t.u16[3] = a;
    t.u16[2] = b;
    t.u16[1] = c;
    t.u16[0] = d;
  } else {
    t.u16[0] = a;
    t.u16[1] = b;
    t.u16[2] = c;
    t.u16[3] = d;
  }
  return t;
}

SIMD_INLINE uint64_t v64_hadd_u8(v64 a) { return a.u8[7] + a.u8[6] + a.u8[5] + a.u8[4] + a.u8[3] + a.u8[2] + a.u8[1] + a.u8[0]; }

SIMD_INLINE v64 v64_cmpgt_s16(v64 a, v64 b) {
  v64 t;
  int c;
  for (c = 0; c < 4; c++) t.s16[c] = -(a.s16[c] > b.s16[c]);
  return t;
}

SIMD_INLINE v64 v64_abs_s16(v64 a) {
  v64 t;
  int c;
  for (c = 0; c < 4; c++) t.u16[c] = (int16_t)a.u16[c] > 0 ? a.u16[c] : -a.u16[c];
  return t;
}

SIMD_INLINE v64 v64_unpacklo_u8_s16(v64 a) {
  v64 t;
  int endian = big_endian() * 4;
  t.s16[3] = (int16_t)a.u8[3 + endian];
  t.s16[2] = (int16_t)a.u8[2 + endian];
  t.s16[1] = (int16_t)a.u8[1 + endian];
  t.s16[0] = (int16_t)a.u8[0 + endian];
  return t;
}

SIMD_INLINE v64 v64_add_16(v64 a, v64 b) {
  v64 t;
  int c;
  for (c = 0; c < 4; c++) t.u16[c] = a.u16[c] + b.u16[c];
  return t;
}

SIMD_INLINE v64 v64_padd_u8(v64 a) {
  v64 t;
  t.u16[0] = (uint16_t)a.u8[0] + (uint16_t)a.u8[1];
  t.u16[1] = (uint16_t)a.u8[2] + (uint16_t)a.u8[3];
  t.u16[2] = (uint16_t)a.u8[4] + (uint16_t)a.u8[5];
  t.u16[3] = (uint16_t)a.u8[6] + (uint16_t)a.u8[7];
  return t;
}

#endif

static inline int iabs(int a) { return a < 0 ? -a : a; }
static inline int max(int a, int b) { return a < b ? b : a; }
static inline int min(int a, int b) { return a < b ? a : b; }
static inline int clip(int a, int b, int c) { return max(min(a, c - 1), b); }

static char *getsavefile(const char *filename) {
  static char savefile[256];

  savefile[0] = 0;
  if (filename) {
    struct stat status;

    FILE *f = fopen(filename, "r");
    if (!f) return savefile;

    unsigned int i;
    for (i = 0; !feof(f) && i < sizeof(savefile) - 1; i++) {
      savefile[i] = fgetc(f);
      if (savefile[i] == '\n') break;
    }
    savefile[i] = 0;
    fclose(f);
  }
  return savefile;
}

/* Do a fast 8x8 transform and check high freq coeffs against threshold */
static int frequency_check(uint8_t *p, int stride, int threshold) {
  v64 thr = v64_dup_16(threshold);
  v64 pppp = v64_from_64(0x0001000100010001LL);
  v64 ppmm = v64_from_64(0x00020001fffffffeLL);
  v64 pmmp = v64_from_64(0x0001ffffffff0001LL);
  v64 pmpm = v64_from_64(0x0001fffe0002ffffLL);

  /* Reduce 8x8 to 4x4 by summing 2x2 squares */
  v64 d0 = v64_add_16(v64_padd_u8(v64_load_unaligned(p + 0 * stride)), v64_padd_u8(v64_load_unaligned(p + 1 * stride)));
  v64 d1 = v64_add_16(v64_padd_u8(v64_load_unaligned(p + 2 * stride)), v64_padd_u8(v64_load_unaligned(p + 3 * stride)));
  v64 d2 = v64_add_16(v64_padd_u8(v64_load_unaligned(p + 4 * stride)), v64_padd_u8(v64_load_unaligned(p + 5 * stride)));
  v64 d3 = v64_add_16(v64_padd_u8(v64_load_unaligned(p + 6 * stride)), v64_padd_u8(v64_load_unaligned(p + 7 * stride)));

  /* Transform */
  v64 tmp0 = v64_from_16(v64_dotp_s16(d0, pppp), v64_dotp_s16(d1, pppp), v64_dotp_s16(d2, pppp), v64_dotp_s16(d3, pppp));
  v64 tmp1 = v64_from_16(v64_dotp_s16(d0, ppmm), v64_dotp_s16(d1, ppmm), v64_dotp_s16(d2, ppmm), v64_dotp_s16(d3, ppmm));
  v64 tmp2 = v64_from_16(v64_dotp_s16(d0, pmmp), v64_dotp_s16(d1, pmmp), v64_dotp_s16(d2, pmmp), v64_dotp_s16(d3, pmmp));
  v64 tmp3 = v64_from_16(v64_dotp_s16(d0, pmpm), v64_dotp_s16(d1, pmpm), v64_dotp_s16(d2, pmpm), v64_dotp_s16(d3, pmpm));

  v64 coeff0 = v64_abs_s16(v64_from_16(0,  // v64_dotp_s16(tmp0, pppp)
                                       0,  // v64_dotp_s16(tmp1, pppp),
                                       v64_dotp_s16(tmp2, pppp), v64_dotp_s16(tmp3, pppp)));
  /* Check four results at a time */
  if (v64_hadd_u8(v64_cmpgt_s16(coeff0, thr))) return 1;

  v64 coeff1 = v64_abs_s16(v64_from_16(0,  // v64_dotp_s16(tmp0, ppmm),
                                       v64_dotp_s16(tmp1, ppmm), v64_dotp_s16(tmp2, ppmm), v64_dotp_s16(tmp3, ppmm)));
  if (v64_hadd_u8(v64_cmpgt_s16(coeff1, thr))) return 1;

  v64 coeff2 =
      v64_abs_s16(v64_from_16(v64_dotp_s16(tmp0, pmmp), v64_dotp_s16(tmp1, pmmp), v64_dotp_s16(tmp2, pmmp), v64_dotp_s16(tmp3, pmmp)));
  if (v64_hadd_u8(v64_cmpgt_s16(coeff2, thr))) return 1;

  v64 coeff3 =
      v64_abs_s16(v64_from_16(v64_dotp_s16(tmp0, pmpm), v64_dotp_s16(tmp1, pmpm), v64_dotp_s16(tmp2, pmpm), v64_dotp_s16(tmp3, pmpm)));
  if (v64_hadd_u8(v64_cmpgt_s16(coeff3, thr))) return 1;

  return 0;
}

typedef struct {
  int raw;                              // 0 = H.264 input, 1 = raw Y8 input
  unsigned int width;                   // Video width
  unsigned int height;                  // Video height
  unsigned int swidth;                  // Scaled width (internal use)
  unsigned int sheight;                 // Scaled height (internal use)
  char *logfile;                        // Optional log file for debugging
  char *maskfile;                       // Optional mask file for video
  char *maxfile;                        // When contains a filename, a max image is written
  char *savefile;                       // Actual file for max image (internal)
  char *ptofile;                        // Optional (but recommended) pto file for converting x,y to
                                        // az,alt
  HuginBase::Panorama *pano;            // Hugin panorama
  HuginBase::PTools::Transform *trafo;  // Hugin transform
  char *execute;                        // Command to run after detection
  char *eventdir;                       // Where to store events (/tmp)
  char *snapshot_dir;                   // Where to store shapshots (/tmp)
  int snapshot_interval;                // Interval for snapshots, seconds (0 = no snapshots)
  int snapshot_integration;             // Integration time for snapshots, frames
  double mintrail_sec;                  // Minimum meteor trail length, seconds (1.2)
  double maxtrail_sec;                  // Maximum meteor trail length, seconds (10.0)
  unsigned int mintrail;                // Minimum meteor trail length, # of frames (internal)
  unsigned int maxtrail;                // Maximum meteor trail length, # of frames (internal)
  double minspeed;                      // Minimum meteor speed, % frame width (0.2)
  double maxspeed;                      // Maximum meteor speed, % frame width (5.0)
  int leveltest;                        // Flag whether to test horizontal movement near horizon
  double minspeedkms;                   // Minimum lateral meteor speed, km/s (2.0)
  double maxspeedkms;                   // Maximum lateral meteor speed, km/s (50.0)
  unsigned int downscale_thr;           // Frame width threshold for downscaling
  int swdec;                            // Use software decoder (0)
  int exit;                             // Exit on error
  int savejpg;                          // Save jpg files
  int saveevent;                        // Do not save event files, display instead
  int nothreads;                        // Single threaded operation (0)
  unsigned int numspots;                // Number of brighest spots to detect (3), max 8
  int brightness;                       // Minimum brightness (16)
  int filter;                           // Hadamard transform threshold (256)
  double dct_threshold;                 // DCT threshold (auto)
  int dct;                              // DCT threshold, # frames (internal)
  int heartbeat;                        // Log timestamp every nth frame (0)
  unsigned int lookahead;               // Frame lookahead / history length (900)
  double peak;                          // Minimum peak brightness multiplier (4.0)
  double corr;                          // Line correlation
  double gnomonic_corr;                 // Gnomonic line correlation
  double spacing_corr;                  // Spacing correlation
  double flash_thr;                     // Threshold for flash detection, off = 0. (1.2)
  int flash;                            // Flash detected (internal use)
  double ptoscale;                      // Az/alt scale factor (internal use)
  double ptowidth;                      // Image width in pto file (internal use)
  double ptoheight;                     // Image width in pto file (internal use)
  time_t lastreport_ts;                 // Timestamp of the last event (internal use)
  int ts_future;                        // Check that video timestamp is not in the future
  int timestamp_top;                    // Look for the timestamp in the top part of the frame
  int old_detection;                    // Use old detection algorith
  int restart;                          // Restart rather than exit
  int thread_timeout;                   // Number of seconds to wait for thread join (120)
  FILE *log;
} config;

static void printhelp() {
  printf(
      "metdetect reads a raw 8 bit greyscale video stream from file or stdin, "
      "looks for\n"
      "meteors, reports any events, and also stores a max frame (a continuous "
      "expsure\n"
      "since the last such store) to file when instructed by an external "
      "application.\n\n"
      "Options:\n"
      "  -C <config file>: Read arguments from a configuration file.\n"
      "  -w <input width>: Width of input stream.  Implies raw input.\n"
      "  -h <input height>: Height of input stream.  Implies raw input.\n"
      "  -m <mask file>: A jpeg file serving as a mask for the input.  Black "
      "pixels in\n"
      "      the mask will remove the corresponding pixel in the input "
      "stream.\n"
      "  -x <max file>: Whenever a filename is written into this file, the max "
      "frame\n"
      "      will be written to that file, the file will be deleted and the "
      "max buffer\n"
      "      will be reset.  An external application could for instance update "
      "this file\n"
      "      once an hour to have metdetect store one hour exposures.\n"
      "  -d <event directory>: When an event occurs, a report will be written "
      "in a\n"
      "      subdirectory in the given event directory.  The date of the event "
      "will be\n"
      "      the name of the subdirectory.  Default: \"/tmp\"\n"
      "  -e: Turn off event reporting to file.\n"
      "  -j <image number>: Write annotated jpeg files of the input.  The "
      "files will be\n"
      "      named frame-<sequence number>.jpg.  Useful for debugging and "
      "offline\n"
      "      analysis.\n"
      "      0: Input image\n"
      "      1: Enhanced input image\n"
      "      2: Difference image\n"
      "      3: Input image with mask\n"
      "      4: Enhanced input image with mask\n"
      "      5: Difference image with mask\n"
      "  -n <number of spots>: Only the brightest moving spots in the input "
      "stream will\n"
      "      be examined.  The number of spots can be changed.  Legal values "
      "are 1 .. 16.\n"
      "      This ensures that meteors can be detected even if there are "
      "brighter\n"
      "      airplanes or other artificial objects moving in the frame.  "
      "Default: 3\n"
      "  -t <minimum trail length>: Minimum duration for a meteor in seconds. "
      "If the\n"
      "      number is very low, false detections get more likely.  If the "
      "number is\n"
      "      high, short lived meteors will not be detected.  Default: 1.2.\n"
      "  -u <maximum trail length>: Maximum duration for a meteor in seconds.\n"
      "      Meteors are rarely visible for more than 10 seconds, so objects "
      "visible for\n"
      "      longer than this are likely airplanes or satellites.  Default: "
      "10.0.\n"
      "  -p <minimum speed>: An object moving slower than this value, given as "
      "percent\n"
      "      of the frame width per frame, will not be considered a meteor.  "
      "Slow moving\n"
      "      objects are frequently airplanes, satellites or noise.  Default: "
      "0.15\n"
      "  -q <maximum speed>: An object moving faster than this value, given as "
      "percent\n"
      "      of the frame width per frame, will not be considered a meteor.  "
      "Apparently\n"
      "      very fast moving objects are likely noise.  Default: 5.0\n"
      "  -P <minimum lateral speed>: An object with a lateral speed in km/s "
      "less than this\n"
      "      value will not be considered a meteor.  Default: 2.0\n"
      "  -Q <maximum lateral speed>: An object with a lateral speed in km/s "
      "greater than this\n"
      "      value will not be considered a meteor.  Default: 50.0\n"
      "  -W <frame width threshold>: If the frame width is equal or larger "
      "than the specified\n"
      "      threshold, the input will be downscaled prior to processing to "
      "improve processing\n"
      "      speed.  Default: 1600.\n"
      "  -l: Do not exclude objects moving horizontally near the horizon\n"
      "  -a <lookahead>: Number of lookahead frames.\n"
      "      This should be at least twice the maximum trail length.  A low "
      "value could\n"
      "      cause the maximum trail length to fail giving false detections.  "
      "A high value\n"
      "      causes latency in the detection.  Events are still detected, but "
      "it could be\n"
      "      minutes after the fact.  Default: 900 (three minutes at 5 "
      "fps).\n");
  printf(
      "  -b <minimum brightness>: The minimum brightness for a spot to be "
      "examined.\n"
      "      A useful range could be 1 (faint) .. 1000 (bright), but it "
      "depends on the\n"
      "      input resolution, light sensitivity, noise level, etc.  Default: "
      "16.  A low\n"
      "      value will increase the chances of detecting faint meteors, but "
      "also increase\n"
      "      the chances for false detections due to noise.\n"
      "  -y <energy threshold>: Threshold used in a Hadamard transform of "
      "potential\n"
      "      objects in the original frame.  A high value reduces the "
      "likelihood\n"
      "      that noise gets interpreted as an object, but also increases the\n"
      "      minimum brightness needed for detection.  Useful range: 0 - "
      "2000.\n"
      "      Default: 256.\n"
      "  -Y <energy threshold>: Threshold used in a DCT of the brightness "
      "values of a\n"
      "      trail.  Meteors brightness changes are low frequency signals "
      "whilst airplanes\n"
      "      are often blinking resulting in a more high frequency signal.  A "
      "high\n"
      "      threshold will reduce the number of events discarded as unlikely "
      "meteors.\n"
      "      Default: auto\n"
      "  -c <correlations>: Correlation thresholds determining a track given "
      "as three\n"
      "      values (%%f,%%f,%%f).  1.0 means a perfect line, 0.0 means that "
      "anything goes.\n"
      "      The first value is the line correlation in the original frame.  "
      "The second is\n"
      "      the spacing correlation.  The third is the line correlation in a "
      "gnomonic\n"
      "      projection (only used in a pto file is specified with the -o "
      "option.  A low\n"
      "      value for the line correlation in the original frame doesn\'t "
      "necessarily mean\n"
      "      that there will be more false detections since the gnomonic "
      "correlation will\n"
      "      will discard most of these.  A low value would often increase the "
      "chances that\n"
      "      curved paths will be detected and discarded.  Default: "
      "0.50,0.90,0.9985.  If\n"
      "      no pto file is available, 0.90 is a better threshold than the "
      "default 0.50.\n"
      "  -f <flash threshold>: Threshold for flash detection.  Must be a value "
      "above\n"
      "      1.0, or 0 to disable.  Higher value for less sensitivity.  "
      "Default: 1.2\n"
      "  -g <minimum peak brightness multiplier>: The minimum peak brightness "
      "for a trail\n"
      "      as a multiplier of the minimum brightness setting (-b).  A high "
      "value lowers\n"
      "      the chances for false detections due to noise happening to line "
      "up.\n"
      "      Default: 4.0\n"
      "  -l <log file>: File to log debug messages.  Default: no logging.\n"
      "  -o <pto file>: Hugin pto file for an equirectangular panorama and the "
      "optical\n"
      "      for the camera lens.  Used to map pixel coordinates to azimuth "
      "and altitude.\n"
      "      Optional.\n"
      "  -r <executable>: Optional executable to run (asynchronously) after an "
      "event is\n"
      "      detected.  The event report file is passed as an argument.  "
      "Ignored with -e.\n"
      "  -k: Use software decoder.\n"
      "  -s: Single threaded operation.\n"
      "  -i: Allow video timestamps in the future.  Useful if the computer or "
      "camera\n"
      "      has a known error.\n"
      "  -T: Look for timestamp in the upper part of the frame instead of the "
      "lower part.\n"
      "  -R: Restart program rather than exit if input is stdin.\n"
      "  -v: Log timestamp every nth frame (0).\n"
      "  -S <interval>: Save a snapshot every nth second as indicated by this "
      "argument.  Not\n"
      "      supported for raw input video.  Stored as <snapshot "
      "dir>/%%Y%%m%%d/%%H/%%M/%%S.jpg\n"
      "      and <snapshor_dir>/snapshot.jpg.  <snapshot_dir> can be specified "
      "with -D.\n"
      "  -I <frames>: Snapshot integration in frames.\n"
      "  -X <errors>: Exit after the specified number of consecutive errors.  "
      "0 = never.\n"
      "  -D <dir>: Directory for snapshots.  Default /tmp.\n"
      "  -R: Restart rather than exit.\n"
      "  -M <name>: Camera model name.  Used to detect timestamps.  Supported "
      "models are:\n"
      "     IP8172, IP816A, IP9171, IP8151, IMX291, IMX291SD, IMX291HD, "
      "IMX307, IMX307SD, IMX307HD.\n"
      "     Default: guess (less reliable timestamp detection).\n"
      "  -A <seconds>: Number of seconds to wait for thread join (120).\n"
      "  -F: Use old detection algorith.\n"
      "  -L: Disable test for near horizontal tracks near the horizon.\n"
      "  -H: This help page.\n"
      "Written by Steinar Midtskogen <steinar@norskmeteornettverk.no> "
      "originally for the\n"
      "Vivotek IP8172 camera using 800x600 or 2560x1920 stream at 5 or 10 fps. "
      " The default\n"
      "values reflect useful settings for this camera.\n");
}

static void debug(FILE *log, const char *format, ...) {
  va_list args;
  va_start(args, format);
  if (log) {
    time_t ltime;
    struct tm result;
    char stime[32];
    ltime = time(nullptr);
    localtime_r(&ltime, &result);
    asctime_r(&result, stime);
    stime[strlen(stime) - 1] = 0;
    fprintf(log, "%s: ", stime);
    vfprintf(log, format, args);
  }
  fflush(log);
  va_end(args);
}

typedef struct {
  double x[16];
  double y[16];
  double az[16];
  double alt[16];
  double gnomx[16];
  double gnomy[16];
  int brightness[16];
  int size[16];
  double timestamp;
  unsigned int framebrightness;
} record;

typedef struct {
  double x;
  double y;
  double az;
  double alt;
  double gnomx;
  double gnomy;
  int brightness;
  int size;
  double timestamp;
  double time;
  unsigned int framebrightness;
  unsigned int neighbours;
  unsigned int closeneighbours;
  unsigned int pos;
} point;

typedef struct {
  int x;
  int y;
  int brightness;
} brightest;

static const uint16_t crc16tab[256] = {
  0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7, 0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef,
  0x1231, 0x0210, 0x3273, 0x2252, 0x52b5, 0x4294, 0x72f7, 0x62d6, 0x9339, 0x8318, 0xb37b, 0xa35a, 0xd3bd, 0xc39c, 0xf3ff, 0xe3de,
  0x2462, 0x3443, 0x0420, 0x1401, 0x64e6, 0x74c7, 0x44a4, 0x5485, 0xa56a, 0xb54b, 0x8528, 0x9509, 0xe5ee, 0xf5cf, 0xc5ac, 0xd58d,
  0x3653, 0x2672, 0x1611, 0x0630, 0x76d7, 0x66f6, 0x5695, 0x46b4, 0xb75b, 0xa77a, 0x9719, 0x8738, 0xf7df, 0xe7fe, 0xd79d, 0xc7bc,
  0x48c4, 0x58e5, 0x6886, 0x78a7, 0x0840, 0x1861, 0x2802, 0x3823, 0xc9cc, 0xd9ed, 0xe98e, 0xf9af, 0x8948, 0x9969, 0xa90a, 0xb92b,
  0x5af5, 0x4ad4, 0x7ab7, 0x6a96, 0x1a71, 0x0a50, 0x3a33, 0x2a12, 0xdbfd, 0xcbdc, 0xfbbf, 0xeb9e, 0x9b79, 0x8b58, 0xbb3b, 0xab1a,
  0x6ca6, 0x7c87, 0x4ce4, 0x5cc5, 0x2c22, 0x3c03, 0x0c60, 0x1c41, 0xedae, 0xfd8f, 0xcdec, 0xddcd, 0xad2a, 0xbd0b, 0x8d68, 0x9d49,
  0x7e97, 0x6eb6, 0x5ed5, 0x4ef4, 0x3e13, 0x2e32, 0x1e51, 0x0e70, 0xff9f, 0xefbe, 0xdfdd, 0xcffc, 0xbf1b, 0xaf3a, 0x9f59, 0x8f78,
  0x9188, 0x81a9, 0xb1ca, 0xa1eb, 0xd10c, 0xc12d, 0xf14e, 0xe16f, 0x1080, 0x00a1, 0x30c2, 0x20e3, 0x5004, 0x4025, 0x7046, 0x6067,
  0x83b9, 0x9398, 0xa3fb, 0xb3da, 0xc33d, 0xd31c, 0xe37f, 0xf35e, 0x02b1, 0x1290, 0x22f3, 0x32d2, 0x4235, 0x5214, 0x6277, 0x7256,
  0xb5ea, 0xa5cb, 0x95a8, 0x8589, 0xf56e, 0xe54f, 0xd52c, 0xc50d, 0x34e2, 0x24c3, 0x14a0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
  0xa7db, 0xb7fa, 0x8799, 0x97b8, 0xe75f, 0xf77e, 0xc71d, 0xd73c, 0x26d3, 0x36f2, 0x0691, 0x16b0, 0x6657, 0x7676, 0x4615, 0x5634,
  0xd94c, 0xc96d, 0xf90e, 0xe92f, 0x99c8, 0x89e9, 0xb98a, 0xa9ab, 0x5844, 0x4865, 0x7806, 0x6827, 0x18c0, 0x08e1, 0x3882, 0x28a3,
  0xcb7d, 0xdb5c, 0xeb3f, 0xfb1e, 0x8bf9, 0x9bd8, 0xabbb, 0xbb9a, 0x4a75, 0x5a54, 0x6a37, 0x7a16, 0x0af1, 0x1ad0, 0x2ab3, 0x3a92,
  0xfd2e, 0xed0f, 0xdd6c, 0xcd4d, 0xbdaa, 0xad8b, 0x9de8, 0x8dc9, 0x7c26, 0x6c07, 0x5c64, 0x4c45, 0x3ca2, 0x2c83, 0x1ce0, 0x0cc1,
  0xef1f, 0xff3e, 0xcf5d, 0xdf7c, 0xaf9b, 0xbfba, 0x8fd9, 0x9ff8, 0x6e17, 0x7e36, 0x4e55, 0x5e74, 0x2e93, 0x3eb2, 0x0ed1, 0x1ef0
};

static uint16_t crc16(const void *buf, int len) {
  const uint8_t *b = (const uint8_t *)buf;
  uint16_t crc = 0;
  for (int c = 0; c < len; c++) crc = (crc << 8) ^ crc16tab[((crc >> 8) ^ *b++) & 255];
  return crc;
}

typedef struct {
  /* "public" fields */
  struct jpeg_error_mgr pub;
  /* for return to caller */
  jmp_buf setjmp_buffer;
} jpegErrorManager;

static char jpegLastErrorMsg[JMSG_LENGTH_MAX];
static void jpegErrorExit(j_common_ptr cinfo) {
  /* cinfo->err actually points to a jpegErrorManager struct */
  jpegErrorManager *myerr = (jpegErrorManager *)cinfo->err;
  /* note : *(cinfo->err) is now equivalent to myerr->pub */

  /* output_message is a method to print an error message */
  /*(* (cinfo->err->output_message) ) (cinfo);*/

  /* Create the message */
  (*(cinfo->err->format_message))(cinfo, jpegLastErrorMsg);

  /* Jump to the setjmp point */
  longjmp(myerr->setjmp_buffer, 1);
}

static int savejpeg(FILE *fp, unsigned char *image, int width, int height, int quality) {
  int i, j;

  JSAMPROW y[16], cb[16], cr[16];
  JSAMPARRAY data[3];
  unsigned char *buf = 0;
  if (width & 15) {
    int width2 = (width + 15) & ~15;
    buf = (unsigned char *)aligned_malloc(width * ((height + 15) & ~15) * 1.5, 16);
    memset(buf, 0, width * ((height + 15) & ~15) * 1.5);
    if (buf) {
      for (int i = 0; i < height; i++) memcpy(buf + i * width, image + i * width2, width);
      memset(buf + width * height, 128, width * height / 2);
    } else
      return 0;
    image = buf;
  }

  struct jpeg_compress_struct cinfo;
  jpegErrorManager jerr;

  data[0] = y;
  data[1] = cb;
  data[2] = cr;

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = jpegErrorExit;
  /* Establish the setjmp return context for my_error_exit to use. */
  if (setjmp(jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    jpeg_destroy_compress(&cinfo);
    fclose(fp);
    if (buf) aligned_free(buf);
    return 0;
  }

  jpeg_create_compress(&cinfo);
  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = 3;
  jpeg_set_defaults(&cinfo);

  jpeg_set_colorspace(&cinfo, JCS_YCbCr);

  cinfo.raw_data_in = TRUE;
  cinfo.comp_info[0].h_samp_factor = 2;
  cinfo.comp_info[0].v_samp_factor = 2;
  cinfo.comp_info[1].h_samp_factor = 1;
  cinfo.comp_info[1].v_samp_factor = 1;
  cinfo.comp_info[2].h_samp_factor = 1;
  cinfo.comp_info[2].v_samp_factor = 1;

  jpeg_set_quality(&cinfo, quality, TRUE);
  cinfo.dct_method = JDCT_FASTEST;

  jpeg_stdio_dest(&cinfo, fp);
  jpeg_start_compress(&cinfo, TRUE);

  for (j = 0; j < height; j += 16) {
    for (i = 0; i < 16; i++) {
      y[i] = image + width * (i + j);
      if (i % 2 == 0) {
        cb[i / 2] = image + width * height + width / 2 * ((i + j) / 2);
        cr[i / 2] = image + width * height + width * height / 4 + width / 2 * ((i + j) / 2);
      }
    }
    jpeg_write_raw_data(&cinfo, data, 16);
  }

  if (buf) aligned_free(buf);

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
  return 1;
}

static time_t gettimestamp(unsigned char *img, int width, int height, int indicator, int threshold) {
  uint8_t img2[width * height / 4];
  for (int i = 0; i < height / 2; i++) {
    for (int j = 0; j < width / 2; j++) {
      img2[i * width / 2 + j] = img[i * 2 * width + j * 2];
    }
  }

  // Vivotek IP8172, IP816A-HP 8x11 font
  static const uint8_t text_0[11] = { 60, 102, 195, 195, 195, 195, 195, 195, 195, 102, 60 };
  static const uint8_t text_1[11] = { 120, 216, 24, 24, 24, 24, 24, 24, 24, 24, 126 };
  static const uint8_t text_2[11] = { 124, 134, 3, 3, 3, 6, 12, 24, 48, 96, 255 };
  static const uint8_t text_3[11] = { 124, 135, 3, 3, 7, 62, 7, 3, 3, 135, 124 };
  static const uint8_t text_4[11] = { 15, 27, 19, 35, 67, 195, 131, 255, 3, 3, 3 };
  static const uint8_t text_5[11] = { 254, 192, 192, 192, 252, 134, 3, 3, 3, 134, 124 };
  static const uint8_t text_6[11] = { 60, 98, 64, 192, 252, 230, 195, 195, 195, 102, 60 };
  static const uint8_t text_7[11] = { 255, 3, 7, 6, 6, 12, 12, 24, 24, 56, 48 };
  static const uint8_t text_8[11] = { 126, 231, 195, 195, 231, 60, 231, 195, 195, 231, 126 };
  static const uint8_t text_9[11] = { 60, 102, 195, 195, 195, 103, 63, 3, 2, 70, 60 };
  static const uint8_t text_bits[10] = { 44, 30, 32, 37, 38, 36, 41, 30, 56, 41 };
  static const uint8_t *text4_bits = text_bits;

  // Vivotek IP9171 8x16 font
  static const uint8_t text2_0[16] = { 28, 62, 127, 103, 227, 227, 227, 227, 227, 227, 227, 227, 103, 127, 62, 28 };
  static const uint8_t text2_1[16] = { 24, 248, 248, 184, 56, 56, 56, 56, 56, 56, 56, 56, 56, 254, 254, 0 };
  static const uint8_t text2_2[16] = { 28, 126, 127, 71, 7, 7, 7, 14, 14, 28, 56, 56, 112, 255, 255, 0 };
  static const uint8_t text2_3[16] = { 56, 126, 126, 6, 6, 6, 30, 60, 62, 7, 7, 7, 7, 254, 254, 56 };
  static const uint8_t text2_4[16] = { 0, 14, 14, 30, 30, 62, 54, 118, 230, 198, 255, 255, 6, 6, 6, 0 };
  static const uint8_t text2_5[16] = { 0, 252, 252, 192, 192, 192, 248, 252, 30, 14, 14, 14, 14, 252, 252, 112 };
  static const uint8_t text2_6[16] = { 14, 63, 63, 112, 96, 96, 254, 255, 247, 227, 227, 99, 99, 127, 62, 28 };
  static const uint8_t text2_7[16] = { 0, 255, 255, 7, 6, 6, 14, 14, 12, 28, 28, 24, 56, 56, 48, 0 };
  static const uint8_t text2_8[16] = { 28, 126, 127, 103, 99, 103, 126, 62, 126, 119, 227, 227, 227, 127, 126, 28 };
  static const uint8_t text2_9[16] = { 28, 126, 127, 231, 227, 227, 227, 231, 127, 127, 3, 7, 7, 126, 124, 56 };
  static const uint8_t text2_bits[10] = { 80, 57, 63, 63, 59, 60, 77, 47, 84, 79 };

  // Vivotek IP8151 8x11 font
  static const uint8_t text3_0[11] = { 60, 66, 66, 66, 66, 66, 66, 66, 60, 0, 0 };
  static const uint8_t text3_1[11] = { 16, 112, 16, 16, 16, 16, 16, 16, 124, 0, 0 };
  static const uint8_t text3_2[11] = { 60, 66, 66, 2, 4, 24, 32, 64, 126, 0, 0 };
  static const uint8_t text3_3[11] = { 60, 66, 2, 2, 28, 2, 2, 66, 60, 0, 0 };
  static const uint8_t text3_4[11] = { 4, 12, 20, 36, 68, 127, 4, 4, 4, 0, 0 };
  static const uint8_t text3_5[11] = { 126, 64, 64, 124, 2, 2, 2, 66, 60, 0, 0 };
  static const uint8_t text3_6[11] = { 28, 32, 64, 124, 66, 66, 66, 66, 60, 0, 0 };
  static const uint8_t text3_7[11] = { 126, 2, 4, 4, 8, 8, 16, 16, 32, 0, 0 };
  static const uint8_t text3_8[11] = { 60, 66, 66, 66, 60, 66, 66, 66, 60, 0, 0 };
  static const uint8_t text3_9[11] = { 60, 66, 66, 66, 66, 62, 2, 4, 56, 0, 0 };
  static const uint8_t text3_bits[10] = { 22, 15, 20, 19, 19, 22, 22, 14, 24, 22 };

  // IMX291 SD 11x15 font
  static const uint16_t text5_0[15] = {
    3840, 8064, 12480, 12480, 24672, 24672, 24672, 24672, 24672, 24672, 24672, 28896, 12480, 8064, 3840
  };
  static const uint16_t text5_1[15] = { 768, 768, 1792, 16128, 15104, 768, 768, 768, 768, 768, 768, 768, 768, 768, 768 };
  static const uint16_t text5_2[15] = { 8064, 16320, 28896, 24672, 96, 96, 192, 192, 896, 1536, 3072, 6144, 12288, 32736, 32736 };
  static const uint16_t text5_3[15] = { 7936, 16256, 29120, 24768, 192, 448, 1920, 1984, 192, 96, 96, 24672, 28896, 16320, 7936 };
  static const uint16_t text5_4[15] = { 384, 896, 1920, 1920, 3456, 6528, 12672, 24960, 49536, 65504, 65504, 384, 384, 384, 384 };
  static const uint16_t text5_5[15] = { 8128, 8128, 6144, 6144, 12288, 16256, 16320, 12512, 96, 96, 96, 24672, 28864, 16320, 7936 };
  static const uint16_t text5_6[15] = {
    3968, 8128, 14560, 12384, 24576, 28544, 32704, 28864, 24672, 24672, 24672, 12384, 14528, 8128, 3840
  };
  static const uint16_t text5_7[15] = { 32736, 32736, 96, 192, 192, 384, 384, 768, 768, 1536, 1536, 1536, 3072, 3072, 6144 };
  static const uint16_t text5_8[15] = { 3840, 8064, 12480, 12480, 12480, 6528, 3840, 8064, 14784, 24672, 24672, 24672, 28896, 16320, 8064 };
  static const uint16_t text5_9[15] = { 3840, 16256, 12736, 24768, 24672, 24672, 24672, 12512, 16352, 8032, 96, 24768, 29120, 16256, 7936 };
  static const uint8_t text5_bits[10] = { 66, 38, 63, 65, 63, 68, 77, 46, 74, 77 };

  // IMX307 SD 8x12 font
  static const uint8_t text7_0[12] = { 0x3c, 0x7e, 0x66, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xc3, 0xe6, 0x7e, 0x3c };
  static const uint8_t text7_1[12] = { 0x0c, 0x1c, 0x7c, 0x7c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c };
  static const uint8_t text7_2[12] = { 0x1c, 0x3e, 0x66, 0x66, 0x06, 0x0e, 0x0c, 0x18, 0x30, 0x70, 0x7e, 0x7e };
  static const uint8_t text7_3[12] = { 0x3c, 0x7e, 0x66, 0x06, 0x1c, 0x1e, 0x03, 0x03, 0x63, 0x63, 0x3e, 0x1c };
  static const uint8_t text7_4[12] = { 0x06, 0x0e, 0x0e, 0x1e, 0x36, 0x76, 0x66, 0xff, 0xff, 0x06, 0x06, 0x06 };
  static const uint8_t text7_5[12] = { 0x7e, 0x7e, 0x60, 0x60, 0x7c, 0x7e, 0x67, 0x03, 0x63, 0x67, 0x3e, 0x1c };
  static const uint8_t text7_6[12] = { 0x1e, 0x3f, 0x33, 0x60, 0x7c, 0x7e, 0x63, 0x63, 0x63, 0x73, 0x3e, 0x1c };
  static const uint8_t text7_7[12] = { 0x7f, 0x7f, 0x03, 0x06, 0x06, 0x0c, 0x0c, 0x0c, 0x18, 0x18, 0x18, 0x30 };
  static const uint8_t text7_8[12] = { 0x3e, 0x7f, 0x63, 0x63, 0x77, 0x3e, 0x3e, 0x73, 0x63, 0x63, 0x3e, 0x1c };
  static const uint8_t text7_9[12] = { 0x1c, 0x3e, 0x67, 0x63, 0x63, 0x63, 0x3f, 0x1f, 0x03, 0x66, 0x7e, 0x3c };
  static const uint8_t text7_bits[10] = { 53, 31, 42, 43, 47, 51, 52, 34, 57, 52 };

  // IMX307 HD 24x36 font
  static const uint32_t text8_0[36] = {
    0x03ffc000, 0x03ffc000, 0x03ffc000, 0x1ffff800, 0x1ffff800, 0x1ffff800, 0x1f81f800, 0x1f81f800, 0x1f81f800,
    0xfc003f00, 0xfc003f00, 0xfc003f00, 0xfc003f00, 0xfc003f00, 0xfc003f00, 0xfc003f00, 0xfc003f00, 0xfc003f00,
    0xfc003f00, 0xfc003f00, 0xfc003f00, 0xfc003f00, 0xfc003f00, 0xfc003f00, 0xfc003f00, 0xfc003f00, 0xfc003f00,
    0xff81f800, 0xff81f800, 0xff81f800, 0x1ffff800, 0x1ffff800, 0x1ffff800, 0x03ffc000, 0x03ffc000, 0x03ffc000,
  };
  static const uint32_t text8_1[36] = {
    0x000fc000, 0x000fc000, 0x000fc000, 0x007fc000, 0x007fc000, 0x007fc000, 0x1fffc000, 0x1fffc000, 0x1fffc000,
    0x1fffc000, 0x1fffc000, 0x1fffc000, 0x000fc000, 0x000fc000, 0x000fc000, 0x000fc000, 0x000fc000, 0x000fc000,
    0x000fc000, 0x000fc000, 0x000fc000, 0x000fc000, 0x000fc000, 0x000fc000, 0x000fc000, 0x000fc000, 0x000fc000,
    0x000fc000, 0x000fc000, 0x000fc000, 0x000fc000, 0x000fc000, 0x000fc000, 0x000fc000, 0x000fc000, 0x000fc000,
  };
  static const uint32_t text8_2[36] = {
    0x007fc000, 0x007fc000, 0x007fc000, 0x03fff800, 0x03fff800, 0x03fff800, 0x1f81f800, 0x1f81f800, 0x1f81f800,
    0x1f81f800, 0x1f81f800, 0x1f81f800, 0x0001f800, 0x0001f800, 0x0001f800, 0x000ff800, 0x000ff800, 0x000ff800,
    0x000fc000, 0x000fc000, 0x000fc000, 0x007e0000, 0x007e0000, 0x007e0000, 0x03f00000, 0x03f00000, 0x03f00000,
    0x1ff00000, 0x1ff00000, 0x1ff00000, 0x1ffff800, 0x1ffff800, 0x1ffff800, 0x1ffff800, 0x1ffff800, 0x1ffff800,
  };
  static const uint32_t text8_3[36] = {
    0x03ffc000, 0x03ffc000, 0x03ffc000, 0x1ffff800, 0x1ffff800, 0x1ffff800, 0x1f81f800, 0x1f81f800, 0x1f81f800,
    0x0001f800, 0x0001f800, 0x0001f800, 0x007fc000, 0x007fc000, 0x007fc000, 0x007ff800, 0x007ff800, 0x007ff800,
    0x00003f00, 0x00003f00, 0x00003f00, 0x00003f00, 0x00003f00, 0x00003f00, 0x1f803f00, 0x1f803f00, 0x1f803f00,
    0x1f803f00, 0x1f803f00, 0x1f803f00, 0x03fff800, 0x03fff800, 0x03fff800, 0x007fc000, 0x007fc000, 0x007fc000,
  };
  static const uint32_t text8_4[36] = {
    0x0001f800, 0x0001f800, 0x0001f800, 0x000ff800, 0x000ff800, 0x000ff800, 0x000ff800, 0x000ff800, 0x000ff800,
    0x007ff800, 0x007ff800, 0x007ff800, 0x03f1f800, 0x03f1f800, 0x03f1f800, 0x1ff1f800, 0x1ff1f800, 0x1ff1f800,
    0x1f81f800, 0x1f81f800, 0x1f81f800, 0xffffff00, 0xffffff00, 0xffffff00, 0xffffff00, 0xffffff00, 0xffffff00,
    0x0001f800, 0x0001f800, 0x0001f800, 0x0001f800, 0x0001f800, 0x0001f800, 0x0001f800, 0x0001f800, 0x0001f800,
  };
  static const uint32_t text8_5[36] = {
    0x1ffff800, 0x1ffff800, 0x1ffff800, 0x1ffff800, 0x1ffff800, 0x1ffff800, 0x1f800000, 0x1f800000, 0x1f800000,
    0x1f800000, 0x1f800000, 0x1f800000, 0x1fffc000, 0x1fffc000, 0x1fffc000, 0x1ffff800, 0x1ffff800, 0x1ffff800,
    0x1f81ff00, 0x1f81ff00, 0x1f81ff00, 0x00003f00, 0x00003f00, 0x00003f00, 0x1f803f00, 0x1f803f00, 0x1f803f00,
    0x1f81ff00, 0x1f81ff00, 0x1f81ff00, 0x03fff800, 0x03fff800, 0x03fff800, 0x007fc000, 0x007fc000, 0x007fc000,
  };
  static const uint32_t text8_6[36] = {
    0x007ff800, 0x007ff800, 0x007ff800, 0x03ffff00, 0x03ffff00, 0x03ffff00, 0x03f03f00, 0x03f03f00, 0x03f03f00,
    0x1f800000, 0x1f800000, 0x1f800000, 0x1fffc000, 0x1fffc000, 0x1fffc000, 0x1ffff800, 0x1ffff800, 0x1ffff800,
    0x1f803f00, 0x1f803f00, 0x1f803f00, 0x1f803f00, 0x1f803f00, 0x1f803f00, 0x1f803f00, 0x1f803f00, 0x1f803f00,
    0x1ff03f00, 0x1ff03f00, 0x1ff03f00, 0x03fff800, 0x03fff800, 0x03fff800, 0x007fc000, 0x007fc000, 0x007fc000,
  };
  static const uint32_t text8_7[36] = {
    0x1fffff00, 0x1fffff00, 0x1fffff00, 0x1fffff00, 0x1fffff00, 0x1fffff00, 0x00003f00, 0x00003f00, 0x00003f00,
    0x0001f800, 0x0001f800, 0x0001f800, 0x0001f800, 0x0001f800, 0x0001f800, 0x000fc000, 0x000fc000, 0x000fc000,
    0x000fc000, 0x000fc000, 0x000fc000, 0x000fc000, 0x000fc000, 0x000fc000, 0x007e0000, 0x007e0000, 0x007e0000,
    0x007e0000, 0x007e0000, 0x007e0000, 0x007e0000, 0x007e0000, 0x007e0000, 0x03f00000, 0x03f00000, 0x03f00000,
  };
  static const uint32_t text8_8[36] = {
    0x03fff800, 0x03fff800, 0x03fff800, 0x1fffff00, 0x1fffff00, 0x1fffff00, 0x1f803f00, 0x1f803f00, 0x1f803f00,
    0x1f803f00, 0x1f803f00, 0x1f803f00, 0x1ff1ff00, 0x1ff1ff00, 0x1ff1ff00, 0x03fff800, 0x03fff800, 0x03fff800,
    0x03fff800, 0x03fff800, 0x03fff800, 0x1ff03f00, 0x1ff03f00, 0x1ff03f00, 0x1f803f00, 0x1f803f00, 0x1f803f00,
    0x1f803f00, 0x1f803f00, 0x1f803f00, 0x03fff800, 0x03fff800, 0x03fff800, 0x007fc000, 0x007fc000, 0x007fc000,
  };
  static const uint32_t text8_9[36] = {
    0x007fc000, 0x007fc000, 0x007fc000, 0x03fff800, 0x03fff800, 0x03fff800, 0x1f81ff00, 0x1f81ff00, 0x1f81ff00,
    0x1f803f00, 0x1f803f00, 0x1f803f00, 0x1f803f00, 0x1f803f00, 0x1f803f00, 0x1f803f00, 0x1f803f00, 0x1f803f00,
    0x03ffff00, 0x03ffff00, 0x03ffff00, 0x007fff00, 0x007fff00, 0x007fff00, 0x00003f00, 0x00003f00, 0x00003f00,
    0x1f81f800, 0x1f81f800, 0x1f81f800, 0x1ffff800, 0x1ffff800, 0x1ffff800, 0x03ffc000, 0x03ffc000, 0x03ffc000,
  };
  static const uint16_t text8_bits[10] = { 53 * 9, 31 * 9, 42 * 9, 43 * 9, 47 * 9, 51 * 9, 52 * 9, 34 * 9, 57 * 9, 52 * 9 };

  // IMX307 HD 16x24 font
  static const uint16_t text9_0[24] = { 0x0ff0, 0x0ff0, 0x3ffc, 0x3ffc, 0x3c3c, 0x3c3c, 0xf00f, 0xf00f, 0xf00f, 0xf00f, 0xf00f, 0xf00f,
                                        0xf00f, 0xf00f, 0xf00f, 0xf00f, 0xf00f, 0xf00f, 0xfc3c, 0xfc3c, 0x3ffc, 0x3ffc, 0x0ff0, 0x0ff0 };
  static const uint16_t text9_1[24] = { 0x00f0, 0x00f0, 0x03f0, 0x03f0, 0x3ff0, 0x3ff0, 0x3ff0, 0x3ff0, 0x00f0, 0x00f0, 0x00f0, 0x00f0,
                                        0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0 };
  static const uint16_t text9_2[24] = { 0x03f0, 0x03f0, 0x0ffc, 0x0ffc, 0x3c3c, 0x3c3c, 0x3c3c, 0x3c3c, 0x003c, 0x003c, 0x00fc, 0x00fc,
                                        0x00f0, 0x00f0, 0x03c0, 0x03c0, 0x0f00, 0x0f00, 0x3f00, 0x3f00, 0x3ffc, 0x3ffc, 0x3ffc, 0x3ffc };
  static const uint16_t text9_3[24] = { 0x0ff0, 0x0ff0, 0x3ffc, 0x3ffc, 0x3c3c, 0x3c3c, 0x003c, 0x003c, 0x03f0, 0x03f0, 0x03fc, 0x03fc,
                                        0x000f, 0x000f, 0x000f, 0x000f, 0x3c0f, 0x3c0f, 0x3c0f, 0x3c0f, 0x0ffc, 0x0ffc, 0x03f0, 0x03f0 };
  static const uint16_t text9_4[24] = { 0x003c, 0x003c, 0x00fc, 0x00fc, 0x00fc, 0x00fc, 0x03fc, 0x03fc, 0x0f3c, 0x0f3c, 0x3f3c, 0x3f3c,
                                        0x3c3c, 0x3c3c, 0xffff, 0xffff, 0xffff, 0xffff, 0x003c, 0x003c, 0x003c, 0x003c, 0x003c, 0x003c };
  static const uint16_t text9_5[24] = { 0x3ffc, 0x3ffc, 0x3ffc, 0x3ffc, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3ff0, 0x3ff0, 0x3ffc, 0x3ffc,
                                        0x3c3f, 0x3c3f, 0x000f, 0x000f, 0x3c0f, 0x3c0f, 0x3c3f, 0x3c3f, 0x0ffc, 0x0ffc, 0x03f0, 0x03f0 };
  static const uint16_t text9_6[24] = { 0x03fc, 0x03fc, 0x0fff, 0x0fff, 0x0f0f, 0x0f0f, 0x3c00, 0x3c00, 0x3ff0, 0x3ff0, 0x3ffc, 0x3ffc,
                                        0x3c0f, 0x3c0f, 0x3c0f, 0x3c0f, 0x3c0f, 0x3c0f, 0x3f0f, 0x3f0f, 0x0ffc, 0x0ffc, 0x03f0, 0x03f0 };
  static const uint16_t text9_7[24] = { 0x3fff, 0x3fff, 0x3fff, 0x3fff, 0x000f, 0x000f, 0x003c, 0x003c, 0x003c, 0x003c, 0x00f0, 0x00f0,
                                        0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x03c0, 0x03c0, 0x03c0, 0x03c0, 0x03c0, 0x03c0, 0x0f00, 0x0f00 };
  static const uint16_t text9_8[24] = { 0x0ffc, 0x0ffc, 0x3fff, 0x3fff, 0x3c0f, 0x3c0f, 0x3c0f, 0x3c0f, 0x3f3f, 0x3f3f, 0x0ffc, 0x0ffc,
                                        0x0ffc, 0x0ffc, 0x3f0f, 0x3f0f, 0x3c0f, 0x3c0f, 0x3c0f, 0x3c0f, 0x0ffc, 0x0ffc, 0x03f0, 0x03f0 };
  static const uint16_t text9_9[24] = { 0x03f0, 0x03f0, 0x0ffc, 0x0ffc, 0x3c3f, 0x3c3f, 0x3c0f, 0x3c0f, 0x3c0f, 0x3c0f, 0x3c0f, 0x3c0f,
                                        0x0fff, 0x0fff, 0x03ff, 0x03ff, 0x000f, 0x000f, 0x3c3c, 0x3c3c, 0x3ffc, 0x3ffc, 0x0ff0, 0x0ff0 };

  static const uint16_t text9_bits[10] = { 53 * 4, 31 * 4, 42 * 4, 43 * 4, 47 * 4, 51 * 4, 52 * 4, 34 * 4, 57 * 4, 52 * 4 };

  static const uint8_t *text_numbers[10] = { text_0, text_1, text_2, text_3, text_4, text_5, text_6, text_7, text_8, text_9 };
  static const uint8_t *text2_numbers[10] = { text2_0, text2_1, text2_2, text2_3, text2_4, text2_5, text2_6, text2_7, text2_8, text2_9 };
  static const uint8_t *text3_numbers[10] = { text3_0, text3_1, text3_2, text3_3, text3_4, text3_5, text3_6, text3_7, text3_8, text3_9 };
  static const uint8_t **text4_numbers = text_numbers;
  static const uint16_t *text5_numbers[10] = { text5_0, text5_1, text5_2, text5_3, text5_4, text5_5, text5_6, text5_7, text5_8, text5_9 };
  static const uint8_t *text7_numbers[10] = { text7_0, text7_1, text7_2, text7_3, text7_4, text7_5, text7_6, text7_7, text7_8, text7_9 };
  static const uint32_t *text8_numbers[10] = { text8_0, text8_1, text8_2, text8_3, text8_4, text8_5, text8_6, text8_7, text8_8, text8_9 };
  static const uint16_t *text9_numbers[10] = { text9_0, text9_1, text9_2, text9_3, text9_4, text9_5, text9_6, text9_7, text9_8, text9_9 };

  // Vivotek IP8172 8x11 number positions
  static const int16_t text_xpos[12] = { 26, 36, 51, 61, 76, 86, 101, 111, 127, 137, 153, 163 };
  static const int16_t text_ypos = -18;

  // Vivotek IP9171 8x16 number positions
  static const int16_t text2_xpos[12] = { 37, 53, 85, 101, 133, 149, 173, 189, 217, 233, 261, 277 };
  static const int16_t text2_ypos = -42;  // Early firmwares: -58

  // Vivotek IP8151 8x11 number positions
  static const int16_t text3_xpos[12] = { 26, 34, 49, 58, 73, 82, 99, 108, 120, 129, 141, 150 };
  static const int16_t text3_ypos = 4;

  // Vivotek IP816A-HP 8x11 number positions
  static const int16_t *text4_xpos = text_xpos;
  static const int16_t text4_ypos = -15;

  // IMX291SD 11x15 number positions
  static const int16_t text5_xpos[12] = { 29, 43, 65, 79, 101, 115, 137, 151, 173, 187, 209, 223 };
  static const int16_t text5_ypos = -28;

  // IMX291HD 22x30 number positions
  static const int16_t text6_xpos[12] = { 29, 43, 65, 79, 101, 115, 137, 151, 173, 187, 209, 223 };
  static const int16_t text6_ypos = -26;

  // IMX307SD 8x12 number positions
  static const int16_t text7_xpos[12] = { 18, 27, 42, 51, 66, 75, 89, 98, 112, 121, 135, 144 };  // 1080P
  static const int16_t text7_ypos = -19;
  // static const int16_t text7_ypos = -23;

  // IMX307HD 24x36 number positions
  static const int16_t text8_xpos[12] = { 54, 81, 126, 153, 198, 225, 267, 294, 336, 363, 405, 432 };
  static const int16_t text8_ypos = -51;

  // IMX307HD 16x24 number positions
  static const int16_t text9_xpos[12] = { 36, 54, 84, 102, 132, 150, 178, 196, 224, 242, 270, 288 };
  static const int16_t text9_ypos = -36;

  const static uint8_t max_numbers[12] = { 9, 9, 1, 9, 3, 9, 2, 9, 5, 9, 5, 9 };
  char text[20];
  uint8_t numbers[12];
  uint8_t numbers2[12];
  uint8_t numbers3[12];
  uint8_t numbers4[12];
  uint8_t numbers5[12];
  uint8_t numbers6[12];
  uint8_t numbers7[12];
  uint8_t numbers8[12];
  uint8_t numbers9[12];

  int tot_err = 0;
  int tot_err2 = 0;
  int tot_err3 = 0;
  int tot_err4 = 0;
  int tot_err5 = 0;
  int tot_err6 = 0;
  int tot_err7 = 0;
  int tot_err8 = 0;
  int tot_err9 = 0;
  for (unsigned int i = 0; i < sizeof(numbers); i++) {
    uint8_t number[11];
    unsigned char *p = img + (height + text_ypos) * width + text_xpos[i];
    for (int k = 0; k < 11; k++) {
      uint8_t bit = 128;
      uint8_t n = 0;
      for (int j = 0; j < 8; j++) {
        n += bit * (*p++ > 192);
        bit >>= 1;
      }
      number[k] = n;
      p += width - 8;
    }

    uint8_t number2[16];
    p = img + (height + text2_ypos) * width + text2_xpos[i];
    for (int k = 0; k < 16; k++) {
      uint8_t bit = 128;
      uint8_t n = 0;
      for (int j = 0; j < 8; j++) {
        n += bit * (*p++ > 192);
        bit >>= 1;
      }
      number2[k] = n;
      p += width - 8;
    }

    uint8_t number3[11];
    p = img + ((text3_ypos < 0 ? height : 0) + text3_ypos) * width + text3_xpos[i];
    for (int k = 0; k < 11; k++) {
      uint8_t bit = 128;
      uint8_t n = 0;
      for (int j = 0; j < 8; j++) {
        n += bit * (*p++ > 192);
        bit >>= 1;
      }
      number3[k] = n;
      p += width - 8;
    }

    uint8_t number4[11];
    p = img + (height + text4_ypos) * width + text4_xpos[i];
    for (int k = 0; k < 11; k++) {
      uint8_t bit = 128;
      uint8_t n = 0;
      for (int j = 0; j < 8; j++) {
        n += bit * (*p++ > 192);
        bit >>= 1;
      }
      number4[k] = n;
      p += width - 8;
    }

    uint16_t number5[15];
    p = img + (height + text5_ypos) * width + text5_xpos[i];
    for (int k = 0; k < 15; k++) {
      uint16_t bit = 1 << 15;
      uint16_t n = 0;
      for (int j = 0; j < 12; j++) {
        // printf("%c", *p > 192 ? '*' : ' ');
        n += bit * (*p++ > 192);
        bit >>= 1;
      }
      // printf("\n");
      number5[k] = n;
      p += width - 12;
    }

    uint16_t number6[15];
    p = img2 + (height / 2 + text6_ypos) * width / 2 + text6_xpos[i];
    for (int k = 0; k < 15; k++) {
      uint16_t bit = 1 << 15;
      uint16_t n = 0;
      for (int j = 0; j < 12; j++) {
        // printf("%c", *p > 192 ? '*' : ' ');
        n += bit * (*p++ > 192);
        bit >>= 1;
      }
      // printf("\n");
      number6[k] = n;
      p += width / 2 - 12;
    }

    uint8_t number7[12];
    p = img + (height + text7_ypos) * width + text7_xpos[i];
    for (int k = 0; k < 12; k++) {
      uint8_t bit = 128;
      uint8_t n = 0;
      for (int j = 0; j < 8; j++) {
        // printf("%c", *p > 192 ? '*' : ' ');
        n += bit * (*p++ > 192);
        bit >>= 1;
      }
      // printf("\n");
      number7[k] = n;
      p += width - 8;
    }

    uint32_t number8[36];
    p = img + (height + text8_ypos) * width + text8_xpos[i];
    for (int k = 0; k < 36; k++) {
      uint32_t bit = 1U << 31;
      uint32_t n = 0;
      for (int j = 0; j < 24; j++) {
        // printf("%c", *p > 192 ? '*' : ' ');
        n += bit * (*p++ > 192);
        bit >>= 1;
      }
      // printf("\n");
      number8[k] = n;
      p += width - 24;
    }

    uint32_t number9[24];
    p = img + (height + text9_ypos) * width + text9_xpos[i];
    for (int k = 0; k < 24; k++) {
      uint16_t bit = 1U << 15;
      uint16_t n = 0;
      for (int j = 0; j < 16; j++) {
        // printf("%c", *p > 192 ? '*' : ' ');
        n += bit * (*p++ > 192);
        bit >>= 1;
      }
      // printf("\n");
      number9[k] = n;
      p += width - 16;
    }

    if (camera == ALL || camera == IP8172) {
      int best_number = -1;
      int best_bitcount = 8 * 16;
      for (int k = 0; k <= max_numbers[i]; k++) {
        int bitcount = text_bits[k];
        // IP8172
        for (int j = 0; j < 11; j++) {
          bitcount -= __builtin_popcount(number[j] & text_numbers[k][j]);
          bitcount += __builtin_popcount(number[j] & ~text_numbers[k][j]);
        }
        bitcount = bitcount * 256 / 11 / 8;
        if (bitcount < best_bitcount) {
          best_number = k;
          best_bitcount = bitcount;
        }
        tot_err += best_bitcount;
      }
      numbers[i] = best_number + 0x30;
    } else
      tot_err = 1 << 30;

    if (camera == ALL || camera == IP9171) {
      int best_number = -1;
      int best_bitcount = 8 * 16;
      for (int k = 0; k <= max_numbers[i]; k++) {
        int bitcount = text_bits[k];
        // IP 9171
        bitcount = text2_bits[k];
        for (int j = 0; j < 16; j++) {
          bitcount -= __builtin_popcount(number2[j] & text2_numbers[k][j]);
          bitcount += __builtin_popcount(number2[j] & ~text2_numbers[k][j]);
        }
        bitcount = bitcount * 256 / 16 / 8;
        if (bitcount <= best_bitcount) {
          best_number = k;
          best_bitcount = bitcount;
        }
        tot_err2 += best_bitcount;
      }
      numbers2[i] = best_number + 0x30;
    } else
      tot_err2 = 1 << 30;

    if (camera == ALL || camera == IP8151) {
      int best_number = -1;
      int best_bitcount = 8 * 16;
      for (int k = 0; k <= max_numbers[i]; k++) {
        int bitcount = text3_bits[k];
        // IP8151
        for (int j = 0; j < 11; j++) {
          bitcount -= __builtin_popcount(number3[j] & text3_numbers[k][j]);
          bitcount += __builtin_popcount(number3[j] & ~text3_numbers[k][j]);
        }
        bitcount = bitcount * 256 / 11 / 8;
        if (bitcount <= best_bitcount) {
          best_number = k;
          best_bitcount = bitcount;
        }
        tot_err3 += best_bitcount;
      }
      numbers3[i] = best_number + 0x30;
    } else
      tot_err3 = 1 << 30;

    if (camera == ALL || camera == IP816A) {
      int best_number = -1;
      int best_bitcount = 8 * 16;
      for (int k = 0; k <= max_numbers[i]; k++) {
        int bitcount = text4_bits[k];
        // IP816A-HP
        for (int j = 0; j < 11; j++) {
          bitcount -= __builtin_popcount(number4[j] & text4_numbers[k][j]);
          bitcount += __builtin_popcount(number4[j] & ~text4_numbers[k][j]);
        }
        bitcount = bitcount * 256 / 11 / 8;
        if (bitcount <= best_bitcount) {
          best_number = k;
          best_bitcount = bitcount;
        }
        tot_err4 += best_bitcount;
      }
      numbers4[i] = best_number + 0x30;
    } else
      tot_err4 = 1 << 30;

    if (camera == ALL || camera == IMX291 || camera == IMX291SD) {
      int best_number = -1;
      int best_bitcount = 16 * 16;
      for (int k = 0; k <= max_numbers[i]; k++) {
        int bitcount = text5_bits[k];
        // IMX291SD
        for (int j = 0; j < 15; j++) {
          bitcount -= __builtin_popcount(number5[j] & text5_numbers[k][j]);
          bitcount += __builtin_popcount(number5[j] & ~text5_numbers[k][j]);
        }
        bitcount = bitcount * 256 / 15 / 11;
        if (bitcount <= best_bitcount) {
          best_number = k;
          best_bitcount = bitcount;
        }
        tot_err5 += best_bitcount;
      }
      numbers5[i] = best_number + 0x30;
    } else
      tot_err5 = 1 << 30;

    if (camera == ALL || camera == IMX291 || camera == IMX291HD) {
      int best_number = -1;
      int best_bitcount = 8 * 12;
      for (int k = 0; k <= max_numbers[i]; k++) {
        int bitcount = text5_bits[k];
        // IMX291HD
        for (int j = 0; j < 15; j++) {
          bitcount -= __builtin_popcount(number6[j] & text5_numbers[k][j]);
          bitcount += __builtin_popcount(number6[j] & ~text5_numbers[k][j]);
        }
        bitcount = bitcount * 256 / 15 / 11;
        if (bitcount <= best_bitcount) {
          best_number = k;
          best_bitcount = bitcount;
        }
        tot_err6 += best_bitcount;
      }
      numbers6[i] = best_number + 0x30;
      // printf("%d: %d\n", best_number, best_bitcount);
    } else
      tot_err6 = 1 << 30;

    if (camera == ALL || camera == IMX307 || camera == IMX307SD) {
      int best_number = -1;
      int best_bitcount = 16 * 16;
      for (int k = 0; k <= max_numbers[i]; k++) {
        int bitcount = text7_bits[k];
        // IMX307SD
        for (int j = 0; j < 12; j++) {
          bitcount -= __builtin_popcount(number7[j] & text7_numbers[k][j]);
          bitcount += __builtin_popcount(number7[j] & ~text7_numbers[k][j]);
        }
        bitcount = bitcount * 256 / 12 / 8;
        if (bitcount <= best_bitcount) {
          best_number = k;
          best_bitcount = bitcount;
        }
        tot_err7 += best_bitcount;
      }
      numbers7[i] = best_number + 0x30;
      // printf("%d: %d\n", best_number, best_bitcount);
    } else
      tot_err7 = 1 << 30;

    if (camera == ALL || camera == IMX307 || camera == IMX307HD) {
      int best_number = -1;
      int best_bitcount = 24 * 32;
      for (int k = 0; k <= max_numbers[i]; k++) {
        int bitcount = text8_bits[k];
        // IMX307HD
        for (int j = 0; j < 36; j++) {
          bitcount -= __builtin_popcount(number8[j] & text8_numbers[k][j]);
          bitcount += __builtin_popcount(number8[j] & ~text8_numbers[k][j]);
        }
        bitcount = bitcount * 256 / 36 / 24;
        if (bitcount <= best_bitcount) {
          best_number = k;
          best_bitcount = bitcount;
        }
        tot_err8 += best_bitcount;
      }
      numbers8[i] = best_number + 0x30;
      // printf("%d: %d\n", best_number, best_bitcount);
    } else
      tot_err8 = 1 << 30;

    if (camera == ALL || camera == IMX307 || camera == IMX307HD) {
      int best_number = -1;
      int best_bitcount = 16 * 24;
      for (int k = 0; k <= max_numbers[i]; k++) {
        int bitcount = text9_bits[k];
        // IMX307HD
        for (int j = 0; j < 24; j++) {
          bitcount -= __builtin_popcount(number9[j] & text9_numbers[k][j]);
          bitcount += __builtin_popcount(number9[j] & ~text9_numbers[k][j]);
        }
        bitcount = bitcount * 256 / 24 / 16;
        if (bitcount <= best_bitcount) {
          best_number = k;
          best_bitcount = bitcount;
        }
        tot_err9 += best_bitcount;
      }
      numbers9[i] = best_number + 0x30;
      // printf("%d: %d\n", best_number, best_bitcount);
    } else
      tot_err9 = 1 << 30;
  }

  if (tot_err < tot_err2 && tot_err < tot_err3 && tot_err < tot_err4 && tot_err < tot_err5 && tot_err < tot_err6 && tot_err < tot_err7 &&
      tot_err < tot_err8 && tot_err < tot_err9)
    sprintf(text, "20%c%c-%c%c-%c%c %c%c:%c%c:%c%c", numbers[0], numbers[1], numbers[2], numbers[3], numbers[4], numbers[5], numbers[6],
            numbers[7], numbers[8], numbers[9], numbers[10], numbers[11]);
  else if (tot_err2 < tot_err3 && tot_err2 < tot_err4 && tot_err2 < tot_err5 && tot_err2 < tot_err6 && tot_err2 < tot_err7 &&
           tot_err2 < tot_err8 && tot_err2 < tot_err9)  // IP9171
    sprintf(text, "20%c%c-%c%c-%c%c %c%c:%c%c:%c%c", numbers2[0], numbers2[1], numbers2[2], numbers2[3], numbers2[4], numbers2[5],
            numbers2[6], numbers2[7], numbers2[8], numbers2[9], numbers2[10], numbers2[11]);
  else if (tot_err3 < tot_err4 && tot_err3 < tot_err5 && tot_err3 < tot_err6 && tot_err3 < tot_err7 && tot_err3 < tot_err8 &&
           tot_err3 < tot_err9)  // IP8151
    sprintf(text, "20%c%c-%c%c-%c%c %c%c:%c%c:%c%c", numbers3[0], numbers3[1], numbers3[2], numbers3[3], numbers3[4], numbers3[5],
            numbers3[6], numbers3[7], numbers3[8], numbers3[9], numbers3[10], numbers3[11]);
  else if (tot_err4 < tot_err5 && tot_err4 < tot_err6 && tot_err4 < tot_err7 && tot_err4 < tot_err8 && tot_err4 < tot_err9)  // IP816A-HP
    sprintf(text, "20%c%c-%c%c-%c%c %c%c:%c%c:%c%c", numbers4[0], numbers4[1], numbers4[2], numbers4[3], numbers4[4], numbers4[5],
            numbers4[6], numbers4[7], numbers4[8], numbers4[9], numbers4[10], numbers4[11]);
  else if (tot_err5 < tot_err6 && tot_err5 < tot_err7 && tot_err5 < tot_err8 && tot_err5 < tot_err9)  // IMX291SD
    sprintf(text, "20%c%c-%c%c-%c%c %c%c:%c%c:%c%c", numbers5[0], numbers5[1], numbers5[2], numbers5[3], numbers5[4], numbers5[5],
            numbers5[6], numbers5[7], numbers5[8], numbers5[9], numbers5[10], numbers5[11]);
  else if (tot_err6 < tot_err7 && tot_err6 < tot_err8 && tot_err6 < tot_err9)  // IMX291HD
    sprintf(text, "20%c%c-%c%c-%c%c %c%c:%c%c:%c%c", numbers6[0], numbers6[1], numbers6[2], numbers6[3], numbers6[4], numbers6[5],
            numbers6[6], numbers6[7], numbers6[8], numbers6[9], numbers6[10], numbers6[11]);
  else if (tot_err7 < tot_err8 && tot_err7 < tot_err9)  // IMX291SD
    sprintf(text, "20%c%c-%c%c-%c%c %c%c:%c%c:%c%c", numbers7[0], numbers7[1], numbers7[2], numbers7[3], numbers7[4], numbers7[5],
            numbers7[6], numbers7[7], numbers7[8], numbers7[9], numbers7[10], numbers7[11]);
  else if (tot_err8 < tot_err9)  // IMX291SD
    sprintf(text, "20%c%c-%c%c-%c%c %c%c:%c%c:%c%c", numbers8[0], numbers8[1], numbers8[2], numbers8[3], numbers8[4], numbers8[5],
            numbers8[6], numbers8[7], numbers8[8], numbers8[9], numbers8[10], numbers8[11]);
  else  // IMX307HD
    sprintf(text, "20%c%c-%c%c-%c%c %c%c:%c%c:%c%c", numbers9[0], numbers9[1], numbers9[2], numbers9[3], numbers9[4], numbers9[5],
            numbers9[6], numbers9[7], numbers9[8], numbers9[9], numbers9[10], numbers9[11]);

  struct tm tm;
  if (!strptime(text, "%Y-%m-%d %H:%M:%S", &tm)) return 0;

  if (indicator) {
    printf("\r%s", text);
    fflush(stdout);
  }

  tm.tm_isdst = -1;

  return mktime(&tm);
}

static uint8_t *loadjpeg(const char *filename, unsigned int *width, unsigned int *height) {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr cerrmgr;
  jpegErrorManager jerr;
  uint8_t *dest;
  JSAMPROW buffer = 0;
  FILE *f;

  if (!filename) return 0;

  f = fopen(filename, "r");
  if (!f) return 0;

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = jpegErrorExit;
  /* Establish the setjmp return context for my_error_exit to use. */
  if (setjmp(jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    jpeg_destroy_decompress(&cinfo);
    fclose(f);
    if (buffer) free(buffer);
    return 0;
  }

  // cinfo.err = jpeg_std_error(&cerrmgr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, f);
  jpeg_read_header(&cinfo, 1);
  cinfo.out_color_space = JCS_YCbCr;
  jpeg_start_decompress(&cinfo);

  *width = cinfo.output_width;
  *height = cinfo.output_height;

  dest = (uint8_t *)aligned_malloc(cinfo.output_width * cinfo.output_height * 1.5, 16);

  if (!dest) {
    jpeg_destroy_decompress(&cinfo);
    fclose(f);
    return 0;
  }

  buffer = (JSAMPROW)malloc(sizeof(JSAMPLE) * cinfo.output_width * cinfo.output_components);

  if (buffer) {
    while (cinfo.output_scanline < cinfo.output_height) {
      jpeg_read_scanlines(&cinfo, &buffer, 1);
      for (unsigned int i = 0; i < *width; i++) dest[(cinfo.output_scanline - 1) * *width + i] = (buffer[i * 3 + 0]);
    }
    memset(dest + *width * *height, 128, *width * *height / 2);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(f);
  if (buffer) free(buffer);
  return dest;
}

static int find_centre(uint8_t *b, int x, int y, int w, int h, int *cx, int *cy, int *cw, int *size) {
  const int radius = w / 128;
  int r = 0;
  int m = 0;
  int s = 0;
  for (int i = max(0, x - radius); i < min(w, x + radius); i++)
    for (int j = max(0, y - radius); j < min(h, y + radius); j++) {
      int v = max(0, b[i + j * w] - 4);
      m = max(m, v);
      r += v;
      s += v > 0;
      *cx += i * v;
      *cy += j * v;
      (*cw) += v;
    }
  *size = s;
  return r * m / 256;
}

// Check if input list is ordered.
static int ordered(int n, const double *a) {
  if (n < 2) return 1;
  int sign = a[0] > a[1] ? -1 : 1;
  for (int i = 2; i < n; i++)
    if (a[i] * sign < a[i - 1] * sign) return 0;
  return 1;
}

static double dist(double x1, double y1, double x2, double y2) { return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)); }

// Input: x, y.  Output: fitted y = ax + b and r (correlation coefficient)
static double linreg(int n, const double *x, const double *y, double *a, double *b, double *r, double *l, double *dr) {
  double rx[n], ry[n];

  // Find the line minimising least squares
  double sumx = 0, sumx2 = 0, sumxy = 0, sumy = 0, sumy2 = 0;

  for (int i = 0; i < n; i++) {
    sumx += x[i];
    sumx2 += x[i] * x[i];
    sumxy += x[i] * y[i];
    sumy += y[i];
  }

  double denom = n * sumx2 - sumx * sumx;

  if (denom == 0)
    *a = *b = 0;
  else {
    *a = (n * sumxy - sumx * sumy) / denom;
    *b = (sumy * sumx2 - sumx * sumxy) / denom;
  }

  // Rotate and shift the line to cross origo at 45 degrees.
  double angle = M_PI / 4 - atan2(*a, 1);
  for (int i = 0; i < n; i++) {
    rx[i] = cos(angle) * x[i] - sin(angle) * (y[i] - *b);
    ry[i] = sin(angle) * x[i] + cos(angle) * (y[i] - *b);
  }

  // Now we can easily check order and we'll have consistent correlation
  if (!ordered(n, rx) && !ordered(n, ry)) {
    *dr = *r = 0;
    return 0;
  }

  // Check spacing
  double avgdist = dist(x[0], y[0], x[n - 1], y[n - 1]) / (n - 1);
  double sd = 0;
  for (int i = 1; i < n; i++) {
    double d = dist(x[i], y[i], x[i - 1], y[i - 1]) / avgdist;
    sd += d * d;
  }
  double distcorr = sqrt(sqrt((n - 1) / sd));

  // Find the correlation coefficient
  sumx = sumx2 = sumxy = sumy = 0;

  for (int i = 0; i < n; i++) {
    sumx += rx[i];
    sumx2 += rx[i] * rx[i];
    sumxy += rx[i] * ry[i];
    sumy += ry[i];
    sumy2 += ry[i] * ry[i];
  }

  *r = fabs((sumxy - sumx * sumy / n) / sqrt((sumx2 - sumx * sumx / n) * (sumy2 - sumy * sumy / n)));
  *l = avgdist;
  *dr = distcorr;

  return *r;
}

static double point_to_line_distance(double x, double y, double a, double b) { return fabs(y - b - a * x) / sqrt(a * a + 1); }

// Input: x, y.  Output: fitted y = ax + b and r (correlation coefficient)
static double simple_linreg(int n, const point *p, double *a, double *b, int comp = 0) {
  // Find the line minimising least squares
  double sumx = 0, sumx2 = 0, sumxy = 0, sumy = 0, sumy2 = 0;

  if (comp == 0)
    for (int i = 0; i < n; i++) {
      sumx += p[i].gnomx;
      sumx2 += p[i].gnomx * p[i].gnomx;
      sumxy += p[i].gnomx * p[i].gnomy;
      sumy += p[i].gnomy;
    }
  else if (comp == 1)
    for (int i = 0; i < n; i++) {
      sumx += p[i].gnomx;
      sumx2 += p[i].gnomx * p[i].gnomx;
      sumxy += p[i].gnomx * p[i].time;
      sumy += p[i].time;
    }
  else
    for (int i = 0; i < n; i++) {
      sumx += p[i].gnomy;
      sumx2 += p[i].gnomy * p[i].gnomy;
      sumxy += p[i].gnomy * p[i].time;
      sumy += p[i].time;
    }

  double denom = n * sumx2 - sumx * sumx;

  if (denom == 0)
    *a = *b = 0;
  else {
    *a = (n * sumxy - sumx * sumy) / denom;
    *b = (sumy * sumx2 - sumx * sumxy) / denom;
  }

  // Find the correlation coefficient
  sumx = sumx2 = sumxy = sumy = 0;

  if (comp == 0)
    for (int i = 0; i < n; i++) {
      sumx += p[i].gnomx;
      sumx2 += p[i].gnomx * p[i].gnomx;
      sumxy += p[i].gnomx * p[i].gnomy;
      sumy += p[i].gnomy;
      sumy2 += p[i].gnomy * p[i].gnomy;
    }
  else if (comp == 1)
    for (int i = 0; i < n; i++) {
      sumx += p[i].gnomx;
      sumx2 += p[i].gnomx * p[i].gnomx;
      sumxy += p[i].gnomx * p[i].time;
      sumy += p[i].time;
      sumy2 += p[i].time * p[i].time;
    }
  else
    for (int i = 0; i < n; i++) {
      sumx += p[i].gnomy;
      sumx2 += p[i].gnomy * p[i].gnomy;
      sumxy += p[i].gnomy * p[i].time;
      sumy += p[i].time;
      sumy2 += p[i].time * p[i].time;
    }

  return fabs((sumxy - sumx * sumy / n) / sqrt((sumx2 - sumx * sumx / n) * (sumy2 - sumy * sumy / n)));
}

// Generic algorithm from https://en.wikipedia.org/wiki/Random_sample_consensus
// Return k and b in f(x) = k*x + b and the correlation.  n is the minimum
// number of data points.
double ransac(int size, const point *p, point *best, int *best_num, double *k, double *b, double fps, double t = 0.01, int n = 3,
              int maxit = 100000) {
  double best_corr = 0, best_model_k = 0, best_model_b = 0;
  int best_model_size = n;
  srand(0);

  for (int it = 0; it++ < maxit;) {
    double t2 = fmod((double)rand() / 10000, t);

    bool visited[size];
    memset(visited, false, sizeof(visited));
    point maybe_inliers[n];
    int maybe_inliers_size;

    // pick n randomly selected values from data
    for (maybe_inliers_size = 0; maybe_inliers_size < n;) {
      int i = rand() % size;
      if (visited[i]) continue;
      int same_time = 0;
      for (int j = 0; j < maybe_inliers_size && !same_time; j++) same_time |= p[i].time == maybe_inliers[j].time;
      if (same_time) continue;
      maybe_inliers[maybe_inliers_size++] = p[i];
      visited[i] = true;
    }

    // model parameters fitted to maybe_inliers
    double maybe_model_k0, maybe_model_b0, maybe_model_k1, maybe_model_b1, maybe_model_k2, maybe_model_b2;
    double r = simple_linreg(maybe_inliers_size, maybe_inliers, &maybe_model_k0, &maybe_model_b0, 0) *
               simple_linreg(maybe_inliers_size, maybe_inliers, &maybe_model_k1, &maybe_model_b1, 1) *
               simple_linreg(maybe_inliers_size, maybe_inliers, &maybe_model_k2, &maybe_model_b2, 2);
    if (r < (1 - t2)) continue;

    point also_inliers[size];
    int also_inliers_size = 0;

    // add points close to the line
    for (int i = 0; i < size; i++)
      if (!visited[i] && point_to_line_distance(p[i].gnomx, p[i].gnomy, maybe_model_k0, maybe_model_b0) < t2 &&
          point_to_line_distance(p[i].gnomx, p[i].time, maybe_model_k1, maybe_model_b1) < 100 * t2 / fps &&
          point_to_line_distance(p[i].gnomy, p[i].time, maybe_model_k2, maybe_model_b2) < 100 * t2 / fps)
        also_inliers[also_inliers_size++] = p[i];

    // parameters fitted to all points in maybe_inliers and also_inliers
    for (int z = 0; z < maybe_inliers_size; z++) also_inliers[also_inliers_size++] = maybe_inliers[z];

    double better_model_k = 0, better_model_b = 0;

    // measure of how well model fits these points
    double dummy;
    r = simple_linreg(also_inliers_size, also_inliers, &better_model_k, &better_model_b, 0) *
        simple_linreg(also_inliers_size, also_inliers, &dummy, &dummy, 1) *
        simple_linreg(also_inliers_size, also_inliers, &dummy, &dummy, 2);

    // Too perfect match indicates multiple duplicate points
    if (r > 0.999999) r = 0;

    r -= 0.01;  //  Don't get stuck at "too" perfect solutions

    // Best so far?  Correct correlation coefficient for number of points.
    if (r > pow(best_corr, (double)also_inliers_size / best_model_size)) {
      best_model_size = also_inliers_size;
      best_model_k = better_model_k;
      best_model_b = better_model_b;
      best_corr = r;
      memcpy(best, also_inliers, sizeof(*also_inliers) * also_inliers_size);
    }
  }

  *k = best_model_k;
  *b = best_model_b;
  *best_num = best_model_size;
  return best_corr;
}

static double midpoint(double az1, double alt1, double az2, double alt2, double *az3, double *alt3) {
  double x1 = az1 * M_PI / 180;
  double x2 = az2 * M_PI / 180;
  double y1 = alt1 * M_PI / 180;
  double y2 = alt2 * M_PI / 180;
  double bx = cos(y2) * cos(x2 - x1);
  double by = cos(y2) * sin(x2 - x1);
  double y3 = atan2(sin(y1) + sin(y2), sqrt((cos(y1) + bx) * (cos(y1) + bx) + by * by));
  double x3 = x1 + atan2(by, cos(y1) + bx);
  double a = sin((y2 - y1) / 2) * sin((y2 - y1) / 2) + cos(y1) * cos(y2) * sin((x2 - x1) / 2) * sin((x2 - x1) / 2);
  double c = 2 * atan2(sqrt(a), sqrt(1 - a));
  *az3 = x3 * 180 / M_PI;
  *alt3 = y3 * 180 / M_PI;
  return c * 180 / M_PI;
}

void saveevent(const config *config, const char *rp, time_t ts) {
  char tsbuf[25];
  char eventdir[strlen(config->eventdir) + 10 + 7];
  char eventfile[strlen(config->eventdir) + 35];
  char stderrfile[strlen(config->eventdir) + 35];

  struct tm *t = localtime(&ts);
  strftime(tsbuf, sizeof(tsbuf), "%Y%m%d/", t);
  snprintf(eventdir, sizeof(eventdir), "%s/%s", config->eventdir, tsbuf);
  struct stat st = { 0 };
  if (config->saveevent && stat(eventdir, &st) == -1) mkdir(eventdir, 0777);
  strftime(tsbuf, sizeof(tsbuf), "%Y%m%d/%H%M%S", t);
  snprintf(eventdir, sizeof(eventdir), "%s/%s", config->eventdir, tsbuf);
  if (config->saveevent && stat(eventdir, &st) == -1) mkdir(eventdir, 0777);

  snprintf(eventfile, sizeof(eventfile), "%s/event.txt", eventdir);
  FILE *f = config->saveevent ? fopen(eventfile, "w") : stdout;

  if (f) {
    fputs(rp, f);

    if (config->saveevent) debug(config->log, "Saving to %s\n", eventfile);

    if (config->saveevent) {
      fclose(f);
      if (config->execute) {
        int pid = fork();
        if (!pid) {
          snprintf(stderrfile, sizeof(stderrfile), "%s/stderr.txt", eventdir);
          int fd = open("/dev/null", O_RDWR);
          FILE *stderrf = fopen(stderrfile, "w");
          dup2(fd, STDIN_FILENO);
          dup2(fd, STDOUT_FILENO);
          dup2(stderrf ? fileno(stderrf) : fd, STDERR_FILENO);
          if (fd > STDERR_FILENO) close(fd);
          if (stderrf) fclose(stderrf);

          char *args[] = { config->execute, eventfile, 0 };
          signal(SIGCHLD, SIG_DFL);
          setpriority(PRIO_PROCESS, 0, 19);
          debug(config->log, "Execution of %s failed.  Error code %d\n", config->execute, execvp(config->execute, args));
          close_dec();
          exit(0);
        } else if (pid > 0)
          debug(config->log, "Executed %s %s as pid %d\n", config->execute, eventfile, pid);
      }
    }
  }
}

static int pointcmp(const void *p1, const void *p2) { return ((point *)p1)->time > ((point *)p2)->time; }

static void detect_trail2(record *history, unsigned int length, config *config, uint8_t *hashtab1, const uint8_t *hashtab2,
                          const uint8_t *hashtab3, int final) {
  int n = min(config->lookahead, length * 2);
  point *track = (point *)malloc(n * config->numspots * sizeof(*track));

  unsigned int c = 0;
  if (!track) return;

  // Track history
  double timestamp = 0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < config->numspots; j++)
      if (history[i].brightness[j] > 0) {
        if (timestamp == 0) timestamp = history[i].timestamp;
        track[c].x = history[i].x[j];
        track[c].y = history[i].y[j];
        track[c].az = history[i].az[j];
        track[c].alt = history[i].alt[j];
        track[c].gnomx = history[i].gnomx[j];
        track[c].gnomy = history[i].gnomy[j];
        track[c].brightness = history[i].brightness[j];
        track[c].size = history[i].size[j];
        track[c].timestamp = history[i].timestamp;
        track[c].time = history[i].timestamp - timestamp;
        track[c].framebrightness = history[i].framebrightness;
        track[c].neighbours = 0;
        track[c].closeneighbours = 0;
        track[c].pos = i;
        c++;
      }

  // Calculate fps
  double mints = DBL_MAX;
  double maxts = 0;
  int minpos = 0;
  int maxpos = 0;
  for (int i = 0; i < c; i++)
    if (track[i].time >= 0 && track[i].brightness > 0) {
      if (track[i].time < mints) {
        mints = track[i].time;
        minpos = track[i].pos;
      }
      if (track[i].time > maxts) {
        maxts = track[i].time;
        maxpos = track[i].pos;
      }
    }
  double fps = (maxpos - minpos) / (maxts - mints);

  // Count neighbours
  for (int i = 0; i < c; i++)
    for (int j = i + 1; j < c; j++) {
      double dist = (track[i].gnomx - track[j].gnomx) * (track[i].gnomx - track[j].gnomx) +
                    (track[i].gnomy - track[j].gnomy) * (track[i].gnomy - track[j].gnomy) +
                    (track[i].time - track[j].time) * (track[i].time - track[j].time) * fps * fps / 1000;

      if (dist < config->minspeed * config->minspeed / 2500 * (track[i].time - track[j].time) * fps) {
        track[i].closeneighbours++;
        track[j].closeneighbours++;
      }

      if (dist < config->maxspeed * config->maxspeed / 1000) {  // Allow 2.5 * maxspeed neighbourhood
        track[i].neighbours++;
        track[j].neighbours++;
      }
    }

  // Remove isolated points and static points
  for (int i = 0; i < c; i++)
    if (track[i].neighbours < max(3, config->mintrail_sec * fps) || track[i].closeneighbours > 1) track[i].brightness = 0;

  // Stricter test for isolated points in time
  int near = 0;
  for (int i = 0; i < c && !near; i++) {
    for (int j = 0; j < c && !near; j++) {
      if (i == j) continue;
      near |= fabs(track[j].timestamp - track[i].timestamp) < 3 / fps;
      if (fabs(track[j].timestamp - track[i].timestamp) > 3 * fps) track[i < c - 1 - j ? i : j].brightness = 0;
    }
    if (!near) track[i].brightness = 0;
  }

  // Cleanup
  int j = 0;
  for (int i = 0; i < c; i++)
    if (track[i].brightness > 0) {
      if (j != i) track[j] = track[i];
      j++;
    }

  if (j < config->mintrail) {
    free(track);
    return;
  }

  // Find best path
  point best_track[j];
  int best_num;
  double a, b;
  double r = ransac(j, track, best_track, &best_num, &a, &b, fps, 1 - config->corr, config->mintrail);
  if (r < pow(config->corr, 3) || best_num > config->maxtrail || best_num < config->mintrail) {
    free(track);
    return;
  }

  memcpy(track, best_track, sizeof(*track) * best_num);

  // Check whether length is reasonable
  // double dur = track[best_num-1].timestamp - track[0].timestamp;
  // if (dur > config->maxtrail_sec || dur <  config->mintrail_sec) {
  //  free(track);
  //  return;
  //}

  // Sort result
  qsort(track, best_num, sizeof(*track), pointcmp);

  // Discard partial tracks, i.e. tracks crossing the beginning or end of the
  // history These have already been detected or will be detected later
  if (track[0].timestamp == history[0].timestamp || track[best_num - 1].timestamp == history[n - 1].timestamp) {
    free(track);
    return;
  }

  memset(hashtab1, 0, 65536 / 8);

  double midaz;
  double midalt;
  double arc = 0;
  arc = midpoint(track[0].az, track[0].alt, track[best_num - 1].az, track[best_num - 1].alt, &midaz, &midalt);

  // Check whether speed is consistent with a meteor at this altitude
  double radius = 6370;  // Earth's radius
  // Start and end altitude above horizon
  double a1 = track[0].alt * M_PI / 180, a2 = track[best_num - 1].alt * M_PI / 180;
  double h1 = 90, h2 = 30;  // Typical meteor altitudes
  // Estimate distance to a meteor given altitude above horizon (angle) and
  // altitude above ground.
  double d1 = (sqrt(4 * radius * radius * sin(a1) * sin(a1) + 8 * h1 * radius) - 2 * radius * sin(a1)) / 2;
  double d2 = (sqrt(4 * radius * radius * sin(a2) * sin(a2) + 8 * h2 * radius) - 2 * radius * sin(a2)) / 2;
  double d = d1 < d2 ? d1 : d2;        // Whichever of start or end that is most close
  double speed = config->minspeedkms;  // Assume at least X km/s lateral motion, should be
                                       // safe unless it's heading our way
  double motion1 = 2 * atan(speed / (2 * d)) * 180 / M_PI;
  // Observed motion across the sky in degrees/s.
  double observed_motion = arc / (track[best_num - 1].timestamp - track[0].timestamp);
  speed = config->maxspeedkms;  // Assume at most X km/s lateral motion
  double motion2 = 2 * atan(speed / (2 * d)) * 180 / M_PI;
  observed_motion = arc / (track[best_num - 1].timestamp - track[0].timestamp);
  if (observed_motion <= motion1 || observed_motion >= motion2) {
    free(track);
    return;
  }

  // Allocate space for event reports
  char rp[16384];
  int rps = sizeof(rp);
  char tsbuf[25];
  char tsbuf2[25];

  // Check if the same track has recently been found
  uint16_t crc = crc16(track, best_num * sizeof(*track));
  if ((hashtab1[crc >> 3] | hashtab2[crc >> 3] | hashtab3[crc >> 3]) & (1 << (crc & 7))) {
    free(track);
    return;
  }

  hashtab1[crc >> 3] |= 1 << (crc & 7);

  // Make report
  time_t ts = (time_t)track[0].timestamp;
  struct tm *t = localtime(&ts);
  strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%d %H:%M:%S", t);

  ts = (time_t)track[best_num - 1].timestamp;
  t = localtime(&ts);

  strftime(tsbuf2, sizeof(tsbuf2), "%Y-%m-%d %H:%M:%S", t);

  unsigned int p = 0;
  p += snprintf(rp + p, rps - p, "[trail]\n");
  p += snprintf(rp + p, rps - p, "frames = %d\n", best_num);
  p += snprintf(rp + p, rps - p, "duration = %.2f\n", track[best_num - 1].timestamp - track[0].timestamp + 0.005);
  p += snprintf(rp + p, rps - p, "slope = %.2f\n", a);
  p += snprintf(rp + p, rps - p, "offset = %.2f\n", b);

  if (config->ptofile && track[0].az >= 0)
    p += snprintf(rp + p, rps - p, "speed = %f\n", arc / (track[best_num - 1].timestamp - track[0].timestamp));
  p += snprintf(rp + p, rps - p, "correlation1 = %f\n", r);
  p += snprintf(rp + p, rps - p, "positions =");
  for (unsigned int i = 0; i < best_num; i++)
    p += snprintf(rp + p, rps - p, " %.1f,%.1f", track[i].x * config->ptowidth / config->width,
                  track[i].y * config->ptoheight / config->height);
  p += snprintf(rp + p, rps - p, "\n");
  p += snprintf(rp + p, rps - p, "timestamps =");
  for (unsigned int i = 0; i < best_num; i++) p += snprintf(rp + p, rps - p, " %.2f", track[i].timestamp);
  p += snprintf(rp + p, rps - p, "\n");
  if (config->ptofile && track[0].az >= 0) {
    p += snprintf(rp + p, rps - p, "coordinates =");
    for (unsigned int i = 0; i < best_num; i++) p += snprintf(rp + p, rps - p, " %.2f,%.2f", track[i].az, track[i].alt);
    p += snprintf(rp + p, rps - p, "\n");
    p += snprintf(rp + p, rps - p, "gnomonic =");
    for (unsigned int i = 0; i < best_num; i++) p += snprintf(rp + p, rps - p, " %f,%f", track[i].gnomx, track[i].gnomy);
    p += snprintf(rp + p, rps - p, "\n");
    p += snprintf(rp + p, rps - p, "midpoint = %.2f,%.2f\n", midaz, midalt);
    p += snprintf(rp + p, rps - p, "arc = %.2f\n", arc);
  }
  p += snprintf(rp + p, rps - p, "brightness =");
  for (unsigned int i = 0; i < best_num; i++) p += snprintf(rp + p, rps - p, " %d", track[i].brightness);
  p += snprintf(rp + p, rps - p, "\n");
  p += snprintf(rp + p, rps - p, "size =");
  for (unsigned int i = 0; i < best_num; i++) p += snprintf(rp + p, rps - p, " %d", track[i].size);
  p += snprintf(rp + p, rps - p, "\n");
  p += snprintf(rp + p, rps - p, "frame_brightness =");
  for (unsigned int i = 0; i < best_num; i++) p += snprintf(rp + p, rps - p, " %.1f", track[i].framebrightness / 2.560);
  p += snprintf(rp + p, rps - p, "\n");

  p += snprintf(rp + p, rps - p, "\n[video]\n");
  p += snprintf(rp + p, rps - p, "start = %s.%02d UTC (%.2f)\n", tsbuf, (int)((track[0].timestamp - (int)track[0].timestamp) * 100),
                track[0].timestamp);
  p += snprintf(rp + p, rps - p, "end = %s.%02d UTC (%.2f)\n", tsbuf2,
                (int)((track[best_num - 1].timestamp - (int)track[best_num - 1].timestamp) * 100), track[best_num - 1].timestamp);

  ts = time(nullptr);
  t = gmtime(&ts);
  strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%d %H:%M:%S", t);
  p += snprintf(rp + p, rps - p, "wallclock = %s UTC\n", tsbuf);
  p += snprintf(rp + p, rps - p, "width = %d\n", config->width);
  p += snprintf(rp + p, rps - p, "height = %d\n", config->height);
  p += snprintf(rp + p, rps - p, "raw = %d\n", config->raw);
  p += snprintf(rp + p, rps - p, "flash = %d\n", config->flash);

  p += snprintf(rp + p, rps - p, "\n[config]\n");
  p += snprintf(rp + p, rps - p, "swidth = %d\n", config->swidth);
  p += snprintf(rp + p, rps - p, "sheight = %d\n", config->sheight);
  p += snprintf(rp + p, rps - p, "swdec = %d\n", config->swdec);
  p += snprintf(rp + p, rps - p, "downscale_thr = %d\n", config->downscale_thr);
  p += snprintf(rp + p, rps - p, "mintrail_sec = %.2f\n", config->mintrail_sec);
  p += snprintf(rp + p, rps - p, "maxtrail_sec = %.2f\n", config->maxtrail_sec);
  p += snprintf(rp + p, rps - p, "mintrail = %d\n", config->mintrail);
  p += snprintf(rp + p, rps - p, "maxtrail = %d\n", config->maxtrail);
  p += snprintf(rp + p, rps - p, "minspeed = %f\n", config->minspeed * config->width / 100);
  p += snprintf(rp + p, rps - p, "maxspeed = %f\n", config->maxspeed * config->width / 100);
  p += snprintf(rp + p, rps - p, "minspeedkms = %f\n", config->minspeedkms);
  p += snprintf(rp + p, rps - p, "maxspeedkms = %f\n", config->maxspeedkms);
  p += snprintf(rp + p, rps - p, "leveltest = %d\n", config->leveltest);
  p += snprintf(rp + p, rps - p, "numspots = %d\n", config->numspots);
  p += snprintf(rp + p, rps - p, "brightness = %d\n", config->brightness);
  p += snprintf(rp + p, rps - p, "flash_thr = %.2f\n", config->flash_thr);
  p += snprintf(rp + p, rps - p, "lookahead = %d\n", config->lookahead);
  p += snprintf(rp + p, rps - p, "exit = %d\n", config->exit);
  p += snprintf(rp + p, rps - p, "peak = %f\n", config->peak);
  p += snprintf(rp + p, rps - p, "filter = %d\n", config->filter);
  p += snprintf(rp + p, rps - p, "dct_threshold = %f\n", config->dct_threshold);
  p += snprintf(rp + p, rps - p, "correlation = %f\n", config->corr);
  p += snprintf(rp + p, rps - p, "spacing correlation = %f\n", config->spacing_corr);
  p += snprintf(rp + p, rps - p, "gnomonic correlation = %f\n", config->gnomonic_corr);
  p += snprintf(rp + p, rps - p, "nothreads = %d\n", config->nothreads);
  p += snprintf(rp + p, rps - p, "lastreport_ts = %u\n", (unsigned int)config->lastreport_ts);
  p += snprintf(rp + p, rps - p, "ts_future = %d\n", config->ts_future);
  p += snprintf(rp + p, rps - p, "snapshot_interval = %d\n", config->snapshot_interval);
  p += snprintf(rp + p, rps - p, "snapshot_integration = %d\n", config->snapshot_integration);
  if (config->logfile) p += snprintf(rp + p, rps - p, "logfile = %s\n", config->logfile);
  if (config->maskfile) p += snprintf(rp + p, rps - p, "maskfile = %s\n", config->maskfile);
  if (config->maxfile) p += snprintf(rp + p, rps - p, "maxfile = %s\n", config->maxfile);
  if (config->savefile) p += snprintf(rp + p, rps - p, "savefile = %s\n", config->savefile);
  if (config->ptofile) {
    p += snprintf(rp + p, rps - p, "ptofile = %s\n", config->ptofile);
    if (config->ptoscale) p += snprintf(rp + p, rps - p, "ptoscale = %f\n", config->ptoscale);
    if (config->ptowidth) p += snprintf(rp + p, rps - p, "ptowidth = %f\n", config->ptowidth);
    if (config->ptoheight) p += snprintf(rp + p, rps - p, "ptoheight = %f\n", config->ptoheight);
  }
  if (config->execute && config->saveevent) p += snprintf(rp + p, rps - p, "execute = %s\n", config->execute);
  if (config->eventdir) p += snprintf(rp + p, rps - p, "eventdir = %s\n", config->eventdir);
  if (config->snapshot_dir) p += snprintf(rp + p, rps - p, "snapshot_dir = %s\n", config->snapshot_dir);

  // Write the best detection of the current series of detections to file
  saveevent(config, rp, (time_t)track[best_num - 1].timestamp);
  config->lastreport_ts = (time_t)track[best_num - 1].timestamp;

  free(track);
}

// An n-dimensional search (n = config->mintrail).  Use recursion since n is not
// fixed.
static void best_paths(double *x, double *y, int *z, int *zz, record *history, signed char *best, signed char *path, unsigned int depth,
                       const config *config, double *bestcorr, int stride) {
  if (depth == config->mintrail) {
    double slope, offset, r, l, dr;
    double corr;
    linreg(config->mintrail, x, y, &slope, &offset, &r, &l, &dr);
    corr = r * dr;
    if (corr > 0) {
      for (int i = 0; i < config->numspots; i++) {
        if (corr > bestcorr[i]) {
          for (int j = config->numspots - 2; j >= i; j--) {
            bestcorr[j + 1] = bestcorr[j];
            memcpy(best + (j + 1) * stride, best + j * stride, sizeof(*best) * config->mintrail);
          }
          bestcorr[i] = corr;
          memcpy(best + i * stride, path, sizeof(*best) * config->mintrail);
          break;
        }
      }
    }
  } else {
    for (unsigned int a = 0; a < config->numspots; a++) {
      if (history[depth].brightness[a] > 0) {
        x[depth] = history[depth].x[a];
        y[depth] = history[depth].y[a];
        z[depth] = history[depth].brightness[a];
        zz[depth] = history[depth].size[a];
        path[depth] = a;
        best_paths(x, y, z, zz, history, best, path, depth + 1, config, bestcorr, stride);
      }
    }
  }
}

static unsigned int dcttest(int *x, int l, int *r) {
  int k;
  int totsum = 0;
  for (k = 0; k < l; k++) {
    int sum = 0;
    double s = k ? 1.0 : sqrt(0.5);
    for (int n = 0; n < l; n++) sum += s * x[n] * cos(3.14159265 * (n + 0.5) * k / l);
    r[k] = abs(sum);
    totsum += r[k];
  }

  int halfsum = 0;
  for (k = 0; k < l && halfsum < totsum / 2; k++) halfsum += r[k];

  return k;
}

// Detect a trail
// Must satisfy:
//   * The points are close to a straight line
//   * The points are ordered (no jumps back and forth)
//   * Consistent spacing between the points
//   * Minimum and maximum length
//   * Minimum and maximum speed
//   * Minimum peak brightness
//   * Not a part of a longer trail
//   * Check speed and how straight the path is if lens data are available
//
// hashtab[2,3] must be the hashtab of the previous calls,
// hashtab1 gets overwritten.
static void detect_trail(record *history, unsigned int length, config *config, uint8_t *hashtab1, const uint8_t *hashtab2,
                         const uint8_t *hashtab3, int final) {
  int len = min(config->lookahead, length * 2);
  signed char best[config->numspots * config->lookahead];
  double slope;
  double offset;
  double corr, r, l, dr;

  double *x = (double *)malloc(config->lookahead * sizeof(double));
  double *y = (double *)malloc(config->lookahead * sizeof(double));
  double *az = (double *)malloc(config->lookahead * sizeof(double));
  double *alt = (double *)malloc(config->lookahead * sizeof(double));
  int *z = (int *)malloc(config->lookahead * sizeof(int));
  int *dct = (int *)malloc(config->lookahead * sizeof(int));
  int *zz = (int *)malloc(config->lookahead * sizeof(int));
  double *bestcorr = (double *)malloc(config->numspots * sizeof(double));

  if (!x || !y || !z || !zz || !bestcorr) {
    fprintf(stderr, "Memory error\n");
    abort();
  }

  memset(hashtab1, 0, 65536 / 8);

  // Allocate space for event reports
  static char rp[16384];
  int rps = sizeof(rp);
  char tsbuf[25];
  char tsbuf2[25];
  unsigned int bestreport = 0;
  unsigned int bestreport_end = 0;
  time_t bestreport_ts = 0;

#if 0
  for (unsigned int i = 0; i < min(config->lookahead, length*2); i++) {
    printf("%d: ", i);
    for (unsigned int j = 0; j < config->numspots; j++)
      if (history[i].x[j] > 0 && history[i].y[j] > 0)
	printf("%.1f,%.1f(%d,%d) ", history[i].x[j], history[i].y[j], history[i].brightness[j], history[i].size[j]);
    printf("\n");
  }
#endif

  for (unsigned int a = 0; a < (unsigned int)min(config->lookahead - config->mintrail, length); a++) {
    if (history[a].brightness[0] <= 0) continue;

    // Update dct, mintrail and maxtrail
    unsigned int c;
    config->dct = 999999999;
    for (c = a + 1; c < length; c++) {
      if (history[c].timestamp - history[a].timestamp >= config->dct_threshold - 0.001 &&
          history[c].timestamp - history[c - 1].timestamp < config->dct_threshold) {
        config->dct = c - a;
        break;
      }
    }
    for (c = a + 1; c < length; c++) {
      if (history[c].timestamp - history[a].timestamp >= config->mintrail_sec - 0.001 &&
          history[c].timestamp - history[c - 1].timestamp < config->mintrail_sec) {
        config->mintrail = min(10, c - a);
        break;
      }
    }
    for (; c < length; c++) {
      if (history[c].timestamp - history[a].timestamp >= config->maxtrail_sec - 0.001 &&
          history[c].timestamp - history[c - 1].timestamp < config->maxtrail_sec) {
        config->maxtrail = c - a;
        break;
      }
    }
    config->mintrail = max((int)(config->mintrail_sec * 10), config->mintrail);
    config->maxtrail = max(10, config->maxtrail);

    // Find optimal path using the minimum trail
    signed char path[config->mintrail];
    int peak;

    memset(bestcorr, 0, config->numspots * sizeof(double));
    best_paths(x + a, y + a, z + a, zz + a, history + a, best, path, 0, config, bestcorr, config->lookahead);

    // Look for a flash
    unsigned int maximum_brightness = 0;
    unsigned int minimum_brightness = 255;
    unsigned int average_brightness = 0;
    for (unsigned int c = a; c < a + config->mintrail; c++) {
      maximum_brightness = max(maximum_brightness, history[c].framebrightness);
      minimum_brightness = min(minimum_brightness, history[c].framebrightness);
      average_brightness += history[c].framebrightness;
    }
    average_brightness /= config->mintrail;
    minimum_brightness += 32;
    maximum_brightness += 32;
    config->flash = config->flash_thr > 0 && average_brightness < 64 &&
                    (history[a].framebrightness + 32) / (history[a + config->mintrail - 1].framebrightness + 32) <
                        maximum_brightness / minimum_brightness / 2 &&
                    fabs(history[a].timestamp - history[a + config->mintrail - 1].timestamp) < config->maxtrail &&
                    maximum_brightness > minimum_brightness * config->flash_thr;

    for (unsigned int c = 0; c < config->numspots; c++) {
      signed char *currbest = best + c * config->lookahead;

      if (bestcorr[c] < config->corr && !config->flash) continue;

      unsigned int b;
      peak = 0;
      for (b = a; b < a + config->mintrail; b++) {
        x[b] = history[b].x[currbest[b - a]];
        y[b] = history[b].y[currbest[b - a]];
        z[b] = history[b].brightness[currbest[b - a]];
        zz[b] = history[b].size[currbest[b - a]];
        az[b] = history[b].az[currbest[b - a]];
        alt[b] = history[b].alt[currbest[b - a]];
        peak = max(peak, z[b]);
      }

      // Test whether the beginning of this path has been taken before
      // If it is, don't qualify as detection, but still try the path
      // so it can marked and later subpaths will be rejected.
      uint16_t crc1 = crc16(x + a, config->mintrail * sizeof(*x));
      uint16_t crc2 = crc16(y + a, config->mintrail * sizeof(*y));
      uint16_t crc = crc1 ^ crc2;
      int overlap = (hashtab1[crc >> 3] | hashtab2[crc >> 3] | hashtab3[crc >> 3]) & (1 << (crc & 7));

      unsigned int skip = 0;
      // Continue the path by taking the best next steps one at a time
      for (; b < config->lookahead && skip < config->mintrail; b++) {
        double bestcorr = 0;
        int stop = 1;

        if (history[b].brightness[0] == 0) {
          x[b] = history[b].x[0] = 2 * x[b - 1] - x[b - 2];
          y[b] = history[b].y[0] = 2 * y[b - 1] - y[b - 2];
          z[b] = 0;
          zz[b] = 0;
          az[b] = history[b].az[0] = 2 * az[b - 1] - az[b - 2];
          alt[b] = history[b].alt[0] = 2 * alt[b - 1] - alt[b - 2];
          currbest[b - a] = 0;
          skip++;
          continue;
        }

        skip = 0;
        for (unsigned int i = 0; i < config->numspots /*&& history[b].brightness[i] > config->brightness*/
             ;
             i++) {
          x[b] = history[b].x[i];
          y[b] = history[b].y[i];
          z[b] = history[b].brightness[i];
          zz[b] = history[b].size[i];
          az[b] = history[b].az[i];
          alt[b] = history[b].alt[i];
          corr = linreg(config->mintrail - 1, x + b - config->mintrail, y + b - config->mintrail, &slope, &offset, &r, &l, &dr);
          dr = sqrt(dr);
          if (corr > bestcorr && dr > config->spacing_corr) {
            stop = 0;
            bestcorr = corr;
            currbest[b - a] = i;
          }
        }

        if (stop) break;

        x[b] = history[b].x[currbest[b - a]];
        y[b] = history[b].y[currbest[b - a]];
        z[b] = history[b].brightness[currbest[b - a]];
        zz[b] = history[b].size[currbest[b - a]];
        az[b] = history[b].az[currbest[b - a]];
        alt[b] = history[b].alt[currbest[b - a]];

        if (dist(x[b], y[b], x[b - 1], y[b - 1]) > config->maxspeed * config->width / 100) break;

        peak = max(peak, z[b]);
      }

      corr = linreg(b - a, x + a, y + a, &slope, &offset, &r, &l, &dr);

      b = min(b, config->lookahead - 1);
      while (!z[b - 1] && b > a) b--;

      // Check if there are other points just before and after nearby this trail
      // and its extensions
      int proximity = 0;
      if (r > config->corr && dr > config->spacing_corr) {
        int n = b - a;
        double speed = dist(x[a], y[a], x[b - 1], y[b - 1]) / n;
        double extx = x[a] + (x[a] - x[b - 1]) / n;
        double exty = y[a] + (y[a] - y[b - 1]) / n;
        double prevx = x[a];
        double prevy = y[a];
        for (int i = a - 1; i >= max(0, a - 8); i--) {
          double mindist = 999999;
          double bestx = 0;
          double besty = 0;
          for (unsigned int k = 0; k < config->numspots; k++)
            if (history[i].brightness[k] > 0) {
              double d = dist(history[i].x[k], history[i].y[k], extx, exty);
              if (d < mindist) {
                mindist = d;
                bestx = history[i].x[k];
                besty = history[i].y[k];
              }

              for (unsigned int j = a; j < b; j++)
                proximity += dist(history[i].x[k], history[i].y[k], x[j], y[j]) < config->minspeed * config->width / 50;
            }
          if (mindist < speed * 2) {
            proximity++;
            extx = 2 * bestx - prevx;
            exty = 2 * besty - prevy;
          } else {
            extx += (x[a] - x[b - 1]) / n;
            exty += (y[a] - y[b - 1]) / n;
          }
          prevx = extx;
          prevy = exty;
        }

        extx = x[b - 1] + (x[b - 1] - x[a]) / n;
        exty = y[b - 1] + (y[b - 1] - y[a]) / n;
        prevx = x[b - 1];
        prevy = y[b - 1];
        for (int i = b; i < min(b + 8, min(config->lookahead - config->mintrail, length)); i++) {
          double mindist = 999999;
          double bestx = 0;
          double besty = 0;
          for (unsigned int k = 0; k < config->numspots; k++)
            if (history[i].brightness[k] > 0) {
              double d = dist(history[i].x[k], history[i].y[k], extx, exty);
              if (d < mindist) {
                mindist = d;
                bestx = history[i].x[k];
                besty = history[i].y[k];
              }
              for (unsigned int j = a; j < b; j++)
                proximity += dist(history[i].x[k], history[i].y[k], x[j], y[j]) < config->minspeed * config->width / 50;
            }
          if (mindist < speed * 2) {
            proximity++;
            extx = 2 * bestx - prevx;
            exty = 2 * besty - prevy;
          } else {
            extx += (x[b - 1] - x[a]) / n;
            exty += (y[b - 1] - y[a]) / n;
          }
          prevx = extx;
          prevy = exty;
        }
      }

#if 0
      printf("%d Best path: ", a);
      for (int i = a; i < b; i++)
	printf("%.1f,%.1f/%d ", x[i], y[i], z[i]);
      printf("\n");
      printf("%.1f > %.1f\n%d <= %d <= %d\n%d > %d\n%.1f < %.1f\nflash %d overlap %d proximity %d\n%.1f <= %.1f <= %.1f\n%f > %f\n%f > %f\n%d >= %d*%f\n", history[a].timestamp, (double)config->lastreport_ts + config->mintrail_sec,
	     config->mintrail, b - a, config->maxtrail, b - a, bestreport, history[b-1].timestamp - history[a].timestamp, config->maxtrail_sec, config->flash, overlap, proximity,
	     config->minspeed * config->width / 100, l, config->maxspeed * config->width / 100, r, config->corr, dr, config->spacing_corr, peak, config->brightness, config->peak);
#endif

      if (fabs(history[a].timestamp - (double)config->lastreport_ts) > config->mintrail_sec && b - a >= config->mintrail &&
          b - a <= config->maxtrail && b - a > bestreport && history[b - 1].timestamp - history[a].timestamp < config->maxtrail_sec &&
          dcttest(z + a, b - a, dct) <= (config->dct > 0 ? config->dct : max(b - a, config->mintrail_sec)) &&
          (config->flash ||
           (!overlap && proximity < b - a && l >= config->minspeed * config->width / 100 && l <= config->maxspeed * config->width / 100 &&
            r > config->corr && dr > config->spacing_corr && peak >= config->brightness * config->peak))) {
        double gnomonic_x[b - a];
        double gnomonic_y[b - a];
        double midaz;
        double midalt;
        double arc = 0;
        double gnomonic_corr = 999;
        int checks_passed = 1;

        // Read pto file to find scale and image size
        if (config->trafo) {
          arc = midpoint(az[a], alt[a], az[b - 1], alt[b - 1], &midaz, &midalt);

          // Make gnomonic (rectilinear) projection
          unsigned int p = b - a;
          for (unsigned int i = 0; i < p; i++) {
            double lmd = az[a + i] * M_PI / 180;
            double phi = alt[a + i] * M_PI / 180;
            double lmd0 = midaz * M_PI / 180;
            double phi1 = midalt * M_PI / 180;
            double c = sin(phi1) * sin(phi) + cos(phi1) * cos(phi) * cos(lmd - lmd0);
            if (c == 0) {
              checks_passed = 0;
              break;
            }
            gnomonic_x[i] = cos(phi) * sin(lmd - lmd0) / c;
            gnomonic_y[i] = (cos(phi1) * sin(phi) - sin(phi1) * cos(phi) * cos(lmd - lmd0)) / c;
          }

          // Check gnomonic fit
          if (checks_passed) {
            double slope, offset, l, dr;
            linreg(p, gnomonic_x, gnomonic_y, &slope, &offset, &gnomonic_corr, &l, &dr);
            if (gnomonic_corr <= config->gnomonic_corr && p > config->mintrail) {
              linreg(p - 1, gnomonic_x, gnomonic_y, &slope, &offset, &gnomonic_corr, &l, &dr);
              b -= (gnomonic_corr > config->gnomonic_corr);
            }
            if (gnomonic_corr <= config->gnomonic_corr && p > config->mintrail) {
              linreg(p - 1, gnomonic_x + 1, gnomonic_y + 1, &slope, &offset, &gnomonic_corr, &l, &dr);
              a += (gnomonic_corr > config->gnomonic_corr);
            }
            if (gnomonic_corr <= config->gnomonic_corr && p - 1 > config->mintrail) {
              linreg(p - 2, gnomonic_x + 1, gnomonic_y + 1, &slope, &offset, &gnomonic_corr, &l, &dr);
              a += (gnomonic_corr > config->gnomonic_corr);
              b -= (gnomonic_corr > config->gnomonic_corr);
            }
          }
          checks_passed &= config->flash || gnomonic_corr > config->gnomonic_corr;

          if (!checks_passed) {
            char tsbuf[25];
            time_t ts = (time_t)history[a].timestamp;
            struct tm *t = localtime(&ts);
            strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%d %H:%M:%S", t);
            debug(config->log, "Detection at %s discarded because of bad gnomonic fit %f.\n", tsbuf, gnomonic_corr, p);
          } else {
            // Check for reasonable altitudes
            checks_passed &= alt[a] > -1.0 && alt[b - 1] > -1.0;

            if (alt[a] && config->leveltest) {
              // Ignore horizontally moving objects near the horizon
              checks_passed &=
                  (fabs(alt[a] - alt[b - 1]) > 1.0 || fabs(az[a] - az[b - 1]) < 3.0 || fabs(az[a] - az[b - 1]) > 180.0 || alt[a] > 5);
              checks_passed &=
                  (fabs(alt[a] - alt[b - 1]) > 0.5 || fabs(az[a] - az[b - 1]) < 1.0 || fabs(az[a] - az[b - 1]) > 180.0 || alt[a] > 5);
            }

            if (!checks_passed) {
              char tsbuf[25];
              time_t ts = (time_t)history[a].timestamp;
              struct tm *t = localtime(&ts);
              strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%d %H:%M:%S", t);
              debug(config->log,
                    "Detection at %s discarded because the flight is too level "
                    "(%.2f,%.2f -> %.2f,%.2f).  Duration = %f.  Frames=%d.\n",
                    tsbuf, az[b - 1], alt[b - 1], az[a], alt[a], history[b - 1].timestamp - history[a].timestamp, b - a);
            } else {
              // Check whether speed is consistent with a meteor at this
              // altitude
              double r = 6370;  // Earth's radius
              // Start and end altitude above horizon
              double a1 = alt[a] * M_PI / 180, a2 = alt[b - 1] * M_PI / 180;
              double h1 = 90, h2 = 30;  // Typical meteor altitudes
              // Estimate distance to a meteor given altitude above horizon
              // (angle) and altitude above ground.
              double d1 = (sqrt(4 * r * r * sin(a1) * sin(a1) + 8 * h1 * r) - 2 * r * sin(a1)) / 2;
              double d2 = (sqrt(4 * r * r * sin(a2) * sin(a2) + 8 * h2 * r) - 2 * r * sin(a2)) / 2;
              double d = d1 < d2 ? d1 : d2;  // Whichever of start or end that is most close

              double speed = config->minspeedkms;  // Assume at least X km/s lateral motion,
                                                   // should be safe unless it's heading our
                                                   // way
              double motion1 = 2 * atan(speed / (2 * d)) * 180 / M_PI;
              // Observed motion across the sky in degrees/s.
              double observed_motion = arc / (history[b - 1].timestamp - history[a].timestamp);
              checks_passed &= config->flash || observed_motion > motion1;

              speed = config->maxspeedkms;  // Assume at most X km/s lateral motion
              double motion2 = 2 * atan(speed / (2 * d)) * 180 / M_PI;
              observed_motion = arc / (history[b - 1].timestamp - history[a].timestamp);
              checks_passed &= observed_motion < motion2;
              if (!checks_passed) {
                char tsbuf[25];
                time_t ts = (time_t)history[a].timestamp;
                struct tm *t = localtime(&ts);
                strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%d %H:%M:%S", t);
                debug(config->log,
                      "Detection at %s discarded because the speed %f deg/s is "
                      "not within %f - %f.  Arc = %f.  Duration = %f (%f - "
                      "%f).  Frames=%d.\n",
                      tsbuf, observed_motion, motion1, motion2, arc, history[b - 1].timestamp - history[a].timestamp, history[a].timestamp,
                      history[b - 1].timestamp, b - a);
              }
            }
          }
        }

        // Discard big daytime objects
        // int size = 0;
        // for (unsigned int i = a; i < b; i++)
        //  size += zz[i];
        // checks_passed &= size * history[a].framebrightness / 2.560 / (b - a)
        // < config->swidth;

        // Detection!
        if (checks_passed) {
          bestreport = b - a;
          bestreport_end = b;
          time_t ts = (time_t)history[a].timestamp;
          bestreport_ts = ts;
          struct tm *t = localtime(&ts);
          strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%d %H:%M:%S", t);

          ts = (time_t)history[b - 1].timestamp;
          t = localtime(&ts);

          strftime(tsbuf2, sizeof(tsbuf2), "%Y-%m-%d %H:%M:%S", t);

          unsigned int p = 0;
          p += snprintf(rp + p, rps - p, "[trail]\n");
          p += snprintf(rp + p, rps - p, "frames = %d\n", b - a);
          p += snprintf(rp + p, rps - p, "duration = %.2f\n", history[b - 1].timestamp - history[a].timestamp + 0.005);
          p += snprintf(rp + p, rps - p, "slope = %.2f\n", slope);
          p += snprintf(rp + p, rps - p, "offset = %.2f\n", offset);

          p += snprintf(rp + p, rps - p, "pixelspeed = %f\n", l);
          if (config->ptofile && az[a] >= 0)
            p += snprintf(rp + p, rps - p, "speed = %f\n", arc / (history[b - 1].timestamp - history[a].timestamp));
          p += snprintf(rp + p, rps - p, "correlation1 = %f\n", r);
          p += snprintf(rp + p, rps - p, "correlation2 = %f\n", dr);
          if (gnomonic_corr < 1.1) p += snprintf(rp + p, rps - p, "correlation3 = %f\n", gnomonic_corr);
          p += snprintf(rp + p, rps - p, "positions =");
          for (unsigned int i = a; i < b; i++)
            p +=
                snprintf(rp + p, rps - p, " %.1f,%.1f", x[i] * config->ptowidth / config->width, y[i] * config->ptoheight / config->height);
          p += snprintf(rp + p, rps - p, "\n");
          p += snprintf(rp + p, rps - p, "timestamps =");
          for (unsigned int i = 0; i < b - a; i++) p += snprintf(rp + p, rps - p, " %.2f", history[a + i].timestamp);
          p += snprintf(rp + p, rps - p, "\n");
          if (config->ptofile && az[a] >= 0) {
            p += snprintf(rp + p, rps - p, "coordinates =");
            for (unsigned int i = a; i < b; i++) p += snprintf(rp + p, rps - p, " %.2f,%.2f", az[i], alt[i]);
            p += snprintf(rp + p, rps - p, "\n");
            p += snprintf(rp + p, rps - p, "gnomonic =");
            for (unsigned int i = 0; i < b - a; i++) p += snprintf(rp + p, rps - p, " %f,%f", gnomonic_x[i], gnomonic_y[i]);
            p += snprintf(rp + p, rps - p, "\n");
            p += snprintf(rp + p, rps - p, "midpoint = %.2f,%.2f\n", midaz, midalt);
            p += snprintf(rp + p, rps - p, "arc = %.2f\n", arc);
          }
          p += snprintf(rp + p, rps - p, "brightness =");
          for (unsigned int i = a; i < b; i++) p += snprintf(rp + p, rps - p, " %d", z[i]);
          p += snprintf(rp + p, rps - p, "\n");
          p += snprintf(rp + p, rps - p, "dct midpoint = %d\n", dcttest(z + a, b - a, dct));
          p += snprintf(rp + p, rps - p, "dct =");
          for (unsigned int i = 0; i < b - a; i++) p += snprintf(rp + p, rps - p, " %d", dct[i]);
          p += snprintf(rp + p, rps - p, "\n");
          p += snprintf(rp + p, rps - p, "size =");
          for (unsigned int i = a; i < b; i++) p += snprintf(rp + p, rps - p, " %d", zz[i]);
          p += snprintf(rp + p, rps - p, "\n");
          p += snprintf(rp + p, rps - p, "frame_brightness =");
          for (unsigned int i = a; i < b; i++) p += snprintf(rp + p, rps - p, " %.1f", history[i].framebrightness / 2.560);
          p += snprintf(rp + p, rps - p, "\n");

          p += snprintf(rp + p, rps - p, "\n[video]\n");
          p += snprintf(rp + p, rps - p, "start = %s.%02d UTC (%.2f)\n", tsbuf,
                        (int)((history[a].timestamp - (int)history[a].timestamp) * 100), history[a].timestamp);
          p += snprintf(rp + p, rps - p, "end = %s.%02d UTC (%.2f)\n", tsbuf2,
                        (int)((history[b - 1].timestamp - (int)history[b - 1].timestamp) * 100), history[b - 1].timestamp);

          ts = time(nullptr);
          t = gmtime(&ts);
          strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%d %H:%M:%S", t);
          p += snprintf(rp + p, rps - p, "wallclock = %s UTC\n", tsbuf);
          p += snprintf(rp + p, rps - p, "width = %d\n", config->width);
          p += snprintf(rp + p, rps - p, "height = %d\n", config->height);
          p += snprintf(rp + p, rps - p, "raw = %d\n", config->raw);
          p += snprintf(rp + p, rps - p, "flash = %d\n", config->flash);

          p += snprintf(rp + p, rps - p, "\n[config]\n");
          p += snprintf(rp + p, rps - p, "swidth = %d\n", config->swidth);
          p += snprintf(rp + p, rps - p, "sheight = %d\n", config->sheight);
          p += snprintf(rp + p, rps - p, "swdec = %d\n", config->swdec);
          p += snprintf(rp + p, rps - p, "downscale_thr = %d\n", config->downscale_thr);
          p += snprintf(rp + p, rps - p, "mintrail_sec = %.2f\n", config->mintrail_sec);
          p += snprintf(rp + p, rps - p, "maxtrail_sec = %.2f\n", config->maxtrail_sec);
          p += snprintf(rp + p, rps - p, "mintrail = %d\n", config->mintrail);
          p += snprintf(rp + p, rps - p, "maxtrail = %d\n", config->maxtrail);
          p += snprintf(rp + p, rps - p, "minspeed = %f\n", config->minspeed * config->width / 100);
          p += snprintf(rp + p, rps - p, "maxspeed = %f\n", config->maxspeed * config->width / 100);
          p += snprintf(rp + p, rps - p, "minspeedkms = %f\n", config->minspeedkms);
          p += snprintf(rp + p, rps - p, "maxspeedkms = %f\n", config->maxspeedkms);
          p += snprintf(rp + p, rps - p, "leveltest = %d\n", config->leveltest);
          p += snprintf(rp + p, rps - p, "numspots = %d\n", config->numspots);
          p += snprintf(rp + p, rps - p, "brightness = %d\n", config->brightness);
          p += snprintf(rp + p, rps - p, "flash_thr = %.2f\n", config->flash_thr);
          p += snprintf(rp + p, rps - p, "lookahead = %d\n", config->lookahead);
          p += snprintf(rp + p, rps - p, "exit = %d\n", config->exit);
          p += snprintf(rp + p, rps - p, "peak = %f\n", config->peak);
          p += snprintf(rp + p, rps - p, "filter = %d\n", config->filter);
          p += snprintf(rp + p, rps - p, "dct_threshold = %f\n", config->dct_threshold);
          p += snprintf(rp + p, rps - p, "correlation = %f\n", config->corr);
          p += snprintf(rp + p, rps - p, "spacing correlation = %f\n", config->spacing_corr);
          p += snprintf(rp + p, rps - p, "gnomonic correlation = %f\n", config->gnomonic_corr);
          p += snprintf(rp + p, rps - p, "nothreads = %d\n", config->nothreads);
          p += snprintf(rp + p, rps - p, "lastreport_ts = %u\n", (unsigned int)config->lastreport_ts);
          p += snprintf(rp + p, rps - p, "ts_future = %d\n", config->ts_future);
          p += snprintf(rp + p, rps - p, "snapshot_interval = %d\n", config->snapshot_interval);
          p += snprintf(rp + p, rps - p, "snapshot_integration = %d\n", config->snapshot_integration);
          if (config->logfile) p += snprintf(rp + p, rps - p, "logfile = %s\n", config->logfile);
          if (config->maskfile) p += snprintf(rp + p, rps - p, "maskfile = %s\n", config->maskfile);
          if (config->maxfile) p += snprintf(rp + p, rps - p, "maxfile = %s\n", config->maxfile);
          if (config->savefile) p += snprintf(rp + p, rps - p, "savefile = %s\n", config->savefile);
          if (config->ptofile) {
            p += snprintf(rp + p, rps - p, "ptofile = %s\n", config->ptofile);
            if (config->ptoscale) p += snprintf(rp + p, rps - p, "ptoscale = %f\n", config->ptoscale);
            if (config->ptowidth) p += snprintf(rp + p, rps - p, "ptowidth = %f\n", config->ptowidth);
            if (config->ptoheight) p += snprintf(rp + p, rps - p, "ptoheight = %f\n", config->ptoheight);
          }
          if (config->execute && config->saveevent) p += snprintf(rp + p, rps - p, "execute = %s\n", config->execute);
          if (config->eventdir) p += snprintf(rp + p, rps - p, "eventdir = %s\n", config->eventdir);
          if (config->snapshot_dir) p += snprintf(rp + p, rps - p, "snapshot_dir = %s\n", config->snapshot_dir);
        }
      }

      // Write the best detection of the current series of detections to file
      if (bestreport_end && a > bestreport_end) {
        bestreport_end = 0;
        bestreport = 0;
        saveevent(config, rp, bestreport_ts);
        config->lastreport_ts = bestreport_ts;
      }

      if (r > config->corr && dr > config->spacing_corr) {
        // Mark this trail as checked
        for (unsigned int i = a; i <= b - config->mintrail; i++) {
          uint16_t crc1 = crc16(x + i, config->mintrail * sizeof(*x));
          uint16_t crc2 = crc16(y + i, config->mintrail * sizeof(*y));
          uint16_t crc = crc1 ^ crc2;
          hashtab1[crc >> 3] |= 1 << (crc & 7);
        }
      }
    }
  }

  // Write the best detection of the current series of detections to file
  if (bestreport_end) {
    saveevent(config, rp, bestreport_ts);
    config->lastreport_ts = bestreport_ts;
  }

  free(bestcorr);
  free(x);
  free(y);
  free(z);
  free(zz);
  free(az);
  free(alt);
  free(dct);
}

typedef struct {
  record *history;
  int length;
  config *conf;
  uint8_t *hashtab1;
  const uint8_t *hashtab2;
  const uint8_t *hashtab3;
  int final;
} detect_args;

static void *detect_trail_thread(void *arg_ptr) {
  detect_args *args = (detect_args *)arg_ptr;
  (args->conf->old_detection ? detect_trail : detect_trail2)(args->history, args->length, args->conf, args->hashtab1, args->hashtab2,
                                                             args->hashtab3, args->final);
  free(args->history);
  return 0;
}

static void detect_trail_async(record *history, int length, config *config, uint8_t *hashtab1, const uint8_t *hashtab2,
                               const uint8_t *hashtab3, int final) {
  static pthread_t detect_thread;
  static int first = 1;
  static detect_args detect_args;

  if (config->nothreads) {
    (config->old_detection ? detect_trail : detect_trail2)(history, length, config, hashtab1, hashtab2, hashtab3, final);
    return;
  }

  record *history2 = (record *)malloc(sizeof(*history2) * config->lookahead);

  if (!history2) {
    (config->old_detection ? detect_trail : detect_trail2)(history, length, config, hashtab1, hashtab2, hashtab3, final);
    return;
  }

  if (!first) {
    struct timespec thread_timeout;
    clock_gettime(CLOCK_REALTIME, &thread_timeout);
    thread_timeout.tv_sec += config->thread_timeout;
    if (pthread_timedjoin_np(detect_thread, nullptr, &thread_timeout) == ETIMEDOUT)
      debug(config->log, "%s(%d): Gave up on thread after %d seconds.\n", __FILE__, __LINE__, config->thread_timeout);
  }

  memcpy(history2, history, sizeof(*history2) * config->lookahead);
  detect_args.history = history2;
  detect_args.length = length;
  detect_args.conf = config;
  detect_args.hashtab1 = hashtab1;
  detect_args.hashtab2 = hashtab2;
  detect_args.hashtab3 = hashtab3;
  detect_args.final = final;

  int status = pthread_create(&detect_thread, nullptr, detect_trail_thread, &detect_args);
  if (status) {
    char buf[64];
    debug(config->log, "Error creating thread: %s\n", strerror_r(status, buf, sizeof(buf)));
    free(history2);
    (config->old_detection ? detect_trail : detect_trail2)(history, length, config, hashtab1, hashtab2, hashtab3, final);
  }
  first = 0;

  if (final) {
    struct timespec thread_timeout;
    clock_gettime(CLOCK_REALTIME, &thread_timeout);
    thread_timeout.tv_sec += config->thread_timeout;
    if (pthread_timedjoin_np(detect_thread, nullptr, &thread_timeout) == ETIMEDOUT)
      debug(config->log, "%s(%d): Gave up on thread after %d seconds.\n", __FILE__, __LINE__, config->thread_timeout);
  }
}

typedef struct {
  unsigned char **frame;
  unsigned char **frame_u;
  unsigned char **frame_v;
  unsigned int width;
  unsigned int height;
} decode_thread_data;

void *decode_thread_func(void *args) {
  decode_thread_data *data = (decode_thread_data *)args;
  unsigned char *h264frame;
  unsigned char *u;
  unsigned char *v;
  unsigned int width = 0, height = 0, stride = 0;
  do h264frame = decode(&width, &height, &stride, 0, 0, &u, &v);
  while (width && height && (width != data->width || height != data->height));  // Don't allow resolution to change

  *data->frame = h264frame;
  *data->frame_u = u;
  *data->frame_v = v;
  return nullptr;
}

void do_snapshot(int snapshot_progress, int width, int height, unsigned char *snapmaxbuf, unsigned char *h264frame,
                 unsigned char *h264frame_u, unsigned char *h264frame_v, int decodestride) {
  unsigned char *buf = (unsigned char *)aligned_malloc(16, 16);
  unsigned char *buf_uv = (unsigned char *)aligned_malloc(16, 16);
  if ((snapshot_progress & (snapshot_progress - 1)) == 0) {
    for (unsigned int j = 0; j < height; j++) {
      for (unsigned int k = 0; k < width; k += 16) {
        v128 a = v128_load_aligned(h264frame + j * decodestride + k);
        v128 b = v128_load_unaligned(snapmaxbuf + j * width + k);
        if (v128_hasgt_u8(a, b)) {
          // Move to ordinary RAM
          v128_store_aligned(buf, a);
          if (!(j & 1) && !h264frame_v) v128_store_aligned(buf_uv, v128_load_aligned(h264frame_u + j / 2 * decodestride + k));
          for (unsigned int l = 0; l < 16; l++) {
            if (snapmaxbuf[j * width + k + l] < buf[l]) {
              snapmaxbuf[j * width + k + l] = buf[l];
              if ((j & 1) || (l & 1)) continue;
              if (h264frame_v) {
                snapmaxbuf[(j / 2) * width / 2 + ((k + l) / 2) + width * height] = h264frame_u[(j / 2) * decodestride / 2 + ((k + l) / 2)];
                snapmaxbuf[(j / 2) * width / 2 + ((k + l) / 2) + width * height + width * height / 4] =
                    h264frame_v[(j / 2) * decodestride / 2 + ((k + l) / 2)];
              } else {
                snapmaxbuf[(j / 2) * width / 2 + ((k + l) / 2) + width * height] = buf_uv[l];
                snapmaxbuf[(j / 2) * width / 2 + ((k + l) / 2) + width * height + width * height / 4] = buf_uv[l + 1];
              }
            }
          }
        }
      }
    }
  } else
    for (unsigned int j = 0; j < height; j++)
      for (unsigned int k = 0; k < width; k += 16)
        if (v128_hasgt_u8(v128_load_aligned(h264frame + j * decodestride + k), v128_load_unaligned(snapmaxbuf + j * width + k)))
          v128_store_unaligned(snapmaxbuf + j * width + k, v128_max_u8(v128_load_unaligned(snapmaxbuf + j * width + k),
                                                                       v128_load_aligned(h264frame + j * decodestride + k)));
  aligned_free(buf);
  aligned_free(buf_uv);
}

typedef struct {
  int snapshot_progress;
  int width;
  int height;
  unsigned char *snapmaxbuf;
  unsigned char *h264frame;
  unsigned char *h264frame_u;
  unsigned char *h264frame_v;
  int decodestride;
} snapshot_thread_data;

void *snapshot_thread_func(void *args) {
  snapshot_thread_data *data = (snapshot_thread_data *)args;
  do_snapshot(data->snapshot_progress, data->width, data->height, data->snapmaxbuf, data->h264frame, data->h264frame_u, data->h264frame_v,
              data->decodestride);
  return nullptr;
}

int parseopts(config *config, int argc, char **argv, char *fargs) {
  int c;
  int size = 0;

  while ((c = getopt(argc, argv,
                     "A:a:b:c:C:d:f:g:y:Y:r:w:h:W:l:m:M:n:x:o:z:t:u:p:q:P:Q:j:"
                     "v:S:I:D:X:HeFksiLTR")) != -1) {
    switch (c) {
      case 'A': config->thread_timeout = atoi(optarg); break;
      case 'a': config->lookahead = atoi(optarg); break;
      case 'b': config->brightness = atoi(optarg); break;
      case 'y': config->filter = atoi(optarg); break;
      case 'Y': config->dct_threshold = strtod(optarg, nullptr); break;
      case 'M':
        if (!strcmp("IP8172", optarg))
          camera = IP8172;
        else if (!strcmp("IP816A", optarg))
          camera = IP816A;
        else if (!strcmp("IP9171", optarg))
          camera = IP9171;
        else if (!strcmp("IP8151", optarg))
          camera = IP8151;
        else if (!strcmp("IMX291", optarg))
          camera = IMX291;
        else if (!strcmp("IMX291SD", optarg))
          camera = IMX291SD;
        else if (!strcmp("IMX291HD", optarg))
          camera = IMX291HD;
        else if (!strcmp("IMX307", optarg))
          camera = IMX307;
        else if (!strcmp("IMX307SD", optarg))
          camera = IMX307SD;
        else if (!strcmp("IMX307HD", optarg))
          camera = IMX307HD;
        else
          fprintf(stderr, "Unknown camera model %s.\n", optarg);
        break;
      case 'c':
        if (sscanf(optarg, "%lf,%lf,%lf", &config->corr, &config->spacing_corr, &config->gnomonic_corr) < 3)
          fprintf(stderr, "Unable to parse -c option: The format must be %%f,%%f,%%f.\n");
        break;
      case 'f': config->flash_thr = strtod(optarg, nullptr); break;
      case 'g': config->peak = strtod(optarg, nullptr); break;
      case 'w':
        config->width = atoi(optarg);
        config->raw = 1;
        break;
      case 'h':
        config->height = atoi(optarg);
        config->raw = 1;
        break;
      case 'v': config->heartbeat = atoi(optarg); break;
      case 't': config->mintrail_sec = strtod(optarg, nullptr); break;
      case 'n': config->numspots = min(atoi(optarg), 16); break;
      case 'u': config->maxtrail_sec = strtod(optarg, nullptr); break;
      case 'j': config->savejpg = atoi(optarg); break;
      case 'X': config->exit = atoi(optarg); break;
      case 'e': config->saveevent = 0; break;
      case 'i': config->ts_future = 0; break;
      case 'k': config->swdec = 1; break;
      case 's': config->nothreads = 1; break;
      case 'p': config->minspeed = strtod(optarg, nullptr); break;
      case 'q': config->maxspeed = strtod(optarg, nullptr); break;
      case 'P': config->minspeedkms = strtod(optarg, nullptr); break;
      case 'Q': config->maxspeedkms = strtod(optarg, nullptr); break;
      case 'l':
        if (config->logfile) free(config->logfile);
        config->logfile = strdup(optarg);
        if (!config->logfile) {
          fprintf(stderr, "Memory error\n");
          return -1;
        }
        break;
      case 'm':
        if (config->maskfile) free(config->maskfile);
        config->maskfile = strdup(optarg);
        if (!config->maskfile) {
          fprintf(stderr, "Memory error\n");
          return -1;
        }
        break;
      case 'W': config->downscale_thr = atoi(optarg); break;
      case 'C':
        if (!fargs) {
          FILE *f = fopen(optarg, "r");
          if (f) {
            fseek(f, 0, SEEK_END);
            size = ftell(f);
            fseek(f, 0, SEEK_SET);
            fargs = (char *)malloc(size + 3);
            if (!fargs || fread(fargs + 2, 1, size, f) != size) {
              fprintf(stderr, "Error when reading config file\n");
              return 1;
            }
            fargs[0] = 'c';  // dummy fargs[0]
            fargs[1] = ' ';
            fargs[size + 2] = 0;
            int prevoptind = optind;
            optind = 1;
            char *fargv[size / 4];
            int fargc = 0;
            char *tmp;
            do fargv[fargc] = strtok_r(fargc ? nullptr : fargs, " \n\r", &tmp);
            while (fargv[fargc] && ++fargc < size / 4);
            if (parseopts(config, fargc, fargv, fargs)) {
              fprintf(stderr, "Error when reading config file\n");
              optind = prevoptind;
              free(fargs);
              return 1;
            }
            optind = prevoptind;
            free(fargs);
          } else {
            fprintf(stderr, "Error when reading config file\n");
            return 1;
          }
        }
        break;
      case 'x':
        if (config->maxfile) free(config->maxfile);
        config->maxfile = strdup(optarg);
        if (!config->maxfile) {
          fprintf(stderr, "Memory error\n");
          return -1;
        }
        break;
      case 'o':
        if (config->ptofile) free(config->ptofile);
        config->ptofile = strdup(optarg);
        if (!config->ptofile) {
          fprintf(stderr, "Memory error\n");
          return -1;
        } else {
          if (config->pano) delete config->pano;
          config->pano = new HuginBase::Panorama;
          std::ifstream ptofile(config->ptofile);
          if (!ptofile.good()) {
            fprintf(stderr, "Could not open pto file %s\n", optarg);
            delete config->pano;
            config->pano = nullptr;
          }
          config->pano->setFilePrefix(hugin_utils::getPathPrefix(optarg));
          if (!config->pano->ReadPTOFile(config->ptofile)) {
            fprintf(stderr, "Could not parse pto file %s\n", optarg);
            delete config->pano;
            config->pano = nullptr;
          }
          config->trafo = new HuginBase::PTools::Transform;
          config->trafo->createInvTransform(config->pano->getSrcImage(0), config->pano->getOptions());
        }
        break;
      case 'r':
        if (config->execute) free(config->execute);
        config->execute = strdup(optarg);
        if (!config->execute) {
          fprintf(stderr, "Memory error\n");
          return -1;
        }
        break;
      case 'd':
        if (config->eventdir) free(config->eventdir);
        config->eventdir = strdup(optarg);
        if (!config->eventdir) {
          fprintf(stderr, "Memory error\n");
          return -1;
        }
        break;
      case 'H': printhelp(); exit(1);
      case 'F': config->old_detection = 1; break;
      case 'L': config->leveltest = 0; break;
      case 'T': config->timestamp_top = 1; break;
      case 'R': config->restart = 1; break;
      case 'S': config->snapshot_interval = atoi(optarg); break;
      case 'I': config->snapshot_integration = atoi(optarg); break;
      case 'D':
        if (config->snapshot_dir) free(config->snapshot_dir);
        config->snapshot_dir = strdup(optarg);
        if (!config->snapshot_dir) {
          fprintf(stderr, "Memory error\n");
          return -1;
        }
        break;
      case '?':
        if (optopt == 'w')
          fprintf(stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint(optopt))
          fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
        return 1;
      default: return 1;
    }
  }
  return 0;
}

int _argc;
char **_argv;

void metdetect(void);

int main(int argc, char **argv) {
  _argc = argc;
  _argv = argv;
  metdetect();
  return 0;
}

void metdetect(void) {
  int argc = _argc;
  char **argv = _argv;
  FILE *i = strcmp(argv[argc - 1], "-") ? fopen(argv[argc - 1], "rb") : stdin;
  uint8_t *__restrict__ origs[2];
  uint8_t *__restrict__ orig;
  uint8_t *__restrict__ orig2;
  uint8_t *__restrict__ jpg_copy = 0;
  uint8_t *__restrict__ bufs[2];
  uint8_t *__restrict__ curr;
  uint8_t *__restrict__ prev;
  uint8_t *__restrict__ diff;
  uint8_t *__restrict__ maxbuf;
  uint8_t *__restrict__ snapmaxbuf = 0;
  uint8_t *add2;
  uint8_t *add;
  uint8_t *mask = 0;
  unsigned int x = 800, y = 600;
  config config;
  record *history;
  int first = 1;
  int errors = 0;
  static uint8_t hashtabs[3][65536 / 8];
  uint8_t *tab1 = hashtabs[0];
  uint8_t *tab2 = hashtabs[1];
  uint8_t *tab3 = hashtabs[2];
  char old_savefile[256];
  char *fargs = nullptr;

  if (argc < 2) {
    printhelp();
    exit(1);
  }

  old_savefile[0] = 0;
  signal(SIGCHLD, SIG_IGN);
  memset(hashtabs, 0, sizeof(hashtabs));
  memset(&config, 0, sizeof(config));
  config.width = x;
  config.height = y;
  config.raw = 0;
  config.numspots = 3;
  config.downscale_thr = 1600;
  config.mintrail_sec = 1.2;
  config.maxtrail_sec = 10.0;
  config.mintrail = 5;
  config.maxtrail = 150;
  config.minspeed = 0.15;
  config.maxspeed = 5.0;
  config.minspeedkms = 2.0;
  config.maxspeedkms = 50.0;
  config.leveltest = 1;
  config.exit = 0;
  config.savejpg = -1;
  config.saveevent = 1;
  config.eventdir = strdup("/tmp");
  config.snapshot_dir = strdup("/tmp");
  config.snapshot_interval = 0;
  config.snapshot_integration = 1;
  config.corr = 0.50;
  config.gnomonic_corr = 0.9985;
  config.spacing_corr = 0.90;
  config.brightness = 6;
  config.peak = 4.0;
  config.maxfile = 0;
  config.logfile = 0;
  config.log = 0;
  config.width = config.height = 0;
  config.lookahead = 900;
  config.ptofile = 0;
  config.pano = 0;
  config.trafo = 0;
  config.ptoscale = 0;
  config.ptowidth = 0;
  config.ptoheight = 0;
  config.filter = 256;
  config.dct_threshold = 0;
  config.savefile = 0;
  config.flash_thr = 1.2;
  config.nothreads = 0;
  config.lastreport_ts = 0;
  config.swdec = 0;
  config.ts_future = 1;
  config.heartbeat = 0;
  config.timestamp_top = 0;
  config.old_detection = 0;
  config.restart = 0;
  config.thread_timeout = 120;
  opterr = 0;

  if (parseopts(&config, argc, argv, fargs)) exit(1);

  if (feof(i)) exit(1);

  if (config.restart && i == stdin) atexit(metdetect);

  x = config.width;
  y = config.height;
  unsigned int decodestride = 0;
  unsigned char *h264frame = 0;
  unsigned char *h264frame2 = 0;
  unsigned char *h264frame_u = 0;
  unsigned char *h264frame_v = 0;

  if (!config.raw) {
    h264frame2 = init_dec(i, &x, &y, &decodestride, config.swdec * (config.nothreads + 1), &h264frame_u, &h264frame_v);
    if (!h264frame2 || !x || !y) {
      fprintf(stderr, "%s: No decoder found.\n", argv[0]);
      close_dec();
      exit(1);
    }
    config.width = x;
    config.height = y;
  }
  if (config.width >= config.downscale_thr) {
    x = (x + 1) >> 1;
    y = (y + 1) >> 1;
  }
  config.swidth = x;
  config.sheight = y;

  unsigned int xa = (x + 15) & ~15;
  unsigned int ya = (y + 15) & ~15;

  if (config.width == 0 || config.height == 0) {
    printhelp();
    close_dec();
    exit(1);
  }

  if (config.flash_thr < 1.01 && config.flash_thr != 0) config.flash_thr = 1.01;

  unsigned char *mask2 = 0;
  unsigned int mx = 0, my = 0;
  if (config.maskfile) mask2 = loadjpeg(config.maskfile, &mx, &my);

  if (!mask2) {
    mask2 = (unsigned char *)aligned_malloc(x * y, 16);
    mx = x;
    my = y;
    if (!mask2) {
      printf("%s: Memory error\n", argv[0]);
      close_dec();
      exit(-1);
    }
    memset(mask2, 255, x * y);
  }

  mask = (uint8_t *)aligned_malloc(xa * y, 16);
  if (!mask) {
    printf("%s: Memory error\n", argv[0]);
    close_dec();
    exit(-1);
  }
  if (xa != x) memset(mask, 255, xa * y);
  for (unsigned int a = 0; a < y; a++)
    for (unsigned int b = 0; b < x; b++) mask[a * xa + b] = mask2[a * my / y * mx + b * mx / x] > 128 ? 255 : 0;
  aligned_free(mask2);

  if (config.logfile) {
    config.log = fopen(config.logfile, "a");
    debug(config.log, "Started logging\n");
  }

  if (config.lookahead < config.maxtrail * 2) config.lookahead = config.maxtrail * 2;

  if (config.lookahead > 18000) config.lookahead = 18000;

  if (config.maxtrail > config.lookahead) config.maxtrail = config.lookahead;

  for (unsigned int i = 0; i < sizeof(bufs) / sizeof(*bufs); i++) {
    bufs[i] = (uint8_t *)aligned_malloc(config.savejpg >= 0 ? xa * ya * 1.5 : xa * ya, 16);
    if (!bufs[i]) {
      printf("%s: Memory error\n", argv[0]);
      close_dec();
      exit(-1);
    }
    memset(bufs[i], 0, xa * y);
    if (config.savejpg >= 0) memset(bufs[i] + xa * y, 128, xa * ya / 2);
  }

  if (config.raw && config.snapshot_interval > 0) {
    printf("%s: Snapshots not supported for raw input, disabling.\n", argv[0]);
    config.snapshot_interval = 0;
  }

  if (config.snapshot_interval > 0) {
    snapmaxbuf = (uint8_t *)aligned_malloc(config.width * config.height * 1.5, 16);
    if (!snapmaxbuf) {
      printf("%s: Could not allocate snapshot buffer.  Disabling snapshots.\n", argv[0]);
      config.snapshot_interval = 0;
    }
  }

  for (unsigned int i = 0; i < sizeof(origs) / sizeof(*origs); i++) {
    origs[i] = (uint8_t *)aligned_malloc(xa * ya, 16);
    if (!origs[i]) {
      printf("%s: Memory error\n", argv[0]);
      close_dec();
      exit(-1);
    }
    memset(origs[i], 0, xa * ya);
  }

  if (config.savejpg >= 0) {
    jpg_copy = (uint8_t *)aligned_malloc(xa * ya * 1.5, 16);
    memset(jpg_copy + xa * y, 128, xa * ya / 2);
  }

  history = (record *)aligned_malloc(sizeof(*history) * config.lookahead, 16);
  orig = origs[0];
  orig2 = origs[1];
  diff = (uint8_t *)aligned_malloc(xa * (config.savejpg >= 0 ? ya * 1.5 : 16), 16);
  maxbuf = (uint8_t *)aligned_malloc(xa * ya * 1.5, 16);
  memset(diff, 0, xa * (config.savejpg >= 0 ? ya * 1.5 : 16));
  memset(maxbuf, 0, xa * ya * 1.5);
  curr = bufs[0];
  prev = bufs[1];
  add2 = (uint8_t *)aligned_malloc(32 + ((x + 15) & ~15), 16);
  add = add2 + 16;

  if (!orig || !diff || !add2 || !maxbuf) {
    printf("%s: Memory error\n", argv[0]);
    close_dec();
    exit(-1);
  }

  if (config.savejpg >= 0) memset(diff + xa * y, 128, xa * ya / 2);
  memset(history, 0, config.lookahead * sizeof(*history));
  memset(maxbuf + xa * y, 128, xa * ya / 2);

  if (!i) {
    printf("%s: File error\n", argv[0]);
    close_dec();
    exit(-1);
  }
  int count = 0;
  static uint64_t framecount = 0;
  uint64_t prevdetect = 0;
  time_t timestamp = 0;
  time_t prevtimestamp = 0;
  time_t prevtimestamp2 = 0;
  time_t snapshot_timestamp = 0;
  int timestamp_fractions = 0;
  int snapshot_progress = 0;
  unsigned int mindist = x * config.maxspeed / 100;
  mindist *= mindist;

  config.savefile = getsavefile(config.maxfile);
  if (*config.savefile) {
    unsigned int x = 0, y = 0;
    unsigned char *prev_max = loadjpeg(config.savefile, &x, &y);
    debug(config.log, "Max file is %s (%d x %d)\n", config.savefile, x, y);
    if (config.width < x || config.height < y || !prev_max) {
      debug(config.log, "Corrupt max %s %d x %d vs %d x %d\n", config.savefile, x, y, config.width, config.height);
      memset(maxbuf, 0, xa * ya);
    } else {
      debug(config.log, "Old max file %s restored\n", config.savefile);
      memcpy(maxbuf, prev_max, xa * ya);
    }
    if (prev_max) aligned_free(prev_max);
  } else
    debug(config.log, "No max file\n");

  int clean_end = 1;

  unsigned char *timestampbuf;
  if (config.width >= config.downscale_thr) {
    int size = ((config.width + 31) & ~31) * 58;
    timestampbuf = (unsigned char *)aligned_malloc(size, 16);
    memset(timestampbuf, 0, size);
  } else
    timestampbuf = config.timestamp_top ? orig : orig + (config.height - 58) * config.width;

  if (!timestampbuf) {
    printf("%s: Memory error\n", argv[0]);
    close_dec();
    exit(-1);
  }

  pthread_t decode_thread;
  int decode_thread_init = 0;
  pthread_t snapshot_thread;
  int snapshot_running = 0;
  do {
    // Dump max file into file named in config.maxfile
    if ((int)timestamp / 8 != (int)prevtimestamp2 / 8 && *config.savefile) {
      FILE *o = fopen(config.savefile, "wb");
      if (o) {
        if (!savejpeg(o, maxbuf, x, y, 90)) debug(config.log, "Unable to write max file %s\n", config.savefile);
        fclose(o);
      }
      char *t = getsavefile(config.maxfile);
      if (*t) {
        if (strncmp(old_savefile, config.savefile, sizeof(old_savefile))) {
          memset(maxbuf, 0, xa * ya);
          memset(maxbuf + xa * y, 128, xa * ya / 2);
          debug(config.log, "New max file %s\n", config.savefile);
        }
        strncpy(old_savefile, config.savefile, sizeof(old_savefile));
        config.savefile = t;
      } else if (config.maxfile)
        debug(config.log, "Unable to read max file location %s\n", config.maxfile);
    }

    brightest brightest[config.numspots];

    // Prepare snapshot
    if (config.snapshot_interval > 0 && !snapshot_progress && timestamp != prevtimestamp2 &&
        iabs(timestamp - snapshot_timestamp) >= config.snapshot_interval) {
      snapshot_progress = 1;
      snapshot_timestamp = timestamp - (timestamp % config.snapshot_interval);
      memset(snapmaxbuf, 0, config.width * config.height);
      memset(snapmaxbuf + config.width * config.height, 128, config.width * config.height / 2);
    }
    prevtimestamp2 = timestamp;

    if (snapshot_progress) {
      if (snapshot_progress > config.snapshot_integration) {
        char buffer[256];
        char buffer2[256];
        char tsbuf1[20];
        char tsbuf2[20];
        struct stat st = { 0 };
        struct tm *t = localtime(&snapshot_timestamp);
        if (snapshot_running) {
          struct timespec thread_timeout;
          clock_gettime(CLOCK_REALTIME, &thread_timeout);
          thread_timeout.tv_sec += config.thread_timeout;
          if (decode_thread_init && pthread_timedjoin_np(decode_thread, nullptr, &thread_timeout) == ETIMEDOUT)
            debug(config.log, "%s(%d): Gave up on thread after %d seconds.\n", __FILE__, __LINE__, config.thread_timeout);
        }

        strftime(tsbuf1, sizeof(tsbuf1), "%Y%m%d", t);
        strftime(tsbuf2, sizeof(tsbuf2), "%H%M%S", t);
        snapshot_progress = 0;
        snprintf(buffer, sizeof(buffer), "%s/%s", config.snapshot_dir, tsbuf1);
        if (stat(buffer, &st) == -1) mkdir(buffer, 0777);
        snprintf(buffer, sizeof(buffer), "%s/%s/%s.jpg", config.snapshot_dir, tsbuf1, tsbuf2);
        FILE *o = fopen(buffer, "wb");
        if (o) {
          if (!savejpeg(o, snapmaxbuf, config.width, config.height, 90)) debug(config.log, "Unable to write snapshot %s\n", buffer);
          fclose(o);
        } else
          debug(config.log, "Unable to open %s for writing (%s)\n", buffer, strerror(errno));
        snprintf(buffer2, sizeof(buffer2), "%s/snapshot.jpg", config.snapshot_dir);
        int input = open(buffer, O_RDONLY);
        if (input != -1) {
          int output = creat(buffer2, 0666);
          if (output != -1) {
            off_t bytes_copied = 0;
            struct stat fileinfo = { 0 };
            fstat(input, &fileinfo);
            int result = sendfile(output, input, &bytes_copied, fileinfo.st_size);
            close(input);
            close(output);
          }
        }
      } else {
        snapshot_progress++;
      }

      if (snapshot_running) {
        struct timespec thread_timeout;
        clock_gettime(CLOCK_REALTIME, &thread_timeout);
        thread_timeout.tv_sec += config.thread_timeout;
        if (decode_thread_init && pthread_timedjoin_np(decode_thread, nullptr, &thread_timeout) == ETIMEDOUT)
          debug(config.log, "%s(%d): Gave up on thread after %d seconds.\n", __FILE__, __LINE__, config.thread_timeout);
      }

      static snapshot_thread_data args;
      args.width = config.width;
      args.height = config.height;
      args.h264frame = h264frame;
      args.h264frame_u = h264frame_u;
      args.h264frame_v = h264frame_v;
      args.snapshot_progress = snapshot_progress;
      args.snapmaxbuf = snapmaxbuf;
      args.decodestride = decodestride;
      snapshot_running = !config.nothreads;
      if (config.nothreads || pthread_create(&snapshot_thread, nullptr, snapshot_thread_func, &args)) {
        snapshot_running = 0;
        do_snapshot(snapshot_progress, config.width, config.height, snapmaxbuf, h264frame, h264frame_u, h264frame_v, decodestride);
      }
    }

    if (!config.raw && !first && !config.nothreads && !ffmpeg_open) {
      struct timespec thread_timeout;
      clock_gettime(CLOCK_REALTIME, &thread_timeout);
      thread_timeout.tv_sec += config.thread_timeout;
      if (decode_thread_init && pthread_timedjoin_np(decode_thread, nullptr, &thread_timeout) == ETIMEDOUT)
        debug(config.log, "%s(%d): Gave up on thread after %d seconds.\n", __FILE__, __LINE__, config.thread_timeout);
    }

    first = 0;
    h264frame = h264frame2;
    // Fetch the next frame asynchronously
    if (!config.raw) {
      static decode_thread_data args;
      args.width = config.width;
      args.height = config.height;
      args.frame = &h264frame2;
      args.frame_u = &h264frame_u;
      args.frame_v = &h264frame_v;
      // Single threaded operation seems faster when using sw decoder, perhaps
      // due to cache smashing
      decode_thread_init = !ffmpeg_open && !config.nothreads && !pthread_create(&decode_thread, nullptr, decode_thread_func, &args);
      if (!decode_thread_init) {
        unsigned int width = 0, height = 0, stride = 0;
        do {
          h264frame = decode(&width, &height, &stride, 0, 0, &h264frame_u, &h264frame_v);
        } while (width && height && (width != config.width || height != config.height));  // Don't allow resolution to change
      }
    }

    // Fetch a new frame
    // TODO: location of the timestamp
    if (config.width >= config.downscale_thr) {
      if (!config.raw) {
        if (h264frame) {
          static int i = 0;

          if (config.timestamp_top) {
            for (unsigned int l = 0; l < 58 / 2; l++) {
              memcpy(timestampbuf + l * 2 * config.width, h264frame + l * 2 * decodestride, x * 2);
              memcpy(timestampbuf + (l * 2 + 1) * config.width, h264frame + (1 + l * 2) * decodestride, x * 2);
              for (unsigned int k = 0; k < x; k += 16) {
                v128 a = v128_shr_u16(v128_add_16(v128_padd_u8(v128_load_unaligned(timestampbuf + k * 2 + l * 2 * config.width)),
                                                  v128_padd_u8(v128_load_unaligned(timestampbuf + k * 2 + (l * 2 + 1) * config.width))),
                                      2);
                v128 b =
                    v128_shr_u16(v128_add_16(v128_padd_u8(v128_load_unaligned(timestampbuf + k * 2 + 16 + l * 2 * config.width)),
                                             v128_padd_u8(v128_load_unaligned(timestampbuf + k * 2 + 16 + (l * 2 + 1) * config.width))),
                                 2);

                v128_store_aligned(orig + l * xa + k, v128_unziplo_8(b, a));
              }
            }
          }

          // No vertical filter to save memory operations
          for (unsigned int j = (config.timestamp_top ? 58 : 0); j < y - (config.timestamp_top ? 0 : 58 / 2); j++) {
            for (unsigned int k = 0; k < x; k += 16) {
              // No vertical filter to save memory operations
              v128 a = v128_shr_u16(v128_padd_u8(v128_load_aligned(h264frame + j * 2 * decodestride + k * 2)), 1);
              v128 b = v128_shr_u16(v128_padd_u8(v128_load_aligned(h264frame + j * 2 * decodestride + k * 2 + 16)), 1);
              v128_store_aligned(orig + j * xa + k, v128_unziplo_8(b, a));
            }
          }

          if (!config.timestamp_top) {
            for (unsigned int l = 0; l < 58 / 2; l++) {
              memcpy(timestampbuf + l * 2 * config.width, h264frame + (y * 2 - 58 + l * 2) * decodestride, x * 2);
              memcpy(timestampbuf + (l * 2 + 1) * config.width, h264frame + (y * 2 - 58 + 1 + l * 2) * decodestride, x * 2);
              for (unsigned int k = 0; k < x; k += 16) {
                v128 a = v128_shr_u16(v128_add_16(v128_padd_u8(v128_load_unaligned(timestampbuf + k * 2 + l * 2 * config.width)),
                                                  v128_padd_u8(v128_load_unaligned(timestampbuf + k * 2 + (l * 2 + 1) * config.width))),
                                      2);
                v128 b =
                    v128_shr_u16(v128_add_16(v128_padd_u8(v128_load_unaligned(timestampbuf + k * 2 + 16 + l * 2 * config.width)),
                                             v128_padd_u8(v128_load_unaligned(timestampbuf + k * 2 + 16 + (l * 2 + 1) * config.width))),
                                 2);

                v128_store_aligned(orig + (y - 58 / 2 + l) * xa + k, v128_unziplo_8(b, a));
              }
            }
          }
        }
      } else {
        int size = ((config.width + 31) & ~31) * 58;
        unsigned char *buf = (unsigned char *)aligned_malloc(size, 16);
        memset(buf, 0, size);

        if (config.timestamp_top) {
          int s = fread(timestampbuf, config.width * 58, 1, i);
          if (s != 1) {
            clean_end = !h264frame;
            break;
          }
          for (unsigned int l = 0; l < 58 / 2; l++) {
            for (unsigned int k = 0; k < x; k += 16) {
              v128 a = v128_shr_u16(v128_add_16(v128_padd_u8(v128_load_unaligned(timestampbuf + k * 2 + l * 2 * config.width)),
                                                v128_padd_u8(v128_load_unaligned(timestampbuf + k * 2 + (l * 2 + 1) * config.width))),
                                    2);
              v128 b = v128_shr_u16(v128_add_16(v128_padd_u8(v128_load_unaligned(timestampbuf + k * 2 + 16 + l * 2 * config.width)),
                                                v128_padd_u8(v128_load_unaligned(timestampbuf + k * 2 + 16 + (l * 2 + 1) * config.width))),
                                    2);

              v128_store_aligned(orig + (y - 58 / 2 + l) * xa + k, v128_unziplo_8(b, a));
            }
          }
        }

        // Downscale by 2
        for (unsigned int j = (config.timestamp_top ? 58 : 0); j < y - (config.timestamp_top ? 0 : 58 / 2); j++) {
          int s = fread(buf, config.width * 2, 1, i);

          if (s != 1) {
            clean_end = 0;
            j = y - 2;
            break;
          }
          if (x == xa)
            for (unsigned int k = 0; k < x; k += 16) {
              v128 a = v128_shr_u16(
                  v128_add_16(v128_padd_u8(v128_load_aligned(buf + k * 2)), v128_padd_u8(v128_load_aligned(buf + config.width + k * 2))),
                  2);
              v128 b = v128_shr_u16(v128_add_16(v128_padd_u8(v128_load_aligned(buf + k * 2 + 16)),
                                                v128_padd_u8(v128_load_aligned(buf + config.width + k * 2 + 16))),
                                    2);

              v128_store_aligned(orig + j * xa + k, v128_unziplo_8(b, a));
            }
          else
            for (unsigned int k = 0; k < x; k += 16) {
              v128 a = v128_shr_u16(v128_add_16(v128_padd_u8(v128_load_unaligned(buf + k * 2)),
                                                v128_padd_u8(v128_load_unaligned(buf + config.width + k * 2))),
                                    2);
              v128 b = v128_shr_u16(v128_add_16(v128_padd_u8(v128_load_unaligned(buf + k * 2 + 16)),
                                                v128_padd_u8(v128_load_unaligned(buf + config.width + k * 2 + 16))),
                                    2);

              v128_store_aligned(orig + j * xa + k, v128_unziplo_8(b, a));
            }
        }
        aligned_free(buf);

        if (!config.timestamp_top) {
          int s = fread(timestampbuf, config.width * 58, 1, i);
          if (s != 1) {
            clean_end = !h264frame;
            break;
          }
          for (unsigned int l = 0; l < 58 / 2; l++) {
            for (unsigned int k = 0; k < x; k += 16) {
              v128 a = v128_shr_u16(v128_add_16(v128_padd_u8(v128_load_unaligned(timestampbuf + k * 2 + l * 2 * config.width)),
                                                v128_padd_u8(v128_load_unaligned(timestampbuf + k * 2 + (l * 2 + 1) * config.width))),
                                    2);
              v128 b = v128_shr_u16(v128_add_16(v128_padd_u8(v128_load_unaligned(timestampbuf + k * 2 + 16 + l * 2 * config.width)),
                                                v128_padd_u8(v128_load_unaligned(timestampbuf + k * 2 + 16 + (l * 2 + 1) * config.width))),
                                    2);

              v128_store_aligned(orig + (y - 58 / 2 + l) * xa + k, v128_unziplo_8(b, a));
            }
          }
        }
      }

      if (!clean_end) break;
    } else {
      if (!config.raw) {
        if (h264frame) {
          if (x == decodestride) {
            memcpy(orig, h264frame, x * y);
          } else {
            for (int i = 0; i < y; i++) memcpy(orig + i * xa, h264frame + i * decodestride, x);
          }
        }
      } else {
        if (x == xa) {
          int s = fread(orig, x * y, 1, i);
          if (s != 1) {
            clean_end = !h264frame;
            break;
          }
        } else {
          for (unsigned int j = 0; j < y; j++) {
            unsigned char buf[x];
            int s = fread(buf, x, 1, i);
            if (s != 1) {
              clean_end = !j && !h264frame;
              break;
            }
            memcpy(orig + j * xa, buf, x);
          }
        }
      }

      if (config.timestamp_top)
        for (unsigned int j = 0; j < 58; j++) memcpy(timestampbuf + j * config.width, orig + j * xa, config.width);
    }

    int end = config.lookahead - 1;

    // Timestamp analysis
    timestamp = 0;
    for (int threshold = 192; threshold > 128 && timestamp < 1388534400; threshold -= 16)
      timestamp = gettimestamp(timestampbuf, config.width, 58, !config.saveevent, threshold);

    static int count = 0;
    static int prevheartbeat = 0;
    if (config.heartbeat && count % config.heartbeat == 0) {
      if (timestamp && timestamp == prevheartbeat) {
        // Not progressing, something is seriously wrong
        debug(config.log, "Timestamps not progressing, giving up.  Restarting decoder.\n");
        struct timespec thread_timeout;
        clock_gettime(CLOCK_REALTIME, &thread_timeout);
        thread_timeout.tv_sec += config.thread_timeout;
        if (decode_thread_init && pthread_timedjoin_np(decode_thread, nullptr, &thread_timeout) == ETIMEDOUT)
          debug(config.log, "%s(%d): Gave up on thread after %d seconds.\n", __FILE__, __LINE__, config.thread_timeout);
        close_dec();
        h264frame2 = init_dec(i, &x, &y, &decodestride, config.swdec * (config.nothreads + 1), &h264frame_u, &h264frame_v);
      }
      char tsbuf[25];
      struct tm *t = localtime(&timestamp);
      strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%d %H:%M:%S", t);
      debug(config.log, "Heartbeat %s\n", tsbuf);
      prevheartbeat = timestamp;
    }

    if ((timestamp < 1388534400 || (config.ts_future && timestamp > time(nullptr) + 3600 * 24)) && timestamp != prevtimestamp ||
        !timestamp && !prevtimestamp) {
      char tsbuf[25];
      struct tm *t = localtime(&timestamp);
      strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%d %H:%M:%S", t);
      debug(config.log, "Illegal timestamp %s.\n", tsbuf);
      t = localtime(&prevtimestamp);
      strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%d %H:%M:%S", t);
      debug(config.log, "  Previously %s.\n", tsbuf);
      timestamp = prevtimestamp;
      if (config.exit > 0) {
        errors++;
        if (errors >= config.exit) {
          debug(config.log, "Unable to read timestamps, giving up.  Restarting decoder.\n");
          struct timespec thread_timeout;
          clock_gettime(CLOCK_REALTIME, &thread_timeout);
          thread_timeout.tv_sec += config.thread_timeout;
          if (decode_thread_init && pthread_timedjoin_np(decode_thread, nullptr, &thread_timeout) == ETIMEDOUT)
            debug(config.log, "%s(%d): Gave up on thread after %d seconds.\n", __FILE__, __LINE__, config.thread_timeout);
          errors = 0;
          close_dec();
          h264frame2 = init_dec(i, &x, &y, &decodestride, config.swdec * (config.nothreads + 1), &h264frame_u, &h264frame_v);
        }
      }
      continue;
    }
    errors = 0;

    if (timestamp == prevtimestamp) {
      timestamp_fractions++;
      history[end].timestamp = timestamp;
    } else {
      for (int i = 0; i <= min(timestamp_fractions, config.lookahead); i++)
        history[end - i].timestamp = (double)timestamp + (double)(i + 1) * (prevtimestamp - timestamp) / (timestamp_fractions + 1);
      timestamp_fractions = 0;
      prevtimestamp = timestamp;
    }
    framecount++;

    for (unsigned int i = 0; i < config.numspots; i++) {
      brightest[i].brightness = 0;
      brightest[i].x = brightest[i].y = -999;
    }

    // Compute 16x2 regional mean of input.

    // Fill up add buffer
    memset(add2, 0, 32 + ((x + 15) & ~15));
    for (unsigned int b = 0; b < x; b += 16) {
      // Use the max of the two most recent frames to avoid detection of dark
      // objects.
      v128 d = v128_padd_u8(v128_max_u8(v128_load_aligned(orig + b), v128_load_aligned(orig2 + b)));
      v128 c = v128_add_16(d, v128_shl_16(d, 3));
      for (int a = 1; a < 8; a++)
        c = v128_add_16(c, v128_padd_u8(v128_max_u8(v128_load_aligned(orig + a * xa + b), v128_load_aligned(orig2 + a * xa + b))));

      v128_store_aligned(add + b, c);
    }

    unsigned int pixelcount = 0;
    unsigned int pixelsum = 0;

    // Use sliding window: add new line, subtract line leaving the window.
    for (unsigned int a = 0; a < y; a++) {
      unsigned int top = max(a - 8, 0);
      unsigned int bot = min(a + 7, y - 1);
      unsigned int aa = a;
      unsigned int a1 = max(0, a - 1);
      if (config.savejpg < 0) {
        aa &= 15;
        a1 &= 15;
      }
      v128 threshold = v128_dup_8(config.brightness);  // + log2i(x >> 6));

      for (unsigned int b = 0; b < x; b++) {
        if (!(b & 15)) {
          v128 c = v128_max_u8(v128_load_aligned(orig + top * xa + b), v128_load_aligned(orig2 + top * xa + b));
          v128 d = v128_max_u8(v128_load_aligned(orig + bot * xa + b), v128_load_aligned(orig2 + bot * xa + b));
          v128 sum = v128_sub_16(v128_add_16(v128_load_aligned(add + b), v128_padd_u8(d)), v128_padd_u8(c));
          v128_store_aligned(add + b, sum);
          sum = v128_shr_u16(sum, 5);
          sum = v128_or(v128_shl_16(sum, 8), sum);

          // Subtract regional 16x2 mean to remove the background.
          // 16x16 gaussian would be better, but that's too slow.
          c = v128_ssub_u8(v128_load_aligned(orig + b + a * xa), sum);
          v128_store_aligned(curr + b + a * xa, c);

          // Difference from previous frame
          d = v128_and(v128_ssub_u8(c, v128_load_aligned(prev + a * xa + b)), v128_load_aligned(mask + a * xa + b));
          v128_store_aligned(diff + aa * xa + b, d);

          // Measure frame brightness
          int masked = mask[a * xa + b];
          pixelcount += !!masked;
          pixelsum += orig[b + a * xa] & masked;

          // Update max frame
          if (config.maxfile)
            v128_store_aligned(maxbuf + a * xa + b,
                               v128_max_u8(v128_load_aligned(orig + a * xa + b), v128_load_aligned(maxbuf + a * xa + b)));

          // Early break if all 16 bytes are below the threshold
          if (!v128_hasgt_u8(d, threshold)) {
            b += 15;
            continue;
          }
        }

        // Very simple despeckle
        unsigned int bb = max(0, b - 1);
        int c = diff[bb + aa * xa];
        if (c - 16 > diff[b + aa * xa] && c - 16 > diff[max(0, b - 2) + aa * xa] && c - 16 > diff[bb + a1 * xa])
          diff[bb + aa * xa] = max(diff[b + aa * xa], diff[(b - 2) + aa * xa]);

        // Find the brightest spot in the diff image
        int brightness = c + diff[b + aa * xa] + diff[b + a1 * xa];
        c = config.numspots - 1;
        if (brightness <= brightest[c].brightness || brightness <= config.brightness) continue;

        // Keep a record of the brightest spots
        for (; c >= 0 && brightness > brightest[c].brightness; c--);
        unsigned int a2 = min(max(a - 4, 0), y - 8);
        unsigned int b2 = min(max(b - 4, 0), x - 8);
        if (++c < config.numspots && frequency_check(orig + a2 * xa + b2, x, config.filter)) {
          unsigned int i, dx, dy;
          for (i = 0; i < c; i++) {
            dx = brightest[i].x - bb;
            dy = brightest[i].y - a;
            if (dx * dx + dy * dy < mindist) break;
          }
          if (i == c) {  // Only if not a nearby brighter spot
            dx = brightest[c].x - bb;
            dy = brightest[c].y - a;
            if (dx * dx + dy * dy > mindist)  // Replace if the weaker is nearby
              memmove(brightest + c + 1, brightest + c, sizeof(*brightest) * (config.numspots - 1 - c));
            brightest[c].brightness = brightness;
            brightest[c].x = bb;
            brightest[c].y = a;

            for (unsigned int i = c + 1; i < config.numspots; i++) {
              unsigned int dx = brightest[i].x - b;
              unsigned int dy = brightest[i].y - a;
              if (dx * dx + dy * dy < mindist) {
                memmove(brightest + i, brightest + i + 1, sizeof(*brightest) * (config.numspots - 1 - i));
                brightest[config.numspots - 1].brightness = 0;
                brightest[config.numspots - 1].x = brightest[config.numspots - 1].y = -999;
              }
            }
          }
        }
      }
    }

    memmove(history, history + 1, end * sizeof(*history));
    memset(history + end, 0, sizeof(*history));
    history[end].framebrightness = pixelsum / pixelcount;

    if (brightest[0].brightness || config.savejpg >= 0) {
      // Search the 16x16 area around the brightest pixel,
      // find the centre and brightness of the entire spot.
      unsigned int j = 0;
      for (unsigned int i = 0; i < config.numspots && brightest[i].brightness; i++) {
        int cx = 0;
        int cy = 0;
        int cw = 0;
        int size = 0;

        history[end].brightness[j] = find_centre(curr, brightest[i].x, brightest[i].y, xa, y, &cx, &cy, &cw, &size);
        history[end].size[j] = size;
        history[end].x[j] = history[end].y[j] = history[end].gnomx[j] = history[end].gnomy[j] = history[end].az[j] = history[end].alt[j] =
            NAN;
        if (history[end].brightness[j]) {
          history[end].x[j] = (double)cx / cw * config.width / x;
          history[end].y[j] = (double)cy / cw * config.height / y;

          if (config.trafo) {
            double w, h, v;
            const HuginBase::SrcPanoImage &img = config.pano->getImage(0);
            const HuginBase::PanoramaOptions &opt = config.pano->getOptions();
            v = opt.getHFOV();
            config.ptowidth = (double)img.getWidth();
            config.ptoheight = (double)img.getHeight();
            w = (double)opt.getWidth();
            h = (double)opt.getHeight();
            if (v > 0) config.ptoscale = w / v;

            // Translate to az and alt
            double xin = history[end].x[j] * config.ptowidth / config.width;
            double yin = history[end].y[j] * config.ptoheight / config.height;
            double xout, yout;
            config.trafo->transformImgCoord(xout, yout, xin, yin);
            history[end].az[j] = xout / config.ptoscale;
            history[end].alt[j] = 90 - yout / config.ptoscale;

            // Use screen centre and calculate gnonomic coordinates
            config.trafo->transformImgCoord(xout, yout, config.ptowidth / 2, config.ptoheight / 2);
            double midaz = xout / config.ptoscale;
            double midalt = 90 - yout / config.ptoscale;

            double lmd = history[end].az[j] * M_PI / 180;
            double phi = history[end].alt[j] * M_PI / 180;
            double lmd0 = midaz * M_PI / 180;
            double phi1 = midalt * M_PI / 180;
            double c = sin(phi1) * sin(phi) + cos(phi1) * cos(phi) * cos(lmd - lmd0);
            if (c != 0) {
              history[end].gnomx[j] = cos(phi) * sin(lmd - lmd0) / c;
              history[end].gnomy[j] = (cos(phi1) * sin(phi) - sin(phi1) * cos(phi) * cos(lmd - lmd0)) / c;
            }
          }
        }
        j++;
      }

      // Annotate image if saving jpg
      if (config.savejpg >= 0) {
        unsigned char *b = config.savejpg == 1 || config.savejpg == 4 ? curr : config.savejpg == 2 || config.savejpg == 5 ? diff : orig;
        memcpy(jpg_copy, b, xa * y);
        b = jpg_copy;
        if (config.savejpg > 2 && mask) {
          for (unsigned int i = 0; i < x; i++)
            for (unsigned int j = 0; j < y; j++) b[j * xa + i] = mask[i + j * xa] ? b[j * xa + i] : 0;
        }

        for (unsigned int i = 0; i < config.numspots && history[end].brightness[i]; i++) {
          int ix = (int)(history[end].x[i] + 0.5) * x / config.width;
          int iy = (int)(history[end].y[i] + 0.5) * y / config.height;
          b[iy * xa + clip(ix - 5, 0, x)] = b[iy * xa + clip(ix - 6, 0, x)] = b[iy * xa + clip(ix - 7, 0, x)] =
              b[iy * xa + clip(ix - 8, 0, x)] = b[iy * xa + clip(ix + 5, 0, x)] = b[iy * xa + clip(ix + 6, 0, x)] =
                  b[iy * xa + clip(ix + 7, 0, x)] = b[iy * xa + clip(ix + 8, 0, x)] = b[ix + clip(iy - 5, 0, y) * xa] =
                      b[ix + clip(iy - 6, 0, y) * xa] = b[ix + clip(iy - 7, 0, y) * xa] = b[ix + clip(iy - 8, 0, y) * xa] =
                          b[ix + clip(iy + 5, 0, y) * xa] = b[ix + clip(iy + 6, 0, y) * xa] = b[ix + clip(iy + 7, 0, y) * xa] =
                              b[ix + clip(iy + 8, 0, y) * xa] = 255 - i * 15;
        }
      }
    }

    // Save jpg file
    if (config.savejpg >= 0) {
      static char buffer[256];
      sprintf(buffer, "frame-%05d.jpg", count);
      FILE *o = fopen(buffer, "wb");
      if (!savejpeg(o, jpg_copy, x, y, 100)) debug(config.log, "Unable to write jpg file %s\n", buffer);
      fclose(o);
    }

    count++;
    curr = bufs[count % (sizeof(bufs) / sizeof(*bufs))];
    prev = bufs[(count + 1) % (sizeof(bufs) / sizeof(*bufs))];
    orig = origs[count % (sizeof(origs) / sizeof(*origs))];
    orig2 = origs[(count + 1) % (sizeof(origs) / sizeof(*origs))];
    if (framecount - prevdetect >= config.lookahead / 2 && framecount >= config.lookahead) {
      int len = framecount - prevdetect;
      if (len > 5) {
        detect_trail_async(history, len, &config, tab1, tab2, tab3, 0);
        prevdetect = framecount;
        tab1 = hashtabs[(framecount + 0) % 3];
        tab2 = hashtabs[(framecount + 1) % 3];
        tab3 = hashtabs[(framecount + 2) % 3];
      }
    }
  } while (h264frame);

  if (config.savefile && *config.savefile) {
    FILE *o = fopen(config.savefile, "wb");
    if (o) {
      if (!savejpeg(o, maxbuf, x, y, 90)) debug(config.log, "Unable to write max file %s\n", config.savefile);
      fclose(o);
    }
  }

  if (clean_end) {
    if (framecount > 5) detect_trail_async(history, config.lookahead, &config, tab1, tab2, tab3, 1);
  }

  debug(config.log, "Exiting.\n");

  for (unsigned int i = 0; i < sizeof(bufs) / sizeof(*bufs); i++) aligned_free(bufs[i]);

  for (unsigned int i = 0; i < sizeof(origs) / sizeof(*origs); i++) aligned_free(origs[i]);

  if (snapmaxbuf) aligned_free(snapmaxbuf);

  if (jpg_copy) aligned_free(jpg_copy);
  aligned_free(history);
  aligned_free(diff);
  aligned_free(maxbuf);
  aligned_free(add2);
  if (config.width >= config.downscale_thr) aligned_free(timestampbuf);
  if (mask) aligned_free(mask);
  if (config.log) fclose(config.log);
  if (i && i != stdin) fclose(i);

  if (!config.raw) {
    if (!config.nothreads && !ffmpeg_open) {
      struct timespec thread_timeout;
      clock_gettime(CLOCK_REALTIME, &thread_timeout);
      thread_timeout.tv_sec += config.thread_timeout;
      if (decode_thread_init && pthread_timedjoin_np(decode_thread, nullptr, &thread_timeout) == ETIMEDOUT)
        debug(config.log, "%s(%d): Gave up on thread after %d seconds.\n", __FILE__, __LINE__, config.thread_timeout);
    }
    close_dec();
  }

  if (fargs) free(fargs);
  if (config.logfile) free(config.logfile);
  if (config.maskfile) free(config.maskfile);
  if (config.maxfile) free(config.maxfile);
  if (config.ptofile) free(config.ptofile);
  if (config.execute) free(config.execute);
  if (config.eventdir) free(config.eventdir);
  if (config.snapshot_dir) free(config.snapshot_dir);
  if (config.pano) delete config.pano;
  if (config.trafo) delete config.trafo;

  if (snapshot_running) {
    struct timespec thread_timeout;
    clock_gettime(CLOCK_REALTIME, &thread_timeout);
    thread_timeout.tv_sec += config.thread_timeout;
    pthread_timedjoin_np(snapshot_thread, nullptr, &thread_timeout);
  }
  exit(0);
}

static unsigned int size = 0;
static unsigned char *p = 0;

static const AVCodec *ffmpeg_codec;
static AVCodecContext *ffmpeg_context = nullptr;
static AVFrame *ffmpeg_frame;
static AVPacket ffmpeg_avpkt;

// Return the next H.264 frame from a file
unsigned char *readh264frame(unsigned int *l) {
  static int first = 2;
  static const unsigned int bufsize = 1024 * 1024;
  static const unsigned int chunksize = 4096;
  static unsigned char buf1_unaligned[bufsize + 15];
  static unsigned char buf2_unaligned[bufsize + 15];
  unsigned char *buf1 = (unsigned char *)((uintptr_t)(buf1_unaligned + 15) & ~15);
  unsigned char *buf2 = (unsigned char *)((uintptr_t)(buf2_unaligned + 15) & ~15);
  static int swap = 0;
  static unsigned int surplus = 0;
  static unsigned int surpluspos = 0;
  unsigned char *curr = (swap ? buf2 : buf1);
  unsigned char *other = swap ? buf1 : buf2;
  unsigned int readbytes = surplus;
  unsigned int pos = 0;

  // Copy extra data from last run
  if (surplus) memcpy(curr, other + surpluspos, surplus);

  do {
    int end, i;
    do {
      unsigned int r = 0;
      int len = chunksize * (1 + (surplus > chunksize)) - surplus;
      if (len < 0) len = 0;
      len += 16 - ((pos + surplus + len) & 15);  // Align end
      r = (unsigned int)fread(curr + pos + surplus, 1, len, inputfile);

      // Avoid start code at the end of the chunk
      while (!feof(inputfile) && v128_haszero_u8(v128_load_aligned(curr + pos + surplus + r - 16)))
        r += (unsigned int)fread(curr + pos + surplus + r, 1, 16, inputfile);

      if (!r && !surplus) break;
      surplus = 0;
      readbytes += r;
      if (!pos || first == 1) pos += 16;
      do {
        // Fast forward if there's no start code in sight
        if (!(pos & 15))
          while (pos <= readbytes - 16 && !v128_haszero_u8(v128_load_aligned(curr + pos))) pos += 16;

        end = ((pos + 16) & ~15);
        if (end > readbytes) end = readbytes;

        // Exact start code search
        pos--;
        while (pos < end &&
               (curr[pos + 0] || curr[pos + 1] || curr[pos + 2] != 1 || (curr[pos + 3] & 0x1f) != 5 && (curr[pos + 3] & 0x1f) != 1))
          pos++;
      } while (pos == end && pos < readbytes);
    } while (pos == end && pos < bufsize - chunksize);
    surplus = readbytes - pos;
  } while (first-- > 1);
  first = 0;
  if (pos >= bufsize - chunksize) return 0;

  // Start code found!
  *l = curr + pos - (swap ? buf2 : buf1);
  swap ^= 1;
  surpluspos = pos;

  return swap ? buf1 : buf2;
}

// Close ffmpeg decoder
int close_ffmpegdec() {
  if (!ffmpeg_open) return 0;
  avcodec_close(ffmpeg_context);
  av_free(ffmpeg_context);
  av_frame_free(&ffmpeg_frame);
  ffmpeg_open = 0;
  return 1;
}

// Replacement for depracated avcodec_decode_video2()
static int avcodec_decode(AVCodecContext *avctx, AVFrame *frame, int *got_frame, AVPacket *pkt) {
  int ret;

  // avctx->get_format = vaapi_get_format;
  // avctx->get_buffer2 = vaapi_get_buffer;

  *got_frame = 0;

  if (pkt) {
    ret = avcodec_send_packet(avctx, pkt);
    if (ret < 0) return ret == AVERROR_EOF ? 0 : ret;
  }

  ret = avcodec_receive_frame(avctx, frame);
  if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) return ret;
  if (ret >= 0) *got_frame = 1;

  return 0;
}

// Decode one frame from file.  Returns Y frame or 0 if end of file reached.
unsigned char *decode_ffmpeg(unsigned int *width, unsigned int *height, unsigned int *stride, unsigned char *h264, unsigned int size,
                             unsigned char **u, unsigned char **v) {
  int len = 0, got_frame;

  if (!ffmpeg_open) return 0;

  do {
    if (h264 && len >= 0) {
      ffmpeg_avpkt.data = h264;
      ffmpeg_avpkt.size = size;
    } else
      ffmpeg_avpkt.data = readh264frame((unsigned int *)&ffmpeg_avpkt.size);

    if (!ffmpeg_avpkt.size) return 0;

    len = avcodec_decode(ffmpeg_context, ffmpeg_frame, &got_frame, &ffmpeg_avpkt);
  } while (len < 0 || !ffmpeg_frame->data[0]);

  *stride = ffmpeg_frame->linesize[0];
  *width = ffmpeg_context->width;
  *height = ffmpeg_context->height;
  if (u) *u = ffmpeg_frame->data[1];
  if (v) *v = ffmpeg_frame->data[2];
  return ffmpeg_frame->data[0];
}

// Init ffmpeg decoder
int init_ffmpegdec(int single_thread) {
#if LIBAVCODEC_VERSION_MAJOR < 58
  avcodec_register_all();
#endif

  av_init_packet(&ffmpeg_avpkt);

#if 0  // This might find hw decoder and fail later
  ffmpeg_codec = avcodec_find_decoder(AV_CODEC_ID_H264);
#else  // Seems to select sw decoder
  ffmpeg_codec = avcodec_find_decoder_by_name("h264");
#endif
  if (!ffmpeg_codec) {
    fprintf(stderr, "Codec not found\n");
    return 0;
  }

  ffmpeg_context = avcodec_alloc_context3(ffmpeg_codec);
  if (!ffmpeg_context) {
    fprintf(stderr, "Could not allocate video codec context\n");
    return 0;
  }

  AVDictionary *dict = nullptr;
  if (!single_thread) av_dict_set(&dict, "threads", "4", 0);
  av_dict_set(&dict, "pix_fmt", "grey", 0);

  if (avcodec_open2(ffmpeg_context, ffmpeg_codec, &dict) < 0) {
    fprintf(stderr, "Could not open codec\n");
    avcodec_close(ffmpeg_context);
    return 0;
  }

  ffmpeg_frame = av_frame_alloc();
  if (!ffmpeg_frame) {
    fprintf(stderr, "Could not allocate video frame\n");
    avcodec_close(ffmpeg_context);
    av_free(ffmpeg_context);
    return 0;
  }
  ffmpeg_open = 1;
  return 1;
}

// Exynox MFC decoder.
// Based on code from <https://github.com/Owersun/mymfc> 2015-07-29.

// TODO: Strip down to what's strictly needed and convert to C99.

#include <dirent.h>
#include <errno.h>
#include <linux/media.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <queue>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>

#ifndef V4L2_CAP_VIDEO_M2M_MPLANE
#define V4L2_CAP_VIDEO_M2M_MPLANE 0x00004000
#endif

typedef struct {
  uint8_t *y;
  uint8_t *uv;
  unsigned int width;
  unsigned int height;
  unsigned int stride;
} picture;

typedef struct {
  int device;
} V4l2Device;

typedef struct {
  int iIndex;
  int iBytesUsed[4];
  void *cPlane[4];
} V4l2SinkBuffer;

class CLinuxV4l2Sink {
 public:
  CLinuxV4l2Sink(V4l2Device *device, enum v4l2_buf_type type);
  ~CLinuxV4l2Sink();

  bool Init(int buffersCount);
  bool GetFormat(v4l2_format *format);
  bool SetFormat(v4l2_format *format);
  bool GetCrop(v4l2_crop *crop);
  bool GetBuffer(V4l2SinkBuffer *buffer);
  bool DequeueBuffer(V4l2SinkBuffer *buffer);
  bool PushBuffer(V4l2SinkBuffer *buffer);
  bool StreamOn(int state);
  bool QueueAll();
  int Poll(int timeout);

 private:
  V4l2Device *device;
  int numplanes;
  std::queue<int> freebuffers;
  enum v4l2_memory memory;
  enum v4l2_buf_type type;
  v4l2_buffer *buffers;
  v4l2_plane *planes;
  unsigned long *addresses;
  bool QueueBuffer(v4l2_buffer *buffer);
};

#ifndef V4L2_CAP_VIDEO_M2M_MPLANE
#define V4L2_CAP_VIDEO_M2M_MPLANE 0x00004000
#endif

class CDVDVideoCodecMFC {
 public:
  CDVDVideoCodecMFC();
  virtual ~CDVDVideoCodecMFC();
  virtual bool decopen(unsigned char *h264, unsigned int size);
  virtual void decclose();
  virtual const picture *Decode(unsigned char *pData, int iSize);

 protected:
  V4l2Device *m_iDecoderHandle;
  CLinuxV4l2Sink *m_MFCCapture;
  CLinuxV4l2Sink *m_MFCOutput;
  V4l2SinkBuffer *m_Buffer;
  V4l2SinkBuffer *m_BufferNowOnScreen;
  picture m_videoBuffer;

  bool OpenDevices();
};

CDVDVideoCodecMFC::CDVDVideoCodecMFC() {
  m_iDecoderHandle = nullptr;
  m_MFCOutput = nullptr;
  m_MFCCapture = nullptr;

  m_Buffer = nullptr;
  m_BufferNowOnScreen = nullptr;

  memset(&m_videoBuffer, 0, sizeof(m_videoBuffer));
}

CDVDVideoCodecMFC::~CDVDVideoCodecMFC() { decclose(); }

bool CDVDVideoCodecMFC::OpenDevices() {
  DIR *dir;

  if ((dir = opendir("/sys/class/video4linux/")) != nullptr) {
    struct dirent *ent;
    while ((ent = readdir(dir)) != nullptr) {
      if (strncmp(ent->d_name, "video", 5) == 0) {
        char *p;
        char name[512];
        char devname[512];
        char sysname[512];
        char drivername[32];
        char target[1024];
        int ret;

        snprintf(sysname, sizeof(devname), "/sys/class/video4linux/%s", ent->d_name);
        snprintf(name, sizeof(sysname), "/sys/class/video4linux/%s/name", ent->d_name);

        FILE *fp = fopen(name, "r");
        if (fgets(drivername, 32, fp) != nullptr) {
          p = strchr(drivername, '\n');
          if (p != nullptr) *p = '\0';
        } else {
          fclose(fp);
          continue;
        }
        fclose(fp);

        ret = readlink(sysname, target, sizeof(target));
        if (ret < 0) continue;
        target[ret] = '\0';
        p = strrchr(target, '/');
        if (p == nullptr) continue;

        snprintf(devname, sizeof(devname), "/dev/%s", ++p);

        if (!m_iDecoderHandle && strstr(drivername, "mfc") != nullptr && strstr(drivername, "dec") != nullptr) {
          int fd = open(devname, O_RDWR | O_NONBLOCK, 0);
          if (fd > -1) {
            v4l2_capability cap;
            memset(&cap, 0, sizeof(cap));
            if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0)
              if (cap.capabilities & V4L2_CAP_STREAMING &&
                  (cap.capabilities & V4L2_CAP_VIDEO_M2M_MPLANE ||
                   (cap.capabilities & (V4L2_CAP_VIDEO_CAPTURE_MPLANE | V4L2_CAP_VIDEO_OUTPUT_MPLANE)))) {
                m_iDecoderHandle = new V4l2Device;
                m_iDecoderHandle->device = fd;
              }
          }
          if (!m_iDecoderHandle) close(fd);
        }
        if (m_iDecoderHandle) {
          closedir(dir);
          return true;
        }
      }
    }
    closedir(dir);
  }

  return false;
}

void CDVDVideoCodecMFC::decclose() {
  delete m_BufferNowOnScreen;
  delete m_Buffer;

  m_Buffer = nullptr;
  m_BufferNowOnScreen = nullptr;

  delete m_MFCCapture;
  delete m_MFCOutput;

  m_MFCOutput = nullptr;
  m_MFCCapture = nullptr;

  if (m_iDecoderHandle) {
    close(m_iDecoderHandle->device);
    delete m_iDecoderHandle;
    m_iDecoderHandle = nullptr;
  }
}

bool CDVDVideoCodecMFC::decopen(unsigned char *h264, unsigned int size) {
  v4l2_format fmt;
  struct v4l2_crop crop;
  V4l2SinkBuffer sinkBuffer;

  if (size >= 2048 * 1024) {
    fprintf(stderr, "Unable to decode stream\n");
    return false;
  }

  m_Buffer = new V4l2SinkBuffer();
  if (!m_Buffer) {
    fprintf(stderr, "Unable to get V4l2SinkBuffer\n");
    return false;
  }
  m_BufferNowOnScreen = new V4l2SinkBuffer();
  if (!m_BufferNowOnScreen) {
    fprintf(stderr, "Unable to get V4l2SinkBuffer\n");
    delete m_Buffer;
    return false;
  }
  m_BufferNowOnScreen->iIndex = -1;
  memset(&m_videoBuffer, 0, sizeof(m_videoBuffer));

  if (!OpenDevices()) {
    fprintf(stderr, "No Exynos MFC Decoder found\n");
    return false;
  }

  // Test NV12 2 Planes Y/CbCr
  memset(&fmt, 0, sizeof(fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12M;
  if (ioctl(m_iDecoderHandle->device, VIDIOC_TRY_FMT, &fmt)) {
    fprintf(stderr, "No suitable format to convert to found\n");
    return false;
  }

  // Create MFC Output sink
  m_MFCOutput = new CLinuxV4l2Sink(m_iDecoderHandle, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE);
  memset(&fmt, 0, sizeof(fmt));
  fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_H264;
  fmt.fmt.pix_mp.plane_fmt[0].sizeimage = 0;

  // Set encoded format
  if (!m_MFCOutput->SetFormat(&fmt)) return false;

  // Init with number of input buffers predefined
  if (!m_MFCOutput->Init(1)) return false;

  // Get empty buffer to fill
  if (!m_MFCOutput->GetBuffer(&sinkBuffer)) return false;

  // Fill it with the header
  sinkBuffer.iBytesUsed[0] = size;
  memcpy(sinkBuffer.cPlane[0], h264, size);

  // Enqueue buffer
  if (!m_MFCOutput->PushBuffer(&sinkBuffer)) return false;

  // Create MFC Capture sink (the one from which decoded frames are read)
  m_MFCCapture = new CLinuxV4l2Sink(m_iDecoderHandle, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE);
  memset(&fmt, 0, sizeof(fmt));

  fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12M;
  if (!m_MFCCapture->SetFormat(&fmt)) return false;

  // Turn on MFC Output with header in it to initialize MFC with all we just
  // setup
  m_MFCOutput->StreamOn(VIDIOC_STREAMON);

  // Initialize MFC Capture
  if (!m_MFCCapture->Init(0)) return false;

  // Queue all buffers (empty) to MFC Capture
  m_MFCCapture->QueueAll();

  // Read the format of MFC Capture
  if (!m_MFCCapture->GetFormat(&fmt) || !m_MFCCapture->GetCrop(&crop)) return false;

  // Turn on MFC Capture
  m_MFCCapture->StreamOn(VIDIOC_STREAMON);

  m_videoBuffer.width = crop.c.width;
  m_videoBuffer.height = crop.c.height;
  m_videoBuffer.stride = fmt.fmt.pix_mp.width;

  m_videoBuffer.y = m_videoBuffer.uv = nullptr;

  return true;
}

const picture *CDVDVideoCodecMFC::Decode(unsigned char *pData, int iSize) {
  if (pData) {
    int demuxer_bytes = iSize;
    uint8_t *demuxer_content = pData;
    m_MFCOutput->Poll(1000 / 3);  // Wait up to 0.3 of a second for buffer availability
    if (m_MFCOutput->GetBuffer(m_Buffer)) {
      m_Buffer->iBytesUsed[0] = demuxer_bytes;
      memcpy((uint8_t *)m_Buffer->cPlane[0], demuxer_content, m_Buffer->iBytesUsed[0]);

      if (!m_MFCOutput->PushBuffer(m_Buffer)) return 0;

    } else {
      if (errno == EAGAIN)
        fprintf(stderr,
                "MFC OUTPUT All buffers are queued and busy, no space "
                "for new frame to decode. Very broken situation. "
                "Current encoded frame will be lost\n");
      else {
        return 0;
      }
    }
  }

  // Get a buffer from MFC Capture
  if (!m_MFCCapture->DequeueBuffer(m_Buffer)) {
    return 0;
  }

  // We got a new buffer to show, so we can enqeue back the buffer wich was on
  // screen
  if (m_BufferNowOnScreen->iIndex > -1) {
    m_MFCCapture->PushBuffer(m_BufferNowOnScreen);
    m_BufferNowOnScreen->iIndex = -1;
  }

  m_videoBuffer.y = (unsigned char *)m_Buffer->cPlane[0];
  m_videoBuffer.uv = (unsigned char *)m_Buffer->cPlane[1];

  std::swap(m_Buffer, m_BufferNowOnScreen);

  return &m_videoBuffer;
}

CLinuxV4l2Sink::CLinuxV4l2Sink(V4l2Device *d, enum v4l2_buf_type t) {
  device = d;
  type = t;
  numplanes = 0;
  addresses = nullptr;
  buffers = nullptr;
  planes = nullptr;
}

CLinuxV4l2Sink::~CLinuxV4l2Sink() {
  StreamOn(VIDIOC_STREAMOFF);

  if (planes) delete[] planes;
  if (buffers) delete[] buffers;
  if (addresses) delete[] addresses;
}

// Init for MMAP buffers
bool CLinuxV4l2Sink::Init(int buffersCount = 0) {
  memory = V4L2_MEMORY_MMAP;

  struct v4l2_format format;
  if (!GetFormat(&format)) return false;

  if (buffersCount == 0 && type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
    struct v4l2_control ctrl;
    ctrl.id = V4L2_CID_MIN_BUFFERS_FOR_CAPTURE;
    if (ioctl(device->device, VIDIOC_G_CTRL, &ctrl)) {
      fprintf(stderr,
              "Type %d, Error getting number of buffers for capture "
              "(V4L2_CID_MIN_BUFFERS_FOR_CAPTURE VIDIOC_G_CTRL)\n",
              type);
      return false;
    }
    buffersCount = ctrl.value + 1;  // One extra buffer
  }

  struct v4l2_requestbuffers reqbuf;
  memset(&reqbuf, 0, sizeof(struct v4l2_requestbuffers));
  reqbuf.type = type;
  reqbuf.memory = memory;
  reqbuf.count = buffersCount;

  if (ioctl(device->device, VIDIOC_REQBUFS, &reqbuf)) {
    fprintf(stderr, "Error requesting buffers. Type %d, Memory %d. (VIDIOC_REQBUFS)\n", type, memory);
    return false;
  }

  buffers = new v4l2_buffer[reqbuf.count];
  planes = new v4l2_plane[numplanes * reqbuf.count];
  addresses = new unsigned long[numplanes * reqbuf.count];

  memset(buffers, 0, reqbuf.count * sizeof(struct v4l2_buffer));
  memset(planes, 0, reqbuf.count * numplanes * sizeof(struct v4l2_plane));

  for (int i = 0; i < reqbuf.count; i++) {
    buffers[i].type = type;
    buffers[i].memory = memory;
    buffers[i].index = i;
    buffers[i].m.planes = &planes[i * numplanes];
    buffers[i].length = numplanes;

    if (ioctl(device->device, VIDIOC_QUERYBUF, &buffers[i])) {
      fprintf(stderr, "Error querying buffers. Type %d, Memory %d. (VIDIOC_QUERYBUF)\n", type, memory);
      return false;
    }

    freebuffers.push(buffers[i].index);
  }

  for (int i = 0; i < reqbuf.count * numplanes; i++) {
    if (planes[i].length) {
      addresses[i] =
          (unsigned long)mmap(nullptr, planes[i].length, PROT_READ | PROT_WRITE, MAP_SHARED, device->device, planes[i].m.mem_offset);
      if (addresses[i] == (unsigned long)MAP_FAILED) return false;
    }
  }

  return true;
}

bool CLinuxV4l2Sink::GetFormat(v4l2_format *format) {
  memset(format, 0, sizeof(struct v4l2_format));
  format->type = type;
  if (ioctl(device->device, VIDIOC_G_FMT, format)) {
    fprintf(stderr, "Error getting sink format. Type %d. (VIDIOC_G_FMT)\n", type);
    return false;
  }
  numplanes = format->fmt.pix_mp.num_planes;
  return true;
}

bool CLinuxV4l2Sink::SetFormat(v4l2_format *format) {
  format->type = type;
  if (ioctl(device->device, VIDIOC_S_FMT, format)) {
    fprintf(stderr, "Error setting sink format. Type %d. (VIDIOC_G_FMT)\n", type);
    return false;
  }
  return true;
}

bool CLinuxV4l2Sink::GetCrop(v4l2_crop *crop) {
  memset(crop, 0, sizeof(struct v4l2_crop));
  crop->type = type;
  if (ioctl(device->device, VIDIOC_G_CROP, crop)) {
    fprintf(stderr, "Error getting sink crop. Type %d. (VIDIOC_G_CROP)\n", type);
    return false;
  }
  return true;
}

bool CLinuxV4l2Sink::StreamOn(int state) {
  if (ioctl(device->device, state, &type)) {
    fprintf(stderr, "Error setting device state to %d, Type %d.\n", state, type);
    return false;
  }
  return true;
}

bool CLinuxV4l2Sink::QueueBuffer(v4l2_buffer *buffer) {
  if (ioctl(device->device, VIDIOC_QBUF, buffer)) {
    fprintf(stderr, "Error queueing buffer. Type %d, Memory %d. Buffer %d, errno %d\n", buffer->type, buffer->memory, buffer->index, errno);
    return false;
  }
  return true;
}

bool CLinuxV4l2Sink::DequeueBuffer(V4l2SinkBuffer *buffer) {
  struct v4l2_buffer buf;
  struct v4l2_plane planes[numplanes];
  memset(&planes, 0, sizeof(struct v4l2_plane) * numplanes);
  memset(&buf, 0, sizeof(struct v4l2_buffer));
  buf.type = type;
  buf.memory = memory;
  buf.m.planes = planes;
  buf.length = numplanes;

  if (ioctl(device->device, VIDIOC_DQBUF, &buf)) {
    if (errno != EAGAIN)
      fprintf(stderr, "Error dequeueing buffer. Type %d, Memory %d. Buffer %d, errno %d\n", buf.type, buf.memory, buf.index, errno);
    return false;
  }

  buffer->iIndex = buf.index;
  for (int i = 0; i < numplanes; i++) buffer->cPlane[i] = (void *)addresses[buffer->iIndex * numplanes + i];
  return true;
}

bool CLinuxV4l2Sink::GetBuffer(V4l2SinkBuffer *buffer) {
  if (freebuffers.empty()) {
    if (!DequeueBuffer(buffer)) return false;
  } else {
    buffer->iIndex = freebuffers.front();
    freebuffers.pop();
    for (int i = 0; i < numplanes; i++) buffer->cPlane[i] = (void *)addresses[buffer->iIndex * numplanes + i];
  }
  return true;
}

bool CLinuxV4l2Sink::PushBuffer(V4l2SinkBuffer *buffer) {
  if (memory == V4L2_MEMORY_USERPTR)
    for (int i = 0; i < numplanes; i++) buffers[buffer->iIndex].m.planes[i].m.userptr = (long unsigned int)buffer->cPlane[i];

  if (type == V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE)
    for (int i = 0; i < numplanes; i++) buffers[buffer->iIndex].m.planes[i].bytesused = buffer->iBytesUsed[i];

  return QueueBuffer(&buffers[buffer->iIndex]);
}

int CLinuxV4l2Sink::Poll(int timeout) {
  struct pollfd p;
  p.fd = device->device;
  p.events = POLLERR;
  (type == V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE) ? p.events |= POLLOUT : p.events |= POLLIN;

  return poll(&p, 1, timeout);
}

bool CLinuxV4l2Sink::QueueAll() {
  while (!freebuffers.empty()) {
    if (!QueueBuffer(&buffers[freebuffers.front()])) return false;
    freebuffers.pop();
  }
  return true;
}

// ---------------------

static CDVDVideoCodecMFC *codec = 0;

// Close MFC decoder
int close_mfcdec() {
  if (!mfc_open) return false;
  if (codec) delete codec;
  mfc_open = 0;
  return true;
}

// Decode one frame from file.  Returns Y frame or 0 if end of file reached.
unsigned char *decode_mfc(unsigned int *width, unsigned int *height, unsigned int *stride, unsigned char *h264, unsigned int s,
                          unsigned char **u, unsigned char **v) {
  const picture *pic;

  if (!mfc_open) return 0;

  do {
    pic = codec->Decode(p, size);

    if (h264) {
      p = h264;
      size = s;
    } else {
      do p = readh264frame(&size);
      while (size && size > 250000);  // Large frames might crash the mfc decoder
    }
  } while (!pic || !pic->y && size);
  *width = pic->width;
  *stride = pic->stride;
  *height = pic->height;
  if (u) *u = size ? pic->uv : 0;
  if (v) *v = 0;
  return size ? pic->y : 0;
}

// Init MFC decoder
int init_mfcdec(unsigned char *h264, unsigned int size) {
  codec = new CDVDVideoCodecMFC();

  if (!codec->decopen(h264, size)) {
    close_mfcdec();
    size = 0;
    mfc_open = 0;
    return 0;
  }
  mfc_open = 1;
  return 1;
}
