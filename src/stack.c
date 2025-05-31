#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdint.h>
#include <stdio.h>
#include <avif/avif.h>
#include <jpeglib.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#if __ARM_NEON__
#include "arm_neon.h"
#endif

// gcc -msse4.2 -O3 -o bin/stack src/stack.c -ljpeg
// gcc -march=armv8-a+crc -O3 -o bin/stack src/stack.c -ljpeg
// gcc -march=armv8-a+crc -mfpu=neon -O3 -o bin/stack src/stack.c -ljpeg

#define ABS(a) ((a) < 0 ? -(a) : (a))

void *aligned_malloc(size_t size, uintptr_t align) {
  void *m = malloc(size + sizeof(void *) + align);
  void **r = (void **)((((uintptr_t)m) + sizeof(void *) + align - 1) & ~(align - 1));
  r[-1] = m;
  return r;
}

void aligned_free(void *p) { free(((void **)p)[-1]); }

inline unsigned int log2i(uint32_t x) { return 31 - __builtin_clz(x); }
#if defined(__aarch64__) || defined(__arm__)
#include <arm_acle.h>
#include <arm_neon.h>
static inline uint32_t u32_crc_u16(uint32_t crc, uint16_t v) { return __crc32ch(crc, v); }
#endif
#ifdef __SSE4_1__
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
inline uint32_t u32_crc_u16(uint32_t crc, uint16_t v) { return (uint32_t)_mm_crc32_u16(crc, v); }
#endif

#define ALIGN(c) __attribute__((aligned(c)))
#define NOISE_SIZE 2048
#define imax(a, b) ((a) > (b) ? (a) : (b))
#define imin(a, b) ((a) < (b) ? (a) : (b))
#define clip(n, low, high) imin((high), imax((n), (low)))

// Adaptive denoise filter.  t = threshold.  log2sizex/y = filter width.
int enhance(uint8_t *restrict image, int width, int height, int t, int log2sizex, int log2sizey, unsigned int dither, uint32_t seed) {
  log2sizex = log2sizex < 3 ? 3 : log2sizex;
  log2sizey = log2sizey < 3 ? 3 : log2sizey;
  log2sizex = log2sizex > 6 ? 6 : log2sizex;
  log2sizey = log2sizey > 6 ? 6 : log2sizey;
  int sizex = 1 << log2sizex;
  int sizey = 1 << log2sizey;
  int size2x = sizex >> 1;
  int size2y = sizey >> 1;
  int shiftx = 6 - log2sizex;
  int shifty = 6 - log2sizey;
  int mask = sizey - 1;
  int i = 0;
  uint16_t *tmp = malloc(width * sizey * sizeof(uint16_t));
  if (tmp == NULL) return 0;
  if (dither < 2) dither = 2;
  dither -= 2;
  if (dither > 11) dither = 11;
  seed |= !seed;
  int dmask = (1 << dither) - 1;
  int doffset = dither ? (1 << (dither - 1)) - 8 : -8;
  ALIGN(32) int16_t noise[NOISE_SIZE + 32];
  for (unsigned int i = 0; i < sizeof(noise) / sizeof(*noise); i++) {
    seed = u32_crc_u16(seed, seed >> 16);
    noise[i] = (int16_t)((seed & dmask) - doffset);
  }

  static int indices[] = { -31, -23, -14, -5, 5, 14, 23, 31 };
  int log2indices = log2i(sizeof(indices) / sizeof(*indices));
  while (i < height + size2y - 1) {
    // Filter horizontally
    int ii = i & mask;
    if (i < height) {
      for (int j = 0; j < width; j++) {
        if (j < size2x || j >= width - size2x) {
          tmp[ii * width + j] = image[i * width + j] << log2indices;
        } else {
          int x = 0;
          for (unsigned int d = 0; d < sizeof(indices) / sizeof(*indices); d++) {
            int l = indices[d] / (1 << shiftx);
            int o = image[i * width + j];
            int v = image[i * width + j + l];
            if (v > o + t || v < o - t) v = o;
            x += v;
          }
          tmp[ii * width + j] = x;
        }
      }
    }

    // Filter vertically
    ii = ++i - size2y;
    if (ii >= 0) {
      if (ii < size2y || i >= height) {
        for (int j = 0; j < width; j++) {
          if (!(j & 31)) seed = u32_crc_u16(seed, seed >> 16);
          int r = noise[(seed & (NOISE_SIZE - 1)) + (j & 31)];
          image[ii * width + j] = clip(((tmp[(ii & mask) * width + j] << log2indices) + r) >> (2 * log2indices), 0, 255);
        }
      } else
        for (int j = 0; j < width; j++) {
          if (ii >= size2y && ii < height - size2y) {
            int x = 0;
            for (unsigned int d = 0; d < sizeof(indices) / sizeof(*indices); d++) {
              int l = indices[d] / (1 << shifty);
              int o = tmp[(ii & mask) * width + j];
              int v = tmp[((ii + l) & mask) * width + j];
              if (v > imin(o + t * 4, 65535) || v < imax(o - t * 4, 0)) v = o;
              x += v;
            }
            if (!(j & 31)) seed = u32_crc_u16(seed, seed >> 16);
            int r = noise[(seed & (NOISE_SIZE - 1)) + (j & 31)];
            image[ii * width + j] = clip((x + r) >> (2 * log2indices), 0, 255);
          }
        }
    }
  }
  free(tmp);
  return 1;
}

static void saveavif(FILE *f, unsigned char *image_data, int width, int height, int quality) {
  // Create an avifImage
  avifImage *image = avifImageCreate(width, height, 8, AVIF_PIXEL_FORMAT_YUV420);
  if (!image) {
    fprintf(stderr, "Failed to create avifImage\n");
    return;
  }

  enhance(image_data, width, height, 8, 5, 5, 6, rand());
  enhance(image_data + width * height, width / 2, height / 2, 16, 4, 4, 0, 0);
  enhance(image_data + width * height + width + height / 4, width / 2, height / 2, 16, 4, 4, 0, 0);

  // Set the image planes
  image->yuvPlanes[AVIF_CHAN_Y] = image_data;
  image->yuvPlanes[AVIF_CHAN_U] = image_data + width * height;
  image->yuvPlanes[AVIF_CHAN_V] = image_data + width * height + (width / 2) * (height / 2);
  image->yuvRowBytes[AVIF_CHAN_Y] = width;
  image->yuvRowBytes[AVIF_CHAN_U] = width / 2;
  image->yuvRowBytes[AVIF_CHAN_V] = width / 2;

  // Create an avifEncoder
  avifEncoder *encoder = avifEncoderCreate();
  if (!encoder) {
    fprintf(stderr, "Failed to create avifEncoder\n");
    avifImageDestroy(image);
    return;
  }

  // Set the encoder quality
  encoder->quality = quality;
  encoder->speed = 7;

  // Encode the image
  avifRWData output = AVIF_DATA_EMPTY;
  avifResult result = avifEncoderWrite(encoder, image, &output);
  if (result != AVIF_RESULT_OK) {
    fprintf(stderr, "Failed to encode image: %s\n", avifResultToString(result));
    avifRWDataFree(&output);
    avifEncoderDestroy(encoder);
    avifImageDestroy(image);
    return;
  }

  // Write the encoded data to the file
  fwrite(output.data, 1, output.size, f);

  // Clean up
  avifRWDataFree(&output);
  avifEncoderDestroy(encoder);
  avifImageDestroy(image);
}

static void savejpeg(FILE *fp, unsigned char *image, int width, int height, int quality) {
  int i, j;

  JSAMPROW y[16], cb[8], cr[8];
  JSAMPARRAY data[3];

  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;

  enhance(image, width, height, 8, 5, 5, 6, rand());
  enhance(image + width * height, width / 2, height / 2, 16, 4, 4, 0, 0);
  enhance(image + width * height + width + height / 4, width / 2, height / 2, 16, 4, 4, 0, 0);

  data[0] = y;
  data[1] = cb;
  data[2] = cr;

  cinfo.err = jpeg_std_error(&jerr);

  jpeg_create_compress(&cinfo);
  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = 3;
  jpeg_set_defaults(&cinfo);

  jpeg_set_colorspace(&cinfo, JCS_YCbCr);

  cinfo.raw_data_in = TRUE;
  cinfo.do_fancy_downsampling = FALSE;
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
      if (j + i < height) {
        y[i] = image + width * (i + j);
        if (i % 2 == 0) {
          cb[i / 2] = image + width * height + width / 2 * ((i + j) / 2);
          cr[i / 2] = image + width * height + width * height / 4 + width / 2 * ((i + j) / 2);
        }
      } else {
        y[i] = y[i - 1];  // Repeat the last valid row
        if (i % 2 == 0) {
          cb[i / 2] = cb[i / 2 - 1];
          cr[i / 2] = cr[i / 2 - 1];
        }
      }
    }
    jpeg_write_raw_data(&cinfo, data, 16);
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
}

int has_extension(const char *filename, const char *extension) {
  if (filename == NULL || extension == NULL) return 0;

  // Find the last occurrence of the dot in the filename
  const char *dot = strrchr(filename, '.');
  if (!dot || dot == filename) return 0;

  // Compare the extension
  return strcmp(dot, extension) == 0;
}

int main(int argc, char **argv) {
  FILE *i = stdin;
  FILE *o = argc == 4 ? fopen(argv[3], "wb") : stdout;
  unsigned char *buf1;
  unsigned char *buf2;
  const int x = atoi(argv[1]);
  const int y = atoi(argv[2]);

  srand(time(NULL));
  buf1 = (unsigned char *)aligned_malloc(x * y * 1.5, 16);
  buf2 = (unsigned char *)aligned_malloc(x * y * 1.5, 16);

  if (!buf1 || !buf2) {
    printf("%s: Memory error\n", argv[0]);
    return -1;
  }

  memset(buf2, 0, x * y);
  memset(buf2 + x * y, 128, x * y / 2);

  if (!i) {
    printf("%s: File error\n", argv[0]);
    return -1;
  }

  while (!feof(i)) {
    int s = fread(buf1, x * y * 1.5, 1, i);
    if (s != 1) break;

    for (int a = 0; a < y; a++)
#if __ARM_NEON__a
      for (int b = 0; b < x; b += 16) {
        int8x16_t v1 = vld1q_u8(buf1 + a * x + b);
        int8x16_t v2 = vld1q_u8(buf2 + a * x + b);
        int8x16_t m = vmaxq_u8(v1, v2);
        int8x16_t c = vqsubq_u8(m, v2);
        if (vget_high_s64(c) | vget_low_s64(c)) {
          vst1q_u8(buf2 + a * x + b, m);
          for (int c = b / 2; c < b / 2 + 8; c++) {
            if (ABS(buf1[x * y + a / 2 * x / 2 + c] - 128) + ABS(buf1[x * y + x / 2 * y / 2 + a / 2 * x / 2 + c] - 128) >
                ABS(buf2[x * y + a / 2 * x / 2 + c] - 128) + ABS(buf2[x * y + x / 2 * y / 2 + a / 2 * x / 2 + c] - 128)) {
              buf2[x * y + a / 2 * x / 2 + c] = buf1[x * y + a / 2 * x / 2 + c];
              buf2[x * y + x / 2 * y / 2 + a / 2 * x / 2 + c] = buf1[x * y + x / 2 * y / 2 + a / 2 * x / 2 + c];
            }
          }
        }
      }
#else
      for (int b = 0; b < x; b++) {
        if (buf1[a * x + b] > buf2[a * x + b]) {
          buf2[a * x + b] = buf1[a * x + b];
          if (ABS(buf1[x * y + a / 2 * x / 2 + b / 2] - 128) + ABS(buf1[x * y + x / 2 * y / 2 + a / 2 * x / 2 + b / 2] - 128) >
              ABS(buf2[x * y + a / 2 * x / 2 + b / 2] - 128) + ABS(buf2[x * y + x / 2 * y / 2 + a / 2 * x / 2 + b / 2] - 128)) {
            buf2[x * y + a / 2 * x / 2 + b / 2] = buf1[x * y + a / 2 * x / 2 + b / 2];
            buf2[x * y + x / 2 * y / 2 + a / 2 * x / 2 + b / 2] = buf1[x * y + x / 2 * y / 2 + a / 2 * x / 2 + b / 2];
          }
        }
      }
#endif
  }

  if (argc == 4 && has_extension(argv[3], ".avif"))
    saveavif(o, buf2, x, y, 60);
  else
    savejpeg(o, buf2, x, y, 80);

  aligned_free(buf1);
  aligned_free(buf2);
  return 0;
}
