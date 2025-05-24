#define _GNU_SOURCE
#include <jpeglib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define ABS(a) ((a) < 0 ? -(a) : (a))
#define CLIP(a) ((a) > 255 ? 255 : (a))

unsigned int coeffs[] = { 1, 12, 66, 220, 495, 792, 924, 792, 495, 220, 66, 12, 1 };

static inline int clip(int v, int min, int max) { return v < min ? min : v > max ? max : v; }

static void blur_h(unsigned char *image, int width, int height) {
  unsigned char tmp[width];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      tmp[x] = (image[clip(x - 6, 0, width - 1) + y * width] * coeffs[0] + image[clip(x - 5, 0, width - 1) + y * width] * coeffs[1] +
                image[clip(x - 4, 0, width - 1) + y * width] * coeffs[2] + image[clip(x - 3, 0, width - 1) + y * width] * coeffs[3] +
                image[clip(x - 2, 0, width - 1) + y * width] * coeffs[4] + image[clip(x - 1, 0, width - 1) + y * width] * coeffs[5] +
                image[clip(x + 0, 0, width - 1) + y * width] * coeffs[6] + image[clip(x + 1, 0, width - 1) + y * width] * coeffs[7] +
                image[clip(x + 2, 0, width - 1) + y * width] * coeffs[8] + image[clip(x + 3, 0, width - 1) + y * width] * coeffs[9] +
                image[clip(x + 4, 0, width - 1) + y * width] * coeffs[10] + image[clip(x + 5, 0, width - 1) + y * width] * coeffs[11] +
                image[clip(x + 6, 0, width - 1) + y * width] * coeffs[12]) >>
               12;
    }
    memcpy(image + y * width, tmp, width);
  }
}

static void blur_v(unsigned char *image, int width, int height) {
  unsigned char tmp[height];
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      tmp[y] = (image[clip(y - 6, 0, height - 1) * width + x] * coeffs[0] + image[clip(y - 5, 0, height - 1) * width + x] * coeffs[1] +
                image[clip(y - 4, 0, height - 1) * width + x] * coeffs[2] + image[clip(y - 3, 0, height - 1) * width + x] * coeffs[3] +
                image[clip(y - 2, 0, height - 1) * width + x] * coeffs[4] + image[clip(y - 1, 0, height - 1) * width + x] * coeffs[5] +
                image[clip(y + 0, 0, height - 1) * width + x] * coeffs[6] + image[clip(y + 1, 0, height - 1) * width + x] * coeffs[7] +
                image[clip(y + 2, 0, height - 1) * width + x] * coeffs[8] + image[clip(y + 3, 0, height - 1) * width + x] * coeffs[9] +
                image[clip(y + 4, 0, height - 1) * width + x] * coeffs[10] + image[clip(y + 5, 0, height - 1) * width + x] * coeffs[11] +
                image[clip(y + 6, 0, height - 1) * width + x] * coeffs[12]) >>
               12;
    }
    for (int y = 0; y < height; y++) image[y * width + x] = tmp[y];
  }
}

static void savejpeg(FILE *fp, unsigned char *image, int width, int height, int quality) {
  int i, j;

  JSAMPROW y[16], cb[16], cr[16];
  JSAMPARRAY data[3];

  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;

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
      y[i] = image + width * (i + j);
      if (i % 2 == 0) {
        cb[i / 2] = image + width * height + width / 2 * ((i + j) / 2);
        cr[i / 2] = image + width * height + width * height / 4 + width / 2 * ((i + j) / 2);
      }
    }
    jpeg_write_raw_data(&cinfo, data, 16);
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
}

int main(int argc, char **argv) {
  FILE *i = stdin;
  FILE *o = stdout;
  unsigned char *buf1;
  unsigned char *buf2;
  unsigned char *buf3;
  const int x = atoi(argv[1]);
  const int y = atoi(argv[2]);
  const unsigned char *filename = argc == 4 ? argv[3] : 0;

  buf1 = aligned_alloc(16, x * y * 1.5);
  buf2 = aligned_alloc(16, x * y * 1.5);
  buf3 = aligned_alloc(16, x * y);

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

    memcpy(buf3, buf1, x * y);
    blur_h(buf3, x, y);
    blur_v(buf3, x, y);

    int avg = 0;
    for (int a = 0; a < x * y; a++) avg += buf3[a];
    avg /= x * y;

    for (int a = 0; a < x * y; a++) buf1[a] = buf1[a] > buf3[a] + avg ? buf1[a] - buf3[a] : 0;

    for (int a = 0; a < x * y; a++) buf2[a] = CLIP(buf2[a] + buf1[a] * (256 - buf2[a]) / 256);
  }

  savejpeg(o, buf2, x, y, 100);

  return 0;
}
