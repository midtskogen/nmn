#define _GNU_SOURCE
#include <jpeglib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define CLIP(a) ((a) < 0 ? 0 : (a))

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
  unsigned int *buf3;
  const int x = atoi(argv[1]);
  const int y = atoi(argv[2]);
  const char *filename = argc == 4 ? argv[3] : 0;

  buf1 = aligned_alloc(16, x * y * 1.5);
  buf2 = aligned_alloc(16, x * y * 1.5);
  buf3 = aligned_alloc(16, x * y * 1.5 * sizeof(unsigned int));

  if (!buf1 || !buf2 || !buf3) {
    printf("%s: Memory error\n", argv[0]);
    return -1;
  }

  memset(buf2, 0, x * y);
  memset(buf2 + x * y, 128, x * y / 2);
  memset(buf3, 0, x * y * 1.5 * sizeof(unsigned int));

  if (!i) {
    printf("%s: File error\n", argv[0]);
    return -1;
  }

  int frames = 0;
  while (!feof(i)) {
    int s = fread(buf1, x * y * 1.5, 1, i);
    if (s != 1) break;

    for (int a = 0; a < x * y * 1.5; a++) buf3[a] += buf1[a];
    frames++;
  }

  for (int a = 0; a < x * y * 1.5; a++) buf2[a] = CLIP((int)buf3[a] / frames - 3);

  savejpeg(o, buf2, x, y, 100);

  return 0;
}
