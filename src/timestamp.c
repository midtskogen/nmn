#define _GNU_SOURCE
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

const char *get_filename_ext(const char *filename) {
  const char *dot = strrchr(filename, '.');
  if (!dot || dot == filename) return "";
  return ++dot;
}

typedef enum { ALL, IP8172, IP816A, IP9171, IP8151, IMX291, IMX291SD, IMX291HD, IMX307, IMX307SD, IMX307HD } camid;

static camid camera = ALL;

static time_t gettimestamp(unsigned char *img, int width, int height) {
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

  // printf("%s\n", text);

  tm.tm_isdst = 0;
  return mktime(&tm) + timegm(&tm) - timelocal(&tm);
}

int parseopts(int argc, char **argv) {
  int c;
  int count = 0;

  while ((c = getopt(argc, argv, "M:")) != -1) {
    switch (c) {
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
        count += 2;
        break;
      case '?':
        if (optopt == 'w')
          fprintf(stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint(optopt))
          fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
        count++;
        return count;
      default: break;
    }
  }
  return count;
}

int main(int argc, char **argv) {
  FILE *fp;
  unsigned int width = 576;
  unsigned int height = 448;
  uint8_t img2[width * height];
  char command[1024];

  int args = 1 + parseopts(argc, argv);
  snprintf(command, sizeof(command),
           "ffmpeg -i %s -vsync 0 -nostdin -vframes 1 -loglevel quiet -vf "
           "crop=%d:%d:0:9999 -pix_fmt gray -f rawvideo -",
           argv[args], width, height);

  fp = popen(command, "r");
  if (fp == NULL) {
    printf("Failed to run command\n");
    exit(1);
  }

  /* Read the output */
  if (fread(img2, sizeof(img2), 1, fp) != 1) {
    printf("Short read\n");
    exit(1);
  }

  printf("%d\n", (unsigned int)gettimestamp(img2, width, height));

  /* close */
  pclose(fp);

  return 0;
}
