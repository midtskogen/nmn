#include <stdio.h>
#include <stdint.h>
#include <jpeglib.h>
#include <setjmp.h>
#include <fstream>
#include "Panorama.h"
#include "PanoToolsInterface.h"
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
}

/* Compile:
 wget http://launchpadlibrarian.net/678162060/libboost-filesystem1.81.0_1.81.0-6_amd64.deb && sudo dpkg -i libboost-filesystem1.81.0_1.81.0-6_amd64.deb && rm libboost-filesystem1.81.0_1.81.0-6_amd64.deb

 cd src
 wget https://sourceforge.net/projects/hugin/files/hugin/hugin-2022.0/hugin-2022.0.0.tar.bz2
 tar xvf hugin-2022.0.0.tar.bz2
 mkdir hugin-2022.0.0/build
 cd hugin-2022.0.0/build
 cmake ..
 cd -
 g++ -march=native -g -Wno-deprecated-declarations -o reproject reproject.c -Isrc/hugin-2022.0.0/build/src -Isrc/hugin-2022.0.0/src/hugin_base/panotools -Isrc/hugin-2021.0.0/build/src -Isrc/hugin-2022.0.0/src/hugin_base -Isrc/hugin-2022.0.0/src/hugin_base/panodata -Isrc/hugin-2022.0.0/src -O6 -L/usr/local/lib -ljpeg -pthread -lavcodec -lz -llzma -lavformat -lswscale -lavutil -lavfilter -lswresample -lavdevice -lbz2 -lx264 -lx265 -Wl,-rpath=/usr/local/lib/hugin /usr/lib/hugin/libhuginbase.so.0.0

# g++ -mavx2 -g -Wno-deprecated-declarations -o bin/reproject src/reproject.c -Isrc/hugin -O6 -L/usr/local/lib -ljpeg -pthread -lavcodec -lz -llzma -lavformat -lswscale -lavutil -lavfilter -lswresample -lavdevice -lbz2 -lx264 -lx265 -Wl,-rpath=/usr/local/lib/hugin /usr/lib/hugin/libhuginbase.so.0.0

# g++ -mavx2 -g -Wno-deprecated-declarations -o bin/reproject src/reproject.c -Isrc/hugin -O6 -ljpeg -pthread -lavcodec -lz -llzma -lavformat -lswscale -lavutil -lavfilter -lswresample -lavdevice -lbz2 -lx264 -lx265 -L/usr/lib/x86_64-linux-gnu/ -Wl,-rpath=/usr/lib/hugin /usr/lib/hugin/libhuginbase.so.0.0

Examples:
 bin/reproject gnomonic.pto gaustatoppen-20230829025919.mp4 out.mp4
 bin/reproject gnomonic.pto gaustatoppen-20230829025919.jpg out.jpg
*/


#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#include <arm_acle.h>
static inline uint32_t u32_crc_u16(uint32_t crc, uint16_t v) { return __crc32ch(crc, v); }
#endif
#ifdef __SSE4_1__
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
inline uint32_t u32_crc_u16(uint32_t crc, uint16_t v) { return (uint32_t)_mm_crc32_u16(crc, v); }
#endif

inline unsigned int log2i(uint32_t x) { return 31 - __builtin_clz(x); }
#define ALIGN(c) __attribute__((aligned(c)))
#define NOISE_SIZE 2048
#define imax(a, b) ((a) > (b) ? (a) : (b))
#define imin(a, b) ((a) < (b) ? (a) : (b))
#define clip(n, low, high) imin((high), imax((n), (low)))

// Adaptive denoise filter.  t = threshold.  log2sizex/y = filter width.
int enhance(uint8_t *image, int width, int height, int t, int log2sizex, int log2sizey, unsigned int dither, uint32_t seed) {
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
  uint16_t *tmp = (uint16_t*)malloc(width * sizey * sizeof(uint16_t));
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

static void *aligned_malloc(size_t size, uintptr_t align)
{
  void *m = malloc(size + sizeof(void*) + align);
  if (!m)
    return m;
  void **r = (void**)((((uintptr_t)m) + sizeof(void*) + align - 1) & ~(align - 1));
  r[-1] = m;
  return r;
}

static void aligned_free(void *p)
{
  free(((void**)p)[-1]);
}

typedef struct {
  /* "public" fields */
  struct jpeg_error_mgr pub;
  /* for return to caller */
  jmp_buf setjmp_buffer;
} jpegErrorManager;

static char jpegLastErrorMsg[JMSG_LENGTH_MAX];
static void jpegErrorExit (j_common_ptr cinfo)
{
  /* cinfo->err actually points to a jpegErrorManager struct */
  jpegErrorManager* myerr = (jpegErrorManager*) cinfo->err;
  /* note : *(cinfo->err) is now equivalent to myerr->pub */

  /* output_message is a method to print an error message */
  /*(* (cinfo->err->output_message) ) (cinfo);*/      

  /* Create the message */
  ( *(cinfo->err->format_message) ) (cinfo, jpegLastErrorMsg);

  /* Jump to the setjmp point */
  longjmp(myerr->setjmp_buffer, 1);
}

static int savejpeg(FILE *fp, unsigned char *image, int width, int height, int quality) {
  int i, j;

  JSAMPROW y[16],cb[16],cr[16];
  JSAMPARRAY data[3];
  unsigned char *buf = 0;
  if (width & 15) {
    int width2 = (width + 15) & ~15;
    buf = (unsigned char*)aligned_malloc(width * ((height + 15) & ~15) * 1.5, 16);
    memset(buf, 0, width * ((height + 15) & ~15) * 1.5);
    if (buf) {
      for (int i = 0; i < height; i++)
        memcpy(buf + i * width, image + i * width2, width);
      memset(buf + width * height, 128, width * height / 2);
    }
    else
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
    if (buf)
      aligned_free(buf);
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

  if (buf)
    aligned_free(buf);

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
  return 1;
}

static uint8_t *loadjpeg(const char *filename, unsigned int *width, unsigned int *height) {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr cerrmgr;
  jpegErrorManager jerr;
  uint8_t *dest;
  JSAMPROW buffer = 0;
  FILE *f;

  if (!filename)
    return 0;

  f = fopen(filename, "r");
  if (!f)
    return 0;

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = jpegErrorExit;
  /* Establish the setjmp return context for my_error_exit to use. */
  if (setjmp(jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error. */
    jpeg_destroy_decompress(&cinfo);
    fclose(f);
    if (buffer)
      free(buffer);
    return 0;
  }

  //cinfo.err = jpeg_std_error(&cerrmgr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, f);
  jpeg_read_header(&cinfo, 1);
  cinfo.out_color_space = JCS_YCbCr;
  jpeg_start_decompress(&cinfo);

  *width = cinfo.output_width;
  *height = cinfo.output_height;

  dest = (uint8_t*)aligned_malloc(cinfo.output_width * cinfo.output_height * 1.5, 16);

  if (!dest) {
    jpeg_destroy_decompress(&cinfo);
    fclose(f);
    return 0;
  }

  buffer = (JSAMPROW)malloc(sizeof(JSAMPLE) * cinfo.output_width * cinfo.output_components);

  if (buffer) {
    while (cinfo.output_scanline < cinfo.output_height) {
      jpeg_read_scanlines(&cinfo, &buffer, 1);
      for (unsigned int i = 0; i < *width; i++) {
        dest[(cinfo.output_scanline - 1) * *width + i] = (buffer[i*3+0]);
	if ((cinfo.output_scanline & 1) && (i & 1)) {
	  dest[(cinfo.output_scanline - 1)/2 * *width/2 + i/2 + *width * *height] = (buffer[i*3+1]);
	  dest[(cinfo.output_scanline - 1)/2 * *width/2 + i/2 + *width * *height + *width * *height / 4] = (buffer[i*3+2]);
	}
      }
    }
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(f);
  if (buffer)
    free(buffer);
  return dest;
}

static uint8_t *reproject(uint8_t *py, uint8_t *pu, uint8_t *pv,
			  int dw, int dh, int sw, int sh, int32_t *map, uint16_t *c01, uint16_t *c23) {
  uint8_t *reprojected = new uint8_t[(int)(dw * dh * 1.5)];

  unsigned int uoff = dw * dh;
  unsigned int voff = uoff + dw * dh / 4;
  for (int y = 0; y < dh; y += 2) {
    for (int x = 0; x < dw; x += 2) {
      int32_t offset0 = map[y * dw + x];
      int32_t offset1 = map[y * dw + x + 1];
      int32_t offset2 = map[y * dw + x + dw];
      int32_t offset3 = map[y * dw + x + 1 + dw];
      int32_t coffset = (offset0 / sw) / 2 * sw/2 + (offset0 % sw) / 2;
      reprojected[y * dw + x] = offset0 < 0 ? 0 :
	(py[offset0 + 0] * (c01[y * dw + x] >> 8) +
	 py[offset0 + 1] * (uint8_t)c01[y * dw + x] +
	 py[offset0 + sw + 0] * (c23[y * dw + x] >> 8) +
	 py[offset0 + sw + 1] * (uint8_t)c23[y * dw + x]) >> 7;
      reprojected[y * dw + x + 1] = offset1 < 0 ? 0 :
	(py[offset1 + 0] * (c01[y * dw + x + 1] >> 8) +
	 py[offset1 + 1] * (uint8_t)c01[y * dw + x + 1] +
	 py[offset1 + sw + 0] * (c23[y * dw + x + 1] >> 8) +
	 py[offset1 + sw + 1] * (uint8_t)c23[y * dw + x + 1]) >> 7;
      reprojected[y * dw + x + dw] = offset2 < 0 ? 0 :
	(py[offset2 + 0] * (c01[y * dw + x + dw] >> 8) +
	 py[offset2 + 1] * (uint8_t)c01[y * dw + x + dw] +
	 py[offset2 + sw + 0] * (c23[y * dw + x + dw] >> 8) +
	 py[offset2 + sw + 1] * (uint8_t)c23[y * dw + x + dw]) >> 7;
      reprojected[y * dw + x + 1 + dw] = offset3 < 0 ? 0 :
	(py[offset3 + 0] * (c01[y * dw + x + 1 + dw] >> 8) +
	 py[offset3 + 1] * (uint8_t)c01[y * dw + x + 1 + dw] +
	 py[offset3 + sw + 0] * (c23[y * dw + x + 1 + dw] >> 8) +
	 py[offset3 + sw + 1] * (uint8_t)c23[y * dw + x + 1 + dw]) >> 7;
      reprojected[y/2 * dw/2 + x/2 + uoff] = offset0 < 0 ? 128 : pu[coffset];
      reprojected[y/2 * dw/2 + x/2 + voff] = offset0 < 0 ? 128 : pv[coffset];
    }
  }
  enhance(reprojected, dw, dh, 16, 5, 5, 6, rand());
  return reprojected;
}

typedef struct StreamingParams {
  char copy_video;
  const char *muxer_opt_key;
  const char *muxer_opt_value;
  const char *video_codec;
  const char *codec_priv_key;
  const char *codec_priv_value;
} StreamingParams;

typedef struct StreamingContext {
  AVFormatContext *avfc;
  const AVCodec *video_avc;
  AVStream *video_avs;
  AVCodecContext *video_avcc;
  int video_index;
  char *filename;
} StreamingContext;

int encode_video(StreamingContext *decoder, StreamingContext *encoder, AVFrame *input_frame) {
  if (input_frame) input_frame->pict_type = AV_PICTURE_TYPE_NONE;

  AVPacket *output_packet = av_packet_alloc();
  if (!output_packet) {
    fprintf(stderr, "Could not allocate memory for output packet\n");
    return -1;
  }

  int response = avcodec_send_frame(encoder->video_avcc, input_frame);

  while (response >= 0) {
    response = avcodec_receive_packet(encoder->video_avcc, output_packet);
    if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
      break;
    } else if (response < 0) {
      fprintf(stderr, "Error while receiving packet from encoder: %d\n", response);
      return -1;
    }

    output_packet->stream_index = decoder->video_index;
    output_packet->duration = encoder->video_avs->time_base.den / encoder->video_avs->time_base.num / decoder->video_avs->avg_frame_rate.num * decoder->video_avs->avg_frame_rate.den;
    
    av_packet_rescale_ts(output_packet, decoder->video_avs->time_base, encoder->video_avs->time_base);
    response = av_interleaved_write_frame(encoder->avfc, output_packet);
    if (response != 0) {
      fprintf(stderr, "Error %d while receiving packet from decoder\n", response);
      return -1;
    }
  }
  av_packet_unref(output_packet);
  av_packet_free(&output_packet);
  return 0;
}

static int decode_packet_and_encode(AVPacket *pPacket, AVCodecContext *pCodecContext, AVFrame *pFrame,
				    int w, int h, int32_t *map, uint16_t *c01, uint16_t *c23,
				    StreamingContext *encoder, StreamingContext *decoder) {
  int response = avcodec_send_packet(pCodecContext, pPacket);

  if (response < 0) {
    fprintf(stderr, "Error while sending a packet to the decoder: %d\n", response);
    return response;
  }

  while (response >= 0) {
    response = avcodec_receive_frame(pCodecContext, pFrame);
    if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
      break;
    } else if (response < 0) {
      fprintf(stderr, "Error while receiving a frame from the decoder: %d\n", response);
      return response;
    }

    if (response >= 0) {
      if (1)
	printf("Frame %d (type=%c, size=%d bytes, format=%d) pts %ld key_frame %d [DTS %d]\n",
	     pCodecContext->frame_number,
	     av_get_picture_type_char(pFrame->pict_type),
	     pFrame->pkt_size,
	     pFrame->format,
	     (long int)pFrame->pts,
	     pFrame->key_frame,
	     pFrame->coded_picture_number
	     );

      if (pFrame->format != AV_PIX_FMT_YUVJ420P && pFrame->format != AV_PIX_FMT_YUV420P) {
	fprintf(stderr, "Not a YUV420 frame (%d)n", pFrame->format);
      }
      uint8_t *reprojected = reproject(pFrame->data[0], pFrame->data[1], pFrame->data[2],
				       w, h, pFrame->width, pFrame->height, map, c01, c23);

      AVFrame pFrame2 = *pFrame;
      pFrame->data[0] = reprojected;
      pFrame->data[1] = reprojected + w * h;
      pFrame->data[2] = reprojected + w * h + w * h / 4;
      pFrame->width = w;
      pFrame->height = h;
      pFrame->linesize[0] = w;
      pFrame->linesize[1] = pFrame->linesize[2] = w/2;
      if (encode_video(decoder, encoder, pFrame)) return -1;
      *pFrame = pFrame2;
      delete[] reprojected;
    }
  }
  return 0;
}

int fill_stream_info(AVStream *avs, AVCodec const **avc, AVCodecContext **avcc) {
  *avc = avcodec_find_decoder(avs->codecpar->codec_id);
  if (!*avc) {
    fprintf(stderr, "Failed to find the codec\n");
    return -1;
  }

  *avcc = avcodec_alloc_context3(*avc);
  if (!*avcc) {
    fprintf(stderr, "Failed to alloc memory for codec context\n");
    return -1;
  }

  if (avcodec_parameters_to_context(*avcc, avs->codecpar) < 0) {
    fprintf(stderr, "Failed to fill codec context");
    return -1;
  }

  if (avcodec_open2(*avcc, *avc, NULL) < 0) {
    fprintf(stderr, "Failed to open codec\n");
    return -1;
  }
  return 0;
}

int prepare_decoder(StreamingContext *sc) {
  for (int i = 0; i < sc->avfc->nb_streams; i++) {
    if (sc->avfc->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      sc->video_avs = sc->avfc->streams[i];
      sc->video_index = i;

      if (fill_stream_info(sc->video_avs, &sc->video_avc, &sc->video_avcc)) {
	return -1;
      }
    } else {
      fprintf(stderr, "Skipping streams other than video\n");
      return -1;
    }
  }

  return 0;
}

int prepare_video_encoder(StreamingContext *sc, AVCodecContext *decoder_ctx, AVRational input_framerate, StreamingParams sp, unsigned int width, unsigned int height) {
  sc->video_avs = avformat_new_stream(sc->avfc, NULL);

  sc->video_avc = avcodec_find_encoder_by_name(sp.video_codec);
  if (!sc->video_avc) {
    fprintf(stderr, "Could not find the proper codec %s\n", sp.video_codec);
    return -1;
  }

  sc->video_avcc = avcodec_alloc_context3(sc->video_avc);
  if (!sc->video_avcc) {
    fprintf(stderr, "Could not allocated memory for codec context\n");
    return -1;
  }

  av_opt_set(sc->video_avcc->priv_data, "preset", "slow", 0);
  if (sp.codec_priv_key && sp.codec_priv_value)
    av_opt_set(sc->video_avcc->priv_data, sp.codec_priv_key, sp.codec_priv_value, 0);

  sc->video_avcc->height = height;
  sc->video_avcc->width = width;
  sc->video_avcc->sample_aspect_ratio = decoder_ctx->sample_aspect_ratio;
  if (sc->video_avc->pix_fmts)
    sc->video_avcc->pix_fmt = sc->video_avc->pix_fmts[0];
  else
    sc->video_avcc->pix_fmt = decoder_ctx->pix_fmt;

  sc->video_avcc->bit_rate = 2 * 1000 * 1000;
  sc->video_avcc->rc_buffer_size = 4 * 1000 * 1000;
  sc->video_avcc->rc_max_rate = 2 * 1000 * 1000;
  sc->video_avcc->rc_min_rate = 2.5 * 1000 * 1000;

  sc->video_avcc->time_base = av_inv_q(input_framerate);
  sc->video_avs->time_base = sc->video_avcc->time_base;

  if (avcodec_open2(sc->video_avcc, sc->video_avc, NULL) < 0) {
    fprintf(stderr, "Could not open the codec\n");
    return -1;
  }
  avcodec_parameters_from_context(sc->video_avs->codecpar, sc->video_avcc);
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc == 4) {
    // Open pto file
    std::ifstream ptofile(argv[1]);

    if (!ptofile.good()) {
      fprintf(stderr, "Could not open pto file %s\n", argv[1]);
      return -1;
    }

    // Parse pto file
    HuginBase::Panorama *pano = new HuginBase::Panorama;;
    pano->setFilePrefix(hugin_utils::getPathPrefix(argv[1]));
    AppBase::DocumentData::ReadWriteError err = pano->readData(ptofile);
    if (err != AppBase::DocumentData::SUCCESSFUL) {
      std::cerr << "Could not parse pto file " << argv[1] << ".  Error code: " << err << std::endl;
      if (err == AppBase::DocumentData::UNKNOWN_ERROR) std::cerr << "UNKNOWN_ERROR" << std::endl;
      if (err == AppBase::DocumentData::INCOMPATIBLE_TYPE) std::cerr << "INCOMPATIBLE_TYPE" << std::endl;
      if (err == AppBase::DocumentData::INVALID_DATA) std::cerr << "INVALID_DATA" << std::endl;
      if (err == AppBase::DocumentData::PARSER_ERROR) std::cerr << "PARSER_ERROR" << std::endl;
      delete pano;
      return -1;
    }

    HuginBase::PTools::Transform *trafo;
    trafo = new HuginBase::PTools::Transform;
    trafo->createTransform(pano->getSrcImage(0), pano->getOptions());
    
    const HuginBase::SrcPanoImage &img = pano->getImage(0);
    const HuginBase::PanoramaOptions &opt = pano->getOptions();
    int ptowidth = img.getWidth();
    int ptoheight = img.getHeight();
    int w = opt.getWidth();
    int h = opt.getHeight();

    // Reproject every pixel
    int32_t *map = new int32_t[w * h];
    uint16_t *c01 = new uint16_t[w * h];
    uint16_t *c23 = new uint16_t[w * h];
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
	double xout, yout;
	unsigned int c[4];
	trafo->transformImgCoord(xout, yout, x, y);
	if (xout < 0 || xout >= ptowidth || yout < 0 || yout >= ptoheight) {
	  // Outside frame
	  map[y*w + x] = -1;
	  c01[y*w + x] = c23[y*w + x] = 128;
	} else {
	  double xint, xfrac = modf(xout, &xint);
	  double yint, yfrac = modf(yout, &yint);
	  c[0] = (int)((1 - xfrac) * (1 - yfrac) * 128 + 0.5);
	  c[1] = (int)(xfrac * (1 - yfrac) * 128 + 0.5);
	  c[2] = (int)((1 - xfrac) * yfrac * 128 + 0.5);
	  c[3] = (int)(xfrac * yfrac * 128 + 0.5);
	  int diff = 128 - c[0] - c[1] - c[2] - c[3];
	  int maxidx = (xfrac >= 0.5) + 2 * (yfrac >= 0.5);
	  c[maxidx] += diff;
	  map[y*w + x] = (int32_t)yint * ptowidth +  (int32_t)xint;
	  c01[y*w + x] = (c[0] << 8) | (c[1] & 0xff);
	  c23[y*w + x] = (c[2] << 8) | (c[3] & 0xff);
	}
      }
    }
    delete trafo;
    delete pano;

    // Open jpeg file
    unsigned int width = 0, height = 0;
    uint8_t *jpeg = loadjpeg(argv[2], &width, &height);

    if (!jpeg) {

      // Set up decoder
      StreamingContext *decoder = (StreamingContext*) calloc(1, sizeof(StreamingContext));
      AVFormatContext *pFormatContext = avformat_alloc_context();
      decoder->avfc = pFormatContext;
      decoder->filename = argv[2];
      if (!pFormatContext) {
	fprintf(stderr, "Could not allocate memory for Format Context\n");
	return -1;
      }
      int err = 0;
      if ((err = avformat_open_input(&pFormatContext, argv[2], NULL, NULL) != 0)) {
	char errbuf[128];
	av_strerror(err, errbuf, sizeof(errbuf));
	fprintf(stderr, "Could not open video file %s (%d): %s\n", argv[2], err, errbuf);
	return -1;
      }
      if (avformat_find_stream_info(pFormatContext,  NULL) < 0) {
	fprintf(stderr, "ERROR could not get the stream info");
	return -1;
      }
      prepare_decoder(decoder);

      const AVCodec *pCodec = NULL;

      AVCodecParameters *pCodecParameters =  NULL;
      int video_stream_index = -1;
      AVRational input_framerate;

      for (int i = 0; i < pFormatContext->nb_streams; i++) {
	AVCodecParameters *pLocalCodecParameters = pFormatContext->streams[i]->codecpar;
	input_framerate = pFormatContext->streams[i]->r_frame_rate;
	printf("AVStream->time_base before open coded %d/%d\n", pFormatContext->streams[i]->time_base.num, pFormatContext->streams[i]->time_base.den);
	printf("AVStream->r_frame_rate before open coded %d/%d\n", pFormatContext->streams[i]->r_frame_rate.num, pFormatContext->streams[i]->r_frame_rate.den);
	printf("AVStream->start_time %ld\n", (long int)pFormatContext->streams[i]->start_time);
	printf("AVStream->duration %ld\n", (long int)pFormatContext->streams[i]->duration);

	printf("finding the proper decoder (CODEC)\n");

	const AVCodec *pLocalCodec = avcodec_find_decoder(pLocalCodecParameters->codec_id);

	if (pLocalCodec == NULL) {
	  fprintf(stderr, "ERROR unsupported codec!\n");
	  continue;
	}

	// when the stream is a video we store its index, codec parameters and codec
	if (pLocalCodecParameters->codec_type == AVMEDIA_TYPE_VIDEO) {
	  if (video_stream_index == -1) {
	    video_stream_index = i;
	    pCodec = pLocalCodec;
	    pCodecParameters = pLocalCodecParameters;
	  }

	  printf("Video Codec: resolution %d x %d\n", pLocalCodecParameters->width, pLocalCodecParameters->height);
	}
	// print its name, id and bitrate
	printf("\tCodec %s ID %d bit_rate %ld\n", pLocalCodec->name, pLocalCodec->id, (long int)pLocalCodecParameters->bit_rate);
      }

      if (video_stream_index == -1) {
	fprintf(stderr, "File %s does not contain a video stream!\n", argv[2]);
	return -1;
      }

      AVCodecContext *pCodecContext = avcodec_alloc_context3(pCodec);
      if (!pCodecContext) {
	fprintf(stderr, "Failed to allocated memory for AVCodecContext\n");
	return -1;
      }

      if (avcodec_parameters_to_context(pCodecContext, pCodecParameters) < 0) {
	fprintf(stderr, "Failed to copy codec params to codec context\n");
	return -1;
      }

      if (avcodec_open2(pCodecContext, pCodec, NULL) < 0) {
	fprintf(stderr, "Failed to open codec through avcodec_open2\n");
	return -1;
      }

      AVFrame *pFrame = av_frame_alloc();
      if (!pFrame) {
	fprintf(stderr, "Failed to allocate memory for AVFrame\n");
	return -1;
      }

      AVPacket *pPacket = av_packet_alloc();
      if (!pPacket) {
	fprintf(stderr, "failed to allocate memory for AVPacket\n");
	return -1;
      }

      // Set up encoder
      StreamingParams sp = {0};
      sp.copy_video = 1;
      sp.video_codec = "libx264";
      sp.codec_priv_key = "x264-params";
      sp.codec_priv_value = "keyint=9999:min-keyint=9999:qp=24:bframes=0:rc-lookahead=0:sync-lookahead=0:threads=4";

      StreamingContext *encoder = (StreamingContext*) calloc(1, sizeof(StreamingContext));
      encoder->filename = argv[3];

      avformat_alloc_output_context2(&encoder->avfc, NULL, NULL, encoder->filename);
      if (!encoder->avfc) {
	fprintf(stderr, "Could not allocate memory for output format\n");
	return -1;
      }

      prepare_video_encoder(encoder, pCodecContext, input_framerate, sp, w, h);

      if (encoder->avfc->oformat->flags & AVFMT_GLOBALHEADER)
	encoder->avfc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

      if (!(encoder->avfc->oformat->flags & AVFMT_NOFILE)) {
	if (avio_open(&encoder->avfc->pb, encoder->filename, AVIO_FLAG_WRITE) < 0) {
	  fprintf(stderr, "Could not open the output file\n");
	  return -1;
	}
      }

      AVDictionary* muxer_opts = NULL;
      if (sp.muxer_opt_key && sp.muxer_opt_value) {
	av_dict_set(&muxer_opts, sp.muxer_opt_key, sp.muxer_opt_value, 0);
      }

      if (avformat_write_header(encoder->avfc, &muxer_opts) < 0) {
	fprintf(stderr, "An error occurred when opening output file\n");
	return -1;
      }


      
      int response = 0;
      while (av_read_frame(pFormatContext, pPacket) >= 0) {
	if (pPacket->stream_index == video_stream_index) {
	  response = decode_packet_and_encode(pPacket, pCodecContext, pFrame, w, h, map, c01, c23,
					      encoder, decoder);
	  if (response < 0)
	    break;
	}
	av_packet_unref(pPacket);
      }

      av_write_trailer(encoder->avfc);

      printf("releasing all the resources\n");

      delete[] c23;
      delete[] c01;
      delete[] map;
      avformat_close_input(&pFormatContext);
      av_packet_free(&pPacket);
      av_frame_free(&pFrame);
      avio_close(encoder->avfc->pb);
      avcodec_free_context(&pCodecContext);
      avformat_free_context(encoder->avfc);
      avcodec_free_context(&encoder->video_avcc);
      free(encoder);
      free(decoder);
      return 0;
    }

    if (ptowidth != width || ptoheight != height) {
      fprintf(stderr, "pto size (%dx%d) does not match jpeg size (%dx%d)\n",
	      ptowidth, ptoheight, width, height);
      aligned_free(jpeg);
      delete[] c23;
      delete[] c01;
      delete[] map;
      return -1;
    }

    uint8_t *reprojected = reproject(jpeg, jpeg + width*height, jpeg + width*height + width*height/4,
				     w, h, width, height, map, c01, c23);
    aligned_free(jpeg);
    delete[] c23;
    delete[] c01;
    delete[] map;

    // Save output jpeg file
    FILE *o = fopen(argv[3], "wb");
    if (!o) {
      fprintf(stderr, "Could not open output file %s\n", argv[3]);
      delete[] reprojected;
      return -1;
    }
    if (!savejpeg(o, reprojected, w, h, 90)) {
      fprintf(stderr, "Could not write output file %s\n", argv[3]);
      delete[] reprojected;
      return -1;
    }

    delete[] reprojected;
  } else {
      fprintf(stderr, "Usage: %s <pto file> <input file (jpeg or mp4)> <output file>\n", argv[0]);
      return -1;
  }
  return 0;
}
