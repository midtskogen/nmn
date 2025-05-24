#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <libavformat/avformat.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/inotify.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

// gcc -Os -o bin/cammon src/cammon.c -lavformat -lavutil -lavcodec -pthread

typedef struct {
  char *maxfile;
  char *prefix;
  char *logfile;
} config;

void *filter(void *input) {
  const char *in_filename = (const char *)input;
  char cmd[1024];
  if (fcntl(stdout->_fileno, F_SETPIPE_SZ, 134217728) == 134217728) {
    int flags = fcntl(stdout->_fileno, F_GETFL, 0);
    if (flags != -1) fcntl(stdout->_fileno, F_SETFL, flags | O_NONBLOCK);
  }
  snprintf(cmd, sizeof(cmd),
           "ffmpeg -vsync 0 -loglevel quiet -nostdin -i %s -f rawvideo "
           "-codec:v copy -codec:a none -bsf:v h264_mp4toannexb -",
           in_filename);
  return system(cmd) ? NULL : input;
}

void *filter_new(void *input) {
  const char *in_filename = (const char *)input;
  AVOutputFormat *ofmt_v = NULL;
  AVFormatContext *ifmt_ctx = NULL, *ofmt_ctx_v = NULL;

  if (fcntl(stdout->_fileno, F_SETPIPE_SZ, 134217728) == 134217728) {
    int flags = fcntl(stdout->_fileno, F_GETFL, 0);
    if (flags != -1) fcntl(stdout->_fileno, F_SETFL, flags | O_NONBLOCK);
  }
  int videoindex = -1;
  av_register_all();
  av_log_set_level(AV_LOG_QUIET);

  // Input
  if (avformat_open_input(&ifmt_ctx, in_filename, 0, 0) < 0) return input;

  // Output
  avformat_alloc_output_context2(&ofmt_ctx_v, NULL, "h264", NULL);
  if (!ofmt_ctx_v) return input;

  ofmt_v = ofmt_ctx_v->oformat;

  for (int i = 0; i < ifmt_ctx->nb_streams; i++) {
    if (ifmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      videoindex = i;
      avformat_new_stream(ofmt_ctx_v, NULL);
      break;
    }
  }

  if (videoindex < 0 || (!(ofmt_v->flags & AVFMT_NOFILE) && avio_open(&ofmt_ctx_v->pb, "/dev/stdout", AVIO_FLAG_WRITE) < 0) ||
      avformat_write_header(ofmt_ctx_v, NULL) < 0)
    return input;

  AVBSFContext *h264bsfc;
  const AVBitStreamFilter *bsf = av_bsf_get_by_name("h264_mp4toannexb");
  if (!bsf || av_bsf_alloc(bsf, &h264bsfc)) return input;

  AVPacket pkt;
  while (av_read_frame(ifmt_ctx, &pkt) >= 0) {
    AVStream *in_stream = ifmt_ctx->streams[pkt.stream_index];

    if (pkt.stream_index == videoindex) {
      avcodec_parameters_copy(h264bsfc->par_in, in_stream->codecpar);
      if (av_bsf_init(h264bsfc) < 0 || av_bsf_send_packet(h264bsfc, &pkt) < 0) {
        av_bsf_free(&h264bsfc);
        av_packet_unref(&pkt);
        return input;
      }

      if (av_bsf_receive_packet(h264bsfc, &pkt) != AVERROR(EAGAIN)) av_interleaved_write_frame(ofmt_ctx_v, &pkt);
    }
    av_packet_unref(&pkt);
  }

  av_bsf_free(&h264bsfc);

  // Write file trailer
  av_write_trailer(ofmt_ctx_v);
  avformat_close_input(&ifmt_ctx);

  /* Close output */
  if (ofmt_ctx_v && !(ofmt_v->flags & AVFMT_NOFILE)) avio_close(ofmt_ctx_v->pb);

  avformat_free_context(ofmt_ctx_v);
  return NULL;
}

#define EVENT_SIZE (sizeof(struct inotify_event))
#define BUF_LEN (1024 * (EVENT_SIZE + 16))

static unsigned long hash(char *str) {
  unsigned long hash = 5381;
  int c;

  while ((c = *str++)) hash = ((hash << 5) + hash) + c;

  return hash;
}

static void printlog(const char *log, const char *format, ...) {
  FILE *f = fopen(log, "a");
  if (f) {
    va_list args;
    char tsbuf[25];
    time_t ts = time(NULL);
    struct tm *t = gmtime(&ts);
    strftime(tsbuf, sizeof(tsbuf), "%Y-%m-%d %H:%M:%S", t);
    fprintf(f, "%s: ", tsbuf);
    va_start(args, format);
    vfprintf(f, format, args);
    va_end(args);
    fclose(f);
  }
}

static int yesterday_today_tomorrow(const char *fmt, size_t len, char *yesterday, char *today, char *tomorrow) {
  time_t ts = time(NULL);
  struct tm *t = gmtime(&ts);
  unsigned long before = hash(today);
  strftime(today, len, fmt, t);
  unsigned long after = hash(today);
  ts -= 24 * 3600;
  t = gmtime(&ts);
  strftime(yesterday, len, fmt, t);
  ts += 48 * 3600;
  t = gmtime(&ts);
  strftime(tomorrow, len, fmt, t);
  struct stat st = { 0 };
  if (stat(tomorrow, &st) == -1) mkdir(tomorrow, 0755);
  return before != after;
}

int *watches = 0;
char **watches_names;
int watches_count;
int watches_size = 128;
int fd = -1;

int find_watch(int wd) {
  for (int i = 0; i < watches_count; i++)
    if (watches[i] == wd) return i;
  return -1;
}

void add_watch(const char *name, const char *logfile) {
  if (!watches) {
    watches = malloc(watches_size * sizeof(int));
    watches_names = malloc(watches_size * sizeof(char *));
    watches_count = 0;
  }

  struct stat statbuf;
  stat(name, &statbuf);
  if (S_ISDIR(statbuf.st_mode)) {
    struct dirent *dent;
    DIR *srcdir = opendir(name);
    while ((dent = readdir(srcdir)) != NULL) {
      struct stat st;
      if (strcmp(dent->d_name, ".") == 0 || strcmp(dent->d_name, "..") == 0) continue;
      if (fstatat(dirfd(srcdir), dent->d_name, &st, 0) < 0) {
        perror(dent->d_name);
        continue;
      }
      if (S_ISDIR(st.st_mode)) {
        char buf[1024];
        snprintf(buf, sizeof(buf), "%s/%s", name, dent->d_name);
        add_watch(buf, logfile);
      }
    }
    closedir(srcdir);
  }
  watches_names[watches_count] = malloc(strlen(name) + 1);
  strcpy(watches_names[watches_count], name);
  watches[watches_count++] = inotify_add_watch(fd, name, IN_CLOSE_WRITE | IN_MOVED_TO);
  printlog(logfile, "Adding watch \"%s\" (%d)\n", name, watches_count);
  if (watches_count == watches_size) {
    watches_size += 128;
    watches = realloc(watches, watches_size);
    char **old_names = watches_names;
    watches_names = malloc(watches_size * sizeof(char *));
    for (int i = 0; i < watches_count; i++) {
      watches_names[i] = malloc(strlen(old_names[i]) + 1);
      strcpy(watches_names[i], old_names[i]);
      free(old_names[i]);
    }
    free(old_names);
  }
}

void rm_watches(const char *prefix, const char *logfile) {
  for (int i = 0; i < watches_count; i++) {
    if (!prefix || !strncmp(prefix, watches_names[i], strlen(prefix))) {
      printlog(logfile, "Removing watch \"%s\"\n", watches[i]);
      inotify_rm_watch(fd, watches[i]);
      watches[i] = -1;
      free(watches_names[i]);
      watches_names[i] = 0;
    }
  }
  int j = 0;
  for (int i = 0; i < watches_count; i++) {
    if (watches[i] >= 0) {
      watches[j] = watches[i];
      watches_names[j] = watches_names[i];
      j++;
    }
  }

  watches_count = j;
  printlog(logfile, "Remaining watches: %d\n", watches_count);
}

static void printhelp(const char *prog) {
  fprintf(stderr, "Usage: %s [options] <dir>\n", prog);
  fprintf(stderr,
          " This program will monitor three directories specified by "
          "<dir> for new files\n");
  fprintf(stderr,
          " and extract the raw h.264 streams from them and output "
          "those to stdout.\n");
  fprintf(stderr,
          " <dir> is a string passed to strftime() to format "
          "yesterday, today and tomorrow.\n");
  fprintf(stderr, " -h:           Print this help and exit.\n");
  fprintf(stderr, " -l <logfile>: Log file,\n");
  fprintf(stderr, " -p <logfile>: Prefix for the files to read,\n");
  fprintf(stderr,
          " -x <maxfile>: File which contains the location of an "
          "hourly \"max\" file used\n");
  fprintf(stderr, "               by metdetect.\n");
  fprintf(stderr, " Example: %s -p full_ /meteor/cam1/%%Y%%m%%d\n", prog);
  fprintf(stderr,
          "         If the current date is 2018-01-01, these "
          "directories will be monitored:\n");
  fprintf(stderr, "         /meteor/cam1/20171231\n");
  fprintf(stderr, "         /meteor/cam1/20180101\n");
  fprintf(stderr, "         /meteor/cam1/20180102\n");
  fprintf(stderr, "         Which directories that get monitored are updated with time.\n");
  fprintf(stderr, "         Only files beginning with full_ will be read.\n");
  fprintf(stderr,
          " The intended use for this program is to provide input for "
          "the metdetect program.\n");
}

int parseopts(config *config, int argc, char **argv) {
  int c;

  while ((c = getopt(argc, argv, "hl:p:x:")) != -1) {
    switch (c) {
      case 'h': printhelp(argv[0]); exit(1);
      case 'l':
        if (config->logfile) free(config->logfile);
        config->logfile = strdup(optarg);
        if (!config->logfile) {
          fprintf(stderr, "Memory error\n");
          return -1;
        }
        break;
      case 'p':
        if (config->prefix) free(config->prefix);
        config->prefix = strdup(optarg);
        if (!config->prefix) {
          fprintf(stderr, "Memory error\n");
          return -1;
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
      case '?':
        if (isprint(optopt))
          fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
        return 1;
      default: return 1;
    }
  }
  return 0;
}

pthread_mutex_t running = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t done = PTHREAD_COND_INITIALIZER;

typedef struct {
  int nfds;
  fd_set *set;
  int retval;
  struct timespec *timeout;
  config *config;
} try_select_data;

void *try_select(void *data) {
  int oldtype;
  try_select_data *d = (try_select_data *)data;
  struct timeval timeout;
  memset(&timeout, 0, sizeof(timeout));
  timeout.tv_sec = 120;

  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &oldtype);
  d->retval = select(d->nfds, d->set, NULL, NULL, &timeout);
  if (!d->retval) printlog(d->config->logfile, "Select timed out.\n");

  pthread_cond_signal(&done);
  return NULL;
}

/* note: this is not thread safe as it uses a global condition/mutex */
int do_or_timeout(try_select_data *data) {
  struct timespec abs_time;
  pthread_t tid;
  int err;

  pthread_mutex_lock(&running);

  /* pthread cond_timedwait expects an absolute time to wait until */
  clock_gettime(CLOCK_REALTIME, &abs_time);
  abs_time.tv_sec += data->timeout->tv_sec;
  abs_time.tv_nsec += data->timeout->tv_nsec;

  pthread_create(&tid, NULL, try_select, data);

  err = pthread_cond_timedwait(&done, &running, &abs_time);

  if (err == ETIMEDOUT) {
    data->retval = -1;
    printlog(data->config->logfile, "Select froze.\n");
  }
  if (!err) pthread_mutex_unlock(&running);

  return err;
}

int main(int argc, char **argv) {
  config config;
  config.maxfile = config.logfile = NULL;
  config.prefix = strdup("");

  if (argc < 2) {
    printhelp(argv[0]);
    exit(1);
  }

  if (parseopts(&config, argc, argv)) return 1;

  char yesterday[1024];
  char today[1024];
  char tomorrow[1024];
  char buffer[BUF_LEN];
  char lastname[1024];
  char *dir = argv[argc - 1];

  memset(yesterday, 0, sizeof(yesterday));
  memset(today, 0, sizeof(today));
  memset(tomorrow, 0, sizeof(tomorrow));
  memset(lastname, 0, sizeof(lastname));

  while (1) {
    fd_set set;
    struct timespec timeout;
    try_select_data data;
    memset(&timeout, 0, sizeof(timeout));
    memset(&data, 0, sizeof(data));
    timeout.tv_sec = 150;

    do {
      char old_yesterday[1024];
      strncpy(old_yesterday, yesterday, sizeof(old_yesterday));
      if (data.retval < 0 || fd < 0 || yesterday_today_tomorrow(dir, sizeof(yesterday), yesterday, today, tomorrow)) {
        if (fd >= 0) {
          rm_watches(NULL, config.logfile);
          close(fd);
        }
        fd = inotify_init();
        if (fd < 0) perror("inotify_init");
        yesterday_today_tomorrow(dir, sizeof(yesterday), yesterday, today, tomorrow);
        printlog(config.logfile, "Initialising watches: \"%s\", \"%s\", \"%s\"\n", yesterday, today, tomorrow);
        add_watch(yesterday, config.logfile);
        add_watch(today, config.logfile);
        add_watch(tomorrow, config.logfile);
      }

      FD_ZERO(&set);
      FD_SET(fd, &set);
      data.nfds = fd + 1;
      data.set = &set;
      data.timeout = &timeout;
      data.config = &config;

      printlog(config.logfile, "Waiting for new file(s)\n");
      do_or_timeout(&data);
    } while (data.retval < 1 || !FD_ISSET(fd, &set));

    int length = read(fd, buffer, BUF_LEN);

    if (length < 0) perror("read");

    printlog(config.logfile, "Read %d bytes\n", length);

    struct inotify_event *event;
    for (int i = 0; i < length; i += EVENT_SIZE + event->len) {
      event = (struct inotify_event *)&buffer[i];
      if (event->len) {
        char name[1024];
        int index = find_watch(event->wd);
        if (index < 0) continue;
        snprintf(name, sizeof(name), "%s/%s", watches_names[index], event->name);
        struct stat statbuf;
        stat(name, &statbuf);
        if (S_ISDIR(statbuf.st_mode)) {
          printlog(config.logfile, "New dir (%d): %s\n", event->mask, name);
          add_watch(name, config.logfile);
        } else {
          printlog(config.logfile, "New file: %s\n", name);
          if ((event->mask & (IN_MOVED_TO | IN_CLOSE_WRITE)) && strncmp(lastname, name, sizeof(lastname)) &&
              !strncmp(config.prefix, event->name, strlen(config.prefix))) {
            strncpy(lastname, name, sizeof(lastname));
            if (config.maxfile) {
              char mf[1024];
              snprintf(mf, sizeof(mf), "%s/max.jpg", watches_names[index]);
              if (access(mf, F_OK) == -1) {
                FILE *f = fopen(config.maxfile, "w");
                if (f) {
                  fprintf(f, "%s\n", mf);
                  fclose(f);
                }
              }
            }
            printlog(config.logfile, "Filtering %s\n", name);

            pthread_t thread;
            struct timespec ts;

            if (clock_gettime(CLOCK_REALTIME, &ts) == -1 || pthread_create(&thread, NULL, filter_new, name)) {
              printlog(config.logfile, "Error creating thread for filter\n");
              filter_new(name);
            } else {
              int s;
              ts.tv_sec += 180;  // Three minute timeout
              if (pthread_timedjoin_np(thread, NULL, &ts)) {
                printlog(config.logfile, "Filter timed out.\n");
                if (pthread_cancel(thread)) {
                  printlog(config.logfile, "Failed to cancel filter.\n");
                  pthread_join(thread, NULL);
                }
              }
            }
            printlog(config.logfile, "Filtered %s\n", name);
          }
        }
      }
    }
  }

  rm_watches(NULL, config.logfile);
  close(fd);

  free(watches);
  free(watches_names);

  if (config.maxfile) free(config.maxfile);
  if (config.logfile) free(config.logfile);

  return 0;
}
