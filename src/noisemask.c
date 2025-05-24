#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

void *aligned_malloc(size_t size, uintptr_t align) {
  void *m = malloc(size + sizeof(void *) + align);
  void **r = (void **)((((uintptr_t)m) + sizeof(void *) + align - 1) & ~(align - 1));
  if (!m) return m;
  r[-1] = m;
  return r;
}

void aligned_free(void *p) { free(((void **)p)[-1]); }

int main(int argc, char **argv) {
  const int levels = 16;
  FILE *i = stdin;
  FILE *o = 0;
  unsigned char *buf1;
  unsigned int *buf2[levels];
  const int x = atoi(argv[1]);
  const int y = atoi(argv[2]);
  int c;

  buf1 = aligned_malloc(x * y * 1.5, 16);
  int ok = !!buf1;
  for (c = 0; c < levels; c++) {
    buf2[c] = aligned_malloc(x * y * sizeof(unsigned int), 16);
    ok |= !!buf2[c];
  }

  if (!ok) {
    printf("%s: Memory error\n", argv[0]);
    return -1;
  }

  for (c = 0; c < levels; c++) memset(buf2[c], 0, x * y * sizeof(unsigned int));

  if (!i) {
    printf("%s: File error\n", argv[0]);
    return -1;
  }

  int frames = 0;
  while (!feof(i)) {
    int s = fread(buf1, x * y * 1.5, 1, i);
    if (s != 1) break;

    for (c = 0; c < levels; c++)
      for (int a = 0; a < x * y; a++) buf2[c][a] += buf1[a] > c;

    frames++;
  }

  for (c = 0; c < levels; c++) {
    int count = 0;
    for (int a = 0; a < x * y; a++) count += buf2[c][a] > (frames / 16 > 3 ? frames / 16 : 3);
    if (count * 100 / x / y < 5) break;
  }
  c -= (c == levels);

  fprintf(stderr, "%d\n", c);

  int count = 0;
  for (int a = 0; a < x * y; a++) {
    buf1[a] = buf2[c][a] > (frames / 16 > 3 ? frames / 16 : 3) ? 0 : 255;
    count += !buf1[a];
  }

  memset(buf1 + x * y, 128, x * y / 2);
  fwrite(buf1, x * y * 1.5, 1, stdout);

  for (c = 0; c < levels; c++) aligned_free(buf2[c]);

  return 0;
}
