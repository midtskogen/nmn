#include <malloc.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define MAXSTARS 8192

typedef struct {
  uint32_t interpolated;
  uint32_t dummy;
  union {
    float f;
    uint32_t i;
  } y;
  union {
    float f;
    uint32_t i;
  } x;
} pos;

const char *readfilerev(const char *name, int *size) {
  struct stat st;

  if (stat(name, &st) == -1) return 0;

  FILE *f = fopen(name, "rb");
  if (!f) return 0;

  char *buffer1 = malloc(st.st_size);
  char *buffer2 = malloc(st.st_size);

  if (fread(buffer1, st.st_size, 1, f) != 1) {
    free(buffer1);
    free(buffer2);
    fclose(f);
    return 0;
  }

  for (int i = 0; i < st.st_size; i++) buffer2[st.st_size - i - 1] = buffer1[i];

  free(buffer1);
  *size = st.st_size;
  fclose(f);
  return buffer2;
}

double dist(const pos *p1, const pos *p2) {
  return sqrt((p1->x.f - p2->x.f) * (p1->x.f - p2->x.f) + (p1->y.f - p2->y.f) * (p1->y.f - p2->y.f));
}

int equal(const pos *p1, const pos *p2) {
  return (p1->x.f - p2->x.f) * (p1->x.f - p2->x.f) + (p1->y.f - p2->y.f) * (p1->y.f - p2->y.f) < 0.5 * 0.5;
}

const pos *find_path(const pos *p, const pos *list, double thr, double min, int max) {
  int found = 0;
  int smallest = 9999999;
  const pos *best = 0;
  int count = 0;
  for (int k = 0; list[k].x.f >= 0 && list[k].y.f >= 0 && k < MAXSTARS; k++) {
    double d = dist(&list[k], p);
    count += d < thr;
    if (d >= min && d < smallest) {
      smallest = d;
      best = &list[k];
    }

    found |= d >= min && d < thr;
  }
  return smallest < thr && count < max ? best : 0;
}

pos **get_stars(int num_frames, double thr, const char **files, int *num_stars) {
  pos p;

  pos **frames = (pos **)malloc(num_frames * sizeof(pos *));
  if (!frames) {
    fprintf(stderr, "Malloc error\n");
    exit(0);
  }
  for (int i = 0; i < num_frames; i++) {
    frames[i] = (pos *)malloc(MAXSTARS * sizeof(pos));
    if (!frames[i]) {
      fprintf(stderr, "Malloc error\n");
      exit(0);
    }
  }
  int cnt = num_frames;
  int total = cnt;

  // Read .axy files
  while (--cnt >= 0) {
    int size = 0;
    const char *file = readfilerev(files[cnt], &size);
    const char *file_start = file;

    if (!file) continue;

    int c = 0;
    while (1) {
      if (size < sizeof(p)) break;

      memcpy(&p, file, sizeof(p));
      file += sizeof(p);
      size -= sizeof(p);

      if (p.x.i == 0x20202020 && p.y.i == 0x20202020) break;

      if (p.x.f > 0 && p.y.f > 0 && c < MAXSTARS - 1) {
        memcpy(&frames[cnt][c], &p, sizeof(p));
        frames[cnt][c++].interpolated = 0;
      }
      frames[cnt][c].x.f = frames[cnt][c].y.f = -1;
    }
    free((void *)file_start);
  }

  pos *interpolated = (pos *)malloc(sizeof(pos) * MAXSTARS);
  int interpolated_count = 0;
  int star = 0;

  if (!interpolated) {
    fprintf(stderr, "Malloc error\n");
    exit(0);
  }

  const pos ***paths;
  paths = (const pos ***)malloc(sizeof(pos **) * MAXSTARS);
  if (!paths) {
    fprintf(stderr, "Malloc error\n");
    exit(0);
  }
  for (int i = 0; i < MAXSTARS; i++) {
    paths[i] = (const pos **)malloc(sizeof(pos *) * total);
    if (!paths[i]) {
      fprintf(stderr, "Malloc error\n");
      exit(0);
    }
  }

  const pos **path = (const pos **)malloc(sizeof(pos *) * total);
  if (!path) {
    fprintf(stderr, "Malloc error\n");
    exit(0);
  }

  for (int j = 0; frames[0][j].x.f >= 0 && frames[0][j].y.f >= 0 && j < MAXSTARS; j++) {
    const pos *curr = &frames[0][j];
    int x = 0;
    path[x++] = curr;
    for (int i = 1; i < total; i++) {
      pos expected;
      const int lookahead = 5;
      const pos *next = 0;
      if (x > 1) {
        double xdiff = path[x - 1]->x.f - path[x - 2]->x.f;
        double ydiff = path[x - 1]->y.f - path[x - 2]->y.f;

        for (int k = 1; k <= lookahead && i + k <= total; k++) {
          expected.x.f = curr->x.f + xdiff * k;
          expected.y.f = curr->y.f + ydiff * k;
          next = find_path(&expected, frames[i + k - 1], thr / 2, 0, 2);
          if (next && interpolated_count < MAXSTARS) {
            interpolated[interpolated_count].x.f = curr->x.f + (next->x.f - curr->x.f) / k;
            interpolated[interpolated_count].y.f = curr->y.f + (next->y.f - curr->y.f) / k;
            interpolated[interpolated_count].interpolated = 1;
            next = &interpolated[interpolated_count++];
            break;
          }
        }
        curr = path[x] = next;
      } else {
        curr = path[x] = find_path(curr, frames[i], thr, 0.5, MAXSTARS);
      }

      if (curr && (dist(path[x], path[x - 1]) > thr || dist(path[x], path[x - 1]) < 0.5)) {
        curr = 0;
        break;
      }
      x++;

      if (!curr) break;
    }
    if (curr) {
      for (int i = 0; i < total; i++) {
        paths[star][i] = path[i];
        // printf("%8.3f %8.3f\n", path[i]->x.f, path[i]->y.f);
      }
      star++;
    }
  }

  pos **copy = (pos **)malloc(sizeof(pos *) * star);
  if (!copy) {
    fprintf(stderr, "Malloc error\n");
    exit(0);
  }
  for (int i = 0; i < star; i++) {
    copy[i] = (pos *)malloc(sizeof(pos) * total);
    if (!copy[i]) {
      fprintf(stderr, "Malloc error\n");
      exit(0);
    }
  }
  *num_stars = star;

  for (int i = 0; i < total; i++)
    for (int j = 0; j < star; j++) copy[j][i] = *paths[j][i];

  free(path);

  for (int i = 0; i < num_frames; i++) free(frames[i]);
  free(frames);

  for (int i = 0; i < MAXSTARS; i++) free(paths[i]);

  free(paths);
  free(interpolated);

  return copy;
}

int is_subset(const pos *sub, int sublen, const pos *super, int superlen) {
  if (sublen > superlen) return 0;

  while (--sublen >= 0) {
    int found = 0;
    for (int i = 0; i < superlen; i++) found |= equal(sub + sublen, super + i);
    if (!found) return 0;
  }
  return 1;
}

int is_equalset(const pos *sub, int sublen, const pos *super, int superlen) {
  return sublen == superlen && is_subset(sub, sublen, super, superlen);
}

typedef struct {
  double n;
  double *x;
  double *y;
} points;

int find_circle(pos *data, double n, double *out_x, double *out_y, double *out_r) {
  int i, iter, IterMAX = 99;

  double Xi, Yi, Zi;
  double Mz, Mxy, Mxx, Myy, Mxz, Myz, Mzz, Cov_xy, Var_z;
  double A0, A1, A2, A22;
  double Dy, xnew, x, ynew, y;
  double DET, Xcenter, Ycenter;
  double meanx = 0, meany = 0;

  for (int i = 0; i < n; i++) {
    meanx += data[i].x.f;
    meany += data[i].y.f;
  }
  meanx /= n;
  meany /= n;

  //     computing moments
  Mxx = Myy = Mxy = Mxz = Myz = Mzz = 0;

  for (i = 0; i < n; i++) {
    Xi = data[i].x.f - meanx;  //  centered x-coordinates
    Yi = data[i].y.f - meany;  //  centered y-coordinates
    Zi = Xi * Xi + Yi * Yi;

    Mxy += Xi * Yi;
    Mxx += Xi * Xi;
    Myy += Yi * Yi;
    Mxz += Xi * Zi;
    Myz += Yi * Zi;
    Mzz += Zi * Zi;
  }
  Mxx /= n;
  Myy /= n;
  Mxy /= n;
  Mxz /= n;
  Myz /= n;
  Mzz /= n;

  //    computing the coefficients of the characteristic polynomial
  Mz = Mxx + Myy;
  Cov_xy = Mxx * Myy - Mxy * Mxy;
  Var_z = Mzz - Mz * Mz;

  A2 = 4 * Cov_xy - 3 * Mz * Mz - Mzz;
  A1 = Var_z * Mz + 4 * Cov_xy * Mz - Mxz * Mxz - Myz * Myz;
  A0 = Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - Var_z * Cov_xy;
  A22 = A2 + A2;

  //    finding the root of the characteristic polynomial
  //    using Newton's method starting at x=0
  //     (it is guaranteed to converge to the right root)

  for (x = 0, y = A0, iter = 0; iter < IterMAX; iter++) {
    Dy = A1 + x * (A22 + 16 * x * x);
    xnew = x - y / Dy;
    if ((xnew == x) || (!isfinite(xnew))) break;
    ynew = A0 + xnew * (A1 + xnew * (A2 + 4 * xnew * xnew));
    if (abs(ynew) >= abs(y)) break;
    x = xnew;
    y = ynew;
  }

  //    computing paramters of the fitting circle
  DET = x * x - x * Mz + Cov_xy;
  Xcenter = (Mxz * (Myy - x) - Myz * Mxy) / DET / 2;
  Ycenter = (Myz * (Mxx - x) - Mxz * Mxy) / DET / 2;

  //       assembling the output
  *out_x = Xcenter + meanx;
  *out_y = Ycenter + meany;
  *out_r = sqrt(Xcenter * Xcenter + Ycenter * Ycenter + Mz - x - x);
  return iter;
}

int main(int argc, const char **argv) {
  double thr = atof(argv[1]);
  int num_frames = argc - 2;
  const int min_size = num_frames - 2 < 10 ? num_frames - 2 : ((num_frames / 10) < 10 ? 10 : num_frames / 10);
  typedef struct {
    int start;
    int end;
    int stars;
    pos **list;
  } pass;

  /* Count passes.  Probably a faster way to compute this... */
  int num_passes = 0;
  for (int start = 0; start < num_frames - min_size; start++)
    for (int end = num_frames - 1; end > start + min_size; end--) num_passes++;

  pass *passes = malloc(sizeof(pass) * num_passes);
  if (!passes) {
    fprintf(stderr, "Malloc error\n");
    exit(0);
  }

  /* Try all possible sequences of a minimum length */
  int c = 0;
  for (int start = 0; start < num_frames - min_size; start++)
    for (int end = num_frames - 1; end > start + min_size; end--) {
      int stars;
      passes[c].start = start;
      passes[c].end = end;
      passes[c].list = get_stars(end - start + 1, thr, argv + 2 + start, &stars);
      passes[c++].stars = stars;
    }

  typedef struct {
    int start;
    int end;
    pos *list;
  } trail;

  trail *unique = malloc(sizeof(trail) * MAXSTARS);
  int unique_trails = 0;

  /* Find the longest unique star trails */
  for (int i = 0; i < num_passes; i++) {
    for (int j = 0; j < passes[i].stars; j++) {
      int found = 0;
      for (int m = 0; m < unique_trails && !found; m++) {
        if (is_subset(unique[m].list, unique[m].end - unique[m].start - 1, passes[i].list[j], passes[i].end - passes[i].start - 1)) {
          unique[m].start = passes[i].start;
          unique[m].end = passes[i].end;
          unique[m].list = passes[i].list[j];
          break;
        }
        found |= is_subset(passes[i].list[j], passes[i].end - passes[i].start - 1, unique[m].list, unique[m].end - unique[m].start - 1);
      }
      if (!found && unique_trails < MAXSTARS) {
        unique[unique_trails].start = passes[i].start;
        unique[unique_trails].end = passes[i].end;
        unique[unique_trails++].list = passes[i].list[j];
      }
      if (unique_trails == MAXSTARS) {
        // Bail out
        i = num_passes;
        j = passes[i].stars;
      }
    }
  }

  typedef struct {
    double x;
    double y;
    double r;
  } circle;

  circle *circles = malloc(sizeof(circle) * unique_trails);

  /* Find trails moving unexpectedly */
  for (int j = 0; j < unique_trails; j++)
    if (find_circle(unique[j].list, unique[j].end - unique[j].start + 1, &circles[j].x, &circles[j].y, &circles[j].r) > 1)
      unique[j].list = 0;

  free(circles);

  /* Filter */
  for (int j = 0; j < unique_trails; j++) {
    int i;
    int count = 0;
    if (unique[j].end - unique[j].start < (min_size < 15 ? min_size : 15)) {
      unique[j].list = 0;
      continue;
    }

    for (i = 6; i <= unique[j].end - unique[j].start && unique[j].list; i++) {
      double dx1 = unique[j].list[i - count].x.f - unique[j].list[i - 5 - count].x.f;
      double dy1 = unique[j].list[i - count].y.f - unique[j].list[i - 5 - count].y.f;
      double dx2 = unique[j].list[i].x.f - unique[j].list[i - 1].x.f;
      double dy2 = unique[j].list[i].y.f - unique[j].list[i - 1].y.f;
      double d = atan2(dy1, dx1) - atan2(dy2, dx2);
      if (fabs(atan2(sin(d), cos(d))) > 3.141592 / 32) {
        count++;
      } else
        count = 0;
    }
    if (count) unique[j].end -= count;
  }

  /* Print result */
  for (int i = 0; i < num_frames; i++) {
    for (int j = 0; j < unique_trails; j++) {
      if (unique[j].list) {
        if (i >= unique[j].start && i <= unique[j].end) {
          printf("\t%07.2f,%07.2f", unique[j].list[i - unique[j].start].x.f, unique[j].list[i - unique[j].start].y.f);
        } else
          printf("\t-------,-------");
      }
    }
    printf("\n");
  }

  for (int i = 0; i < num_passes; i++) {
    for (int j = 0; j < passes[i].stars; j++) free(passes[i].list[j]);
    free(passes[i].list);
  }
  free(passes);
  free(unique);

  return 0;
}
