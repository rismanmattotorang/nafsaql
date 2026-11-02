// Memory-bandwidth ceiling of the machine (Phase 17): multi-threaded sequential read, AVX2 loads, 1 GiB, best of 5.
// cc -O3 -mavx2 -o /tmp/membw scripts/membw.c -lpthread && /tmp/membw
// Measured on the reference machine (i3-8100, DDR4, shared host at load 5): 12.9 / 17.2 / 18.4 / 19.5 GB/s at 1 / 2 / 3 / 4 threads.
// Multi-threaded sequential read bandwidth (AVX2 loads, 4 accumulators), thread counts 1..4, 1 GiB buffer.
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <stdatomic.h>
static atomic_int ready, go;
#define BYTES (1ULL<<30)
static char *buf;
typedef struct { size_t off, len; volatile float sink; int reps; } job_t;
static void *work(void *p) {
  job_t *j = (job_t *)p;
  __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
  atomic_fetch_add(&ready, 1); while (!atomic_load(&go)) {}
  for (int r = 0; r < j->reps; r++) {
    const float *x = (const float *)(buf + j->off);
    size_t n = j->len / sizeof(float);
    for (size_t i = 0; i + 32 <= n; i += 32) {
      a0 = _mm256_add_ps(a0, _mm256_load_ps(x + i));
      a1 = _mm256_add_ps(a1, _mm256_load_ps(x + i + 8));
      a2 = _mm256_add_ps(a2, _mm256_load_ps(x + i + 16));
      a3 = _mm256_add_ps(a3, _mm256_load_ps(x + i + 24));
    }
  }
  float t[8]; _mm256_storeu_ps(t, _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
  j->sink = t[0];
  return 0;
}
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
int main(void) {
  buf = aligned_alloc(64, BYTES); memset(buf, 1, BYTES);
  for (int th = 1; th <= 4; th++) {
    double best = 0;
    for (int trial = 0; trial < 5; trial++) {
      pthread_t t[4]; job_t j[4]; atomic_store(&ready, 0); atomic_store(&go, 0);
      for (int i = 0; i < th; i++) { j[i].off = (BYTES / th) * i; j[i].len = BYTES / th; j[i].reps = 3; pthread_create(&t[i], 0, work, &j[i]); }
      while (atomic_load(&ready) < th) {} double t0 = now(); atomic_store(&go, 1);
      for (int i = 0; i < th; i++) pthread_join(t[i], 0);
      double dt = now() - t0, gbs = (double)BYTES * 3 / dt / 1e9;
      if (gbs > best) best = gbs;
    }
    printf("threads %d: %.1f GB/s (best of 5)\n", th, best);
  }
  return 0;
}
