#ifndef NEON_SHIM_H
#define NEON_SHIM_H
 
// Build with -DWINDOWS_VER / /DWINDOWS_VER.
// COMPILE WITH G++
// WINDOWS_VER requires AVX2 + FMA: /arch:AVX2 (MSVC) or -mavx2 -mfma (clang/gcc).
 
#include <cstdlib>
#include <cstddef>
 
#ifdef WINDOWS_VER
 
#include <immintrin.h>

#define SHIM_INLINE static inline __attribute__((always_inline))
 
typedef __m128 float32x4_t;
typedef __m128 float32x2_t;
 
struct float32x4x2_t {
  float32x4_t val[2];
};
 
SHIM_INLINE float32x4_t vld1q_f32(const float* p) { return _mm_loadu_ps(p); }
SHIM_INLINE void vst1q_f32(float* p, float32x4_t v) { _mm_storeu_ps(p, v); }
SHIM_INLINE float32x4_t vdupq_n_f32(float x) { return _mm_set1_ps(x); }
 
SHIM_INLINE float32x4_t vaddq_f32(float32x4_t a, float32x4_t b) {
  return _mm_add_ps(a, b);
}
SHIM_INLINE float32x4_t vmulq_f32(float32x4_t a, float32x4_t b) {
  return _mm_mul_ps(a, b);
}
 
SHIM_INLINE float32x4_t vfmaq_f32(float32x4_t acc, float32x4_t a,
                                  float32x4_t b) {
  return _mm_fmadd_ps(a, b, acc);
}
SHIM_INLINE float32x4_t vmlaq_f32(float32x4_t acc, float32x4_t a,
                                  float32x4_t b) {
  return _mm_fmadd_ps(a, b, acc);
}
 
SHIM_INLINE float vaddvq_f32(float32x4_t a) {
  __m128 t = _mm_add_ps(a, _mm_movehl_ps(a, a));  // [a0+a2, a1+a3, ..]
  t = _mm_add_ss(t, _mm_shuffle_ps(t, t, 1));     // lane0 + lane1
  return _mm_cvtss_f32(t);
}
 
SHIM_INLINE float32x2_t vget_low_f32(float32x4_t a) { return a; }
SHIM_INLINE float32x2_t vget_high_f32(float32x4_t a) {
  return _mm_movehl_ps(a, a);
}
SHIM_INLINE float32x4_t vcombine_f32(float32x2_t lo, float32x2_t hi) {
  return _mm_movelh_ps(lo, hi);
}
 
// vtrnq_f32(a, b): val[0] = [a0 b0 a2 b2], val[1] = [a1 b1 a3 b3]
// 1 shift + 1 blend per output (blend runs on more ports than shuffles).
SHIM_INLINE float32x4x2_t vtrnq_f32(float32x4_t a, float32x4_t b) {
  float32x4x2_t r;
  r.val[0] = _mm_blend_ps(
      a, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(b), 4)), 0xA);
  r.val[1] = _mm_blend_ps(
      _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a), 4)), b, 0xA);
  return r;
}
 
#else 
#include <arm_neon.h>
#endif
 
inline void* tensor_alloc(size_t alignment, size_t size) {
#ifdef WINDOWS_VER
  return _aligned_malloc(size, alignment);  // no size-multiple requirement
#else
  return aligned_alloc(alignment, size);
#endif
}
 
inline void tensor_free(void* p) {
#ifdef WINDOWS_VER
  _aligned_free(p);
#else
  free(p);
#endif
}
 
#endif