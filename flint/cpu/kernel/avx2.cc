// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <assert.h>
#include <immintrin.h>
#include <stdint.h>

#include "flint/cpu/kernel/abstract.h"

namespace fl {
namespace op {
namespace cpu {
namespace kernel {

#if LIBWAIFU_KERNEL_MSVC
inline float libwaifu_cvtsh_ss(uint16_t sh) {
  __m128i vh = _mm_set1_epi16(sh);
  __m128 vs = _mm_cvtph_ps(vh);
  return _mm_cvtss_f32(vs);
}
#endif

LIBWAIFU_KERNEL_FORCE_INLINE float hsum(__m256 ymm) {
  __m128 x = _mm256_castps256_ps128(ymm);
  x = _mm_add_ps(x, _mm256_extractf128_ps(ymm, 1));
  x = _mm_add_ps(x, _mm_movehl_ps(x, x));
  x = _mm_add_ps(x, _mm_movehdup_ps(x));
  return _mm_cvtss_f32(x);
}

void sgemm6x16Avx2Kernel(int64_t kc, const float *a, const float *b, float *c, int64_t rs_c) {
  // a: kc x MR
  // b: kc x NR

  // C: MR x NR (6 x 2 ymmX)
  __m256 c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
  __m256 a00, b00, b01;

  float *pc = c;
  c00 = _mm256_loadu_ps(pc);
  c01 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c10 = _mm256_loadu_ps(pc);
  c11 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c20 = _mm256_loadu_ps(pc);
  c21 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c30 = _mm256_loadu_ps(pc);
  c31 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c40 = _mm256_loadu_ps(pc);
  c41 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  c50 = _mm256_loadu_ps(pc);
  c51 = _mm256_loadu_ps(pc + 8);
  pc += rs_c;

  const float *pa = a;
  const float *pb = b;
  for (int k = 0; k < kc; ++k) {
    b00 = _mm256_loadu_ps(pb);
    b01 = _mm256_loadu_ps(pb + 8);
    a00 = _mm256_broadcast_ss(pa);

    c00 = _mm256_fmadd_ps(a00, b00, c00);
    c01 = _mm256_fmadd_ps(a00, b01, c01);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c10 = _mm256_fmadd_ps(a00, b00, c10);
    c11 = _mm256_fmadd_ps(a00, b01, c11);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c20 = _mm256_fmadd_ps(a00, b00, c20);
    c21 = _mm256_fmadd_ps(a00, b01, c21);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c30 = _mm256_fmadd_ps(a00, b00, c30);
    c31 = _mm256_fmadd_ps(a00, b01, c31);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c40 = _mm256_fmadd_ps(a00, b00, c40);
    c41 = _mm256_fmadd_ps(a00, b01, c41);
    pa += 1;

    a00 = _mm256_broadcast_ss(pa);
    c50 = _mm256_fmadd_ps(a00, b00, c50);
    c51 = _mm256_fmadd_ps(a00, b01, c51);
    pa += 1;

    pb += 16;
  }

  pc = c;
  _mm256_storeu_ps(pc, c00);
  _mm256_storeu_ps(pc + 8, c01);
  pc += rs_c;

  _mm256_storeu_ps(pc, c10);
  _mm256_storeu_ps(pc + 8, c11);
  pc += rs_c;

  _mm256_storeu_ps(pc, c20);
  _mm256_storeu_ps(pc + 8, c21);
  pc += rs_c;

  _mm256_storeu_ps(pc, c30);
  _mm256_storeu_ps(pc + 8, c31);
  pc += rs_c;

  _mm256_storeu_ps(pc, c40);
  _mm256_storeu_ps(pc + 8, c41);
  pc += rs_c;

  _mm256_storeu_ps(pc, c50);
  _mm256_storeu_ps(pc + 8, c51);
  pc += rs_c;
}

void saxpyAvx2Kernel(int64_t n, float a, const float *x, float *y) {
  __m256 a00 = _mm256_broadcast_ss(&a);
  __m256 x00, y00;

  int64_t nb = n / 8;
  int64_t nr = n % 8;

  const float *px = x;
  float *py = y;
  for (int i = 0; i < nb; ++i) {
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_loadu_ps(py);

    y00 = _mm256_fmadd_ps(a00, x00, y00);
    _mm256_storeu_ps(py, y00);

    px += 8;
    py += 8;
  }

  for (int i = 0; i < nr; ++i) {
    *py++ += a * *px++;
  }
}

float sdotAvx2Kernel(int64_t n, const float *x, const float *y) {
  __m256 x00, y00, a00;

  a00 = _mm256_setzero_ps();

  int64_t nb = n / 8;
  int64_t nr = n % 8;

  const float *px = x;
  const float *py = y;
  for (int i = 0; i < nb; ++i) {
    x00 = _mm256_loadu_ps(px);
    y00 = _mm256_loadu_ps(py);
    a00 = _mm256_fmadd_ps(x00, y00, a00);

    px += 8;
    py += 8;
  }

  // unroll a00
  float sum = hsum(a00);
  for (int i = 0; i < nr; ++i) {
    sum += *px++ * *py++;
  }

  return sum;
}

LIBWAIFU_KERNEL_FORCE_INLINE float half2float(Float16 half) {
#if LIBWAIFU_KERNEL_MSVC
  return libwaifu_cvtsh_ss(*reinterpret_cast<uint16_t *>(&half));
#else
  return _cvtsh_ss(*reinterpret_cast<uint16_t *>(&half));
#endif
}

// dot of a fp32 x with a fp16 y. This is the w16a32 GEMV path: x is the activation, y a row of
// the fp16 weight, converted eight at a time on the way into the FMA.
float shdotAvx2Kernel(int64_t n, const float *x, const Float16 *y) {
  __m256 a00 = _mm256_setzero_ps();

  int64_t nb = n / 8;
  int64_t nr = n % 8;

  const float *px = x;
  const Float16 *py = y;
  for (int64_t i = 0; i < nb; ++i) {
    __m256 x00 = _mm256_loadu_ps(px);
    __m256 y00 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(py)));
    a00 = _mm256_fmadd_ps(x00, y00, a00);

    px += 8;
    py += 8;
  }

  float sum = hsum(a00);
  for (int64_t i = 0; i < nr; ++i) {
    sum += *px++ * half2float(*py++);
  }

  return sum;
}

// y (fp32) += a * x (fp16). The transposed half of the w16a32 GEMV path.
void hsaxpyAvx2Kernel(int64_t n, float a, const Float16 *x, float *y) {
  __m256 a00 = _mm256_broadcast_ss(&a);

  int64_t nb = n / 8;
  int64_t nr = n % 8;

  const Float16 *px = x;
  float *py = y;
  for (int64_t i = 0; i < nb; ++i) {
    __m256 x00 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(px)));
    __m256 y00 = _mm256_loadu_ps(py);

    y00 = _mm256_fmadd_ps(a00, x00, y00);
    _mm256_storeu_ps(py, y00);

    px += 8;
    py += 8;
  }

  for (int64_t i = 0; i < nr; ++i) {
    *py++ += a * half2float(*px++);
  }
}

// transpose the 8x8 block held in v0..v7 in place: on return v[i] holds what was element i of
// each of the eight inputs.
LIBWAIFU_KERNEL_FORCE_INLINE void transpose8x8Avx2(__m256 *v) {
  __m256 t0 = _mm256_unpacklo_ps(v[0], v[1]);
  __m256 t1 = _mm256_unpackhi_ps(v[0], v[1]);
  __m256 t2 = _mm256_unpacklo_ps(v[2], v[3]);
  __m256 t3 = _mm256_unpackhi_ps(v[2], v[3]);
  __m256 t4 = _mm256_unpacklo_ps(v[4], v[5]);
  __m256 t5 = _mm256_unpackhi_ps(v[4], v[5]);
  __m256 t6 = _mm256_unpacklo_ps(v[6], v[7]);
  __m256 t7 = _mm256_unpackhi_ps(v[6], v[7]);

  __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
  __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xee);
  __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
  __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xee);
  __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
  __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xee);
  __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
  __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xee);

  v[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
  v[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
  v[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
  v[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
  v[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
  v[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
  v[6] = _mm256_permute2f128_ps(s2, s6, 0x31);
  v[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
}

// tgt[r * tgtStride + c] = float(src[r + c * srcStride]).
//
// This is the packing loop for a transposed source, which is what both the A and the B pack hit
// in the layouts a Linear uses. Read that way, the source is numCols rows of numRows contiguous
// elements and the target is their transpose, so it is an ordinary out-of-place transpose with
// the conversion, if any, folded in. Doing it eight by eight keeps both sides at vector width:
// eight loads, an 8x8 transpose in registers, eight 32 byte stores. The element at a time version
// cannot vectorize the strided load at all, and for fp16 calls out to a software half_to_float
// for every element on top of that.
//
// load reads eight consecutive source elements as floats, cvtOne reads one. r is the outer loop,
// so the target is written straight through, eight rows at a time.
template<typename Ts, typename Load, typename CvtOne>
LIBWAIFU_KERNEL_FORCE_INLINE void packTransposeAvx2(
    int numRows,
    int numCols,
    const Ts *src,
    int64_t srcStride,
    float *tgt,
    int64_t tgtStride,
    Load load,
    CvtOne cvtOne) {
  int nr8 = numRows & ~7;
  int nc8 = numCols & ~7;

  for (int r0 = 0; r0 < nr8; r0 += 8) {
    for (int c0 = 0; c0 < nc8; c0 += 8) {
      const Ts *ps = src + r0 + static_cast<int64_t>(c0) * srcStride;

      // v[j] holds rows r0..r0+7 of column c0+j; after the transpose, v[i] holds row r0+i.
      __m256 v[8];
      for (int j = 0; j < 8; ++j) v[j] = load(ps + j * srcStride);
      transpose8x8Avx2(v);

      float *pt = tgt + static_cast<int64_t>(r0) * tgtStride + c0;
      for (int i = 0; i < 8; ++i) _mm256_storeu_ps(pt + i * tgtStride, v[i]);
    }
  }

  // the columns no whole tile covered, then the rows. MR is 12, so the column tail is real.
  for (int r = 0; r < nr8; ++r) {
    for (int c = nc8; c < numCols; ++c) {
      tgt[r * tgtStride + c] = cvtOne(src[r + c * srcStride]);
    }
  }
  for (int r = nr8; r < numRows; ++r) {
    for (int c = 0; c < numCols; ++c) {
      tgt[r * tgtStride + c] = cvtOne(src[r + c * srcStride]);
    }
  }
}

void hspackTransposeAvx2Kernel(
    int numRows,
    int numCols,
    const Float16 *src,
    int64_t srcStride,
    float *tgt,
    int64_t tgtStride) {
  packTransposeAvx2(
      numRows,
      numCols,
      src,
      srcStride,
      tgt,
      tgtStride,
      [](const Float16 *p) {
        return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p)));
      },
      [](Float16 v) { return half2float(v); });
}

void spackTransposeAvx2Kernel(
    int numRows,
    int numCols,
    const float *src,
    int64_t srcStride,
    float *tgt,
    int64_t tgtStride) {
  packTransposeAvx2(
      numRows,
      numCols,
      src,
      srcStride,
      tgt,
      tgtStride,
      [](const float *p) { return _mm256_loadu_ps(p); },
      [](float v) { return v; });
}

void hscvtAvx2Kernel(int64_t n, const Float16 *x, float *y) {
  int nb = n / 8;
  for (int i = 0; i < nb; ++i) {
    __m128i x0 = _mm_loadu_si128((const __m128i *)x);
    __m256 y0 = _mm256_cvtph_ps(x0);
    _mm256_storeu_ps(y, y0);

    x += 8;
    y += 8;
  }

  int nr = n % 8;
  if (nr == 0) return;

  Float16 xr[8];
  float yr[8];
  for (int i = 0; i < nr; ++i) {
    xr[i] = x[i];
  }
  __m128i x0 = _mm_loadu_si128((const __m128i *)xr);
  __m256 y0 = _mm256_cvtph_ps(x0);
  _mm256_storeu_ps(yr, y0);
  for (int i = 0; i < nr; ++i) {
    y[i] = yr[i];
  }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace fl
