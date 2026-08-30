// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include <arm_fp16.h>
#include <arm_neon.h>
#include <assert.h>
#include <stdint.h>

#include "flint/cpu/kernel/abstract.h"

namespace fl {
namespace op {
namespace cpu {
namespace kernel {

#define LIBWAIFU_GemmFloat6x16AsimdhpKernel_LdC(m) \
  c##m##0 = vld1q_f32(pc);                         \
  c##m##1 = vld1q_f32(pc + 4);                     \
  c##m##2 = vld1q_f32(pc + 8);                     \
  c##m##3 = vld1q_f32(pc + 12);                    \
  pc += rs_c;

#define LIBWAIFU_GemmFloat6x16AsimdhpKernel_FmaRow(m)  \
  a00 = vdupq_n_f32(pa[m]);                            \
  c##m##0 = vfmaq_f32(c##m##0, a00, b00);              \
  c##m##1 = vfmaq_f32(c##m##1, a00, b01);              \
  c##m##2 = vfmaq_f32(c##m##2, a00, b02);              \
  c##m##3 = vfmaq_f32(c##m##3, a00, b03);

#define LIBWAIFU_GemmFloat6x16AsimdhpKernel_StC(m) \
  vst1q_f32(pc, c##m##0);                          \
  vst1q_f32(pc + 4, c##m##1);                      \
  vst1q_f32(pc + 8, c##m##2);                      \
  vst1q_f32(pc + 12, c##m##3);                     \
  pc += rs_c;

/// The float counterpart of hgemm12x16: 6 rows of 16 columns, which is 24 accumulators of four
/// floats. With the four b vectors and the broadcast a that is 29 of the 32 vector registers,
/// so nothing spills. 12 rows would not fit, which is why this is 6 wide where the half kernel
/// is 12.
void sgemm6x16AsimdhpKernel(int64_t kc, const float *a, const float *b, float *c, int64_t rs_c) {
  // a: kc x MR
  // b: kc x NR

  // C: MR x NR (6 x 4 float32x4_t)
  float32x4_t c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33,
      c40, c41, c42, c43, c50, c51, c52, c53;
  float32x4_t a00, b00, b01, b02, b03;

  float *pc = c;
  LIBWAIFU_GemmFloat6x16AsimdhpKernel_LdC(0);
  LIBWAIFU_GemmFloat6x16AsimdhpKernel_LdC(1);
  LIBWAIFU_GemmFloat6x16AsimdhpKernel_LdC(2);
  LIBWAIFU_GemmFloat6x16AsimdhpKernel_LdC(3);
  LIBWAIFU_GemmFloat6x16AsimdhpKernel_LdC(4);
  LIBWAIFU_GemmFloat6x16AsimdhpKernel_LdC(5);

  const float *pa = a;
  const float *pb = b;
  for (int64_t k = 0; k < kc; ++k) {
    b00 = vld1q_f32(pb);
    b01 = vld1q_f32(pb + 4);
    b02 = vld1q_f32(pb + 8);
    b03 = vld1q_f32(pb + 12);

    LIBWAIFU_GemmFloat6x16AsimdhpKernel_FmaRow(0);
    LIBWAIFU_GemmFloat6x16AsimdhpKernel_FmaRow(1);
    LIBWAIFU_GemmFloat6x16AsimdhpKernel_FmaRow(2);
    LIBWAIFU_GemmFloat6x16AsimdhpKernel_FmaRow(3);
    LIBWAIFU_GemmFloat6x16AsimdhpKernel_FmaRow(4);
    LIBWAIFU_GemmFloat6x16AsimdhpKernel_FmaRow(5);

    pa += 6;
    pb += 16;
  }

  pc = c;
  LIBWAIFU_GemmFloat6x16AsimdhpKernel_StC(0);
  LIBWAIFU_GemmFloat6x16AsimdhpKernel_StC(1);
  LIBWAIFU_GemmFloat6x16AsimdhpKernel_StC(2);
  LIBWAIFU_GemmFloat6x16AsimdhpKernel_StC(3);
  LIBWAIFU_GemmFloat6x16AsimdhpKernel_StC(4);
  LIBWAIFU_GemmFloat6x16AsimdhpKernel_StC(5);
}

/// A float activation against a half weight, accumulated in float. The weight is widened as it is
/// loaded rather than beforehand, so the row of B is only ever read at its stored size.
float shdotAsimdhpKernel(int64_t n, const float *x, const Float16 *y) {
  float32x4_t sum0 = vdupq_n_f32(0), sum1 = vdupq_n_f32(0);
  float16x8_t y00;

  int64_t nb = n / 8;
  int64_t nr = n % 8;

  const float *px = x;
  const __fp16 *py = reinterpret_cast<const __fp16 *>(y);
  for (int64_t i = 0; i < nb; ++i) {
    y00 = vld1q_f16(py);
    sum0 = vfmaq_f32(sum0, vld1q_f32(px), vcvt_f32_f16(vget_low_f16(y00)));
    sum1 = vfmaq_f32(sum1, vld1q_f32(px + 4), vcvt_f32_f16(vget_high_f16(y00)));

    px += 8;
    py += 8;
  }

  float sum = vaddvq_f32(vaddq_f32(sum0, sum1));
  for (int64_t i = 0; i < nr; ++i) {
    sum += *px * static_cast<float>(*py);
    ++px;
    ++py;
  }

  return sum;
}

/// As hsaxpyAsimdhpKernel, but the scalar arrives in float and stays there. Rounding it to half
/// first would throw away precision the caller has, for nothing.
void hsaxpyFloatAsimdhpKernel(int64_t n, float a, const Float16 *x, float *y) {
  float32x4_t a00 = vdupq_n_f32(a);
  float32x4_t x00, y00;

  int64_t nb = n / 4;
  int64_t nr = n % 4;

  const __fp16 *px = reinterpret_cast<const __fp16 *>(x);
  float *py = y;
  for (int64_t i = 0; i < nb; ++i) {
    x00 = vcvt_f32_f16(vld1_f16(px));
    y00 = vld1q_f32(py);

    y00 = vfmaq_f32(y00, x00, a00);
    vst1q_f32(py, y00);

    px += 4;
    py += 4;
  }

  for (int64_t i = 0; i < nr; ++i) {
    *py += a * static_cast<float>(*px);
    ++px;
    ++py;
  }
}

void hsaxpyAsimdhpKernel(int64_t n, Float16 a, const Float16 *x, float *y) {
  float32x4_t a00 = vcvt_f32_f16(vld1_dup_f16(reinterpret_cast<__fp16 *>(&a)));
  float32x4_t x00, y00;

  int64_t nb = n / 4;
  int64_t nr = n % 4;

  const __fp16 *px = reinterpret_cast<const __fp16 *>(x);
  float *py = y;
  for (int i = 0; i < nb; ++i) {
    x00 = vcvt_f32_f16(vld1_f16(px));
    y00 = vld1q_f32(py);

    y00 = vfmaq_f32(y00, x00, a00);
    vst1q_f32(py, y00);

    px += 4;
    py += 4;
  }

  for (int i = 0; i < nr; ++i) {
    *py += a * *px;
    ++px;
    ++py;
  }
}

#define LIBWAIFU_DotHalfAsimdhpKernel_FmaBlock \
  x00 = vld1q_f16(px);                       \
  y00 = vld1q_f16(py);                       \
  ha00 = vfmaq_f16(ha00, x00, y00);          \
  px += 8;                                   \
  py += 8;

Float16 hdotAsimdhpKernel(int64_t n, const Float16 *x, const Float16 *y) {
  float16x8_t x00, y00, ha00;
  float32x4_t sa00, sa01;

  sa00 = vdupq_n_f32(0);
  sa01 = vdupq_n_f32(0);

  int64_t nb = n / 64;
  int64_t nr = n % 64;

  const __fp16 *px = reinterpret_cast<const __fp16 *>(x);
  const __fp16 *py = reinterpret_cast<const __fp16 *>(y);
  for (int i = 0; i < nb; ++i) {
    ha00 = vdupq_n_f16(0);

    LIBWAIFU_DotHalfAsimdhpKernel_FmaBlock;
    LIBWAIFU_DotHalfAsimdhpKernel_FmaBlock;
    LIBWAIFU_DotHalfAsimdhpKernel_FmaBlock;
    LIBWAIFU_DotHalfAsimdhpKernel_FmaBlock;
    LIBWAIFU_DotHalfAsimdhpKernel_FmaBlock;
    LIBWAIFU_DotHalfAsimdhpKernel_FmaBlock;
    LIBWAIFU_DotHalfAsimdhpKernel_FmaBlock;
    LIBWAIFU_DotHalfAsimdhpKernel_FmaBlock;

    sa00 = vaddq_f32(sa00, vcvt_f32_f16(vget_low_f16(ha00)));
    sa01 = vaddq_f32(sa01, vcvt_f32_f16(vget_high_f16(ha00)));
  }

  __fp16 hsum1 = 0.0;
  for (int i = 0; i < nr; ++i) {
    hsum1 = vfmah_f16(hsum1, *px, *py);
    ++px;
    ++py;
  }
  sa00 = vaddq_f32(sa00, vcvt_f32_f16(vset_lane_f16(hsum1, vdup_n_f16(0), 0)));

  // unroll a00
  sa00 = vpaddq_f32(sa00, sa01);
  sa00 = vpaddq_f32(sa00, sa00);
  sa00 = vpaddq_f32(sa00, sa00);
  float sum0 = vgetq_lane_f32(sa00, 0);

  return vget_lane_f16(vcvt_f16_f32(vsetq_lane_f32(sum0, vdupq_n_f32(0), 0)), 0);
}

#define LIBWAIFU_GemmHalf12x16AsimdhpKernel_FmaBlock(m, r, rl) \
  c##m##0 = vfmaq_lane_f16(c##m##0, b00, ra0##r, rl);        \
  c##m##1 = vfmaq_lane_f16(c##m##1, b01, ra0##r, rl);

#define LIBWAIFU_GemmHalf12x16AsimdhpKernel_LdStC(m) \
  a00 = vld1q_f16(pc);                             \
  c##m##0 = vaddq_f16(a00, c##m##0);               \
  vst1q_f16(pc, c##m##0);                          \
  a00 = vld1q_f16(pc + 8);                         \
  c##m##1 = vaddq_f16(a00, c##m##1);               \
  vst1q_f16(pc + 8, c##m##1);                      \
  pc += rs_c;

void hgemm12x16AsimdhpKernel(
    int64_t kc,
    const Float16 *a,
    const Float16 *b,
    Float16 *c,
    int64_t rs_c) {
  // a: kc x MR
  // b: kc x NR

  // C: MR x NR
  float16x8_t c00 = vdupq_n_f16(0), c01 = vdupq_n_f16(0), c10 = vdupq_n_f16(0),
              c11 = vdupq_n_f16(0), c20 = vdupq_n_f16(0), c21 = vdupq_n_f16(0),
              c30 = vdupq_n_f16(0), c31 = vdupq_n_f16(0), c40 = vdupq_n_f16(0),
              c41 = vdupq_n_f16(0), c50 = vdupq_n_f16(0), c51 = vdupq_n_f16(0),
              c60 = vdupq_n_f16(0), c61 = vdupq_n_f16(0), c70 = vdupq_n_f16(0),
              c71 = vdupq_n_f16(0), c80 = vdupq_n_f16(0), c81 = vdupq_n_f16(0),
              c90 = vdupq_n_f16(0), c91 = vdupq_n_f16(0), ca0 = vdupq_n_f16(0),
              ca1 = vdupq_n_f16(0), cb0 = vdupq_n_f16(0), cb1 = vdupq_n_f16(0);
  float16x8_t a00, b00, b01;
  float16x4_t ra00, ra01, ra02;

  const __fp16 *pa = reinterpret_cast<const __fp16 *>(a);
  const __fp16 *pb = reinterpret_cast<const __fp16 *>(b);

  for (int64_t k = 0; k < kc; ++k) {
    ra00 = vld1_f16(pa);
    pa += 4;
    ra01 = vld1_f16(pa);
    pa += 4;
    ra02 = vld1_f16(pa);
    pa += 4;

    b00 = vld1q_f16(pb);
    pb += 8;
    b01 = vld1q_f16(pb);
    pb += 8;

    LIBWAIFU_GemmHalf12x16AsimdhpKernel_FmaBlock(0, 0, 0);
    LIBWAIFU_GemmHalf12x16AsimdhpKernel_FmaBlock(1, 0, 1);
    LIBWAIFU_GemmHalf12x16AsimdhpKernel_FmaBlock(2, 0, 2);
    LIBWAIFU_GemmHalf12x16AsimdhpKernel_FmaBlock(3, 0, 3);
    LIBWAIFU_GemmHalf12x16AsimdhpKernel_FmaBlock(4, 1, 0);
    LIBWAIFU_GemmHalf12x16AsimdhpKernel_FmaBlock(5, 1, 1);
    LIBWAIFU_GemmHalf12x16AsimdhpKernel_FmaBlock(6, 1, 2);
    LIBWAIFU_GemmHalf12x16AsimdhpKernel_FmaBlock(7, 1, 3);
    LIBWAIFU_GemmHalf12x16AsimdhpKernel_FmaBlock(8, 2, 0);
    LIBWAIFU_GemmHalf12x16AsimdhpKernel_FmaBlock(9, 2, 1);
    LIBWAIFU_GemmHalf12x16AsimdhpKernel_FmaBlock(a, 2, 2);
    LIBWAIFU_GemmHalf12x16AsimdhpKernel_FmaBlock(b, 2, 3);
  }

  __fp16 *pc = reinterpret_cast<__fp16 *>(c);

  LIBWAIFU_GemmHalf12x16AsimdhpKernel_LdStC(0);
  LIBWAIFU_GemmHalf12x16AsimdhpKernel_LdStC(1);
  LIBWAIFU_GemmHalf12x16AsimdhpKernel_LdStC(2);
  LIBWAIFU_GemmHalf12x16AsimdhpKernel_LdStC(3);
  LIBWAIFU_GemmHalf12x16AsimdhpKernel_LdStC(4);
  LIBWAIFU_GemmHalf12x16AsimdhpKernel_LdStC(5);
  LIBWAIFU_GemmHalf12x16AsimdhpKernel_LdStC(6);
  LIBWAIFU_GemmHalf12x16AsimdhpKernel_LdStC(7);
  LIBWAIFU_GemmHalf12x16AsimdhpKernel_LdStC(8);
  LIBWAIFU_GemmHalf12x16AsimdhpKernel_LdStC(9);
  LIBWAIFU_GemmHalf12x16AsimdhpKernel_LdStC(a);
  LIBWAIFU_GemmHalf12x16AsimdhpKernel_LdStC(b);
}

#define LIBWAIFU_CvtHalfToFloatAsimdhpKernel_CvtBlock \
  x00 = vld1q_f16(px);                              \
  y00 = vcvt_f32_f16(vget_low_f16(x00));            \
  y01 = vcvt_f32_f16(vget_high_f16(x00));           \
  vst1q_f32(py, y00);                               \
  vst1q_f32(py + 4, y01);                           \
  px += 8;                                          \
  py += 8;

void hscvtAsimdhpKernel(int64_t n, const Float16 *x, float *y) {
  float16x8_t x00;
  float32x4_t y00, y01;

  int64_t nb = n / 8;
  int64_t nr = n % 8;

  const __fp16 *px = reinterpret_cast<const __fp16 *>(x);
  float *py = y;
  for (int64_t i = 0; i < nb; ++i) {
    LIBWAIFU_CvtHalfToFloatAsimdhpKernel_CvtBlock;
  }

  for (int64_t i = 0; i < nr; ++i) {
    *py = *reinterpret_cast<const Float16 *>(px);
    ++py;
    ++px;
  }
}

#define LIBWAIFU_CvtFloatToHalfAsimdhpKernel_CvtBlock         \
  x00 = vld1q_f32(px);                                      \
  x01 = vld1q_f32(px + 4);                                  \
  y00 = vcombine_f16(vcvt_f16_f32(x00), vcvt_f16_f32(x01)); \
  vst1q_f16(py, y00);                                       \
  px += 8;                                                  \
  py += 8;

void shcvtAsimdhpKernel(int64_t n, const float *x, Float16 *y) {
  float32x4_t x00, x01;
  float16x8_t y00;

  int64_t nb = n / 8;
  int64_t nr = n % 8;

  const float *px = x;
  __fp16 *py = reinterpret_cast<__fp16 *>(y);
  for (int64_t i = 0; i < nb; ++i) {
    LIBWAIFU_CvtFloatToHalfAsimdhpKernel_CvtBlock;
  }

  for (int64_t i = 0; i < nr; ++i) {
    *reinterpret_cast<Float16 *>(py) = *px;
    ++py;
    ++px;
  }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace fl
