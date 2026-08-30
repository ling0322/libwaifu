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


#include "flint/cpu/kernel/asimdfhm.h"

#include <arm_fp16.h>
#include <arm_neon.h>
#include <stdint.h>

namespace fl {
namespace op {
namespace cpu {
namespace kernel {

#define LIBWAIFU_GemmHalf6x16AsimdfhmKernel_LdC(m) \
  h00 = vld1q_f16(pc);                             \
  h01 = vld1q_f16(pc + 8);                         \
  c##m##0 = vcvt_f32_f16(vget_low_f16(h00));       \
  c##m##1 = vcvt_f32_f16(vget_high_f16(h00));      \
  c##m##2 = vcvt_f32_f16(vget_low_f16(h01));       \
  c##m##3 = vcvt_f32_f16(vget_high_f16(h01));      \
  pc += rs_c;

// fmlal widens as it multiplies, so b stays in half and only the four sums each instruction
// lands in are float. That is why sixteen columns take four of them where the ASIMDHP kernel
// needs four conversions and four multiplies.
#define LIBWAIFU_GemmHalf6x16AsimdfhmKernel_FmaRow(m, v, lane) \
  c##m##0 = vfmlalq_lane_low_f16(c##m##0, b00, v, lane);       \
  c##m##1 = vfmlalq_lane_high_f16(c##m##1, b00, v, lane);      \
  c##m##2 = vfmlalq_lane_low_f16(c##m##2, b01, v, lane);       \
  c##m##3 = vfmlalq_lane_high_f16(c##m##3, b01, v, lane);

#define LIBWAIFU_GemmHalf6x16AsimdfhmKernel_StC(m)                               \
  vst1q_f16(pc, vcombine_f16(vcvt_f16_f32(c##m##0), vcvt_f16_f32(c##m##1)));     \
  vst1q_f16(pc + 8, vcombine_f16(vcvt_f16_f32(c##m##2), vcvt_f16_f32(c##m##3))); \
  pc += rs_c;

/// hgemm6x16AsimdhpKernel with the widening folded into the multiply.
///
/// FEAT_FHM is optional in ARMv8.2 and plenty of parts do not have it, so this sits beside the
/// plain ASIMDHP kernel rather than replacing it; findBestCpuMathBackend picks between them. The
/// target attribute is what makes the intrinsics available without asking for fp16fml across the
/// whole build, which would fault on the parts that lack it.
///
/// The tile is the same six by sixteen, and for the same reason: the accumulator is float either
/// way, and four floats to a register is what bounds it. fmlal saves the conversions, not the
/// register pressure.
__attribute__((target("fp16fml"))) void hgemm6x16AsimdfhmKernel(
    int64_t kc,
    const Float16 *a,
    const Float16 *b,
    Float16 *c,
    int64_t rs_c) {
  // a: kc x MR
  // b: kc x NR

  // C: MR x NR (6 x 4 float32x4_t)
  float32x4_t c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, c30, c31, c32, c33,
      c40, c41, c42, c43, c50, c51, c52, c53;
  float16x8_t h00, h01, b00, b01;
  float16x4_t av0, av1;

  __fp16 *pc = reinterpret_cast<__fp16 *>(c);
  LIBWAIFU_GemmHalf6x16AsimdfhmKernel_LdC(0);
  LIBWAIFU_GemmHalf6x16AsimdfhmKernel_LdC(1);
  LIBWAIFU_GemmHalf6x16AsimdfhmKernel_LdC(2);
  LIBWAIFU_GemmHalf6x16AsimdfhmKernel_LdC(3);
  LIBWAIFU_GemmHalf6x16AsimdfhmKernel_LdC(4);
  LIBWAIFU_GemmHalf6x16AsimdfhmKernel_LdC(5);

  const __fp16 *pa = reinterpret_cast<const __fp16 *>(a);
  const __fp16 *pb = reinterpret_cast<const __fp16 *>(b);
  for (int64_t k = 0; k < kc; ++k) {
    b00 = vld1q_f16(pb);
    b01 = vld1q_f16(pb + 8);

    // Two overlapping four-lane loads rather than one of eight: the row of A is six wide, and
    // reading eight would run off the end of the panel on the last pass.
    av0 = vld1_f16(pa);
    av1 = vld1_f16(pa + 2);

    LIBWAIFU_GemmHalf6x16AsimdfhmKernel_FmaRow(0, av0, 0);
    LIBWAIFU_GemmHalf6x16AsimdfhmKernel_FmaRow(1, av0, 1);
    LIBWAIFU_GemmHalf6x16AsimdfhmKernel_FmaRow(2, av0, 2);
    LIBWAIFU_GemmHalf6x16AsimdfhmKernel_FmaRow(3, av0, 3);
    LIBWAIFU_GemmHalf6x16AsimdfhmKernel_FmaRow(4, av1, 2);
    LIBWAIFU_GemmHalf6x16AsimdfhmKernel_FmaRow(5, av1, 3);

    pa += 6;
    pb += 16;
  }

  pc = reinterpret_cast<__fp16 *>(c);
  LIBWAIFU_GemmHalf6x16AsimdfhmKernel_StC(0);
  LIBWAIFU_GemmHalf6x16AsimdfhmKernel_StC(1);
  LIBWAIFU_GemmHalf6x16AsimdfhmKernel_StC(2);
  LIBWAIFU_GemmHalf6x16AsimdfhmKernel_StC(3);
  LIBWAIFU_GemmHalf6x16AsimdfhmKernel_StC(4);
  LIBWAIFU_GemmHalf6x16AsimdfhmKernel_StC(5);
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace fl
