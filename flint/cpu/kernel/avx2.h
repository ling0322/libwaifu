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

#pragma once

#include <stdint.h>

#include "lutil/log.h"
#include "flint/cpu/kernel/abstract.h"

namespace fl {
namespace op {
namespace cpu {
namespace kernel {

void hscvtAvx2Kernel(int64_t n, const Float16 *x, float *y);
void sgemm6x16Avx2Kernel(int64_t kc, const float *a, const float *b, float *c, int64_t rs_c);
float sdotAvx2Kernel(int64_t n, const float *x, const float *y);
void saxpyAvx2Kernel(int64_t n, float a, const float *x, float *y);
float shdotAvx2Kernel(int64_t n, const float *x, const Float16 *y);
void hsaxpyAvx2Kernel(int64_t n, float a, const Float16 *x, float *y);
void hspackTransposeAvx2Kernel(
    int numRows,
    int numCols,
    const Float16 *src,
    int64_t srcStride,
    float *tgt,
    int64_t tgtStride);
void spackTransposeAvx2Kernel(
    int numRows,
    int numCols,
    const float *src,
    int64_t srcStride,
    float *tgt,
    int64_t tgtStride);

template<>
inline void packTransposeKernel<float, float, CpuMathBackend::AVX2>(
    int numRows,
    int numCols,
    const float *src,
    int64_t srcStride,
    float *tgt,
    int64_t tgtStride) {
  return spackTransposeAvx2Kernel(numRows, numCols, src, srcStride, tgt, tgtStride);
}

template<>
inline void packTransposeKernel<Float16, float, CpuMathBackend::AVX2>(
    int numRows,
    int numCols,
    const Float16 *src,
    int64_t srcStride,
    float *tgt,
    int64_t tgtStride) {
  return hspackTransposeAvx2Kernel(numRows, numCols, src, srcStride, tgt, tgtStride);
}

template<>
inline void cvtKernel<Float16, float, CpuMathBackend::AVX2>(
    int n,
    const Float16 *x,
    int64_t offsetX,
    float *y,
    int64_t offsetY) {
  return hscvtAvx2Kernel(n, x + offsetX, y + offsetY);
}
template<>
inline void gemmKernel<float, float, float, 6, 16, CpuMathBackend::AVX2>(
    int64_t kc,
    const float *a,
    const float *b,
    float *c,
    int64_t rs_c) {
  return sgemm6x16Avx2Kernel(kc, a, b, c, rs_c);
}

template<>
inline float dotKernel<float, float, float, CpuMathBackend::AVX2>(
    int64_t n,
    const float *x,
    const float *y,
    int64_t offsetY) {
  return sdotAvx2Kernel(n, x, y + offsetY);
}
template<>
inline void axpyKernel<float, float, float, CpuMathBackend::AVX2>(
    int64_t n,
    float a,
    const float *x,
    int64_t offsetX,
    float *y) {
  return saxpyAvx2Kernel(n, a, x + offsetX, y);
}

template<>
inline float dotKernel<float, float, Float16, CpuMathBackend::AVX2>(
    int64_t n,
    const float *x,
    const Float16 *y,
    int64_t offsetY) {
  return shdotAvx2Kernel(n, x, y + offsetY);
}
template<>
inline void axpyKernel<float, Float16, float, CpuMathBackend::AVX2>(
    int64_t n,
    float a,
    const Float16 *x,
    int64_t offsetX,
    float *y) {
  return hsaxpyAvx2Kernel(n, a, x + offsetX, y);
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace fl
