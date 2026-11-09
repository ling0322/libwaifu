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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "flint/cuda/common.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cuda {

class Gemm {
 public:
  virtual ~Gemm() = default;

  virtual void hgemm(
      bool transA,
      bool transB,
      int m,
      int n,
      int k,
      __half alpha,
      const __half *A,
      int lda,
      const __half *B,
      int ldb,
      __half beta,
      __half *C,
      int ldc) {
    NOT_IMPL();
  }

  /// C = alpha A B + beta S: the GEMM above with a source read through its own leading dimension
  /// rather than C's -- 0 for one row read down every row of C, which is a bias. Returns false,
  /// with nothing done, where the backend cannot read the operands that way; the caller then puts
  /// the source into C and calls hgemm() with beta.
  virtual bool hgemmFrom(
      bool transA,
      bool transB,
      int m,
      int n,
      int k,
      float alpha,
      const __half *A,
      int lda,
      const __half *B,
      int ldb,
      float beta,
      const __half *S,
      int lds,
      __half *C,
      int ldc) {
    return false;
  }

  virtual void sgemm(
      bool transA,
      bool transB,
      int m,
      int n,
      int k,
      float alpha,
      const float *A,
      int lda,
      const float *B,
      int ldb,
      float beta,
      float *C,
      int ldc) {
    NOT_IMPL();
  }

  /// Half operands into a float C, accumulated in float: the sum as it was before half would round
  /// it. See Operators::matmulWidensHalf.
  virtual void hgemmIntoFloat(
      bool transA,
      bool transB,
      int m,
      int n,
      int k,
      float alpha,
      const __half *A,
      int lda,
      const __half *B,
      int ldb,
      float beta,
      float *C,
      int ldc) {
    NOT_IMPL();
  }

  virtual void hgemmArray(
      bool transA,
      bool transB,
      int m,
      int n,
      int k,
      __half alpha,
      const __half *const *arrayA,
      int lda,
      const __half *const *arrayB,
      int ldb,
      __half beta,
      __half *const *arrayC,
      int ldc,
      int batchSize) {
    NOT_IMPL();
  }

  virtual void hgemmIntoFloatArray(
      bool transA,
      bool transB,
      int m,
      int n,
      int k,
      float alpha,
      const __half *const *arrayA,
      int lda,
      const __half *const *arrayB,
      int ldb,
      float beta,
      float *const *arrayC,
      int ldc,
      int batchSize) {
    NOT_IMPL();
  }

  virtual void sgemmArray(
      bool transA,
      bool transB,
      int m,
      int n,
      int k,
      float alpha,
      const float *const *arrayA,
      int lda,
      const float *const *arrayB,
      int ldb,
      float beta,
      float *const *arrayC,
      int ldc,
      int batchSize) {
    NOT_IMPL();
  }
};

}  // namespace cuda
}  // namespace op
}  // namespace fl
