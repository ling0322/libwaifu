// The MIT License (MIT)
//
// Copyright (c) 2023-2024 Xiaoyang Chen
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

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lutil/shared_library.h"
#include "flint/cuda/common.h"
#include "flint/cuda/gemm.h"
#include "flint/cuda/matvec.h"
#include "flint/tensor.h"
#include "flint/tensor_view.h"

namespace fl {
namespace op {
namespace cuda {

class MatMul {
 public:
  static std::shared_ptr<MatMul> create();
  static std::shared_ptr<MatMul> createCutlass();
  static std::shared_ptr<MatMul> createCublas();

  /// C = A @ B, written into `C`: contiguous, A's batch and rows by B's columns, of A's type -- or
  /// float where A and B are both half, the sum as it was before half would round it.
  void apply(const TensorView &A, const TensorView &B, const TensorView &C);

  /// C = A @ B + addend, `addend` of C's type broadcast over C's leading dimensions -- a bias (N),
  /// or a residual of C's shape. It is written into C first and the GEMM adds the product to it
  /// (beta = 1) -- the vector kernel to its float sum -- so the two are rounded once, as torch's
  /// addmm rounds them. An empty `addend` is apply() above.
  void apply(
      const TensorView &A,
      const TensorView &B,
      const TensorView &addend,
      const TensorView &C);

 protected:
  std::shared_ptr<Gemm> _gemm;

  // The float paths, in `T`, which is <half> or <float>, into a `TC` that is `T` or, for <half>,
  // <float>. They differ only in which cuBLAS call they end at and in the vector kernel, which
  // exists for half alone; everything about the shapes is the same, so they are one body rather
  // than three. Defined in matmul.cc and used only there.
  // `bias` is empty, or (N) of TC, contiguous, to add to every row; `beta` is 0, or 1 to add the
  // product to what C already holds -- see apply().
  template<typename T, typename TC = T>
  void gemm(
      const TensorView &A,
      const TensorView &B,
      const TensorView &bias,
      float beta,
      const TensorView &C);
  template<typename T, typename TC = T>
  void bmm(
      const TensorView &A,
      const TensorView &B,
      const TensorView &bias,
      float beta,
      const TensorView &C);
  template<typename T, typename TC = T>
  void matmulFloat(
      const TensorView &A,
      const TensorView &B,
      const TensorView &bias,
      float beta,
      const TensorView &C);
  template<typename T, typename TC = T>
  void bmmToGemm(
      const TensorView &A,
      const TensorView &B,
      const TensorView &bias,
      float beta,
      const TensorView &C);
  template<typename T>
  std::vector<const T *> getBatch(const TensorView &A, int nBatchDim);
};

}  // namespace cuda
}  // namespace op
}  // namespace fl
