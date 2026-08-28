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

namespace fl {
namespace op {
namespace cuda {

class MatMul {
 public:
  static std::shared_ptr<MatMul> create();
  static std::shared_ptr<MatMul> createCutlass();
  static std::shared_ptr<MatMul> createCublas();

  Tensor apply(const Tensor &A, const Tensor &B);
  Tensor applyNarrowPrecision(
      const Tensor &A,
      const Tensor &sfA,
      const Tensor &B,
      const Tensor &sfB);

 protected:
  std::shared_ptr<Gemm> _gemm;

  // The float paths, in `T`, which is <half> or <float>. The two differ only in which cuBLAS
  // call they end at and in the vector kernel, which exists for half alone; everything about the
  // shapes is the same, so they are one body rather than two. Defined in matmul.cc and used only
  // there.
  template<typename T>
  Tensor gemm(Tensor A, Tensor B);
  template<typename T>
  Tensor bmm(Tensor A, Tensor B);
  template<typename T>
  Tensor matmulFloat(const Tensor &A, const Tensor &B);
  template<typename T>
  Tensor bmmToGemm(const Tensor &A, const Tensor &B);
  template<typename T>
  std::vector<const T *> getBatch(const Tensor &A, int nBatchDim);

  Tensor matmulQ4(const Tensor &A, const Tensor &B);
  Tensor gemmQ4(const Tensor &A, const Tensor &B);
  Tensor bmmToGemmQ4(const Tensor &A, const Tensor &B);

  Tensor matmulMxfp4(const Tensor &A, const Tensor &sfA, const Tensor &B, const Tensor &sfB);
};

}  // namespace cuda
}  // namespace op
}  // namespace fl
