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

#include "flint/cpu/matmul.h"

#include "lutil/strings.h"
#include "flint/cpu/accessor.h"
#include "flint/cpu/common.h"
#include "flint/cpu/kernel/interface.h"
#include "flint/cpu/fill.h"

namespace fl {
namespace op {
namespace cpu {

std::vector<int> getBmmOutputShape(const TensorView &A, const TensorView &B) {
  CHECK(A.getDim() >= B.getDim());
  CHECK(A.getDim() > 2 && A.getDim() <= 4 && B.getDim() >= 2);
  std::vector<int> shape;

  // broadcast B
  int broadcastDims = A.getDim() - B.getDim();
  for (int i = 0; i < broadcastDims; ++i) {
    shape.push_back(A.getShape(i));
  }

  // batch dim: B.shape(i) == A.shape(broadcastDims + i)
  int batchDims = B.getDim() - 2;
  for (int i = 0; i < batchDims; ++i) {
    CHECK(A.getShape(broadcastDims + i) == B.getShape(i));
    shape.push_back(B.getShape(i));
  }

  shape.push_back(A.getShape(-2));
  shape.push_back(B.getShape(-1));

  return shape;
}

GEMMArgs generateGemmArgs(const TensorView &A, const TensorView &B, const TensorView &C) {
  CHECK(A.getDim() >= B.getDim() && A.getDim() == C.getDim());
  CHECK(B.getDim() >= 2);
  CHECK(A.getShape(-2) == C.getShape(-2));
  CHECK(A.getShape(-1) == B.getShape(-2));
  CHECK(B.getShape(-1) == C.getShape(-1));

  bool transA, transB;
  int lda, ldb;
  if (A.getStride(-1) == 1) {
    transA = false;
    lda = A.getStride(-2);
  } else if (A.getStride(-2) == 1) {
    transA = true;
    lda = A.getStride(-1);
  } else {
    NOT_IMPL();
  }

  if (B.getStride(-1) == 1) {
    transB = false;
    ldb = B.getStride(-2);
  } else if (B.getStride(-2) == 1) {
    transB = true;
    ldb = B.getStride(-1);
  } else {
    NOT_IMPL();
  }

  int m = A.getShape(-2);
  int k = A.getShape(-1);
  int n = B.getShape(-1);
  int ldc = C.getStride(-2);

  GEMMArgs gemmArgs;
  gemmArgs.K = k;
  gemmArgs.lda = lda;
  gemmArgs.ldb = ldb;
  gemmArgs.ldc = ldc;
  gemmArgs.M = m;
  gemmArgs.N = n;
  gemmArgs.transA = transA;
  gemmArgs.transB = transB;

  return gemmArgs;
}

template<typename T>
void callGemm(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const T *A,
    int lda,
    const T *B,
    int ldb,
    T *C,
    int ldc,
    kernel::Mode mode);

template<>
inline void callGemm<float>(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    float *C,
    int ldc,
    kernel::Mode mode) {
  return kernel::gemmFloat(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, mode);
}

template<>
inline void callGemm<Float16>(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const Float16 *A,
    int lda,
    const Float16 *B,
    int ldb,
    Float16 *C,
    int ldc,
    kernel::Mode mode) {
  const kernel::Float16 *xA = reinterpret_cast<const kernel::Float16 *>(A);
  const kernel::Float16 *xB = reinterpret_cast<const kernel::Float16 *>(B);
  kernel::Float16 *xC = reinterpret_cast<kernel::Float16 *>(C);
  return kernel::gemmHalf(transA, transB, M, N, K, xA, lda, xB, ldb, xC, ldc, mode);
}

template<typename T>
void gemm(const TensorView &A, const TensorView &B, const TensorView &C) {
  CHECK(A.getDim() == B.getDim() && A.getDim() == 2);
  C.throwIfInvalidShape({A.getShape(0), B.getShape(1)}, "gemm");

  // The kernels accumulate into C rather than overwrite it.
  fill(C, 0.0f);

  GEMMArgs gemmArgs = generateGemmArgs(A, B, C);
  callGemm<T>(
      gemmArgs.transA,
      gemmArgs.transB,
      gemmArgs.M,
      gemmArgs.N,
      gemmArgs.K,
      getDataPtrCpu<T>(A),
      gemmArgs.lda,
      getDataPtrCpu<T>(B),
      gemmArgs.ldb,
      getDataPtrCpu<T>(C),
      gemmArgs.ldc,
      kernel::Mode::OMP);
}

/// A float activation against a half weight, which is how a model is held on the CPU: the weight
/// is left as the file stored it and only what flows between the layers is widened.
///
/// x64 has no half arithmetic to speak of, so the alternative is widening the weights as they are
/// read, which doubles the model -- 13.74 GB against 6.97 for SDXL. The micro-kernel converts the
/// weight as it packs it, so nothing is widened in memory and the arithmetic is the float32 it
/// would have been anyway.
void gemmHalfWeight(const TensorView &A, const TensorView &B, const TensorView &C) {
  CHECK(A.getDim() == 2 && B.getDim() == 2);
  CHECK(A.getDType() == DType::kFloat && B.getDType() == DType::kFloat16);
  C.throwIfInvalidShape({A.getShape(0), B.getShape(1)}, "gemm");

  // The kernel accumulates into C rather than overwrite it.
  fill(C, 0.0f);

  GEMMArgs gemmArgs = generateGemmArgs(A, B, C);
  kernel::gemmHalfWeightFloat(
      gemmArgs.transA,
      gemmArgs.transB,
      gemmArgs.M,
      gemmArgs.N,
      gemmArgs.K,
      getDataPtrCpu<float>(A),
      gemmArgs.lda,
      reinterpret_cast<const kernel::Float16 *>(getDataPtrCpu<Float16>(B)),
      gemmArgs.ldb,
      getDataPtrCpu<float>(C),
      gemmArgs.ldc,
      kernel::Mode::OMP);
}

/// The same for an activation of any rank against a two dimensional weight, which is what a
/// linear layer hands over: everything but the last axis is one long batch of rows.
void matmulHalfWeight(const TensorView &A, const TensorView &B, const TensorView &C) {
  if (A.getDim() == 2) return gemmHalfWeight(A, B, C);

  if (A.getDim() > 2 && B.getDim() == 2 && A.isContiguous() && C.isContiguous()) {
    return gemmHalfWeight(A.view({-1, A.getShape(-1)}), B, C.view({-1, B.getShape(1)}));
  }

  // Two batched operands are two activations -- attention's scores by its values -- and neither
  // of those is a weight, so there is nothing here for a half one to be.
  NOT_IMPL();
}

template<typename T>
void bmmNx2(const TensorView &A, const TensorView &B, const TensorView &C) {
  CHECK(C.isContiguous());
  gemm<T>(A.view({-1, A.getShape(-1)}), B, C.view({-1, B.getShape(1)}));
}

template<typename T>
void bmm(const TensorView &A, const TensorView &B, const TensorView &C) {
  TensorView xB = B;
  if (A.getDim() != B.getDim()) xB = expandBatchDims(B, A.getShape());
  C.throwIfInvalidShape(getBmmOutputShape(A, xB), "bmm");

  // The kernels accumulate into C rather than overwrite it.
  fill(C, 0.0f);

  TensorList<const T, 2> mA = TensorList<const T, 2>::fromTensor(A);
  TensorList<const T, 2> mB = TensorList<const T, 2>::fromTensor(xB);
  TensorList<T, 2> mC = TensorList<T, 2>::fromTensor(C);

  GEMMArgs gemmArgs = generateGemmArgs(A, xB, C);

  // broadcast B.
  CHECK(mA.getLength() == mC.getLength());
  CHECK(mA.getLength() % mB.getLength() == 0);

  const T *const *mAp = mA.getDataPtrList().data();
  const T *const *mBp = mB.getDataPtrList().data();
  T *const *mCp = mC.getDataPtrList().data();

  int numMatrices = mA.getLength();
#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < numMatrices; ++i) {
    callGemm<T>(
        gemmArgs.transA,
        gemmArgs.transB,
        gemmArgs.M,
        gemmArgs.N,
        gemmArgs.K,
        mAp[i],
        gemmArgs.lda,
        mBp[i],
        gemmArgs.ldb,
        mCp[i],
        gemmArgs.ldc,
        kernel::Mode::SingleThread);
  }
}

template<typename T>
void matmulFloat(const TensorView &A, const TensorView &B, const TensorView &C) {
  if (A.getDim() == 2 && B.getDim() == 2) {
    gemm<T>(A, B, C);
  } else if (A.getDim() > 2 && A.isContiguous() && B.getDim() == 2 && C.isContiguous()) {
    bmmNx2<T>(A, B, C);
  } else if (A.getDim() >= 2 && B.getDim() >= 2) {
    bmm<T>(A, B, C);
  } else {
    NOT_IMPL();
  }
}

void matmul(const TensorView &A, const TensorView &B, const TensorView &C) {
  DType typeA = A.getDType();
  DType typeB = B.getDType();

  if (typeA == DType::kFloat && typeB == DType::kFloat) return matmulFloat<float>(A, B, C);
  if (typeA == DType::kFloat && typeB == DType::kFloat16) return matmulHalfWeight(A, B, C);

#if LUT_CPU_ARCH == LUT_AARCH64
  if (typeA == DType::kFloat16 && typeB == DType::kFloat16) return matmulFloat<Float16>(A, B, C);
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
