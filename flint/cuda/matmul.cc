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

#include "flint/cuda/matmul.h"

#include <cuda_fp16.hpp>

#include <type_traits>

#include "lutil/strings.h"
#include "flint/cpu/common.h"
#include "flint/cpu/matmul.h"
#include "flint/cuda/common.h"
#include "flint/cuda/dequant.h"
#include "flint/cuda/fill.h"
#include "flint/cuda/gemm_cublas.h"
#include "flint/cuda/gemm_cutlass.h"
#include "flint/cuda/matvec.h"
#include "flint/dtype.h"

namespace fl {
namespace op {
namespace cuda {

std::shared_ptr<MatMul> MatMul::create() {
  std::shared_ptr<MatMul> mm;
  std::string err0, err1;

  try {
    mm = createCublas();
    LOG(INFO) << "Use GEMM from cuBLAS.";
    return mm;
  } catch (const lut::Error &e) {
    LOG(DEBUG) << "Load cublas extension failed with message: " << e.what();
    err0 = e.what();
  }

  try {
    mm = createCutlass();
    LOG(INFO) << "Use GEMM from cutlass.";
    return mm;
  } catch (const lut::Error &e) {
    LOG(DEBUG) << "Load cublas extension failed with message: " << e.what();
    err1 = e.what();
  }

  throw lut::AbortedError("unable to create MatMul operator: " + err0 + "; " + err1);
}

std::shared_ptr<MatMul> MatMul::createCublas() {
  std::shared_ptr<MatMul> mm{new MatMul()};
  mm->_gemm = CublasGemm::create();

  return mm;
}

std::shared_ptr<MatMul> MatMul::createCutlass() {
  std::shared_ptr<MatMul> mm{new MatMul()};

#ifdef LIBWAIFU_CUTLASS_ENABLED
  mm->_gemm = CutlassGemm::create();
#else
  throw lut::AbortedError("Cutlass MatMul operator not enabled.");
#endif

  return mm;
}

Tensor MatMul::apply(const Tensor &A, const Tensor &B) {
  CHECK(A.getDevice().getType() == Device::kCuda);
  CHECK(B.getDevice().getType() == Device::kCuda);

  if (A.getDType() == DType::kFloat16 && B.getDType() == DType::kFloat16) {
    return matmulFloat<half>(A, B);
  } else if (A.getDType() == DType::kFloat && B.getDType() == DType::kFloat) {
    return matmulFloat<float>(A, B);
  } else {
    NOT_IMPL();
  }
}

Tensor MatMul::applyNarrowPrecision(
    const Tensor &A,
    const Tensor &sfA,
    const Tensor &B,
    const Tensor &sfB) {
  CHECK(A.getDevice().getType() == Device::kCuda);
  CHECK(B.getDevice().getType() == Device::kCuda);
  CHECK(sfA.getDevice().getType() == Device::kCuda);
  CHECK(sfB.getDevice().getType() == Device::kCuda);

  if (A.getDType() == DType::kFp4E2M0x2 && B.getDType() == DType::kFp4E2M0x2 &&
      sfA.getDType() == DType::kUInt8 && sfB.getDType() == DType::kUInt8) {
    return matmulMxfp4(A, sfA, B, sfB);
  }

  NOT_IMPL();
}

Tensor MatMul::matmulMxfp4(const Tensor &A, const Tensor &sfA, const Tensor &B, const Tensor &sfB) {
  CHECK(A.getDim() == B.getDim() && A.getDim() == 2);
  Tensor C = createCudaTensorHalf({A.getShape(0), B.getShape(1) * 2});
  fill(C, 0.0f);

  int m = A.getShape(0);
  int k = A.getShape(1) * 2;
  int n = B.getShape(1) * 2;
  CHECK(k == B.getShape(1) * 2);

  float alpha = 1.0;

  _gemm->gemmMxfp4Bf16(
      m,
      n,
      k,
      alpha,
      getDataPtrCuda<Fp4E2M0x2>(A),
      getDataPtrCuda<UInt8>(sfA),
      getDataPtrCuda<Fp4E2M0x2>(B),
      getDataPtrCuda<UInt8>(sfB),
      getDataPtrCuda<Float16>(C));

  LL_CUDA_SYNCHRONIZE();

  return C;
}

/// A GEMM call in `T`. The interface names its two arms after the BLAS letters rather than
/// taking a type, so this is where the letter is chosen.
template<typename T>
void callGemm(
    Gemm *gemm,
    bool transA,
    bool transB,
    int m,
    int n,
    int k,
    float alpha,
    const T *A,
    int lda,
    const T *B,
    int ldb,
    float beta,
    T *C,
    int ldc);

template<>
void callGemm<half>(
    Gemm *gemm,
    bool transA,
    bool transB,
    int m,
    int n,
    int k,
    float alpha,
    const half *A,
    int lda,
    const half *B,
    int ldb,
    float beta,
    half *C,
    int ldc) {
  gemm->hgemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
void callGemm<float>(
    Gemm *gemm,
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
  gemm->sgemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<typename T>
void callGemmArray(
    Gemm *gemm,
    bool transA,
    bool transB,
    int m,
    int n,
    int k,
    float alpha,
    const T *const *arrayA,
    int lda,
    const T *const *arrayB,
    int ldb,
    float beta,
    T *const *arrayC,
    int ldc,
    int batchSize);

template<>
void callGemmArray<half>(
    Gemm *gemm,
    bool transA,
    bool transB,
    int m,
    int n,
    int k,
    float alpha,
    const half *const *arrayA,
    int lda,
    const half *const *arrayB,
    int ldb,
    float beta,
    half *const *arrayC,
    int ldc,
    int batchSize) {
  gemm->hgemmArray(
      transA, transB, m, n, k, alpha, arrayA, lda, arrayB, ldb, beta, arrayC, ldc, batchSize);
}

template<>
void callGemmArray<float>(
    Gemm *gemm,
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
  gemm->sgemmArray(
      transA, transB, m, n, k, alpha, arrayA, lda, arrayB, ldb, beta, arrayC, ldc, batchSize);
}

template<typename T>
Tensor MatMul::matmulFloat(const Tensor &A, const Tensor &B) {
  CHECK(A.getDType() == B.getDType() && A.getDType() == DType::getType<T>());
  if (A.getDim() == 2 && B.getDim() == 2) {
    // A single row against a transposed weight is the decode-step shape, and the vector kernel
    // is faster for it. Every condition gemvHalf asserts has to be checked here, not just the
    // transposed layout: a contiguous B with one column also has getStride(0) == 1, and the
    // kernel loads eight halves at a time. Anything it cannot take falls through to the GEMM.
    // There is no float kernel behind it, so a float matmul is always the GEMM.
    if constexpr (std::is_same<T, half>::value) {
      if (A.getShape(0) == 1 && A.getStride(1) == 1 && B.getStride(0) == 1 &&
          B.getStride(1) == B.getShape(0) && B.getShape(0) % 8 == 0) {
        return gemvHalf(A.subtensor(0), B);
      }
    }
    return gemm<T>(A, B);
  } else if (A.getDim() > 2 && B.getDim() == 2 && A.isContiguous()) {
    return bmmToGemm<T>(A, B);
  } else if (A.getDim() >= 2 && B.getDim() >= 2 && A.getDim() >= B.getDim()) {
    return bmm<T>(A, B);
  } else {
    NOT_IMPL();
  }
}

template<typename T>
std::vector<const T *> getBatchImpl1(const Tensor &A) {
  const T *base = getDataPtrCuda<T>(A);

  int stride0 = A.getStride(0);
  std::vector<const T *> batch;
  for (int i = 0; i < A.getShape(0); ++i) {
    batch.push_back(base + i * stride0);
  }
  return batch;
}

template<typename T>
std::vector<const T *> getBatchImpl2(const Tensor &A) {
  const T *base = getDataPtrCuda<T>(A);

  int stride0 = A.getStride(0);
  int stride1 = A.getStride(1);
  std::vector<const T *> batch;
  for (int i = 0; i < A.getShape(0); ++i) {
    for (int j = 0; j < A.getShape(1); ++j) {
      batch.push_back(base + i * stride0 + j * stride1);
    }
  }
  return batch;
}

template<typename T>
std::vector<const T *> MatMul::getBatch(const Tensor &A, int nBatchDim) {
  if (nBatchDim == 1) return getBatchImpl1<T>(A);
  if (nBatchDim == 2) return getBatchImpl2<T>(A);

  NOT_IMPL();
}

template<typename T>
Tensor MatMul::bmmToGemm(const Tensor &A, const Tensor &B) {
  std::vector<int> shape = A.getShape();

  Tensor xA = A.view({-1, A.getShape(-1)});
  Tensor xC = gemm<T>(xA, B);

  shape.back() = B.getShape(1);
  return xC.view(shape);
}

template<typename T>
Tensor MatMul::bmm(Tensor A, Tensor B) {
  Tensor xB = B;
  if (A.getDim() != B.getDim()) xB = op::cpu::expandBatchDims(B, A.getShape());

  std::vector<int> shapeC = op::cpu::getBmmOutputShape(A, xB);
  Tensor C = createCudaTensor<T>(shapeC);

  int nBatchDim = A.getDim() - 2;

  op::cpu::GEMMArgs gemmArgs = op::cpu::generateGemmArgs(A, xB, C);
  std::vector<const T *> batchA = getBatch<T>(A, nBatchDim);
  std::vector<const T *> batchB = getBatch<T>(xB, nBatchDim);
  std::vector<const T *> batchC = getBatch<T>(C, nBatchDim);
  CHECK(batchA.size() == batchB.size() && batchA.size() == batchC.size());

  int64_t nb = batchA.size();
  lut::c_ptr<const T *> arrayA = llynCudaAlloc<const T *>(nb);
  lut::c_ptr<const T *> arrayB = llynCudaAlloc<const T *>(nb);
  lut::c_ptr<T *> arrayC = llynCudaAlloc<T *>(nb);

  int64_t nc = sizeof(void *) * nb;
  LL_CHECK_CUDA_STATUS(cudaMemcpy(arrayA.get(), batchA.data(), nc, cudaMemcpyHostToDevice));
  LL_CHECK_CUDA_STATUS(cudaMemcpy(arrayB.get(), batchB.data(), nc, cudaMemcpyHostToDevice));
  LL_CHECK_CUDA_STATUS(cudaMemcpy(arrayC.get(), batchC.data(), nc, cudaMemcpyHostToDevice));

  callGemmArray<T>(
      _gemm.get(),
      gemmArgs.transA,
      gemmArgs.transB,
      gemmArgs.M,
      gemmArgs.N,
      gemmArgs.K,
      1.0f,
      arrayA.get(),
      gemmArgs.lda,
      arrayB.get(),
      gemmArgs.ldb,
      0.0f,
      arrayC.get(),
      gemmArgs.ldc,
      nb);

  LL_CUDA_SYNCHRONIZE();
  return C;
}

template<typename T>
Tensor MatMul::gemm(Tensor A, Tensor B) {
  CHECK(A.getDim() == B.getDim() && A.getDim() == 2);
  Tensor C = createCudaTensor<T>({A.getShape(0), B.getShape(1)});

  op::cpu::GEMMArgs gemmArgs = op::cpu::generateGemmArgs(A, B, C);
  callGemm<T>(
      _gemm.get(),
      gemmArgs.transA,
      gemmArgs.transB,
      gemmArgs.M,
      gemmArgs.N,
      gemmArgs.K,
      1.0f,
      getDataPtrCuda<T>(A),
      gemmArgs.lda,
      getDataPtrCuda<T>(B),
      gemmArgs.ldb,
      0.0f,
      getDataPtrCuda<T>(C),
      gemmArgs.ldc);
  LL_CUDA_SYNCHRONIZE();

  return C;
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
