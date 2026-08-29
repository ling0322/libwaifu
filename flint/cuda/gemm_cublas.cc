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

#include "flint/cuda/gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/shared_library.h"
#include "lutil/strings.h"

#define LL_CHECK_CUBLAS_STATUS(x)                                                          \
  {                                                                                        \
    cublasStatus_t status = x;                                                             \
    if (status != CUBLAS_STATUS_SUCCESS) {                                                 \
      LOG(ERROR) << "Error while calling: " << #x << ": " << cublasGetErrorString(status); \
      throw lut::AbortedError(cublasGetErrorString(status));                               \
    }                                                                                      \
  }

extern "C" {

typedef CUBLASAPI cublasStatus_t CUBLASWINAPI (*cublasGemmBatchedExFunc_t)(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const void *alpha,
    const void *const Aarray[],
    cudaDataType Atype,
    int lda,
    const void *const Barray[],
    cudaDataType Btype,
    int ldb,
    const void *beta,
    void *const Carray[],
    cudaDataType Ctype,
    int ldc,
    int batchCount,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo);

typedef CUBLASAPI cublasStatus_t CUBLASWINAPI (*cublasGetPropertyFunc_t)(
    libraryPropertyType type,
    int *value);

typedef CUBLASAPI cublasStatus_t CUBLASWINAPI (*cublasGemmExFunc_t)(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const void *alpha,
    const void *A,
    cudaDataType Atype,
    int lda,
    const void *B,
    cudaDataType Btype,
    int ldb,
    const void *beta,
    void *C,
    cudaDataType Ctype,
    int ldc,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo);
}

typedef CUBLASAPI cublasStatus_t CUBLASWINAPI (*cublasCreateFunc_t)(cublasHandle_t *handle);
typedef CUBLASAPI cublasStatus_t CUBLASWINAPI (*cublasDestroyFunc_t)(cublasHandle_t handle);

namespace fl {
namespace op {
namespace cuda {

const char *cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    default:
      return "Unknown cuBLAS error";
  }
}

/// The oldest cuBLAS this may call, which is the one CUDA 11 shipped.
///
/// Not caution. `cublasGemmEx` took a `cudaDataType` for its compute type until CUDA 11, where it
/// became a `cublasComputeType_t` -- the same symbol, a different meaning for the same argument.
/// An older library loads and resolves every name this asks for, and then gets handed
/// CUBLAS_COMPUTE_32F where it expects a data type. cuBLAS's major version follows CUDA's, so
/// this is the version test for it.
constexpr int kMinimumMajorVersion = 11;

class CublasGemm::Impl {
 public:
  cublasGemmBatchedExFunc_t _cublasGemmBatchedEx;
  cublasGemmExFunc_t _cublasGemmEx;
  cublasCreateFunc_t _cublasCreate;
  cublasDestroyFunc_t _cublasDestroy;
  cublasHandle_t _handle;

  static std::unique_ptr<Impl> create() {
    std::unique_ptr<Impl> impl = std::make_unique<Impl>();

    impl->_libCublas = lut::SharedLibrary::open("cublas");

    // Asked before anything is called, so that a library too old to mean what this means is
    // refused rather than driven with arguments it will read as something else.
    cublasGetPropertyFunc_t getProperty = impl->_libCublas->getFunc<cublasGetPropertyFunc_t>(
        "cublasGetProperty");
    int major = 0;
    int minor = 0;
    LL_CHECK_CUBLAS_STATUS(getProperty(MAJOR_VERSION, &major));
    LL_CHECK_CUBLAS_STATUS(getProperty(MINOR_VERSION, &minor));
    if (major < kMinimumMajorVersion) {
      throw lut::AbortedError(lut::sprintf(
          "cuBLAS is %d.%d, and %d.0 is the oldest that reads cublasGemmEx's compute type the "
          "way this calls it",
          major,
          minor,
          kMinimumMajorVersion));
    }
    LOG(INFO) << "cuBLAS " << major << "." << minor;

    impl->_cublasGemmBatchedEx = impl->_libCublas->getFunc<cublasGemmBatchedExFunc_t>(
        "cublasGemmBatchedEx");
    impl->_cublasGemmEx = impl->_libCublas->getFunc<cublasGemmExFunc_t>("cublasGemmEx");
    impl->_cublasCreate = impl->_libCublas->getFunc<cublasCreateFunc_t>("cublasCreate_v2");
    impl->_cublasDestroy = impl->_libCublas->getFunc<cublasDestroyFunc_t>("cublasDestroy_v2");

    LL_CHECK_CUBLAS_STATUS(impl->_cublasCreate(&impl->_handle));
    return impl;
  }

  Impl()
      : _handle(nullptr) {
  }

  ~Impl() {
    if (_cublasDestroy && _handle) {
      cublasStatus_t status = _cublasDestroy(_handle);
      if (status != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "Error while calling cublasDestroy(): " << cublasGetErrorString(status);
      }

      _handle = nullptr;
    }
  }

 private:
  std::unique_ptr<lut::SharedLibrary> _libCublas;
};

std::shared_ptr<Gemm> CublasGemm::create() {
  std::shared_ptr<CublasGemm> mm{new CublasGemm()};
  mm->_impl = Impl::create();

  return mm;
}

void CublasGemm::hgemm(
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
  float alphaFp32 = static_cast<float>(alpha);
  float betaFp32 = static_cast<float>(beta);

  LL_CHECK_CUBLAS_STATUS(_impl->_cublasGemmEx(
      _impl->_handle,
      transB ? CUBLAS_OP_T : CUBLAS_OP_N,
      transA ? CUBLAS_OP_T : CUBLAS_OP_N,
      n,
      m,
      k,
      &alphaFp32,
      B,
      CUDA_R_16F,
      ldb,
      A,
      CUDA_R_16F,
      lda,
      &betaFp32,
      C,
      CUDA_R_16F,
      ldc,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT));
}

/// The same call with both operands and the result in float. The compute type stays
/// CUBLAS_COMPUTE_32F rather than its TF32 variant: a float GEMM is asked for where half has run
/// out of range, and TF32 carries fewer mantissa bits than half does.
void CublasGemm::sgemm(
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
  LL_CHECK_CUBLAS_STATUS(_impl->_cublasGemmEx(
      _impl->_handle,
      transB ? CUBLAS_OP_T : CUBLAS_OP_N,
      transA ? CUBLAS_OP_T : CUBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      B,
      CUDA_R_32F,
      ldb,
      A,
      CUDA_R_32F,
      lda,
      &beta,
      C,
      CUDA_R_32F,
      ldc,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT));
}

void CublasGemm::hgemmArray(
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
  float alphaFp32 = static_cast<float>(alpha);
  float betaFp32 = static_cast<float>(beta);

  LL_CHECK_CUBLAS_STATUS(_impl->_cublasGemmBatchedEx(
      _impl->_handle,
      transB ? CUBLAS_OP_T : CUBLAS_OP_N,
      transA ? CUBLAS_OP_T : CUBLAS_OP_N,
      n,
      m,
      k,
      &alphaFp32,
      reinterpret_cast<const void *const *>(arrayB),
      CUDA_R_16F,
      ldb,
      reinterpret_cast<const void *const *>(arrayA),
      CUDA_R_16F,
      lda,
      &betaFp32,
      reinterpret_cast<void *const *>(arrayC),
      CUDA_R_16F,
      ldc,
      batchSize,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT));
}

void CublasGemm::sgemmArray(
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
  LL_CHECK_CUBLAS_STATUS(_impl->_cublasGemmBatchedEx(
      _impl->_handle,
      transB ? CUBLAS_OP_T : CUBLAS_OP_N,
      transA ? CUBLAS_OP_T : CUBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      reinterpret_cast<const void *const *>(arrayB),
      CUDA_R_32F,
      ldb,
      reinterpret_cast<const void *const *>(arrayA),
      CUDA_R_32F,
      lda,
      &beta,
      reinterpret_cast<void *const *>(arrayC),
      CUDA_R_32F,
      ldc,
      batchSize,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT));
}

}  // namespace cuda
}  // namespace op
}  // namespace fl