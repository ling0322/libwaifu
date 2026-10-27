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

#include "flint/cuda/common.h"

#include <cuda_fp16.h>

#include <atomic>

#include "flint/cuda/cuda_tensor_data.h"
#include "flint/tensor.h"
#include "lutil/log.h"

namespace fl {
namespace op {
namespace cuda {

int gMaxThreadsPerSM = 0;
int gNumSM = 0;

namespace {

// What llynCudaMalloc() has handed out and not had back, and the most it may come to; 0 for none.
// Atomic rather than locked: an allocation adds its bytes first and takes them away again if it
// is refused, so that two threads asking at once cannot both find room for one.
std::atomic<int64_t> gAllocatedBytes{0};
std::atomic<int64_t> gResidentBytes{0};
std::atomic<int64_t> gPeakAllocatedBytes{0};
std::atomic<int64_t> gMemoryLimit{0};

// What a limit of 0 means: the line the card puts on what may be handed out. Read only when the
// limit is set, never on the way to an allocation, so the hot path stays one atomic load.
std::atomic<int64_t> gDefaultMemoryLimit{0};

// Counts `bytes` as handed out, or throws where the limit would be passed and counts nothing.
// Returns what was handed out before, for a refusal to say.
int64_t take(int64_t bytes) {
  int64_t before = gAllocatedBytes.fetch_add(bytes);
  int64_t limit = gMemoryLimit.load();
  if (limit > 0 && before + bytes > limit) {
    gAllocatedBytes.fetch_sub(bytes);
    throw lut::OutOfMemoryError(lut::sprintf(
        "the CUDA memory limit is %d bytes, %d of them in use, and %d more were asked for",
        limit,
        before,
        bytes));
  }
  int64_t now = before + bytes;
  int64_t peak = gPeakAllocatedBytes.load();
  while (now > peak && !gPeakAllocatedBytes.compare_exchange_weak(peak, now)) {
  }
  return before;
}

// Takes back what take() counted, where the driver did not give it, and throws what the driver
// said: out of memory as itself, anything else as an abort.
[[noreturn]] void refused(cudaError_t err, int64_t bytes, int64_t before) {
  gAllocatedBytes.fetch_sub(bytes);
  // The error is the call's, and returned by it, but it is also left as the thread's last one,
  // where the next kernel launch that asks would find it and report it as its own.
  cudaGetLastError();
  if (err == cudaErrorMemoryAllocation) {
    throw lut::OutOfMemoryError(lut::sprintf(
        "the CUDA device could not allocate %d bytes (%d held by flint)",
        bytes,
        before));
  }
  throw lut::AbortedError(cudaGetErrorString(err));
}

}  // namespace

void *llynCudaMalloc(size_t size, cudaStream_t stream) {
  int64_t bytes = static_cast<int64_t>(size);
  int64_t before = take(bytes);

  void *ptr = nullptr;
#ifdef LIBWAIFU_CUDA_MALLOC_ASYNC_ENABLED
  cudaError_t err = cudaMallocAsync(&ptr, size, stream);
#else
  cudaError_t err = cudaMalloc(&ptr, size);
#endif
  if (err != cudaSuccess) refused(err, bytes, before);
  return ptr;
}

void llynCudaFree(void *ptr, size_t size, cudaStream_t stream) {
#ifdef LIBWAIFU_CUDA_MALLOC_ASYNC_ENABLED
  cudaError_t err = cudaFreeAsync(ptr, stream);
#else
  cudaError_t err = cudaFree(ptr);
#endif
  gAllocatedBytes.fetch_sub(static_cast<int64_t>(size));
  if (err != cudaSuccess) {
    LOG(ERROR) << "Error while freeing CUDA memory: " << cudaGetErrorString(err);
  }
}

void *llynCudaMallocResident(size_t size) {
  int64_t bytes = static_cast<int64_t>(size);
  int64_t before = take(bytes);

  void *ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  if (err != cudaSuccess) refused(err, bytes, before);
  gResidentBytes.fetch_add(bytes);
  return ptr;
}

void llynCudaFreeResident(void *ptr, size_t size) {
  cudaError_t err = cudaFree(ptr);
  gAllocatedBytes.fetch_sub(static_cast<int64_t>(size));
  gResidentBytes.fetch_sub(static_cast<int64_t>(size));
  if (err != cudaSuccess) {
    LOG(ERROR) << "Error while freeing CUDA memory: " << cudaGetErrorString(err);
  }
}

void setCudaMemoryLimit(int64_t bytes) {
  gMemoryLimit.store(bytes > 0 ? bytes : gDefaultMemoryLimit.load());
}

void setCudaDefaultMemoryLimit(int64_t bytes) {
  gDefaultMemoryLimit.store(bytes);
  // Nobody has asked for one of their own yet, so this is the one in force.
  int64_t none = 0;
  gMemoryLimit.compare_exchange_strong(none, bytes);
}

int64_t getCudaMemoryLimit() {
  return gMemoryLimit.load();
}

int64_t getCudaAllocatedBytes() {
  return gAllocatedBytes.load();
}

int64_t getCudaResidentBytes() {
  return gResidentBytes.load();
}

int64_t getCudaPeakAllocatedBytes() {
  return gPeakAllocatedBytes.load();
}

void resetCudaPeakAllocatedBytes() {
  gPeakAllocatedBytes.store(gAllocatedBytes.load());
}

Tensor createCudaTensor(lut::Span<const int> shape, DType dtype) {
  auto tensorShape = std::make_shared<TensorShape>(shape);
  auto data = CudaTensorData::create(tensorShape->getNumEl(), dtype);

  return Tensor::create(tensorShape, std::move(data));
}

template<typename T>
float elemImpl(const TensorView &tensor) {
  T v;
  LL_CHECK_CUDA_STATUS(
      cudaMemcpy(&v, getDataPtrCuda<T>(tensor), sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}
float elem(const TensorView &tensor) {
  CHECK(tensor.getDim() == 1 && tensor.getShape(0) == 1);

  if (tensor.getDType() == DType::kFloat16) return elemImpl<half>(tensor);
  if (tensor.getDType() == DType::kFloat) return elemImpl<float>(tensor);

  NOT_IMPL();
}

bool elemBool(const TensorView &tensor) {
  CHECK(tensor.getDim() == 1 && tensor.getShape(0) == 1);
  CHECK(tensor.getDType() == DType::kBool);

  bool v;
  LL_CHECK_CUDA_STATUS(
      cudaMemcpy(&v, getDataPtrCuda<BoolType>(tensor), sizeof(BoolType), cudaMemcpyDeviceToHost));

  return v;
}

int getCudaDeviceAttribute(cudaDeviceAttr attr) {
  int value;
  LL_CHECK_CUDA_STATUS(cudaDeviceGetAttribute(&value, attr, 0));
  return value;
}

int getCudaDeviceCount() {
  int value;

  cudaError_t status = cudaGetDeviceCount(&value);
  if (status != cudaSuccess) {
    return 0;
  }

  return value;
}

dim3 getGrid1D(int numel, int blockSize) {
  if (!gMaxThreadsPerSM) {
    gMaxThreadsPerSM = getCudaDeviceAttribute(cudaDevAttrMaxThreadsPerMultiProcessor);
  }
  if (!gNumSM) {
    gNumSM = getCudaDeviceAttribute(cudaDevAttrMultiProcessorCount);
  }

  int maxThreadsPerSM = gMaxThreadsPerSM;
  int numSM = gNumSM;
  int numBlock = (numel + blockSize - 1) / blockSize;
  int deviceNumBlock = maxThreadsPerSM * numSM / blockSize;
  CHECK(deviceNumBlock > 0);

  dim3 grid(std::min(numBlock, deviceNumBlock));

  return grid;
}

int getCudaArch() {
  // TODO: too slow
  cudaDeviceProp prop;
  LL_CHECK_CUDA_STATUS(cudaGetDeviceProperties(&prop, 0));
  return prop.major * 10 + prop.minor;
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
