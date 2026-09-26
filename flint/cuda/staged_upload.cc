// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
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

#include "flint/cuda/staged_upload.h"

#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "flint/cuda/common.h"

namespace fl {
namespace op {
namespace cuda {

namespace {

/// memcpy, split evenly between up to `threads` threads.
void parallelMemcpy(std::byte *dest, const std::byte *src, int64_t n, int threads) {
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
  {
    int64_t numThreads = omp_get_num_threads();
    int64_t per = (n + numThreads - 1) / numThreads;
    int64_t begin = std::min(n, per * omp_get_thread_num());
    int64_t end = std::min(n, begin + per);
    if (begin < end) std::memcpy(dest + begin, src + begin, end - begin);
  }
#else
  (void)threads;
  std::memcpy(dest, src, n);
#endif
}

}  // namespace

StagedUpload::StagedUpload()
    : _buffers{nullptr, nullptr},
      _sent{nullptr, nullptr},
      _next(0) {
  for (int i = 0; i < 2; ++i) {
    LL_CHECK_CUDA_STATUS(
        cudaHostAlloc(reinterpret_cast<void **>(&_buffers[i]), ChunkBytes, cudaHostAllocDefault));
    LL_CHECK_CUDA_STATUS(cudaEventCreateWithFlags(&_sent[i], cudaEventDisableTiming));
  }
}

StagedUpload *StagedUpload::getInstance() {
  static StagedUpload *instance = new StagedUpload();
  return instance;
}

void StagedUpload::copy(void *dest, const void *src, int64_t n, cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(_mutex);

  auto *to = static_cast<std::byte *>(dest);
  auto *from = static_cast<const std::byte *>(src);
  for (int64_t offset = 0; offset < n; offset += ChunkBytes) {
    int64_t size = std::min(ChunkBytes, n - offset);
    int k = _next;
    _next ^= 1;

    // The copy engine has to be done reading this buffer before it is written again. An event
    // that was never recorded counts as reached, which is the state a fresh buffer is in.
    LL_CHECK_CUDA_STATUS(cudaEventSynchronize(_sent[k]));
    parallelMemcpy(_buffers[k], from + offset, size, NumThreads);

    LL_CHECK_CUDA_STATUS(
        cudaMemcpyAsync(to + offset, _buffers[k], size, cudaMemcpyHostToDevice, stream));
    LL_CHECK_CUDA_STATUS(cudaEventRecord(_sent[k], stream));
  }
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
