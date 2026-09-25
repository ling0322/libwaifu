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

// Inclusive prefix sum along the last dimension.
//
// One block per row. The block walks its row a tile of BLOCK_DIM elements at a time, each thread
// holding one element, and scans each tile with cub::BlockScan. What carries one tile's total into
// the next is BlockScan's prefix callback: thread 0 is handed the tile's aggregate, returns the
// running total so far as the tile's prefix, and adds the aggregate to it. So any row length works
// with one launch, and the sum is in float32 whatever the storage type.
//
// This is the simple layout, and it is the right one for what flint scans: a few hundred rows of a
// few thousand elements at most (a vocoder's phase over one sentence is nine rows of a few hundred).
// A row far longer than there are rows to spread across the card would want a device-wide scan
// instead -- cub::DeviceScan per row, or a decoupled look-back -- and that is not written here.

#include <cuda_fp16.h>

#include <limits>

#include <cub/cub.cuh>

#include "flint/cuda/common.h"
#include "flint/cuda/scan.h"
#include "flint/operators.h"

namespace fl {
namespace op {
namespace cuda {

namespace {

/// The running total of a row, handed to each tile as its prefix.
struct RunningTotal {
  float total;

  __device__ explicit RunningTotal(float start)
      : total(start) {
  }

  __device__ float operator()(float aggregate) {
    float prefix = total;
    total += aggregate;
    return prefix;
  }
};

template<typename T, int BLOCK_DIM>
__global__ void cumsumRowsKernel(const T *__restrict__ in, T *__restrict__ out, int64_t length) {
  using BlockScan = cub::BlockScan<float, BLOCK_DIM>;
  __shared__ typename BlockScan::TempStorage storage;

  const T *row = in + blockIdx.x * length;
  T *outRow = out + blockIdx.x * length;

  RunningTotal carry(0.0f);
  for (int64_t start = 0; start < length; start += BLOCK_DIM) {
    int64_t index = start + threadIdx.x;
    float value = index < length ? static_cast<float>(row[index]) : 0.0f;

    float scanned;
    BlockScan(storage).InclusiveSum(value, scanned, carry);
    // The storage is reused by the next tile's scan; every thread has to be done reading it.
    __syncthreads();

    if (index < length) outRow[index] = static_cast<T>(scanned);
  }
}

template<typename T>
Tensor cumsumImpl(const Tensor &A) {
  CHECK(A.isContiguous());
  CHECK(A.getDim() >= 1);

  Tensor C = createCudaTensor<T>(A.getShape());
  int64_t length = A.getShape(-1);
  if (length == 0 || A.getNumEl() == 0) return C;

  int64_t rows = A.getNumEl() / length;
  CHECK(rows <= std::numeric_limits<int>::max());

  constexpr int blockDim = 256;
  cumsumRowsKernel<T, blockDim><<<static_cast<unsigned>(rows), blockDim>>>(
      getDataPtrCuda<T>(A),
      getDataPtrCuda<T>(C),
      length);

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return C;
}

}  // namespace

Tensor cumsumLastDim(const Tensor &A) {
  if (A.getDType() == DType::kFloat) return cumsumImpl<float>(A);
  if (A.getDType() == DType::kFloat16) return cumsumImpl<half>(A);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
