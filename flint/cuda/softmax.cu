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

#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>
#include <math.h>

#include <type_traits>

#include "flint/cuda/accessor.h"
#include "flint/cuda/common.h"
#include "flint/functional.h"

namespace fl {
namespace op {
namespace cuda {

struct MaxFloat {
  __device__ float operator()(float a, float b) const {
    return fmaxf(a, b);
  }
};

/// One block per row. VECTORIZED reads and writes a half2 at a time, which needs an even width,
/// operands aligned for it, and a half tensor: two elements to a machine word is a half's own
/// arrangement, so it is never instantiated for float.
template<typename T, int BLOCK_SIZE, bool VECTORIZED>
__global__ void softmaxFusedKernel(
    const T *__restrict__ input,
    T *__restrict__ output,
    int width) {
  int rowOffset = blockIdx.x * width;
  float threadMax = -INFINITY;

  if constexpr (VECTORIZED) {
    const half2 *input2 = reinterpret_cast<const half2 *>(input + rowOffset);
    int width2 = width / 2;
    for (int i = threadIdx.x; i < width2; i += BLOCK_SIZE) {
      float2 value = __half22float2(input2[i]);
      threadMax = fmaxf(threadMax, fmaxf(value.x, value.y));
    }
  } else {
    for (int i = threadIdx.x; i < width; i += BLOCK_SIZE) {
      threadMax = fmaxf(threadMax, static_cast<float>(input[rowOffset + i]));
    }
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ float rowMax;
  __shared__ float invSum;
  threadMax = BlockReduce(tempStorage).Reduce(threadMax, MaxFloat{});
  if (threadIdx.x == 0) rowMax = threadMax;
  __syncthreads();

  float threadSum = 0.0f;
  if constexpr (VECTORIZED) {
    const half2 *input2 = reinterpret_cast<const half2 *>(input + rowOffset);
    int width2 = width / 2;
    for (int i = threadIdx.x; i < width2; i += BLOCK_SIZE) {
      float2 value = __half22float2(input2[i]);
      threadSum += expf(value.x - rowMax) + expf(value.y - rowMax);
    }
  } else {
    for (int i = threadIdx.x; i < width; i += BLOCK_SIZE) {
      threadSum += expf(static_cast<float>(input[rowOffset + i]) - rowMax);
    }
  }

  threadSum = BlockReduce(tempStorage).Sum(threadSum);
  if (threadIdx.x == 0) invSum = 1.0f / threadSum;
  __syncthreads();

  if constexpr (VECTORIZED) {
    const half2 *input2 = reinterpret_cast<const half2 *>(input + rowOffset);
    half2 *output2 = reinterpret_cast<half2 *>(output + rowOffset);
    int width2 = width / 2;
    for (int i = threadIdx.x; i < width2; i += BLOCK_SIZE) {
      float2 value = __half22float2(input2[i]);
      output2[i] = __floats2half2_rn(
          expf(value.x - rowMax) * invSum,
          expf(value.y - rowMax) * invSum);
    }
  } else {
    for (int i = threadIdx.x; i < width; i += BLOCK_SIZE) {
      output[rowOffset + i] =
          static_cast<T>(expf(static_cast<float>(input[rowOffset + i]) - rowMax) * invSum);
    }
  }
}

/// One warp per row, for rows short enough to sit in registers. The block-per-row kernel above
/// gives a row 256 threads and two block-wide reductions no matter how short it is, which for the
/// 77-wide rows a text prompt produces leaves seven eighths of the threads idle and spends the
/// time in cub rather than on memory: 63 GB/s where the part can do 448. A warp needs no shared
/// memory and no __syncthreads, and reading the row once into registers means the two reductions
/// and the write cost nothing more to fetch.
///
/// PER_THREAD is how much of a row one lane keeps, so this handles widths up to 32 * PER_THREAD.
/// Rows wider than that stay with the block-per-row kernel, which is already at the memory limit
/// there and has nowhere better to go.
/// A warp is 32 lanes, which the full shuffle mask below takes as given.
constexpr int kWarpSize = 32;

template<typename T, int ROWS_PER_BLOCK, int PER_THREAD>
__global__ void softmaxWarpKernel(
    const T *__restrict__ input,
    T *__restrict__ output,
    int width,
    int rows) {
  // Every lane of a warp shares threadIdx.y, so a warp leaves here whole and the shuffles below
  // keep a full mask.
  int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.y;
  if (row >= rows) return;

  const T *rowInput = input + static_cast<int64_t>(row) * width;
  T *rowOutput = output + static_cast<int64_t>(row) * width;

  float held[PER_THREAD];
  float threadMax = -INFINITY;
#pragma unroll
  for (int j = 0; j < PER_THREAD; ++j) {
    int i = threadIdx.x + j * kWarpSize;
    held[j] = i < width ? static_cast<float>(rowInput[i]) : -INFINITY;
    threadMax = fmaxf(threadMax, held[j]);
  }
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    threadMax = fmaxf(threadMax, __shfl_xor_sync(0xffffffff, threadMax, offset));
  }

  float threadSum = 0.0f;
#pragma unroll
  for (int j = 0; j < PER_THREAD; ++j) {
    int i = threadIdx.x + j * kWarpSize;
    // The lanes past the end of the row take no part in the sum. Held values are left as they
    // are, so a row that is genuinely all -INFINITY comes out the same as it does above.
    held[j] = expf(held[j] - threadMax);
    if (i < width) threadSum += held[j];
  }
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    threadSum += __shfl_xor_sync(0xffffffff, threadSum, offset);
  }

  float invSum = 1.0f / threadSum;
#pragma unroll
  for (int j = 0; j < PER_THREAD; ++j) {
    int i = threadIdx.x + j * kWarpSize;
    if (i < width) rowOutput[i] = static_cast<T>(held[j] * invSum);
  }
}

template<typename T, int BLOCK_SIZE>
__global__ void softmaxStridedKernel(
    PackedTensorAccessor<const T, 3> input,
    PackedTensorAccessor<T, 3> output) {
  int width = input.getShape(2);
  int y = blockIdx.x % input.getShape(1);
  int z = blockIdx.x / input.getShape(1);

  float threadMax = -INFINITY;
  for (int x = threadIdx.x; x < width; x += BLOCK_SIZE) {
    threadMax = fmaxf(threadMax, static_cast<float>(input[z][y][x]));
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ float rowMax;
  __shared__ float invSum;
  threadMax = BlockReduce(tempStorage).Reduce(threadMax, MaxFloat{});
  if (threadIdx.x == 0) rowMax = threadMax;
  __syncthreads();

  float threadSum = 0.0f;
  for (int x = threadIdx.x; x < width; x += BLOCK_SIZE) {
    threadSum += expf(static_cast<float>(input[z][y][x]) - rowMax);
  }

  threadSum = BlockReduce(tempStorage).Sum(threadSum);
  if (threadIdx.x == 0) invSum = 1.0f / threadSum;
  __syncthreads();

  for (int x = threadIdx.x; x < width; x += BLOCK_SIZE) {
    output[z][y][x] = static_cast<T>(expf(static_cast<float>(input[z][y][x]) - rowMax) * invSum);
  }
}

template<typename T>
Tensor softmaxStrided3D(Tensor A) {
  CHECK(A.getDType() == DType::getType<T>());
  CHECK(A.getDim() == 3);

  Tensor C = createCudaTensor<T>(A.getShape());

  constexpr int blockSize = 256;
  int rows = A.getShape(0) * A.getShape(1);
  softmaxStridedKernel<T, blockSize><<<rows, blockSize>>>(A, C);
  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return C;
}

template<typename T>
Tensor softmax1D(Tensor A) {
  Tensor xA = A.view({1, 1, A.getShape(0)});
  Tensor C = softmaxStrided3D<T>(xA);

  return C.view({C.getShape(2)});
}

template<typename T>
Tensor softmax2D(Tensor A) {
  Tensor xA = A.view({1, A.getShape(0), A.getShape(1)});
  Tensor C = softmaxStrided3D<T>(xA);

  return C.view({C.getShape(1), C.getShape(2)});
}

template<typename T>
Tensor softmax4D(Tensor A) {
  std::vector<int> shape = A.getShape();

  Tensor xA = A.view({-1, A.getShape(2), A.getShape(3)});
  Tensor C = softmaxStrided3D<T>(xA);

  return C.view(shape);
}

template<typename T>
Tensor softmaxContiguous(Tensor A) {
  int width = A.getShape(-1);
  int64_t numel = A.getNumEl();
  CHECK(numel < std::numeric_limits<int>::max());
  int rows = static_cast<int>(numel / width);

  Tensor C = createCudaTensor<T>(A.getShape());
  const T *input = getDataPtrCuda<T>(A);
  T *output = getDataPtrCuda<T>(C);

  // A short row goes to a warp rather than a block. Attention hands softmax rows as short as the
  // prompt is -- 77 for SDXL -- and a block of 256 spends nine tenths of its time not fetching.
  constexpr int rowsPerBlock = 4;
  constexpr int perThread = 4;
  if (width <= kWarpSize * perThread) {
    dim3 block(kWarpSize, rowsPerBlock);
    int blocks = (rows + rowsPerBlock - 1) / rowsPerBlock;
    softmaxWarpKernel<T, rowsPerBlock, perThread><<<blocks, block>>>(input, output, width, rows);
    LL_CUDA_SYNCHRONIZE();
    LL_CHECK_CUDA_STATUS(cudaGetLastError());
    return C;
  }

  constexpr int blockSize = 256;
  bool useHalf2 = std::is_same<T, half>::value && width % 2 == 0 &&
                  reinterpret_cast<uintptr_t>(input) % alignof(half2) == 0 &&
                  reinterpret_cast<uintptr_t>(output) % alignof(half2) == 0;
  if (useHalf2) {
    softmaxFusedKernel<T, blockSize, std::is_same<T, half>::value>
        <<<rows, blockSize>>>(input, output, width);
  } else {
    softmaxFusedKernel<T, blockSize, false><<<rows, blockSize>>>(input, output, width);
  }

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return C;
}

template<typename T>
Tensor softmaxImpl(Tensor A) {
  if (A.isContiguous()) return softmaxContiguous<T>(A);
  if (A.getDim() == 1) return softmax1D<T>(A);
  if (A.getDim() == 2) return softmax2D<T>(A);
  if (A.getDim() == 3) return softmaxStrided3D<T>(A);
  if (A.getDim() == 4) return softmax4D<T>(A);

  NOT_IMPL();
}

Tensor softmax(Tensor A) {
  if (A.getDType() == DType::kFloat16) return softmaxImpl<half>(A);
  if (A.getDType() == DType::kFloat) return softmaxImpl<float>(A);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
