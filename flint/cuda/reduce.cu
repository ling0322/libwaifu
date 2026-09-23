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

#include <assert.h>
#include <cuda_fp16.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/cub.cuh>
#include <cuda/std/cmath>
#include <cuda/std/limits>

#include "flint/cuda/accessor.h"
#include "flint/cuda/common.h"
#include "flint/cuda/reduce.h"
#include "flint/operators.h"

namespace fl {
namespace op {
namespace cuda {

template<MapReduceType MR_TYPE, typename TIn, typename TOut>
struct MapOp {
  __device__ TOut operator()(const TIn &x) const {
    if constexpr (
        MR_TYPE == MapReduceType::SUM || MR_TYPE == MapReduceType::MAX ||
        MR_TYPE == MapReduceType::MIN || MR_TYPE == MapReduceType::ALL) {
      return TOut(x);
    } else if constexpr (MR_TYPE == MapReduceType::SUM_EXP) {
      return expf(TOut(x));
    } else if constexpr (MR_TYPE == MapReduceType::SUM_SQUARE) {
      return TOut(x) * TOut(x);
    } else {
      __trap();
    }
  }
};

template<MapReduceType MR_TYPE, typename T>
struct ReduceOp {
  __device__ T operator()(const T &a, const T &b) const {
    if constexpr (
        MR_TYPE == MapReduceType::SUM || MR_TYPE == MapReduceType::SUM_EXP ||
        MR_TYPE == MapReduceType::SUM_SQUARE) {
      return a + b;
    } else if constexpr (MR_TYPE == MapReduceType::ALL) {
      return a && b;
    } else if constexpr (MR_TYPE == MapReduceType::MAX) {
      return std::max(a, b);
    } else if constexpr (MR_TYPE == MapReduceType::MIN) {
      return std::min(a, b);
    } else {
      __trap();
    }
  }
};

template<typename T, MapReduceType MR_TYPE>
__device__ __host__ T getReduceInitial() {
  if constexpr (
      MR_TYPE == MapReduceType::SUM || MR_TYPE == MapReduceType::SUM_EXP ||
      MR_TYPE == MapReduceType::SUM_SQUARE) {
    return T(0);
  } else if constexpr (MR_TYPE == MapReduceType::MAX) {
    return -::cuda::std::numeric_limits<float>::infinity();
  } else if constexpr (MR_TYPE == MapReduceType::MIN) {
    return ::cuda::std::numeric_limits<float>::infinity();
  } else if constexpr (MR_TYPE == MapReduceType::ALL) {
    return true;
  } else {
#ifdef __CUDA_ARCH__
    __trap();
#else
    NOT_IMPL();
#endif
  }
}

// One block per row, reducing the row's elements into one.
//
// The row is the block's x index. It used to be two indices, y and z, over a 3D view of the input
// -- which held at most 65 535 rows on each axis and refused any input of rank four or more
// outright. Every input is now read as `(rows, last)`, and x holds 2^31 - 1 blocks.
template<typename TIn, typename TOut, MapReduceType REDUCE_TYPE, int BLOCK_DIM>
__global__ void reduceRowsKernel(
    PackedTensorAccessor<const TIn, 2> input,
    PackedTensorAccessor<TOut, 1> output) {
  assert(blockDim.x == BLOCK_DIM);
  MapOp<REDUCE_TYPE, TIn, TOut> mapOp;
  ReduceOp<REDUCE_TYPE, TOut> reduceOp;

  int64_t row = blockIdx.x;

  TOut elemReduce = getReduceInitial<TOut, REDUCE_TYPE>();
  for (int offset = 0; offset < input.getShape(1); offset += BLOCK_DIM) {
    int idx = offset + threadIdx.x;
    if (idx < input.getShape(1)) {
      // element-wise map.
      TIn elemIn = input[row][idx];
      TOut elemMap = mapOp(elemIn);
      elemReduce = reduceOp(elemReduce, elemMap);
    }
  }

  using BlockReduce = cub::BlockReduce<TOut, BLOCK_DIM>;
  using TempStorage = typename BlockReduce::TempStorage;

  __shared__ TempStorage tempStorage;
  BlockReduce blockReduce{tempStorage};
  elemReduce = blockReduce.Reduce(elemReduce, reduceOp);

  if (threadIdx.x == 0) output[row] = elemReduce;
}

template<MapReduceType MR_TYPE, typename TIn, typename TOut>
Tensor reduceAllImpl(const Tensor &A) {
  CHECK(A.isContiguous());
  CHECK(A.getDType() == DType::getType<TIn>());

  int64_t numel = A.getNumEl();

  Tensor C = createCudaTensor<TOut>({1});
  TOut *result = getDataPtrCuda<TOut>(C);

  thrust::device_ptr<const TIn> pdata = thrust::device_pointer_cast(getDataPtrCuda<TIn>(A));
  thrust::device_vector<TIn> data(pdata, pdata + numel);
  auto iter = thrust::make_transform_iterator(data.begin(), MapOp<MR_TYPE, TIn, TOut>{});

  void *tempStorage = nullptr;
  size_t tempStorageBytes = 0;

  // Step 1: Query temp storage size
  cub::DeviceReduce::Reduce(
      tempStorage,
      tempStorageBytes,
      getDataPtrCuda<TIn>(A),
      result,
      numel,
      ReduceOp<MR_TYPE, TOut>{},
      0.0);

  lut::c_ptr<std::byte> tempStoragePtr = llynCudaAlloc<std::byte>(tempStorageBytes);
  tempStorage = tempStoragePtr.get();

  // Step 2: Run the reduction
  cub::DeviceReduce::Reduce(
      tempStorage,
      tempStorageBytes,
      iter,
      result,
      numel,
      ReduceOp<MR_TYPE, TOut>{},
      getReduceInitial<TOut, MR_TYPE>());

  return C;
}

template<MapReduceType MR_TYPE, typename TIn, typename TOut>
Tensor reduceLastDim3DImpl(Tensor A) {
  CHECK(A.getDType() == DType::getType<TIn>());
  CHECK(A.getDim() >= 1);

  // Everything before the last axis is one axis of rows. Folding them needs them laid out as one,
  // which is a copy for a strided input and nothing for a contiguous one.
  std::vector<int> shape = A.getShape();
  int last = shape.back();
  shape.pop_back();

  int rows = 1;
  for (int size : shape) rows *= size;

  Tensor flat = A.isContiguous() ? A : getOperators(Device::kCuda)->contiguous(A);
  flat = flat.view({rows, last});

  Tensor C = createCudaTensor<TOut>({rows});
  if (rows > 0) {
    constexpr int blockSize = 256;
    reduceRowsKernel<TIn, TOut, MR_TYPE, blockSize><<<rows, blockSize>>>(flat, C);

    LL_CUDA_SYNCHRONIZE();
    LL_CHECK_CUDA_STATUS(cudaGetLastError());
  }

  // The reduced axis is gone and the rest come back. A 1D input reduces to one element, which
  // stays a 1D tensor of one rather than becoming a scalar -- what it always was.
  if (shape.empty()) return C;
  return C.view(shape);
}

Tensor reduceLastDim(Tensor A, DType outType, MapReduceType reduceType) {
  DType inType = A.getDType();

  if (inType == DType::kFloat16 && outType == DType::kFloat && reduceType == MapReduceType::SUM_EXP)
    return reduceLastDim3DImpl<MapReduceType::SUM_EXP, half, float>(A);
  if (inType == DType::kFloat16 && outType == DType::kFloat && reduceType == MapReduceType::SUM)
    return reduceLastDim3DImpl<MapReduceType::SUM, half, float>(A);
  if (inType == DType::kFloat16 && outType == DType::kFloat &&
      reduceType == MapReduceType::SUM_SQUARE)
    return reduceLastDim3DImpl<MapReduceType::SUM_SQUARE, half, float>(A);
  if (inType == DType::kFloat16 && outType == DType::kFloat16 && reduceType == MapReduceType::MAX)
    return reduceLastDim3DImpl<MapReduceType::MAX, half, half>(A);
  if (inType == DType::kFloat16 && outType == DType::kFloat16 && reduceType == MapReduceType::MIN)
    return reduceLastDim3DImpl<MapReduceType::MIN, half, half>(A);

  // The same reductions over float32, for a model that runs in full precision on the card -- the
  // kernels were always generic over the type, and only this dispatch was half-only.
  if (inType == DType::kFloat && outType == DType::kFloat) {
    switch (reduceType) {
      case MapReduceType::SUM_EXP:
        return reduceLastDim3DImpl<MapReduceType::SUM_EXP, float, float>(A);
      case MapReduceType::SUM:
        return reduceLastDim3DImpl<MapReduceType::SUM, float, float>(A);
      case MapReduceType::SUM_SQUARE:
        return reduceLastDim3DImpl<MapReduceType::SUM_SQUARE, float, float>(A);
      case MapReduceType::MAX:
        return reduceLastDim3DImpl<MapReduceType::MAX, float, float>(A);
      case MapReduceType::MIN:
        return reduceLastDim3DImpl<MapReduceType::MIN, float, float>(A);
      default:
        break;
    }
  }

  NOT_IMPL();
}

Tensor reduceAll(Tensor A, DType outType, MapReduceType reduceType) {
  DType inType = A.getDType();

  if (inType == DType::kFloat16 && outType == DType::kFloat && reduceType == MapReduceType::SUM)
    return reduceAllImpl<MapReduceType::SUM, half, float>(A);
  if (inType == DType::kFloat && outType == DType::kFloat && reduceType == MapReduceType::SUM)
    return reduceAllImpl<MapReduceType::SUM, float, float>(A);
  if (inType == DType::kBool && outType == DType::kBool && reduceType == MapReduceType::ALL)
    return reduceAllImpl<MapReduceType::ALL, BoolType, BoolType>(A);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
