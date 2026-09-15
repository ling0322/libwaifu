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

// The three normalizations, which are one formula: (x - mean) / sqrt(variance + eps) * w + b over
// some slice of the tensor. An RMS norm is the case where the mean is left at zero and the scale
// comes from the mean square instead of the variance, which is why it shares every kernel here
// rather than having its own. What differs between them is the slice: the last dimension for
// layerNorm and rmsNorm, a group of channels and the space it covers for groupNorm.

#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>

#include <type_traits>

#include "lutil/error.h"
#include "lutil/strings.h"
#include "flint/cuda/accessor.h"
#include "flint/cuda/common.h"
#include "flint/cuda/norm.h"

namespace fl {
namespace op {
namespace cuda {
namespace {

constexpr int kBlockSize = 256;

/// What a slice is rescaled by.
struct Moments {
  float mean;
  float invStd;
};

/// Finish the block's partial sums. With SUBTRACT_MEAN off the mean stays at zero and the scale
/// comes from the mean square, which is the RMS form; the sum is then never accumulated and never
/// reduced, so that case costs exactly what it did before the two were shared.
template<int BLOCK_SIZE, bool SUBTRACT_MEAN>
__forceinline__ __device__ Moments
reduceMoments(float sum, float sumSquare, int64_t count, float eps) {
  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage squareStorage;
  __shared__ Moments shared;

  sumSquare = BlockReduce(squareStorage).Sum(sumSquare);
  if constexpr (SUBTRACT_MEAN) {
    __shared__ typename BlockReduce::TempStorage sumStorage;
    sum = BlockReduce(sumStorage).Sum(sum);
  }

  if (threadIdx.x == 0) {
    float mean = SUBTRACT_MEAN ? sum / count : 0.0f;

    // The variance as E[x^2] - E[x]^2. Accumulating in float over a few thousand values keeps the
    // cancellation well away from anything half could tell apart, and a negative result can still
    // fall out of rounding, which the epsilon covers.
    float variance = sumSquare / count - mean * mean;
    shared.mean = mean;
    shared.invStd = rsqrtf(variance > 0.0f ? variance + eps : eps);
  }
  __syncthreads();

  return shared;
}

/// One block per row of a contiguous tensor. VECTORIZED reads and writes a half2 at a time, which
/// needs an even width, operands aligned for it, and a half tensor: it is the only case where two
/// elements fit one machine word, so it is never instantiated for float.
template<typename T, int BLOCK_SIZE, bool SUBTRACT_MEAN, bool VECTORIZED>
__global__ void normRowKernel(
    const T *__restrict__ input,
    const T *__restrict__ weight,
    const T *__restrict__ bias,
    T *__restrict__ output,
    int hiddenSize,
    int weightStride,
    int biasStride,
    float eps) {
  int64_t rowOffset = static_cast<int64_t>(blockIdx.x) * hiddenSize;
  float sum = 0.0f;
  float sumSquare = 0.0f;

  if constexpr (VECTORIZED) {
    const half2 *input2 = reinterpret_cast<const half2 *>(input + rowOffset);
    int width = hiddenSize / 2;
    for (int i = threadIdx.x; i < width; i += BLOCK_SIZE) {
      float2 value = __half22float2(input2[i]);
      sumSquare += value.x * value.x + value.y * value.y;
      if constexpr (SUBTRACT_MEAN) sum += value.x + value.y;
    }
  } else {
    for (int i = threadIdx.x; i < hiddenSize; i += BLOCK_SIZE) {
      float value = static_cast<float>(input[rowOffset + i]);
      sumSquare += value * value;
      if constexpr (SUBTRACT_MEAN) sum += value;
    }
  }

  Moments moments = reduceMoments<BLOCK_SIZE, SUBTRACT_MEAN>(sum, sumSquare, hiddenSize, eps);

  if constexpr (VECTORIZED) {
    const half2 *input2 = reinterpret_cast<const half2 *>(input + rowOffset);
    const half2 *weight2 = reinterpret_cast<const half2 *>(weight);
    const half2 *bias2 = reinterpret_cast<const half2 *>(bias);
    half2 *output2 = reinterpret_cast<half2 *>(output + rowOffset);

    int width = hiddenSize / 2;
    for (int i = threadIdx.x; i < width; i += BLOCK_SIZE) {
      float2 value = __half22float2(input2[i]);
      float x = (value.x - moments.mean) * moments.invStd;
      float y = (value.y - moments.mean) * moments.invStd;
      if (weight) {
        float2 scale = __half22float2(weight2[i]);
        x *= scale.x;
        y *= scale.y;
      }
      if (bias) {
        float2 shift = __half22float2(bias2[i]);
        x += shift.x;
        y += shift.y;
      }
      output2[i] = __floats2half2_rn(x, y);
    }
  } else {
    for (int i = threadIdx.x; i < hiddenSize; i += BLOCK_SIZE) {
      float value = (static_cast<float>(input[rowOffset + i]) - moments.mean) * moments.invStd;
      if (weight) value *= static_cast<float>(weight[i * weightStride]);
      if (bias) value += static_cast<float>(bias[i * biasStride]);

      output[rowOffset + i] = static_cast<T>(value);
    }
  }
}

/// The same, for a tensor whose rows are not packed, read through its strides.
template<typename T, int BLOCK_SIZE, bool SUBTRACT_MEAN>
__global__ void normRowStridedKernel(
    PackedTensorAccessor<const T, 3> inputTensor,
    const T *__restrict__ weight,
    const T *__restrict__ bias,
    PackedTensorAccessor<T, 3> outputTensor,
    int weightStride,
    int biasStride,
    float eps) {
  int hiddenSize = inputTensor.getShape(2);
  int y = blockIdx.x % inputTensor.getShape(1);
  int z = blockIdx.x / inputTensor.getShape(1);

  float sum = 0.0f;
  float sumSquare = 0.0f;
  for (int x = threadIdx.x; x < hiddenSize; x += BLOCK_SIZE) {
    float value = static_cast<float>(inputTensor[z][y][x]);
    sumSquare += value * value;
    if constexpr (SUBTRACT_MEAN) sum += value;
  }

  Moments moments = reduceMoments<BLOCK_SIZE, SUBTRACT_MEAN>(sum, sumSquare, hiddenSize, eps);

  for (int x = threadIdx.x; x < hiddenSize; x += BLOCK_SIZE) {
    float value = (static_cast<float>(inputTensor[z][y][x]) - moments.mean) * moments.invStd;
    if (weight) value *= static_cast<float>(weight[x * weightStride]);
    if (bias) value += static_cast<float>(bias[x * biasStride]);

    outputTensor[z][y][x] = static_cast<T>(value);
  }
}

/// One block per (image, group). A group covers `channelPerGroup` channels of `spatial` pixels
/// each, and they are contiguous, so the whole group is one run of memory.
template<typename T, int BLOCK_SIZE>
__global__ void groupNormKernel(
    const T *__restrict__ input,
    const T *__restrict__ weight,
    const T *__restrict__ bias,
    T *__restrict__ output,
    int channelPerGroup,
    int spatial,
    int groups,
    float eps) {
  int group = blockIdx.x % groups;
  int64_t groupSize = static_cast<int64_t>(channelPerGroup) * spatial;
  int64_t offset = static_cast<int64_t>(blockIdx.x) * groupSize;

  float sum = 0.0f;
  float sumSquare = 0.0f;
  for (int64_t i = threadIdx.x; i < groupSize; i += BLOCK_SIZE) {
    float value = static_cast<float>(input[offset + i]);
    sum += value;
    sumSquare += value * value;
  }

  Moments moments = reduceMoments<BLOCK_SIZE, true>(sum, sumSquare, groupSize, eps);

  // The scale and the shift are per channel, not per group, so the channel this element belongs
  // to has to be recovered from where it sits inside the group.
  for (int64_t i = threadIdx.x; i < groupSize; i += BLOCK_SIZE) {
    float value = (static_cast<float>(input[offset + i]) - moments.mean) * moments.invStd;
    int channel = group * channelPerGroup + static_cast<int>(i / spatial);
    if (weight) value *= static_cast<float>(weight[channel]);
    if (bias) value += static_cast<float>(bias[channel]);

    output[offset + i] = static_cast<T>(value);
  }
}

/// The weight and the bias are read in the input's own type rather than converted, so a norm is
/// asked for in one precision throughout.
void checkNormOperand(const Tensor &x, const char *what, int expected, DType dtype) {
  if (x.empty()) return;

  if (x.getDType() != dtype) {
    THROW(InvalidArg, lut::sprintf("%s is not %s", what, dtype.toString()));
  }
  if (x.getDim() != 1) {
    THROW(InvalidArg, lut::sprintf("%s is not one dimensional", what));
  }
  if (x.getNumEl() != expected) {
    THROW(
        InvalidArg,
        lut::sprintf("%s holds %d values, not %d", what, int(x.getNumEl()), expected));
  }
}

/// A norm's weight and bias are read one element per position, so a view of a wider tensor works
/// as long as its step is carried along. A stride of one is the ordinary case and the only one the
/// vectorized arm can take.
int strideOf(const Tensor &x) {
  return x.empty() ? 1 : x.getStride(0);
}

template<typename T>
const T *dataOrNull(const Tensor &x) {
  return x.empty() ? nullptr : getDataPtrCuda<T>(x);
}

template<typename T>
bool isHalf2Aligned(const T *p) {
  return reinterpret_cast<uintptr_t>(p) % alignof(half2) == 0;
}

/// Normalize over the last dimension, which is what layerNorm and rmsNorm both do.
template<typename T, bool SUBTRACT_MEAN>
Tensor normOverLastDimImpl(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    float eps) {
  DType dtype = DType::getType<T>();
  int hiddenSize = input.getShape(-1);
  checkNormOperand(weight, "the norm weight", hiddenSize, dtype);
  checkNormOperand(bias, "the norm bias", hiddenSize, dtype);

  int64_t numRow64 = input.getNumEl() / hiddenSize;
  if (numRow64 > std::numeric_limits<int32_t>::max()) THROW(InvalidArg, "this norm: too many rows");
  int numRow = static_cast<int>(numRow64);

  Tensor output = createCudaTensor<T>(input.getShape());
  const T *weightData = dataOrNull<T>(weight);
  const T *biasData = dataOrNull<T>(bias);

  int weightStride = strideOf(weight);
  int biasStride = strideOf(bias);

  if (!input.isContiguous()) {
    // The strided kernel indexes three dimensions, so anything else is folded into them first.
    Tensor input3D = input.getDim() == 3 ? input : input.unsqueeze(0);
    Tensor output3D = output.getDim() == 3 ? output : output.unsqueeze(0);
    if (input3D.getDim() != 3) THROW(InvalidArg, "this norm: an input of this rank is not strided");

    normRowStridedKernel<T, kBlockSize, SUBTRACT_MEAN><<<numRow, kBlockSize>>>(
        input3D,
        weightData,
        biasData,
        output3D,
        weightStride,
        biasStride,
        eps);
  } else {
    const T *inputData = getDataPtrCuda<T>(input);
    T *outputData = getDataPtrCuda<T>(output);

    // Two elements to a machine word is a half's own arrangement; a float row is read one at a
    // time whatever it is aligned to.
    bool vectorized = std::is_same<T, half>::value && hiddenSize % 2 == 0 && weightStride == 1 &&
                      biasStride == 1 && isHalf2Aligned(inputData) &&
                      isHalf2Aligned(outputData) && (!weightData || isHalf2Aligned(weightData)) &&
                      (!biasData || isHalf2Aligned(biasData));

    if (vectorized) {
      normRowKernel<T, kBlockSize, SUBTRACT_MEAN, std::is_same<T, half>::value>
          <<<numRow, kBlockSize>>>(
              inputData, weightData, biasData, outputData, hiddenSize, 1, 1, eps);
    } else {
      normRowKernel<T, kBlockSize, SUBTRACT_MEAN, false><<<numRow, kBlockSize>>>(
          inputData, weightData, biasData, outputData, hiddenSize, weightStride, biasStride, eps);
    }
  }

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return output;
}

template<bool SUBTRACT_MEAN>
Tensor normOverLastDim(const Tensor &input, const Tensor &weight, const Tensor &bias, float eps) {
  if (input.getDim() < 1) THROW(InvalidArg, "this norm takes an input of at least one dimension");

  if (input.getDType() == DType::kFloat16) {
    return normOverLastDimImpl<half, SUBTRACT_MEAN>(input, weight, bias, eps);
  }
  if (input.getDType() == DType::kFloat) {
    return normOverLastDimImpl<float, SUBTRACT_MEAN>(input, weight, bias, eps);
  }

  THROW(InvalidArg, "this norm takes a <half> or <float> input");
}

template<typename T>
Tensor groupNormImpl(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    int groups,
    float eps) {
  int batch = input.getShape(0);
  int channel = input.getShape(1);
  if (groups < 1 || channel % groups != 0) {
    THROW(
        InvalidArg,
        lut::sprintf("groupNorm: %d channels do not divide into %d groups", channel, groups));
  }

  DType dtype = DType::getType<T>();
  checkNormOperand(weight, "the groupNorm weight", channel, dtype);
  checkNormOperand(bias, "the groupNorm bias", channel, dtype);

  int spatial = input.getShape(2) * input.getShape(3);
  Tensor output = createCudaTensor<T>(input.getShape());

  groupNormKernel<T, kBlockSize><<<batch * groups, kBlockSize>>>(
      getDataPtrCuda<T>(input),
      dataOrNull<T>(weight),
      dataOrNull<T>(bias),
      getDataPtrCuda<T>(output),
      channel / groups,
      spatial,
      groups,
      eps);

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return output;
}

}  // namespace

Tensor rmsNorm(const Tensor &input, const Tensor &weight, float eps) {
  if (weight.empty()) THROW(InvalidArg, "rmsNorm needs a weight");

  return normOverLastDim<false>(input, weight, Tensor(), eps);
}

Tensor layerNorm(const Tensor &input, const Tensor &weight, const Tensor &bias, float eps) {
  return normOverLastDim<true>(input, weight, bias, eps);
}

Tensor groupNorm(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    int groups,
    float eps) {
  if (input.getDim() != 4) THROW(InvalidArg, "groupNorm takes a 4-D input, as (N, C, H, W)");
  LL_CHECK_CONTIGUOUS(input);

  if (input.getDType() == DType::kFloat16) {
    return groupNormImpl<half>(input, weight, bias, groups, eps);
  }
  if (input.getDType() == DType::kFloat) {
    return groupNormImpl<float>(input, weight, bias, groups, eps);
  }

  THROW(InvalidArg, "groupNorm takes a <half> or <float> input");
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
