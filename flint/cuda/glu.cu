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

#include <algorithm>
#include <type_traits>

#include "flint/cuda/common.h"
#include "flint/cuda/glu.h"
#include "flint/operators.h"

namespace fl {
namespace op {
namespace cuda {

/// Which activation the gate half goes through. The two halves are used the same way either way:
/// the first is the gate, the second is the value, and the result is the value times the
/// activated gate.
enum class GateOp { SILU, GELU };

template<GateOp OP>
__forceinline__ __device__ float applyGate(float x) {
  if constexpr (OP == GateOp::SILU) {
    return x / (1.0f + expf(-x));
  } else {
    // The exact GELU, so that this matches torch.nn.GELU() rather than its tanh approximation.
    return x * 0.5f * (1.0f + erff(x * 0.70710678118654752f));
  }
}

// Both storage types are read into float and written back from it, so the arithmetic is float32
// whichever one is stored.
__forceinline__ __device__ float toFloat(half x) {
  return __half2float(x);
}

__forceinline__ __device__ float toFloat(float x) {
  return x;
}

template<typename T>
__forceinline__ __device__ T fromFloat(float x);

template<>
__forceinline__ __device__ half fromFloat<half>(float x) {
  return __float2half(x);
}

template<>
__forceinline__ __device__ float fromFloat<float>(float x) {
  return x;
}

/// A contiguous input read as `rows` rows of `inputWidth`. The rows are spread over the grid's y
/// and z axes, each at most 65535 long, so the kernel skips the rows past the end that rounding
/// the grid up adds. VECTORIZED reads and writes pairs as half2, and is only instantiated for half.
template<typename T, bool VECTORIZED, GateOp OP>
__global__ void gatedLinearContiguousKernel(
    const T *__restrict__ input,
    T *__restrict__ output,
    int64_t rows,
    int inputWidth,
    int outputWidth) {
  int64_t row = static_cast<int64_t>(blockIdx.z) * gridDim.y + blockIdx.y;
  if (row >= rows) return;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if constexpr (VECTORIZED) {
    static_assert(std::is_same_v<T, half>, "half2 is for half");
    int x2 = x * 2;
    if (x2 >= outputWidth) return;

    int64_t inputOffset = row * inputWidth + x2;
    int64_t outputOffset = row * outputWidth + x2;
    float2 gate = __half22float2(*reinterpret_cast<const half2 *>(input + inputOffset));
    float2 value =
        __half22float2(*reinterpret_cast<const half2 *>(input + inputOffset + outputWidth));
    *reinterpret_cast<half2 *>(output + outputOffset) = __floats2half2_rn(
        value.x * applyGate<OP>(gate.x),
        value.y * applyGate<OP>(gate.y));
  } else {
    if (x >= outputWidth) return;

    int64_t inputOffset = row * inputWidth + x;
    float gate = toFloat(input[inputOffset]);
    float value = toFloat(input[inputOffset + outputWidth]);
    output[row * outputWidth + x] = fromFloat<T>(value * applyGate<OP>(gate));
  }
}

template<typename T, bool VECTORIZED, GateOp OP>
__global__ void gatedLinearStridedKernel(
    const T *__restrict__ input,
    T *__restrict__ output,
    int inputStride0,
    int inputStride1,
    int inputStride2,
    int outputWidth) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int64_t inputOffset =
      static_cast<int64_t>(z) * inputStride0 + static_cast<int64_t>(y) * inputStride1;
  int64_t outputOffset = (static_cast<int64_t>(z) * gridDim.y + y) * outputWidth;

  if constexpr (VECTORIZED) {
    static_assert(std::is_same_v<T, half>, "half2 is for half");
    int x2 = x * 2;
    if (x2 >= outputWidth) return;

    float2 gate = __half22float2(*reinterpret_cast<const half2 *>(input + inputOffset + x2));
    float2 value = __half22float2(
        *reinterpret_cast<const half2 *>(input + inputOffset + outputWidth + x2));
    *reinterpret_cast<half2 *>(output + outputOffset + x2) = __floats2half2_rn(
        value.x * applyGate<OP>(gate.x),
        value.y * applyGate<OP>(gate.y));
  } else {
    if (x >= outputWidth) return;

    int64_t gateOffset = inputOffset + static_cast<int64_t>(x) * inputStride2;
    float gate = toFloat(input[gateOffset]);
    float value = toFloat(input[gateOffset + static_cast<int64_t>(outputWidth) * inputStride2]);
    output[outputOffset + x] = fromFloat<T>(value * applyGate<OP>(gate));
  }
}

template<typename T>
bool alignedForHalf2(const T *pointer) {
  return reinterpret_cast<uintptr_t>(pointer) % alignof(half2) == 0;
}

/// Any rank, contiguous: every leading dimension folded into rows.
template<typename T, GateOp OP>
Tensor gatedLinearContiguous(const Tensor &tensor) {
  std::vector<Tensor::ShapeType> shapeC = tensor.getShape();
  shapeC.back() /= 2;
  Tensor C = createCudaTensor<T>(shapeC);

  int inputWidth = tensor.getShape(-1);
  int outputWidth = inputWidth / 2;
  int64_t rows = inputWidth == 0 ? 0 : tensor.getNumEl() / inputWidth;
  if (rows == 0 || outputWidth == 0) return C;

  constexpr int blockSize = 256;
  constexpr int64_t gridLimit = 65535;
  dim3 d;
  d.y = static_cast<unsigned>(std::min(rows, gridLimit));
  d.z = static_cast<unsigned>((rows + d.y - 1) / d.y);
  CHECK(d.z <= gridLimit);

  const T *input = getDataPtrCuda<T>(tensor);
  T *output = getDataPtrCuda<T>(C);
  bool useHalf2 = false;
  if constexpr (std::is_same_v<T, half>) {
    useHalf2 = outputWidth % 2 == 0 && alignedForHalf2(input) && alignedForHalf2(output);
  }

  if (useHalf2) {
    if constexpr (std::is_same_v<T, half>) {
      d.x = (outputWidth / 2 + blockSize - 1) / blockSize;
      gatedLinearContiguousKernel<T, true, OP>
          <<<d, blockSize>>>(input, output, rows, inputWidth, outputWidth);
    }
  } else {
    d.x = (outputWidth + blockSize - 1) / blockSize;
    gatedLinearContiguousKernel<T, false, OP>
        <<<d, blockSize>>>(input, output, rows, inputWidth, outputWidth);
  }

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return C;
}

/// Rank 3 and strided, read through its strides without a copy.
template<typename T, GateOp OP>
Tensor gatedLinearStrided3D(const Tensor &tensor) {
  std::vector<Tensor::ShapeType> shapeC = tensor.getShape();
  shapeC.back() /= 2;
  Tensor C = createCudaTensor<T>(shapeC);
  if (C.getNumEl() == 0) return C;

  constexpr int blockSize = 256;
  dim3 d;
  d.z = C.getShape(0);
  d.y = C.getShape(1);

  const T *input = getDataPtrCuda<T>(tensor);
  T *output = getDataPtrCuda<T>(C);
  int inputStride0 = tensor.getStride(0);
  int inputStride1 = tensor.getStride(1);
  int inputStride2 = tensor.getStride(2);
  int outputWidth = C.getShape(2);

  bool useHalf2 = false;
  if constexpr (std::is_same_v<T, half>) {
    useHalf2 = inputStride2 == 1 && outputWidth % 2 == 0 && inputStride0 % 2 == 0 &&
               inputStride1 % 2 == 0 && alignedForHalf2(input) && alignedForHalf2(output);
  }

  if (useHalf2) {
    if constexpr (std::is_same_v<T, half>) {
      d.x = (outputWidth / 2 + blockSize - 1) / blockSize;
      gatedLinearStridedKernel<T, true, OP><<<d, blockSize>>>(
          input, output, inputStride0, inputStride1, inputStride2, outputWidth);
    }
  } else {
    d.x = (outputWidth + blockSize - 1) / blockSize;
    gatedLinearStridedKernel<T, false, OP><<<d, blockSize>>>(
        input, output, inputStride0, inputStride1, inputStride2, outputWidth);
  }

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return C;
}

template<typename T, GateOp OP>
Tensor gatedLinearOf(const Tensor &tensor) {
  if (tensor.isContiguous()) return gatedLinearContiguous<T, OP>(tensor);

  // A strided view of rank 3 is read through its strides; rank 2 is that with one leading row.
  if (tensor.getDim() == 3) return gatedLinearStrided3D<T, OP>(tensor);
  if (tensor.getDim() == 2) return gatedLinearStrided3D<T, OP>(tensor.unsqueeze(0)).subtensor(0);

  // Any other strided rank is copied whole first.
  return gatedLinearContiguous<T, OP>(getOperators(Device::kCuda)->contiguous(tensor));
}

template<GateOp OP>
Tensor gatedLinear(const Tensor &tensor) {
  CHECK(tensor.getDevice().getType() == Device::kCuda);
  CHECK(tensor.getDim() >= 1);
  CHECK(tensor.getShape(-1) % 2 == 0);

  // Checked rather than assumed: a float32 tensor read as half is garbage, not an error.
  if (tensor.getDType() == DType::kFloat16) return gatedLinearOf<half, OP>(tensor);
  if (tensor.getDType() == DType::kFloat) return gatedLinearOf<float, OP>(tensor);

  THROW(InvalidArg, "swiglu and geglu take a float16 or float tensor on CUDA");
}

Tensor swiglu(const Tensor &tensor) {
  return gatedLinear<GateOp::SILU>(tensor);
}

Tensor geglu(const Tensor &tensor) {
  return gatedLinear<GateOp::GELU>(tensor);
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
