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

#include <cuda_fp16.h>

#include "lutil/error.h"
#include "lutil/strings.h"
#include "flint/cuda/common.h"
#include "flint/cuda/upsample.h"

namespace fl {
namespace op {
namespace cuda {
namespace {

/// One thread per output pixel, reading the input pixel it copies. Driving this from the output
/// rather than the input is what keeps the writes contiguous, and the reads land on the same
/// input pixel for `scale` neighbouring threads, which the cache is happy to serve.
///
/// The index arithmetic goes through FastDivmod: the divisors are the same for every thread, so
/// each division becomes a multiply-high plus a shift instead of the integer divide the hardware
/// has no unit for. Both divisors are loop-invariant, but nvcc cannot strength-reduce them on its
/// own since it only learns their values at launch.
template<typename T>
__global__ void upsampleNearest2dKernel(
    const T *__restrict__ input,
    T *__restrict__ output,
    int inputW,
    FastDivmod outputWDivmod,
    FastDivmod scaleDivmod,
    uint32_t numel) {
  uint32_t stride = blockDim.x * gridDim.x;

  for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
    // `outRow` counts output rows across the whole batch, that is plane * outputH + outY. Its
    // quotient by `scale` is the matching input row, plane * inputH + outY / scale, because
    // outputH = inputH * scale makes the plane part an exact multiple of `scale`; that folds the
    // plane and the row into one division rather than two.
    uint32_t outRow, outX;
    outputWDivmod.divmod(idx, outRow, outX);

    uint32_t inRow, inX, remainder;
    scaleDivmod.divmod(outRow, inRow, remainder);
    scaleDivmod.divmod(outX, inX, remainder);

    output[idx] = input[inRow * inputW + inX];
  }
}

/// The pixel is only ever copied, never arithmetic on, so the element type is carried through
/// rather than converted at either end.
template<typename T>
Tensor upsampleNearest2dImpl(const Tensor &input, int scale) {
  int inputH = input.getShape(2);
  int inputW = input.getShape(3);
  int outputH = inputH * scale;
  int outputW = inputW * scale;

  Tensor output = createCudaTensor<T>({input.getShape(0), input.getShape(1), outputH, outputW});
  int64_t numel = output.getNumEl();
  if (numel > std::numeric_limits<int32_t>::max()) {
    THROW(InvalidArg, "upsampleNearest2d: the result is too large");
  }

  constexpr int kBlockSize = 256;
  dim3 grid = getGrid1D(static_cast<int>(numel), kBlockSize);
  upsampleNearest2dKernel<T><<<grid, kBlockSize>>>(
      getDataPtrCuda<T>(input),
      getDataPtrCuda<T>(output),
      inputW,
      FastDivmod(outputW),
      FastDivmod(scale),
      static_cast<uint32_t>(numel));

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return output;
}

}  // namespace

Tensor upsampleNearest2d(const Tensor &input, int scale) {
  if (input.getDim() != 4) {
    THROW(InvalidArg, "upsampleNearest2d takes a 4-D input, as (N, C, H, W)");
  }
  if (scale < 1) THROW(InvalidArg, "upsampleNearest2d: the scale is below one");
  LL_CHECK_CONTIGUOUS(input);

  if (input.getDType() == DType::kFloat16) return upsampleNearest2dImpl<half>(input, scale);
  if (input.getDType() == DType::kFloat) return upsampleNearest2dImpl<float>(input, scale);

  THROW(InvalidArg, "upsampleNearest2d takes a <half> or <float> input");
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
