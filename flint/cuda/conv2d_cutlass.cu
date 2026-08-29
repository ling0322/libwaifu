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

// The convolution on CUTLASS, which is what a build without cuDNN runs.
//
// CUTLASS convolves in NHWC and nothing else -- its activation and filter iterators are written
// for it, and there is no NCHW kernel to ask for -- while the rest of the library is NCHW. So the
// activations are permuted in and out around the kernel. That is not as expensive as it sounds,
// and it is paid back: measured against cuDNN on the shapes SDXL's U-Net runs, the NHWC kernel is
// 7% to 17% faster, because NHWC is the layout the tensor cores want and cuDNN is the one having
// to work around a layout here. With a tiled permute on either side the whole thing still comes
// out ahead on four of the five shapes measured.
//
// A 1x1 convolution with unit stride and no padding is not convolved at all. In NCHW it is a
// matrix multiply -- the weight is (K, C) and the pixels are (C, H * W) already -- so it goes to
// the GEMM, which also means no permute and no CUTLASS convolution for a quarter of the calls
// SDXL makes.

#include "flint/cuda/conv2d_cutlass.h"

#ifdef LIBWAIFU_CUTLASS_ENABLED

#include <cuda_fp16.h>

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"

#include "lutil/error.h"
#include "lutil/strings.h"
#include "flint/cuda/common.h"
#include "flint/functional.h"

#define LL_CHECK_CUTLASS(x)                                                              \
  {                                                                                      \
    cutlass::Status status = x;                                                          \
    if (status != cutlass::Status::kSuccess) {                                           \
      THROW(Aborted, lut::sprintf("%s failed: %s", #x, cutlassGetStatusString(status))); \
    }                                                                                    \
  }

namespace fl {
namespace op {
namespace cuda {
namespace {

namespace cc = cutlass::conv;
using cutlass::layout::TensorNHWC;

/// One convolution kernel, named by what it can take rather than by its shape.
///
/// The problem size is a runtime argument to a CUTLASS convolution -- the kernel extent, the
/// stride and the padding are all in `Conv2dProblemSize` -- so a 1x1 and a 3x3 and a strided one
/// are the same instantiation. What is not runtime is the alignment, which is why there are two
/// half kernels rather than one.
template<
    typename T,
    typename OpClass,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    int Alignment,
    int EpilogueVector,
    cc::IteratorAlgorithm Algorithm,
    int Stages>
using Fprop = cc::device::ImplicitGemmConvolution<typename cc::kernel::DefaultConv2dFprop<
    T,
    TensorNHWC,
    T,
    TensorNHWC,
    T,
    TensorNHWC,
    float,
    OpClass,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<T, EpilogueVector, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
    Stages,
    cutlass::arch::OpMultiplyAdd,
    Algorithm,
    cc::StrideSupport::kStrided,
    Alignment,
    Alignment>::Kernel>;

/// Half on the tensor cores, eight channels at a time. Everything SDXL runs but its first and
/// last convolution goes here.
using HalfAligned = Fprop<
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    8,
    8,
    cc::IteratorAlgorithm::kOptimized,
    3>;

/// The same one channel at a time, for the convolutions that touch the latent: it is four
/// channels deep, so neither the count going in nor the count coming out divides by eight. One
/// rather than four so that this takes any count at all -- the operator is not SDXL's alone, and
/// a three channel image is a perfectly ordinary thing to convolve. Two stages rather than three
/// for the same reason: the deeper pipeline stages its loads with `cp.async`, which moves four,
/// eight or sixteen bytes and cannot be asked for the two that one half is.
using HalfNarrow = Fprop<
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    1,
    1,
    cc::IteratorAlgorithm::kAnalytic,
    2>;

/// Float on the SIMT pipeline, one channel at a time, which takes any shape. The autoencoder is
/// the only thing that runs in float32 and it ends on a three channel image, so this one has to
/// take a width nothing divides.
using FloatAny = Fprop<
    float,
    cutlass::arch::OpClassSimt,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    1,
    1,
    cc::IteratorAlgorithm::kAnalytic,
    2>;

/// What CUTLASS calls the type the rest of the library calls `T`. Layout compatible, distinct to
/// the compiler, so the pointers are reinterpreted where they cross over.
template<typename T>
struct CutlassType;
template<>
struct CutlassType<half> {
  using Type = cutlass::half_t;
};
template<>
struct CutlassType<float> {
  using Type = float;
};

constexpr int kTile = 32;
constexpr int kTileRows = 8;

/// NCHW to NHWC and back, which for one image is transposing a C by H*W matrix.
///
/// Through shared memory so that both ends are coalesced: read along the source's rows, write
/// along the destination's. The row is padded by one element to miss the bank conflict that a
/// power-of-two stride would otherwise put on every column read. The library's own strided copy
/// would do this correctly and about seven times slower -- 146 GB/s against something over 900 --
/// because it reads one element per thread wherever the strides land it.
template<typename T>
__global__ void transposeKernel(
    const T *__restrict__ in,
    T *__restrict__ out,
    int rows,
    int cols) {
  __shared__ T tile[kTile][kTile + 1];

  const T *inImage = in + static_cast<int64_t>(blockIdx.z) * rows * cols;
  T *outImage = out + static_cast<int64_t>(blockIdx.z) * rows * cols;

  int x = blockIdx.x * kTile + threadIdx.x;
  int y = blockIdx.y * kTile + threadIdx.y;
  for (int j = 0; j < kTile; j += kTileRows) {
    if (x < cols && y + j < rows) {
      tile[threadIdx.y + j][threadIdx.x] = inImage[static_cast<int64_t>(y + j) * cols + x];
    }
  }
  __syncthreads();

  x = blockIdx.y * kTile + threadIdx.x;
  y = blockIdx.x * kTile + threadIdx.y;
  for (int j = 0; j < kTile; j += kTileRows) {
    if (x < rows && y + j < cols) {
      outImage[static_cast<int64_t>(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

/// Transpose `batch` matrices of `rows` by `cols`, in place of nothing: `out` is separate.
template<typename T>
void transpose(const T *in, T *out, int batch, int rows, int cols) {
  dim3 grid((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile, batch);
  dim3 block(kTile, kTileRows);
  transposeKernel<T><<<grid, block>>>(in, out, rows, cols);
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

/// One value per output channel, added to every pixel of it. The result is still NHWC here, so
/// the channel is the fastest axis and the read of the bias is the one thing that repeats.
template<typename T>
__global__ void addBiasNhwcKernel(T *data, const T *bias, int channels, int64_t numel) {
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += stride) {
    data[i] = static_cast<T>(static_cast<float>(data[i]) + static_cast<float>(bias[i % channels]));
  }
}

template<typename T>
void addBiasNhwc(Tensor &nhwc, const Tensor &bias, int channels) {
  int64_t numel = nhwc.getNumEl();
  constexpr int kBlock = 256;
  dim3 grid = getGrid1D(static_cast<int>(std::min<int64_t>(numel, 1 << 20)), kBlock);
  addBiasNhwcKernel<T><<<grid, kBlock>>>(
      getDataPtrCuda<T>(nhwc),
      getDataPtrCuda<T>(bias),
      channels,
      numel);
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

/// Run one CUTLASS convolution over tensors already in NHWC.
template<typename Conv, typename T>
void convolve(
    const T *activation,
    const T *filter,
    T *output,
    const cc::Conv2dProblemSize &problem) {
  using Element = typename CutlassType<T>::Type;

  // The iterators' own TensorRef is not over a const element, so the inputs go in as mutable
  // pointers. Nothing writes through them.
  cutlass::TensorRef<Element, TensorNHWC> refA(
      const_cast<Element *>(reinterpret_cast<Element const *>(activation)),
      TensorNHWC::packed(problem.activation_extent()));
  cutlass::TensorRef<Element, TensorNHWC> refB(
      const_cast<Element *>(reinterpret_cast<Element const *>(filter)),
      TensorNHWC::packed(problem.filter_extent()));
  cutlass::TensorRef<Element, TensorNHWC> refD(
      reinterpret_cast<Element *>(output),
      TensorNHWC::packed(problem.output_extent()));

  Conv op;
  typename Conv::Arguments args{problem, refA, refB, refD, refD, {1.0f, 0.0f}};

  LL_CHECK_CUTLASS(op.can_implement(args));

  size_t workspaceSize = Conv::get_workspace_size(args);
  lut::c_ptr<int8_t> workspace;
  if (workspaceSize) workspace = llynCudaAlloc<int8_t>(static_cast<int64_t>(workspaceSize));

  LL_CHECK_CUTLASS(op.initialize(args, workspace.get()));
  LL_CHECK_CUTLASS(op());
}

/// The whole of it for one element type: permute in, convolve, add the bias, permute out.
template<typename T>
Tensor conv2dImpl(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    const Conv2dOptions &options,
    bool narrow) {
  int n = input.getShape(0);
  int c = input.getShape(1);
  int h = input.getShape(2);
  int w = input.getShape(3);
  int k = weight.getShape(0);
  int r = weight.getShape(2);
  int s = weight.getShape(3);

  int outH = (h + 2 * options.padding - options.dilation * (r - 1) - 1) / options.stride + 1;
  int outW = (w + 2 * options.padding - options.dilation * (s - 1) - 1) / options.stride + 1;
  if (outH < 1 || outW < 1) {
    THROW(InvalidArg, "conv2d: the input is smaller than the kernel reaches");
  }

  // The filter is (K, C, R, S) and CUTLASS reads (K, R, S, C), which is the same transpose the
  // activation needs, once per output channel. It is done on every call: a weight is not owned
  // here and nothing says it will still be there next time, and it is small next to what the
  // convolution then does with it.
  Tensor nhwcInput = createCudaTensor<T>({n, h, w, c});
  Tensor nhwcWeight = createCudaTensor<T>({k, r, s, c});
  Tensor nhwcOutput = createCudaTensor<T>({n, outH, outW, k});

  transpose<T>(getDataPtrCuda<T>(input), getDataPtrCuda<T>(nhwcInput), n, c, h * w);
  transpose<T>(getDataPtrCuda<T>(weight), getDataPtrCuda<T>(nhwcWeight), k, c, r * s);

  cc::Conv2dProblemSize problem(
      {n, h, w, c},
      {k, r, s, c},
      {options.padding, options.padding, options.padding, options.padding},
      {options.stride, options.stride},
      {options.dilation, options.dilation},
      {n, outH, outW, k},
      cc::Mode::kCrossCorrelation,
      1);

  const T *activation = getDataPtrCuda<T>(nhwcInput);
  const T *filter = getDataPtrCuda<T>(nhwcWeight);
  T *out = getDataPtrCuda<T>(nhwcOutput);
  if constexpr (std::is_same<T, float>::value) {
    convolve<FloatAny, T>(activation, filter, out, problem);
  } else if (narrow) {
    convolve<HalfNarrow, T>(activation, filter, out, problem);
  } else {
    convolve<HalfAligned, T>(activation, filter, out, problem);
  }

  if (!bias.empty()) addBiasNhwc<T>(nhwcOutput, bias, k);

  Tensor output = createCudaTensor<T>({n, k, outH, outW});
  transpose<T>(getDataPtrCuda<T>(nhwcOutput), getDataPtrCuda<T>(output), n, outH * outW, k);

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return output;
}

/// A 1x1 convolution with unit stride and no padding, which is a matrix multiply wearing a hat.
///
/// In NCHW the pixels of one image are already (C, H * W) and the weight is already (K, C), so
/// the product is (K, H * W), which is the answer in the layout it was wanted in. No permute, no
/// convolution kernel, and it is a quarter of the convolutions SDXL runs.
Tensor conv1x1(const Tensor &input, const Tensor &weight, const Tensor &bias) {
  int n = input.getShape(0);
  int c = input.getShape(1);
  int h = input.getShape(2);
  int w = input.getShape(3);
  int k = weight.getShape(0);

  // (K, C) by (C, H * W), which is (K, H * W): the answer, already in NCHW. One image at a time,
  // because the weight is the same for all of them and matmul does not broadcast a smaller left
  // operand over a batched right one.
  Tensor flatWeight = weight.view({k, c});
  Tensor result;
  for (int i = 0; i < n; ++i) {
    Tensor image = n == 1 ? input.view({c, h * w}) : input.subtensor(i).view({c, h * w});
    Tensor product = F::matmul(flatWeight, image);
    result = result.empty() ? product : F::cat(result, product, 0);
  }

  result = result.view({n, k, h, w});
  if (!bias.empty()) result = F::add(result, bias.view({1, k, 1, 1}));

  return result;
}

}  // namespace

bool isConv2dCutlassAvailable() {
  return true;
}

Tensor conv2dCutlass(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    const Conv2dOptions &options) {
  if (input.getDim() != 4) THROW(InvalidArg, "conv2d takes a 4-D input, as (N, C, H, W)");
  if (weight.getDim() != 4) THROW(InvalidArg, "conv2d takes a 4-D weight, as (K, C, R, S)");
  if (input.getDType() != weight.getDType()) {
    THROW(InvalidArg, "conv2d: the input and the weight are of different types");
  }
  if (options.groups != 1) {
    THROW(InvalidArg, "conv2d on CUTLASS: a grouped convolution is not implemented");
  }
  if (options.stride < 1 || options.dilation < 1) {
    THROW(InvalidArg, "conv2d: the stride and the dilation are below one");
  }
  if (options.padding < 0) THROW(InvalidArg, "conv2d: the padding is negative");
  LL_CHECK_CONTIGUOUS(input);
  LL_CHECK_CONTIGUOUS(weight);

  if (input.getShape(1) != weight.getShape(1)) {
    THROW(
        InvalidArg,
        lut::sprintf(
            "conv2d: an input of %d channels does not match a weight of %d",
            input.getShape(1),
            weight.getShape(1)));
  }
  if (!bias.empty()) {
    if (bias.getNumEl() != weight.getShape(0)) {
      THROW(InvalidArg, "conv2d: the bias does not match the output channels");
    }
    if (bias.getDType() != input.getDType()) {
      THROW(InvalidArg, "conv2d: the bias and the input are of different types");
    }
  }

  int c = input.getShape(1);
  int k = weight.getShape(0);
  bool isPointwise = weight.getShape(2) == 1 && weight.getShape(3) == 1 && options.stride == 1 &&
                     options.padding == 0 && options.dilation == 1;
  if (isPointwise) return conv1x1(input, weight, bias);

  bool narrow = c % 8 != 0 || k % 8 != 0;
  if (input.getDType() == DType::kFloat16) {
    return conv2dImpl<half>(input, weight, bias, options, narrow);
  }
  if (input.getDType() == DType::kFloat) {
    return conv2dImpl<float>(input, weight, bias, options, narrow);
  }

  THROW(InvalidArg, "conv2d takes a <half> or <float> input");
}

}  // namespace cuda
}  // namespace op
}  // namespace fl

#else  // LIBWAIFU_CUTLASS_ENABLED

namespace fl {
namespace op {
namespace cuda {

bool isConv2dCutlassAvailable() {
  return false;
}

Tensor conv2dCutlass(const Tensor &, const Tensor &, const Tensor &, const Conv2dOptions &) {
  THROW(Aborted, "conv2d on CUTLASS: this build has no CUTLASS");
}

}  // namespace cuda
}  // namespace op
}  // namespace fl

#endif  // LIBWAIFU_CUTLASS_ENABLED
