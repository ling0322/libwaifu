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
//
// The 1-D convolution is the same machinery over an image one row tall. It is here rather than
// a composition of `conv2d` because `conv2d` pads both axes alike, and the axis of length one must
// not be padded -- a problem size takes the two paddings separately, so here it simply is not. It
// also takes groups, which `conv2d` does not: the depthwise case, as many groups as channels, has
// a CUTLASS kernel of its own and is what a conformer asks for; any other group count is one
// ungrouped convolution per group, each reading and writing its own channels of the tensors the
// whole convolution shares.

#include "flint/cuda/conv1d.h"
#include "flint/cuda/conv2d_cutlass.h"

#include <cuda_fp16.h>

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/default_depthwise_fprop.h"

#include "lutil/error.h"
#include "lutil/strings.h"
#include "flint/cuda/common.h"
#include "flint/operators.h"

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

/// A depthwise convolution: as many groups as channels, one channel each, in and out. The implicit
/// GEMM's reduction is then only the kernel's taps, too short for a tensor core to be worth it, so
/// this is on the SIMT pipeline in both types. It reads one element at a time, so it takes any
/// channel count. `T` is CUTLASS's element type, not the library's.
template<typename T>
using Depthwise = cc::device::ImplicitGemmConvolution<typename cc::kernel::DefaultDepthwiseFprop<
    T,
    TensorNHWC,
    T,
    TensorNHWC,
    T,
    TensorNHWC,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<T, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cc::IteratorAlgorithm::kAnalytic,
    cc::StrideSupport::kStrided>::Kernel>;

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
///
/// The layouts are the caller's rather than packed from the problem, so that a group can be a
/// window onto tensors wider than it: its channels are a run inside every pixel of the whole, and
/// the stride from one pixel to the next is the whole's channel count, not the group's.
template<typename Conv, typename T>
void convolve(
    const T *activation,
    TensorNHWC activationLayout,
    const T *filter,
    TensorNHWC filterLayout,
    T *output,
    TensorNHWC outputLayout,
    const cc::Conv2dProblemSize &problem) {
  using Element = typename CutlassType<T>::Type;

  // The iterators' own TensorRef is not over a const element, so the inputs go in as mutable
  // pointers. Nothing writes through them.
  cutlass::TensorRef<Element, TensorNHWC> refA(
      const_cast<Element *>(reinterpret_cast<Element const *>(activation)),
      activationLayout);
  cutlass::TensorRef<Element, TensorNHWC> refB(
      const_cast<Element *>(reinterpret_cast<Element const *>(filter)),
      filterLayout);
  cutlass::TensorRef<Element, TensorNHWC> refD(reinterpret_cast<Element *>(output), outputLayout);

  Conv op;
  typename Conv::Arguments args{problem, refA, refB, refD, refD, {1.0f, 0.0f}};

  LL_CHECK_CUTLASS(op.can_implement(args));

  size_t workspaceSize = Conv::get_workspace_size(args);
  lut::c_ptr<int8_t> workspace;
  if (workspaceSize) workspace = llynCudaAlloc<int8_t>(static_cast<int64_t>(workspaceSize));

  LL_CHECK_CUTLASS(op.initialize(args, workspace.get()));
  LL_CHECK_CUTLASS(op());
}

/// The same over tensors that are exactly the problem's extent, which is every call but a group.
template<typename Conv, typename T>
void convolvePacked(
    const T *activation,
    const T *filter,
    T *output,
    const cc::Conv2dProblemSize &problem) {
  convolve<Conv, T>(
      activation,
      TensorNHWC::packed(problem.activation_extent()),
      filter,
      TensorNHWC::packed(problem.filter_extent()),
      output,
      TensorNHWC::packed(problem.output_extent()),
      problem);
}

/// How a convolution steps over its input, one axis at a time. `conv2d` asks for the same on both;
/// `conv1d` asks for nothing at all on the height, which is one.
struct Geometry {
  int padH;
  int padW;
  int strideH;
  int strideW;
  int dilationH;
  int dilationW;
  int groups;
};

/// The whole of it for one element type: permute in, convolve, add the bias, permute out.
///
/// `input` is (N, C, H, W) and `weight` (K, C / groups, R, S), both contiguous and both already
/// checked by the caller; what is left to refuse here is a kernel that reaches past the input.
/// `name` is the operator the caller asked for, for the message that refusal carries.
template<typename T>
Tensor convNchw(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    const Geometry &g,
    const char *name) {
  int n = input.getShape(0);
  int c = input.getShape(1);
  int h = input.getShape(2);
  int w = input.getShape(3);
  int k = weight.getShape(0);
  int groupC = weight.getShape(1);
  int r = weight.getShape(2);
  int s = weight.getShape(3);
  int groupK = k / g.groups;

  int outH = (h + 2 * g.padH - g.dilationH * (r - 1) - 1) / g.strideH + 1;
  int outW = (w + 2 * g.padW - g.dilationW * (s - 1) - 1) / g.strideW + 1;
  if (h + 2 * g.padH < g.dilationH * (r - 1) + 1 || w + 2 * g.padW < g.dilationW * (s - 1) + 1) {
    THROW(InvalidArg, lut::sprintf("%s: the input is smaller than the kernel reaches", name));
  }

  // The filter is (K, C / groups, R, S) and CUTLASS reads (K, R, S, C / groups), which is the same
  // transpose the activation needs, once per output channel. It is done on every call: a weight
  // is not owned here and nothing says it will still be there next time, and it is small next to
  // what the convolution then does with it.
  Tensor nhwcInput = createCudaTensor<T>({n, h, w, c});
  Tensor nhwcWeight = createCudaTensor<T>({k, r, s, groupC});
  Tensor nhwcOutput = createCudaTensor<T>({n, outH, outW, k});

  transpose<T>(getDataPtrCuda<T>(input), getDataPtrCuda<T>(nhwcInput), n, c, h * w);
  transpose<T>(getDataPtrCuda<T>(weight), getDataPtrCuda<T>(nhwcWeight), k, groupC, r * s);

  const T *activation = getDataPtrCuda<T>(nhwcInput);
  const T *filter = getDataPtrCuda<T>(nhwcWeight);
  T *out = getDataPtrCuda<T>(nhwcOutput);

  // One problem for one group of the convolution, or for all of it when there is one group.
  auto problemOf = [&](int problemC, int problemK, int problemGroups) {
    return cc::Conv2dProblemSize(
        {n, h, w, problemC},
        {problemK, r, s, problemC / problemGroups},
        {g.padH, g.padH, g.padW, g.padW},
        {g.strideH, g.strideW},
        {g.dilationH, g.dilationW},
        {n, outH, outW, problemK},
        cc::Mode::kCrossCorrelation,
        1,
        problemGroups);
  };

  bool isDepthwise = g.groups > 1 && g.groups == c && g.groups == k;
  if (g.groups == 1) {
    cc::Conv2dProblemSize problem = problemOf(c, k, 1);
    if constexpr (std::is_same<T, float>::value) {
      convolvePacked<FloatAny, T>(activation, filter, out, problem);
    } else if (c % 8 != 0 || k % 8 != 0) {
      convolvePacked<HalfNarrow, T>(activation, filter, out, problem);
    } else {
      convolvePacked<HalfAligned, T>(activation, filter, out, problem);
    }
  } else if (isDepthwise) {
    using Kernel = Depthwise<typename CutlassType<T>::Type>;
    convolvePacked<Kernel, T>(activation, filter, out, problemOf(c, k, g.groups));
  } else {
    // Each group is an ungrouped convolution of its own channels. The activation and the output
    // are the whole convolution's, seen through a layout that steps a whole pixel -- all C, or all
    // K, channels -- from one position to the next, and starting at the group's first channel.
    // The filter needs no such care: a group's output channels are a contiguous run of it. The
    // one channel at a time kernels are the ones that take any start and any stride.
    cc::Conv2dProblemSize problem = problemOf(groupC, groupK, 1);
    TensorNHWC activationLayout(c, w * c, h * w * c);
    TensorNHWC filterLayout = TensorNHWC::packed(problem.filter_extent());
    TensorNHWC outputLayout(k, outW * k, outH * outW * k);
    int64_t filterPerGroup = static_cast<int64_t>(groupK) * r * s * groupC;
    for (int group = 0; group < g.groups; ++group) {
      const T *groupActivation = activation + static_cast<int64_t>(group) * groupC;
      const T *groupFilter = filter + group * filterPerGroup;
      T *groupOut = out + static_cast<int64_t>(group) * groupK;
      if constexpr (std::is_same<T, float>::value) {
        convolve<FloatAny, T>(
            groupActivation,
            activationLayout,
            groupFilter,
            filterLayout,
            groupOut,
            outputLayout,
            problem);
      } else {
        convolve<HalfNarrow, T>(
            groupActivation,
            activationLayout,
            groupFilter,
            filterLayout,
            groupOut,
            outputLayout,
            problem);
      }
    }
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

  // Everything here is on the card, so the multiply and the join are the card's own operators'.
  Operators *ops = getOperators(Device::kCuda);

  // (K, C) by (C, H * W), which is (K, H * W): the answer, already in NCHW. One image at a time,
  // because the weight is the same for all of them and matmul does not broadcast a smaller left
  // operand over a batched right one.
  Tensor flatWeight = weight.view({k, c});
  Tensor result;
  for (int i = 0; i < n; ++i) {
    Tensor image = n == 1 ? input.view({c, h * w}) : input.subtensor(i).view({c, h * w});
    Tensor product = ops->matmul(flatWeight, image);
    result = result.empty() ? product : ops->cat(result, product, 0);
  }

  result = result.view({n, k, h, w});
  if (!bias.empty()) result = ops->add(result, bias.view({1, k, 1, 1}));

  return result;
}

/// `convNchw` for whichever of the two types `input` is in.
Tensor convNchwAnyType(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    const Geometry &g,
    const char *name) {
  if (input.getDType() == DType::kFloat16) return convNchw<half>(input, weight, bias, g, name);
  if (input.getDType() == DType::kFloat) return convNchw<float>(input, weight, bias, g, name);

  THROW(InvalidArg, lut::sprintf("%s takes a <half> or <float> input", name));
}

}  // namespace

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

  bool isPointwise = weight.getShape(2) == 1 && weight.getShape(3) == 1 && options.stride == 1 &&
                     options.padding == 0 && options.dilation == 1;
  if (isPointwise) return conv1x1(input, weight, bias);

  Geometry geometry{
      options.padding,
      options.padding,
      options.stride,
      options.stride,
      options.dilation,
      options.dilation,
      1};
  return convNchwAnyType(input, weight, bias, geometry, "conv2d");
}

Tensor conv1d(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    const Conv1dOptions &options) {
  if (input.getDim() != 3) THROW(InvalidArg, "conv1d takes a 3-D input, as (N, C, L)");
  if (weight.getDim() != 3) THROW(InvalidArg, "conv1d takes a 3-D weight, as (K, C / groups, R)");
  if (input.getDType() != weight.getDType()) {
    THROW(InvalidArg, "conv1d: the input and the weight are of different types");
  }
  if (options.stride < 1 || options.dilation < 1) {
    THROW(InvalidArg, "conv1d: the stride and the dilation are below one");
  }
  if (options.padding < 0) THROW(InvalidArg, "conv1d: the padding is negative");
  LL_CHECK_CONTIGUOUS(input);
  LL_CHECK_CONTIGUOUS(weight);

  int n = input.getShape(0);
  int c = input.getShape(1);
  int l = input.getShape(2);
  int k = weight.getShape(0);
  int r = weight.getShape(2);
  int groups = options.groups;
  if (groups < 1 || c % groups != 0 || k % groups != 0) {
    THROW(
        InvalidArg,
        lut::sprintf(
            "conv1d: %d groups do not divide %d input and %d output channels",
            groups,
            c,
            k));
  }
  if (weight.getShape(1) != c / groups) {
    THROW(
        InvalidArg,
        lut::sprintf(
            "conv1d: an input of %d channels in %d groups does not match a weight of %d",
            c,
            groups,
            weight.getShape(1)));
  }
  if (!bias.empty()) {
    if (bias.getNumEl() != k) {
      THROW(InvalidArg, "conv1d: the bias does not match the output channels");
    }
    if (bias.getDType() != input.getDType()) {
      THROW(InvalidArg, "conv1d: the bias and the input are of different types");
    }
  }

  // A signal is an image one row tall and a kernel one tap tall, and the height is neither padded
  // nor stepped over.
  Tensor image = input.view({n, c, 1, l});
  Tensor filter = weight.view({k, c / groups, 1, r});

  Tensor output;
  bool isPointwise = groups == 1 && r == 1 && options.stride == 1 && options.padding == 0;
  if (isPointwise) {
    output = conv1x1(image, filter, bias);
  } else {
    Geometry geometry{0, options.padding, 1, options.stride, 1, options.dilation, groups};
    output = convNchwAnyType(image, filter, bias, geometry, "conv1d");
  }

  return output.view({n, k, output.getShape(3)});
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
