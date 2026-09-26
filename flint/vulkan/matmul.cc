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

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "flint/vulkan/common.h"
#include "flint/vulkan/ops.h"

namespace fl {
namespace op {
namespace vulkan {

namespace {

// The tile of the output each workgroup takes: gemm_core.glsl's, and gemm_coopmat_core.glsl's.
constexpr int kTile = 64;
constexpr int kCoopmatTile = 128;

struct GemmPush {
  uint64_t c;
  uint64_t a;
  uint64_t b;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t strideAm;
  uint32_t strideAk;
  uint32_t strideBk;
  uint32_t strideBn;
  uint32_t batch1;
  uint32_t strideA0;
  uint32_t strideA1;
  uint32_t strideB0;
  uint32_t strideB1;
};

struct CoopmatGemmPush {
  uint64_t c;
  uint64_t a;
  uint64_t b;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t strideAm;
  uint32_t strideB;
  uint32_t alongN;
  uint32_t batch1;
  uint32_t strideA0;
  uint32_t strideA1;
  uint32_t strideB0;
  uint32_t strideB1;
};

struct ConvPush {
  uint64_t c;
  uint64_t input;
  uint64_t weight;
  uint64_t bias;
  uint32_t channels;
  uint32_t height;
  uint32_t width;
  uint32_t outChannels;
  uint32_t outHeight;
  uint32_t outWidth;
  uint32_t kernelHeight;
  uint32_t kernelWidth;
  uint32_t strideH;
  uint32_t strideW;
  int32_t paddingH;
  int32_t paddingW;
  uint32_t dilationH;
  uint32_t dilationW;
  uint32_t groups;
};

uint32_t ceilDiv(int64_t a, int64_t b) {
  return static_cast<uint32_t>((a + b - 1) / b);
}

// Whether the cooperative matrix kernels may take `tensor`: half precision, on a device that has
// them, and starting on sixteen bytes, which is what their eight-element loads need.
bool coopmatOperand(const Tensor &tensor) {
  return tensor.getDType() == DType::kFloat16 &&
         getContext(tensor)->getInfo().cooperativeMatrix && getAddress(tensor) % 16 == 0;
}

// The product on cooperative matrices, when the operands are laid out so that it can read them
// eight elements at a time: every stride but the unit one a multiple of eight, A along K and B
// along either. Returns false, having done nothing, when they are not.
bool coopmatGemm(
    const Tensor &a,
    const Tensor &b,
    const GemmPush &gemm,
    uint32_t numBatches,
    const Tensor &output) {
  if (!coopmatOperand(a) || !coopmatOperand(b)) return false;

  auto eighths = [](uint32_t stride) { return stride % 8 == 0; };
  bool aAlongK = gemm.strideAk == 1 && eighths(gemm.strideAm);
  bool bAlongK = gemm.strideBk == 1 && eighths(gemm.strideBn);
  bool bAlongN = gemm.strideBn == 1 && eighths(gemm.strideBk) && gemm.N % 8 == 0;
  bool batches = eighths(gemm.strideA0) && eighths(gemm.strideA1) && eighths(gemm.strideB0) &&
                 eighths(gemm.strideB1);
  if (gemm.K % 8 != 0 || !aAlongK || !(bAlongK || bAlongN) || !batches) return false;

  CoopmatGemmPush push{};
  push.c = gemm.c;
  push.a = gemm.a;
  push.b = gemm.b;
  push.M = gemm.M;
  push.N = gemm.N;
  push.K = gemm.K;
  push.strideAm = gemm.strideAm;
  push.alongN = bAlongK ? 0 : 1;
  push.strideB = bAlongK ? gemm.strideBn : gemm.strideBk;
  push.batch1 = gemm.batch1;
  push.strideA0 = gemm.strideA0;
  push.strideA1 = gemm.strideA1;
  push.strideB0 = gemm.strideB0;
  push.strideB1 = gemm.strideB1;

  getContext(output)->dispatch(
      "gemm_coopmat_f16",
      &push,
      sizeof(push),
      ceilDiv(gemm.N, kCoopmatTile),
      ceilDiv(gemm.M, kCoopmatTile),
      numBatches);
  return true;
}

// Batched GEMM over batch dimensions that have already been collapsed to at most two.
Tensor gemm(const Tensor &a, const Tensor &b, const Layout &batch, const std::vector<int> &shape) {
  int M = a.getShape(-2);
  int K = a.getShape(-1);
  int N = b.getShape(-1);

  Tensor output = createTensor(shape, a.getDType());
  if (output.getNumEl() == 0) return output;
  if (K == 0) {
    fill(output, 0.0f);
    return output;
  }

  GemmPush push{};
  push.c = getAddress(output);
  push.a = getAddress(a);
  push.b = getAddress(b);
  push.M = static_cast<uint32_t>(M);
  push.N = static_cast<uint32_t>(N);
  push.K = static_cast<uint32_t>(K);
  push.strideAm = static_cast<uint32_t>(a.getStride(-2));
  push.strideAk = static_cast<uint32_t>(a.getStride(-1));
  push.strideBk = static_cast<uint32_t>(b.getStride(-2));
  push.strideBn = static_cast<uint32_t>(b.getStride(-1));

  // The kernel takes the batch as (b0, b1), so a single batch dimension is b1 alone.
  uint32_t numBatches = 1;
  if (batch.ndim == 2) {
    push.batch1 = batch.shape[1];
    push.strideA0 = batch.strides[0][0];
    push.strideB0 = batch.strides[1][0];
    push.strideA1 = batch.strides[0][1];
    push.strideB1 = batch.strides[1][1];
    numBatches = batch.shape[0] * batch.shape[1];
  } else {
    push.batch1 = batch.shape[0];
    push.strideA1 = batch.strides[0][0];
    push.strideB1 = batch.strides[1][0];
    numBatches = batch.shape[0];
  }

  if (coopmatGemm(a, b, push, numBatches, output)) return output;

  getContext(a)->dispatch(
      kernelName("gemm", a.getDType()).c_str(),
      &push,
      sizeof(push),
      ceilDiv(N, kTile),
      ceilDiv(M, kTile),
      numBatches);
  return output;
}

// The same prefix of dimensions prepended to `b` that `a` has and it lacks, with a stride of zero.
Tensor expandBatch(const Tensor &b, const Tensor &a) {
  std::vector<TensorShape::Elem> elems;
  for (int d = 0; d < a.getDim() - b.getDim(); ++d) elems.push_back({a.getShape(d), 0});
  for (int d = 0; d < b.getDim(); ++d) elems.push_back({b.getShape(d), b.getStride(d)});
  return Tensor::create(
      std::make_shared<TensorShape>(lut::makeConstSpan(elems)),
      b.getInternalData(),
      b.getInternalOffset());
}

// (N, C, H, W) as (N, H, W, C), contiguous: channels last, the layout conv2d_coopmat.comp reads
// eight channels at a time from. The same permutation takes a weight (K, C, R, S) to (K, R, S, C).
Tensor channelsLast(const Tensor &tensor) {
  std::vector<TensorShape::Elem> elems = {
      {tensor.getShape(0), tensor.getStride(0)},
      {tensor.getShape(2), tensor.getStride(2)},
      {tensor.getShape(3), tensor.getStride(3)},
      {tensor.getShape(1), tensor.getStride(1)}};
  Tensor permuted = Tensor::create(
      std::make_shared<TensorShape>(lut::makeConstSpan(elems)),
      tensor.getInternalData(),
      tensor.getInternalOffset());
  return makeContiguous(permuted);
}

Tensor conv(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    int strideH,
    int strideW,
    int paddingH,
    int paddingW,
    int dilationH,
    int dilationW,
    int groups,
    const char *name) {
  checkFloat(input.getDType(), name);
  if (weight.getDType() != input.getDType()) {
    throw lut::InvalidArgError(lut::sprintf("%s: the weight and input differ in dtype", name));
  }
  if (strideH < 1 || strideW < 1 || dilationH < 1 || dilationW < 1 || paddingH < 0 ||
      paddingW < 0) {
    throw lut::InvalidArgError(lut::sprintf("%s: invalid stride, padding or dilation", name));
  }

  int N = input.getShape(0);
  int C = input.getShape(1);
  int H = input.getShape(2);
  int W = input.getShape(3);
  int K = weight.getShape(0);
  int R = weight.getShape(2);
  int S = weight.getShape(3);
  if (groups < 1 || C % groups != 0 || K % groups != 0 || weight.getShape(1) != C / groups) {
    throw lut::InvalidArgError(lut::sprintf(
        "%s: a weight of %s does not fit %d channels in %d groups",
        name,
        weight.getShapeString(),
        C,
        groups));
  }

  int P = (H + 2 * paddingH - dilationH * (R - 1) - 1) / strideH + 1;
  int Q = (W + 2 * paddingW - dilationW * (S - 1) - 1) / strideW + 1;
  if (P <= 0 || Q <= 0) {
    throw lut::InvalidArgError(lut::sprintf("%s: the kernel is larger than the input", name));
  }

  // Half precision on a device with cooperative matrices goes channels last first, which costs a
  // copy of the input and the weight and is what lets the product read them eight at a time.
  bool coopmat = input.getDType() == DType::kFloat16 &&
                 getContext(input)->getInfo().cooperativeMatrix && (C / groups) % 8 == 0;
  Tensor x = coopmat ? channelsLast(input) : makeContiguous(input);
  Tensor w = coopmat ? channelsLast(weight) : makeContiguous(weight);
  Tensor keepBias;
  uint64_t biasAddress = 0;
  if (!bias.empty()) {
    if (bias.getNumEl() != K || bias.getDType() != input.getDType()) {
      throw lut::InvalidArgError(lut::sprintf(
          "%s: the bias must be %d elements of the input's dtype",
          name,
          K));
    }
    keepBias = makeContiguous(bias);
    biasAddress = getAddress(keepBias);
  }

  Tensor output = createTensor({N, K, P, Q}, input.getDType());
  ConvPush push{};
  push.c = getAddress(output);
  push.input = getAddress(x);
  push.weight = getAddress(w);
  push.bias = biasAddress;
  push.channels = C;
  push.height = H;
  push.width = W;
  push.outChannels = K;
  push.outHeight = P;
  push.outWidth = Q;
  push.kernelHeight = R;
  push.kernelWidth = S;
  push.strideH = strideH;
  push.strideW = strideW;
  push.paddingH = paddingH;
  push.paddingW = paddingW;
  push.dilationH = dilationH;
  push.dilationW = dilationW;
  push.groups = groups;

  int tile = coopmat ? kCoopmatTile : kTile;
  std::string kernel = coopmat ? "conv2d_coopmat_f16" : kernelName("conv2d", x.getDType());
  getContext(x)->dispatch(
      kernel.c_str(),
      &push,
      sizeof(push),
      ceilDiv(K / groups, tile),
      ceilDiv(static_cast<int64_t>(P) * Q, tile),
      static_cast<uint32_t>(N * groups));
  return output;
}

}  // namespace

Tensor matmul(const Tensor &a, const Tensor &b) {
  checkFloat(a.getDType(), "matmul");
  if (a.getDType() != b.getDType()) {
    throw lut::InvalidArgError(lut::sprintf(
        "matmul on the Vulkan device takes two of one dtype, not %s and %s",
        a.getDType().toString(),
        b.getDType().toString()));
  }
  if (a.getDim() < 2 || b.getDim() < 2 || a.getDim() < b.getDim()) {
    throw lut::InvalidArgError(lut::sprintf(
        "matmul: unable to multiply %s by %s",
        a.getShapeString(),
        b.getShapeString()));
  }
  if (a.getShape(-1) != b.getShape(-2)) {
    throw lut::InvalidArgError(lut::sprintf(
        "matmul: %s and %s do not agree on the inner dimension",
        a.getShapeString(),
        b.getShapeString()));
  }

  std::vector<int> shape = a.getShape();
  shape.back() = b.getShape(-1);

  // A stack of rows times one matrix is one taller product, when the rows stack evenly.
  if (b.getDim() == 2 && a.getDim() > 2 && a.isContiguous()) {
    Tensor flat = a.view({-1, a.getShape(-1)});
    Layout single{};
    single.ndim = 1;
    single.shape[0] = 1;
    return gemm(flat, b, single, {flat.getShape(0), b.getShape(-1)}).view(shape);
  }

  Tensor expanded = expandBatch(b, a);
  std::vector<int> batchShape;
  std::vector<int> stridesA;
  std::vector<int> stridesB;
  for (int d = 0; d < a.getDim() - 2; ++d) {
    if (expanded.getShape(d) != a.getShape(d)) {
      throw lut::InvalidArgError(lut::sprintf(
          "matmul: the batch of %s does not match %s",
          b.getShapeString(),
          a.getShapeString()));
    }
    batchShape.push_back(a.getShape(d));
    stridesA.push_back(a.getStride(d));
    stridesB.push_back(expanded.getStride(d));
  }

  Layout batch;
  if (!collapse(batchShape, {stridesA, stridesB}, &batch) || batch.ndim > 2) {
    // More batch dimensions than the kernel walks, which do not merge: one leading slice at a
    // time into the rows of the answer.
    Tensor output = createTensor(shape, a.getDType());
    for (int i = 0; i < a.getShape(0); ++i) {
      copy(matmul(a.subtensor(i), expanded.subtensor(i)), output.subtensor(i));
    }
    return output;
  }

  return gemm(a, expanded, batch, shape);
}

Tensor conv2d(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
  if (input.getDim() != 4 || weight.getDim() != 4) {
    throw lut::InvalidArgError("conv2d takes an input (N, C, H, W) and a weight (K, C, R, S)");
  }
  return conv(
      input,
      weight,
      bias,
      stride,
      stride,
      padding,
      padding,
      dilation,
      dilation,
      groups,
      "conv2d");
}

Tensor conv1d(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
  if (input.getDim() != 3 || weight.getDim() != 3) {
    throw lut::InvalidArgError("conv1d takes an input (N, C, L) and a weight (K, C, R)");
  }

  Tensor output = conv(
      input.unsqueeze(2),
      weight.unsqueeze(2),
      bias,
      1,
      stride,
      0,
      padding,
      1,
      dilation,
      groups,
      "conv1d");
  return output.squeeze(2);
}

}  // namespace vulkan
}  // namespace op
}  // namespace fl
