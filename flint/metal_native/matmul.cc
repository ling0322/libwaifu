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

#include <string>
#include <vector>

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "flint/cpu/common.h"
#include "flint/metal_native/common.h"
#include "flint/metal_native/ops.h"

namespace fl {
namespace op {
namespace metal_native {

namespace {

// The tile of the output each threadgroup takes, and its threads: kernels/gemm.h's.
constexpr int kTile = 64;
constexpr uint32_t kGemmThreads = 128;

// Keep each of these in step with the struct of the same name in kernels/matmul.metal.

struct GemmParams {
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
  uint32_t vecA;
  uint32_t vecB;
  uint32_t addKind;
};

// The tile of the output each threadgroup of kernels/nax/matmul.metal takes, and its threads:
// four simdgroups.
constexpr int kNaxTileM = 64;
constexpr int kNaxTileN = 128;
constexpr uint32_t kNaxThreads = 128;

// And of kernels/nax/conv2d.metal: output channels by output pixels, and the rows of the
// reduction it gathers at a time.
constexpr int kNaxConvTileM = 64;
constexpr int kNaxConvTileN = 128;
constexpr int kNaxConvTileK = 32;

// In step with NaxGemmParams in kernels/nax/matmul.metal.
struct NaxGemmParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t lda;
  uint32_t ldb;
  uint32_t batch1;
  uint32_t strideA0;
  uint32_t strideA1;
  uint32_t strideB0;
  uint32_t strideB1;
  uint32_t addKind;
};

// In step with kVecAlongK and kVecAlongMN in kernels/matmul.metal.
constexpr uint32_t kVecAlongK = 1;
constexpr uint32_t kVecAlongMN = 2;

// In step with kAddNone, kAddColumns and kAddWhole in kernels/matmul.metal: what the kernels add to
// the product's float sum before they round it -- nothing, a value per column, or a contiguous
// tensor of the output's shape.
constexpr uint32_t kAddNone = 0;
constexpr uint32_t kAddColumns = 1;
constexpr uint32_t kAddWhole = 2;

// What is added to a product, as the kernels read it: `values` (N) for kAddColumns, and for
// kAddWhole, contiguous and of the output's shape.
struct Addend {
  TensorView values;
  uint32_t kind = kAddNone;

  // This addend for the leading slice `i` of the output.
  Addend subtensor(int i) const {
    return kind == kAddWhole ? Addend{values.subtensor(i), kind} : *this;
  }
};

struct ConvParams {
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
  uint32_t hasBias;
  uint32_t pointwise;
  uint32_t vecWeight;
  uint32_t vecInput;
};

uint32_t ceilDiv(int64_t a, int64_t b) {
  return static_cast<uint32_t>((a + b - 1) / b);
}

// Whether an operand may be read four elements at a time, and along which dimension: the one of
// unit stride, `strideK` along K and `strideMN` along M or N, so long as everything else -- the
// other stride, the batch strides and where the operand starts -- is a whole number of fours.
uint32_t vectorLayout(
    const TensorView &operand,
    uint32_t strideK,
    uint32_t strideMN,
    uint32_t batchStride0,
    uint32_t batchStride1) {
  auto fours = [](int64_t x) { return x % 4 == 0; };
  if (!fours(operand.getInternalOffset()) || !fours(batchStride0) || !fours(batchStride1)) {
    return 0;
  }
  if (strideK == 1 && fours(strideMN)) return kVecAlongK;
  if (strideMN == 1 && fours(strideK)) return kVecAlongMN;
  return 0;
}

// What a kernel's name gains for its output: nothing where it is the operands' type, "_f32" where
// a product of half is kept in float.
std::string outputSuffix(const TensorView &a, const TensorView &output) {
  return output.getDType() == a.getDType() ? "" : "_f32";
}

// The product on the neural accelerators, where there are any and the operands are laid out for
// them: A along K, B along N or along K. Returns false, having done nothing, when not.
bool naxGemm(
    const TensorView &a,
    const TensorView &b,
    const GemmParams &gemm,
    uint32_t numBatches,
    const Addend &addend,
    const TensorView &output) {
  if (!getContext(a)->hasNax() || gemm.strideAk != 1) return false;

  bool alongN = gemm.strideBn == 1;
  bool alongK = gemm.strideBk == 1;
  if (!alongN && !alongK) return false;

  NaxGemmParams params{};
  params.M = gemm.M;
  params.N = gemm.N;
  params.K = gemm.K;
  params.lda = gemm.strideAm;
  params.ldb = alongN ? gemm.strideBk : gemm.strideBn;
  params.batch1 = gemm.batch1;
  params.strideA0 = gemm.strideA0;
  params.strideA1 = gemm.strideA1;
  params.strideB0 = gemm.strideB0;
  params.strideB1 = gemm.strideB1;
  params.addKind = addend.kind;

  // With an addend, the kernels that hold the sum to add it before they round it; without, the ones
  // that have matmul2d write the product itself.
  bool adds = addend.kind != kAddNone;
  const char *variant = adds ? (alongN ? "nax_matmul_add_nn" : "nax_matmul_add_nt")
                             : (alongN ? "nax_matmul_nn" : "nax_matmul_nt");
  std::string kernel = kernelName(variant, a.getDType()) + outputSuffix(a, output);
  uint32_t groupsX = ceilDiv(gemm.N, kNaxTileN);
  uint32_t groupsY = ceilDiv(gemm.M, kNaxTileM);
  if (adds) {
    getContext(a)->dispatchGroups(
        kernel.c_str(),
        {getArg(output), getArg(a), getArg(b), getArg(addend.values)},
        &params,
        sizeof(params),
        groupsX,
        groupsY,
        numBatches,
        kNaxThreads);
  } else {
    getContext(a)->dispatchGroups(
        kernel.c_str(),
        {getArg(output), getArg(a), getArg(b)},
        &params,
        sizeof(params),
        groupsX,
        groupsY,
        numBatches,
        kNaxThreads);
  }
  return true;
}

// Batched GEMM over batch dimensions that have already been collapsed to at most two.
// `output` is contiguous, each batch's (M, N) after the one before.
void gemm(
    const TensorView &a,
    const TensorView &b,
    const Layout &batch,
    const Addend &addend,
    const TensorView &output) {
  int M = a.getShape(-2);
  int K = a.getShape(-1);
  int N = b.getShape(-1);

  if (output.getNumEl() == 0) return;
  if (K == 0) {
    // A product over nothing is zero, and what is added to it is the answer.
    if (addend.kind == kAddNone) {
      fill(output, 0.0f);
    } else {
      std::vector<int> shape = output.getShape();
      copy(op::cpu::expandBatchDims(addend.values, shape).expand(shape), output);
    }
    return;
  }

  GemmParams params{};
  params.M = static_cast<uint32_t>(M);
  params.N = static_cast<uint32_t>(N);
  params.K = static_cast<uint32_t>(K);
  params.strideAm = static_cast<uint32_t>(a.getStride(-2));
  params.strideAk = static_cast<uint32_t>(a.getStride(-1));
  params.strideBk = static_cast<uint32_t>(b.getStride(-2));
  params.strideBn = static_cast<uint32_t>(b.getStride(-1));

  // The kernel takes the batch as (b0, b1), so a single batch dimension is b1 alone.
  uint32_t numBatches = 1;
  if (batch.ndim == 2) {
    params.batch1 = batch.shape[1];
    params.strideA0 = batch.strides[0][0];
    params.strideB0 = batch.strides[1][0];
    params.strideA1 = batch.strides[0][1];
    params.strideB1 = batch.strides[1][1];
    numBatches = batch.shape[0] * batch.shape[1];
  } else {
    params.batch1 = batch.shape[0];
    params.strideA1 = batch.strides[0][0];
    params.strideB1 = batch.strides[1][0];
    numBatches = batch.shape[0];
  }
  params.vecA = vectorLayout(a, params.strideAk, params.strideAm, params.strideA0, params.strideA1);
  params.vecB = vectorLayout(b, params.strideBk, params.strideBn, params.strideB0, params.strideB1);
  params.addKind = addend.kind;

  if (naxGemm(a, b, params, numBatches, addend, output)) return;

  // The output stands in for an addend that is not there; the kernel is told not to read it.
  const TensorView &added = addend.kind == kAddNone ? output : addend.values;
  getContext(a)->dispatchGroups(
      (kernelName("matmul", a.getDType()) + outputSuffix(a, output)).c_str(),
      {getArg(output), getArg(a), getArg(b), getArg(added)},
      &params,
      sizeof(params),
      ceilDiv(N, kTile),
      ceilDiv(M, kTile),
      numBatches,
      kGemmThreads);
}

// The same prefix of dimensions prepended to `b` that `a` has and it lacks, with a stride of zero.
TensorView expandBatch(const TensorView &b, const TensorView &a) {
  std::vector<TensorShape::Elem> elems;
  for (int d = 0; d < a.getDim() - b.getDim(); ++d) elems.push_back({a.getShape(d), 0});
  for (int d = 0; d < b.getDim(); ++d) elems.push_back({b.getShape(d), b.getStride(d)});
  return restride(b, lut::makeConstSpan(elems));
}

void conv(
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    int strideH,
    int strideW,
    int paddingH,
    int paddingW,
    int dilationH,
    int dilationW,
    int groups,
    const char *name,
    const TensorView &output) {
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

  checkOutput(output, {N, K, P, Q}, input.getDType(), name);
  if (output.getNumEl() == 0) return;

  Contiguous x(input);
  Contiguous w(weight);
  Contiguous b;
  if (!bias.empty()) {
    if (bias.getNumEl() != K || bias.getDType() != input.getDType()) {
      throw lut::InvalidArgError(lut::sprintf(
          "%s: the bias must be %d elements of the input's dtype",
          name,
          K));
    }
    b = Contiguous(bias);
  }

  ConvParams params{};
  params.channels = C;
  params.height = H;
  params.width = W;
  params.outChannels = K;
  params.outHeight = P;
  params.outWidth = Q;
  params.kernelHeight = R;
  params.kernelWidth = S;
  params.strideH = strideH;
  params.strideW = strideW;
  params.paddingH = paddingH;
  params.paddingW = paddingW;
  params.dilationH = dilationH;
  params.dilationW = dilationW;
  params.groups = groups;
  params.hasBias = b.empty() ? 0 : 1;

  // Four at a time where every row starts on a multiple of four elements.
  auto fours = [](int64_t n) { return n % 4 == 0; };
  int64_t reduction = static_cast<int64_t>(C / groups) * R * S;
  params.pointwise = R == 1 && S == 1 && strideH == 1 && strideW == 1 && paddingH == 0 &&
                     paddingW == 0;
  params.vecWeight = fours(reduction) && fours(w.getInternalOffset());
  params.vecInput = params.pointwise && fours(static_cast<int64_t>(H) * W) &&
                    fours(x.getInternalOffset());

  // On the neural accelerators where there are any, with the same parameters.
  if (getContext(x)->hasNax()) {
    // A gathered reduction that is not whole chunks has a kernel of its own, which ends on a
    // short one.
    const char *variant = "nax_conv2d";
    if (params.pointwise) {
      variant = "nax_conv2d_pointwise";
    } else if (reduction % kNaxConvTileK != 0) {
      variant = "nax_conv2d_short";
    }
    std::string kernel = kernelName(variant, x.getDType());
    getContext(x)->dispatchGroups(
        kernel.c_str(),
        {getArg(output), getArg(x), getArg(w), b.empty() ? getArg(x) : getArg(b)},
        &params,
        sizeof(params),
        ceilDiv(static_cast<int64_t>(P) * Q, kNaxConvTileN),
        ceilDiv(K / groups, kNaxConvTileM),
        static_cast<uint32_t>(N * groups),
        kNaxThreads);
    return;
  }

  // The input stands in for a bias that is not there; the kernel is told not to read it.
  getContext(x)->dispatchGroups(
      kernelName("conv2d", x.getDType()).c_str(),
      {getArg(output), getArg(x), getArg(w), b.empty() ? getArg(x) : getArg(b)},
      &params,
      sizeof(params),
      ceilDiv(static_cast<int64_t>(P) * Q, kTile),
      ceilDiv(K / groups, kTile),
      static_cast<uint32_t>(N * groups),
      kGemmThreads);
}

// The product, with `addend` added to it before it is rounded, into `out`, which matmul checked.
void product(
    const TensorView &a,
    const TensorView &b,
    const Addend &addend,
    const TensorView &out) {
  // A stack of rows times one matrix is one taller product, when the rows stack evenly.
  if (b.getDim() == 2 && a.getDim() > 2 && a.isContiguous()) {
    // The rows counted rather than inferred: a view cannot infer them from a K of zero.
    int rows = 1;
    for (int d = 0; d < a.getDim() - 1; ++d) rows *= a.getShape(d);
    int N = b.getShape(-1);
    TensorView flat = a.view({rows, a.getShape(-1)});
    Layout single{};
    single.ndim = 1;
    single.shape[0] = 1;
    Addend flatAddend = addend;
    if (addend.kind == kAddWhole) flatAddend.values = addend.values.view({rows, N});
    gemm(flat, b, single, flatAddend, out.view({rows, N}));
    return;
  }

  TensorView expanded = expandBatch(b, a);
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
    for (int i = 0; i < a.getShape(0); ++i) {
      product(a.subtensor(i), expanded.subtensor(i), addend.subtensor(i), out.subtensor(i));
    }
    return;
  }

  gemm(a, expanded, batch, addend, out);
}

}  // namespace

void matmul(const TensorView &a, const TensorView &b, const TensorView &out) {
  matmulAdd(a, b, TensorView(), out);
}

void matmulAdd(
    const TensorView &a,
    const TensorView &b,
    const TensorView &addend,
    const TensorView &out) {
  checkFloat(a.getDType(), "matmul");
  if (a.getDType() != b.getDType()) {
    throw lut::InvalidArgError(lut::sprintf(
        "matmul on the Metal device takes two of one dtype, not %s and %s",
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
  // Half times half may be kept in float, as the sum was: what a product half cannot hold needs.
  bool wide = a.getDType() == DType::kFloat16 && out.getDType() == DType::kFloat;
  checkOutput(out, shape, wide ? DType(DType::kFloat) : a.getDType(), "matmul");

  if (addend.empty()) return product(a, b, Addend{}, out);
  if (addend.getDType() != out.getDType() || addend.getDim() > out.getDim()) {
    throw lut::InvalidArgError(lut::sprintf(
        "matmul: an addend of %s %s does not fit an output of %s %s",
        addend.getDType().toString(),
        addend.getShapeString(),
        out.getDType().toString(),
        out.getShapeString()));
  }

  // One value for each column, however many dimensions it is written in, is a bias, which the
  // kernels read by column. Anything else is read element by element, as the output is laid out:
  // a residual of its shape where it is, and anything broadcast, copied out to that first.
  int N = shape.back();
  if (addend.getShape(-1) == N && addend.getNumEl() == N) {
    Contiguous bias(addend);
    product(a, b, Addend{bias.view({N}), kAddColumns}, out);
  } else {
    Contiguous whole(op::cpu::expandBatchDims(addend, shape).expand(shape));
    product(a, b, Addend{whole, kAddWhole}, out);
  }
}

void conv2d(
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    const TensorView &out) {
  if (input.getDim() != 4 || weight.getDim() != 4) {
    throw lut::InvalidArgError("conv2d takes an input (N, C, H, W) and a weight (K, C, R, S)");
  }
  conv(
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
      "conv2d",
      out);
}

void conv1d(
    const TensorView &input,
    const TensorView &weight,
    const TensorView &bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    const TensorView &out) {
  if (input.getDim() != 3 || weight.getDim() != 3) {
    throw lut::InvalidArgError("conv1d takes an input (N, C, L) and a weight (K, C, R)");
  }
  if (out.getDim() != 3) throw lut::InvalidArgError("conv1d writes (N, K, Lout)");

  conv(
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
      "conv1d",
      out.unsqueeze(2));
}

}  // namespace metal_native
}  // namespace op
}  // namespace fl
