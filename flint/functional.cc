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

#include "flint/functional.h"

#include <math.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "lutil/error.h"
#include "lutil/strings.h"
#include "flint/cpu/cpu_tensor_data.h"
#include "flint/operators.h"
#ifdef LIBWAIFU_CUDA_ENABLED
#include "flint/cuda/cuda_host_tensor_data.h"
#include "flint/cuda/cuda_tensor_data.h"
#endif
#ifdef LIBWAIFU_METAL_ENABLED
#include "flint/metal_native/metal_tensor_data.h"
#endif
#ifdef LIBWAIFU_VULKAN_ENABLED
#include "flint/vulkan/vulkan_tensor_data.h"
#endif

namespace fl {
namespace F {

namespace {

int64_t numelOf(lut::Span<const int> shape) {
  int64_t numel = 1;
  for (int size : shape) {
    if (size < 0) THROW(InvalidArg, lut::sprintf("a tensor cannot have a dimension of %d", size));
    numel *= size;
  }
  return numel;
}

/// `input` <(B, KVH, L, D)> with each key-value head repeated for the query heads that share it,
/// so that grouped-query attention can be done as plain attention.
Tensor expandKeyValueHeads(Operators *op, TensorView input, int numHeads) {
  int batchSize = input.getShape(0);
  int numKeyValueHeads = input.getShape(1);
  int length = input.getShape(2);
  int headDim = input.getShape(3);
  int groupSize = numHeads / numKeyValueHeads;

  TensorView expanded =
      input.unsqueeze(2).expand({batchSize, numKeyValueHeads, groupSize, length, headDim});
  Tensor output = emptyLike(expanded);
  op->copy(expanded, output);

  return output.view({batchSize, numHeads, length, headDim});
}

/// `A` (..., m, k) times `B` (..., k, n), into a fresh tensor of `A`'s type on its device.
Tensor product(Operators *op, TensorView A, TensorView B) {
  std::vector<int> shape = A.getShape();
  shape.back() = B.getShape(-1);
  Tensor out = empty(A.getDevice(), shape, A.getDType());
  op->matmul(A, B, out);
  return out;
}

/// `input` times `scale`, into a fresh tensor like it.
Tensor scaled(Operators *op, TensorView input, float scale) {
  Tensor out = emptyLike(input);
  op->mul(input, scale, out);
  return out;
}

/// How many score elements one block of queries may hold at once, counting every head and every
/// sequence in the batch -- the whole tensor the two matmuls hand each other, not one head's
/// share of it. 128M of them is 256MB in half, and the softmax needs a second copy of it.
///
/// This is a bound on memory and nothing else. Blocking was measured on every shape SDXL runs and
/// a bigger block was faster every time, cache or no cache: the score matrix is streamed once
/// either way, and what the block size really moves is which kernel cuBLAS picks for the second
/// matmul, which is not something to steer by. So the budget is set where it stops an allocation
/// nobody meant to make, and left clear of everything that runs.
constexpr int64_t kAttentionScoreLimit = 128 * 1024 * 1024;

}  // namespace

// --- Allocation -----------------------------------------------------------------------------

std::unique_ptr<TensorData> allocate(Device device, int64_t numel, DType dtype, AllocHint hint) {
  if (numel > TensorData::MaxNumEl) {
    THROW(InvalidArg, lut::sprintf("a tensor of %d elements is too large", numel));
  }
  numel = std::max<int64_t>(numel, 1);

  switch (device.getType()) {
    case Device::kCpu:
      return op::cpu::CpuTensorData::create(numel, dtype);
#ifdef LIBWAIFU_CUDA_ENABLED
    case Device::kCuda:
      return op::cuda::CudaTensorData::create(numel, dtype, 0, hint);
    case Device::kCudaHost:
      return op::cuda::CudaHostTensorData::create(numel, dtype);
#endif
#ifdef LIBWAIFU_METAL_ENABLED
    case Device::kMetal:
      return op::metal_native::MetalTensorData::create(numel, dtype, hint);
#endif
#ifdef LIBWAIFU_VULKAN_ENABLED
    case Device::kVulkan:
      return op::vulkan::VulkanTensorData::create(numel, dtype);
#endif
    default:
      THROW(
          NotImplemented,
          lut::sprintf("this build cannot allocate on the %s device", device.getName()));
  }
}

Tensor empty(Device device, lut::Span<const int> shape, DType dtype) {
  int64_t numel = numelOf(shape);
  return Tensor::create(std::make_shared<TensorShape>(shape), allocate(device, numel, dtype));
}

Tensor emptyLike(const TensorView &input) {
  std::vector<int> shape = input.getShape();
  return empty(input.getDevice(), shape, input.getDType());
}

// --- Attention ------------------------------------------------------------------------------

void composedAttention(
    Operators *op,
    TensorView q,
    TensorView k,
    TensorView v,
    bool causal,
    TensorView out) {
  int numHeads = q.getShape(1);
  int numKeyValueHeads = k.getShape(1);
  int queryLength = q.getShape(2);
  int keyValueLength = k.getShape(2);
  int headDim = q.getShape(3);

  Tensor expandedK, expandedV;
  if (numHeads != numKeyValueHeads) {
    expandedK = expandKeyValueHeads(op, k, numHeads);
    expandedV = expandKeyValueHeads(op, v, numHeads);
    k = expandedK;
    v = expandedV;
  }

  // Scaling both q and k keeps the scores in range for half precision.
  float scale = sqrtf(1.0f / sqrtf(1.0f * headDim));
  Tensor scaledK = scaled(op, k, scale).transpose(-2, -1);

  // The score matrix is the whole cost of doing it this way: a VAE decoding a 1024 by 1024 image
  // attends over 16384 positions, and holding all of that at once is half a gigabyte before the
  // softmax needs a second copy. Since each output row depends only on its own row of scores, the
  // queries are taken a block at a time instead once that gets out of hand. The answer is the
  // same to the bit -- no running maximum is needed, because a block still sees every key.
  //
  // What a row of queries costs is a row of scores in every head of every sequence, so that is
  // what the budget is divided by.
  int64_t scoresPerQuery = static_cast<int64_t>(q.getShape(0)) * numHeads * keyValueLength;

  // What the budget affords is rounded down to a power of two rather than taken where the
  // division landed. The second matmul takes its kernel from the number of query rows handed to
  // it, and the choice is not monotone in that number: 640 rows of scores against the values
  // measured 837us where 512 measured 219, for a quarter more work. The powers of two were
  // uniformly among the good ones.
  int blockSize = queryLength;
  if (scoresPerQuery * queryLength > kAttentionScoreLimit) {
    int64_t budget = kAttentionScoreLimit / scoresPerQuery;
    blockSize = 1;
    while (blockSize * 2 <= budget) blockSize *= 2;
  }

  // The probabilities of one block of queries, against every key.
  auto probabilities = [&](TensorView blockQ, int begin, int end) {
    Tensor scores = product(op, scaled(op, blockQ, scale), scaledK);
    // A single query attends to the whole history, so it needs no mask. A block of them is masked
    // against where it sits, not where the whole query is.
    if (causal && queryLength > 1) {
      Tensor fullMask = empty(
          q.getDevice(), {keyValueLength, keyValueLength}, op->getDefaultFloatType());
      op->causalMask(fullMask);
      TensorView mask = TensorView(fullMask).slice(
          0, {keyValueLength - queryLength + begin, keyValueLength - queryLength + end});

      // The mask comes in the device's *default* float type, which is not always the type the
      // scores are in: on aarch64 that default is half, and a model running in float32 has
      // float32 scores.
      Tensor typedMask;
      if (mask.getDType() != scores.getDType()) {
        typedMask = empty(q.getDevice(), mask.getShape(), scores.getDType());
        op->cast(mask, typedMask);
        mask = typedMask;
      }

      // One mask for every sequence and head: seen at the scores' shape by repeating it.
      while (mask.getDim() < scores.getDim()) mask = mask.unsqueeze(0);
      std::vector<int> scoreShape = scores.getShape();
      Tensor masked = emptyLike(scores);
      op->add(scores, mask.expand(scoreShape), masked);
      scores = masked;
    }
    Tensor probabilities = emptyLike(scores);
    op->softmax(scores, probabilities);
    return probabilities;
  };

  if (blockSize >= queryLength) {
    op->matmul(probabilities(q, 0, queryLength), v, out);
    return;
  }

  // Each block's answer is written into the rows of `out` it belongs to. Joining the blocks as they
  // arrive would copy everything finished so far on every pass, which is quadratic in the number
  // of blocks: the VAE's 64 of them spent 1.7ms of an 18ms call doing nothing else.
  for (int begin = 0; begin < queryLength; begin += blockSize) {
    int end = std::min(begin + blockSize, queryLength);
    Tensor blockOutput = product(op, probabilities(q.slice(-2, {begin, end}), begin, end), v);
    op->copy(blockOutput, out.slice(-2, {begin, end}));
  }
}

// --- Signal ---------------------------------------------------------------------------------

namespace {

/// `values`, a row-major float array of `shape` on the host, as a tensor of `dtype` on `device`.
Tensor hostConstant(
    Operators *op,
    Device device,
    DType dtype,
    lut::Span<const int> shape,
    const std::vector<float> &values) {
  Tensor host = empty(Device(Device::kCpu), shape, DType::kFloat);
  CHECK(host.getNumEl() == static_cast<int64_t>(values.size()));
  std::copy(
      values.begin(),
      values.end(),
      host.getInternalData()->getData<float>(host.getInternalOffset()));

  Tensor moved = host;
  if (device.getType() != Device::kCpu) {
    moved = empty(device, shape, DType::kFloat);
    op->transfer(host, moved);
  }
  if (dtype == DType::kFloat) return moved;

  Tensor cast = empty(device, shape, dtype);
  op->cast(moved, cast);
  return cast;
}

/// The elements of `input`, on whatever device, as floats on the host.
std::vector<float> readFloats(Operators *op, TensorView input) {
  Tensor asFloat = empty(input.getDevice(), input.getShape(), DType::kFloat);
  if (input.getDType() == DType::kFloat) {
    op->copy(input, asFloat);
  } else {
    op->cast(input, asFloat);
  }

  Tensor host = asFloat;
  if (input.getDevice().getType() != Device::kCpu) {
    host = empty(Device(Device::kCpu), input.getShape(), DType::kFloat);
    op->transfer(asFloat, host);
    op->synchronize();
  }

  const float *data = host.getInternalData()->getData<float>(host.getInternalOffset());
  return std::vector<float>(data, data + host.getNumEl());
}

/// `value` everywhere in `shape`, as one element of `dtype` on `device` read through an expanded
/// view -- what a binary operator takes as its right operand.
Tensor scalar(Operators *op, Device device, DType dtype, float value, lut::Span<const int> shape) {
  Tensor one = empty(device, {1}, dtype);
  op->fill(one, value);
  return one.view(std::vector<int>(shape.size(), 1)).expand(shape);
}

/// A contiguous tensor of `shape` and `dtype` on `device`, all zeros.
Tensor zeros(Operators *op, Device device, lut::Span<const int> shape, DType dtype) {
  Tensor out = empty(device, shape, dtype);
  op->fill(out, 0.0f);
  return out;
}

/// `shape` with its last dimension replaced by `size`.
std::vector<int> withLast(std::vector<int> shape, int size) {
  shape.back() = size;
  return shape;
}

constexpr double kPi = 3.14159265358979323846;

/// The bank `composedStft` convolves by, (2 * bins, 1, nFft): every bin's windowed cosine, then
/// every bin's windowed negative sine. With `inverse` it is the bank `composedIstft` overlap-adds
/// by instead: scaled by 1 / nFft, and doubled for every bin but the two that are their own
/// conjugate, which is what recovers the half of the spectrum the forward transform did not keep.
std::vector<float> sinusoidBank(const std::vector<float> &window, bool inverse) {
  int nFft = static_cast<int>(window.size());
  int bins = nFft / 2 + 1;
  std::vector<float> values(2 * bins * nFft);

  for (int bin = 0; bin < bins; ++bin) {
    double scale = 1.0;
    if (inverse) {
      bool ownConjugate = bin == 0 || (nFft % 2 == 0 && bin == nFft / 2);
      scale = (ownConjugate ? 1.0 : 2.0) / nFft;
    }

    for (int n = 0; n < nFft; ++n) {
      double phase = 2.0 * kPi * bin * n / nFft;
      double w = window[n];
      values[bin * nFft + n] = static_cast<float>(scale * w * cos(phase));
      values[(bins + bin) * nFft + n] = static_cast<float>(-scale * w * sin(phase));
    }
  }

  return values;
}

}  // namespace

int checkConvTranspose1d(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int outputPadding,
    int groups,
    TensorView out) {
  if (input.getDim() != 3 || weight.getDim() != 3) {
    THROW(InvalidArg, "convTranspose1d: the input and the weight must be 3-D");
  }
  if (stride < 1 || padding < 0 || outputPadding < 0 || groups < 1) {
    THROW(InvalidArg, "convTranspose1d: a stride, padding or group count out of range");
  }
  if (outputPadding >= stride) {
    THROW(InvalidArg, "convTranspose1d: an output padding is less than the stride");
  }

  int channels = input.getShape(1);
  if (channels % groups != 0 || weight.getShape(0) != channels) {
    THROW(InvalidArg, "convTranspose1d: the weight does not match the input's channels");
  }
  int outChannels = weight.getShape(1) * groups;
  if (!bias.empty() && (bias.getDim() != 1 || bias.getShape(0) != outChannels)) {
    THROW(InvalidArg, "convTranspose1d: the bias is not one per output channel");
  }
  if (bias.getDType() != input.getDType() && !bias.empty()) {
    THROW(InvalidArg, "convTranspose1d: the bias is not of the input's type");
  }
  if (weight.getDType() != input.getDType() || out.getDType() != input.getDType()) {
    THROW(InvalidArg, "convTranspose1d: the weight and out are not of the input's type");
  }

  int length = input.getShape(2);
  int outLength = (length - 1) * stride - 2 * padding + weight.getShape(2) + outputPadding;
  if (outLength < 1) THROW(InvalidArg, "convTranspose1d: the output would be empty");
  out.throwIfInvalidShape({input.getShape(0), outChannels, outLength}, "convTranspose1d");

  return outLength;
}

Tensor transposedConvolutionColumns(
    Operators *op,
    TensorView input,
    TensorView weight,
    int groups,
    int group) {
  int batch = input.getShape(0);
  int groupIn = input.getShape(1) / groups;
  int length = input.getShape(2);
  int groupOut = weight.getShape(1);
  int kernel = weight.getShape(2);
  DType dtype = input.getDType();
  Device device = input.getDevice();

  Tensor rows = empty(device, {batch, length, groupIn}, dtype);
  op->copy(input.slice(1, {group * groupIn, (group + 1) * groupIn}).transpose(1, 2), rows);

  // A group of a contiguous weight is a contiguous run of it, which is the case every model has;
  // anything else is copied into one first.
  TensorView w = weight.slice(0, {group * groupIn, (group + 1) * groupIn});
  Tensor contiguousWeight;
  if (!w.isContiguous()) {
    contiguousWeight = empty(device, {groupIn, groupOut, kernel}, dtype);
    op->copy(w, contiguousWeight);
    w = contiguousWeight;
  }

  Tensor columns = empty(device, {batch * length, groupOut * kernel}, dtype);
  op->matmul(rows.view({batch * length, groupIn}), w.view({groupIn, groupOut * kernel}), columns);
  return columns;
}

void composedConvTranspose1d(
    Operators *op,
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int outputPadding,
    int groups,
    TensorView out) {
  int outLength =
      checkConvTranspose1d(input, weight, bias, stride, padding, outputPadding, groups, out);

  int batch = input.getShape(0);
  int channels = input.getShape(1);
  int length = input.getShape(2);
  int groupOut = weight.getShape(1);
  int kernel = weight.getShape(2);
  int groupIn = channels / groups;

  DType dtype = input.getDType();
  Device device = input.getDevice();

  // The kernel rounded up to a whole number of strides, so that it cuts evenly into pieces. The
  // zeros it is padded with contribute nothing. Every piece lands `length * stride` positions,
  // shifted by its own index of strides; one more stride of room covers an output padding.
  int pieces = (kernel + stride - 1) / stride;
  int padded = pieces * stride;
  int fullLength = (length + pieces) * stride;

  for (int g = 0; g < groups; ++g) {
    TensorView x = input.slice(1, {g * groupIn, (g + 1) * groupIn});
    TensorView w = weight.slice(0, {g * groupIn, (g + 1) * groupIn});

    // (N * L, C) @ (C, K * padded): every contribution of every input position.
    Tensor rows = empty(device, {batch, length, groupIn}, dtype);
    op->copy(x.transpose(1, 2), rows);
    Tensor paddedWeight = zeros(op, device, {groupIn, groupOut, padded}, dtype);
    op->copy(w, paddedWeight.slice(2, {0, kernel}));

    Tensor columns = empty(device, {batch * length, groupOut * padded}, dtype);
    op->matmul(
        rows.view({batch * length, groupIn}),
        paddedWeight.view({groupIn, groupOut * padded}),
        columns);
    TensorView split = columns.view({batch, length, groupOut, pieces, stride});

    Tensor total = zeros(op, device, {batch, groupOut, fullLength}, dtype);
    Tensor shifted = empty(device, {batch, groupOut, fullLength}, dtype);
    Tensor part = empty(device, {batch, groupOut, length, 1, stride}, dtype);
    for (int piece = 0; piece < pieces; ++piece) {
      // (N, L, K, 1, stride) -> (N, K, L, 1, stride) -> (N, K, L * stride), at its shift.
      op->copy(split.slice(3, {piece, piece + 1}).transpose(1, 2), part);
      op->fill(shifted, 0.0f);
      op->copy(
          part.view({batch, groupOut, length * stride}),
          shifted.slice(2, {piece * stride, (piece + length) * stride}));
      op->add(total, shifted, total);
    }

    op->copy(
        total.slice(2, {padding, padding + outLength}),
        out.slice(1, {g * groupOut, (g + 1) * groupOut}));
  }

  if (!bias.empty()) {
    op->add(out, bias.view({1, groupOut * groups, 1}).expand(out.getShape()), out);
  }
}

void composedSnake(
    Operators *op,
    TensorView input,
    TensorView alpha,
    TensorView beta,
    float eps,
    TensorView out) {
  if (input.getDim() != 3) THROW(InvalidArg, "snake: the input must be (N, C, L)");
  int channels = input.getShape(1);
  if (alpha.getNumEl() != channels || (!beta.empty() && beta.getNumEl() != channels)) {
    THROW(InvalidArg, "snake: alpha and beta are one per channel");
  }

  DType dtype = input.getDType();
  Device device = input.getDevice();
  std::vector<int> shape = input.getShape();
  auto perChannel = [&](TensorView parameter) {
    return parameter.view({1, channels, 1}).expand(shape);
  };

  // The epsilon is added to the divisor rather than guarding the division, which is what the
  // reference does, for the channel whose parameter has decayed to zero.
  TensorView divisorOf = beta.empty() ? alpha : beta;
  Tensor divisor = empty(device, {channels}, dtype);
  op->add(divisorOf.view({channels}), scalar(op, device, dtype, eps, {channels}), divisor);

  Tensor a = emptyLike(input);
  Tensor b = emptyLike(input);
  op->mul(input, perChannel(alpha), a);
  op->sin(a, b);
  op->square(b, a);
  op->divTensor(a, perChannel(divisor), b);
  op->add(input, b, out);
}

void composedPad1d(
    Operators *op,
    TensorView input,
    int left,
    int right,
    PadMode mode,
    TensorView out) {
  if (input.getDim() < 1) THROW(InvalidArg, "pad1d: the input has no dimension to pad");
  int length = input.getShape(-1);
  if (left < 0 || right < 0) THROW(InvalidArg, "pad1d: a padding cannot be negative");
  if (mode == PadMode::kReflect && (left >= length || right >= length)) {
    THROW(InvalidArg, "pad1d: a reflection pads by less than the input is long");
  }
  std::vector<int> shape = input.getShape();
  out.throwIfInvalidShape(withLast(shape, left + length + right), "pad1d");

  DType dtype = input.getDType();
  Device device = input.getDevice();
  int last = input.getDim() - 1;
  op->copy(input, out.slice(last, {left, left + length}));

  // One edge: `width` positions of `out` from `at`, made from the input starting at `begin` --
  // the sample to repeat, or the first of the run to mirror.
  auto edge = [&](int width, int at, int begin) {
    if (width == 0) return;
    TensorView target = out.slice(last, {at, at + width});

    switch (mode) {
      case PadMode::kZero:
        op->copy(zeros(op, device, withLast(shape, width), dtype), target);
        break;

      case PadMode::kReplicate:
        op->copy(input.slice(last, {begin, begin + 1}).expand(withLast(shape, width)), target);
        break;

      case PadMode::kReflect: {
        std::vector<float> antiDiagonal(width * width, 0.0f);
        for (int row = 0; row < width; ++row) antiDiagonal[row * width + width - 1 - row] = 1.0f;
        Tensor flip = hostConstant(op, device, dtype, {width, width}, antiDiagonal);

        int rows = static_cast<int>(input.getNumEl() / length);
        Tensor run = empty(device, withLast(shape, width), dtype);
        op->copy(input.slice(last, {begin, begin + width}), run);
        Tensor mirrored = empty(device, withLast(shape, width), dtype);
        op->matmul(run.view({rows, width}), flip, mirrored.view({rows, width}));
        op->copy(mirrored, target);
        break;
      }

      default:
        THROW(InvalidArg, "pad1d: no such mode");
    }
  };

  // A mirror is about the first and last samples, which are therefore not repeated.
  bool reflect = mode == PadMode::kReflect;
  edge(left, 0, reflect ? 1 : 0);
  edge(right, left + length, reflect ? length - 1 - right : length - 1);
}

void composedClamp(Operators *op, TensorView input, float lo, float hi, TensorView out) {
  if (lo > hi) THROW(InvalidArg, "clamp: the lower bound is above the upper one");
  out.throwIfInvalidShape(input.getShape(), "clamp");

  DType dtype = input.getDType();
  Device device = input.getDevice();
  std::vector<int> shape = input.getShape();
  bool hasLo = !std::isinf(lo);
  bool hasHi = !std::isinf(hi);
  if (!hasLo && !hasHi) {
    op->copy(input, out);
    return;
  }

  // Every element beside its bound, (n, 2), and the bound picked out by the `max` or `min` that
  // reduces the last dimension. A reduction hands back one of the two as it is, so a value inside
  // the interval comes out exactly itself and one beyond it exactly the bound. Arithmetic would
  // not: `lo + relu(x - lo)` rounds `x - lo` to the input's type first, and 1e-4 in half comes
  // back from a lower bound of -0.5 as 0.
  int n = static_cast<int>(input.getNumEl());
  Tensor dense;
  TensorView x = input;
  if (!x.isContiguous()) {
    dense = emptyLike(input);
    op->copy(input, dense);
    x = dense;
  }

  Tensor pair = empty(device, {n, 2}, dtype);
  Tensor picked = empty(device, {n}, dtype);
  auto bound = [&](TensorView from, float value, bool upper) {
    op->copy(from.view({n, 1}), pair.slice(1, {0, 1}));
    op->copy(scalar(op, device, dtype, value, {n, 1}), pair.slice(1, {1, 2}));
    if (upper) {
      op->min(pair, picked);
    } else {
      op->max(pair, picked);
    }
  };
  if (hasLo) {
    bound(x, lo, false);
    x = picked;
  }
  if (hasHi) bound(x, hi, true);
  op->copy(picked.view(shape), out);

  // Whether a reduction keeps a NaN is the device's business, and torch.clamp keeps it, as the
  // kernels do. `exp(-|x|) * 0` is 0 for every number and either infinity -- exp(-inf) is 0 --
  // and NaN for a NaN alone: added on, it puts the NaN back and leaves the rest exactly as it
  // was. Not `x * 0`, which is a NaN of an infinity too.
  Tensor a = emptyLike(input);
  Tensor b = emptyLike(input);
  op->abs(input, a);
  op->neg(a, b);
  op->exp(b, a);
  op->mul(a, 0.0f, b);
  op->add(out, b, out);
}

void composedStft(
    Operators *op,
    TensorView input,
    TensorView window,
    int nFft,
    int hop,
    bool centered,
    TensorView out) {
  if (input.getDim() != 3 || input.getShape(1) != 1) {
    THROW(InvalidArg, "stft: the input must be (N, 1, L)");
  }
  if (nFft < 1 || hop < 1) THROW(InvalidArg, "stft: nFft and hop must be at least 1");
  if (window.getNumEl() != nFft) THROW(InvalidArg, "stft: the window is not nFft long");

  DType dtype = input.getDType();
  Device device = input.getDevice();
  int bins = nFft / 2 + 1;
  Tensor bank = hostConstant(
      op, device, dtype, {2 * bins, 1, nFft}, sinusoidBank(readFloats(op, window), false));

  TensorView signal = input;
  Tensor padded;
  if (centered) {
    int half = nFft / 2;
    padded = empty(device, {input.getShape(0), 1, input.getShape(2) + 2 * half}, dtype);
    op->pad1d(input, half, half, PadMode::kReflect, padded);
    signal = padded;
  }

  // One channel in, two per bin out, a stride of the hop: a frame is what the kernel covers and
  // the hop is how far it moves, which is the definition of the transform.
  op->conv1d(signal, bank, TensorView(), hop, 0, 1, 1, out);
}

void composedIstft(
    Operators *op,
    TensorView spectrum,
    TensorView window,
    int nFft,
    int hop,
    bool centered,
    TensorView out) {
  int bins = nFft / 2 + 1;
  if (spectrum.getDim() != 3 || spectrum.getShape(1) != 2 * bins) {
    THROW(InvalidArg, "istft: the spectrum must be (N, 2 * (nFft / 2 + 1), frames)");
  }
  if (nFft < 1 || hop < 1) THROW(InvalidArg, "istft: nFft and hop must be at least 1");
  if (window.getNumEl() != nFft) THROW(InvalidArg, "istft: the window is not nFft long");

  DType dtype = spectrum.getDType();
  Device device = spectrum.getDevice();
  int batch = spectrum.getShape(0);
  int frames = spectrum.getShape(2);
  int fullLength = (frames - 1) * hop + nFft;
  int trim = centered ? nFft / 2 : 0;
  out.throwIfInvalidShape({batch, 1, fullLength - 2 * trim}, "istft");

  std::vector<float> taps = readFloats(op, window);
  Tensor bank = hostConstant(op, device, dtype, {2 * bins, 1, nFft}, sinusoidBank(taps, true));

  // What the two windows together multiply each sample by, once every frame covering it is added
  // up. A position no frame covers -- the first sample of a Hann window, which `centered` trims --
  // is kept off zero rather than divided by.
  std::vector<float> envelope(fullLength, 0.0f);
  for (int frame = 0; frame < frames; ++frame) {
    for (int n = 0; n < nFft; ++n) envelope[frame * hop + n] += taps[n] * taps[n];
  }
  for (float &value : envelope) value = std::max(value, 1e-11f);

  Tensor wave = empty(device, {batch, 1, fullLength}, dtype);
  op->convTranspose1d(spectrum, bank, TensorView(), hop, 0, 0, 1, wave);

  // Divided in float32 whatever the spectrum's type. Toward the ends the envelope falls far below
  // what half can hold -- the floor above, and w[1]^2 of a 1024-point Hann at 9e-11 -- and a
  // divisor that rounded to zero makes a NaN of the sample, or an infinity.
  Tensor overlap = hostConstant(op, device, DType::kFloat, {1, 1, fullLength}, envelope);
  Tensor normalized = wave;
  if (dtype != DType::kFloat) {
    normalized = empty(device, {batch, 1, fullLength}, DType::kFloat);
    op->cast(wave, normalized);
  }
  op->divTensor(normalized, overlap.expand({batch, 1, fullLength}), normalized);

  TensorView kept = normalized.slice(2, {trim, fullLength - trim});
  if (dtype == DType::kFloat) {
    op->copy(kept, out);
  } else {
    Tensor packed = empty(device, out.getShape(), DType::kFloat);
    op->copy(kept, packed);
    op->cast(packed, out);
  }
}

}  // namespace F
}  // namespace fl
