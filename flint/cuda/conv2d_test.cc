// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cstdlib>
#include <string>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/span.h"
#include "flint/cuda/conv2d.h"
#include "flint/cuda/conv2d_cutlass.h"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {
namespace {

struct Shape4 {
  int n;
  int c;
  int h;
  int w;
};

/// Values that vary without a pattern a convolution could accidentally satisfy, and without
/// pulling in a random number generator.
std::vector<float> spread(int count, uint32_t seed) {
  std::vector<float> values;
  uint32_t state = seed | 1;
  for (int i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    values.push_back(float(state >> 8) / float(1 << 24) * 2.0f - 1.0f);
  }
  return values;
}

/// The convolution written out as its definition, which is what the cuDNN one is checked against.
/// Slow on purpose: nothing here should be clever enough to share a mistake with the kernel.
std::vector<float> referenceConv2d(
    const std::vector<float> &x,
    Shape4 in,
    const std::vector<float> &weight,
    Shape4 filter,
    const std::vector<float> *bias,
    const op::cuda::Conv2dOptions &options,
    Shape4 &out) {
  int r = filter.h;
  int s = filter.w;
  out.n = in.n;
  out.c = filter.n;
  out.h = (in.h + 2 * options.padding - options.dilation * (r - 1) - 1) / options.stride + 1;
  out.w = (in.w + 2 * options.padding - options.dilation * (s - 1) - 1) / options.stride + 1;

  int inChannelPerGroup = in.c / options.groups;
  int outChannelPerGroup = out.c / options.groups;

  std::vector<float> y(size_t(out.n) * out.c * out.h * out.w, 0.0f);
  for (int n = 0; n < out.n; ++n) {
    for (int k = 0; k < out.c; ++k) {
      int group = k / outChannelPerGroup;
      for (int oh = 0; oh < out.h; ++oh) {
        for (int ow = 0; ow < out.w; ++ow) {
          float sum = bias ? (*bias)[k] : 0.0f;
          for (int ci = 0; ci < inChannelPerGroup; ++ci) {
            int c = group * inChannelPerGroup + ci;
            for (int kh = 0; kh < r; ++kh) {
              for (int kw = 0; kw < s; ++kw) {
                int ih = oh * options.stride - options.padding + kh * options.dilation;
                int iw = ow * options.stride - options.padding + kw * options.dilation;
                if (ih < 0 || ih >= in.h || iw < 0 || iw >= in.w) continue;

                float a = x[((size_t(n) * in.c + c) * in.h + ih) * in.w + iw];
                float b = weight[((size_t(k) * inChannelPerGroup + ci) * r + kh) * s + kw];
                sum += a * b;
              }
            }
          }
          y[((size_t(n) * out.c + k) * out.h + oh) * out.w + ow] = sum;
        }
      }
    }
  }

  return y;
}

Tensor toCuda(const Tensor &x, DType dtype) {
  return F::cast(F::toDevice(Device::getCuda(), x), dtype);
}

Tensor toCpuFloat(const Tensor &x) {
  return F::toDevice(Device::getCpu(), F::cast(x, DType::kFloat));
}

bool skipUnavailable() {
  if (!isOperatorsAvailable(Device::kCuda)) return true;
  return !op::cuda::isConv2dAvailable();
}

/// Which implementation to check. The operator convolves on CUTLASS, and the cases are checked
/// against a written-out reference rather than against another library, so a mistake the two
/// could share is not a mistake either of them can hide.
using Conv2dFn = Tensor (*)(
    const Tensor &,
    const Tensor &,
    const Tensor &,
    const op::cuda::Conv2dOptions &);

/// Runs one convolution both ways and says whether they agree.
bool matchesReference(
    Shape4 in,
    Shape4 filter,
    bool withBias,
    const op::cuda::Conv2dOptions &options,
    DType dtype,
    float rtol,
    Conv2dFn convolve = op::cuda::conv2d) {
  std::vector<float> x = spread(in.n * in.c * in.h * in.w, 3);
  std::vector<float> w = spread(filter.n * filter.c * filter.h * filter.w, 5);
  std::vector<float> b = spread(filter.n, 7);

  Shape4 out{};
  std::vector<float> expected =
      referenceConv2d(x, in, w, filter, withBias ? &b : nullptr, options, out);

  Tensor xCuda = toCuda(
      Tensor::create<float>({in.n, in.c, in.h, in.w}, lut::makeConstSpan(x)),
      dtype);
  Tensor wCuda = toCuda(
      Tensor::create<float>({filter.n, filter.c, filter.h, filter.w}, lut::makeConstSpan(w)),
      dtype);
  Tensor bCuda = withBias ? toCuda(Tensor::create<float>({filter.n}, lut::makeConstSpan(b)), dtype)
                          : Tensor();

  Tensor actual = convolve(xCuda, wCuda, bCuda, options);
  if (actual.getShape() != std::vector<int>{out.n, out.c, out.h, out.w}) return false;

  Tensor reference = Tensor::create<float>(
      {out.n, out.c, out.h, out.w},
      lut::makeConstSpan(expected));

  return F::allClose(toCpuFloat(actual), reference, rtol);
}

}  // namespace

CATCH_TEST_CASE("test conv2d (kernel shapes)", "[fl][op][cuda][conv2d]") {
  if (skipUnavailable()) CATCH_SKIP("conv2d not available");

  // 1x1, which is a per-pixel matrix multiply, and the 3x3 that keeps its resolution: between
  // them these are almost every convolution in a diffusion U-Net.
  CATCH_REQUIRE(matchesReference(
      {2, 8, 7, 5},
      {4, 8, 1, 1},
      false,
      {1, 0, 1, 1},
      DType::kFloat16,
      2e-2f));
  CATCH_REQUIRE(matchesReference(
      {2, 8, 7, 5},
      {4, 8, 3, 3},
      false,
      {1, 1, 1, 1},
      DType::kFloat16,
      2e-2f));

  // A kernel that shrinks its input, and one that reaches the edge exactly.
  CATCH_REQUIRE(matchesReference(
      {1, 3, 8, 8},
      {6, 3, 3, 3},
      false,
      {1, 0, 1, 1},
      DType::kFloat16,
      2e-2f));
  CATCH_REQUIRE(matchesReference(
      {1, 4, 5, 5},
      {2, 4, 5, 5},
      false,
      {1, 0, 1, 1},
      DType::kFloat16,
      2e-2f));

  // Non-square input, and a kernel that is not square either.
  CATCH_REQUIRE(matchesReference(
      {1, 4, 9, 4},
      {8, 4, 3, 1},
      false,
      {1, 0, 1, 1},
      DType::kFloat16,
      2e-2f));
}

CATCH_TEST_CASE("test conv2d (stride, padding, dilation)", "[fl][op][cuda][conv2d]") {
  if (skipUnavailable()) CATCH_SKIP("conv2d not available");

  // The downsample of a U-Net: 3x3 taken two pixels at a time.
  CATCH_REQUIRE(matchesReference(
      {2, 8, 8, 8},
      {8, 8, 3, 3},
      false,
      {2, 0, 1, 1},
      DType::kFloat16,
      2e-2f));
  CATCH_REQUIRE(matchesReference(
      {1, 4, 9, 9},
      {4, 4, 3, 3},
      false,
      {2, 1, 1, 1},
      DType::kFloat16,
      2e-2f));

  // Padding wider than the kernel reaches, so most of the output sees only zeros.
  CATCH_REQUIRE(matchesReference(
      {1, 2, 3, 3},
      {2, 2, 3, 3},
      false,
      {1, 2, 1, 1},
      DType::kFloat16,
      2e-2f));

  // Dilation, which spreads the kernel out over the input without growing it.
  CATCH_REQUIRE(matchesReference(
      {1, 4, 9, 9},
      {4, 4, 3, 3},
      false,
      {1, 2, 2, 1},
      DType::kFloat16,
      2e-2f));
  // Several groups are nobody's here any more: CUTLASS does one and the operator is CUTLASS, so
  // what there is to check is that asking is refused. That case is with the rest of its own
  // below.
}

CATCH_TEST_CASE("test conv2d (bias)", "[fl][op][cuda][conv2d]") {
  if (skipUnavailable()) CATCH_SKIP("conv2d not available");

  // The bias is one value per output channel, spread over the batch and both spatial axes.
  CATCH_REQUIRE(matchesReference(
      {2, 4, 6, 6},
      {8, 4, 3, 3},
      true,
      {1, 1, 1, 1},
      DType::kFloat16,
      2e-2f));
  CATCH_REQUIRE(matchesReference(
      {1, 4, 5, 5},
      {4, 4, 1, 1},
      true,
      {1, 0, 1, 1},
      DType::kFloat16,
      2e-2f));
}

CATCH_TEST_CASE("test conv2d (float)", "[fl][op][cuda][conv2d]") {
  if (skipUnavailable()) CATCH_SKIP("conv2d not available");

  // A VAE decoder is the reason to have this path: it overflows float16 on real weights, so it is
  // run in float, and the tolerance says as much.
  CATCH_REQUIRE(matchesReference(
      {1, 4, 8, 8},
      {8, 4, 3, 3},
      true,
      {1, 1, 1, 1},
      DType::kFloat,
      1e-5f));
  CATCH_REQUIRE(matchesReference(
      {2, 6, 5, 7},
      {4, 6, 3, 3},
      false,
      {2, 1, 1, 1},
      DType::kFloat,
      1e-5f));
}

CATCH_TEST_CASE("test conv2d (a shape it cannot take)", "[fl][op][cuda][conv2d]") {
  if (skipUnavailable()) CATCH_SKIP("conv2d not available");

  Tensor x = toCuda(F::rand({1, 4, 8, 8}, DType::kFloat), DType::kFloat16);
  Tensor w = toCuda(F::rand({4, 4, 3, 3}, DType::kFloat), DType::kFloat16);
  Tensor noBias;

  // Wrong rank, channels that do not match the weight, and a kernel larger than what it is given
  // are all things a caller can recover from, so none of them may end the process.
  CATCH_REQUIRE_THROWS(op::cuda::conv2d(x.view({1, 4, 64}), w, noBias, {1, 1, 1, 1}));
  CATCH_REQUIRE_THROWS(
      op::cuda::conv2d(toCuda(F::rand({1, 5, 8, 8}, DType::kFloat), DType::kFloat16), w, noBias,
                       {1, 1, 1, 1}));
  CATCH_REQUIRE_THROWS(op::cuda::conv2d(
      toCuda(F::rand({1, 4, 2, 2}, DType::kFloat), DType::kFloat16),
      w,
      noBias,
      {1, 0, 1, 1}));

  // And the operator still works afterwards.
  CATCH_REQUIRE(op::cuda::conv2d(x, w, noBias, {1, 1, 1, 1}).getShape() ==
                std::vector<int>{1, 4, 8, 8});
}

CATCH_TEST_CASE("test conv2d (through the operator interface)", "[fl][op][cuda][conv2d]") {
  if (skipUnavailable()) CATCH_SKIP("conv2d not available");

  // The same convolution reached the way a layer would reach it.
  Tensor x = toCuda(F::rand({2, 4, 8, 8}, DType::kFloat), DType::kFloat16);
  Tensor w = toCuda(F::rand({8, 4, 3, 3}, DType::kFloat), DType::kFloat16);
  Tensor b = toCuda(F::rand({8}, DType::kFloat), DType::kFloat16);

  Tensor viaOperators = F::conv2d(x, w, b, 1, 1, 1, 1);
  Tensor direct = op::cuda::conv2d(x, w, b, {1, 1, 1, 1});

  CATCH_REQUIRE(viaOperators.getShape() == std::vector<int>{2, 8, 8, 8});
  CATCH_REQUIRE(F::allClose(toCpuFloat(viaOperators), toCpuFloat(direct), 1e-6f, 1e-6f));

  // The host has its own convolution now, so the operator answers there rather than refusing.
  // What it gets is checked against a written-out reference in cpu/conv2d_test.cc; here it is
  // enough that the two devices agree, since each has been checked against that definition
  // separately.
  Tensor cpuX = F::toDevice(Device::getCpu(), F::cast(x, DType::kFloat));
  Tensor cpuW = F::toDevice(Device::getCpu(), F::cast(w, DType::kFloat));
  Tensor cpuB = F::toDevice(Device::getCpu(), F::cast(b, DType::kFloat));
  Tensor onHost = F::conv2d(cpuX, cpuW, cpuB, 1, 1, 1, 1);
  CATCH_REQUIRE(onHost.getShape() == std::vector<int>{2, 8, 8, 8});
  CATCH_REQUIRE(F::allClose(onHost, toCpuFloat(direct), 2e-2f));
}

CATCH_TEST_CASE("test conv2d (cutlass)", "[fl][op][cuda][cutlass][conv2d]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");
  if (!op::cuda::isConv2dCutlassAvailable()) CATCH_SKIP("this build has no CUTLASS");

  auto cutlass = op::cuda::conv2dCutlass;

  // The three parameter combinations SDXL asks for and nothing else: a 3x3 that keeps its
  // resolution, a 3x3 that halves it, and the 1x1 that is a matrix multiply rather than a
  // convolution -- which is a different code path here, so it is not the same test twice.
  CATCH_REQUIRE(matchesReference(
      {1, 16, 12, 12}, {32, 16, 3, 3}, false, {1, 1, 1, 1}, DType::kFloat16, 2e-2f, cutlass));
  CATCH_REQUIRE(matchesReference(
      {1, 16, 12, 12}, {32, 16, 3, 3}, false, {2, 1, 1, 1}, DType::kFloat16, 2e-2f, cutlass));
  CATCH_REQUIRE(matchesReference(
      {1, 16, 12, 12}, {32, 16, 1, 1}, false, {1, 0, 1, 1}, DType::kFloat16, 2e-2f, cutlass));

  // With a bias, which is added after the convolution rather than by it.
  CATCH_REQUIRE(matchesReference(
      {1, 16, 12, 12}, {32, 16, 3, 3}, true, {1, 1, 1, 1}, DType::kFloat16, 2e-2f, cutlass));
  CATCH_REQUIRE(matchesReference(
      {1, 16, 12, 12}, {32, 16, 1, 1}, true, {1, 0, 1, 1}, DType::kFloat16, 2e-2f, cutlass));

  // Channel counts that eight does not divide, which is the other kernel: SDXL's latent is four
  // channels deep and its image is three, so both ends of the model land here.
  CATCH_REQUIRE(matchesReference(
      {1, 4, 12, 12}, {32, 4, 3, 3}, true, {1, 1, 1, 1}, DType::kFloat16, 2e-2f, cutlass));
  CATCH_REQUIRE(matchesReference(
      {1, 16, 12, 12}, {4, 16, 3, 3}, true, {1, 1, 1, 1}, DType::kFloat16, 2e-2f, cutlass));

  // More than one image, which the permute has to keep apart.
  CATCH_REQUIRE(matchesReference(
      {3, 16, 10, 10}, {32, 16, 3, 3}, true, {1, 1, 1, 1}, DType::kFloat16, 2e-2f, cutlass));

  // Not square, so that a transposed height and width would show.
  CATCH_REQUIRE(matchesReference(
      {1, 16, 14, 6}, {32, 16, 3, 3}, false, {1, 1, 1, 1}, DType::kFloat16, 2e-2f, cutlass));
}

CATCH_TEST_CASE("test conv2d (cutlass refuses what it cannot do)", "[fl][op][cuda][cutlass][conv2d]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");
  if (!op::cuda::isConv2dCutlassAvailable()) CATCH_SKIP("this build has no CUTLASS");

  // A grouped convolution needs another instantiation and nothing here asks for one, so it is
  // refused rather than answered wrongly. Nothing else about the operator is narrower than cuDNN.
  Tensor x = F::cast(F::toDevice(Device::getCuda(), F::rand({2, 8, 6, 6}, DType::kFloat)), DType::kFloat16);
  Tensor w = F::cast(F::toDevice(Device::getCuda(), F::rand({8, 4, 3, 3}, DType::kFloat)), DType::kFloat16);
  CATCH_REQUIRE_THROWS(op::cuda::conv2dCutlass(x, w, Tensor(), {1, 1, 1, 2}));
}

CATCH_TEST_CASE("test conv2d (cutlass, float)", "[fl][op][cuda][cutlass][conv2d]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");
  if (!op::cuda::isConv2dCutlassAvailable()) CATCH_SKIP("this build has no CUTLASS");

  auto cutlass = op::cuda::conv2dCutlass;

  // The autoencoder runs in float32 and is almost all 3x3, ending on a three channel image.
  CATCH_REQUIRE(matchesReference(
      {1, 16, 12, 12}, {32, 16, 3, 3}, true, {1, 1, 1, 1}, DType::kFloat, 1e-4f, cutlass));
  CATCH_REQUIRE(matchesReference(
      {1, 16, 12, 12}, {3, 16, 3, 3}, true, {1, 1, 1, 1}, DType::kFloat, 1e-4f, cutlass));
  CATCH_REQUIRE(matchesReference(
      {1, 4, 12, 12}, {16, 4, 1, 1}, true, {1, 0, 1, 1}, DType::kFloat, 1e-4f, cutlass));
}

}  // namespace fl
