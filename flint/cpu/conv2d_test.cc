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

// The CPU convolution against its own definition. Written out here rather than compared with the
// CUDA one: the two are meant to agree, so comparing them would hide a misreading they share.

#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/span.h"
#include "flint/functional.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {
namespace {

struct Shape4 {
  int n;
  int c;
  int h;
  int w;
};

struct Options {
  int stride;
  int padding;
  int dilation;
  int groups;
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

/// The convolution as its definition: every output is a sum over the window it sees, and anything
/// outside the image is zero. Slow on purpose -- nothing here should be clever enough to share a
/// mistake with the im2col.
std::vector<float> reference(
    const std::vector<float> &x,
    Shape4 in,
    const std::vector<float> &weight,
    Shape4 filter,
    const std::vector<float> *bias,
    Options options,
    Shape4 &out) {
  int r = filter.h;
  int s = filter.w;
  out.n = in.n;
  out.c = filter.n;
  out.h = (in.h + 2 * options.padding - options.dilation * (r - 1) - 1) / options.stride + 1;
  out.w = (in.w + 2 * options.padding - options.dilation * (s - 1) - 1) / options.stride + 1;

  int channelsPerGroup = in.c / options.groups;
  int filtersPerGroup = filter.n / options.groups;

  std::vector<float> result(size_t(out.n) * out.c * out.h * out.w, 0.0f);
  for (int n = 0; n < out.n; ++n) {
    for (int k = 0; k < out.c; ++k) {
      int group = k / filtersPerGroup;
      for (int oy = 0; oy < out.h; ++oy) {
        for (int ox = 0; ox < out.w; ++ox) {
          double sum = bias ? (*bias)[k] : 0.0;
          for (int c = 0; c < channelsPerGroup; ++c) {
            int inChannel = group * channelsPerGroup + c;
            for (int i = 0; i < r; ++i) {
              for (int j = 0; j < s; ++j) {
                int y = oy * options.stride - options.padding + i * options.dilation;
                int xx = ox * options.stride - options.padding + j * options.dilation;
                if (y < 0 || y >= in.h || xx < 0 || xx >= in.w) continue;

                sum += double(x[((size_t(n) * in.c + inChannel) * in.h + y) * in.w + xx]) *
                       double(weight[((size_t(k) * channelsPerGroup + c) * r + i) * s + j]);
              }
            }
          }
          result[((size_t(n) * out.c + k) * out.h + oy) * out.w + ox] = float(sum);
        }
      }
    }
  }

  return result;
}

bool matchesReference(Shape4 in, Shape4 filter, bool withBias, Options options) {
  std::vector<float> x = spread(in.n * in.c * in.h * in.w, 3);
  std::vector<float> w = spread(filter.n * filter.c * filter.h * filter.w, 5);
  std::vector<float> b = spread(filter.n, 7);

  Shape4 out{};
  std::vector<float> expected =
      reference(x, in, w, filter, withBias ? &b : nullptr, options, out);

  Tensor actual = F::conv2d(
      Tensor::create<float>({in.n, in.c, in.h, in.w}, lut::makeConstSpan(x)),
      Tensor::create<float>({filter.n, filter.c, filter.h, filter.w}, lut::makeConstSpan(w)),
      withBias ? Tensor::create<float>({filter.n}, lut::makeConstSpan(b)) : Tensor(),
      options.stride,
      options.padding,
      options.dilation,
      options.groups);

  if (actual.getShape() != std::vector<int>{out.n, out.c, out.h, out.w}) return false;

  return F::allClose(
      actual,
      Tensor::create<float>({out.n, out.c, out.h, out.w}, lut::makeConstSpan(expected)),
      1e-5f);
}

}  // namespace

CATCH_TEST_CASE("test conv2d on the CPU", "[core][nn][operators]") {
  // The 1x1 and the 3x3 that keeps its resolution: between them, almost every convolution in a
  // diffusion U-Net and its autoencoder.
  CATCH_REQUIRE(matchesReference({2, 8, 7, 5}, {4, 8, 1, 1}, false, {1, 0, 1, 1}));
  CATCH_REQUIRE(matchesReference({2, 8, 7, 5}, {4, 8, 3, 3}, false, {1, 1, 1, 1}));

  // With a bias, which is added to every pixel of its own channel.
  CATCH_REQUIRE(matchesReference({2, 8, 7, 5}, {4, 8, 3, 3}, true, {1, 1, 1, 1}));

  // Halving the resolution, which is how a U-Net goes down a level.
  CATCH_REQUIRE(matchesReference({1, 8, 8, 8}, {16, 8, 3, 3}, true, {2, 1, 1, 1}));

  // A kernel that shrinks its input, and one that reaches the edge exactly.
  CATCH_REQUIRE(matchesReference({1, 3, 8, 8}, {6, 3, 3, 3}, false, {1, 0, 1, 1}));
  CATCH_REQUIRE(matchesReference({1, 4, 5, 5}, {2, 4, 5, 5}, false, {1, 0, 1, 1}));

  // Not square, and a kernel that is not square either, so a transposed axis would show.
  CATCH_REQUIRE(matchesReference({1, 4, 9, 4}, {8, 4, 3, 1}, false, {1, 0, 1, 1}));

  // Padding wider than the kernel reaches, so most of the output sees only zeros.
  CATCH_REQUIRE(matchesReference({1, 2, 3, 3}, {2, 2, 3, 3}, false, {1, 2, 1, 1}));

  // Dilation, and both at once.
  CATCH_REQUIRE(matchesReference({1, 4, 9, 9}, {4, 4, 3, 3}, false, {1, 2, 2, 1}));
  CATCH_REQUIRE(matchesReference({1, 4, 9, 9}, {4, 4, 3, 3}, true, {2, 2, 2, 1}));

  // Groups, including one per channel.
  CATCH_REQUIRE(matchesReference({2, 8, 6, 6}, {8, 4, 3, 3}, false, {1, 1, 1, 2}));
  CATCH_REQUIRE(matchesReference({1, 6, 5, 5}, {6, 1, 3, 3}, true, {1, 1, 1, 6}));
}

CATCH_TEST_CASE("test conv2d on the CPU (more pixels than one block)", "[core][nn][operators]") {
  // The im2col is built a block of output pixels at a time, and the block is sized from the
  // channel count, so a wide enough image is more than one block and the seam between them is
  // somewhere a mistake would sit.
  CATCH_REQUIRE(matchesReference({1, 64, 40, 40}, {32, 64, 3, 3}, true, {1, 1, 1, 1}));
}

CATCH_TEST_CASE("test conv2d on the CPU (a shape it cannot take)", "[core][nn][operators]") {
  Tensor x = F::rand({2, 4, 8, 8}, DType::kFloat);
  Tensor w = F::rand({8, 4, 3, 3}, DType::kFloat);

  // Channels that do not match the weight, a kernel larger than the input, and a group count the
  // channels do not divide into: all a caller's to fix rather than to guess at.
  CATCH_REQUIRE_THROWS(F::conv2d(F::rand({2, 5, 8, 8}, DType::kFloat), w, Tensor(), 1, 1, 1, 1));
  CATCH_REQUIRE_THROWS(
      F::conv2d(F::rand({1, 4, 2, 2}, DType::kFloat), w, Tensor(), 1, 0, 1, 1));
  CATCH_REQUIRE_THROWS(F::conv2d(x, w, Tensor(), 1, 1, 1, 3));
  CATCH_REQUIRE_THROWS(F::conv2d(x, w, F::rand({4}, DType::kFloat), 1, 1, 1, 1));
}

CATCH_TEST_CASE("test conv2d on the CPU (half weight)", "[core][nn][operators]") {
  // A float activation against a half weight, which is how a model is held on the host: x64 has
  // no half arithmetic, so widening the weights as they are read would double the model. The
  // micro-kernel converts the weight as it packs it and the arithmetic is the float32 it would
  // have been either way, so what this compares is the two kernels rather than the rounding --
  // both are given the same weight, one holding it as half and one as that half widened again.
  struct Case {
    Shape4 in;
    Shape4 filter;
    Options options;
  };

  const Case cases[] = {
      {{2, 8, 7, 5}, {4, 8, 3, 3}, {1, 1, 1, 1}},
      {{1, 8, 8, 8}, {16, 8, 3, 3}, {2, 1, 1, 1}},
      {{2, 8, 7, 5}, {4, 8, 1, 1}, {1, 0, 1, 1}},
      {{2, 8, 6, 6}, {8, 4, 3, 3}, {1, 1, 1, 2}},
      {{1, 64, 40, 40}, {32, 64, 3, 3}, {1, 1, 1, 1}},
  };

  for (const Case &c : cases) {
    std::vector<float> x = spread(c.in.n * c.in.c * c.in.h * c.in.w, 3);
    std::vector<float> w = spread(c.filter.n * c.filter.c * c.filter.h * c.filter.w, 5);
    std::vector<float> b = spread(c.filter.n, 7);

    Tensor xT = Tensor::create<float>({c.in.n, c.in.c, c.in.h, c.in.w}, lut::makeConstSpan(x));
    Tensor wT = Tensor::create<float>(
        {c.filter.n, c.filter.c, c.filter.h, c.filter.w},
        lut::makeConstSpan(w));
    Tensor bT = Tensor::create<float>({c.filter.n}, lut::makeConstSpan(b));

    Tensor half = F::cast(wT, DType::kFloat16);
    Tensor widened = F::cast(half, DType::kFloat);

    Tensor expected = F::conv2d(
        xT, widened, bT, c.options.stride, c.options.padding, c.options.dilation, c.options.groups);
    Tensor actual = F::conv2d(
        xT, half, bT, c.options.stride, c.options.padding, c.options.dilation, c.options.groups);

    CATCH_REQUIRE(actual.getShape() == expected.getShape());
    CATCH_REQUIRE(actual.getDType() == DType::kFloat);
    CATCH_REQUIRE(F::allClose(actual, expected, 1e-5f));
  }
}

#if LUT_CPU_ARCH == LUT_AARCH64
CATCH_TEST_CASE("test conv2d on the CPU (half throughout)", "[core][nn][operators]") {
  // Half is the default float type on this architecture, so this is the path the U-Net takes.
  // The reference is the longhand convolution in float, and the tolerance is what half can hold
  // rather than what the kernel can: the GEMM sums in float, so what is left is the rounding of
  // the inputs and of the result.
  struct Case {
    Shape4 in;
    Shape4 filter;
    Options options;
  };

  const Case cases[] = {
      {{2, 8, 7, 5}, {4, 8, 3, 3}, {1, 1, 1, 1}},
      {{1, 8, 8, 8}, {16, 8, 3, 3}, {2, 1, 1, 1}},
      {{2, 8, 7, 5}, {4, 8, 1, 1}, {1, 0, 1, 1}},
      {{2, 8, 6, 6}, {8, 4, 3, 3}, {1, 1, 1, 2}},
      {{1, 64, 40, 40}, {32, 64, 3, 3}, {1, 1, 1, 1}},
  };

  for (const Case &c : cases) {
    std::vector<float> x = spread(c.in.n * c.in.c * c.in.h * c.in.w, 3);
    std::vector<float> w = spread(c.filter.n * c.filter.c * c.filter.h * c.filter.w, 5);
    std::vector<float> b = spread(c.filter.n, 7);

    Shape4 out{};
    std::vector<float> expected = reference(x, c.in, w, c.filter, &b, c.options, out);

    Tensor xT = F::cast(
        Tensor::create<float>({c.in.n, c.in.c, c.in.h, c.in.w}, lut::makeConstSpan(x)),
        DType::kFloat16);
    Tensor wT = F::cast(
        Tensor::create<float>(
            {c.filter.n, c.filter.c, c.filter.h, c.filter.w},
            lut::makeConstSpan(w)),
        DType::kFloat16);
    Tensor bT =
        F::cast(Tensor::create<float>({c.filter.n}, lut::makeConstSpan(b)), DType::kFloat16);

    Tensor actual = F::conv2d(
        xT, wT, bT, c.options.stride, c.options.padding, c.options.dilation, c.options.groups);

    CATCH_REQUIRE(actual.getDType() == DType::kFloat16);
    CATCH_REQUIRE(actual.getShape() == std::vector<int>{out.n, out.c, out.h, out.w});
    CATCH_REQUIRE(
        F::allClose(
            F::cast(actual, DType::kFloat),
            Tensor::create<float>({out.n, out.c, out.h, out.w}, lut::makeConstSpan(expected)),
            5e-3f,
            5e-3f));
  }
}
#endif  // LUT_CPU_ARCH == LUT_AARCH64

}  // namespace cpu
}  // namespace op
}  // namespace fl
