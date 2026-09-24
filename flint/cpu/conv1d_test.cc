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

// The CPU 1-D convolution against its own definition. Written out longhand here rather than
// compared with `conv2d` over a one-row image, which is what the implementation *is*: a test that
// asked the same question the code asks would agree with any misreading of the padding, and the
// padding across the two axes is the whole of what conv1d does differently.

#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/span.h"
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {

namespace {

Operators *cpuOps() {
  return getOperators(Device::kCpu);
}

}  // namespace

namespace op {
namespace cpu {
namespace {

struct Shape3 {
  int n;
  int c;
  int l;
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
/// off either end of the signal is zero.
std::vector<float> reference(
    const std::vector<float> &x,
    Shape3 in,
    const std::vector<float> &weight,
    Shape3 filter,
    const std::vector<float> *bias,
    Options options,
    Shape3 &out) {
  int r = filter.l;
  out.n = in.n;
  out.c = filter.n;
  out.l = (in.l + 2 * options.padding - options.dilation * (r - 1) - 1) / options.stride + 1;

  int channelsPerGroup = in.c / options.groups;
  int filtersPerGroup = filter.n / options.groups;

  std::vector<float> result(size_t(out.n) * out.c * out.l, 0.0f);
  for (int n = 0; n < out.n; ++n) {
    for (int k = 0; k < out.c; ++k) {
      int group = k / filtersPerGroup;
      for (int ot = 0; ot < out.l; ++ot) {
        double sum = bias ? (*bias)[k] : 0.0;
        for (int c = 0; c < channelsPerGroup; ++c) {
          int inChannel = group * channelsPerGroup + c;
          for (int i = 0; i < r; ++i) {
            int t = ot * options.stride - options.padding + i * options.dilation;
            if (t < 0 || t >= in.l) continue;

            sum += double(x[(size_t(n) * in.c + inChannel) * in.l + t]) *
                   double(weight[(size_t(k) * channelsPerGroup + c) * r + i]);
          }
        }
        result[(size_t(n) * out.c + k) * out.l + ot] = float(sum);
      }
    }
  }

  return result;
}

bool matchesReference(Shape3 in, Shape3 filter, bool withBias, Options options) {
  std::vector<float> x = spread(in.n * in.c * in.l, 3);
  std::vector<float> w = spread(filter.n * filter.c * filter.l, 5);
  std::vector<float> b = spread(filter.n, 7);

  Shape3 out{};
  std::vector<float> expected =
      reference(x, in, w, filter, withBias ? &b : nullptr, options, out);

  Tensor actual = cpuOps()->conv1d(
      Tensor::create<float>({in.n, in.c, in.l}, lut::makeConstSpan(x)),
      Tensor::create<float>({filter.n, filter.c, filter.l}, lut::makeConstSpan(w)),
      withBias ? Tensor::create<float>({filter.n}, lut::makeConstSpan(b)) : Tensor(),
      options.stride,
      options.padding,
      options.dilation,
      options.groups);

  if (actual.getShape() != std::vector<int>{out.n, out.c, out.l}) return false;

  return cpuOps()->allClose(
      actual,
      Tensor::create<float>({out.n, out.c, out.l}, lut::makeConstSpan(expected)),
      1e-5f);
}

}  // namespace

CATCH_TEST_CASE("test conv1d on the CPU", "[core][nn][operators]") {
  // The kernel of one and the kernel of three that keeps its length: between them, most of what
  // a speech model convolves.
  CATCH_REQUIRE(matchesReference({2, 8, 12}, {4, 8, 1}, false, {1, 0, 1, 1}));
  CATCH_REQUIRE(matchesReference({2, 8, 12}, {4, 8, 3}, false, {1, 1, 1, 1}));

  // With a bias, which is added to every sample of its own channel.
  CATCH_REQUIRE(matchesReference({2, 8, 12}, {4, 8, 3}, true, {1, 1, 1, 1}));

  // Striding, which is how a codec goes down a rate.
  CATCH_REQUIRE(matchesReference({1, 8, 16}, {16, 8, 5}, true, {2, 2, 1, 1}));

  // A kernel that shrinks its input, and one that reaches the end exactly.
  CATCH_REQUIRE(matchesReference({1, 3, 8}, {6, 3, 3}, false, {1, 0, 1, 1}));
  CATCH_REQUIRE(matchesReference({1, 4, 5}, {2, 4, 5}, false, {1, 0, 1, 1}));

  // Padding wider than the kernel reaches, so most of the output sees only zeros. This is the
  // case a square 2-D padding gets wrong, because it would pad the single row as well.
  CATCH_REQUIRE(matchesReference({1, 2, 3}, {2, 2, 3}, false, {1, 2, 1, 1}));

  // Dilation, which is what a conformer's wide receptive field is made of, and both at once.
  CATCH_REQUIRE(matchesReference({1, 4, 20}, {4, 4, 3}, false, {1, 2, 2, 1}));
  CATCH_REQUIRE(matchesReference({1, 4, 20}, {4, 4, 3}, true, {2, 2, 4, 1}));

  // Groups, and the depthwise case -- one group per channel -- which is every grouped convolution
  // IndexTTS-2.5 asks for.
  CATCH_REQUIRE(matchesReference({2, 8, 10}, {8, 4, 3}, false, {1, 1, 1, 2}));
  CATCH_REQUIRE(matchesReference({1, 6, 9}, {6, 1, 3}, true, {1, 1, 1, 6}));
  CATCH_REQUIRE(matchesReference({1, 16, 24}, {16, 1, 5}, true, {1, 2, 1, 16}));
}

CATCH_TEST_CASE("test conv1d on the CPU (more samples than one block)", "[core][nn][operators]") {
  // The im2col is built a block of output positions at a time, sized from the channel count, so a
  // long enough signal is more than one block and the seam between them is somewhere a mistake
  // would sit.
  CATCH_REQUIRE(matchesReference({1, 64, 1600}, {32, 64, 3}, true, {1, 1, 1, 1}));
}

CATCH_TEST_CASE("test conv1d on the CPU (a shape it cannot take)", "[core][nn][operators]") {
  Tensor x = cpuOps()->rand({2, 4, 8}, DType::kFloat);
  Tensor w = cpuOps()->rand({8, 4, 3}, DType::kFloat);

  // A 4-D input, channels that do not match the weight, a kernel longer than the input, a group
  // count the channels do not divide into, and a bias of the wrong width.
  CATCH_REQUIRE_THROWS(
      cpuOps()->conv1d(cpuOps()->rand({2, 4, 8, 8}, DType::kFloat), w, Tensor(), 1, 1, 1, 1));
  CATCH_REQUIRE_THROWS(
      cpuOps()->conv1d(cpuOps()->rand({2, 5, 8}, DType::kFloat), w, Tensor(), 1, 1, 1, 1));
  CATCH_REQUIRE_THROWS(
      cpuOps()->conv1d(cpuOps()->rand({1, 4, 2}, DType::kFloat), w, Tensor(), 1, 0, 1, 1));
  CATCH_REQUIRE_THROWS(cpuOps()->conv1d(x, w, Tensor(), 1, 1, 1, 3));
  CATCH_REQUIRE_THROWS(cpuOps()->conv1d(x, w, cpuOps()->rand({4}, DType::kFloat), 1, 1, 1, 1));
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
