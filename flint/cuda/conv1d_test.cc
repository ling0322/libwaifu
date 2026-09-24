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

#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/span.h"
#include "flint/cuda/conv1d.h"
#include "flint/device.h"
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {

namespace {

Operators *cudaOps() {
  return getOperators(Device::kCuda);
}

Operators *cpuOps() {
  return getOperators(Device::kCpu);
}

/// One convolution to check: an input of `n` signals of `c` channels and `l` samples, and a
/// weight of `k` filters `r` taps long.
struct Case {
  int n;
  int c;
  int l;
  int k;
  int r;
  op::cuda::Conv1dOptions options;
  bool withBias;
};

/// Values that vary without a pattern a convolution could accidentally satisfy.
std::vector<float> spread(int count, uint32_t seed) {
  std::vector<float> values;
  uint32_t state = seed | 1;
  for (int i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    values.push_back(float(state >> 8) / float(1 << 24) * 2.0f - 1.0f);
  }
  return values;
}

/// The convolution written out as its definition, slow on purpose so that it shares no mistake
/// with the kernel. Returns the output length through `outL`.
std::vector<float> referenceConv1d(
    const Case &cs,
    const std::vector<float> &x,
    const std::vector<float> &w,
    const std::vector<float> &b,
    int &outL) {
  const op::cuda::Conv1dOptions &o = cs.options;
  outL = (cs.l + 2 * o.padding - o.dilation * (cs.r - 1) - 1) / o.stride + 1;
  int groupC = cs.c / o.groups;
  int groupK = cs.k / o.groups;

  std::vector<float> y(size_t(cs.n) * cs.k * outL, 0.0f);
  for (int n = 0; n < cs.n; ++n) {
    for (int k = 0; k < cs.k; ++k) {
      int group = k / groupK;
      for (int t = 0; t < outL; ++t) {
        float sum = cs.withBias ? b[k] : 0.0f;
        for (int ci = 0; ci < groupC; ++ci) {
          int c = group * groupC + ci;
          for (int tap = 0; tap < cs.r; ++tap) {
            int at = t * o.stride - o.padding + tap * o.dilation;
            if (at < 0 || at >= cs.l) continue;
            sum += x[(size_t(n) * cs.c + c) * cs.l + at] * w[(size_t(k) * groupC + ci) * cs.r + tap];
          }
        }
        y[(size_t(n) * cs.k + k) * outL + t] = sum;
      }
    }
  }

  return y;
}

Tensor toCuda(const Tensor &x, DType dtype) {
  return cudaOps()->cast(cudaOps()->toDevice(Device::getCuda(), x), dtype);
}

Tensor toCpuFloat(const Tensor &x) {
  return cudaOps()->toDevice(Device::getCpu(), cudaOps()->cast(x, DType::kFloat));
}

/// Runs one case on the card and says whether it agrees with the definition.
bool matchesReference(const Case &cs, DType dtype) {
  int groupC = cs.c / cs.options.groups;
  std::vector<float> x = spread(cs.n * cs.c * cs.l, 3);
  std::vector<float> w = spread(cs.k * groupC * cs.r, 5);
  std::vector<float> b = spread(cs.k, 7);

  int outL = 0;
  std::vector<float> expected = referenceConv1d(cs, x, w, b, outL);

  Tensor xCuda = toCuda(Tensor::create<float>({cs.n, cs.c, cs.l}, lut::makeConstSpan(x)), dtype);
  Tensor wCuda = toCuda(Tensor::create<float>({cs.k, groupC, cs.r}, lut::makeConstSpan(w)), dtype);
  Tensor bCuda = cs.withBias ? toCuda(Tensor::create<float>({cs.k}, lut::makeConstSpan(b)), dtype)
                             : Tensor();

  Tensor actual = op::cuda::conv1d(xCuda, wCuda, bCuda, cs.options);
  if (actual.getShape() != std::vector<int>{cs.n, cs.k, outL}) return false;

  Tensor reference = Tensor::create<float>({cs.n, cs.k, outL}, lut::makeConstSpan(expected));
  float tolerance = dtype == DType::kFloat ? 1e-4f : 2e-2f;
  return cpuOps()->allClose(toCpuFloat(actual), reference, tolerance, tolerance);
}

bool skipUnavailable() {
  return !isOperatorsAvailable(Device::kCuda);
}

}  // namespace

CATCH_TEST_CASE("test conv1d (one group)", "[fl][op][cuda][conv1d]") {
  if (skipUnavailable()) CATCH_SKIP("cuda device not available");

  for (DType dtype : {DType::kFloat16, DType::kFloat}) {
    // Channel counts eight divides, which is the tensor core kernel in half, and a kernel that
    // keeps the length.
    CATCH_REQUIRE(matchesReference({2, 16, 40, 32, 3, {1, 1, 1, 1}, true}, dtype));

    // Stride, dilation, and a padding wider than the kernel reaches: the padding is only on the
    // time axis, which a square conv2d padding could not have said.
    CATCH_REQUIRE(matchesReference({1, 16, 41, 32, 5, {2, 2, 1, 1}, false}, dtype));
    CATCH_REQUIRE(matchesReference({1, 16, 40, 32, 3, {1, 3, 3, 1}, true}, dtype));
    CATCH_REQUIRE(matchesReference({1, 8, 9, 8, 3, {1, 4, 1, 1}, false}, dtype));

    // Channel counts nothing divides, which is the one channel at a time kernel.
    CATCH_REQUIRE(matchesReference({2, 3, 25, 5, 7, {1, 3, 1, 1}, true}, dtype));

    // One tap, unit stride, no padding: a matrix multiply, and a different path here.
    CATCH_REQUIRE(matchesReference({3, 16, 20, 24, 1, {1, 0, 1, 1}, true}, dtype));
    // One tap with a stride is not one.
    CATCH_REQUIRE(matchesReference({1, 16, 20, 24, 1, {2, 0, 1, 1}, false}, dtype));

    // A kernel exactly as long as the signal.
    CATCH_REQUIRE(matchesReference({1, 8, 7, 8, 7, {1, 0, 1, 1}, false}, dtype));
  }
}

CATCH_TEST_CASE("test conv1d (depthwise)", "[fl][op][cuda][conv1d]") {
  if (skipUnavailable()) CATCH_SKIP("cuda device not available");

  for (DType dtype : {DType::kFloat16, DType::kFloat}) {
    // The conformer's: 31 taps, padded to keep the length, one group per channel.
    CATCH_REQUIRE(matchesReference({2, 64, 50, 64, 31, {1, 15, 1, 64}, true}, dtype));
    CATCH_REQUIRE(matchesReference({1, 16, 37, 16, 3, {1, 1, 1, 16}, false}, dtype));

    // And with a stride and a dilation, which it takes as well.
    CATCH_REQUIRE(matchesReference({1, 16, 40, 16, 5, {2, 2, 1, 16}, true}, dtype));
    CATCH_REQUIRE(matchesReference({1, 16, 40, 16, 3, {1, 3, 3, 16}, false}, dtype));

    // A channel count nothing divides, and more channels than one threadblock covers.
    CATCH_REQUIRE(matchesReference({1, 70, 21, 70, 3, {1, 1, 1, 70}, true}, dtype));
    CATCH_REQUIRE(matchesReference({2, 6, 30, 6, 5, {1, 2, 1, 6}, true}, dtype));
  }
}

CATCH_TEST_CASE("test conv1d (groups)", "[fl][op][cuda][conv1d]") {
  if (skipUnavailable()) CATCH_SKIP("cuda device not available");

  for (DType dtype : {DType::kFloat16, DType::kFloat}) {
    // Groups of several channels, where every group reads and writes a window of channels in
    // tensors shared with the others -- a mistake in that window lands in another group.
    CATCH_REQUIRE(matchesReference({2, 16, 30, 8, 3, {1, 1, 1, 4}, true}, dtype));
    CATCH_REQUIRE(matchesReference({1, 6, 33, 9, 5, {2, 2, 1, 3}, false}, dtype));

    // A channel multiplier: one input channel per group, several output channels each.
    CATCH_REQUIRE(matchesReference({1, 4, 20, 12, 3, {1, 1, 2, 4}, true}, dtype));

    // Two groups with one tap, which is not the matrix multiply path since that is one group.
    CATCH_REQUIRE(matchesReference({2, 8, 12, 8, 1, {1, 0, 1, 2}, true}, dtype));
  }
}

CATCH_TEST_CASE("test conv1d (a shape it cannot take)", "[fl][op][cuda][conv1d]") {
  if (skipUnavailable()) CATCH_SKIP("cuda device not available");

  Tensor x = toCuda(cpuOps()->rand({1, 8, 16}, DType::kFloat), DType::kFloat16);
  Tensor w = toCuda(cpuOps()->rand({8, 8, 3}, DType::kFloat), DType::kFloat16);
  Tensor noBias;

  // Every one of these is something a caller can recover from, so none may end the process.
  CATCH_REQUIRE_THROWS(op::cuda::conv1d(x.view({1, 8, 4, 4}), w, noBias, {1, 1, 1, 1}));
  CATCH_REQUIRE_THROWS(op::cuda::conv1d(x, w.view({8, 8, 3, 1}), noBias, {1, 1, 1, 1}));
  CATCH_REQUIRE_THROWS(op::cuda::conv1d(x, w, noBias, {1, 1, 1, 3}));
  CATCH_REQUIRE_THROWS(op::cuda::conv1d(x, w, noBias, {1, 1, 1, 2}));
  CATCH_REQUIRE_THROWS(op::cuda::conv1d(x, w, noBias, {0, 1, 1, 1}));
  CATCH_REQUIRE_THROWS(op::cuda::conv1d(x, w, noBias, {1, -1, 1, 1}));
  CATCH_REQUIRE_THROWS(op::cuda::conv1d(
      toCuda(cpuOps()->rand({1, 8, 2}, DType::kFloat), DType::kFloat16),
      w,
      noBias,
      {1, 0, 1, 1}));
  CATCH_REQUIRE_THROWS(op::cuda::conv1d(
      x,
      w,
      toCuda(cpuOps()->rand({4}, DType::kFloat), DType::kFloat16),
      {1, 1, 1, 1}));

  // And it still works afterwards.
  CATCH_REQUIRE(op::cuda::conv1d(x, w, noBias, {1, 1, 1, 1}).getShape() ==
                std::vector<int>{1, 8, 16});
}

CATCH_TEST_CASE("test conv1d (through the operator interface)", "[fl][op][cuda][conv1d]") {
  if (skipUnavailable()) CATCH_SKIP("cuda device not available");

  Tensor x = toCuda(cpuOps()->rand({2, 16, 24}, DType::kFloat), DType::kFloat16);
  Tensor w = toCuda(cpuOps()->rand({16, 1, 5}, DType::kFloat), DType::kFloat16);
  Tensor b = toCuda(cpuOps()->rand({16}, DType::kFloat), DType::kFloat16);

  Tensor viaOperators = cudaOps()->conv1d(x, w, b, 1, 2, 1, 16);
  Tensor direct = op::cuda::conv1d(x, w, b, {1, 2, 1, 16});

  CATCH_REQUIRE(viaOperators.getShape() == std::vector<int>{2, 16, 24});
  CATCH_REQUIRE(cpuOps()->allClose(toCpuFloat(viaOperators), toCpuFloat(direct), 1e-6f, 1e-6f));
}

}  // namespace fl
