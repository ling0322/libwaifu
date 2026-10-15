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

// The CPU kernels of flint/cpu/signal.h and the compositions in flint/functional.h that every
// other device without a kernel runs, each against the operation's own definition written out
// longhand. Two implementations agreeing with each other would agree on a shared misreading; both
// agreeing with the definition would not.

#include <math.h>

#include <cmath>
#include <limits>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/span.h"
#include "flint/functional.h"
#include "flint/operators.h"
#include "flint/tensor.h"
#include "flint/test_functional.h"

namespace fl {
namespace op {
namespace cpu {
namespace {

constexpr double kPi = 3.14159265358979323846;

Operators *cpuOps() {
  return getOperators(Device::kCpu);
}

/// Values that vary without a pattern an operator could accidentally satisfy.
std::vector<float> spread(int count, uint32_t seed) {
  std::vector<float> values;
  uint32_t state = seed | 1;
  for (int i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    values.push_back(float(state >> 8) / float(1 << 24) * 2.0f - 1.0f);
  }
  return values;
}

Tensor tensorOf(std::initializer_list<int> shape, const std::vector<float> &values) {
  return Tensor::create<float>(shape, lut::makeConstSpan(values));
}

std::vector<float> valuesOf(const Tensor &tensor) {
  Tensor dense = F::contiguous(cpuOps(), tensor);
  const float *data = dense.getInternalData()->getData<float>(dense.getInternalOffset());
  return std::vector<float>(data, data + dense.getNumEl());
}

void requireClose(const std::vector<float> &actual, const std::vector<float> &expected, float tol) {
  CATCH_REQUIRE(actual.size() == expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    CATCH_INFO("element " << i << ": " << actual[i] << " against " << expected[i]);
    if (std::isnan(expected[i]) || std::isinf(expected[i])) {
      // Exact, a NaN included: what is asked is that it came through as itself.
      CATCH_REQUIRE((std::isnan(actual[i]) ? std::isnan(expected[i]) : actual[i] == expected[i]));
      continue;
    }
    CATCH_REQUIRE(fabsf(actual[i] - expected[i]) <= tol * (1.0f + fabsf(expected[i])));
  }
}

struct Transposed {
  int n, c, l, k, r, stride, padding, outputPadding, groups;
  bool withBias;
};

/// A transposed convolution as its definition: input position `l` adds the whole kernel to the
/// output starting at `l * stride - padding`, and whatever lands outside it is dropped.
std::vector<float> transposedReference(
    const Transposed &t,
    const std::vector<float> &x,
    const std::vector<float> &w,
    const std::vector<float> &b) {
  int groupIn = t.c / t.groups;
  int groupOut = t.k / t.groups;
  int outLength = (t.l - 1) * t.stride - 2 * t.padding + t.r + t.outputPadding;
  std::vector<float> out(t.n * t.k * outLength, 0.0f);

  for (int n = 0; n < t.n; ++n) {
    for (int c = 0; c < t.c; ++c) {
      int g = c / groupIn;
      for (int l = 0; l < t.l; ++l) {
        for (int k = 0; k < groupOut; ++k) {
          for (int r = 0; r < t.r; ++r) {
            int o = l * t.stride + r - t.padding;
            if (o < 0 || o >= outLength) continue;
            out[(n * t.k + g * groupOut + k) * outLength + o] +=
                x[(n * t.c + c) * t.l + l] * w[(c * groupOut + k) * t.r + r];
          }
        }
      }
    }
  }
  if (t.withBias) {
    for (int n = 0; n < t.n; ++n) {
      for (int k = 0; k < t.k; ++k) {
        for (int o = 0; o < outLength; ++o) out[(n * t.k + k) * outLength + o] += b[k];
      }
    }
  }
  return out;
}

}  // namespace

CATCH_TEST_CASE("convTranspose1d matches its definition, kernel and composition", "[op][cpu][signal]") {
  // The vocoder's shape (kernel twice the stride), a kernel shorter than the stride, one that
  // does not divide by it with an output padding, grouped, and the one-tap zero stuffing.
  Transposed cases[] = {
      {2, 4, 9, 6, 8, 4, 2, 0, 1, true},
      {1, 3, 5, 2, 2, 3, 0, 0, 1, false},
      {2, 4, 7, 6, 7, 3, 2, 1, 2, true},
      {1, 1, 6, 1, 1, 3, 0, 0, 1, false},
  };

  for (const Transposed &t : cases) {
    CATCH_INFO("stride " << t.stride << " kernel " << t.r << " groups " << t.groups);
    std::vector<float> x = spread(t.n * t.c * t.l, 1);
    std::vector<float> w = spread(t.c * (t.k / t.groups) * t.r, 2);
    std::vector<float> b = spread(t.k, 3);
    Tensor input = tensorOf({t.n, t.c, t.l}, x);
    Tensor weight = tensorOf({t.c, t.k / t.groups, t.r}, w);
    Tensor bias = t.withBias ? tensorOf({t.k}, b) : Tensor();
    std::vector<float> expected = transposedReference(t, x, w, b);

    Tensor kernel = F::convTranspose1d(
        cpuOps(), input, weight, bias, t.stride, t.padding, t.outputPadding, t.groups);
    requireClose(valuesOf(kernel), expected, 1e-5f);

    Tensor composed = F::zeros(Device::kCpu, kernel.getShape(), DType::kFloat);
    F::composedConvTranspose1d(
        cpuOps(), input, weight, bias, t.stride, t.padding, t.outputPadding, t.groups, composed);
    requireClose(valuesOf(composed), expected, 1e-5f);
  }
}

CATCH_TEST_CASE("snake matches its definition, kernel and composition", "[op][cpu][signal]") {
  int n = 2, c = 3, l = 11;
  std::vector<float> x = spread(n * c * l, 4);
  std::vector<float> alpha = {0.5f, 1.0f, 2.5f};
  std::vector<float> beta = {0.25f, 3.0f, 1.0f};
  float eps = 1e-6f;

  for (bool withBeta : {false, true}) {
    CATCH_INFO("beta " << withBeta);
    std::vector<float> expected(x.size());
    for (int i = 0; i < n * c * l; ++i) {
      int ch = (i / l) % c;
      float s = sinf(alpha[ch] * x[i]);
      expected[i] = x[i] + s * s / ((withBeta ? beta : alpha)[ch] + eps);
    }

    Tensor input = tensorOf({n, c, l}, x);
    Tensor a = tensorOf({c}, alpha);
    Tensor b = withBeta ? tensorOf({c}, beta) : Tensor();

    requireClose(valuesOf(F::snake(cpuOps(), input, a, b, eps)), expected, 1e-5f);

    Tensor composed = F::zeros(Device::kCpu, {n, c, l}, DType::kFloat);
    F::composedSnake(cpuOps(), input, a, b, eps, composed);
    requireClose(valuesOf(composed), expected, 1e-5f);
  }
}

CATCH_TEST_CASE("pad1d fills as each mode says, kernel and composition", "[op][cpu][signal]") {
  // Two rows, so that a mirror that reached across a row would show.
  Tensor input = tensorOf({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});

  struct Case {
    PadMode mode;
    int left, right;
    std::vector<float> expected;
  } cases[] = {
      {PadMode::kZero, 2, 1, {0, 0, 1, 2, 3, 4, 0, 0, 0, 5, 6, 7, 8, 0}},
      {PadMode::kReflect, 3, 2, {4, 3, 2, 1, 2, 3, 4, 3, 2, 8, 7, 6, 5, 6, 7, 8, 7, 6}},
      // Further than the input is long, which only a replication can do.
      {PadMode::kReplicate, 5, 1, {1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 5, 5, 5, 5, 5, 5, 6, 7, 8, 8}},
      {PadMode::kReflect, 0, 0, {1, 2, 3, 4, 5, 6, 7, 8}},
  };

  for (const Case &c : cases) {
    CATCH_INFO("mode " << int(c.mode) << " left " << c.left << " right " << c.right);
    requireClose(valuesOf(F::pad1d(cpuOps(), input, c.left, c.right, c.mode)), c.expected, 0);

    Tensor composed = F::zeros(
        Device::kCpu, {2, 4 + c.left + c.right}, DType::kFloat);
    F::composedPad1d(cpuOps(), input, c.left, c.right, c.mode, composed);
    requireClose(valuesOf(composed), c.expected, 0);
  }

  CATCH_REQUIRE_THROWS(F::pad1d(cpuOps(), input, 4, 0, PadMode::kReflect));
}

CATCH_TEST_CASE("clamp bounds either side or both, kernel and composition", "[op][cpu][signal]") {
  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();
  // Two dimensions, so that a bound broadcast as if the input were a vector would show. A NaN
  // stays one, as torch.clamp has it, and an infinity goes to whichever bound it is beyond.
  std::vector<float> x = {-3.0f, -0.99f, -0.5f, 0.0f, 0.25f, 0.99f, 7.0f, 0.5f, nan, inf, -inf, 1.0f};
  Tensor input = tensorOf({3, 4}, x);

  for (auto [lo, hi] : std::vector<std::pair<float, float>>{
           {-0.99f, 0.99f}, {1e-5f, inf}, {-inf, 0.3f}, {-inf, inf}}) {
    CATCH_INFO("lo " << lo << " hi " << hi);
    std::vector<float> expected;
    for (float v : x) expected.push_back(std::isnan(v) ? v : std::min(std::max(v, lo), hi));

    requireClose(valuesOf(F::clamp(cpuOps(), input, lo, hi)), expected, 0);

    Tensor composed = F::zeros(Device::kCpu, {3, 4}, DType::kFloat);
    F::composedClamp(cpuOps(), input, lo, hi, composed);
    requireClose(valuesOf(composed), expected, 1e-6f);
  }
}

CATCH_TEST_CASE("stft is the windowed DFT of each frame", "[op][cpu][signal]") {
  int nFft = 16, hop = 4, length = 50;
  int bins = nFft / 2 + 1;
  std::vector<float> x = spread(length, 5);
  std::vector<float> window(nFft);
  for (int i = 0; i < nFft; ++i) window[i] = 0.5f - 0.5f * cosf(2.0f * float(kPi) * i / nFft);

  for (bool centered : {false, true}) {
    CATCH_INFO("centered " << centered);
    // The signal torch.stft would frame: reflected by half a window at each end when centred.
    std::vector<float> framed = x;
    if (centered) {
      framed.clear();
      for (int i = nFft / 2; i > 0; --i) framed.push_back(x[i]);
      framed.insert(framed.end(), x.begin(), x.end());
      for (int i = 1; i <= nFft / 2; ++i) framed.push_back(x[length - 1 - i]);
    }
    int frames = (int(framed.size()) - nFft) / hop + 1;

    std::vector<float> expected(2 * bins * frames);
    for (int t = 0; t < frames; ++t) {
      for (int k = 0; k < bins; ++k) {
        double re = 0, im = 0;
        for (int n = 0; n < nFft; ++n) {
          double phase = 2.0 * kPi * k * n / nFft;
          re += window[n] * framed[t * hop + n] * cos(phase);
          im -= window[n] * framed[t * hop + n] * sin(phase);
        }
        expected[k * frames + t] = float(re);
        expected[(bins + k) * frames + t] = float(im);
      }
    }

    Tensor spectrum = F::stft(cpuOps(), tensorOf({1, 1, length}, x), tensorOf({nFft}, window),
                              nFft, hop, centered);
    CATCH_REQUIRE(spectrum.getShape() == std::vector<int>{1, 2 * bins, frames});
    requireClose(valuesOf(spectrum), expected, 1e-4f);
  }
}

CATCH_TEST_CASE("istft undoes stft", "[op][cpu][signal]") {
  int nFft = 16, hop = 4, length = 64;
  std::vector<float> x = spread(2 * length, 6);
  std::vector<float> window(nFft);
  for (int i = 0; i < nFft; ++i) window[i] = 0.5f - 0.5f * cosf(2.0f * float(kPi) * i / nFft);
  Tensor w = tensorOf({nFft}, window);

  Tensor spectrum = F::stft(cpuOps(), tensorOf({2, 1, length}, x), w, nFft, hop, true);
  Tensor back = F::istft(cpuOps(), spectrum, w, nFft, hop, true);
  CATCH_REQUIRE(back.getShape() == std::vector<int>{2, 1, length});
  requireClose(valuesOf(back), x, 1e-4f);
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
