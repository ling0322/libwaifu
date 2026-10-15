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

// The CUDA kernels of flint/cuda/signal.h against the CPU's, which flint/cpu/signal_test.cc checks
// against the definitions, in float and in half. And the compositions every device without a
// kernel runs, run here on the GPU: they move constants to the device and read the window back,
// which the CPU never has to, and a GPU is where that is tested before Metal relies on it.

#include <math.h>

#include <cmath>
#include <limits>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/span.h"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"
#include "flint/tensor.h"
#include "flint/test_functional.h"

namespace fl {
namespace {

Operators *cudaOps() {
  return getOperators(Device::kCuda);
}

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

Tensor onCpu(std::initializer_list<int> shape, const std::vector<float> &values) {
  return Tensor::create<float>(shape, lut::makeConstSpan(values));
}

/// `x`, a float tensor on the CPU, on the GPU as `dtype`.
Tensor onCuda(const Tensor &x, DType dtype) {
  Tensor moved = F::toDevice(cudaOps(), x, Device::getCuda());
  return dtype == DType::kFloat ? moved : F::cast(cudaOps(), moved, dtype);
}

Tensor backToCpu(const Tensor &x) {
  return F::toDevice(cudaOps(), F::cast(cudaOps(), x, DType::kFloat), Device::getCpu());
}

bool closeTo(const Tensor &cuda, const Tensor &cpu, DType dtype) {
  float tol = dtype == DType::kFloat ? 1e-4f : 2e-2f;
  return cpuOps()->allClose(backToCpu(cuda), cpu, tol, tol);
}

constexpr double kPi = 3.14159265358979323846;

std::vector<float> hann(int n) {
  std::vector<float> window(n);
  for (int i = 0; i < n; ++i) window[i] = float(0.5 - 0.5 * cos(2.0 * kPi * i / n));
  return window;
}

}  // namespace

CATCH_TEST_CASE("CUDA convTranspose1d agrees with the CPU", "[op][cuda][signal]") {
  struct Case {
    int n, c, l, k, r, stride, padding, outputPadding, groups;
  } cases[] = {
      // A BigVGAN upsampling stage, and the one-tap zero stuffing, grouped, and odd.
      {1, 64, 40, 32, 16, 8, 4, 0, 1},
      {2, 3, 17, 3, 1, 4, 0, 0, 3},
      {2, 6, 13, 4, 7, 3, 2, 1, 2},
  };

  for (const Case &t : cases) {
    CATCH_INFO("stride " << t.stride << " kernel " << t.r << " groups " << t.groups);
    Tensor x = onCpu({t.n, t.c, t.l}, spread(t.n * t.c * t.l, 1));
    Tensor w = onCpu({t.c, t.k / t.groups, t.r}, spread(t.c * t.k / t.groups * t.r, 2));
    Tensor b = onCpu({t.k}, spread(t.k, 3));
    Tensor expected =
        F::convTranspose1d(cpuOps(), x, w, b, t.stride, t.padding, t.outputPadding, t.groups);

    for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
      Tensor got = F::convTranspose1d(
          cudaOps(),
          onCuda(x, dtype),
          onCuda(w, dtype),
          onCuda(b, dtype),
          t.stride,
          t.padding,
          t.outputPadding,
          t.groups);
      CATCH_REQUIRE(closeTo(got, expected, dtype));
    }

    Tensor composed = F::zeros(Device::getCuda(), expected.getShape(), DType::kFloat);
    F::composedConvTranspose1d(
        cudaOps(),
        onCuda(x, DType::kFloat),
        onCuda(w, DType::kFloat),
        onCuda(b, DType::kFloat),
        t.stride,
        t.padding,
        t.outputPadding,
        t.groups,
        composed);
    CATCH_REQUIRE(closeTo(composed, expected, DType::kFloat));
  }
}

CATCH_TEST_CASE("CUDA snake agrees with the CPU", "[op][cuda][signal]") {
  int n = 2, c = 24, l = 301;
  Tensor x = onCpu({n, c, l}, spread(n * c * l, 4));
  std::vector<float> alpha = spread(c, 5), beta = spread(c, 6);
  for (float &v : alpha) v = 0.2f + fabsf(v);
  for (float &v : beta) v = 0.2f + fabsf(v);
  Tensor a = onCpu({c}, alpha), b = onCpu({c}, beta);

  for (bool withBeta : {false, true}) {
    Tensor expected = F::snake(cpuOps(), x, a, withBeta ? b : Tensor(), 1e-9f);
    for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
      Tensor got = F::snake(
          cudaOps(),
          onCuda(x, dtype),
          onCuda(a, dtype),
          withBeta ? onCuda(b, dtype) : Tensor(),
          1e-9f);
      CATCH_REQUIRE(closeTo(got, expected, dtype));
    }

    Tensor composed = F::zeros(Device::getCuda(), {n, c, l}, DType::kFloat);
    F::composedSnake(
        cudaOps(),
        onCuda(x, DType::kFloat),
        onCuda(a, DType::kFloat),
        withBeta ? onCuda(b, DType::kFloat) : Tensor(),
        1e-9f,
        composed);
    CATCH_REQUIRE(closeTo(composed, expected, DType::kFloat));
  }
}

CATCH_TEST_CASE("CUDA pad1d agrees with the CPU", "[op][cuda][signal]") {
  Tensor x = onCpu({2, 3, 37}, spread(2 * 3 * 37, 7));
  struct Case {
    PadMode mode;
    int left, right;
  } cases[] = {
      {PadMode::kZero, 3, 0},
      {PadMode::kReflect, 18, 36},
      {PadMode::kReplicate, 50, 2},
  };

  for (const Case &p : cases) {
    CATCH_INFO("mode " << int(p.mode));
    Tensor expected = F::pad1d(cpuOps(), x, p.left, p.right, p.mode);
    for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
      CATCH_REQUIRE(
          closeTo(F::pad1d(cudaOps(), onCuda(x, dtype), p.left, p.right, p.mode), expected, dtype));
    }

    Tensor composed = F::zeros(Device::getCuda(), expected.getShape(), DType::kFloat);
    F::composedPad1d(cudaOps(), onCuda(x, DType::kFloat), p.left, p.right, p.mode, composed);
    CATCH_REQUIRE(closeTo(composed, expected, DType::kFloat));
  }
}

CATCH_TEST_CASE("CUDA clamp agrees with the CPU", "[op][cuda][signal]") {
  float inf = std::numeric_limits<float>::infinity();
  Tensor x = onCpu({5, 77}, spread(5 * 77, 8));

  for (auto [lo, hi] : std::vector<std::pair<float, float>>{{-0.5f, 0.25f}, {0.1f, inf}, {-inf, -0.2f}}) {
    Tensor expected = F::clamp(cpuOps(), x, lo, hi);
    for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
      CATCH_REQUIRE(closeTo(F::clamp(cudaOps(), onCuda(x, dtype), lo, hi), expected, dtype));
    }

    Tensor composed = F::zeros(Device::getCuda(), {5, 77}, DType::kFloat);
    F::composedClamp(cudaOps(), onCuda(x, DType::kFloat), lo, hi, composed);
    CATCH_REQUIRE(closeTo(composed, expected, DType::kFloat));
  }

  // In half, kernel and composition both exactly the input rounded to half and then clamped: a value
  // inside the interval is itself, however small -- `x - lo + lo` would make 0 of 1e-4 against a
  // bound of -0.5 -- and one beyond it is exactly the bound.
  Tensor small = onCpu({1, 8}, {1e-4f, -1e-4f, 3e-5f, 0.2499f, -0.4999f, -1e4f, 1e4f, 99.9f});
  for (auto [lo, hi] : std::vector<std::pair<float, float>>{{-0.5f, 0.25f}, {-inf, 100.0f}}) {
    CATCH_INFO("lo " << lo << " hi " << hi);
    Tensor rounded = backToCpu(onCuda(small, DType::kFloat16));
    Tensor expected = F::clamp(cpuOps(), rounded, lo, hi);
    // The bounds too are only what half can hold of them.
    Tensor halfBounds = backToCpu(onCuda(onCpu({2}, {std::isinf(lo) ? 0.0f : lo, hi}), DType::kFloat16));
    const float *b = halfBounds.getInternalData()->getData<float>(halfBounds.getInternalOffset());
    if (!std::isinf(lo)) expected = F::clamp(cpuOps(), rounded, b[0], b[1]);
    else expected = F::clamp(cpuOps(), rounded, lo, b[1]);

    Tensor kernel = backToCpu(F::clamp(cudaOps(), onCuda(small, DType::kFloat16), lo, hi));
    Tensor composed = F::zeros(Device::getCuda(), {1, 8}, DType::kFloat16);
    F::composedClamp(cudaOps(), onCuda(small, DType::kFloat16), lo, hi, composed);
    for (const Tensor &got : {kernel, backToCpu(composed)}) {
      const float *v = got.getInternalData()->getData<float>(got.getInternalOffset());
      const float *e = expected.getInternalData()->getData<float>(expected.getInternalOffset());
      for (int i = 0; i < 8; ++i) {
        CATCH_INFO("element " << i << ": " << v[i] << " against " << e[i]);
        CATCH_REQUIRE(v[i] == e[i]);
      }
    }
  }

  // A NaN stays one, as on the CPU, and an infinity goes to the bound it is beyond: fmaxf and
  // fminf on their own would make a bound of the NaN.
  float nan = std::numeric_limits<float>::quiet_NaN();
  Tensor special = onCpu({1, 4}, {nan, inf, -inf, 0.1f});
  for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
    Tensor kernel = backToCpu(F::clamp(cudaOps(), onCuda(special, dtype), -0.5f, 0.25f));
    Tensor composed = F::zeros(Device::getCuda(), {1, 4}, dtype);
    F::composedClamp(cudaOps(), onCuda(special, dtype), -0.5f, 0.25f, composed);
    for (const Tensor &got : {kernel, backToCpu(composed)}) {
      const float *v = got.getInternalData()->getData<float>(got.getInternalOffset());
      CATCH_REQUIRE(std::isnan(v[0]));
      CATCH_REQUIRE(v[1] == 0.25f);
      CATCH_REQUIRE(v[2] == -0.5f);
      CATCH_REQUIRE(fabsf(v[3] - 0.1f) < 1e-3f);
    }
  }
}

CATCH_TEST_CASE("CUDA stft and istft agree with the CPU", "[op][cuda][signal]") {
  // HiFT's own transform, sixteen points at a hop of four, and a mel spectrogram's.
  for (auto [nFft, hop] : std::vector<std::pair<int, int>>{{16, 4}, {1024, 256}}) {
    CATCH_INFO("nFft " << nFft);
    int length = 4 * nFft;
    Tensor x = onCpu({2, 1, length}, spread(2 * length, 9));
    Tensor window = onCpu({nFft}, hann(nFft));

    Tensor expected = F::stft(cpuOps(), x, window, nFft, hop, true);
    Tensor spectrum =
        F::stft(cudaOps(), onCuda(x, DType::kFloat), onCuda(window, DType::kFloat), nFft, hop, true);
    CATCH_REQUIRE(cpuOps()->allClose(backToCpu(spectrum), expected, 1e-3f, 1e-3f));

    Tensor back = F::istft(cudaOps(), spectrum, onCuda(window, DType::kFloat), nFft, hop, true);
    CATCH_REQUIRE(cpuOps()->allClose(backToCpu(back), x, 1e-3f, 1e-3f));
  }
}

CATCH_TEST_CASE("CUDA istft in half has no NaN at the ends", "[op][cuda][signal]") {
  // Not centred, so the ends are kept: there the squared window's overlap is far below what half
  // can hold -- zero at the first sample of a Hann window, 9e-11 at the next -- and dividing by it
  // in half made NaNs. Away from the ends the round trip is the signal, to what half allows.
  int nFft = 1024, hop = 256, length = 8 * nFft;
  Tensor x = onCpu({1, 1, length}, spread(length, 10));
  Tensor window = onCpu({nFft}, hann(nFft));

  for (bool centered : {false, true}) {
    CATCH_INFO("centered " << centered);
    DType h = DType::kFloat16;
    Tensor spectrum = F::stft(cudaOps(), onCuda(x, h), onCuda(window, h), nFft, hop, centered);
    Tensor back = backToCpu(F::istft(cudaOps(), spectrum, onCuda(window, h), nFft, hop, centered));
    CATCH_REQUIRE(back.getNumEl() == length);

    const float *got = back.getInternalData()->getData<float>(back.getInternalOffset());
    const float *want = x.getInternalData()->getData<float>(x.getInternalOffset());
    double worst = 0;
    for (int i = 0; i < length; ++i) {
      CATCH_INFO("sample " << i);
      CATCH_REQUIRE(std::isfinite(got[i]));
      // Covered by a whole window's worth of frames either side.
      if (centered || (i >= nFft && i < length - nFft)) {
        worst = std::max(worst, double(fabsf(got[i] - want[i])));
      }
    }
    CATCH_REQUIRE(worst < 2e-2);
  }
}

}  // namespace fl
