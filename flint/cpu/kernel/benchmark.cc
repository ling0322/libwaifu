// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
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

#include "flint/bench.h"

#ifdef MKL_ENABLED
#include <mkl.h>
#endif

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <functional>
#include <string>

#include "lutil/attributes.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "lutil/time.h"
#include "flint/cpu/kernel/abstract.h"
#include "flint/cpu/kernel/gemm.h"
#include "flint/cpu/kernel/interface.h"
#include "flint/cpu/kernel/util.h"

namespace fl {
namespace op {
namespace cpu {
namespace kernel {

#if LUT_CPU_ARCH == LUT_AMD64

struct SdxlShape {
  const char *what;
  int m, n, k;
};

const SdxlShape kSdxlShapes[] = {
    {"ff.gate      1024x10240x1280", 1024, 10240, 1280},
    {"ff.out       1024x1280x5120", 1024, 1280, 5120},
    {"attn proj    1024x1280x1280", 1024, 1280, 1280},
    {"attn qkv     1024x3840x1280", 1024, 3840, 1280},
    {"ff.gate      4096x5120x640", 4096, 5120, 640},
    {"ff.out       4096x640x2560", 4096, 640, 2560},
    {"attn proj    4096x640x640", 4096, 640, 640},
    {"attn qkv     4096x1920x640", 4096, 1920, 640},
    {"cross kv     77x2560x2048", 77, 2560, 2048},
    {"cross kv     77x1280x2048", 77, 1280, 2048},
};

// enough loops to spend a fraction of a second on a shape, so the small ones are not read off the
// timer's noise floor and the big ones still get enough tries for the fastest to mean something.
int sdxlNumLoops(const SdxlShape &shape) {
  double flop = 2.0 * shape.m * shape.n * shape.k;
  int n = static_cast<int>(1.0e11 / flop);
  return std::min(std::max(n, 8), 100);
}

// time fn numLoops times and keep the fastest. A mean over a couple of runs swings by more than
// ten times on a machine that is doing anything else at all; the fastest run is the one that was
// interfered with least, which is the number this benchmark is trying to report.
double benchmarkFastest(int numLoops, const std::function<void()> &fn) {
  fn();  // warm up: first touch of C, and the weight pulled in once.

  double best = std::numeric_limits<double>::max();
  for (int i = 0; i < numLoops; ++i) {
    double t0 = lut::now();
    fn();
    best = std::min(best, lut::now() - t0);
  }
  return best;
}

void fillRandom(std::vector<float> &x) {
  for (float &v : x) v = (rand() % 2000 - 1000) / 1000.0f;
}

LL_BENCHMARK(bench::Group::kSdxlCpu, "SDXL GEMM") {
  std::string header =
      lut::sprintf("%-36s %10s %9s %10s %9s", "shape", "fp32", "GF/s", "w16a32", "GF/s");
#ifdef MKL_ENABLED
  header += lut::sprintf(" %10s %9s", "mkl", "GF/s");
#endif
  bench::print("MxNxK, times in ms\n%s\n", header.c_str());

  for (const SdxlShape &shape : kSdxlShapes) {
    int M = shape.m, N = shape.n, K = shape.k;
    int numLoops = sdxlNumLoops(shape);
    double flop = 2.0 * M * N * K;

    // B is (N, K), the layout a Linear's weight is stored in, so transB.
    std::vector<float> dA(M * K), dB(K * N), dC(M * N);
    std::vector<Float16> dBh(K * N);
    fillRandom(dA);
    fillRandom(dB);
    for (int i = 0; i < K * N; ++i) dBh[i] = cvt_s2h(dB[i]);

    double dFloat = benchmarkFastest(numLoops, [&] {
      gemmFloat(false, true, M, N, K, dA.data(), K, dB.data(), K, dC.data(), N, Mode::OMP);
    });
    double dHalfWeight = benchmarkFastest(numLoops, [&] {
      gemmHalfWeightFloat(false, true, M, N, K, dA.data(), K, dBh.data(), K, dC.data(), N,
                          Mode::OMP);
    });

    std::string line = lut::sprintf(
        "%-30s %10.2f %9.1f %10.2f %9.1f",
        shape.what,
        dFloat * 1e3,
        flop / dFloat / 1e9,
        dHalfWeight * 1e3,
        flop / dHalfWeight / 1e9);
#ifdef MKL_ENABLED
    double dMkl = benchmarkFastest(numLoops, [&] {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, dA.data(), K, dB.data(),
                  K, 0.0f, dC.data(), N);
    });
    line += lut::sprintf(" %10.2f %9.1f", dMkl * 1e3, flop / dMkl / 1e9);
#endif
    bench::print("%s\n", line.c_str());
  }
}

#endif  // LUT_CPU_ARCH == LUT_AMD64

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace fl
