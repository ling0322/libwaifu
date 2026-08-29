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

#include "../../../third_party/catch2/catch_amalgamated.hpp"

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

void benchmarkPack(Block<float> A, Block<float> Ap, int KC) {
  double t0 = lut::now();
  int kb = (A.numRows + KC - 1) / KC;
  int lastKc = A.numRows % KC;
  for (int i = 0; i < kb; ++i) {
    int kc = (i != kb - 1 || lastKc == 0) ? KC : lastKc;
    Block<float> Ai = A.sliceRow(i * KC, kc);
    Pack<float, float, CpuMathBackend::DEFAULT, Mode::OMP>(Ai, Ap, Ap.stride);
  }
  LOG(INFO) << lut::sprintf(
      "pack (%d, %d) stride=%d KC=%d T=%d: %f",
      A.numRows,
      A.numCols,
      A.stride,
      KC,
      A.transposed,
      lut::now() - t0);
}

double benchmarkSgemm(int M, int K, int N, int numLoops = 2) {
  std::vector<float> dA(M * K);
  std::vector<float> dB(K * N);
  std::vector<float> dC(M * N);

  double t0 = lut::now();
  for (int i = 0; i < numLoops; ++i)
    fl::op::cpu::kernel::gemmFloat(
        false,
        true,
        M,
        N,
        K,
        dA.data(),
        K,
        dB.data(),
        K,
        dC.data(),
        N,
        Mode::OMP);

  double dt = (lut::now() - t0) / numLoops;
  return dt;
}

double benchmarkHgemm(int M, int K, int N, int numLoops = 2) {
  std::vector<Float16> dA(M * K);
  std::vector<Float16> dB(K * N);
  std::vector<Float16> dC(M * N);

  double t0 = lut::now();
  for (int i = 0; i < numLoops; ++i)
    fl::op::cpu::kernel::gemmHalf(
        false,
        true,
        M,
        N,
        K,
        dA.data(),
        K,
        dB.data(),
        K,
        dC.data(),
        N,
        Mode::OMP);

  double dt = (lut::now() - t0) / numLoops;
  return dt;
}

#ifdef MKL_ENABLED
double benchmarkMklSgemm(int M, int K, int N, int numLoops = 2) {
  std::vector<float> dA(M * K);
  std::vector<float> dB(K * N);
  std::vector<float> dC(M * N);

  double t0 = lut::now();
  for (int i = 0; i < numLoops; ++i)
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        M,
        N,
        K,
        1.0f,
        dA.data(),
        K,
        dB.data(),
        K,
        0.0f,
        dC.data(),
        N);

  double dt = (lut::now() - t0) / numLoops;
  return dt;
}
#endif

CATCH_TEST_CASE("benchmark Pack", "[benchmark][cpu_kernel][pack]") {
  constexpr int ROW = 4096;
  constexpr int COL = 4096;
  constexpr int NR = 16;
  constexpr int NC = 512;

  std::vector<float> dA(ROW * COL);
  std::vector<float> dAp(ROW * NC);

  Block<float> A = Block<float>{dA.data(), ROW, ROW, COL, true};
  Block<float> Ap = Block<float>{dAp.data(), NR, ROW * NC / NR, NR, false};
  benchmarkPack(A, Ap, NC);
}

int gemmBenchmarkShapes[][4] = {
    {17, 4096, 27392, 2},
    {17, 13696, 4096, 2},
    {4096, 4096, 4096, 10},
    {1, 4096, 27392, 10},
    {1, 13696, 4096, 10},
    {0, 0, 0, 0}};

#if LUT_CPU_ARCH == LUT_AMD64

CATCH_TEST_CASE("benchmark SGEMM", "[benchmark][cpu_kernel][sgemm]") {
  int (*pshape)[4];

  for (pshape = &gemmBenchmarkShapes[0]; **pshape != 0; ++pshape) {
    int m = (*pshape)[0];
    int k = (*pshape)[1];
    int n = (*pshape)[2];
    int numLoops = (*pshape)[3];

    double dWaifu = benchmarkSgemm(m, k, n, numLoops);
#ifdef MKL_ENABLED
    double dMkl = benchmarkMklSgemm(m, k, n, numLoops);
    LOG(INFO) << lut::sprintf("SGEMM (M,K,N)=(%d,%d,%d): mkl=%f libwaifu=%f", m, k, n, dMkl, dWaifu);
#else
    LOG(INFO) << lut::sprintf("SGEMM (M,K,N)=(%d,%d,%d): libwaifu=%f", m, k, n, dWaifu);
#endif
  }
}

// Every GEMM the SDXL U-Net runs at 1024 by 1024, in the proportion it runs them. This is the same
// table the CUDA benchmark uses, and the counts came from there: they are per denoising step and
// already include both passes classifier free guidance makes. These ten shapes are 99.6% of a
// step's multiply-adds. The text encoders and the autoencoder are left out -- they run once an
// image rather than once a step, and are 0.3% of the work.
//
// Read what it says with care, for the same reason the CUDA one carries: it runs one shape many
// times over, so the weight stays in cache from one iteration to the next and a real run never
// gets that. On a CPU that flatters the w16a32 column hardest, because half the point of a fp16
// weight is that it is half the bytes to pull from memory, and here it is pulled once. Use it to
// see where a shape stands, and a whole image to decide anything.
struct SdxlShape {
  const char *what;
  int m, n, k, perStep;
};

const SdxlShape kSdxlShapes[] = {
    {"ff.gate      1024x10240x1280", 1024, 10240, 1280, 120},
    {"ff.out       1024x1280x5120", 1024, 1280, 5120, 120},
    {"attn proj    1024x1280x1280", 1024, 1280, 1280, 384},
    {"attn qkv     1024x3840x1280", 1024, 3840, 1280, 120},
    {"ff.gate      4096x5120x640", 4096, 5120, 640, 20},
    {"ff.out       4096x640x2560", 4096, 640, 2560, 20},
    {"attn proj    4096x640x640", 4096, 640, 640, 80},
    {"attn qkv     4096x1920x640", 4096, 1920, 640, 20},
    {"cross kv     77x2560x2048", 77, 2560, 2048, 120},
    {"cross kv     77x1280x2048", 77, 1280, 2048, 20},
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

CATCH_TEST_CASE("SDXL GEMM CPU benchmarks", "[benchmark][cpu_kernel][sdxl]") {
  std::string header =
      lut::sprintf("%-30s %10s %9s %10s %9s", "shape", "fp32", "GF/s", "w16a32", "GF/s");
#ifdef MKL_ENABLED
  header += lut::sprintf(" %10s %9s", "mkl", "GF/s");
#endif
  std::printf("\nSDXL GEMM CPU benchmarks (MxNxK, times in ms)\n%s\n", header.c_str());

  double stepUsFloat = 0.0;
  double stepUsHalfWeight = 0.0;
  double stepFlop = 0.0;
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
    std::printf("%s\n", line.c_str());

    stepUsFloat += dFloat * 1e6 * shape.perStep;
    stepUsHalfWeight += dHalfWeight * 1e6 * shape.perStep;
    stepFlop += flop * shape.perStep;
  }

  std::printf(
      "%-30s %10.1f %9.1f %10.1f %9.1f\n",
      "one denoising step",
      stepUsFloat / 1e3,
      stepFlop / stepUsFloat / 1e3,
      stepUsHalfWeight / 1e3,
      stepFlop / stepUsHalfWeight / 1e3);
  std::printf(
      "%-30s   fp32 %.1f s, w16a32 %.1f s\n",
      "thirty steps",
      stepUsFloat * 30 / 1e6,
      stepUsHalfWeight * 30 / 1e6);
}

#endif  // LUT_CPU_ARCH == LUT_AMD64

#if LUT_CPU_ARCH == LUT_AARCH64

CATCH_TEST_CASE("benchmark HGEMM", "[benchmark][cpu][cpu_kernel][hgemm]") {
  int (*pshape)[4];

  for (pshape = &gemmBenchmarkShapes[0]; **pshape != 0; ++pshape) {
    int m = (*pshape)[0];
    int k = (*pshape)[1];
    int n = (*pshape)[2];
    int numLoops = (*pshape)[3];

    double dWaifu = benchmarkHgemm(m, k, n, numLoops);
    LOG(INFO) << lut::sprintf("HGEMM (M,K,N)=(%d,%d,%d): libwaifu=%f", m, k, n, dWaifu);
  }
}

#endif  // LUT_CPU_ARCH == LUT_AMD64

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace fl
