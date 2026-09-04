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

// The convolutions SDXL runs, on the processor. The GEMM half of the same group is a level down,
// against the kernels themselves; this one goes through the operator, because a convolution on
// x64 is a packing and a GEMM and the arrangement around them is most of what there is to measure.

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include "flint/bench.h"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {
namespace {

/// How long a batch of runs should take before it is worth timing. Anything shorter and what is
/// measured is the clock and the thread pool waking up: a norm at half a millisecond came out
/// four times slower than the same norm on four times the data, which is the shape of noise
/// rather than of an answer.
constexpr double kBatchMs = 20.0;

/// How many batches to take the fastest of. Seven rather than three because on a machine with as
/// many threads as it has hyperthreads, a batch every so often comes out several times slower --
/// the parallel region waking up, not the operator -- and the slow ones do not average out with
/// the fast ones, they only add to them. The fastest of enough tries is the operator on its own,
/// which is the number a benchmark is for.
constexpr int kRuns = 7;

template<typename Fn>
double onceMs(Fn &&fn) {
  auto begin = std::chrono::steady_clock::now();
  fn();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
      .count();
}

/// The fastest of a few batches, per call. The fastest rather than the mean, because on a
/// processor the slow runs are the machine doing something else and the fast one is the closest
/// this gets to the operator on its own.
template<typename Fn>
double fastestMs(Fn &&fn) {
  fn();  // warm the caches, and do not time it

  // Doubled until a batch is long enough to time, rather than worked out from one run: that run
  // is the cold one, and an estimate off it comes out too small exactly where the operator is
  // fastest and the estimate matters most.
  int loops = 1;
  double took = 0.0;
  while (true) {
    took = onceMs([&] {
      for (int i = 0; i < loops; ++i) fn();
    });
    if (took >= kBatchMs || loops >= (1 << 20)) break;
    loops *= 2;
  }

  double best = took;
  for (int run = 1; run < kRuns; ++run) {
    double again = onceMs([&] {
      for (int i = 0; i < loops; ++i) fn();
    });
    if (again < best) best = again;
  }
  return best / loops;
}

void benchmarkConv2d(
    const std::string &name,
    int batch,
    int inChannel,
    int outChannel,
    int size,
    int kernel,
    int stride) {
  int padding = kernel / 2;
  Tensor input = F::rand({batch, inChannel, size, size}, DType::kFloat, Device::getCpu());
  Tensor weight = F::rand({outChannel, inChannel, kernel, kernel}, DType::kFloat, Device::getCpu());
  Tensor bias = F::rand({outChannel}, DType::kFloat, Device::getCpu());

  double milliseconds =
      fastestMs([&] { F::conv2d(input, weight, bias, stride, padding, 1, 1); });

  int outSize = (size + 2 * padding - kernel) / stride + 1;
  double flop = 2.0 * batch * outChannel * outSize * outSize * inChannel * kernel * kernel;
  bench::print(
      "%-36s %10.2f ms %10.1f\n", name.c_str(), milliseconds, flop / (milliseconds * 1.0e6));
}

/// The bytes an operator has to move at least once: what it reads plus what it writes. A norm or
/// an activation is held up by memory rather than by the arithmetic, so this over the time is the
/// number worth reading.
void printBandwidth(const std::string &name, double milliseconds, double bytes) {
  bench::print(
      "%-44s %10.2f ms  %8.1f GB/s\n", name.c_str(), milliseconds, bytes / (milliseconds * 1.0e6));
}

/// Elements times four, which is what float32 costs. The processor has no half kernels here, so
/// a CPU run widens the weights as it reads them and works in float throughout.
double floatBytes(std::initializer_list<int> shape) {
  double count = 1;
  for (int extent : shape) count *= extent;
  return 4.0 * count;
}

Tensor randFloat(std::initializer_list<int> shape) {
  return F::rand(shape, DType::kFloat, Device::getCpu());
}

}  // namespace

LL_BENCHMARK(bench::Group::kSdxlCpu, "SDXL elementwise") {
  // The same set the CUDA group runs, minus attention, which the processor has no operator for.
  // None of it is much arithmetic; all of it reads a whole tensor and writes one.
  //
  // Per call, not per step: how many times a step runs each of these has not been counted the way
  // the GEMM table's counts were, so these cannot be added up into a step.
  bench::print("float32, per call\n");

  struct Level {
    const char *what;
    int channels;
    int size;
  };
  const Level levels[] = {
      {"128x128x320", 320, 128},
      {"64x64x640", 640, 64},
      {"32x32x1280", 1280, 32},
  };

  for (const Level &level : levels) {
    Tensor x = randFloat({1, level.channels, level.size, level.size});
    Tensor scale = randFloat({level.channels});
    Tensor shift = randFloat({level.channels});
    double moved = 2 * floatBytes({1, level.channels, level.size, level.size});

    printBandwidth(
        std::string("group_norm   ") + level.what,
        fastestMs([&] { F::groupNorm(x, scale, shift, 32, 1e-5f); }),
        moved);
    printBandwidth(
        std::string("silu         ") + level.what,
        fastestMs([&] { F::silu(x); }),
        moved);

    Tensor other = randFloat({1, level.channels, level.size, level.size});
    printBandwidth(
        std::string("add          ") + level.what,
        fastestMs([&] { F::add(x, other); }),
        1.5 * moved);
  }

  // The transformer levels: SDXL attends at 64 by 64 and at 32 by 32 and not below.
  struct Block {
    const char *what;
    int tokens;
    int width;
  };
  const Block blocks[] = {{"64x64x640", 4096, 640}, {"32x32x1280", 1024, 1280}};

  for (const Block &block : blocks) {
    Tensor hidden = randFloat({1, block.tokens, block.width});
    Tensor scale = randFloat({block.width});
    Tensor shift = randFloat({block.width});
    printBandwidth(
        std::string("layer_norm   ") + block.what,
        fastestMs([&] { F::layerNorm(hidden, scale, shift, 1e-5f); }),
        2 * floatBytes({1, block.tokens, block.width}));

    int inner = block.width * 4;
    Tensor gated = randFloat({1, block.tokens, 2 * inner});
    printBandwidth(
        std::string("geglu        ") + block.what,
        fastestMs([&] { F::geglu(gated); }),
        1.5 * floatBytes({1, block.tokens, 2 * inner}));
  }

  const Level upsampled[] = {{"32x32x1280 to 64x64", 1280, 32}, {"64x64x640 to 128x128", 640, 64}};
  for (const Level &level : upsampled) {
    Tensor x = randFloat({1, level.channels, level.size, level.size});
    printBandwidth(
        std::string("upsample     ") + level.what,
        fastestMs([&] { F::upsampleNearest2d(x, 2); }),
        5 * floatBytes({1, level.channels, level.size, level.size}));
  }
}

LL_BENCHMARK(bench::Group::kSdxlCpu, "SDXL convolution") {
  // The same shapes the CUDA group runs, so that the two tables can be read against each other.
  // Float32 rather than half: x64 has no half kernels here, and a CPU run widens the weights as
  // it reads them.
  bench::print("float32, one group\n%-36s %13s %10s\n", "shape", "time", "GFLOP/s");
  benchmarkConv2d("unet 128x128x320 3x3", 1, 320, 320, 128, 3, 1);
  benchmarkConv2d("unet 64x64x640 3x3", 1, 640, 640, 64, 3, 1);
  benchmarkConv2d("unet 32x32x1280 3x3", 1, 1280, 1280, 32, 3, 1);
  benchmarkConv2d("unet 128x128x320 downsample 3x3/2", 1, 320, 320, 128, 3, 2);
  benchmarkConv2d("unet 64x64x640 1x1", 1, 640, 640, 64, 1, 1);
  benchmarkConv2d("vae 512x512x256 3x3", 1, 256, 256, 512, 3, 1);
}

}  // namespace fl
