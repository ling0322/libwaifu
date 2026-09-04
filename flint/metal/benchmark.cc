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

// The same SDXL shapes the CUDA and CPU benchmarks run, on the Metal backend through MLX. Half
// precision throughout, which is what the Metal operators use in practice.

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

constexpr double kBatchMs = 20.0;
constexpr int kRuns = 7;

template<typename Fn>
double onceMs(Fn &&fn) {
  auto begin = std::chrono::steady_clock::now();
  fn();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
      .count();
}

/// The fastest of a few batches, per call. A synchronisation point inside the lambda is needed
/// so that the wall clock measures work done rather than work submitted.
template<typename Fn>
double fastestMs(Fn &&fn) {
  fn();

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

Tensor randHalf(std::initializer_list<int> shape) {
  return F::rand(shape, DType::kFloat16, Device::getMetal());
}

/// Wait for the Metal GPU to finish all queued work, so the wall clock measures computation
/// rather than submission.
void sync() {
  getOperators(Device::kMetal)->synchronize();
}

void printMatmul(const std::string &name, double milliseconds, int m, int n, int k) {
  double tflops = 2.0 * m * n * k / (milliseconds * 1.0e9);
  bench::print("%-36s %10.2f ms %10.2f TFLOP/s\n", name.c_str(), milliseconds, tflops);
}

void printBandwidth(const std::string &name, double milliseconds, double bytes) {
  bench::print(
      "%-44s %10.2f ms  %8.1f GB/s\n", name.c_str(), milliseconds, bytes / (milliseconds * 1.0e6));
}

double halfBytes(std::initializer_list<int> shape) {
  double count = 1;
  for (int extent : shape) count *= extent;
  return 2.0 * count;
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
  Tensor input = randHalf({batch, inChannel, size, size});
  Tensor weight = randHalf({outChannel, inChannel, kernel, kernel});
  Tensor bias = randHalf({outChannel});

  double milliseconds = fastestMs([&] {
    Tensor out = F::conv2d(input, weight, bias, stride, padding, 1, 1);
    sync();
  });

  int outSize = (size + 2 * padding - kernel) / stride + 1;
  double flop = 2.0 * batch * outChannel * outSize * outSize * inChannel * kernel * kernel;
  bench::print(
      "%-36s %10.2f ms %10.1f GFLOP/s\n", name.c_str(), milliseconds, flop / (milliseconds * 1.0e6));
}

}  // namespace

LL_BENCHMARK(bench::Group::kSdxlMetal, "SDXL GEMM") {
  if (!isOperatorsAvailable(Device::kMetal)) LL_BENCHMARK_SKIP("metal device not available");

  bench::print("half precision\n%-36s %13s %10s\n", "shape", "time", "TFLOP/s");

  struct Shape {
    const char *what;
    int m, n, k;
  };

  const Shape shapes[] = {
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

  for (const Shape &shape : shapes) {
    Tensor input = randHalf({shape.m, shape.k});
    Tensor weight = randHalf({shape.n, shape.k}).transpose(0, 1);

    double milliseconds = fastestMs([&] {
      Tensor out = F::matmul(input, weight);
      sync();
    });
    printMatmul(shape.what, milliseconds, shape.m, shape.n, shape.k);
  }
}

LL_BENCHMARK(bench::Group::kSdxlMetal, "SDXL elementwise") {
  if (!isOperatorsAvailable(Device::kMetal)) LL_BENCHMARK_SKIP("metal device not available");

  bench::print("half precision, per call\n");

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
    Tensor x = randHalf({1, level.channels, level.size, level.size});
    Tensor scale = randHalf({level.channels});
    Tensor shift = randHalf({level.channels});
    double moved = 2 * halfBytes({1, level.channels, level.size, level.size});

    printBandwidth(
        std::string("group_norm   ") + level.what,
        fastestMs([&] {
          Tensor out = F::groupNorm(x, scale, shift, 32, 1e-5f);
          sync();
        }),
        moved);
    printBandwidth(
        std::string("silu         ") + level.what,
        fastestMs([&] {
          Tensor out = F::silu(x);
          sync();
        }),
        moved);

    Tensor other = randHalf({1, level.channels, level.size, level.size});
    printBandwidth(
        std::string("add          ") + level.what,
        fastestMs([&] {
          Tensor out = F::add(x, other);
          sync();
        }),
        1.5 * moved);
  }

  struct Attention {
    const char *what;
    int tokens;
    int heads;
  };
  const Attention attentions[] = {{"64x64x640", 4096, 10}, {"32x32x1280", 1024, 20}};

  for (const Attention &attention : attentions) {
    int width = attention.heads * 64;
    Tensor hidden = randHalf({1, attention.tokens, width});
    Tensor scale = randHalf({width});
    Tensor shift = randHalf({width});
    printBandwidth(
        std::string("layer_norm   ") + attention.what,
        fastestMs([&] {
          Tensor out = F::layerNorm(hidden, scale, shift, 1e-5f);
          sync();
        }),
        2 * halfBytes({1, attention.tokens, width}));

    Tensor q = randHalf({1, attention.heads, attention.tokens, 64});
    Tensor k = randHalf({1, attention.heads, attention.tokens, 64});
    Tensor v = randHalf({1, attention.heads, attention.tokens, 64});
    double flop = 4.0 * attention.heads * attention.tokens * attention.tokens * 64;
    double milliseconds = fastestMs([&] {
      Tensor out = F::attention(q, k, v, false);
      sync();
    });
    bench::print(
        "%-44s %10.2f ms  %8.2f TFLOP/s\n",
        (std::string("self attention ") + attention.what).c_str(),
        milliseconds,
        flop / (milliseconds * 1.0e9));

    Tensor ck = randHalf({1, attention.heads, 77, 64});
    Tensor cv = randHalf({1, attention.heads, 77, 64});
    double crossFlop = 4.0 * attention.heads * attention.tokens * 77 * 64;
    milliseconds = fastestMs([&] {
      Tensor out = F::attention(q, ck, cv, false);
      sync();
    });
    bench::print(
        "%-44s %10.2f ms  %8.2f TFLOP/s\n",
        (std::string("cross attention ") + attention.what).c_str(),
        milliseconds,
        crossFlop / (milliseconds * 1.0e9));

    int inner = width * 4;
    Tensor gated = randHalf({1, attention.tokens, 2 * inner});
    printBandwidth(
        std::string("geglu        ") + attention.what,
        fastestMs([&] {
          Tensor out = F::geglu(gated);
          sync();
        }),
        1.5 * halfBytes({1, attention.tokens, 2 * inner}));
  }

  const Level upsampled[] = {{"32x32x1280 to 64x64", 1280, 32}, {"64x64x640 to 128x128", 640, 64}};
  for (const Level &level : upsampled) {
    Tensor x = randHalf({1, level.channels, level.size, level.size});
    printBandwidth(
        std::string("upsample     ") + level.what,
        fastestMs([&] {
          Tensor out = F::upsampleNearest2d(x, 2);
          sync();
        }),
        5 * halfBytes({1, level.channels, level.size, level.size}));
  }
}

LL_BENCHMARK(bench::Group::kSdxlMetal, "SDXL convolution") {
  if (!isOperatorsAvailable(Device::kMetal)) LL_BENCHMARK_SKIP("metal device not available");

  bench::print("half precision, one group\n%-36s %13s %10s\n", "shape", "time", "GFLOP/s");
  benchmarkConv2d("unet 128x128x320 3x3", 1, 320, 320, 128, 3, 1);
  benchmarkConv2d("unet 64x64x640 3x3", 1, 640, 640, 64, 3, 1);
  benchmarkConv2d("unet 32x32x1280 3x3", 1, 1280, 1280, 32, 3, 1);
  benchmarkConv2d("unet 128x128x320 downsample 3x3/2", 1, 320, 320, 128, 3, 2);
  benchmarkConv2d("unet 64x64x640 1x1", 1, 640, 640, 64, 1, 1);
  benchmarkConv2d("vae 512x512x256 3x3", 1, 256, 256, 512, 3, 1);
}

}  // namespace fl
