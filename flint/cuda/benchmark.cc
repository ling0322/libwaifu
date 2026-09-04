#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <chrono>

#include "flint/bench.h"
#include "lutil/span.h"
#include "lutil/strings.h"
#include "flint/cuda/common.h"
#include "flint/cuda/conv2d.h"
#ifdef LIBWAIFU_CUDNN_ENABLED
#include "flint/cuda/conv2d_cudnn.h"
#endif
#include "flint/cuda/cuda_operators.h"
#include "flint/functional.h"
#include "flint/tensor.h"

namespace fl {
namespace {

constexpr int NumWarmup = 5;
constexpr int NumIterations = 20;

class CudaEvent {
 public:
  CudaEvent() { LL_CHECK_CUDA_STATUS(cudaEventCreate(&_event)); }

  ~CudaEvent() { cudaEventDestroy(_event); }

  operator cudaEvent_t() const { return _event; }

 private:
  cudaEvent_t _event;
};

template <typename Fn>
float benchmarkCuda(Fn &&fn) {
  for (int i = 0; i < NumWarmup; ++i) fn();
  LL_CHECK_CUDA_STATUS(cudaDeviceSynchronize());

  CudaEvent begin;
  CudaEvent end;
  LL_CHECK_CUDA_STATUS(cudaEventRecord(begin));
  for (int i = 0; i < NumIterations; ++i) fn();
  LL_CHECK_CUDA_STATUS(cudaEventRecord(end));
  LL_CHECK_CUDA_STATUS(cudaEventSynchronize(end));

  float totalMs;
  LL_CHECK_CUDA_STATUS(cudaEventElapsedTime(&totalMs, begin, end));
  return totalMs / NumIterations;
}

void printMatmul(const std::string &name, float milliseconds, int m, int n, int k) {
  double tflops = 2.0 * m * n * k / (milliseconds * 1.0e9);
  bench::print("%-44s %10.3f us  %8.2f TFLOP/s\n", name.c_str(), milliseconds * 1000.0f, tflops);
}

Tensor randHalf(const std::shared_ptr<Operators> &operators, std::initializer_list<int> shape) {
  return operators->rand(shape, DType::kFloat16);
}

/// The bytes an operator has to move at least once: what it reads plus what it writes. A norm or
/// an activation is held up by the bus rather than by the arithmetic, so this over the time is
/// the number worth reading, and the card's peak is what to read it against.
void printBandwidth(const std::string &name, float milliseconds, double bytes) {
  bench::print(
      "%-44s %10.3f us  %8.1f GB/s\n",
      name.c_str(),
      milliseconds * 1000.0f,
      bytes / (milliseconds * 1.0e6));
}

/// Elements times two, which is what half precision costs.
double halfBytes(std::initializer_list<int> shape) {
  double count = 1;
  for (int extent : shape) count *= extent;
  return 2.0 * count;
}

void benchmarkConv2d(
    const std::shared_ptr<Operators> &operators,
    const std::string &name,
    int batch,
    int inChannel,
    int outChannel,
    int size,
    int kernel,
    int stride) {
  int padding = kernel / 2;
  Tensor input = randHalf(operators, {batch, inChannel, size, size});
  Tensor weight = randHalf(operators, {outChannel, inChannel, kernel, kernel});
  Tensor bias = randHalf(operators, {outChannel});

  int outSize = (size + 2 * padding - kernel) / stride + 1;
  double flop = 2.0 * batch * outChannel * outSize * outSize * inChannel * kernel * kernel;

  float milliseconds = benchmarkCuda(
      [&] { op::cuda::conv2d(input, weight, bias, {stride, padding, 1, 1}); });
  std::string line = lut::sprintf("%-36s", name.c_str());
  line += lut::sprintf(" %10.1f us %10.2f", milliseconds * 1000.0f, flop / (milliseconds * 1.0e9));

#ifdef LIBWAIFU_CUDNN_ENABLED
  // The same convolution on cuDNN, in a column of its own, because a number for one
  // implementation says nothing on its own: what is worth knowing is whether the one that runs is
  // behind the one that does not. Left out rather than reported as zero where the library is not
  // on the machine, since a build having cuDNN and a machine having it are two different things.
  if (op::cuda::isConv2dCudnnAvailable()) {
    float reference = benchmarkCuda(
        [&] { op::cuda::conv2dCudnn(input, weight, bias, {stride, padding, 1, 1}); });
    line += lut::sprintf(" %10.1f us %10.2f", reference * 1000.0f, flop / (reference * 1.0e9));
  }
#endif  // LIBWAIFU_CUDNN_ENABLED

  bench::print("%s\n", line.c_str());
}

}  // namespace

LL_BENCHMARK(bench::Group::kSdxlCuda, "SDXL GEMM") {
  // Every GEMM shape the U-Net runs at 1024 by 1024. These ten are 99.6% of a step's
  // multiply-adds; the text encoders and the autoencoder are left out, being 0.3% of an image and
  // float32 besides.
  //
  // One line per shape and no total. A total would have to be a weighted sum, and a weighted sum
  // of these alone reads as what a step costs while leaving out the convolutions, the attention
  // and the norms -- a number wrong in a direction nobody checks.
  //
  // Both backends in the one run, side by side, rather than one per run under LIBWAIFU_GEMM:
  // which of them is ahead on a shape is the whole question, and two runs minutes apart on a
  // machine doing other things is a poor way to ask it.
  //
  // Read what it says with care: it runs one shape fifty times over, so a weight stays in L2 from
  // one iteration to the next, which a real run never gets. That flatters whatever moves the
  // least data per multiply-add, and it flattered a 64 by 64 CUTLASS tile into looking faster
  // than the 128 by 128 one it is a percent slower than on a whole image. Use it to see where a
  // shape stands, and a whole image to decide anything.
  if (!isOperatorsAvailable(Device::kCuda)) LL_BENCHMARK_SKIP("cuda device not available");

  // Made here rather than taken from getOperatorsSharedPtr, which has already settled on one.
  // Either may be missing -- a build without CUTLASS, a machine whose cuBLAS will not load -- and
  // a missing one leaves its column out rather than the benchmark.
  std::shared_ptr<Operators> cublas;
  std::shared_ptr<Operators> cutlass;
  try {
    cublas = op::cuda::CudaOperators::create(op::cuda::CudaOperators::OPT_CUBLAS_GEMM);
  } catch (const lut::Error &error) {
    bench::print("cuBLAS is not usable here: %s\n", error.what());
  }
  try {
    cutlass = op::cuda::CudaOperators::create(op::cuda::CudaOperators::OPT_CUTLASS_GEMM);
  } catch (const lut::Error &error) {
    bench::print("CUTLASS is not usable here: %s\n", error.what());
  }
  if (!cublas && !cutlass) LL_BENCHMARK_SKIP("neither GEMM backend is usable");

  std::shared_ptr<Operators> operators = cublas ? cublas : cutlass;

  std::string header = lut::sprintf("%-36s", "shape");
  if (cublas) header += lut::sprintf(" %13s %10s", "cublas", "TFLOP/s");
  if (cutlass) header += lut::sprintf(" %13s %10s", "cutlass", "TFLOP/s");
  bench::print("half precision\n%s\n", header.c_str());

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
    Tensor input = randHalf(operators, {shape.m, shape.k});
    Tensor weight = randHalf(operators, {shape.n, shape.k}).transpose(0, 1);
    double flop = 2.0 * shape.m * shape.n * shape.k;

    std::string line = lut::sprintf("%-36s", shape.what);
    for (const std::shared_ptr<Operators> &backend : {cublas, cutlass}) {
      if (!backend) continue;

      float milliseconds = benchmarkCuda([&] { backend->matmul(input, weight); });
      line += lut::sprintf(
          " %10.1f us %10.2f",
          milliseconds * 1000.0f,
          flop / (milliseconds * 1.0e9));
    }
    bench::print("%s\n", line.c_str());
  }
}

LL_BENCHMARK(bench::Group::kSdxlCuda, "SDXL elementwise") {
  if (!isOperatorsAvailable(Device::kCuda)) LL_BENCHMARK_SKIP("cuda device not available");
  std::shared_ptr<Operators> operators = getOperatorsSharedPtr(Device::kCuda);

  // Everything a step runs that is not a GEMM or a convolution. None of it is much arithmetic --
  // the GEMMs are 99.6% of the multiply-adds -- and all of it reads a whole tensor and writes
  // one, which on this card is 448 GB/s away from free. What the column says is how close to the
  // bus each of them gets.
  //
  // Per call, not per step: how many times a step runs each of these has not been counted the way
  // the GEMM table's counts were, so these cannot be added up into a step.
  // The same tensors twenty times over, so after the first pass they are in L2 and the column
  // reads above what the bus can do. Take it as a ceiling and a way of telling these apart from
  // each other, not as what they get on a whole image.
  bench::print("half precision, per call, warm in cache\n");

  // The three resolutions the U-Net works at, at a 1024 by 1024 image.
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
    Tensor x = randHalf(operators, {1, level.channels, level.size, level.size});
    Tensor scale = randHalf(operators, {level.channels});
    Tensor shift = randHalf(operators, {level.channels});
    double moved = 2 * halfBytes({1, level.channels, level.size, level.size});

    printBandwidth(
        std::string("group_norm   ") + level.what,
        benchmarkCuda([&] { F::groupNorm(x, scale, shift, 32, 1e-5f); }),
        moved);
    printBandwidth(
        std::string("silu         ") + level.what,
        benchmarkCuda([&] { F::silu(x); }),
        moved);

    // Three tensors rather than two: a residual reads both of its inputs.
    Tensor other = randHalf(operators, {1, level.channels, level.size, level.size});
    printBandwidth(
        std::string("add          ") + level.what,
        benchmarkCuda([&] { F::add(x, other); }),
        1.5 * moved);
  }

  // The transformer levels. SDXL attends at 64 by 64 and at 32 by 32 and not below, and a head is
  // 64 wide throughout, so the head count is the channel count over 64.
  struct Attention {
    const char *what;
    int tokens;
    int heads;
  };
  const Attention attentions[] = {{"64x64x640", 4096, 10}, {"32x32x1280", 1024, 20}};

  for (const Attention &attention : attentions) {
    int width = attention.heads * 64;
    Tensor hidden = randHalf(operators, {1, attention.tokens, width});
    Tensor scale = randHalf(operators, {width});
    Tensor shift = randHalf(operators, {width});
    printBandwidth(
        std::string("layer_norm   ") + attention.what,
        benchmarkCuda([&] { F::layerNorm(hidden, scale, shift, 1e-5f); }),
        2 * halfBytes({1, attention.tokens, width}));

    Tensor q = randHalf(operators, {1, attention.heads, attention.tokens, 64});
    Tensor k = randHalf(operators, {1, attention.heads, attention.tokens, 64});
    Tensor v = randHalf(operators, {1, attention.heads, attention.tokens, 64});
    double flop = 4.0 * attention.heads * attention.tokens * attention.tokens * 64;
    float milliseconds = benchmarkCuda([&] { F::attention(q, k, v, false); });
    bench::print(
        "%-44s %10.3f us  %8.2f TFLOP/s\n",
        (std::string("self attention ") + attention.what).c_str(),
        milliseconds * 1000.0f,
        flop / (milliseconds * 1.0e9));

    // Cross attention reads the prompt, which is 77 tokens however big the picture is.
    Tensor ck = randHalf(operators, {1, attention.heads, 77, 64});
    Tensor cv = randHalf(operators, {1, attention.heads, 77, 64});
    double crossFlop = 4.0 * attention.heads * attention.tokens * 77 * 64;
    milliseconds = benchmarkCuda([&] { F::attention(q, ck, cv, false); });
    bench::print(
        "%-44s %10.3f us  %8.2f TFLOP/s\n",
        (std::string("cross attention ") + attention.what).c_str(),
        milliseconds * 1000.0f,
        crossFlop / (milliseconds * 1.0e9));

    // The feed-forward gate, which halves what it reads.
    int inner = width * 4;
    Tensor gated = randHalf(operators, {1, attention.tokens, 2 * inner});
    printBandwidth(
        std::string("geglu        ") + attention.what,
        benchmarkCuda([&] { F::geglu(gated); }),
        1.5 * halfBytes({1, attention.tokens, 2 * inner}));
  }

  // Upsampling, which reads one pixel and writes four.
  const Level upsampled[] = {{"32x32x1280 to 64x64", 1280, 32}, {"64x64x640 to 128x128", 640, 64}};
  for (const Level &level : upsampled) {
    Tensor x = randHalf(operators, {1, level.channels, level.size, level.size});
    printBandwidth(
        std::string("upsample     ") + level.what,
        benchmarkCuda([&] { F::upsampleNearest2d(x, 2); }),
        5 * halfBytes({1, level.channels, level.size, level.size}));
  }
}

LL_BENCHMARK(bench::Group::kSdxlCuda, "SDXL convolution") {
  if (!isOperatorsAvailable(Device::kCuda)) LL_BENCHMARK_SKIP("cuda device not available");
  if (!op::cuda::isConv2dAvailable()) LL_BENCHMARK_SKIP("this build cannot convolve");
  std::shared_ptr<Operators> operators = getOperatorsSharedPtr(Device::kCuda);

  // SDXL shapes at a 1024 by 1024 image, whose latent is 128 by 128.
  //
  // The cuDNN column appears beside each one where the library is on the machine, which is what
  // says whether the kernels that actually run are behind the ones that do not. Its own case
  // rather than the tail of another, so that it can be run on its own and does not have to pay
  // for a page of unrelated allocations first.
  std::string header = lut::sprintf("%-36s %13s %10s", "shape", "cutlass", "TFLOP/s");
#ifdef LIBWAIFU_CUDNN_ENABLED
  if (op::cuda::isConv2dCudnnAvailable()) {
    header += lut::sprintf(" %13s %10s", "cudnn", "TFLOP/s");
  }
#endif  // LIBWAIFU_CUDNN_ENABLED
  bench::print("half precision, one group\n%s\n", header.c_str());
  benchmarkConv2d(operators, "unet 128x128x320 3x3", 1, 320, 320, 128, 3, 1);
  benchmarkConv2d(operators, "unet 64x64x640 3x3", 1, 640, 640, 64, 3, 1);
  benchmarkConv2d(operators, "unet 32x32x1280 3x3", 1, 1280, 1280, 32, 3, 1);
  benchmarkConv2d(operators, "unet 128x128x320 downsample 3x3/2", 1, 320, 320, 128, 3, 2);
  benchmarkConv2d(operators, "unet 64x64x640 1x1", 1, 640, 640, 64, 1, 1);
  benchmarkConv2d(operators, "vae 512x512x256 3x3", 1, 256, 256, 512, 3, 1);
}

}  // namespace fl
