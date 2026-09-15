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

#include "flint/operators.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "flint/cpu/cpu_operators.h"
#include "flint/cpu/kernel/interface.h"
#include "flint/cuda/cuda_operators.h"
#ifdef LIBWAIFU_MLX_ENABLED
#include "flint/metal/metal_operators.h"
#endif

namespace fl {

namespace {

Tensor expandKeyValueHeads(Operators *op, Tensor input, int numHeads) {
  int batchSize = input.getShape(0);
  int numKeyValueHeads = input.getShape(1);
  int length = input.getShape(2);
  int headDim = input.getShape(3);
  int groupSize = numHeads / numKeyValueHeads;

  Tensor expanded =
      input.unsqueeze(2).expand({batchSize, numKeyValueHeads, groupSize, length, headDim});
  Tensor output = op->tensorLike(expanded);
  op->copy(expanded, output);

  return output.view({batchSize, numHeads, length, headDim});
}

}  // namespace

Tensor Operators::arangeLong(LongType begin, LongType end, LongType step) {
  NOT_IMPL();
}

Tensor Operators::lookup(Tensor table, Tensor indices) {
  NOT_IMPL();
}

Tensor Operators::matmul(Tensor a, Tensor b) {
  NOT_IMPL();
}

Tensor Operators::matmulNarrowPrecision(Tensor A, Tensor sfA, Tensor B, Tensor sfB) {
  NOT_IMPL();
}

Tensor Operators::layerNorm(Tensor input, Tensor weight, Tensor bias, float eps) {
  THROW(NotImplemented, "layerNorm is only available on the CUDA device");
  return Tensor();
}

Tensor Operators::groupNorm(Tensor input, Tensor weight, Tensor bias, int groups, float eps) {
  THROW(NotImplemented, "groupNorm is only available on the CUDA device");
  return Tensor();
}

Tensor Operators::upsampleNearest2d(Tensor input, int scale) {
  THROW(NotImplemented, "upsampleNearest2d is only available on the CUDA device");
  return Tensor();
}

Tensor Operators::geglu(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::conv2d(
    Tensor input,
    Tensor weight,
    Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
  // A device that has no convolution is something a caller can work around, so it is told rather
  // than killed. NOT_IMPL() aborts, which is right for a case nobody can act on and wrong here.
  THROW(NotImplemented, "conv2d is only available on the CUDA device, in a build with cuDNN");
  return Tensor();
}

Tensor Operators::gatedDeltaNetPrefill(
    Tensor q,
    Tensor k,
    Tensor v,
    Tensor g,
    Tensor beta,
    Tensor cuSeqlens,
    Tensor stateSlots,
    Tensor state) {
  NOT_IMPL();
}

Tensor Operators::mul(Tensor input, float other) {
  NOT_IMPL();
}

Tensor Operators::div(Tensor input, float other) {
  NOT_IMPL();
}

Tensor Operators::mod(Tensor input, LongType other) {
  NOT_IMPL();
}

Tensor Operators::eq(Tensor input, Tensor other) {
  NOT_IMPL();
}

Tensor Operators::mul(Tensor input, Tensor other) {
  NOT_IMPL();
}

Tensor Operators::softmax(Tensor input) {
  NOT_IMPL();
}

/// How many score elements one block of queries may hold at once, counting every head and every
/// sequence in the batch -- the whole tensor the two matmuls hand each other, not one head's
/// share of it. 128M of them is 256MB in half, and the softmax needs a second copy of it.
///
/// This is a bound on memory and nothing else. Blocking was measured on every shape SDXL runs and
/// a bigger block was faster every time, cache or no cache: the score matrix is streamed once
/// either way, and what the block size really moves is which kernel cuBLAS picks for the second
/// matmul, which is not something to steer by. So the budget is set where it stops an allocation
/// nobody meant to make, and left clear of everything that runs.
constexpr int64_t kAttentionScoreLimit = 128 * 1024 * 1024;

Tensor Operators::attention(Tensor q, Tensor k, Tensor v, bool causal) {
  CHECK(q.getDim() == 4 && k.getDim() == 4 && v.getDim() == 4);

  int numHeads = q.getShape(1);
  int numKeyValueHeads = k.getShape(1);
  int queryLength = q.getShape(2);
  int keyValueLength = k.getShape(2);
  int headDim = q.getShape(3);
  CHECK(numHeads % numKeyValueHeads == 0);

  if (numHeads != numKeyValueHeads) {
    k = expandKeyValueHeads(this, k, numHeads);
    v = expandKeyValueHeads(this, v, numHeads);
  }

  // Scaling both q and k keeps the scores in range for half precision.
  float scale = sqrtf(1.0f / sqrtf(1.0f * headDim));
  Tensor scaledK = mul(k, scale).transpose(-2, -1);

  // The score matrix is the whole cost of doing it this way: a VAE decoding a 1024 by 1024 image
  // attends over 16384 positions, and holding all of that at once is half a gigabyte before the
  // softmax needs a second copy. Since each output row depends only on its own row of scores, the
  // queries are taken a block at a time instead once that gets out of hand. The answer is the
  // same to the bit -- no running maximum is needed, because a block still sees every key.
  //
  // What a row of queries costs is a row of scores in every head of every sequence, so that is
  // what the budget is divided by. Dividing by the key length alone, as this once did, left the
  // heads out, so the thing it called a limit was not one: ten heads held ten times it, and forty
  // would have held forty.
  int64_t scoresPerQuery = static_cast<int64_t>(q.getShape(0)) * numHeads * keyValueLength;

  // What the budget affords is then rounded down to a power of two rather than taken where the
  // division landed. The second matmul takes its kernel from the number of query rows handed to
  // it, and the choice is not monotone in that number: 640 rows of scores against the values
  // measured 837us where 512 measured 219, for a quarter more work. The powers of two were
  // uniformly among the good ones.
  int blockSize = queryLength;
  if (scoresPerQuery * queryLength > kAttentionScoreLimit) {
    int64_t budget = kAttentionScoreLimit / scoresPerQuery;
    blockSize = 1;
    while (blockSize * 2 <= budget) blockSize *= 2;
  }

  auto attendBlock = [&](Tensor blockQ, int begin, int end) {
    Tensor scores = matmul(mul(blockQ, scale), scaledK);

    // A single query attends to the whole history, so it needs no mask. A block of them is masked
    // against where it sits, not where the whole query is.
    if (causal && queryLength > 1) {
      Tensor mask = causalMask(keyValueLength)
                        .slice(0, {keyValueLength - queryLength + begin,
                                   keyValueLength - queryLength + end});
      scores = add(scores, mask);
    }

    return matmul(softmax(scores), v);
  };

  if (blockSize >= queryLength) return attendBlock(q, 0, queryLength);

  // The answer is made once and each block written into the rows it belongs to. Joining the
  // blocks as they arrive copies everything finished so far on every pass, which is quadratic in
  // the number of blocks: the VAE's 64 of them spent 1.7ms of an 18ms call doing nothing else.
  Tensor output = tensor({q.getShape(0), numHeads, queryLength, headDim}, q.getDType());
  for (int begin = 0; begin < queryLength; begin += blockSize) {
    int end = std::min(begin + blockSize, queryLength);
    Tensor blockOutput = attendBlock(q.slice(-2, {begin, end}), begin, end);
    Tensor destination = output.slice(-2, {begin, end});
    copy(blockOutput, destination);
  }

  return output;
}

Tensor Operators::sum(Tensor input, int dim) {
  NOT_IMPL();
}

Tensor Operators::max(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::square(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::min(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::divTensor(Tensor input, Tensor other) {
  NOT_IMPL();
}

Tensor Operators::neg(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::abs(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::exp(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::sqrt(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::rsqrt(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::sigmoid(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::tanh(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::relu(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::gelu(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::silu(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::sin(Tensor input) {
  NOT_IMPL();
}
Tensor Operators::cos(Tensor input) {
  NOT_IMPL();
}
Tensor Operators::quickGelu(Tensor input) {
  NOT_IMPL();
}

void Operators::fill(Tensor input, float value) {
  NOT_IMPL();
}

Tensor Operators::add(Tensor a, Tensor b) {
  NOT_IMPL();
}

Tensor Operators::sub(Tensor a, Tensor b) {
  NOT_IMPL();
}

Tensor Operators::subFloat(Tensor input, float other) {
  NOT_IMPL();
}

float Operators::elem(Tensor tensor) {
  NOT_IMPL();
}

bool Operators::elemBool(Tensor tensor) {
  NOT_IMPL();
}

Tensor Operators::tensor(lut::Span<const int> shape, DType dtype) {
  NOT_IMPL();
}

Tensor Operators::hostTensor(lut::Span<const int> shape, DType dtype) {
  NOT_IMPL();
}

Tensor Operators::tensorLike(Tensor input) {
  NOT_IMPL();
}

Tensor Operators::zeros(lut::Span<const int> shape, DType dtype) {
  NOT_IMPL();
}

MemorySnapshot Operators::captureMemorySnapshot() {
  NOT_IMPL();
}

void Operators::resetPeakMemoryStats() {
  NOT_IMPL();
}

bool Operators::allClose(Tensor A, Tensor B, float rtol, float atol) {
  NOT_IMPL();
}

bool Operators::all(Tensor A) {
  NOT_IMPL();
}

void Operators::print(Tensor tensor) {
  NOT_IMPL();
}

Tensor Operators::rmsNorm(Tensor input, Tensor weight, float eps) {
  NOT_IMPL();
}

void Operators::rotaryEmbedding(Tensor positions, Tensor query, Tensor key, Tensor rotaryCache) {
  NOT_IMPL();
}

Tensor Operators::pagedAttention(
    Tensor q,
    Tensor keyCache,
    Tensor valueCache,
    Tensor blockTable,
    Tensor cuSeqlensQ,
    Tensor seqlensK,
    int maxQLen,
    int maxKLen,
    bool causal) {
  NOT_IMPL();
}

void Operators::storeKVCache(
    Tensor k,
    Tensor v,
    Tensor keyCache,
    Tensor valueCache,
    Tensor slotMapping) {
  NOT_IMPL();
}

Tensor Operators::sample(Tensor logits, Tensor temperatures, Tensor topKs, Tensor topPs) {
  NOT_IMPL();
}

Tensor Operators::causalMask(int max_len) {
  NOT_IMPL();
}

void Operators::copy(Tensor src, Tensor dest) {
  NOT_IMPL();
}

Tensor Operators::swiglu(Tensor A) {
  NOT_IMPL();
}

Tensor Operators::toDevice(Device device, Tensor tensor) {
  NOT_IMPL();
}

void Operators::repetitionPenalty(Tensor logits, Tensor history, float weight) {
  NOT_IMPL();
}

Tensor Operators::cast(Tensor tensor, DType dtype) {
  NOT_IMPL();
}

DType Operators::getDefaultFloatType() {
  NOT_IMPL();
}

void Operators::synchronize() {
}

void Operators::manualSeed(uint64_t seed) {
  NOT_IMPL();
}

Tensor Operators::rand(lut::Span<const int> shape, DType dtype) {
  NOT_IMPL();
}

Tensor Operators::randNormal(lut::Span<const int> shape) {
  NOT_IMPL();
}

// One per Device::Type, and cuda-host is deliberately left empty: it names memory rather than a
// processor, so every operator asked for on it should report that it is not implemented.
std::shared_ptr<Operators> gOperatorsForDevice[Device::NumDeviceType] = {
    nullptr,
    nullptr,
    nullptr,
    nullptr};

static std::atomic<bool> gInitialized{false};

namespace {

/// Which GEMM backend to build the CUDA operators with, read from LIBWAIFU_GEMM.
///
/// The two are picked between at startup and never mixed, which is what makes one run comparable
/// with another. Left unset, the operators choose for themselves, which is cuBLAS wherever it can
/// be loaded.
int gemmOptionsFromEnvironment() {
#ifdef LIBWAIFU_CUDA_ENABLED
  const char *choice = std::getenv("LIBWAIFU_GEMM");
  if (!choice) return 0;

  std::string name = choice;
  if (name == "cublas") {
    LOG(INFO) << "LIBWAIFU_GEMM=cublas";
    return op::cuda::CudaOperators::OPT_CUBLAS_GEMM;
  }
  if (name == "cutlass") {
    LOG(INFO) << "LIBWAIFU_GEMM=cutlass";
    return op::cuda::CudaOperators::OPT_CUTLASS_GEMM;
  }

  // Naming a backend that is not there should say so rather than quietly measure the other one.
  throw lut::AbortedError(
      lut::sprintf("LIBWAIFU_GEMM is \"%s\", which is neither cublas nor cutlass", choice));
#else
  return 0;
#endif
}

}  // namespace

void initOperators() {
  op::cpu::kernel::init();

#ifdef _OPENMP
  LOG(INFO) << "OMP max_threads = " << omp_get_max_threads();
#endif

  if (!gInitialized.exchange(true)) {
    CHECK(!gOperatorsForDevice[Device::kCpu]);
    gOperatorsForDevice[Device::kCpu] = std::make_shared<op::cpu::CPUOperators>();
#ifdef LIBWAIFU_CUDA_ENABLED
    CHECK(!gOperatorsForDevice[Device::kCuda]);
    gOperatorsForDevice[Device::kCuda] =
        op::cuda::CudaOperators::create(gemmOptionsFromEnvironment());
#endif
#ifdef LIBWAIFU_MLX_ENABLED
    // Unlike CUDA, a build with MLX still has to cope with there being no GPU to talk to, so
    // the operators are only registered when MLX can actually reach one.
    if (op::metal::MetalOperators::isAvailable()) {
      CHECK(!gOperatorsForDevice[Device::kMetal]);
      gOperatorsForDevice[Device::kMetal] = op::metal::MetalOperators::create();
    }
#endif
  }
}

Operators *getOperators(Device::Type deviceType) {
  if (!gInitialized) throw lut::AbortedError("call getOperators() before initialization");
  if (!gOperatorsForDevice[deviceType]) {
    std::string deviceName = Device(deviceType).getName();
    throw lut::NotImplementedError(lut::sprintf("%s operators not implemented", deviceName));
  }

  return gOperatorsForDevice[deviceType].get();
}

std::shared_ptr<Operators> getOperatorsSharedPtr(Device::Type deviceType) {
  if (!gInitialized) throw lut::AbortedError("call getOperators() before initialization");
  if (!gOperatorsForDevice[deviceType]) {
    std::string deviceName = Device(deviceType).getName();
    throw lut::NotImplementedError(lut::sprintf("%s operators not implemented", deviceName));
  }

  return gOperatorsForDevice[deviceType];
}

bool isOperatorsAvailable(Device::Type deviceType) {
  if (!gInitialized) throw lut::AbortedError("call isOperatorsAvailable() before initialization");
  if (!gOperatorsForDevice[deviceType]) {
    return false;
  } else {
    return true;
  }
}

void destroyOperators() {
  op::cpu::kernel::destroy();

  if (gInitialized.exchange(false)) {
    for (int i = 0; i < Device::NumDeviceType; ++i) {
      gOperatorsForDevice[i] = nullptr;
    }
  }
}

}  // namespace fl
