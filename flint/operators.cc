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

#include <math.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "flint/cpu/cpu_operators.h"
#include "flint/cpu/kernel/interface.h"
#ifdef LIBWAIFU_CUDA_ENABLED
#include "flint/cuda/cuda_operators.h"
#endif
#include "flint/functional.h"
#ifdef LIBWAIFU_METAL_ENABLED
#include "flint/metal_native/metal_operators.h"
#endif
#ifdef LIBWAIFU_VULKAN_ENABLED
#include "flint/vulkan/vulkan_operators.h"
#endif

namespace fl {

namespace {

// --- Attention's composition ------------------------------------------------------------
//
// What Operators::attention runs on a device whose kernels do not take what it was handed.
// It is kept until every device has kernels of its own, and goes then with everything here.

using F::empty;
using F::emptyLike;

/// `input` <(B, KVH, L, D)> with each key-value head repeated for the query heads that share it,
/// so that grouped-query attention can be done as plain attention.
Tensor expandKeyValueHeads(Operators *op, TensorView input, int numHeads) {
  int batchSize = input.getShape(0);
  int numKeyValueHeads = input.getShape(1);
  int length = input.getShape(2);
  int headDim = input.getShape(3);
  int groupSize = numHeads / numKeyValueHeads;

  TensorView expanded =
      input.unsqueeze(2).expand({batchSize, numKeyValueHeads, groupSize, length, headDim});
  Tensor output = emptyLike(expanded);
  op->copy(expanded, output);

  return output.view({batchSize, numHeads, length, headDim});
}

/// `A` (..., m, k) times `B` (..., k, n), into a fresh tensor of `A`'s type on its device.
Tensor product(Operators *op, TensorView A, TensorView B) {
  std::vector<int> shape = A.getShape();
  shape.back() = B.getShape(-1);
  Tensor out = empty(A.getDevice(), shape, A.getDType());
  op->matmul(A, B, out);
  return out;
}

/// `input` times `scale`, into a fresh tensor like it.
Tensor scaled(Operators *op, TensorView input, float scale) {
  Tensor out = emptyLike(input);
  op->mul(input, scale, out);
  return out;
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

/// The mask Operators::attention was handed, refused where it is not one. See `MaskType`.
void checkMask(TensorView q, TensorView k, MaskType maskType, int blockSize) {
  int queries = q.getShape(2);
  int keys = k.getShape(2);
  switch (maskType) {
    case MaskType::kNone:
    case MaskType::kCausal:
      if (blockSize != 0) {
        THROW(
            InvalidArg,
            lut::sprintf("attention: a block size of %d without a block mask", blockSize));
      }
      return;
    case MaskType::kBlockCausal:
    case MaskType::kBlockDiagonal:
      if (blockSize <= 0) {
        THROW(InvalidArg, lut::sprintf("attention: a block mask of %d keys a block", blockSize));
      }
      if (maskType == MaskType::kBlockCausal && keys < queries) {
        THROW(
            InvalidArg,
            lut::sprintf(
                "attention: a block causal mask over %d keys and %d queries, which leaves the "
                "first queries before every key",
                keys,
                queries));
      }
      if (maskType == MaskType::kBlockDiagonal && keys != queries) {
        THROW(
            InvalidArg,
            lut::sprintf(
                "attention: a block diagonal mask is self-attention, and this is %d queries "
                "over %d keys",
                queries,
                keys));
      }
      return;
  }
  THROW(InvalidArg, lut::sprintf("attention: no mask type %d", static_cast<int>(maskType)));
}

void composedAttention(
    Operators *op,
    TensorView q,
    TensorView k,
    TensorView v,
    bool causal,
    TensorView out) {
  int numHeads = q.getShape(1);
  int numKeyValueHeads = k.getShape(1);
  int queryLength = q.getShape(2);
  int keyValueLength = k.getShape(2);
  int headDim = q.getShape(3);

  Tensor expandedK, expandedV;
  if (numHeads != numKeyValueHeads) {
    expandedK = expandKeyValueHeads(op, k, numHeads);
    expandedV = expandKeyValueHeads(op, v, numHeads);
    k = expandedK;
    v = expandedV;
  }

  // Scaling both q and k keeps the scores in range for half precision.
  float scale = sqrtf(1.0f / sqrtf(1.0f * headDim));
  Tensor scaledK = scaled(op, k, scale).transpose(-2, -1);

  // The score matrix is the whole cost of doing it this way: a VAE decoding a 1024 by 1024 image
  // attends over 16384 positions, and holding all of that at once is half a gigabyte before the
  // softmax needs a second copy. Since each output row depends only on its own row of scores, the
  // queries are taken a block at a time instead once that gets out of hand. The answer is the
  // same to the bit -- no running maximum is needed, because a block still sees every key.
  //
  // What a row of queries costs is a row of scores in every head of every sequence, so that is
  // what the budget is divided by.
  int64_t scoresPerQuery = static_cast<int64_t>(q.getShape(0)) * numHeads * keyValueLength;

  // What the budget affords is rounded down to a power of two rather than taken where the
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

  // The probabilities of one block of queries, against every key.
  auto probabilities = [&](TensorView blockQ, int begin, int end) {
    Tensor scores = product(op, scaled(op, blockQ, scale), scaledK);
    // A single query attends to the whole history, so it needs no mask. A block of them is masked
    // against where it sits, not where the whole query is.
    if (causal && queryLength > 1) {
      Tensor fullMask = empty(
          q.getDevice(), {keyValueLength, keyValueLength}, op->getDefaultFloatType());
      op->causalMask(fullMask);
      TensorView mask = TensorView(fullMask).slice(
          0, {keyValueLength - queryLength + begin, keyValueLength - queryLength + end});

      // The mask comes in the device's *default* float type, which is not always the type the
      // scores are in: on aarch64 that default is half, and a model running in float32 has
      // float32 scores.
      Tensor typedMask;
      if (mask.getDType() != scores.getDType()) {
        typedMask = empty(q.getDevice(), mask.getShape(), scores.getDType());
        op->cast(mask, typedMask);
        mask = typedMask;
      }

      // One mask for every sequence and head: seen at the scores' shape by repeating it.
      while (mask.getDim() < scores.getDim()) mask = mask.unsqueeze(0);
      std::vector<int> scoreShape = scores.getShape();
      Tensor masked = emptyLike(scores);
      op->add(scores, mask.expand(scoreShape), masked);
      scores = masked;
    }
    Tensor probabilities = emptyLike(scores);
    op->softmax(scores, probabilities);
    return probabilities;
  };

  if (blockSize >= queryLength) {
    op->matmul(probabilities(q, 0, queryLength), v, out);
    return;
  }

  // Each block's answer is written into the rows of `out` it belongs to. Joining the blocks as they
  // arrive would copy everything finished so far on every pass, which is quadratic in the number
  // of blocks: the VAE's 64 of them spent 1.7ms of an 18ms call doing nothing else.
  for (int begin = 0; begin < queryLength; begin += blockSize) {
    int end = std::min(begin + blockSize, queryLength);
    Tensor blockOutput = product(op, probabilities(q.slice(-2, {begin, end}), begin, end), v);
    op->copy(blockOutput, out.slice(-2, {begin, end}));
  }
}

}  // namespace


// What a device that has no kernel for something says when asked for it: a NotImplementedError
// naming the operator, which the C API hands back as an error rather than letting it abort the
// process. A missing kernel is a fact about the device and not a broken invariant, so whoever is
// driving it -- a server running one job of many -- gets to fail that job and carry on.
#define NO_KERNEL() THROW(NotImplemented, std::string(__func__) + " has no kernel on this device")

void Operators::arangeLong(LongType begin, LongType step, TensorView out) {
  NO_KERNEL();
}

void Operators::lookup(TensorView table, TensorView indices, TensorView out) {
  NO_KERNEL();
}

void Operators::rotaryEmbedding(
    TensorView positions,
    TensorView query,
    TensorView key,
    TensorView rotaryCache) {
  NO_KERNEL();
}

void Operators::rmsNorm(TensorView input, TensorView weight, float eps, TensorView out) {
  NO_KERNEL();
}

void Operators::layerNorm(
    TensorView input,
    TensorView weight,
    TensorView bias,
    float eps,
    TensorView out) {
  THROW(NotImplemented, "layerNorm is only available on the CUDA device");
}

void Operators::groupNorm(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int groups,
    float eps,
    TensorView out) {
  THROW(NotImplemented, "groupNorm is only available on the CUDA device");
}

void Operators::upsampleNearest2d(TensorView input, int scale, TensorView out) {
  THROW(NotImplemented, "upsampleNearest2d is only available on the CUDA device");
}

void Operators::upsampleNearest1d(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::geglu(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::matmul(TensorView A, TensorView B, TensorView out) {
  NO_KERNEL();
}

void Operators::conv2d(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    TensorView out) {
  THROW(NotImplemented, "conv2d is only available on the CUDA device, in a build with cuDNN");
}

void Operators::conv1d(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    TensorView out) {
  THROW(
      NotImplemented,
      "conv1d is not available on this device; the CPU, CUDA and Metal all have it");
}

void Operators::convTranspose1d(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int outputPadding,
    int groups,
    TensorView out) {
  F::composedConvTranspose1d(this, input, weight, bias, stride, padding, outputPadding, groups, out);
}

void Operators::snake(
    TensorView input,
    TensorView alpha,
    TensorView beta,
    float eps,
    TensorView out) {
  F::composedSnake(this, input, alpha, beta, eps, out);
}

void Operators::pad1d(TensorView input, int left, int right, PadMode mode, TensorView out) {
  F::composedPad1d(this, input, left, right, mode, out);
}

void Operators::clamp(TensorView input, float lo, float hi, TensorView out) {
  F::composedClamp(this, input, lo, hi, out);
}

void Operators::stft(
    TensorView input,
    TensorView window,
    int nFft,
    int hop,
    bool centered,
    TensorView out) {
  F::composedStft(this, input, window, nFft, hop, centered, out);
}

void Operators::istft(
    TensorView spectrum,
    TensorView window,
    int nFft,
    int hop,
    bool centered,
    TensorView out) {
  F::composedIstft(this, spectrum, window, nFft, hop, centered, out);
}

void Operators::mul(TensorView input, float other, TensorView out) {
  NO_KERNEL();
}

void Operators::div(TensorView input, float other, TensorView out) {
  NO_KERNEL();
}

void Operators::mod(TensorView input, LongType other, TensorView out) {
  NO_KERNEL();
}

void Operators::mul(TensorView input, TensorView other, TensorView out) {
  NO_KERNEL();
}

void Operators::softmax(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::attention(
    TensorView q,
    TensorView k,
    TensorView v,
    MaskType maskType,
    int blockSize,
    TensorView out) {
  checkMask(q, k, maskType, blockSize);
  if (maskType == MaskType::kBlockCausal || maskType == MaskType::kBlockDiagonal) {
    THROW(
        NotImplemented,
        lut::sprintf(
            "attention with a block %s mask has no kernel on this device for these operands, "
            "and no composition",
            maskType == MaskType::kBlockCausal ? "causal" : "diagonal"));
  }
  composedAttention(this, q, k, v, maskType == MaskType::kCausal, out);
}

void Operators::pagedAttention(
    TensorView q,
    TensorView keyCache,
    TensorView valueCache,
    TensorView blockTable,
    TensorView cuSeqlensQ,
    TensorView seqlensK,
    int maxQLen,
    int maxKLen,
    bool causal,
    TensorView out) {
  NO_KERNEL();
}

void Operators::storeKVCache(
    TensorView k,
    TensorView v,
    TensorView keyCache,
    TensorView valueCache,
    TensorView slotMapping) {
  NO_KERNEL();
}

void Operators::gatedDeltaNetPrefill(
    TensorView q,
    TensorView k,
    TensorView v,
    TensorView g,
    TensorView beta,
    TensorView cuSeqlens,
    TensorView stateSlots,
    TensorView state,
    TensorView out) {
  NO_KERNEL();
}

void Operators::sample(
    TensorView logits,
    TensorView temperatures,
    TensorView topKs,
    TensorView topPs,
    TensorView out) {
  NO_KERNEL();
}

void Operators::add(TensorView input, TensorView other, TensorView out) {
  NO_KERNEL();
}

void Operators::sub(TensorView input, TensorView other, TensorView out) {
  NO_KERNEL();
}

void Operators::sum(TensorView input, int dim, TensorView out) {
  NO_KERNEL();
}

void Operators::cumsum(TensorView input, int dim, TensorView out) {
  NO_KERNEL();
}

void Operators::max(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::eq(TensorView input, TensorView other, TensorView out) {
  NO_KERNEL();
}

void Operators::square(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::min(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::divTensor(TensorView input, TensorView other, TensorView out) {
  NO_KERNEL();
}

void Operators::neg(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::abs(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::exp(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::log(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::round(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::sqrt(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::rsqrt(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::sigmoid(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::tanh(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::relu(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::gelu(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::silu(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::sin(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::cos(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::quickGelu(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::fill(TensorView input, float value) {
  NO_KERNEL();
}

bool Operators::all(TensorView A) {
  NO_KERNEL();
}

bool Operators::allClose(TensorView A, TensorView B, float rtol, float atol) {
  NO_KERNEL();
}

void Operators::print(TensorView tensor) {
  NO_KERNEL();
}

void Operators::causalMask(TensorView out) {
  NO_KERNEL();
}

void Operators::copy(TensorView src, TensorView dest) {
  NO_KERNEL();
}

void Operators::swiglu(TensorView A, TensorView out) {
  NO_KERNEL();
}

void Operators::transfer(TensorView src, TensorView dest) {
  NO_KERNEL();
}

float Operators::elem(TensorView tensor) {
  NO_KERNEL();
}

bool Operators::elemBool(TensorView tensor) {
  NO_KERNEL();
}

void Operators::repetitionPenalty(TensorView logits, TensorView history, float weight) {
  NO_KERNEL();
}

void Operators::cast(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::rand(TensorView out) {
  NO_KERNEL();
}

void Operators::randNormal(TensorView out) {
  NO_KERNEL();
}

void Operators::manualSeed(uint64_t seed) {
  NO_KERNEL();
}

MemorySnapshot Operators::captureMemorySnapshot() {
  NO_KERNEL();
}

void Operators::resetPeakMemoryStats() {
  NO_KERNEL();
}

// Not NO_KERNEL(): a device whose allocator holds nothing back has nothing to hand over, and has
// answered this question by already having done it.
void Operators::releaseUnusedMemory() {
}

void Operators::setMemoryLimit(int64_t bytes) {
  NO_KERNEL();
}

MemoryBudget Operators::getMemoryBudget() {
  NO_KERNEL();
}

DType Operators::getDefaultFloatType() {
  NO_KERNEL();
}

void Operators::synchronize() {
}

std::shared_ptr<Operators> gOperatorsForDevice[Device::NumDeviceType] = {
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr};

static std::atomic<bool> gInitialized{false};

#ifdef _OPENMP
// The thread that loads flint -- the main thread, for a program, a test binary or an app that
// links it -- made a root of the OpenMP runtime for as long as it lives, by asking it something
// that registers the asker (omp_get_max_threads does; not every omp_ call does).
//
// A libomp linked in statically, which is what the macOS build makes from LLVM's source, shuts
// the runtime down when the last root thread exits (__kmp_internal_end_thread; the shared library
// leaves that to its destructor instead), and starts it again for the next thread that asks.
// Threads that come and go while others start -- each test of a binary on a thread of its own,
// an engine thread replaced by another -- then catch a shutdown halfway through a start, and the
// process aborts in ___kmp_free (kmp_alloc.cpp). A root that never exits means the last root
// never does either. waifu/tests/openmp.rs is the case.
//
// Here, in the file initOperators is in, because the linker leaves out an object of a static
// library that nothing refers to -- and its static initializers with it.
static const int gOpenMPRootThread = omp_get_max_threads();
#endif

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
    gOperatorsForDevice[Device::kCuda] = op::cuda::CudaOperators::create();
#endif
#ifdef LIBWAIFU_METAL_ENABLED
    // Unlike CUDA, a build with Metal still has to cope with there being no GPU it can run on,
    // so the operators are only registered where there is one.
    if (op::metal_native::MetalOperators::isAvailable()) {
      CHECK(!gOperatorsForDevice[Device::kMetal]);
      gOperatorsForDevice[Device::kMetal] = op::metal_native::MetalOperators::create();
    }
#endif
#ifdef LIBWAIFU_VULKAN_ENABLED
    // Registered the same way, only where there is a device: a Vulkan build loads the loader
    // itself, and so runs on machines that have none.
    if (op::vulkan::VulkanOperators::isAvailable()) {
      CHECK(!gOperatorsForDevice[Device::kVulkan]);
      gOperatorsForDevice[Device::kVulkan] = op::vulkan::VulkanOperators::create();
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
