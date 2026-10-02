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

#include "flint/functional.h"

#include <math.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "lutil/error.h"
#include "lutil/strings.h"
#include "flint/cpu/cpu_tensor_data.h"
#include "flint/operators.h"
#ifdef LIBWAIFU_CUDA_ENABLED
#include "flint/cuda/cuda_host_tensor_data.h"
#include "flint/cuda/cuda_tensor_data.h"
#endif
#ifdef LIBWAIFU_MLX_ENABLED
#include "flint/metal/metal_tensor_data.h"
#endif
#ifdef LIBWAIFU_VULKAN_ENABLED
#include "flint/vulkan/vulkan_tensor_data.h"
#endif

namespace fl {
namespace F {

namespace {

int64_t numelOf(lut::Span<const int> shape) {
  int64_t numel = 1;
  for (int size : shape) {
    if (size < 0) THROW(InvalidArg, lut::sprintf("a tensor cannot have a dimension of %d", size));
    numel *= size;
  }
  return numel;
}

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

}  // namespace

// --- Allocation -----------------------------------------------------------------------------

std::unique_ptr<TensorData> allocate(Device device, int64_t numel, DType dtype) {
  if (numel > TensorData::MaxNumEl) {
    THROW(InvalidArg, lut::sprintf("a tensor of %d elements is too large", numel));
  }
  numel = std::max<int64_t>(numel, 1);

  switch (device.getType()) {
    case Device::kCpu:
      return op::cpu::CpuTensorData::create(numel, dtype);
#ifdef LIBWAIFU_CUDA_ENABLED
    case Device::kCuda:
      return op::cuda::CudaTensorData::create(numel, dtype);
    case Device::kCudaHost:
      return op::cuda::CudaHostTensorData::create(numel, dtype);
#endif
#ifdef LIBWAIFU_MLX_ENABLED
    case Device::kMetal:
      return op::metal::MetalTensorData::create(numel, dtype);
#endif
#ifdef LIBWAIFU_VULKAN_ENABLED
    case Device::kVulkan:
      return op::vulkan::VulkanTensorData::create(numel, dtype);
#endif
    default:
      THROW(
          NotImplemented,
          lut::sprintf("this build cannot allocate on the %s device", device.getName()));
  }
}

Tensor empty(Device device, lut::Span<const int> shape, DType dtype) {
  int64_t numel = numelOf(shape);
  return Tensor::create(std::make_shared<TensorShape>(shape), allocate(device, numel, dtype));
}

Tensor emptyLike(const TensorView &input) {
  std::vector<int> shape = input.getShape();
  return empty(input.getDevice(), shape, input.getDType());
}

// --- Attention ------------------------------------------------------------------------------

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

}  // namespace F
}  // namespace fl
