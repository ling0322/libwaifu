// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Exercises CudaOperators::attention (cuda_operators.cc), which uses FlashAttention when it is
// compiled in and falls back to the portable operator otherwise -- unlike flash_attn_test.cc,
// this file is built whenever WITH_CUDA is on so both paths stay covered.

#include "catch2/catch_amalgamated.hpp"
#include "flint/test_functional.h"
#include "flint/cuda/attention.h"
#include "flint/device.h"
#include "flint/operators.h"

namespace fl {

namespace {

/// The CUDA operators, which the calls here that run on CUDA are asked of.
Operators *cudaOps() {
  return getOperators(Device::kCuda);
}

/// The CPU operators, which the calls here that run on CPU are asked of.
Operators *cpuOps() {
  return getOperators(Device::kCpu);
}

Tensor toCuda(const Tensor &a) {
  return F::cast(cudaOps(), F::toDevice(cudaOps(), a, Device::getCuda()), DType::kFloat16);
}

Tensor toCpu(const Tensor &a) {
  return F::toDevice(cudaOps(), F::cast(cudaOps(), a, DType::kFloat), Device::getCpu());
}

}  // namespace

CATCH_TEST_CASE("test CUDA attention", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  auto runCase = [](int numHeads,
                    int numKeyValueHeads,
                    int queryLength,
                    int keyValueLength,
                    int headDim,
                    bool causal) {
    // a model feeds attention with [batch, length, heads, headDim] transposed to
    // [batch, heads, length, headDim], so the inputs are not contiguous.
    Tensor q = F::rand(Device::getCpu(), {1, queryLength, numHeads, headDim}, DType::kFloat);
    Tensor k = F::rand(Device::getCpu(), {1, keyValueLength, numKeyValueHeads, headDim}, DType::kFloat);
    Tensor v = F::rand(Device::getCpu(), {1, keyValueLength, numKeyValueHeads, headDim}, DType::kFloat);
    Tensor xr =
      F::attention(cpuOps(), q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal);

    Tensor x = F::attention(cudaOps(), 
        toCuda(q).transpose(1, 2),
        toCuda(k).transpose(1, 2),
        toCuda(v).transpose(1, 2),
        causal);

    return cpuOps()->allClose(toCpu(x), xr, 5e-3f);
  };

  // headDim 128 goes to FlashAttention.
  CATCH_REQUIRE(runCase(8, 8, 8, 8, 128, false));
  CATCH_REQUIRE(runCase(8, 8, 128, 128, 128, false));
  CATCH_REQUIRE(runCase(8, 8, 8, 8, 128, true));
  CATCH_REQUIRE(runCase(8, 2, 8, 8, 128, false));
  CATCH_REQUIRE(runCase(8, 2, 6, 10, 128, true));
  CATCH_REQUIRE(runCase(8, 2, 1, 10, 128, true));

  // long enough to make the operator split the keys.
  CATCH_REQUIRE(runCase(8, 2, 1, 2048, 128, true));
  CATCH_REQUIRE(runCase(8, 2, 4, 1024, 128, true));

  // headDim 16 is unsupported by FlashAttention and falls back to the portable path.
  CATCH_REQUIRE(runCase(8, 8, 16, 16, 16, false));
  CATCH_REQUIRE(runCase(8, 2, 6, 10, 16, true));
}

namespace {

bool runAttentionCase(
    int numHeads,
    int numKeyValueHeads,
    int queryLength,
    int keyValueLength,
    int headDim,
    bool causal) {
  Tensor q = F::rand(Device::getCpu(), {1, queryLength, numHeads, headDim}, DType::kFloat);
  Tensor k = F::rand(Device::getCpu(), {1, keyValueLength, numKeyValueHeads, headDim}, DType::kFloat);
  Tensor v = F::rand(Device::getCpu(), {1, keyValueLength, numKeyValueHeads, headDim}, DType::kFloat);
  Tensor xr =
      F::attention(cpuOps(), q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal);

  Tensor x = F::attention(cudaOps(), 
      toCuda(q).transpose(1, 2),
      toCuda(k).transpose(1, 2),
      toCuda(v).transpose(1, 2),
      causal);

  return cpuOps()->allClose(toCpu(x), xr, 5e-3f);
}

}  // namespace

/// Attention written out, with no blocking anywhere: the reference the blocked fallback has to
/// reproduce. Kept small enough that the whole score matrix is affordable here.
Tensor wholeAttention(const Tensor &q, const Tensor &k, const Tensor &v, bool causal) {
  int headDim = q.getShape(3);
  int queryLength = q.getShape(2);
  int keyValueLength = k.getShape(2);

  float scale = sqrtf(1.0f / sqrtf(1.0f * headDim));
  Tensor scores =
      F::matmul(cudaOps(), F::mul(cudaOps(), q, scale), F::mul(cudaOps(), k, scale).transpose(-2, -1));
  if (causal && queryLength > 1) {
    Tensor mask = F::causalMask(Device::getCuda(), keyValueLength)
                      .slice(0, {keyValueLength - queryLength, keyValueLength});
    scores = F::add(cudaOps(), scores, mask);
  }

  return F::matmul(cudaOps(), F::softmax(cudaOps(), scores), v);
}

CATCH_TEST_CASE("test CUDA attention (blocked over the queries)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A head of 32 is one the compiled FlashAttention kernels will not take, so this is the
  // fallback. Past the operator's score budget it takes the queries a block at a time rather than
  // holding the whole matrix, and the answer may not change for it: each output row sees every
  // key either way, so there is nothing to approximate.
  //
  // A score matrix is one per head, so the budget is reached here on eight heads of a length that
  // one head would fit whole -- and reached with a matrix a tenth the size that one head would
  // need to reach it, which is what keeps this test affordable. The rule that counted only the
  // query and key lengths took this in a single block and held eight times what it meant to.
  Tensor q = toCuda(F::rand(Device::getCpu(), {1, 8, 4200, 32}, DType::kFloat));
  Tensor k = toCuda(F::rand(Device::getCpu(), {1, 8, 4200, 32}, DType::kFloat));
  Tensor v = toCuda(F::rand(Device::getCpu(), {1, 8, 4200, 32}, DType::kFloat));

  Tensor actual = F::attention(cudaOps(), q, k, v, false);
  CATCH_REQUIRE(actual.getShape() == std::vector<int>{1, 8, 4200, 32});
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(actual), toCpu(wholeAttention(q, k, v, false)), 5e-3));

  // The same length in one head, which stays under the budget and is taken whole.
  Tensor q1 = toCuda(F::rand(Device::getCpu(), {1, 1, 4200, 32}, DType::kFloat));
  Tensor k1 = toCuda(F::rand(Device::getCpu(), {1, 1, 4200, 32}, DType::kFloat));
  Tensor v1 = toCuda(F::rand(Device::getCpu(), {1, 1, 4200, 32}, DType::kFloat));

  Tensor whole = F::attention(cudaOps(), q1, k1, v1, false);
  CATCH_REQUIRE(whole.getShape() == std::vector<int>{1, 1, 4200, 32});
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(whole), toCpu(wholeAttention(q1, k1, v1, false)), 5e-3));
}

CATCH_TEST_CASE("test CUDA attention (blocked and causal)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Blocking moves where a block of queries sits in the sequence, so the mask has to move with
  // it. Getting that wrong lets a position see what follows it, which is the one thing the mask
  // exists to prevent and which no shape check would catch. Eight heads over a length that does
  // not divide by the block size leaves a short block at the end, where the mask sits furthest
  // from where it was cut.
  Tensor q = toCuda(F::rand(Device::getCpu(), {1, 8, 4200, 32}, DType::kFloat));
  Tensor k = toCuda(F::rand(Device::getCpu(), {1, 8, 4200, 32}, DType::kFloat));
  Tensor v = toCuda(F::rand(Device::getCpu(), {1, 8, 4200, 32}, DType::kFloat));

  Tensor actual = F::attention(cudaOps(), q, k, v, true);
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(actual), toCpu(wholeAttention(q, k, v, true)), 5e-3));
}

CATCH_TEST_CASE("test CUDA attention (head dims)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // FlashAttention only has kernels for a few head dims; the rest fall back to the portable
  // operator. Both sides of that split have to agree with the CPU reference.
  for (bool causal : {false, true}) {
    for (int headDim : {16, 32, 64, 128, 256}) {
      CATCH_INFO("headDim = " << headDim << ", causal = " << causal);
      CATCH_REQUIRE(runAttentionCase(4, 4, 8, 8, headDim, causal));
      CATCH_REQUIRE(runAttentionCase(4, 2, 6, 10, headDim, causal));
    }
  }
}

CATCH_TEST_CASE("test CUDA attention (degenerate lengths)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A single query against a single key is the shortest possible attention, and with one head it
  // removes the head axis from the indexing arithmetic entirely.
  for (bool causal : {false, true}) {
    CATCH_INFO("causal = " << causal);
    CATCH_REQUIRE(runAttentionCase(1, 1, 1, 1, 128, causal));
    CATCH_REQUIRE(runAttentionCase(1, 1, 1, 16, 128, causal));
    CATCH_REQUIRE(runAttentionCase(8, 8, 1, 1, 128, causal));
    // one query against a long history, the decode-step shape.
    CATCH_REQUIRE(runAttentionCase(8, 2, 1, 513, 128, causal));
  }
}

CATCH_TEST_CASE("test CUDA attention (grouped query ratios)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Every head count divisible by the key/value head count is a different grouping ratio, and
  // the ratio decides which key head each query head reads.
  for (bool causal : {false, true}) {
    CATCH_INFO("causal = " << causal);
    CATCH_REQUIRE(runAttentionCase(8, 1, 4, 12, 128, causal));  // all heads share one kv head
    CATCH_REQUIRE(runAttentionCase(8, 4, 4, 12, 128, causal));
    CATCH_REQUIRE(runAttentionCase(8, 8, 4, 12, 128, causal));  // no grouping
    CATCH_REQUIRE(runAttentionCase(12, 4, 4, 12, 128, causal));
  }
}

CATCH_TEST_CASE("test CUDA attention (causal boundaries)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Key lengths on and just off the kernel's tile boundaries, where the causal masking of the
  // final partial tile is easiest to get wrong.
  for (int keyLength : {127, 128, 129, 255, 256, 257}) {
    CATCH_INFO("keyLength = " << keyLength);
    CATCH_REQUIRE(runAttentionCase(8, 2, 1, keyLength, 128, true));
    CATCH_REQUIRE(runAttentionCase(8, 2, 4, keyLength, 128, true));
  }

  // a fresh prefill, where the query range is the whole key range.
  for (int length : {1, 2, 128, 129}) {
    CATCH_INFO("length = " << length);
    CATCH_REQUIRE(runAttentionCase(8, 2, length, length, 128, true));
  }
}

namespace {

/// As runAttentionCase, with a batch: each sequence of it is read through its own batch stride,
/// which a batch of one never exercises.
bool runBatchedCase(
    int batch,
    int numHeads,
    int numKeyValueHeads,
    int queryLength,
    int keyValueLength,
    int headDim,
    bool causal) {
  Tensor q = F::rand(Device::getCpu(), {batch, queryLength, numHeads, headDim}, DType::kFloat);
  Tensor k =
      F::rand(Device::getCpu(), {batch, keyValueLength, numKeyValueHeads, headDim}, DType::kFloat);
  Tensor v =
      F::rand(Device::getCpu(), {batch, keyValueLength, numKeyValueHeads, headDim}, DType::kFloat);
  Tensor xr =
      F::attention(cpuOps(), q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal);

  Tensor x = F::attention(
      cudaOps(),
      toCuda(q).transpose(1, 2),
      toCuda(k).transpose(1, 2),
      toCuda(v).transpose(1, 2),
      causal);

  return cpuOps()->allClose(toCpu(x), xr, 5e-3f);
}

}  // namespace

CATCH_TEST_CASE("test CUDA attention (fused kernel tiles)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The fused kernel takes 64 queries and 64 keys at a time, a warp to sixteen of the queries.
  // Lengths on and off those boundaries, in every head dimension it is built for: a block of
  // queries that is short, a warp of it that holds nothing, and a block of keys that is.
  for (bool causal : {false, true}) {
    for (int headDim : {32, 64, 128}) {
      for (int queryLength : {1, 15, 16, 17, 63, 64, 65, 200}) {
        for (int keyValueLength : {queryLength, queryLength + 1, queryLength + 77, 300}) {
          if (keyValueLength < queryLength) continue;
          CATCH_INFO(
              "headDim = " << headDim << ", causal = " << causal << ", L = " << queryLength
                           << ", S = " << keyValueLength);
          CATCH_REQUIRE(
              runBatchedCase(2, 4, 2, queryLength, keyValueLength, headDim, causal));
        }
      }
    }
  }
}

namespace {

/// The fused kernel alone, its keys split `splits` ways, against the CPU's attention. Grouped
/// query heads are packed into the rows of one CTA, so a group of seven with one query is seven
/// rows and with five queries thirty-five, which a CTA of one warp no longer holds.
bool runSplitCase(
    int numHeads,
    int numKeyValueHeads,
    int queryLength,
    int keyValueLength,
    int headDim,
    bool causal,
    int splits) {
  Tensor q = F::rand(Device::getCpu(), {2, queryLength, numHeads, headDim}, DType::kFloat);
  Tensor k = F::rand(Device::getCpu(), {2, keyValueLength, numKeyValueHeads, headDim}, DType::kFloat);
  Tensor v = F::rand(Device::getCpu(), {2, keyValueLength, numKeyValueHeads, headDim}, DType::kFloat);
  Tensor xr =
      F::attention(cpuOps(), q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal);

  Tensor x = F::empty(Device::getCuda(), {2, numHeads, queryLength, headDim}, DType::kFloat16);
  bool ran = op::cuda::attention(
      toCuda(q).transpose(1, 2),
      toCuda(k).transpose(1, 2),
      toCuda(v).transpose(1, 2),
      causal,
      x,
      splits);
  return ran && cpuOps()->allClose(toCpu(x), xr, 5e-3f);
}

}  // namespace

CATCH_TEST_CASE("test CUDA attention (grouped heads packed, keys split)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Qwen2's fourteen heads to two, which packs seven heads to a row block; one, two and three
  // queries are one warp's rows, five are a CTA of four warps. Key lengths on and off a block of
  // 64, split one way, a few, and more ways than there are blocks, which is cut back to one block
  // a split.
  for (bool causal : {false, true}) {
    for (int headDim : {32, 64, 128}) {
      for (int queryLength : {1, 2, 3, 5}) {
        for (int keyValueLength : {5, 63, 64, 65, 1000}) {
          if (keyValueLength < queryLength) continue;
          for (int splits : {1, 2, 3, 7, 100}) {
            CATCH_INFO(
                "headDim = " << headDim << ", causal = " << causal << ", L = " << queryLength
                             << ", S = " << keyValueLength << ", splits = " << splits);
            CATCH_REQUIRE(
                runSplitCase(14, 2, queryLength, keyValueLength, headDim, causal, splits));
          }
        }
      }
    }
  }

  // Multi-head attention, a group of one, from a single query to a warp's sixteen and past it.
  for (bool causal : {false, true}) {
    for (int queryLength : {1, 7, 16, 17}) {
      for (int splits : {1, 4}) {
        CATCH_INFO("MHA, causal = " << causal << ", L = " << queryLength << ", splits = " << splits);
        CATCH_REQUIRE(runSplitCase(8, 8, queryLength, 700, 64, causal, splits));
      }
    }
  }

  // A group of every query head to one key-value head.
  CATCH_REQUIRE(runSplitCase(16, 1, 1, 333, 128, true, 0));
  CATCH_REQUIRE(runSplitCase(16, 1, 4, 333, 128, true, 0));
}

CATCH_TEST_CASE("test CUDA attention (scores past half's range)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Every query is 64 across the head and every key a constant across it, so a score is 64 * 64 *
  // the key's constant / 8: 81920 for the key of 160 and at most 76800 for the rest, all past what
  // half holds and with the largest 5000 clear of the next. The answer is that key's value row,
  // exactly. A score matrix written out in half -- the composition's -- is infinite there and its
  // softmax not a number.
  constexpr int kQueries = 4;
  constexpr int kKeys = 10;
  constexpr int kHeadDim = 64;
  constexpr int kLargest = 3;
  Tensor q = F::empty(Device::getCpu(), {1, 1, kQueries, kHeadDim}, DType::kFloat);
  Tensor k = F::empty(Device::getCpu(), {1, 1, kKeys, kHeadDim}, DType::kFloat);
  Tensor v = F::rand(Device::getCpu(), {1, 1, kKeys, kHeadDim}, DType::kFloat);
  float *pq = q.getInternalData()->getData<float>(q.getInternalOffset());
  float *pk = k.getInternalData()->getData<float>(k.getInternalOffset());
  for (int i = 0; i < kQueries * kHeadDim; ++i) pq[i] = 64.0f;
  for (int key = 0; key < kKeys; ++key) {
    float value = key == kLargest ? 160.0f : 100.0f + 5.0f * key;
    for (int d = 0; d < kHeadDim; ++d) pk[key * kHeadDim + d] = value;
  }

  Tensor x = toCpu(F::attention(cudaOps(), toCuda(q), toCuda(k), toCuda(v), false));

  // The value row, through half and back, as the kernel was handed it.
  Tensor rounded = toCpu(toCuda(v));
  Tensor expected = F::empty(Device::getCpu(), {1, 1, kQueries, kHeadDim}, DType::kFloat);
  const float *pv = rounded.getInternalData()->getData<float>(rounded.getInternalOffset());
  float *pe = expected.getInternalData()->getData<float>(expected.getInternalOffset());
  for (int query = 0; query < kQueries; ++query) {
    for (int d = 0; d < kHeadDim; ++d) pe[query * kHeadDim + d] = pv[kLargest * kHeadDim + d];
  }
  CATCH_REQUIRE(cpuOps()->allClose(x, expected, 1e-3f));
}

}  // namespace fl
