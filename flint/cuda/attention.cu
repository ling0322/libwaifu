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

// Scaled dot product attention in one pass over the keys, the score matrix never written out: the
// online softmax of FlashAttention, written after flint/metal_native/kernels/attention.metal.
//
// # Rows
//
// The query heads that share a key-value head attend to the same keys, so a CTA takes its rows
// from all of them: the rows of key-value head h are every (query i, head g of its group), query
// first -- row r is query r / group of head h * group + r % group. A CTA holds WARPS * 16 of those
// rows, a warp sixteen. Under multi-head attention, a group of one, that is the rows of one head
// in order, as any attention kernel has them. Under grouped-query attention it is what keeps the
// rows full when there are few queries: a decode step of Qwen2's seven heads to a group is seven
// rows of one CTA rather than one row of each of seven, which also reads the keys once rather than
// seven times. Query first, so that the rows of a CTA are a run of neighbouring queries and a
// causal mask still ends the walk over the keys early.
//
// # Keys
//
// A CTA walks its keys kBlockN at a time. For each block a warp works out its scores S = Q K^T,
// the softmax of the scores so far, and O = O * rescale + P V, all in float, on mma.sync m16n8k16.
// Where the rows are too few to fill the card the keys are split as well, flash-decoding's way:
// the grid is (row blocks * splits, key-value heads, batch), each CTA walks one split, and writes
// its unnormalized O with its rows' maximum and sum; combineKernel then weighs the splits against
// each other, which is the same rescaling the walk does from one block to the next.
//
// # Registers
//
// What Metal has to do through threadgroup memory is done here in registers, because mma.sync,
// unlike a simdgroup matrix, says where each element is (the layout is the one
// gated_delta_net_mma.cu checks against a reference GEMM):
//
// - A lane of an m16n8 accumulator holds two rows of it, lane / 4 and lane / 4 + 8, and the four
//   lanes sharing lane / 4 hold all of those rows' columns between them. So the maximum of a row is
//   two butterfly shuffles over lanes ^1 and ^2, and scaling a row of O is a multiply on the
//   registers that hold it.
// - Two adjacent m16n8 accumulators, cast to half, are the A operand of the 16 by 16 tile they
//   cover. So the probabilities go from the softmax into P V without leaving the registers.
//
// The queries are read once, into the A operands of every product of the CTA's lifetime. The keys
// and values go through shared memory, each row padded by sixteen bytes so that the eight rows one
// operand load reads land on different banks; the keys are read back as the B operand directly,
// since a row of K is a column of K^T, and the values transposed on the way in by ldmatrix.

#include "flint/cuda/attention.h"

#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

#include <algorithm>
#include <type_traits>

#include "flint/cuda/common.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cuda {

namespace {

constexpr int kBlockN = 64;
constexpr int kMinArch = 80;

struct Params {
  const half *q;
  const half *k;
  const half *v;
  half *out;
  int numHeads;
  int numKeyValueHeads;
  int headsPerKeyValueHead;
  int queryLength;
  int keyLength;
  // headsPerKeyValueHead * queryLength: the rows of one key-value head.
  int rows;
  // How many CTAs of rows a key-value head has, and how many ways its keys are split.
  int rowBlocks;
  int splits;
  int keysPerSplit;
  // Every tensor's last dimension is contiguous; these are the others, in elements.
  int64_t qStrideB;
  int64_t qStrideH;
  int64_t qStrideL;
  int64_t kStrideB;
  int64_t kStrideH;
  int64_t kStrideL;
  int64_t vStrideB;
  int64_t vStrideH;
  int64_t vStrideL;
  // 1 / sqrt(D), times log2(e): the scores go into exp2.
  float scaleLog2;
  // The mask, as Operators::attention has it: 0 none, otherwise blocks of this many keys from key
  // 0, a query seeing its own block and every block before it. 1 is the causal mask, under which
  // the last query sees every key and the first keyLength - queryLength + 1.
  int blockSize;
  // Where a split writes, when there is more than one: <float>(batch, kvHeads, splits, rows, D),
  // and the maximum and the sum of each row, <float>(batch, kvHeads, splits, rows).
  float *partialOut;
  float *partialMax;
  float *partialSum;
};

/// One m16n8 accumulator: four floats a lane.
struct Acc {
  float x[4];
};

[[maybe_unused]] __device__ inline void mma16816(Acc &d, const unsigned *a, const unsigned *b) {
#if __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d.x[0]), "+f"(d.x[1]), "+f"(d.x[2]), "+f"(d.x[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

/// The B operands of two adjacent eight-column tiles out of a row-major 16 by 16 block of shared
/// memory whose rows are the contraction, transposed on the way in. ldmatrix's four matrices are
/// (k 0-7, n 0-7), (k 8-15, n 0-7), (k 0-7, n 8-15), (k 8-15, n 8-15) under this address pattern,
/// so the first tile's operand is registers 0 and 1 and the second's 2 and 3.
[[maybe_unused]] __device__ inline void loadBTrans(unsigned *r, const half *src, int ld, int lane) {
#if __CUDA_ARCH__ >= 800
  const half *p = src + (lane & 15) * ld + (lane >> 4) * 8;
  unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#endif
}

[[maybe_unused]] __device__ inline unsigned pack(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<unsigned *>(&h);
}

/// Where a row of a key-value head comes from: its query and its head, or nowhere for a row past
/// the last.
struct Row {
  bool valid;
  int query;
  int head;
};

/// The last key a query at `position` -- its index plus keyLength - queryLength -- may see: the
/// last of its block, or of the keys, whichever comes first. Under the causal mask, a block of
/// one, that is the position itself.
__device__ inline int lastKeyOf(const Params &p, int position) {
  if (p.blockSize == 0) return p.keyLength - 1;
  // The causal mask without the division, which is most of what this costs and buys it nothing.
  if (p.blockSize == 1) return position;
  return min(p.keyLength - 1, (position / p.blockSize) * p.blockSize + p.blockSize - 1);
}

__device__ inline Row rowOf(const Params &p, int kvHead, int row) {
  Row r;
  r.valid = row < p.rows;
  r.query = r.valid ? row / p.headsPerKeyValueHead : 0;
  r.head = kvHead * p.headsPerKeyValueHead + (r.valid ? row % p.headsPerKeyValueHead : 0);
  return r;
}

template<int D, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) attentionKernel(Params p) {
#if __CUDA_ARCH__ >= 800
  constexpr int kThreads = WARPS * 32;
  constexpr int kBlockM = WARPS * 16;
  constexpr int kStride = D + 8;
  constexpr int kSegments = D / 8;
  __shared__ __align__(16) half keys[kBlockN * kStride];
  __shared__ __align__(16) half values[kBlockN * kStride];

  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int rowBlock = blockIdx.x % p.rowBlocks;
  int split = blockIdx.x / p.rowBlocks;
  int kvHead = blockIdx.y;
  int batch = blockIdx.z;
  int r0 = rowBlock * kBlockM;
  const half *kBase = p.k + batch * p.kStrideB + kvHead * p.kStrideH;
  const half *vBase = p.v + batch * p.vStrideB + kvHead * p.vStrideH;

  // The two rows this lane holds of every accumulator, and the first of its two columns.
  int firstRow = r0 + warp * 16 + (lane >> 2);
  Row rows[2] = {rowOf(p, kvHead, firstRow), rowOf(p, kvHead, firstRow + 8)};
  int column = (lane & 3) * 2;

  // The queries, straight into A operands: rows 0 and 1, columns column and column + 8 of each
  // sixteen of the head.
  unsigned query[D / 16][4];
  auto queryPair = [&](const Row &row, int d) -> unsigned {
    if (!row.valid) return 0;
    const half *q = p.q + batch * p.qStrideB + row.head * p.qStrideH + row.query * p.qStrideL;
    return *reinterpret_cast<const unsigned *>(q + d);
  };
#pragma unroll
  for (int kk = 0; kk < D / 16; ++kk) {
    query[kk][0] = queryPair(rows[0], kk * 16 + column);
    query[kk][1] = queryPair(rows[1], kk * 16 + column);
    query[kk][2] = queryPair(rows[0], kk * 16 + column + 8);
    query[kk][3] = queryPair(rows[1], kk * 16 + column + 8);
  }

  Acc o[D / 8];
#pragma unroll
  for (int dt = 0; dt < D / 8; ++dt) {
    o[dt].x[0] = o[dt].x[1] = o[dt].x[2] = o[dt].x[3] = 0.0f;
  }

  // Each row's running maximum, in the exp2 domain, and this lane's share of its running sum: the
  // share is rescaled by the row's own factor, which all four lanes agree on, so the shares are
  // only added up at the end.
  float rowMax[2] = {-INFINITY, -INFINITY};
  float rowSum[2] = {0.0f, 0.0f};
  int offset = p.keyLength - p.queryLength;
  int lastKey[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    lastKey[r] = lastKeyOf(p, rows[r].query + offset);
  }

  // This split's keys, and no further than the last any row of this CTA may see.
  int keyBegin = split * p.keysPerSplit;
  int keyEnd = min(p.keyLength, keyBegin + p.keysPerSplit);
  if (p.blockSize > 0) {
    int lastQuery = min(p.queryLength - 1, (r0 + kBlockM - 1) / p.headsPerKeyValueHead);
    keyEnd = min(keyEnd, lastKeyOf(p, lastQuery + offset) + 1);
  }

  for (int k0 = keyBegin; k0 < keyEnd; k0 += kBlockN) {
    __syncthreads();
    for (int i = threadIdx.x; i < kBlockN * kSegments; i += kThreads) {
      int key = i / kSegments;
      int d = (i - key * kSegments) * 8;
      uint4 keyRow = make_uint4(0, 0, 0, 0);
      uint4 valueRow = make_uint4(0, 0, 0, 0);
      if (k0 + key < keyEnd) {
        keyRow = *reinterpret_cast<const uint4 *>(kBase + (k0 + key) * p.kStrideL + d);
        valueRow = *reinterpret_cast<const uint4 *>(vBase + (k0 + key) * p.vStrideL + d);
      }
      *reinterpret_cast<uint4 *>(keys + key * kStride + d) = keyRow;
      *reinterpret_cast<uint4 *>(values + key * kStride + d) = valueRow;
    }
    __syncthreads();

    // S = Q K^T, eight keys a tile. The B operand of a tile is K^T's (k, n) = K's (n, k), and two
    // adjacent k of one n are adjacent in K's row: one 32-bit load each.
    Acc s[kBlockN / 8];
#pragma unroll
    for (int nt = 0; nt < kBlockN / 8; ++nt) {
      s[nt].x[0] = s[nt].x[1] = s[nt].x[2] = s[nt].x[3] = 0.0f;
      const half *keyRow = keys + (nt * 8 + (lane >> 2)) * kStride + column;
#pragma unroll
      for (int kk = 0; kk < D / 16; ++kk) {
        unsigned b[2];
        b[0] = *reinterpret_cast<const unsigned *>(keyRow + kk * 16);
        b[1] = *reinterpret_cast<const unsigned *>(keyRow + kk * 16 + 8);
        mma16816(s[nt], query[kk], b);
      }
    }

    // The online softmax of the block, row by row: masked, scaled, its maximum across the four
    // lanes that share the row, and the probabilities in place of the scores.
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float blockMax = -INFINITY;
#pragma unroll
      for (int nt = 0; nt < kBlockN / 8; ++nt) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          int key = k0 + nt * 8 + column + j;
          float &x = s[nt].x[2 * r + j];
          x = key < keyEnd && key <= lastKey[r] ? x * p.scaleLog2 : -INFINITY;
          blockMax = fmaxf(blockMax, x);
        }
      }
      blockMax = fmaxf(blockMax, __shfl_xor_sync(0xffffffff, blockMax, 1));
      blockMax = fmaxf(blockMax, __shfl_xor_sync(0xffffffff, blockMax, 2));

      float newMax = fmaxf(rowMax[r], blockMax);
      // A row that has seen no key yet keeps nothing, and adds nothing.
      float rescale = newMax == -INFINITY ? 1.0f : exp2f(rowMax[r] - newMax);
      float blockSum = 0.0f;
#pragma unroll
      for (int nt = 0; nt < kBlockN / 8; ++nt) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          float &x = s[nt].x[2 * r + j];
          x = newMax == -INFINITY ? 0.0f : exp2f(x - newMax);
          blockSum += x;
        }
      }
      rowSum[r] = rowSum[r] * rescale + blockSum;
      rowMax[r] = newMax;
#pragma unroll
      for (int dt = 0; dt < D / 8; ++dt) {
        o[dt].x[2 * r] *= rescale;
        o[dt].x[2 * r + 1] *= rescale;
      }
    }

    // O += P V, sixteen keys a step. Tiles 2t and 2t + 1 of the probabilities, cast to half, are
    // the A operand of keys 16t to 16t + 15.
#pragma unroll
    for (int kt = 0; kt < kBlockN / 16; ++kt) {
      unsigned a[4];
      a[0] = pack(s[2 * kt].x[0], s[2 * kt].x[1]);
      a[1] = pack(s[2 * kt].x[2], s[2 * kt].x[3]);
      a[2] = pack(s[2 * kt + 1].x[0], s[2 * kt + 1].x[1]);
      a[3] = pack(s[2 * kt + 1].x[2], s[2 * kt + 1].x[3]);
#pragma unroll
      for (int dt = 0; dt < D / 8; dt += 2) {
        unsigned b[4];
        loadBTrans(b, values + kt * 16 * kStride + dt * 8, kStride, lane);
        mma16816(o[dt], a, b);
        mma16816(o[dt + 1], a, b + 2);
      }
    }
  }

  // Each row's sum, the shares added up first. One split is the answer, O over that sum, written
  // out as pairs of halves; one of several is O as it stands, with the maximum and the sum the
  // combine weighs it by.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    float sum = rowSum[r];
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    // Worked out again rather than held through the walk, where it would be registers for nothing.
    int index = r0 + warp * 16 + (lane >> 2) + 8 * r;
    Row row = rowOf(p, kvHead, index);
    if (!row.valid) continue;

    if (p.splits == 1) {
      float inverse = sum > 0.0f ? 1.0f / sum : 0.0f;
      half *outRow =
          p.out +
          ((static_cast<int64_t>(batch) * p.numHeads + row.head) * p.queryLength + row.query) * D;
#pragma unroll
      for (int dt = 0; dt < D / 8; ++dt) {
        *reinterpret_cast<unsigned *>(outRow + dt * 8 + column) =
            pack(o[dt].x[2 * r] * inverse, o[dt].x[2 * r + 1] * inverse);
      }
    } else {
      int64_t at =
          ((static_cast<int64_t>(batch) * p.numKeyValueHeads + kvHead) * p.splits + split) *
              p.rows +
          index;
      float *outRow = p.partialOut + at * D;
#pragma unroll
      for (int dt = 0; dt < D / 8; ++dt) {
        *reinterpret_cast<float2 *>(outRow + dt * 8 + column) =
            make_float2(o[dt].x[2 * r], o[dt].x[2 * r + 1]);
      }
      if ((lane & 3) == 0) {
        p.partialMax[at] = rowMax[r];
        p.partialSum[at] = sum;
      }
    }
  }
#endif
}

/// The splits of one row put together: a CTA a row of one key-value head, a thread a column of
/// it. Each split's O and sum are weighed by how far its maximum is below the largest of them, as
/// the walk over one split weighs one block of keys against the ones before.
template<int D>
__global__ void __launch_bounds__(D) combineKernel(Params p) {
  int row = blockIdx.x;
  int kvHead = blockIdx.y;
  int batch = blockIdx.z;
  int d = threadIdx.x;
  int64_t first =
      (static_cast<int64_t>(batch) * p.numKeyValueHeads + kvHead) * p.splits * p.rows + row;

  float largest = -INFINITY;
  for (int split = 0; split < p.splits; ++split) {
    largest = fmaxf(largest, p.partialMax[first + static_cast<int64_t>(split) * p.rows]);
  }
  float sum = 0.0f;
  float out = 0.0f;
  if (largest != -INFINITY) {
    for (int split = 0; split < p.splits; ++split) {
      int64_t at = first + static_cast<int64_t>(split) * p.rows;
      float weight = exp2f(p.partialMax[at] - largest);
      sum += weight * p.partialSum[at];
      out += weight * p.partialOut[at * D + d];
    }
  }

  int query = row / p.headsPerKeyValueHead;
  int head = kvHead * p.headsPerKeyValueHead + row % p.headsPerKeyValueHead;
  p.out[((static_cast<int64_t>(batch) * p.numHeads + head) * p.queryLength + query) * D + d] =
      __float2half(sum > 0.0f ? out / sum : 0.0f);
}

bool isAligned(const TensorView &x, int64_t elements) {
  uintptr_t address = reinterpret_cast<uintptr_t>(getDataPtrCuda<half>(x));
  if (address % (elements * sizeof(half)) != 0) return false;
  for (int d = 0; d < 3; ++d) {
    if (x.getStride(d) % elements != 0) return false;
  }
  return true;
}

bool hasTensorCores() {
  static const bool has = [] {
    int major = 0;
    int minor = 0;
    LL_CHECK_CUDA_STATUS(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    LL_CHECK_CUDA_STATUS(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0));
    return major * 10 + minor >= kMinArch;
  }();
  return has;
}

int multiprocessors() {
  static const int count = [] {
    int n = 0;
    LL_CHECK_CUDA_STATUS(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0));
    return n;
  }();
  return count;
}

}  // namespace

bool attention(
    const TensorView &q,
    const TensorView &k,
    const TensorView &v,
    int blockSize,
    const TensorView &out) {
  return attention(q, k, v, blockSize, out, 0);
}

bool attention(
    const TensorView &q,
    const TensorView &k,
    const TensorView &v,
    int blockSize,
    const TensorView &out,
    int splits) {
  if (q.getDim() != 4 || k.getDim() != 4 || v.getDim() != 4 || out.getDim() != 4) return false;
  if (q.getDType() != DType::kFloat16 || k.getDType() != DType::kFloat16 ||
      v.getDType() != DType::kFloat16 || out.getDType() != DType::kFloat16) {
    return false;
  }

  int batch = q.getShape(0);
  int numHeads = q.getShape(1);
  int queryLength = q.getShape(2);
  int headDim = q.getShape(3);
  int numKeyValueHeads = k.getShape(1);
  int keyLength = k.getShape(2);

  if (headDim != 32 && headDim != 64 && headDim != 128) return false;
  if (k.getShape() != v.getShape() || k.getShape(0) != batch || k.getShape(3) != headDim) {
    return false;
  }
  if (numKeyValueHeads == 0 || numHeads % numKeyValueHeads != 0) return false;
  if (queryLength == 0 || keyLength == 0) return false;
  if (blockSize < 0 || (blockSize > 0 && keyLength < queryLength)) return false;
  if (q.getStride(3) != 1 || k.getStride(3) != 1 || v.getStride(3) != 1) return false;
  if (!out.isContiguous() || out.getShape() != q.getShape()) return false;

  // The keys and values are read sixteen bytes at a time, the queries and the output four.
  if (!isAligned(k, 8) || !isAligned(v, 8) || !isAligned(q, 2)) return false;
  if (reinterpret_cast<uintptr_t>(getDataPtrCuda<half>(out)) % 4 != 0) return false;
  if (numKeyValueHeads > 65535 || batch > 65535) return false;
  int64_t rows = static_cast<int64_t>(queryLength) * (numHeads / numKeyValueHeads);
  if (rows > (1 << 30)) return false;
  if (!hasTensorCores()) return false;

  Params params;
  params.q = getDataPtrCuda<half>(q);
  params.k = getDataPtrCuda<half>(k);
  params.v = getDataPtrCuda<half>(v);
  params.out = getDataPtrCuda<half>(out);
  params.numHeads = numHeads;
  params.numKeyValueHeads = numKeyValueHeads;
  params.headsPerKeyValueHead = numHeads / numKeyValueHeads;
  params.queryLength = queryLength;
  params.keyLength = keyLength;
  params.rows = static_cast<int>(rows);
  params.qStrideB = q.getStride(0);
  params.qStrideH = q.getStride(1);
  params.qStrideL = q.getStride(2);
  params.kStrideB = k.getStride(0);
  params.kStrideH = k.getStride(1);
  params.kStrideL = k.getStride(2);
  params.vStrideB = v.getStride(0);
  params.vStrideH = v.getStride(1);
  params.vStrideL = v.getStride(2);
  params.scaleLog2 = 1.4426950408889634f / sqrtf(static_cast<float>(headDim));
  params.blockSize = blockSize;
  params.partialOut = nullptr;
  params.partialMax = nullptr;
  params.partialSum = nullptr;

  // A warp where the rows of a key-value head fit in one, four otherwise.
  int warps = params.rows <= 16 ? 1 : 4;
  params.rowBlocks = (params.rows + warps * 16 - 1) / (warps * 16);

  // Split the keys only where the CTAs without a split would not fill the card once -- and then as
  // many ways as it takes to give every multiprocessor two, never a split of less than a block. A
  // split costs a second launch and a trip through memory for every row; Qwen2's prefill of 300,
  // 66 CTAs on 36 multiprocessors, measured half as slow again split two ways as not at all.
  int blocks = (keyLength + kBlockN - 1) / kBlockN;
  if (splits <= 0) {
    int64_t ctas = static_cast<int64_t>(params.rowBlocks) * numKeyValueHeads * batch;
    int64_t sms = multiprocessors();
    splits = ctas >= sms ? 1 : static_cast<int>((2 * sms + ctas - 1) / ctas);
  }
  splits = std::max(1, std::min(splits, blocks));
  params.keysPerSplit = ((blocks + splits - 1) / splits) * kBlockN;
  params.splits = (keyLength + params.keysPerSplit - 1) / params.keysPerSplit;
  if (static_cast<int64_t>(params.rowBlocks) * params.splits > (1ll << 31) - 1) return false;

  Tensor partialOut;
  Tensor partialMax;
  Tensor partialSum;
  if (params.splits > 1) {
    int partials = static_cast<int>(
        static_cast<int64_t>(batch) * numKeyValueHeads * params.splits * params.rows);
    partialOut = createCudaTensor<float>({partials, headDim});
    partialMax = createCudaTensor<float>({partials});
    partialSum = createCudaTensor<float>({partials});
    params.partialOut = getDataPtrCuda<float>(partialOut);
    params.partialMax = getDataPtrCuda<float>(partialMax);
    params.partialSum = getDataPtrCuda<float>(partialSum);
  }

  dim3 grid(params.rowBlocks * params.splits, numKeyValueHeads, batch);
  auto run = [&](auto d) {
    constexpr int D = decltype(d)::value;
    if (warps == 1) {
      attentionKernel<D, 1><<<grid, 32>>>(params);
    } else {
      attentionKernel<D, 4><<<grid, 128>>>(params);
    }
    if (params.splits > 1) {
      combineKernel<D><<<dim3(params.rows, numKeyValueHeads, batch), D>>>(params);
    }
  };
  switch (headDim) {
    case 32:
      run(std::integral_constant<int, 32>());
      break;
    case 64:
      run(std::integral_constant<int, 64>());
      break;
    case 128:
      run(std::integral_constant<int, 128>());
      break;
  }
  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return true;
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
