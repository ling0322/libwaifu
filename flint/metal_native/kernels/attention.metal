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
// online softmax of FlashAttention. A threadgroup takes kBlockQ queries of one head of one batch
// -- the grid is (query blocks, heads, batch) -- and walks the keys kBlockK at a time: every
// simdgroup has eight of the queries, and for each block of keys works out their scores S = Q K^T,
// the softmax of the scores so far, and O = O * rescale + P V, all in float.
//
// Everything a simdgroup does to rows -- the maximum and sum of each row of S, and scaling the rows
// of O -- goes through threadgroup memory or a diagonal matrix, never the layout of a simdgroup
// matrix's elements across its lanes, which is Metal's to choose.

#include <metal_simdgroup_matrix>

#include "common.h"

/// In step with AttentionParams in attention.cc.
struct AttentionParams {
  uint numHeads;
  uint headsPerKeyValueHead;  // numHeads / numKeyValueHeads
  uint queryLength;
  uint keyLength;
  // Every tensor's last dimension is contiguous; these are the others, in elements.
  uint qStrideB;
  uint qStrideH;
  uint qStrideL;
  uint kStrideB;
  uint kStrideH;
  uint kStrideL;
  uint vStrideB;
  uint vStrideH;
  uint vStrideL;
  float scale;
  // The mask, as Operators::attention has it: a MaskType -- 0 none, 1 causal, 2 block causal, 3
  // block diagonal -- and for the block masks how many keys a block is, counted from key 0. Under
  // the causal mask the last query sees every key and the first keyLength - queryLength + 1; under
  // a block diagonal one a query sees its own block only.
  uint maskType;
  uint blockSize;
};

constant constexpr uint kMaskNone = 0;
constant constexpr uint kMaskCausal = 1;
constant constexpr uint kMaskBlockDiagonal = 3;

/// The first key a query at `position` -- its index plus keyLength - queryLength -- may see: the
/// first of its own block under a block diagonal mask, and key 0 under every other.
inline uint firstKeyOf(constant AttentionParams &p, uint position) {
  if (p.maskType != kMaskBlockDiagonal) return 0;
  return (position / p.blockSize) * p.blockSize;
}

/// The last key a query at `position` may see: every key with no mask, the position itself under
/// the causal one, and the last of its block, or of the keys, whichever comes first, under either
/// block mask.
inline uint lastKeyOf(constant AttentionParams &p, uint position) {
  if (p.maskType == kMaskNone) return p.keyLength - 1;
  if (p.maskType == kMaskCausal) return position;
  return min(p.keyLength - 1, (position / p.blockSize) * p.blockSize + p.blockSize - 1);
}

constant constexpr uint kBlockQ = 32;
constant constexpr uint kBlockK = 32;
constant constexpr uint kSimdgroups = 4;
constant constexpr uint kThreads = kSimdgroups * 32;

template <typename T, uint D>
kernel void attention(
    device T *out [[buffer(0)]],
    device const T *q [[buffer(1)]],
    device const T *k [[buffer(2)]],
    device const T *v [[buffer(3)]],
    constant AttentionParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  // Rows unpadded. Padding them to spread a column of 8 x 8 blocks over the banks, as on CUDA,
  // made this kernel 16 to 29% slower on an M5 Pro -- but only with the keys and the values
  // padded alike, to a stride not a power of two: D + 4, D + 8 and D + 16 all were, D + 8 for the
  // keys beside D + 4 for the values was not. Whatever that is, it is the compiler's or the GPU's,
  // not the banks', and unpadded is the fastest measured.
  constexpr uint kStride = D;
  threadgroup T keys[kBlockK * kStride];    // and the queries, before the first block
  threadgroup T values[kBlockK * kStride];
  threadgroup float scores[kSimdgroups * 8 * kBlockK];
  threadgroup T probabilities[kSimdgroups * 8 * kBlockK];
  threadgroup float diagonal[kSimdgroups * 64];

  uint batch = group.z;
  uint head = group.y;
  uint q0 = group.x * kBlockQ;
  uint kvHead = head / p.headsPerKeyValueHead;
  device const T *qBase = q + batch * p.qStrideB + head * p.qStrideH;
  device const T *kBase = k + batch * p.kStrideB + kvHead * p.kStrideH;
  device const T *vBase = v + batch * p.vStrideB + kvHead * p.vStrideH;

  threadgroup float *myScores = scores + sg * 8 * kBlockK;
  threadgroup T *myProbabilities = probabilities + sg * 8 * kBlockK;
  threadgroup float *myDiagonal = diagonal + sg * 64;

  // The queries, through the keys' tile, into registers: eight rows a simdgroup.
  for (uint i = tid; i < kBlockQ * D; i += kThreads) {
    uint row = i / D;
    uint d = i - row * D;
    keys[row * kStride + d] = q0 + row < p.queryLength ? qBase[(q0 + row) * p.qStrideL + d] : T(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_matrix<T, 8, 8> query[D / 8];
  for (uint d = 0; d < D / 8; ++d) simdgroup_load(query[d], keys + sg * 8 * kStride + d * 8, kStride);

  simdgroup_matrix<float, 8, 8> o[D / 8];
  for (uint d = 0; d < D / 8; ++d) o[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  // The softmax runs a row to four lanes, eight scores to a lane; each lane keeps its row's
  // running maximum and sum.
  uint row = lane / 4;
  uint quarter = lane % 4;
  uint queryIndex = q0 + sg * 8 + row;
  float rowMax = -INFINITY;
  float rowSum = 0.0f;
  // The keys this query may see. A row past the last query sees what the last one does; it is
  // never written.
  uint offset = p.keyLength - p.queryLength;
  uint position = min(queryIndex, p.queryLength - 1) + offset;
  uint firstKey = firstKeyOf(p, position);
  uint lastKey = lastKeyOf(p, position);

  // The keys any row of this threadgroup may see: the rows are queries in order, so the first
  // row's first key and the last row's last key bound them all. Blocks of keys before the one
  // and after the other have nothing to add. The same for every thread of the group, so that each
  // reaches every barrier in the loop.
  uint keyBegin = (firstKeyOf(p, q0 + offset) / kBlockK) * kBlockK;
  uint keyEnd = lastKeyOf(p, min(q0 + kBlockQ, p.queryLength) - 1 + offset) + 1;

  for (uint k0 = keyBegin; k0 < keyEnd; k0 += kBlockK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < kBlockK * D; i += kThreads) {
      uint key = i / D;
      uint d = i - key * D;
      bool in = k0 + key < p.keyLength;
      keys[key * kStride + d] = in ? kBase[(k0 + key) * p.kStrideL + d] : T(0);
      values[key * kStride + d] = in ? vBase[(k0 + key) * p.vStrideL + d] : T(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S = Q K^T, eight keys at a time.
    for (uint j = 0; j < kBlockK / 8; ++j) {
      simdgroup_matrix<float, 8, 8> s = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      for (uint d = 0; d < D / 8; ++d) {
        simdgroup_matrix<T, 8, 8> keyT;
        simdgroup_load(keyT, keys + j * 8 * kStride + d * 8, kStride, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(s, query[d], keyT, s);
      }
      simdgroup_store(s, myScores + j * 8, kBlockK);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // The online softmax of this lane's eight scores.
    float x[8];
    float blockMax = -INFINITY;
    for (uint c = 0; c < 8; ++c) {
      uint column = quarter * 8 + c;
      uint key = k0 + column;
      bool seen = key >= firstKey && key <= lastKey;
      x[c] = seen ? myScores[row * kBlockK + column] * p.scale : -INFINITY;
      blockMax = max(blockMax, x[c]);
    }
    blockMax = max(blockMax, simd_shuffle_xor(blockMax, 1));
    blockMax = max(blockMax, simd_shuffle_xor(blockMax, 2));

    float newMax = max(rowMax, blockMax);
    // A row that has seen no key yet keeps nothing, and adds nothing.
    float rescale = newMax == -INFINITY ? 1.0f : exp(rowMax - newMax);
    float blockSum = 0.0f;
    for (uint c = 0; c < 8; ++c) {
      float e = newMax == -INFINITY ? 0.0f : exp(x[c] - newMax);
      blockSum += e;
      myProbabilities[row * kBlockK + quarter * 8 + c] = T(e);
    }
    blockSum += simd_shuffle_xor(blockSum, 1);
    blockSum += simd_shuffle_xor(blockSum, 2);
    rowSum = rowSum * rescale + blockSum;
    rowMax = newMax;

    // diag(rescale), to scale the rows of O by.
    myDiagonal[lane] = 0.0f;
    myDiagonal[lane + 32] = 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (quarter == 0) myDiagonal[row * 8 + row] = rescale;
    simdgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> scale;
    simdgroup_load(scale, myDiagonal, 8);
    simdgroup_matrix<T, 8, 8> probability[kBlockK / 8];
    for (uint j = 0; j < kBlockK / 8; ++j) {
      simdgroup_load(probability[j], myProbabilities + j * 8, kBlockK);
    }

    // O = diag(rescale) O + P V.
    for (uint d = 0; d < D / 8; ++d) {
      simdgroup_multiply(o[d], scale, o[d]);
      for (uint j = 0; j < kBlockK / 8; ++j) {
        simdgroup_matrix<T, 8, 8> value;
        simdgroup_load(value, values + j * 8 * kStride + d * 8, kStride);
        simdgroup_multiply_accumulate(o[d], probability[j], value, o[d]);
      }
    }
  }

  // O / the sum of each row, written out through the scores' tile, the rows that are queries.
  simdgroup_barrier(mem_flags::mem_threadgroup);
  myDiagonal[lane] = 0.0f;
  myDiagonal[lane + 32] = 0.0f;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (quarter == 0) myDiagonal[row * 8 + row] = rowSum > 0.0f ? 1.0f / rowSum : 0.0f;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_matrix<float, 8, 8> normalize;
  simdgroup_load(normalize, myDiagonal, 8);

  uint outRow = p.numHeads * p.queryLength;  // the output is (B, H, Lq, D), contiguous
  device T *outBase = out + (batch * outRow + head * p.queryLength) * D;
  for (uint d = 0; d < D / 8; ++d) {
    simdgroup_multiply(o[d], normalize, o[d]);
    simdgroup_store(o[d], myScores, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 64; e += 32) {
      uint r = e / 8;
      uint index = q0 + sg * 8 + r;
      if (index < p.queryLength) outBase[index * D + d * 8 + e % 8] = T(myScores[e]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define FLINT_ATTENTION(T, S, D)                                                                \
  template [[host_name("attention_" #D "_" S)]] kernel void attention<T, D>(                    \
      device T *, device const T *, device const T *, device const T *,                         \
      constant AttentionParams &, uint3, uint, uint, uint);

FLINT_ATTENTION(half, "f16", 32)
FLINT_ATTENTION(half, "f16", 64)
FLINT_ATTENTION(half, "f16", 128)
FLINT_ATTENTION(float, "f32", 32)
FLINT_ATTENTION(float, "f32", 64)
