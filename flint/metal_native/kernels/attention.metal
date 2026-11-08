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
// -- the grid is (query blocks, heads, batch) -- and walks the keys a block at a time: every
// simdgroup has eight of the queries, and for each block of keys works out their scores S = Q K^T,
// the softmax of the scores so far, and O = O * rescale + P V, all in float.
//
// The way of it is MLX's steel_attention (MIT, Copyright 2023-2025 Apple Inc.): the queries staged
// once, the keys staged transposed and the values after them in the same threadgroup memory, and
// the scores and O held in simdgroup matrices whose elements the softmax works on where they are.
// A lane holds two elements of an 8 x 8 matrix: row r, columns c and c + 1, where (c, r) is
// fragmentCoord(); the four lanes of a row differ in bits 0 and 3. That layout is Metal's to
// choose, so attention.cc tries the kernel against an answer worked out on the host before it uses
// it, and a GPU that lays the matrices out otherwise runs the composition instead.

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

// In step with kBlockQ and kThreads in attention.cc.
constant constexpr uint kBlockQ = 32;
constant constexpr uint kSimdgroups = 4;
constant constexpr uint kThreads = kSimdgroups * 32;

#define FLINT_UNROLL _Pragma("clang loop unroll(full)")

typedef simdgroup_matrix<float, 8, 8> Matrix;

/// Where this lane's elements of an 8 x 8 matrix are: (column, row) of the first of them.
inline short2 fragmentCoord(ushort lane) {
  short quad = lane / 4;
  short row = (quad & 4) + ((lane / 2) % 4);
  short column = (quad & 2) * 2 + (lane % 2) * 2;
  return short2(column, row);
}

/// The 8 x 8 matrix at `src` in threadgroup memory, its rows `ld` apart, in float.
template <typename T>
inline Matrix loadMatrix(threadgroup const T *src, uint ld, short2 at) {
  Matrix m;
  threadgroup const T *element = src + at.y * ld + at.x;
  m.thread_elements()[0] = float(element[0]);
  m.thread_elements()[1] = float(element[1]);
  return m;
}

/// The largest, or the sum, of a row over the four lanes that hold it.
inline float rowMax(float x) {
  x = max(x, simd_shuffle_xor(x, ushort(1)));
  return max(x, simd_shuffle_xor(x, ushort(8)));
}

inline float rowSum(float x) {
  x += simd_shuffle_xor(x, ushort(1));
  return x + simd_shuffle_xor(x, ushort(8));
}

/// `rows` rows of D from `src`, `ld` apart, into threadgroup memory: row r, column d at
/// dst[r * rowStride + d * columnStride]. Rows from `valid` on are zeros, and not read.
template <typename T, uint D, uint kRows, bool kChecked>
inline void stage(
    threadgroup T *dst,
    uint rowStride,
    uint columnStride,
    device const T *src,
    uint ld,
    uint valid,
    uint tid) {
  for (uint i = tid; i < kRows * D; i += kThreads) {
    uint r = i / D;
    uint d = i - r * D;
    dst[r * rowStride + d * columnStride] = !kChecked || r < valid ? src[r * ld + d] : T(0);
  }
}

template <typename T, uint D>
kernel void attention(
    device T *out [[buffer(0)]],
    device const T *q [[buffer(1)]],
    device const T *k [[buffer(2)]],
    device const T *v [[buffer(3)]],
    constant AttentionParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  // Keys a step: sixteen where a row is 256 bytes or more -- half heads of 128, float ones of 64
  // and 128 -- and thirty-two below. On an M5 Pro with its accelerators off: sixteen took a half
  // head of 128 from 44.5 ms to 35.6 and a float one of 64 from 23.6 to 19.8 over 4096 keys, and
  // cost the narrower rows a few percent; sixty-four was slower everywhere. Padding the staged
  // rows, which MLX does, was slower too -- by a fifth to a third for those 256-byte rows -- and
  // holding the queries in registers rather than reading them again each step, by 4 to 8%.
  constexpr uint kBlockK = D * sizeof(T) >= 256 ? 16 : 32;
  constexpr uint kD = D / 8;
  constexpr uint kK = kBlockK / 8;
  // The staged tiles: the queries (kBlockQ, D); the keys transposed, (D, kBlockK); the values
  // (kBlockK, D), in the same memory as the keys once the scores are taken.
  constexpr uint kStrideQ = D;
  constexpr uint kStrideK = kBlockK;
  constexpr uint kStrideV = D;
  constexpr uint kKeyValues = D * kStrideK > kBlockK * kStrideV ? D * kStrideK : kBlockK * kStrideV;
  threadgroup T queries[kBlockQ * kStrideQ];
  threadgroup T keyValues[kKeyValues];

  uint batch = group.z;
  uint head = group.y;
  uint q0 = group.x * kBlockQ;
  uint kvHead = head / p.headsPerKeyValueHead;
  device const T *qBase = q + batch * p.qStrideB + head * p.qStrideH;
  device const T *kBase = k + batch * p.kStrideB + kvHead * p.kStrideH;
  device const T *vBase = v + batch * p.vStrideB + kvHead * p.vStrideH;

  // The queries, once: rows past the last are zeros.
  uint queriesLeft = p.queryLength - q0;
  if (queriesLeft < kBlockQ) {
    stage<T, D, kBlockQ, true>(queries, kStrideQ, 1, qBase + q0 * p.qStrideL, p.qStrideL, queriesLeft, tid);
  } else {
    stage<T, D, kBlockQ, false>(queries, kStrideQ, 1, qBase + q0 * p.qStrideL, p.qStrideL, kBlockQ, tid);
  }

  // This lane's row, and the keys it may see; a row past the last query sees what the last one
  // does, and is never written. Wraps where there are fewer keys than queries, which only no mask
  // allows, and which no mask does not read.
  short2 at = fragmentCoord(lane);
  uint offset = p.keyLength - p.queryLength;
  uint last = p.queryLength - 1;
  uint position = min(q0 + sg * 8 + uint(at.y), last) + offset;
  uint firstKey = firstKeyOf(p, position);
  uint lastKey = lastKeyOf(p, position);

  // The keys any row of this threadgroup sees, and those every row does, where a block needs no
  // mask. The same for every thread of the group, so that each reaches every barrier in the loop.
  uint firstRow = q0 + offset, lastRow = min(q0 + kBlockQ - 1, last) + offset;
  uint keyBegin = (firstKeyOf(p, firstRow) / kBlockK) * kBlockK;
  uint keyEnd = lastKeyOf(p, lastRow) + 1;
  uint seenFrom = firstKeyOf(p, lastRow), seenTo = lastKeyOf(p, firstRow);

  // The scores go into exp2, so the scale carries log2(e).
  float scale = p.scale * 1.4426950408889634f;
  Matrix o[kD];
  FLINT_UNROLL
  for (uint d = 0; d < kD; ++d) o[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  float rowMaxSoFar = -INFINITY;
  float rowSumSoFar = 0.0f;
  threadgroup const T *myQueries = queries + sg * 8 * kStrideQ;

  for (uint k0 = keyBegin; k0 < keyEnd; k0 += kBlockK) {
    uint keysLeft = p.keyLength - k0;
    bool lastKeys = keysLeft < kBlockK;

    // The keys, transposed: key j, dimension d at keyValues[d * kStrideK + j].
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lastKeys) {
      stage<T, D, kBlockK, true>(keyValues, 1, kStrideK, kBase + k0 * p.kStrideL, p.kStrideL, keysLeft, tid);
    } else {
      stage<T, D, kBlockK, false>(keyValues, 1, kStrideK, kBase + k0 * p.kStrideL, p.kStrideL, kBlockK, tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S = Q K^T, eight dimensions at a time.
    Matrix s[kK];
    FLINT_UNROLL
    for (uint t = 0; t < kK; ++t) s[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    FLINT_UNROLL
    for (uint d = 0; d < kD; ++d) {
      Matrix query = loadMatrix(myQueries + d * 8, kStrideQ, at);
      FLINT_UNROLL
      for (uint t = 0; t < kK; ++t) {
        Matrix keyT = loadMatrix(keyValues + d * 8 * kStrideK + t * 8, kStrideK, at);
        simdgroup_multiply_accumulate(s[t], query, keyT, s[t]);
      }
    }

    // The mask where a row misses any of the block, the scale, and the row's largest score.
    float blockMax = -INFINITY;
    bool whole = k0 >= seenFrom && k0 + kBlockK - 1 <= seenTo;
    FLINT_UNROLL
    for (uint t = 0; t < kK; ++t) {
      FLINT_UNROLL
      for (short j = 0; j < 2; ++j) {
        uint key = k0 + t * 8 + at.x + j;
        float x = s[t].thread_elements()[j] * scale;
        x = whole || (key >= firstKey && key <= lastKey) ? x : -INFINITY;
        s[t].thread_elements()[j] = x;
        blockMax = max(blockMax, x);
      }
    }
    float newMax = max(rowMaxSoFar, rowMax(blockMax));

    // A row that has seen no key yet keeps nothing, and adds nothing: a block diagonal mask leaves
    // a row of a later block nothing in the first keys its threadgroup walks.
    float rescale = newMax == -INFINITY ? 1.0f : exp2(rowMaxSoFar - newMax);
    float blockSum = 0.0f;
    FLINT_UNROLL
    for (uint t = 0; t < kK; ++t) {
      FLINT_UNROLL
      for (short j = 0; j < 2; ++j) {
        float e = newMax == -INFINITY ? 0.0f : exp2(s[t].thread_elements()[j] - newMax);
        s[t].thread_elements()[j] = e;
        blockSum += e;
      }
    }
    rowSumSoFar = rowSumSoFar * rescale + rowSum(blockSum);
    rowMaxSoFar = newMax;
    FLINT_UNROLL
    for (uint d = 0; d < kD; ++d) {
      o[d].thread_elements()[0] *= rescale;
      o[d].thread_elements()[1] *= rescale;
    }

    // The values, where the keys were: key j, dimension d at keyValues[j * kStrideV + d].
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lastKeys) {
      stage<T, D, kBlockK, true>(keyValues, kStrideV, 1, vBase + k0 * p.vStrideL, p.vStrideL, keysLeft, tid);
    } else {
      stage<T, D, kBlockK, false>(keyValues, kStrideV, 1, vBase + k0 * p.vStrideL, p.vStrideL, kBlockK, tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O += P V. Keys past the last are zeros, under P's zeros.
    FLINT_UNROLL
    for (uint d = 0; d < kD; ++d) {
      FLINT_UNROLL
      for (uint t = 0; t < kK; ++t) {
        Matrix value = loadMatrix(keyValues + t * 8 * kStrideV + d * 8, kStrideV, at);
        simdgroup_multiply_accumulate(o[d], s[t], value, o[d]);
      }
    }
  }

  // O / the sum of the row, the rows that are queries. The output is (B, H, Lq, D), contiguous.
  float inverse = rowSumSoFar > 0.0f ? 1.0f / rowSumSoFar : 0.0f;
  uint row = q0 + sg * 8 + uint(at.y);
  if (row < p.queryLength) {
    device T *outRow = out + ((batch * p.numHeads + head) * p.queryLength + row) * D;
    FLINT_UNROLL
    for (uint d = 0; d < kD; ++d) {
      outRow[d * 8 + at.x] = T(o[d].thread_elements()[0] * inverse);
      outRow[d * 8 + at.x + 1] = T(o[d].thread_elements()[1] * inverse);
    }
  }
}

#define FLINT_ATTENTION(T, S, D)                                                                \
  template [[host_name("attention_" #D "_" S)]] kernel void attention<T, D>(                    \
      device T *, device const T *, device const T *, device const T *,                         \
      constant AttentionParams &, uint3, uint, ushort, ushort);

FLINT_ATTENTION(half, "f16", 32)
FLINT_ATTENTION(half, "f16", 64)
FLINT_ATTENTION(half, "f16", 128)
FLINT_ATTENTION(float, "f32", 32)
FLINT_ATTENTION(float, "f32", 64)
FLINT_ATTENTION(float, "f32", 128)
