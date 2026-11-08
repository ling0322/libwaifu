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

// Scaled dot product attention on the neural accelerators: kernels/attention.metal's online
// softmax, with both products Metal 4's matmul2d on 16 x 16 fragments held in registers. The way
// of it is MLX's steel_attention_nax (MIT, Copyright 2025 Apple Inc.): no threadgroup memory, the
// operands read from the device straight into fragments, and the softmax done on the fragments'
// elements where they are.
//
// A threadgroup takes kBlockQ queries of one head of one batch -- the grid is (query blocks,
// heads, batch) -- and each of its four simdgroups sixteen of them, walking the keys kBlockK at a
// time on its own: S = Q K^T in float, the softmax of the scores so far, O = O * rescale + P V.
// The scores are float throughout, so a model whose scores leave half's range -- CosyVoice3's DiT
// reaches 97,000 -- is answered as in float.
//
// A lane holds eight elements of a 16 x 16 fragment: rows r and r + 8, four columns from c, where
// (c, r) is fragmentCoord(). That is the layout of the cooperative tensors of a 16 x 32 x 16
// matmul2d with relaxed precision, as MLX reads them -- Metal calls it the implementation's, and
// without relaxed precision it is another, which this does not read. The tests check the answer
// against a reference, so a GPU that lays them out otherwise fails there rather than quietly.
//
// Half only: relaxed precision takes float operands at about half's precision -- 2e-4 off over
// 300 keys, where flint holds float to 1e-4 -- so float is kernels/attention.metal's.

#include <metal_simdgroup>
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

/// In step with AttentionParams in attention.cc, which the kernel in kernels/attention.metal
/// takes too.
struct AttentionParams {
  uint numHeads;
  uint headsPerKeyValueHead;
  uint queryLength;
  uint keyLength;
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
  uint maskType;
  uint blockSize;
};

constant constexpr uint kMaskNone = 0;
constant constexpr uint kMaskCausal = 1;
constant constexpr uint kMaskBlockDiagonal = 3;

/// The first and the last key a query at `position` -- its index plus keyLength - queryLength --
/// may see, as kernels/attention.metal has them.
inline uint firstKeyOf(constant AttentionParams &p, uint position) {
  if (p.maskType != kMaskBlockDiagonal) return 0;
  return (position / p.blockSize) * p.blockSize;
}

inline uint lastKeyOf(constant AttentionParams &p, uint position) {
  if (p.maskType == kMaskNone) return p.keyLength - 1;
  if (p.maskType == kMaskCausal) return position;
  return min(p.keyLength - 1, (position / p.blockSize) * p.blockSize + p.blockSize - 1);
}

// In step with kNaxBlockQ in attention.cc.
constant constexpr int kBlockQ = 64;
constant constexpr int kBlockK = 32;

template <typename T>
using Frag = vec<T, 8>;

/// Where this lane's elements of a fragment are: (column, row) of the first of them.
inline short2 fragmentCoord(ushort lane) {
  short quad = lane >> 2;
  short row = (quad & 4) | ((lane >> 1) & 3);
  short column = ((quad & 2) | (lane & 1)) * 4;
  return short2(column, row);
}

#define FLINT_UNROLL _Pragma("clang loop unroll(full)")

/// The fragment at `src`, its rows `ld` apart. Checked, rows from `rows` on are zeros, and not
/// read; unchecked, every row is there, which is every block but the last of the keys or queries.
template <bool kChecked, typename T>
inline Frag<T> load(device const T *src, uint ld, short2 at, int rows) {
  Frag<T> f;
  FLINT_UNROLL
  for (short i = 0; i < 2; ++i) {
    int r = at.y + i * 8;
    device const T *row = src + r * ld + at.x;
    FLINT_UNROLL
    for (short j = 0; j < 4; ++j) f[i * 4 + j] = !kChecked || r < rows ? row[j] : T(0);
  }
  return f;
}

/// c0 and c1, side by side 16 x 32, plus a (16 x 16) times b0 and b1 side by side (16 x 32): b
/// as it is, or b0 and b1 each the transpose of the fragment held.
template <typename C, typename A, typename B, bool kTransposeB>
inline void multiplyAccumulate(
    thread Frag<C> &c0,
    thread Frag<C> &c1,
    thread const Frag<A> &a,
    thread const Frag<B> &b0,
    thread const Frag<B> &b1) {
  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, kTransposeB, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, execution_simdgroup> op;
  auto left = op.template get_left_input_cooperative_tensor<A, B, C>();
  auto right = op.template get_right_input_cooperative_tensor<A, B, C>();
  auto out = op.template get_destination_cooperative_tensor<
      remove_addrspace_t<decltype(left)>, remove_addrspace_t<decltype(right)>, C>();
  FLINT_UNROLL
  for (short i = 0; i < 8; ++i) {
    left[i] = a[i];
    right[i] = b0[i];
    right[8 + i] = b1[i];
    out[i] = c0[i];
    out[8 + i] = c1[i];
  }
  op.run(left, right, out);
  FLINT_UNROLL
  for (short i = 0; i < 8; ++i) {
    c0[i] = out[i];
    c1[i] = out[8 + i];
  }
}

/// The largest, or the sum, of a row over the four lanes that hold it: they differ in bits 0 and 3.
inline float rowMax(float x) {
  x = max(x, simd_shuffle_xor(x, ushort(1)));
  return max(x, simd_shuffle_xor(x, ushort(8)));
}

inline float rowSum(float x) {
  x += simd_shuffle_xor(x, ushort(1));
  return x + simd_shuffle_xor(x, ushort(8));
}

/// S = Q K^T: sixteen queries by kBlockK keys, sixteen dimensions at a time.
template <bool kChecked, typename T, int D>
inline void scoresOf(
    thread Frag<float> *s,
    device const T *qRows,
    uint qld,
    int queries,
    device const T *kRows,
    uint kld,
    int keys,
    short2 at) {
  FLINT_UNROLL
  for (int t = 0; t < kBlockK / 16; ++t) s[t] = Frag<float>(0.0f);
  FLINT_UNROLL
  for (int d = 0; d < D / 16; ++d) {
    Frag<T> qf = load<kChecked>(qRows + d * 16, qld, at, queries);
    FLINT_UNROLL
    for (int t = 0; t < kBlockK / 16; t += 2) {
      Frag<T> k0 = load<kChecked>(kRows + (t * 16) * kld + d * 16, kld, at, keys - t * 16);
      Frag<T> k1 = load<kChecked>(kRows + (t * 16 + 16) * kld + d * 16, kld, at, keys - t * 16 - 16);
      multiplyAccumulate<float, T, T, true>(s[t], s[t + 1], qf, k0, k1);
    }
  }
}

/// O += P V, thirty-two dimensions at a time. Keys past the last are zeros, under P's zeros.
template <bool kChecked, typename T, int D>
inline void accumulate(
    thread Frag<float> *o,
    thread const Frag<float> *s,
    device const T *vRows,
    uint vld,
    int keys,
    short2 at) {
  FLINT_UNROLL
  for (int d = 0; d < D / 16; d += 2) {
    FLINT_UNROLL
    for (int t = 0; t < kBlockK / 16; ++t) {
      device const T *rows = vRows + (t * 16) * vld + d * 16;
      Frag<T> v0 = load<kChecked>(rows, vld, at, keys - t * 16);
      Frag<T> v1 = load<kChecked>(rows + 16, vld, at, keys - t * 16);
      multiplyAccumulate<float, float, T, false>(o[d], o[d + 1], s[t], v0, v1);
    }
  }
}

template <typename T, int D>
kernel void nax_attention(
    device T *out [[buffer(0)]],
    device const T *q [[buffer(1)]],
    device const T *k [[buffer(2)]],
    device const T *v [[buffer(3)]],
    constant AttentionParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int kD = D / 16;
  constexpr int kK = kBlockK / 16;

  uint batch = group.z;
  uint head = group.y;
  uint q0 = group.x * kBlockQ + sg * 16;
  // A simdgroup past the last query has nothing to do, and no barrier to wait at.
  if (q0 >= p.queryLength) return;
  uint kvHead = head / p.headsPerKeyValueHead;
  device const T *qBase = q + batch * p.qStrideB + head * p.qStrideH + q0 * p.qStrideL;
  device const T *kBase = k + batch * p.kStrideB + kvHead * p.kStrideH;
  device const T *vBase = v + batch * p.vStrideB + kvHead * p.vStrideH;
  int queries = int(p.queryLength - q0);

  // This lane's two rows, and the keys each may see; a row past the last query sees what the last
  // one does, and is never written. Wraps where there are fewer keys than queries, which only no
  // mask allows, and which no mask does not read.
  short2 at = fragmentCoord(lane);
  uint offset = p.keyLength - p.queryLength;
  uint last = p.queryLength - 1;
  uint positionA = min(q0 + uint(at.y), last) + offset;
  uint positionB = min(q0 + uint(at.y) + 8, last) + offset;
  uint firstA = firstKeyOf(p, positionA), lastA = lastKeyOf(p, positionA);
  uint firstB = firstKeyOf(p, positionB), lastB = lastKeyOf(p, positionB);

  // The keys any row of this simdgroup sees: the rows are queries in order, so the first row's
  // first key and the last row's last key bound them all. And those every row sees: from the last
  // row's first key to the first row's last, where a block needs no mask.
  uint firstRow = q0 + offset, lastRow = min(q0 + 15, last) + offset;
  uint keyBegin = (firstKeyOf(p, firstRow) / kBlockK) * kBlockK;
  uint keyEnd = lastKeyOf(p, lastRow) + 1;
  uint seenFrom = firstKeyOf(p, lastRow), seenTo = lastKeyOf(p, firstRow);

  // The scores go into exp2, so the scale carries log2(e).
  float scale = p.scale * 1.4426950408889634f;
  Frag<float> o[kD];
  for (int d = 0; d < kD; ++d) o[d] = Frag<float>(0.0f);
  float maxA = -INFINITY, maxB = -INFINITY;
  float sumA = 0.0f, sumB = 0.0f;

  for (uint k0 = keyBegin; k0 < keyEnd; k0 += kBlockK) {
    int keys = int(p.keyLength - k0);
    device const T *kBlock = kBase + k0 * p.kStrideL;
    device const T *vBlock = vBase + k0 * p.vStrideL;

    bool lastKeys = keys < kBlockK;
    Frag<float> s[kK];
    if (lastKeys || queries < 16) {
      scoresOf<true, T, D>(s, qBase, p.qStrideL, queries, kBlock, p.kStrideL, keys, at);
    } else {
      scoresOf<false, T, D>(s, qBase, p.qStrideL, queries, kBlock, p.kStrideL, keys, at);
    }

    // The mask where a row misses any of the block, the scale, and each row's largest score.
    float blockMaxA = -INFINITY, blockMaxB = -INFINITY;
    if (k0 >= seenFrom && k0 + kBlockK - 1 <= seenTo) {
      FLINT_UNROLL
      for (int t = 0; t < kK; ++t) {
        FLINT_UNROLL
        for (short j = 0; j < 4; ++j) {
          s[t][j] *= scale;
          s[t][4 + j] *= scale;
          blockMaxA = max(blockMaxA, s[t][j]);
          blockMaxB = max(blockMaxB, s[t][4 + j]);
        }
      }
    } else {
      FLINT_UNROLL
      for (int t = 0; t < kK; ++t) {
        FLINT_UNROLL
        for (short j = 0; j < 4; ++j) {
          uint key = k0 + t * 16 + at.x + j;
          float a = key >= firstA && key <= lastA ? s[t][j] * scale : -INFINITY;
          float b = key >= firstB && key <= lastB ? s[t][4 + j] * scale : -INFINITY;
          s[t][j] = a;
          s[t][4 + j] = b;
          blockMaxA = max(blockMaxA, a);
          blockMaxB = max(blockMaxB, b);
        }
      }
    }
    float newMaxA = max(maxA, rowMax(blockMaxA));
    float newMaxB = max(maxB, rowMax(blockMaxB));

    // A row that has seen no key yet keeps nothing, and adds nothing: a block diagonal mask leaves
    // a row of a later block nothing in the first keys its simdgroup walks.
    float rescaleA = newMaxA == -INFINITY ? 1.0f : exp2(maxA - newMaxA);
    float rescaleB = newMaxB == -INFINITY ? 1.0f : exp2(maxB - newMaxB);
    float blockSumA = 0.0f, blockSumB = 0.0f;
    FLINT_UNROLL
    for (int t = 0; t < kK; ++t) {
      FLINT_UNROLL
      for (short j = 0; j < 4; ++j) {
        float a = newMaxA == -INFINITY ? 0.0f : exp2(s[t][j] - newMaxA);
        float b = newMaxB == -INFINITY ? 0.0f : exp2(s[t][4 + j] - newMaxB);
        s[t][j] = a;
        s[t][4 + j] = b;
        blockSumA += a;
        blockSumB += b;
      }
    }
    sumA = sumA * rescaleA + rowSum(blockSumA);
    sumB = sumB * rescaleB + rowSum(blockSumB);
    maxA = newMaxA;
    maxB = newMaxB;
    FLINT_UNROLL
    for (int d = 0; d < kD; ++d) {
      FLINT_UNROLL
      for (short j = 0; j < 4; ++j) {
        o[d][j] *= rescaleA;
        o[d][4 + j] *= rescaleB;
      }
    }

    if (lastKeys) {
      accumulate<true, T, D>(o, s, vBlock, p.vStrideL, keys, at);
    } else {
      accumulate<false, T, D>(o, s, vBlock, p.vStrideL, keys, at);
    }
  }

  // O / the sum of each row, the rows that are queries. The output is (B, H, Lq, D), contiguous.
  float inverseA = sumA > 0.0f ? 1.0f / sumA : 0.0f;
  float inverseB = sumB > 0.0f ? 1.0f / sumB : 0.0f;
  device T *outBase = out + ((batch * p.numHeads + head) * p.queryLength + q0) * D;
  for (int d = 0; d < kD; ++d) {
    for (short i = 0; i < 2; ++i) {
      int r = at.y + i * 8;
      if (r >= queries) continue;
      float inverse = i == 0 ? inverseA : inverseB;
      for (short j = 0; j < 4; ++j) outBase[r * D + d * 16 + at.x + j] = T(o[d][i * 4 + j] * inverse);
    }
  }
}

#define FLINT_NAX_ATTENTION(T, S, D)                                                            \
  template [[host_name("nax_attention_" #D "_" S)]] kernel void nax_attention<T, D>(            \
      device T *, device const T *, device const T *, device const T *,                         \
      constant AttentionParams &, uint3, ushort, ushort);

FLINT_NAX_ATTENTION(half, "f16", 32)
FLINT_NAX_ATTENTION(half, "f16", 64)
FLINT_NAX_ATTENTION(half, "f16", 128)
