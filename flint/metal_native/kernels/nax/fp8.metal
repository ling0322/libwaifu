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

// The half activation times an E4M3 weight on the neural accelerators, as kernels/fp8.metal has it:
// C (M, N) = A (M, K) W^T, W (N, K) with a scale a row or one for all of it. The activation is
// read where it is. The weight is read kTileK columns at a time, widened to half -- exactly -- into
// threadgroup memory, a row of W to a row of the tile, and the next chunk read into registers
// while matmul2d works on this one. The sum is in float, and the scale is applied on the way out.
//
// Metal 4.1 has a format for E4M3 that matmul2d reads directly; this is Metal 4.0, which does not.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

/// In step with Fp8MatmulParams in fp8.cc, and kernels/fp8.metal's.
struct Fp8MatmulParams {
  uint M;
  uint N;
  uint K;
  uint perRow;
  uint vecA;
  uint vecB;  // the weight's rows start on a multiple of four bytes
};

// In step with kNaxFp8TileM and kNaxFp8TileN in fp8.cc.
constant constexpr int kTileM = 64;   // activation rows
constant constexpr int kTileN = 128;  // weight rows
constant constexpr int kTileK = 32;
constant constexpr uint kThreads = 128;
// Each thread's share of a chunk of the weight: kTileN x kTileK bytes, four at a time.
constant constexpr uint kLoads = kTileN * kTileK / 4 / kThreads;

using Extents = dextents<int32_t, 2>;

/// As kernels/fp8.metal's: sign, exponent and mantissa in a half's places are 2^-8 of the value.
inline half decodeE4M3(uchar code) {
  ushort bits = ushort((code & 0x80) << 8) | ushort((code & 0x7f) << 7);
  return as_type<half>(bits) * 256.0h;
}

/// A chunk of the weight on its way: four codes a load, zero past an edge.
struct Chunk {
  uchar4 codes[kLoads];
};

inline void readChunk(
    thread Chunk &chunk,
    device const uchar *w,
    constant Fp8MatmulParams &p,
    uint n0,
    uint k0,
    uint tid) {
  bool inside = p.vecB && n0 + kTileN <= p.N && k0 + kTileK <= p.K;
  for (uint j = 0; j < kLoads; ++j) {
    uint v = tid + kThreads * j;
    uint n = v / (kTileK / 4);
    uint k = (v % (kTileK / 4)) * 4;
    if (inside) {
      chunk.codes[j] = *reinterpret_cast<device const uchar4 *>(w + (n0 + n) * p.K + k0 + k);
      continue;
    }
    uchar4 codes = uchar4(0);
    for (uint e = 0; e < 4; ++e) {
      if (n0 + n < p.N && k0 + k + e < p.K) codes[e] = w[(n0 + n) * p.K + k0 + k + e];
    }
    chunk.codes[j] = codes;
  }
}

// Row n of the tile is row n0 + n of W, the kTileK columns of the chunk.
inline void writeChunk(thread const Chunk &chunk, threadgroup half *tile, uint tid) {
  for (uint j = 0; j < kLoads; ++j) {
    uint v = tid + kThreads * j;
    uint n = v / (kTileK / 4);
    uint k = (v % (kTileK / 4)) * 4;
    for (uint e = 0; e < 4; ++e) tile[n * kTileK + k + e] = decodeE4M3(chunk.codes[j][e]);
  }
}

kernel void nax_fp8_matmul(
    device half *c [[buffer(0)]],
    device half *a [[buffer(1)]],
    device const uchar *w [[buffer(2)]],
    device const float *scale [[buffer(3)]],
    constant Fp8MatmulParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup half tile[kTileN * kTileK];

  uint m0 = group.y * kTileM;
  uint n0 = group.x * kTileN;

  // Extents run from the innermost dimension out: the activation is (K, M) to a tensor, and the
  // tile, W's rows across, (K, N) -- the right operand transposed.
  tensor<device half, Extents, tensor_inline> A(
      a, Extents(p.K, p.M), array<int32_t, 2>{1, int32_t(p.K)});
  tensor<threadgroup half, Extents, tensor_inline> B(
      tile, Extents(kTileK, kTileN), array<int32_t, 2>{1, kTileK});

  constexpr auto descriptor = matmul2d_descriptor(
      kTileM,
      kTileN,
      kTileK,
      false,
      true,
      false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<4>> op;

  auto first = A.slice(0, m0);
  auto sum = op.template get_destination_cooperative_tensor<decltype(first), decltype(B), float>();
  for (uint16_t i = 0; i < sum.get_capacity(); ++i) {
    if (sum.is_valid_element(i)) sum[i] = 0.0f;
  }

  Chunk chunk;
  readChunk(chunk, w, p, n0, 0, tid);
  for (uint k0 = 0; k0 < p.K; k0 += kTileK) {
    writeChunk(chunk, tile, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (k0 + kTileK < p.K) readChunk(chunk, w, p, n0, k0 + kTileK, tid);
    auto tileA = A.slice(k0, m0);
    op.run(tileA, B, sum);
    // The tile is written again for the next chunk; every simdgroup has to be done reading it.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint16_t i = 0; i < sum.get_capacity(); ++i) {
    if (!sum.is_valid_element(i)) continue;
    auto at = sum.get_multidimensional_index(i);  // (n, m) within the tile
    uint n = n0 + uint(at[0]);
    uint m = m0 + uint(at[1]);
    if (m >= p.M || n >= p.N) continue;
    c[m * p.N + n] = half(sum[i] * scale[p.perRow ? n : 0]);
  }
}
