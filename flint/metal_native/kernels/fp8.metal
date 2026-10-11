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

// The product of a half activation and an FP8 weight on gemm.h: C (M, N) = A (M, K) W^T (K, N),
// the weight W (N, K) in E4M3 with a scale for each of its rows -- each of C's columns -- or one
// for the whole of it. The weight is read a byte an element and widened to half as it is staged,
// which is exact: every E4M3 value is a half. The scale is applied to the sum on the way out,
// where it is one multiply an element rather than one a term.

#include "gemm.h"

/// In step with Fp8MatmulParams in fp8.cc.
struct Fp8MatmulParams {
  uint M;
  uint N;
  uint K;
  uint perRow;  // a scale for each row of the weight, or one for the whole of it
  uint vecA;    // the activation's rows start on a multiple of four elements
  uint vecB;    // and the weight's
};

/// An E4M3 code as a half. Sign, exponent and mantissa moved into a half's places make that half
/// 2^-8 of the value -- the two exponents' biases are 7 and 15 -- subnormals included, so a
/// multiply by 256, exact, finishes it. E4M3's one NaN, 0x7f, comes out as 480 rather than NaN:
/// no weight holds it.
inline half decodeE4M3(uchar code) {
  ushort bits = ushort((code & 0x80) << 8) | ushort((code & 0x7f) << 7);
  return as_type<half>(bits) * 256.0h;
}

struct Fp8MatmulProblem {
  device half *c;
  device const half *a;
  device const uchar *w;
  device const float *scale;
  constant Fp8MatmulParams *p;

  static constexpr constant uint kTileK = TileK<half>::value;
  static constexpr constant uint kStrideA = StrideA<half>::value;
  static constexpr constant uint kPerThreadA = kTileM * kTileK / kGemmThreads;
  static constexpr constant uint kPerThreadB = kTileK * kTileN / kGemmThreads;

  bool hasRowBias() const {
    return false;
  }
  void stageRowBias(threadgroup float *, uint, uint) const {}

  void stageA(threadgroup half *tile, uint m0, uint k0, uint tid) const {
    if (p->vecA && m0 + kTileM <= p->M && k0 + kTileK <= p->K) {
      for (uint j = 0; j < kPerThreadA / 4; ++j) {
        uint v = tid + kGemmThreads * j;
        uint m = v / (kTileK / 4);
        uint k = (v % (kTileK / 4)) * 4;
        half4 x = *reinterpret_cast<device const half4 *>(a + (m0 + m) * p->K + k0 + k);
        for (uint e = 0; e < 4; ++e) tile[m * kStrideA + k + e] = x[e];
      }
      return;
    }
    for (uint j = 0; j < kPerThreadA; ++j) {
      uint k = tid % kTileK;
      uint m = tid / kTileK + (kGemmThreads / kTileK) * j;
      bool in = m0 + m < p->M && k0 + k < p->K;
      tile[m * kStrideA + k] = in ? a[(m0 + m) * p->K + k0 + k] : half(0);
    }
  }

  // The weight is laid out along k: row n of W is column n of B.
  void stageB(threadgroup half *tile, uint k0, uint n0, uint tid) const {
    if (p->vecB && n0 + kTileN <= p->N && k0 + kTileK <= p->K) {
      for (uint j = 0; j < kPerThreadB / 4; ++j) {
        uint v = tid + kGemmThreads * j;
        uint n = v / (kTileK / 4);
        uint k = (v % (kTileK / 4)) * 4;
        uchar4 codes = *reinterpret_cast<device const uchar4 *>(w + (n0 + n) * p->K + k0 + k);
        for (uint e = 0; e < 4; ++e) tile[(k + e) * kStrideB + n] = decodeE4M3(codes[e]);
      }
      return;
    }
    for (uint j = 0; j < kPerThreadB; ++j) {
      uint k = tid % kTileK;
      uint n = tid / kTileK + (kGemmThreads / kTileK) * j;
      bool in = n0 + n < p->N && k0 + k < p->K;
      tile[k * kStrideB + n] = in ? decodeE4M3(w[(n0 + n) * p->K + k0 + k]) : half(0);
    }
  }

  // Every element is scaled by its column's scale on the way out, so none goes as a block.
  bool storeBlock(simdgroup_matrix<float, 8, 8>, uint, uint) const {
    return false;
  }
  void storeC(uint m, uint n, float value) const {
    c[m * p->N + n] = half(value * scale[p->perRow ? n : 0]);
  }
};

kernel void fp8_matmul(
    device half *c [[buffer(0)]],
    device const half *a [[buffer(1)]],
    device const uchar *w [[buffer(2)]],
    device const float *scale [[buffer(3)]],
    constant Fp8MatmulParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  FLINT_GEMM_SHARED(half)

  Fp8MatmulProblem problem;
  problem.c = c;
  problem.a = a;
  problem.w = w;
  problem.scale = scale;
  problem.p = &p;
  gemm<half>(problem, p.M, p.N, p.K, group.xy, tid, sg, lane, tileA, tileB, scratch);
}
