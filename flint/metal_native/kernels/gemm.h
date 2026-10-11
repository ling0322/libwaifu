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

// A tiled matrix multiply on simdgroup matrices, C (M, N) = A (M, K) B (K, N), the operands in
// their own type T and the sum in float.
//
// A threadgroup of kGemmThreads threads -- four simdgroups, two by two -- computes a kTileM x
// kTileN tile of C, TileK<T> at a time, staging both operands through threadgroup memory; each
// simdgroup holds a 32 x 32 quarter of the tile as four by four 8 x 8 accumulators. The sizes are
// the ones that do well on every Apple GPU from the M1 on, not the best for any one of them: a
// tile of 64 x 64 keeps the threadgroup memory under 16 KB, so that a core runs two threadgroups
// at once even on the M1.
//
// The caller hands in a Problem that makes it a particular product:
//
//   bool hasRowBias()
//   void stageRowBias(threadgroup float *table, uint m0, uint tid)
//       whether C starts from a value per row rather than zero, and if it does, that value for
//       rows m0 : m0 + kTileM written into `table`, (kTileM, 8), each row's value eight times
//   void stageA(threadgroup T *tile, uint m0, uint k0, uint tid)
//       A[m0 : m0 + kTileM, k0 : k0 + TileK<T>] into `tile`, (kTileM, kStrideA<T>), m by k, with
//       zeros where it runs past A
//   void stageB(threadgroup T *tile, uint k0, uint n0, uint tid)
//       B[k0 : k0 + TileK<T>, n0 : n0 + kTileN] into `tile`, (TileK<T>, kStrideB), k by n
//   bool storeBlock(simdgroup_matrix<float, 8, 8> block, uint m, uint n)
//       writes the 8 x 8 block of C at (m, n) by itself, if it can; false leaves it to
//   void storeC(uint m, uint n, float value)
//       which writes one element, already known to be in range
//
// and the threadgroup memory, which only a kernel can declare: FLINT_GEMM_SHARED(T) declares it.

#pragma once

#include <metal_simdgroup_matrix>

#include "common.h"

constant constexpr uint kTileM = 64;
constant constexpr uint kTileN = 64;
constant constexpr uint kGemmThreads = 128;

// The depth of a step: 32 for half and 16 for float, the same number of bytes.
template <typename T>
struct TileK {
  static constant constexpr uint value = 64 / sizeof(T);
};

// The rows of the staged tiles are padded, so that the simdgroup loads of one column of 8 x 8
// blocks do not all fall on the same banks.
template <typename T>
struct StrideA {
  static constant constexpr uint value = TileK<T>::value + 8;
};
constant constexpr uint kStrideB = kTileN + 8;

// The scratch holds the row bias table at the start, and an 8 x 8 for each simdgroup at the end.
#define FLINT_GEMM_SHARED(T)                                   \
  threadgroup T tileA[kTileM * StrideA<T>::value];             \
  threadgroup T tileB[TileK<T>::value * kStrideB];             \
  threadgroup float scratch[kTileM * 8];

template <typename T, typename Problem>
inline void gemm(
    thread const Problem &problem,
    uint M,
    uint N,
    uint K,
    uint2 tile,
    uint tid,
    uint sg,
    uint lane,
    threadgroup T *tileA,
    threadgroup T *tileB,
    threadgroup float *scratch) {
  constexpr uint kTileK = TileK<T>::value;
  constexpr uint kStrideA = StrideA<T>::value;

  uint m0 = tile.y * kTileM;
  uint n0 = tile.x * kTileN;
  uint sgM = (sg / 2) * 32;
  uint sgN = (sg % 2) * 32;

  simdgroup_matrix<float, 8, 8> acc[4][4];
  if (problem.hasRowBias()) {
    problem.stageRowBias(scratch, m0, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
      for (uint j = 0; j < 4; ++j) simdgroup_load(acc[i][j], scratch + (sgM + 8 * i) * 8, 8);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } else {
    for (uint i = 0; i < 4; ++i) {
      for (uint j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }

  for (uint k0 = 0; k0 < K; k0 += kTileK) {
    problem.stageA(tileA, m0, k0, tid);
    problem.stageB(tileB, k0, n0, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = 0; kk < kTileK; kk += 8) {
      simdgroup_matrix<T, 8, 8> a[4];
      simdgroup_matrix<T, 8, 8> b[4];
      for (uint i = 0; i < 4; ++i) {
        simdgroup_load(a[i], tileA + (sgM + 8 * i) * kStrideA + kk, kStrideA);
      }
      for (uint j = 0; j < 4; ++j) simdgroup_load(b[j], tileB + kk * kStrideB + sgN + 8 * j, kStrideB);
      for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // A block the problem cannot write by itself goes out through an 8 x 8 of threadgroup memory
  // the simdgroup has to itself, which leaves where in a simdgroup matrix each element is to
  // Metal.
  threadgroup float *block = scratch + sg * 64;
  for (uint i = 0; i < 4; ++i) {
    for (uint j = 0; j < 4; ++j) {
      uint m = m0 + sgM + 8 * i;
      uint n = n0 + sgN + 8 * j;
      if (m >= M || n >= N) continue;
      if (m + 8 <= M && n + 8 <= N && problem.storeBlock(acc[i][j], m, n)) continue;

      simdgroup_store(acc[i][j], block, 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < 64; e += 32) {
        uint mm = m + e / 8;
        uint nn = n + e % 8;
        if (mm < M && nn < N) problem.storeC(mm, nn, block[e]);
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

/// `block` as T, element by element: the same matrix in the same layout, whatever that is.
template <typename T>
inline simdgroup_matrix<T, 8, 8> convertBlock(simdgroup_matrix<float, 8, 8> block) {
  simdgroup_matrix<T, 8, 8> out;
  out.thread_elements()[0] = T(block.thread_elements()[0]);
  out.thread_elements()[1] = T(block.thread_elements()[1]);
  return out;
}
