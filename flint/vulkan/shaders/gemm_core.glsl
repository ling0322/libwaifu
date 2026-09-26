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

// A tiled matrix multiply, C (M, N) = A (M, K) B (K, N), accumulated in float.
//
// Each workgroup of 256 threads computes a TILE_M x TILE_N tile of C, TILE_K at a time, staging
// both operands through shared memory as floats; each thread holds a 4 x 4 block of the tile,
// its rows and columns strided by 16 so that neighbouring threads read neighbouring words of
// shared memory. The includer supplies what makes it a particular product:
//
//   float loadA(uint m, uint k)   the operands, already known to be in range
//   float loadB(uint k, uint n)
//   void storeC(uint m, uint n, float value)
//   bool aAlongM()                whether consecutive m (rather than k) of A are adjacent in
//   bool bAlongN()                memory, and the same for n of B -- which decides how the
//                                 threads share out a load, so that adjacent threads read
//                                 adjacent addresses

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16

layout(local_size_x = 256) in;

shared float tileA[TILE_K][TILE_M + 4];
shared float tileB[TILE_K][TILE_N + 4];

void gemm(uint M, uint N, uint K) {
  uint tid = gl_LocalInvocationID.x;
  uint tx = tid % 16;
  uint ty = tid / 16;
  uint m0 = gl_WorkGroupID.y * TILE_M;
  uint n0 = gl_WorkGroupID.x * TILE_N;

  float acc[4][4];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.0;
  }

  bool alongM = aAlongM();
  bool alongN = bAlongN();

  for (uint k0 = 0; k0 < K; k0 += TILE_K) {
    // Each operand's tile is 1024 elements, four for each thread.
    for (uint j = 0; j < 4; ++j) {
      uint m, k;
      if (alongM) {
        m = tid % TILE_M;
        k = tid / TILE_M + 4 * j;
      } else {
        k = tid % TILE_K;
        m = tid / TILE_K + 16 * j;
      }
      tileA[k][m] = (m0 + m < M && k0 + k < K) ? loadA(m0 + m, k0 + k) : 0.0;

      uint n;
      if (alongN) {
        n = tid % TILE_N;
        k = tid / TILE_N + 4 * j;
      } else {
        k = tid % TILE_K;
        n = tid / TILE_K + 16 * j;
      }
      tileB[k][n] = (n0 + n < N && k0 + k < K) ? loadB(k0 + k, n0 + n) : 0.0;
    }
    barrier();

    for (uint k = 0; k < TILE_K; ++k) {
      float a[4];
      float b[4];
      for (uint i = 0; i < 4; ++i) {
        a[i] = tileA[k][ty + 16 * i];
        b[i] = tileB[k][tx + 16 * i];
      }
      for (uint i = 0; i < 4; ++i) {
        for (uint j = 0; j < 4; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
      }
    }
    barrier();
  }

  for (uint i = 0; i < 4; ++i) {
    uint m = m0 + ty + 16 * i;
    if (m >= M) continue;
    for (uint j = 0; j < 4; ++j) {
      uint n = n0 + tx + 16 * j;
      if (n < N) storeC(m, n, acc[i][j]);
    }
  }
}
