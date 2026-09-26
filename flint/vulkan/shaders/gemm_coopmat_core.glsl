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

// A half precision product, C (M, N) = A (M, K) B (K, N) + bias (N) accumulated in float, through
// cooperative matrices: 16 x 16 x 16 tiles multiplied by a subgroup together, which is what drives
// the tensor cores on the cards that have them.
//
// What makes it fast is how the operands arrive, sixteen bytes -- eight halves -- at a time, and
// the next slice of K read into registers while the current one is being multiplied. So the
// includer hands over eight elements at once, as a uvec4, each function returning zeros for
// whatever lies outside the operand, with K a multiple of eight:
//
//   uvec4 loadA8(uint m, uint k)    A[m, k .. k + 8), k a multiple of eight
//   uvec4 loadB8K(uint n, uint k)   B[k .. k + 8, n], when B is laid out along K
//   uvec4 loadB8N(uint k, uint n)   B[k, n .. n + 8), when it is laid out along N
//   bool bAlongN()                  which of the two
//   float biasOf(uint n)            what column n starts from: its bias, or zero
//   void storeTile(tile, m, n)      a whole 16 x 16 tile of C whose corner is (m, n)
//   void storeC(uint m, uint n, float value)
//                                   one element, for the tiles on the edges of C
//
// Defining C_ALONG_M says that consecutive m rather than n of C are adjacent in memory, which
// decides the order the elements on the edges go out in.
//
// A workgroup is eight subgroups of 32 and computes a 128 x 128 tile of C, TILE_K of K at a
// time. The subgroups stand in a 4 x 2 grid, each owning 32 x 64 of the tile: two rows of four
// 16 x 16 accumulators. Shared memory holds the tiles as uvec4, which the cooperative loads count
// in, each row padded by one uvec4 -- eight halves -- so that rows do not all start on the same
// bank.
//
// Every loop is marked for unrolling. Without that the kernel ran at half the speed on an RTX
// 5060 Ti, most likely because a loop left rolled indexes the accumulator array at run time,
// which keeps it out of registers.

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_control_flow_attributes : require

#define TILE_M 128
#define TILE_N 128
#define TILE_K 64
#define SUBGROUP_M 32
#define SUBGROUP_N 64

#define CHUNKS_K (TILE_K / 8)            // uvec4 in TILE_K halves
#define CHUNKS_N (TILE_N / 8)
#define ROW_A (CHUNKS_K + 1)             // uvec4 in a row of tileA
#define ROW_BK (CHUNKS_K + 1)            // in a row of tileB laid out along K, one row per n
#define ROW_BN (CHUNKS_N + 1)            // in a row of tileB laid out along N, one row per k
#define LOADS (TILE_M * CHUNKS_K / 256)  // uvec4 of each operand each thread loads per slice

#define Accumulator coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>
#define HalfTile coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>

layout(local_size_x = 256) in;

shared uvec4 tileA[TILE_M * ROW_A];
shared uvec4 tileB[TILE_N * ROW_BK];  // at least TILE_K * ROW_BN
shared float scratch[8 * 16 * 16];    // a 16 x 16 tile for each subgroup

void gemm(uint M, uint N, uint K) {
  uint tid = gl_LocalInvocationID.x;
  uint subgroup = gl_SubgroupID;
  uint lane = gl_SubgroupInvocationID;
  uint m0 = gl_WorkGroupID.y * TILE_M;
  uint n0 = gl_WorkGroupID.x * TILE_N;
  uint subM = (subgroup / 2) * SUBGROUP_M;
  uint subN = (subgroup % 2) * SUBGROUP_N;
  uint base = subgroup * 256;
  bool alongN = bAlongN();

  // The accumulators start from the bias, a column of it at a time through the subgroup's tile
  // of scratch, so that nothing has to add it on the way out.
  Accumulator acc[2][4];
  [[unroll]] for (uint j = 0; j < 4; ++j) {
    [[unroll]] for (uint e = 0; e < 256; e += 32) {
      uint n = n0 + subN + 16 * j + (e + lane) % 16;
      scratch[base + e + lane] = n < N ? biasOf(n) : 0.0;
    }
    subgroupMemoryBarrierShared();
    subgroupBarrier();
    coopMatLoad(acc[0][j], scratch, base, 16, gl_CooperativeMatrixLayoutRowMajor);
    acc[1][j] = acc[0][j];
    subgroupBarrier();
  }

  // Each operand's tile is LOADS uvec4 for each thread. For A and for B along K a thread takes
  // chunk `c` of row `r` of 128 rows of CHUNKS_K; for B along N, of TILE_K rows of CHUNKS_N.
  uvec4 nextA[LOADS];
  uvec4 nextB[LOADS];

  // The slice of K starting at k0, into registers.
  #define FETCH(k0)                                                                    \
    [[unroll]] for (uint j = 0; j < LOADS; ++j) {                                      \
      uint index = tid + 256 * j;                                                      \
      uint r = index / CHUNKS_K;                                                       \
      uint c = index % CHUNKS_K;                                                       \
      nextA[j] = loadA8(m0 + r, (k0) + 8 * c);                                         \
      if (alongN) {                                                                    \
        nextB[j] = loadB8N((k0) + index / CHUNKS_N, n0 + 8 * (index % CHUNKS_N));      \
      } else {                                                                         \
        nextB[j] = loadB8K(n0 + r, (k0) + 8 * c);                                      \
      }                                                                                \
    }

  FETCH(0);
  for (uint k0 = 0; k0 < K; k0 += TILE_K) {
    [[unroll]] for (uint j = 0; j < LOADS; ++j) {
      uint index = tid + 256 * j;
      tileA[(index / CHUNKS_K) * ROW_A + index % CHUNKS_K] = nextA[j];
      if (alongN) {
        tileB[(index / CHUNKS_N) * ROW_BN + index % CHUNKS_N] = nextB[j];
      } else {
        tileB[(index / CHUNKS_K) * ROW_BK + index % CHUNKS_K] = nextB[j];
      }
    }
    barrier();

    if (k0 + TILE_K < K) {
      FETCH(k0 + TILE_K);
    }

    [[unroll]] for (uint kk = 0; kk < CHUNKS_K; kk += 2) {
      coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a[2];
      coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> b[4];
      [[unroll]] for (uint i = 0; i < 2; ++i) {
        coopMatLoad(a[i], tileA, (subM + 16 * i) * ROW_A + kk, ROW_A,
                    gl_CooperativeMatrixLayoutRowMajor);
      }
      [[unroll]] for (uint j = 0; j < 4; ++j) {
        if (alongN) {
          coopMatLoad(b[j], tileB, (8 * kk) * ROW_BN + (subN + 16 * j) / 8, ROW_BN,
                      gl_CooperativeMatrixLayoutRowMajor);
        } else {
          coopMatLoad(b[j], tileB, (subN + 16 * j) * ROW_BK + kk, ROW_BK,
                      gl_CooperativeMatrixLayoutColumnMajor);
        }
      }
      [[unroll]] for (uint i = 0; i < 2; ++i) {
        [[unroll]] for (uint j = 0; j < 4; ++j) acc[i][j] = coopMatMulAdd(a[i], b[j], acc[i][j]);
      }
    }
    barrier();
  }

  // A tile wholly inside C goes out as a tile. One on its edges goes through the subgroup's
  // scratch, so that storeC() sees one element at a time and what lies past the edge is left
  // alone.
  [[unroll]] for (uint i = 0; i < 2; ++i) {
    [[unroll]] for (uint j = 0; j < 4; ++j) {
      uint mTile = m0 + subM + 16 * i;
      uint nTile = n0 + subN + 16 * j;
      if (mTile + 16 <= M && nTile + 16 <= N) {
        storeTile(HalfTile(acc[i][j]), mTile, nTile);
        continue;
      }

      coopMatStore(acc[i][j], scratch, base, 16, gl_CooperativeMatrixLayoutRowMajor);
      subgroupMemoryBarrierShared();
      subgroupBarrier();
      [[unroll]] for (uint e = 0; e < 256; e += 32) {
#ifdef C_ALONG_M
        uint row = (e + lane) % 16;
        uint column = (e + lane) / 16;
#else
        uint row = (e + lane) / 16;
        uint column = (e + lane) % 16;
#endif
        uint m = mTile + row;
        uint n = nTile + column;
        if (m < M && n < N) storeC(m, n, scratch[base + row * 16 + column]);
      }
      subgroupBarrier();
    }
  }
}
