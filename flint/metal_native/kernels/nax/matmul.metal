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

// The batched matrix multiply on the neural accelerators, through Metal 4's matmul2d: for every
// batch (b0, b1), C[b0, b1] = A[b0, b1] B[b0, b1], with C contiguous. A is laid out along K, its
// rows lda apart; B along N, its rows (of K) ldb apart, or -- the NT kernels, a linear layer's
// weight -- along K, its rows (of N) ldb apart. A batch stride of zero broadcasts. The batch is the
// grid's z, b0 * batch1 + b1.
//
// A threadgroup of four simdgroups computes a kTileM x kTileN tile of C over the whole of K, which
// matmul2d walks itself, edges included. The sum is the accelerators', in float.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

/// In step with NaxGemmParams in matmul.cc.
struct NaxGemmParams {
  uint M;
  uint N;
  uint K;
  uint lda;
  uint ldb;
  uint batch1;
  uint strideA0;
  uint strideA1;
  uint strideB0;
  uint strideB1;
};

// Chosen on an M5 Pro over SDXL's products, among 32, 64 and 128 on each side. In step with
// kNaxTileM and kNaxTileN in matmul.cc.
constant constexpr int kTileM = 64;
constant constexpr int kTileN = 128;

using Extents = dextents<int32_t, 2>;

template <typename T, bool kTransposeB>
kernel void nax_matmul(
    device T *c [[buffer(0)]],
    device T *a [[buffer(1)]],
    device T *b [[buffer(2)]],
    constant NaxGemmParams &p [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]]) {
  uint batch = group.z;
  uint b0 = batch / p.batch1;
  uint b1 = batch - b0 * p.batch1;

  // A tensor's extents run from the innermost dimension out: A is (K, M) to it, and C (N, M).
  tensor<device T, Extents, tensor_inline> A(
      a + b0 * p.strideA0 + b1 * p.strideA1,
      Extents(p.K, p.M),
      array<int32_t, 2>{1, int32_t(p.lda)});
  tensor<device T, Extents, tensor_inline> B(
      b + b0 * p.strideB0 + b1 * p.strideB1,
      kTransposeB ? Extents(p.K, p.N) : Extents(p.N, p.K),
      array<int32_t, 2>{1, int32_t(p.ldb)});
  tensor<device T, Extents, tensor_inline> C(
      c + batch * p.M * p.N,
      Extents(p.N, p.M),
      array<int32_t, 2>{1, int32_t(p.N)});

  constexpr auto descriptor =
      matmul2d_descriptor(kTileM, kTileN, int(dynamic_extent), false, kTransposeB, false);
  matmul2d<descriptor, execution_simdgroups<4>> op;

  int m0 = group.y * kTileM;
  int n0 = group.x * kTileN;
  auto tileA = A.slice(0, m0);
  auto tileB = kTransposeB ? B.slice(0, n0) : B.slice(n0, 0);
  auto tileC = C.slice(n0, m0);
  op.run(tileA, tileB, tileC);
}

#define FLINT_NAX_MATMUL(T, S)                                                                  \
  template [[host_name("nax_matmul_nn_" S)]] kernel void nax_matmul<T, false>(                  \
      device T *, device T *, device T *, constant NaxGemmParams &, uint3);                     \
  template [[host_name("nax_matmul_nt_" S)]] kernel void nax_matmul<T, true>(                   \
      device T *, device T *, device T *, constant NaxGemmParams &, uint3);

FLINT_NAX_MATMUL(half, "f16")
FLINT_NAX_MATMUL(float, "f32")
