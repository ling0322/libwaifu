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
//
// The _add kernels add an addend to that sum before it is rounded to C, so that the two are rounded
// once: a value per column -- a linear layer's bias -- or one laid out as C is, a residual. The sum
// is held in registers for it, a cooperative tensor, and written out an element at a time; the
// kernels with nothing to add have matmul2d write C itself, as they did.

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
  uint addKind;
};

// In step with kAddColumns in kernels/matmul.metal and matmul.cc; anything else is kAddWhole.
constant constexpr uint kAddColumns = 1;

// Chosen on an M5 Pro over SDXL's products, among 32, 64 and 128 on each side. In step with
// kNaxTileM and kNaxTileN in matmul.cc.
constant constexpr int kTileM = 64;
constant constexpr int kTileN = 128;

using Extents = dextents<int32_t, 2>;

// C in `C`: T's own, or float for a product of half kept as the sum was.
template <typename T, typename C, bool kTransposeB>
kernel void nax_matmul(
    device C *c [[buffer(0)]],
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
  tensor<device C, Extents, tensor_inline> product(
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
  auto tileC = product.slice(n0, m0);
  op.run(tileA, tileB, tileC);
}

// C = A B + the addend, in C's type: the sum, and the addend added to it, in float, rounded once.
template <typename T, typename C, bool kTransposeB>
kernel void nax_matmul_add(
    device C *c [[buffer(0)]],
    device T *a [[buffer(1)]],
    device T *b [[buffer(2)]],
    device C *add [[buffer(3)]],
    constant NaxGemmParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]]) {
  uint batch = group.z;
  uint b0 = batch / p.batch1;
  uint b1 = batch - b0 * p.batch1;

  tensor<device T, Extents, tensor_inline> A(
      a + b0 * p.strideA0 + b1 * p.strideA1,
      Extents(p.K, p.M),
      array<int32_t, 2>{1, int32_t(p.lda)});
  tensor<device T, Extents, tensor_inline> B(
      b + b0 * p.strideB0 + b1 * p.strideB1,
      kTransposeB ? Extents(p.K, p.N) : Extents(p.N, p.K),
      array<int32_t, 2>{1, int32_t(p.ldb)});

  constexpr auto descriptor =
      matmul2d_descriptor(kTileM, kTileN, int(dynamic_extent), false, kTransposeB, false);
  matmul2d<descriptor, execution_simdgroups<4>> op;

  int m0 = group.y * kTileM;
  int n0 = group.x * kTileN;
  auto tileA = A.slice(0, m0);
  auto tileB = kTransposeB ? B.slice(0, n0) : B.slice(n0, 0);
  using TileA = decltype(tileA);
  using TileB = decltype(tileB);
  auto sum = op.template get_destination_cooperative_tensor<TileA, TileB, float>();
  op.run(tileA, tileB, sum);

  // The addend, (N, M) to a tensor as C is: a bias is the same row of N at every m, a row stride
  // of zero. It is read into registers laid out as the sum is, which is what lets the two be added
  // element by element; the answer is rounded to C in a third such tensor, which is written out
  // whole, the edges of C left alone.
  bool columns = p.addKind == kAddColumns;
  tensor<device C, Extents, tensor_inline> addend(
      add + (columns ? 0 : batch * p.M * p.N),
      Extents(p.N, p.M),
      array<int32_t, 2>{1, columns ? 0 : int32_t(p.N)});
  tensor<device C, Extents, tensor_inline> product(
      c + batch * p.M * p.N,
      Extents(p.N, p.M),
      array<int32_t, 2>{1, int32_t(p.N)});
  auto more = op.template get_destination_cooperative_tensor<TileA, TileB, C>();
  more.load(addend.slice(n0, m0));
  auto out = op.template get_destination_cooperative_tensor<TileA, TileB, C>();
#pragma clang loop unroll(full)
  for (uint16_t i = 0; i < sum.get_capacity(); ++i) {
    if (sum.is_valid_element(i)) out[i] = C(sum[i] + float(more[i]));
  }
  out.store(product.slice(n0, m0));
}

#define FLINT_NAX_MATMUL(T, C, S)                                                               \
  template [[host_name("nax_matmul_nn_" S)]] kernel void nax_matmul<T, C, false>(               \
      device C *, device T *, device T *, constant NaxGemmParams &, uint3);                     \
  template [[host_name("nax_matmul_nt_" S)]] kernel void nax_matmul<T, C, true>(                \
      device C *, device T *, device T *, constant NaxGemmParams &, uint3);                     \
  template [[host_name("nax_matmul_add_nn_" S)]] kernel void nax_matmul_add<T, C, false>(       \
      device C *, device T *, device T *, device C *, constant NaxGemmParams &, uint3);   \
  template [[host_name("nax_matmul_add_nt_" S)]] kernel void nax_matmul_add<T, C, true>(        \
      device C *, device T *, device T *, device C *, constant NaxGemmParams &, uint3);

FLINT_NAX_MATMUL(half, half, "f16")
FLINT_NAX_MATMUL(float, float, "f32")
FLINT_NAX_MATMUL(half, float, "f16_f32")
