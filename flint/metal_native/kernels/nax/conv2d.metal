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

// 2-D convolution on the neural accelerators, as kernels/matmul.metal's conv2d has it: an implicit
// GEMM, for one group of one image at a time -- the grid's z, image * groups + group --
//
//   C (K, P * Q) = weight (K, C * R * S) unfold(input) (C * R * S, P * Q)
//
// with the input (N, C, H, W) and weight (K, C / groups, R, S) contiguous and the output
// (N, K, P, Q) contiguous, which is C row by row. The weight is read where it is. The unfolded
// input is gathered kTileK rows at a time into threadgroup memory -- or, for a 1 x 1 kernel of
// stride one and no padding, which leaves the input as it is, read where it is too -- and matmul2d
// multiplies the two into a sum the threadgroup holds in float. The bias is added to that sum as
// it is written out.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

/// In step with ConvParams in matmul.cc, and kernels/matmul.metal's.
struct ConvParams {
  uint channels;     // C, all groups
  uint height;
  uint width;
  uint outChannels;  // K, all groups
  uint outHeight;
  uint outWidth;
  uint kernelHeight;
  uint kernelWidth;
  uint strideH;
  uint strideW;
  int paddingH;
  int paddingW;
  uint dilationH;
  uint dilationW;
  uint groups;
  uint hasBias;
  uint pointwise;
  uint vecWeight;
  uint vecInput;
};

// In step with kNaxConvTileM and kNaxConvTileN in matmul.cc.
constant constexpr int kTileM = 64;   // output channels
constant constexpr int kTileN = 128;  // output pixels
constant constexpr int kTileK = 32;
constant constexpr uint kThreads = 128;
static_assert(kThreads == uint(kTileN), "a thread to each pixel of the tile");

using Extents = dextents<int32_t, 2>;

// One pixel's column of the unfolded input, read kTileK rows at a time from the top down: what a
// thread gathers. Adjacent threads have adjacent pixels, so they read adjacent input pixels when
// the stride is one. The column carries where it is through the kernel's columns, rows and
// channels from one chunk to the next rather than dividing for each element.
template <typename T>
struct Column {
  device const T *input;
  constant ConvParams *p;
  uint K;
  bool valid;  // a pixel of the output, rather than past its end
  int y0;
  int x0;
  uint k = 0;
  uint channel = 0;
  uint r = 0;
  uint s = 0;

  Column(device const T *input, constant ConvParams *p, uint K, uint N, uint pixel)
      : input(input), p(p), K(K), valid(pixel < N) {
    uint row = pixel / p->outWidth;
    uint column = pixel - row * p->outWidth;
    y0 = int(row * p->strideH) - p->paddingH;
    x0 = int(column * p->strideW) - p->paddingW;
  }

  // The next kTileK rows, zero past the bottom or the edge of the image.
  void next(thread T *values) {
    for (uint kk = 0; kk < uint(kTileK); ++kk) {
      T value = T(0);
      if (valid && k < K) {
        int y = y0 + int(r * p->dilationH);
        int x = x0 + int(s * p->dilationW);
        if (y >= 0 && y < int(p->height) && x >= 0 && x < int(p->width)) {
          value = input[(channel * p->height + uint(y)) * p->width + uint(x)];
        }
      }
      values[kk] = value;

      ++k;
      if (++s == p->kernelWidth) {
        s = 0;
        if (++r == p->kernelHeight) {
          r = 0;
          ++channel;
        }
      }
    }
  }
};

// The sum, plus the bias, written to the rows m0 : m0 + kTileM and columns n0 : n0 + kTileN of C
// that are in it.
template <typename T, typename Sum>
inline void writeOut(
    thread Sum &sum,
    device T *c,
    device const T *bias,
    bool hasBias,
    uint M,
    uint N,
    uint m0,
    uint n0) {
  for (uint16_t i = 0; i < sum.get_capacity(); ++i) {
    if (!sum.is_valid_element(i)) continue;
    auto at = sum.get_multidimensional_index(i);  // (n, m) within the tile
    uint n = n0 + uint(at[0]);
    uint m = m0 + uint(at[1]);
    if (m >= M || n >= N) continue;
    float value = sum[i];
    if (hasBias) value += float(bias[m]);
    c[m * N + n] = T(value);
  }
}

template <typename T, bool kPointwise>
kernel void nax_conv2d(
    device T *c [[buffer(0)]],
    device T *input [[buffer(1)]],
    device T *weight [[buffer(2)]],
    device T *bias [[buffer(3)]],
    constant ConvParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup T tileB[kTileK * kTileN];

  uint image = group.z / p.groups;
  uint g = group.z - image * p.groups;
  uint channelsPerGroup = p.channels / p.groups;
  uint M = p.outChannels / p.groups;
  uint N = p.outHeight * p.outWidth;
  uint K = channelsPerGroup * p.kernelHeight * p.kernelWidth;

  device T *out = c + (image * p.outChannels + g * M) * N;
  device T *x = input + (image * p.channels + g * channelsPerGroup) * p.height * p.width;
  device T *w = weight + g * M * K;
  device T *b = bias + (p.hasBias ? g * M : 0);

  uint m0 = group.y * kTileM;
  uint n0 = group.x * kTileN;

  // Extents run from the innermost dimension out: the weight is (K, M) to a tensor.
  tensor<device T, Extents, tensor_inline> A(w, Extents(K, M), array<int32_t, 2>{1, int32_t(K)});

  constexpr auto descriptor = matmul2d_descriptor(
      kTileM,
      kTileN,
      kTileK,
      false,
      false,
      false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<4>> op;

  if (kPointwise) {
    // The input is the unfolded input already: (N, K) to a tensor, a channel to a row.
    tensor<device T, Extents, tensor_inline> B(x, Extents(N, K), array<int32_t, 2>{1, int32_t(N)});
    auto first = A.slice(0, m0);
    auto firstB = B.slice(n0, 0);
    auto sum = op.template get_destination_cooperative_tensor<
        decltype(first), decltype(firstB), float>();
    for (uint16_t i = 0; i < sum.get_capacity(); ++i) {
      if (sum.is_valid_element(i)) sum[i] = 0.0f;
    }
    for (uint k0 = 0; k0 < K; k0 += kTileK) {
      auto tileA = A.slice(k0, m0);
      auto tileB = B.slice(n0, k0);
      op.run(tileA, tileB, sum);
    }
    writeOut(sum, out, b, p.hasBias != 0, M, N, m0, n0);
    return;
  }

  tensor<threadgroup T, Extents, tensor_inline> B(
      tileB,
      Extents(kTileN, kTileK),
      array<int32_t, 2>{1, kTileN});
  auto first = A.slice(0, m0);
  auto sum = op.template get_destination_cooperative_tensor<decltype(first), decltype(B), float>();
  for (uint16_t i = 0; i < sum.get_capacity(); ++i) {
    if (sum.is_valid_element(i)) sum[i] = 0.0f;
  }

  // The next chunk is read into registers before the product of this one starts, so that the
  // reads are on their way while the accelerators work, and are written into the tile only after.
  // (Two tiles, a chunk to each in turn, would save a barrier a chunk; on an M5 Pro the threadgroup
  // memory that costs made it slower.)
  Column<T> column(x, &p, K, N, n0 + tid);
  T values[kTileK];
  column.next(values);
  for (uint k0 = 0; k0 < K; k0 += kTileK) {
    for (uint kk = 0; kk < uint(kTileK); ++kk) tileB[kk * kTileN + tid] = values[kk];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (k0 + kTileK < K) column.next(values);
    auto tileA = A.slice(k0, m0);
    op.run(tileA, B, sum);
    // The tile is written again for the next rows; every simdgroup has to be done reading it.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  writeOut(sum, out, b, p.hasBias != 0, M, N, m0, n0);
}

#define FLINT_NAX_CONV2D(T, S)                                                                  \
  template [[host_name("nax_conv2d_" S)]] kernel void nax_conv2d<T, false>(                     \
      device T *, device T *, device T *, device T *, constant ConvParams &, uint3, uint);       \
  template [[host_name("nax_conv2d_pointwise_" S)]] kernel void nax_conv2d<T, true>(            \
      device T *, device T *, device T *, device T *, constant ConvParams &, uint3, uint);

FLINT_NAX_CONV2D(half, "f16")
FLINT_NAX_CONV2D(float, "f32")
