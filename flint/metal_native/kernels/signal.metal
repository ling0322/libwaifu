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

// The kernels a vocoder is built from: the second half of a transposed 1-D convolution, the snake
// activation, padding the last dimension and clamping. One thread an output element, and every
// tensor contiguous unless said otherwise. The Vulkan backend's shaders, in Metal's terms.

#include "common.h"

// Keep each of these in step with the struct of the same name in signal.cc.

struct OverlapAdd1dParams {
  uint numel;    // of C
  uint inWidth;  // L
  uint stride;
  uint padding;
  uint taps;  // R; `kernel` is a keyword in Metal
  uint channels;
  uint outWidth;
  uint batchStride;
  uint channelStride;
  uint hasBias;
};

struct SnakeParams {
  uint numel;
  uint width;  // L
  uint channels;
  uint hasBeta;
  float eps;
};

struct Pad1dParams {
  uint numel;  // of C
  uint width;  // of A's rows
  uint outWidth;
  uint left;
  uint mode;
};

struct ClampParams {
  uint numel;
  float lo;
  float hi;
};

// col2im: C (N, K, Lout) from the contiguous columns A (N * L, K * R), gathering the taps that
// land on each output element and adding the bias (K) when there is one. Input position `l` lands
// tap `r` on `l * stride + r - padding`, so the taps reaching `o` are those congruent to
// `o + padding` modulo the stride. C's last dimension is contiguous; its batch and channel strides
// are given, since a group writes its own channels of the result. In float.
template <typename T>
kernel void overlap_add1d(
    device T *c [[buffer(0)]],
    device const T *a [[buffer(1)]],
    device const T *bias [[buffer(2)]],
    constant OverlapAdd1dParams &p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.numel) return;
  uint o = i % p.outWidth;
  uint plane = i / p.outWidth;
  uint k = plane % p.channels;
  uint n = plane / p.channels;

  uint row = p.channels * p.taps;
  uint base = n * p.inWidth * row + k * p.taps;
  int at = int(o + p.padding);
  int stride = int(p.stride);

  float sum = p.hasBias != 0 ? Elem<T>::load(bias, k) : 0.0f;
  for (int r = at % stride; r < int(p.taps); r += stride) {
    int l = (at - r) / stride;
    if (l < 0) break;
    if (l < int(p.inWidth)) sum += Elem<T>::load(a, base + uint(l) * row + uint(r));
  }
  Elem<T>::store(c, n * p.batchStride + k * p.channelStride + o, sum);
}

// x + sin(alpha * x)^2 / (beta + eps) per channel of A (N, C, L), into C. alpha and beta are (C);
// without beta, beta is alpha, the plain snake. Built without fast math, Metal's sin holds its
// precision over the whole range, so the angle needs no reducing first as the Vulkan shader's
// does. In float.
template <typename T>
kernel void snake(
    device T *c [[buffer(0)]],
    device const T *a [[buffer(1)]],
    device const T *alpha [[buffer(2)]],
    device const T *beta [[buffer(3)]],
    constant SnakeParams &p [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.numel) return;
  uint channel = (i / p.width) % p.channels;
  float x = Elem<T>::load(a, i);
  float s = sin(Elem<T>::load(alpha, channel) * x);
  float divisor = Elem<T>::load(p.hasBeta != 0 ? beta : alpha, channel) + p.eps;
  Elem<T>::store(c, i, x + s * s / divisor);
}

// A (rows, width) padded on its last dimension into C (rows, left + width + right): mode 0 with
// zeros, 1 mirrored about the end samples, 2 with the end samples repeated -- PadMode's values.
// Elements are moved, never computed on, so they are carried as they are stored.
template <typename T>
kernel void pad1d(
    device T *c [[buffer(0)]],
    device const T *a [[buffer(1)]],
    constant Pad1dParams &p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.numel) return;
  uint row = i / p.outWidth;
  int last = int(p.width) - 1;
  int at = int(i % p.outWidth) - int(p.left);

  if (at < 0 || at > last) {
    if (p.mode == 0u) {
      c[i] = T(0);
      return;
    }
    at = p.mode == 2u ? clamp(at, 0, last) : (at < 0 ? -at : 2 * last - at);
  }
  c[i] = a[row * p.width + uint(at)];
}

// min(max(x, lo), hi) of A into C. Either bound may be infinite. A NaN stays one, as torch.clamp
// has it, where Metal's min and max would give the other operand.
template <typename T>
kernel void clamp_bounds(
    device T *c [[buffer(0)]],
    device const T *a [[buffer(1)]],
    constant ClampParams &p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.numel) return;
  float x = Elem<T>::load(a, i);
  Elem<T>::store(c, i, isnan(x) ? x : min(max(x, p.lo), p.hi));
}

#define FLINT_SIGNAL_KERNELS(T, S)                                                              \
  template [[host_name("overlap_add1d_" S)]] kernel void overlap_add1d<T>(                      \
      device T *, device const T *, device const T *, constant OverlapAdd1dParams &, uint);     \
  template [[host_name("snake_" S)]] kernel void snake<T>(                                      \
      device T *,                                                                               \
      device const T *,                                                                         \
      device const T *,                                                                         \
      device const T *,                                                                         \
      constant SnakeParams &,                                                                   \
      uint);                                                                                    \
  template [[host_name("pad1d_" S)]] kernel void pad1d<T>(                                      \
      device T *, device const T *, constant Pad1dParams &, uint);                              \
  template [[host_name("clamp_" S)]] kernel void clamp_bounds<T>(                               \
      device T *, device const T *, constant ClampParams &, uint);

FLINT_SIGNAL_KERNELS(float, "f32")
FLINT_SIGNAL_KERNELS(half, "f16")
