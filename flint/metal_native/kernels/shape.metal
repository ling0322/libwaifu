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

// The kernels that move elements about: lookup, nearest upsampling, the gated linear units and the
// rotary embedding. One thread an output element (a pair of them for the rotary embedding), and
// every tensor contiguous unless said otherwise.

#include "common.h"

// Keep each of these in step with the struct of the same name in shape.cc.

struct LookupParams {
  uint numel;  // of C
  uint width;  // D
};

struct UpsampleParams {
  uint numel;  // of C
  uint height;
  uint width;
  uint scale;
};

struct Upsample1dParams {
  uint numel;  // of C
  uint width;  // of A's rows
  uint size;
  float scale;
};

struct GluParams {
  uint numel;      // of C
  uint halfWidth;  // D
  uint op;
};

struct RotaryParams {
  uint numPairs;  // T * heads * headDim / 2
  uint heads;
  uint headDim;
  uint rowStride;
  uint cacheStride;
};

// C[r, :] = table[indices[r], :], the table (rows, D) and the indices longs.
template <typename T>
kernel void lookup(
    device T *c [[buffer(0)]],
    device const T *table [[buffer(1)]],
    device const long *indices [[buffer(2)]],
    constant LookupParams &p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.numel) return;
  uint row = i / p.width;
  uint column = i - row * p.width;
  c[i] = table[uint(indices[row]) * p.width + column];
}

// A (N, C, H, W) by an integer scale into C (N, C, H * scale, W * scale).
template <typename T>
kernel void upsample(
    device T *c [[buffer(0)]],
    device const T *a [[buffer(1)]],
    constant UpsampleParams &p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.numel) return;
  uint outWidth = p.width * p.scale;
  uint outHeight = p.height * p.scale;
  uint x = i % outWidth;
  uint y = (i / outWidth) % outHeight;
  uint plane = i / (outWidth * outHeight);
  c[i] = a[(plane * p.height + y / p.scale) * p.width + x / p.scale];
}

// The last dimension of A (rows, width) resized to C (rows, size), each output element copying
// input min(floor(j * scale), width - 1) with scale = float(width) / size -- torch's
// `F.interpolate(size=size, mode="nearest")`. Built without fast math, the product is one rounded
// float multiply, so the index is torch's even where it lands a hair under a whole number.
template <typename T>
kernel void upsample1d(
    device T *c [[buffer(0)]],
    device const T *a [[buffer(1)]],
    constant Upsample1dParams &p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.numel) return;
  uint j = i % p.size;
  uint row = i / p.size;
  float position = float(j) * p.scale;
  uint source = min(uint(floor(position)), p.width - 1u);
  c[i] = a[row * p.width + source];
}

// A (..., 2D) into C (..., D), C = A[..., D:] * act(A[..., :D]) -- the first half is the gate.
// GEGLU when op is 1, SwiGLU when it is 0. In float.
template <typename T>
kernel void glu(
    device T *c [[buffer(0)]],
    device const T *a [[buffer(1)]],
    constant GluParams &p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.numel) return;
  uint row = i / p.halfWidth;
  uint column = i - row * p.halfWidth;
  uint base = row * 2 * p.halfWidth;

  float gate = Elem<T>::load(a, base + column);
  float value = Elem<T>::load(a, base + p.halfWidth + column);
  float activated = p.op == 1 ? gelu(gate) : silu(gate);
  Elem<T>::store(c, i, value * activated);
}

// Rotary embedding of A (T, heads, headDim) in place, element d paired with d + headDim / 2. A's
// rows may be spaced by any stride -- a slice of a fused projection -- but each row's heads are
// contiguous. The cache is (positions, 2 * headDim): cosines, then sines, a full head of each.
// In float.
template <typename T>
kernel void rotary(
    device T *a [[buffer(0)]],
    device const T *cache [[buffer(1)]],
    device const long *positions [[buffer(2)]],
    constant RotaryParams &p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.numPairs) return;
  uint halfWidth = p.headDim / 2;
  uint d = i % halfWidth;
  uint head = (i / halfWidth) % p.heads;
  uint t = i / (halfWidth * p.heads);

  uint cacheRow = uint(positions[t]) * p.cacheStride;
  uint base = t * p.rowStride + head * p.headDim;

  float x1 = Elem<T>::load(a, base + d);
  float x2 = Elem<T>::load(a, base + d + halfWidth);
  float cos1 = Elem<T>::load(cache, cacheRow + d);
  float cos2 = Elem<T>::load(cache, cacheRow + d + halfWidth);
  float sin1 = Elem<T>::load(cache, cacheRow + p.headDim + d);
  float sin2 = Elem<T>::load(cache, cacheRow + p.headDim + d + halfWidth);

  Elem<T>::store(a, base + d, x1 * cos1 - x2 * sin1);
  Elem<T>::store(a, base + d + halfWidth, x2 * cos2 + x1 * sin2);
}

#define FLINT_SHAPE_KERNELS(T, S)                                                               \
  template [[host_name("lookup_" S)]] kernel void lookup<T>(                                    \
      device T *, device const T *, device const long *, constant LookupParams &, uint);        \
  template [[host_name("upsample_" S)]] kernel void upsample<T>(                                \
      device T *, device const T *, constant UpsampleParams &, uint);                           \
  template [[host_name("upsample1d_" S)]] kernel void upsample1d<T>(                            \
      device T *, device const T *, constant Upsample1dParams &, uint);                         \
  template [[host_name("glu_" S)]] kernel void glu<T>(                                          \
      device T *, device const T *, constant GluParams &, uint);                                \
  template [[host_name("rotary_" S)]] kernel void rotary<T>(                                    \
      device T *, device const T *, device const long *, constant RotaryParams &, uint);

FLINT_SHAPE_KERNELS(float, "f32")
FLINT_SHAPE_KERNELS(half, "f16")
