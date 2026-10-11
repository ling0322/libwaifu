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


// What every kernel shares: how an element of each storage type is read and written, how a strided
// index is worked out, and the few functions Metal does not have.
//
// Arithmetic is done in float throughout, as the Vulkan backend's is: half and the integer types
// are only ever storage. A kernel is written once as a template over its storage types, and each
// type it is built for is named in the library as `<kernel>_<suffix>` -- `unary_f16` -- which is
// what Context::dispatch() is asked for.

#pragma once

#include <metal_stdlib>

using namespace metal;

/// The most dimensions a strided kernel takes, after collapse() on the host has merged what it can.
constant constexpr uint kMaxDims = 6;

/// A bool tensor's element: one byte, zero or one. A type of its own so that it is told apart from
/// uint8 when a value is written to it.
struct Bool8 {
  uchar v;
};

/// Reading and writing one element of a storage type, as a float and, for the integer types, as a
/// long -- which is what a copy between two integer types goes through, losing nothing a float
/// would.
template <typename T>
struct Elem {
  static float load(device const T *p, uint i) {
    return float(p[i]);
  }
  static void store(device T *p, uint i, float x) {
    p[i] = T(x);
  }
};

#define FLINT_INTEGER_ELEM(T)                          \
  template <>                                          \
  struct Elem<T> {                                     \
    static float load(device const T *p, uint i) {     \
      return float(p[i]);                              \
    }                                                  \
    static void store(device T *p, uint i, float x) {  \
      p[i] = T(long(x));                               \
    }                                                  \
    static long loadLong(device const T *p, uint i) {  \
      return long(p[i]);                               \
    }                                                  \
    static void storeLong(device T *p, uint i, long x) { \
      p[i] = T(x);                                     \
    }                                                  \
  };

FLINT_INTEGER_ELEM(long)
FLINT_INTEGER_ELEM(int)
FLINT_INTEGER_ELEM(uchar)
FLINT_INTEGER_ELEM(char)

template <>
struct Elem<Bool8> {
  static float load(device const Bool8 *p, uint i) {
    return p[i].v != 0 ? 1.0f : 0.0f;
  }
  static void store(device Bool8 *p, uint i, float x) {
    p[i].v = x != 0.0f ? 1 : 0;
  }
};

/// Where element `i` of a tensor is, in elements from its start, given the collapsed shape and the
/// tensor's strides over it: the last dimension varies fastest.
inline uint stridedOffset(uint i, uint ndim, constant uint *shape, constant uint *stride) {
  uint offset = 0;
  uint rest = i;
  for (int d = int(ndim) - 1; d >= 0; --d) {
    uint size = shape[d];
    uint next = rest / size;
    offset += (rest - next * size) * stride[d];
    rest = next;
  }
  return offset;
}

// erf, which Metal does not have: Abramowitz and Stegun 7.1.26, whose error is under 1.5e-7
// everywhere -- about what a float can resolve of a value near one. The Vulkan backend's, so that
// the two agree.
inline float erfApprox(float x) {
  float s = sign(x);
  float a = abs(x);
  float t = 1.0f / (1.0f + 0.3275911f * a);
  float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t -
                     0.284496736f) * t + 0.254829592f) * t * exp(-a * a);
  return s * y;
}

inline float gelu(float x) {
  return 0.5f * x * (1.0f + erfApprox(x * 0.70710678118654752f));
}

inline float silu(float x) {
  return x / (1.0f + exp(-x));
}
