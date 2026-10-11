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


// The elementwise kernels: unary and binary arithmetic, eq, mod, fill, copy between any two types,
// and arange. Each reads its inputs through strides and writes its output, contiguous unless it is
// fill or copy, element by element: one thread per element.

#include "common.h"

// Keep each of these in step with the struct of the same name in elementwise.cc.

struct UnaryParams {
  uint numel;
  uint ndim;
  uint op;
  float scalar;
  uint shape[kMaxDims];
  uint strideA[kMaxDims];
};

struct BinaryParams {
  uint numel;
  uint ndim;
  uint op;
  uint pad;
  uint shape[kMaxDims];
  uint strideA[kMaxDims];
  uint strideB[kMaxDims];
};

struct CopyParams {
  uint numel;
  uint ndim;
  uint shape[kMaxDims];
  uint strideA[kMaxDims];
  uint strideC[kMaxDims];
};

struct FillParams {
  uint numel;
  uint ndim;
  float value;
  uint shape[kMaxDims];
  uint strideC[kMaxDims];
};

struct ModParams {
  long other;
  uint numel;
};

struct ArangeParams {
  long begin;
  long step;
  uint numel;
};

// In step with UnaryOp in ops.h.
constant constexpr uint kNeg = 0;
constant constexpr uint kAbs = 1;
constant constexpr uint kExp = 2;
constant constexpr uint kSqrt = 3;
constant constexpr uint kRsqrt = 4;
constant constexpr uint kSquare = 5;
constant constexpr uint kSigmoid = 6;
constant constexpr uint kTanh = 7;
constant constexpr uint kRelu = 8;
constant constexpr uint kGelu = 9;
constant constexpr uint kSilu = 10;
constant constexpr uint kSin = 11;
constant constexpr uint kCos = 12;
constant constexpr uint kQuickGelu = 13;
constant constexpr uint kMulScalar = 14;
constant constexpr uint kDivScalar = 15;
constant constexpr uint kLog = 17;
constant constexpr uint kRound = 18;

// In step with BinaryOp in ops.h.
constant constexpr uint kAdd = 0;
constant constexpr uint kSub = 1;
constant constexpr uint kMul = 2;

inline float naturalLog(float x) {
  if (x > 0.0f && !isinf(x)) return log(x);
  if (x == 0.0f) return -INFINITY;
  if (isinf(x) && x > 0.0f) return x;
  return NAN;  // for negatives and NaN alike
}

template <typename T>
kernel void unary(
    device T *c [[buffer(0)]],
    device const T *a [[buffer(1)]],
    constant UnaryParams &p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.numel) return;
  float x = Elem<T>::load(a, stridedOffset(i, p.ndim, p.shape, p.strideA));
  float y;
  switch (p.op) {
    case kNeg: y = -x; break;
    case kAbs: y = abs(x); break;
    case kExp: y = exp(x); break;
    case kLog: y = naturalLog(x); break;
    // rint, the tie to the even neighbour, as torch.round takes it.
    case kRound: y = rint(x); break;
    case kSqrt: y = sqrt(x); break;
    case kRsqrt: y = rsqrt(x); break;
    case kSquare: y = x * x; break;
    case kSigmoid: y = 1.0f / (1.0f + exp(-x)); break;
    // Clamped first, as on Vulkan: nothing a float can tell apart from +-1 is lost.
    case kTanh: y = tanh(clamp(x, -15.0f, 15.0f)); break;
    case kRelu: y = x > 0.0f ? x : 0.0f; break;
    case kGelu: y = gelu(x); break;
    case kSilu: y = silu(x); break;
    case kSin: y = sin(x); break;
    case kCos: y = cos(x); break;
    case kQuickGelu: y = x / (1.0f + exp(-1.702f * x)); break;
    case kMulScalar: y = x * p.scalar; break;
    case kDivScalar: y = x / p.scalar; break;
    default: y = x; break;
  }
  Elem<T>::store(c, i, y);
}

template <typename T>
kernel void binary(
    device T *c [[buffer(0)]],
    device const T *a [[buffer(1)]],
    device const T *b [[buffer(2)]],
    constant BinaryParams &p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.numel) return;
  float x = Elem<T>::load(a, stridedOffset(i, p.ndim, p.shape, p.strideA));
  float y = Elem<T>::load(b, stridedOffset(i, p.ndim, p.shape, p.strideB));
  float z;
  switch (p.op) {
    case kAdd: z = x + y; break;
    case kSub: z = x - y; break;
    case kMul: z = x * y; break;
    default: z = x / y; break;
  }
  Elem<T>::store(c, i, z);
}

// int64 arithmetic in int64, not through a float that would lose what is past its 24 bits.
kernel void binary_i64(
    device long *c [[buffer(0)]],
    device const long *a [[buffer(1)]],
    device const long *b [[buffer(2)]],
    constant BinaryParams &p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.numel) return;
  long x = a[stridedOffset(i, p.ndim, p.shape, p.strideA)];
  long y = b[stridedOffset(i, p.ndim, p.shape, p.strideB)];
  long z;
  switch (p.op) {
    case kAdd: z = x + y; break;
    case kSub: z = x - y; break;
    case kMul: z = x * y; break;
    default: z = x / y; break;
  }
  c[i] = z;
}

// eq of two floats compares their values as floats; of two integers, as longs.
template <typename T>
kernel void equal(
    device Bool8 *c [[buffer(0)]],
    device const T *a [[buffer(1)]],
    device const T *b [[buffer(2)]],
    constant BinaryParams &p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.numel) return;
  float x = Elem<T>::load(a, stridedOffset(i, p.ndim, p.shape, p.strideA));
  float y = Elem<T>::load(b, stridedOffset(i, p.ndim, p.shape, p.strideB));
  c[i].v = x == y ? 1 : 0;
}

template <typename T>
kernel void equal_integer(
    device Bool8 *c [[buffer(0)]],
    device const T *a [[buffer(1)]],
    device const T *b [[buffer(2)]],
    constant BinaryParams &p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.numel) return;
  long x = Elem<T>::loadLong(a, stridedOffset(i, p.ndim, p.shape, p.strideA));
  long y = Elem<T>::loadLong(b, stridedOffset(i, p.ndim, p.shape, p.strideB));
  c[i].v = x == y ? 1 : 0;
}

kernel void mod_i64(
    device long *c [[buffer(0)]],
    device const long *a [[buffer(1)]],
    constant ModParams &p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.numel) return;
  // C's remainder, which takes the sign of the dividend, as the Vulkan and CPU operators do.
  long x = a[i];
  c[i] = x - (x / p.other) * p.other;
}

template <typename T>
kernel void fill(
    device T *c [[buffer(0)]],
    constant FillParams &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.numel) return;
  Elem<T>::store(c, stridedOffset(i, p.ndim, p.shape, p.strideC), p.value);
}

// A copy goes through a float, unless both ends are integers, when it goes through a long, or
// both bool, when the byte is copied as it is.
template <typename A, typename C>
struct Convert {
  static void apply(device const A *a, uint ia, device C *c, uint ic) {
    Elem<C>::store(c, ic, Elem<A>::load(a, ia));
  }
};

#define FLINT_INTEGER_CONVERT(A, C)                                     \
  template <>                                                           \
  struct Convert<A, C> {                                                \
    static void apply(device const A *a, uint ia, device C *c, uint ic) { \
      Elem<C>::storeLong(c, ic, Elem<A>::loadLong(a, ia));              \
    }                                                                   \
  };

FLINT_INTEGER_CONVERT(long, long)
FLINT_INTEGER_CONVERT(int, int)
FLINT_INTEGER_CONVERT(uchar, uchar)
FLINT_INTEGER_CONVERT(char, char)
FLINT_INTEGER_CONVERT(int, long)
FLINT_INTEGER_CONVERT(long, int)

template <>
struct Convert<Bool8, Bool8> {
  static void apply(device const Bool8 *a, uint ia, device Bool8 *c, uint ic) {
    c[ic] = a[ia];
  }
};

template <typename A, typename C>
kernel void copy(
    device C *c [[buffer(0)]],
    device const A *a [[buffer(1)]],
    constant CopyParams &p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.numel) return;
  Convert<A, C>::apply(
      a,
      stridedOffset(i, p.ndim, p.shape, p.strideA),
      c,
      stridedOffset(i, p.ndim, p.shape, p.strideC));
}

// A contiguous run of bytes, a word at a time where both ends allow it.
kernel void copy_words(
    device uint *c [[buffer(0)]],
    device const uint *a [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i < count) c[i] = a[i];
}

kernel void copy_bytes(
    device uchar *c [[buffer(0)]],
    device const uchar *a [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i < count) c[i] = a[i];
}

kernel void arange_i64(
    device long *c [[buffer(0)]],
    constant ArangeParams &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  if (i < p.numel) c[i] = p.begin + long(i) * p.step;
}

// --- the types each is built for ----------------------------------------------------------------

#define FLINT_FLOAT_KERNELS(T, S)                                                               \
  template [[host_name("unary_" S)]] kernel void unary<T>(                                      \
      device T *, device const T *, constant UnaryParams &, uint);                              \
  template [[host_name("binary_" S)]] kernel void binary<T>(                                    \
      device T *, device const T *, device const T *, constant BinaryParams &, uint);           \
  template [[host_name("equal_" S)]] kernel void equal<T>(                                      \
      device Bool8 *, device const T *, device const T *, constant BinaryParams &, uint);

FLINT_FLOAT_KERNELS(float, "f32")
FLINT_FLOAT_KERNELS(half, "f16")

template [[host_name("equal_i64")]] kernel void equal_integer<long>(
    device Bool8 *, device const long *, device const long *, constant BinaryParams &, uint);
template [[host_name("equal_u8")]] kernel void equal_integer<uchar>(
    device Bool8 *, device const uchar *, device const uchar *, constant BinaryParams &, uint);

#define FLINT_FILL(T, S)                                    \
  template [[host_name("fill_" S)]] kernel void fill<T>(    \
      device T *, constant FillParams &, uint);

FLINT_FILL(float, "f32")
FLINT_FILL(half, "f16")
FLINT_FILL(long, "i64")
FLINT_FILL(int, "i32")
FLINT_FILL(uchar, "u8")
FLINT_FILL(char, "i8")
FLINT_FILL(Bool8, "bool")

#define FLINT_COPY(A, C, S)                                              \
  template [[host_name("copy_" S)]] kernel void copy<A, C>(              \
      device C *, device const A *, constant CopyParams &, uint);

// The pairs the Vulkan backend builds, and no others.
FLINT_COPY(float, float, "f32_f32")
FLINT_COPY(half, half, "f16_f16")
FLINT_COPY(float, half, "f32_f16")
FLINT_COPY(half, float, "f16_f32")
FLINT_COPY(long, long, "i64_i64")
FLINT_COPY(int, int, "i32_i32")
FLINT_COPY(uchar, uchar, "u8_u8")
FLINT_COPY(char, char, "i8_i8")
FLINT_COPY(Bool8, Bool8, "bool_bool")
FLINT_COPY(long, float, "i64_f32")
FLINT_COPY(long, half, "i64_f16")
FLINT_COPY(int, float, "i32_f32")
FLINT_COPY(int, half, "i32_f16")
FLINT_COPY(int, long, "i32_i64")
FLINT_COPY(long, int, "i64_i32")
FLINT_COPY(Bool8, float, "bool_f32")
FLINT_COPY(Bool8, half, "bool_f16")
FLINT_COPY(float, Bool8, "f32_bool")
FLINT_COPY(half, Bool8, "f16_bool")
FLINT_COPY(uchar, float, "u8_f32")
FLINT_COPY(uchar, half, "u8_f16")
FLINT_COPY(char, float, "i8_f32")
FLINT_COPY(char, half, "i8_f16")
FLINT_COPY(float, long, "f32_i64")
FLINT_COPY(half, long, "f16_i64")
