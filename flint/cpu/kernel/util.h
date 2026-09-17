// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
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

#pragma once

#include <stdint.h>

#include <cstring>
#include <limits>
#include <memory>

#include "lutil/c_ptr.h"
#include "lutil/platform.h"
#include "lutil/span.h"
#include "flint/cpu/kernel/abstract.h"

namespace fl {
namespace op {
namespace cpu {
namespace kernel {

// copy vector x to y.
template<typename T>
void copyVec(int n, const T *x, int incx, T *y, int incy) {
  for (int i = 0; i < n; ++i) {
    y[i * incy] = x[i * incx];
  }
}
// allocate n single float and returns the holder. the memory is 32 byte aligned.
template<typename T>
lut::c_ptr<T> alignedAlloc(int64_t n) {
  return lut::c_ptr<T>(
      reinterpret_cast<T *>(lut::alloc32ByteAlignedMem(sizeof(T) * n)),
      lut::free32ByteAlignedMem);
}

float cvt_h2s(Float16 vh);
Float16 cvt_s2h(float vf);

/// E4M3 -- four exponent bits, three mantissa bits, a bias of seven -- to float32. Exact: every
/// E4M3 value is a float32 value. 0x7f and 0xff are the format's NaNs; it has no infinities.
///
/// Inline rather than beside cvt_h2s in util.cc, because this one is called once per element from
/// the packing loop in block.h and there is no vectorized pack kernel for E4M3 to fall into.
inline float cvt_f8_s(Fp8E4M3 v) {
  uint32_t exp = (v.v >> 3) & 0xfu;
  uint32_t man = v.v & 0x7u;

  float mag;
  if (exp == 0) {
    // Subnormal: man * 2^-9, which float32 holds exactly, zero included.
    mag = static_cast<float>(man) * 0x1p-9f;
  } else if (exp == 0xf && man == 0x7) {
    mag = std::numeric_limits<float>::quiet_NaN();
  } else {
    // The exponent field moves from a bias of 7 to float32's 127, and the three mantissa bits to
    // the top of its twenty-three.
    uint32_t bits = ((exp + 120u) << 23) | (man << 20);
    std::memcpy(&mag, &bits, sizeof(mag));
  }

  return (v.v & 0x80u) ? -mag : mag;
}

/// float32 to E4M3, round to nearest even, saturating to +-448 rather than reaching the NaN code.
///
/// Checked against CUDA's __nv_cvt_float_to_fp8(__NV_SATFINITE, __NV_E4M3) over all 2^32 float bit
/// patterns, and against __nv_cvt_fp8_to_halfraw over all 256 codes, so a weight quantized here
/// and one quantized on the card are the same bytes.
inline Fp8E4M3 cvt_s_f8(float vf) {
  uint32_t bits;
  std::memcpy(&bits, &vf, sizeof(bits));

  uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
  uint32_t mag = bits & 0x7fffffffu;

  if (mag > 0x7f800000u) return Fp8E4M3{static_cast<uint8_t>(sign | 0x7fu)};  // NaN
  if ((mag >> 23) == 0) return Fp8E4M3{sign};  // zero, or a float subnormal: far below this range

  int exp = static_cast<int>(mag >> 23) - 127;
  uint32_t sig = (1u << 23) | (mag & 0x7fffffu);

  // Three mantissa bits at an exponent of at least -6. Below that E4M3 goes subnormal and keeps
  // fewer of them, which is the same thing as dropping more bits here.
  int shift = 20;
  if (exp < -6) {
    shift += -6 - exp;
    exp = -6;
  }
  if (shift > 31) return Fp8E4M3{sign};  // under half of the smallest subnormal

  uint32_t roundBit = 1u << (shift - 1);
  uint32_t rem = sig & ((1u << shift) - 1);
  uint32_t q = sig >> shift;
  if (rem > roundBit || (rem == roundBit && (q & 1u))) ++q;

  // q carries its own overflow, and both carries are the same addition: at 16 it rolls into the
  // exponent field, and at 8 the largest subnormal becomes the smallest normal.
  uint32_t code = static_cast<uint32_t>((exp + 6) << 3) + q;
  if (code > 0x7eu) code = 0x7eu;

  return Fp8E4M3{static_cast<uint8_t>(sign | code)};
}

template<typename T>
T cvtf(float v);
template<>
inline float cvtf(float v) {
  return v;
}
template<>
inline Float16 cvtf(float v) {
  return cvt_s2h(v);
}

template<typename T>
T cvtf(Fp8E4M3 v);
template<>
inline float cvtf(Fp8E4M3 v) {
  return cvt_f8_s(v);
}

template<typename T>
T cvtf(Float16 v);
template<>
inline float cvtf(Float16 v) {
  return cvt_h2s(v);
}
template<>
inline Float16 cvtf(Float16 v) {
  return v;
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace fl
