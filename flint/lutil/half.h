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
#include <string.h>

// F16C where the compiler has it: the build gives x64 -mf16c (see CMakeLists.txt), and cl has the
// intrinsics regardless. A CUDA device pass is left the portable code, which it never calls.
#if !defined(__CUDA_ARCH__) && (defined(__F16C__) || (defined(_MSC_VER) && defined(_M_X64)))
#include <immintrin.h>
#define LUT_HALF_F16C 1
#else
#define LUT_HALF_F16C 0
#endif

namespace lut {

// Inline, since x64's fl::Float16 converts through these on every load and store of a half.
// The portable conversions are from https://gist.github.com/rygorous/2156668
// License: https://creativecommons.org/publicdomain/zero/1.0/

namespace internal {

inline uint32_t floatBits(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof(u));
  return u;
}

inline float bitsFloat(uint32_t u) {
  float f;
  memcpy(&f, &u, sizeof(f));
  return f;
}

// Rounds to nearest, but a tie goes away from zero rather than to the even half; F16C, which x64
// builds use instead, has IEEE's rule.
inline uint16_t cvtss_sh_soft(float v) {
  constexpr uint32_t f32infty = 255u << 23;
  constexpr uint32_t f16infty = 31u << 23;
  constexpr uint32_t magic = 15u << 23;
  constexpr uint32_t sign_mask = 0x80000000u;
  constexpr uint32_t round_mask = ~0xfffu;

  uint32_t u = internal::floatBits(v);
  uint32_t sign = u & sign_mask;
  u ^= sign;

  uint16_t o;
  if (u >= f32infty) {                        // Inf or NaN (all exponent bits set)
    o = (u > f32infty) ? 0x7e00 : 0x7c00;     // NaN->qNaN and Inf->Inf
  } else {                                    // (De)normalized number or zero
    u &= round_mask;
    u = internal::floatBits(internal::bitsFloat(u) * internal::bitsFloat(magic));
    u -= round_mask;
    if (u > f16infty) u = f16infty;  // Clamp to signed infinity if overflowed

    o = static_cast<uint16_t>(u >> 13);  // Take the bits!
  }

  return static_cast<uint16_t>(o | (sign >> 16));
}

inline float cvtsh_ss_soft(uint16_t v) {
  constexpr uint32_t magic = 113u << 23;
  constexpr uint32_t shifted_exp = 0x7c00u << 13;  // exponent mask after shift

  uint32_t o = (v & 0x7fffu) << 13;  // exponent/mantissa bits
  uint32_t exp = shifted_exp & o;    // just the exponent
  o += (127 - 15) << 23;             // exponent adjust

  // handle exponent special cases
  if (exp == shifted_exp) {  // Inf/NaN?
    o += (128 - 16) << 23;   // extra exp adjust
  } else if (exp == 0) {     // Zero/Denormal?
    o += 1 << 23;            // extra exp adjust
    o = internal::floatBits(internal::bitsFloat(o) - internal::bitsFloat(magic));  // renormalize
  }

  o |= (v & 0x8000u) << 16;  // sign bit
  return internal::bitsFloat(o);
}

}  // namespace internal

/// @brief Convert from float to float16: to nearest, ties to even where F16C does it.
/// @param v value in float32.
/// @return value in float16.
inline uint16_t cvtss_sh(float v) {
#if LUT_HALF_F16C
  __m128i h = _mm_cvtps_ph(_mm_set_ss(v), _MM_FROUND_TO_NEAREST_INT);
  return static_cast<uint16_t>(_mm_cvtsi128_si32(h));
#else
  return internal::cvtss_sh_soft(v);
#endif
}

/// @brief Convert from float16 to float32.
/// @param v value in float16.
/// @return value in float32.
inline float cvtsh_ss(uint16_t v) {
#if LUT_HALF_F16C
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(v)));
#else
  return internal::cvtsh_ss_soft(v);
#endif
}

}  // namespace lut
