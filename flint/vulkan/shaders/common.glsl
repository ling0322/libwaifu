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

// Shared by every kernel. A kernel reaches its tensors through buffer device addresses handed to
// it as push constants, so there are no descriptor sets anywhere; this file declares the buffer
// types those addresses are read through.
//
// A kernel's element types are fixed when it is compiled. Slot A, B and C each get a type from
// the build (A_T, A_ALIGN, and A_BOOL or A_INTEGER where they apply -- see shaders.cmake), and
// the macros below read a slot's element as a float, or write one from a float, whatever that
// type is. Arithmetic is done in float throughout: half and 8-bit values are only ever storage.

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

#define MAX_DIMS 6

// The index of this thread among all of a dispatchLinear() launch, whose workgroups are folded
// into two dimensions when there are more than one dimension may hold. A macro, because the
// workgroup size it reads is only declared by the kernel, after this file.
#define globalIndex()                                                    \
  ((gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * gl_WorkGroupSize.x + \
   gl_LocalInvocationID.x)

#ifdef A_T
layout(buffer_reference, std430, buffer_reference_align = A_ALIGN) buffer BufferA {
  A_T v[];
};
#if defined(A_BOOL)
#define LOAD_A(address, i) (uint(BufferA(address).v[i]) != 0u ? 1.0 : 0.0)
#else
#define LOAD_A(address, i) float(BufferA(address).v[i])
#endif
#define LOAD_A_RAW(address, i) BufferA(address).v[i]
#endif

#ifdef B_T
layout(buffer_reference, std430, buffer_reference_align = B_ALIGN) buffer BufferB {
  B_T v[];
};
#if defined(B_BOOL)
#define LOAD_B(address, i) (uint(BufferB(address).v[i]) != 0u ? 1.0 : 0.0)
#else
#define LOAD_B(address, i) float(BufferB(address).v[i])
#endif
#define LOAD_B_RAW(address, i) BufferB(address).v[i]
#endif

#ifdef C_T
layout(buffer_reference, std430, buffer_reference_align = C_ALIGN) buffer BufferC {
  C_T v[];
};
#if defined(C_BOOL)
#define STORE_C(address, i, x) BufferC(address).v[i] = C_T(uint((x) != 0.0))
#elif defined(C_INTEGER)
#define STORE_C(address, i, x) BufferC(address).v[i] = C_T(int64_t(x))
#else
#define STORE_C(address, i, x) BufferC(address).v[i] = C_T(x)
#endif
#define STORE_C_RAW(address, i, x) BufferC(address).v[i] = (x)
#endif

// Float buffers of either width, for kernels that also read or write tensors whose type does
// not follow a slot -- statistics, weights of a fixed type and the like.
layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuffer {
  float v[];
};
layout(buffer_reference, std430, buffer_reference_align = 8) buffer LongBuffer {
  int64_t v[];
};

// erf, which GLSL does not have: Abramowitz and Stegun 7.1.26, whose error is under 1.5e-7
// everywhere -- about what a float can resolve of a value near one.
float erfApprox(float x) {
  float s = sign(x);
  float a = abs(x);
  float t = 1.0 / (1.0 + 0.3275911 * a);
  float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t +
                   0.254829592) *
                      t * exp(-a * a);
  return s * y;
}

float gelu(float x) {
  return 0.5 * x * (1.0 + erfApprox(x * 0.70710678118654752));
}

float silu(float x) {
  return x / (1.0 + exp(-x));
}
