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

// Uniform or normal random floats from Philox4x32-10, the generator flint/cuda/rand.cu and the
// Vulkan backend use and in the same arrangement, so that a seed draws the same numbers on every
// device: element i of a draw is word i % 4 of the block at counter `base + i / 4`, keyed by the
// seed. Normals are Box-Muller over word pairs (0, 1) and (2, 3).

#include "common.h"

/// In step with RandParams in rand.cc.
struct RandParams {
  ulong seed;
  ulong base;
  uint numel;
  uint normal;
};

constant constexpr uint kM0 = 0xD2511F53u;
constant constexpr uint kM1 = 0xCD9E8D57u;
constant constexpr uint kW0 = 0x9E3779B9u;
constant constexpr uint kW1 = 0xBB67AE85u;
constant constexpr float kTwoPow32Inv = 2.3283064e-10f;
constant constexpr float kTwoPi = 6.283185307179586f;

inline uint4 philox(uint4 ctr, uint k0, uint k1) {
  for (int round = 0; round < 10; ++round) {
    if (round > 0) {
      k0 += kW0;
      k1 += kW1;
    }
    uint hi0 = mulhi(kM0, ctr.x);
    uint lo0 = kM0 * ctr.x;
    uint hi1 = mulhi(kM1, ctr.z);
    uint lo1 = kM1 * ctr.z;
    ctr = uint4(hi1 ^ ctr.y ^ k0, lo1, hi0 ^ ctr.w ^ k1, lo0);
  }
  return ctr;
}

inline float uniformFromBits(uint bits) {
  return float(bits) * kTwoPow32Inv + kTwoPow32Inv / 2.0f;
}

inline float2 normalFromBits(uint b0, uint b1) {
  float radius = sqrt(-2.0f * log(uniformFromBits(b0)));
  float angle = kTwoPi * uniformFromBits(b1);
  return radius * float2(cos(angle), sin(angle));
}

kernel void rand_f32(
    device float *c [[buffer(0)]],
    constant RandParams &p [[buffer(1)]],
    uint block [[thread_position_in_grid]]) {
  uint first = block * 4;
  if (first >= p.numel) return;

  ulong position = p.base + ulong(block);
  uint4 counter = uint4(uint(position), uint(position >> 32), 0u, 0u);
  uint4 bits = philox(counter, uint(p.seed), uint(p.seed >> 32));

  float4 values;
  if (p.normal != 0) {
    values = float4(normalFromBits(bits.x, bits.y), normalFromBits(bits.z, bits.w));
  } else {
    values = float4(
        uniformFromBits(bits.x),
        uniformFromBits(bits.y),
        uniformFromBits(bits.z),
        uniformFromBits(bits.w));
  }

  for (uint i = 0; i < 4 && first + i < p.numel; ++i) c[first + i] = values[i];
}
