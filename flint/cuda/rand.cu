// The MIT License (MIT)
//
// Copyright (c) 2025 Xiaoyang Chen
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

#include "flint/cuda/rand.h"

#include <math.h>
#include <stdint.h>

#include <memory>

#include "flint/cuda/cast.h"
#include "flint/cuda/common.h"

namespace fl {
namespace op {
namespace cuda {
namespace {

// Philox4x32-10, from Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" (SC11). It is
// written out here rather than called in cuRAND because cuRAND is a DLL on Windows -- NVIDIA
// ships no static cuRAND for that host -- and the import is resolved by the loader before main()
// runs, so a machine with a driver but no CUDA toolkit could not start the binary at all, CPU
// paths included. What was used from that library is all below; this is the generator cuRAND was
// asked for by name, CURAND_RNG_PSEUDO_PHILOX4_32_10.
//
// It is a keyed bijection on a 128-bit counter rather than a state that is stirred, so a draw is
// a pure function of (seed, position). Nothing lives on the card between calls, and threads need
// no coordination: each one works out where its own values sit in the stream and goes straight
// to them.
//
// The stream is not cuRAND's. The two agree on the bijection and differ in how a position is
// laid into the counter, which each implementation keeps to itself, so the same seed draws a
// different latent than it did while this file called cuRAND.

constexpr uint32_t kPhiloxM0 = 0xD2511F53;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9;  // the golden ratio, in Weyl-sequence form
constexpr uint32_t kPhiloxW1 = 0xBB67AE85;  // sqrt(3) - 1, likewise

constexpr int kPhiloxRounds = 10;

/// 2^-32. A draw is scaled by this and offset by half of it, which lands it in (0, 1) rather than
/// [0, 1): randNormal takes the logarithm of one, and log(0) is not a float this can return.
constexpr float kTwoPow32Inv = 2.3283064e-10f;

constexpr float kTwoPi = 6.283185307179586f;

__host__ __device__ inline uint32_t mulhi32(uint32_t a, uint32_t b) {
#ifdef __CUDA_ARCH__
  return __umulhi(a, b);
#else
  return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
#endif
}

/// One round: two 32x32->64 multiplies, and a shuffle that crosses each product's halves into the
/// other lane's word. The diffusion is in the multiply; the key only ever enters by xor.
__host__ __device__ inline void philoxRound(uint32_t ctr[4], uint32_t k0, uint32_t k1) {
  uint32_t hi0 = mulhi32(kPhiloxM0, ctr[0]);
  uint32_t lo0 = kPhiloxM0 * ctr[0];
  uint32_t hi1 = mulhi32(kPhiloxM1, ctr[2]);
  uint32_t lo1 = kPhiloxM1 * ctr[2];

  uint32_t c0 = hi1 ^ ctr[1] ^ k0;
  uint32_t c1 = lo1;
  uint32_t c2 = hi0 ^ ctr[3] ^ k1;
  uint32_t c3 = lo0;

  ctr[0] = c0;
  ctr[1] = c1;
  ctr[2] = c2;
  ctr[3] = c3;
}

/// Ten rounds over `ctr` keyed by (k0, k1), which is the whole generator. The key is bumped by the
/// two Weyl constants before every round but the first, so one 64-bit seed keys the ten rounds
/// ten different ways.
__host__ __device__ inline void philox4x32_10(
    const uint32_t ctr[4],
    uint32_t k0,
    uint32_t k1,
    uint32_t out[4]) {
  out[0] = ctr[0];
  out[1] = ctr[1];
  out[2] = ctr[2];
  out[3] = ctr[3];

  for (int round = 0; round < kPhiloxRounds; ++round) {
    if (round > 0) {
      k0 += kPhiloxW0;
      k1 += kPhiloxW1;
    }
    philoxRound(out, k0, k1);
  }
}

/// The four words that sit at `position` in `seed`'s stream. Position goes in the low pair of
/// counter words and the high pair stays zero: a run would have to draw 2^66 values before the
/// low pair wrapped, and the words left at zero cost nothing, since the counter is diffused
/// whole.
__host__ __device__ inline void philoxAt(uint64_t seed, uint64_t position, uint32_t out[4]) {
  uint32_t ctr[4] = {
      static_cast<uint32_t>(position),
      static_cast<uint32_t>(position >> 32),
      0u,
      0u};
  philox4x32_10(ctr, static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32), out);
}

__device__ inline float uniformFromBits(uint32_t bits) {
  return bits * kTwoPow32Inv + kTwoPow32Inv / 2.0f;
}

/// Box-Muller: a pair of uniforms becomes a pair of independent standard normals, one on each axis
/// of a point drawn at a uniform angle and a Rayleigh radius. Four words is two pairs, so a draw
/// fills its block exactly and no bits are thrown away.
__device__ inline void normalFromBitPair(uint32_t b0, uint32_t b1, float &z0, float &z1) {
  float radius = sqrtf(-2.0f * logf(uniformFromBits(b0)));
  float angle = kTwoPi * uniformFromBits(b1);

  float sinAngle;
  float cosAngle;
  sincosf(angle, &sinAngle, &cosAngle);

  z0 = radius * cosAngle;
  z1 = radius * sinAngle;
}

/// `n` values starting at block `base` of `seed`'s stream. The loop is grid-stride and counts in
/// blocks of four, so the last block is the only one that can run past the end of the tensor.
__global__ void uniformKernel(float *__restrict__ out, int n, uint64_t seed, uint64_t base) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int numBlocks = (n + 3) / 4;

  for (int block = idx; block < numBlocks; block += stride) {
    uint32_t bits[4];
    philoxAt(seed, base + block, bits);

    int offset = block * 4;
    for (int i = 0; i < 4 && offset + i < n; ++i) {
      out[offset + i] = uniformFromBits(bits[i]);
    }
  }
}

__global__ void normalKernel(float *__restrict__ out, int n, uint64_t seed, uint64_t base) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int numBlocks = (n + 3) / 4;

  for (int block = idx; block < numBlocks; block += stride) {
    uint32_t bits[4];
    philoxAt(seed, base + block, bits);

    float z[4];
    normalFromBitPair(bits[0], bits[1], z[0], z[1]);
    normalFromBitPair(bits[2], bits[3], z[2], z[3]);

    int offset = block * 4;
    for (int i = 0; i < 4 && offset + i < n; ++i) {
      out[offset + i] = z[i];
    }
  }
}

/// How many four-wide blocks `numel` values occupy. A draw always consumes whole blocks, so the
/// next one starts on a boundary and two draws in a run can never share a position.
inline uint64_t blocksFor(int numel) {
  return static_cast<uint64_t>((numel + 3) / 4);
}

}  // namespace

class Rand::Impl {
 public:
  ~Impl() = default;
  static std::unique_ptr<Impl> newImpl();

  Tensor randNormal(lut::Span<const int> shape);
  Tensor rand(lut::Span<const int> shape);
  void setSeed(uint64_t seed);

 private:
  Impl() = default;

  uint64_t _seed = 0;

  /// Where the next draw begins, counted in four-wide blocks. This is the only state a generator
  /// keeps: it is what makes two draws in a row differ, and what setSeed puts back to the start.
  uint64_t _position = 0;
};

Tensor Rand::Impl::randNormal(lut::Span<const int> shape) {
  Tensor result = createCudaTensorFloat(shape);
  int numel = static_cast<int>(result.getNumEl());

  constexpr int blockSize = 256;
  dim3 grid = getGrid1D(static_cast<int>(blocksFor(numel)), blockSize);

  normalKernel<<<grid, blockSize>>>(getDataPtrCuda<float>(result), numel, _seed, _position);
  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  _position += blocksFor(numel);

  return castFloatToHalf(result);
}

Tensor Rand::Impl::rand(lut::Span<const int> shape) {
  Tensor result = createCudaTensorFloat(shape);
  int numel = static_cast<int>(result.getNumEl());

  constexpr int blockSize = 256;
  dim3 grid = getGrid1D(static_cast<int>(blocksFor(numel)), blockSize);

  uniformKernel<<<grid, blockSize>>>(getDataPtrCuda<float>(result), numel, _seed, _position);
  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  _position += blocksFor(numel);

  return result;
}

void Rand::Impl::setSeed(uint64_t seed) {
  _seed = seed;

  // Back to the start of the stream, rather than only on to a new one: a caller who seeds twice
  // with the same number wants the same draws both times.
  _position = 0;
}

std::unique_ptr<Rand::Impl> Rand::Impl::newImpl() {
  return std::unique_ptr<Impl>{new Impl()};
}

std::shared_ptr<Rand> Rand::newRand() {
  std::shared_ptr<Rand> rand{new Rand()};
  rand->_impl = Rand::Impl::newImpl();

  return rand;
}

Tensor Rand::randNormal(lut::Span<const int> shape) {
  return _impl->randNormal(shape);
}

Tensor Rand::rand(lut::Span<const int> shape) {
  return _impl->rand(shape);
}

void Rand::setSeed(uint64_t seed) {
  _impl->setSeed(seed);
}

void philox4x32_10ForTest(const uint32_t ctr[4], const uint32_t key[2], uint32_t out[4]) {
  philox4x32_10(ctr, key[0], key[1], out);
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
