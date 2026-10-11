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

// The products on gemm.h: the batched matrix multiply, and convolution as an implicit GEMM.

#include "gemm.h"

// Keep each of these in step with the struct of the same name in matmul.cc.

// How an operand may be read four elements at a time, which matmul.cc works out: zero when it may
// not, or along which of its dimensions -- the one of unit stride, every other stride and where
// it starts being a multiple of four elements.
constant constexpr uint kVecAlongK = 1;
constant constexpr uint kVecAlongMN = 2;

struct GemmParams {
  uint M;
  uint N;
  uint K;
  uint strideAm;
  uint strideAk;
  uint strideBk;
  uint strideBn;
  uint batch1;
  uint strideA0;
  uint strideA1;
  uint strideB0;
  uint strideB1;
  uint vecA;
  uint vecB;
};

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
  uint pointwise;  // a 1 x 1 kernel of stride one and no padding: the unfolded input is the input
  uint vecWeight;  // the weight rows may be read four at a time
  uint vecInput;   // pointwise, and the input rows may be read four at a time
};

// For every batch (b0, b1), C[b0, b1] = A[b0, b1] B[b0, b1], A and B with any strides -- a batch
// stride of zero broadcasts -- and C contiguous. The batch is the grid's z, b0 * batch1 + b1.
template <typename T>
struct MatmulProblem {
  device T *c;
  device const T *a;
  device const T *b;
  constant GemmParams *p;

  static constexpr constant uint kTileK = TileK<T>::value;
  static constexpr constant uint kStrideA = StrideA<T>::value;
  // Elements of a staged tile each thread loads.
  static constexpr constant uint kPerThreadA = kTileM * kTileK / kGemmThreads;
  static constexpr constant uint kPerThreadB = kTileK * kTileN / kGemmThreads;

  T loadA(uint m, uint k) const {
    return a[m * p->strideAm + k * p->strideAk];
  }
  T loadB(uint k, uint n) const {
    return b[k * p->strideBk + n * p->strideBn];
  }
  vec<T, 4> loadA4(uint m, uint k) const {
    return *reinterpret_cast<device const vec<T, 4> *>(a + m * p->strideAm + k * p->strideAk);
  }

  bool hasRowBias() const {
    return false;
  }
  void stageRowBias(threadgroup float *, uint, uint) const {}
  vec<T, 4> loadB4(uint k, uint n) const {
    return *reinterpret_cast<device const vec<T, 4> *>(b + k * p->strideBk + n * p->strideBn);
  }

  void stageA(threadgroup T *tile, uint m0, uint k0, uint tid) const {
    bool inside = m0 + kTileM <= p->M && k0 + kTileK <= p->K;
    if (inside && p->vecA == kVecAlongK) {
      for (uint j = 0; j < kPerThreadA / 4; ++j) {
        uint v = tid + kGemmThreads * j;
        uint m = v / (kTileK / 4);
        uint k = (v % (kTileK / 4)) * 4;
        vec<T, 4> x = loadA4(m0 + m, k0 + k);
        for (uint e = 0; e < 4; ++e) tile[m * kStrideA + k + e] = x[e];
      }
    } else if (inside && p->vecA == kVecAlongMN) {
      for (uint j = 0; j < kPerThreadA / 4; ++j) {
        uint v = tid + kGemmThreads * j;
        uint k = v / (kTileM / 4);
        uint m = (v % (kTileM / 4)) * 4;
        vec<T, 4> x = loadA4(m0 + m, k0 + k);
        for (uint e = 0; e < 4; ++e) tile[(m + e) * kStrideA + k] = x[e];
      }
    } else {
      // Adjacent threads read adjacent addresses when they can.
      bool alongM = p->strideAm == 1 && p->strideAk != 1;
      for (uint j = 0; j < kPerThreadA; ++j) {
        uint m, k;
        if (alongM) {
          m = tid % kTileM;
          k = tid / kTileM + (kGemmThreads / kTileM) * j;
        } else {
          k = tid % kTileK;
          m = tid / kTileK + (kGemmThreads / kTileK) * j;
        }
        bool in = m0 + m < p->M && k0 + k < p->K;
        tile[m * kStrideA + k] = in ? loadA(m0 + m, k0 + k) : T(0);
      }
    }
  }

  void stageB(threadgroup T *tile, uint k0, uint n0, uint tid) const {
    bool inside = n0 + kTileN <= p->N && k0 + kTileK <= p->K;
    if (inside && p->vecB == kVecAlongMN) {
      for (uint j = 0; j < kPerThreadB / 4; ++j) {
        uint v = tid + kGemmThreads * j;
        uint k = v / (kTileN / 4);
        uint n = (v % (kTileN / 4)) * 4;
        vec<T, 4> x = loadB4(k0 + k, n0 + n);
        for (uint e = 0; e < 4; ++e) tile[k * kStrideB + n + e] = x[e];
      }
    } else if (inside && p->vecB == kVecAlongK) {
      for (uint j = 0; j < kPerThreadB / 4; ++j) {
        uint v = tid + kGemmThreads * j;
        uint n = v / (kTileK / 4);
        uint k = (v % (kTileK / 4)) * 4;
        vec<T, 4> x = loadB4(k0 + k, n0 + n);
        for (uint e = 0; e < 4; ++e) tile[(k + e) * kStrideB + n] = x[e];
      }
    } else {
      bool alongN = p->strideBn == 1;
      for (uint j = 0; j < kPerThreadB; ++j) {
        uint k, n;
        if (alongN) {
          n = tid % kTileN;
          k = tid / kTileN + (kGemmThreads / kTileN) * j;
        } else {
          k = tid % kTileK;
          n = tid / kTileK + (kGemmThreads / kTileK) * j;
        }
        bool in = n0 + n < p->N && k0 + k < p->K;
        tile[k * kStrideB + n] = in ? loadB(k0 + k, n0 + n) : T(0);
      }
    }
  }

  bool storeBlock(simdgroup_matrix<float, 8, 8> block, uint m, uint n) const {
    simdgroup_store(convertBlock<T>(block), c + m * p->N + n, p->N);
    return true;
  }
  void storeC(uint m, uint n, float value) const {
    Elem<T>::store(c, m * p->N + n, value);
  }
};

template <typename T>
kernel void matmul(
    device T *c [[buffer(0)]],
    device const T *a [[buffer(1)]],
    device const T *b [[buffer(2)]],
    constant GemmParams &p [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  FLINT_GEMM_SHARED(T)

  uint batch = group.z;
  uint b0 = batch / p.batch1;
  uint b1 = batch - b0 * p.batch1;

  MatmulProblem<T> problem;
  problem.c = c + batch * p.M * p.N;
  problem.a = a + b0 * p.strideA0 + b1 * p.strideA1;
  problem.b = b + b0 * p.strideB0 + b1 * p.strideB1;
  problem.p = &p;
  gemm<T>(problem, p.M, p.N, p.K, group.xy, tid, sg, lane, tileA, tileB, scratch);
}

// 2-D convolution, never writing the unfolded input out. For one group of one image at a time --
// the grid's z, image * groups + group -- the product is
//
//   C (K, P * Q) = weight (K, C * R * S) unfold(input) (C * R * S, P * Q)
//
// taken per group, with the input (N, C, H, W) and weight (K, C / groups, R, S) contiguous, and
// the output (N, K, P, Q) contiguous -- which is C, row by row, so that it is written a block at a
// time. The bias, when hasBias is set, is where each row of C starts. Stride, padding and dilation
// are per axis, which is what lets a 1-D convolution be this one over an image one row tall.
template <typename T>
struct ConvProblem {
  device T *c;           // this image's output for this group
  device const T *input;   // this image's input channels for this group
  device const T *weight;  // this group's weight
  device const T *bias;    // and its bias
  constant ConvParams *p;
  uint M;  // output channels of a group
  uint N;  // P * Q
  uint K;  // C / groups * R * S

  static constexpr constant uint kTileK = TileK<T>::value;
  static constexpr constant uint kStrideA = StrideA<T>::value;
  static constexpr constant uint kPerThreadA = kTileM * kTileK / kGemmThreads;
  static constexpr constant uint kPerThreadB = kTileK * kTileN / kGemmThreads;

  bool hasRowBias() const {
    return p->hasBias != 0;
  }
  void stageRowBias(threadgroup float *table, uint m0, uint tid) const {
    for (uint i = tid; i < kTileM * 8; i += kGemmThreads) {
      uint m = m0 + i / 8;
      table[i] = m < M ? Elem<T>::load(bias, m) : 0.0f;
    }
  }

  // The weight, (M, K) and contiguous.
  void stageA(threadgroup T *tile, uint m0, uint k0, uint tid) const {
    if (p->vecWeight && m0 + kTileM <= M && k0 + kTileK <= K) {
      for (uint j = 0; j < kPerThreadA / 4; ++j) {
        uint v = tid + kGemmThreads * j;
        uint m = v / (kTileK / 4);
        uint k = (v % (kTileK / 4)) * 4;
        vec<T, 4> x = *reinterpret_cast<device const vec<T, 4> *>(weight + (m0 + m) * K + k0 + k);
        for (uint e = 0; e < 4; ++e) tile[m * kStrideA + k + e] = x[e];
      }
      return;
    }
    for (uint j = 0; j < kPerThreadA; ++j) {
      uint k = tid % kTileK;
      uint m = tid / kTileK + (kGemmThreads / kTileK) * j;
      bool in = m0 + m < M && k0 + k < K;
      tile[m * kStrideA + k] = in ? weight[(m0 + m) * K + k0 + k] : T(0);
    }
  }

  // The unfolded input, (K, N), gathered. Each thread keeps to one output pixel n, so that
  // adjacent threads read adjacent input pixels when the stride is one, and walks k two at a time
  // from where it starts, carrying through the kernel's columns, rows and channels rather than
  // dividing for every element.
  void stageB(threadgroup T *tile, uint k0, uint n0, uint tid) const {
    if (p->pointwise) {
      if (p->vecInput && n0 + kTileN <= N && k0 + kTileK <= K) {
        for (uint j = 0; j < kPerThreadB / 4; ++j) {
          uint v = tid + kGemmThreads * j;
          uint k = v / (kTileN / 4);
          uint n = (v % (kTileN / 4)) * 4;
          vec<T, 4> x = *reinterpret_cast<device const vec<T, 4> *>(input + (k0 + k) * N + n0 + n);
          for (uint e = 0; e < 4; ++e) tile[k * kStrideB + n + e] = x[e];
        }
        return;
      }
      for (uint j = 0; j < kPerThreadB; ++j) {
        uint n = tid % kTileN;
        uint k = tid / kTileN + (kGemmThreads / kTileN) * j;
        bool in = n0 + n < N && k0 + k < K;
        tile[k * kStrideB + n] = in ? input[(k0 + k) * N + n0 + n] : T(0);
      }
      return;
    }

    constexpr uint kRows = kGemmThreads / kTileN;  // the k each thread starts from, 0 or 1
    uint n = tid % kTileN;
    uint k = tid / kTileN;
    uint pixel = n0 + n;

    uint row = pixel / p->outWidth;
    uint column = pixel - row * p->outWidth;
    int y0 = int(row * p->strideH) - p->paddingH;
    int x0 = int(column * p->strideW) - p->paddingW;

    uint kernelSize = p->kernelHeight * p->kernelWidth;
    uint first = k0 + k;
    uint channel = first / kernelSize;
    uint rs = first - channel * kernelSize;
    uint r = rs / p->kernelWidth;
    uint s = rs - r * p->kernelWidth;

    for (uint j = 0; j < kPerThreadB; ++j) {
      uint kk = k + kRows * j;
      T value = T(0);
      if (pixel < N && k0 + kk < K) {
        int y = y0 + int(r * p->dilationH);
        int x = x0 + int(s * p->dilationW);
        if (y >= 0 && y < int(p->height) && x >= 0 && x < int(p->width)) {
          value = input[(channel * p->height + uint(y)) * p->width + uint(x)];
        }
      }
      tile[kk * kStrideB + n] = value;

      s += kRows;
      while (s >= p->kernelWidth) {
        s -= p->kernelWidth;
        if (++r == p->kernelHeight) {
          r = 0;
          ++channel;
        }
      }
    }
  }

  bool storeBlock(simdgroup_matrix<float, 8, 8> block, uint m, uint n) const {
    simdgroup_store(convertBlock<T>(block), c + m * N + n, N);
    return true;
  }
  void storeC(uint m, uint n, float value) const {
    Elem<T>::store(c, m * N + n, value);
  }
};

template <typename T>
kernel void conv2d(
    device T *c [[buffer(0)]],
    device const T *input [[buffer(1)]],
    device const T *weight [[buffer(2)]],
    device const T *bias [[buffer(3)]],
    constant ConvParams &p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  FLINT_GEMM_SHARED(T)

  uint image = group.z / p.groups;
  uint g = group.z - image * p.groups;
  uint channelsPerGroup = p.channels / p.groups;
  uint outChannelsPerGroup = p.outChannels / p.groups;
  uint spatial = p.outHeight * p.outWidth;

  ConvProblem<T> problem;
  problem.M = outChannelsPerGroup;
  problem.N = spatial;
  problem.K = channelsPerGroup * p.kernelHeight * p.kernelWidth;
  problem.c = c + (image * p.outChannels + g * outChannelsPerGroup) * spatial;
  problem.input = input + (image * p.channels + g * channelsPerGroup) * p.height * p.width;
  problem.weight = weight + g * outChannelsPerGroup * problem.K;
  problem.bias = bias + (p.hasBias ? g * outChannelsPerGroup : 0);
  problem.p = &p;
  gemm<T>(problem, problem.M, problem.N, problem.K, group.xy, tid, sg, lane, tileA, tileB, scratch);
}

#define FLINT_GEMM_KERNELS(T, S)                                                                \
  template [[host_name("matmul_" S)]] kernel void matmul<T>(                                    \
      device T *, device const T *, device const T *, constant GemmParams &, uint3, uint, uint,  \
      uint);                                                                                    \
  template [[host_name("conv2d_" S)]] kernel void conv2d<T>(                                    \
      device T *, device const T *, device const T *, device const T *, constant ConvParams &,  \
      uint3, uint, uint, uint);

FLINT_GEMM_KERNELS(float, "f32")
FLINT_GEMM_KERNELS(half, "f16")
