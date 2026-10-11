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

// Layer, RMS and group norm: one threadgroup a row (or a group), statistics in float whatever the
// storage type. The weight and bias are A's type; each is used only when its flag in the params is
// set, and the input is bound in the place of one that is not there.
//
// The variance is E[x^2] - E[x]^2, and eps goes inside the square root, as on the other devices.

#include "rows.h"

/// In step with NormParams in norm.cc.
struct NormParams {
  uint numRows;
  uint rowLength;
  float eps;
  uint hasWeight;
  uint hasBias;
};

/// In step with GroupNormParams in norm.cc.
struct GroupNormParams {
  uint numGroups;        // N * groups
  uint groupLength;      // (C / groups) * H * W
  uint spatial;          // H * W
  uint channelsPerGroup;
  uint groups;
  float eps;
  uint hasWeight;
  uint hasBias;
};

inline float invStdOf(float2 moments, float mean, float eps) {
  float variance = moments.y - mean * mean;
  return rsqrt(variance > 0.0f ? variance + eps : eps);
}

template <typename T>
inline float2 momentsOf(
    device const T *a,
    uint base,
    uint length,
    uint tid,
    threadgroup float *scratch,
    threadgroup float *scratch2) {
  float2 sums = float2(0.0f);
  for (uint i = tid; i < length; i += kRowThreads) {
    float x = Elem<T>::load(a, base + i);
    sums += float2(x, x * x);
  }
  return sumRow2(sums, tid, scratch, scratch2) / float(length);
}

template <typename T, bool kRms>
kernel void norm(
    device T *c [[buffer(0)]],
    device const T *a [[buffer(1)]],
    device const T *weight [[buffer(2)]],
    device const T *bias [[buffer(3)]],
    constant NormParams &p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[kRowThreads];
  threadgroup float scratch2[kRowThreads];
  if (row >= p.numRows) return;
  uint base = row * p.rowLength;

  float2 moments = momentsOf(a, base, p.rowLength, tid, scratch, scratch2);
  float mean = kRms ? 0.0f : moments.x;
  float invStd = invStdOf(moments, mean, p.eps);

  for (uint i = tid; i < p.rowLength; i += kRowThreads) {
    float y = (Elem<T>::load(a, base + i) - mean) * invStd;
    if (p.hasWeight) y *= Elem<T>::load(weight, i);
    if (p.hasBias) y += Elem<T>::load(bias, i);
    Elem<T>::store(c, base + i, y);
  }
}

template <typename T>
kernel void group_norm(
    device T *c [[buffer(0)]],
    device const T *a [[buffer(1)]],
    device const T *weight [[buffer(2)]],
    device const T *bias [[buffer(3)]],
    constant GroupNormParams &p [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[kRowThreads];
  threadgroup float scratch2[kRowThreads];
  if (group >= p.numGroups) return;
  uint base = group * p.groupLength;

  float2 moments = momentsOf(a, base, p.groupLength, tid, scratch, scratch2);
  float mean = moments.x;
  float invStd = invStdOf(moments, mean, p.eps);

  uint firstChannel = (group % p.groups) * p.channelsPerGroup;
  for (uint i = tid; i < p.groupLength; i += kRowThreads) {
    float y = (Elem<T>::load(a, base + i) - mean) * invStd;
    uint channel = firstChannel + i / p.spatial;
    if (p.hasWeight) y *= Elem<T>::load(weight, channel);
    if (p.hasBias) y += Elem<T>::load(bias, channel);
    Elem<T>::store(c, base + i, y);
  }
}

#define FLINT_NORM_KERNELS(T, S)                                                                \
  template [[host_name("layer_norm_" S)]] kernel void norm<T, false>(                           \
      device T *, device const T *, device const T *, device const T *, constant NormParams &, \
      uint, uint);                                                                              \
  template [[host_name("rms_norm_" S)]] kernel void norm<T, true>(                              \
      device T *, device const T *, device const T *, device const T *, constant NormParams &, \
      uint, uint);                                                                              \
  template [[host_name("group_norm_" S)]] kernel void group_norm<T>(                            \
      device T *, device const T *, device const T *, device const T *,                         \
      constant GroupNormParams &, uint, uint);

FLINT_NORM_KERNELS(float, "f32")
FLINT_NORM_KERNELS(half, "f16")
