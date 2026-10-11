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


// The reductions of the last dimension, softmax and cumsum: one threadgroup a row, accumulated in
// float whatever the storage type.

#include "rows.h"

// Sum, max or min of each row. For bool, `all`: whether every element of the row is true.
template <typename A, typename C>
kernel void reduce(
    device C *c [[buffer(0)]],
    device const A *a [[buffer(1)]],
    constant RowParams &p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[kRowThreads];
  if (row >= p.numRows) return;
  uint base = row * p.rowLength;

  float x = p.op == kReduceSum ? 0.0f : (p.op == kReduceMax ? -INFINITY : INFINITY);
  for (uint i = tid; i < p.rowLength; i += kRowThreads) {
    x = combine(x, Elem<A>::load(a, base + i), p.op);
  }
  x = reduceRow(x, p.op, tid, scratch);
  if (tid == 0) Elem<C>::store(c, row, x);
}

kernel void reduce_bool(
    device Bool8 *c [[buffer(0)]],
    device const Bool8 *a [[buffer(1)]],
    constant RowParams &p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[kRowThreads];
  if (row >= p.numRows) return;
  uint base = row * p.rowLength;

  float x = 1.0f;
  for (uint i = tid; i < p.rowLength; i += kRowThreads) x = min(x, Elem<Bool8>::load(a, base + i));
  x = reduceRow(x, kReduceMin, tid, scratch);
  if (tid == 0) Elem<Bool8>::store(c, row, x);
}

template <typename T>
kernel void softmax(
    device T *c [[buffer(0)]],
    device const T *a [[buffer(1)]],
    constant RowParams &p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[kRowThreads];
  if (row >= p.numRows) return;
  uint base = row * p.rowLength;

  float m = -INFINITY;
  for (uint i = tid; i < p.rowLength; i += kRowThreads) m = max(m, Elem<T>::load(a, base + i));
  m = reduceRow(m, kReduceMax, tid, scratch);

  float s = 0.0f;
  for (uint i = tid; i < p.rowLength; i += kRowThreads) s += exp(Elem<T>::load(a, base + i) - m);
  s = reduceRow(s, kReduceSum, tid, scratch);

  float scale = 1.0f / s;
  for (uint i = tid; i < p.rowLength; i += kRowThreads) {
    Elem<T>::store(c, base + i, exp(Elem<T>::load(a, base + i) - m) * scale);
  }
}

// The inclusive prefix sum of each row, a tile of kRowThreads elements at a time: each tile is
// scanned in threadgroup memory by doubling -- every step adds the value `offset` places back,
// offset 1, 2, 4, ... -- and the running total of the tiles before carries into the next one, so a
// row of any length is one dispatch. The row length is the same for every thread, so every barrier
// is reached by all of them.
template <typename T>
kernel void scan(
    device T *c [[buffer(0)]],
    device const T *a [[buffer(1)]],
    constant RowParams &p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[kRowThreads];
  if (row >= p.numRows) return;
  uint base = row * p.rowLength;

  float carry = 0.0f;
  for (uint start = 0; start < p.rowLength; start += kRowThreads) {
    uint i = start + tid;
    scratch[tid] = i < p.rowLength ? Elem<T>::load(a, base + i) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = 1; offset < kRowThreads; offset <<= 1) {
      float behind = tid >= offset ? scratch[tid - offset] : 0.0f;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      scratch[tid] += behind;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i < p.rowLength) Elem<T>::store(c, base + i, carry + scratch[tid]);
    carry += scratch[kRowThreads - 1];
    // The scratch is refilled by the next tile; every thread has to be done reading it.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define FLINT_ROW_KERNELS(T, S)                                                                 \
  template [[host_name("reduce_" S)]] kernel void reduce<T, T>(                                 \
      device T *, device const T *, constant RowParams &, uint, uint);                          \
  template [[host_name("softmax_" S)]] kernel void softmax<T>(                                  \
      device T *, device const T *, constant RowParams &, uint, uint);                          \
  template [[host_name("scan_" S)]] kernel void scan<T>(                                        \
      device T *, device const T *, constant RowParams &, uint, uint);

FLINT_ROW_KERNELS(float, "f32")
FLINT_ROW_KERNELS(half, "f16")
