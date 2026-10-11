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


// Kernels that give each row of a contiguous tensor one threadgroup: the reductions, softmax,
// cumsum and the norms. A threadgroup is kRowThreads threads, one row is threadgroup
// `row` of the grid, and what the threads have each worked out is combined through threadgroup
// memory the kernel declares and hands in.

#pragma once

#include "common.h"

constant constexpr uint kRowThreads = 256;

// In step with ReduceOp in ops.h.
constant constexpr uint kReduceSum = 0;
constant constexpr uint kReduceMax = 1;
constant constexpr uint kReduceMin = 2;

/// What a kernel of rows is handed besides its buffers. In step with RowParams in reduce.cc.
struct RowParams {
  uint numRows;
  uint rowLength;
  uint op;
};

inline float combine(float x, float y, uint op) {
  if (op == kReduceMax) return max(x, y);
  if (op == kReduceMin) return min(x, y);
  return x + y;
}

/// `x` combined over every thread of the threadgroup, as every thread's answer.
inline float reduceRow(float x, uint op, uint tid, threadgroup float *scratch) {
  scratch[tid] = x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = kRowThreads / 2; s > 0; s >>= 1) {
    if (tid < s) scratch[tid] = combine(scratch[tid], scratch[tid + s], op);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float result = scratch[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

/// Two sums at once, for a mean and a mean of squares.
inline float2 sumRow2(float2 x, uint tid, threadgroup float *scratch, threadgroup float *scratch2) {
  scratch[tid] = x.x;
  scratch2[tid] = x.y;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = kRowThreads / 2; s > 0; s >>= 1) {
    if (tid < s) {
      scratch[tid] += scratch[tid + s];
      scratch2[tid] += scratch2[tid + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float2 result = float2(scratch[0], scratch2[0]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}
