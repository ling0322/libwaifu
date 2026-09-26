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

// For kernels that give each row a workgroup of ROW_THREADS threads and reduce over it.

#define ROW_THREADS 256

layout(local_size_x = ROW_THREADS) in;

shared float rowScratch[ROW_THREADS];
shared float rowScratch2[ROW_THREADS];

// The row this workgroup is for. Rows are dispatched the way dispatchLinear() dispatches threads,
// so there may be more of them than one dimension of workgroups holds.
uint rowIndex() {
  return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
}

#define REDUCE_SUM 0
#define REDUCE_MAX 1
#define REDUCE_MIN 2

float combine(float x, float y, uint op) {
  if (op == REDUCE_MAX) return max(x, y);
  if (op == REDUCE_MIN) return min(x, y);
  return x + y;
}

// Every thread's `x` combined, returned to every thread.
float reduceRow(float x, uint op) {
  uint tid = gl_LocalInvocationID.x;
  rowScratch[tid] = x;
  barrier();
  for (uint s = ROW_THREADS / 2; s > 0; s >>= 1) {
    if (tid < s) rowScratch[tid] = combine(rowScratch[tid], rowScratch[tid + s], op);
    barrier();
  }
  float result = rowScratch[0];
  barrier();
  return result;
}

// Two sums at once, for the mean and the mean square.
vec2 sumRow2(vec2 x) {
  uint tid = gl_LocalInvocationID.x;
  rowScratch[tid] = x.x;
  rowScratch2[tid] = x.y;
  barrier();
  for (uint s = ROW_THREADS / 2; s > 0; s >>= 1) {
    if (tid < s) {
      rowScratch[tid] += rowScratch[tid + s];
      rowScratch2[tid] += rowScratch2[tid + s];
    }
    barrier();
  }
  vec2 result = vec2(rowScratch[0], rowScratch2[0]);
  barrier();
  return result;
}
