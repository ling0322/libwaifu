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

#include <cuda_fp16.h>

#include "flint/cuda/accessor.h"
#include "flint/cuda/common.h"
#include "flint/cuda/lookup.h"
#include "flint/operators.h"

namespace fl {
namespace op {
namespace cuda {

// One thread per element of the output, on a grid that is one-dimensional.
//
// The grid used to be two- and three-dimensional, with the ids along y and z. Those two axes hold
// at most 65 535 blocks each, so a lookup of more ids than that -- w2v-bert's relative positions
// are one per *pair* of frames, a quarter of a million for ten seconds of speech -- failed to
// launch at all, with `invalid configuration argument` and no hint that the count was the
// problem. The x axis holds 2^31 - 1 blocks, which is more than any tensor here will have
// elements.
template<typename T>
__global__ void lookupKernel(
    const T *embd,
    const int64_t *ids,
    T *dst,
    int64_t rows,
    int64_t width,
    int64_t tableRows) {
  int64_t element = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= rows * width) return;

  int64_t row = element / width;
  int64_t column = element % width;

  int64_t index = ids[row];
  assert(index >= 0 && index < tableRows);
  dst[element] = embd[index * width + column];
}

// Every id at once, whatever shape they arrive in: one row of `embdTable` per id, the result shaped
// like `input` with the row's width after it.
template<typename T>
Tensor lookupRows(const Tensor &embdTable, const Tensor &input) {
  std::vector<Tensor::ShapeType> shape = input.getShape();
  shape.push_back(embdTable.getShape(1));
  Tensor dst = createCudaTensor<T>(shape);

  // Both are read as flat arrays, so both have to be laid out as one.
  Operators *ops = getOperators(Device::kCuda);
  Tensor table = embdTable.isContiguous() ? embdTable : ops->contiguous(embdTable);
  Tensor ids = input.isContiguous() ? input : ops->contiguous(input);

  int64_t rows = ids.getNumEl();
  int64_t width = table.getShape(1);
  int64_t total = rows * width;
  if (total == 0) return dst;

  constexpr int blockSize = 256;
  int64_t blocks = (total + blockSize - 1) / blockSize;

  lookupKernel<T><<<static_cast<unsigned int>(blocks), blockSize>>>(
      getDataPtrCuda<T>(table),
      getDataPtrCuda<int64_t>(ids),
      getDataPtrCuda<T>(dst),
      rows,
      width,
      table.getShape(0));
  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return dst;
}

Tensor lookup(const Tensor &embdTable, const Tensor &input) {
  CHECK(input.getDType() == DType::kLong);
  CHECK(input.getDevice().getType() == Device::kCuda);
  CHECK(embdTable.getDevice().getType() == Device::kCuda);
  CHECK(embdTable.getDim() == 2);

  // Ids are 2D for a batch of sequences and 1D for a packed one; either way one embedding row
  // comes out per id.
  if (input.getDim() == 1 || input.getDim() == 2) {
    if (embdTable.getDType() == DType::kFloat16) return lookupRows<half>(embdTable, input);
    if (embdTable.getDType() == DType::kFloat) return lookupRows<float>(embdTable, input);
  }

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
