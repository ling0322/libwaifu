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

#include "flint/cpu/lookup.h"

#include "flint/cpu/accessor.h"
#include "flint/cpu/common.h"
#include "flint/cpu/copy.h"

namespace fl {
namespace op {
namespace cpu {

template<typename T>
void lookupKernel2D(const TensorView &table, const TensorView &indices, const TensorView &xC) {
  CHECK(table.getDim() == 2 && indices.getDim() == 2);

  int vocabSize = table.getShape(0);
  int d0 = indices.getShape(0);
  int d1 = indices.getShape(1);
  int embdDim = table.getShape(1);
  xC.throwIfInvalidShape({d0, d1, embdDim}, "lookup");

  TensorAccessor<const T, 2> A = table;
  TensorAccessor<const LongType, 2> B = indices;
  TensorAccessor<T, 3> C = xC;

  for (int i = 0; i < d0; ++i) {
    for (int j = 0; j < d1; ++j) {
      int64_t index = B[i][j];
      CHECK(index < vocabSize) << "indices out of range";

      copyVector(C[i][j], A[index]);
    }
  }
}

template<typename T>
void lookupKernel1D(const TensorView &table, const TensorView &indices, const TensorView &xC) {
  CHECK(table.getDim() == 2 && indices.getDim() == 1);

  int vocabSize = table.getShape(0);
  int d0 = indices.getShape(0);
  int embdDim = table.getShape(1);
  xC.throwIfInvalidShape({d0, embdDim}, "lookup");

  TensorAccessor<const T, 2> A = table;
  TensorAccessor<const LongType, 1> B = indices;
  TensorAccessor<T, 2> C = xC;

  for (int i = 0; i < d0; ++i) {
    int64_t index = B[i];
    CHECK(index < vocabSize) << "indices out of range";

    copyVector(C[i], A[index]);
  }
}

void lookup(const TensorView &table, const TensorView &indices, const TensorView &C) {
  // a packed batch of ids is 1D, one embedding row comes out per id.
  if (indices.getDim() == 1) {
    if (table.getDType() == DType::kFloat) return lookupKernel1D<float>(table, indices, C);

    // A lookup copies a row; it does no arithmetic on it, so a half table needs nothing of the
    // architecture and hands back half. What to do with that is the caller's -- an embedding layer
    // converts to whatever it works in, which is where the decision belongs.
    if (table.getDType() == DType::kFloat16) return lookupKernel1D<Float16>(table, indices, C);
    NOT_IMPL();
  }

  if (table.getDType() == DType::kFloat) return lookupKernel2D<float>(table, indices, C);
  if (table.getDType() == DType::kFloat16) return lookupKernel2D<Float16>(table, indices, C);
  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
