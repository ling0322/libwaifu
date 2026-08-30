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

#include <math.h>

#include "lutil/error.h"
#include "lutil/log.h"
#include "flint/metal/common.h"
#include "flint/metal/ops.h"

namespace fl {
namespace op {
namespace metal {

Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v, bool causal) {
  CHECK(q.getDim() == 4) << "attention expects (batch, numHeads, length, headDim)";

  // MLX takes the same [batch, heads, length, headDim] layout flint documents, and handles
  // grouped-query attention by broadcasting the key and value heads, so no expansion here.
  float scale = 1.0f / sqrtf(static_cast<float>(q.getShape(-1)));

  return fromMlxArray(
      mlx::core::fast::scaled_dot_product_attention(
          toMlxArray(q),
          toMlxArray(k),
          toMlxArray(v),
          scale,
          causal ? "causal" : ""));
}

}  // namespace metal
}  // namespace op
}  // namespace fl
