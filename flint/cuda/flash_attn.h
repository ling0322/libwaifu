// The MIT License (MIT)
//
// Copyright (c) 2023-2025 Xiaoyang Chen
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

#pragma once

#include "flint/tensor.h"
#include "flint/tensor_view.h"

namespace fl {
namespace op {
namespace cuda {

/// Runs FlashAttention on q [b, nHead, L, D] and k, v [b, nKvHead, S, D], writing `out`
/// <half>[b, nHead, L, D], contiguous. Returns false, with `out` untouched, if no compiled kernel
/// matches the inputs.
bool flashAttention(
    const TensorView &q,
    const TensorView &k,
    const TensorView &v,
    bool causal,
    const TensorView &out);

/// Runs FlashAttention of a packed batch of queries over a paged KV cache. q is
/// [totalQLen, nHead, D], keyCache and valueCache are [nBlock, blockSize, nKvHead, D], blockTable
/// is <int>[nSeq, maxNumBlock], cuSeqlensQ is <int>[nSeq + 1] and seqlensK is <int>[nSeq].
/// maxQLen and maxKLen bound the per-sequence lengths; the kernel sizes its grid from them.
/// The result is written into `out` <half>[totalQLen, nHead, D], contiguous. Returns false if no
/// compiled kernel matches the inputs.
bool pagedFlashAttention(
    const TensorView &q,
    const TensorView &keyCache,
    const TensorView &valueCache,
    const TensorView &blockTable,
    const TensorView &cuSeqlensQ,
    const TensorView &seqlensK,
    int maxQLen,
    int maxKLen,
    bool causal,
    const TensorView &out);

}  // namespace cuda
}  // namespace op
}  // namespace fl
