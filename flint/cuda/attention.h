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

#pragma once

#include "flint/tensor_view.h"

namespace fl {
namespace op {
namespace cuda {

/// Scaled dot product attention in one pass over the keys, on the tensor cores: q <half>[b, nHead,
/// L, D] and k, v <half>[b, nKvHead, S, D], into `out` <half>[b, nHead, L, D], contiguous. nKvHead
/// divides nHead for grouped-query attention. `blockSize` is the mask as Operators::attention has
/// it: 0 none, 1 causal, aligned to the bottom right of the score matrix, and a larger one causal
/// between blocks of that many keys counted from key 0 and whole within them.
///
/// The scores, their softmax and the output's accumulator are all float, so what half precision
/// cannot hold -- a score past 65504, or one that has to keep more than eleven bits -- never has to
/// be held in it. That is the difference from the composition in flint/functional.h, which writes
/// the score matrix out in the inputs' type.
///
/// The query heads that share a key-value head are taken together, nHead / nKvHead of them, so
/// that a decode step fills its CTAs and reads its keys once; and where the CTAs are still too few
/// for the card the keys are split between several, whose answers are put together after. Both
/// are worked out from the shapes. See attention.cu.
///
/// Returns false, with `out` untouched, for anything it was not built for: a type other than half,
/// a head dimension other than 32, 64 or 128, a last dimension that is not contiguous, operands not
/// aligned to sixteen bytes, a mask over fewer keys than queries, or a device before Ampere. The
/// caller then runs the composition, which has no block larger than one.
bool attention(
    const TensorView &q,
    const TensorView &k,
    const TensorView &v,
    int blockSize,
    const TensorView &out);

/// The same, with the keys split `splits` ways -- at most one way a block of 64 keys -- rather
/// than as many as the card wants; 0 is the card's choice. For the tests and the benchmark, which
/// have to reach every split however the card would have chosen.
bool attention(
    const TensorView &q,
    const TensorView &k,
    const TensorView &v,
    int blockSize,
    const TensorView &out,
    int splits);

}  // namespace cuda
}  // namespace op
}  // namespace fl
