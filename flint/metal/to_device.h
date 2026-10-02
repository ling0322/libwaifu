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

#pragma once

#include "flint/tensor_view.h"

namespace fl {
namespace op {
namespace metal {

/// @brief Copy contiguous `src` into contiguous `dest`, one on the CPU and the other on Metal,
/// both of one dtype and one number of elements.
///
/// Both sides address the same unified memory, so this is a memcpy rather than the staged
/// transfer a discrete GPU needs. It still copies: the two devices own their buffers separately,
/// and sharing one would make a CPU write visible to a tensor nobody expected to change.
/// Exactly `src.getNumEl()` elements move, from `src`'s offset to `dest`'s.
void transfer(const TensorView &src, const TensorView &dest);

}  // namespace metal
}  // namespace op
}  // namespace fl
