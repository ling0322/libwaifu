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

#include <stddef.h>

namespace fl {
namespace op {
namespace metal {

/// @brief Hand MLX the mlx.metallib carried inside this binary's __TEXT,__metallib section,
///        the way a CUDA build carries its fatbin.
/// Idempotent, but only effective before the first MLX operation: MLX builds its default
/// library once, when the Metal device is first constructed, and caches it from then on.
/// @throw lut::AbortedError if the binary was linked without the section.
void useEmbeddedMetallib();

/// @brief Size in bytes of the embedded mlx.metallib, or 0 if there is no such section.
/// Exposed so a caller can report the cost rather than having to parse the Mach-O itself.
size_t getEmbeddedMetallibSize();

}  // namespace metal
}  // namespace op
}  // namespace fl
