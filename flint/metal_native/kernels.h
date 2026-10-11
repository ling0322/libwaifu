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

#include <stddef.h>

namespace fl {
namespace op {
namespace metal_native {

/// Every kernel in kernels/, compiled into one Metal library when flint was built and carried in
/// it as bytes, so that whatever links flint has its kernels with no file to find at run time.
/// Defined in a source file the build writes, see kernels/kernels.cmake.
extern const unsigned char kMetallib[];
extern const size_t kMetallibSize;

/// The kernels in kernels/nax/, the same way: those that run on the neural accelerators of the
/// GPUs that have them, in Metal 4, for macOS 26.2 and later. No bytes when the toolchain flint
/// was built with has no Metal 4.
extern const unsigned char kNaxMetallib[];
extern const size_t kNaxMetallibSize;

}  // namespace metal_native
}  // namespace op
}  // namespace fl
