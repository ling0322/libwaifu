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

// metal-cpp's own definitions -- the selectors and classes its headers name -- which exactly one
// source file in a program defines.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "flint/metal_native/metal_cpp.h"

namespace fl {
namespace op {
namespace metal_native {

AutoreleasePool::AutoreleasePool()
    : _pool(NS::AutoreleasePool::alloc()->init()) {
}

AutoreleasePool::~AutoreleasePool() {
  _pool->release();
}

std::string describe(NS::Error *error) {
  if (!error) return "no reason given";
  return error->localizedDescription()->utf8String();
}

}  // namespace metal_native
}  // namespace op
}  // namespace fl
