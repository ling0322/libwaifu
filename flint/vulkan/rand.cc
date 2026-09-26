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

#include "lutil/error.h"
#include "lutil/log.h"
#include "flint/vulkan/common.h"
#include "flint/vulkan/ops.h"

namespace fl {
namespace op {
namespace vulkan {

namespace {

struct RandPush {
  uint64_t c;
  uint64_t seed;
  uint64_t base;
  uint32_t numel;
  uint32_t normal;
};

}  // namespace

Tensor Rand::draw(lut::Span<const int> shape, bool normal) {
  Tensor output = createTensor(shape, DType::kFloat);
  int64_t numel = output.getNumEl();
  int64_t numBlocks = (numel + 3) / 4;

  RandPush push{};
  push.c = getAddress(output);
  push.seed = _seed;
  push.base = _position;
  push.numel = static_cast<uint32_t>(numel);
  push.normal = normal ? 1 : 0;
  getContext(output)->dispatchLinear("rand_f32", &push, sizeof(push), numBlocks);

  _position += static_cast<uint64_t>(numBlocks);
  return output;
}

Tensor Rand::uniform(lut::Span<const int> shape) {
  return draw(shape, false);
}

Tensor Rand::normal(lut::Span<const int> shape) {
  return draw(shape, true);
}

void Rand::setSeed(uint64_t seed) {
  _seed = seed;
  _position = 0;
}

}  // namespace vulkan
}  // namespace op
}  // namespace fl
