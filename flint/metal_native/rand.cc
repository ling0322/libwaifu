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
#include "flint/metal_native/common.h"
#include "flint/metal_native/ops.h"

namespace fl {
namespace op {
namespace metal_native {

namespace {

// In step with RandParams in kernels/rand.metal.
struct RandParams {
  uint64_t seed;
  uint64_t base;
  uint32_t numel;
  uint32_t normal;
};

}  // namespace

void Rand::draw(const TensorView &output, bool normal) {
  checkOutput(output, output.getShape(), DType::kFloat, normal ? "randNormal" : "rand");
  int64_t numel = output.getNumEl();
  if (numel == 0) return;
  int64_t numBlocks = (numel + 3) / 4;

  RandParams params{};
  params.seed = _seed;
  params.base = _position;
  params.numel = static_cast<uint32_t>(numel);
  params.normal = normal ? 1 : 0;
  getContext(output)->dispatch("rand_f32", {getArg(output)}, &params, sizeof(params), numBlocks);

  _position += static_cast<uint64_t>(numBlocks);
}

void Rand::uniform(const TensorView &out) {
  draw(out, false);
}

void Rand::normal(const TensorView &out) {
  draw(out, true);
}

void Rand::setSeed(uint64_t seed) {
  _seed = seed;
  _position = 0;
}

}  // namespace metal_native
}  // namespace op
}  // namespace fl
