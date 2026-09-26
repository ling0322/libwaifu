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
#include "lutil/strings.h"
#include "flint/vulkan/common.h"
#include "flint/vulkan/ops.h"

namespace fl {
namespace op {
namespace vulkan {

namespace {

struct LookupPush {
  uint64_t c;
  uint64_t table;
  uint64_t indices;
  uint32_t numel;
  uint32_t width;
};

struct UpsamplePush {
  uint64_t c;
  uint64_t a;
  uint32_t numel;
  uint32_t height;
  uint32_t width;
  uint32_t scale;
};

struct GluPush {
  uint64_t c;
  uint64_t a;
  uint32_t numel;
  uint32_t halfWidth;
  uint32_t op;
};

struct RotaryPush {
  uint64_t a;
  uint64_t cache;
  uint64_t positions;
  uint32_t numPairs;
  uint32_t heads;
  uint32_t headDim;
  uint32_t rowStride;
  uint32_t cacheStride;
};

Tensor glu(const Tensor &input, uint32_t op, const char *name) {
  checkFloat(input.getDType(), name);
  if (input.getDim() < 1 || input.getShape(-1) % 2 != 0) {
    throw lut::InvalidArgError(lut::sprintf("%s needs an even last dimension", name));
  }

  Tensor x = makeContiguous(input);
  std::vector<int> shape = x.getShape();
  shape.back() /= 2;
  Tensor output = createTensor(shape, x.getDType());

  GluPush push{};
  push.c = getAddress(output);
  push.a = getAddress(x);
  push.numel = static_cast<uint32_t>(output.getNumEl());
  push.halfWidth = static_cast<uint32_t>(shape.back());
  push.op = op;
  getContext(x)->dispatchLinear(
      kernelName("glu", x.getDType()).c_str(),
      &push,
      sizeof(push),
      push.numel);
  return output;
}

void rotate(const Tensor &positions, const Tensor &x, const Tensor &cache) {
  if (x.getDim() != 3 || x.getStride(2) != 1 || x.getStride(1) != x.getShape(2)) {
    throw lut::InvalidArgError(
        "rotaryEmbedding takes (T, heads, headDim) with each row's heads contiguous");
  }
  if (x.getDType() != cache.getDType()) {
    throw lut::InvalidArgError("rotaryEmbedding: the cache and the tensor differ in dtype");
  }

  int headDim = x.getShape(2);
  RotaryPush push{};
  push.a = getAddress(x);
  push.cache = getAddress(cache);
  push.positions = getAddress(positions);
  push.numPairs = static_cast<uint32_t>(x.getNumEl() / 2);
  push.heads = static_cast<uint32_t>(x.getShape(1));
  push.headDim = static_cast<uint32_t>(headDim);
  push.rowStride = static_cast<uint32_t>(x.getStride(0));
  push.cacheStride = static_cast<uint32_t>(cache.getStride(0));
  getContext(x)->dispatchLinear(
      kernelName("rotary", x.getDType()).c_str(),
      &push,
      sizeof(push),
      push.numPairs);
}

}  // namespace

Tensor lookup(const Tensor &table, const Tensor &indices) {
  if (indices.getDType() != DType::kLong) throw lut::InvalidArgError("lookup takes int64 indices");
  if (table.getDim() != 2) throw lut::InvalidArgError("lookup takes a table of (rows, width)");
  checkFloat(table.getDType(), "lookup");

  Tensor t = makeContiguous(table);
  Tensor ids = makeContiguous(indices);
  std::vector<int> shape = ids.getShape();
  shape.push_back(t.getShape(1));
  Tensor output = createTensor(shape, t.getDType());
  if (output.getNumEl() == 0) return output;

  LookupPush push{};
  push.c = getAddress(output);
  push.table = getAddress(t);
  push.indices = getAddress(ids);
  push.numel = static_cast<uint32_t>(output.getNumEl());
  push.width = static_cast<uint32_t>(t.getShape(1));
  getContext(t)->dispatchLinear(
      kernelName("lookup", t.getDType()).c_str(),
      &push,
      sizeof(push),
      push.numel);
  return output;
}

Tensor upsampleNearest2d(const Tensor &input, int scale) {
  checkFloat(input.getDType(), "upsampleNearest2d");
  if (input.getDim() != 4) throw lut::InvalidArgError("upsampleNearest2d takes (N, C, H, W)");
  if (scale < 1) throw lut::InvalidArgError("upsampleNearest2d: the scale must be positive");

  Tensor x = makeContiguous(input);
  int height = x.getShape(2);
  int width = x.getShape(3);
  Tensor output = createTensor(
      {x.getShape(0), x.getShape(1), height * scale, width * scale},
      x.getDType());
  if (output.getNumEl() == 0) return output;

  UpsamplePush push{};
  push.c = getAddress(output);
  push.a = getAddress(x);
  push.numel = static_cast<uint32_t>(output.getNumEl());
  push.height = static_cast<uint32_t>(height);
  push.width = static_cast<uint32_t>(width);
  push.scale = static_cast<uint32_t>(scale);
  getContext(x)->dispatchLinear(
      kernelName("upsample", x.getDType()).c_str(),
      &push,
      sizeof(push),
      push.numel);
  return output;
}

Tensor geglu(const Tensor &input) {
  return glu(input, 1, "geglu");
}

Tensor swiglu(const Tensor &input) {
  return glu(input, 0, "swiglu");
}

void rotaryEmbedding(
    const Tensor &positions,
    const Tensor &query,
    const Tensor &key,
    const Tensor &rotaryCache) {
  if (positions.getDType() != DType::kLong || positions.getDim() != 1) {
    throw lut::InvalidArgError("rotaryEmbedding takes a vector of int64 positions");
  }
  if (rotaryCache.getDim() != 2 || rotaryCache.getStride(1) != 1) {
    throw lut::InvalidArgError("rotaryEmbedding takes a cache of (positions, 2 * headDim)");
  }
  checkFloat(query.getDType(), "rotaryEmbedding");
  if (positions.getShape(0) == 0) return;

  Tensor ids = makeContiguous(positions);
  rotate(ids, query, rotaryCache);
  rotate(ids, key, rotaryCache);
}

}  // namespace vulkan
}  // namespace op
}  // namespace fl
