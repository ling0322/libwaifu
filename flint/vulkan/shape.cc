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

#include <vector>

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

struct Upsample1dPush {
  uint64_t c;
  uint64_t a;
  uint32_t numel;
  uint32_t width;
  uint32_t size;
  float scale;
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

void glu(const TensorView &input, uint32_t op, const char *name, const TensorView &output) {
  checkFloat(input.getDType(), name);
  if (input.getDim() < 1 || input.getShape(-1) % 2 != 0) {
    throw lut::InvalidArgError(lut::sprintf("%s needs an even last dimension", name));
  }

  std::vector<int> shape = input.getShape();
  shape.back() /= 2;
  checkOutput(output, shape, input.getDType(), name);
  if (output.getNumEl() == 0) return;

  Tensor keep;
  TensorView x = makeContiguous(input, &keep);
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
}

void rotate(const TensorView &positions, const TensorView &x, const TensorView &cache) {
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

void lookup(const TensorView &table, const TensorView &indices, const TensorView &output) {
  if (indices.getDType() != DType::kLong) throw lut::InvalidArgError("lookup takes int64 indices");
  if (table.getDim() != 2) throw lut::InvalidArgError("lookup takes a table of (rows, width)");
  checkFloat(table.getDType(), "lookup");

  std::vector<int> shape = indices.getShape();
  shape.push_back(table.getShape(1));
  checkOutput(output, shape, table.getDType(), "lookup");
  if (output.getNumEl() == 0) return;

  Tensor keepTable, keepIndices;
  TensorView t = makeContiguous(table, &keepTable);
  TensorView ids = makeContiguous(indices, &keepIndices);
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
}

void upsampleNearest2d(const TensorView &input, int scale, const TensorView &output) {
  checkFloat(input.getDType(), "upsampleNearest2d");
  if (input.getDim() != 4) throw lut::InvalidArgError("upsampleNearest2d takes (N, C, H, W)");
  if (scale < 1) throw lut::InvalidArgError("upsampleNearest2d: the scale must be positive");

  int height = input.getShape(2);
  int width = input.getShape(3);
  checkOutput(
      output,
      {input.getShape(0), input.getShape(1), height * scale, width * scale},
      input.getDType(),
      "upsampleNearest2d");
  if (output.getNumEl() == 0) return;

  Tensor keep;
  TensorView x = makeContiguous(input, &keep);
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
}

void upsampleNearest1d(const TensorView &input, const TensorView &output) {
  checkFloat(input.getDType(), "upsampleNearest1d");
  if (input.getDim() < 1) {
    throw lut::InvalidArgError("upsampleNearest1d takes at least one dimension");
  }
  if (output.getDim() < 1) throw lut::InvalidArgError("upsampleNearest1d: the output is a scalar");
  int size = output.getShape(-1);
  if (size < 1) throw lut::InvalidArgError("upsampleNearest1d: the size must be positive");
  if (input.getShape(-1) < 1) throw lut::InvalidArgError("upsampleNearest1d: the input is empty");

  int length = input.getShape(-1);
  std::vector<int> shape = input.getShape();
  shape.back() = size;
  checkOutput(output, shape, input.getDType(), "upsampleNearest1d");
  if (output.getNumEl() == 0) return;

  Tensor keep;
  TensorView x = makeContiguous(input, &keep);
  Upsample1dPush push{};
  push.c = getAddress(output);
  push.a = getAddress(x);
  push.numel = static_cast<uint32_t>(output.getNumEl());
  push.width = static_cast<uint32_t>(length);
  push.size = static_cast<uint32_t>(size);
  push.scale = static_cast<float>(length) / static_cast<float>(size);
  getContext(x)->dispatchLinear(
      kernelName("upsample1d", x.getDType()).c_str(),
      &push,
      sizeof(push),
      push.numel);
}

void geglu(const TensorView &input, const TensorView &out) {
  glu(input, 1, "geglu", out);
}

void swiglu(const TensorView &input, const TensorView &out) {
  glu(input, 0, "swiglu", out);
}

void rotaryEmbedding(
    const TensorView &positions,
    const TensorView &query,
    const TensorView &key,
    const TensorView &rotaryCache) {
  if (positions.getDType() != DType::kLong || positions.getDim() != 1) {
    throw lut::InvalidArgError("rotaryEmbedding takes a vector of int64 positions");
  }
  if (rotaryCache.getDim() != 2 || rotaryCache.getStride(1) != 1) {
    throw lut::InvalidArgError("rotaryEmbedding takes a cache of (positions, 2 * headDim)");
  }
  checkFloat(query.getDType(), "rotaryEmbedding");
  if (positions.getShape(0) == 0) return;

  Tensor keep;
  TensorView ids = makeContiguous(positions, &keep);
  rotate(ids, query, rotaryCache);
  rotate(ids, key, rotaryCache);
}

}  // namespace vulkan
}  // namespace op
}  // namespace fl
