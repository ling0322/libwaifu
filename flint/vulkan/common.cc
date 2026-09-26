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

#include "flint/vulkan/common.h"

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "flint/vulkan/ops.h"
#include "flint/vulkan/vulkan_tensor_data.h"

namespace fl {
namespace op {
namespace vulkan {

namespace {

const VulkanTensorData *getData(const Tensor &tensor) {
  const VulkanTensorData *data = dynamic_cast<const VulkanTensorData *>(
      tensor.getInternalData().get());
  if (!data) {
    throw lut::AbortedError(lut::sprintf(
        "expected a tensor on the Vulkan device, got one on %s",
        tensor.getDevice().getName()));
  }
  return data;
}

}  // namespace

Tensor createTensor(lut::Span<const int> shape, DType dtype) {
  auto tensorShape = std::make_shared<TensorShape>(shape);
  int64_t numel = tensorShape->getNumEl();

  // A tensor with nothing in it still needs somewhere to be, so it gets the smallest buffer.
  std::shared_ptr<TensorData> data = VulkanTensorData::create(std::max<int64_t>(numel, 1), dtype);
  return Tensor::create(tensorShape, data);
}

Context *getContext(const Tensor &tensor) {
  return getData(tensor)->getContext();
}

const Buffer &getBuffer(const Tensor &tensor) {
  return getData(tensor)->getBuffer();
}

int64_t getByteOffset(const Tensor &tensor) {
  return tensor.getDType().getTotalSize(tensor.getInternalOffset());
}

uint64_t getAddress(const Tensor &tensor) {
  return getBuffer(tensor).address + static_cast<uint64_t>(getByteOffset(tensor));
}

std::string getTypeSuffix(DType dtype) {
  switch (dtype) {
    case DType::kFloat:
      return "f32";
    case DType::kFloat16:
      return "f16";
    case DType::kLong:
      return "i64";
    case DType::kInt32:
      return "i32";
    case DType::kUInt8:
      return "u8";
    case DType::kInt8:
      return "i8";
    case DType::kBool:
      return "bool";
    default:
      throw lut::NotImplementedError(
          lut::sprintf("the Vulkan device has no kernels for %s", dtype.toString()));
  }
}

std::string kernelName(const char *base, DType dtype) {
  return std::string(base) + "_" + getTypeSuffix(dtype);
}

void checkFloat(DType dtype, const char *op) {
  if (dtype != DType::kFloat && dtype != DType::kFloat16) {
    throw lut::InvalidArgError(lut::sprintf(
        "%s on the Vulkan device takes float32 or float16, not %s",
        op,
        dtype.toString()));
  }
}

std::vector<int> getStrides(const Tensor &tensor) {
  std::vector<int> strides;
  for (int d = 0; d < tensor.getDim(); ++d) strides.push_back(tensor.getStride(d));
  return strides;
}

bool collapse(
    const std::vector<int> &shape,
    std::initializer_list<std::vector<int>> strides,
    Layout *layout) {
  int numTensors = static_cast<int>(strides.size());
  CHECK(numTensors <= 3);

  // Dimensions of size one go first: they add nothing to an index.
  std::vector<int> dims;
  for (int d = 0; d < static_cast<int>(shape.size()); ++d) {
    if (shape[d] != 1) dims.push_back(d);
  }

  std::vector<int64_t> mergedShape;
  std::vector<std::vector<int64_t>> mergedStrides(numTensors);
  for (int d : dims) {
    // Merge into the dimension before when every tensor steps over this one exactly as far as
    // that one's stride says it should.
    bool merge = !mergedShape.empty();
    int t = 0;
    for (const std::vector<int> &stride : strides) {
      if (merge && mergedStrides[t].back() != static_cast<int64_t>(stride[d]) * shape[d]) {
        merge = false;
      }
      ++t;
    }

    t = 0;
    if (merge) {
      mergedShape.back() *= shape[d];
      for (const std::vector<int> &stride : strides) mergedStrides[t++].back() = stride[d];
    } else {
      mergedShape.push_back(shape[d]);
      for (const std::vector<int> &stride : strides) mergedStrides[t++].push_back(stride[d]);
    }
  }

  if (mergedShape.size() > kMaxDims) return false;

  // A tensor of one element is still one dimension of one.
  if (mergedShape.empty()) {
    mergedShape.push_back(1);
    for (int t = 0; t < numTensors; ++t) mergedStrides[t].push_back(0);
  }

  layout->ndim = static_cast<int>(mergedShape.size());
  for (int d = 0; d < kMaxDims; ++d) {
    bool used = d < layout->ndim;
    layout->shape[d] = used ? static_cast<uint32_t>(mergedShape[d]) : 1;
    for (int t = 0; t < 3; ++t) {
      layout->strides[t][d] = (used && t < numTensors) ? static_cast<uint32_t>(
                                                             mergedStrides[t][d])
                                                       : 0;
    }
  }
  return true;
}

Tensor makeContiguous(const Tensor &tensor) {
  if (tensor.isContiguous()) return tensor;

  Tensor output = createTensor(tensor.getShape(), tensor.getDType());
  copy(tensor, output);
  return output;
}

}  // namespace vulkan
}  // namespace op
}  // namespace fl
