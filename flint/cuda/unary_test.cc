// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cmath>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/device.h"
#include "flint/operators.h"

namespace fl {

namespace {

/// The CUDA operators, which the calls here that run on CUDA are asked of.
Operators *cudaOps() {
  return getOperators(Device::kCuda);
}

/// The CPU operators, which the calls here that run on CPU are asked of.
Operators *cpuOps() {
  return getOperators(Device::kCpu);
}

Tensor toCuda(const Tensor &a) {
  return cudaOps()->cast(cudaOps()->toDevice(Device::getCuda(), a), DType::kFloat16);
}

Tensor toCpu(const Tensor &a) {
  return cudaOps()->toDevice(Device::getCpu(), cudaOps()->cast(a, DType::kFloat));
}

using UnaryFn = Tensor (Operators::*)(Tensor);

struct Case {
  const char *name;
  UnaryFn fn;
};

/// Every element-wise function, so a new one added to the enum without a CUDA arm shows up here
/// rather than at a call site.
const std::vector<Case> &allUnary() {
  static const std::vector<Case> cases = {
      {"neg", &Operators::neg},
      {"abs", &Operators::abs},
      {"exp", &Operators::exp},
      {"square", &Operators::square},
      {"sigmoid", &Operators::sigmoid},
      {"tanh", &Operators::tanh},
      {"relu", &Operators::relu},
      {"gelu", &Operators::gelu},
      {"silu", &Operators::silu},
      {"sin", &Operators::sin},
      {"cos", &Operators::cos},
      {"quickGelu", &Operators::quickGelu},
  };
  return cases;
}

}  // namespace

CATCH_TEST_CASE("test CUDA unary operators", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Values spread either side of zero, including magnitudes where the saturating functions have
  // flattened. exp(8) is still inside the half range; much beyond that it would overflow.
  Tensor a = Tensor::create<float>(
      {2, 5},
      {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.5f, -2.5f, 8.0f, -8.0f, 0.125f});
  Tensor x = toCuda(a);

  for (const Case &c : allUnary()) {
    CATCH_INFO("op = " << c.name);
    CATCH_REQUIRE(cpuOps()->allClose(
        toCpu((cudaOps()->*c.fn)(x)),
        (cpuOps()->*c.fn)(a),
        5e-3,
        5e-3));
  }
}

CATCH_TEST_CASE("test CUDA unary operators (positive domain)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // sqrt and rsqrt need a non-negative input, so they get their own values.
  Tensor a = Tensor::create<float>({5}, {0.25f, 1.0f, 2.0f, 9.0f, 0.0625f});
  Tensor x = toCuda(a);

  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(cudaOps()->sqrt(x)), cpuOps()->sqrt(a), 5e-3, 5e-3));
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(cudaOps()->rsqrt(x)), cpuOps()->rsqrt(a), 5e-3, 5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(toCpu(cudaOps()->log(x)), cpuOps()->log(a), 5e-3, 5e-3));
}

CATCH_TEST_CASE("test CUDA log", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Float32 across the range a mel energy covers -- from the 1e-10 floor Whisper clamps to up to
  // thousands -- against the CPU, to float32's precision rather than half's.
  std::vector<float> values;
  for (int i = 0; i < 300; ++i) values.push_back(std::pow(10.0f, -10.0f + i * 0.045f));
  Tensor a = Tensor::create<float>({3, 100}, values);
  Tensor got = cudaOps()->log(cudaOps()->toDevice(Device::getCuda(), a));
  CATCH_REQUIRE(got.getDType() == DType::kFloat);
  CATCH_REQUIRE(cpuOps()->allClose(
      cudaOps()->toDevice(Device::getCpu(), got), cpuOps()->log(a), 1e-6, 1e-5));

  // A strided view, and log(0) = -inf, as on the CPU.
  Tensor strided = cudaOps()->log(cudaOps()->toDevice(Device::getCuda(), a).transpose(0, 1));
  CATCH_REQUIRE(cpuOps()->allClose(
      cudaOps()->toDevice(Device::getCpu(), cudaOps()->contiguous(strided)),
      cpuOps()->log(cpuOps()->contiguous(a.transpose(0, 1))),
      1e-6,
      1e-5));
  Tensor zero = cudaOps()->log(cudaOps()->toDevice(Device::getCuda(), Tensor::create<float>({1}, {0.0f})));
  float value = cudaOps()->elem(zero);
  CATCH_REQUIRE(std::isinf(value));
  CATCH_REQUIRE(value < 0.0f);
}

CATCH_TEST_CASE("test CUDA unary operators (shapes and strides)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The kernel has a packed arm and an accessor arm per rank; walk both, and a size that makes
  // every thread go round the grid-stride loop more than once.
  for (std::vector<int> shape : std::vector<std::vector<int>>{
           {1},
           {17},
           {4, 5},
           {2, 3, 4},
           {2, 3, 4, 5},
           {256, 1024}}) {
    Tensor a = cpuOps()->rand(shape, DType::kFloat);
    CATCH_INFO("shape rank = " << shape.size());
    CATCH_REQUIRE(
        cpuOps()->allClose(toCpu(cudaOps()->neg(toCuda(a))), cpuOps()->neg(a), 5e-3, 5e-3));
    CATCH_REQUIRE(
        cpuOps()->allClose(toCpu(cudaOps()->silu(toCuda(a))), cpuOps()->silu(a), 5e-3, 5e-3));
  }

  // a strided view has to be read through its strides, not as a flat buffer.
  Tensor a = cpuOps()->rand({2, 3, 4}, DType::kFloat);
  Tensor strided = toCuda(a).transpose(0, 2);
  CATCH_REQUIRE(!strided.isContiguous());
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(cudaOps()->neg(strided)),
      cpuOps()->neg(a.transpose(0, 2)),
      5e-3,
      5e-3));
}

CATCH_TEST_CASE("test CUDA unary operators (float tensors)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The kernels are instantiated for float as well as half; moving without a cast selects it.
  Tensor a = cpuOps()->rand({3, 8}, DType::kFloat);
  Tensor x = cudaOps()->toDevice(Device::getCuda(), a);
  CATCH_REQUIRE(x.getDType() == DType::kFloat);

  for (const Case &c : allUnary()) {
    Tensor actual = (cudaOps()->*c.fn)(x);
    CATCH_INFO("op = " << c.name);
    CATCH_REQUIRE(actual.getDType() == DType::kFloat);
    CATCH_REQUIRE(cpuOps()->allClose(
        cudaOps()->toDevice(Device::getCpu(), actual),
        (cpuOps()->*c.fn)(a),
        1e-4,
        1e-5));
  }
}

CATCH_TEST_CASE("test CUDA unary operators (known values)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Fixed points that separate the activations from one another: at x = 0 relu, gelu and silu all
  // give 0 while sigmoid gives 0.5, and a large negative input drives all three to 0.
  Tensor a = Tensor::create<float>({3}, {0.0f, -10.0f, 1.0f});
  Tensor x = toCuda(a);

  Tensor reluOut = toCpu(cudaOps()->relu(x));
  const float *relu = reluOut.getInternalData()->getData<float>(reluOut.getInternalOffset());
  CATCH_REQUIRE(relu[0] == 0.0f);
  CATCH_REQUIRE(relu[1] == 0.0f);
  CATCH_REQUIRE(std::fabs(relu[2] - 1.0f) < 1e-2f);

  Tensor sigmoidOut = toCpu(cudaOps()->sigmoid(x));
  const float *sigmoid =
      sigmoidOut.getInternalData()->getData<float>(sigmoidOut.getInternalOffset());
  CATCH_REQUIRE(std::fabs(sigmoid[0] - 0.5f) < 1e-2f);
  CATCH_REQUIRE(sigmoid[1] < 1e-2f);

  // gelu and silu both pass through the origin, unlike sigmoid.
  Tensor geluOut = toCpu(cudaOps()->gelu(x));
  Tensor siluOut = toCpu(cudaOps()->silu(x));
  CATCH_REQUIRE(geluOut.getInternalData()->getData<float>(geluOut.getInternalOffset())[0] == 0.0f);
  CATCH_REQUIRE(siluOut.getInternalData()->getData<float>(siluOut.getInternalOffset())[0] == 0.0f);
}

CATCH_TEST_CASE("test CUDA div (element-wise)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = Tensor::create<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor b = Tensor::create<float>({2, 3}, {2.0f, 4.0f, 4.0f, 8.0f, 10.0f, 3.0f});
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(cudaOps()->divTensor(toCuda(a), toCuda(b))),
      cpuOps()->divTensor(a, b),
      5e-3,
      5e-3));

  // the divisor broadcasts over the leading dimensions, which takes the strided kernel.
  Tensor row = Tensor::create<float>({3}, {1.0f, 2.0f, 4.0f});
  CATCH_REQUIRE(cpuOps()->allClose(
      toCpu(cudaOps()->divTensor(toCuda(a), toCuda(row))),
      cpuOps()->divTensor(a, row),
      5e-3,
      5e-3));

  // dividing by itself is 1 everywhere, which a kernel that dropped the divisor would not give.
  Tensor ones = toCpu(cudaOps()->divTensor(toCuda(a), toCuda(a)));
  Tensor expected = cpuOps()->tensor({2, 3}, DType::kFloat);
  cpuOps()->fill(expected, 1.0f);
  CATCH_REQUIRE(cpuOps()->allClose(ones, expected, 5e-3, 5e-3));
}

CATCH_TEST_CASE("test CUDA min", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // An all-positive row would still look right if min returned the initial +inf only when every
  // element is larger, so include a row that is entirely negative.
  Tensor a = Tensor::create<float>({2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f});
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(cudaOps()->min(toCuda(a))), cpuOps()->min(a), 5e-3, 5e-3));
  CATCH_REQUIRE(
      cpuOps()->allClose(toCpu(cudaOps()->max(toCuda(a))), cpuOps()->max(a), 5e-3, 5e-3));

  // widths either side of the block size, where the reduction loops.
  for (int width : {1, 255, 256, 257, 1000}) {
    Tensor b = cpuOps()->rand({2, 3, width}, DType::kFloat);
    CATCH_INFO("width = " << width);
    CATCH_REQUIRE(
        cpuOps()->allClose(toCpu(cudaOps()->min(toCuda(b))), cpuOps()->min(b), 5e-3, 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA round and cast to int64", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Ties to even, then truncated to int64: the two steps the speech tokenizer's quantizer takes
  // from a projection to a token id. Half holds every value here exactly.
  std::vector<float> in = {-2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 0.4f, 0.6f, -0.6f, 3.7f};
  std::vector<float> rounded = {-2.0f, -2.0f, 0.0f, 0.0f, 2.0f, 2.0f, 0.0f, 1.0f, -1.0f, 4.0f};
  std::vector<float> truncated = {-2.0f, -1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 0.0f, 3.0f};
  Tensor x = Tensor::create<float>({2, 5}, in);

  for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
    CATCH_INFO("from " << dtype.toString());
    Tensor onCard = cudaOps()->cast(cudaOps()->toDevice(Device::getCuda(), x), dtype);

    // Exact, element by element: allClose compares with a strict `<`, so no tolerance says "equal".
    Tensor r = cudaOps()->round(onCard);
    CATCH_REQUIRE(r.getDType() == dtype);
    Tensor back = toCpu(r);
    const float *values = back.getInternalData()->getData<float>(back.getInternalOffset());
    for (size_t i = 0; i < in.size(); ++i) {
      CATCH_INFO("round(" << in[i] << ")");
      CATCH_REQUIRE(values[i] == rounded[i]);
    }

    auto asLongs = [](const Tensor &ids) {
      Tensor host = cudaOps()->toDevice(Device::getCpu(), ids);
      const LongType *data = host.getInternalData()->getData<LongType>(host.getInternalOffset());
      return std::vector<LongType>(data, data + host.getNumEl());
    };

    Tensor ids = cudaOps()->cast(r, DType::kLong);
    CATCH_REQUIRE(ids.getDType() == DType::kLong);
    CATCH_REQUIRE(ids.getShape() == std::vector<int>{2, 5});
    std::vector<LongType> got = asLongs(ids);
    for (size_t i = 0; i < in.size(); ++i) CATCH_REQUIRE(got[i] == LongType(rounded[i]));

    // Cast alone truncates toward zero.
    got = asLongs(cudaOps()->cast(onCard, DType::kLong));
    for (size_t i = 0; i < in.size(); ++i) {
      CATCH_INFO(in[i]);
      CATCH_REQUIRE(got[i] == LongType(truncated[i]));
    }
  }

  // Ids off the card go straight into lookup, which is what the cast is for.
  Tensor table = cudaOps()->toDevice(
      Device::getCuda(), Tensor::create<float>({3, 2}, {0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f}));
  Tensor idsFloat = cudaOps()->toDevice(Device::getCuda(), Tensor::create<float>({2}, {2.4f, 0.6f}));
  Tensor rows = cudaOps()->lookup(table, cudaOps()->cast(cudaOps()->round(idsFloat), DType::kLong));
  CATCH_REQUIRE(cpuOps()->allClose(
      cudaOps()->toDevice(Device::getCpu(), rows),
      Tensor::create<float>({2, 2}, {2.0f, 2.0f, 1.0f, 1.0f})));
}

}  // namespace fl
