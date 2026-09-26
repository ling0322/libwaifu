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
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {

namespace {

/// The CPU operators, which is what the calls in this file are asked of.
Operators *cpuOps() {
  return getOperators(Device::kCpu);
}

}  // namespace

namespace op {
namespace cpu {

namespace {

/// The values each function is checked at: zero, both signs, and magnitudes far enough out that a
/// saturating function has flattened.
const std::vector<float> &probes() {
  static const std::vector<float> values = {
      0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.5f, -2.5f, 8.0f, -8.0f, 0.125f};
  return values;
}

Tensor probeTensor() {
  return Tensor::create<float>({static_cast<int>(probes().size())}, probes());
}

void checkAgainst(Tensor actual, float (*reference)(float), const char *name) {
  const float *data = actual.getInternalData()->getData<float>(actual.getInternalOffset());
  for (size_t i = 0; i < probes().size(); ++i) {
    CATCH_INFO(name << " at x = " << probes()[i]);
    CATCH_REQUIRE(std::fabs(data[i] - reference(probes()[i])) < 1e-5f);
  }
}

}  // namespace

CATCH_TEST_CASE("test CPU unary operators", "[core][nn][operators]") {
  Tensor x = probeTensor();

  checkAgainst(cpuOps()->neg(x), [](float v) { return -v; }, "neg");
  checkAgainst(cpuOps()->abs(x), [](float v) { return std::fabs(v); }, "abs");
  checkAgainst(cpuOps()->exp(x), [](float v) { return std::exp(v); }, "exp");
  checkAgainst(cpuOps()->square(x), [](float v) { return v * v; }, "square");
  checkAgainst(cpuOps()->tanh(x), [](float v) { return std::tanh(v); }, "tanh");
  checkAgainst(cpuOps()->relu(x), [](float v) { return v > 0.0f ? v : 0.0f; }, "relu");
  checkAgainst(
      cpuOps()->sigmoid(x),
      [](float v) { return 1.0f / (1.0f + std::exp(-v)); },
      "sigmoid");
  checkAgainst(
      cpuOps()->silu(x),
      [](float v) { return v / (1.0f + std::exp(-v)); },
      "silu");
  checkAgainst(
      cpuOps()->gelu(x),
      [](float v) { return v * 0.5f * (1.0f + std::erf(v * 0.70710678118654752f)); },
      "gelu");
  checkAgainst(cpuOps()->sin(x), [](float v) { return std::sin(v); }, "sin");
  checkAgainst(cpuOps()->cos(x), [](float v) { return std::cos(v); }, "cos");
  checkAgainst(
      cpuOps()->quickGelu(x),
      [](float v) { return v / (1.0f + std::exp(-1.702f * v)); },
      "quickGelu");
}

CATCH_TEST_CASE("test CPU unary operators (positive domain)", "[core][nn][operators]") {
  // sqrt and rsqrt are only defined for non-negative and positive inputs, so they get their own
  // probes rather than the shared set.
  std::vector<float> values = {0.25f, 1.0f, 2.0f, 9.0f, 1e-4f};
  Tensor x = Tensor::create<float>({static_cast<int>(values.size())}, values);

  Tensor rootTensor = cpuOps()->sqrt(x);
  Tensor invRootTensor = cpuOps()->rsqrt(x);
  const float *root = rootTensor.getInternalData()->getData<float>(rootTensor.getInternalOffset());
  const float *invRoot =
      invRootTensor.getInternalData()->getData<float>(invRootTensor.getInternalOffset());

  for (size_t i = 0; i < values.size(); ++i) {
    CATCH_INFO("x = " << values[i]);
    CATCH_REQUIRE(std::fabs(root[i] - std::sqrt(values[i])) < 1e-5f);
    CATCH_REQUIRE(std::fabs(invRoot[i] - 1.0f / std::sqrt(values[i])) < 1e-3f);
  }

  Tensor logTensor = cpuOps()->log(x);
  const float *logged = logTensor.getInternalData()->getData<float>(logTensor.getInternalOffset());
  for (size_t i = 0; i < values.size(); ++i) {
    CATCH_INFO("log at x = " << values[i]);
    CATCH_REQUIRE(std::fabs(logged[i] - std::log(values[i])) < 1e-5f);
  }

  // The edges a spectrogram floors its energies to stay away from: log(0) is -inf, and a negative
  // number is NaN rather than anything that could pass for a value.
  Tensor edges = cpuOps()->log(Tensor::create<float>({2}, {0.0f, -1.0f}));
  const float *edge = edges.getInternalData()->getData<float>(edges.getInternalOffset());
  CATCH_REQUIRE(std::isinf(edge[0]));
  CATCH_REQUIRE(edge[0] < 0.0f);
  CATCH_REQUIRE(std::isnan(edge[1]));

  // sqrt(0) is 0 rather than NaN. Read the value directly: elem() has no CPU implementation.
  Tensor zero = cpuOps()->sqrt(Tensor::create<float>({1}, {0.0f}));
  CATCH_REQUIRE(zero.getInternalData()->getData<float>(zero.getInternalOffset())[0] == 0.0f);
}

CATCH_TEST_CASE("test CPU unary operators (shapes)", "[core][nn][operators]") {
  // The kernel walks the tensor a row at a time, so a shape has to survive unchanged and every
  // element has to be visited, including in a strided view.
  Tensor a = cpuOps()->rand({2, 3, 4}, DType::kFloat);
  Tensor negated = cpuOps()->neg(a);
  CATCH_REQUIRE(negated.getShape() == std::vector<int>{2, 3, 4});
  CATCH_REQUIRE(
      cpuOps()->allClose(cpuOps()->add(a, negated), cpuOps()->zeros({2, 3, 4}, DType::kFloat)));

  Tensor strided = a.transpose(0, 2);
  Tensor stridedNeg = cpuOps()->neg(strided);
  CATCH_REQUIRE(stridedNeg.getShape() == std::vector<int>{4, 3, 2});
  CATCH_REQUIRE(cpuOps()->allClose(
      cpuOps()->add(strided, stridedNeg),
      cpuOps()->zeros({4, 3, 2}, DType::kFloat)));

  // applying a function twice is not the same as applying it once, so a no-op kernel fails here.
  CATCH_REQUIRE(cpuOps()->allClose(cpuOps()->neg(negated), a));
}

CATCH_TEST_CASE("test CPU div and min", "[core][nn][operators]") {
  Tensor a = Tensor::create<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor b = Tensor::create<float>({2, 3}, {2.0f, 4.0f, 4.0f, 8.0f, 10.0f, 3.0f});

  CATCH_REQUIRE(cpuOps()->allClose(
      cpuOps()->divTensor(a, b),
      Tensor::create<float>({2, 3}, {0.5f, 0.5f, 0.75f, 0.5f, 0.5f, 2.0f})));

  // the divisor is broadcast over the leading dimensions the way mul does it.
  Tensor row = Tensor::create<float>({3}, {1.0f, 2.0f, 4.0f});
  CATCH_REQUIRE(cpuOps()->allClose(
      cpuOps()->divTensor(a, row),
      Tensor::create<float>({2, 3}, {1.0f, 1.0f, 0.75f, 4.0f, 2.5f, 1.5f})));

  // min drops the last dimension the way max does, and must not report the initial +inf.
  Tensor c = Tensor::create<float>({2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f});
  CATCH_REQUIRE(cpuOps()->allClose(cpuOps()->min(c), Tensor::create<float>({2}, {1.0f, -4.0f})));
  CATCH_REQUIRE(cpuOps()->allClose(cpuOps()->max(c), Tensor::create<float>({2}, {4.0f, -1.0f})));

  // a single-element row is both the min and the max.
  Tensor one = Tensor::create<float>({2, 1}, {7.0f, -7.0f});
  CATCH_REQUIRE(cpuOps()->allClose(cpuOps()->min(one), Tensor::create<float>({2}, {7.0f, -7.0f})));
}

CATCH_TEST_CASE("test CPU arange", "[core][nn][operators]") {
  Tensor x = cpuOps()->arangeLong(0, 10, 2);
  CATCH_REQUIRE(x.getShape() == std::vector<int>{5});
  CATCH_REQUIRE(x.getDType() == DType::kLong);

  const LongType *data = x.getInternalData()->getData<LongType>(x.getInternalOffset());
  for (int i = 0; i < 5; ++i) {
    CATCH_INFO("i = " << i);
    CATCH_REQUIRE(data[i] == 2 * i);
  }

  // a step that does not divide the span evenly stops short rather than overshooting, and a
  // negative step counts down.
  CATCH_REQUIRE(cpuOps()->arangeLong(0, 10, 3).getShape() == std::vector<int>{3});
  Tensor down = cpuOps()->arangeLong(10, 0, -2);
  CATCH_REQUIRE(down.getShape() == std::vector<int>{5});
  CATCH_REQUIRE(
      down.getInternalData()->getData<LongType>(down.getInternalOffset())[0] == 10);
}

CATCH_TEST_CASE("test CPU randn", "[core][nn][operators]") {
  // An odd element count is the case the pair-at-a-time Gaussian fill has to pad for.
  for (int count : {1, 2, 3, 4096}) {
    Tensor x = cpuOps()->randNormal({count});
    CATCH_INFO("count = " << count);
    CATCH_REQUIRE(x.getShape() == std::vector<int>{count});
    CATCH_REQUIRE(x.getDType() == DType::kFloat);
  }

  Tensor x = cpuOps()->randNormal({8192});
  const float *data = x.getInternalData()->getData<float>(x.getInternalOffset());
  double sum = 0.0;
  double sumSquare = 0.0;
  for (int i = 0; i < x.getNumEl(); ++i) {
    CATCH_REQUIRE(!std::isnan(data[i]));
    sum += data[i];
    sumSquare += static_cast<double>(data[i]) * data[i];
  }

  double mean = sum / x.getNumEl();
  double stddev = std::sqrt(sumSquare / x.getNumEl() - mean * mean);
  CATCH_REQUIRE(std::fabs(mean) < 0.1);
  CATCH_REQUIRE(std::fabs(stddev - 1.0) < 0.1);
}

CATCH_TEST_CASE("test CPU elem and scalar div", "[core][nn][operators]") {
  CATCH_REQUIRE(cpuOps()->elem(Tensor::create<float>({1}, {1.5f})) == 1.5f);
  CATCH_REQUIRE(cpuOps()->elem(Tensor::create<float>({1}, {0.0f})) == 0.0f);
  CATCH_REQUIRE(cpuOps()->elem(Tensor::create<float>({1}, {-2.5f})) == -2.5f);

  Tensor a = Tensor::create<float>({2, 2}, {1.0f, 2.0f, 4.0f, 8.0f});
  CATCH_REQUIRE(cpuOps()->allClose(
      cpuOps()->div(a, 2.0f),
      Tensor::create<float>({2, 2}, {0.5f, 1.0f, 2.0f, 4.0f})));
  // dividing by one leaves the tensor alone.
  CATCH_REQUIRE(cpuOps()->allClose(cpuOps()->div(a, 1.0f), a));
}

CATCH_TEST_CASE("test CPU mod", "[core][nn][operators]") {
  Tensor ids = Tensor::create<LongType>({2, 4}, {0, 3, 4, 5, 7, 8, 99, 100});
  Tensor x = cpuOps()->mod(ids, 4);

  const LongType *data = x.getInternalData()->getData<LongType>(x.getInternalOffset());
  const LongType expected[] = {0, 3, 0, 1, 3, 0, 3, 0};
  for (int i = 0; i < 8; ++i) {
    CATCH_INFO("i = " << i);
    CATCH_REQUIRE(data[i] == expected[i]);
  }
}

CATCH_TEST_CASE("test CPU eq and all", "[core][nn][operators]") {
  // Tensor::create is not instantiated for UInt8, so build these through the allocator and
  // write the bytes in.
  auto makeUInt8 = [](std::vector<uint8_t> values) {
    Tensor x = cpuOps()->tensor({static_cast<int>(values.size())}, DType::kUInt8);
    UInt8 *data = x.getInternalData()->getData<UInt8>(x.getInternalOffset());
    for (size_t i = 0; i < values.size(); ++i) data[i].v = values[i];
    return x;
  };

  Tensor a = makeUInt8({1, 2, 3, 4});
  Tensor same = makeUInt8({1, 2, 3, 4});
  Tensor different = makeUInt8({1, 2, 9, 4});

  CATCH_REQUIRE(cpuOps()->all(cpuOps()->eq(a, same)));
  CATCH_REQUIRE(!cpuOps()->all(cpuOps()->eq(a, different)));

  // eq answers per element, so the one mismatch has to be the only false.
  Tensor mask = cpuOps()->eq(a, different);
  CATCH_REQUIRE(mask.getDType() == DType::kBool);
  const BoolType *data = mask.getInternalData()->getData<BoolType>(mask.getInternalOffset());
  CATCH_REQUIRE(data[0]);
  CATCH_REQUIRE(data[1]);
  CATCH_REQUIRE(!data[2]);
  CATCH_REQUIRE(data[3]);
}

CATCH_TEST_CASE("test CPU round", "[core][nn][operators]") {
  // Every tie goes to the even neighbour -- torch.round -- where C's round() would take -2.5 to
  // -3 and 2.5 to 3. Float only: this backend's unary kernels take half on aarch64 alone, and the
  // CUDA and Metal tests check half.
  std::vector<float> in = {-2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 0.4f, 0.6f, -0.6f, 3.7f};
  std::vector<float> want = {-2.0f, -2.0f, 0.0f, 0.0f, 2.0f, 2.0f, 0.0f, 1.0f, -1.0f, 4.0f};
  Tensor x = Tensor::create<float>({static_cast<int>(in.size())}, in);

  Tensor rounded = cpuOps()->round(x);
  CATCH_REQUIRE(rounded.getDType() == DType::kFloat);
  const float *got = rounded.getInternalData()->getData<float>(rounded.getInternalOffset());
  for (size_t i = 0; i < in.size(); ++i) {
    CATCH_INFO("round(" << in[i] << ")");
    CATCH_REQUIRE(got[i] == want[i]);
  }
}

CATCH_TEST_CASE("test CPU cast to int64", "[core][nn][operators]") {
  // Truncated toward zero, as torch's .long() is -- a caller that wants the nearest integer rounds
  // first -- up to the 6560 a speech token reaches.
  std::vector<float> in = {2.9f, -2.9f, 0.0f, 6560.0f, 1e6f, -0.5f};
  std::vector<LongType> want = {2, -2, 0, 6560, 1000000, 0};
  Tensor x = Tensor::create<float>({2, 3}, in);

  for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
    Tensor source = cpuOps()->cast(x, dtype);
    Tensor ids = cpuOps()->cast(source, DType::kLong);
    CATCH_REQUIRE(ids.getDType() == DType::kLong);
    CATCH_REQUIRE(ids.getShape() == std::vector<int>{2, 3});

    const LongType *got = ids.getInternalData()->getData<LongType>(ids.getInternalOffset());
    for (size_t i = 0; i < in.size(); ++i) {
      CATCH_INFO(in[i] << " from " << dtype.toString());
      // 1e6 is past half's range and becomes infinity there, so that one is float's alone.
      if (dtype == DType::kFloat16 && in[i] > 65504.0f) continue;
      CATCH_REQUIRE(got[i] == want[i]);
    }
  }
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
