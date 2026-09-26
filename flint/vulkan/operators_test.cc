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

#include <math.h>
#include <stdint.h>

#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/time.h"
#include "flint/device.h"
#include "flint/operators.h"

// Every operator the Vulkan device has, checked against the CPU's answer to the same question.
// Half precision is checked against the CPU in float, which is what it approximates.

namespace fl {

namespace {

Operators *vk() {
  return getOperators(Device::kVulkan);
}

Operators *cpu() {
  return getOperators(Device::kCpu);
}

Tensor toVulkan(const Tensor &a, DType dtype = DType::kFloat) {
  return vk()->cast(vk()->toDevice(Device::getVulkan(), a), dtype);
}

Tensor toCpu(const Tensor &a) {
  Tensor x = a.getDType().isFloat() ? vk()->cast(a, DType::kFloat) : a;
  return vk()->toDevice(Device::getCpu(), x);
}

// A random float tensor on the CPU, spread over [-2, 2) so that signs and saturation are seen.
Tensor randn(std::initializer_list<int> shape) {
  return cpu()->subFloat(cpu()->mul(cpu()->rand(shape, DType::kFloat), 4.0f), 2.0f);
}

bool close(const Tensor &vulkan, const Tensor &reference, float rtol, float atol) {
  return cpu()->allClose(toCpu(vulkan), reference, rtol, atol);
}

std::vector<float> values(const Tensor &a) {
  Tensor x = cpu()->contiguous(a.getDevice().getType() == Device::kCpu ? a : toCpu(a));
  const float *data = x.getInternalData()->getData<float>(x.getInternalOffset());
  return std::vector<float>(data, data + x.getNumEl());
}

#define SKIP_WITHOUT_VULKAN() \
  if (!isOperatorsAvailable(Device::kVulkan)) CATCH_SKIP("vulkan device not available")

}  // namespace

CATCH_TEST_CASE("test Vulkan toDevice", "[op][vulkan]") {
  SKIP_WITHOUT_VULKAN();

  Tensor a = randn({3, 7, 5});
  CATCH_REQUIRE(cpu()->allClose(toCpu(toVulkan(a)), a, 1e-6f, 1e-6f));

  // Strided on the host, and strided on the device on the way back.
  Tensor at = a.transpose(0, 2).slice(1, {1, 6});
  CATCH_REQUIRE(cpu()->allClose(toCpu(toVulkan(at)), at, 1e-6f, 1e-6f));
  CATCH_REQUIRE(cpu()->allClose(toCpu(toVulkan(a).transpose(0, 2).slice(1, {1, 6})), at, 1e-6f, 1e-6f));

  Tensor half = toVulkan(a, DType::kFloat16);
  CATCH_REQUIRE(half.getDType() == DType::kFloat16);
  CATCH_REQUIRE(close(half, a, 1e-3f, 1e-3f));

  Tensor longs = cpu()->arangeLong(-5, 100, 3);
  Tensor back = toCpu(vk()->toDevice(Device::getVulkan(), longs));
  CATCH_REQUIRE(back.getDType() == DType::kLong);
  for (int i = 0; i < longs.getShape(0); ++i) {
    CATCH_REQUIRE(back.getInternalData()->getData<LongType>(0)[i] == -5 + 3 * i);
  }
}

CATCH_TEST_CASE("test Vulkan binary operators", "[op][vulkan]") {
  SKIP_WITHOUT_VULKAN();

  Tensor a = randn({2, 5, 10});
  Tensor b = randn({5});
  Tensor at = a.transpose(2, 1).slice(1, {1, 9});

  for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
    CATCH_INFO("dtype = " << dtype.toString());
    float tol = dtype == DType::kFloat ? 1e-5f : 5e-3f;
    Tensor x = toVulkan(a, dtype).transpose(2, 1).slice(1, {1, 9});
    Tensor y = toVulkan(b, dtype);

    CATCH_REQUIRE(close(vk()->add(x, y), cpu()->add(at, b), tol, tol));
    CATCH_REQUIRE(close(vk()->sub(x, y), cpu()->sub(at, b), tol, tol));
    CATCH_REQUIRE(close(vk()->mul(x, y), cpu()->mul(at, b), tol, tol));
    CATCH_REQUIRE(close(vk()->divTensor(x, y), cpu()->divTensor(at, b), 1e-2f, tol));
    CATCH_REQUIRE(close(vk()->mul(x, 0.1f), cpu()->mul(at, 0.1f), tol, tol));
    CATCH_REQUIRE(close(vk()->div(x, 4.0f), cpu()->div(at, 4.0f), tol, tol));
    CATCH_REQUIRE(close(vk()->subFloat(x, 0.5f), cpu()->subFloat(at, 0.5f), tol, tol));

    // A bias broadcast over (N, C, H, W), the pattern a convolution adds its bias in.
    Tensor image = randn({2, 3, 4, 5});
    Tensor bias = randn({3});
    Tensor vkSum = vk()->add(toVulkan(image, dtype), toVulkan(bias, dtype).view({1, 3, 1, 1}));
    CATCH_REQUIRE(close(vkSum, cpu()->add(image, bias.view({1, 3, 1, 1})), tol, tol));
  }
}

CATCH_TEST_CASE("test Vulkan integer and bool operators", "[op][vulkan]") {
  SKIP_WITHOUT_VULKAN();

  Tensor longs = vk()->arangeLong(-7, 20, 2);
  Tensor remainders = toCpu(vk()->mod(longs, 5));
  for (int i = 0; i < longs.getShape(0); ++i) {
    LongType x = -7 + 2 * i;
    CATCH_REQUIRE(remainders.getInternalData()->getData<LongType>(0)[i] == x % 5);
  }

  Tensor sum = toCpu(vk()->add(longs, longs));
  CATCH_REQUIRE(sum.getInternalData()->getData<LongType>(0)[3] == 2 * (-7 + 6));

  Tensor a = toVulkan(randn({4, 6}));
  Tensor same = vk()->eq(a, a);
  CATCH_REQUIRE(same.getDType() == DType::kBool);
  CATCH_REQUIRE(vk()->all(same));
  CATCH_REQUIRE(!vk()->all(vk()->eq(a, vk()->add(a, a))));
  CATCH_REQUIRE(vk()->all(vk()->eq(longs, longs)));
}

CATCH_TEST_CASE("test Vulkan unary operators", "[op][vulkan]") {
  SKIP_WITHOUT_VULKAN();

  Tensor a = randn({2, 5, 10});
  Tensor positive = cpu()->subFloat(cpu()->abs(a), -0.1f);
  Tensor at = a.transpose(2, 1).slice(1, {1, 9});

  for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
    CATCH_INFO("dtype = " << dtype.toString());
    float tol = dtype == DType::kFloat ? 1e-5f : 5e-3f;
    Tensor x = toVulkan(a, dtype).transpose(2, 1).slice(1, {1, 9});
    Tensor p = toVulkan(positive, dtype);

    CATCH_REQUIRE(close(vk()->neg(x), cpu()->neg(at), tol, tol));
    CATCH_REQUIRE(close(vk()->abs(x), cpu()->abs(at), tol, tol));
    CATCH_REQUIRE(close(vk()->exp(x), cpu()->exp(at), tol, tol));
    CATCH_REQUIRE(close(vk()->square(x), cpu()->square(at), tol, tol));
    CATCH_REQUIRE(close(vk()->sigmoid(x), cpu()->sigmoid(at), tol, tol));
    CATCH_REQUIRE(close(vk()->tanh(x), cpu()->tanh(at), tol, tol));
    CATCH_REQUIRE(close(vk()->relu(x), cpu()->relu(at), tol, tol));
    CATCH_REQUIRE(close(vk()->gelu(x), cpu()->gelu(at), tol, tol));
    CATCH_REQUIRE(close(vk()->silu(x), cpu()->silu(at), tol, tol));
    CATCH_REQUIRE(close(vk()->quickGelu(x), cpu()->quickGelu(at), tol, tol));
    CATCH_REQUIRE(close(vk()->sin(x), cpu()->sin(at), tol, tol));
    CATCH_REQUIRE(close(vk()->cos(x), cpu()->cos(at), tol, tol));
    CATCH_REQUIRE(close(vk()->sqrt(p), cpu()->sqrt(positive), tol, tol));
    CATCH_REQUIRE(close(vk()->rsqrt(p), cpu()->rsqrt(positive), tol, tol));
    CATCH_REQUIRE(close(vk()->log(p), cpu()->log(positive), tol, tol));
  }
}

CATCH_TEST_CASE("test Vulkan log at its edges", "[op][vulkan]") {
  SKIP_WITHOUT_VULKAN();

  // GLSL leaves log undefined at and below zero, so these are the kernel's own answers, and they
  // have to be the ones logf gives.
  Tensor x = Tensor::create<float>({5}, {0.0f, -1.0f, INFINITY, 1.0f, 2.718281828f});
  for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
    CATCH_INFO("dtype = " << dtype.toString());
    std::vector<float> y = values(vk()->log(toVulkan(x, dtype)));
    CATCH_REQUIRE((isinf(y[0]) && y[0] < 0));
    CATCH_REQUIRE(isnan(y[1]));
    CATCH_REQUIRE((isinf(y[2]) && y[2] > 0));
    CATCH_REQUIRE(y[3] == 0.0f);
    CATCH_REQUIRE(fabsf(y[4] - 1.0f) < 1e-3f);
  }
}

CATCH_TEST_CASE("test Vulkan round and cast to int64", "[op][vulkan]") {
  SKIP_WITHOUT_VULKAN();

  // Ties to even, as torch.round; half holds every value exactly, so both types answer exactly.
  Tensor x = Tensor::create<float>({10}, {-2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 0.4f, 0.6f, -0.6f, 3.7f});
  const std::vector<float> rounded = {-2.0f, -2.0f, 0.0f, 0.0f, 2.0f, 2.0f, 0.0f, 1.0f, -1.0f, 4.0f};
  const std::vector<LongType> truncated = {-2, -1, 0, 0, 1, 2, 0, 0, 0, 3};

  for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
    CATCH_INFO(dtype.toString());
    Tensor onDevice = toVulkan(x, dtype);

    Tensor r = vk()->round(onDevice);
    CATCH_REQUIRE(r.getDType() == dtype);
    CATCH_REQUIRE(values(r) == rounded);

    // To int64, which truncates toward zero; rounded first it is the nearest id.
    auto asLongs = [](const Tensor &ids) {
      Tensor host = toCpu(ids);
      const LongType *data = host.getInternalData()->getData<LongType>(host.getInternalOffset());
      return std::vector<LongType>(data, data + host.getNumEl());
    };
    Tensor ids = vk()->cast(r, DType::kLong);
    CATCH_REQUIRE(ids.getDType() == DType::kLong);
    std::vector<LongType> nearest(rounded.begin(), rounded.end());
    CATCH_REQUIRE(asLongs(ids) == nearest);
    CATCH_REQUIRE(asLongs(vk()->cast(onDevice, DType::kLong)) == truncated);
  }
}

CATCH_TEST_CASE("test Vulkan softmax and reductions", "[op][vulkan]") {
  SKIP_WITHOUT_VULKAN();

  for (int lastDim : {1, 8, 255, 1280, 5000}) {
    CATCH_INFO("lastDim = " << lastDim);
    Tensor a = randn({3, 4, lastDim});
    for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
      CATCH_INFO("dtype = " << dtype.toString());
      float tol = dtype == DType::kFloat ? 1e-5f : 5e-3f;
      Tensor x = toVulkan(a, dtype);
      CATCH_REQUIRE(close(vk()->softmax(x), cpu()->softmax(a), tol, tol));
      CATCH_REQUIRE(close(vk()->max(x), cpu()->max(a), tol, tol));
      CATCH_REQUIRE(close(vk()->min(x), cpu()->min(a), tol, tol));
      // A sum of thousands of elements reaches the tens, where a half is only good to a
      // hundredth, so the tolerance is that of the result's type rather than of the inputs'.
      float sumTol = dtype == DType::kFloat ? 1e-3f : 1e-2f;
      CATCH_REQUIRE(close(vk()->sum(x, -1), cpu()->sum(a, -1), sumTol, sumTol));
    }
  }

  // Along a dimension other than the last, and over everything.
  Tensor a = randn({3, 4, 5});
  Tensor x = toVulkan(a);
  CATCH_REQUIRE(close(vk()->sum(x, 0), cpu()->sum(a, 0), 1e-5f, 1e-5f));
  CATCH_REQUIRE(close(vk()->sum(x, 1), cpu()->sum(a, 1), 1e-5f, 1e-5f));

  float total = 0.0f;
  for (float v : values(a)) total += v;
  Tensor all = vk()->sum(x, None);
  CATCH_REQUIRE(all.getNumEl() == 1);
  CATCH_REQUIRE(fabsf(vk()->elem(all) - total) < 1e-4f);
}

CATCH_TEST_CASE("test Vulkan norms", "[op][vulkan]") {
  SKIP_WITHOUT_VULKAN();

  Tensor a = randn({3, 5, 320});
  Tensor weight = randn({320});
  Tensor bias = randn({320});
  Tensor image = randn({2, 64, 9, 7});
  Tensor channelWeight = randn({64});
  Tensor channelBias = randn({64});

  for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
    CATCH_INFO("dtype = " << dtype.toString());
    float tol = dtype == DType::kFloat ? 1e-4f : 2e-2f;
    Tensor x = toVulkan(a, dtype);
    Tensor w = toVulkan(weight, dtype);
    Tensor b = toVulkan(bias, dtype);

    CATCH_REQUIRE(close(
        vk()->layerNorm(x, w, b, 1e-5f),
        cpu()->layerNorm(a, weight, bias, 1e-5f),
        tol,
        tol));
    CATCH_REQUIRE(close(
        vk()->layerNorm(x, Tensor(), Tensor(), 1e-6f),
        cpu()->layerNorm(a, Tensor(), Tensor(), 1e-6f),
        tol,
        tol));
    CATCH_REQUIRE(close(vk()->rmsNorm(x, w, 1e-6f), cpu()->rmsNorm(a, weight, 1e-6f), tol, tol));

    Tensor vkImage = toVulkan(image, dtype);
    CATCH_REQUIRE(close(
        vk()->groupNorm(
            vkImage,
            toVulkan(channelWeight, dtype),
            toVulkan(channelBias, dtype),
            32,
            1e-5f),
        cpu()->groupNorm(image, channelWeight, channelBias, 32, 1e-5f),
        tol,
        tol));
    CATCH_REQUIRE(close(
        vk()->groupNorm(vkImage, Tensor(), Tensor(), 8, 1e-6f),
        cpu()->groupNorm(image, Tensor(), Tensor(), 8, 1e-6f),
        tol,
        tol));
  }
}

CATCH_TEST_CASE("test Vulkan matmul", "[op][vulkan]") {
  SKIP_WITHOUT_VULKAN();

  struct Case {
    std::vector<int> shapeA;
    std::vector<int> shapeB;
    bool transposeB;
  };
  std::vector<Case> cases = {
      {{1, 1}, {1, 1}, false},
      {{17, 33}, {33, 5}, false},
      {{64, 128}, {128, 64}, false},
      {{300, 517}, {129, 517}, true},  // a weight, as the layers hand it in
      {{2, 77, 768}, {320, 768}, true},
      {{3, 40, 64}, {3, 64, 50}, false},
      {{2, 5, 40, 64}, {2, 5, 64, 50}, false},
      {{2, 5, 40, 64}, {64, 50}, false},
      {{1, 4096}, {4096, 1}, false},
      // Shapes the cooperative matrix kernel takes, with edges that are not whole tiles: B along
      // N in a batch of batches, and B along K.
      {{2, 5, 200, 64}, {2, 5, 64, 72}, false},
      {{3, 300, 256}, {136, 256}, true},
  };

  for (const Case &c : cases) {
    Tensor a = cpu()->rand(c.shapeA, DType::kFloat);
    Tensor b = cpu()->rand(c.shapeB, DType::kFloat);
    Tensor bt = c.transposeB ? b.transpose(-1, -2) : b;
    Tensor reference = cpu()->matmul(a, bt);

    for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
      CATCH_INFO("A = " << a.getShapeString() << ", B = " << bt.getShapeString()
                        << ", dtype = " << dtype.toString());
      Tensor x = toVulkan(a, dtype);
      Tensor y = toVulkan(b, dtype);
      if (c.transposeB) y = y.transpose(-1, -2);
      float tol = dtype == DType::kFloat ? 1e-4f : 2e-2f;
      CATCH_REQUIRE(close(vk()->matmul(x, y), reference, tol, tol));
    }
  }

  // A transposed on the left, as attention's second product sees it.
  Tensor a = cpu()->rand({64, 48}, DType::kFloat);
  Tensor b = cpu()->rand({64, 32}, DType::kFloat);
  CATCH_REQUIRE(close(
      vk()->matmul(toVulkan(a).transpose(0, 1), toVulkan(b)),
      cpu()->matmul(a.transpose(0, 1), b),
      1e-4f,
      1e-4f));
}

CATCH_TEST_CASE("test Vulkan convolutions", "[op][vulkan]") {
  SKIP_WITHOUT_VULKAN();

  struct Case {
    int N, C, H, W, K, R, stride, padding, dilation, groups;
  };
  std::vector<Case> cases = {
      {1, 4, 8, 8, 8, 3, 1, 1, 1, 1},
      {2, 16, 13, 11, 32, 3, 2, 1, 1, 1},
      {1, 32, 9, 9, 16, 1, 1, 0, 1, 1},
      {1, 8, 10, 10, 8, 3, 1, 2, 2, 1},
      {2, 8, 7, 7, 12, 3, 1, 1, 1, 4},
      {1, 3, 20, 20, 5, 5, 3, 0, 1, 1},
  };

  for (const Case &c : cases) {
    Tensor input = randn({c.N, c.C, c.H, c.W});
    Tensor weight = randn({c.K, c.C / c.groups, c.R, c.R});
    Tensor bias = randn({c.K});
    Tensor reference =
        cpu()->conv2d(input, weight, bias, c.stride, c.padding, c.dilation, c.groups);
    Tensor noBias =
        cpu()->conv2d(input, weight, Tensor(), c.stride, c.padding, c.dilation, c.groups);

    for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
      CATCH_INFO("input = " << input.getShapeString() << ", weight = "
                            << weight.getShapeString() << ", stride = " << c.stride
                            << ", padding = " << c.padding << ", dilation = " << c.dilation
                            << ", groups = " << c.groups << ", dtype = " << dtype.toString());
      float tol = dtype == DType::kFloat ? 1e-4f : 3e-2f;
      Tensor x = toVulkan(input, dtype);
      Tensor w = toVulkan(weight, dtype);
      CATCH_REQUIRE(close(
          vk()->conv2d(x, w, toVulkan(bias, dtype), c.stride, c.padding, c.dilation, c.groups),
          reference,
          tol,
          tol));
      CATCH_REQUIRE(close(
          vk()->conv2d(x, w, Tensor(), c.stride, c.padding, c.dilation, c.groups),
          noBias,
          tol,
          tol));
    }
  }

  Tensor signal = randn({2, 6, 50});
  Tensor kernel = randn({6, 1, 7});
  Tensor reference = cpu()->conv1d(signal, kernel, Tensor(), 1, 3, 1, 6);
  CATCH_REQUIRE(close(
      vk()->conv1d(toVulkan(signal), toVulkan(kernel), Tensor(), 1, 3, 1, 6),
      reference,
      1e-4f,
      1e-4f));

  Tensor dense = randn({4, 6, 3});
  Tensor denseBias = randn({4});
  CATCH_REQUIRE(close(
      vk()->conv1d(toVulkan(signal), toVulkan(dense), toVulkan(denseBias), 2, 1, 2, 1),
      cpu()->conv1d(signal, dense, denseBias, 2, 1, 2, 1),
      1e-4f,
      1e-4f));
}

CATCH_TEST_CASE("test Vulkan attention", "[op][vulkan]") {
  SKIP_WITHOUT_VULKAN();

  Tensor q = randn({2, 4, 37, 32});
  Tensor k = randn({2, 2, 37, 32});
  Tensor v = randn({2, 2, 37, 32});
  for (bool causal : {false, true}) {
    CATCH_INFO("causal = " << causal);
    Tensor reference = cpu()->attention(q, k, v, causal);
    CATCH_REQUIRE(close(
        vk()->attention(toVulkan(q), toVulkan(k), toVulkan(v), causal),
        reference,
        1e-4f,
        1e-4f));
    CATCH_REQUIRE(close(
        vk()->attention(
            toVulkan(q, DType::kFloat16),
            toVulkan(k, DType::kFloat16),
            toVulkan(v, DType::kFloat16),
            causal),
        reference,
        3e-2f,
        3e-2f));
  }
}

CATCH_TEST_CASE("test Vulkan shape operators", "[op][vulkan]") {
  SKIP_WITHOUT_VULKAN();

  Tensor table = randn({10, 7});
  Tensor ids = Tensor::create<LongType>({2, 3}, {3, 0, 9, 9, 1, 4});
  CATCH_REQUIRE(close(
      vk()->lookup(toVulkan(table, DType::kFloat16), vk()->toDevice(Device::getVulkan(), ids)),
      cpu()->lookup(table, ids),
      1e-3f,
      1e-3f));

  Tensor image = randn({2, 3, 4, 5});
  CATCH_REQUIRE(close(
      vk()->upsampleNearest2d(toVulkan(image), 2),
      cpu()->upsampleNearest2d(image, 2),
      1e-6f,
      1e-6f));

  Tensor gated = randn({3, 4, 16});
  CATCH_REQUIRE(close(vk()->geglu(toVulkan(gated)), cpu()->geglu(gated), 1e-5f, 1e-5f));
  CATCH_REQUIRE(close(vk()->swiglu(toVulkan(gated)), cpu()->swiglu(gated), 1e-5f, 1e-5f));

  Tensor a = randn({3, 4});
  Tensor b = randn({3, 2});
  CATCH_REQUIRE(close(vk()->cat(toVulkan(a), toVulkan(b), 1), cpu()->cat(a, b, 1), 1e-6f, 1e-6f));
  CATCH_REQUIRE(close(
      vk()->contiguous(toVulkan(image).transpose(1, 3)),
      cpu()->contiguous(image.transpose(1, 3)),
      1e-6f,
      1e-6f));

  Tensor zeros = vk()->zeros({3, 5}, DType::kFloat16);
  CATCH_REQUIRE(close(zeros, cpu()->zeros({3, 5}, DType::kFloat), 1e-6f, 1e-6f));

  Tensor filled = vk()->tensor({4, 6}, DType::kFloat);
  vk()->fill(filled, 2.5f);
  vk()->fill(filled.slice(1, {2, 4}), -1.0f);
  Tensor expected = cpu()->tensor({4, 6}, DType::kFloat);
  cpu()->fill(expected, 2.5f);
  cpu()->fill(expected.slice(1, {2, 4}), -1.0f);
  CATCH_REQUIRE(close(filled, expected, 1e-6f, 1e-6f));

  Tensor mask = vk()->causalMask(5);
  std::vector<float> maskValues = values(mask);
  CATCH_REQUIRE(maskValues[0 * 5 + 0] == 0.0f);
  CATCH_REQUIRE(maskValues[1 * 5 + 0] == 0.0f);
  CATCH_REQUIRE(isinf(maskValues[0 * 5 + 1]));
}

CATCH_TEST_CASE("test Vulkan rotary embedding", "[op][vulkan]") {
  SKIP_WITHOUT_VULKAN();

  int T = 5, heads = 3, headDim = 8, maxPos = 16;
  Tensor q = randn({T, heads, headDim});
  Tensor cache = randn({maxPos, 2 * headDim});
  std::vector<LongType> positionValues = {0, 3, 7, 7, 15};
  Tensor positions = Tensor::create<LongType>({T}, positionValues);

  Tensor x = toVulkan(q, DType::kFloat16);
  vk()->rotaryEmbedding(
      vk()->toDevice(Device::getVulkan(), positions),
      x,
      toVulkan(q, DType::kFloat16),
      toVulkan(cache, DType::kFloat16));

  std::vector<float> input = values(q);
  std::vector<float> table = values(cache);
  std::vector<float> output = values(x);
  int half = headDim / 2;
  for (int t = 0; t < T; ++t) {
    const float *row = &table[positionValues[t] * 2 * headDim];
    for (int h = 0; h < heads; ++h) {
      const float *v = &input[(t * heads + h) * headDim];
      const float *y = &output[(t * heads + h) * headDim];
      for (int d = 0; d < half; ++d) {
        float first = v[d] * row[d] - v[d + half] * row[headDim + d];
        float second = v[d + half] * row[d + half] + v[d] * row[headDim + d + half];
        CATCH_REQUIRE(fabsf(y[d] - first) < 3e-2f);
        CATCH_REQUIRE(fabsf(y[d + half] - second) < 3e-2f);
      }
    }
  }
}

namespace {

// Philox4x32-10 on the host, to check the kernel's numbers against.
void philox(uint64_t seed, uint64_t position, uint32_t out[4]) {
  uint32_t ctr[4] = {uint32_t(position), uint32_t(position >> 32), 0, 0};
  uint32_t k0 = uint32_t(seed), k1 = uint32_t(seed >> 32);
  for (int round = 0; round < 10; ++round) {
    if (round > 0) {
      k0 += 0x9E3779B9;
      k1 += 0xBB67AE85;
    }
    uint64_t p0 = uint64_t(0xD2511F53) * ctr[0];
    uint64_t p1 = uint64_t(0xCD9E8D57) * ctr[2];
    uint32_t next[4] = {
        uint32_t(p1 >> 32) ^ ctr[1] ^ k0,
        uint32_t(p1),
        uint32_t(p0 >> 32) ^ ctr[3] ^ k1,
        uint32_t(p0)};
    for (int i = 0; i < 4; ++i) ctr[i] = next[i];
  }
  for (int i = 0; i < 4; ++i) out[i] = ctr[i];
}

}  // namespace

CATCH_TEST_CASE("test Vulkan rand draws the CUDA operators' numbers", "[op][vulkan]") {
  SKIP_WITHOUT_VULKAN();

  // Random123's known answer for a zero counter and key: this is the generator it names.
  uint32_t known[4];
  philox(0, 0, known);
  CATCH_REQUIRE(known[0] == 0x6627e8d5u);
  CATCH_REQUIRE(known[1] == 0xe169c58du);
  CATCH_REQUIRE(known[2] == 0xbc57ac4cu);
  CATCH_REQUIRE(known[3] == 0x9b00dbd8u);

  const float kTwoPow32Inv = 2.3283064e-10f;
  vk()->manualSeed(1234);
  vk()->rand({5}, DType::kFloat);  // moves the counter on by two blocks
  std::vector<float> uniform = values(vk()->rand({9}, DType::kFloat));
  for (int i = 0; i < 9; ++i) {
    uint32_t bits[4];
    philox(1234, 2 + i / 4, bits);
    CATCH_REQUIRE(uniform[i] == bits[i % 4] * kTwoPow32Inv + kTwoPow32Inv / 2.0f);
  }

  // Normal, which only has to agree to within what the device's transcendentals round to.
  vk()->manualSeed(99);
  Tensor normal = vk()->randNormal({2, 50000});
  CATCH_REQUIRE(normal.getDType() == DType::kFloat);
  std::vector<float> z = values(normal);
  uint32_t bits[4];
  philox(99, 0, bits);
  float radius = sqrtf(-2.0f * logf(bits[0] * kTwoPow32Inv + kTwoPow32Inv / 2.0f));
  float angle = 6.283185307179586f * (bits[1] * kTwoPow32Inv + kTwoPow32Inv / 2.0f);
  CATCH_REQUIRE(fabsf(z[0] - radius * cosf(angle)) < 1e-4f);
  CATCH_REQUIRE(fabsf(z[1] - radius * sinf(angle)) < 1e-4f);

  double mean = 0.0, meanSquare = 0.0;
  for (float v : z) {
    mean += v;
    meanSquare += v * v;
  }
  mean /= z.size();
  meanSquare /= z.size();
  CATCH_REQUIRE(fabs(mean) < 0.02);
  CATCH_REQUIRE(fabs(meanSquare - 1.0) < 0.02);
}

CATCH_TEST_CASE("test Vulkan memory statistics", "[op][vulkan]") {
  SKIP_WITHOUT_VULKAN();

  vk()->releaseUnusedMemory();
  vk()->resetPeakMemoryStats();
  MemorySnapshot before = vk()->captureMemorySnapshot();
  CATCH_REQUIRE(before.getTotalMemory() > 0);
  {
    Tensor big = vk()->zeros({64, 1024, 1024}, DType::kFloat16);
    MemorySnapshot during = vk()->captureMemorySnapshot();
    CATCH_REQUIRE(during.getAllocatedMemory() - before.getAllocatedMemory() >= 128 << 20);
  }
  MemorySnapshot after = vk()->captureMemorySnapshot();
  CATCH_REQUIRE(after.getAllocatedMemory() == before.getAllocatedMemory());
  CATCH_REQUIRE(after.getPeakAllocatedMemory() >= before.getAllocatedMemory() + (128 << 20));
  vk()->releaseUnusedMemory();
}

// Not a test: how fast the products are, on shapes an SDXL U-Net at 1024 x 1024 multiplies.
// Hidden, so it runs only when asked for: ./build/unittest "[vulkan-benchmark]".
CATCH_TEST_CASE("benchmark Vulkan matmul and conv2d", "[.][vulkan-benchmark]") {
  SKIP_WITHOUT_VULKAN();

  auto time = [](auto &&run) {
    run();
    vk()->synchronize();
    int repeat = 10;
    double t0 = lut::now();
    for (int i = 0; i < repeat; ++i) run();
    vk()->synchronize();
    return (lut::now() - t0) / repeat;
  };

  struct Case {
    const char *name;
    std::vector<int> a;
    std::vector<int> b;
    bool transposeB;
  };
  std::vector<Case> cases = {
      {"linear 8192x640x640", {8192, 640}, {640, 640}, true},
      {"linear 8192x5120x640", {8192, 640}, {5120, 640}, true},
      {"linear 2048x1280x1280", {2048, 1280}, {1280, 1280}, true},
      {"q.kT 20x4096x4096x64", {20, 4096, 64}, {20, 4096, 64}, true},
      {"p.v 20x4096x64x4096", {20, 4096, 4096}, {20, 4096, 64}, false},
  };
  for (const Case &c : cases) {
    Tensor a = vk()->cast(vk()->rand(c.a, DType::kFloat), DType::kFloat16);
    Tensor b = vk()->cast(vk()->rand(c.b, DType::kFloat), DType::kFloat16);
    Tensor bt = c.transposeB ? b.transpose(-1, -2) : b;
    double seconds = time([&]() { vk()->matmul(a, bt); });
    double flops = 2.0 * a.getNumEl() * bt.getShape(-1);
    printf("%-28s %8.3f ms %7.1f TFLOPS\n", c.name, seconds * 1e3, flops / seconds / 1e12);
  }

  struct ConvCase {
    const char *name;
    int N, C, H, K;
    DType dtype;
  };
  std::vector<ConvCase> convs = {
      {"conv 3x3 2x320x128 f16", 2, 320, 128, 320, DType::kFloat16},
      {"conv 3x3 2x640x64 f16", 2, 640, 64, 640, DType::kFloat16},
      {"conv 3x3 1x256x512 f32", 1, 256, 512, 256, DType::kFloat},
  };
  for (const ConvCase &c : convs) {
    Tensor x = vk()->cast(vk()->rand({c.N, c.C, c.H, c.H}, DType::kFloat), c.dtype);
    Tensor w = vk()->cast(vk()->rand({c.K, c.C, 3, 3}, DType::kFloat), c.dtype);
    double seconds = time([&]() { vk()->conv2d(x, w, Tensor(), 1, 1, 1, 1); });
    double flops = 2.0 * c.N * c.H * c.H * c.K * c.C * 9;
    printf("%-28s %8.3f ms %7.1f TFLOPS\n", c.name, seconds * 1e3, flops / seconds / 1e12);
  }
}

}  // namespace fl
