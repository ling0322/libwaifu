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

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/error.h"
#include "lutil/time.h"
#include "flint/test_functional.h"
#include "flint/device.h"
#include "flint/operators.h"
#include "flint/metal_native/context.h"
#include "flint/metal_native/ops.h"

// Every operator the Metal device has, checked against the CPU's answer to the same question.
// Half precision is checked against the CPU in float, which is what it approximates.

namespace fl {

namespace {

Operators *mtl() {
  return getOperators(Device::kMetal);
}

Operators *cpu() {
  return getOperators(Device::kCpu);
}

Tensor toMetal(const Tensor &a, DType dtype = DType::kFloat) {
  return F::cast(mtl(), F::toDevice(mtl(), a, Device::getMetal()), dtype);
}

Tensor toCpu(const Tensor &a) {
  Tensor x = a.getDType().isFloat() ? F::cast(mtl(), a, DType::kFloat) : a;
  return F::toDevice(mtl(), x, Device::getCpu());
}

// `x + value` on the CPU, the value broadcast over `x`.
Tensor plus(const Tensor &x, float value) {
  Tensor scalar = F::zeros(Device::getCpu(), {1}, x.getDType());
  cpu()->fill(scalar, value);
  return F::add(cpu(), x, scalar);
}

// A random float tensor on the CPU, spread over [-2, 2) so that signs and saturation are seen.
Tensor randn(std::initializer_list<int> shape) {
  return plus(F::mul(cpu(), F::rand(Device::getCpu(), shape, DType::kFloat), 4.0f), -2.0f);
}

bool close(const Tensor &metal, const Tensor &reference, float rtol, float atol) {
  return cpu()->allClose(toCpu(metal), reference, rtol, atol);
}

std::vector<float> values(const Tensor &a) {
  Tensor x = F::contiguous(cpu(), a.getDevice().getType() == Device::kCpu ? a : toCpu(a));
  const float *data = x.getInternalData()->getData<float>(x.getInternalOffset());
  return std::vector<float>(data, data + x.getNumEl());
}

#define SKIP_WITHOUT_METAL() \
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available")

}  // namespace

CATCH_TEST_CASE("test Metal toDevice", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  Tensor a = randn({3, 7, 5});
  CATCH_REQUIRE(cpu()->allClose(toCpu(toMetal(a)), a, 1e-6f, 1e-6f));

  // Strided on the host, and strided on the device on the way back.
  Tensor at = a.transpose(0, 2).slice(1, {1, 6});
  CATCH_REQUIRE(cpu()->allClose(toCpu(toMetal(at)), at, 1e-6f, 1e-6f));
  CATCH_REQUIRE(cpu()->allClose(toCpu(toMetal(a).transpose(0, 2).slice(1, {1, 6})), at, 1e-6f, 1e-6f));

  Tensor half = toMetal(a, DType::kFloat16);
  CATCH_REQUIRE(half.getDType() == DType::kFloat16);
  CATCH_REQUIRE(close(half, a, 1e-3f, 1e-3f));

  Tensor longs = F::arangeLong(Device::getCpu(), -5, 100, 3);
  Tensor back = toCpu(F::toDevice(mtl(), longs, Device::getMetal()));
  CATCH_REQUIRE(back.getDType() == DType::kLong);
  for (int i = 0; i < longs.getShape(0); ++i) {
    CATCH_REQUIRE(back.getInternalData()->getData<LongType>(0)[i] == -5 + 3 * i);
  }
}

CATCH_TEST_CASE("test Metal binary operators", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  Tensor a = randn({2, 5, 10});
  Tensor b = randn({5});
  Tensor at = a.transpose(2, 1).slice(1, {1, 9});

  for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
    CATCH_INFO("dtype = " << dtype.toString());
    float tol = dtype == DType::kFloat ? 1e-5f : 5e-3f;
    Tensor x = toMetal(a, dtype).transpose(2, 1).slice(1, {1, 9});
    Tensor y = toMetal(b, dtype);

    CATCH_REQUIRE(close(F::add(mtl(), x, y), F::add(cpu(), at, b), tol, tol));
    CATCH_REQUIRE(close(F::sub(mtl(), x, y), F::sub(cpu(), at, b), tol, tol));
    CATCH_REQUIRE(close(F::mul(mtl(), x, y), F::mul(cpu(), at, b), tol, tol));
    CATCH_REQUIRE(close(F::divTensor(mtl(), x, y), F::divTensor(cpu(), at, b), 1e-2f, tol));
    CATCH_REQUIRE(close(F::mul(mtl(), x, 0.1f), F::mul(cpu(), at, 0.1f), tol, tol));
    CATCH_REQUIRE(close(F::div(mtl(), x, 4.0f), F::div(cpu(), at, 4.0f), tol, tol));

    // A bias broadcast over (N, C, H, W), the pattern a convolution adds its bias in.
    Tensor image = randn({2, 3, 4, 5});
    Tensor bias = randn({3});
    Tensor mtlSum = F::add(mtl(), toMetal(image, dtype), toMetal(bias, dtype).view({1, 3, 1, 1}));
    CATCH_REQUIRE(close(mtlSum, F::add(cpu(), image, bias.view({1, 3, 1, 1})), tol, tol));
  }
}

CATCH_TEST_CASE("test Metal integer and bool operators", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  Tensor longs = F::arangeLong(Device(Device::kMetal), -7, 20, 2);
  Tensor remainders = toCpu(F::mod(mtl(), longs, 5));
  for (int i = 0; i < longs.getShape(0); ++i) {
    LongType x = -7 + 2 * i;
    CATCH_REQUIRE(remainders.getInternalData()->getData<LongType>(0)[i] == x % 5);
  }

  Tensor sum = toCpu(F::add(mtl(), longs, longs));
  CATCH_REQUIRE(sum.getInternalData()->getData<LongType>(0)[3] == 2 * (-7 + 6));

  Tensor a = toMetal(randn({4, 6}));
  Tensor same = F::eq(mtl(), a, a);
  CATCH_REQUIRE(same.getDType() == DType::kBool);
  CATCH_REQUIRE(mtl()->all(same));
  CATCH_REQUIRE(!mtl()->all(F::eq(mtl(), a, F::add(mtl(), a, a))));
  CATCH_REQUIRE(mtl()->all(F::eq(mtl(), longs, longs)));
}

CATCH_TEST_CASE("test Metal unary operators", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  Tensor a = randn({2, 5, 10});
  Tensor positive = plus(F::abs(cpu(), a), 0.1f);
  Tensor at = a.transpose(2, 1).slice(1, {1, 9});

  for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
    CATCH_INFO("dtype = " << dtype.toString());
    float tol = dtype == DType::kFloat ? 1e-5f : 5e-3f;
    Tensor x = toMetal(a, dtype).transpose(2, 1).slice(1, {1, 9});
    Tensor p = toMetal(positive, dtype);

    CATCH_REQUIRE(close(F::neg(mtl(), x), F::neg(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(F::abs(mtl(), x), F::abs(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(F::exp(mtl(), x), F::exp(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(F::square(mtl(), x), F::square(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(F::sigmoid(mtl(), x), F::sigmoid(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(F::tanh(mtl(), x), F::tanh(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(F::relu(mtl(), x), F::relu(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(F::gelu(mtl(), x), F::gelu(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(F::silu(mtl(), x), F::silu(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(F::quickGelu(mtl(), x), F::quickGelu(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(F::sin(mtl(), x), F::sin(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(F::cos(mtl(), x), F::cos(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(F::sqrt(mtl(), p), F::sqrt(cpu(), positive), tol, tol));
    CATCH_REQUIRE(close(F::rsqrt(mtl(), p), F::rsqrt(cpu(), positive), tol, tol));
    CATCH_REQUIRE(close(F::log(mtl(), p), F::log(cpu(), positive), tol, tol));
  }
}

CATCH_TEST_CASE("test Metal log at its edges", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  // GLSL leaves log undefined at and below zero, so these are the kernel's own answers, and they
  // have to be the ones logf gives.
  Tensor x = Tensor::create<float>({5}, {0.0f, -1.0f, INFINITY, 1.0f, 2.718281828f});
  for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
    CATCH_INFO("dtype = " << dtype.toString());
    std::vector<float> y = values(F::log(mtl(), toMetal(x, dtype)));
    CATCH_REQUIRE((isinf(y[0]) && y[0] < 0));
    CATCH_REQUIRE(isnan(y[1]));
    CATCH_REQUIRE((isinf(y[2]) && y[2] > 0));
    CATCH_REQUIRE(y[3] == 0.0f);
    CATCH_REQUIRE(fabsf(y[4] - 1.0f) < 1e-3f);
  }
}

CATCH_TEST_CASE("test Metal round and cast to int64", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  // Ties to even, as torch.round; half holds every value exactly, so both types answer exactly.
  Tensor x = Tensor::create<float>({10}, {-2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 0.4f, 0.6f, -0.6f, 3.7f});
  const std::vector<float> rounded = {-2.0f, -2.0f, 0.0f, 0.0f, 2.0f, 2.0f, 0.0f, 1.0f, -1.0f, 4.0f};
  const std::vector<LongType> truncated = {-2, -1, 0, 0, 1, 2, 0, 0, 0, 3};

  for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
    CATCH_INFO(dtype.toString());
    Tensor onDevice = toMetal(x, dtype);

    Tensor r = F::round(mtl(), onDevice);
    CATCH_REQUIRE(r.getDType() == dtype);
    CATCH_REQUIRE(values(r) == rounded);

    // To int64, which truncates toward zero; rounded first it is the nearest id.
    auto asLongs = [](const Tensor &ids) {
      Tensor host = toCpu(ids);
      const LongType *data = host.getInternalData()->getData<LongType>(host.getInternalOffset());
      return std::vector<LongType>(data, data + host.getNumEl());
    };
    Tensor ids = F::cast(mtl(), r, DType::kLong);
    CATCH_REQUIRE(ids.getDType() == DType::kLong);
    std::vector<LongType> nearest(rounded.begin(), rounded.end());
    CATCH_REQUIRE(asLongs(ids) == nearest);
    CATCH_REQUIRE(asLongs(F::cast(mtl(), onDevice, DType::kLong)) == truncated);
  }
}


CATCH_TEST_CASE("test Metal softmax and reductions", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  for (int lastDim : {1, 8, 255, 1280, 5000}) {
    CATCH_INFO("lastDim = " << lastDim);
    Tensor a = randn({3, 4, lastDim});
    for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
      CATCH_INFO("dtype = " << dtype.toString());
      float tol = dtype == DType::kFloat ? 1e-5f : 5e-3f;
      Tensor x = toMetal(a, dtype);
      CATCH_REQUIRE(close(F::softmax(mtl(), x), F::softmax(cpu(), a), tol, tol));
      CATCH_REQUIRE(close(F::max(mtl(), x), F::max(cpu(), a), tol, tol));
      CATCH_REQUIRE(close(F::min(mtl(), x), F::min(cpu(), a), tol, tol));
      // A sum of thousands of elements reaches the tens, where a half is only good to a
      // hundredth, so the tolerance is that of the result's type rather than of the inputs'.
      float sumTol = dtype == DType::kFloat ? 1e-3f : 1e-2f;
      CATCH_REQUIRE(close(F::sum(mtl(), x, -1), F::sum(cpu(), a, -1), sumTol, sumTol));
    }
  }

  // Along a dimension other than the last, and over everything.
  Tensor a = randn({3, 4, 5});
  Tensor x = toMetal(a);
  CATCH_REQUIRE(close(F::sum(mtl(), x, 0), F::sum(cpu(), a, 0), 1e-5f, 1e-5f));
  CATCH_REQUIRE(close(F::sum(mtl(), x, 1), F::sum(cpu(), a, 1), 1e-5f, 1e-5f));

  float total = 0.0f;
  for (float v : values(a)) total += v;
  Tensor all = F::sum(mtl(), x.view({-1}), 0);
  CATCH_REQUIRE(all.getNumEl() == 1);
  CATCH_REQUIRE(fabsf(mtl()->elem(all) - total) < 1e-4f);
}

CATCH_TEST_CASE("test Metal cumsum", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  // Rows shorter than a 256-wide tile, exactly one, just past it, and several with a ragged end:
  // the carry from one tile into the next is what the longer ones check.
  for (int length : {1, 7, 256, 257, 1000, 4099}) {
    CATCH_INFO("length = " << length);
    Tensor a = F::rand(Device::getCpu(), {3, 5, length}, DType::kFloat);
    Tensor got = F::cumsum(mtl(), toMetal(a), -1);
    CATCH_REQUIRE(got.getShape() == a.getShape());
    CATCH_REQUIRE(close(got, F::cumsum(cpu(), a, -1), 1e-4f, 1e-4f));
  }

  // Every dimension, counted from either end.
  Tensor b = F::rand(Device::getCpu(), {4, 300, 9}, DType::kFloat);
  for (int dim : {0, 1, 2, -1, -2, -3}) {
    CATCH_INFO("dim = " << dim);
    CATCH_REQUIRE(close(F::cumsum(mtl(), toMetal(b), dim), F::cumsum(cpu(), b, dim), 1e-4f, 1e-4f));
  }

  // A strided view, scanned along its middle dimension.
  Tensor strided = toMetal(b).transpose(0, 2);
  CATCH_REQUIRE(close(
      F::cumsum(mtl(), strided, 1), F::cumsum(cpu(), F::contiguous(cpu(), b.transpose(0, 2)), 1), 1e-4f, 1e-4f));

  // Half, accumulated in float and rounded once: within half a unit of the ~300 a row sums to.
  Tensor c = F::rand(Device::getCpu(), {2, 600}, DType::kFloat);
  Tensor half = F::cumsum(mtl(), toMetal(c, DType::kFloat16), -1);
  CATCH_REQUIRE(half.getDType() == DType::kFloat16);
  Tensor want = F::cumsum(cpu(), F::cast(cpu(), F::cast(cpu(), c, DType::kFloat16), DType::kFloat), -1);
  CATCH_REQUIRE(close(half, want, 1e-3f, 1e-3f));

  // Known values, so a dropped or double-counted element is a wrong total and not noise.
  Tensor d = Tensor::create<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  CATCH_REQUIRE(values(F::cumsum(mtl(), toMetal(d), -1)) ==
                std::vector<float>({1.0f, 3.0f, 6.0f, 4.0f, 9.0f, 15.0f}));
  CATCH_REQUIRE(values(F::cumsum(mtl(), toMetal(d), 0)) ==
                std::vector<float>({1.0f, 2.0f, 3.0f, 5.0f, 7.0f, 9.0f}));
}

CATCH_TEST_CASE("test Metal norms", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  Tensor a = randn({3, 5, 320});
  Tensor weight = randn({320});
  Tensor bias = randn({320});
  Tensor image = randn({2, 64, 9, 7});
  Tensor channelWeight = randn({64});
  Tensor channelBias = randn({64});

  for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
    CATCH_INFO("dtype = " << dtype.toString());
    float tol = dtype == DType::kFloat ? 1e-4f : 2e-2f;
    Tensor x = toMetal(a, dtype);
    Tensor w = toMetal(weight, dtype);
    Tensor b = toMetal(bias, dtype);

    CATCH_REQUIRE(close(
        F::layerNorm(mtl(), x, w, b, 1e-5f),
        F::layerNorm(cpu(), a, weight, bias, 1e-5f),
        tol,
        tol));
    CATCH_REQUIRE(close(
        F::layerNorm(mtl(), x, Tensor(), Tensor(), 1e-6f),
        F::layerNorm(cpu(), a, Tensor(), Tensor(), 1e-6f),
        tol,
        tol));
    CATCH_REQUIRE(close(F::rmsNorm(mtl(), x, w, 1e-6f), F::rmsNorm(cpu(), a, weight, 1e-6f), tol, tol));

    Tensor mtlImage = toMetal(image, dtype);
    CATCH_REQUIRE(close(
        F::groupNorm(mtl(), 
            mtlImage,
            toMetal(channelWeight, dtype),
            toMetal(channelBias, dtype),
            32,
            1e-5f),
        F::groupNorm(cpu(), image, channelWeight, channelBias, 32, 1e-5f),
        tol,
        tol));
    CATCH_REQUIRE(close(
        F::groupNorm(mtl(), mtlImage, Tensor(), Tensor(), 8, 1e-6f),
        F::groupNorm(cpu(), image, Tensor(), Tensor(), 8, 1e-6f),
        tol,
        tol));
  }
}

namespace {

// The products against the CPU's, on whichever kernels the Metal device has chosen.
void checkMatmul() {
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
      // Edges that are not whole tiles: B along
      // N in a batch of batches, and B along K.
      {{2, 5, 200, 64}, {2, 5, 64, 72}, false},
      {{3, 300, 256}, {136, 256}, true},
  };

  for (const Case &c : cases) {
    Tensor a = F::rand(Device::getCpu(), c.shapeA, DType::kFloat);
    Tensor b = F::rand(Device::getCpu(), c.shapeB, DType::kFloat);
    Tensor bt = c.transposeB ? b.transpose(-1, -2) : b;
    Tensor reference = F::matmul(cpu(), a, bt);

    for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
      CATCH_INFO("A = " << a.getShapeString() << ", B = " << bt.getShapeString()
                        << ", dtype = " << dtype.toString());
      Tensor x = toMetal(a, dtype);
      Tensor y = toMetal(b, dtype);
      if (c.transposeB) y = y.transpose(-1, -2);
      float tol = dtype == DType::kFloat ? 1e-4f : 2e-2f;
      CATCH_REQUIRE(close(F::matmul(mtl(), x, y), reference, tol, tol));
    }
  }

  // A transposed on the left, as attention's second product sees it.
  Tensor a = F::rand(Device::getCpu(), {64, 48}, DType::kFloat);
  Tensor b = F::rand(Device::getCpu(), {64, 32}, DType::kFloat);
  CATCH_REQUIRE(close(
      F::matmul(mtl(), toMetal(a).transpose(0, 1), toMetal(b)),
      F::matmul(cpu(), a.transpose(0, 1), b),
      1e-4f,
      1e-4f));
}

// For as long as it lives, the Metal device runs none of the kernels for the neural
// accelerators, so that those they stand in for are tested on a GPU that has them too.
class WithoutNax {
 public:
  WithoutNax() {
    op::metal_native::Context::get()->setNaxEnabled(false);
  }
  ~WithoutNax() {
    op::metal_native::Context::get()->setNaxEnabled(true);
  }
};

}  // namespace

CATCH_TEST_CASE("test Metal matmul", "[op][metal]") {
  SKIP_WITHOUT_METAL();
  CATCH_INFO("neural accelerators: " << op::metal_native::Context::get()->hasNax());
  checkMatmul();
}

CATCH_TEST_CASE("test Metal matmul without the neural accelerators", "[op][metal]") {
  SKIP_WITHOUT_METAL();
  WithoutNax without;
  checkMatmul();
}

namespace {

// The convolutions against the CPU's, on whichever kernels the Metal device has chosen.
void checkConvolutions() {
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
        F::conv2d(cpu(), input, weight, bias, c.stride, c.padding, c.dilation, c.groups);
    Tensor noBias =
        F::conv2d(cpu(), input, weight, Tensor(), c.stride, c.padding, c.dilation, c.groups);

    for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
      CATCH_INFO("input = " << input.getShapeString() << ", weight = "
                            << weight.getShapeString() << ", stride = " << c.stride
                            << ", padding = " << c.padding << ", dilation = " << c.dilation
                            << ", groups = " << c.groups << ", dtype = " << dtype.toString());
      float tol = dtype == DType::kFloat ? 1e-4f : 3e-2f;
      Tensor x = toMetal(input, dtype);
      Tensor w = toMetal(weight, dtype);
      CATCH_REQUIRE(close(
          F::conv2d(mtl(), x, w, toMetal(bias, dtype), c.stride, c.padding, c.dilation, c.groups),
          reference,
          tol,
          tol));
      CATCH_REQUIRE(close(
          F::conv2d(mtl(), x, w, Tensor(), c.stride, c.padding, c.dilation, c.groups),
          noBias,
          tol,
          tol));
    }
  }

  Tensor signal = randn({2, 6, 50});
  Tensor kernel = randn({6, 1, 7});
  Tensor reference = F::conv1d(cpu(), signal, kernel, Tensor(), 1, 3, 1, 6);
  CATCH_REQUIRE(close(
      F::conv1d(mtl(), toMetal(signal), toMetal(kernel), Tensor(), 1, 3, 1, 6),
      reference,
      1e-4f,
      1e-4f));

  Tensor dense = randn({4, 6, 3});
  Tensor denseBias = randn({4});
  CATCH_REQUIRE(close(
      F::conv1d(mtl(), toMetal(signal), toMetal(dense), toMetal(denseBias), 2, 1, 2, 1),
      F::conv1d(cpu(), signal, dense, denseBias, 2, 1, 2, 1),
      1e-4f,
      1e-4f));
}

}  // namespace

CATCH_TEST_CASE("test Metal convolutions", "[op][metal]") {
  SKIP_WITHOUT_METAL();
  checkConvolutions();
}

CATCH_TEST_CASE("test Metal convolutions without the neural accelerators", "[op][metal]") {
  SKIP_WITHOUT_METAL();
  WithoutNax without;
  checkConvolutions();
}

// The signal kernels, each against the CPU, in both float types: a grouped transposed
// convolution with an output padding and a kernel that is not a whole number of strides, a snake
// with and without beta, every padding mode -- a replication wider than the signal among them --
// and a clamp bounded on one side and on both. Then a transform and its inverse, which compose
// them.
CATCH_TEST_CASE("test Metal signal operators", "[op][metal]") {
  SKIP_WITHOUT_METAL();
  float inf = std::numeric_limits<float>::infinity();

  Tensor x = randn({2, 6, 29});
  Tensor w = randn({6, 2, 7});
  Tensor b = randn({4});
  Tensor upsample = randn({1, 16, 40});
  Tensor upsampleWeight = randn({16, 8, 16});
  Tensor alpha = plus(F::abs(cpu(), randn({6})), 0.2f);
  Tensor beta = plus(F::abs(cpu(), randn({6})), 0.2f);

  for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
    CATCH_INFO("dtype = " << dtype.toString());
    float tol = dtype == DType::kFloat ? 1e-4f : 2e-2f;

    CATCH_REQUIRE(close(
        F::convTranspose1d(
            mtl(), toMetal(x, dtype), toMetal(w, dtype), toMetal(b, dtype), 3, 2, 1, 2),
        F::convTranspose1d(cpu(), x, w, b, 3, 2, 1, 2),
        tol,
        tol));
    // A vocoder's: the kernel twice the stride, no bias.
    CATCH_REQUIRE(close(
        F::convTranspose1d(
            mtl(), toMetal(upsample, dtype), toMetal(upsampleWeight, dtype), Tensor(), 8, 4, 0, 1),
        F::convTranspose1d(cpu(), upsample, upsampleWeight, Tensor(), 8, 4, 0, 1),
        tol,
        tol));

    for (bool withBeta : {false, true}) {
      CATCH_REQUIRE(close(
          F::snake(
              mtl(),
              toMetal(x, dtype),
              toMetal(alpha, dtype),
              withBeta ? toMetal(beta, dtype) : Tensor(),
              1e-9f),
          F::snake(cpu(), x, alpha, withBeta ? beta : Tensor(), 1e-9f),
          tol,
          tol));
    }

    // Arguments to twenty radians, where an approximate sin would drift.
    Tensor far = F::mul(cpu(), x, 5.0f);
    Tensor wide = plus(F::abs(cpu(), randn({6})), 1.0f);
    CATCH_REQUIRE(close(
        F::snake(mtl(), toMetal(far, dtype), toMetal(wide, dtype), Tensor(), 1e-9f),
        F::snake(cpu(), toCpu(toMetal(far, dtype)), toCpu(toMetal(wide, dtype)), Tensor(), 1e-9f),
        tol,
        tol));

    // A padding moves elements and computes nothing, so it is exact once the CPU's answer has
    // been through the same type.
    Tensor rounded = toCpu(toMetal(x, dtype));
    for (PadMode mode : {PadMode::kZero, PadMode::kReflect, PadMode::kReplicate}) {
      CATCH_REQUIRE(close(
          F::pad1d(mtl(), toMetal(x, dtype), 5, 9, mode),
          F::pad1d(cpu(), rounded, 5, 9, mode),
          0.0f,
          1e-6f));
    }
    CATCH_REQUIRE(close(
        F::pad1d(mtl(), toMetal(x, dtype), 40, 33, PadMode::kReplicate),
        F::pad1d(cpu(), rounded, 40, 33, PadMode::kReplicate),
        0.0f,
        1e-6f));

    CATCH_REQUIRE(close(
        F::clamp(mtl(), toMetal(x, dtype), -0.5f, inf),
        F::clamp(cpu(), rounded, -0.5f, inf),
        0.0f,
        1e-6f));
    CATCH_REQUIRE(close(
        F::clamp(mtl(), toMetal(x, dtype), -1.0f, 0.5f),
        F::clamp(cpu(), rounded, -1.0f, 0.5f),
        0.0f,
        1e-6f));

    // A NaN stays one, which Metal's min and max do not keep; an infinity goes to its bound.
    Tensor special = F::zeros(Device::getCpu(), {4}, DType::kFloat);
    float *raw = special.getInternalData()->getData<float>(special.getInternalOffset());
    raw[0] = std::numeric_limits<float>::quiet_NaN();
    raw[1] = inf;
    raw[2] = -inf;
    raw[3] = 0.125f;
    std::vector<float> clamped = values(F::clamp(mtl(), toMetal(special, dtype), -0.5f, 0.25f));
    CATCH_REQUIRE(std::isnan(clamped[0]));
    CATCH_REQUIRE(clamped[1] == 0.25f);
    CATCH_REQUIRE(clamped[2] == -0.5f);
    CATCH_REQUIRE(clamped[3] == 0.125f);
  }

  // A transform and its inverse, which every device composes out of the operators above.
  int nFft = 16, hop = 4;
  Tensor wave = randn({2, 1, 64});
  Tensor window = F::zeros(Device::getCpu(), {nFft}, DType::kFloat);
  float *taps = window.getInternalData()->getData<float>(window.getInternalOffset());
  for (int i = 0; i < nFft; ++i) taps[i] = 0.5f - 0.5f * cosf(2.0f * 3.14159265f * i / nFft);
  Tensor spectrum = F::stft(mtl(), toMetal(wave), toMetal(window), nFft, hop, true);
  CATCH_REQUIRE(close(spectrum, F::stft(cpu(), wave, window, nFft, hop, true), 1e-4f, 1e-4f));
  CATCH_REQUIRE(
      close(F::istft(mtl(), spectrum, toMetal(window), nFft, hop, true), wave, 1e-4f, 1e-4f));

  // In half and not centred, the ends are kept, where the window's overlap is below what half can
  // hold: dividing by it there made NaNs.
  Tensor halfSpectrum = F::stft(
      mtl(), toMetal(wave, DType::kFloat16), toMetal(window, DType::kFloat16), nFft, hop, false);
  std::vector<float> back =
      values(F::istft(mtl(), halfSpectrum, toMetal(window, DType::kFloat16), nFft, hop, false));
  std::vector<float> want = values(wave);
  for (size_t i = 0; i < back.size(); ++i) {
    CATCH_INFO("sample " << i);
    CATCH_REQUIRE(std::isfinite(back[i]));
    size_t at = i % 64;
    if (at >= size_t(nFft) && at < 64 - size_t(nFft)) {
      CATCH_REQUIRE(fabsf(back[i] - want[i]) < 2e-2f);
    }
  }
}

CATCH_TEST_CASE("test Metal attention", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  Tensor q = randn({2, 4, 37, 32});
  Tensor k = randn({2, 2, 37, 32});
  Tensor v = randn({2, 2, 37, 32});
  for (bool causal : {false, true}) {
    CATCH_INFO("causal = " << causal);
    Tensor reference = F::attention(cpu(), q, k, v, causal);
    CATCH_REQUIRE(close(
        F::attention(mtl(), toMetal(q), toMetal(k), toMetal(v), causal),
        reference,
        1e-4f,
        1e-4f));
    CATCH_REQUIRE(close(
        F::attention(mtl(), 
            toMetal(q, DType::kFloat16),
            toMetal(k, DType::kFloat16),
            toMetal(v, DType::kFloat16),
            causal),
        reference,
        3e-2f,
        3e-2f));
  }
}

CATCH_TEST_CASE("test Metal attention of other widths and lengths", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  struct Case {
    int heads, keyValueHeads, queryLength, keyLength, headDim;
    bool causal;
  };
  std::vector<Case> cases = {
      {3, 3, 70, 77, 64, false},   // cross attention: fewer queries than keys
      {2, 1, 33, 90, 64, true},    // causal over a longer history
      {2, 2, 40, 40, 128, false},
      {2, 2, 45, 45, 80, false},   // no kernel for 80: the composition
  };
  for (const Case &c : cases) {
    CATCH_INFO("heads = " << c.heads << "/" << c.keyValueHeads << ", lengths = " << c.queryLength
                          << "/" << c.keyLength << ", headDim = " << c.headDim
                          << ", causal = " << c.causal);
    // The queries as a model hands them in: (B, L, H, D), viewed as (B, H, L, D).
    Tensor q = randn({1, c.queryLength, c.heads, c.headDim});
    Tensor k = randn({1, c.keyValueHeads, c.keyLength, c.headDim});
    Tensor v = randn({1, c.keyValueHeads, c.keyLength, c.headDim});
    Tensor reference = F::attention(cpu(), q.transpose(1, 2), k, v, c.causal);

    Tensor mq = toMetal(q, DType::kFloat16).transpose(1, 2);
    Tensor got = F::attention(
        mtl(),
        mq,
        toMetal(k, DType::kFloat16),
        toMetal(v, DType::kFloat16),
        c.causal);
    CATCH_REQUIRE(close(got, reference, 3e-2f, 3e-2f));
  }
}

namespace {

// What an E4M3 code means, worked out the long way: the kernel's bit trick is not what this
// checks it against.
float decodeE4M3(uint8_t code) {
  int sign = code >> 7;
  int exponent = (code >> 3) & 0xf;
  int mantissa = code & 0x7;
  float magnitude = exponent == 0 ? ldexpf(static_cast<float>(mantissa), -9)
                                  : ldexpf(1.0f + mantissa / 8.0f, exponent - 7);
  return sign ? -magnitude : magnitude;
}

// The fp8 products against the CPU's, on whichever kernel the Metal device has chosen.
void checkFp8Matmul() {
  struct Case {
    std::vector<int> shapeA;  // (..., k)
    int rows;
  };
  std::vector<Case> cases = {
      {{37, 48}, 72},       // nothing a whole tile, nothing four wide
      {{37, 50}, 72},       // k not a multiple of four: one element at a time
      {{128, 256}, 128},    // whole tiles, read four at a time
      {{2, 5, 64}, 96},     // a batch
      {{1, 3072}, 3072},    // one row of a long product, as a single token has it
  };

  std::mt19937 rng(7);
  for (const Case &c : cases) {
    int k = c.shapeA.back();
    CATCH_INFO("a has " << c.shapeA.size() << " dimensions, k = " << k << ", rows = " << c.rows);

    // Codes from all of E4M3 but its NaNs, and a scale a weight's would be.
    Tensor codes = F::empty(Device::getCpu(), {c.rows, k}, DType::kFp8E4M3);
    Fp8E4M3 *code = codes.getInternalData()->getData<Fp8E4M3>(0);
    std::vector<float> weight(static_cast<size_t>(c.rows) * k);
    for (size_t i = 0; i < weight.size(); ++i) {
      uint8_t x = static_cast<uint8_t>(rng() % 256);
      if ((x & 0x7f) == 0x7f) x = 0x3c;
      code[i].v = x;
      weight[i] = decodeE4M3(x);
    }
    float scale = 0.001f * (1 + static_cast<int>(rng() % 100));
    Tensor scales = Tensor::create<float>({1}, {scale});

    // The activation in half, from [-2, 2), and the product the long way, in float.
    Tensor uniform = F::rand(Device::getCpu(), c.shapeA, DType::kFloat);
    Tensor a = toMetal(plus(F::mul(cpu(), uniform, 4.0f), -2.0f), DType::kFloat16);
    std::vector<float> activation = values(a);
    int m = static_cast<int>(activation.size()) / k;
    std::vector<float> reference(static_cast<size_t>(m) * c.rows);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < c.rows; ++j) {
        double sum = 0;
        for (int l = 0; l < k; ++l) sum += double(activation[i * k + l]) * weight[j * k + l];
        reference[i * c.rows + j] = static_cast<float>(sum * scale);
      }
    }

    std::vector<int> outShape = c.shapeA;
    outShape.back() = c.rows;
    Tensor out = F::empty(Device::getMetal(), outShape, DType::kFloat16);
    op::metal_native::fp8Matmul(
        a,
        F::toDevice(mtl(), codes, Device::getMetal()),
        F::toDevice(mtl(), scales, Device::getMetal()),
        out);
    std::vector<float> got = values(out);

    // Against the float product: what is left is the one rounding to half of the answer.
    double err = 0, ref = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      err += (double(got[i]) - reference[i]) * (double(got[i]) - reference[i]);
      ref += double(reference[i]) * reference[i];
    }
    CATCH_CHECK(sqrt(err / ref) < 1e-3);
  }
}

}  // namespace

CATCH_TEST_CASE("test Metal fp8 matmul", "[op][metal]") {
  SKIP_WITHOUT_METAL();
  checkFp8Matmul();
}

CATCH_TEST_CASE("test Metal fp8 matmul without the neural accelerators", "[op][metal]") {
  SKIP_WITHOUT_METAL();
  WithoutNax without;
  checkFp8Matmul();
}

CATCH_TEST_CASE("test Metal shape operators", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  Tensor table = randn({10, 7});
  Tensor ids = Tensor::create<LongType>({2, 3}, {3, 0, 9, 9, 1, 4});
  CATCH_REQUIRE(close(
      F::lookup(mtl(), toMetal(table, DType::kFloat16), F::toDevice(mtl(), ids, Device::getMetal())),
      F::lookup(cpu(), table, ids),
      1e-3f,
      1e-3f));

  Tensor image = randn({2, 3, 4, 5});
  CATCH_REQUIRE(close(
      F::upsampleNearest2d(mtl(), toMetal(image), 2),
      F::upsampleNearest2d(cpu(), image, 2),
      1e-6f,
      1e-6f));

  Tensor frames = randn({2, 3, 90});
  CATCH_REQUIRE(close(
      F::upsampleNearest1d(mtl(), toMetal(frames), 154),
      F::upsampleNearest1d(cpu(), frames, 154),
      1e-6f,
      1e-6f));
  CATCH_REQUIRE(close(
      F::upsampleNearest1d(mtl(), toMetal(frames), 41),
      F::upsampleNearest1d(cpu(), frames, 41),
      1e-6f,
      1e-6f));

  Tensor gated = randn({3, 4, 16});
  CATCH_REQUIRE(close(F::geglu(mtl(), toMetal(gated)), F::geglu(cpu(), gated), 1e-5f, 1e-5f));
  CATCH_REQUIRE(close(F::swiglu(mtl(), toMetal(gated)), F::swiglu(cpu(), gated), 1e-5f, 1e-5f));

  Tensor a = randn({3, 4});
  Tensor b = randn({3, 2});
  CATCH_REQUIRE(close(F::cat(mtl(), toMetal(a), toMetal(b), 1), F::cat(cpu(), a, b, 1), 1e-6f, 1e-6f));
  CATCH_REQUIRE(close(
      F::contiguous(mtl(), toMetal(image).transpose(1, 3)),
      F::contiguous(cpu(), image.transpose(1, 3)),
      1e-6f,
      1e-6f));

  Tensor zeros = F::zeros(Device(Device::kMetal), {3, 5}, DType::kFloat16);
  CATCH_REQUIRE(close(zeros, F::zeros(Device::getCpu(), {3, 5}, DType::kFloat), 1e-6f, 1e-6f));

  Tensor filled = F::empty(Device(Device::kMetal), {4, 6}, DType::kFloat);
  mtl()->fill(filled, 2.5f);
  mtl()->fill(filled.slice(1, {2, 4}), -1.0f);
  Tensor expected = F::empty(Device::getCpu(), {4, 6}, DType::kFloat);
  cpu()->fill(expected, 2.5f);
  cpu()->fill(expected.slice(1, {2, 4}), -1.0f);
  CATCH_REQUIRE(close(filled, expected, 1e-6f, 1e-6f));

  Tensor mask = F::causalMask(Device(Device::kMetal), 5);
  std::vector<float> maskValues = values(mask);
  CATCH_REQUIRE(maskValues[0 * 5 + 0] == 0.0f);
  CATCH_REQUIRE(maskValues[1 * 5 + 0] == 0.0f);
  CATCH_REQUIRE(isinf(maskValues[0 * 5 + 1]));
}

CATCH_TEST_CASE("test Metal rotary embedding", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  int T = 5, heads = 3, headDim = 8, maxPos = 16;
  Tensor q = randn({T, heads, headDim});
  Tensor cache = randn({maxPos, 2 * headDim});
  std::vector<LongType> positionValues = {0, 3, 7, 7, 15};
  Tensor positions = Tensor::create<LongType>({T}, positionValues);

  Tensor x = toMetal(q, DType::kFloat16);
  mtl()->rotaryEmbedding(
      F::toDevice(mtl(), positions, Device::getMetal()),
      x,
      toMetal(q, DType::kFloat16),
      toMetal(cache, DType::kFloat16));

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

CATCH_TEST_CASE("test Metal rand draws the CUDA operators' numbers", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  // Random123's known answer for a zero counter and key: this is the generator it names.
  uint32_t known[4];
  philox(0, 0, known);
  CATCH_REQUIRE(known[0] == 0x6627e8d5u);
  CATCH_REQUIRE(known[1] == 0xe169c58du);
  CATCH_REQUIRE(known[2] == 0xbc57ac4cu);
  CATCH_REQUIRE(known[3] == 0x9b00dbd8u);

  const float kTwoPow32Inv = 2.3283064e-10f;
  mtl()->manualSeed(1234);
  F::rand(Device(Device::kMetal), {5}, DType::kFloat);  // moves the counter on by two blocks
  std::vector<float> uniform = values(F::rand(Device(Device::kMetal), {9}, DType::kFloat));
  for (int i = 0; i < 9; ++i) {
    uint32_t bits[4];
    philox(1234, 2 + i / 4, bits);
    CATCH_REQUIRE(uniform[i] == bits[i % 4] * kTwoPow32Inv + kTwoPow32Inv / 2.0f);
  }

  // Normal, which only has to agree to within what the device's transcendentals round to.
  mtl()->manualSeed(99);
  Tensor normal = F::randNormal(Device(Device::kMetal), {2, 50000});
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

CATCH_TEST_CASE("test Metal memory statistics", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  mtl()->releaseUnusedMemory();
  mtl()->resetPeakMemoryStats();
  MemorySnapshot before = mtl()->captureMemorySnapshot();
  CATCH_REQUIRE(before.getTotalMemory() > 0);
  {
    Tensor big = F::zeros(Device(Device::kMetal), {64, 1024, 1024}, DType::kFloat16);
    MemorySnapshot during = mtl()->captureMemorySnapshot();
    CATCH_REQUIRE(during.getAllocatedMemory() - before.getAllocatedMemory() >= 128 << 20);
  }
  MemorySnapshot after = mtl()->captureMemorySnapshot();
  CATCH_REQUIRE(after.getAllocatedMemory() == before.getAllocatedMemory());
  CATCH_REQUIRE(after.getPeakAllocatedMemory() >= before.getAllocatedMemory() + (128 << 20));
  mtl()->releaseUnusedMemory();
}


CATCH_TEST_CASE("test Metal allocation the device cannot give is out of memory", "[op][metal]") {
  SKIP_WITHOUT_METAL();
  using op::metal_native::Buffer;

  // A terabyte: more than any Mac's buffer can be, so the device refuses it however much is free.
  op::metal_native::VramAllocator &allocator = op::metal_native::Context::get()->getAllocator();
  int64_t before = allocator.getAllocatedBytes();
  CATCH_REQUIRE_THROWS_AS(allocator.allocate(int64_t(1) << 40), lut::OutOfMemoryError);
  CATCH_REQUIRE(allocator.getAllocatedBytes() == before);

  // Its own kind of error, so that a caller can tell it from every other failure.
  try {
    allocator.allocate(int64_t(1) << 40);
  } catch (const lut::Error &e) {
    CATCH_REQUIRE(e.getCode() == lut::ErrorCode::OutOfMemory);
  }

  // And nothing is left broken by it: the next allocation that fits is had as before.
  Buffer taken = allocator.allocate(1 << 20);
  CATCH_REQUIRE(taken->size >= 1 << 20);
  allocator.free(taken);
  CATCH_REQUIRE(allocator.getAllocatedBytes() == before);
}

CATCH_TEST_CASE("test Metal allocation past the memory limit is out of memory", "[op][metal]") {
  SKIP_WITHOUT_METAL();
  using op::metal_native::Buffer;

  op::metal_native::VramAllocator &allocator = op::metal_native::Context::get()->getAllocator();
  allocator.releaseUnusedMemory();

  // Room for one 600 MB tensor and not two. Larger than a chunk, so no chunk already held -- every
  // one of them in use, now that the empty ones have gone -- has room for it.
  constexpr int64_t kBig = 600 << 20;
  allocator.setMemoryLimit(allocator.getHeldBytes() + (1 << 30));

  Buffer first = allocator.allocate(kBig);
  CATCH_REQUIRE(allocator.getHeldBytes() <= allocator.getMemoryLimit());
  CATCH_REQUIRE_THROWS_AS(allocator.allocate(kBig), lut::OutOfMemoryError);
  CATCH_REQUIRE(allocator.getHeldBytes() <= allocator.getMemoryLimit());

  // Given back, the bytes are there for the next tensor, inside the limit.
  allocator.free(first);
  Buffer again = allocator.allocate(kBig);
  allocator.free(again);

  // And no limit is the device's limit again.
  allocator.setMemoryLimit(0);
  Buffer a = allocator.allocate(kBig);
  Buffer b = allocator.allocate(kBig);
  allocator.free(a);
  allocator.free(b);
  allocator.releaseUnusedMemory();
}

namespace {

// A float tensor of `shape` on Metal, allocated as `hint` says.
Tensor metalTensor(std::vector<int> shape, AllocHint hint) {
  int64_t numel = 1;
  for (int n : shape) numel *= n;
  return Tensor::create(
      std::make_shared<TensorShape>(lut::makeConstSpan(shape)),
      F::allocate(Device::getMetal(), numel, DType::kFloat, hint));
}

}  // namespace

CATCH_TEST_CASE("test Metal a large tensor's chunk is kept for one that fills it", "[op][metal]") {
  SKIP_WITHOUT_METAL();
  using op::metal_native::Buffer;
  op::metal_native::VramAllocator &allocator = op::metal_native::Context::get()->getAllocator();
  allocator.releaseUnusedMemory();

  // 900 MB: a chunk of its own, which stays when the tensor goes.
  Buffer large = allocator.allocate(900 << 20);
  op::metal_native::Chunk *chunk = large->block->chunk;
  CATCH_REQUIRE(chunk->dedicated);
  allocator.free(large);

  // A small tensor is not cut from it.
  Buffer small = allocator.allocate(1 << 20);
  CATCH_CHECK(small->block->chunk != chunk);

  // One that fills four fifths of it and more has it, the same chunk.
  Buffer filling = allocator.allocate(800 << 20);
  CATCH_CHECK(filling->block->chunk == chunk);
  allocator.free(filling);

  // One that would leave more than a fifth of it empty has a chunk of its own instead -- made once
  // the empty 900 MB one has gone back, so that what is held is it, the small one's and no more.
  // (The chunk is told by its size: the one that went may leave its address to the new one.)
  int64_t smallChunk = small->block->chunk->size;
  Buffer smaller = allocator.allocate(600 << 20);
  CATCH_CHECK(smaller->block->chunk->size == 600 << 20);
  CATCH_CHECK(allocator.getHeldBytes() == smallChunk + (600 << 20));

  allocator.free(smaller);
  allocator.free(small);
  allocator.releaseUnusedMemory();
}

CATCH_TEST_CASE("test Metal weights have buffers of their own", "[op][metal]") {
  SKIP_WITHOUT_METAL();
  std::shared_ptr<op::metal_native::Context> context = op::metal_native::Context::get();
  op::metal_native::VramAllocator &allocator = context->getAllocator();
  allocator.releaseUnusedMemory();
  int64_t held = allocator.getHeldBytes();
  int64_t allocated = allocator.getAllocatedBytes();

  {
    // 20 KB: a buffer of its own, of whole pages, and no chunk.
    Tensor weight = metalTensor({1024, 5}, AllocHint::kWeight);
    int64_t taken = allocator.getHeldBytes() - held;
    CATCH_REQUIRE(taken >= 1024 * 5 * 4);
    CATCH_REQUIRE(taken < (1 << 20));
    CATCH_REQUIRE(allocator.getAllocatedBytes() - allocated == 1024 * 5 * 4);

    // Read and written like any other tensor's.
    mtl()->fill(weight, 1.5f);
    for (float value : values(weight)) CATCH_REQUIRE(value == 1.5f);

    // Written again and let go of at once, with the fill still being encoded: the free waits for
    // that work, since the command buffers do not keep what they use, and then gives the buffer
    // back.
    mtl()->fill(weight, 2.5f);
  }
  CATCH_REQUIRE(allocator.getAllocatedBytes() == allocated);
  CATCH_REQUIRE(allocator.getHeldBytes() == held);
}

CATCH_TEST_CASE("test Metal weights count against the memory limit", "[op][metal]") {
  SKIP_WITHOUT_METAL();
  op::metal_native::VramAllocator &allocator = op::metal_native::Context::get()->getAllocator();
  allocator.releaseUnusedMemory();

  // Room for one 600 KB weight and not two.
  allocator.setMemoryLimit(allocator.getHeldBytes() + (1 << 20));
  {
    Tensor first = metalTensor({600, 256}, AllocHint::kWeight);
    CATCH_REQUIRE_THROWS_AS(metalTensor({600, 256}, AllocHint::kWeight), lut::OutOfMemoryError);
  }
  // Let go of, and never used by the device, it is back at once: there is room for the next.
  Tensor again = metalTensor({600, 256}, AllocHint::kWeight);
  allocator.setMemoryLimit(0);
}

// Not a test: how fast the products are, on shapes an SDXL U-Net at 1024 x 1024 multiplies.
// Hidden, so it runs only when asked for: ./build/unittest "[metal-benchmark]".
CATCH_TEST_CASE("benchmark Metal matmul and conv2d", "[.][metal-benchmark]") {
  SKIP_WITHOUT_METAL();

  auto time = [](auto &&run) {
    run();
    mtl()->synchronize();
    int repeat = 10;
    double t0 = lut::now();
    for (int i = 0; i < repeat; ++i) run();
    mtl()->synchronize();
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
    Tensor a = F::cast(mtl(), F::rand(Device(Device::kMetal), c.a, DType::kFloat), DType::kFloat16);
    Tensor b = F::cast(mtl(), F::rand(Device(Device::kMetal), c.b, DType::kFloat), DType::kFloat16);
    Tensor bt = c.transposeB ? b.transpose(-1, -2) : b;
    double seconds = time([&]() { F::matmul(mtl(), a, bt); });
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
    Tensor x = F::cast(mtl(), F::rand(Device(Device::kMetal), {c.N, c.C, c.H, c.H}, DType::kFloat), c.dtype);
    Tensor w = F::cast(mtl(), F::rand(Device(Device::kMetal), {c.K, c.C, 3, 3}, DType::kFloat), c.dtype);
    double seconds = time([&]() { F::conv2d(mtl(), x, w, Tensor(), 1, 1, 1, 1); });
    double flops = 2.0 * c.N * c.H * c.H * c.K * c.C * 9;
    printf("%-28s %8.3f ms %7.1f TFLOPS\n", c.name, seconds * 1e3, flops / seconds / 1e12);
  }
}

// Not a test: how fast the fp8 product is, with and without the neural accelerators, on the
// shapes qwen-image 2.1 multiplies at 1024 x 1024. ./build/unittest "[metal-benchmark]".
CATCH_TEST_CASE("benchmark Metal fp8 matmul", "[.][metal-benchmark]") {
  SKIP_WITHOUT_METAL();

  auto time = [](auto &&run) {
    run();
    mtl()->synchronize();
    int repeat = 10;
    double t0 = lut::now();
    for (int i = 0; i < repeat; ++i) run();
    mtl()->synchronize();
    return (lut::now() - t0) / repeat;
  };

  struct Case {
    const char *name;
    int m, k, rows;
  };
  std::vector<Case> cases = {
      {"dit qkv/out 4096x4096x4096", 4096, 4096, 4096},
      {"dit mlp up 4096x12288x4096", 4096, 4096, 12288},
      {"dit mlp down 4096x4096x12288", 4096, 12288, 4096},
      {"text mlp up 300x12288x4096", 300, 4096, 12288},
  };
  std::mt19937 rng(7);
  for (const Case &c : cases) {
    Tensor codes = F::empty(Device::getCpu(), {c.rows, c.k}, DType::kFp8E4M3);
    Fp8E4M3 *code = codes.getInternalData()->getData<Fp8E4M3>(0);
    for (int64_t i = 0; i < static_cast<int64_t>(c.rows) * c.k; ++i) {
      uint8_t x = static_cast<uint8_t>(rng() % 256);
      code[i].v = (x & 0x7f) == 0x7f ? 0x3c : x;
    }
    Tensor weight = F::toDevice(mtl(), codes, Device::getMetal());
    Tensor scales = F::empty(Device::getMetal(), {1}, DType::kFloat);
    mtl()->fill(scales, 0.01f);
    Tensor a = toMetal(F::rand(Device::getCpu(), {c.m, c.k}, DType::kFloat), DType::kFloat16);
    Tensor out = F::empty(Device::getMetal(), {c.m, c.rows}, DType::kFloat16);

    auto run = [&]() { op::metal_native::fp8Matmul(a, weight, scales, out); };
    double flops = 2.0 * c.m * c.k * c.rows;
    double withNax = time(run);
    double without;
    {
      WithoutNax off;
      without = time(run);
    }
    printf("%-30s accelerators %7.2f ms %6.2f TFLOP/s   simdgroup %7.2f ms %6.2f TFLOP/s\n",
           c.name, withNax * 1e3, flops / withNax / 1e12, without * 1e3, flops / without / 1e12);
  }
}

}  // namespace fl
