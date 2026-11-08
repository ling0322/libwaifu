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

#include <algorithm>
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
  return fl::test::cast(mtl(), fl::test::toDevice(mtl(), a, Device::getMetal()), dtype);
}

Tensor toCpu(const Tensor &a) {
  Tensor x = a.getDType().isFloat() ? fl::test::cast(mtl(), a, DType::kFloat) : a;
  return fl::test::toDevice(mtl(), x, Device::getCpu());
}

// `x + value` on the CPU, the value broadcast over `x`.
Tensor plus(const Tensor &x, float value) {
  Tensor scalar = fl::test::zeros(Device::getCpu(), {1}, x.getDType());
  cpu()->fill(scalar, value);
  return fl::test::add(cpu(), x, scalar);
}

// A random float tensor on the CPU, spread over [-2, 2) so that signs and saturation are seen.
Tensor randn(std::initializer_list<int> shape) {
  return plus(fl::test::mul(cpu(), fl::test::rand(Device::getCpu(), shape, DType::kFloat), 4.0f), -2.0f);
}

bool close(const Tensor &metal, const Tensor &reference, float rtol, float atol) {
  return cpu()->allClose(toCpu(metal), reference, rtol, atol);
}

std::vector<float> values(const Tensor &a) {
  Tensor x = fl::test::contiguous(cpu(), a.getDevice().getType() == Device::kCpu ? a : toCpu(a));
  const float *data = x.getInternalData()->getData<float>(x.getInternalOffset());
  return std::vector<float>(data, data + x.getNumEl());
}

#define SKIP_WITHOUT_METAL() \
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available")

/// Runs `body(nax)` with the kernels on the neural accelerators, where this GPU has them, and then
/// without, so that a test checks both the kernel and the one it stands in for. They are on again
/// afterwards, however it ends.
template<typename Fn>
void eachKernel(Fn &&body) {
  std::shared_ptr<op::metal_native::Context> context = op::metal_native::Context::get();
  struct On {
    std::shared_ptr<op::metal_native::Context> context;
    ~On() { context->setNaxEnabled(true); }
  } on{context};
  context->setNaxEnabled(true);
  bool hasNax = context->hasNax();
  for (bool nax : {true, false}) {
    if (nax && !hasNax) continue;
    context->setNaxEnabled(nax);
    body(nax);
  }
}

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

  Tensor longs = fl::test::arangeLong(Device::getCpu(), -5, 100, 3);
  Tensor back = toCpu(fl::test::toDevice(mtl(), longs, Device::getMetal()));
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

    CATCH_REQUIRE(close(fl::test::add(mtl(), x, y), fl::test::add(cpu(), at, b), tol, tol));
    CATCH_REQUIRE(close(fl::test::sub(mtl(), x, y), fl::test::sub(cpu(), at, b), tol, tol));
    CATCH_REQUIRE(close(fl::test::mul(mtl(), x, y), fl::test::mul(cpu(), at, b), tol, tol));
    CATCH_REQUIRE(close(fl::test::divTensor(mtl(), x, y), fl::test::divTensor(cpu(), at, b), 1e-2f, tol));
    CATCH_REQUIRE(close(fl::test::mul(mtl(), x, 0.1f), fl::test::mul(cpu(), at, 0.1f), tol, tol));
    CATCH_REQUIRE(close(fl::test::div(mtl(), x, 4.0f), fl::test::div(cpu(), at, 4.0f), tol, tol));

    // A bias broadcast over (N, C, H, W), the pattern a convolution adds its bias in.
    Tensor image = randn({2, 3, 4, 5});
    Tensor bias = randn({3});
    Tensor mtlSum = fl::test::add(mtl(), toMetal(image, dtype), toMetal(bias, dtype).view({1, 3, 1, 1}));
    CATCH_REQUIRE(close(mtlSum, fl::test::add(cpu(), image, bias.view({1, 3, 1, 1})), tol, tol));
  }
}

CATCH_TEST_CASE("test Metal integer and bool operators", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  Tensor longs = fl::test::arangeLong(Device(Device::kMetal), -7, 20, 2);
  Tensor remainders = toCpu(fl::test::mod(mtl(), longs, 5));
  for (int i = 0; i < longs.getShape(0); ++i) {
    LongType x = -7 + 2 * i;
    CATCH_REQUIRE(remainders.getInternalData()->getData<LongType>(0)[i] == x % 5);
  }

  Tensor sum = toCpu(fl::test::add(mtl(), longs, longs));
  CATCH_REQUIRE(sum.getInternalData()->getData<LongType>(0)[3] == 2 * (-7 + 6));

  Tensor a = toMetal(randn({4, 6}));
  Tensor same = fl::test::eq(mtl(), a, a);
  CATCH_REQUIRE(same.getDType() == DType::kBool);
  CATCH_REQUIRE(mtl()->all(same));
  CATCH_REQUIRE(!mtl()->all(fl::test::eq(mtl(), a, fl::test::add(mtl(), a, a))));
  CATCH_REQUIRE(mtl()->all(fl::test::eq(mtl(), longs, longs)));
}

CATCH_TEST_CASE("test Metal unary operators", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  Tensor a = randn({2, 5, 10});
  Tensor positive = plus(fl::test::abs(cpu(), a), 0.1f);
  Tensor at = a.transpose(2, 1).slice(1, {1, 9});

  for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
    CATCH_INFO("dtype = " << dtype.toString());
    float tol = dtype == DType::kFloat ? 1e-5f : 5e-3f;
    Tensor x = toMetal(a, dtype).transpose(2, 1).slice(1, {1, 9});
    Tensor p = toMetal(positive, dtype);

    CATCH_REQUIRE(close(fl::test::neg(mtl(), x), fl::test::neg(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(fl::test::abs(mtl(), x), fl::test::abs(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(fl::test::exp(mtl(), x), fl::test::exp(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(fl::test::square(mtl(), x), fl::test::square(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(fl::test::sigmoid(mtl(), x), fl::test::sigmoid(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(fl::test::tanh(mtl(), x), fl::test::tanh(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(fl::test::relu(mtl(), x), fl::test::relu(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(fl::test::gelu(mtl(), x), fl::test::gelu(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(fl::test::silu(mtl(), x), fl::test::silu(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(fl::test::quickGelu(mtl(), x), fl::test::quickGelu(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(fl::test::sin(mtl(), x), fl::test::sin(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(fl::test::cos(mtl(), x), fl::test::cos(cpu(), at), tol, tol));
    CATCH_REQUIRE(close(fl::test::sqrt(mtl(), p), fl::test::sqrt(cpu(), positive), tol, tol));
    CATCH_REQUIRE(close(fl::test::rsqrt(mtl(), p), fl::test::rsqrt(cpu(), positive), tol, tol));
    CATCH_REQUIRE(close(fl::test::log(mtl(), p), fl::test::log(cpu(), positive), tol, tol));
  }
}

CATCH_TEST_CASE("test Metal log at its edges", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  // GLSL leaves log undefined at and below zero, so these are the kernel's own answers, and they
  // have to be the ones logf gives.
  Tensor x = Tensor::create<float>({5}, {0.0f, -1.0f, INFINITY, 1.0f, 2.718281828f});
  for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
    CATCH_INFO("dtype = " << dtype.toString());
    std::vector<float> y = values(fl::test::log(mtl(), toMetal(x, dtype)));
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

    Tensor r = fl::test::round(mtl(), onDevice);
    CATCH_REQUIRE(r.getDType() == dtype);
    CATCH_REQUIRE(values(r) == rounded);

    // To int64, which truncates toward zero; rounded first it is the nearest id.
    auto asLongs = [](const Tensor &ids) {
      Tensor host = toCpu(ids);
      const LongType *data = host.getInternalData()->getData<LongType>(host.getInternalOffset());
      return std::vector<LongType>(data, data + host.getNumEl());
    };
    Tensor ids = fl::test::cast(mtl(), r, DType::kLong);
    CATCH_REQUIRE(ids.getDType() == DType::kLong);
    std::vector<LongType> nearest(rounded.begin(), rounded.end());
    CATCH_REQUIRE(asLongs(ids) == nearest);
    CATCH_REQUIRE(asLongs(fl::test::cast(mtl(), onDevice, DType::kLong)) == truncated);
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
      CATCH_REQUIRE(close(fl::test::softmax(mtl(), x), fl::test::softmax(cpu(), a), tol, tol));
      CATCH_REQUIRE(close(fl::test::max(mtl(), x), fl::test::max(cpu(), a), tol, tol));
      CATCH_REQUIRE(close(fl::test::min(mtl(), x), fl::test::min(cpu(), a), tol, tol));
      // A sum of thousands of elements reaches the tens, where a half is only good to a
      // hundredth, so the tolerance is that of the result's type rather than of the inputs'.
      float sumTol = dtype == DType::kFloat ? 1e-3f : 1e-2f;
      CATCH_REQUIRE(close(fl::test::sum(mtl(), x, -1), fl::test::sum(cpu(), a, -1), sumTol, sumTol));
    }
  }

  // Along a dimension other than the last, and over everything.
  Tensor a = randn({3, 4, 5});
  Tensor x = toMetal(a);
  CATCH_REQUIRE(close(fl::test::sum(mtl(), x, 0), fl::test::sum(cpu(), a, 0), 1e-5f, 1e-5f));
  CATCH_REQUIRE(close(fl::test::sum(mtl(), x, 1), fl::test::sum(cpu(), a, 1), 1e-5f, 1e-5f));

  float total = 0.0f;
  for (float v : values(a)) total += v;
  Tensor all = fl::test::sum(mtl(), x.view({-1}), 0);
  CATCH_REQUIRE(all.getNumEl() == 1);
  CATCH_REQUIRE(fabsf(mtl()->elem(all) - total) < 1e-4f);
}

CATCH_TEST_CASE("test Metal cumsum", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  // Rows shorter than a 256-wide tile, exactly one, just past it, and several with a ragged end:
  // the carry from one tile into the next is what the longer ones check.
  for (int length : {1, 7, 256, 257, 1000, 4099}) {
    CATCH_INFO("length = " << length);
    Tensor a = fl::test::rand(Device::getCpu(), {3, 5, length}, DType::kFloat);
    Tensor got = fl::test::cumsum(mtl(), toMetal(a), -1);
    CATCH_REQUIRE(got.getShape() == a.getShape());
    CATCH_REQUIRE(close(got, fl::test::cumsum(cpu(), a, -1), 1e-4f, 1e-4f));
  }

  // Every dimension, counted from either end.
  Tensor b = fl::test::rand(Device::getCpu(), {4, 300, 9}, DType::kFloat);
  for (int dim : {0, 1, 2, -1, -2, -3}) {
    CATCH_INFO("dim = " << dim);
    CATCH_REQUIRE(close(fl::test::cumsum(mtl(), toMetal(b), dim), fl::test::cumsum(cpu(), b, dim), 1e-4f, 1e-4f));
  }

  // A strided view, scanned along its middle dimension.
  Tensor strided = toMetal(b).transpose(0, 2);
  CATCH_REQUIRE(close(
      fl::test::cumsum(mtl(), strided, 1), fl::test::cumsum(cpu(), fl::test::contiguous(cpu(), b.transpose(0, 2)), 1), 1e-4f, 1e-4f));

  // Half, accumulated in float and rounded once: within half a unit of the ~300 a row sums to.
  Tensor c = fl::test::rand(Device::getCpu(), {2, 600}, DType::kFloat);
  Tensor half = fl::test::cumsum(mtl(), toMetal(c, DType::kFloat16), -1);
  CATCH_REQUIRE(half.getDType() == DType::kFloat16);
  Tensor want = fl::test::cumsum(cpu(), fl::test::cast(cpu(), fl::test::cast(cpu(), c, DType::kFloat16), DType::kFloat), -1);
  CATCH_REQUIRE(close(half, want, 1e-3f, 1e-3f));

  // Known values, so a dropped or double-counted element is a wrong total and not noise.
  Tensor d = Tensor::create<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  CATCH_REQUIRE(values(fl::test::cumsum(mtl(), toMetal(d), -1)) ==
                std::vector<float>({1.0f, 3.0f, 6.0f, 4.0f, 9.0f, 15.0f}));
  CATCH_REQUIRE(values(fl::test::cumsum(mtl(), toMetal(d), 0)) ==
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
        fl::test::layerNorm(mtl(), x, w, b, 1e-5f),
        fl::test::layerNorm(cpu(), a, weight, bias, 1e-5f),
        tol,
        tol));
    CATCH_REQUIRE(close(
        fl::test::layerNorm(mtl(), x, Tensor(), Tensor(), 1e-6f),
        fl::test::layerNorm(cpu(), a, Tensor(), Tensor(), 1e-6f),
        tol,
        tol));
    CATCH_REQUIRE(close(fl::test::rmsNorm(mtl(), x, w, 1e-6f), fl::test::rmsNorm(cpu(), a, weight, 1e-6f), tol, tol));

    Tensor mtlImage = toMetal(image, dtype);
    CATCH_REQUIRE(close(
        fl::test::groupNorm(mtl(), 
            mtlImage,
            toMetal(channelWeight, dtype),
            toMetal(channelBias, dtype),
            32,
            1e-5f),
        fl::test::groupNorm(cpu(), image, channelWeight, channelBias, 32, 1e-5f),
        tol,
        tol));
    CATCH_REQUIRE(close(
        fl::test::groupNorm(mtl(), mtlImage, Tensor(), Tensor(), 8, 1e-6f),
        fl::test::groupNorm(cpu(), image, Tensor(), Tensor(), 8, 1e-6f),
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
    Tensor a = fl::test::rand(Device::getCpu(), c.shapeA, DType::kFloat);
    Tensor b = fl::test::rand(Device::getCpu(), c.shapeB, DType::kFloat);
    Tensor bt = c.transposeB ? b.transpose(-1, -2) : b;
    Tensor reference = fl::test::matmul(cpu(), a, bt);

    for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
      CATCH_INFO("A = " << a.getShapeString() << ", B = " << bt.getShapeString()
                        << ", dtype = " << dtype.toString());
      Tensor x = toMetal(a, dtype);
      Tensor y = toMetal(b, dtype);
      if (c.transposeB) y = y.transpose(-1, -2);
      float tol = dtype == DType::kFloat ? 1e-4f : 2e-2f;
      CATCH_REQUIRE(close(fl::test::matmul(mtl(), x, y), reference, tol, tol));
    }
  }

  // A transposed on the left, as attention's second product sees it.
  Tensor a = fl::test::rand(Device::getCpu(), {64, 48}, DType::kFloat);
  Tensor b = fl::test::rand(Device::getCpu(), {64, 32}, DType::kFloat);
  CATCH_REQUIRE(close(
      fl::test::matmul(mtl(), toMetal(a).transpose(0, 1), toMetal(b)),
      fl::test::matmul(cpu(), a.transpose(0, 1), b),
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

// `a` on the Metal device in `dtype`, with NaNs after it in its buffer: a kernel that reads past
// its end has them in what it writes.
Tensor toMetalWithNaNAfter(const Tensor &a, DType dtype) {
  std::vector<float> data = values(a);
  int n = static_cast<int>(data.size());
  Tensor padded = fl::test::zeros(Device::getCpu(), {n + 64}, DType::kFloat);
  float *raw = padded.getInternalData()->getData<float>(padded.getInternalOffset());
  std::copy(data.begin(), data.end(), raw);
  std::fill(raw + n, raw + n + 64, std::numeric_limits<float>::quiet_NaN());
  std::vector<int> shape = a.getShape();
  return toMetal(padded, dtype).slice({0, n}).view(shape);
}

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
      // One by one, with fewer channels than a chunk of the reduction and with a chunk and a bit:
      // the pointwise kernel's, and then, with a stride, the gathering kernel's short one.
      {2, 4, 9, 9, 8, 1, 1, 0, 1, 1},
      {2, 36, 6, 6, 8, 1, 1, 0, 1, 1},
      {2, 36, 9, 9, 8, 1, 2, 0, 1, 1},
  };

  for (const Case &c : cases) {
    Tensor input = randn({c.N, c.C, c.H, c.W});
    Tensor weight = randn({c.K, c.C / c.groups, c.R, c.R});
    Tensor bias = randn({c.K});
    Tensor reference =
        fl::test::conv2d(cpu(), input, weight, bias, c.stride, c.padding, c.dilation, c.groups);
    Tensor noBias =
        fl::test::conv2d(cpu(), input, weight, Tensor(), c.stride, c.padding, c.dilation, c.groups);

    for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
      CATCH_INFO("input = " << input.getShapeString() << ", weight = "
                            << weight.getShapeString() << ", stride = " << c.stride
                            << ", padding = " << c.padding << ", dilation = " << c.dilation
                            << ", groups = " << c.groups << ", dtype = " << dtype.toString());
      float tol = dtype == DType::kFloat ? 1e-4f : 3e-2f;
      Tensor x = toMetal(input, dtype);
      Tensor w = toMetal(weight, dtype);
      CATCH_REQUIRE(close(
          fl::test::conv2d(mtl(), x, w, toMetal(bias, dtype), c.stride, c.padding, c.dilation, c.groups),
          reference,
          tol,
          tol));
      CATCH_REQUIRE(close(
          fl::test::conv2d(mtl(), x, w, Tensor(), c.stride, c.padding, c.dilation, c.groups),
          noBias,
          tol,
          tol));

      // Nothing past the end of the weight or the input is read, so NaNs there do not get in.
      Tensor poisoned = fl::test::conv2d(
          mtl(),
          toMetalWithNaNAfter(input, dtype),
          toMetalWithNaNAfter(weight, dtype),
          toMetal(bias, dtype),
          c.stride,
          c.padding,
          c.dilation,
          c.groups);
      CATCH_REQUIRE(close(poisoned, reference, tol, tol));
    }
  }

  Tensor signal = randn({2, 6, 50});
  Tensor kernel = randn({6, 1, 7});
  Tensor reference = fl::test::conv1d(cpu(), signal, kernel, Tensor(), 1, 3, 1, 6);
  CATCH_REQUIRE(close(
      fl::test::conv1d(mtl(), toMetal(signal), toMetal(kernel), Tensor(), 1, 3, 1, 6),
      reference,
      1e-4f,
      1e-4f));

  Tensor dense = randn({4, 6, 3});
  Tensor denseBias = randn({4});
  CATCH_REQUIRE(close(
      fl::test::conv1d(mtl(), toMetal(signal), toMetal(dense), toMetal(denseBias), 2, 1, 2, 1),
      fl::test::conv1d(cpu(), signal, dense, denseBias, 2, 1, 2, 1),
      1e-4f,
      1e-4f));
}

}  // namespace

CATCH_TEST_CASE("test Metal matmul (half kept in float)", "[op][metal]") {
  SKIP_WITHOUT_METAL();
  CATCH_REQUIRE(mtl()->matmulWidensHalf());

  // Half times half into float, as the composed attention takes its scores: the sum as it was,
  // past what half holds. A's first column is 300 and B's first row 1500, so every product is
  // 450,000 and more; the rest is random. K is 75, not a whole number of any kernel's steps. B as
  // it is, and along K -- the transpose of an (N, K), as the keys are -- each as one matrix and as
  // a batch of (2, 3) the way attention hands them over.
  auto loud = [](std::initializer_list<int> shape, bool column, float big) {
    Tensor x = randn(shape);
    std::vector<int> dims = x.getShape();
    int rows = dims[dims.size() - 2], cols = dims.back();
    int64_t matrices = x.getNumEl() / (static_cast<int64_t>(rows) * cols);
    float *data = x.getInternalData()->getData<float>(x.getInternalOffset());
    for (int64_t m = 0; m < matrices; ++m) {
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
          if ((column ? c : r) == 0) data[(m * rows + r) * cols + c] = big;
        }
      }
    }
    return x;
  };
  for (bool batched : {false, true}) {
    for (bool alongK : {false, true}) {
      Tensor a = batched ? loud({2, 3, 70, 75}, true, 300.0f) : loud({70, 75}, true, 300.0f);
      Tensor b;
      if (alongK) {
        // (N, K) with its first column big, viewed as (K, N).
        Tensor nk = batched ? loud({2, 3, 33, 75}, true, 1500.0f) : loud({33, 75}, true, 1500.0f);
        b = toMetal(nk, DType::kFloat16).transpose(-2, -1);
      } else {
        Tensor kn = batched ? loud({2, 3, 75, 33}, false, 1500.0f) : loud({75, 33}, false, 1500.0f);
        b = toMetal(kn, DType::kFloat16);
      }
      Tensor ah = toMetal(a, DType::kFloat16);
      Tensor reference = fl::test::matmul(cpu(), toCpu(ah), fl::test::contiguous(cpu(), toCpu(b)));
      eachKernel([&](bool nax) {
        CATCH_INFO("nax = " << nax << ", batched = " << batched << ", B along K = " << alongK);
        std::vector<int> shape = ah.getShape();
        shape.back() = b.getShape(-1);
        Tensor x = F::empty(Device::getMetal(), shape, DType::kFloat);
        mtl()->matmul(ah, b, x);
        std::vector<float> got = values(x);
        CATCH_REQUIRE(std::all_of(got.begin(), got.end(), [](float y) { return std::isfinite(y); }));
        CATCH_REQUIRE(close(x, reference, 1e-3f, 1e-2f));
      });
    }
  }
}

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
  Tensor alpha = plus(fl::test::abs(cpu(), randn({6})), 0.2f);
  Tensor beta = plus(fl::test::abs(cpu(), randn({6})), 0.2f);

  for (DType dtype : {DType(DType::kFloat), DType(DType::kFloat16)}) {
    CATCH_INFO("dtype = " << dtype.toString());
    float tol = dtype == DType::kFloat ? 1e-4f : 2e-2f;

    CATCH_REQUIRE(close(
        fl::test::convTranspose1d(
            mtl(), toMetal(x, dtype), toMetal(w, dtype), toMetal(b, dtype), 3, 2, 1, 2),
        fl::test::convTranspose1d(cpu(), x, w, b, 3, 2, 1, 2),
        tol,
        tol));
    // A vocoder's: the kernel twice the stride, no bias.
    CATCH_REQUIRE(close(
        fl::test::convTranspose1d(
            mtl(), toMetal(upsample, dtype), toMetal(upsampleWeight, dtype), Tensor(), 8, 4, 0, 1),
        fl::test::convTranspose1d(cpu(), upsample, upsampleWeight, Tensor(), 8, 4, 0, 1),
        tol,
        tol));

    for (bool withBeta : {false, true}) {
      CATCH_REQUIRE(close(
          fl::test::snake(
              mtl(),
              toMetal(x, dtype),
              toMetal(alpha, dtype),
              withBeta ? toMetal(beta, dtype) : Tensor(),
              1e-9f),
          fl::test::snake(cpu(), x, alpha, withBeta ? beta : Tensor(), 1e-9f),
          tol,
          tol));
    }

    // Arguments to twenty radians, where an approximate sin would drift.
    Tensor far = fl::test::mul(cpu(), x, 5.0f);
    Tensor wide = plus(fl::test::abs(cpu(), randn({6})), 1.0f);
    CATCH_REQUIRE(close(
        fl::test::snake(mtl(), toMetal(far, dtype), toMetal(wide, dtype), Tensor(), 1e-9f),
        fl::test::snake(cpu(), toCpu(toMetal(far, dtype)), toCpu(toMetal(wide, dtype)), Tensor(), 1e-9f),
        tol,
        tol));

    // A padding moves elements and computes nothing, so it is exact once the CPU's answer has
    // been through the same type.
    Tensor rounded = toCpu(toMetal(x, dtype));
    for (PadMode mode : {PadMode::kZero, PadMode::kReflect, PadMode::kReplicate}) {
      CATCH_REQUIRE(close(
          fl::test::pad1d(mtl(), toMetal(x, dtype), 5, 9, mode),
          fl::test::pad1d(cpu(), rounded, 5, 9, mode),
          0.0f,
          1e-6f));
    }
    CATCH_REQUIRE(close(
        fl::test::pad1d(mtl(), toMetal(x, dtype), 40, 33, PadMode::kReplicate),
        fl::test::pad1d(cpu(), rounded, 40, 33, PadMode::kReplicate),
        0.0f,
        1e-6f));

    CATCH_REQUIRE(close(
        fl::test::clamp(mtl(), toMetal(x, dtype), -0.5f, inf),
        fl::test::clamp(cpu(), rounded, -0.5f, inf),
        0.0f,
        1e-6f));
    CATCH_REQUIRE(close(
        fl::test::clamp(mtl(), toMetal(x, dtype), -1.0f, 0.5f),
        fl::test::clamp(cpu(), rounded, -1.0f, 0.5f),
        0.0f,
        1e-6f));

    // A NaN stays one, which Metal's min and max do not keep; an infinity goes to its bound.
    Tensor special = fl::test::zeros(Device::getCpu(), {4}, DType::kFloat);
    float *raw = special.getInternalData()->getData<float>(special.getInternalOffset());
    raw[0] = std::numeric_limits<float>::quiet_NaN();
    raw[1] = inf;
    raw[2] = -inf;
    raw[3] = 0.125f;
    std::vector<float> clamped = values(fl::test::clamp(mtl(), toMetal(special, dtype), -0.5f, 0.25f));
    CATCH_REQUIRE(std::isnan(clamped[0]));
    CATCH_REQUIRE(clamped[1] == 0.25f);
    CATCH_REQUIRE(clamped[2] == -0.5f);
    CATCH_REQUIRE(clamped[3] == 0.125f);
  }

  // A transform and its inverse, which every device composes out of the operators above.
  int nFft = 16, hop = 4;
  Tensor wave = randn({2, 1, 64});
  Tensor window = fl::test::zeros(Device::getCpu(), {nFft}, DType::kFloat);
  float *taps = window.getInternalData()->getData<float>(window.getInternalOffset());
  for (int i = 0; i < nFft; ++i) taps[i] = 0.5f - 0.5f * cosf(2.0f * 3.14159265f * i / nFft);
  Tensor spectrum = fl::test::stft(mtl(), toMetal(wave), toMetal(window), nFft, hop, true);
  CATCH_REQUIRE(close(spectrum, fl::test::stft(cpu(), wave, window, nFft, hop, true), 1e-4f, 1e-4f));
  CATCH_REQUIRE(
      close(fl::test::istft(mtl(), spectrum, toMetal(window), nFft, hop, true), wave, 1e-4f, 1e-4f));

  // In half and not centred, the ends are kept, where the window's overlap is below what half can
  // hold: dividing by it there made NaNs.
  Tensor halfSpectrum = fl::test::stft(
      mtl(), toMetal(wave, DType::kFloat16), toMetal(window, DType::kFloat16), nFft, hop, false);
  std::vector<float> back =
      values(fl::test::istft(mtl(), halfSpectrum, toMetal(window, DType::kFloat16), nFft, hop, false));
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
  eachKernel([&](bool nax) {
  for (bool causal : {false, true}) {
    CATCH_INFO("nax = " << nax << ", causal = " << causal);
    Tensor reference = fl::test::attention(cpu(), q, k, v, fl::test::causalIf(causal));
    CATCH_REQUIRE(close(
        fl::test::attention(mtl(), toMetal(q), toMetal(k), toMetal(v), fl::test::causalIf(causal)),
        reference,
        1e-4f,
        1e-4f));
    CATCH_REQUIRE(close(
        fl::test::attention(mtl(), 
            toMetal(q, DType::kFloat16),
            toMetal(k, DType::kFloat16),
            toMetal(v, DType::kFloat16),
            fl::test::causalIf(causal)),
        reference,
        3e-2f,
        3e-2f));
  }
  });
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
  eachKernel([&](bool nax) {
  for (const Case &c : cases) {
    CATCH_INFO("nax = " << nax << ", heads = " << c.heads << "/" << c.keyValueHeads << ", lengths = " << c.queryLength
                          << "/" << c.keyLength << ", headDim = " << c.headDim
                          << ", causal = " << c.causal);
    // The queries as a model hands them in: (B, L, H, D), viewed as (B, H, L, D).
    Tensor q = randn({1, c.queryLength, c.heads, c.headDim});
    Tensor k = randn({1, c.keyValueHeads, c.keyLength, c.headDim});
    Tensor v = randn({1, c.keyValueHeads, c.keyLength, c.headDim});
    Tensor reference = fl::test::attention(cpu(), q.transpose(1, 2), k, v, fl::test::causalIf(c.causal));

    Tensor mq = toMetal(q, DType::kFloat16).transpose(1, 2);
    Tensor got = fl::test::attention(
        mtl(),
        mq,
        toMetal(k, DType::kFloat16),
        toMetal(v, DType::kFloat16),
        fl::test::causalIf(c.causal));
    CATCH_REQUIRE(close(got, reference, 3e-2f, 3e-2f));
  }
  });
}

namespace {

/// Attention under `maskType` in blocks of `blockSize`, written out one query at a time on the
/// host: the reference the kernel's block masks are checked against, since the composition has
/// none. q is (N, H, L, D), k and v (N, KVH, S, D), float; the answer (N, H, L, D).
Tensor maskReference(
    const Tensor &q,
    const Tensor &k,
    const Tensor &v,
    MaskType maskType,
    int blockSize) {
  int batch = q.getShape(0), numHeads = q.getShape(1), queryLength = q.getShape(2);
  int headDim = q.getShape(3), numKeyValueHeads = k.getShape(1), keyLength = k.getShape(2);
  int group = numHeads / numKeyValueHeads;
  std::vector<float> pq = values(q), pk = values(k), pv = values(v);
  Tensor out = fl::test::zeros(Device::getCpu(), {batch, numHeads, queryLength, headDim}, DType::kFloat);
  float *po = out.getInternalData()->getData<float>(out.getInternalOffset());

  float scale = 1.0f / sqrtf(static_cast<float>(headDim));
  std::vector<float> scores(keyLength);
  for (int n = 0; n < batch; ++n) {
    for (int h = 0; h < numHeads; ++h) {
      int kvh = h / group;
      const float *kh = pk.data() + static_cast<size_t>(n * numKeyValueHeads + kvh) * keyLength * headDim;
      const float *vh = pv.data() + static_cast<size_t>(n * numKeyValueHeads + kvh) * keyLength * headDim;
      for (int i = 0; i < queryLength; ++i) {
        int position = i + keyLength - queryLength;
        int firstKey = 0;
        int lastKey = keyLength - 1;
        if (maskType == MaskType::kCausal) lastKey = position;
        if (maskType == MaskType::kBlockCausal || maskType == MaskType::kBlockDiagonal) {
          int block = position / blockSize;
          lastKey = std::min(keyLength - 1, block * blockSize + blockSize - 1);
          if (maskType == MaskType::kBlockDiagonal) firstKey = block * blockSize;
        }
        const float *qi = pq.data() + (static_cast<size_t>(n * numHeads + h) * queryLength + i) * headDim;
        float largest = -INFINITY;
        for (int j = firstKey; j <= lastKey; ++j) {
          float dot = 0.0f;
          for (int d = 0; d < headDim; ++d) dot += qi[d] * kh[j * headDim + d];
          scores[j] = dot * scale;
          largest = std::max(largest, scores[j]);
        }
        float sum = 0.0f;
        for (int j = firstKey; j <= lastKey; ++j) {
          scores[j] = expf(scores[j] - largest);
          sum += scores[j];
        }
        float *oi = po + (static_cast<size_t>(n * numHeads + h) * queryLength + i) * headDim;
        for (int j = firstKey; j <= lastKey; ++j) {
          for (int d = 0; d < headDim; ++d) oi[d] += scores[j] / sum * vh[j * headDim + d];
        }
      }
    }
  }
  return out;
}

/// The kernel itself under `maskType` in blocks of `blockSize`, in `dtype`, against the reference
/// on the same operands rounded as the kernel reads them. False where the kernel did not take it.
bool runBlockCase(
    int numHeads,
    int numKeyValueHeads,
    int queryLength,
    int keyLength,
    int headDim,
    MaskType maskType,
    int blockSize,
    DType dtype = DType::kFloat16) {
  // In [0, 1), as the CUDA test draws them: a softmax that is not all on one key.
  auto drawn = [dtype](std::initializer_list<int> shape) {
    return toMetal(fl::test::rand(Device::getCpu(), shape, DType::kFloat), dtype);
  };
  Tensor q = drawn({2, numHeads, queryLength, headDim});
  Tensor k = drawn({2, numKeyValueHeads, keyLength, headDim});
  Tensor v = drawn({2, numKeyValueHeads, keyLength, headDim});
  Tensor reference = maskReference(toCpu(q), toCpu(k), toCpu(v), maskType, blockSize);
  float tol = dtype == DType::kFloat ? 1e-4f : 5e-3f;
  bool good = true;
  eachKernel([&](bool nax) {
    Tensor x = F::empty(Device::getMetal(), {2, numHeads, queryLength, headDim}, dtype);
    bool ran = op::metal_native::attention(q, k, v, maskType, blockSize, x);
    if (!ran || !close(x, reference, tol, tol)) {
      CATCH_UNSCOPED_INFO("failed with nax = " << nax);
      good = false;
    }
  });
  return good;
}

}  // namespace

CATCH_TEST_CASE("test Metal attention (block-causal mask)", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  // Blocks of one -- the causal mask, against a reference that is not the composition -- of a few
  // keys, of the 50 frames CosyVoice3's flow streams in, of exactly two of the kernel's blocks of
  // 32 keys, and larger than every key there is. Query lengths that are a decode step, a short
  // run, a threadgroup's worth and past it; key lengths equal to them, past them by a little, and
  // past the most keys the kernel takes without a block mask.
  for (int blockSize : {1, 2, 7, 50, 64, 1000}) {
    for (int headDim : {32, 64, 128}) {
      for (int queryLength : {1, 17, 32, 130}) {
        for (int keyLength : {queryLength, queryLength + 37, 300}) {
          if (keyLength < queryLength) continue;
          CATCH_INFO(
              "block = " << blockSize << ", headDim = " << headDim << ", L = " << queryLength
                         << ", S = " << keyLength);
          CATCH_REQUIRE(runBlockCase(
              4, 2, queryLength, keyLength, headDim, MaskType::kBlockCausal, blockSize));
        }
      }
    }
  }

  // In float, at the widths the kernel has in float.
  for (int headDim : {32, 64}) {
    for (int keyLength : {49, 50, 51, 130, 300}) {
      CATCH_INFO("float, headDim = " << headDim << ", S = " << keyLength);
      CATCH_REQUIRE(runBlockCase(
          4, 2, keyLength, keyLength, headDim, MaskType::kBlockCausal, 50, DType::kFloat));
      CATCH_REQUIRE(runBlockCase(
          4, 2, 17, keyLength, headDim, MaskType::kBlockCausal, 50, DType::kFloat));
    }
  }

  // CosyVoice3's DiT: sixteen heads of 64 in blocks of 50, over a prompt and a sentence of about
  // fourteen seconds of frames.
  CATCH_REQUIRE(runBlockCase(16, 16, 700, 700, 64, MaskType::kBlockCausal, 50));
}

CATCH_TEST_CASE("test Metal attention (block-diagonal mask)", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  // Self-attention in windows: blocks of one -- every query its own key alone -- of a few keys,
  // of exactly the kernel's 32 and past it, of Qwen3-ASR's 104, and larger than every key there
  // is, which is no mask. Lengths a single query, a block of keys and one past it, a short last
  // window and a long run.
  for (int blockSize : {1, 7, 32, 50, 104, 1000}) {
    for (int headDim : {32, 64, 128}) {
      for (int length : {1, 17, 32, 33, 130, 179, 300}) {
        CATCH_INFO("block = " << blockSize << ", headDim = " << headDim << ", L = S = " << length);
        CATCH_REQUIRE(
            runBlockCase(4, 2, length, length, headDim, MaskType::kBlockDiagonal, blockSize));
      }
    }
  }
  for (int length : {45, 104, 179, 300}) {
    CATCH_INFO("float, L = S = " << length);
    CATCH_REQUIRE(
        runBlockCase(4, 2, length, length, 64, MaskType::kBlockDiagonal, 104, DType::kFloat));
  }

  // Through MetalOperators::attention, which takes it to the kernel: the queries as a model hands
  // them in, (B, L, H, D) viewed as (B, H, L, D).
  Tensor q = fl::test::rand(Device::getCpu(), {1, 179, 14, 64}, DType::kFloat);
  Tensor k = fl::test::rand(Device::getCpu(), {1, 14, 179, 64}, DType::kFloat);
  Tensor v = fl::test::rand(Device::getCpu(), {1, 14, 179, 64}, DType::kFloat);
  Tensor x = fl::test::attention(
      mtl(),
      toMetal(q, DType::kFloat16).transpose(1, 2),
      toMetal(k, DType::kFloat16),
      toMetal(v, DType::kFloat16),
      MaskType::kBlockDiagonal,
      104);
  Tensor qc = fl::test::contiguous(cpu(), q.transpose(1, 2));
  CATCH_REQUIRE(close(x, maskReference(qc, k, v, MaskType::kBlockDiagonal, 104), 5e-3f, 5e-3f));
}

CATCH_TEST_CASE("test Metal attention (a block past every key is no mask)", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  // With as many queries as keys and a block at least as long as the keys, every query sees
  // every key under either block mask: the same work as no mask at all, to the bit.
  auto drawn = [](std::initializer_list<int> shape) {
    return toMetal(fl::test::rand(Device::getCpu(), shape, DType::kFloat), DType::kFloat16);
  };
  Tensor q = drawn({2, 8, 200, 64});
  Tensor k = drawn({2, 8, 200, 64});
  Tensor v = drawn({2, 8, 200, 64});
  eachKernel([&](bool nax) {
    CATCH_INFO("nax = " << nax);
    Tensor unmasked = F::empty(Device::getMetal(), {2, 8, 200, 64}, DType::kFloat16);
    Tensor blocked = F::empty(Device::getMetal(), {2, 8, 200, 64}, DType::kFloat16);
    Tensor windowed = F::empty(Device::getMetal(), {2, 8, 200, 64}, DType::kFloat16);
    CATCH_REQUIRE(op::metal_native::attention(q, k, v, MaskType::kNone, 0, unmasked));
    CATCH_REQUIRE(op::metal_native::attention(q, k, v, MaskType::kBlockCausal, 200, blocked));
    CATCH_REQUIRE(op::metal_native::attention(q, k, v, MaskType::kBlockDiagonal, 200, windowed));
    // allClose's bounds are strict, so "to the bit" is a difference below the smallest float.
    CATCH_REQUIRE(cpu()->allClose(toCpu(unmasked), toCpu(blocked), 0.0f, 1e-38f));
    CATCH_REQUIRE(cpu()->allClose(toCpu(unmasked), toCpu(windowed), 0.0f, 1e-38f));
  });
}

CATCH_TEST_CASE("test Metal attention (scores past half's range)", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  // CosyVoice3's DiT: a channel or two every query and key carry large, so that every score is
  // past what half holds -- 97,000 there -- while the softmax, which a constant added to a row does
  // not move, is decided by the rest. Here the last two channels are 300 in the queries and 1500 in
  // the keys, both exact in half: scores of 112,500 with a head of 64, 79,500 with 128. In float
  // that is an ordinary attention; scores held in half are infinities, and their softmax NaN --
  // which is what the composition did before it kept them in float.
  auto sunk = [](int length, int headDim, float big) {
    Tensor x = randn({1, 2, length, headDim});
    float *data = x.getInternalData()->getData<float>(x.getInternalOffset());
    for (int row = 0; row < 2 * length; ++row) {
      data[row * headDim + headDim - 1] = big;
      data[row * headDim + headDim - 2] = big;
    }
    return x;
  };
  struct Case {
    MaskType mask;
    int block;
  };
  const Case masks[] = {{MaskType::kNone, 0}, {MaskType::kCausal, 0}, {MaskType::kBlockCausal, 50}};
  // Heads the kernels take, and one as wide as a VAE's, which only the composition does; its keys
  // carry 4500 -- exact in half too -- for scores past half there as well.
  for (int headDim : {64, 128, 512}) {
    float big = headDim == 512 ? 4500.0f : 1500.0f;
    // A length the kernels have always taken, and one past the 256 keys the composition took
    // without a block mask, which only a kernel on the neural accelerators takes now.
    for (int length : {100, 600}) {
      Tensor q = toMetal(sunk(length, headDim, 300.0f), DType::kFloat16);
      Tensor k = toMetal(sunk(length, headDim, big), DType::kFloat16);
      Tensor v = toMetal(randn({1, 2, length, headDim}), DType::kFloat16);
      for (const Case &c : masks) {
        // The composition has no block mask, and a head of 512 has no kernel.
        if (headDim == 512 && c.mask == MaskType::kBlockCausal) continue;
        Tensor reference = maskReference(toCpu(q), toCpu(k), toCpu(v), c.mask, c.block);
        auto check = [&](const Tensor &x) {
          std::vector<float> got = values(x);
          CATCH_REQUIRE(std::all_of(got.begin(), got.end(), [](float y) { return std::isfinite(y); }));
          CATCH_REQUIRE(close(x, reference, 2e-2f, 2e-2f));
        };
        eachKernel([&](bool nax) {
          CATCH_INFO(
              "nax = " << nax << ", headDim = " << headDim << ", length = " << length
                       << ", mask = " << static_cast<int>(c.mask));
          check(fl::test::attention(mtl(), q, k, v, c.mask, c.block));
          // And the composition itself, which keeps its scores in float in half.
          if (c.mask != MaskType::kBlockCausal) {
            Tensor x = F::empty(Device::getMetal(), q.getShape(), DType::kFloat16);
            mtl()->Operators::attention(q, k, v, c.mask, c.block, x);
            check(x);
          }
        });
      }
    }
  }
}

CATCH_TEST_CASE("test Metal attention on the neural accelerators answers here", "[op][metal]") {
  SKIP_WITHOUT_METAL();
  // A GPU the tests run on that has the accelerators is one the kernel was written for: were its
  // layout another, attention would quietly run the other kernel, and every test above would pass
  // without asking the one on the accelerators anything.
  if (!op::metal_native::Context::get()->hasNax()) CATCH_SKIP("no neural accelerators here");
  CATCH_REQUIRE(op::metal_native::naxAttentionAnswers());
}

CATCH_TEST_CASE("test Metal attention refuses the masks it does not define", "[op][metal]") {
  SKIP_WITHOUT_METAL();

  // Shapes the kernel takes -- half and float, heads of 64 and 32, a few dozen keys -- so that a
  // mask that is not one reaches the kernel's dispatch rather than the composition's, where a
  // block size was once run as no mask at all.
  Tensor qh = toMetal(randn({1, 2, 16, 64}), DType::kFloat16);
  Tensor kh = toMetal(randn({1, 2, 16, 64}), DType::kFloat16);
  Tensor shorter = toMetal(randn({1, 2, 8, 64}), DType::kFloat16);
  Tensor longer = toMetal(randn({1, 2, 24, 64}), DType::kFloat16);
  CATCH_REQUIRE_THROWS(fl::test::attention(mtl(), qh, kh, kh, MaskType::kNone, 4));
  CATCH_REQUIRE_THROWS(fl::test::attention(mtl(), qh, kh, kh, static_cast<MaskType>(7)));
  CATCH_REQUIRE_THROWS(fl::test::attention(mtl(), qh, kh, kh, MaskType::kBlockCausal, 0));
  CATCH_REQUIRE_THROWS(fl::test::attention(mtl(), qh, shorter, shorter, MaskType::kBlockCausal, 4));
  CATCH_REQUIRE_THROWS(fl::test::attention(mtl(), qh, longer, longer, MaskType::kBlockDiagonal, 4));
  // Both block masks are the kernel's, here.
  CATCH_REQUIRE_NOTHROW(fl::test::attention(mtl(), qh, kh, kh, MaskType::kBlockCausal, 4));
  CATCH_REQUIRE_NOTHROW(fl::test::attention(mtl(), qh, kh, kh, MaskType::kBlockDiagonal, 4));

  Tensor qf = toMetal(randn({2, 4, 37, 32}));
  Tensor kf = toMetal(randn({2, 2, 37, 32}));
  CATCH_REQUIRE_THROWS(fl::test::attention(mtl(), qf, kf, kf, MaskType::kCausal, 1));

  // A head the kernel was not built for goes to the composition, which has no block mask.
  Tensor q80 = toMetal(randn({1, 2, 16, 80}), DType::kFloat16);
  CATCH_REQUIRE_NOTHROW(fl::test::attention(mtl(), q80, q80, q80, MaskType::kCausal));
  CATCH_REQUIRE_THROWS(fl::test::attention(mtl(), q80, q80, q80, MaskType::kBlockCausal, 4));
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
    Tensor uniform = fl::test::rand(Device::getCpu(), c.shapeA, DType::kFloat);
    Tensor a = toMetal(plus(fl::test::mul(cpu(), uniform, 4.0f), -2.0f), DType::kFloat16);
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
        fl::test::toDevice(mtl(), codes, Device::getMetal()),
        fl::test::toDevice(mtl(), scales, Device::getMetal()),
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
      fl::test::lookup(mtl(), toMetal(table, DType::kFloat16), fl::test::toDevice(mtl(), ids, Device::getMetal())),
      fl::test::lookup(cpu(), table, ids),
      1e-3f,
      1e-3f));

  Tensor image = randn({2, 3, 4, 5});
  CATCH_REQUIRE(close(
      fl::test::upsampleNearest2d(mtl(), toMetal(image), 2),
      fl::test::upsampleNearest2d(cpu(), image, 2),
      1e-6f,
      1e-6f));

  Tensor frames = randn({2, 3, 90});
  CATCH_REQUIRE(close(
      fl::test::upsampleNearest1d(mtl(), toMetal(frames), 154),
      fl::test::upsampleNearest1d(cpu(), frames, 154),
      1e-6f,
      1e-6f));
  CATCH_REQUIRE(close(
      fl::test::upsampleNearest1d(mtl(), toMetal(frames), 41),
      fl::test::upsampleNearest1d(cpu(), frames, 41),
      1e-6f,
      1e-6f));

  Tensor gated = randn({3, 4, 16});
  CATCH_REQUIRE(close(fl::test::geglu(mtl(), toMetal(gated)), fl::test::geglu(cpu(), gated), 1e-5f, 1e-5f));
  CATCH_REQUIRE(close(fl::test::swiglu(mtl(), toMetal(gated)), fl::test::swiglu(cpu(), gated), 1e-5f, 1e-5f));

  Tensor a = randn({3, 4});
  Tensor b = randn({3, 2});
  CATCH_REQUIRE(close(fl::test::cat(mtl(), toMetal(a), toMetal(b), 1), fl::test::cat(cpu(), a, b, 1), 1e-6f, 1e-6f));
  CATCH_REQUIRE(close(
      fl::test::contiguous(mtl(), toMetal(image).transpose(1, 3)),
      fl::test::contiguous(cpu(), image.transpose(1, 3)),
      1e-6f,
      1e-6f));

  Tensor zeros = fl::test::zeros(Device(Device::kMetal), {3, 5}, DType::kFloat16);
  CATCH_REQUIRE(close(zeros, fl::test::zeros(Device::getCpu(), {3, 5}, DType::kFloat), 1e-6f, 1e-6f));

  Tensor filled = F::empty(Device(Device::kMetal), {4, 6}, DType::kFloat);
  mtl()->fill(filled, 2.5f);
  mtl()->fill(filled.slice(1, {2, 4}), -1.0f);
  Tensor expected = F::empty(Device::getCpu(), {4, 6}, DType::kFloat);
  cpu()->fill(expected, 2.5f);
  cpu()->fill(expected.slice(1, {2, 4}), -1.0f);
  CATCH_REQUIRE(close(filled, expected, 1e-6f, 1e-6f));

  Tensor mask = fl::test::causalMask(Device(Device::kMetal), 5);
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
      fl::test::toDevice(mtl(), positions, Device::getMetal()),
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
  fl::test::rand(Device(Device::kMetal), {5}, DType::kFloat);  // moves the counter on by two blocks
  std::vector<float> uniform = values(fl::test::rand(Device(Device::kMetal), {9}, DType::kFloat));
  for (int i = 0; i < 9; ++i) {
    uint32_t bits[4];
    philox(1234, 2 + i / 4, bits);
    CATCH_REQUIRE(uniform[i] == bits[i % 4] * kTwoPow32Inv + kTwoPow32Inv / 2.0f);
  }

  // Normal, which only has to agree to within what the device's transcendentals round to.
  mtl()->manualSeed(99);
  Tensor normal = fl::test::randNormal(Device(Device::kMetal), {2, 50000});
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
    Tensor big = fl::test::zeros(Device(Device::kMetal), {64, 1024, 1024}, DType::kFloat16);
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

  // What the budget says under it: the limit, and what the tensors hold against it.
  MemoryBudget limited = allocator.getMemoryBudget();
  CATCH_REQUIRE(limited.capacity == allocator.getMemoryLimit());
  CATCH_REQUIRE(limited.allocated == allocator.getAllocatedBytes());

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
    Tensor a = fl::test::cast(mtl(), fl::test::rand(Device(Device::kMetal), c.a, DType::kFloat), DType::kFloat16);
    Tensor b = fl::test::cast(mtl(), fl::test::rand(Device(Device::kMetal), c.b, DType::kFloat), DType::kFloat16);
    Tensor bt = c.transposeB ? b.transpose(-1, -2) : b;
    double seconds = time([&]() { fl::test::matmul(mtl(), a, bt); });
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
      // C * 9 not a multiple of the reduction's chunk: a VAE's convolutions in from the latent or
      // the picture.
      {"conv 3x3 1x4x128 f16", 1, 4, 128, 512, DType::kFloat16},
      {"conv 3x3 1x16x128 f16", 1, 16, 128, 512, DType::kFloat16},
      {"conv 3x3 1x3x1024 f16", 1, 3, 1024, 128, DType::kFloat16},
  };
  for (const ConvCase &c : convs) {
    Tensor x = fl::test::cast(mtl(), fl::test::rand(Device(Device::kMetal), {c.N, c.C, c.H, c.H}, DType::kFloat), c.dtype);
    Tensor w = fl::test::cast(mtl(), fl::test::rand(Device(Device::kMetal), {c.K, c.C, 3, 3}, DType::kFloat), c.dtype);
    double seconds = time([&]() { fl::test::conv2d(mtl(), x, w, Tensor(), 1, 1, 1, 1); });
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
    Tensor weight = fl::test::toDevice(mtl(), codes, Device::getMetal());
    Tensor scales = F::empty(Device::getMetal(), {1}, DType::kFloat);
    mtl()->fill(scales, 0.01f);
    Tensor a = toMetal(fl::test::rand(Device::getCpu(), {c.m, c.k}, DType::kFloat), DType::kFloat16);
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
