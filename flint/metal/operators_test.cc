// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
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

#include <algorithm>
#include <cmath>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"

namespace fl {
namespace op {
namespace metal {
namespace {

// The Metal operators run in float16, the same as CUDA's, so the reference the results are held
// against is the float32 CPU one and the tolerances are the ones the CUDA tests use.
Tensor toMetal(const Tensor &a) {
  return F::cast(F::toDevice(Device::getMetal(), a), DType::kFloat16);
}

Tensor toCpu(const Tensor &a) {
  return F::toDevice(Device::getCpu(), F::cast(a, DType::kFloat));
}

/// Read a tensor out as plain floats, so a test can hold a result against a reference worked out
/// here rather than against another backend. layerNorm, groupNorm, conv2d and upsampleNearest2d
/// have no CPU implementation to compare with -- they are CUDA-only -- and a reference written
/// out longhand is a stricter check anyway: it shares no code with what it is testing.
std::vector<float> readFloats(const Tensor &a) {
  Tensor c = F::contiguous(toCpu(a));
  const float *data = c.getInternalData()->getData<float>(c.getInternalOffset());
  return std::vector<float>(data, data + c.getNumEl());
}

void requireClose(
    const std::vector<float> &actual,
    const std::vector<float> &expected,
    float tolerance) {
  CATCH_REQUIRE(actual.size() == expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    CATCH_INFO("element " << i << ": " << actual[i] << " vs " << expected[i]);
    CATCH_REQUIRE(std::fabs(actual[i] - expected[i]) < tolerance);
  }
}

}  // namespace

CATCH_TEST_CASE("test Metal device transfer round trip", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({3, 7, 5}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(F::toDevice(Device::getCpu(), F::toDevice(Device::getMetal(), a)), a));

  // Half the point of the backend is that a fp32 tensor can go over, be worked on in fp16 and
  // come back; the round trip through both dtypes is what the operator tests below rely on.
  CATCH_REQUIRE(F::allClose(toCpu(toMetal(a)), a, 5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal binary operators", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({2, 5, 10}, DType::kFloat);
  Tensor b = F::rand({5}, DType::kFloat);

  // Transposed and then sliced: a non-contiguous view with an offset, which is what the
  // as_strided bridge has to carry over to MLX intact. A backend that quietly made its inputs
  // contiguous would pass a dense test and fail this one.
  Tensor at = a.transpose(2, 1).slice(1, {1, 9});
  Tensor xt = toMetal(a).transpose(2, 1).slice(1, {1, 9});
  Tensor y = toMetal(b);

  CATCH_REQUIRE(F::allClose(toCpu(F::add(xt, y)), F::add(at, b), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::sub(xt, y)), F::sub(at, b), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::mul(xt, y)), F::mul(at, b), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::mul(xt, 0.1f)), F::mul(at, 0.1f), 1e-3, 1e-4));
  CATCH_REQUIRE(F::allClose(toCpu(F::div(xt, 4.0f)), F::div(at, 4.0f), 1e-3, 1e-4));
}

CATCH_TEST_CASE("test Metal unary operators", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({2, 5, 10}, DType::kFloat);
  Tensor at = a.transpose(2, 1).slice(1, {1, 9});
  Tensor xt = toMetal(a).transpose(2, 1).slice(1, {1, 9});

  CATCH_REQUIRE(F::allClose(toCpu(F::neg(xt)), F::neg(at), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::exp(xt)), F::exp(at), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::sqrt(xt)), F::sqrt(at), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::sigmoid(xt)), F::sigmoid(at), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::tanh(xt)), F::tanh(at), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::gelu(xt)), F::gelu(at), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::silu(xt)), F::silu(at), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::quickGelu(xt)), F::quickGelu(at), 5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal matmul", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({10, 20}, DType::kFloat);
  Tensor b = F::rand({20, 40}, DType::kFloat);
  CATCH_REQUIRE(
      F::allClose(toCpu(F::matmul(toMetal(a), toMetal(b))), F::matmul(a, b), 5e-2, 5e-2));

  // Batched, and against a transposed right hand side, which is how attention projections and
  // the SDXL linear layers actually call it.
  Tensor c = F::rand({5, 10, 20}, DType::kFloat);
  Tensor d = F::rand({40, 20}, DType::kFloat);
  CATCH_REQUIRE(
      F::allClose(
          toCpu(F::matmul(toMetal(c), toMetal(d).transpose(-1, -2))),
          F::matmul(c, d.transpose(-1, -2)),
          5e-2,
          5e-2));
}

CATCH_TEST_CASE("test Metal softmax and sum", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({4, 6, 8}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(toCpu(F::softmax(toMetal(a))), F::softmax(a), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::sum(toMetal(a), -1)), F::sum(a, -1), 5e-2, 5e-2));
}

CATCH_TEST_CASE("test Metal layerNorm", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  constexpr int kRows = 3;
  constexpr int kCols = 16;
  constexpr float kEps = 1e-5f;

  Tensor a = F::rand({kRows, kCols}, DType::kFloat);
  Tensor weight = F::rand({kCols}, DType::kFloat);
  Tensor bias = F::rand({kCols}, DType::kFloat);

  std::vector<float> x = readFloats(a);
  std::vector<float> w = readFloats(weight);
  std::vector<float> b = readFloats(bias);

  std::vector<float> expected(x.size());
  for (int row = 0; row < kRows; ++row) {
    const float *r = x.data() + row * kCols;
    float mean = 0.0f;
    for (int i = 0; i < kCols; ++i) mean += r[i];
    mean /= kCols;

    float var = 0.0f;
    for (int i = 0; i < kCols; ++i) var += (r[i] - mean) * (r[i] - mean);
    var /= kCols;

    float scale = 1.0f / std::sqrt(var + kEps);
    for (int i = 0; i < kCols; ++i) {
      expected[row * kCols + i] = (r[i] - mean) * scale * w[i] + b[i];
    }
  }

  Tensor got = F::layerNorm(toMetal(a), toMetal(weight), toMetal(bias), kEps);
  requireClose(readFloats(got), expected, 5e-3f);
}

CATCH_TEST_CASE("test Metal groupNorm", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  constexpr int kBatch = 2;
  constexpr int kChannels = 8;
  constexpr int kGroups = 4;
  constexpr int kHeight = 3;
  constexpr int kWidth = 3;
  constexpr int kSpatial = kHeight * kWidth;
  constexpr int kPerGroup = kChannels / kGroups;
  constexpr float kEps = 1e-5f;

  Tensor a = F::rand({kBatch, kChannels, kHeight, kWidth}, DType::kFloat);
  Tensor weight = F::rand({kChannels}, DType::kFloat);
  Tensor bias = F::rand({kChannels}, DType::kFloat);

  std::vector<float> x = readFloats(a);
  std::vector<float> w = readFloats(weight);
  std::vector<float> b = readFloats(bias);

  std::vector<float> expected(x.size());
  for (int n = 0; n < kBatch; ++n) {
    for (int g = 0; g < kGroups; ++g) {
      // Each group covers kPerGroup channels and all the space under them, and is normalized
      // over that whole block together -- that is what makes it group rather than layer norm.
      int base = n * kChannels * kSpatial + g * kPerGroup * kSpatial;
      int count = kPerGroup * kSpatial;

      float mean = 0.0f;
      for (int i = 0; i < count; ++i) mean += x[base + i];
      mean /= count;

      float var = 0.0f;
      for (int i = 0; i < count; ++i) var += (x[base + i] - mean) * (x[base + i] - mean);
      var /= count;

      float scale = 1.0f / std::sqrt(var + kEps);
      for (int i = 0; i < count; ++i) {
        int channel = g * kPerGroup + i / kSpatial;
        expected[base + i] = (x[base + i] - mean) * scale * w[channel] + b[channel];
      }
    }
  }

  Tensor got = F::groupNorm(toMetal(a), toMetal(weight), toMetal(bias), kGroups, kEps);
  requireClose(readFloats(got), expected, 5e-3f);
}

CATCH_TEST_CASE("test Metal conv2d", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  constexpr int kBatch = 2;
  constexpr int kInC = 3;
  constexpr int kOutC = 4;
  constexpr int kH = 5;
  constexpr int kW = 5;
  constexpr int kK = 3;      // square kernel
  constexpr int kPad = 1;    // same padding, so the output keeps kH x kW

  Tensor input = F::rand({kBatch, kInC, kH, kW}, DType::kFloat);
  Tensor weight = F::rand({kOutC, kInC, kK, kK}, DType::kFloat);
  Tensor bias = F::rand({kOutC}, DType::kFloat);

  std::vector<float> x = readFloats(input);
  std::vector<float> w = readFloats(weight);
  std::vector<float> b = readFloats(bias);

  // The definition, written out: this is what the NCHW-to-NHWC transposes around MLX's conv2d
  // have to come back to.
  std::vector<float> expected(kBatch * kOutC * kH * kW, 0.0f);
  for (int n = 0; n < kBatch; ++n) {
    for (int oc = 0; oc < kOutC; ++oc) {
      for (int oh = 0; oh < kH; ++oh) {
        for (int ow = 0; ow < kW; ++ow) {
          float sum = b[oc];
          for (int ic = 0; ic < kInC; ++ic) {
            for (int r = 0; r < kK; ++r) {
              for (int c = 0; c < kK; ++c) {
                int ih = oh + r - kPad;
                int iw = ow + c - kPad;
                if (ih < 0 || ih >= kH || iw < 0 || iw >= kW) continue;
                sum += x[((n * kInC + ic) * kH + ih) * kW + iw] *
                       w[((oc * kInC + ic) * kK + r) * kK + c];
              }
            }
          }
          expected[((n * kOutC + oc) * kH + oh) * kW + ow] = sum;
        }
      }
    }
  }

  Tensor got = F::conv2d(toMetal(input), toMetal(weight), toMetal(bias), 1, kPad, 1, 1);
  requireClose(readFloats(got), expected, 5e-2f);
}

CATCH_TEST_CASE("test Metal attention", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor q = F::rand({2, 4, 8, 16}, DType::kFloat);
  Tensor k = F::rand({2, 4, 8, 16}, DType::kFloat);
  Tensor v = F::rand({2, 4, 8, 16}, DType::kFloat);

  CATCH_REQUIRE(
      F::allClose(
          toCpu(F::attention(toMetal(q), toMetal(k), toMetal(v), false)),
          F::attention(q, k, v, false),
          5e-2,
          5e-2));
}

CATCH_TEST_CASE("test Metal upsampleNearest2d", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  constexpr int kBatch = 2;
  constexpr int kChannels = 3;
  constexpr int kH = 4;
  constexpr int kW = 5;
  constexpr int kScale = 2;

  Tensor image = F::rand({kBatch, kChannels, kH, kW}, DType::kFloat);
  std::vector<float> x = readFloats(image);

  std::vector<float> expected(kBatch * kChannels * kH * kScale * kW * kScale);
  for (int i = 0; i < kBatch * kChannels; ++i) {
    for (int oh = 0; oh < kH * kScale; ++oh) {
      for (int ow = 0; ow < kW * kScale; ++ow) {
        expected[(i * kH * kScale + oh) * kW * kScale + ow] =
            x[(i * kH + oh / kScale) * kW + ow / kScale];
      }
    }
  }

  Tensor got = F::upsampleNearest2d(toMetal(image), kScale);
  requireClose(readFloats(got), expected, 5e-3f);
}

CATCH_TEST_CASE("test Metal geglu", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor gated = F::rand({3, 5, 16}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(toCpu(F::geglu(toMetal(gated))), F::geglu(gated), 5e-3, 5e-3));
}

CATCH_TEST_CASE("test Metal lookup", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor table = F::rand({20, 8}, DType::kFloat);
  Tensor indices = Tensor::create<LongType>({2, 3}, {0, 5, 19, 3, 11, 7});

  CATCH_REQUIRE(
      F::allClose(
          toCpu(F::lookup(toMetal(table), F::toDevice(Device::getMetal(), indices))),
          F::lookup(table, indices),
          5e-3,
          5e-3));
}


// Every operator the SDXL pipeline actually reaches for, in one place: if this passes, the
// question "can the model run on Metal" is down to composition rather than coverage.
CATCH_TEST_CASE("test Metal covers what SDXL calls", "[op][metal]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  Tensor a = F::rand({2, 4, 6}, DType::kFloat);
  Tensor b = F::rand({2, 4, 6}, DType::kFloat);
  Tensor xa = toMetal(a);
  Tensor xb = toMetal(b);

  CATCH_SECTION("cat") {
    CATCH_REQUIRE(F::allClose(toCpu(F::cat(xa, xb, -1)), F::cat(a, b, -1), 5e-3, 5e-3));
    CATCH_REQUIRE(F::allClose(toCpu(F::cat(xa, xb, 1)), F::cat(a, b, 1), 5e-3, 5e-3));
  }

  CATCH_SECTION("contiguous of a view") {
    CATCH_REQUIRE(
        F::allClose(
            toCpu(F::contiguous(xa.transpose(0, 2))),
            F::contiguous(a.transpose(0, 2)),
            5e-3,
            5e-3));
  }

  CATCH_SECTION("view and unsqueeze") {
    CATCH_REQUIRE(F::allClose(toCpu(xa.view({2, 24})), a.view({2, 24}), 5e-3, 5e-3));
    CATCH_REQUIRE(F::allClose(toCpu(xa.unsqueeze(1)), a.unsqueeze(1), 5e-3, 5e-3));
  }

  CATCH_SECTION("randn and manualSeed") {
    F::manualSeed(Device::getMetal(), 42);
    Tensor noise = F::randn({2, 3, 4}, Device::getMetal());
    CATCH_REQUIRE(noise.getDevice().getType() == Device::kMetal);
    CATCH_REQUIRE(noise.getNumEl() == 24);
  }
}

CATCH_TEST_CASE("test Metal at VAE scale in fp16", "[op][metal][probe]") {
  if (!isOperatorsAvailable(Device::kMetal)) CATCH_SKIP("metal device not available");

  auto countNaN = [](const Tensor &t) {
    std::vector<float> v = readFloats(t);
    return std::count_if(v.begin(), v.end(), [](float x) { return std::isnan(x); });
  };

  CATCH_SECTION("groupNorm over a large plane") {
    // A VAE decoder normalizes 128 channels over 256x256 of space in 32 groups. The sum of
    // squares behind the variance is over 260k elements per group, which is where fp16 runs out.
    Tensor a = F::rand({1, 128, 256, 256}, DType::kFloat);
    Tensor w = F::rand({128}, DType::kFloat);
    Tensor b = F::rand({128}, DType::kFloat);

    Tensor got = F::groupNorm(toMetal(a), toMetal(w), toMetal(b), 32, 1e-5);
    CATCH_INFO("groupNorm NaN count " << countNaN(got));
    CATCH_REQUIRE(countNaN(got) == 0);
  }

  CATCH_SECTION("layerNorm over a wide row") {
    Tensor a = F::rand({2, 64, 8192}, DType::kFloat);
    Tensor w = F::rand({8192}, DType::kFloat);
    Tensor b = F::rand({8192}, DType::kFloat);
    Tensor got = F::layerNorm(toMetal(a), toMetal(w), toMetal(b), 1e-5);
    CATCH_INFO("layerNorm NaN count " << countNaN(got));
    CATCH_REQUIRE(countNaN(got) == 0);
  }

  CATCH_SECTION("attention over a long sequence") {
    // The VAE attends over every position of its feature map at once.
    Tensor q = F::rand({1, 1, 4096, 64}, DType::kFloat);
    Tensor k = F::rand({1, 1, 4096, 64}, DType::kFloat);
    Tensor v = F::rand({1, 1, 4096, 64}, DType::kFloat);

    Tensor got = F::attention(toMetal(q), toMetal(k), toMetal(v), false);
    CATCH_INFO("attention NaN count " << countNaN(got));
    CATCH_REQUIRE(countNaN(got) == 0);
  }
}

}  // namespace metal
}  // namespace op
}  // namespace fl
