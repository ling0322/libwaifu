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

#include <math.h>
#include <string.h>

#include <functional>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/half.h"
#include "flint/test_functional.h"
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

// the CPU fp16 kernels are checked against the fp32 kernels on the same device.
Tensor toFp16(const Tensor &a) {
  return F::cast(cpuOps(), a, DType::kFloat16);
}

Tensor toFp32(const Tensor &a) {
  return F::cast(cpuOps(), a, DType::kFloat);
}

}  // namespace

CATCH_TEST_CASE("test CPU fp16 binary operators", "[op][cpu][float16]") {
  Tensor a = F::rand(Device::getCpu(), {2, 5, 10}, DType::kFloat);
  Tensor b = F::rand(Device::getCpu(), {5}, DType::kFloat);
  Tensor at = a.transpose(2, 1).slice(1, {1, 9});
  Tensor xt = toFp16(a).transpose(2, 1).slice(1, {1, 9});
  Tensor y = toFp16(b);

  CATCH_REQUIRE(cpuOps()->allClose(toFp32(F::add(cpuOps(), xt, y)), F::add(cpuOps(), at, b), 5e-3));
  CATCH_REQUIRE(cpuOps()->allClose(toFp32(F::mul(cpuOps(), xt, y)), F::mul(cpuOps(), at, b), 5e-3));
}

CATCH_TEST_CASE("test CPU fp16 copy operators", "[op][cpu][float16]") {
  Tensor a = F::rand(Device::getCpu(), {2, 10, 50}, DType::kFloat);
  Tensor x = toFp16(a).transpose(1, 0);
  Tensor dest = F::emptyLike(x);
  cpuOps()->copy(x, dest);
  CATCH_REQUIRE(cpuOps()->allClose(toFp32(dest).transpose(1, 0), a));

  Tensor b = F::rand(Device::getCpu(), {10, 2, 5, 20}, DType::kFloat);
  Tensor expanded = toFp16(b).unsqueeze(1).expand({10, 4, 2, 5, 20});
  Tensor dest5d = F::emptyLike(expanded);
  cpuOps()->copy(expanded, dest5d);
  CATCH_REQUIRE(
      cpuOps()->allClose(
          toFp32(dest5d),
          F::contiguous(cpuOps(), b.unsqueeze(1).expand({10, 4, 2, 5, 20}))));
}

CATCH_TEST_CASE("test CPU fp16 matmul operators", "[op][cpu][float16]") {
  auto runCase = [](std::initializer_list<int> shapeA, std::initializer_list<int> shapeB) {
    Tensor a = F::rand(Device::getCpu(), shapeA, DType::kFloat);
    Tensor b = F::rand(Device::getCpu(), shapeB, DType::kFloat);
    Tensor xr = F::matmul(cpuOps(), a, b.slice(-1, {8, 28}).transpose(-1, -2));

    Tensor y = toFp16(b).slice(-1, {8, 28}).transpose(-1, -2);
    Tensor x = F::matmul(cpuOps(), toFp16(a), y);

    return cpuOps()->allClose(toFp32(x), xr, 5e-2);
  };

  CATCH_REQUIRE(runCase({10, 20}, {40, 30}));
  CATCH_REQUIRE(runCase({5, 10, 20}, {40, 30}));
  CATCH_REQUIRE(runCase({5, 10, 5, 20}, {10, 40, 30}));
}

CATCH_TEST_CASE("test CPU fp16 rmsNorm operator", "[op][cpu][float16]") {
  Tensor a = F::rand(Device::getCpu(), {2, 5, 10}, DType::kFloat);
  Tensor w = F::rand(Device::getCpu(), {10}, DType::kFloat);
  Tensor x = F::rmsNorm(cpuOps(), toFp16(a), toFp16(w), 1e-5);

  CATCH_REQUIRE(cpuOps()->allClose(toFp32(x), F::rmsNorm(cpuOps(), a, w, 1e-5), 5e-2));
}

CATCH_TEST_CASE("test CPU fp16 activation operators", "[op][cpu][float16]") {
  Tensor a = F::rand(Device::getCpu(), {2, 5, 150}, DType::kFloat);

  CATCH_REQUIRE(
      cpuOps()->allClose(toFp32(F::softmax(cpuOps(), toFp16(a))), F::softmax(cpuOps(), a), 5e-2));
  CATCH_REQUIRE(cpuOps()->allClose(toFp32(F::swiglu(cpuOps(), toFp16(a))), F::swiglu(cpuOps(), a), 5e-2));
}

namespace {

using HalfOp = std::function<Tensor(const std::vector<Tensor> &)>;

/// Runs `fn` on half inputs and on the very same values in float, and asks that the half answer
/// is half too and is the float one rounded. The inputs are rounded to half first, so what is
/// measured is the operator and not the rounding of what it was given. Inputs that are not float
/// -- indices -- are handed to both unchanged.
void requireHalfAgrees(
    const char *name,
    HalfOp fn,
    const std::vector<Tensor> &inputs,
    float rtol = 1e-2f,
    float atol = 1e-2f) {
  std::vector<Tensor> half, wide;
  for (const Tensor &t : inputs) {
    bool isFloat = !t.empty() && t.getDType() == DType::kFloat;
    Tensor h = isFloat ? toFp16(F::contiguous(cpuOps(), t)) : t;
    half.push_back(h);
    wide.push_back(isFloat ? toFp32(h) : t);
  }

  CATCH_INFO(name);
  Tensor got = fn(half);
  CATCH_REQUIRE(got.getDType() == DType::kFloat16);
  CATCH_REQUIRE(cpuOps()->allClose(toFp32(got), fn(wide), rtol, atol));
}

Tensor signedRand(std::initializer_list<int> shape) {
  return F::randNormal(Device::getCpu(), shape);
}

Tensor positiveRand(std::initializer_list<int> shape) {
  return F::exp(cpuOps(), F::randNormal(Device::getCpu(), shape));
}

}  // namespace

#if LUT_HALF_F16C
/// x64 converts with F16C, and the portable conversions are what it replaced. Half to float is
/// exact, so the two agree on every half. Float to half they agree on everything but a tie: F16C
/// takes it to the even half, as IEEE, aarch64 and torch do, where the portable code takes it away
/// from zero -- which differs on 31744 floats of the 2^32, checked once by brute force. A NaN only has to stay one:
/// F16C keeps its payload where the portable code makes it the one quiet NaN.
CATCH_TEST_CASE("test F16C conversions agree with the portable ones", "[op][cpu][float16]") {
  for (uint32_t h = 0; h < 65536; ++h) {
    float fast = lut::cvtsh_ss(static_cast<uint16_t>(h));
    float soft = lut::internal::cvtsh_ss_soft(static_cast<uint16_t>(h));
    CATCH_INFO("half " << h);
    if (isnan(soft)) {
      CATCH_REQUIRE(isnan(fast));
    } else {
      CATCH_REQUIRE(memcmp(&fast, &soft, sizeof(float)) == 0);
    }
  }

  // A float's mantissa in steps of 2^-13 is eighths of a normal half's last place, so this walks
  // round-down, tie and round-up for every exponent from below the subnormals to past the
  // largest half, both signs.
  int ties = 0;
  for (uint32_t bits = 0; bits < 0x80000000u; bits += 0x400u) {
    for (uint32_t sign : {0u, 0x80000000u}) {
      uint32_t u = bits | sign;
      float f;
      memcpy(&f, &u, sizeof(f));
      uint16_t fast = lut::cvtss_sh(f);
      uint16_t soft = lut::internal::cvtss_sh_soft(f);
      CATCH_INFO("float bits " << u);
      if (isnan(f)) {
        CATCH_REQUIRE((fast & 0x7c00) == 0x7c00);
        CATCH_REQUIRE((fast & 0x03ff) != 0);
      } else if (fast != soft) {
        // f is exactly between the two, F16C has the even one and the portable code the one
        // farther from zero.
        double x = fabs(double(f));
        double a = lut::cvtsh_ss(fast & 0x7fff), b = lut::cvtsh_ss(soft & 0x7fff);
        CATCH_REQUIRE((fast & 1) == 0);
        CATCH_REQUIRE(a < b);
        CATCH_REQUIRE(x - a == b - x);
        ++ties;
      }
    }
  }
  CATCH_REQUIRE(ties > 0);
}
#endif  // LUT_HALF_F16C

CATCH_TEST_CASE("test CPU fp16 is the default float type", "[op][cpu][float16]") {
  CATCH_REQUIRE(cpuOps()->getDefaultFloatType() == DType::kFloat16);

  Tensor mask = F::causalMask(Device::getCpu(), 5);
  CATCH_REQUIRE(mask.getDType() == DType::kFloat16);
  Tensor wide = toFp32(mask);
  const float *m = wide.getInternalData()->getData<float>(wide.getInternalOffset());
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      if (j <= i) {
        CATCH_REQUIRE(m[i * 5 + j] == 0.0f);
      } else {
        CATCH_REQUIRE(std::isinf(m[i * 5 + j]));
        CATCH_REQUIRE(m[i * 5 + j] < 0.0f);
      }
    }
  }

  Tensor filled = F::zeros(Device::getCpu(), {3, 7}, DType::kFloat16);
  cpuOps()->fill(filled, 1.5f);
  Tensor filledWide = toFp32(filled);
  const float *f = filledWide.getInternalData()->getData<float>(filledWide.getInternalOffset());
  for (int i = 0; i < 21; ++i) CATCH_REQUIRE(f[i] == 1.5f);
}

CATCH_TEST_CASE("test CPU fp16 unary operators", "[op][cpu][float16]") {
  Tensor x = signedRand({4, 37});
  Tensor p = positiveRand({4, 37});

  requireHalfAgrees("neg", [](auto &t) { return F::neg(cpuOps(), t[0]); }, {x});
  requireHalfAgrees("abs", [](auto &t) { return F::abs(cpuOps(), t[0]); }, {x});
  requireHalfAgrees("exp", [](auto &t) { return F::exp(cpuOps(), t[0]); }, {x});
  requireHalfAgrees("log", [](auto &t) { return F::log(cpuOps(), t[0]); }, {p});
  requireHalfAgrees("round", [](auto &t) { return F::round(cpuOps(), F::mul(cpuOps(), t[0], 4.0f)); }, {x});
  requireHalfAgrees("square", [](auto &t) { return F::square(cpuOps(), t[0]); }, {x});
  requireHalfAgrees("sqrt", [](auto &t) { return F::sqrt(cpuOps(), t[0]); }, {p});
  requireHalfAgrees("rsqrt", [](auto &t) { return F::rsqrt(cpuOps(), t[0]); }, {p});
  requireHalfAgrees("sigmoid", [](auto &t) { return F::sigmoid(cpuOps(), t[0]); }, {x});
  requireHalfAgrees("tanh", [](auto &t) { return F::tanh(cpuOps(), t[0]); }, {x});
  requireHalfAgrees("relu", [](auto &t) { return F::relu(cpuOps(), t[0]); }, {x});
  requireHalfAgrees("gelu", [](auto &t) { return F::gelu(cpuOps(), t[0]); }, {x});
  requireHalfAgrees("silu", [](auto &t) { return F::silu(cpuOps(), t[0]); }, {x});
  requireHalfAgrees("sin", [](auto &t) { return F::sin(cpuOps(), t[0]); }, {x});
  requireHalfAgrees("cos", [](auto &t) { return F::cos(cpuOps(), t[0]); }, {x});
  requireHalfAgrees("quickGelu", [](auto &t) { return F::quickGelu(cpuOps(), t[0]); }, {x});
  requireHalfAgrees("clamp", [](auto &t) { return F::clamp(cpuOps(), t[0], -0.5f, 0.75f); }, {x});
  requireHalfAgrees("mul scalar", [](auto &t) { return F::mul(cpuOps(), t[0], 3.0f); }, {x});
  requireHalfAgrees("div scalar", [](auto &t) { return F::div(cpuOps(), t[0], 3.0f); }, {x});
  requireHalfAgrees("geglu", [](auto &t) { return F::geglu(cpuOps(), t[0]); }, {x.slice(1, {0, 36})});
}

CATCH_TEST_CASE("test CPU fp16 binary operators, all four", "[op][cpu][float16]") {
  Tensor a = signedRand({3, 5, 16});
  Tensor b = positiveRand({16});

  requireHalfAgrees("add", [](auto &t) { return F::add(cpuOps(), t[0], t[1]); }, {a, b});
  requireHalfAgrees("sub", [](auto &t) { return F::sub(cpuOps(), t[0], t[1]); }, {a, b});
  requireHalfAgrees("mul", [](auto &t) { return F::mul(cpuOps(), t[0], t[1]); }, {a, b});
  requireHalfAgrees("divTensor", [](auto &t) { return F::divTensor(cpuOps(), t[0], t[1]); }, {a, b});
}

CATCH_TEST_CASE("test CPU fp16 reductions", "[op][cpu][float16]") {
  Tensor x = signedRand({3, 6, 64});

  requireHalfAgrees("sum last", [](auto &t) { return F::sum(cpuOps(), t[0], -1); }, {x}, 2e-2f, 5e-2f);
  requireHalfAgrees("sum first", [](auto &t) { return F::sum(cpuOps(), t[0], 0); }, {x}, 2e-2f, 5e-2f);
  requireHalfAgrees("cumsum", [](auto &t) { return F::cumsum(cpuOps(), t[0], 1); }, {x});
  requireHalfAgrees("max", [](auto &t) { return F::max(cpuOps(), t[0]); }, {x});
  requireHalfAgrees("min", [](auto &t) { return F::min(cpuOps(), t[0]); }, {x});
  requireHalfAgrees("softmax", [](auto &t) { return F::softmax(cpuOps(), t[0]); }, {x});
}

CATCH_TEST_CASE("test CPU fp16 normalizations and resampling", "[op][cpu][float16]") {
  Tensor x = signedRand({2, 8, 6, 5});
  Tensor w = positiveRand({8});
  Tensor b = signedRand({8});
  Tensor row = signedRand({4, 7, 40});
  Tensor rw = positiveRand({40});
  Tensor rb = signedRand({40});

  requireHalfAgrees(
      "layerNorm", [](auto &t) { return F::layerNorm(cpuOps(), t[0], t[1], t[2], 1e-5f); },
      {row, rw, rb}, 2e-2f, 2e-2f);
  requireHalfAgrees(
      "rmsNorm", [](auto &t) { return F::rmsNorm(cpuOps(), t[0], t[1], 1e-5f); }, {row, rw},
      2e-2f, 2e-2f);
  requireHalfAgrees(
      "groupNorm", [](auto &t) { return F::groupNorm(cpuOps(), t[0], t[1], t[2], 4, 1e-5f); },
      {x, w, b}, 2e-2f, 2e-2f);
  requireHalfAgrees(
      "upsampleNearest2d", [](auto &t) { return F::upsampleNearest2d(cpuOps(), t[0], 2); }, {x});
  requireHalfAgrees(
      "upsampleNearest1d", [](auto &t) { return F::upsampleNearest1d(cpuOps(), t[0], 17); },
      {row.slice(2, {0, 7})});
}

CATCH_TEST_CASE("test CPU fp16 signal operators", "[op][cpu][float16]") {
  Tensor x = signedRand({2, 4, 23});
  Tensor w = signedRand({6, 4, 3});
  Tensor tw = signedRand({4, 3, 4});
  Tensor b = signedRand({6});
  Tensor alpha = positiveRand({4});
  Tensor beta = positiveRand({4});

  requireHalfAgrees(
      "conv1d", [](auto &t) { return F::conv1d(cpuOps(), t[0], t[1], t[2], 2, 1, 1, 1); },
      {x, w, b}, 2e-2f, 2e-2f);
  requireHalfAgrees(
      "convTranspose1d",
      [](auto &t) { return F::convTranspose1d(cpuOps(), t[0], t[1], Tensor(), 2, 1, 1, 1); },
      {x, tw}, 2e-2f, 2e-2f);
  requireHalfAgrees(
      "snake", [](auto &t) { return F::snake(cpuOps(), t[0], t[1], t[2], 1e-6f); },
      {x, alpha, beta});
  requireHalfAgrees(
      "pad1d", [](auto &t) { return F::pad1d(cpuOps(), t[0], 3, 2, PadMode::kReflect); }, {x});
}

CATCH_TEST_CASE("test CPU fp16 lookup, attention and sampling", "[op][cpu][float16]") {
  Tensor table = signedRand({50, 16});
  Tensor indices = Tensor::create<LongType>({2, 3}, {0, 7, 49, 3, 3, 12});
  requireHalfAgrees(
      "lookup", [](auto &t) { return F::lookup(cpuOps(), t[0], t[1]); }, {table, indices}, 1e-6f, 1e-6f);

  Tensor q = signedRand({1, 4, 9, 32});
  Tensor k = signedRand({1, 2, 9, 32});
  Tensor v = signedRand({1, 2, 9, 32});
  for (bool causal : {false, true}) {
    CATCH_INFO("causal " << causal);
    requireHalfAgrees(
        "attention", [causal](auto &t) { return F::attention(cpuOps(), t[0], t[1], t[2], causal); },
        {q, k, v}, 2e-2f, 2e-2f);
  }

  // Greedy sampling of half logits picks what it picks from the same logits in float.
  Tensor logits = signedRand({3, 100});
  Tensor temperatures = Tensor::create<float>({3}, {0.0f, 0.0f, 0.0f});
  Tensor topKs = Tensor::create<IntType>({3}, {-1, -1, -1});
  Tensor topPs = Tensor::create<float>({3}, {1.0f, 1.0f, 1.0f});
  Tensor h = toFp16(logits);
  Tensor fromHalf = F::sample(cpuOps(), h, temperatures, topKs, topPs);
  Tensor fromFloat = F::sample(cpuOps(), toFp32(h), temperatures, topKs, topPs);
  const LongType *sh = fromHalf.getInternalData()->getData<LongType>(fromHalf.getInternalOffset());
  const LongType *sf = fromFloat.getInternalData()->getData<LongType>(fromFloat.getInternalOffset());
  for (int i = 0; i < 3; ++i) CATCH_REQUIRE(sh[i] == sf[i]);

  // The repetition penalty is in place, on the logits of either type.
  Tensor history = Tensor::create<LongType>({3, 2}, {1, 5, 2, 2, 99, 0});
  Tensor penalizedHalf = F::contiguous(cpuOps(), h);
  Tensor penalizedFloat = toFp32(h);
  cpuOps()->repetitionPenalty(penalizedHalf, history, 1.3f);
  cpuOps()->repetitionPenalty(penalizedFloat, history, 1.3f);
  CATCH_REQUIRE(cpuOps()->allClose(toFp32(penalizedHalf), penalizedFloat, 1e-2f, 1e-2f));

  // allClose and print take half themselves.
  CATCH_REQUIRE(cpuOps()->allClose(h, h));
  cpuOps()->print(h.slice(0, {0, 1}));
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
