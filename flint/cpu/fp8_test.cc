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
#include "lutil/error.h"
#include "lutil/span.h"
#include "flint/cpu/fp8.h"
#include "flint/cpu/kernel/util.h"
#include "flint/device.h"
#include "flint/fp8.h"
#include "flint/functional.h"
#include "flint/tensor.h"

namespace fl {
namespace {

/// allClose measures the largest difference against the mean magnitude, which says nothing useful
/// when one row of the result is thousands of times larger than the rest. This weighs every
/// element by its own size instead.
double relativeRmse(const Tensor &x, const Tensor &reference) {
  const float *pa = x.getInternalData()->getData<float>(x.getInternalOffset());
  const float *pb = reference.getInternalData()->getData<float>(reference.getInternalOffset());

  double squaredError = 0.0;
  double squaredReference = 0.0;
  for (int64_t i = 0; i < x.getNumEl(); ++i) {
    double diff = double(pa[i]) - double(pb[i]);
    squaredError += diff * diff;
    squaredReference += double(pb[i]) * double(pb[i]);
  }

  return std::sqrt(squaredError / squaredReference);
}

bool allFinite(const Tensor &x) {
  const float *data = x.getInternalData()->getData<float>(x.getInternalOffset());
  for (int64_t i = 0; i < x.getNumEl(); ++i) {
    if (!std::isfinite(data[i])) return false;
  }
  return true;
}

/// Thirty-two magnitudes E4M3 holds exactly, the largest of them 448. A row of these has a scale
/// of exactly one, so quantizing it is a round trip rather than an approximation, and anything
/// the quantizer gets wrong -- the scale arithmetic, the element order -- turns it into a wrong
/// answer rather than into slightly more error.
std::vector<float> exactE4m3Row(float rowScale) {
  const float codes[32] = {0.0f,    448.0f,  -448.0f, 1.0f,   -1.0f,  1.5f,   2.0f,   3.0f,
                           4.0f,    6.0f,    8.0f,    12.0f,  16.0f,  24.0f,  32.0f,  48.0f,
                           64.0f,   96.0f,   128.0f,  192.0f, 256.0f, 384.0f, 416.0f, 320.0f,
                           -256.0f, -0.5f,   0.5f,    0.25f,  0.125f, -96.0f, 224.0f, -3.0f};

  std::vector<float> row;
  for (float code : codes) row.push_back(code * rowScale);
  return row;
}

/// A reference for the multiply that does not share a line of code with it: the weight read back
/// as float and multiplied by the ordinary float GEMM.
Tensor referenceGemm(const Tensor &a, const Fp8Operand &w) {
  return F::matmul(a, op::cpu::dequantFp8ToFloat(w).transpose(0, 1));
}

}  // namespace

CATCH_TEST_CASE("test cpu fp8 conversion (every code)", "[fl][op][cpu][fp8]") {
  // E4M3 has 256 of them, so there is no reason to test a sample. Decoding is checked against the
  // format's own definition written out here -- sign, four exponent bits with a bias of seven,
  // three mantissa bits, and subnormals at an exponent of -6 -- rather than against another
  // implementation of the same table.
  for (int i = 0; i < 256; ++i) {
    CATCH_INFO("code = " << i);
    op::cpu::kernel::Fp8E4M3 code{static_cast<uint8_t>(i)};
    float got = op::cpu::kernel::cvt_f8_s(code);

    int sign = (i & 0x80) ? -1 : 1;
    int exp = (i >> 3) & 0xf;
    int man = i & 0x7;

    if (exp == 0xf && man == 0x7) {
      CATCH_REQUIRE(std::isnan(got));
      continue;
    }

    double want = exp == 0 ? std::ldexp(man, -9) : std::ldexp(8 + man, exp - 7 - 3);
    CATCH_REQUIRE(got == float(sign * want));

    // And every code encodes back to itself, which is what says the two directions agree.
    if (!(i == 0x80)) {  // negative zero encodes to positive zero's code
      CATCH_REQUIRE(op::cpu::kernel::cvt_s_f8(got).v == i);
    }
  }
}

CATCH_TEST_CASE("test cpu fp8 conversion (rounding and saturation)", "[fl][op][cpu][fp8]") {
  auto encode = [](float v) { return int(op::cpu::kernel::cvt_s_f8(v).v); };

  // Round to nearest even at a tie, on both sides of the last mantissa bit. Between 1.0 (code
  // 0x38, mantissa 000) and 1.125 (0x39, mantissa 001) the tie is 1.0625, which goes to the even
  // one; between 1.125 and 1.25 (0x3a) it is 1.1875, which goes to 0x3a.
  CATCH_REQUIRE(encode(1.0625f) == 0x38);
  CATCH_REQUIRE(encode(1.1875f) == 0x3a);
  CATCH_REQUIRE(encode(1.0624f) == 0x38);
  CATCH_REQUIRE(encode(1.0626f) == 0x39);

  // Saturation rather than the NaN code: 448 is the largest finite magnitude, and anything above
  // it comes back as 448 with its sign.
  CATCH_REQUIRE(encode(448.0f) == 0x7e);
  CATCH_REQUIRE(encode(1.0e30f) == 0x7e);
  CATCH_REQUIRE(encode(-1.0e30f) == 0xfe);
  CATCH_REQUIRE(encode(std::numeric_limits<float>::infinity()) == 0x7e);
  CATCH_REQUIRE(encode(-std::numeric_limits<float>::infinity()) == 0xfe);

  // The subnormal range, and the rollover from the largest subnormal into the smallest normal.
  CATCH_REQUIRE(encode(0x1p-9f) == 0x01);
  CATCH_REQUIRE(encode(0x1p-6f) == 0x08);
  CATCH_REQUIRE(encode(7 * 0x1p-9f) == 0x07);
  // Half of the smallest subnormal is a tie, and zero is the even side of it.
  CATCH_REQUIRE(encode(0x1p-10f) == 0x00);
  CATCH_REQUIRE(encode(0x1.8p-10f) == 0x01);
  CATCH_REQUIRE(encode(1.0e-30f) == 0x00);

  CATCH_REQUIRE((encode(std::numeric_limits<float>::quiet_NaN()) & 0x7f) == 0x7f);
}

CATCH_TEST_CASE("test cpu fp8 quantizer round trip", "[fl][op][cpu][fp8]") {
  // Four rows, each a different power of two away from the next, so the per channel scale is the
  // only thing that can tell them apart.
  std::vector<float> data;
  for (float rowScale : {1.0f, 8.0f, 0.03125f, 64.0f}) {
    std::vector<float> row = exactE4m3Row(rowScale);
    data.insert(data.end(), row.begin(), row.end());
  }
  Tensor w = Tensor::create<float>({4, 32}, lut::makeConstSpan(data));

  Fp8Operand q = op::cpu::quantizeFp8(w);
  CATCH_REQUIRE(q.data.getShape() == std::vector<int>{4, 32});
  CATCH_REQUIRE(q.data.getDType() == DType::kFp8E4M3);
  CATCH_REQUIRE(q.channelScale.getShape() == std::vector<int>{4});

  // rowAmax / 448, and every row's amax is 448 times the row's own scale.
  Tensor scale = Tensor::create<float>({4}, {1.0f, 8.0f, 0.03125f, 64.0f});
  CATCH_REQUIRE(F::allClose(q.channelScale, scale, 1e-6f, 1e-6f));

  Tensor x = op::cpu::dequantFp8ToFloat(q);
  CATCH_REQUIRE(x.getShape() == std::vector<int>{4, 32});
  CATCH_REQUIRE(F::allClose(x, w, 1e-6f, 1e-6f));
}

CATCH_TEST_CASE("test cpu fp8 quantizer (zero row)", "[fl][op][cpu][fp8]") {
  // A row with nothing in it drives its own scale to zero, and the scale is a divisor on the way
  // in. This is the case that turns into NaN if the guard is missing.
  std::vector<float> data(2 * 32, 0.0f);
  for (int i = 32; i < 64; ++i) data[i] = 1.0f;

  Tensor w = Tensor::create<float>({2, 32}, lut::makeConstSpan(data));
  Tensor x = op::cpu::dequantFp8ToFloat(op::cpu::quantizeFp8(w));

  CATCH_REQUIRE(allFinite(x));
  CATCH_REQUIRE(F::allClose(x, w, 1e-6f, 1e-6f));
}

CATCH_TEST_CASE("test cpu fp8 quantizer (half input)", "[fl][op][cpu][fp8]") {
  // A weight arrives from a package in the type the file stored it in, which for a model on this
  // device is half. Quantizing it must not need a float copy of the whole thing first.
  Tensor w = F::randn({8, 64});
  Tensor h = F::cast(w, DType::kFloat16);

  Fp8Operand fromHalf = op::cpu::quantizeFp8(h);
  Fp8Operand fromFloat = op::cpu::quantizeFp8(F::cast(h, DType::kFloat));

  CATCH_REQUIRE(F::allClose(fromHalf.channelScale, fromFloat.channelScale, 1e-6f, 1e-6f));
  CATCH_REQUIRE(F::allClose(
      op::cpu::dequantFp8ToFloat(fromHalf),
      op::cpu::dequantFp8ToFloat(fromFloat),
      1e-6f,
      1e-6f));
}

CATCH_TEST_CASE("test cpu gemmFp8 (shapes)", "[fl][op][cpu][fp8]") {
  auto runCase = [](int m, int n, int k) {
    CATCH_INFO("m = " << m << ", n = " << n << ", k = " << k);

    Tensor a = F::randn({m, k});
    Fp8Operand w = op::cpu::quantizeFp8(F::randn({n, k}));

    Tensor actual = op::cpu::gemmFp8(a, w);
    CATCH_REQUIRE(actual.getShape() == std::vector<int>{m, n});
    CATCH_REQUIRE(allFinite(actual));

    // Against the same weight read back as float and multiplied by the ordinary GEMM, so what is
    // compared is the packing and the scale rather than the format's own error.
    CATCH_REQUIRE(relativeRmse(actual, referenceGemm(a, w)) < 1e-5);
  };

  // A single row, which is the case the block path is shaped least like.
  runCase(1, 16, 16);
  runCase(1, 512, 1024);
  // Squares, and residues on each axis.
  runCase(16, 16, 16);
  runCase(17, 33, 65);
  runCase(64, 64, 64);
  // Wider than one NC panel and deeper than one KC panel, which is where the packing loop runs
  // more than once.
  runCase(8, 5120, 1024);
  runCase(129, 264, 96);
  // Odd k, which the CUDA path refuses and this one does not.
  runCase(4, 8, 1);
  runCase(4, 8, 7);
  runCase(3, 8, 68);
}

CATCH_TEST_CASE("test cpu gemmFp8 (batch axes)", "[fl][op][cpu][fp8]") {
  Fp8Operand w = op::cpu::quantizeFp8(F::randn({64, 32}));

  Tensor a3 = F::randn({2, 3, 32});
  Tensor out3 = op::cpu::gemmFp8(a3, w);
  CATCH_REQUIRE(out3.getShape() == std::vector<int>{2, 3, 64});
  CATCH_REQUIRE(F::allClose(out3.view({-1, 64}), op::cpu::gemmFp8(a3.view({-1, 32}), w)));

  Tensor a4 = F::randn({2, 3, 5, 32});
  CATCH_REQUIRE(op::cpu::gemmFp8(a4, w).getShape() == std::vector<int>{2, 3, 5, 64});
}

CATCH_TEST_CASE("test cpu gemmFp8 (channel scale per column)", "[fl][op][cpu][fp8]") {
  // A weight whose channels differ by four orders of magnitude, against an activation of ones:
  // each column of the result is then that channel's own row sum, so a scale applied to the wrong
  // axis -- or to the panel rather than to the channel -- cannot look right by accident. Each row
  // here is a constant, so quantizing it is exact and the only error left is the arithmetic's.
  std::vector<float> data(16 * 16);
  for (int r = 0; r < 16; ++r) {
    for (int i = 0; i < 16; ++i) data[r * 16 + i] = std::pow(10.0f, float(r % 5) - 2.0f);
  }

  Tensor w = Tensor::create<float>({16, 16}, lut::makeConstSpan(data));
  Fp8Operand q = op::cpu::quantizeFp8(w);

  std::vector<float> ones(4 * 16, 1.0f);
  Tensor a = Tensor::create<float>({4, 16}, lut::makeConstSpan(ones));
  Tensor out = op::cpu::gemmFp8(a, q);

  const float *got = out.getInternalData()->getData<float>(out.getInternalOffset());
  for (int m = 0; m < 4; ++m) {
    for (int r = 0; r < 16; ++r) {
      float want = 16.0f * std::pow(10.0f, float(r % 5) - 2.0f);
      CATCH_INFO("row " << m << ", channel " << r);
      CATCH_REQUIRE(std::fabs(got[m * 16 + r] - want) <= want * 1e-5f);
    }
  }
}

CATCH_TEST_CASE("test makeFp8Operand rejects what it cannot use", "[fl][op][cpu][fp8]") {
  Fp8Operand q = op::cpu::quantizeFp8(F::randn({8, 32}));

  CATCH_REQUIRE_NOTHROW(makeFp8Operand(q.data, q.channelScale));
  // The data where the scale belongs.
  CATCH_REQUIRE_THROWS_AS(makeFp8Operand(q.channelScale, q.channelScale), lut::Error);
  // One scale for a weight that has eight channels.
  CATCH_REQUIRE_THROWS_AS(makeFp8Operand(q.data, q.channelScale.slice(0, {0, 1})), lut::Error);
}

}  // namespace fl
