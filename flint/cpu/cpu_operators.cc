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

#include "flint/cpu/cpu_operators.h"

#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>

#include "lutil/random.h"
#include "flint/cpu/all_close.h"
#include "flint/cpu/binary_op.h"
#include "flint/cpu/cast.h"
#include "flint/cpu/common.h"
#include "flint/cpu/copy.h"
#include "flint/cpu/fill.h"
#include "flint/cpu/gated_delta_net.h"
#include "flint/cpu/kernel/interface.h"
#include "flint/cpu/lookup.h"
#include "flint/cpu/matmul.h"
#include "flint/cpu/normalizations.h"
#include "flint/cpu/conv1d.h"
#include "flint/cpu/conv2d.h"
#include "flint/cpu/upsample.h"
#include "flint/cpu/print.h"
#include "flint/cpu/rand.h"
#include "flint/cpu/reduce.h"
#include "flint/cpu/scan.h"
#include "flint/cpu/repetition_penalty.h"
#include "flint/cpu/softmax.h"
#include "flint/cpu/glu.h"
#include "flint/cpu/tensor.h"
#include "flint/cpu/transform.h"
#include "flint/cpu/unary.h"
#include "flint/functional.h"
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {

namespace {

/// A packed copy of `input`, or `input` itself if it is packed already, so the loops below can
/// walk it as a flat array. A temporary of this backend's own.
TensorView contiguousCpu(const TensorView &input, Tensor &holder) {
  if (input.isContiguous()) return input;

  holder = F::emptyLike(input);
  op::cpu::copy(input, holder);
  return holder;
}

}  // namespace

CPUOperators::CPUOperators() {
}

// -- class CPUOperators ----------

void CPUOperators::rand(TensorView out) {
  op::cpu::rand(out, &_rand, 0, 1);
}

void CPUOperators::matmul(TensorView A, TensorView B, TensorView out) {
  cpu::matmul(A, B, out);
}

void CPUOperators::gatedDeltaNetPrefill(
    TensorView q,
    TensorView k,
    TensorView v,
    TensorView g,
    TensorView beta,
    TensorView cuSeqlens,
    TensorView stateSlots,
    TensorView state,
    TensorView out) {
  cpu::gatedDeltaNetPrefill(q, k, v, g, beta, cuSeqlens, stateSlots, state, out);
}

void CPUOperators::print(TensorView tensor) {
  cpu::print(tensor);
}

void CPUOperators::add(TensorView input, TensorView other, TensorView out) {
  cpu::binaryOp(input, other, BinaryOp::ADD, out);
}

void CPUOperators::sub(TensorView input, TensorView other, TensorView out) {
  cpu::binaryOp(input, other, BinaryOp::SUB, out);
}

void CPUOperators::subFloat(TensorView input, float other, TensorView out) {
  op::cpu::transform(input, 1.0f, -other, out);
}

void CPUOperators::softmax(TensorView input, TensorView out) {
  cpu::softmax(input, out);
}

void CPUOperators::sample(
    TensorView logits,
    TensorView temperatures,
    TensorView topKs,
    TensorView topPs,
    TensorView out) {
  CHECK(logits.getDevice().isHost() && logits.getDim() == 2);
  CHECK(logits.isContiguous());
  int rows = logits.getShape(0);
  int vocabSize = logits.getShape(1);
  CHECK(temperatures.getDevice().isHost() &&
        temperatures.getDType() == DType::kFloat && temperatures.isContiguous() &&
        temperatures.getShape() == std::vector<int>{rows});
  CHECK(topKs.getDevice().isHost() &&
        topKs.getDType() == DType::kInt32 && topKs.isContiguous() &&
        topKs.getShape() == std::vector<int>{rows});
  CHECK(topPs.getDevice().isHost() &&
        topPs.getDType() == DType::kFloat && topPs.isContiguous() &&
        topPs.getShape() == std::vector<int>{rows});
  CHECK(out.getDType() == DType::kLong && out.isContiguous());
  out.throwIfInvalidShape({rows}, "sample");

  if (logits.getDType() == DType::kFloat16) {
    Tensor wide = F::empty(Device::getCpu(), logits.getShape(), DType::kFloat);
    cpu::cast(logits, wide);
    return sample(wide, temperatures, topKs, topPs, out);
  }
  CHECK(logits.getDType() == DType::kFloat);

  const float *logitData = getDataPtrCpu<float>(logits);
  const float *temperatureData = getDataPtrCpu<float>(temperatures);
  const IntType *topKData = getDataPtrCpu<IntType>(topKs);
  const float *topPData = getDataPtrCpu<float>(topPs);
  LongType *sampled = getDataPtrCpu<LongType>(out);
  std::vector<int> labels(vocabSize);
  std::vector<float> weights(vocabSize);

  for (int row = 0; row < rows; ++row) {
    float temperature = temperatureData[row];
    int topK = topKData[row];
    float topP = topPData[row];
    CHECK(std::isfinite(temperature) && temperature >= 0.0f);
    CHECK(topK >= -1 && topK <= vocabSize);
    CHECK(topP > 0.0f && topP <= 1.0f);

    const float *rowLogits = logitData + static_cast<int64_t>(row) * vocabSize;
    auto normalizedLogit = [&](int label) {
      float value = rowLogits[label];
      return std::isnan(value) ? -std::numeric_limits<float>::infinity() : value;
    };
    std::iota(labels.begin(), labels.end(), 0);
    std::sort(labels.begin(), labels.end(), [&](int left, int right) {
      return normalizedLogit(left) > normalizedLogit(right);
    });

    if (temperature == 0.0f) {
      sampled[row] = labels[0];
      continue;
    }

    int effectiveTopK = topK <= 0 ? vocabSize : topK;
    float maxLogit = normalizedLogit(labels[0]);
    auto samplingWeight = [&](int label) {
      float logit = normalizedLogit(label);
      if (std::isinf(maxLogit)) return logit == maxLogit ? 1.0f : 0.0f;
      return std::exp((logit - maxLogit) / temperature);
    };
    float totalWeight = 0.0f;
    for (int i = 0; i < effectiveTopK; ++i) {
      weights[i] = samplingWeight(labels[i]);
      totalWeight += weights[i];
    }

    float selectedWeight = 0.0f;
    int selectedCount = effectiveTopK;
    for (int i = 0; i < effectiveTopK; ++i) {
      selectedWeight += weights[i];
      if (selectedWeight >= topP * totalWeight) {
        selectedCount = i + 1;
        break;
      }
    }

    float draw = _rand.nextFloat() * selectedWeight;
    float cumulative = 0.0f;
    sampled[row] = labels[selectedCount - 1];
    for (int i = 0; i < selectedCount; ++i) {
      cumulative += weights[i];
      if (draw < cumulative) {
        sampled[row] = labels[i];
        break;
      }
    }
  }
}

bool CPUOperators::allClose(TensorView A, TensorView B, float rtol, float atol) {
  return cpu::allClose(A, B, rtol, atol);
}

void CPUOperators::mul(TensorView A, float k, TensorView out) {
  op::cpu::transform(A, k, 0.0f, out);
}

void CPUOperators::mul(TensorView A, TensorView B, TensorView out) {
  op::cpu::binaryOp(A, B, BinaryOp::MUL, out);
}

void CPUOperators::lookup(TensorView table, TensorView indices, TensorView out) {
  cpu::lookup(table, indices, out);
}

void CPUOperators::fill(TensorView input, float value) {
  cpu::fill(input, value);
}

void CPUOperators::sum(TensorView input, int dim, TensorView out) {
  int ndim = input.getDim();
  if (dim < 0) dim += ndim;
  CHECK(dim >= 0 && dim < ndim);

  if (dim == ndim - 1) {
    cpu::reduce(input, MapReduceType::SUM, out);
    return;
  }

  // Not a single transpose of `dim` with the last: swapping them back afterwards puts the old last
  // dimension in the wrong place on any rank-4 input summed over a dimension before the third from
  // the end. (.., D, ..) is viewed as (outer, D, inner) instead, turned to (outer, inner, D), and
  // reduced to (outer, inner), which is `out` with every other dimension where it was.
  std::vector<int> shape = input.getShape();
  int outer = 1, inner = 1;
  for (int d = 0; d < dim; ++d) outer *= shape[d];
  for (int d = dim + 1; d < ndim; ++d) inner *= shape[d];

  Tensor packed;
  TensorView grouped = contiguousCpu(input, packed).view({outer, shape[dim], inner});
  Tensor turned = F::emptyLike(grouped.transpose(1, 2));
  cpu::copy(grouped.transpose(1, 2), turned);
  cpu::reduce(turned, MapReduceType::SUM, out.view({outer, inner}));
}

void CPUOperators::cumsum(TensorView input, int dim, TensorView out) {
  int ndim = input.getDim();
  if (dim < 0) dim += ndim;
  CHECK(dim >= 0 && dim < ndim);

  // Float16 arithmetic is an aarch64 feature on this backend, so a half tensor is scanned in
  // float32 and narrowed once at the end -- which is also where the precision is wanted.
  DType dtype = input.getDType();
  Tensor packed;
  TensorView source = contiguousCpu(input, packed);
  Tensor wide;
  if (dtype == DType::kFloat16) {
    wide = F::empty(Device::getCpu(), source.getShape(), DType::kFloat);
    cpu::cast(source, wide);
    source = wide;
  }

  // The scan is along the last dimension of a packed tensor, so any other is turned to the last
  // first, and the result turned back as it is written out.
  Tensor turned;
  if (dim != ndim - 1) {
    turned = F::emptyLike(source.transpose(dim, ndim - 1));
    cpu::copy(source.transpose(dim, ndim - 1), turned);
    source = turned;
  }

  Tensor scanned = F::emptyLike(source);
  cpu::cumsumLastDim(source, scanned);
  TensorView result = dim == ndim - 1 ? TensorView(scanned) : scanned.transpose(dim, ndim - 1);

  if (dtype == DType::kFloat16) {
    Tensor narrow = F::empty(Device::getCpu(), result.getShape(), DType::kFloat);
    cpu::copy(result, narrow);
    cpu::cast(narrow, out);
  } else {
    cpu::copy(result, out);
  }
}

void CPUOperators::max(TensorView input, TensorView out) {
  cpu::reduce(input, MapReduceType::MAX, out);
}

void CPUOperators::min(TensorView input, TensorView out) {
  cpu::reduce(input, MapReduceType::MIN, out);
}

void CPUOperators::square(TensorView input, TensorView out) {
  cpu::unaryOp(input, cpu::UnaryOp::SQUARE, out);
}

void CPUOperators::divTensor(TensorView input, TensorView other, TensorView out) {
  cpu::binaryOp(input, other, BinaryOp::DIV, out);
}

void CPUOperators::neg(TensorView input, TensorView out) {
  cpu::unaryOp(input, cpu::UnaryOp::NEG, out);
}

void CPUOperators::abs(TensorView input, TensorView out) {
  cpu::unaryOp(input, cpu::UnaryOp::ABS, out);
}

void CPUOperators::exp(TensorView input, TensorView out) {
  cpu::unaryOp(input, cpu::UnaryOp::EXP, out);
}

void CPUOperators::log(TensorView input, TensorView out) {
  cpu::unaryOp(input, cpu::UnaryOp::LOG, out);
}

void CPUOperators::round(TensorView input, TensorView out) {
  cpu::unaryOp(input, cpu::UnaryOp::ROUND, out);
}

void CPUOperators::sqrt(TensorView input, TensorView out) {
  cpu::unaryOp(input, cpu::UnaryOp::SQRT, out);
}

void CPUOperators::rsqrt(TensorView input, TensorView out) {
  cpu::unaryOp(input, cpu::UnaryOp::RSQRT, out);
}

void CPUOperators::sigmoid(TensorView input, TensorView out) {
  cpu::unaryOp(input, cpu::UnaryOp::SIGMOID, out);
}

void CPUOperators::tanh(TensorView input, TensorView out) {
  cpu::unaryOp(input, cpu::UnaryOp::TANH, out);
}

void CPUOperators::relu(TensorView input, TensorView out) {
  cpu::unaryOp(input, cpu::UnaryOp::RELU, out);
}

void CPUOperators::gelu(TensorView input, TensorView out) {
  cpu::unaryOp(input, cpu::UnaryOp::GELU, out);
}

void CPUOperators::silu(TensorView input, TensorView out) {
  cpu::unaryOp(input, cpu::UnaryOp::SILU, out);
}

void CPUOperators::sin(TensorView input, TensorView out) {
  cpu::unaryOp(input, cpu::UnaryOp::SIN, out);
}

void CPUOperators::cos(TensorView input, TensorView out) {
  cpu::unaryOp(input, cpu::UnaryOp::COS, out);
}

void CPUOperators::quickGelu(TensorView input, TensorView out) {
  cpu::unaryOp(input, cpu::UnaryOp::QUICK_GELU, out);
}

void CPUOperators::div(TensorView input, float other, TensorView out) {
  op::cpu::transform(input, 1.0f / other, 0.0f, out);
}

void CPUOperators::arangeLong(LongType begin, LongType step, TensorView out) {
  CHECK(out.getDType() == DType::kLong && out.getDim() == 1 && out.isContiguous());

  LongType *data = getDataPtrCpu<LongType>(out);
  int64_t numel = out.getNumEl();
  for (int64_t i = 0; i < numel; ++i) {
    data[i] = begin + step * i;
  }
}

void CPUOperators::randNormal(TensorView out) {
  CHECK(out.getDType() == DType::kFloat && out.isContiguous());
  int64_t numel = out.getNumEl();
  float *data = getDataPtrCpu<float>(out);

  // fillGaussian works in pairs, so an odd count is filled one element long and truncated.
  if (numel % 2 == 0) {
    _rand.fillGaussian(lut::Span<float>(data, numel));
  } else {
    std::vector<float> padded(numel + 1);
    _rand.fillGaussian(lut::makeSpan(padded));
    std::copy(padded.begin(), padded.begin() + numel, data);
  }
}

float CPUOperators::elem(TensorView tensor) {
  CHECK(tensor.getNumEl() == 1);
  CHECK(tensor.getDType() == DType::kFloat);

  return getDataPtrCpu<float>(tensor)[0];
}

bool CPUOperators::elemBool(TensorView tensor) {
  CHECK(tensor.getNumEl() == 1);
  CHECK(tensor.getDType() == DType::kBool);

  return getDataPtrCpu<BoolType>(tensor)[0];
}

void CPUOperators::mod(TensorView input, LongType other, TensorView out) {
  CHECK(input.getDType() == DType::kLong && out.getDType() == DType::kLong);
  CHECK(other != 0);
  CHECK(out.isContiguous());
  out.throwIfInvalidShape(input.getShape(), "mod");

  Tensor packed;
  TensorView x = contiguousCpu(input, packed);
  const LongType *src = getDataPtrCpu<LongType>(x);
  LongType *dest = getDataPtrCpu<LongType>(out);
  for (int64_t i = 0; i < out.getNumEl(); ++i) {
    dest[i] = src[i] % other;
  }
}

void CPUOperators::eq(TensorView input, TensorView other, TensorView out) {
  // Matches the CUDA backend, which compares <uint8> tensors and answers in <bool>.
  CHECK(input.getDType() == DType::kUInt8 && other.getDType() == DType::kUInt8);
  CHECK(out.getDType() == DType::kBool && out.isContiguous());
  input.throwIfInvalidShape(other.getShape(), "eq");
  out.throwIfInvalidShape(input.getShape(), "eq");

  Tensor packedA, packedB;
  TensorView a = contiguousCpu(input, packedA);
  TensorView b = contiguousCpu(other, packedB);

  const UInt8 *pa = getDataPtrCpu<UInt8>(a);
  const UInt8 *pb = getDataPtrCpu<UInt8>(b);
  BoolType *pc = getDataPtrCpu<BoolType>(out);
  for (int64_t i = 0; i < out.getNumEl(); ++i) {
    pc[i] = pa[i] == pb[i];
  }
}

bool CPUOperators::all(TensorView A) {
  CHECK(A.getDType() == DType::kBool);

  Tensor packed;
  TensorView x = contiguousCpu(A, packed);
  const BoolType *data = getDataPtrCpu<BoolType>(x);
  for (int64_t i = 0; i < x.getNumEl(); ++i) {
    if (!data[i]) return false;
  }

  return true;
}

void CPUOperators::repetitionPenalty(TensorView logits, TensorView history, float weight) {
  CHECK(history.getDType() == DType::kLong);

  cpu::repetitionPenalty(logits, history, weight);
}

void CPUOperators::rmsNorm(TensorView input, TensorView weight, float eps, TensorView out) {
  CHECK(input.getDType() == weight.getDType());

  cpu::rmsNorm(input, weight, eps, out);
}

void CPUOperators::layerNorm(
    TensorView input,
    TensorView weight,
    TensorView bias,
    float eps,
    TensorView out) {
  cpu::layerNorm(input, weight, bias, eps, out);
}

void CPUOperators::groupNorm(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int groups,
    float eps,
    TensorView out) {
  cpu::groupNorm(input, weight, bias, groups, eps, out);
}

void CPUOperators::upsampleNearest2d(TensorView input, int scale, TensorView out) {
  cpu::upsampleNearest2d(input, scale, out);
}

void CPUOperators::upsampleNearest1d(TensorView input, TensorView out) {
  cpu::upsampleNearest1d(input, out);
}

void CPUOperators::conv2d(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    TensorView out) {
  cpu::conv2d(input, weight, bias, stride, padding, dilation, groups, out);
}

void CPUOperators::conv1d(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    TensorView out) {
  cpu::conv1d(input, weight, bias, stride, padding, dilation, groups, out);
}

void CPUOperators::causalMask(TensorView out) {
  op::cpu::causalMask(out);
}

void CPUOperators::copy(TensorView src, TensorView dest) {
  cpu::copy(src, dest);
}

void CPUOperators::swiglu(TensorView A, TensorView out) {
  cpu::swiglu(A, out);
}

void CPUOperators::geglu(TensorView input, TensorView out) {
  cpu::geglu(input, out);
}

void CPUOperators::cast(TensorView input, TensorView out) {
  cpu::cast(input, out);
}

void CPUOperators::manualSeed(uint64_t seed) {
  _rand.reset(seed);
}

DType CPUOperators::getDefaultFloatType() {
  return DType::getType<cpu::DefaultFloatType>();
}

// the CPU backend allocates through the system allocator and tracks nothing.
MemorySnapshot CPUOperators::captureMemorySnapshot() {
  return MemorySnapshot(0, 0, 0, 0);
}

void CPUOperators::resetPeakMemoryStats() {
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
