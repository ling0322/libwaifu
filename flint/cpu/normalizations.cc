// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include "flint/cpu/normalizations.h"

#include <cmath>

#include "lutil/error.h"
#include "lutil/strings.h"
#include "flint/cpu/accessor.h"
#include "flint/cpu/common.h"
#include "flint/cpu/tensor.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {

template<typename T>
Tensor rmsNormKernel(const Tensor &tensor, const Tensor &weight, float eps) {
  CHECK(weight.getDim() == 1);
  CHECK(tensor.getShape(-1) == weight.getShape(0));

  Tensor C = tensorLike(tensor);

  TensorList<const T, 1> vA = TensorList<const T, 1>::fromTensor(tensor);
  TensorList<T, 1> vC = TensorList<T, 1>::fromTensor(C);
  CHECK(vA.getLength() == vC.getLength());

  TensorAccessor<const T, 1> w = weight;

  int numRows = vA.getLength();
#pragma omp parallel for schedule(dynamic, 1)
  for (int j = 0; j < numRows; ++j) {
    TensorAccessor<const T, 1> a = vA.getTensor(j);
    TensorAccessor<T, 1> c = vC.getTensor(j);

    float sum = 0.0;
    for (int i = 0; i < a.getShape(0); ++i) {
      float va = a[i];
      sum += va * va;
    }
    float mean = sum / a.getShape(0);
    float rms = std::sqrt(mean + eps);

    // compute rms-norm
    for (int i = 0; i < a.getShape(0); ++i) {
      float va = a[i];
      float vw = w[i];
      c[i] = static_cast<T>(a[i] * w[i] / rms);
    }
  }

  return C;
}

/// The weight and the bias are optional and read one value per position, so an empty tensor
/// stands for "leave this alone" rather than for a tensor of ones or zeros.
template<typename T>
const T *dataOrNull(const Tensor &x) {
  return x.empty() ? nullptr : x.getInternalData()->getData<T>(x.getInternalOffset());
}

void checkNormOperand(const Tensor &x, const char *what, int expected, DType dtype) {
  if (x.empty()) return;

  if (x.getDType() != dtype) {
    THROW(InvalidArg, lut::sprintf("%s is not the same type as the input", what));
  }
  if (x.getDim() != 1) THROW(InvalidArg, lut::sprintf("%s is not one dimensional", what));
  if (x.getNumEl() != expected) {
    THROW(
        InvalidArg,
        lut::sprintf("%s holds %d values, not %d", what, int(x.getNumEl()), expected));
  }
}

/// The same shape of loop as the RMS one above, with the mean subtracted rather than left at
/// zero. Accumulated in float whatever the elements are: over a few thousand values that keeps
/// the cancellation in `E[x^2] - E[x]^2` well away from anything half could tell apart, and a
/// negative result can still fall out of rounding, which the epsilon covers.
template<typename T>
Tensor layerNormKernel(const Tensor &tensor, const Tensor &weight, const Tensor &bias, float eps) {
  int hiddenSize = tensor.getShape(-1);
  checkNormOperand(weight, "the layerNorm weight", hiddenSize, tensor.getDType());
  checkNormOperand(bias, "the layerNorm bias", hiddenSize, tensor.getDType());

  Tensor C = tensorLike(tensor);

  TensorList<const T, 1> vA = TensorList<const T, 1>::fromTensor(tensor);
  TensorList<T, 1> vC = TensorList<T, 1>::fromTensor(C);
  CHECK(vA.getLength() == vC.getLength());

  const T *w = dataOrNull<T>(weight);
  const T *b = dataOrNull<T>(bias);

  int numRows = vA.getLength();
#pragma omp parallel for schedule(dynamic, 1)
  for (int j = 0; j < numRows; ++j) {
    TensorAccessor<const T, 1> a = vA.getTensor(j);
    TensorAccessor<T, 1> c = vC.getTensor(j);

    int width = a.getShape(0);
    float sum = 0.0f;
    float sumSquare = 0.0f;
    for (int i = 0; i < width; ++i) {
      float va = a[i];
      sum += va;
      sumSquare += va * va;
    }

    float mean = sum / width;
    float variance = sumSquare / width - mean * mean;
    float invStd = 1.0f / std::sqrt(variance > 0.0f ? variance + eps : eps);

    for (int i = 0; i < width; ++i) {
      float value = (float(a[i]) - mean) * invStd;
      if (w) value *= float(w[i]);
      if (b) value += float(b[i]);
      c[i] = static_cast<T>(value);
    }
  }

  return C;
}

/// One (image, group) at a time. A group covers `channelPerGroup` channels of `spatial` pixels
/// each and they are contiguous, so the whole group is one run of memory -- which is why this
/// indexes the buffer rather than going through TensorList, whose rows are the last dimension.
template<typename T>
Tensor groupNormKernel(
    const Tensor &tensor,
    const Tensor &weight,
    const Tensor &bias,
    int groups,
    float eps) {
  int batch = tensor.getShape(0);
  int channels = tensor.getShape(1);
  int spatial = tensor.getShape(2) * tensor.getShape(3);

  checkNormOperand(weight, "the groupNorm weight", channels, tensor.getDType());
  checkNormOperand(bias, "the groupNorm bias", channels, tensor.getDType());

  Tensor C = tensorLike(tensor);
  const T *in = tensor.getInternalData()->getData<T>(tensor.getInternalOffset());
  T *out = C.getInternalData()->getData<T>(C.getInternalOffset());
  const T *w = dataOrNull<T>(weight);
  const T *b = dataOrNull<T>(bias);

  int channelPerGroup = channels / groups;
  int64_t groupSize = static_cast<int64_t>(channelPerGroup) * spatial;
  int blocks = batch * groups;

#pragma omp parallel for schedule(dynamic, 1)
  for (int block = 0; block < blocks; ++block) {
    int group = block % groups;
    const T *x = in + static_cast<int64_t>(block) * groupSize;
    T *y = out + static_cast<int64_t>(block) * groupSize;

    float sum = 0.0f;
    float sumSquare = 0.0f;
    for (int64_t i = 0; i < groupSize; ++i) {
      float value = x[i];
      sum += value;
      sumSquare += value * value;
    }

    float mean = sum / groupSize;
    float variance = sumSquare / groupSize - mean * mean;
    float invStd = 1.0f / std::sqrt(variance > 0.0f ? variance + eps : eps);

    // The scale and the shift are per channel rather than per group, so which channel an element
    // belongs to has to be recovered from where it sits inside the group.
    for (int64_t i = 0; i < groupSize; ++i) {
      float value = (float(x[i]) - mean) * invStd;
      int channel = group * channelPerGroup + static_cast<int>(i / spatial);
      if (w) value *= float(w[channel]);
      if (b) value += float(b[channel]);
      y[i] = static_cast<T>(value);
    }
  }

  return C;
}

Tensor rmsNorm(Tensor tensor, Tensor weight, float eps) {
  if (tensor.getDType() == DType::kFloat) return rmsNormKernel<float>(tensor, weight, eps);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (tensor.getDType() == DType::kFloat16) return rmsNormKernel<Float16>(tensor, weight, eps);
#endif

  NOT_IMPL();
}

Tensor layerNorm(Tensor tensor, Tensor weight, Tensor bias, float eps) {
  if (tensor.getDim() < 1) {
    THROW(InvalidArg, "layerNorm takes an input of at least one dimension");
  }

  if (tensor.getDType() == DType::kFloat) {
    return layerNormKernel<float>(tensor, weight, bias, eps);
  }
#if LUT_CPU_ARCH == LUT_AARCH64
  if (tensor.getDType() == DType::kFloat16) {
    return layerNormKernel<Float16>(tensor, weight, bias, eps);
  }
#endif

  NOT_IMPL();
}

Tensor groupNorm(Tensor tensor, Tensor weight, Tensor bias, int groups, float eps) {
  if (tensor.getDim() != 4) THROW(InvalidArg, "groupNorm takes a 4-D input, as (N, C, H, W)");
  if (!tensor.isContiguous()) THROW(InvalidArg, "groupNorm takes a contiguous input");
  if (groups < 1 || tensor.getShape(1) % groups != 0) {
    THROW(
        InvalidArg,
        lut::sprintf(
            "groupNorm: %d channels do not divide into %d groups",
            tensor.getShape(1),
            groups));
  }

  if (tensor.getDType() == DType::kFloat) {
    return groupNormKernel<float>(tensor, weight, bias, groups, eps);
  }
#if LUT_CPU_ARCH == LUT_AARCH64
  if (tensor.getDType() == DType::kFloat16) {
    return groupNormKernel<Float16>(tensor, weight, bias, groups, eps);
  }
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
