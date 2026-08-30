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

// The convolution as an im2col and a GEMM, which is the shortest route to a fast one: the packed
// block GEMM in cpu/kernel is already tuned, and a convolution laid out as a matrix multiply is
// the same arithmetic.
//
// A weight of (K, C, R, S) read as (K, C * R * S), against a matrix whose column is one output
// pixel and whose rows are the C * R * S input values that pixel sums over, gives (K, P * Q) --
// which is the image in NCHW, in the layout it was wanted in, with nothing to permute afterwards.
//
// That matrix is R * S times larger than the image it came from, so it is never built whole. A
// 1024 by 1024 output of 128 channels would want (128 * 9) by (1024 * 1024) floats, which is 4.8
// GB. It is built a block of output pixels at a time instead, into a buffer each thread keeps and
// reuses, sized so that the block is a few megabytes whatever the channel count.

#include "flint/cpu/conv2d.h"

#include <algorithm>
#include <vector>

#include "lutil/error.h"
#include "lutil/strings.h"
#include "flint/cpu/common.h"
#include "flint/cpu/kernel/interface.h"
#include "flint/cpu/tensor.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {
namespace {

/// How much of one thread's im2col buffer to keep live. Small enough that a machine's worth of
/// threads is a few hundred megabytes at the widest convolution SDXL runs, large enough that the
/// GEMM it feeds is not a sliver: at the U-Net's deepest block a column is 11520 values, which
/// this makes 182 columns wide, and at the autoencoder's widest it is 1820.
constexpr int64_t kBlockBytes = 8 << 20;
constexpr int kMinBlockPixels = 32;

struct Problem {
  int batch;
  int inChannels;
  int inH;
  int inW;
  int outChannels;
  int filterH;
  int filterW;
  int outH;
  int outW;
  int stride;
  int padding;
  int dilation;
  int groups;
};

/// Gather the input values that `count` output pixels starting at `first` sum over.
///
/// One row of `col` is one (channel, filter row, filter column) of the weight, and one column is
/// one output pixel, so this is the arrangement the GEMM below reads without transposing either
/// operand. Written row by row rather than column by column: along a row the output pixels step
/// through the input by `stride`, which is a stride of one for almost everything a diffusion
/// model convolves, so the reads run forwards through the image.
template<typename T>
void im2col(
    const T *image,
    const Problem &p,
    int channelBegin,
    int channelCount,
    int first,
    int count,
    T *col) {
  int spatial = p.inH * p.inW;

  for (int c = 0; c < channelCount; ++c) {
    const T *plane = image + static_cast<int64_t>(channelBegin + c) * spatial;

    for (int r = 0; r < p.filterH; ++r) {
      for (int s = 0; s < p.filterW; ++s) {
        T *row = col + (static_cast<int64_t>(c * p.filterH + r) * p.filterW + s) * count;

        for (int i = 0; i < count; ++i) {
          int pixel = first + i;
          int y = (pixel / p.outW) * p.stride - p.padding + r * p.dilation;
          int x = (pixel % p.outW) * p.stride - p.padding + s * p.dilation;

          // Outside the image is the zero the padding stands for.
          bool inside = y >= 0 && y < p.inH && x >= 0 && x < p.inW;
          row[i] = inside ? plane[static_cast<int64_t>(y) * p.inW + x] : T(0.0f);
        }
      }
    }
  }
}

/// A float activation against a weight of either type.
///
/// The mixed micro-kernel takes its float operand first and its half one second, and a convolution
/// wants the weight first, so that case is computed as its transpose -- `(count, filters)` from
/// `col' * weight'` -- which is what puts `col` in the slot the kernel keeps for a float. The
/// block that comes back is then read down its columns rather than along its rows on the way into
/// the image, which is a strided read of something small enough to stay in cache.
Tensor conv2dImpl(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    const Problem &p) {
  bool halfWeight = weight.getDType() == DType::kFloat16;
  Tensor output = tensor({p.batch, p.outChannels, p.outH, p.outW}, input.getDType());

  const float *in = input.getInternalData()->getData<float>(input.getInternalOffset());
  const void *w = halfWeight
                      ? static_cast<const void *>(
                            weight.getInternalData()->getData<Float16>(weight.getInternalOffset()))
                      : static_cast<const void *>(
                            weight.getInternalData()->getData<float>(weight.getInternalOffset()));
  float *out = output.getInternalData()->getData<float>(output.getInternalOffset());
  const float *biasData =
      bias.empty() ? nullptr : bias.getInternalData()->getData<float>(bias.getInternalOffset());

  int channelsPerGroup = p.inChannels / p.groups;
  int filtersPerGroup = p.outChannels / p.groups;
  int64_t columnHeight = static_cast<int64_t>(channelsPerGroup) * p.filterH * p.filterW;
  int64_t outSpatial = static_cast<int64_t>(p.outH) * p.outW;

  int blockPixels = static_cast<int>(kBlockBytes / (columnHeight * sizeof(float)));
  blockPixels = std::max(kMinBlockPixels, blockPixels);
  blockPixels = static_cast<int>(std::min<int64_t>(blockPixels, outSpatial));

  int64_t blocksPerImage = (outSpatial + blockPixels - 1) / blockPixels;
  int64_t totalBlocks = static_cast<int64_t>(p.batch) * p.groups * blocksPerImage;

#pragma omp parallel
  {
    // One buffer per thread rather than per block: the shapes do not change between blocks, and
    // this is the allocation that would otherwise happen thousands of times an image.
    std::vector<float> col(columnHeight * blockPixels);
    std::vector<float> product(static_cast<int64_t>(filtersPerGroup) * blockPixels);

#pragma omp for schedule(dynamic, 1)
    for (int64_t block = 0; block < totalBlocks; ++block) {
      int64_t within = block % blocksPerImage;
      int64_t rest = block / blocksPerImage;
      int group = static_cast<int>(rest % p.groups);
      int image = static_cast<int>(rest / p.groups);

      int first = static_cast<int>(within * blockPixels);
      int count = static_cast<int>(std::min<int64_t>(blockPixels, outSpatial - first));

      const float *imageData =
          in + static_cast<int64_t>(image) * p.inChannels * p.inH * p.inW;
      im2col<float>(
          imageData,
          p,
          group * channelsPerGroup,
          channelsPerGroup,
          first,
          count,
          col.data());

      // The GEMM accumulates into its output rather than overwriting it -- `cpu::matmul` builds a
      // zeroed C for exactly this reason -- and this buffer is reused for every block the thread
      // takes, so it has to start at zero each time.
      std::fill(product.begin(), product.begin() + static_cast<int64_t>(filtersPerGroup) * count,
                0.0f);

      // (filters, columnHeight) by (columnHeight, count). The weight of one group is already
      // contiguous, so it is read where it lies rather than gathered.
int64_t groupOffset = static_cast<int64_t>(group) * filtersPerGroup * columnHeight;
      if (halfWeight) {
        kernel::gemmHalfWeightFloat(
            true,
            true,
            count,
            filtersPerGroup,
            static_cast<int>(columnHeight),
            col.data(),
            count,
            reinterpret_cast<const kernel::Float16 *>(w) + groupOffset,
            static_cast<int>(columnHeight),
            product.data(),
            filtersPerGroup,
            kernel::Mode::SingleThread);
      } else {
        kernel::gemmFloat(
            false,
            false,
            filtersPerGroup,
            count,
            static_cast<int>(columnHeight),
            static_cast<const float *>(w) + groupOffset,
            static_cast<int>(columnHeight),
            col.data(),
            count,
            product.data(),
            count,
            kernel::Mode::SingleThread);
      }

      // The product is (filters, count) and the output row it belongs in is outSpatial long, so
      // each row lands at its own offset rather than the whole block being one run.
      for (int f = 0; f < filtersPerGroup; ++f) {
        int channel = group * filtersPerGroup + f;
        float *destination =
            out + (static_cast<int64_t>(image) * p.outChannels + channel) * outSpatial + first;
        const float *source = product.data() + static_cast<int64_t>(f) * count;

        float b = biasData ? biasData[channel] : 0.0f;
        if (halfWeight) {
          // The block is (count, filters) there, so this filter's values are a column of it.
          const float *column = product.data() + f;
          for (int i = 0; i < count; ++i) destination[i] = column[i * filtersPerGroup] + b;
        } else if (biasData) {
          for (int i = 0; i < count; ++i) destination[i] = source[i] + b;
        } else {
          std::copy(source, source + count, destination);
        }
      }
    }
  }

  return output;
}

}  // namespace

Tensor conv2d(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    int stride,
    int padding,
    int dilation,
    int groups) {
  if (input.getDim() != 4) THROW(InvalidArg, "conv2d takes a 4-D input, as (N, C, H, W)");
  if (weight.getDim() != 4) THROW(InvalidArg, "conv2d takes a 4-D weight, as (K, C, R, S)");
  // A float activation may be multiplied by a half weight, which is how a model is held here.
  if (input.getDType() != weight.getDType() &&
      !(input.getDType() == DType::kFloat && weight.getDType() == DType::kFloat16)) {
    THROW(InvalidArg, "conv2d: the input and the weight are of different types");
  }
  if (groups < 1) THROW(InvalidArg, "conv2d: the group count is below one");
  if (stride < 1 || dilation < 1) {
    THROW(InvalidArg, "conv2d: the stride and the dilation are below one");
  }
  if (padding < 0) THROW(InvalidArg, "conv2d: the padding is negative");
  if (!input.isContiguous() || !weight.isContiguous()) {
    THROW(InvalidArg, "conv2d takes contiguous tensors");
  }

  if (input.getShape(1) != weight.getShape(1) * groups) {
    THROW(
        InvalidArg,
        lut::sprintf(
            "conv2d: an input of %d channels does not match a weight of %d by %d groups",
            input.getShape(1),
            weight.getShape(1),
            groups));
  }
  if (weight.getShape(0) % groups != 0) {
    THROW(InvalidArg, "conv2d: the filters do not divide into the groups");
  }
  if (!bias.empty()) {
    if (bias.getNumEl() != weight.getShape(0)) {
      THROW(InvalidArg, "conv2d: the bias does not match the output channels");
    }
    if (bias.getDType() != input.getDType()) {
      THROW(InvalidArg, "conv2d: the bias and the input are of different types");
    }
  }

  Problem p{};
  p.batch = input.getShape(0);
  p.inChannels = input.getShape(1);
  p.inH = input.getShape(2);
  p.inW = input.getShape(3);
  p.outChannels = weight.getShape(0);
  p.filterH = weight.getShape(2);
  p.filterW = weight.getShape(3);
  p.stride = stride;
  p.padding = padding;
  p.dilation = dilation;
  p.groups = groups;
  p.outH = (p.inH + 2 * padding - dilation * (p.filterH - 1) - 1) / stride + 1;
  p.outW = (p.inW + 2 * padding - dilation * (p.filterW - 1) - 1) / stride + 1;

  if (p.outH < 1 || p.outW < 1) {
    THROW(InvalidArg, "conv2d: the input is smaller than the kernel reaches");
  }

  if (input.getDType() == DType::kFloat) return conv2dImpl(input, weight, bias, p);

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
