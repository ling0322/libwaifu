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

#pragma once

#include <stdint.h>

#include "lutil/random.h"
#include "flint/device.h"
#include "flint/memory.h"
#include "flint/tensor.h"

namespace fl {

// base functional interface to apply operators for Tensor
class Operators {
 public:
  virtual ~Operators() = default;

  virtual Tensor arangeLong(LongType begin, LongType end, LongType step);
  virtual Tensor lookup(Tensor table, Tensor indices);
  virtual void rotaryEmbedding(Tensor positions, Tensor query, Tensor key, Tensor rotaryCache);
  virtual Tensor rmsNorm(Tensor input, Tensor weight, float eps);

  /// Normalize the last dimension of `input` to zero mean and unit variance, then scale by
  /// `weight` and shift by `bias`; either may be empty.
  virtual Tensor layerNorm(Tensor input, Tensor weight, Tensor bias, float eps);

  /// Normalize `input` <float16>(N, C, H, W) over each group of channels and the space it covers,
  /// then scale and shift per channel. `weight` and `bias` are (C), and either may be empty.
  virtual Tensor groupNorm(Tensor input, Tensor weight, Tensor bias, int groups, float eps);

  /// Repeat each pixel of `input` <float16>(N, C, H, W) `scale` times along both spatial axes.
  virtual Tensor upsampleNearest2d(Tensor input, int scale);

  /// The gated linear unit of `swiglu` with a GELU in place of the SiLU.
  virtual Tensor geglu(Tensor input);

  virtual Tensor matmul(Tensor A, Tensor B);

  /// 2-D convolution of `input` <float16|float>(N, C, H, W) by `weight` (K, C / groups, R, S),
  /// with an optional per-channel `bias` (K). Square stride, padding and dilation throughout.
  virtual Tensor conv2d(
      Tensor input,
      Tensor weight,
      Tensor bias,
      int stride,
      int padding,
      int dilation,
      int groups);

  /// 1-D convolution of `input` <float16|float>(N, C, L) by `weight` (K, C / groups, R), with an
  /// optional per-channel `bias` (K). The result is (N, K, Lout), Lout being
  /// (L + 2 * padding - dilation * (R - 1) - 1) / stride + 1.
  ///
  /// `groups == C == K` is the depthwise case. It gets no operator of its own -- it is this one
  /// with every group a single channel -- but it is the case a speech model actually asks for and
  /// the one a backend has most to gain from specializing.
  ///
  /// Every backend implements this. CUDA on CUTLASS, groups and all; the CPU as the 2-D
  /// convolution over an image one row tall that it is, sharing `conv.cc`'s im2col and GEMM with
  /// `conv2d` and differing only in padding the length and not the height; Metal as MLX's own,
  /// with the operands turned channels last and back.
  virtual Tensor conv1d(
      Tensor input,
      Tensor weight,
      Tensor bias,
      int stride,
      int padding,
      int dilation,
      int groups);

  /// Transposed 1-D convolution of `input` <float16|float>(N, C, L) by `weight`
  /// (C, K / groups, R), with an optional per-channel `bias` (K). The result is (N, K, Lout),
  /// Lout being (L - 1) * stride - 2 * padding + R + outputPadding: input position `l`
  /// contributes the whole kernel starting at `l * stride`. What a vocoder upsamples with.
  ///
  /// No backend implements this. `waifu::audio::conv_transpose1d` composes it as one `matmul`
  /// and ceil(R / stride) shifted additions, for one group only.
  virtual Tensor convTranspose1d(
      Tensor input,
      Tensor weight,
      Tensor bias,
      int stride,
      int padding,
      int outputPadding,
      int groups);

  /// The activation a BigVGAN is built from, per channel of `input` <float16|float>(N, C, L):
  /// `x + sin(alpha * x)^2 / (beta + eps)`. `alpha` and `beta` are (C) and already exponentiated
  /// -- a checkpoint trained with `alpha_logscale` stores their logarithm, and raising it belongs
  /// to whoever reads the weight. An empty `beta` means beta is alpha, the plain snake.
  ///
  /// No backend implements this. `waifu::audio::snake` composes it out of the elementwise
  /// operators, which costs about six passes over the tensor where a fused kernel would cost one.
  /// Of everything named here this is the one whose kernel would most likely pay for itself: it
  /// is memory bound and a vocoder applies it eighteen times per upsampling stage.
  virtual Tensor snake(Tensor input, Tensor alpha, Tensor beta, float eps);

  /// Short time Fourier transform of `input` <float>(N, 1, L) against `window` (nFft), as
  /// <float>(N, 2 * (nFft / 2 + 1), frames) -- every bin's real part, then every bin's imaginary
  /// part, because a tensor here holds real numbers. Only half the spectrum: the input is real,
  /// so the other half is its conjugate. `centered` pads by half a window at both ends, by
  /// reflection, which is what `torch.stft` does by default.
  ///
  /// No backend implements this. `waifu::audio::stft` composes it as a convolution against a bank
  /// of windowed sinusoids, which is a GEMM of O(nFft^2) per frame where a fast transform would
  /// be O(nFft log nFft). A backend with an FFT -- cuFFT, vDSP, MLX -- is the reason this is named
  /// here rather than left a composition forever.
  virtual Tensor stft(Tensor input, Tensor window, int nFft, int hop, bool centered);

  /// The inverse of `stft`: `spectrum` <float>(N, 2 * (nFft / 2 + 1), frames) against `window`
  /// (nFft), as <float>(N, 1, L). Overlap-add, divided by the window's own overlap so that the
  /// inverse of a transform is the signal it was taken of.
  ///
  /// No backend implements this. `waifu::audio::istft` composes it out of `conv_transpose1d`.
  virtual Tensor istft(Tensor spectrum, Tensor window, int nFft, int hop, bool centered);

  /// Solve the lower triangular systems L X = B. l is <float>(..., N, N) and b is
  /// <float>(..., N, M) with the same batch dimensions.
  virtual Tensor mul(Tensor input, float other);
  virtual Tensor div(Tensor input, float other);
  virtual Tensor mod(Tensor input, LongType other);
  virtual Tensor mul(Tensor input, Tensor other);
  virtual Tensor softmax(Tensor input);

  /// Scaled dot product attention. q is [batch, numHeads, queryLength, headDim], k and v are
  /// [batch, numKeyValueHeads, keyValueLength, headDim]. numHeads is a multiple of
  /// numKeyValueHeads, so grouped-query and multi-query attention need no expanded k and v.
  virtual Tensor attention(Tensor q, Tensor k, Tensor v, bool causal);

    /// Scaled dot product attention of a packed (varlen) batch of queries over a paged KV cache.
    virtual Tensor pagedAttention(
      Tensor q,
      Tensor keyCache,
      Tensor valueCache,
      Tensor blockTable,
      Tensor cuSeqlensQ,
      Tensor seqlensK,
      int maxQLen,
      int maxKLen,
      bool causal);

    /// Scatter packed keys and values into their paged KV-cache slots.
    virtual void storeKVCache(
      Tensor k,
      Tensor v,
      Tensor keyCache,
      Tensor valueCache,
      Tensor slotMapping);

    /// Gated DeltaNet linear attention over a packed (varlen) batch, prefill form.
    virtual Tensor gatedDeltaNetPrefill(
      Tensor q,
      Tensor k,
      Tensor v,
      Tensor g,
      Tensor beta,
      Tensor cuSeqlens,
      Tensor stateSlots,
      Tensor state);

    virtual Tensor sample(Tensor logits, Tensor temperatures, Tensor topKs, Tensor topPs);
  virtual Tensor add(Tensor input, Tensor other);
  virtual Tensor sub(Tensor input, Tensor other);
  virtual Tensor subFloat(Tensor input, float other);
  virtual Tensor sum(Tensor input, int dim);

  /// Inclusive prefix sum along `dim`, which may be negative to count from the back: element `i`
  /// of the result is the sum of elements `0..=i` of the input. Same shape and dtype as `input`;
  /// float16 is accumulated in float32.
  virtual Tensor cumsum(Tensor input, int dim);
  virtual Tensor max(Tensor input);
  virtual Tensor eq(Tensor input, Tensor other);
  virtual Tensor square(Tensor input);
  virtual Tensor min(Tensor input);
  virtual Tensor divTensor(Tensor input, Tensor other);
  virtual Tensor neg(Tensor input);
  virtual Tensor abs(Tensor input);
  virtual Tensor exp(Tensor input);
  virtual Tensor sqrt(Tensor input);
  virtual Tensor rsqrt(Tensor input);
  virtual Tensor sigmoid(Tensor input);
  virtual Tensor tanh(Tensor input);
  virtual Tensor relu(Tensor input);
  virtual Tensor gelu(Tensor input);
  virtual Tensor silu(Tensor input);
  virtual Tensor sin(Tensor input);
  virtual Tensor cos(Tensor input);
  virtual Tensor quickGelu(Tensor input);
  virtual void fill(Tensor input, float value);
  virtual Tensor tensor(lut::Span<const int> shape, DType dtype);

  /// @brief Create an uninitialized tensor in host memory that this device can read directly,
  /// without the driver staging it through a buffer of its own.
  ///
  /// It belongs here rather than on the host's own operators because it is this device that
  /// decides what "directly" costs and which calls arrange it. Only the CUDA operators implement
  /// it; on the CPU there is nothing to arrange.
  virtual Tensor hostTensor(lut::Span<const int> shape, DType dtype);
  virtual Tensor tensorLike(Tensor input);
  virtual Tensor zeros(lut::Span<const int> shape, DType dtype);
  virtual bool all(Tensor A);
  /// Whether every pair of elements is within `rtol` relative and `atol` absolute tolerance.
  ///
  /// The tolerances have defaults because most callers mean the same pair and saying so at every
  /// call is noise. They belong to this declaration alone: an override must not restate them, and
  /// a call gets them only through an `Operators *`, which is how everything here is reached.
  virtual bool allClose(Tensor A, Tensor B, float rtol = 1e-3, float atol = 1e-5);
  virtual void print(Tensor tensor);
  virtual Tensor causalMask(int max_len);
  virtual void copy(Tensor src, Tensor dest);
  virtual Tensor swiglu(Tensor A);
  virtual Tensor toDevice(Device device, Tensor tensor);

  /// A contiguous tensor holding the same elements as `input`, in storage of this device's own.
  ///
  /// Not virtual, and not a kernel: it is `tensorLike` followed by `copy`, which every device has
  /// and which already knows how to read strides. A device with something better to do here would
  /// override `copy`, not this.
  Tensor contiguous(Tensor input);

  /// `A` and `B` joined along `dim`, the one dimension they may disagree on.
  ///
  /// Not virtual either: it allocates the joined tensor and copies each half into its own slice
  /// of it, so a device that can copy can do this.
  Tensor cat(Tensor A, Tensor B, int dim);

  virtual float elem(Tensor tensor);
  virtual bool elemBool(Tensor tensor);
  virtual void repetitionPenalty(Tensor logits, Tensor history, float weight);
  virtual Tensor cast(Tensor tensor, DType dtype);
  virtual Tensor rand(lut::Span<const int> shape, DType dtype);
  virtual Tensor randNormal(lut::Span<const int> shape);
  virtual void manualSeed(uint64_t seed);

  virtual MemorySnapshot captureMemorySnapshot();
  virtual void resetPeakMemoryStats();

  /// Hand every byte no tensor is using back to the driver, so that another process on the same
  /// device may have it. Does nothing where an allocator gives memory back as each tensor goes,
  /// which is what the CPU does and what CUDA does when it is built without its pool.
  ///
  /// Only worth calling where something large has just been let go of and nothing here will ask
  /// for it again -- a model taken off the card. Between two runs of one model it is the wrong
  /// call: it gives back the very blocks the next run would have reused, and the next run asks
  /// the driver for them again.
  virtual void releaseUnusedMemory();

  virtual DType getDefaultFloatType();

  /// Wait until all previously submitted work on this device has finished. A no-op on the CPU,
  /// where every call already blocks until completion.
  virtual void synchronize();
};

Operators *getOperators(Device::Type deviceType);
std::shared_ptr<Operators> getOperatorsSharedPtr(Device::Type deviceType);

bool isOperatorsAvailable(Device::Type deviceType);
void initOperators();
void destroyOperators();

}  // namespace fl
