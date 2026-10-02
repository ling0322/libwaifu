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

#include <memory>

#include "lutil/random.h"
#include "flint/device.h"
#include "flint/memory.h"
#include "flint/tensor.h"
#include "flint/tensor_view.h"

namespace fl {

/// @brief What a device computes with: one kernel per method, reading the views it is given and
/// writing the one it is told to write.
///
/// Nothing here allocates a tensor anyone else sees, and nothing here decides what shape or type
/// a result has. Both belong to `flint/functional.h`, which works out the result's shape and type
/// from the inputs, allocates it on the right device, and then hands every tensor in as a
/// TensorView -- the output included, as the last argument. An operator checks what it is handed
/// against what it can do and writes the output; that is the whole of its job. Temporaries a
/// kernel needs for itself it still makes for itself, and they never leave it.
///
/// The views borrow their storage for the call and no longer. No operator keeps one after it
/// returns: work a device queues instead of running is ordered against later work on the same
/// memory, which is what lets the caller drop a tensor the moment the call is made.
class Operators {
 public:
  virtual ~Operators() = default;

  /// Write `begin`, `begin + step`, ... into `out` <int64>(n).
  virtual void arangeLong(LongType begin, LongType step, TensorView out);
  virtual void lookup(TensorView table, TensorView indices, TensorView out);
  virtual void rotaryEmbedding(
      TensorView positions,
      TensorView query,
      TensorView key,
      TensorView rotaryCache);
  virtual void rmsNorm(TensorView input, TensorView weight, float eps, TensorView out);

  /// Normalize the last dimension of `input` to zero mean and unit variance, then scale by
  /// `weight` and shift by `bias`; either may be empty.
  virtual void layerNorm(
      TensorView input,
      TensorView weight,
      TensorView bias,
      float eps,
      TensorView out);

  /// Normalize `input` <float16>(N, C, H, W) over each group of channels and the space it covers,
  /// then scale and shift per channel. `weight` and `bias` are (C), and either may be empty.
  virtual void groupNorm(
      TensorView input,
      TensorView weight,
      TensorView bias,
      int groups,
      float eps,
      TensorView out);

  /// Repeat each pixel of `input` <float16>(N, C, H, W) `scale` times along both spatial axes.
  virtual void upsampleNearest2d(TensorView input, int scale, TensorView out);

  /// Resize the last dimension of `input` to `size` by taking, for output `j`, the input element
  /// `min(floor(j * scale), length - 1)` with `scale = float(length) / size` worked out and
  /// multiplied in float32 -- `F.interpolate(size=..., mode="nearest")`, frame for frame,
  /// including where the float32 product floors one below an exact integer. `size` is the length
  /// of `out`'s last dimension.
  virtual void upsampleNearest1d(TensorView input, TensorView out);

  /// The gated linear unit of `swiglu` with a GELU in place of the SiLU.
  virtual void geglu(TensorView input, TensorView out);

  virtual void matmul(TensorView A, TensorView B, TensorView out);

  /// 2-D convolution of `input` <float16|float>(N, C, H, W) by `weight` (K, C / groups, R, S),
  /// with an optional per-channel `bias` (K). Square stride, padding and dilation throughout.
  virtual void conv2d(
      TensorView input,
      TensorView weight,
      TensorView bias,
      int stride,
      int padding,
      int dilation,
      int groups,
      TensorView out);

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
  virtual void conv1d(
      TensorView input,
      TensorView weight,
      TensorView bias,
      int stride,
      int padding,
      int dilation,
      int groups,
      TensorView out);

  /// Transposed 1-D convolution of `input` <float16|float>(N, C, L) by `weight`
  /// (C, K / groups, R), with an optional per-channel `bias` (K). The result is (N, K, Lout),
  /// Lout being (L - 1) * stride - 2 * padding + R + outputPadding: input position `l`
  /// contributes the whole kernel starting at `l * stride`. What a vocoder upsamples with.
  ///
  /// No backend implements this. `waifu::audio::conv_transpose1d` composes it as one `matmul`
  /// and ceil(R / stride) shifted additions, for one group only.
  virtual void convTranspose1d(
      TensorView input,
      TensorView weight,
      TensorView bias,
      int stride,
      int padding,
      int outputPadding,
      int groups,
      TensorView out);

  /// The activation a BigVGAN is built from, per channel of `input` <float16|float>(N, C, L):
  /// `x + sin(alpha * x)^2 / (beta + eps)`. `alpha` and `beta` are (C) and already exponentiated
  /// -- a checkpoint trained with `alpha_logscale` stores their logarithm, and raising it belongs
  /// to whoever reads the weight. An empty `beta` means beta is alpha, the plain snake.
  ///
  /// No backend implements this. `waifu::audio::snake` composes it out of the elementwise
  /// operators, which costs about six passes over the tensor where a fused kernel would cost one.
  /// Of everything named here this is the one whose kernel would most likely pay for itself: it
  /// is memory bound and a vocoder applies it eighteen times per upsampling stage.
  virtual void snake(
      TensorView input,
      TensorView alpha,
      TensorView beta,
      float eps,
      TensorView out);

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
  virtual void stft(
      TensorView input,
      TensorView window,
      int nFft,
      int hop,
      bool centered,
      TensorView out);

  /// The inverse of `stft`: `spectrum` <float>(N, 2 * (nFft / 2 + 1), frames) against `window`
  /// (nFft), as <float>(N, 1, L). Overlap-add, divided by the window's own overlap so that the
  /// inverse of a transform is the signal it was taken of.
  ///
  /// No backend implements this. `waifu::audio::istft` composes it out of `conv_transpose1d`.
  virtual void istft(
      TensorView spectrum,
      TensorView window,
      int nFft,
      int hop,
      bool centered,
      TensorView out);

  virtual void mul(TensorView input, float other, TensorView out);
  virtual void div(TensorView input, float other, TensorView out);
  virtual void mod(TensorView input, LongType other, TensorView out);
  virtual void mul(TensorView input, TensorView other, TensorView out);
  virtual void softmax(TensorView input, TensorView out);

  /// Scaled dot product attention. q is [batch, numHeads, queryLength, headDim], k and v are
  /// [batch, numKeyValueHeads, keyValueLength, headDim]. numHeads is a multiple of
  /// numKeyValueHeads, so grouped-query and multi-query attention need no expanded k and v.
  ///
  /// A device without a kernel of its own gets the composition in `flint/functional.h` --
  /// matmul, softmax, matmul, a block of queries at a time -- written into `out`.
  virtual void attention(TensorView q, TensorView k, TensorView v, bool causal, TensorView out);

  /// Scaled dot product attention of a packed (varlen) batch of queries over a paged KV cache.
  virtual void pagedAttention(
      TensorView q,
      TensorView keyCache,
      TensorView valueCache,
      TensorView blockTable,
      TensorView cuSeqlensQ,
      TensorView seqlensK,
      int maxQLen,
      int maxKLen,
      bool causal,
      TensorView out);

  /// Scatter packed keys and values into their paged KV-cache slots.
  virtual void storeKVCache(
      TensorView k,
      TensorView v,
      TensorView keyCache,
      TensorView valueCache,
      TensorView slotMapping);

  /// Gated DeltaNet linear attention over a packed (varlen) batch, prefill form. `state` is
  /// updated in place; the attention output is written to `out`.
  virtual void gatedDeltaNetPrefill(
      TensorView q,
      TensorView k,
      TensorView v,
      TensorView g,
      TensorView beta,
      TensorView cuSeqlens,
      TensorView stateSlots,
      TensorView state,
      TensorView out);

  virtual void sample(
      TensorView logits,
      TensorView temperatures,
      TensorView topKs,
      TensorView topPs,
      TensorView out);
  virtual void add(TensorView input, TensorView other, TensorView out);
  virtual void sub(TensorView input, TensorView other, TensorView out);
  virtual void sum(TensorView input, int dim, TensorView out);

  /// Inclusive prefix sum along `dim`, which may be negative to count from the back: element `i`
  /// of the result is the sum of elements `0..=i` of the input. Same shape and dtype as `input`;
  /// float16 is accumulated in float32.
  virtual void cumsum(TensorView input, int dim, TensorView out);
  virtual void max(TensorView input, TensorView out);
  virtual void eq(TensorView input, TensorView other, TensorView out);
  virtual void square(TensorView input, TensorView out);
  virtual void min(TensorView input, TensorView out);
  virtual void divTensor(TensorView input, TensorView other, TensorView out);
  virtual void neg(TensorView input, TensorView out);
  virtual void abs(TensorView input, TensorView out);
  virtual void exp(TensorView input, TensorView out);
  /// The natural logarithm; zero gives -inf and a negative number NaN, as `logf` does.
  virtual void log(TensorView input, TensorView out);
  /// To the nearest integer, a tie going to the even one -- 0.5 to 0, 1.5 and 2.5 to 2 -- which is
  /// torch.round and not C's round(). The result keeps the input's float type.
  virtual void round(TensorView input, TensorView out);
  virtual void sqrt(TensorView input, TensorView out);
  virtual void rsqrt(TensorView input, TensorView out);
  virtual void sigmoid(TensorView input, TensorView out);
  virtual void tanh(TensorView input, TensorView out);
  virtual void relu(TensorView input, TensorView out);
  virtual void gelu(TensorView input, TensorView out);
  virtual void silu(TensorView input, TensorView out);
  virtual void sin(TensorView input, TensorView out);
  virtual void cos(TensorView input, TensorView out);
  virtual void quickGelu(TensorView input, TensorView out);
  virtual void fill(TensorView input, float value);

  virtual bool all(TensorView A);
  /// Whether every pair of elements is within `rtol` relative and `atol` absolute tolerance.
  ///
  /// The tolerances have defaults because most callers mean the same pair and saying so at every
  /// call is noise. They belong to this declaration alone: an override must not restate them, and
  /// a call gets them only through an `Operators *`, which is how everything here is reached.
  virtual bool allClose(TensorView A, TensorView B, float rtol = 1e-3, float atol = 1e-5);
  virtual void print(TensorView tensor);

  /// Write the causal mask of `out` <default float>(n, n): zero on and below the diagonal, minus
  /// infinity above it.
  virtual void causalMask(TensorView out);

  /// Copy the elements of `src` into `dest`, both on this device and of one shape; either may
  /// have any strides.
  virtual void copy(TensorView src, TensorView dest);
  virtual void swiglu(TensorView A, TensorView out);

  /// Copy the elements of contiguous `src` into contiguous `dest` of the same shape and type, the
  /// two on different devices, one of them this one. What moving a tensor between devices is once
  /// `dest` has been allocated where it is going.
  virtual void transfer(TensorView src, TensorView dest);

  virtual float elem(TensorView tensor);
  virtual bool elemBool(TensorView tensor);
  virtual void repetitionPenalty(TensorView logits, TensorView history, float weight);

  /// Convert the elements of `input` to the type of `out`, which is of the same shape.
  virtual void cast(TensorView input, TensorView out);

  /// Fill `out` with numbers drawn uniformly from [0, 1).
  virtual void rand(TensorView out);
  /// Fill `out` <float>, with numbers drawn from the standard normal distribution.
  virtual void randNormal(TensorView out);
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
