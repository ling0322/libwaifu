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

/// C interface to flint, meant for language bindings. It deals in storage and in views over it, and
/// keeps no tensors of its own: what a tensor is -- which storage, which shape, who holds it -- is
/// the binding's business, and every operation reads and writes views the binding made. Every
/// function is thread-safe in the sense that it has no shared mutable state of its own, but a
/// single handle must not be used from two threads at once.

#pragma once

#include <stdint.h>

#ifdef _WIN32
#ifdef LIBWAIFU_EXPORTS
#define FLAPI __declspec(dllexport)
#else  // LIBWAIFU_EXPORTS
#define FLAPI __declspec(dllimport)
#endif  // LIBWAIFU_EXPORTS
#else   // _WIN32
#ifdef LIBWAIFU_EXPORTS
#define FLAPI __attribute__((visibility("default")))
#else  // LIBWAIFU_EXPORTS
#define FLAPI
#endif  // LIBWAIFU_EXPORTS
#endif  // _WIN32

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define FL_OK 0
#define FL_ERROR_INVALID_ARG 0x0100
#define FL_ERROR_ABORTED 0x0102


/// Storage: a run of elements of one dtype on one device, and nothing about their shape.
///
/// Made by fl_tensor_data_create() or fl_tensor_data_borrow() and freed by
/// fl_tensor_data_destroy(). The caller owns it: destroying it frees the memory at once, so every
/// view made from it has to be finished with first.
typedef struct fl_tensor_data_impl_t *fl_tensor_data_t;

/// A shape, strides and an offset over some storage -- what every operator reads and writes.
///
/// Made by fl_tensor_view_create() and released by fl_tensor_view_destroy(). A view owns nothing:
/// the fl_tensor_data_t it was made from has to outlive it, and destroying that first leaves the
/// view pointing at freed memory.
typedef struct fl_tensor_view_impl_t *fl_tensor_view_t;

/// A copy that is still on its way. Made by fl_transfer_async(), seen through by
/// fl_transfer_wait() or fl_transfer_wait_sync(), and freed by fl_transfer_destroy() either way.
typedef struct fl_transfer_impl_t *fl_transfer_t;

/// The operators of one device: the backend that computes, and what every operation below is
/// asked of rather than working it out from the views it was handed.
///
/// Made by fl_operators_create() and released by fl_operators_destroy(). The operators themselves
/// belong to the library and are shared by the whole process, so a second handle on the same
/// device makes no second backend and costs nothing; what a handle owns is a reference that keeps
/// them alive.
typedef struct fl_operators_impl_t *fl_operators_t;

typedef enum fl_dtype_t {
  FL_DTYPE_UNKNOWN = 0,
  FL_DTYPE_FLOAT = 1,
  FL_DTYPE_LONG = 2,
  FL_DTYPE_UINT8 = 3,
  FL_DTYPE_FLOAT16 = 4,
  FL_DTYPE_INT8 = 6,
  FL_DTYPE_FP4E2M0X2 = 7,
  FL_DTYPE_BOOL = 8,
  FL_DTYPE_INT32 = 9,
  FL_DTYPE_FP8E4M3 = 10,
} fl_dtype_t;

typedef enum fl_device_type_t {
  FL_DEVICE_CPU = 0,
  FL_DEVICE_CUDA = 1,
  /// Host memory page-locked by the CUDA driver. The CPU may read and write it, but it carries no
  /// operators: it is where weights wait to be copied to the GPU, not somewhere to compute.
  FL_DEVICE_CUDA_HOST = 2,
  FL_DEVICE_METAL = 3,
  FL_DEVICE_VULKAN = 4,
  /// Last, and one past the devices, so that adding one moves this and nothing else.
  FL_DEVICE_UNKNOWN = 5,
} fl_device_type_t;

/// Select the operator backends for the machine. Call once before anything else here; later calls
/// do nothing. Failures are reported through fl_get_last_error_code().
FLAPI void fl_init(void);

/// Whether this build has operators for `device` and the machine can run them. Call fl_init()
/// first; a device that is not available is reported as zero rather than as an error.
FLAPI int32_t fl_is_device_available(fl_device_type_t device, int32_t *out);

/// Error code of the last call made on this thread, or zero if it succeeded.
FLAPI int32_t fl_get_last_error_code(void);

/// Message of the last call made on this thread. Owned by the library, valid until the next call
/// on the same thread, and never NULL.
FLAPI const char *fl_get_last_error_message(void);

/// A handle on the operators of `device`. Call fl_init() first.
///
/// A device this build or this machine has no operators for is an error; asking about that
/// without failing is what fl_is_device_available() is for. FL_DEVICE_CUDA_HOST names memory
/// rather than a processor and so has no operators of its own.
FLAPI int32_t fl_operators_create(fl_device_type_t device, fl_operators_t *out);

/// Release a handle. The operators outlive it. Passing NULL has no effect.
FLAPI void fl_operators_destroy(fl_operators_t operators);

/// The device these operators compute on, which is what fl_operators_create() was given.
FLAPI int32_t fl_operators_get_device(fl_operators_t operators, fl_device_type_t *out);

/// The float type these operators work in by default.
FLAPI int32_t fl_get_default_float_type(fl_operators_t operators, fl_dtype_t *out);

// --- Storage ----------------------------------------------------------------------------------

/// Allocate uninitialized storage for `numel` elements of `dtype` on `device`. FL_DEVICE_CUDA_HOST
/// is page-locked host memory, which only a CUDA build can allocate.
/// @param numel at least one.
FLAPI int32_t fl_tensor_data_create(
    fl_device_type_t device,
    fl_dtype_t dtype,
    int64_t numel,
    fl_tensor_data_t *out);

/// Storage on the CPU that *is* `numel` elements of `dtype` at `data`, rather than a copy of them:
/// a weight file mapped into memory, read where it lies.
///
/// The bytes stay the caller's. Nothing here frees them, so they have to outlive the storage and
/// every view of it; and they are only ever read, so a view of them must not be written to. The
/// address has to be aligned to the element size, which the CPU kernels read in.
/// @param numel at least one.
FLAPI int32_t fl_tensor_data_borrow(
    const void *data,
    fl_dtype_t dtype,
    int64_t numel,
    fl_tensor_data_t *out);

/// Free the storage. Views made from it dangle afterwards and must not be used. Passing NULL has
/// no effect.
FLAPI void fl_tensor_data_destroy(fl_tensor_data_t data);

/// How many elements the storage holds.
FLAPI int32_t fl_tensor_data_get_numel(fl_tensor_data_t data, int64_t *out);
FLAPI int32_t fl_tensor_data_get_dtype(fl_tensor_data_t data, fl_dtype_t *out);
FLAPI int32_t fl_tensor_data_get_device(fl_tensor_data_t data, fl_device_type_t *out);

/// The address of the first element of storage on the host -- the CPU's memory or page-locked
/// memory -- for the caller to read or fill. Storage on a device has no address the caller may
/// touch and is refused. Storage made by fl_tensor_data_borrow() is handed back as it was given,
/// and is still only to be read.
FLAPI int32_t fl_tensor_data_get_host_ptr(fl_tensor_data_t data, void **out);

// --- Views ------------------------------------------------------------------------------------

/// Make a view of `data`: `ndim` dimensions of sizes `shape` and strides `stride`, starting
/// `offset` elements into the storage. Strides and the offset count elements, not bytes, and a
/// stride may be zero to repeat elements along a dimension.
///
/// Refused unless every element the view can reach lies inside the storage: `offset` plus the sum
/// of `(shape[i] - 1) * stride[i]` has to be below the storage's element count. The view does not
/// keep `data` alive; see fl_tensor_view_t.
/// @param shape,stride one entry per dimension; may be NULL only when `ndim` is zero.
FLAPI int32_t fl_tensor_view_create(
    fl_tensor_data_t data,
    const int32_t *shape,
    const int32_t *stride,
    int32_t ndim,
    int64_t offset,
    fl_tensor_view_t *out);

/// Release a view. The storage it was made from is not touched. Passing NULL has no effect.
FLAPI void fl_tensor_view_destroy(fl_tensor_view_t view);

FLAPI int32_t fl_tensor_view_get_dim(fl_tensor_view_t view, int32_t *out);

/// Size and stride of dimension `dim`, which may be negative to count from the back.
FLAPI int32_t fl_tensor_view_get_shape(fl_tensor_view_t view, int32_t dim, int32_t *out);
FLAPI int32_t fl_tensor_view_get_stride(fl_tensor_view_t view, int32_t dim, int32_t *out);

/// Where the view's first element is in its storage, in elements.
FLAPI int32_t fl_tensor_view_get_offset(fl_tensor_view_t view, int64_t *out);
FLAPI int32_t fl_tensor_view_get_dtype(fl_tensor_view_t view, fl_dtype_t *out);
FLAPI int32_t fl_tensor_view_get_device(fl_tensor_view_t view, fl_device_type_t *out);
FLAPI int32_t fl_tensor_view_is_contiguous(fl_tensor_view_t view, int32_t *out);

// --- Operations -------------------------------------------------------------------------------
//
// Every operation reads the views it is handed and writes its result into `out`, a view the
// caller made over storage it allocated: nothing here allocates a result. Deciding the shape and
// type of `out` is the caller's part; what is checked here is that every view is on the
// operators' own device, and for the operations whose result has the input's shape, that `out`
// has it too. A view an operation may be given or not is NULL when it is not.

/// Copy the elements of `src` into `dest`, which must have the same shape and dtype. Both on the
/// operators' device, or both on the host for the CPU operators.
FLAPI int32_t fl_copy(fl_operators_t operators, fl_tensor_view_t src, fl_tensor_view_t dest);

/// Convert the elements of `input` into `out`'s dtype.
FLAPI int32_t fl_cast(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);

/// Copy contiguous `src` into contiguous `dest` on another device, of the same shape and dtype.
/// `operators` are the ones that own the transfer: the accelerator's, whichever side of it the
/// copy starts from.
FLAPI int32_t fl_transfer(fl_operators_t operators, fl_tensor_view_t src, fl_tensor_view_t dest);

/// Start copying contiguous `src`, in page-locked host memory, to `device`, which has to be
/// FL_DEVICE_CUDA, and return before it has arrived. The storage it fills comes back in `dest`,
/// owned by the caller as any other, holding exactly `src`'s elements; it must not be read before
/// the transfer has been waited on. `src`'s storage has to stay alive until then too.
FLAPI int32_t fl_transfer_async(
    fl_tensor_view_t src,
    fl_device_type_t device,
    fl_tensor_data_t *dest,
    fl_transfer_t *out);

/// Order the work that follows behind the copy, without stopping the caller. Take it where the
/// result is about to be used, since that is where the dependency lands.
FLAPI int32_t fl_transfer_wait(fl_transfer_t transfer);

/// The same, except that it does not return until the copy has finished. For a caller about to
/// read the bytes itself.
FLAPI int32_t fl_transfer_wait_sync(fl_transfer_t transfer);

/// Free the handle, waited on or not. A transfer never waited on is a fetch that turned out not to
/// be wanted; its storage may still be destroyed afterwards, and goes back in the copy's order.
FLAPI void fl_transfer_destroy(fl_transfer_t transfer);

/// Fill every element of `tensor` with `value`.
FLAPI int32_t fl_fill(fl_operators_t operators, fl_tensor_view_t tensor, float value);

/// Fill `out` with uniform random numbers in [0, 1).
FLAPI int32_t fl_rand(fl_operators_t operators, fl_tensor_view_t out);

/// Fill `out`, a float view, from a normal distribution with mean 0 and variance 1.
FLAPI int32_t fl_randn(fl_operators_t operators, fl_tensor_view_t out);

/// Seed the random number generator these operators draw from in fl_rand() and fl_randn().
FLAPI int32_t fl_manual_seed(fl_operators_t operators, uint64_t seed);

/// Write `begin`, `begin + step`, ... into `out`, a <long> vector.
FLAPI int32_t fl_arange(fl_operators_t operators, int64_t begin, int64_t step, fl_tensor_view_t out);

/// Write a causal mask into `out` (n, n): -inf where a position may not attend and 0 elsewhere.
FLAPI int32_t fl_causal_mask(fl_operators_t operators, fl_tensor_view_t out);

/// Rows of `table` (V, D) named by `indices` <long>(...), into `out` (..., D).
FLAPI int32_t fl_lookup(
    fl_operators_t operators,
    fl_tensor_view_t table,
    fl_tensor_view_t indices,
    fl_tensor_view_t out);

/// Apply NeoX-style rotary embedding to `query` and `key` in place. `positions` is
/// <long>(numTokens) and `rotary_cache` (maxPositions, 2 * headDim).
FLAPI int32_t fl_rotary_embedding(
    fl_operators_t operators,
    fl_tensor_view_t positions,
    fl_tensor_view_t query,
    fl_tensor_view_t key,
    fl_tensor_view_t rotary_cache);

/// Root mean square normalization over the last dimension, scaled by `weight` (D).
FLAPI int32_t fl_rms_norm(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t weight,
    float eps,
    fl_tensor_view_t out);

/// Normalize the last dimension to zero mean and unit variance, then scale by `weight` and shift
/// by `bias`, either of which may be NULL.
FLAPI int32_t fl_layer_norm(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t weight,
    fl_tensor_view_t bias,
    float eps,
    fl_tensor_view_t out);

/// Normalize `input` (N, C, H, W) over each group of channels, then scale and shift per channel.
FLAPI int32_t fl_group_norm(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t weight,
    fl_tensor_view_t bias,
    int32_t groups,
    float eps,
    fl_tensor_view_t out);

/// 2-D convolution of `input` (N, C, H, W) by `weight` (K, C / groups, R, S) into `out`
/// (N, K, H', W'). `bias` (K) may be NULL.
FLAPI int32_t fl_conv2d(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t weight,
    fl_tensor_view_t bias,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t groups,
    fl_tensor_view_t out);

/// 1-D convolution of `input` (N, C, L) by `weight` (K, C / groups, R) into `out` (N, K, L').
FLAPI int32_t fl_conv1d(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t weight,
    fl_tensor_view_t bias,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t groups,
    fl_tensor_view_t out);

/// Transposed 1-D convolution of `input` (N, C, L) by `weight` (C, K / groups, R).
FLAPI int32_t fl_conv_transpose1d(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t weight,
    fl_tensor_view_t bias,
    int32_t stride,
    int32_t padding,
    int32_t output_padding,
    int32_t groups,
    fl_tensor_view_t out);

/// x + sin(alpha * x)^2 / (beta + eps), per channel of `input` (N, C, L). `beta` may be NULL.
FLAPI int32_t fl_snake(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t alpha,
    fl_tensor_view_t beta,
    float eps,
    fl_tensor_view_t out);

/// Short time Fourier transform of `input` (N, 1, L) against `window` (n_fft), into `out`
/// (N, 2 * (n_fft / 2 + 1), frames).
FLAPI int32_t fl_stft(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t window,
    int32_t n_fft,
    int32_t hop,
    int32_t centered,
    fl_tensor_view_t out);

/// The inverse of fl_stft(), into `out` (N, 1, L).
FLAPI int32_t fl_istft(
    fl_operators_t operators,
    fl_tensor_view_t spectrum,
    fl_tensor_view_t window,
    int32_t n_fft,
    int32_t hop,
    int32_t centered,
    fl_tensor_view_t out);

/// Repeat each pixel of `input` (N, C, H, W) `scale` times along both spatial axes.
FLAPI int32_t fl_upsample_nearest2d(
    fl_operators_t operators,
    fl_tensor_view_t input,
    int32_t scale,
    fl_tensor_view_t out);

/// Resize the last dimension of `input` to `out`'s, each output copying the nearest input before
/// it, as torch's interpolate(mode="nearest") does.
FLAPI int32_t fl_upsample_nearest1d(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t out);

/// Matrix multiplication, batched over the leading dimensions.
FLAPI int32_t fl_matmul(
    fl_operators_t operators,
    fl_tensor_view_t a,
    fl_tensor_view_t b,
    fl_tensor_view_t out);

/// Whether `device` can run the FP8 matrix multiplication, which today means CUDA on sm_80 or
/// newer. Every other device is reported as zero rather than as an error.
FLAPI int32_t fl_fp8_available(fl_device_type_t device, int32_t *out);

/// Quantize `x` <float16>(rows, k) to E4M3 with one scale per row, writing the codes into `data`
/// <fp8e4m3>(rows, k) and the scales into `channel_scale` <float>(rows). k a multiple of 16.
FLAPI int32_t fl_fp8_quantize(
    fl_tensor_view_t x,
    fl_tensor_view_t data,
    fl_tensor_view_t channel_scale);

/// The inverse of fl_fp8_quantize(), into `out` <float16>(rows, k).
FLAPI int32_t fl_fp8_dequantize(
    fl_tensor_view_t data,
    fl_tensor_view_t channel_scale,
    fl_tensor_view_t out);

/// `a` <float16>(..., k) times the transpose of the FP8 weight `data` (rows, k) with one scale per
/// row, into `out` <float16>(..., rows). rows a multiple of 8, k of 16.
FLAPI int32_t fl_fp8_matmul(
    fl_tensor_view_t a,
    fl_tensor_view_t data,
    fl_tensor_view_t channel_scale,
    fl_tensor_view_t out);

/// fl_fp8_matmul() for a weight with one <float> scale for the whole tensor.
FLAPI int32_t fl_fp8_matmul_tensor_scale(
    fl_tensor_view_t a,
    fl_tensor_view_t data,
    fl_tensor_view_t scale,
    fl_tensor_view_t out);

/// Element-wise binary operations. `b` has `a`'s shape -- a caller broadcasting a smaller operand
/// hands over a view of it expanded to `a`'s shape -- and so has `out`, which fl_eq() writes as
/// <bool> and the others in `a`'s dtype.
FLAPI int32_t fl_add(fl_operators_t operators, fl_tensor_view_t a, fl_tensor_view_t b, fl_tensor_view_t out);
FLAPI int32_t fl_sub(fl_operators_t operators, fl_tensor_view_t a, fl_tensor_view_t b, fl_tensor_view_t out);
FLAPI int32_t fl_mul(fl_operators_t operators, fl_tensor_view_t a, fl_tensor_view_t b, fl_tensor_view_t out);
FLAPI int32_t fl_div(fl_operators_t operators, fl_tensor_view_t a, fl_tensor_view_t b, fl_tensor_view_t out);
FLAPI int32_t fl_eq(fl_operators_t operators, fl_tensor_view_t a, fl_tensor_view_t b, fl_tensor_view_t out);

/// Element-wise `input` * `other`, `input` / `other` and, for <long>, `input` % `other`.
FLAPI int32_t fl_mul_scalar(fl_operators_t operators, fl_tensor_view_t input, float other, fl_tensor_view_t out);
FLAPI int32_t fl_div_scalar(fl_operators_t operators, fl_tensor_view_t input, float other, fl_tensor_view_t out);
FLAPI int32_t fl_mod_scalar(fl_operators_t operators, fl_tensor_view_t input, int64_t other, fl_tensor_view_t out);

/// Element-wise unary operations, `out` of `input`'s shape and dtype: x^2, -x, |x|, e^x, the
/// natural logarithm, rounding half to even, sqrt(x), 1/sqrt(x), 1/(1+e^-x), tanh(x), max(x, 0),
/// the exact GELU, x * sigmoid(x), sin(x), cos(x), and x * sigmoid(1.702 * x).
FLAPI int32_t fl_square(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);
FLAPI int32_t fl_neg(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);
FLAPI int32_t fl_abs(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);
FLAPI int32_t fl_exp(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);
FLAPI int32_t fl_log(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);
FLAPI int32_t fl_round(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);
FLAPI int32_t fl_sqrt(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);
FLAPI int32_t fl_rsqrt(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);
FLAPI int32_t fl_sigmoid(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);
FLAPI int32_t fl_tanh(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);
FLAPI int32_t fl_relu(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);
FLAPI int32_t fl_gelu(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);
FLAPI int32_t fl_silu(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);
FLAPI int32_t fl_sin(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);
FLAPI int32_t fl_cos(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);
FLAPI int32_t fl_quick_gelu(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);

/// Softmax over the last dimension, into `out` of `input`'s shape.
FLAPI int32_t fl_softmax(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);

/// Gated linear units over the last dimension of `input` (..., D), D even, into `out` (..., D / 2):
/// swish and GELU gated respectively.
FLAPI int32_t fl_swiglu(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);
FLAPI int32_t fl_geglu(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);

/// Sum over dimension `dim`, non-negative, into `out`, `input`'s shape without it.
FLAPI int32_t fl_sum(fl_operators_t operators, fl_tensor_view_t input, int32_t dim, fl_tensor_view_t out);

/// Inclusive prefix sum along dimension `dim`, non-negative, into `out` of `input`'s shape.
FLAPI int32_t fl_cumsum(fl_operators_t operators, fl_tensor_view_t input, int32_t dim, fl_tensor_view_t out);

/// Largest and smallest element of the last dimension, into `out`, `input`'s shape without it.
FLAPI int32_t fl_max(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);
FLAPI int32_t fl_min(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out);

/// Scaled dot product attention. `q` is (N, nHead, L, D) and `k` and `v` are (N, nKvHead, S, D),
/// where nKvHead may divide nHead; `out` is (N, nHead, L, D). `causal` masks the future
/// positions, aligned to the bottom right of the score matrix.
FLAPI int32_t fl_attention(
    fl_operators_t operators,
    fl_tensor_view_t q,
    fl_tensor_view_t k,
    fl_tensor_view_t v,
    int32_t causal,
    fl_tensor_view_t out);

/// Whether this build has fl_paged_attention(), which rides on the FlashAttention kernels.
FLAPI int32_t fl_paged_attention_available(int32_t *out);

/// Scaled dot product attention over a packed batch of queries `q` (totalQLen, nHead, D) reading a
/// paged KV cache, into `out` of `q`'s shape.
FLAPI int32_t fl_paged_attention(
    fl_operators_t operators,
    fl_tensor_view_t q,
    fl_tensor_view_t key_cache,
    fl_tensor_view_t value_cache,
    fl_tensor_view_t block_table,
    fl_tensor_view_t cu_seqlens_q,
    fl_tensor_view_t seqlens_k,
    int32_t max_q_len,
    int32_t max_k_len,
    int32_t causal,
    fl_tensor_view_t out);

/// Scatter `k` and `v` (numTokens, nKvHead, D) into the paged caches at `slot_mapping`.
FLAPI int32_t fl_store_kv_cache(
    fl_operators_t operators,
    fl_tensor_view_t k,
    fl_tensor_view_t v,
    fl_tensor_view_t key_cache,
    fl_tensor_view_t value_cache,
    fl_tensor_view_t slot_mapping);

/// Sample one label per row of `logits` (rows, vocab) with per-row parameters, into `out`
/// <long>(rows).
FLAPI int32_t fl_sample_with_params(
    fl_operators_t operators,
    fl_tensor_view_t logits,
    fl_tensor_view_t temperatures,
    fl_tensor_view_t top_ks,
    fl_tensor_view_t top_ps,
    fl_tensor_view_t out);

/// Divide the logits of the tokens in `history` by `weight`, in place.
FLAPI int32_t fl_repetition_penalty(
    fl_operators_t operators,
    fl_tensor_view_t logits,
    fl_tensor_view_t history,
    float weight);

/// Whether every pair of elements is within `rtol` relative and `atol` absolute tolerance.
FLAPI int32_t fl_all_close(
    fl_operators_t operators,
    fl_tensor_view_t a,
    fl_tensor_view_t b,
    float rtol,
    float atol,
    int32_t *out);

/// Whether every element of a <bool> view is true.
FLAPI int32_t fl_all(fl_operators_t operators, fl_tensor_view_t tensor, int32_t *out);

/// The single element of a one-element view, as a float.
FLAPI int32_t fl_elem(fl_operators_t operators, fl_tensor_view_t tensor, float *out);

/// Print the view to stdout.
FLAPI int32_t fl_print(fl_operators_t operators, fl_tensor_view_t tensor);

/// The memory usage of one device. A device that does not report its usage, which is what the CPU
/// backend does, reports zero for every field.
typedef struct fl_memory_snapshot_t {
  int64_t total;
  int64_t free;
  int64_t allocated;
  int64_t peak_allocated;
} fl_memory_snapshot_t;

/// Measure the memory usage of `device`.
FLAPI int32_t fl_memory_capture(fl_device_type_t device, fl_memory_snapshot_t *out);

/// Set the peak allocated bytes of `device` back to zero, so that the next measurement covers only
/// what happens from here.
FLAPI int32_t fl_memory_reset_peak_stats(fl_device_type_t device);

/// Give every byte of `device` that no tensor is using back to the driver, so that another process
/// may have it. A device whose allocator hands memory back as each tensor goes does nothing. Worth
/// calling only where something large has just been let go of and nothing is about to ask for it
/// again -- a model taken off the card -- since it hands back the blocks the next allocation would
/// otherwise have reused.
FLAPI int32_t fl_memory_release_unused(fl_device_type_t device);

/// What fl_set_fatal_handler() registers.
typedef void (*fl_fatal_handler_t)(void);

/// Register `handler` to run just before the library ends the process.
///
/// Little does. A check that fails inside an operator -- two tensors on different devices meeting
/// at a convolution, a shape no kernel was written for -- comes back as FL_ERROR_ABORTED with the
/// message, so a broken invariant reaches a caller the same way a bad argument does. What is left
/// is the path with nothing to report to: reaching code that was never written, where there is no
/// call still standing to return a code to.
///
/// That one prints what went wrong and calls abort(), and the message lands on top of whatever a
/// caller that owns the screen had drawn. This runs first, before anything is printed, which is
/// that caller's one chance to give the screen back.
///
/// It runs on whichever thread failed, and the process is going down either way, so it should do
/// the one thing it is there for and return. Passing NULL clears it.
FLAPI void fl_set_fatal_handler(fl_fatal_handler_t handler);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
