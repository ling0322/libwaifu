//! The operations of `fl::Operators`, one function per method.
//!
//! The C interface takes the operators every call runs on rather than working them out from
//! the tensors it was handed. These functions do that working out: an operation runs on the
//! device its inputs live on, so the operators are those of the first tensor it reads.
//!
//! Every function here takes its inputs by reference and returns a new tensor, except for the few
//! that write into a tensor the caller already has, which take it as `&mut`. That `&mut` is a
//! statement of intent rather than a guarantee: a tensor shares its storage with its clones and
//! its views, so an operation that writes in place can be observed through any of them.
//!
//! An operation runs on the device its inputs live on, and all of them must agree on that device.
//!
//! # Shapes are checked fatally
//!
//! These functions return a [`Result`], but it does not cover as much as a Rust API usually would.
//! The operators check the shapes and dtypes they are handed with the C++ library's own fatal
//! check, which prints a message and aborts the process; so does reaching an operation the device
//! has no kernel for. Neither is an unwind the binding could catch and turn into an `Err`, so the
//! shapes each function documents are a requirement on the caller rather than something to hand
//! over and let the error path sort out.
//!
//! ```no_run
//! use waifu::flint::{functional as F, Tensor};
//!
//! let a = Tensor::from_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0])?;
//! let b = Tensor::from_f32(&[2, 2], &[1.0, 0.0, 0.0, 1.0])?;
//! assert_eq!(F::matmul(&a, &b)?.to_vec_f32()?, vec![1.0, 2.0, 3.0, 4.0]);
//! # Ok::<(), waifu::flint::Error>(())
//! ```

use super::operators::{copy_operators, operators_of, raw_operators};
use super::{check, ffi, DType, Device, Nvfp4Tensor, Result, Tensor};

/// Reduce over the last dimension, the default of [`sum`] and [`max`].
pub const LAST_DIM: i32 = -1;

/// A 1-D tensor holding the values of `[begin, end)` taken `step` at a time.
pub fn arange(begin: i64, end: i64, step: i64, device: Device) -> Result<Tensor> {
    let operators = raw_operators(device)?;
    Tensor::produce(|out| unsafe { ffi::fl_arange(operators, begin, end, step, out) })
}

/// A tensor filled with uniform random numbers in `[0, 1)`.
pub fn rand(shape: &[i32], dtype: DType, device: Device) -> Result<Tensor> {
    let operators = raw_operators(device)?;
    Tensor::produce(|out| unsafe {
        ffi::fl_rand(
            operators,
            shape.as_ptr(),
            shape.len() as i32,
            dtype as i32,
            out,
        )
    })
}

/// A float tensor drawn from a normal distribution with mean 0 and variance 1.
pub fn randn(shape: &[i32], device: Device) -> Result<Tensor> {
    let operators = raw_operators(device)?;
    Tensor::produce(|out| unsafe {
        ffi::fl_randn(operators, shape.as_ptr(), shape.len() as i32, out)
    })
}

/// Seed the generator that [`rand`] and [`randn`] draw from on `device`.
pub fn manual_seed(device: Device, seed: u64) -> Result<()> {
    let operators = raw_operators(device)?;
    check(unsafe { ffi::fl_manual_seed(operators, seed) })
}

/// The rows of `table` `<float>(V, D)` named by `indices` `<long>(..)`, which gains a trailing
/// dimension of `D`.
pub fn lookup(table: &Tensor, indices: &Tensor) -> Result<Tensor> {
    let operators = operators_of(table)?;
    Tensor::produce(|out| unsafe { ffi::fl_lookup(operators, table.raw, indices.raw, out) })
}

/// Apply NeoX-style rotary embedding to `query` and `key` in place.
///
/// `positions` is `<long>(numTokens)`, `query` and `key` are `<float>(numTokens, numHeads,
/// headDim)`, and `rotary_cache` is `<float>(maxPositions, 2 * headDim)`, each row holding a
/// cosine half followed by a sine half.
pub fn rotary_embedding(
    positions: &Tensor,
    query: &mut Tensor,
    key: &mut Tensor,
    rotary_cache: &Tensor,
) -> Result<()> {
    let operators = operators_of(positions)?;
    check(unsafe {
        ffi::fl_rotary_embedding(
            operators,
            positions.raw,
            query.raw,
            key.raw,
            rotary_cache.raw,
        )
    })
}

/// Root mean square layer normalization over the last dimension of `input`, scaled by `weight`
/// `<float>(D)`.
pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_rms_norm(operators, input.raw, weight.raw, eps, out) })
}

/// Matrix multiplication, batched over the leading dimensions.
/// Normalize the last dimension of `input` to zero mean and unit variance, then scale by `weight`
/// and shift by `bias`. Unlike an RMS norm this subtracts the mean, which is what CLIP and a
/// diffusion U-Net both do.
pub fn layer_norm(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let weight = weight.map(|t| t.raw).unwrap_or(std::ptr::null_mut());
    let bias = bias.map(|t| t.raw).unwrap_or(std::ptr::null_mut());

    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe {
        ffi::fl_layer_norm(operators, input.raw, weight, bias, eps, out)
    })
}

/// x * sigmoid(1.702 * x), which OpenAI's CLIP uses in place of GELU.
pub fn quick_gelu(input: &Tensor) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_quick_gelu(operators, input.raw, out) })
}

/// A 2-D convolution of `input` `(N, C, H, W)` by `weight` `(K, C / groups, R, S)`, with an
/// optional per-channel bias. The stride, padding and dilation are square.
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
) -> Result<Tensor> {
    let bias = bias.map(|t| t.raw).unwrap_or(std::ptr::null_mut());
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe {
        ffi::fl_conv2d(
            operators, input.raw, weight.raw, bias, stride, padding, dilation, groups, out,
        )
    })
}

/// A 1-D convolution of `input` `(N, C, L)` by `weight` `(K, C / groups, R)`.
///
/// No device implements this yet, so every call fails; [`crate::audio::conv1d`] is what computes
/// a 1-D convolution today, out of [`conv2d`]. This is the seat a kernel takes when one is
/// written. See `Operators::conv1d`.
pub fn conv1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
) -> Result<Tensor> {
    let bias = bias.map(|t| t.raw).unwrap_or(std::ptr::null_mut());
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe {
        ffi::fl_conv1d(
            operators, input.raw, weight.raw, bias, stride, padding, dilation, groups, out,
        )
    })
}

/// A transposed 1-D convolution of `input` `(N, C, L)` by `weight` `(C, K / groups, R)`.
///
/// No device implements this yet; [`crate::audio::conv_transpose1d`] is what computes one today.
pub fn conv_transpose1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: i32,
    padding: i32,
    output_padding: i32,
    groups: i32,
) -> Result<Tensor> {
    let bias = bias.map(|t| t.raw).unwrap_or(std::ptr::null_mut());
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe {
        ffi::fl_conv_transpose1d(
            operators,
            input.raw,
            weight.raw,
            bias,
            stride,
            padding,
            output_padding,
            groups,
            out,
        )
    })
}

/// `x + sin(alpha * x)^2 / (beta + eps)`, per channel of `input` `(N, C, L)`.
///
/// No device implements this yet; [`crate::audio::snake`] is what computes one today.
pub fn snake(input: &Tensor, alpha: &Tensor, beta: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    let beta = beta.map(|t| t.raw).unwrap_or(std::ptr::null_mut());
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_snake(operators, input.raw, alpha.raw, beta, eps, out) })
}

/// The short time Fourier transform of `input` `(N, 1, L)` against `window` `(n_fft)`.
///
/// No device implements this yet; [`crate::audio::stft`] is what computes one today.
pub fn stft(
    input: &Tensor,
    window: &Tensor,
    n_fft: i32,
    hop: i32,
    centered: bool,
) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe {
        ffi::fl_stft(
            operators,
            input.raw,
            window.raw,
            n_fft,
            hop,
            centered as i32,
            out,
        )
    })
}

/// The inverse of [`stft`].
///
/// No device implements this yet; [`crate::audio::istft`] is what computes one today.
pub fn istft(
    spectrum: &Tensor,
    window: &Tensor,
    n_fft: i32,
    hop: i32,
    centered: bool,
) -> Result<Tensor> {
    let operators = operators_of(spectrum)?;
    Tensor::produce(|out| unsafe {
        ffi::fl_istft(
            operators,
            spectrum.raw,
            window.raw,
            n_fft,
            hop,
            centered as i32,
            out,
        )
    })
}

/// Normalize `input` `(N, C, H, W)` over each group of channels together with the space it covers,
/// then scale and shift per channel.
pub fn group_norm(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    groups: i32,
    eps: f32,
) -> Result<Tensor> {
    let weight = weight.map(|t| t.raw).unwrap_or(std::ptr::null_mut());
    let bias = bias.map(|t| t.raw).unwrap_or(std::ptr::null_mut());
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe {
        ffi::fl_group_norm(operators, input.raw, weight, bias, groups, eps, out)
    })
}

/// Repeat each pixel of `input` `(N, C, H, W)` `scale` times along both spatial axes.
pub fn upsample_nearest2d(input: &Tensor, scale: i32) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_upsample_nearest2d(operators, input.raw, scale, out) })
}

pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let operators = operators_of(a)?;
    Tensor::produce(|out| unsafe { ffi::fl_matmul(operators, a.raw, b.raw, out) })
}

/// `a` `<float16>(..., k)` times the transpose of an NVFP4 `weight` `(rows, k)`, as
/// `<float16>(..., rows)`. `a` is quantized on the way in, because the block scaled tensor cores
/// take no other kind of operand; `rows` has to be a multiple of 8.
pub fn nvfp4_matmul(a: &Tensor, weight: &Nvfp4Tensor) -> Result<Tensor> {
    Tensor::produce(|out| unsafe {
        ffi::fl_nvfp4_matmul(
            a.raw,
            weight.data.raw,
            weight.block_scale.raw,
            weight.global_scale.raw,
            out,
        )
    })
}

/// `a` `(..., k)` times the transpose of an FP8 weight `(rows, k)`, as `(..., rows)`, in the
/// float type the weight's device computes in.
///
/// The weight arrives as the two tensors it is made of -- `<fp8e4m3>(rows, k)` and the
/// `<float>(rows)` scale, row `r` meaning `weight[r] * channel_scale[r]` -- rather than as one
/// value holding both. [`Fp8Tensor`](super::Fp8Tensor) is what quantizing produces and hands the
/// pair out of; a weight a package stored quantized never becomes one, because it arrives as two
/// ordinary tensors under two ordinary names and there is nothing to be gained by pairing them up
/// again on the way to a call that takes them apart.
///
/// Unlike [`nvfp4_matmul`] the activation is not narrowed: the multiply happens at full width
/// either way and only the weight is narrow. So this is `<float16>` in and out, with `rows` a
/// multiple of 8 and `k` a multiple of 16. CUDA is the only device with the kernels for it.
pub fn fp8_matmul(a: &Tensor, weight: &Tensor, channel_scale: &Tensor) -> Result<Tensor> {
    Tensor::produce(|out| unsafe { ffi::fl_fp8_matmul(a.raw, weight.raw, channel_scale.raw, out) })
}

/// [`fp8_matmul`] for a weight with one scale for the whole tensor rather than one per row:
/// `a` `(..., k)` times the transpose of `<fp8e4m3>(rows, k)`, times the one `<float>` in
/// `scale`, as `(..., rows)`.
///
/// This is how a checkpoint quantized elsewhere usually arrives -- a `weight` of E4M3 codes and a
/// single `weight_scale` -- and it is multiplied as stored. `scale` holds one element, whatever its
/// shape, on the weight's device; the kernel reads it there. The same constraints as
/// [`fp8_matmul`] otherwise: `<float16>` in and out, `rows` a multiple of 8, `k` of 16, CUDA only.
pub fn fp8_matmul_tensor_scale(a: &Tensor, weight: &Tensor, scale: &Tensor) -> Result<Tensor> {
    Tensor::produce(|out| unsafe {
        ffi::fl_fp8_matmul_tensor_scale(a.raw, weight.raw, scale.raw, out)
    })
}

/// Element-wise `a * b`, broadcasting `b` over the leading dimensions of `a`.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let operators = operators_of(a)?;
    Tensor::produce(|out| unsafe { ffi::fl_mul(operators, a.raw, b.raw, out) })
}

/// Element-wise `a / b`, broadcasting `b` over the leading dimensions of `a`.
pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let operators = operators_of(a)?;
    Tensor::produce(|out| unsafe { ffi::fl_div(operators, a.raw, b.raw, out) })
}

/// Element-wise `a + b`, broadcasting `b` over the leading dimensions of `a`.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let operators = operators_of(a)?;
    Tensor::produce(|out| unsafe { ffi::fl_add(operators, a.raw, b.raw, out) })
}

/// Element-wise `a - b`, broadcasting `b` over the leading dimensions of `a`.
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let operators = operators_of(a)?;
    Tensor::produce(|out| unsafe { ffi::fl_sub(operators, a.raw, b.raw, out) })
}

/// Element-wise `a == b`, as a [`DType::Bool`] tensor of the same shape.
pub fn eq(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let operators = operators_of(a)?;
    Tensor::produce(|out| unsafe { ffi::fl_eq(operators, a.raw, b.raw, out) })
}

/// Element-wise `input * other`.
pub fn mul_scalar(input: &Tensor, other: f32) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_mul_scalar(operators, input.raw, other, out) })
}

/// Element-wise `input / other`.
pub fn div_scalar(input: &Tensor, other: f32) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_div_scalar(operators, input.raw, other, out) })
}

/// Element-wise `input % other`, for a [`DType::Long`] tensor.
pub fn mod_scalar(input: &Tensor, other: i64) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_mod_scalar(operators, input.raw, other, out) })
}

/// Element-wise `input` squared.
pub fn square(input: &Tensor) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_square(operators, input.raw, out) })
}

/// Element-wise `-x`.
pub fn neg(input: &Tensor) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_neg(operators, input.raw, out) })
}

/// Element-wise `|x|`.
pub fn abs(input: &Tensor) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_abs(operators, input.raw, out) })
}

/// Element-wise `e^x`.
pub fn exp(input: &Tensor) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_exp(operators, input.raw, out) })
}

/// Element-wise square root.
pub fn sqrt(input: &Tensor) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_sqrt(operators, input.raw, out) })
}

/// Element-wise reciprocal square root, `1/sqrt(x)`.
pub fn rsqrt(input: &Tensor) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_rsqrt(operators, input.raw, out) })
}

/// Element-wise `1/(1 + e^-x)`.
pub fn sigmoid(input: &Tensor) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_sigmoid(operators, input.raw, out) })
}

/// Element-wise hyperbolic tangent.
pub fn tanh(input: &Tensor) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_tanh(operators, input.raw, out) })
}

/// Element-wise `max(x, 0)`.
pub fn relu(input: &Tensor) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_relu(operators, input.raw, out) })
}

/// Element-wise exact GELU, `x * P(X <= x)`. Not the tanh approximation.
pub fn gelu(input: &Tensor) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_gelu(operators, input.raw, out) })
}

/// Element-wise `x * sigmoid(x)`.
pub fn silu(input: &Tensor) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_silu(operators, input.raw, out) })
}

/// Element-wise sine, in radians.
pub fn sin(input: &Tensor) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_sin(operators, input.raw, out) })
}

/// Element-wise cosine, in radians.
pub fn cos(input: &Tensor) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_cos(operators, input.raw, out) })
}

/// Softmax over the last dimension.
pub fn softmax(input: &Tensor) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_softmax(operators, input.raw, out) })
}

/// Swish-gated linear unit over the last dimension of `input` `<float>(..., D)`, which must be
/// even and which the result halves: `swiglu(x) = swish(x[..D / 2]) * x[D / 2..]`.
pub fn swiglu(input: &Tensor) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_swiglu(operators, input.raw, out) })
}

/// The same gating with a GELU: `geglu(x) = gelu(x[..D / 2]) * x[D / 2..]`. A diffusion U-Net's
/// feed forward is written the other way round, with the gate second, so the exporter swaps the
/// two halves of the projection on the way out.
pub fn geglu(input: &Tensor) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_geglu(operators, input.raw, out) })
}

/// Sum over dimension `dim`, which may be negative to count from the back and which the result
/// drops. Pass [`LAST_DIM`] for the common case.
pub fn sum(input: &Tensor, dim: i32) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_sum(operators, input.raw, dim, out) })
}

/// The largest element of dimension `dim`, which the result drops the same way [`sum`] does.
pub fn max(input: &Tensor, dim: i32) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_max(operators, input.raw, dim, out) })
}

/// Smallest element of dimension `dim`, which the result drops the way [`sum`] does.
pub fn min(input: &Tensor, dim: i32) -> Result<Tensor> {
    let operators = operators_of(input)?;
    Tensor::produce(|out| unsafe { ffi::fl_min(operators, input.raw, dim, out) })
}

/// Concatenate `a` and `b` along `dim`. They must agree on every other dimension.
pub fn cat(a: &Tensor, b: &Tensor, dim: i32) -> Result<Tensor> {
    let operators = operators_of(a)?;
    Tensor::produce(|out| unsafe { ffi::fl_cat(operators, a.raw, b.raw, dim, out) })
}

/// A `<float>(max_len, max_len)` mask holding `-inf` where a position may not attend and `0`
/// where it may.
pub fn causal_mask(max_len: i32, device: Device) -> Result<Tensor> {
    let operators = raw_operators(device)?;
    Tensor::produce(|out| unsafe { ffi::fl_causal_mask(operators, max_len, out) })
}

/// Scaled dot product attention.
///
/// `q` is `<float>(N, nHead, L, D)` and `k` and `v` are `<float>(N, nKvHead, S, D)`, where
/// `nKvHead` may divide `nHead` for grouped-query attention rather than being expanded first.
/// `causal` masks the future positions, aligned to the bottom right of the score matrix.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    let operators = operators_of(q)?;
    Tensor::produce(|out| unsafe {
        ffi::fl_attention(operators, q.raw, k.raw, v.raw, causal as i32, out)
    })
}

/// The keys and values of one forward pass, and where they belong in a paged KV cache.
///
/// Held together because [`paged_attention`] needs all six of them, and passing them positionally
/// makes two tensors of the same shape easy to swap by mistake.
pub struct PagedKvCache<'a> {
    /// `<float>(nBlock, blockSize, nKvHead, D)`: the key block pool.
    pub key_cache: &'a Tensor,
    /// `<float>(nBlock, blockSize, nKvHead, D)`: the value block pool.
    pub value_cache: &'a Tensor,
    /// `<int>(nSeq, maxNumBlock)`: the blocks each sequence owns, in token order.
    pub block_table: &'a Tensor,
    /// `<int>(nSeq + 1)`: the exclusive prefix sum of the query lengths.
    pub cu_seqlens_q: &'a Tensor,
    /// `<int>(nSeq)`: the number of cached tokens each sequence attends to.
    pub seqlens_k: &'a Tensor,
    /// The longest query length in the batch.
    pub max_q_len: i32,
    /// The largest value in `seqlens_k`.
    pub max_k_len: i32,
}

/// Whether this build has [`paged_attention`] at all. It rides on the FlashAttention kernels,
/// which `WITH_FLASH_ATTN=ON` compiles in; without them the call only reports an error, since
/// there is no portable paged attention to fall back to the way [`attention`] does.
pub fn paged_attention_available() -> bool {
    super::init();

    let mut available: i32 = 0;
    match check(unsafe { ffi::fl_paged_attention_available(&mut available) }) {
        Ok(()) => available != 0,
        Err(_) => false,
    }
}

/// Scaled dot product attention over a packed batch of queries reading a paged KV cache.
///
/// `q` is `<float>(totalQLen, nHead, D)`, the queries of every sequence packed back to back.
/// Sequence `i` owns the blocks named by row `i` of `cache.block_table` and attends to the first
/// `cache.seqlens_k[i]` tokens they hold; the tokens it had before this call are that count minus
/// its query length, which is where `causal` starts masking.
pub fn paged_attention(q: &Tensor, cache: &PagedKvCache<'_>, causal: bool) -> Result<Tensor> {
    let operators = operators_of(q)?;
    Tensor::produce(|out| unsafe {
        ffi::fl_paged_attention(
            operators,
            q.raw,
            cache.key_cache.raw,
            cache.value_cache.raw,
            cache.block_table.raw,
            cache.cu_seqlens_q.raw,
            cache.seqlens_k.raw,
            cache.max_q_len,
            cache.max_k_len,
            causal as i32,
            out,
        )
    })
}

/// Scatter the keys and values of a forward pass into a paged KV cache, so that a later
/// [`paged_attention`] reads them back.
///
/// `k` and `v` are `<float>(numTokens, nKvHead, D)`, packed like the queries, and `slot_mapping`
/// is `<int>(numTokens)` holding `blockId * blockSize + offset` for each token.
pub fn store_kv_cache(
    k: &Tensor,
    v: &Tensor,
    key_cache: &mut Tensor,
    value_cache: &mut Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    let operators = operators_of(k)?;
    check(unsafe {
        ffi::fl_store_kv_cache(
            operators,
            k.raw,
            v.raw,
            key_cache.raw,
            value_cache.raw,
            slot_mapping.raw,
        )
    })
}

/// Sample one label per row of `logits` `<float>(rows, vocabSize)` with per-row parameters.
///
/// `temperatures` and `top_ps` are `<float>(rows)` and `top_ks` is `<int>(rows)`. A temperature of
/// zero selects greedily, and a `top_k` of zero or less keeps every label.
pub fn sample_with_params(
    logits: &Tensor,
    temperatures: &Tensor,
    top_ks: &Tensor,
    top_ps: &Tensor,
) -> Result<Tensor> {
    let operators = operators_of(logits)?;
    Tensor::produce(|out| unsafe {
        ffi::fl_sample_with_params(
            operators,
            logits.raw,
            temperatures.raw,
            top_ks.raw,
            top_ps.raw,
            out,
        )
    })
}

/// Divide the logits of the tokens in `history` `<long>(N, historyLen)` by `weight`, penalizing
/// the ones already generated. `logits` `<float>(N, vocabSize)` is written in place.
pub fn repetition_penalty(logits: &mut Tensor, history: &Tensor, weight: f32) -> Result<()> {
    let operators = operators_of(logits)?;
    check(unsafe { ffi::fl_repetition_penalty(operators, logits.raw, history.raw, weight) })
}

/// Copy the elements of `src` into `dest`, which must have the same shape.
pub fn copy(src: &Tensor, dest: &mut Tensor) -> Result<()> {
    let operators = copy_operators(src.try_device()?, dest.try_device()?)?;
    check(unsafe { ffi::fl_copy(operators, src.raw, dest.raw) })
}

/// Fill every element of `tensor` with `value`, in place.
pub fn fill(tensor: &mut Tensor, value: f32) -> Result<()> {
    let operators = operators_of(tensor)?;
    check(unsafe { ffi::fl_fill(operators, tensor.raw, value) })
}

/// Whether every pair of elements is within `rtol` relative and `atol` absolute tolerance. The
/// tolerances match the C++ defaults, which are looser than a bit-for-bit comparison.
pub fn all_close(a: &Tensor, b: &Tensor) -> Result<bool> {
    all_close_with_tolerance(a, b, 1e-3, 1e-5)
}

/// [`all_close`] with the tolerances spelled out.
pub fn all_close_with_tolerance(a: &Tensor, b: &Tensor, rtol: f32, atol: f32) -> Result<bool> {
    let mut value: i32 = 0;
    let operators = operators_of(a)?;
    check(unsafe { ffi::fl_all_close(operators, a.raw, b.raw, rtol, atol, &mut value) })?;
    Ok(value != 0)
}

/// Whether every element of a [`DType::Bool`] tensor is true.
pub fn all(tensor: &Tensor) -> Result<bool> {
    let mut value: i32 = 0;
    let operators = operators_of(tensor)?;
    check(unsafe { ffi::fl_all(operators, tensor.raw, &mut value) })?;
    Ok(value != 0)
}

/// The single element of a one-element tensor, as an `f32`.
pub fn elem(tensor: &Tensor) -> Result<f32> {
    let mut value = 0.0f32;
    let operators = operators_of(tensor)?;
    check(unsafe { ffi::fl_elem(operators, tensor.raw, &mut value) })?;
    Ok(value)
}

/// The float type the operators of `device` work in by default.
pub fn default_float_type(device: Device) -> Result<DType> {
    let operators = raw_operators(device)?;
    let mut raw: i32 = 0;
    check(unsafe { ffi::fl_get_default_float_type(operators, &mut raw) })?;
    DType::from_raw(raw)
}

/// Print the tensor to stdout.
pub fn print(tensor: &Tensor) -> Result<()> {
    let operators = operators_of(tensor)?;
    check(unsafe { ffi::fl_print(operators, tensor.raw) })
}
