//! The operations of `fl::Operators`, one function per method.
//!
//! The C interface takes the operators every call runs on and the view every call writes into,
//! and works out neither. These functions do that working out: an operation runs on the device
//! its inputs live on, so the operators are those of the first tensor it reads, and the shape and
//! type of what it returns follow from what it was handed, as each function says.
//!
//! Every function here takes its inputs by reference and returns a new tensor, except for the few
//! that write into a tensor the caller already has, which take it as `&mut`. That `&mut` is a
//! statement of intent rather than a guarantee: a tensor shares its storage with its clones and
//! its views, so an operation that writes in place can be observed through any of them.
//!
//! An operation runs on the device its inputs live on, and all of them must agree on that device.
//!
//! # Shapes are checked, mostly
//!
//! The shape of every result is worked out here, and what that needs of the inputs is checked
//! here too and reported as an `Err`. What a kernel needs beyond that -- a layout it was not
//! written for, a dtype it has no instance for -- is checked by the C++ library's own fatal check,
//! which prints a message and aborts the process; so does reaching an operation the device has no
//! kernel for. Neither is an unwind the binding could catch and turn into an `Err`, so the shapes
//! and dtypes each function documents are a requirement on the caller rather than something to
//! hand over and let the error path sort out.
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
use super::{check, ffi, DType, Device, Error, Result, Tensor};

/// Reduce over the last dimension, the default of [`sum`] and [`max`].
pub const LAST_DIM: i32 = -1;

/// The view of an optional tensor: NULL where there is none.
fn optional(tensor: Option<&Tensor>) -> ffi::FlTensorView {
    tensor.map(Tensor::raw).unwrap_or(std::ptr::null_mut())
}

/// A fresh tensor of `input`'s shape and dtype, on its device: what every elementwise operation
/// writes.
fn like(input: &Tensor) -> Result<Tensor> {
    Tensor::empty(&input.shape(), input.dtype(), input.device())
}

/// A fresh tensor of `shape` and `dtype` on `input`'s device.
fn beside(input: &Tensor, shape: &[i32], dtype: DType) -> Result<Tensor> {
    Tensor::empty(shape, dtype, input.device())
}

/// `dim` of `input`, which may be negative to count from the back, as a non-negative dimension.
fn real_dim(input: &Tensor, dim: i32, name: &str) -> Result<i32> {
    let rank = input.dim()?;
    if dim < -rank || dim >= rank {
        return Err(Error::invalid(format!("{name}: no dimension {dim} in a {rank}-D tensor")));
    }
    Ok(if dim < 0 { dim + rank } else { dim })
}

/// `input`'s shape with dimension `dim` -- already non-negative -- dropped. A tensor with nothing
/// left is (1), which is what every reduction of a vector has always given.
fn without_dim(input: &Tensor, dim: i32) -> Vec<i32> {
    let mut shape = input.shape();
    shape.remove(dim as usize);
    if shape.is_empty() {
        shape.push(1);
    }
    shape
}

/// `other` seen at `input`'s shape: `other` may have fewer dimensions, and a dimension of one
/// where `input` has more. Only `other` grows; `input`'s shape is the result's.
fn broadcast_to(other: &Tensor, input: &Tensor, name: &str) -> Result<Tensor> {
    let target = input.shape();
    let refuse = || {
        Error::invalid(format!(
            "{name}: {:?} cannot be broadcast to {:?}",
            other.shape(),
            target
        ))
    };
    if other.dim()? > input.dim()? {
        return Err(refuse());
    }

    let mut broadcast = other.clone();
    while broadcast.dim()? < input.dim()? {
        broadcast = broadcast.unsqueeze(0)?;
    }
    for (d, &size) in target.iter().enumerate() {
        let have = broadcast.shape()[d];
        if have != size && have != 1 {
            return Err(refuse());
        }
    }
    if broadcast.shape() == target {
        Ok(broadcast)
    } else {
        broadcast.expand(&target)
    }
}

/// The length a convolution leaves, refused below one.
fn convolved_length(
    length: i32,
    kernel: i32,
    stride: i32,
    padding: i32,
    dilation: i32,
    name: &str,
) -> Result<i32> {
    let out = (length + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
    if out < 1 {
        return Err(Error::invalid(format!("{name}: the output would be {out} long")));
    }
    Ok(out)
}

fn check_convolution(stride: i32, padding: i32, dilation: i32, groups: i32, name: &str) -> Result<()> {
    if groups < 1 {
        return Err(Error::invalid(format!("{name}: groups must be at least 1")));
    }
    if stride < 1 || dilation < 1 {
        return Err(Error::invalid(format!("{name}: stride and dilation must be at least 1")));
    }
    if padding < 0 {
        return Err(Error::invalid(format!("{name}: padding cannot be negative")));
    }
    Ok(())
}

/// A 1-D tensor holding the values of `[begin, end)` taken `step` at a time.
pub fn arange(begin: i64, end: i64, step: i64, device: Device) -> Result<Tensor> {
    if step == 0 {
        return Err(Error::invalid("arange: the step cannot be zero"));
    }
    let numel = (end - begin) / step;
    if numel < 0 || numel >= i32::MAX as i64 {
        return Err(Error::invalid(format!(
            "arange: {numel} elements from {begin} to {end} by {step}"
        )));
    }

    let operators = raw_operators(device)?;
    let out = Tensor::empty(&[numel as i32], DType::Long, device)?;
    check(unsafe { ffi::fl_arange(operators, begin, step, out.raw()) })?;
    Ok(out)
}

/// A tensor filled with uniform random numbers in `[0, 1)`.
pub fn rand(shape: &[i32], dtype: DType, device: Device) -> Result<Tensor> {
    let operators = raw_operators(device)?;
    let out = Tensor::empty(shape, dtype, device)?;
    check(unsafe { ffi::fl_rand(operators, out.raw()) })?;
    Ok(out)
}

/// A float tensor drawn from a normal distribution with mean 0 and variance 1.
pub fn randn(shape: &[i32], device: Device) -> Result<Tensor> {
    let operators = raw_operators(device)?;
    let out = Tensor::empty(shape, DType::Float, device)?;
    check(unsafe { ffi::fl_randn(operators, out.raw()) })?;
    Ok(out)
}

/// Seed the generator that [`rand`] and [`randn`] draw from on `device`.
pub fn manual_seed(device: Device, seed: u64) -> Result<()> {
    let operators = raw_operators(device)?;
    check(unsafe { ffi::fl_manual_seed(operators, seed) })
}

/// The rows of `table` `<float>(V, D)` named by `indices` `<long>(..)`, which gains a trailing
/// dimension of `D`.
pub fn lookup(table: &Tensor, indices: &Tensor) -> Result<Tensor> {
    if table.dim()? != 2 {
        return Err(Error::invalid("lookup: the table must be 2-D"));
    }
    let mut shape = indices.shape();
    shape.push(table.shape_at(1)?);

    let operators = operators_of(table)?;
    let out = beside(table, &shape, table.dtype())?;
    check(unsafe { ffi::fl_lookup(operators, table.raw(), indices.raw(), out.raw()) })?;
    Ok(out)
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
            positions.raw(),
            query.raw(),
            key.raw(),
            rotary_cache.raw(),
        )
    })
}

/// Root mean square layer normalization over the last dimension of `input`, scaled by `weight`
/// `<float>(D)`.
pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let operators = operators_of(input)?;
    let out = like(input)?;
    check(unsafe { ffi::fl_rms_norm(operators, input.raw(), weight.raw(), eps, out.raw()) })?;
    Ok(out)
}

/// Normalize the last dimension of `input` to zero mean and unit variance, then scale by `weight`
/// and shift by `bias`. Unlike an RMS norm this subtracts the mean, which is what CLIP and a
/// diffusion U-Net both do.
pub fn layer_norm(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let operators = operators_of(input)?;
    let out = like(input)?;
    check(unsafe {
        ffi::fl_layer_norm(operators, input.raw(), optional(weight), optional(bias), eps, out.raw())
    })?;
    Ok(out)
}

/// x * sigmoid(1.702 * x), which OpenAI's CLIP uses in place of GELU.
pub fn quick_gelu(input: &Tensor) -> Result<Tensor> {
    unary(input, ffi::fl_quick_gelu)
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
    if input.dim()? != 4 || weight.dim()? != 4 {
        return Err(Error::invalid("conv2d: the input and the weight must be 4-D"));
    }
    check_convolution(stride, padding, dilation, groups, "conv2d")?;
    let (x, w) = (input.shape(), weight.shape());
    let height = convolved_length(x[2], w[2], stride, padding, dilation, "conv2d")?;
    let width = convolved_length(x[3], w[3], stride, padding, dilation, "conv2d")?;

    let operators = operators_of(input)?;
    let out = beside(input, &[x[0], w[0], height, width], input.dtype())?;
    check(unsafe {
        ffi::fl_conv2d(
            operators,
            input.raw(),
            weight.raw(),
            optional(bias),
            stride,
            padding,
            dilation,
            groups,
            out.raw(),
        )
    })?;
    Ok(out)
}

/// A 1-D convolution of `input` `(N, C, L)` by `weight` `(K, C / groups, R)`.
///
/// CUDA only, on CUTLASS; on any other device every call fails. [`crate::audio::conv1d`] is what
/// computes a 1-D convolution everywhere today, out of [`conv2d`]. See `Operators::conv1d`.
pub fn conv1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: i32,
    padding: i32,
    dilation: i32,
    groups: i32,
) -> Result<Tensor> {
    if input.dim()? != 3 || weight.dim()? != 3 {
        return Err(Error::invalid("conv1d: the input and the weight must be 3-D"));
    }
    check_convolution(stride, padding, dilation, groups, "conv1d")?;
    let (x, w) = (input.shape(), weight.shape());
    let length = convolved_length(x[2], w[2], stride, padding, dilation, "conv1d")?;

    let operators = operators_of(input)?;
    let out = beside(input, &[x[0], w[0], length], input.dtype())?;
    check(unsafe {
        ffi::fl_conv1d(
            operators,
            input.raw(),
            weight.raw(),
            optional(bias),
            stride,
            padding,
            dilation,
            groups,
            out.raw(),
        )
    })?;
    Ok(out)
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
    if input.dim()? != 3 || weight.dim()? != 3 {
        return Err(Error::invalid("convTranspose1d: the input and the weight must be 3-D"));
    }
    check_convolution(stride, padding, 1, groups, "convTranspose1d")?;
    let (x, w) = (input.shape(), weight.shape());
    let length = (x[2] - 1) * stride - 2 * padding + w[2] + output_padding;
    if length < 1 {
        return Err(Error::invalid(format!(
            "convTranspose1d: the output would be {length} long"
        )));
    }

    let operators = operators_of(input)?;
    let out = beside(input, &[x[0], w[1] * groups, length], input.dtype())?;
    check(unsafe {
        ffi::fl_conv_transpose1d(
            operators,
            input.raw(),
            weight.raw(),
            optional(bias),
            stride,
            padding,
            output_padding,
            groups,
            out.raw(),
        )
    })?;
    Ok(out)
}

/// `x + sin(alpha * x)^2 / (beta + eps)`, per channel of `input` `(N, C, L)`.
///
/// No device implements this yet; [`crate::audio::snake`] is what computes one today.
pub fn snake(input: &Tensor, alpha: &Tensor, beta: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    let operators = operators_of(input)?;
    let out = like(input)?;
    check(unsafe {
        ffi::fl_snake(operators, input.raw(), alpha.raw(), optional(beta), eps, out.raw())
    })?;
    Ok(out)
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
    if input.dim()? != 3 {
        return Err(Error::invalid("stft: the input must be (N, 1, L)"));
    }
    if n_fft < 1 || hop < 1 {
        return Err(Error::invalid("stft: nFft and hop must be at least 1"));
    }
    let length = input.shape_at(2)? + if centered { 2 * (n_fft / 2) } else { 0 };
    if length < n_fft {
        return Err(Error::invalid("stft: the input is shorter than one window"));
    }
    let frames = (length - n_fft) / hop + 1;

    let operators = operators_of(input)?;
    let shape = [input.shape_at(0)?, 2 * (n_fft / 2 + 1), frames];
    let out = beside(input, &shape, input.dtype())?;
    check(unsafe {
        ffi::fl_stft(
            operators,
            input.raw(),
            window.raw(),
            n_fft,
            hop,
            centered as i32,
            out.raw(),
        )
    })?;
    Ok(out)
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
    if spectrum.dim()? != 3 {
        return Err(Error::invalid("istft: the spectrum must be (N, bins, frames)"));
    }
    if n_fft < 1 || hop < 1 {
        return Err(Error::invalid("istft: nFft and hop must be at least 1"));
    }
    let length =
        n_fft + hop * (spectrum.shape_at(2)? - 1) - if centered { 2 * (n_fft / 2) } else { 0 };
    if length < 1 {
        return Err(Error::invalid("istft: the spectrum is too short to invert"));
    }

    let operators = operators_of(spectrum)?;
    let out = beside(spectrum, &[spectrum.shape_at(0)?, 1, length], spectrum.dtype())?;
    check(unsafe {
        ffi::fl_istft(
            operators,
            spectrum.raw(),
            window.raw(),
            n_fft,
            hop,
            centered as i32,
            out.raw(),
        )
    })?;
    Ok(out)
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
    let operators = operators_of(input)?;
    let out = like(input)?;
    check(unsafe {
        ffi::fl_group_norm(
            operators,
            input.raw(),
            optional(weight),
            optional(bias),
            groups,
            eps,
            out.raw(),
        )
    })?;
    Ok(out)
}

/// Repeat each pixel of `input` `(N, C, H, W)` `scale` times along both spatial axes.
pub fn upsample_nearest2d(input: &Tensor, scale: i32) -> Result<Tensor> {
    if input.dim()? != 4 {
        return Err(Error::invalid("upsampleNearest2d: the input is not 4-D"));
    }
    if scale < 1 {
        return Err(Error::invalid("upsampleNearest2d: the scale must be at least 1"));
    }
    let x = input.shape();

    let operators = operators_of(input)?;
    let out = beside(input, &[x[0], x[1], x[2] * scale, x[3] * scale], input.dtype())?;
    check(unsafe { ffi::fl_upsample_nearest2d(operators, input.raw(), scale, out.raw()) })?;
    Ok(out)
}

/// Resize the last dimension of `input` to `size` as `F.interpolate(size=size, mode="nearest")`
/// does: output `j` copies input `min(floor(j * scale), length - 1)`, with `scale` the float32
/// ratio `length / size` and the product taken in float32 -- torch's index, frame for frame.
pub fn upsample_nearest1d(input: &Tensor, size: i32) -> Result<Tensor> {
    if input.dim()? < 1 {
        return Err(Error::invalid("upsampleNearest1d: the input has no dimensions"));
    }
    if size < 1 {
        return Err(Error::invalid("upsampleNearest1d: the size must be at least 1"));
    }
    let mut shape = input.shape();
    *shape.last_mut().expect("at least one dimension") = size;

    let operators = operators_of(input)?;
    let out = beside(input, &shape, input.dtype())?;
    check(unsafe { ffi::fl_upsample_nearest1d(operators, input.raw(), out.raw()) })?;
    Ok(out)
}

/// Matrix multiplication, batched over the leading dimensions. `b` may have fewer dimensions than
/// `a`, and is then seen at `a`'s batch.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (sa, sb) = (a.shape(), b.shape());
    let refuse = || Error::invalid(format!("matmul: {sa:?} by {sb:?}"));
    if sa.len() < 2 || sb.len() < 2 {
        return Err(Error::invalid("matmul: both operands need 2 dimensions"));
    }
    if sa[sa.len() - 1] != sb[sb.len() - 2] {
        return Err(refuse());
    }
    if sb.len() > 2 {
        if sa.len() < sb.len() {
            return Err(refuse());
        }
        let offset = sa.len() - sb.len();
        for d in 0..sb.len() - 2 {
            if sb[d] != sa[d + offset] {
                return Err(Error::invalid(format!(
                    "matmul: the batches of {sa:?} and {sb:?} differ"
                )));
            }
        }
    }

    let mut shape = sa.clone();
    *shape.last_mut().expect("two dimensions") = sb[sb.len() - 1];
    let operators = operators_of(a)?;
    let out = beside(a, &shape, a.dtype())?;
    check(unsafe { ffi::fl_matmul(operators, a.raw(), b.raw(), out.raw()) })?;
    Ok(out)
}

/// What an FP8 product of `a` `(..., k)` by a weight of `(rows, k)` writes: `<float16>(..., rows)`
/// on the weight's device.
fn fp8_result(a: &Tensor, weight: &Tensor) -> Result<Tensor> {
    if a.dim()? < 1 || weight.dim()? < 1 {
        return Err(Error::invalid("fp8 matmul: an operand has no dimensions"));
    }
    let mut shape = a.shape();
    *shape.last_mut().expect("at least one dimension") = weight.shape_at(0)?;
    Tensor::empty(&shape, DType::Float16, weight.device())
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
/// The activation is not narrowed: the multiply happens at full width and only the weight is
/// narrow. So this is `<float16>` in and out, with `rows` a
/// multiple of 8 and `k` a multiple of 16. CUDA is the only device with the kernels for it.
pub fn fp8_matmul(a: &Tensor, weight: &Tensor, channel_scale: &Tensor) -> Result<Tensor> {
    let out = fp8_result(a, weight)?;
    check(unsafe { ffi::fl_fp8_matmul(a.raw(), weight.raw(), channel_scale.raw(), out.raw()) })?;
    Ok(out)
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
    let out = fp8_result(a, weight)?;
    check(unsafe {
        ffi::fl_fp8_matmul_tensor_scale(a.raw(), weight.raw(), scale.raw(), out.raw())
    })?;
    Ok(out)
}

type BinaryFn =
    unsafe extern "C" fn(ffi::FlOperators, ffi::FlTensorView, ffi::FlTensorView, ffi::FlTensorView) -> i32;

/// An elementwise operation of `a` and `b`, `b` broadcast to `a`'s shape, into a tensor like `a`.
fn binary(a: &Tensor, b: &Tensor, name: &str, call: BinaryFn) -> Result<Tensor> {
    if a.dtype() != b.dtype() {
        return Err(Error::invalid(format!("{name}: the two operands differ in type")));
    }
    let b = broadcast_to(b, a, name)?;
    let operators = operators_of(a)?;
    let out = like(a)?;
    check(unsafe { call(operators, a.raw(), b.raw(), out.raw()) })?;
    Ok(out)
}

/// Element-wise `a * b`, broadcasting `b` over the leading dimensions of `a`.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary(a, b, "mul", ffi::fl_mul)
}

/// Element-wise `a / b`, broadcasting `b` over the leading dimensions of `a`.
pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary(a, b, "divTensor", ffi::fl_div)
}

/// Element-wise `a + b`, broadcasting `b` over the leading dimensions of `a`.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary(a, b, "add", ffi::fl_add)
}

/// Element-wise `a - b`, broadcasting `b` over the leading dimensions of `a`.
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary(a, b, "sub", ffi::fl_sub)
}

/// Element-wise `a == b`, as a [`DType::Bool`] tensor of the same shape.
pub fn eq(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.dtype() != b.dtype() {
        return Err(Error::invalid("eq: the two operands differ in type"));
    }
    if a.shape() != b.shape() {
        return Err(Error::invalid(format!(
            "eq: {:?} and {:?} differ in shape",
            a.shape(),
            b.shape()
        )));
    }
    let operators = operators_of(a)?;
    let out = beside(a, &a.shape(), DType::Bool)?;
    check(unsafe { ffi::fl_eq(operators, a.raw(), b.raw(), out.raw()) })?;
    Ok(out)
}

/// Element-wise `input * other`.
pub fn mul_scalar(input: &Tensor, other: f32) -> Result<Tensor> {
    let operators = operators_of(input)?;
    let out = like(input)?;
    check(unsafe { ffi::fl_mul_scalar(operators, input.raw(), other, out.raw()) })?;
    Ok(out)
}

/// Element-wise `input / other`.
pub fn div_scalar(input: &Tensor, other: f32) -> Result<Tensor> {
    let operators = operators_of(input)?;
    let out = like(input)?;
    check(unsafe { ffi::fl_div_scalar(operators, input.raw(), other, out.raw()) })?;
    Ok(out)
}

/// Element-wise `input % other`, for a [`DType::Long`] tensor.
pub fn mod_scalar(input: &Tensor, other: i64) -> Result<Tensor> {
    if other == 0 {
        return Err(Error::invalid("mod: by zero"));
    }
    let operators = operators_of(input)?;
    let out = like(input)?;
    check(unsafe { ffi::fl_mod_scalar(operators, input.raw(), other, out.raw()) })?;
    Ok(out)
}

type UnaryFn = unsafe extern "C" fn(ffi::FlOperators, ffi::FlTensorView, ffi::FlTensorView) -> i32;

/// An elementwise operation of `input` into a tensor like it.
fn unary(input: &Tensor, call: UnaryFn) -> Result<Tensor> {
    let operators = operators_of(input)?;
    let out = like(input)?;
    check(unsafe { call(operators, input.raw(), out.raw()) })?;
    Ok(out)
}

/// Element-wise `input` squared.
pub fn square(input: &Tensor) -> Result<Tensor> {
    unary(input, ffi::fl_square)
}

/// Element-wise `-x`.
pub fn neg(input: &Tensor) -> Result<Tensor> {
    unary(input, ffi::fl_neg)
}

/// Element-wise `|x|`.
pub fn abs(input: &Tensor) -> Result<Tensor> {
    unary(input, ffi::fl_abs)
}

/// Element-wise `e^x`.
pub fn exp(input: &Tensor) -> Result<Tensor> {
    unary(input, ffi::fl_exp)
}

/// Element-wise natural logarithm. Zero gives minus infinity and a negative number NaN.
pub fn log(input: &Tensor) -> Result<Tensor> {
    unary(input, ffi::fl_log)
}

/// Element-wise to the nearest integer, a tie to the even one, as `torch.round`. Still a float.
pub fn round(input: &Tensor) -> Result<Tensor> {
    unary(input, ffi::fl_round)
}

/// Element-wise square root.
pub fn sqrt(input: &Tensor) -> Result<Tensor> {
    unary(input, ffi::fl_sqrt)
}

/// Element-wise reciprocal square root, `1/sqrt(x)`.
pub fn rsqrt(input: &Tensor) -> Result<Tensor> {
    unary(input, ffi::fl_rsqrt)
}

/// Element-wise `1/(1 + e^-x)`.
pub fn sigmoid(input: &Tensor) -> Result<Tensor> {
    unary(input, ffi::fl_sigmoid)
}

/// Element-wise hyperbolic tangent.
pub fn tanh(input: &Tensor) -> Result<Tensor> {
    unary(input, ffi::fl_tanh)
}

/// Element-wise `max(x, 0)`.
pub fn relu(input: &Tensor) -> Result<Tensor> {
    unary(input, ffi::fl_relu)
}

/// Element-wise exact GELU, `x * P(X <= x)`. Not the tanh approximation.
pub fn gelu(input: &Tensor) -> Result<Tensor> {
    unary(input, ffi::fl_gelu)
}

/// Element-wise `x * sigmoid(x)`.
pub fn silu(input: &Tensor) -> Result<Tensor> {
    unary(input, ffi::fl_silu)
}

/// Element-wise sine, in radians.
pub fn sin(input: &Tensor) -> Result<Tensor> {
    unary(input, ffi::fl_sin)
}

/// Element-wise cosine, in radians.
pub fn cos(input: &Tensor) -> Result<Tensor> {
    unary(input, ffi::fl_cos)
}

/// Softmax over the last dimension.
pub fn softmax(input: &Tensor) -> Result<Tensor> {
    unary(input, ffi::fl_softmax)
}

/// The half a gated linear unit leaves: the last dimension halved.
fn gated(input: &Tensor, name: &str, call: UnaryFn) -> Result<Tensor> {
    let mut shape = input.shape();
    match shape.last_mut() {
        Some(last) if *last % 2 == 0 => *last /= 2,
        _ => {
            return Err(Error::invalid(format!(
                "{name}: the last dimension of {:?} is not even",
                input.shape()
            )))
        }
    }
    let operators = operators_of(input)?;
    let out = beside(input, &shape, input.dtype())?;
    check(unsafe { call(operators, input.raw(), out.raw()) })?;
    Ok(out)
}

/// Swish-gated linear unit over the last dimension of `input` `<float>(..., D)`, which must be
/// even and which the result halves: `swiglu(x) = swish(x[..D / 2]) * x[D / 2..]`.
pub fn swiglu(input: &Tensor) -> Result<Tensor> {
    gated(input, "swiglu", ffi::fl_swiglu)
}

/// The same gating with a GELU: `geglu(x) = gelu(x[..D / 2]) * x[D / 2..]`. A diffusion U-Net's
/// feed forward is written the other way round, with the gate second, so the exporter swaps the
/// two halves of the projection on the way out.
pub fn geglu(input: &Tensor) -> Result<Tensor> {
    gated(input, "geglu", ffi::fl_geglu)
}

/// Sum over dimension `dim`, which may be negative to count from the back and which the result
/// drops. Pass [`LAST_DIM`] for the common case.
pub fn sum(input: &Tensor, dim: i32) -> Result<Tensor> {
    let dim = real_dim(input, dim, "sum")?;
    let operators = operators_of(input)?;
    let out = beside(input, &without_dim(input, dim), input.dtype())?;
    check(unsafe { ffi::fl_sum(operators, input.raw(), dim, out.raw()) })?;
    Ok(out)
}

/// The inclusive prefix sum along `dim`, which may be negative to count from the back: element `i`
/// of the result is the sum of elements `0..=i`. Same shape and dtype as `input`; a float16 input
/// is accumulated in float32.
pub fn cumsum(input: &Tensor, dim: i32) -> Result<Tensor> {
    let dim = real_dim(input, dim, "fl_cumsum")?;
    let operators = operators_of(input)?;
    let out = like(input)?;
    check(unsafe { ffi::fl_cumsum(operators, input.raw(), dim, out.raw()) })?;
    Ok(out)
}

/// The largest or smallest element of the last dimension, which the result drops. `dim` names
/// that dimension, and any other is refused rather than quietly handed the last.
fn reduce_last(input: &Tensor, dim: i32, name: &str, call: UnaryFn) -> Result<Tensor> {
    let last = input.dim()? - 1;
    if dim != -1 && dim != last {
        return Err(Error::invalid(format!(
            "{name} reduces the last dimension, not dimension {dim}"
        )));
    }
    let last = real_dim(input, -1, name)?;
    let operators = operators_of(input)?;
    let out = beside(input, &without_dim(input, last), input.dtype())?;
    check(unsafe { call(operators, input.raw(), out.raw()) })?;
    Ok(out)
}

/// The largest element of dimension `dim`, which the result drops the same way [`sum`] does.
pub fn max(input: &Tensor, dim: i32) -> Result<Tensor> {
    reduce_last(input, dim, "fl_max", ffi::fl_max)
}

/// Smallest element of dimension `dim`, which the result drops the way [`sum`] does.
pub fn min(input: &Tensor, dim: i32) -> Result<Tensor> {
    reduce_last(input, dim, "fl_min", ffi::fl_min)
}

/// Concatenate `a` and `b` along `dim`. They must agree on every other dimension.
pub fn cat(a: &Tensor, b: &Tensor, dim: i32) -> Result<Tensor> {
    if a.dtype() != b.dtype() {
        return Err(Error::invalid("cat: the two halves differ in type"));
    }
    if a.dim()? != b.dim()? {
        return Err(Error::invalid("cat: the two halves differ in rank"));
    }
    let dim = real_dim(a, dim, "cat")?;
    let (sa, sb) = (a.shape(), b.shape());
    for d in 0..sa.len() {
        if d != dim as usize && sa[d] != sb[d] {
            return Err(Error::invalid(format!(
                "cat: {sa:?} and {sb:?} differ in more than dimension {dim}"
            )));
        }
    }

    let (da, db) = (sa[dim as usize], sb[dim as usize]);
    let mut shape = sa.clone();
    shape[dim as usize] = da + db;
    let out = beside(a, &shape, a.dtype())?;
    copy(a, &mut out.slice(dim, 0, da)?)?;
    copy(b, &mut out.slice(dim, da, da + db)?)?;
    Ok(out)
}

/// A `(max_len, max_len)` mask holding `-inf` where a position may not attend and `0` where it
/// may, in the device's default float type.
pub fn causal_mask(max_len: i32, device: Device) -> Result<Tensor> {
    if max_len < 1 {
        return Err(Error::invalid(format!("causalMask: a mask of {max_len} positions")));
    }
    let operators = raw_operators(device)?;
    let out = Tensor::empty(&[max_len, max_len], default_float_type(device)?, device)?;
    check(unsafe { ffi::fl_causal_mask(operators, out.raw()) })?;
    Ok(out)
}

/// Scaled dot product attention.
///
/// `q` is `<float>(N, nHead, L, D)` and `k` and `v` are `<float>(N, nKvHead, S, D)`, where
/// `nKvHead` may divide `nHead` for grouped-query attention rather than being expanded first.
/// `causal` masks the future positions, aligned to the bottom right of the score matrix.
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    if q.dim()? != 4 || k.dim()? != 4 || v.dim()? != 4 {
        return Err(Error::invalid("attention: q, k and v must be 4-D"));
    }
    let (sq, sk, sv) = (q.shape(), k.shape(), v.shape());
    if sk[1] == 0 || sq[1] % sk[1] != 0 {
        return Err(Error::invalid(
            "attention: the query heads are not a multiple of the key-value heads",
        ));
    }

    let operators = operators_of(q)?;
    let out = beside(q, &[sq[0], sq[1], sq[2], sv[3]], q.dtype())?;
    check(unsafe {
        ffi::fl_attention(operators, q.raw(), k.raw(), v.raw(), causal as i32, out.raw())
    })?;
    Ok(out)
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
    if q.dim()? != 3 {
        return Err(Error::invalid("pagedAttention: q must be (tokens, heads, headDim)"));
    }
    let operators = operators_of(q)?;
    let out = like(q)?;
    check(unsafe {
        ffi::fl_paged_attention(
            operators,
            q.raw(),
            cache.key_cache.raw(),
            cache.value_cache.raw(),
            cache.block_table.raw(),
            cache.cu_seqlens_q.raw(),
            cache.seqlens_k.raw(),
            cache.max_q_len,
            cache.max_k_len,
            causal as i32,
            out.raw(),
        )
    })?;
    Ok(out)
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
            k.raw(),
            v.raw(),
            key_cache.raw(),
            value_cache.raw(),
            slot_mapping.raw(),
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
    if logits.dim()? != 2 {
        return Err(Error::invalid("sample: the logits must be (rows, vocabulary)"));
    }
    let operators = operators_of(logits)?;
    let out = beside(logits, &[logits.shape_at(0)?], DType::Long)?;
    check(unsafe {
        ffi::fl_sample_with_params(
            operators,
            logits.raw(),
            temperatures.raw(),
            top_ks.raw(),
            top_ps.raw(),
            out.raw(),
        )
    })?;
    Ok(out)
}

/// Divide the logits of the tokens in `history` `<long>(N, historyLen)` by `weight`, penalizing
/// the ones already generated. `logits` `<float>(N, vocabSize)` is written in place.
pub fn repetition_penalty(logits: &mut Tensor, history: &Tensor, weight: f32) -> Result<()> {
    let operators = operators_of(logits)?;
    check(unsafe { ffi::fl_repetition_penalty(operators, logits.raw(), history.raw(), weight) })
}

/// Copy the elements of `src` into `dest`, which must have the same shape.
pub fn copy(src: &Tensor, dest: &mut Tensor) -> Result<()> {
    let operators = copy_operators(src.try_device()?, dest.try_device()?)?;
    check(unsafe { ffi::fl_copy(operators, src.raw(), dest.raw()) })
}

/// Fill every element of `tensor` with `value`, in place.
pub fn fill(tensor: &mut Tensor, value: f32) -> Result<()> {
    let operators = operators_of(tensor)?;
    check(unsafe { ffi::fl_fill(operators, tensor.raw(), value) })
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
    check(unsafe { ffi::fl_all_close(operators, a.raw(), b.raw(), rtol, atol, &mut value) })?;
    Ok(value != 0)
}

/// Whether every element of a [`DType::Bool`] tensor is true.
pub fn all(tensor: &Tensor) -> Result<bool> {
    let mut value: i32 = 0;
    let operators = operators_of(tensor)?;
    check(unsafe { ffi::fl_all(operators, tensor.raw(), &mut value) })?;
    Ok(value != 0)
}

/// The single element of a one-element tensor, as an `f32`.
pub fn elem(tensor: &Tensor) -> Result<f32> {
    let mut value = 0.0f32;
    let operators = operators_of(tensor)?;
    check(unsafe { ffi::fl_elem(operators, tensor.raw(), &mut value) })?;
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
    check(unsafe { ffi::fl_print(operators, tensor.raw()) })
}
