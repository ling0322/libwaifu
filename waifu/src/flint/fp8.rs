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

//! A weight held in E4M3 with one scale per output channel.

use super::{check, ffi, init, Device, Result, Tensor};

/// What a package calls the scale beside a weight it stored quantized: the elements are
/// `"…weight"` and the scale is `"…weight.scale"`.
///
/// The pairing is a naming convention because safetensors has nowhere else to put it -- a tensor's
/// header holds a dtype, a shape and two offsets, and the only free text in the format is one
/// string map for the file as a whole. So the convention is the format, and reading a package
/// enforces it -- [`read_safetensors`](crate::read_safetensors) and
/// [`ParamSource::from_files`](super::ParamSource::from_files) alike: an `<fp8e4m3>` tensor with no
/// scale beside it is refused there rather than found out by a kernel much later.
pub const CHANNEL_SCALE_SUFFIX: &str = ".scale";

/// A tensor quantized to E4M3: one byte per element, and one `float` scale per row, held together
/// because quantizing produces both at once and a row means what it means only against its own
/// scale.
///
/// This is what [`Fp8Tensor::quantize`] hands back, and not what an FP8 weight *is*. A package
/// that stored one quantized holds two ordinary tensors under two ordinary names -- `"…weight"`
/// and `"…weight.scale"` -- which a graph loads as two ordinary weights and hands to
/// [`functional::fp8_matmul`](super::functional::fp8_matmul) as two arguments. Nothing below the
/// C interface has ever taken a pair as one thing either: `fl_fp8_matmul` takes two handles and
/// puts them together on the other side. So this type is a convenience for the one path that
/// makes both pieces in the same breath, and [`Fp8Tensor::data`] and
/// [`Fp8Tensor::channel_scale`] are how they get back out.
///
/// Where [`super::Nvfp4Tensor`] narrows both operands and multiplies on the block scaled tensor
/// cores, this narrows the weight alone: the multiply is the ordinary one, and the weight is
/// widened on its way into it. So it buys bandwidth rather than arithmetic -- about twice the
/// speed of the half GEMM where a weight is read once and multiplied by few rows, and a little
/// slower than it where the rows are many. It costs about 2.6e-2 of relative RMSE. See
/// `docs/fp8.md`.
///
/// CUDA is the only device with the kernels. The format is not the card's -- the bytes would mean
/// the same on a processor, and a package that stores a weight quantized says nothing about where
/// it will be multiplied -- but what reads them is, for now.
#[derive(Debug)]
pub struct Fp8Tensor {
    data: Tensor,
    channel_scale: Tensor,
}

impl Fp8Tensor {
    /// Whether `device` can quantize and multiply in FP8, which today is CUDA and nothing else.
    /// The kernel is written against the sm_80 tensor cores, so unlike NVFP4 it needs no more
    /// than Ampere -- but a device with no FP8 kernels at all answers no here rather than failing
    /// further in.
    pub fn is_available(device: Device) -> bool {
        init();

        let mut available: i32 = 0;
        match check(unsafe { ffi::fl_fp8_available(device as ffi::FlDeviceType, &mut available) }) {
            Ok(()) => available != 0,
            Err(_) => false,
        }
    }

    /// Quantize a contiguous two dimensional `<float16>` CUDA tensor whose `k` 16 divides.
    pub fn quantize(x: &Tensor) -> Result<Fp8Tensor> {
        let mut data: ffi::FlTensor = std::ptr::null_mut();
        let mut channel_scale: ffi::FlTensor = std::ptr::null_mut();

        check(unsafe { ffi::fl_fp8_quantize(x.raw, &mut data, &mut channel_scale) })?;

        // The C side hands over both handles or neither, so there is no half-owned state here to
        // unwind.
        Ok(unsafe {
            Fp8Tensor {
                data: Tensor::from_raw(data),
                channel_scale: Tensor::from_raw(channel_scale),
            }
        })
    }

    /// The quantized elements, `<fp8e4m3>(rows, k)`.
    pub fn data(&self) -> &Tensor {
        &self.data
    }

    /// The scale each row means itself against, `<float>(rows)`.
    pub fn channel_scale(&self) -> &Tensor {
        &self.channel_scale
    }

    /// The `(rows, k)` this was quantized from.
    pub fn shape(&self) -> Result<(i32, i32)> {
        Ok((self.data.shape_at(0)?, self.data.shape_at(1)?))
    }

    /// Back to `<float16>(rows, k)`, carrying the quantization error with it. Mostly useful for
    /// seeing how much of that error there is.
    pub fn dequantize(&self) -> Result<Tensor> {
        Tensor::produce(|out| unsafe {
            ffi::fl_fp8_dequantize(self.data.raw, self.channel_scale.raw, out)
        })
    }
}
