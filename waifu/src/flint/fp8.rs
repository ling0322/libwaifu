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

/// A tensor quantized to E4M3: one byte per element, and one `float` scale per row. The two
/// pieces travel together because the multiply needs both, and a row means what it means only
/// against its own scale.
///
/// Where [`super::Nvfp4Tensor`] narrows both operands and multiplies on the block scaled tensor
/// cores, this narrows the weight alone: the multiply is the ordinary one, and the weight is
/// widened on its way into it. So it buys bandwidth rather than arithmetic -- on CUDA about twice
/// the speed of the half GEMM where a weight is read once and multiplied by few rows, and a little
/// slower than it where the rows are many. It costs about 2.6e-2 of relative RMSE. See
/// `docs/fp8.md`.
///
/// Both devices have it. On the CPU there is no FP8 arithmetic to reach for at all, so it buys no
/// speed there and the weight taking one byte an element rather than two is the whole of it. The
/// two quantizers agree byte for byte, so a weight means the same thing on either.
#[derive(Debug)]
pub struct Fp8Tensor {
    pub(super) data: Tensor,
    pub(super) channel_scale: Tensor,
}

impl Fp8Tensor {
    /// Whether `device` can quantize and multiply in FP8. [`Device::Cpu`] always can; the CUDA
    /// kernel is written against the sm_80 tensor cores, so unlike NVFP4 it needs no more than
    /// Ampere.
    pub fn is_available(device: Device) -> bool {
        init();

        let mut available: i32 = 0;
        match check(unsafe { ffi::fl_fp8_available(device as ffi::FlDeviceType, &mut available) }) {
            Ok(()) => available != 0,
            Err(_) => false,
        }
    }

    /// Quantize a contiguous two dimensional `x`, in the float type its device computes in:
    /// `<float16>` on CUDA, where `k` also has to be a multiple of 16, and either `<float>` or
    /// `<float16>` on the CPU, where nothing has to divide.
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

    /// The `(rows, k)` this was quantized from.
    pub fn shape(&self) -> Result<(i32, i32)> {
        Ok((self.data.shape_at(0)?, self.data.shape_at(1)?))
    }

    /// Back to `(rows, k)` in its device's float type -- `<float16>` on CUDA, `<float>` on the
    /// CPU -- carrying the quantization error with it. Mostly useful for seeing how much of that
    /// error there is.
    pub fn dequantize(&self) -> Result<Tensor> {
        Tensor::produce(|out| unsafe {
            ffi::fl_fp8_dequantize(self.data.raw, self.channel_scale.raw, out)
        })
    }
}
