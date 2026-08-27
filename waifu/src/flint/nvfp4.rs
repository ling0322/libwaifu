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

//! A tensor held in NVFP4, for the block scaled tensor cores.

use super::{check, ffi, init, Result, Tensor};

/// A tensor quantized to NVFP4: E2M1 elements, an E4M3 scale for every 16 of them, and one scale
/// for the tensor as a whole. The three pieces travel together because the multiply needs all of
/// them, and the scales are interleaved rather than laid out row by row, so they are only useful
/// to the kernels that expect that arrangement.
///
/// The quantization error is real: on normally distributed data a single quantized operand costs
/// about 9.5e-2 of relative RMSE, and two of them about 1.3e-1. See `docs/nvfp4.md`.
#[derive(Debug)]
pub struct Nvfp4Tensor {
    pub(super) data: Tensor,
    pub(super) block_scale: Tensor,
    pub(super) global_scale: Tensor,
}

impl Nvfp4Tensor {
    /// Whether this build and this GPU can quantize and multiply in NVFP4. The tensor core
    /// instruction is specific to sm_120a, and the build needs CUTLASS, so this is false far more
    /// often than a CUDA device is absent.
    pub fn is_available() -> bool {
        init();

        let mut available: i32 = 0;
        match check(unsafe { ffi::fl_nvfp4_available(&mut available) }) {
            Ok(()) => available != 0,
            Err(_) => false,
        }
    }

    /// Quantize `x`, a contiguous `<float16>(rows, k)` CUDA tensor whose `k` 32 divides.
    pub fn quantize(x: &Tensor) -> Result<Nvfp4Tensor> {
        let mut data: ffi::FlTensor = std::ptr::null_mut();
        let mut block_scale: ffi::FlTensor = std::ptr::null_mut();
        let mut global_scale: ffi::FlTensor = std::ptr::null_mut();

        check(unsafe {
            ffi::fl_nvfp4_quantize(x.raw, &mut data, &mut block_scale, &mut global_scale)
        })?;

        // The C side hands over all three handles or none, so there is no half-owned state here
        // to unwind.
        Ok(unsafe {
            Nvfp4Tensor {
                data: Tensor::from_raw(data),
                block_scale: Tensor::from_raw(block_scale),
                global_scale: Tensor::from_raw(global_scale),
            }
        })
    }

    /// The `(rows, k)` this was quantized from.
    pub fn shape(&self) -> Result<(i32, i32)> {
        Ok((self.data.shape_at(0)?, self.data.shape_at(1)? * 2))
    }

    /// Back to `<float16>(rows, k)`, carrying the quantization error with it. Mostly useful for
    /// seeing how much of that error there is.
    pub fn dequantize(&self) -> Result<Tensor> {
        Tensor::produce(|out| unsafe {
            ffi::fl_nvfp4_dequantize(
                self.data.raw,
                self.block_scale.raw,
                self.global_scale.raw,
                out,
            )
        })
    }
}
