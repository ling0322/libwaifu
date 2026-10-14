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

//! A weight held in E4M3 with one scale for the whole of it.

use super::{check, ffi, init, Device};

/// What a package calls the scale beside a weight it stored quantized: the elements are
/// `"…weight"` and the scale is `"…weight.scale"`.
///
/// The pairing is a naming convention because safetensors has nowhere else to put it -- a tensor's
/// header holds a dtype, a shape and two offsets, and the only free text in the format is one
/// string map for the file as a whole. So the convention is the format, and reading a package
/// enforces it -- [`read_safetensors`](crate::read_safetensors) and
/// [`Weights::from_files`](super::Weights::from_files) alike: an `<fp8e4m3>` tensor with no
/// scale beside it is refused there rather than found out by a kernel much later.
pub const SCALE_SUFFIX: &str = ".scale";

/// Whether `device` can multiply by an FP8 weight -- see
/// [`functional::fp8_matmul`](super::functional::fp8_matmul) -- which today is CUDA on sm_80 or
/// newer, and Metal. A device with no FP8 kernels answers no here rather than failing further in.
///
/// The narrow weight buys bandwidth rather than arithmetic: the multiply is the ordinary one, and
/// the weight is widened on its way into it. See `docs/fp8.md`.
pub fn fp8_available(device: Device) -> bool {
    init();

    let mut available: i32 = 0;
    match check(unsafe { ffi::fl_fp8_available(device as ffi::FlDeviceType, &mut available) }) {
        Ok(()) => available != 0,
        Err(_) => false,
    }
}
