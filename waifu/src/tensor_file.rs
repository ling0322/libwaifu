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

//! Reading tensors out of safetensors files.
//!
//! A model's weights are safetensors files, named by the whole name --
//! `"model.layer0.attn.weight"` and not `"weight"` -- and a model too large for one file is
//! written as several, which a [`Manifest`](crate::Manifest) names in order. They are read into
//! one namespace, so which file a tensor was written to is not something the model has to know.
//!
//! # One format
//!
//! There used to be two, both of them this project's own: a `tdicv2` stream holding every tensor
//! of a model in one archive entry, and a later layout of one entry per tensor. Both are gone, and
//! so is the zip they lived in. The weights of a model are safetensors files now -- the format
//! every other diffusion runtime already reads and writes -- and this file is the whole of reading
//! them.
//!
//! What that buys is not in here. It is that a weight can be looked at with anything: `torch`,
//! `numpy`, a hex editor, the `safetensors` viewer on the hub. A format of one's own has to earn
//! that cost, and two formats of one's own never did.
//!
//! # What reading costs
//!
//! One file at a time. A file is read whole, each of its tensors is handed on as it is built, and
//! the file's bytes are let go of before the next file is read. So what a caller keeps is the
//! caller's -- a model puts each weight on the card, or page-locks it, as it arrives -- and the
//! reading itself costs one file on top of that, not the whole package.
//!
//! There is no object in between that holds a model's files. A model is read once, into the
//! [`ParamSource`](crate::flint::ParamSource) it runs from, and nothing afterwards reads the disk.

use std::collections::HashMap;
use std::path::Path;

use safetensors::tensor::{Dtype, SafeTensors};

use crate::error::{Error, Result};
use crate::flint::{DType, Tensor, CHANNEL_SCALE_SUFFIX};

/// The largest rank and dimension a stored tensor may have, which keep a corrupt file from asking
/// for an unreasonable allocation.
const MAX_RANK: usize = 16;
const MAX_DIM: usize = 1048576;

/// How many bytes a safetensors file spends saying how long its header is.
const HEADER_LENGTH_BYTES: usize = 8;

/// Every tensor of the safetensors files at `paths`, exactly as they were written, on the host.
///
/// For what is not a model: the reference activations a test checks a pass against, a file being
/// looked into. A model is read with
/// [`Weights::from_files`](crate::flint::Weights::from_files) instead, which puts each
/// weight where it will be used as it is read rather than holding all of them here first.
pub fn read_safetensors(paths: &[impl AsRef<Path>]) -> Result<HashMap<String, Tensor>> {
    collect(paths, &mut Ok)
}

/// The same, out of one file's bytes already in hand rather than off the disk.
pub fn parse_safetensors(bytes: &[u8]) -> Result<HashMap<String, Tensor>> {
    collect_bytes(bytes, &mut Ok)
}

/// Every tensor of the files at `paths`, each passed through `place` as it is read, keyed by its
/// whole name.
///
/// One file at a time: its bytes are gone before the next one is read, so the high-water mark is
/// what `place` keeps and one file, not the package twice over.
///
/// A name in two files is refused rather than replaced -- a model whose weights depend on the
/// order its files were listed in is worse than one that will not load -- and every quantized
/// weight has to have its scales beside it; see [`check_fp8_pairs`].
pub(crate) fn collect(
    paths: &[impl AsRef<Path>],
    place: &mut dyn FnMut(Tensor) -> Result<Tensor>,
) -> Result<HashMap<String, Tensor>> {
    let mut tensors = HashMap::new();

    for path in paths {
        let path = path.as_ref();
        let blame = |error: Error| Error::format(format!("{}: {}", path.display(), error));

        let bytes = std::fs::read(path)
            .map_err(|error| Error::format(format!("{}: {error}", path.display())))?;
        each_in(&bytes, &mut |name, tensor| {
            keep(&mut tensors, name, place(tensor)?)
        })
        .map_err(blame)?;
    }

    check_fp8_pairs(&tensors)?;
    Ok(tensors)
}

/// [`collect`], for one file whose bytes are already in hand.
pub(crate) fn collect_bytes(
    bytes: &[u8],
    place: &mut dyn FnMut(Tensor) -> Result<Tensor>,
) -> Result<HashMap<String, Tensor>> {
    let mut tensors = HashMap::new();
    each_in(bytes, &mut |name, tensor| {
        keep(&mut tensors, name, place(tensor)?)
    })?;

    check_fp8_pairs(&tensors)?;
    Ok(tensors)
}

/// Hand every tensor one file holds to `each`, in the order its header lists them.
fn each_in(bytes: &[u8], each: &mut dyn FnMut(String, Tensor) -> Result<()>) -> Result<()> {
    let (header_length, metadata) = SafeTensors::read_metadata(bytes)
        .map_err(|error| Error::format(format!("not a safetensors file: {error}")))?;

    // Where the data section begins, which the header's offsets are relative to.
    let base = HEADER_LENGTH_BYTES + header_length;

    for (name, info) in metadata.tensors() {
        let blame = |error: Error| Error::format(format!("tensor {name:?}: {error}"));

        let shape = dimensions(&info.shape).map_err(blame)?;
        let dtype = dtype(info.dtype).map_err(blame)?;
        let range = base + info.data_offsets.0..base + info.data_offsets.1;

        // `read_metadata` has already checked that the offsets lie inside the file and that they
        // account for it exactly, so this cannot fail -- but the slice below would panic rather
        // than fail if it ever did, which is not what a corrupt package should do.
        if range.end > bytes.len() {
            return Err(blame(Error::format(
                "its bytes are not inside the file".to_string(),
            )));
        }

        let tensor = Tensor::from_bytes(&shape, dtype, &bytes[range])
            .map_err(|error| Error::format(format!("tensor {name:?}: {error}")))?;
        each(name, tensor)?;
    }

    Ok(())
}

/// Keep `tensor` under `name`, unless something already is.
fn keep(tensors: &mut HashMap<String, Tensor>, name: String, tensor: Tensor) -> Result<()> {
    if tensors.contains_key(&name) {
        return Err(Error::format(format!(
            "tensor {name:?} is in more than one file of this model"
        )));
    }

    tensors.insert(name, tensor);
    Ok(())
}

/// The shape a header claims, as the dimensions a tensor is built from.
fn dimensions(shape: &[usize]) -> Result<Vec<i32>> {
    if shape.len() > MAX_RANK {
        return Err(Error::format(format!("rank {} is out of range", shape.len())));
    }

    let mut dimensions = Vec::with_capacity(shape.len());
    for &size in shape {
        if size == 0 || size >= MAX_DIM {
            return Err(Error::format(format!("dimension {size} is out of range")));
        }
        dimensions.push(size as i32);
    }

    Ok(dimensions)
}

/// What a safetensors element type is called here.
///
/// Only what an exporter writes. A checkpoint straight off the hub is usually bfloat16, which is
/// not one of them: what this reads is a model exported for it, and the exporter is where the
/// narrowing to float16 happens. Saying so here beats loading a model whose weights were
/// reinterpreted as the wrong type and drawing noise.
///
/// `F8_E4M3` is the format `docs/fp8.md` describes, and is the one element type here that means
/// nothing on its own: the tensor beside it holding its scales is what makes it a weight. See
/// [`check_fp8_pairs`].
fn dtype(dtype: Dtype) -> Result<DType> {
    match dtype {
        Dtype::F32 => Ok(DType::Float),
        Dtype::F16 => Ok(DType::Float16),
        Dtype::F8_E4M3 => Ok(DType::Fp8E4M3),
        Dtype::I64 => Ok(DType::Long),
        Dtype::I32 => Ok(DType::Int32),
        Dtype::U8 => Ok(DType::UInt8),
        Dtype::I8 => Ok(DType::Int8),
        Dtype::BOOL => Ok(DType::Bool),
        other => Err(Error::format(format!(
            "{other:?} is not an element type this reads; the exporter writes float16, float32, \
             float8_e4m3 and int64"
        ))),
    }
}

/// That every `<fp8e4m3>` tensor has the scales it means itself against beside it.
///
/// A quantized weight is two tensors -- `"…weight"` and `"…weight.scale"` -- and the file format
/// has nowhere to say they belong together, so the name is the whole of the pairing and this is
/// where it is checked. Without it a package that lost its scales, or an exporter that wrote them
/// under another name, would load: the elements are a perfectly good tensor on their own, and what
/// would go wrong is a picture drawn from a weight scaled by nothing, several layers away from
/// anything that could name the cause.
///
/// Run over the whole model once it is read rather than over each file, since a weight and its
/// scale may have been written to different ones. Where each tensor went does not change the
/// answer: placing one moves it, and keeps its type and its shape.
///
/// The other direction is deliberately not checked. A `"…scale"` with no FP8 tensor beside it is
/// an ordinary tensor with an unlucky name, and refusing one would make a suffix this library
/// chose into a word no other exporter may use.
fn check_fp8_pairs(tensors: &HashMap<String, Tensor>) -> Result<()> {
    for (name, tensor) in tensors {
        if tensor.dtype() != DType::Fp8E4M3 {
            continue;
        }

        let scale_name = format!("{name}{CHANNEL_SCALE_SUFFIX}");
        let Some(scale) = tensors.get(&scale_name) else {
            return Err(Error::format(format!(
                "tensor {name:?} is float8_e4m3 and there is no {scale_name:?} beside it; a \
                 quantized weight is the elements and the scales, and the elements alone do not \
                 say what they are worth"
            )));
        };

        let rows = tensor.shape().first().copied().unwrap_or(0);
        if scale.dtype() != DType::Float || scale.shape() != [rows] {
            return Err(Error::format(format!(
                "tensor {scale_name:?} is {:?}{:?}, and the scales of {name:?} have to be \
                 <float>[{rows}] -- one per row",
                scale.dtype(),
                scale.shape(),
            )));
        }
    }

    Ok(())
}
