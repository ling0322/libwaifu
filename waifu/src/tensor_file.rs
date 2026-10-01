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
//! Nothing is copied. Each file is mapped into memory, and each of its tensors is a CPU tensor
//! whose storage is that mapping -- the page cache itself; see [`Tensor::borrowing`]. So what a
//! caller keeps is the caller's: a model that puts its weights on the card copies each one there
//! straight out of the mapping, and one that keeps them on the host -- low-vram, or the CPU --
//! keeps the mapping and nothing else. The host holds no copy the caller did not ask for, and a
//! package two processes have open is in memory once.
//!
//! Two things follow. The disk is read when a page is first touched rather than when the package
//! is opened, and a mapping touched out of order is, on a spinning disk, a seek for every weight.
//! So tensors are taken out of a file in the order they lie in it, and each is read then: a weight
//! moved to the card is read by the move, and one kept on the host is read through by
//! [`Weights`](crate::flint::Weights) as it arrives. And a file must not change while it is
//! mapped -- one truncated or rewritten in place faults the next time a weight in it is read.
//!
//! There is no object in between that holds a model's files. A model is read once, into the
//! [`ParamSource`](crate::flint::ParamSource) it runs from; the mappings go with the last tensor
//! on them.

use std::collections::HashMap;
use std::fs::File;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use safetensors::tensor::{Dtype, SafeTensors};

use crate::error::{Error, Result};
use crate::flint::{DType, Device, Residency, Tensor, Weights, CHANNEL_SCALE_SUFFIX};

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
    let mut tensors = HashMap::new();
    each_mapped(paths, &mut |name, tensor| keep(&mut tensors, name, tensor))?;
    check_fp8_pairs(&tensors)?;
    Ok(tensors)
}

/// The same, out of one file's bytes already in hand rather than off the disk.
pub fn parse_safetensors(bytes: &[u8]) -> Result<HashMap<String, Tensor>> {
    let mut tensors = HashMap::new();
    each_copied(bytes, &mut |name, tensor| keep(&mut tensors, name, tensor))?;
    check_fp8_pairs(&tensors)?;
    Ok(tensors)
}

/// A model's weights out of the files at `paths`, for a model running on `device` with its
/// weights held the way `residency` says -- what [`Weights::from_files`] is.
///
/// Each tensor borrows its file's mapping rather than a copy of it, and [`Weights::insert`] puts
/// it where it waits as it arrives, so what the host holds is what the weights keep: nothing, for
/// a weight moved to the card. Every quantized weight has to have its scales beside it; see
/// [`check_fp8_pairs`].
pub(crate) fn collect(
    paths: &[impl AsRef<Path>],
    device: Device,
    residency: Residency,
) -> Result<Weights> {
    let mut weights = Weights::empty(device, residency)?;
    each_mapped(paths, &mut |name, tensor| weights.insert(name, tensor))?;
    check_fp8_pairs(weights.tensors())?;
    Ok(weights)
}

/// [`collect`], for one file whose bytes are already in hand.
pub(crate) fn collect_bytes(bytes: &[u8], device: Device, residency: Residency) -> Result<Weights> {
    let mut weights = Weights::empty(device, residency)?;
    each_copied(bytes, &mut |name, tensor| weights.insert(name, tensor))?;
    check_fp8_pairs(weights.tensors())?;
    Ok(weights)
}

/// Hand every tensor of the files at `paths` to `each`, file by file and in the order each file
/// lies in, each one borrowing its file's mapping.
fn each_mapped(
    paths: &[impl AsRef<Path>],
    each: &mut dyn FnMut(String, Tensor) -> Result<()>,
) -> Result<()> {
    for path in paths {
        let path = path.as_ref();
        let blame = |error: Error| Error::format(format!("{}: {}", path.display(), error));

        let map = Arc::new(map(path).map_err(blame)?);
        each_in(&map, &mut |name, stored| {
            let tensor = borrowed(&map, stored)
                .map_err(|error| Error::format(format!("tensor {name:?}: {error}")))?;
            each(name, tensor)
        })
        .map_err(blame)?;
    }
    Ok(())
}

/// Hand every tensor of one file's `bytes` to `each`, in the order it lies in, each one a copy.
fn each_copied(bytes: &[u8], each: &mut dyn FnMut(String, Tensor) -> Result<()>) -> Result<()> {
    each_in(bytes, &mut |name, stored| {
        let tensor = Tensor::from_bytes(&stored.shape, stored.dtype, &bytes[stored.range])
            .map_err(|error| Error::format(format!("tensor {name:?}: {error}")))?;
        each(name, tensor)
    })
}

/// `path`, mapped read-only.
///
/// Nothing is read here: a page is read when it is first touched, which is why the tensors are
/// handed on in file order -- see the module documentation. `madvise(MADV_WILLNEED)` would ask the
/// kernel to start reading ahead, and does not do it: it reads one readahead window, 8 MB of a
/// 6.5 GB file, measured.
fn map(path: &Path) -> Result<Mmap> {
    let file = File::open(path).map_err(|error| Error::format(error.to_string()))?;

    // SAFETY: the mapping is only read, and stays valid for as long as the file's contents do.
    // What would break it is the file being truncated or rewritten in place while it is held --
    // the module documentation says why a package is not -- and that is a fault on the read that
    // finds it, not memory that silently changes under a weight.
    unsafe { Mmap::map(&file) }.map_err(|error| Error::format(error.to_string()))
}

/// A tensor out of `map`, as a view of it where its elements are aligned and as a copy where they
/// are not.
///
/// The safetensors writers align every tensor to its element size, but the format does not
/// promise it, and the CPU kernels read elements as their own type. A mapping starts on a page, so
/// whether a tensor is aligned is whether its offset into the file is.
fn borrowed(map: &Arc<Mmap>, stored: Stored) -> Result<Tensor> {
    let alignment = stored.dtype.total_size(1).max(1) as usize;
    if stored.range.start % alignment == 0 {
        Ok(Tensor::borrowing(&stored.shape, stored.dtype, map, stored.range)?)
    } else {
        Ok(Tensor::from_bytes(&stored.shape, stored.dtype, &map[stored.range])?)
    }
}

/// One tensor as a header describes it: what it is, and where its bytes are in the file.
struct Stored {
    shape: Vec<i32>,
    dtype: DType,
    range: Range<usize>,
}

/// Hand every tensor one file holds to `each`, in the order its header lists them.
fn each_in(bytes: &[u8], each: &mut dyn FnMut(String, Stored) -> Result<()>) -> Result<()> {
    let (header_length, metadata) = SafeTensors::read_metadata(bytes)
        .map_err(|error| Error::format(format!("not a safetensors file: {error}")))?;

    // Where the data section begins, which the header's offsets are relative to.
    let base = HEADER_LENGTH_BYTES + header_length;

    // In the order they lie in the file, not the order the header's index happens to hand them
    // out in: taking a mapped file apart out of order is a seek for every tensor.
    let mut stored: Vec<_> = metadata.tensors().into_iter().collect();
    stored.sort_by_key(|(_, info)| info.data_offsets.0);

    for (name, info) in stored {
        let blame = |error: Error| Error::format(format!("tensor {name:?}: {error}"));

        let shape = dimensions(&info.shape).map_err(blame)?;
        let dtype = dtype(info.dtype).map_err(blame)?;
        let range = base + info.data_offsets.0..base + info.data_offsets.1;

        // `read_metadata` has already checked that the offsets lie inside the file and that they
        // account for it exactly, so this cannot fail -- but the slice a caller takes of it would
        // panic rather than fail if it ever did, which is not what a corrupt package should do.
        if range.end > bytes.len() {
            return Err(blame(Error::format(
                "its bytes are not inside the file".to_string(),
            )));
        }

        each(name, Stored { shape, dtype, range })?;
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
/// A scale of `<float>[rows]` (one per output channel, [`WeightFormat::Fp8`](crate::flint::WeightFormat::Fp8))
/// or `<float>[1]` (one for the whole weight, [`WeightFormat::Fp8TensorScale`](crate::flint::WeightFormat::Fp8TensorScale))
/// both pass here: which one a model actually wants is a configuration's call, not this reader's --
/// see [`WeightFormat`](crate::flint::WeightFormat) -- and by the time a graph is built for it,
/// [`Linear::graph`](crate::Linear::graph) asks for the scale at the exact shape its format means,
/// so a package that names the wrong shape for what its manifest declares fails there instead.
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
        let shape = scale.shape();
        if scale.dtype() != DType::Float || (shape != [rows] && shape != [1]) {
            return Err(Error::format(format!(
                "tensor {scale_name:?} is {:?}{:?}, and the scales of {name:?} have to be \
                 <float>[{rows}] -- one per row -- or <float>[1] -- one for the whole weight",
                scale.dtype(),
                scale.shape(),
            )));
        }
    }

    Ok(())
}
