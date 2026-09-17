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

//! Reading the parameters of a model, which are safetensors files.
//!
//! A [`ParamFile`] is a model's tensors read onto the host and handed out by name -- the whole
//! name, `"model.layer0.attn.weight"` and not `"weight"`. Which namespace is being read is a
//! [`Graph`](crate::flint::Graph)'s, and where a weight goes and in what precision is
//! [`resident`](crate::flint::resident)'s: what comes out of here is what the files hold.
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
//! A model too large for one file is written as several, which a
//! [`Manifest`](crate::Manifest) names in order. They are read into one namespace, so which file
//! a tensor was written to is not something the model has to know.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use std::rc::Rc;

use safetensors::tensor::{Dtype, SafeTensors};

use crate::error::{Error, Result};
use crate::flint::{DType, Tensor, CHANNEL_SCALE_SUFFIX};

/// The largest rank and dimension a stored tensor may have, which keep a corrupt file from asking
/// for an unreasonable allocation.
const MAX_RANK: usize = 16;
const MAX_DIM: usize = 1048576;

/// The one namespace a model's tensors end up in, whichever files they came out of.
type Tensors = HashMap<String, Tensor>;

/// The parameters of a model, read onto the host.
///
/// The whole of every file is read when the file is made and every tensor in it is built as it
/// goes. There is nothing to look up afterwards and nothing left on disk -- a name is a tensor
/// that is already there.
///
/// Cloning one shares the tensors rather than copying them, and so does [`ParamFile::get`]: what
/// it hands over is another handle on storage the file holds, which is what makes handing the
/// same weight to two models cost nothing.
#[derive(Clone)]
pub struct ParamFile {
    tensors: Rc<Tensors>,
}

impl ParamFile {
    /// Read a model held in one or more safetensors files.
    ///
    /// They are read in order into one namespace, so which file a tensor was written to is not
    /// something the model has to know. A name appearing in two of them is refused: which won
    /// would otherwise depend on the order they were listed in.
    pub fn open(paths: &[impl AsRef<Path>]) -> Result<ParamFile> {
        let mut tensors = Tensors::new();

        for path in paths {
            let path = path.as_ref();
            let bytes = std::fs::read(path)
                .map_err(|error| Error::format(format!("{}: {error}", path.display())))?;

            read_into(&bytes, &mut tensors)
                .map_err(|error| Error::format(format!("{}: {error}", path.display())))?;
        }
        check_fp8_pairs(&tensors)?;

        Ok(ParamFile {
            tensors: Rc::new(tensors),
        })
    }

    /// The same, out of bytes already in hand rather than off the disk.
    pub fn parse(bytes: &[u8]) -> Result<ParamFile> {
        let mut tensors = Tensors::new();
        read_into(bytes, &mut tensors)?;
        check_fp8_pairs(&tensors)?;

        Ok(ParamFile {
            tensors: Rc::new(tensors),
        })
    }

    /// The tensor the file calls `name`, checked against the shape the caller expects.
    ///
    /// `name` is the whole name, since that is all a file has. What comes back is what was
    /// written: the element type the exporter chose, on the host the bytes were read onto.
    /// Putting it where the model runs is [`resident`](crate::flint::resident)'s to do.
    pub fn get(&self, name: &str, shape: &[i32]) -> Result<Tensor> {
        let tensor = self.get_unchecked(name)?;
        if tensor.shape() != shape {
            return Err(Error::model(format!(
                "tensor {name:?} has shape {:?}, expected {shape:?}",
                tensor.shape(),
            )));
        }
        Ok(tensor)
    }

    /// The same, whatever shape it turns out to have.
    pub fn get_unchecked(&self, name: &str) -> Result<Tensor> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| Error::model(format!("tensor {name:?} not found in model")))
    }

    /// Whether the file holds a tensor called `name`.
    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Every tensor the file held, taken out of it.
    ///
    /// For a reader that is going to put the whole file somewhere else and wants to let go of it
    /// as it goes rather than at the end: taking a tensor out of what comes back drops this file's
    /// handle on it, so the bytes go once the copy is made. What
    /// [`Pinned::read`](crate::flint::Pinned::read) is for, where both copies are host memory and
    /// holding the weights twice is the difference between fitting and not.
    ///
    /// A file someone else still has a handle on cannot be emptied, and hands over a map of
    /// handles instead. That is correct and costs nothing extra -- it is what cloning a file has
    /// always done -- but the sharer still holds what it holds.
    pub fn take(self) -> Tensors {
        Rc::try_unwrap(self.tensors).unwrap_or_else(|shared| (*shared).clone())
    }

    /// The whole names of every tensor the file held, in order. For finding out what a model
    /// actually calls things when it fails to find what it expected.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// How many tensors the file held.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

impl fmt::Debug for ParamFile {
    /// How much is in the file rather than every tensor in it: a model holds hundreds, and this
    /// is read in the middle of an error message.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParamFile")
            .field("tensors", &self.tensors.len())
            .finish()
    }
}

/// Every tensor of one safetensors file, into the namespace they share.
fn read_into(bytes: &[u8], tensors: &mut Tensors) -> Result<()> {
    let file = SafeTensors::deserialize(bytes)
        .map_err(|error| Error::format(format!("not a safetensors file: {error}")))?;

    for (name, view) in file.tensors() {
        let tensor =
            build(&view).map_err(|error| Error::format(format!("tensor {name:?}: {error}")))?;

        // A name that is already there is refused rather than replaced: a model whose weights
        // depend on which file was read first is worse than one that will not load.
        if tensors.insert(name.clone(), tensor).is_some() {
            return Err(Error::format(format!(
                "tensor {name:?} is in more than one file of this model"
            )));
        }
    }
    Ok(())
}

/// One tensor, as the shape and type its header claims.
fn build(view: &safetensors::tensor::TensorView) -> Result<Tensor> {
    let shape = view.shape();
    if shape.len() > MAX_RANK {
        return Err(Error::format(format!(
            "rank {} is out of range",
            shape.len()
        )));
    }

    let mut dimensions = Vec::with_capacity(shape.len());
    for &size in shape {
        if size == 0 || size >= MAX_DIM {
            return Err(Error::format(format!("dimension {size} is out of range")));
        }
        dimensions.push(size as i32);
    }

    Ok(Tensor::from_bytes(
        &dimensions,
        dtype(view.dtype())?,
        view.data(),
    )?)
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
/// Run over the whole model rather than over each file, since a weight and its scale may have been
/// written to different ones.
///
/// The other direction is deliberately not checked. A `"…scale"` with no FP8 tensor beside it is
/// an ordinary tensor with an unlucky name, and refusing one would make a suffix this library
/// chose into a word no other exporter may use.
fn check_fp8_pairs(tensors: &Tensors) -> Result<()> {
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
