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
//! A [`ParamFile`] is a model's tensors, named by the whole name --
//! `"model.layer0.attn.weight"` and not `"weight"`. Which namespace is being read is a
//! [`Graph`](crate::flint::Graph)'s, and where a weight goes and in what precision is the
//! [`Ir`](crate::flint::Ir)'s: what comes out of here is what the files hold.
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
//!
//! # What opening one costs
//!
//! The headers, and nothing else. Opening maps each file and parses the table at the front of it;
//! the bytes of a weight are read at the moment something asks for that weight, and not before.
//!
//! This is what lets a package be larger than the memory it is read on. The older form of this
//! file read every byte of every file up front and built every tensor as it went, so a 26 GB model
//! was 26 GB of ordinary host memory before the first instruction ran -- which is not a slow way
//! to start such a model so much as no way to start it at all.
//!
//! It also moves the decision. When a weight crosses into memory, whether it is page-locked on the
//! way, and when it is let go of again are things the [`Ir`](crate::flint::Ir) says, one
//! instruction at a time, and this is the layer that makes them sayable: a name here is a range of
//! a mapping, so asking for one twice is two reads and asking for none is no reads.
//!
//! The mapping is not the cache. The kernel's page cache is -- the pages of a package are touched
//! once per sampler step in the same order every step, which is the access pattern it is best at
//! -- so nothing here keeps a copy of what it has already handed out.

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::ops::Range;
use std::path::Path;
use std::rc::Rc;

use memmap2::Mmap;
use safetensors::tensor::{Dtype, SafeTensors};

use crate::error::{Error, Result};
use crate::flint::{DType, Tensor, CHANNEL_SCALE_SUFFIX};

/// The largest rank and dimension a stored tensor may have, which keep a corrupt file from asking
/// for an unreasonable allocation.
const MAX_RANK: usize = 16;
const MAX_DIM: usize = 1048576;

/// How many bytes a safetensors file spends saying how long its header is.
const HEADER_LENGTH_BYTES: usize = 8;

/// Where one tensor's bytes are, which is all that is kept about it until it is asked for.
#[derive(Clone, Debug)]
struct Entry {
    /// Which of the model's files holds it, as an index into [`ParamFile::stores`].
    store: usize,
    dtype: DType,
    shape: Vec<i32>,
    /// The bytes, as a range of that file rather than of its data section: what is recorded is
    /// what a read is issued against, so that reading one involves no arithmetic to get wrong.
    range: Range<usize>,
}

impl Entry {
    /// How many bytes the tensor is, which is the length of its range.
    fn nbytes(&self) -> usize {
        self.range.end - self.range.start
    }
}

/// One file of a model, mapped.
///
/// Either a mapping of a file on disk, or bytes the caller already had. The second is what
/// [`ParamFile::parse`] is given -- a package read from somewhere that is not a path -- and it
/// behaves the same way from here on, since what the rest of this wants of either is a slice.
enum Store {
    Mapped(Mmap),
    Held(Vec<u8>),
}

impl Store {
    fn bytes(&self) -> &[u8] {
        match self {
            Store::Mapped(mapping) => mapping,
            Store::Held(bytes) => bytes,
        }
    }
}

impl fmt::Debug for Store {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (kind, len) = match self {
            Store::Mapped(mapping) => ("mapped", mapping.len()),
            Store::Held(bytes) => ("held", bytes.len()),
        };
        write!(f, "{kind}({len} bytes)")
    }
}

/// The parameters of a model, as the files that hold them.
///
/// Cloning one shares the mapping rather than making a second, which is what lets the halves of a
/// model -- SDXL is four passes over one package -- be handed the same file without any of them
/// having to know that the others were.
///
/// What [`ParamFile::get`] hands over is a tensor of its own: the bytes are copied out of the
/// mapping, because a tensor owns its storage and the mapping is read-only. So two calls for the
/// same name are two tensors, where the older form of this file would have handed out two handles
/// on one. Nothing relied on that, and the thing that would have -- a model read once and run
/// many times -- is served by where a weight comes to rest rather than by this.
#[derive(Clone)]
pub struct ParamFile {
    entries: Rc<HashMap<String, Entry>>,
    stores: Rc<Vec<Store>>,
}

impl ParamFile {
    /// Read the headers of a model held in one or more safetensors files.
    ///
    /// They are read in order into one namespace, so which file a tensor was written to is not
    /// something the model has to know. A name appearing in two of them is refused: which won
    /// would otherwise depend on the order they were listed in.
    ///
    /// Only the headers. The weights themselves are read when they are asked for -- see the note
    /// on what opening costs at the top of this file.
    pub fn open(paths: &[impl AsRef<Path>]) -> Result<ParamFile> {
        let mut entries = HashMap::new();
        let mut stores = Vec::with_capacity(paths.len());

        for path in paths {
            let path = path.as_ref();
            let blame = |error: Error| Error::format(format!("{}: {}", path.display(), error));

            let file = File::open(path)
                .map_err(|error| Error::format(format!("{}: {error}", path.display())))?;

            // Safety: mapping a file is unsound in exactly one way -- something else truncating
            // or writing it underneath us -- and a package is a read-only artifact that nothing
            // in this process writes. The same assumption every runtime that mmaps a checkpoint
            // makes, and the reason this is not offered for a file the caller is still writing.
            let mapping = unsafe { Mmap::map(&file) }
                .map_err(|error| Error::format(format!("{}: {error}", path.display())))?;

            index_into(Store::Mapped(mapping), &mut stores, &mut entries).map_err(blame)?;
        }
        check_fp8_pairs(&entries)?;

        Ok(ParamFile {
            entries: Rc::new(entries),
            stores: Rc::new(stores),
        })
    }

    /// The same, out of bytes already in hand rather than off the disk.
    ///
    /// The bytes are kept, since what is handed out later are ranges of them.
    pub fn parse(bytes: &[u8]) -> Result<ParamFile> {
        let mut entries = HashMap::new();
        let mut stores = Vec::with_capacity(1);
        index_into(Store::Held(bytes.to_vec()), &mut stores, &mut entries)?;
        check_fp8_pairs(&entries)?;

        Ok(ParamFile {
            entries: Rc::new(entries),
            stores: Rc::new(stores),
        })
    }

    /// The tensor the file calls `name`, checked against the shape the caller expects.
    ///
    /// `name` is the whole name, since that is all a file has. What comes back is what was
    /// written: the element type the exporter chose, on the host the bytes were read onto.
    /// Putting it where the model runs is the [`Ir`](crate::flint::Ir)'s to do.
    pub fn get(&self, name: &str, shape: &[i32]) -> Result<Tensor> {
        let entry = self.entry(name)?;
        if entry.shape != shape {
            return Err(Error::model(format!(
                "tensor {name:?} has shape {:?}, expected {shape:?}",
                entry.shape,
            )));
        }

        self.build(name, entry)
    }

    /// The same, whatever shape it turns out to have.
    pub fn get_unchecked(&self, name: &str) -> Result<Tensor> {
        self.build(name, self.entry(name)?)
    }

    /// Read `name` into storage the caller has already allocated.
    ///
    /// The one way to get a weight into memory that is not ordinary: `destination` may be
    /// page-locked host memory, and then this is the read that fills it, rather than a read into
    /// a buffer of this file's choosing followed by a copy into that one. For a model being
    /// streamed a weight at a time, those two differ by the whole model memcpy'd once per step.
    ///
    /// `destination` has to be a host tensor of exactly this weight's shape and element type: what
    /// is written is the bytes as the file holds them, and nothing here converts anything.
    pub fn read_into(&self, name: &str, destination: &mut Tensor) -> Result<()> {
        let entry = self.entry(name)?;

        if destination.shape() != entry.shape {
            return Err(Error::model(format!(
                "tensor {name:?} has shape {:?}, but the storage given for it is {:?}",
                entry.shape,
                destination.shape(),
            )));
        }
        if destination.dtype() != entry.dtype {
            return Err(Error::model(format!(
                "tensor {name:?} is {:?}, but the storage given for it is {:?}",
                entry.dtype,
                destination.dtype(),
            )));
        }

        let bytes = destination.host_bytes_mut()?;
        if bytes.len() != entry.nbytes() {
            return Err(Error::model(format!(
                "tensor {name:?} is {} bytes, but the storage given for it is {}",
                entry.nbytes(),
                bytes.len(),
            )));
        }

        bytes.copy_from_slice(self.bytes_of(entry));
        Ok(())
    }

    /// Whether the file holds a tensor called `name`.
    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// What shape the tensor called `name` is, without reading it.
    ///
    /// For a model that reads its own architecture out of the package rather than out of a
    /// configuration -- Anima's autoencoder counts its stages by looking for them. Answered from
    /// the header, so asking costs nothing: the point of a lazily read file is undone by a caller
    /// that has to read a weight to find out how big it is.
    pub fn shape_of(&self, name: &str) -> Option<Vec<i32>> {
        self.entries.get(name).map(|entry| entry.shape.clone())
    }

    /// What element type the tensor called `name` is, without reading it.
    pub fn dtype_of(&self, name: &str) -> Option<DType> {
        self.entries.get(name).map(|entry| entry.dtype)
    }

    /// The whole names of every tensor the file holds, in order. For finding out what a model
    /// actually calls things when it fails to find what it expected.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.entries.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// How many tensors the file holds.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entry for `name`, or what is wrong with asking for it.
    fn entry(&self, name: &str) -> Result<&Entry> {
        self.entries
            .get(name)
            .ok_or_else(|| Error::model(format!("tensor {name:?} not found in model")))
    }

    /// The bytes an entry points at.
    ///
    /// The range was checked against the length of its store when the header was read, so this
    /// does not have to be fallible.
    fn bytes_of(&self, entry: &Entry) -> &[u8] {
        &self.stores[entry.store].bytes()[entry.range.clone()]
    }

    /// One tensor, copied out of the mapping onto the host.
    fn build(&self, name: &str, entry: &Entry) -> Result<Tensor> {
        Tensor::from_bytes(&entry.shape, entry.dtype, self.bytes_of(entry))
            .map_err(|error| Error::format(format!("tensor {name:?}: {error}")))
    }
}

impl fmt::Debug for ParamFile {
    /// How much is in the file rather than every tensor in it: a model holds hundreds, and this
    /// is read in the middle of an error message.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParamFile")
            .field("tensors", &self.entries.len())
            .field("files", &self.stores.len())
            .finish()
    }
}

/// Read one file's header and record where each of its tensors is.
///
/// The store is pushed whether or not it turns out to hold anything, so that the indices already
/// handed out to entries stay correct.
fn index_into(
    store: Store,
    stores: &mut Vec<Store>,
    entries: &mut HashMap<String, Entry>,
) -> Result<()> {
    let index = stores.len();
    let bytes = store.bytes();

    let (header_length, metadata) = SafeTensors::read_metadata(bytes)
        .map_err(|error| Error::format(format!("not a safetensors file: {error}")))?;

    // Where the data section begins, which the header's offsets are relative to.
    let base = HEADER_LENGTH_BYTES + header_length;

    let mut found = Vec::with_capacity(metadata.tensors().len());
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

        found.push((
            name.clone(),
            Entry {
                store: index,
                dtype,
                shape,
                range,
            },
        ));
    }

    stores.push(store);

    for (name, entry) in found {
        // A name that is already there is refused rather than replaced: a model whose weights
        // depend on which file was read first is worse than one that will not load.
        if entries.insert(name.clone(), entry).is_some() {
            return Err(Error::format(format!(
                "tensor {name:?} is in more than one file of this model"
            )));
        }
    }

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
/// Run over the whole model rather than over each file, since a weight and its scale may have been
/// written to different ones.
///
/// The other direction is deliberately not checked. A `"…scale"` with no FP8 tensor beside it is
/// an ordinary tensor with an unlucky name, and refusing one would make a suffix this library
/// chose into a word no other exporter may use.
/// Asked of the index rather than of the tensors, so that a package is checked without any of it
/// being read: the header says the type and the shape, which is the whole of the question.
fn check_fp8_pairs(entries: &HashMap<String, Entry>) -> Result<()> {
    for (name, entry) in entries {
        if entry.dtype != DType::Fp8E4M3 {
            continue;
        }

        let scale_name = format!("{name}{CHANNEL_SCALE_SUFFIX}");
        let Some(scale) = entries.get(&scale_name) else {
            return Err(Error::format(format!(
                "tensor {name:?} is float8_e4m3 and there is no {scale_name:?} beside it; a \
                 quantized weight is the elements and the scales, and the elements alone do not \
                 say what they are worth"
            )));
        };

        let rows = entry.shape.first().copied().unwrap_or(0);
        if scale.dtype != DType::Float || scale.shape != [rows] {
            return Err(Error::format(format!(
                "tensor {scale_name:?} is {:?}{:?}, and the scales of {name:?} have to be \
                 <float>[{rows}] -- one per row",
                scale.dtype, scale.shape,
            )));
        }
    }

    Ok(())
}
