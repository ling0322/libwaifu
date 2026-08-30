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

//! Reading the parameters of a model.
//!
//! A package holds its tensors in one `tdicv2` stream: a flat table of name to tensor, written
//! with the shape and the element type of each. A [`VarBuilder`] reads that table once and then
//! hands out views of it, one per module, so that a layer asks for `"weight"` and gets the tensor
//! the whole name of which is `"model.layer0.attn.weight"`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::rc::Rc;

use crate::flint::{DType, Device, Tensor};

use crate::error::{Error, Result};
use crate::reader::BinaryRead;
use crate::zip_file::{EntryHandle, ZipFile};

/// The header every parameter file starts with.
const MAGIC: &str = "llyn::tdicv2    ";

/// The magic number that closes a tensor's data, as a check that it was read to the right place.
const DATA_MAGIC: i16 = 0x55aa;

/// The largest rank and dimension a stored tensor may have, which keep a corrupt file from asking
/// for an unreasonable allocation.
const MAX_RANK: i16 = 16;
const MAX_DIM: i32 = 1048576;

/// Where one tensor's bytes are and what to read them as.
#[derive(Clone, Debug)]
struct Record {
    source: usize,
    offset: u64,
    length: usize,
    shape: Vec<i32>,
    dtype: DType,
}

/// Somewhere a tensor's bytes can be read from when they are wanted.
enum Source {
    /// A parameter file already in memory, which is what a test hands over and what a small one
    /// is not worth doing anything cleverer with.
    Memory(Vec<u8>),
    /// An entry of a package on disk, read a tensor at a time. A `RefCell` because the handle
    /// seeks and a builder is shared by every layer that reads through it; the library is single
    /// threaded per device, which is the same assumption everything else here makes.
    Entry(RefCell<EntryHandle>),
}

impl Source {
    fn read(&self, offset: u64, length: usize) -> Result<Vec<u8>> {
        match self {
            Source::Memory(bytes) => {
                let end = offset as usize + length;
                if end > bytes.len() {
                    return Err(Error::format("a tensor runs past the end of the parameters"));
                }
                Ok(bytes[offset as usize..end].to_vec())
            }
            Source::Entry(handle) => handle.borrow_mut().read_at(offset, length),
        }
    }
}

/// The parameters of a model, and where in them this builder points.
///
/// What it holds is an index rather than the parameters: walking a file says what is in it and
/// where, and a tensor is read when a layer asks for it. A model is most of a package -- SDXL's
/// is 6.97 GB of 6.97 -- so reading it all in first means holding it twice for the length of the
/// load, once here and once on the device it is being copied to. This way the host holds one
/// tensor at a time, and a tensor nothing asks for is never read at all.
///
/// Cloning a builder or narrowing it with [`VarBuilder::with_name`] shares the index and the
/// files rather than copying them.
#[derive(Clone)]
pub struct VarBuilder {
    records: Rc<HashMap<String, Record>>,
    sources: Rc<Vec<Source>>,
    namespace: String,
    device: Device,
    float_type: DType,
}

impl VarBuilder {
    /// Read a parameter file, moving what it holds onto `device` as each tensor is asked for.
    ///
    /// Float tensors are cast to `float_type` on the way out, so that a file written in one
    /// precision can drive a device that works in another.
    pub fn from_reader(
        reader: &mut impl Read,
        device: Device,
        float_type: DType,
    ) -> Result<VarBuilder> {
        VarBuilder::from_readers([reader], device, float_type)
    }

    /// The same, over a model written as several files rather than one.
    ///
    /// Each reader is a whole parameter file in its own right -- its own header, its own records,
    /// its own end -- holding a part of the model. They are read in order and their tensors put
    /// into one namespace, so which part a tensor was written to is not something the model has to
    /// know. A name appearing in two of them is refused: it would otherwise depend on the order.
    pub fn from_readers<R: Read>(
        readers: impl IntoIterator<Item = R>,
        device: Device,
        float_type: DType,
    ) -> Result<VarBuilder> {
        let mut sources = Vec::new();
        let mut records = HashMap::new();
        for mut reader in readers {
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes)?;
            index_params(sources.len(), &mut Cursor::new(&bytes), &mut records)?;
            sources.push(Source::Memory(bytes));
        }

        Ok(VarBuilder {
            records: Rc::new(records),
            sources: Rc::new(sources),
            namespace: String::new(),
            device,
            float_type,
        })
    }

    /// Read the index of a model held in one or more packages, leaving the tensors on disk.
    ///
    /// This is the one to use for a model: nothing is read until a layer asks, so the host never
    /// holds more than the tensor being handed over. `entry` is the name the parameters go under
    /// inside each package, which is the same in all of them.
    pub fn from_packages(
        packages: &[ZipFile],
        entry: &str,
        device: Device,
        float_type: DType,
    ) -> Result<VarBuilder> {
        let mut sources = Vec::new();
        let mut records = HashMap::new();
        for package in packages {
            let mut handle = package.entry_handle(entry)?;
            index_params(sources.len(), &mut handle, &mut records)?;
            sources.push(Source::Entry(RefCell::new(handle)));
        }

        Ok(VarBuilder {
            records: Rc::new(records),
            sources: Rc::new(sources),
            namespace: String::new(),
            device,
            float_type,
        })
    }

    /// A builder pointing at the `name` sub-namespace of this one.
    pub fn with_name(&self, name: &str) -> VarBuilder {
        let mut child = self.clone();
        child.namespace = self.name_of(name);
        child
    }

    /// The same builder, handing out float tensors in `float_type` instead.
    ///
    /// One model does not have to be in one precision. SDXL's autoencoder is the case this exists
    /// for: it is marked `force_upcast` and overflows half, so it is built from a float32 view of
    /// the same file the rest of the model is read from in half.
    pub fn with_float_type(&self, float_type: DType) -> VarBuilder {
        let mut child = self.clone();
        child.float_type = float_type;
        child
    }

    /// The same, in this builder's float type whatever the file held it as.
    ///
    /// [`VarBuilder::get`] leaves a matrix as it was stored, since a model is mostly matrices and
    /// widening them is what doubles it. That is the wrong answer for a parameter small enough
    /// not to matter and awkward enough to leave narrow -- a position embedding is added to what
    /// comes out of the token table, and an addition would rather have two of the same thing.
    pub fn get_widened(&self, name: &str, shape: &[i32]) -> Result<Tensor> {
        let tensor = self.get(name, shape)?;
        if tensor.dtype() == DType::Float || tensor.dtype() == DType::Float16 {
            Ok(tensor.cast(self.float_type)?)
        } else {
            Ok(tensor)
        }
    }

    /// The tensor called `name` here, checked against the shape the caller expects.
    pub fn get(&self, name: &str, shape: &[i32]) -> Result<Tensor> {
        let tensor = self.get_unchecked(name)?;
        if tensor.shape() != shape {
            return Err(Error::model(format!(
                "tensor {:?} has shape {:?}, expected {:?}",
                self.name_of(name),
                tensor.shape(),
                shape
            )));
        }
        Ok(tensor)
    }

    /// The tensor called `name` here, whatever shape it turns out to have.
    pub fn get_unchecked(&self, name: &str) -> Result<Tensor> {
        let full_name = self.name_of(name);
        let record = self
            .records
            .get(&full_name)
            .ok_or_else(|| Error::model(format!("tensor {full_name:?} not found in model")))?;

        let bytes = self.sources[record.source].read(record.offset, record.length)?;
        let tensor = Tensor::from_bytes(&record.shape, record.dtype, &bytes)?;

        let tensor = tensor.to_device(self.device)?;
        if tensor.dtype() != DType::Float && tensor.dtype() != DType::Float16 {
            return Ok(tensor);
        }

        // A matrix or a convolution kernel is left as the file stored it; everything else is
        // widened to what this builder was asked for.
        //
        // The two are not the same kind of thing. A rank of two or more is a weight, and weights
        // are the model: 6.964 GB of SDXL's 6.969, over 803 tensors. The rest is a bias or a
        // norm's scale, one value per channel, 1340 of them and 5 MB between them.
        //
        // So the weights stay where they are and the small things widen. On a device whose float
        // type is already the file's this changes nothing; on one whose is not -- x64, which has
        // no half arithmetic and would otherwise read a 6.97 GB model as 13.74 -- it is the
        // difference between holding the model once and holding it twice. What multiplies them
        // takes a half weight against a float activation and converts as it packs, so the
        // arithmetic is the same either way.
        if tensor.dim()? >= 2 {
            Ok(tensor)
        } else {
            Ok(tensor.cast(self.float_type)?)
        }
    }

    /// Whether a tensor called `name` is here.
    pub fn has(&self, name: &str) -> bool {
        self.records.contains_key(&self.name_of(name))
    }

    /// The whole name of `name` in this namespace.
    pub fn name_of(&self, name: &str) -> String {
        if self.namespace.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.namespace, name)
        }
    }

    /// The name of this namespace itself, for a message about what a module was missing.
    pub fn name(&self) -> &str {
        &self.namespace
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn float_type(&self) -> DType {
        self.float_type
    }

    /// The whole names of every tensor the file held, in order. For finding out what a package
    /// actually calls things when a model fails to find what it expected.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.records.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// How many tensors the file held.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

impl fmt::Debug for VarBuilder {
    /// Names the namespace and how much is in the file, rather than every tensor in it: a model
    /// holds hundreds, and this is read in the middle of an error message.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VarBuilder")
            .field("namespace", &self.namespace)
            .field("tensors", &self.records.len())
            .field("device", &self.device)
            .field("float_type", &self.float_type)
            .finish()
    }
}

/// Reads one whole parameter file into `params`.
/// Walk one parameter file and say what is in it and where, without reading any of it.
///
/// The data of each tensor is stepped over rather than loaded, so this costs a seek per tensor
/// instead of the size of the model. What it leaves behind is enough to read any one of them
/// later: which source it is in, where its bytes start, and what shape and type to read them as.
fn index_params(
    source: usize,
    reader: &mut (impl Read + Seek),
    records: &mut HashMap<String, Record>,
) -> Result<()> {
    reader.expect_tag(MAGIC)?;
    reader.expect_tag("<d> ")?;

    let mut tag = reader.read_string(4)?;
    while tag != "</d>" {
        if tag != "<r> " {
            return Err(Error::format(format!(
                "expected a record or the end of the parameters, got {tag:?}"
            )));
        }

        let (name, record) = index_named_tensor(source, reader)?;
        if records.insert(name.clone(), record).is_some() {
            return Err(Error::format(format!(
                "tensor {name:?} is in more than one part of this model"
            )));
        }

        reader.expect_tag("</r>")?;
        tag = reader.read_string(4)?;
    }

    Ok(())
}

/// Where one `name -> tensor` record's data begins, and what to read it as.
fn index_named_tensor(source: usize, reader: &mut (impl Read + Seek)) -> Result<(String, Record)> {
    let name_length = reader.read_i16()?;
    if name_length <= 0 {
        return Err(Error::format("tensor name is empty"));
    }

    let name = reader.read_string(name_length as usize)?;
    reader.expect_tag("tnsr")?;

    let rank = reader.read_i16()?;
    if !(0..=MAX_RANK).contains(&rank) {
        return Err(Error::format(format!("tensor rank {rank} is out of range")));
    }

    let mut shape = Vec::with_capacity(rank as usize);
    for _ in 0..rank {
        let size = reader.read_i32()?;
        if size <= 0 || size >= MAX_DIM {
            return Err(Error::format(format!(
                "tensor dimension {size} is out of range"
            )));
        }
        shape.push(size);
    }

    reader.expect_tag("tdat")?;
    let slots = reader.read_i32()?;
    if slots != 1 {
        return Err(Error::format(format!(
            "tensor holds {slots} data slots, expected 1"
        )));
    }

    let dtype = DType::from_code(reader.read_i16()? as i32)
        .map_err(|error| Error::format(format!("tensor has an unknown element type: {error}")))?;
    let numel = reader.read_i64()?;
    let expected: i64 = shape.iter().map(|&size| size as i64).product();
    if numel != expected {
        return Err(Error::format(format!(
            "tensor holds {numel} elements but its shape calls for {expected}"
        )));
    }

    let length = dtype.total_size(numel) as u64;
    let offset = reader.stream_position()?;
    reader.seek(SeekFrom::Current(length as i64))?;

    // The magic that closes the data is checked here rather than when the tensor is read: a file
    // that is wrong should say so once, while it is being walked, and not once per layer.
    if reader.read_i16()? != DATA_MAGIC {
        return Err(Error::format(
            "tensor data did not end where it should have",
        ));
    }

    Ok((
        name,
        Record {
            source,
            offset,
            length: length as usize,
            shape,
            dtype,
        },
    ))
}

