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

//! A [`Graph`] as the instructions that run it.
//!
//! A graph says what a pass computes. An [`Ir`] says how it is run: a flat list of
//! [`Inst`]ructions, walked from the front, holding two kinds of thing. `%3 = matmul(%1, %2)`
//! runs one node of the graph and puts what it made away under its value; `free %1` says that
//! nothing is going to read `%1` again and lets go of it.
//!
//! ```text
//! ir {
//!   %0 = input("latent")
//!   %1 = load("conv_in.weight", [320, 4, 3, 3])
//!   %2 = conv2d(%0, %1, stride=1, padding=1, dilation=1, groups=1)
//!   free %0
//!   free %1
//!   %3 = silu(%2)
//!   free %2
//!   -> sample = %3
//! }
//! ```
//!
//! Freeing being an instruction rather than something a runner works out as it goes is the point
//! of compiling at all: where a value dies is decided once, by a walk of the graph, and is then
//! there to be read -- printed, checked in a test, counted -- instead of being a thing that only
//! happens. It also leaves the runner with nothing to decide, which is what will make it possible
//! to say later that a `free` gives a buffer back rather than merely dropping a handle.
//!
//! Those two decisions -- which nodes run, and where each value dies -- are the whole of what an
//! IR settles. The arithmetic is the same work in the same order through the same
//! [`functional`](super::functional) calls the eager code makes, which is what lets
//! [`Ir::run`] and the `forward` it stands for agree to the last bit rather than to a
//! tolerance.
//!
//! # Why freeing has to be said at all
//!
//! An eager pass gives memory back without being asked: an intermediate is a Rust local, and it
//! goes when the local does. A slot table does not. Something that walks a graph and keeps each
//! result under the value that produced it holds every intermediate of the whole pass at once,
//! which is worse than the code it replaces -- so the `free` instructions are not an
//! optimization, they are what makes running a graph cost what running the pass costs.
//!
//! What they buy on top of that is the weights. A model's parameters are `load` instructions
//! like any other, read out of the package the model was written to, and freed at the
//! instruction that read one last. So the two ends of a weight's life are both written down, and
//! a run holds no more of a model at once than the pass is actually using -- which is what a
//! model too large for the device it runs on needs and what an eager pass, whose weights are all
//! read before the first of them is used, cannot say.
//!
//! # Where a weight comes from, and when
//!
//! [`Residency`] decides, and it decides it in the [`ParamSource`] rather than in the IR:
//!
//! * [`Residency::Device`] brings every weight across when the source is built -- see
//!   [`resident`] -- and a `load` stays the node it was, answered by a lookup. What the card
//!   holds is the model, which is what a model that fits wants and what twenty sampler steps
//!   would otherwise pay for twenty times over.
//! * [`Residency::LowVram`] brings nothing across early. There a `load` compiles to three
//!   instructions -- [`Inst::Read`] onto the host, [`Inst::Upload`] to the device, and the frees
//!   that go with them -- written into the pass in front of the instruction that reads the
//!   weight. A weight does not arrive in one step, and the two copies are finished with at
//!   different times: the host copy is dead the moment the move has run, the device copy lives
//!   until the last operator that reads it, and one instruction could not say both. So the card
//!   holds what the pass is using rather than the model, and the host holds one weight rather
//!   than the package.
//!
//! There used to be a third thing in between: the kept weights were compiled into a prologue of
//! their own, run once into a table, and every pass began by copying that table into its slots.
//! It bought nothing a lookup does not -- the same bytes on the same card at the same time -- and
//! it cost a second instruction list, a type to carry the table, and a rule about which values a
//! pass was allowed to free. A load that is not an instruction needs none of that.
//!
//! # What is deliberately not here
//!
//! *Shapes.* An IR is compiled from a graph alone, so it does not know what any value is the
//! shape of, and cannot say how much memory a run holds at once. A wrongly built graph is still
//! found out where it runs -- but by then the instruction knows which line built it, which is
//! what [`Graph::site`] was paid for.
//!
//! *Storage, as opposed to values.* `view`, `transpose`, `slice`, `subtensor`, `squeeze` and
//! `unsqueeze` share their input's storage, so `free` does not mean bytes going back: a view that
//! outlives the value it was taken from keeps those bytes, and it is [`Tensor`]'s own reference
//! count and not this that gets that right. Which is enough while a `free` only drops a handle.
//! The day it wants to report how much is held, or to hand an instruction a buffer to write into,
//! it will have to know which values share storage, and that is a thing to add here when there is
//! something that reads it.
//!
//! *Rewriting.* Nodes run in the order the graph holds them, one kernel each. Fusing two of them
//! needs a kernel that does both, and where there is no such kernel there is nothing for a pass
//! to rewrite into.

use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use super::{functional as F, Bound, DType, Device, Tensor};
use crate::error::{Error, Result};
use crate::param_file::ParamFile;

use super::graph::{Graph, Site};
use super::op::{Binary, Extent, Op, Reduce, Scalar, Unary, Value};

/// One instruction of an [`Ir`].
#[derive(Clone, Debug)]
pub enum Inst {
    /// Run one node of the graph, and keep what it produces under `value`.
    Run {
        value: Value,
        op: Op,
        /// Where the node was built, which is what an error about this instruction says.
        site: Site,
    },
    /// Read the weight `name` out of the package, onto the host.
    ///
    /// The first of the three hops a weight makes, and the one that touches the disk: what comes
    /// back is a tensor of ordinary host memory, or of the page-locked kind when `pinned`, in the
    /// element type the package wrote.
    ///
    /// `pinned` is not a property of the weight but of what is about to happen to it. A weight
    /// that crosses to the device once and stays there is read into ordinary memory, because
    /// page-locking it would hold a second copy of it for the length of one move. A weight that
    /// crosses on every pass is read into the page-locked kind, because the driver reads that
    /// directly where it has to stage a pageable source through a bounce buffer of its own.
    Read {
        value: Value,
        name: String,
        shape: Vec<i32>,
        pinned: bool,
        /// Where the load was built, which is what an error about this instruction says. The same
        /// thing a `Run`'s site is, and here for the same reason: a package that does not hold
        /// what a model asks for should name the line that asked.
        site: Site,
    },
    /// Let `value` stand for `source`, which is the same weight under another node.
    ///
    /// A graph may name one weight from more than one place -- a tied embedding, a block written
    /// twice -- and each of those is a `load` node of its own with a value of its own. Bringing
    /// each across would put one weight on the card once per node that named it, so the first is
    /// brought across and the rest are made to stand for it, right where it arrives.
    ///
    /// Two slots then hold handles on one tensor, which is what a tensor is built to allow: the
    /// storage goes when the last of them does, so the frees the compiler already emits for each
    /// of them are exactly right and none of them had to know about the others.
    Alias { value: Value, source: Value },
    /// Move `source` from the host onto the device this run is for.
    ///
    /// The second hop, and the one that costs the bus. Split from the read above rather than done
    /// with it because the two ends have different lifetimes: the host copy is dead the moment
    /// this instruction has run, and the device copy lives until the last operator that reads it.
    /// Written as two instructions, those are two `free`s in two places, which is the whole of why
    /// a package larger than the card can be run at all.
    Upload { value: Value, source: Value },
    /// Let go of `value`, which nothing left to run is going to read.
    ///
    /// Always after the instruction that read it last, so a value is alive from the `Run` that
    /// made it up to and including the one that freed it. A value the graph names as an output is
    /// never freed, since the run itself is the thing still to read it.
    ///
    /// This may free what the instruction just before it produced, for a node that runs without
    /// anything reading its result. Only two kinds do: an input, which the caller hands over
    /// whether or not the pass wants it, and a draw, which moves the device's generator on.
    Free { value: Value },
}

impl fmt::Display for Inst {
    /// As the line an IR prints for it. The alternate form, `{:#}`, writes where a `Run`'s node
    /// came from after it, the way `{:#}` on a [`Graph`] does.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Inst::Run { value, op, site } => {
                write!(f, "{value} = {op}")?;
                if f.alternate() {
                    write!(f, "  ; {site}")?;
                }

                Ok(())
            }
            Inst::Read {
                value,
                name,
                shape,
                pinned,
                ..
            } => {
                let where_to = if *pinned { "pinned" } else { "host" };
                write!(f, "{value} = read {name:?} {shape:?} -> {where_to}")
            }
            Inst::Alias { value, source } => write!(f, "{value} = alias {source}"),
            Inst::Upload { value, source } => write!(f, "{value} = upload {source}"),
            Inst::Free { value } => write!(f, "free {value}"),
        }
    }
}

/// Where a [`load`](Op::Load) gets its weight, and what kind of memory it comes to rest in.
///
/// A load is not one thing happening but three: the package is read, the bytes come to rest on the
/// host, and that host copy crosses to the device. Only the first of them is asked of a source.
/// The other two are instructions -- [`Inst::Read`] is this call, [`Inst::Upload`] is the move --
/// and where each of them sits is the [`Ir`]'s to decide and not this trait's to guess.
///
/// A weight is asked for by the name the package holds it under, because that is all a graph
/// knows: there is no build step that could have handed out anything shorter.
pub trait ParamSource {
    /// The weight called `name`, on the host, which the caller expects to have `shape`.
    ///
    /// `pinned` asks for page-locked host memory rather than ordinary memory. It is a request
    /// about what is about to happen to the weight and not about the weight: the driver's copy
    /// engine reads page-locked memory directly where it has to stage a pageable source through a
    /// bounce buffer of its own first, so a weight that crosses the bus on every pass asks for it
    /// and one that crosses once does not.
    ///
    /// A source with nothing to page-lock -- anything that is not a package bound for a CUDA
    /// device -- reads into ordinary memory instead. There being nothing else for it to be given.
    fn read(&self, name: &str, shape: &[i32], pinned: bool) -> Result<Tensor>;

    /// The weight called `name`, ready for the graph to read on the device the run is going to.
    ///
    /// What an [`Op::Load`] is answered out of when the weights are kept, where [`read`] above is
    /// the first hop of a streamed one. The difference is where the tensor comes to rest: this
    /// hands back something the operators can read, and `read` hands back a host copy that an
    /// [`Inst::Upload`] still has to move.
    ///
    /// Kept weights are therefore a lookup and not a read -- the package was walked once when the
    /// source was built -- which is the whole of why [`Residency::Device`] needs no instructions
    /// of its own.
    ///
    /// [`read`]: ParamSource::read
    fn load(&self, name: &str, shape: &[i32]) -> Result<Tensor>;

    /// The device a weight goes to once it has been read, which is the one the model runs on.
    fn device(&self) -> Device;

    /// Where this model's weights wait between the instructions that read them.
    ///
    /// Read when the graph is compiled rather than when it is run: what it decides is whether a
    /// load is a lookup in a source that already holds the weight, or the three instructions that
    /// fetch one, which run once per step. See [`Ir::compile`].
    fn residency(&self) -> Residency;

    /// What shape the weight called `name` is, if there is one here.
    ///
    /// For a model that reads its own architecture out of the package rather than out of a
    /// configuration -- Anima's autoencoder counts its stages by looking for them -- which has to
    /// ask before it knows what it is asking for. Answered without handing the weight over, for
    /// the same reason `check` is: reading a weight is a read of the disk, and finding out how big
    /// one is should not be.
    fn shape_of(&self, name: &str) -> Option<Vec<i32>>;

    /// Whether there is a weight called `name` here at all, whatever shape it turns out to be.
    ///
    /// For the half of a model a package may or may not hold -- SDXL's VAE encoder is one -- where
    /// the question has to be answered before there is anything to ask it about a shape.
    fn has(&self, name: &str) -> bool {
        self.shape_of(name).is_some()
    }

    /// Whether `name` is here with `shape`, answered without handing the weight over.
    ///
    /// Apart from `read` because for a source where reading costs something the two are not the
    /// same question at all: checking a model's eight hundred weights by reading each one and
    /// dropping it would pull the whole package off the disk to learn nothing but that the package
    /// is the right one. See [`check_parameters`], which is the only caller.
    fn check(&self, name: &str, shape: &[i32]) -> Result<()>;
}

impl ParamSource for HashMap<String, Tensor> {
    /// A state dict that is already in memory, which hands over another handle on what it holds
    /// rather than reading anything. Nothing in it is page-locked, and nothing here pretends it
    /// is: what `pinned` buys is a faster copy, and a source that cannot offer one says so by
    /// handing back what it has.
    fn read(&self, name: &str, shape: &[i32], _pinned: bool) -> Result<Tensor> {
        Ok(held(self, name, shape)?.clone())
    }

    /// The same lookup, which is what a kept weight's load is: whoever built this map put the
    /// tensors where the run wants them, and a handle on one costs no bytes.
    fn load(&self, name: &str, shape: &[i32]) -> Result<Tensor> {
        Ok(held(self, name, shape)?.clone())
    }

    /// Wherever its tensors already are, which whoever built it decided. An empty one has nothing
    /// to be asked about and answers for the host.
    fn device(&self) -> Device {
        self.values()
            .next()
            .map_or(Device::Cpu, |tensor| tensor.device())
    }

    /// Kept, which is the only thing a map in memory can be: there is no package behind it to read
    /// a weight out of a second time.
    fn residency(&self) -> Residency {
        Residency::Device
    }

    fn shape_of(&self, name: &str) -> Option<Vec<i32>> {
        self.get(name).map(|tensor| tensor.shape())
    }

    fn has(&self, name: &str) -> bool {
        self.contains_key(name)
    }

    fn check(&self, name: &str, shape: &[i32]) -> Result<()> {
        held(self, name, shape).map(drop)
    }
}

/// The tensor a state dict holds under `name`, or what is wrong with asking for it.
///
/// Borrowed rather than cloned, so that taking a handle stays the caller's decision: `check` wants
/// the two questions asked and nothing else to happen.
fn held<'a>(weights: &'a HashMap<String, Tensor>, name: &str, shape: &[i32]) -> Result<&'a Tensor> {
    let tensor = weights
        .get(name)
        .ok_or_else(|| Error::model(format!("tensor {name:?} not found in model")))?;

    if tensor.shape() != shape {
        return Err(Error::model(format!(
            "tensor {name:?} has shape {:?}, expected {shape:?}",
            tensor.shape()
        )));
    }

    Ok(tensor)
}

/// Everything a run is given: where the weights come from, and what the inputs stand for.
///
/// One type rather than two arguments because the two are one call's worth of thing, and are
/// handed on together -- [`Ir::run`] passes the whole of it down to every instruction, so a run
/// that turns out to need something else later, somewhere to hand a buffer back or something to
/// report its time to, is a field here rather than another argument at every call site in the
/// library.
///
/// It is built one input at a time and stood in a local of its own, so that a call reads as the
/// two things it is -- what the run is given, and then the run:
///
/// ```
/// # use waifu::flint::{Graph, Ir, Residency, RunContext, Tensor};
/// # use std::collections::HashMap;
/// # let g = Graph::new();
/// # let x = g.input("hidden");
/// # g.output("hidden", g.silu(x));
/// # let ir = Ir::compile(&g, Residency::Device);
/// # let weights = HashMap::new();
/// # let hidden = Tensor::from_f32(&[1, 2], &[1.0, -1.0])?;
/// let context = RunContext::new(&weights).input("hidden", &hidden);
/// let outputs = ir.run(&context)?;
/// # Ok::<(), waifu::Error>(())
/// ```
///
/// It holds the names and borrows the tensors, so the one thing it costs is the list itself --
/// five pairs for the U-Net, against the pass they are handed to.
pub struct RunContext<'a> {
    params: &'a dyn ParamSource,
    inputs: Vec<(&'a str, &'a Tensor)>,
}

impl<'a> RunContext<'a> {
    /// A run that loads every [`Op::Load`]'s weight from `params`, with nothing kept and no
    /// inputs yet.
    pub fn new(params: &'a dyn ParamSource) -> Self {
        RunContext {
            params,
            inputs: Vec::new(),
        }
    }

    /// The tensor the graph's input called `name` stands for.
    ///
    /// One call for each of [`Ir::inputs`], in any order; [`Ir::run`] refuses a context that
    /// names anything else, or that leaves one out.
    pub fn input(mut self, name: &'a str, tensor: &'a Tensor) -> Self {
        self.inputs.push((name, tensor));
        self
    }

    /// Where every [`Op::Load`] reads its weight from.
    pub fn params(&self) -> &'a dyn ParamSource {
        self.params
    }

    /// The inputs, as the caller named them.
    pub fn inputs(&self) -> &[(&'a str, &'a Tensor)] {
        &self.inputs
    }

    /// The tensor given for the input called `name`, if one was.
    pub fn tensor(&self, name: &str) -> Option<&'a Tensor> {
        self.inputs
            .iter()
            .find(|(given, _)| *given == name)
            .map(|(_, tensor)| *tensor)
    }
}

/// A model's weights: the package they are in, and where they are going.
///
/// What a [`load`](Op::Load) is answered out of, and it holds no weight. The package is mapped
/// rather than read, so making one costs the headers and nothing else; every byte after that is
/// read at the instruction that asked for it and let go of at the instruction that finished with
/// it.
///
/// That is the difference from what this replaced. The older form read the whole package onto the
/// device when the model was built, which is what a model that fits wants and what a model that
/// does not fit cannot do at all; a second form page-locked the whole package on the host, which
/// fits but reserves the model's size in memory no other process may have. Neither decision is
/// made here any more. Both are the [`Ir`]'s, one weight at a time, and what is left here is the
/// two questions an instruction cannot answer for itself: what the package holds, and where this
/// model is running.
pub struct Weights {
    file: ParamFile,
    device: Device,
    residency: Residency,
    /// What the device computes in. Worked out once because `retype` is asked about every weight
    /// read and the answer cannot change within a model.
    computes_in: DType,
}

impl Weights {
    /// The weights of `file`, for a model running on `device` with its loads compiled the way
    /// `residency` says.
    ///
    /// Refuses the pair that cannot be built rather than quietly running as the other one: a
    /// low-vram run waits in page-locked host memory, which is CUDA's to hand out.
    pub fn new(file: ParamFile, device: Device, residency: Residency) -> Result<Weights> {
        if !residency.works_on(device) {
            return Err(Error::model(format!(
                "a low-vram run waits in page-locked host memory, which is cuda's to hand out, \
                 so it has nothing to offer {}",
                device.name()
            )));
        }

        Ok(Weights {
            computes_in: F::default_float_type(device)?,
            file,
            device,
            residency,
        })
    }

    /// How many weights the package holds.
    pub fn len(&self) -> usize {
        self.file.len()
    }

    pub fn is_empty(&self) -> bool {
        self.file.is_empty()
    }
}

impl ParamSource for Weights {
    /// One weight, off the disk and onto the host.
    ///
    /// Three things happen, and they are here in this order:
    ///
    /// 1. storage is allocated, in ordinary host memory or the page-locked kind, in the element
    ///    type the exporter wrote;
    /// 2. [`ParamFile::read_into`] fills it, which is a read of the mapping straight into those
    ///    bytes rather than into a buffer that would then have to be copied into them;
    /// 3. `retype` changes the element type, in the one case a device cannot take it.
    ///
    /// The last is where anything is decided at all, and doing it here rather than after the move
    /// is what keeps it cheap: a conversion happens on a tensor that has not been sent anywhere.
    fn read(&self, name: &str, shape: &[i32], pinned: bool) -> Result<Tensor> {
        // Page-locked memory is the CUDA driver's to hand out, so a run going anywhere else reads
        // into ordinary memory whatever it was asked for. Saying so here rather than refusing is
        // right because `pinned` is an optimization: what it changes is the speed of the move
        // below it, and a device with no such move loses nothing by ignoring it.
        let into = if pinned && self.device == Device::Cuda {
            Device::CudaHost
        } else {
            Device::Cpu
        };

        let dtype = self.file.dtype_of(name).ok_or_else(|| {
            Error::model(format!("tensor {name:?} not found in model"))
        })?;

        let mut tensor = Tensor::empty(shape, dtype, into)?;
        self.file.read_into(name, &mut tensor)?;

        retype(tensor, self.computes_in)
    }

    /// The same read, and then the hop a streamed weight makes as its own instruction.
    ///
    /// Not what a low-vram run uses: there the two hops are written into the pass so that the
    /// host copy can be freed between them, which is the point of writing them apart. This is
    /// here so that the trait is whole -- a package can answer a kept load as well as a streamed
    /// one -- and so that a caller compiling for [`Residency::Device`] against a package rather
    /// than against a map gets the weight it asked for rather than an error.
    fn load(&self, name: &str, shape: &[i32]) -> Result<Tensor> {
        Ok(self.read(name, shape, false)?.to_device(self.device)?)
    }

    fn device(&self) -> Device {
        self.device
    }

    fn residency(&self) -> Residency {
        self.residency
    }

    /// Off the header, so asking moves nothing and touches no disk.
    fn shape_of(&self, name: &str) -> Option<Vec<i32>> {
        self.file.shape_of(name)
    }

    fn has(&self, name: &str) -> bool {
        self.file.has(name)
    }

    /// Asked of the header too, which is the reason `check` is not `read` with its result
    /// dropped: checking a model this way reads none of it.
    fn check(&self, name: &str, shape: &[i32]) -> Result<()> {
        match self.file.shape_of(name) {
            None => Err(Error::model(format!("tensor {name:?} not found in model"))),
            Some(found) if found != shape => Err(Error::model(format!(
                "tensor {name:?} has shape {found:?}, expected {shape:?}"
            ))),
            Some(_) => Ok(()),
        }
    }
}

impl fmt::Debug for Weights {
    /// How much is in the package and where it is going, rather than every weight: a model holds
    /// hundreds of them and they are named in the IR that reads them.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Weights")
            .field("tensors", &self.file.len())
            .field("device", &self.device)
            .field("residency", &self.residency)
            .finish()
    }
}

/// Every weight of `file`, on `device`, for a model that fits there and will keep them.
///
/// The package is walked once, here, and what comes back is a map an [`Op::Load`] is a lookup in.
/// That is the whole of what [`Residency::Device`] needs: no instruction brings a weight across
/// at run time, because they are all across before the first run.
///
/// Each weight makes the same three steps a streamed one does -- read, `retype`, move -- with the
/// conversion before the move, because it is cheaper on a tensor that has not been sent anywhere.
/// What the host holds afterwards is nothing: the read is a copy out of the mapping and the copy
/// goes as the device one is made.
///
/// A map rather than a table of slots, keyed the way the package keys it, so that a weight two
/// nodes name is one tensor with two handles on it and neither node had to know about the other.
pub fn resident(file: &ParamFile, device: Device) -> Result<HashMap<String, Tensor>> {
    let source = Weights::new(file.clone(), device, Residency::Device)?;

    let mut weights = HashMap::with_capacity(file.len());
    for name in file.names() {
        let shape = file
            .shape_of(name)
            .ok_or_else(|| Error::model(format!("tensor {name:?} not found in model")))?;

        weights.insert(name.to_string(), source.load(name, &shape)?);
    }

    Ok(weights)
}

/// Where a model's weights wait between the instructions that read them.
///
/// The one question a caller has to answer that the package cannot: both modes run the same IR
/// over the same weights and draw the same picture, and they differ only in what the card is
/// holding while they do it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Residency {
    /// On the device, moved once and kept. Fastest, and what a card the model fits on wants.
    #[default]
    Device,
    /// Nowhere, between passes. A weight is read out of the package into page-locked host memory
    /// at the [`Inst::Read`] that wants it, crosses the bus at the [`Inst::Upload`] after that,
    /// and both copies are given back at the `free` after its last reader. For a card the model
    /// does not fit on: what the card holds is what the pass is using rather than the model.
    LowVram,
}

impl Residency {
    /// The weights of `file`, for a model on `device` that will hold them the way this says.
    ///
    /// Behind an [`Rc`] because that is what the halves of a model share: SDXL is four passes
    /// built out of one package, and the point of opening it once is that they all read the same
    /// one afterwards.
    ///
    /// Nothing is read here whichever this is. What the two differ in is where the instructions
    /// that read them end up, which is [`Ir::compile`]'s to arrange.
    pub fn read(self, file: ParamFile, device: Device) -> Result<Rc<dyn ParamSource>> {
        match self {
            // Across now, once, and kept: what comes back answers a load by looking it up.
            Residency::Device => Ok(Rc::new(resident(&file, device)?)),
            // Nothing across yet. The instructions in each pass do it, one weight at a time.
            Residency::LowVram => Ok(Rc::new(Weights::new(file, device, self)?)),
        }
    }

    /// Whether a run on `device` can be had this way.
    ///
    /// [`Residency::LowVram`] waits in memory only CUDA hands out, so it is the one answer that
    /// depends on where the run is going. Nothing above offers the pair that cannot be built --
    /// the command line and both screens name the two together, as one device -- so this is the
    /// check that says so for a caller reaching the library directly.
    pub fn works_on(self, device: Device) -> bool {
        match self {
            Residency::Device => true,
            Residency::LowVram => device == Device::Cuda,
        }
    }
}

/// Say now whether `weights` can answer everything `graph` will ask for.
///
/// A graph names its weights and reads none of them, and a [`ParamSource`] is filled without
/// consulting any graph, so the two are only brought together by a `load` instruction running.
/// That would make a package that does not hold what a model asks for -- or holds it in another
/// shape -- something found out on the way to a picture rather than when the model was asked for.
///
/// This asks the same questions the first run would and throws the answers away. It is
/// [`ParamSource::check`] rather than a read per weight precisely so that it stays free to ask:
/// a read is a read of the disk, and a package is checked before any of it is wanted.
pub fn check_parameters(graph: &Graph, weights: &dyn ParamSource) -> Result<()> {
    for (_, op) in graph.nodes() {
        if let Op::Load { name, shape } = op {
            weights.check(&name, &shape)?;
        }
    }

    Ok(())
}

/// The one thing about a weight that reading it decides, and the one case it decides it in.
///
/// A per-channel vector -- a bias, or a normalization's scale -- has to be the type the activation
/// beside it is, because that is what the operators that read one require: a `layer_norm` or a
/// `group_norm` refuses a weight of another type outright, and a convolution refuses a bias of
/// one. The matrices are not like that. A GEMM takes a half weight against a float activation and
/// converts it as it packs, which every backend here says so in as many words, so a weight of rank
/// two or more is left exactly as the file wrote it -- and it has to be, since those are the
/// model: 6.964 GB of SDXL's 6.969, against 5 MB for the 1340 vectors.
///
/// So the case is a half weight on a device that cannot compute in half, and x64 is the whole of
/// it: it has no half kernels, so it reads a half package with float activations. A device that
/// computes in half leaves a half package alone, and a package written in float is left alone
/// everywhere -- which is what keeps the autoencoder, written in float32 while the rest of SDXL
/// is half, from being narrowed to meet a type it does not run in.
fn retype(tensor: Tensor, computes_in: DType) -> Result<Tensor> {
    if tensor.dtype() != DType::Float16 || computes_in == DType::Float16 || tensor.dim()? >= 2 {
        return Ok(tensor);
    }

    Ok(tensor.cast(computes_in)?)
}

/// A graph as the instructions that run it.
///
/// [`Ir::compile`] makes one and [`Ir::run`] runs it. An IR is worth keeping: SDXL walks the
/// same U-Net twenty or more times for one picture, and compiling is a walk of the graph that
/// does not have to happen again between steps.
///
/// # What its values mean
///
/// A [`Value`] in an IR is the one the graph handed out, not a renumbering of it, so a printed
/// IR and the printed graph it came from name the same node the same way and can be read side
/// by side. A run therefore keeps room for every value the graph has, including the ones no
/// instruction produces; that is one empty slot each, and it buys being able to compare the two.
#[derive(Clone, Debug)]
pub struct Ir {
    insts: Vec<Inst>,
    /// How many slots a run needs. Every value the graph had -- see the type documentation -- and
    /// after them the ones compiling handed out for the host copy of a streamed weight.
    width: usize,
    inputs: Vec<(String, Value)>,
    outputs: Vec<(String, Value)>,
}

impl Ir {
    /// Work out how to run `graph`, with its weights held the way `residency` says.
    ///
    /// Cannot fail: a graph already guarantees that every operand exists and comes before the
    /// node reading it, which is everything this needs of it.
    ///
    /// # What residency decides
    ///
    /// Whether a [`Op::Load`] is an instruction at all:
    ///
    /// * [`Residency::Device`] compiles it to nothing of its own. The weights were brought across
    ///   when the source was built -- see [`resident`] -- so the load stays the node it was and is
    ///   answered by a lookup, exactly as every model in this library read its weights before
    ///   there was a choice of residency.
    /// * [`Residency::LowVram`] compiles it to three instructions -- read the weight onto the
    ///   host, move it to the device, let the host copy go -- written into the pass, in front of
    ///   the instruction that reads it, with the device copy freed after the last one that does.
    ///   The read is into page-locked memory, which is what makes the move that follows a copy the
    ///   driver can make directly. So the card holds what the pass is using rather than the model,
    ///   and the host holds one weight rather than the package.
    ///
    /// One list either way. A pass is what it says it is, and nothing runs before it.
    pub fn compile(graph: &Graph, residency: Residency) -> Ir {
        // The whole graph, once. Every node here is either moved into an instruction below or
        // dropped as one nothing reads, and asking for them one at a time would copy each of them
        // a second time.
        let nodes = graph.nodes();
        let outputs = graph.outputs();

        let kept = residency == Residency::Device;

        // Which `load` brings a weight across, and which ones stand for it. Keyed by what a load
        // asks for rather than by name alone: two loads of one name in two shapes is a model that
        // is wrong about its own weights, and it should fail at the second read rather than
        // quietly be handed the first one's tensor.
        //
        // Only worth working out for a streamed load, where bringing one weight across twice in a
        // pass is a second trip over the bus. A kept load is a lookup, so two nodes naming one
        // weight are two handles on the tensor the source already holds and neither costs
        // anything to answer.
        let mut first_load: HashMap<(&str, &[i32]), Value> = HashMap::new();
        let mut repeats: HashMap<usize, Vec<Value>> = HashMap::new();
        let mut is_repeat = vec![false; nodes.len()];
        for (value, op) in &nodes {
            if let (false, Op::Load { name, shape }) = (kept, op) {
                match first_load.get(&(name.as_str(), shape.as_slice())) {
                    Some(brought) => {
                        repeats.entry(brought.slot()).or_default().push(*value);
                        is_repeat[value.slot()] = true;
                    }
                    None => {
                        first_load.insert((name.as_str(), shape.as_slice()), *value);
                    }
                }
            }
        }

        let live = live_values(&nodes, &outputs);
        let mut remaining = remaining_reads(&nodes, &live, &outputs);

        // Every value the graph handed out, and room after them for the ones this hands out: a
        // streamed weight needs a second slot for its host copy, which no node ever named.
        let mut width = nodes.len();

        let mut insts = Vec::new();

        for (value, op) in nodes {
            if !live[value.slot()] {
                continue;
            }

            // A streamed load is three instructions rather than one, and the only node that is.
            // A kept one is no special case at all: it falls through to the `Inst::Run` below and
            // is answered by a lookup in the source, which is what `resident` built.
            if let (false, Op::Load { name, shape }) = (kept, &op) {
                // A repeat was dealt with where the weight arrived, several instructions ago.
                if is_repeat[value.slot()] {
                    continue;
                }

                let staging = Value::at(width);
                width += 1;

                let hops = [
                    Inst::Read {
                        value: staging,
                        name: name.clone(),
                        shape: shape.clone(),
                        pinned: !kept,
                        site: graph.site(value),
                    },
                    Inst::Upload {
                        value,
                        source: staging,
                    },
                    // The host copy is dead the moment the move above has run, whichever list
                    // this is on. It is the device copy whose life the two modes differ over.
                    Inst::Free { value: staging },
                ];

                // The nodes that named the same weight, made to stand for it here rather than
                // where they asked: under streaming the tensor this aliases is freed at its own
                // last reader, which may come first.
                let standing_for = repeats.remove(&value.slot()).unwrap_or_default();
                let also = standing_for.iter().map(|repeat| Inst::Alias {
                    value: *repeat,
                    source: value,
                });

                insts.extend(hops);
                insts.extend(also);

                // A load reads no operand, so there is nothing to count down here. The free for
                // what it produced is emitted below by whichever instruction reads it last.
                continue;
            }

            // Reading an operand for the last time is what frees it. An operand named twice by
            // one node -- `g.add(x, x)` -- is counted twice and so falls to zero once.
            let operands = op.operands();
            insts.push(Inst::Run {
                value,
                op,
                site: graph.site(value),
            });

            for operand in operands {
                remaining[operand.slot()] -= 1;
                if remaining[operand.slot()] == 0 {
                    insts.push(Inst::Free { value: operand });
                }
            }

            // A node that ran for its effect rather than its result. See `Inst::Free`.
            if remaining[value.slot()] == 0 {
                insts.push(Inst::Free { value });
            }
        }

        Ir {
            insts,
            width,
            inputs: graph.inputs(),
            outputs,
        }
    }

    /// Run the instructions, and hand back what the graph named as its outputs.
    ///
    /// `context` holds a tensor for each of [`Ir::inputs`] and the [`ParamSource`] every load
    /// reads its weight from -- by lookup where the weights are kept, and through the
    /// [`Inst::Read`] in front of the reader where they are streamed. The outputs come back in
    /// the order [`Graph::output`] named them.
    ///
    /// # Errors
    ///
    /// If the inputs are not exactly the ones the IR asks for, or if an instruction fails -- in
    /// which case the error says which node it was and which line built it.
    pub fn run(&self, context: &RunContext<'_>) -> Result<Vec<(String, Tensor)>> {
        self.check_inputs(context.inputs())?;

        // Empty, every run. What a kept weight costs here is the handle the lookup hands back and
        // the free that follows its last reader; the storage is the source's, and it stays
        // because the source still holds it.
        let mut slots: Vec<Option<Tensor>> = vec![None; self.width];

        for inst in &self.insts {
            step(inst, &mut slots, context)?;
        }

        Ok(self
            .outputs
            .iter()
            .map(|(name, value)| {
                let tensor = slots[value.slot()]
                    .clone()
                    .expect("a value the graph names as an output is never freed");

                (name.clone(), tensor)
            })
            .collect())
    }

    /// The instructions, in the order they run.
    pub fn insts(&self) -> &[Inst] {
        &self.insts
    }

    /// How many instructions there are, counting the frees.
    pub fn len(&self) -> usize {
        self.insts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.insts.is_empty()
    }

    /// What the caller has to hand to [`Ir::run`], in the order the graph asked for it.
    ///
    /// Every input the graph asked for, including one the pass turns out not to read: what the
    /// caller has to supply is the graph's contract, and not a thing to have change under them
    /// because a node was dropped.
    pub fn inputs(&self) -> &[(String, Value)] {
        &self.inputs
    }

    /// What the IR is for, in the order the graph named it.
    pub fn outputs(&self) -> &[(String, Value)] {
        &self.outputs
    }

    /// Refuses anything but exactly the inputs the IR asks for.
    ///
    /// A name it does not know is as much a mistake as a name it is missing, and the more likely
    /// one: a caller that spells an input wrongly has handed over a tensor and left the one the
    /// IR wanted out, and only the second half of that would be noticed otherwise.
    fn check_inputs(&self, inputs: &[(&str, &Tensor)]) -> Result<()> {
        for (name, _) in &self.inputs {
            if !inputs.iter().any(|(given, _)| given == name) {
                return Err(Error::model(format!(
                    "this graph takes an input called {name:?}, which was not given"
                )));
            }
        }

        for (given, _) in inputs {
            if !self.inputs.iter().any(|(name, _)| name == given) {
                return Err(Error::model(format!(
                    "this graph takes no input called {given:?}"
                )));
            }
        }

        Ok(())
    }
}

impl fmt::Display for Ir {
    /// The instructions, one per line, and then what the run hands back.
    ///
    /// One block, because there is one list: what a run does is what it says, and a weight that
    /// was brought across before any of it shows up as the `load` it always was.
    ///
    /// The alternate form, `{:#}`, writes where each node was built after it, the way `{:#}` on a
    /// [`Graph`] does.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ir {{")?;
        for inst in &self.insts {
            match f.alternate() {
                true => writeln!(f, "  {inst:#}")?,
                false => writeln!(f, "  {inst}")?,
            }
        }

        for (name, value) in &self.outputs {
            writeln!(f, "  -> {name} = {value}")?;
        }

        write!(f, "}}")
    }
}

/// Says which node failed, and where it was built, in front of what went wrong.
///
/// The whole of what [`Graph::site`] is for. Without it a failure in the middle of ten thousand
/// nodes reports the shape it did not like and nothing about which of the model's lines asked for
/// it.
fn blame(value: Value, op: &Op, site: &Site, error: Error) -> Error {
    Error::model(format!("{value} = {op}, built at {site}: {error}"))
}

/// The same for a read, which has a name and a shape where a node has an operation.
///
/// Written to read like the `load` node it was compiled from, since that is the thing whoever is
/// reading this wrote.
fn blame_read(name: &str, shape: &[i32], site: &Site, error: Error) -> Error {
    Error::model(format!("load({name:?}, {shape:?}), built at {site}: {error}"))
}

/// Run one instruction against the slots.
///
/// The whole of what a run does.
fn step(inst: &Inst, slots: &mut [Option<Tensor>], context: &RunContext<'_>) -> Result<()> {
    match inst {
        Inst::Run { value, op, site } => {
            let tensor = execute(op, slots, context)
                .map_err(|error| blame(*value, op, site, error))?;

            slots[value.slot()] = Some(tensor);
        }
        Inst::Read {
            value,
            name,
            shape,
            pinned,
            site,
        } => {
            let tensor = context
                .params()
                .read(name, shape, *pinned)
                .map_err(|error| blame_read(name, shape, site, error))?;

            slots[value.slot()] = Some(tensor);
        }
        Inst::Alias { value, source } => {
            slots[value.slot()] = Some(read(slots, *source).clone());
        }
        Inst::Upload { value, source } => {
            let host = read(slots, *source);
            slots[value.slot()] = Some(host.to_device(context.params().device())?);
        }
        Inst::Free { value } => slots[value.slot()] = None,
    }

    Ok(())
}

/// Run one operation, reading its operands out of the slots.
fn execute(op: &Op, slots: &[Option<Tensor>], context: &RunContext<'_>) -> Result<Tensor> {
    let get = |value: &Value| read(slots, *value);

    let tensor = match op {
        Op::Input { name } => context
            .tensor(name)
            .cloned()
            .expect("run checked that every input was given"),
        // Compiled into an `Inst::Read` and an `Inst::Upload` rather than run here, so that where
        // a weight comes to rest is something the IR says. `compile` puts no load in an
        // `Inst::Run`, which is why this is unreachable rather than a second way of doing it.
        // Answered by the source, which under [`Residency::Device`] is a lookup in the map
        // `resident` filled. A streamed load never reaches here: `compile` turns that one into a
        // read and an upload, so this is the kept case and nothing else.
        Op::Load { name, shape } => context.params().load(name, shape)?,
        Op::Constant { tensor } => tensor.clone(),

        Op::Zeros {
            shape,
            dtype,
            device,
        } => Tensor::zeros(&sizes(slots, shape)?, *dtype, *device)?,
        Op::Arange {
            begin,
            end,
            step,
            device,
        } => F::arange(*begin, *end, *step, *device)?,
        Op::Rand {
            shape,
            dtype,
            device,
        } => F::rand(&sizes(slots, shape)?, *dtype, *device)?,
        Op::Randn { shape, device } => F::randn(&sizes(slots, shape)?, *device)?,
        Op::CausalMask { max_len, device } => F::causal_mask(size(slots, *max_len)?, *device)?,

        Op::Unary { kind, input } => {
            let input = get(input);
            match kind {
                Unary::Neg => F::neg(input)?,
                Unary::Abs => F::abs(input)?,
                Unary::Exp => F::exp(input)?,
                Unary::Sqrt => F::sqrt(input)?,
                Unary::Rsqrt => F::rsqrt(input)?,
                Unary::Square => F::square(input)?,
                Unary::Sigmoid => F::sigmoid(input)?,
                Unary::Tanh => F::tanh(input)?,
                Unary::Relu => F::relu(input)?,
                Unary::Gelu => F::gelu(input)?,
                Unary::QuickGelu => F::quick_gelu(input)?,
                Unary::Silu => F::silu(input)?,
                Unary::Sin => F::sin(input)?,
                Unary::Cos => F::cos(input)?,
                Unary::Softmax => F::softmax(input)?,
                Unary::Swiglu => F::swiglu(input)?,
                Unary::Geglu => F::geglu(input)?,
            }
        }
        Op::Binary { kind, lhs, rhs } => {
            let (lhs, rhs) = (get(lhs), get(rhs));
            match kind {
                Binary::Add => F::add(lhs, rhs)?,
                Binary::Sub => F::sub(lhs, rhs)?,
                Binary::Mul => F::mul(lhs, rhs)?,
                Binary::Div => F::div(lhs, rhs)?,
                Binary::Eq => F::eq(lhs, rhs)?,
            }
        }
        Op::Scalar { kind, input, other } => {
            let input = get(input);
            match kind {
                Scalar::Mul => F::mul_scalar(input, *other)?,
                Scalar::Div => F::div_scalar(input, *other)?,
            }
        }
        Op::ModScalar { input, other } => F::mod_scalar(get(input), *other)?,
        Op::Reduce { kind, input, dim } => {
            let input = get(input);
            match kind {
                Reduce::Sum => F::sum(input, *dim)?,
                Reduce::Max => F::max(input, *dim)?,
                Reduce::Min => F::min(input, *dim)?,
            }
        }

        Op::LayerNorm {
            input,
            weight,
            bias,
            eps,
        } => F::layer_norm(
            get(input),
            weight.as_ref().map(get),
            bias.as_ref().map(get),
            *eps,
        )?,
        Op::RmsNorm { input, weight, eps } => F::rms_norm(get(input), get(weight), *eps)?,
        Op::GroupNorm {
            input,
            weight,
            bias,
            groups,
            eps,
        } => F::group_norm(
            get(input),
            weight.as_ref().map(get),
            bias.as_ref().map(get),
            *groups,
            *eps,
        )?,

        Op::Matmul { lhs, rhs } => F::matmul(get(lhs), get(rhs))?,
        Op::Fp8Matmul {
            lhs,
            weight,
            channel_scale,
        } => F::fp8_matmul(get(lhs), get(weight), get(channel_scale))?,
        Op::Lookup { table, indices } => F::lookup(get(table), get(indices))?,
        Op::Conv2d {
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        } => F::conv2d(
            get(input),
            get(weight),
            bias.as_ref().map(get),
            *stride,
            *padding,
            *dilation,
            *groups,
        )?,
        Op::UpsampleNearest2d { input, scale } => F::upsample_nearest2d(get(input), *scale)?,
        Op::Conv1d {
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        } => F::conv1d(
            get(input),
            get(weight),
            bias.as_ref().map(get),
            *stride,
            *padding,
            *dilation,
            *groups,
        )?,
        Op::ConvTranspose1d {
            input,
            weight,
            bias,
            stride,
            padding,
            output_padding,
            groups,
        } => F::conv_transpose1d(
            get(input),
            get(weight),
            bias.as_ref().map(get),
            *stride,
            *padding,
            *output_padding,
            *groups,
        )?,
        Op::Snake {
            input,
            alpha,
            beta,
            eps,
        } => F::snake(get(input), get(alpha), beta.as_ref().map(get), *eps)?,
        Op::Stft {
            input,
            window,
            n_fft,
            hop,
            centered,
        } => F::stft(get(input), get(window), *n_fft, *hop, *centered)?,
        Op::Istft {
            spectrum,
            window,
            n_fft,
            hop,
            centered,
        } => F::istft(get(spectrum), get(window), *n_fft, *hop, *centered)?,
        Op::Attention {
            query,
            key,
            value,
            causal,
        } => F::attention(get(query), get(key), get(value), *causal)?,
        Op::Cat { lhs, rhs, dim } => F::cat(get(lhs), get(rhs), *dim)?,

        Op::View { input, shape } => get(input).view(&sizes(slots, shape)?)?,
        Op::Transpose { input, dim0, dim1 } => get(input).transpose(*dim0, *dim1)?,
        Op::Slice {
            input,
            dim,
            begin,
            end,
        } => get(input).slice(*dim, bound(slots, *begin)?, bound(slots, *end)?)?,
        Op::Subtensor { input, index } => get(input).subtensor(*index)?,
        Op::Unsqueeze { input, dim } => get(input).unsqueeze(*dim)?,
        Op::Squeeze { input, dim } => get(input).squeeze(*dim)?,
        Op::Contiguous { input } => get(input).contiguous()?,
        Op::Cast { input, dtype } => get(input).cast(*dtype)?,
        Op::ToDevice { input, device } => get(input).to_device(*device)?,
    };

    Ok(tensor)
}

/// The size an [`Extent`] stands for, now that there are tensors to read it off.
fn size(slots: &[Option<Tensor>], extent: Extent) -> Result<i32> {
    match extent {
        Extent::At(size) => Ok(size),
        Extent::Of { value, dim } => Ok(read(slots, value).shape_at(dim)?),
        Extent::Prod { value, from, to } => {
            if from < 0 || to < from {
                return Err(Error::model(format!(
                    "{from}..{to} is not a range of dimensions to multiply together"
                )));
            }

            // An empty range is one element, which is what an empty product is and what a
            // reshape that folds no dimensions should get.
            let tensor = read(slots, value);
            (from..to).try_fold(1, |size, dim| Ok(size * tensor.shape_at(dim)?))
        }
        Extent::End => Err(Error::model(
            "an open end is a slice bound and not a size, and cannot stand in a shape",
        )),
    }
}

/// A whole shape, resolved.
fn sizes(slots: &[Option<Tensor>], shape: &[Extent]) -> Result<Vec<i32>> {
    shape.iter().map(|extent| size(slots, *extent)).collect()
}

/// The same for one end of a slice, where an open end is the thing an [`Extent`] cannot be a
/// size for and is exactly what is wanted.
fn bound(slots: &[Option<Tensor>], extent: Extent) -> Result<Bound> {
    match extent {
        Extent::End => Ok(Bound::End),
        extent => Ok(Bound::At(size(slots, extent)?)),
    }
}

/// The tensor a value stands for, which an IR has already made sure is there.
///
/// An IR frees a value after the instruction that reads it last and never before, so a slot that
/// is empty here is this module having got its own arithmetic wrong, not anything the caller did.
fn read(slots: &[Option<Tensor>], value: Value) -> &Tensor {
    slots[value.slot()]
        .as_ref()
        .unwrap_or_else(|| panic!("{value} was freed before the instruction that reads it"))
}

/// Which values a run has to compute: the outputs, whatever they are computed from, and the nodes
/// that run whether or not anything reads them.
///
/// One pass from the back is enough. Every operand is a node that was already there when the one
/// reading it was added, so by the time the walk reaches a node it has already been past
/// everything that could have read it.
fn live_values(nodes: &[(Value, Op)], outputs: &[(String, Value)]) -> Vec<bool> {
    let mut live = vec![false; nodes.len()];
    for (_, value) in outputs {
        live[value.slot()] = true;
    }

    for (value, op) in nodes.iter().rev() {
        if is_root(op) {
            live[value.slot()] = true;
        }

        if !live[value.slot()] {
            continue;
        }

        for operand in op.operands() {
            live[operand.slot()] = true;
        }
    }

    live
}

/// Whether a node runs even if nothing reads what it produces.
fn is_root(op: &Op) -> bool {
    match op {
        // The caller hands one of these over rather than the IR computing it, so there is
        // nothing to save by leaving it out, and `Ir::inputs` promises to ask for it either
        // way.
        Op::Input { .. } => true,

        // A draw moves the device's generator on. One that nothing reads is still a draw, and
        // dropping it would change every number drawn after it -- which is exactly the kind of
        // difference between an IR and the pass it stands for that must not exist.
        Op::Rand { .. } | Op::Randn { .. } => true,

        _ => false,
    }
}

/// How many times each value is still to be read once the run starts, counting only the nodes
/// that will run and counting each naming as an output as a read that never happens.
///
/// Not [`Graph::use_counts`], which counts every reader the graph has: a reader that is not going
/// to run does not keep its operand alive, and counting it would hold a tensor for the length of
/// the pass.
fn remaining_reads(
    nodes: &[(Value, Op)],
    live: &[bool],
    outputs: &[(String, Value)],
) -> Vec<usize> {
    let mut counts = vec![0; nodes.len()];
    for (value, op) in nodes {
        if !live[value.slot()] {
            continue;
        }

        for operand in op.operands() {
            counts[operand.slot()] += 1;
        }
    }

    // Never taken back, which is what keeps an output in its slot until the run hands it over.
    for (_, value) in outputs {
        counts[value.slot()] += 1;
    }

    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::cell::Cell;

    use crate::flint::Device;

    /// A state dict holding `tensors`, which is what a `load` reads out of.
    fn state_dict(tensors: &[(&str, &[i32], &[f32])]) -> HashMap<String, Tensor> {
        tensors
            .iter()
            .map(|(name, shape, values)| {
                (name.to_string(), Tensor::from_f32(shape, values).unwrap())
            })
            .collect()
    }

    /// A state dict that counts what it was asked, for telling the two questions apart.
    struct Counting {
        weights: HashMap<String, Tensor>,
        loads: Cell<usize>,
        checks: Cell<usize>,
    }

    impl Counting {
        fn over(weights: HashMap<String, Tensor>) -> Counting {
            Counting {
                weights,
                loads: Cell::new(0),
                checks: Cell::new(0),
            }
        }
    }

    impl ParamSource for Counting {
        fn read(&self, name: &str, shape: &[i32], pinned: bool) -> Result<Tensor> {
            self.loads.set(self.loads.get() + 1);
            self.weights.read(name, shape, pinned)
        }

        /// Counted the same way: a kept load is still a load, and what these tests weigh is how
        /// many times a weight is asked for.
        fn load(&self, name: &str, shape: &[i32]) -> Result<Tensor> {
            self.loads.set(self.loads.get() + 1);
            self.weights.load(name, shape)
        }

        fn device(&self) -> Device {
            self.weights.device()
        }

        fn residency(&self) -> Residency {
            Residency::Device
        }

        fn shape_of(&self, name: &str) -> Option<Vec<i32>> {
            self.weights.shape_of(name)
        }

        fn has(&self, name: &str) -> bool {
            self.weights.has(name)
        }

        fn check(&self, name: &str, shape: &[i32]) -> Result<()> {
            self.checks.set(self.checks.get() + 1);
            self.weights.check(name, shape)
        }
    }

    /// A parameter file holding nothing, for the checks that happen before one is read.
    ///
    /// A safetensors file with an empty header, which is the smallest one there is: eight bytes
    /// saying the header is two long, and then `{}`.
    fn empty_file() -> ParamFile {
        let mut bytes = 2u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        ParamFile::parse(&bytes).unwrap()
    }

    /// One transformer feed forward, which reads its input twice and so has a value that outlives
    /// the instruction after the one that made it.
    fn feed_forward() -> Graph {
        let g = Graph::new();
        let x = g.input("hidden");
        let weight = g.load("fc.weight", &[2, 2]);

        let transposed = g.transpose(weight, 0, 1);
        let up = g.matmul(x, transposed);
        let up = g.silu(up);
        let out = g.add(x, up);
        g.output("hidden", out);

        g
    }

    /// Compile, bring the kept weights across, and run: the three calls a caller makes, as one.
    ///
    fn compile_and_run(
        graph: &Graph,
        params: &dyn ParamSource,
        context: RunContext<'_>,
    ) -> Result<Vec<(String, Tensor)>> {
        Ir::compile(graph, params.residency()).run(&context)
    }

    /// The instructions as they are printed, which is the readable way to say what an IR is.
    fn listing(ir: &Ir) -> Vec<String> {
        ir.insts().iter().map(Inst::to_string).collect()
    }

    fn produced(insts: &[Inst]) -> Vec<Value> {
        insts
            .iter()
            .filter_map(|inst| match inst {
                Inst::Run { value, .. }
                | Inst::Read { value, .. }
                | Inst::Upload { value, .. }
                | Inst::Alias { value, .. } => Some(*value),
                Inst::Free { .. } => None,
            })
            .collect()
    }

    fn freed(insts: &[Inst]) -> Vec<Value> {
        insts
            .iter()
            .filter_map(|inst| match inst {
                Inst::Free { value } => Some(*value),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn runs_the_nodes_in_order_and_frees_each_value_where_it_dies() {
        let ir = Ir::compile(&feed_forward(), Residency::Device);

        // A kept weight is the `load` node it always was: no instruction brings it across,
        // because the source it is looked up in already holds it. The handle this pass takes on
        // it is freed at its last reader like any other value; the storage is the source's.
        assert_eq!(
            listing(&ir),
            [
                "%0 = input(\"hidden\")",
                "%1 = load(\"fc.weight\", [2, 2])",
                "%2 = transpose(%1, 0, 1)",
                "free %1",
                "%3 = matmul(%0, %2)",
                "free %2",
                "%4 = silu(%3)",
                "free %3",
                "%5 = add(%0, %4)",
                // The residual is where the input is finally done with, four instructions after
                // it was put in place.
                "free %0",
                "free %4",
            ]
        );
    }

    #[test]
    fn brings_a_streamed_weight_across_inside_the_pass_and_frees_both_copies() {
        let ir = Ir::compile(&feed_forward(), Residency::LowVram);

        assert_eq!(
            listing(&ir),
            [
                "%0 = input(\"hidden\")",
                // Page-locked, because this copy is about to cross the bus and will do so again
                // on every pass after this one.
                "%6 = read \"fc.weight\" [2, 2] -> pinned",
                "%1 = upload %6",
                "free %6",
                "%2 = transpose(%1, 0, 1)",
                // And the device copy goes at its last reader, exactly as any other value does.
                "free %1",
                "%3 = matmul(%0, %2)",
                "free %2",
                "%4 = silu(%3)",
                "free %3",
                "%5 = add(%0, %4)",
                "free %0",
                "free %4",
            ]
        );
    }

    #[test]
    fn runs_every_node_the_graph_holds_and_asks_for_what_it_asked_for() {
        let g = feed_forward();
        let ir = Ir::compile(&g, Residency::Device);

        // Every node the graph had, loads included: a kept load is an instruction like the rest.
        assert_eq!(produced(ir.insts()).len(), g.len());
        assert_eq!(ir.inputs(), g.inputs());
        assert_eq!(ir.outputs(), g.outputs());
    }

    #[test]
    fn never_frees_what_the_graph_names_as_an_output() {
        let ir = Ir::compile(&feed_forward(), Residency::Device);
        let output = ir.outputs()[0].1;

        assert!(!freed(ir.insts()).contains(&output));
    }

    #[test]
    fn frees_a_value_read_twice_by_one_node_once() {
        let g = Graph::new();
        let x = g.input("x");
        let doubled = g.add(x, x);
        g.output("x", doubled);

        let ir = Ir::compile(&g, Residency::Device);

        assert_eq!(
            listing(&ir),
            ["%0 = input(\"x\")", "%1 = add(%0, %0)", "free %0"]
        );
    }

    #[test]
    fn leaves_out_a_node_nothing_will_read() {
        let g = Graph::new();
        let x = g.input("x");
        let wanted = g.silu(x);
        let spare = g.gelu(x);
        g.output("y", wanted);

        let ir = Ir::compile(&g, Residency::Device);

        assert_eq!(g.len(), 3);
        assert!(!produced(ir.insts()).contains(&spare));
        // Dropping the second reader is also what lets the input go a step earlier.
        assert_eq!(
            listing(&ir),
            ["%0 = input(\"x\")", "%1 = silu(%0)", "free %0"]
        );
    }

    #[test]
    fn keeps_a_draw_nothing_reads_because_it_moves_the_generator_on() {
        let g = Graph::new();
        let x = g.input("x");
        let noise = g.randn([2, 2], Device::Cpu);
        let y = g.silu(x);
        g.output("y", y);

        let ir = Ir::compile(&g, Residency::Device);

        assert_eq!(produced(ir.insts()), vec![x, noise, y]);
        // Drawn, and then let go of at once: nothing is going to read it.
        assert_eq!(listing(&ir)[2], "free %1");
    }

    #[test]
    fn asks_for_an_input_the_pass_turns_out_not_to_read() {
        let g = Graph::new();
        let x = g.input("x");
        let spare = g.input("spare");
        g.output("x", x);

        let ir = Ir::compile(&g, Residency::Device);

        assert_eq!(ir.inputs().len(), 2);
        assert_eq!(produced(ir.insts()), vec![x, spare]);
        assert_eq!(freed(ir.insts()), vec![spare]);
    }

    #[test]
    fn reads_a_size_off_a_value_when_it_runs() {
        let g = Graph::new();
        let x = g.input("x");
        let table = g.input("table");

        // As many rows of the table as `x` is long, whatever that turns out to be.
        let rows = g.slice(table, 0, 0, Extent::of(x, 0));
        g.output("rows", rows);

        // The value a size is read off is a value the node reads, so it is an operand and is not
        // let go of before the instruction that reads it.
        let plan = listing(&Ir::compile(&g, Residency::Device));
        assert_eq!(
            plan,
            [
                "%0 = input(\"x\")",
                "%1 = input(\"table\")",
                "%2 = slice(%1, dim=0, 0..dim(%0, 0))",
                "free %1",
                "free %0",
            ]
        );

        let table = Tensor::from_f32(&[4, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let two = Tensor::from_i64(&[2], &[0, 0]).unwrap();
        let params = state_dict(&[]);
        let context = RunContext::new(&params)
            .input("x", &two)
            .input("table", &table);
        let outputs = compile_and_run(&g, &params, context).unwrap();

        assert_eq!(outputs[0].1.to_vec_f32().unwrap(), [1.0, 2.0, 3.0, 4.0]);

        // The same instructions again, and a different answer, which is the whole point.
        let three = Tensor::from_i64(&[3], &[0, 0, 0]).unwrap();
        let context = RunContext::new(&params)
            .input("x", &three)
            .input("table", &table);
        let outputs = compile_and_run(&g, &params, context).unwrap();

        assert_eq!(
            outputs[0].1.to_vec_f32().unwrap(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn reshapes_to_a_length_it_was_not_told() {
        let g = Graph::new();
        let x = g.input("x");
        g.output(
            "x",
            g.view(x, [Extent::At(1), Extent::of(x, 0), Extent::At(1)]),
        );

        let ir = Ir::compile(&g, Residency::Device);
        let params = state_dict(&[]);
        for length in [2, 6] {
            let x = Tensor::from_f32(&[length], &vec![1.0; length as usize]).unwrap();
            let context = RunContext::new(&params).input("x", &x);
            let outputs = ir.run(&context).unwrap();

            assert_eq!(outputs[0].1.shape(), [1, length, 1]);
        }
    }

    #[test]
    fn computes_what_the_eager_pass_computes() {
        let hidden = Tensor::from_f32(&[2, 2], &[1.0, -2.0, 0.5, 3.0]).unwrap();
        let params = state_dict(&[("fc.weight", &[2, 2], &[0.25, -1.5, 2.0, 0.75])]);
        let weight = params["fc.weight"].clone();

        let context = RunContext::new(&params).input("hidden", &hidden);
        let outputs = compile_and_run(&feed_forward(), &params, context).unwrap();

        // The same calls in the same order, written out. Nothing was reassociated, so this is an
        // equality and not a tolerance.
        let transposed = weight.transpose(0, 1).unwrap();
        let up = F::silu(&F::matmul(&hidden, &transposed).unwrap()).unwrap();
        let eager = F::add(&hidden, &up).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].0, "hidden");
        assert_eq!(
            outputs[0].1.to_vec_f32().unwrap(),
            eager.to_vec_f32().unwrap()
        );
    }

    #[test]
    fn hands_back_every_output_under_the_name_it_was_given() {
        let g = Graph::new();
        let x = g.input("hidden");
        let pooled = g.sum(x, -1);
        g.output("hidden", x);
        g.output("pooled", pooled);

        let hidden = Tensor::from_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let params = state_dict(&[]);
        let context = RunContext::new(&params).input("hidden", &hidden);

        let outputs = compile_and_run(&g, &params, context).unwrap();

        let names: Vec<&str> = outputs.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["hidden", "pooled"]);
        assert_eq!(outputs[1].1.to_vec_f32().unwrap(), [3.0, 7.0]);
    }

    #[test]
    fn refuses_an_input_it_did_not_ask_for_and_one_it_is_not_given() {
        let g = Graph::new();
        let x = g.input("hidden");
        g.output("hidden", x);

        let ir = Ir::compile(&g, Residency::Device);
        let params = state_dict(&[]);
        let tensor = Tensor::from_f32(&[1], &[1.0]).unwrap();

        let missing = ir
            .run(&RunContext::new(&params))
            .unwrap_err()
            .to_string();
        assert!(
            missing.contains("\"hidden\", which was not given"),
            "{missing}"
        );

        let stray = ir
            .run(
                &RunContext::new(&params)
                    .input("hidden", &tensor)
                    .input("hidde", &tensor),
            )
            .unwrap_err()
            .to_string();
        assert!(stray.contains("no input called \"hidde\""), "{stray}");
    }

    #[test]
    fn says_which_node_failed_and_where_it_was_built() {
        let g = Graph::new();
        let line = line!() + 1;
        let weight = g.load("fc.weight", &[2, 2]);
        g.output("weight", weight);

        // A state dict that does not hold what the graph asks for, which is a failure that comes
        // back rather than one the library aborts on. It comes back from the run now rather than
        // from a load of its own, and says the same thing: a kept load is an instruction in the
        // pass, so what blames it is what blames any other.
        let error = Ir::compile(&g, Residency::Device)
            .run(&RunContext::new(&state_dict(&[])))
            .unwrap_err()
            .to_string();

        // The value it would have produced is named too, which a read could not say: the node is
        // the instruction now, so the blame is the one every other node gets.
        assert!(
            error.contains("load(\"fc.weight\", [2, 2]), built at "),
            "{error}"
        );
        assert!(error.contains(&format!("ir.rs:{line}")), "{error}");
        assert!(error.contains("not found in model"), "{error}");
    }

    #[test]
    fn loads_a_weight_under_the_name_the_namespace_makes_of_it() {
        let g = Graph::new();
        let x = g.input("hidden");
        let attn = g.subgraph("down1").subgraph("attn");
        let weight = attn.load("to_q.weight", &[2, 2]);
        let scaled = attn.matmul(x, weight);

        // The handle that was never narrowed still names the root, whatever the one beside it is
        // doing.
        let bias = g.load("out.bias", &[2, 2]);
        let out = g.add(scaled, bias);
        g.output("hidden", out);

        assert_eq!(g.parameters(), vec!["down1.attn.to_q.weight", "out.bias"]);

        let hidden = Tensor::from_f32(&[2, 2], &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let params = state_dict(&[
            ("down1.attn.to_q.weight", &[2, 2], &[1.0, 2.0, 3.0, 4.0]),
            ("out.bias", &[2, 2], &[10.0, 10.0, 10.0, 10.0]),
        ]);

        let outputs = compile_and_run(
            &g,
            &params,
            RunContext::new(&params).input("hidden", &hidden),
        )
        .unwrap();
        assert_eq!(outputs[0].1.to_vec_f32().unwrap(), [11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn refuses_a_weight_the_state_dict_holds_in_another_shape() {
        let g = Graph::new();
        let weight = g.load("fc.weight", &[2, 2]);
        g.output("weight", weight);

        let params = state_dict(&[("fc.weight", &[4], &[1.0, 2.0, 3.0, 4.0])]);
        let error = Ir::compile(&g, Residency::Device)
            .run(&RunContext::new(&params))
            .unwrap_err()
            .to_string();

        assert!(error.contains("has shape [4], expected [2, 2]"), "{error}");
    }

    #[test]
    fn prints_the_instructions_and_then_what_the_run_hands_back() {
        let g = Graph::new();
        let x = g.input("latent");
        let weight = g.load("conv_in.weight", &[320, 4, 3, 3]);

        let x = g.conv2d(x, weight, None, 1, 1, 1, 1);
        let x = g.silu(x);
        g.output("sample", x);

        assert_eq!(
            Ir::compile(&g, Residency::Device).to_string(),
            concat!(
                "ir {\n",
                "  %0 = input(\"latent\")\n",
                "  %1 = load(\"conv_in.weight\", [320, 4, 3, 3])\n",
                "  %2 = conv2d(%0, %1, stride=1, padding=1, dilation=1, groups=1)\n",
                "  free %0\n",
                "  free %1\n",
                "  %3 = silu(%2)\n",
                "  free %2\n",
                "  -> sample = %3\n",
                "}"
            )
        );
    }

    #[test]
    fn prints_where_a_node_was_built_only_when_asked() {
        let ir = Ir::compile(&feed_forward(), Residency::Device);

        assert!(!ir.to_string().contains("ir.rs:"));

        let annotated = format!("{ir:#}");
        assert!(
            annotated.contains("%2 = transpose(%1, 0, 1)  ; "),
            "{annotated}"
        );
        assert!(annotated.contains("ir.rs:"), "{annotated}");
        // A free is not a node and has nowhere to have been built.
        assert!(annotated.contains("\n  free %2\n"), "{annotated}");
    }

    #[test]
    fn checks_a_graph_without_loading_a_single_weight() {
        let source = Counting::over(state_dict(&[("fc.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0])]));

        check_parameters(&feed_forward(), &source).unwrap();

        assert_eq!(source.checks.get(), 1);
        // The whole reason `check` is not `load` with its answer thrown away: under
        // `Residency::LowVram` a load is a weight crossing the bus, and checking a model would
        // otherwise move the whole of it to learn that the package is the right one.
        assert_eq!(source.loads.get(), 0, "checking a model must not move it");
    }

    #[test]
    fn a_check_is_still_the_two_questions_a_load_asks() {
        let source = Counting::over(state_dict(&[("fc.weight", &[2, 1], &[1.0, 0.0])]));

        // The right name in the wrong shape, which is the mistake this is here to catch early.
        let error = check_parameters(&feed_forward(), &source)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("has shape [2, 1], expected [2, 2]"),
            "{error}"
        );

        let missing = check_parameters(&feed_forward(), &Counting::over(state_dict(&[])))
            .unwrap_err()
            .to_string();
        assert!(missing.contains("not found in model"), "{missing}");
    }

    #[test]
    fn says_that_page_locked_memory_is_cudas_to_hand_out() {
        // Refused before the file is touched, so an empty one is enough to ask with.
        for device in [Device::Cpu, Device::Metal] {
            let error = Weights::new(empty_file(), device, Residency::LowVram)
                .unwrap_err()
                .to_string();

            assert!(error.contains("page-locked"), "{error}");
            assert!(error.contains(device.name()), "{error}");
        }
    }

    #[test]
    fn keeps_the_weights_on_the_device_unless_asked_otherwise() {
        assert_eq!(Residency::default(), Residency::Device);
    }
}
