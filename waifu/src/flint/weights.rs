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

//! A model's weights, read out of its package and kept on the device as far as they fit, and
//! what a graph's `load` instructions are answered from. See the IR's documentation for how a run
//! reads them: [`Ir`](super::Ir).

use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use super::graph::Graph;
use super::op::Op;
use super::{AllocHint, Device, MemorySnapshot, Tensor};
use crate::error::{Error, Result};
use crate::tensor_file;

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

/// A model's weights, read out of its package and kept on the device as far as they were asked to
/// be and it allows: see [`Weights::from_files`].
///
/// One map of tensors, some on the card and the rest in ordinary host memory -- or all of them on
/// the card. What differs is only whether a [`load`](Weights::load) has to move the weight -- a
/// kept one is already where the run wants it and is handed over as a handle, and a waiting one is
/// copied across the bus, to be given back at the `free` after its last reader.
///
/// # What low-vram costs
///
/// The copy, once per load per run, which is the whole model across the bus for every sampler
/// step. So it is slower than keeping the weights on the card and always will be; it is the mode
/// for when the alternative is not drawing at all.
///
/// And the model's size in host memory, for as long as the model is held -- the page cache the
/// package is mapped from, which is not page-locked and is not a copy: the kernel may drop it
/// under pressure and read it again from the file. It used to be locked, because the driver sends
/// a pageable source at well under the bus's rate. Flint's own upload does not (see `flint/cuda/staged_upload.h`): it stages
/// through two small locked buffers filled by several threads and sends as fast as a locked
/// source. Locking the model therefore bought nothing, and cost a limit: Windows locks no more
/// than half of RAM for the GPU, so a 32 GB machine could not hold a 17 GB model this way. And it
/// cost a copy: locked memory is the process's own, where the mapping is shared with the page
/// cache the file is already in.
///
/// # On Metal
///
/// The same, except that the package is not read when the model is. Metal's memory is the
/// machine's, so a model read into the page cache up front is the whole model in the memory
/// low-vram is there to spare, and on a machine smaller than the model the end of the package
/// pushes out its beginning before the first step asks for it. So each weight stays its file's
/// mapping, untouched, and a load reads it off the disk as it copies it into a buffer of the
/// device's. Pages read that way are clean, and the kernel drops them
/// rather than writing them anywhere when memory runs short.
///
/// The copy is the host's and synchronous (see `Context::upload` in `flint/metal_native`): it
/// waits for the device to finish with the buffer it writes into, and the device waits while the
/// host reads the disk. A weight's own pages are never handed to the device instead: a Metal
/// buffer over a file mapping is read in whole and wired before any work that names it runs, so
/// wrapping the package would hold all of it, and wrapping each weight is what a later version
/// can do to take the host out of the way.
pub struct Weights {
    /// Keyed the way the package keys them, which is what a graph's `load` asks for. Behind a
    /// cell, because a weight on the device may be moved back to the host in the middle of a run,
    /// through the `Weights` the run shares: see [`Weights::evict_one`].
    tensors: RefCell<HashMap<String, Tensor>>,

    /// The names of the weights `keep_the_smallest_on_device` put on the device, in the order it
    /// put them there, smallest first: a stack, whose top -- the largest -- is the first to go
    /// back to the host when the device runs out. See [`Weights::evict_one`].
    device_tensors: RefCell<Vec<String>>,

    /// Where a run wants a weight, which is the device the model was built for. Where any one
    /// weight waits between runs is its own tensor's to say.
    device: Device,
}

impl Weights {
    /// Every weight of the safetensors files at `paths`, for a model running on `device`, with up
    /// to `device_weight_bytes` of them kept on it -- `u64::MAX`, the usual answer, for as many as
    /// fit -- and the rest on the host.
    ///
    /// The one way a model's weights are read. Each file is mapped rather than read, and each of
    /// its weights is put where it is going straight out of the mapping, so the host holds no
    /// copy of its own: a weight moved to the card leaves nothing behind, and one kept on the host
    /// is the page cache. See [`tensor_file`](crate::tensor_file).
    ///
    /// The weights kept are the smallest, no more of them than the device has free -- the room its
    /// memory limit leaves, or, without one, its budget -- and a run that runs short moves the
    /// largest of them back off it: see [`Weights::evict_one`]. Those not kept are each moved onto
    /// the device by the load that reads it and given back after its last reader; on Metal, whose
    /// memory is the machine's, "the host" is the package's file, so a model larger than the
    /// machine still runs. 0 keeps none.
    ///
    /// Smallest first because what a load from the host costs is mostly not its bytes: each one
    /// is a copy the run waits for, and a model's norms, biases and scales are hundreds of loads
    /// that are each a wait and almost nothing to keep. So the first few hundred megabytes take
    /// most of the waits away, and what is left to cross over per pass is the large projections,
    /// whose copies are the ones that run at the bus's rate.
    ///
    /// What each device can do of that is [`kept_on_device`]'s to say: on the processor and
    /// Vulkan, and on CUDA asked to fill it, every weight is on the device.
    pub fn from_files(
        paths: &[impl AsRef<Path>],
        device: Device,
        device_weight_bytes: u64,
    ) -> Result<Weights> {
        tensor_file::collect(paths, device, device_weight_bytes)
    }

    /// What `device` makes of being asked to keep `device_weight_bytes` of the weights: `None`
    /// for every weight on the device as it is read, or the bytes of them to keep there.
    pub(crate) fn kept_on(device: Device, device_weight_bytes: u64) -> Option<u64> {
        kept_on_device(device, device_weight_bytes)
    }

    /// `tensors` as they are, for a run on `device`: not moved here, so a load
    /// hands back a handle on one already on `device` and copies one that is not.
    ///
    /// For weights that were not read out of a package -- made up by a test, or none at all for
    /// a graph that loads nothing.
    pub fn from_tensors(tensors: HashMap<String, Tensor>, device: Device) -> Result<Weights> {
        Ok(Weights {
            tensors: RefCell::new(tensors),
            ..Weights::empty(device)
        })
    }

    /// A weight for every [`Op::Load`] in `graph`, each made by `make` out of its name and shape,
    /// for a run on `device`.
    ///
    /// For weights made up rather than read: asking the graph what it loads means there is no
    /// second list of a model's weights to fall out of step with the first. A name loaded more
    /// than once is made once.
    pub fn for_graph(
        graph: &Graph,
        device: Device,
        mut make: impl FnMut(&str, &[i32]) -> Result<Tensor>,
    ) -> Result<Weights> {
        let mut tensors = HashMap::new();
        for (_, op) in graph.nodes() {
            if let Op::Load { name, shape } = op {
                if !tensors.contains_key(&name) {
                    let tensor = make(&name, &shape)?;
                    tensors.insert(name, tensor);
                }
            }
        }

        Weights::from_tensors(tensors, device)
    }

    /// How many weights there are.
    pub fn len(&self) -> usize {
        self.tensors.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.borrow().is_empty()
    }

    /// No weights yet, for a model running on `device`. [`tensor_file`](crate::tensor_file) fills
    /// it with [`Weights::insert`].
    pub(crate) fn empty(device: Device) -> Weights {
        Weights {
            tensors: RefCell::new(HashMap::new()),
            device_tensors: RefCell::new(Vec::new()),
            device,
        }
    }

    /// Keep `tensor` under `name`, in the type the file holds it in, and put on the device -- or,
    /// `on_host`, left where it is until a load wants it.
    ///
    /// A name already held is refused rather than replaced -- a model whose weights depend on the
    /// order its files were listed in is worse than one that will not load -- and refused before
    /// anything is moved.
    pub(crate) fn insert(&mut self, name: String, tensor: Tensor, on_host: bool) -> Result<()> {
        let tensors = self.tensors.get_mut();
        if tensors.contains_key(&name) {
            return Err(Error::format(format!(
                "tensor {name:?} is in more than one file of this model"
            )));
        }

        let tensor = match on_host {
            false => tensor.to_device_as(self.device, AllocHint::Weight)?,
            // On the host, a weight is its file's mapping, and nothing below reads it until a run
            // does -- so it is read here, in the order the package lies in. Except on Metal, where
            // the host's memory is the device's too and the package stays on the disk until a
            // load wants it; see the type documentation.
            true => {
                if self.device != Device::Metal {
                    read_through(&tensor)?;
                }
                tensor
            }
        };

        tensors.insert(name, tensor);
        Ok(())
    }

    /// Once every weight is in on the host: the smallest put on the device and kept there, as many
    /// as fit in `device_weight_bytes` and in what the device has free -- the room its memory
    /// limit leaves, or, without one, its budget. Given `u64::MAX` bytes, that is as many as the
    /// device has room for. See [`smallest_within`].
    ///
    /// After the whole package rather than as each weight comes, because which are the smallest
    /// is a question about all of them. They are moved in `order`, the order the package lies in,
    /// so that the disk is read front to back rather than smallest first. Each one moved leaves its
    /// host copy behind for the page cache to drop, as a resident model's weights do. A device that
    /// turns out to be full before the last of them -- each weight takes whole pages, which the
    /// sizes do not count -- keeps those it has.
    pub(crate) fn keep_the_smallest_on_device(
        &mut self,
        device_weight_bytes: u64,
        order: &[String],
    ) -> Result<()> {
        if device_weight_bytes == 0 {
            return Ok(());
        }

        let free = MemorySnapshot::capture(self.device)?.free.max(0) as u64;
        let allowed = device_weight_bytes.min(free);

        let tensors = self.tensors.get_mut();
        let mut sizes = Vec::with_capacity(tensors.len());
        for (name, tensor) in tensors.iter() {
            sizes.push((name.clone(), tensor.nbytes()? as u64));
        }
        let (chosen, _) = smallest_within(sizes, allowed);

        // Moved in the order they lie in, and stacked smallest first, as smallest_within hands
        // them back, so that the largest is on top.
        let wanted: std::collections::HashSet<&String> = chosen.iter().collect();
        let mut bytes = 0u64;
        let mut moved = std::collections::HashSet::new();
        for name in order.iter().filter(|name| wanted.contains(name)) {
            let tensor = tensors.get_mut(name).expect("a name the map was asked for");
            match tensor.to_device_as(self.device, AllocHint::Weight) {
                Ok(on_device) => {
                    bytes += on_device.nbytes()? as u64;
                    *tensor = on_device;
                    moved.insert(name.clone());
                }
                Err(error) if error.is_out_of_memory() => break,
                Err(error) => return Err(error.into()),
            }
        }
        let device_tensors = self.device_tensors.get_mut();
        device_tensors.extend(chosen.into_iter().filter(|name| moved.contains(name)));

        crate::log::write(
            crate::log::Level::Info,
            "weights",
            &format!(
                "{} of {} weights kept on {}: {} MB of the {} MB there was room for",
                device_tensors.len(),
                tensors.len(),
                self.device.name(),
                bytes / 1_000_000,
                allowed / 1_000_000
            ),
        );
        Ok(())
    }

    /// The weight called `name`, which the caller expects to have `shape`, on the device the run
    /// is going to: a handle on it if it is already there, and the copy across the bus if it is
    /// waiting on the host. Either way the run lets go of it after the weight's last reader.
    ///
    /// Synchronous, because the instruction after this one reads what it returns and there is
    /// nothing else for the host to be doing in between.
    pub fn load(&self, name: &str, shape: &[i32]) -> Result<Tensor> {
        // The map let go of before the device is asked for anything: running out on the way
        // there is answered by moving a weight off it, which is a change to the map.
        let tensor = held(&self.tensors.borrow(), name, shape)?.clone();
        Ok(tensor.to_device_as(self.device, AllocHint::Weight)?)
    }

    /// Move the weight on top of [`device_tensors`](Weights) -- the largest of those kept on the
    /// device -- back to the host, and say which it was; `None` where none is left to move.
    ///
    /// For when the device runs out in the middle of a run: the weight is copied back to the host
    /// and the next load of it copies it across again, as a low-vram load does. Its memory on the
    /// device goes back once nothing holds it -- at once, unless an instruction that is running
    /// has it in hand, and then after that instruction.
    pub fn evict_one(&self) -> Result<Option<String>> {
        let Some(name) = self.device_tensors.borrow_mut().pop() else {
            return Ok(None);
        };
        let mut tensors = self.tensors.borrow_mut();
        let tensor = tensors.get_mut(&name).expect("a name the map holds");
        *tensor = tensor.to_device(Device::Cpu)?;
        Ok(Some(name))
    }

    /// What shape the weight called `name` is, if there is one here.
    ///
    /// For a model that reads its own architecture out of the package rather than out of a
    /// configuration -- Anima's autoencoder counts its stages by looking for them -- which has to
    /// ask before it knows what it is asking for. Answered without handing the weight over, for
    /// the same reason `check` is.
    pub fn shape_of(&self, name: &str) -> Option<Vec<i32>> {
        self.tensors.borrow().get(name).map(|tensor| tensor.shape())
    }

    /// Whether there is a weight called `name` here at all, whatever shape it turns out to be.
    ///
    /// For the half of a model a package may or may not hold -- SDXL's VAE encoder is one -- where
    /// the question has to be answered before there is anything to ask it about a shape.
    pub fn has(&self, name: &str) -> bool {
        self.tensors.borrow().contains_key(name)
    }

    /// Whether `name` is here with `shape`, answered without handing the weight over.
    ///
    /// Asked of the map rather than being `load` with its result dropped: checking a low-vram
    /// model's eight hundred weights by moving each one across the bus would be a whole pass's
    /// worth of copying to learn that the package is the right one. See [`check_parameters`],
    /// which is the only caller.
    pub fn check(&self, name: &str, shape: &[i32]) -> Result<()> {
        held(&self.tensors.borrow(), name, shape).map(drop)
    }

    /// Every weight, by name, for the checks made over the whole package once it is read.
    pub(crate) fn tensors(&self) -> Ref<'_, HashMap<String, Tensor>> {
        self.tensors.borrow()
    }
}

impl fmt::Debug for Weights {
    /// How many there are and where they are going, rather than every weight: a model holds
    /// hundreds of them and they are named in the IR that reads them.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Weights")
            .field("tensors", &self.tensors.borrow().len())
            .field("device", &self.device)
            .finish()
    }
}

/// What `device` makes of being asked to keep `device_weight_bytes` of a model's weights: `None`
/// for every weight on the device as it is read, and otherwise every weight on the host and as
/// many bytes of the smallest of them as that on the device, moved there once the package is in.
///
/// Keeping some off the device needs a device whose memory a weight can be moved into and given
/// back from one load at a time -- CUDA's, through its staged upload, and Metal's. The processor
/// already holds the weights where it reads them, and Vulkan has no such path, so on those every
/// weight is on the device. And filling the device -- `u64::MAX` -- needs one that can make room
/// again, which only Metal can as things are: a card has no limit here, and running out there is
/// not told apart from any other failure. So on CUDA filling is every weight on the card; a number
/// of bytes, none among them, is kept as asked.
fn kept_on_device(device: Device, device_weight_bytes: u64) -> Option<u64> {
    match device {
        Device::Metal => Some(device_weight_bytes),
        Device::Cuda if device_weight_bytes != u64::MAX => Some(device_weight_bytes),
        _ => None,
    }
}

/// Which of the weights `sizes` names, by name and bytes, to keep on the device within `budget`
/// bytes, and how many bytes that comes to: the smallest first, for as long as the next fits.
///
/// Once one does not fit, none after it can, since they are larger. Weights of the same size go
/// by name, so that the same package and the same budget keep the same weights every time.
fn smallest_within(mut sizes: Vec<(String, u64)>, budget: u64) -> (Vec<String>, u64) {
    sizes.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    let mut kept = Vec::new();
    let mut total = 0u64;
    for (name, bytes) in sizes {
        if total.saturating_add(bytes) > budget {
            break;
        }
        total += bytes;
        kept.push(name);
    }
    (kept, total)
}

/// Say now whether `weights` can answer everything `graph` will ask for.
///
/// A graph names its weights and reads none of them, and [`Weights`] are read without
/// consulting any graph, so the two are only brought together by a `load` instruction running.
/// That would make a package that does not hold what a model asks for -- or holds it in another
/// shape -- something found out on the way to a picture rather than when the model was asked for.
///
/// This asks the same questions the first run would and throws the answers away. It is
/// [`Weights::check`] rather than a read per weight precisely so that it stays free to ask:
/// a read is a read of the disk, and a package is checked before any of it is wanted.
pub fn check_parameters(graph: &Graph, weights: &Weights) -> Result<()> {
    for (_, op) in graph.nodes() {
        if let Op::Load { name, shape } = op {
            weights.check(&name, &shape)?;
        }
    }

    Ok(())
}

/// Read every page of a host tensor's bytes once.
///
/// For a weight that is its file's mapping: the disk is read when a page is first touched, and
/// left to the run that would be in the order the graph asks for weights -- on a spinning disk, a
/// seek for each. Read as the package is taken apart, the file goes in front to back instead, at
/// the disk's rate. A weight already in the page cache costs a walk over its page tables.
///
/// A weight moved to the card needs none of this: moving it reads it, in the same order.
fn read_through(tensor: &Tensor) -> Result<()> {
    const PAGE: usize = 4096;

    let bytes = tensor.host_bytes()?;
    // One byte of each page faults the page in; the sum keeps the reads from being dropped.
    let mut sum = 0u8;
    for at in (0..bytes.len()).step_by(PAGE) {
        sum = sum.wrapping_add(unsafe { std::ptr::read_volatile(bytes.as_ptr().add(at)) });
    }
    std::hint::black_box(sum);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Weights on the host holding `tensors`.
    fn weights(tensors: &[(&str, &[i32], &[f32])]) -> Weights {
        let tensors = tensors
            .iter()
            .map(|(name, shape, values)| {
                (name.to_string(), Tensor::from_f32(shape, values).unwrap())
            })
            .collect();
        Weights::from_tensors(tensors, Device::Cpu).unwrap()
    }

    /// A graph that loads one weight, `fc.weight` of [2, 2], and hands it back.
    fn loads_a_weight() -> Graph {
        let g = Graph::new();
        let weight = g.load("fc.weight", &[2, 2]);
        g.output("weight", weight);
        g
    }

    #[test]
    fn a_check_is_still_the_two_questions_a_load_asks() {
        let source = weights(&[("fc.weight", &[2, 1], &[1.0, 0.0])]);

        // The right name in the wrong shape, which is the mistake this is here to catch early.
        let error = check_parameters(&loads_a_weight(), &source)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("has shape [2, 1], expected [2, 2]"),
            "{error}"
        );

        let missing = check_parameters(&loads_a_weight(), &weights(&[]))
            .unwrap_err()
            .to_string();
        assert!(missing.contains("not found in model"), "{missing}");
    }

    #[test]
    fn each_device_does_what_it_can_of_what_is_asked() {
        let (fill, some, none) = (u64::MAX, 1 << 30, 0);

        // Metal does all of it.
        for asked in [fill, some, none] {
            assert_eq!(kept_on_device(Device::Metal, asked), Some(asked));
        }
        // CUDA keeps a number of bytes, none among them, but cannot make room again, so filling it
        // is all of the weights on the card.
        assert_eq!(kept_on_device(Device::Cuda, fill), None);
        assert_eq!(kept_on_device(Device::Cuda, some), Some(some));
        assert_eq!(kept_on_device(Device::Cuda, none), Some(none));
        // The processor and Vulkan have every weight on the device, however they are asked.
        for device in [Device::Cpu, Device::Vulkan] {
            for asked in [fill, some, none] {
                assert_eq!(kept_on_device(device, asked), None);
            }
        }
    }

    #[test]
    fn a_hybrid_model_keeps_its_smallest_weights_on_the_device() {
        let sizes = |list: &[(&str, u64)]| {
            list.iter()
                .map(|(name, bytes)| (name.to_string(), *bytes))
                .collect::<Vec<_>>()
        };
        let model = sizes(&[
            ("proj", 400),
            ("norm", 8),
            ("bias", 16),
            ("scale", 8),
            ("mlp", 200),
        ]);

        // The smallest first, ties by name, until the next one would not fit.
        assert_eq!(
            smallest_within(model.clone(), 240),
            (
                vec!["norm".into(), "scale".into(), "bias".into(), "mlp".into()],
                232
            )
        );
        // Stopping at the first that does not fit, not skipping it for a later one that would.
        assert_eq!(
            smallest_within(model.clone(), 100),
            (vec!["norm".into(), "scale".into(), "bias".into()], 32)
        );
        assert_eq!(smallest_within(model.clone(), 0), (vec![], 0));
        assert_eq!(smallest_within(model, u64::MAX).0.len(), 5);
    }
}
