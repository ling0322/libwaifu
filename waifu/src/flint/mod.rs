//! Safe Rust bindings for the native flint tensor library.
//!
//! A [`Tensor`] owns a handle to shape metadata plus a reference to storage that other tensors may
//! share. Reshaping operations such as [`Tensor::view`] or [`Tensor::transpose`] return a new
//! tensor over the same elements rather than copying them, and `clone` does the same, so cloning a
//! tensor is cheap but does not give you an independent copy of the data.
//!
//! ```no_run
//! use waifu::flint::{self, DType, Device, Tensor};
//!
//! flint::init();
//! let x = Tensor::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
//! assert_eq!(x.shape(), vec![2, 3]);
//! assert_eq!(x.dtype(), DType::Float);
//! assert_eq!(x.device(), Device::Cpu);
//! assert_eq!(x.transpose(0, 1)?.to_vec_f32()?, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
//! # Ok::<(), waifu::flint::Error>(())
//! ```
//!
//! The operations that work on tensors rather than describe them live in [`functional`], one
//! function per method of the native `Operators`:
//!
//! ```no_run
//! use waifu::flint::{functional as F, Tensor};
//!
//! let x = Tensor::from_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0])?;
//! let sums = F::sum(&x, F::LAST_DIM)?;
//! assert_eq!(sums.to_vec_f32()?, vec![3.0, 7.0]);
//! # Ok::<(), waifu::flint::Error>(())
//! ```
//!
//! # A pass written down, rather than run
//!
//! Everything above computes as it is called. A pass written that way -- [`functional::matmul`]
//! and the rest, one call at a time -- exists only while it is running, and nothing can be asked
//! about it before it starts: what it will read, how much it will hold at once, which of its
//! values are still wanted at a given point. The answers are in the shape of the Rust code, which
//! is not a thing that can be read at run time.
//!
//! A [`Graph`] is that same pass written down. Each method adds a node and hands back the
//! [`Value`] it produces, so building one reads like the arithmetic it stands for and computes
//! none of it:
//!
//! ```
//! use waifu::flint::Graph;
//!
//! let g = Graph::new();
//! let x = g.input("hidden");
//! let scale = g.input("scale");
//!
//! let normed = g.layer_norm(x, Some(scale), None, 1e-5);
//! let gated = g.silu(normed);
//! g.output("hidden", gated);
//! ```
//!
//! A graph does not run itself, and there is deliberately no room in it for anything that would.
//! What runs one is an [`Ir`], which is the same pass again with two things decided that a graph
//! does not decide: which nodes are worth running, and where each value is finished with.
//!
//! ```
//! # use waifu::flint::{Graph, Ir, RunContext, Tensor};
//! # use std::collections::HashMap;
//! # let g = Graph::new();
//! # let x = g.input("hidden");
//! # let gated = g.silu(x);
//! # g.output("hidden", gated);
//! // The same graph again, and this time run. The weights come from the source the run is
//! // given, which is also what decides where they wait.
//! let ir = Ir::compile(&g);
//!
//! let hidden = Tensor::from_f32(&[1, 2], &[1.0, -1.0])?;
//! let weights = HashMap::new();
//!
//! let context = RunContext::new(&weights).input("hidden", &hidden);
//! let outputs = ir.run(&context)?;
//! assert_eq!(outputs[0].0, "hidden");
//! # Ok::<(), waifu::Error>(())
//! ```
//!
//! See [`Graph`] for what it does and does not promise about what it holds, and [`Ir`] for what
//! running one costs and what it does not yet do.
//!
//! # Threading
//!
//! [`Tensor`] is deliberately neither `Send` nor `Sync`. The underlying operators keep per-device
//! state that is not prepared for concurrent use, so a tensor stays on the thread that made it.

mod ffi;
mod fp8;
pub mod functional;
mod graph;
mod ir;
mod op;
mod operators;

pub use fp8::{Fp8Tensor, CHANNEL_SCALE_SUFFIX};
pub use graph::{Graph, Site, WeightFormat};
pub use ir::{check_parameters, Inst, Ir, ParamSource, Residency, RunContext, Weights};
pub use operators::Operators;
pub use op::{Binary, Extent, Op, Reduce, Scalar, Unary, Value};

use operators::{operators_of, transfer_operators};

use std::cell::RefCell;
use std::ffi::CStr;
use std::fmt;
use std::os::raw::c_void;
use std::rc::Rc;
use std::sync::Once;

/// Element type of a tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum DType {
    Float = 1,
    Long = 2,
    UInt8 = 3,
    Float16 = 4,
    Int8 = 6,
    Bool = 8,
    Int32 = 9,
    Fp8E4M3 = 10,
}

impl DType {
    /// The number this type is written as, in a model file or over the C interface.
    pub fn code(self) -> i32 {
        self as i32
    }

    /// The type `code` names, as [`DType::code`] wrote it.
    pub fn from_code(code: i32) -> Result<DType> {
        DType::from_raw(code)
    }

    /// The number of bytes `numel` elements of this type occupy once packed together.
    pub fn total_size(self, numel: i64) -> i64 {
        match self {
            DType::Float | DType::Int32 => 4 * numel,
            DType::Float16 => 2 * numel,
            DType::Long => 8 * numel,
            DType::UInt8 | DType::Int8 | DType::Bool | DType::Fp8E4M3 => numel,
        }
    }

    fn from_raw(raw: i32) -> Result<DType> {
        match raw {
            1 => Ok(DType::Float),
            2 => Ok(DType::Long),
            3 => Ok(DType::UInt8),
            4 => Ok(DType::Float16),
            6 => Ok(DType::Int8),
            8 => Ok(DType::Bool),
            9 => Ok(DType::Int32),
            10 => Ok(DType::Fp8E4M3),
            other => Err(Error::unsupported(format!("unknown dtype {other}"))),
        }
    }
}

/// Where a tensor's storage lives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum Device {
    Cpu = 0,
    Cuda = 1,
    /// Host memory page-locked by the CUDA driver. Readable by the CPU and carrying no operators
    /// of its own: it is where weights wait to be copied to the GPU.
    CudaHost = 2,
    Metal = 3,
    Vulkan = 4,
}

impl Device {
    /// What this device is called on the command line and on screen.
    pub fn name(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
            Device::CudaHost => "cuda-host",
            Device::Metal => "metal",
            Device::Vulkan => "vulkan",
        }
    }

    /// Whether this build has operators for the device and the machine can run them.
    ///
    /// Worth asking before running anything: the operators a device is missing end the process
    /// rather than reporting an error, so a caller that can fall back should check first.
    pub fn is_available(self) -> bool {
        init();
        let mut available: i32 = 0;
        match check(unsafe { ffi::fl_is_device_available(self as i32, &mut available) }) {
            Ok(()) => available != 0,
            Err(_) => false,
        }
    }

    fn from_raw(raw: i32) -> Result<Device> {
        match raw {
            0 => Ok(Device::Cpu),
            1 => Ok(Device::Cuda),
            2 => Ok(Device::CudaHost),
            3 => Ok(Device::Metal),
            4 => Ok(Device::Vulkan),
            other => Err(Error::unsupported(format!("unknown device {other}"))),
        }
    }
}

/// One end of a slice range. [`Bound::End`] leaves that end of the dimension where it is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bound {
    /// A position, which may be negative to count from the back.
    At(i32),
    /// The start or the end of the dimension, whichever side this bound is on.
    End,
}

impl From<i32> for Bound {
    fn from(index: i32) -> Bound {
        Bound::At(index)
    }
}

/// A failure reported by the library.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error {
    code: i32,
    message: String,
}

impl Error {
    /// The C error code, or zero for a failure the binding itself detected.
    pub fn code(&self) -> i32 {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    /// Whether the call was rejected for an argument it could not accept.
    pub fn is_invalid_arg(&self) -> bool {
        self.code == ffi::FL_ERROR_INVALID_ARG
    }

    /// Whether the call was accepted but could not be carried out, such as running out of memory
    /// or asking for a device the build does not support.
    pub fn is_aborted(&self) -> bool {
        self.code == ffi::FL_ERROR_ABORTED
    }

    fn unsupported(message: String) -> Error {
        Error { code: 0, message }
    }

    /// An argument the binding refused before asking the library, reported as the library would.
    fn invalid(message: impl Into<String>) -> Error {
        Error {
            code: ffi::FL_ERROR_INVALID_ARG,
            message: message.into(),
        }
    }

    /// Reads the error the last C call left on this thread. Only called right after one failed.
    fn last() -> Error {
        // Safety: the pointer is owned by the library and stays valid until this thread makes
        // another call into it, which cannot happen while we are copying it out.
        let message = unsafe {
            let raw = ffi::fl_get_last_error_message();
            if raw.is_null() {
                String::new()
            } else {
                CStr::from_ptr(raw).to_string_lossy().into_owned()
            }
        };
        let code = unsafe { ffi::fl_get_last_error_code() };
        Error {
            code,
            message: if message.is_empty() {
                "unknown error".to_string()
            } else {
                message
            },
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (code 0x{:04x})", self.message, self.code)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Turns a status code into a `Result`, picking up the message the failing call left behind.
fn check(status: i32) -> Result<()> {
    if status == ffi::FL_OK {
        Ok(())
    } else {
        Err(Error::last())
    }
}

static INIT: Once = Once::new();

/// Select the operator backends for this machine.
///
/// Every constructor calls this, so it is only worth calling directly to get the cost out of the
/// way at a point of your choosing. Repeated calls do nothing.
/// Run `handler` just before the library ends the process.
///
/// Little does. A check that fails inside an operator -- two tensors on different devices meeting
/// at a convolution, a shape no kernel was written for -- comes back as an `Err`, the same as a
/// bad argument does, so a broken invariant is something you can handle rather than something
/// that happens to you. What is left is the path with nothing to return a `Result` to: reaching
/// code that was never written.
///
/// That one prints what went wrong and calls `abort()`, and the message lands on top of whatever
/// a caller that owns the screen had drawn, unreadable. This is that caller's one chance to put
/// the screen back. It runs before anything is printed, on whichever thread failed, and nothing
/// runs after it but the message, the stack trace and the abort -- so it should do the one thing
/// it is there for and return.
pub fn on_fatal(handler: extern "C" fn()) {
    init();
    unsafe { ffi::fl_set_fatal_handler(Some(handler)) };
}

pub fn init() {
    INIT.call_once(|| unsafe { ffi::fl_init() });
}

/// The memory usage of one device.
///
/// A device that does not report its usage, which is what the CPU backend does, reports zero
/// everywhere; a caller that has to size an allocation from this should check [`total`] first.
///
/// [`total`]: MemorySnapshot::total
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MemorySnapshot {
    /// The memory the device has.
    pub total: i64,
    /// The memory no process has reserved yet. Memory this process already took from the driver
    /// is not free even once its tensors are gone, since the allocator holds on to it for reuse.
    pub free: i64,
    /// The bytes the tensors of this process hold.
    pub allocated: i64,
    /// The largest [`allocated`](MemorySnapshot::allocated) reached since the last
    /// [`MemorySnapshot::reset_peak_stats`].
    pub peak_allocated: i64,
}

impl MemorySnapshot {
    /// Measure the memory usage of `device`.
    pub fn capture(device: Device) -> Result<MemorySnapshot> {
        init();
        let mut raw = ffi::FlMemorySnapshot::default();
        check(unsafe { ffi::fl_memory_capture(device as i32, &mut raw) })?;
        Ok(MemorySnapshot {
            total: raw.total,
            free: raw.free,
            allocated: raw.allocated,
            peak_allocated: raw.peak_allocated,
        })
    }

    /// Set the peak of `device` back to zero, so that the next measurement covers only what
    /// happens from here. This is how the size of one forward pass is measured.
    pub fn reset_peak_stats(device: Device) -> Result<()> {
        init();
        check(unsafe { ffi::fl_memory_reset_peak_stats(device as i32) })
    }

    /// Give every byte of `device` that no tensor holds back to the driver, which is what makes
    /// [`free`](MemorySnapshot::free) count it and what lets another process have it.
    ///
    /// For after a model has been let go of and nothing is about to want it back. It is the wrong
    /// call between two runs of one model: what it hands over is the memory the next run would
    /// have taken straight out of the allocator, and that run then waits for the driver again.
    ///
    /// A device that gives memory back as each tensor goes -- the CPU, or a CUDA build without its
    /// pool -- has nothing to do here and says so by doing nothing.
    pub fn release_unused(device: Device) -> Result<()> {
        init();
        check(unsafe { ffi::fl_memory_release_unused(device as i32) })
    }
}

/// Storage: a run of elements of one dtype on one device, owned here and shared by every tensor
/// over it. Freed when the last of them goes.
struct Storage {
    raw: ffi::FlTensorData,
    dtype: DType,
    device: Device,
    /// Whether the elements are somebody else's bytes, borrowed to be read where they lie.
    read_only: bool,
    /// What borrowed storage reads from. Dropped after `raw` is destroyed, which is what keeps the
    /// bytes alive for exactly as long as anything may read them.
    _owner: Option<Box<dyn std::any::Any>>,
}

impl Storage {
    /// Uninitialized storage for `numel` elements, at least one.
    fn allocate(numel: i64, dtype: DType, device: Device) -> Result<Rc<Storage>> {
        init();
        let mut raw: ffi::FlTensorData = std::ptr::null_mut();
        check(unsafe {
            ffi::fl_tensor_data_create(device as i32, dtype as i32, numel.max(1), &mut raw)
        })?;
        Ok(Rc::new(Storage {
            raw,
            dtype,
            device,
            read_only: false,
            _owner: None,
        }))
    }

    /// The address of the first element, for storage on the host.
    fn host_ptr(&self) -> Result<*mut u8> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        check(unsafe { ffi::fl_tensor_data_get_host_ptr(self.raw, &mut ptr) })?;
        Ok(ptr as *mut u8)
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        // Safety: made by a successful C call and destroyed exactly once, here. Every view of it
        // belongs to a tensor that holds this storage, so none outlives it.
        unsafe { ffi::fl_tensor_data_destroy(self.raw) };
    }
}

/// Where a tensor's elements are in its storage, and the view handle that says so to the library.
struct Layout {
    shape: Vec<i32>,
    stride: Vec<i32>,
    offset: i64,
    raw: ffi::FlTensorView,
}

impl Drop for Layout {
    fn drop(&mut self) {
        // Safety: made by a successful C call and destroyed exactly once, here.
        unsafe { ffi::fl_tensor_view_destroy(self.raw) };
    }
}

/// Row-major strides for `shape`.
fn contiguous_strides(shape: &[i32]) -> Vec<i32> {
    let mut stride = vec![0; shape.len()];
    let mut step: i64 = 1;
    for d in (0..shape.len()).rev() {
        stride[d] = step as i32;
        step *= shape[d] as i64;
    }
    stride
}

/// The number of elements `shape` holds, counted the way flint counts them: a shape of no
/// dimensions holds none.
fn numel_of(shape: &[i32]) -> i64 {
    if shape.is_empty() {
        0
    } else {
        shape.iter().map(|&n| n as i64).product()
    }
}

/// `shape` with a dimension of -1 worked out from `numel`, refused if it cannot be.
fn real_shape(numel: i64, shape: &[i32]) -> Result<Vec<i32>> {
    let mut real = shape.to_vec();
    let mut infer = None;
    let mut known: i64 = 1;
    for (d, &n) in shape.iter().enumerate() {
        if n < 0 {
            if infer.is_some() {
                return Err(Error::invalid("a view infers more than one dimension"));
            }
            infer = Some(d);
        } else {
            known *= n as i64;
        }
    }

    match infer {
        Some(d) => {
            if known == 0 || numel % known != 0 {
                return Err(Error::invalid(format!(
                    "a view of {numel} elements cannot infer a dimension of {known}"
                )));
            }
            real[d] = (numel / known) as i32;
        }
        None if numel != known => {
            return Err(Error::invalid(format!(
                "invalid view: {numel} elements cannot be seen as {known}"
            )));
        }
        None => {}
    }
    Ok(real)
}

/// A tensor: a shape over storage that other tensors may share.
///
/// The storage goes away once the last tensor referring to it is gone. A tensor is a reference to
/// that storage plus where its elements are in it, so cloning one, or taking a view of it, copies
/// neither the elements nor the storage.
#[derive(Clone)]
pub struct Tensor {
    storage: Rc<Storage>,
    layout: Rc<Layout>,
}

impl Tensor {
    /// A tensor over `storage` with the given layout, checked by the library against the storage.
    fn over(storage: Rc<Storage>, shape: Vec<i32>, stride: Vec<i32>, offset: i64) -> Result<Tensor> {
        let mut raw: ffi::FlTensorView = std::ptr::null_mut();
        check(unsafe {
            ffi::fl_tensor_view_create(
                storage.raw,
                shape.as_ptr(),
                stride.as_ptr(),
                shape.len() as i32,
                offset,
                &mut raw,
            )
        })?;
        Ok(Tensor {
            storage,
            layout: Rc::new(Layout {
                shape,
                stride,
                offset,
                raw,
            }),
        })
    }

    /// The same storage seen through another layout.
    fn with_layout(&self, shape: Vec<i32>, stride: Vec<i32>, offset: i64) -> Result<Tensor> {
        Tensor::over(Rc::clone(&self.storage), shape, stride, offset)
    }

    /// The view handle the C interface reads and writes this tensor through.
    pub(crate) fn raw(&self) -> ffi::FlTensorView {
        self.layout.raw
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: &[i32], dtype: DType, device: Device) -> Result<Tensor> {
        let mut tensor = Tensor::empty(shape, dtype, device)?;
        functional::fill(&mut tensor, 0.0)?;
        Ok(tensor)
    }

    /// Create a tensor without writing anything into it.
    ///
    /// For storage that is about to be overwritten in full, such as a KV cache pool: zeroing tens
    /// of gigabytes that the first forward pass overwrites anyway is a cost with nothing to show
    /// for it. Reading an element before writing it gives whatever the allocator handed back, so
    /// this is only worth using when every element is written first.
    pub fn empty(shape: &[i32], dtype: DType, device: Device) -> Result<Tensor> {
        if let Some(&n) = shape.iter().find(|&&n| n < 0) {
            return Err(Error::invalid(format!("a tensor cannot have a dimension of {n}")));
        }
        let numel: i64 = shape.iter().map(|&n| n as i64).product();
        let storage = Storage::allocate(numel, dtype, device)?;
        Tensor::over(storage, shape.to_vec(), contiguous_strides(shape), 0)
    }

    /// Create a CPU tensor holding a copy of `data`, laid out row-major.
    ///
    /// Fails if `data` does not hold exactly as many elements as `shape` describes.
    pub fn from_f32(shape: &[i32], data: &[f32]) -> Result<Tensor> {
        Tensor::from_elements(shape, data, DType::Float)
    }

    /// Create a CPU tensor of 64-bit integers holding a copy of `data`.
    pub fn from_i64(shape: &[i32], data: &[i64]) -> Result<Tensor> {
        Tensor::from_elements(shape, data, DType::Long)
    }

    /// Create a CPU tensor of 32-bit integers holding a copy of `data`, the type the paged
    /// attention operations take their block tables and sequence lengths in.
    pub fn from_i32(shape: &[i32], data: &[i32]) -> Result<Tensor> {
        Tensor::from_elements(shape, data, DType::Int32)
    }

    /// Create a CPU tensor of bytes holding a copy of `data`, the type the element-wise
    /// comparisons take their inputs in.
    pub fn from_u8(shape: &[i32], data: &[u8]) -> Result<Tensor> {
        Tensor::from_elements(shape, data, DType::UInt8)
    }

    /// Create a CPU tensor of `dtype` over the raw bytes of `data`, laid out row-major.
    ///
    /// The element types a tensor can hold are not all types Rust has a use for on its own, so
    /// this is how a quantized or half precision tensor gets built: hand over the bytes as they
    /// were stored. `data` must be exactly as long as `shape` and `dtype` describe.
    pub fn from_bytes(shape: &[i32], dtype: DType, data: &[u8]) -> Result<Tensor> {
        Tensor::from_elements(shape, data, dtype)
    }

    /// A CPU tensor of `dtype` that *is* `range` of `owner`'s bytes, rather than a copy of them.
    ///
    /// For a weight file mapped into memory: the mapping becomes the tensor's storage, so reading
    /// a package copies nothing. The storage holds a reference to `owner`, so the bytes live as
    /// long as the tensor and every view of it.
    ///
    /// The tensor is read-only -- [`Tensor::host_bytes_mut`] refuses it -- and the range has to
    /// start on a multiple of the element size, which a safetensors writer arranges and this
    /// refuses otherwise; [`tensor_file`](crate::tensor_file) copies such a tensor instead.
    pub(crate) fn borrowing<T>(
        shape: &[i32],
        dtype: DType,
        owner: &std::sync::Arc<T>,
        range: std::ops::Range<usize>,
    ) -> Result<Tensor>
    where
        T: AsRef<[u8]> + Send + Sync + 'static,
    {
        init();
        let bytes = &owner.as_ref().as_ref()[range];
        let numel = numel_of(shape);
        if numel <= 0 {
            return Err(Error::invalid("a borrowed tensor has at least one element"));
        }
        let expected = dtype.total_size(numel);
        if bytes.len() as i64 != expected {
            return Err(Error::invalid(format!(
                "{} bytes do not hold {:?} of {dtype:?}: expected {expected}",
                bytes.len(),
                shape
            )));
        }

        let mut raw: ffi::FlTensorData = std::ptr::null_mut();
        check(unsafe {
            ffi::fl_tensor_data_borrow(bytes.as_ptr() as *const c_void, dtype as i32, numel, &mut raw)
        })?;
        let storage = Rc::new(Storage {
            raw,
            dtype,
            device: Device::Cpu,
            read_only: true,
            _owner: Some(Box::new(std::sync::Arc::clone(owner))),
        });
        Tensor::over(storage, shape.to_vec(), contiguous_strides(shape), 0)
    }

    fn from_elements<T>(shape: &[i32], data: &[T], dtype: DType) -> Result<Tensor> {
        let mut tensor = Tensor::empty(shape, dtype, Device::Cpu)?;
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        let dest = tensor.host_bytes_mut()?;
        if dest.len() != bytes.len() {
            return Err(Error::invalid(format!(
                "data_size does not match the shape and dtype: expected {} bytes, got {}",
                dest.len(),
                bytes.len()
            )));
        }
        dest.copy_from_slice(bytes);
        Ok(tensor)
    }

    /// The bytes of a contiguous host tensor, where they lie.
    fn host_run(&self) -> Result<(*mut u8, usize)> {
        if !matches!(self.storage.device, Device::Cpu | Device::CudaHost) {
            return Err(Error::invalid(format!(
                "the bytes of a tensor on {} have no address the caller may touch",
                self.storage.device.name()
            )));
        }
        if !self.is_contiguous() {
            return Err(Error::invalid("a non-contiguous tensor's bytes are not one run"));
        }
        let nbytes = self.nbytes()? as usize;
        if nbytes == 0 {
            return Ok((std::ptr::NonNull::dangling().as_ptr(), 0));
        }
        let start = self.storage.host_ptr()?;
        let skip = self.storage.dtype.total_size(self.layout.offset) as usize;
        Ok((unsafe { start.add(skip) }, nbytes))
    }

    /// The tensor's own bytes, to be written where they lie.
    ///
    /// For storage filled by something that writes into a buffer of the caller's choosing -- a
    /// file read is the case this exists for -- rather than read into a buffer of its own and
    /// copied in afterwards. For a model's weights those two differ by the whole model memcpy'd a
    /// second time, which is the difference between streaming a package and holding it twice.
    ///
    /// Only a contiguous tensor on the host, which is the CPU's memory or the page-locked host
    /// memory CUDA hands out. One on the device is refused: it has no address this side may touch.
    /// So is one [`borrowing`](Tensor::borrowing) a mapped file, whose bytes may only be read.
    ///
    /// # What the borrow does and does not say
    ///
    /// The slice borrows `self`, so it cannot outlive this handle, and `&mut self` keeps a second
    /// borrow from being taken through the same one. It does not make the bytes exclusively this
    /// handle's: storage is shared between handles, so a tensor cloned beforehand still addresses
    /// them. What this is for is filling storage that nothing else has seen yet.
    pub fn host_bytes_mut(&mut self) -> Result<&mut [u8]> {
        if self.storage.read_only {
            return Err(Error::invalid(
                "the tensor's bytes are borrowed read-only, so there is no writable pointer to hand out",
            ));
        }
        let (ptr, len) = self.host_run()?;
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
    }

    /// The tensor's own bytes, to be read where they lie.
    ///
    /// [`Tensor::host_bytes_mut`] for reading, and so given for a tensor
    /// [`borrowing`](Tensor::borrowing) a mapped file too. The same refusals otherwise: a tensor on
    /// the device, and a non-contiguous one.
    pub fn host_bytes(&self) -> Result<&[u8]> {
        let (ptr, len) = self.host_run()?;
        Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Dimension `dim`, which may be negative to count from the back, as an index.
    fn real_dim(&self, dim: i32) -> Result<usize> {
        let rank = self.layout.shape.len() as i32;
        let real = if dim < 0 { dim + rank } else { dim };
        if real < 0 || real >= rank {
            return Err(Error::invalid(format!("no dimension {dim} in a {rank}-D tensor")));
        }
        Ok(real as usize)
    }

    /// Position `index` of dimension `dim`, which may be negative to count from the back and may
    /// be one past the end.
    fn real_index(&self, dim: usize, index: i32) -> Result<i32> {
        let size = self.layout.shape[dim];
        let real = if index >= 0 { index } else { size + index };
        if real < 0 || real > size {
            return Err(Error::invalid(format!(
                "index {index} is outside a dimension of {size}"
            )));
        }
        Ok(real)
    }

    /// Number of dimensions.
    pub fn dim(&self) -> Result<i32> {
        Ok(self.layout.shape.len() as i32)
    }

    /// Size of dimension `dim`, which may be negative to count from the back.
    pub fn shape_at(&self, dim: i32) -> Result<i32> {
        Ok(self.layout.shape[self.real_dim(dim)?])
    }

    /// Sizes of every dimension.
    pub fn shape(&self) -> Vec<i32> {
        self.layout.shape.clone()
    }

    /// Stride of dimension `dim`, in elements.
    pub fn stride(&self, dim: i32) -> Result<i32> {
        Ok(self.layout.stride[self.real_dim(dim)?])
    }

    /// Total number of elements.
    pub fn numel(&self) -> i64 {
        numel_of(&self.layout.shape)
    }

    pub fn dtype(&self) -> DType {
        self.storage.dtype
    }

    pub fn try_dtype(&self) -> Result<DType> {
        Ok(self.storage.dtype)
    }

    pub fn device(&self) -> Device {
        self.storage.device
    }

    pub fn try_device(&self) -> Result<Device> {
        Ok(self.storage.device)
    }

    /// Whether the elements sit next to each other in memory, in row-major order.
    pub fn is_contiguous(&self) -> bool {
        let mut numel: i64 = 1;
        for d in (0..self.layout.shape.len()).rev() {
            let size = self.layout.shape[d];
            if numel != self.layout.stride[d] as i64 && size != 1 {
                return false;
            }
            numel *= size as i64;
        }
        true
    }

    /// Read the same elements under a new shape, sharing the storage. One dimension may be -1 to
    /// be worked out from the others. A tensor that is not contiguous can still be viewed as long
    /// as no new dimension straddles a gap in its layout.
    pub fn view(&self, shape: &[i32]) -> Result<Tensor> {
        let shape = real_shape(self.numel(), shape)?;
        if self.is_contiguous() {
            let stride = contiguous_strides(&shape);
            return self.with_layout(shape, stride, self.layout.offset);
        }

        // Runs of dimensions that are contiguous with each other, merged into one, from the back.
        let mut merged: Vec<(i32, i32)> = Vec::new();
        for d in (0..self.layout.shape.len()).rev() {
            let (size, stride) = (self.layout.shape[d], self.layout.stride[d]);
            if stride == 0 {
                return Err(Error::invalid("unable to change the view of an expanded tensor"));
            }
            let joins = d + 1 < self.layout.shape.len()
                && self.layout.stride[d + 1] as i64 * self.layout.shape[d + 1] as i64
                    == stride as i64;
            match merged.last_mut() {
                Some(last) if joins => last.0 *= size,
                _ => merged.push((size, stride)),
            }
        }

        // Each merged run split into the requested dimensions that fill it exactly.
        let mut view: Vec<(i32, i32)> = Vec::new();
        let mut requested = shape.iter().rev().peekable();
        for &(size, stride) in &merged {
            let mut numel = 1;
            while let Some(&&n) = requested.peek() {
                if n * numel > size {
                    break;
                }
                view.push((n, numel * stride));
                numel *= n;
                requested.next();
            }
            if numel != size {
                return Err(Error::invalid("unable to get this view of the tensor"));
            }
        }
        view.reverse();
        let (shape, stride) = view.into_iter().unzip();
        self.with_layout(shape, stride, self.layout.offset)
    }

    /// See every dimension of one at `shape`'s size by repeating its element, sharing the storage.
    /// `shape` has as many dimensions as the tensor, and only a dimension of one may grow.
    pub fn expand(&self, shape: &[i32]) -> Result<Tensor> {
        if shape.len() != self.layout.shape.len() {
            return Err(Error::invalid(format!(
                "expand: {:?} has another rank than {:?}",
                shape, self.layout.shape
            )));
        }
        let mut stride = self.layout.stride.clone();
        for d in 0..shape.len() {
            if shape[d] != self.layout.shape[d] {
                if self.layout.shape[d] != 1 {
                    return Err(Error::invalid(format!(
                        "expand: dimension {d} holds {} elements, and only a single one can grow",
                        self.layout.shape[d]
                    )));
                }
                stride[d] = 0;
            }
        }
        self.with_layout(shape.to_vec(), stride, self.layout.offset)
    }

    /// Exchange two dimensions, sharing the storage. The result is usually not contiguous.
    pub fn transpose(&self, dim0: i32, dim1: i32) -> Result<Tensor> {
        let (d0, d1) = (self.real_dim(dim0)?, self.real_dim(dim1)?);
        let mut shape = self.layout.shape.clone();
        let mut stride = self.layout.stride.clone();
        shape.swap(d0, d1);
        stride.swap(d0, d1);
        self.with_layout(shape, stride, self.layout.offset)
    }

    /// Take the half-open range `[begin, end)` of dimension `dim`, sharing the storage.
    ///
    /// Both bounds accept a plain `i32`, negative to count from the back, or [`Bound::End`] to
    /// leave that side alone.
    pub fn slice(
        &self,
        dim: i32,
        begin: impl Into<Bound>,
        end: impl Into<Bound>,
    ) -> Result<Tensor> {
        let d = self.real_dim(dim)?;
        let size = self.layout.shape[d];
        let begin = match begin.into() {
            Bound::At(index) => self.real_index(d, index)?,
            Bound::End => 0,
        };
        let end = match end.into() {
            Bound::At(index) => self.real_index(d, index)?,
            Bound::End => size,
        };
        if begin >= end {
            return Err(Error::invalid(format!(
                "slice: [{begin}, {end}) is not within a dimension of {size}"
            )));
        }

        let mut shape = self.layout.shape.clone();
        shape[d] = end - begin;
        let offset = self.layout.offset + self.layout.stride[d] as i64 * begin as i64;
        self.with_layout(shape, self.layout.stride.clone(), offset)
    }

    /// Take one entry of the first dimension, dropping that dimension.
    pub fn subtensor(&self, index: i32) -> Result<Tensor> {
        let d = self.real_dim(0)?;
        let index = self.real_index(d, index)?;
        if index >= self.layout.shape[0] {
            return Err(Error::invalid(format!(
                "subtensor: {index} is not within a dimension of {}",
                self.layout.shape[0]
            )));
        }
        let offset = self.layout.offset + self.layout.stride[0] as i64 * index as i64;
        self.with_layout(
            self.layout.shape[1..].to_vec(),
            self.layout.stride[1..].to_vec(),
            offset,
        )
    }

    /// Add a dimension of size one at `dim`.
    pub fn unsqueeze(&self, dim: i32) -> Result<Tensor> {
        let rank = self.layout.shape.len();
        let d = if dim == rank as i32 { rank } else { self.real_dim(dim)? };
        let stride = if rank == 0 {
            1
        } else if d == 0 {
            self.layout.stride[0] * self.layout.shape[0]
        } else {
            self.layout.stride[d - 1]
        };

        let mut shapes = self.layout.shape.clone();
        let mut strides = self.layout.stride.clone();
        shapes.insert(d, 1);
        strides.insert(d, stride);
        self.with_layout(shapes, strides, self.layout.offset)
    }

    /// Remove the dimension at `dim`, which must have size one.
    pub fn squeeze(&self, dim: i32) -> Result<Tensor> {
        let d = self.real_dim(dim)?;
        if self.layout.shape[d] != 1 {
            return Err(Error::invalid(format!(
                "squeeze: dimension {dim} holds {} elements, not one",
                self.layout.shape[d]
            )));
        }
        let mut shape = self.layout.shape.clone();
        let mut stride = self.layout.stride.clone();
        shape.remove(d);
        stride.remove(d);
        self.with_layout(shape, stride, self.layout.offset)
    }

    /// Return a contiguous tensor with the same elements, copying only if needed.
    pub fn contiguous(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        let mut out = Tensor::empty(&self.layout.shape, self.dtype(), self.device())?;
        functional::copy(self, &mut out)?;
        Ok(out)
    }

    /// Copy the tensor to another device.
    pub fn to_device(&self, device: Device) -> Result<Tensor> {
        // Already there, so nothing is copied and no operator is asked anything.
        if self.device() == device {
            return Ok(self.clone());
        }

        // A transfer is a plain copy of one run of bytes, so a strided tensor is packed first, on
        // the side it is on -- the host's operators for either kind of host memory.
        let source = self.contiguous()?;
        let dest = Tensor::empty(&source.layout.shape, source.dtype(), device)?;
        let operators = transfer_operators(source.device(), device)?;
        check(unsafe { ffi::fl_transfer(operators, source.raw(), dest.raw()) })?;
        Ok(dest)
    }

    /// Start copying to `device` and return before the bytes have arrived.
    ///
    /// Only from [`Device::CudaHost`] to [`Device::Cuda`]; every other pair is an error rather
    /// than a synchronous copy under an asynchronous name. The tensor is inside the
    /// [`FutureTensor`] that comes back and is had by taking it, which is also what sees the copy
    /// through.
    pub fn to_device_async(&self, device: Device) -> Result<FutureTensor> {
        let mut data: ffi::FlTensorData = std::ptr::null_mut();
        let mut transfer: ffi::FlTransfer = std::ptr::null_mut();
        check(unsafe {
            ffi::fl_transfer_async(self.raw(), device as i32, &mut data, &mut transfer)
        })?;

        // Owned from here, whatever happens next: the storage by its Rc and the copy by the future.
        let storage = Rc::new(Storage {
            raw: data,
            dtype: self.dtype(),
            device,
            read_only: false,
            _owner: None,
        });
        let pending = PendingTransfer(transfer);
        let shape = self.layout.shape.clone();
        let tensor = Tensor::over(storage, shape.clone(), contiguous_strides(&shape), 0)?;
        Ok(FutureTensor {
            pending,
            tensor,
            source: RefCell::new(Some(self.clone())),
        })
    }

    /// Convert the elements to another data type.
    pub fn cast(&self, dtype: DType) -> Result<Tensor> {
        if self.dtype() == dtype {
            return Ok(self.clone());
        }
        let out = Tensor::empty(&self.layout.shape, dtype, self.device())?;
        let operators = operators_of(self)?;
        check(unsafe { ffi::fl_cast(operators, self.raw(), out.raw()) })?;
        Ok(out)
    }

    /// Number of bytes the elements occupy once packed together.
    pub fn nbytes(&self) -> Result<i64> {
        Ok(self.dtype().total_size(self.numel()))
    }

    /// Copy the elements out in row-major order, bringing them back from the device and packing
    /// them first if needed.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        self.to_vec(DType::Float)
    }

    /// Copy 64-bit integer elements out in row-major order.
    pub fn to_vec_i64(&self) -> Result<Vec<i64>> {
        self.to_vec(DType::Long)
    }

    /// Copy 32-bit integer elements out in row-major order.
    pub fn to_vec_i32(&self) -> Result<Vec<i32>> {
        self.to_vec(DType::Int32)
    }

    /// Copy byte elements out in row-major order.
    pub fn to_vec_u8(&self) -> Result<Vec<u8>> {
        self.to_vec(DType::UInt8)
    }

    /// Copy boolean elements out in row-major order, as the comparisons in
    /// [`functional`] produce them.
    pub fn to_vec_bool(&self) -> Result<Vec<bool>> {
        // Read the bytes rather than `bool` itself: a byte that is neither 0 nor 1 is a valid u8
        // but not a valid bool, and nothing here can promise the library never writes one.
        let bytes: Vec<u8> = self.to_vec(DType::Bool)?;
        Ok(bytes.into_iter().map(|byte| byte != 0).collect())
    }

    fn to_vec<T: Default + Clone>(&self, expected: DType) -> Result<Vec<T>> {
        let dtype = self.try_dtype()?;
        if dtype != expected {
            return Err(Error::unsupported(format!(
                "tensor holds {dtype:?}, not {expected:?}; cast it first"
            )));
        }

        let nbytes = self.nbytes()? as usize;
        let mut values = vec![T::default(); nbytes / std::mem::size_of::<T>()];
        if nbytes == 0 {
            return Ok(values);
        }

        // Packed where it lies and moved afterwards, rather than the other way round: a transfer
        // reads one contiguous run, so a strided tensor is packed on the device it is on.
        let mut source = self.contiguous()?;
        if !matches!(source.device(), Device::Cpu | Device::CudaHost) {
            source = source.to_device(Device::Cpu)?;
        }
        let bytes = source.host_bytes()?;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), values.as_mut_ptr() as *mut u8, nbytes)
        };
        Ok(values)
    }
}

/// The handle of a copy still on its way, destroyed once whatever holds it is done with it.
struct PendingTransfer(ffi::FlTransfer);

impl Drop for PendingTransfer {
    fn drop(&mut self) {
        // Safety: made by a successful C call and destroyed exactly once, here.
        unsafe { ffi::fl_transfer_destroy(self.0) };
    }
}

/// A copy that is still on its way, and the tensor it is filling.
///
/// Made by [`Tensor::to_device_async`]. The tensor comes out of [`FutureTensor::take`] or
/// [`FutureTensor::take_sync`] and there is no other way to reach it, so a tensor whose bytes
/// have not arrived cannot be handed to an operator. Dropping one without taking it is a fetch
/// that turned out not to be wanted: it costs the bandwidth already spent and nothing else.
pub struct FutureTensor {
    // Declared first so that it goes first: the copy's handle before the storage it writes.
    pending: PendingTransfer,
    tensor: Tensor,
    /// What the copy reads, held until it has been seen through.
    source: RefCell<Option<Tensor>>,
}

impl FutureTensor {
    /// Order the work that follows behind the copy and hand the tensor over.
    ///
    /// The caller does not stop here; what it arranges is that work enqueued from now on runs
    /// after the copy. Take it where the tensor is about to be used rather than where the copy
    /// was started, since that is where the dependency lands.
    pub fn take(&self) -> Result<Tensor> {
        check(unsafe { ffi::fl_transfer_wait(self.pending.0) })?;
        self.source.borrow_mut().take();
        Ok(self.tensor.clone())
    }

    /// The same, except that it does not return until the copy has finished. For a caller about
    /// to read the bytes itself rather than enqueue work that reads them.
    pub fn take_sync(&self) -> Result<Tensor> {
        check(unsafe { ffi::fl_transfer_wait_sync(self.pending.0) })?;
        self.source.borrow_mut().take();
        Ok(self.tensor.clone())
    }
}

impl fmt::Debug for FutureTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FutureTensor").finish_non_exhaustive()
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape())
            .field("dtype", &self.try_dtype().ok())
            .field("device", &self.try_device().ok())
            .field("contiguous", &self.is_contiguous())
            .finish()
    }
}
