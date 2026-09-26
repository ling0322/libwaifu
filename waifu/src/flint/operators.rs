//! The operators of one device: the backend a call actually runs on.
//!
//! The C interface asks for these rather than working them out from the tensors it was handed, so
//! something on this side has to hold them. That is what this module is: [`Operators::create`]
//! makes a handle, and everything else here is about which handle a given call belongs to.
//!
//! The handles themselves are made once per device per thread and kept for the life of the
//! thread, because the operators behind them are the library's and shared by the whole process --
//! a second handle is another reference rather than another backend. A [`Tensor`] is neither
//! `Send` nor `Sync` for the same reason the operators are not, so the per-thread cache never has
//! to be shared.
//!
//! [`Tensor`]: super::Tensor

use std::cell::RefCell;
use std::fmt;
use std::marker::PhantomData;

use super::{check, ffi, init, Device, Error, Result};

/// One slot per [`Device`], which the enum numbers from zero.
const DEVICE_COUNT: usize = 5;

/// A handle on the operators of one device.
///
/// Dropping it releases the handle; the operators outlive it, since they belong to the library.
pub struct Operators {
    raw: ffi::FlOperators,
    /// Keeps the type off `Send`/`Sync`, since the operators behind it are not ready for either.
    _not_sync: PhantomData<*const ()>,
}

impl Operators {
    /// The operators of `device`.
    ///
    /// A device this build or this machine has no operators for is an error;
    /// [`Device::is_available`] is the way to ask about that without failing. [`Device::CudaHost`]
    /// names memory rather than a processor and so has none of its own: page-locked host memory
    /// is made by the CUDA operators, which are the ones that know how.
    pub fn create(device: Device) -> Result<Operators> {
        init();
        let mut raw: ffi::FlOperators = std::ptr::null_mut();
        check(unsafe { ffi::fl_operators_create(device as i32, &mut raw) })?;
        debug_assert!(!raw.is_null(), "a successful call must produce a handle");
        Ok(Operators {
            raw,
            _not_sync: PhantomData,
        })
    }

    /// The device these compute on, which is what [`Operators::create`] was given.
    pub fn device(&self) -> Result<Device> {
        let mut raw: ffi::FlDeviceType = 0;
        check(unsafe { ffi::fl_operators_get_device(self.raw, &mut raw) })?;
        Device::from_raw(raw)
    }
}

impl fmt::Debug for Operators {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.device() {
            Ok(device) => write!(f, "Operators({})", device.name()),
            Err(error) => write!(f, "Operators(<{error}>)"),
        }
    }
}

impl Drop for Operators {
    fn drop(&mut self) {
        unsafe { ffi::fl_operators_destroy(self.raw) };
    }
}

thread_local! {
    /// One handle per device, made on first use and kept for the life of the thread.
    static PER_DEVICE: RefCell<[Option<Operators>; DEVICE_COUNT]> =
        RefCell::new([None, None, None, None, None]);
}

/// The handle this thread uses for `device`, making it if this is the first call.
///
/// What comes back is borrowed from the thread-local cache, which nothing ever clears, so it
/// stays valid until the thread ends. It is returned rather than passed to a closure so that a
/// caller may hold two of them -- a transfer has two ends -- without nesting the borrow.
pub(crate) fn raw_operators(device: Device) -> Result<ffi::FlOperators> {
    PER_DEVICE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let slot = &mut cache[device as usize];
        if slot.is_none() {
            *slot = Some(Operators::create(device)?);
        }
        Ok(slot.as_ref().expect("just filled").raw)
    })
}

/// The handle for the device `tensor` is on.
pub(crate) fn operators_of(tensor: &super::Tensor) -> Result<ffi::FlOperators> {
    let device = tensor.try_device()?;
    if device == Device::CudaHost {
        return Err(Error::unsupported(
            "page-locked host memory carries no operators: it is where weights wait to be copied \
             to the GPU, not somewhere to compute"
                .to_string(),
        ));
    }
    raw_operators(device)
}

/// The handle that owns a transfer between `from` and `to`.
///
/// It is the accelerator's on whichever end is not the CPU: that side is the one that knows how
/// either end was allocated, and page-locked host memory is the CUDA driver's however much it
/// looks like the CPU's. Two ends on the same accelerator are its own.
pub(crate) fn transfer_operators(from: Device, to: Device) -> Result<ffi::FlOperators> {
    let cuda_side = |device: Device| matches!(device, Device::Cuda | Device::CudaHost);

    if cuda_side(from) || cuda_side(to) {
        raw_operators(Device::Cuda)
    } else if from == Device::Metal || to == Device::Metal {
        raw_operators(Device::Metal)
    } else if from == Device::Vulkan || to == Device::Vulkan {
        raw_operators(Device::Vulkan)
    } else {
        raw_operators(Device::Cpu)
    }
}

/// The handle that reads a tensor's bytes back out to the host.
///
/// Host memory is the CPU's to pack and to read, whoever page-locked it. Anything on a device has
/// to cross the bus first, and that transfer is the device's own.
pub(crate) fn readback_operators(tensor: &super::Tensor) -> Result<ffi::FlOperators> {
    match tensor.try_device()? {
        Device::Cpu | Device::CudaHost => raw_operators(Device::Cpu),
        device => raw_operators(device),
    }
}

/// The handle that owns a copy between two tensors that are already where they belong.
///
/// Host memory is host memory whoever page-locked it, so a copy with host at both ends is the
/// CPU's to make. What this may not do is cross the bus; that is [`Tensor::to_device`].
///
/// [`Tensor::to_device`]: super::Tensor::to_device
pub(crate) fn copy_operators(src: Device, dest: Device) -> Result<ffi::FlOperators> {
    let is_host = |device: Device| matches!(device, Device::Cpu | Device::CudaHost);

    if is_host(src) && is_host(dest) {
        raw_operators(Device::Cpu)
    } else if src == dest {
        raw_operators(src)
    } else {
        Err(Error::unsupported(format!(
            "a copy does not cross devices: {} to {} is to_device()",
            src.name(),
            dest.name()
        )))
    }
}
