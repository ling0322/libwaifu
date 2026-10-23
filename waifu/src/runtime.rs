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

//! Where a run goes: which device, and where the weights wait.

use crate::flint::{MemoryBudget, MemorySnapshot};
use crate::Device;

/// Where a run goes: a device.
///
/// Its own type rather than a [`Device`] because it is what a run is asked of, and the weights a
/// run keeps on the device are the runtime's to say. Every device keeps what fits there, copies
/// the rest over as each is read, and lets weights go when a run runs short, so there is no second
/// name for a device with the weights somewhere else.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Runtime {
    device: Device,
}

impl Runtime {
    /// Every place a run can go, in the order the flag's own usage lists them.
    pub const ALL: [Runtime; 4] = [
        Runtime::on(Device::Cpu),
        Runtime::on(Device::Cuda),
        Runtime::on(Device::Metal),
        Runtime::on(Device::Vulkan),
    ];

    /// A device, with as many of the weights on it as fit, which each device makes what it can of
    /// -- see [`Weights::from_files`](crate::Weights::from_files).
    ///
    /// On Metal, CUDA and Vulkan the smallest weights fill most of what the memory limit allows --
    /// all of the device where there is none -- as the model is read; every other weight is copied
    /// into the device's memory at the instruction that reads it, and a run that runs short lets
    /// weights on the device go. So a model that fits is all on the device, and one larger than
    /// the card still draws. See [`Weights`](crate::Weights) and [`set_memory_limit`]. On the
    /// processor every weight is where it is read.
    pub const fn on(device: Device) -> Runtime {
        Runtime { device }
    }

    pub fn device(self) -> Device {
        self.device
    }

    /// How many bytes of a model's weights to keep on the device: as many as fit, always. A
    /// smaller share is a memory limit's to ask for -- see [`set_memory_limit`] -- rather than a
    /// runtime's.
    pub fn device_weight_bytes(self) -> u64 {
        u64::MAX
    }

    /// What it is called on the command line and on screen.
    pub fn name(self) -> &'static str {
        self.device.name()
    }
}

/// Where the model should run, as `-device` spells it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DeviceOption {
    /// The first accelerator there is a device for -- CUDA, then Metal, then Vulkan -- and the CPU
    /// otherwise.
    #[default]
    Auto,
    Cpu,
    Cuda,
    Metal,
    Vulkan,
}

impl DeviceOption {
    /// The runtime this actually means, which is the one question `auto` leaves open.
    pub fn resolve(self) -> Runtime {
        match self {
            DeviceOption::Cpu => Runtime::on(Device::Cpu),
            DeviceOption::Cuda => Runtime::on(Device::Cuda),
            DeviceOption::Metal => Runtime::on(Device::Metal),
            DeviceOption::Vulkan => Runtime::on(Device::Vulkan),
            DeviceOption::Auto => {
                // At most one of CUDA and Metal is ever built, so the order between those two only
                // decides which check runs first. Vulkan comes last because it runs on the same
                // cards as either, through kernels of its own that are not the fastest those
                // cards have.
                if Device::Cuda.is_available() {
                    Runtime::on(Device::Cuda)
                } else if Device::Metal.is_available() {
                    Runtime::on(Device::Metal)
                } else if Device::Vulkan.is_available() {
                    Runtime::on(Device::Vulkan)
                } else {
                    Runtime::on(Device::Cpu)
                }
            }
        }
    }
}

/// Limit the memory of the device `device` runs on -- the tensors of whatever runs there, the
/// weights kept on it among them -- to `bytes`; 0 for no limit but the device's.
///
/// The process's, not an engine's: there is one allocator per device. A model read afterwards
/// keeps its smallest weights on the device within the limit, and a run that then runs short lets
/// weights on the device go. Metal, CUDA and Vulkan have a limit; the processor does not.
pub fn set_memory_limit(device: DeviceOption, bytes: u64) -> Result<(), String> {
    let device = device.resolve().device();
    MemorySnapshot::set_limit(device, bytes.min(i64::MAX as u64) as i64)
        .map_err(|error| format!("{} takes no memory limit: {error}", device.name()))?;
    crate::log::write(
        crate::log::Level::Info,
        &match bytes {
            0 => format!("no memory limit on {}", device.name()),
            _ => format!(
                "memory limit on {}: {} MB",
                device.name(),
                bytes / 1_000_000
            ),
        },
    );
    Ok(())
}

/// What the allocator of `device` may still take, where this build and this machine can run there
/// and it says: see [`MemoryBudget`]. `None` for the processor, which keeps no count.
///
/// The first question to CUDA builds a context on the card, a few hundred megabytes of it -- which
/// is why this is only asked of a device a run can go to.
pub fn remaining(device: Device, available: &[Runtime]) -> Option<u64> {
    if device == Device::Cpu || !available.contains(&Runtime::on(device)) {
        return None;
    }
    match MemoryBudget::capture(device) {
        Ok(budget) if budget.capacity > 0 => Some(budget.remaining()),
        _ => None,
    }
}

impl Runtime {
    /// Every place a run can go that this build and this machine can take it.
    ///
    /// The first call starts the tensor library, which says what it found on stderr.
    pub fn available() -> Vec<Runtime> {
        Runtime::ALL
            .into_iter()
            .filter(|runtime| runtime.device().is_available())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_device_has_one_name_and_keeps_what_fits() {
        for option in [
            DeviceOption::Cpu,
            DeviceOption::Cuda,
            DeviceOption::Metal,
            DeviceOption::Vulkan,
        ] {
            assert_eq!(option.resolve().device_weight_bytes(), u64::MAX);
        }
        assert_eq!(DeviceOption::Metal.resolve().device(), Device::Metal);

        let names: Vec<&str> = Runtime::ALL.iter().map(|runtime| runtime.name()).collect();
        assert_eq!(names, ["cpu", "cuda", "metal", "vulkan"]);
    }
}
