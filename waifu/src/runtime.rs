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

use crate::flint::MemorySnapshot;
use crate::Device;

/// Where a run goes: a device, and where the weights wait between the steps that read them.
///
/// One name rather than two answers. Where the weights wait is a question about a card the model
/// does not fit on, so it has one device it can be asked of and one reason to ask it, and asking
/// it beside the device made a second box whose rows were struck out three times out of four. As
/// a device name -- `cuda_cpu_offload`, cuda with the weights kept on the host -- the pair that
/// cannot be built cannot be said either, and there is one list on screen instead of two. Metal
/// needs no second name: it keeps on the device what fits and the rest in the model's files.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Runtime {
    device: Device,
    /// How many bytes of a model's weights to keep on the device: see
    /// [`Weights::from_files`](crate::Weights::from_files).
    device_weight_bytes: u64,
}

impl Runtime {
    /// Every place a run can go, in the order the flag's own usage lists them.
    pub const ALL: [Runtime; 5] = [
        Runtime::on(Device::Cpu),
        Runtime::on(Device::Cuda),
        Runtime::CUDA_CPU_OFFLOAD,
        Runtime::on(Device::Metal),
        Runtime::on(Device::Vulkan),
    ];

    /// Cuda, with the package held on the host and each weight moved onto the card at the
    /// instruction that reads it. For a card the model does not fit on.
    pub const CUDA_CPU_OFFLOAD: Runtime = Runtime {
        device: Device::Cuda,
        device_weight_bytes: 0,
    };

    /// A device, with as many of the weights on it as fit, which each device makes what it can of
    /// -- see [`Weights::from_files`](crate::Weights::from_files).
    ///
    /// On Metal the smallest weights fill what the memory limit leaves -- the device's budget where
    /// there is none -- as the model is read; every other weight is read off the disk into the
    /// device's memory at the instruction that reads it, and a run that runs short moves weights
    /// back off the device. So a model that fits is all on the device, and one larger than the
    /// machine still draws. See [`Weights`](crate::Weights) and [`set_memory_limit`]. Everywhere
    /// else, as things are, every weight is on the device.
    pub const fn on(device: Device) -> Runtime {
        Runtime {
            device,
            device_weight_bytes: u64::MAX,
        }
    }

    pub fn device(self) -> Device {
        self.device
    }

    pub fn device_weight_bytes(self) -> u64 {
        self.device_weight_bytes
    }

    /// What it is called on the command line and on screen.
    pub fn name(self) -> &'static str {
        match self == Runtime::CUDA_CPU_OFFLOAD {
            true => "cuda_cpu_offload",
            false => self.device.name(),
        }
    }

    /// What the drawing screen's device list says beside the name of one this machine can use.
    pub fn about(self) -> &'static str {
        match self == Runtime::CUDA_CPU_OFFLOAD {
            // The row that has to say more than that. It is the slow answer, and the slow answer
            // picked by someone who was not told is a run that looks broken.
            true => "ready, slower: for a model larger than the card",
            false => "ready",
        }
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
    CudaCpuOffload,
    Metal,
    Vulkan,
}

impl DeviceOption {
    /// The runtime this actually means, which is the one question `auto` leaves open.
    pub fn resolve(self) -> Runtime {
        match self {
            DeviceOption::Cpu => Runtime::on(Device::Cpu),
            DeviceOption::Cuda => Runtime::on(Device::Cuda),
            DeviceOption::CudaCpuOffload => Runtime::CUDA_CPU_OFFLOAD,
            DeviceOption::Metal => Runtime::on(Device::Metal),
            DeviceOption::Vulkan => Runtime::on(Device::Vulkan),
            // Never an offload one: with no model in hand there is nothing to measure against
            // the card. Which one suits a model is the caller's: see `waifu_devices_json`.
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

/// How much of each card is free, where the card said. `None` is a card that was not asked or
/// did not answer, which is taken as room enough: that is what `auto` assumed before it measured.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Room {
    pub cuda: Option<u64>,
    pub vulkan: Option<u64>,
}

impl Room {
    /// Asks the cards among `available` how much of them is free.
    ///
    /// Not Metal: that backend has no snapshot, and asking for one ends the process. Its memory is
    /// the machine's, so there is no separate card for a model not to fit on.
    ///
    /// The first question to CUDA builds a context on the card, a few hundred megabytes of it --
    /// which is why this is only asked for `auto`, where a card is about to be used anyway.
    pub fn measure(available: &[Runtime]) -> Room {
        let free = |device: Device| match available.contains(&Runtime::on(device)) {
            true => match MemorySnapshot::capture(device) {
                Ok(memory) if memory.total > 0 => Some(memory.free.max(0) as u64),
                _ => None,
            },
            false => None,
        };
        Room {
            cuda: free(Device::Cuda),
            vulkan: free(Device::Vulkan),
        }
    }
}

/// Limit the memory of the device `device` runs on -- the tensors of whatever runs there, the
/// weights an offload runtime keeps on it, and what it keeps for reuse -- to `bytes`; 0 for no
/// limit but the device's.
///
/// The process's, not an engine's: there is one allocator per device. Metal fills what the limit
/// leaves with weights when it reads a model, and a run that then runs short moves weights back
/// off the device. Only Metal has a limit.
pub fn set_memory_limit(device: DeviceOption, bytes: u64) -> Result<(), String> {
    let device = device.resolve().device();
    MemorySnapshot::set_limit(device, bytes.min(i64::MAX as u64) as i64)
        .map_err(|error| format!("{} takes no memory limit: {error}", device.name()))
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
    fn metal_fills_what_it_has_and_cuda_keeps_its_two_ways() {
        // Metal keeps on the device what fits and the rest in the files, under one name.
        let metal = DeviceOption::Metal.resolve();
        assert_eq!(metal.device(), Device::Metal);
        assert_eq!(metal.device_weight_bytes(), u64::MAX);
        assert_eq!(metal.name(), "metal");
        assert_eq!(metal.about(), "ready");

        // Cuda's two names are as they were: the card -- asked to fill it, which a card makes
        // all of the weights on it -- or nothing on it.
        assert_eq!(DeviceOption::Cuda.resolve().device_weight_bytes(), u64::MAX);
        assert_eq!(
            DeviceOption::CudaCpuOffload.resolve().device_weight_bytes(),
            0
        );

        let names: Vec<&str> = Runtime::ALL.iter().map(|runtime| runtime.name()).collect();
        assert_eq!(
            names,
            ["cpu", "cuda", "cuda_cpu_offload", "metal", "vulkan"]
        );
    }
}
