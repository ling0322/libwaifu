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
use crate::{Device, Residency};

/// Where a run goes: a device, and where the weights wait between the steps that read them.
///
/// One name rather than two answers. Where the weights wait is a question about a card the model
/// does not fit on, so it has one device it can be asked of and one reason to ask it, and asking
/// it beside the device made a second box whose rows were struck out three times out of four. As
/// a device name -- `cuda_cpu_offload`, cuda with the weights kept on the host, and
/// `metal_disk_offload`, metal with them left on the disk -- the pair that cannot be built cannot
/// be said either, and there is one list on screen instead of two.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Runtime {
    device: Device,
    residency: Residency,
}

impl Runtime {
    /// Every place a run can go, in the order the flag's own usage lists them.
    pub const ALL: [Runtime; 6] = [
        Runtime::on(Device::Cpu),
        Runtime::on(Device::Cuda),
        Runtime::CUDA_CPU_OFFLOAD,
        Runtime::on(Device::Metal),
        Runtime::METAL_DISK_OFFLOAD,
        Runtime::on(Device::Vulkan),
    ];

    /// Cuda, with the package held on the host and each weight moved onto the card at the
    /// instruction that reads it. For a card the model does not fit on.
    pub const CUDA_CPU_OFFLOAD: Runtime = Runtime {
        device: Device::Cuda,
        residency: Residency::Hybrid {
            device_weight_bytes: 0,
        },
    };

    /// Metal, with the package left in its file and each weight read off the disk into the
    /// device's memory at the instruction that reads it. Metal's memory is the machine's, so this
    /// is for a model larger than the machine; see [`Weights`](crate::Weights).
    pub const METAL_DISK_OFFLOAD: Runtime = Runtime {
        device: Device::Metal,
        residency: Residency::Hybrid {
            device_weight_bytes: 0,
        },
    };

    /// A device with the weights on it, which is what every name but the offload ones means.
    pub const fn on(device: Device) -> Runtime {
        Runtime {
            device,
            residency: Residency::Device,
        }
    }

    pub fn device(self) -> Device {
        self.device
    }

    pub fn residency(self) -> Residency {
        self.residency
    }

    /// This offload runtime with the smallest weights kept on the device, as many as fit in
    /// `bytes`: [`Residency::Hybrid`]. Nothing is the runtime as it is.
    ///
    /// Refused for a runtime that is not an offload one, since it keeps the whole model on the
    /// device already and there is nothing a number of bytes would change.
    pub fn with_device_weight_bytes(self, bytes: u64) -> Result<Runtime, String> {
        match (self.residency, bytes) {
            (_, 0) => Ok(self),
            (Residency::Hybrid { .. }, _) => Ok(Runtime {
                residency: Residency::Hybrid {
                    device_weight_bytes: bytes,
                },
                ..self
            }),
            (Residency::Device, _) => Err(format!(
                "{} keeps every weight on the device already: the bytes of weights to keep there \
                 are for cuda_cpu_offload and metal_disk_offload",
                self.name()
            )),
        }
    }

    /// What it is called on the command line and on screen. A hybrid runtime goes by its offload
    /// device's name: it is that device with a number beside it.
    pub fn name(self) -> &'static str {
        match (self.device, self.residency) {
            (device, Residency::Device) => device.name(),
            (Device::Metal, Residency::Hybrid { .. }) => "metal_disk_offload",
            (_, Residency::Hybrid { .. }) => "cuda_cpu_offload",
        }
    }

    /// What the drawing screen's device list says beside the name of one this machine can use.
    pub fn about(self) -> &'static str {
        match (self.device, self.residency) {
            (_, Residency::Device) => "ready",
            // The rows that have to say more than that. They are the slow answers, and the slow
            // answer picked by someone who was not told is a run that looks broken.
            (Device::Metal, Residency::Hybrid { .. }) => {
                "ready, slower: for a model larger than the memory"
            }
            (_, Residency::Hybrid { .. }) => "ready, slower: for a model larger than the card",
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
    MetalDiskOffload,
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
            DeviceOption::MetalDiskOffload => Runtime::METAL_DISK_OFFLOAD,
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
