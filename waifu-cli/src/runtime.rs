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
//!
//! What this machine can run on is libwaifu's to say, through `waifu_devices_json`; which of them
//! `auto` means is this tool's, and it is here.

/// A device a run can be sent to, as libwaifu names them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Device {
    Cpu,
    Cuda,
    Metal,
    Vulkan,
}

impl Device {
    pub fn name(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
            Device::Metal => "metal",
            Device::Vulkan => "vulkan",
        }
    }
}

use crate::c::{self, WaifuDevice};

/// Where a run goes: a device, and where the weights wait between the steps that read them.
///
/// One name rather than two answers. As a device name -- `cuda_cpu_offload`, cuda with none of the
/// weights kept on the card -- the pair that cannot be built cannot be said either, and there is
/// one list on screen instead of two. Every other device needs no second name: libwaifu keeps on
/// it what fits and lets weights go when a run runs short.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Runtime {
    device: Device,
    /// None of the weights kept on the card, each moved onto it as it is used: cuda_cpu_offload.
    offload: bool,
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
    /// instruction that reads it, so that the card holds nothing of the model between steps.
    pub const CUDA_CPU_OFFLOAD: Runtime = Runtime {
        device: Device::Cuda,
        offload: true,
    };

    /// A device with as many of the weights on it as fit, which is what every name but the
    /// offload one means.
    pub const fn on(device: Device) -> Runtime {
        Runtime {
            device,
            offload: false,
        }
    }

    pub fn device(self) -> Device {
        self.device
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn offload(self) -> bool {
        self.offload
    }

    /// What it is called on the command line and on screen.
    pub fn name(self) -> &'static str {
        match self.offload {
            false => self.device.name(),
            true => "cuda_cpu_offload",
        }
    }

    /// What the drawing screen's device list says beside the name of one this machine can use.
    pub fn about(self) -> &'static str {
        match self.offload {
            false => "ready",
            // The row that has to say more than that. It is the slow answer, and the slow answer
            // picked by someone who was not told is a run that looks broken.
            true => "ready, slower: keeps none of the model on the card",
        }
    }
}

/// Where the model should run, as `-device` spells it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceOption {
    /// The first accelerator there is a device for -- CUDA, then Metal, then Vulkan -- and the CPU
    /// otherwise.
    Auto,
    Cpu,
    Cuda,
    CudaCpuOffload,
    Metal,
    Vulkan,
}

impl DeviceOption {
    /// The runtime this actually means on a machine that can run on `available`, which is the
    /// one question `auto` leaves open.
    pub fn resolve(self, available: &[Runtime]) -> Runtime {
        match self {
            DeviceOption::Cpu => Runtime::on(Device::Cpu),
            DeviceOption::Cuda => Runtime::on(Device::Cuda),
            DeviceOption::CudaCpuOffload => Runtime::CUDA_CPU_OFFLOAD,
            DeviceOption::Metal => Runtime::on(Device::Metal),
            DeviceOption::Vulkan => Runtime::on(Device::Vulkan),
            // Never the offload one: the card keeps what fits of a model and lets weights go when
            // a run runs short, which is faster than keeping none of it there.
            //
            // At most one of CUDA and Metal is ever built, so the order between those two only
            // decides which check runs first. Vulkan comes last because it runs on the same cards
            // as either, through kernels of its own that are not the fastest those cards have.
            DeviceOption::Auto => [Device::Cuda, Device::Metal, Device::Vulkan]
                .into_iter()
                .map(Runtime::on)
                .find(|runtime| available.contains(runtime))
                .unwrap_or(Runtime::on(Device::Cpu)),
        }
    }
}

impl Runtime {
    /// The device libwaifu's engine is started on for this.
    pub fn waifu_device(self) -> WaifuDevice {
        match (self.device, self.offload) {
            (Device::Cpu, _) => WaifuDevice::WAIFU_DEVICE_CPU,
            (Device::Cuda, false) => WaifuDevice::WAIFU_DEVICE_CUDA,
            (Device::Cuda, true) => WaifuDevice::WAIFU_DEVICE_CUDA_CPU_OFFLOAD,
            (Device::Metal, _) => WaifuDevice::WAIFU_DEVICE_METAL,
            (Device::Vulkan, _) => WaifuDevice::WAIFU_DEVICE_VULKAN,
        }
    }

    /// The one called `name`, as `waifu_devices_json` and the -device flag name them.
    pub fn named(name: &str) -> Option<Runtime> {
        Runtime::ALL
            .into_iter()
            .find(|runtime| runtime.name() == name)
    }
}

/// What this machine can run on, asked of libwaifu.
///
/// The first time CUDA is asked it builds a context on the card, a few hundred megabytes of it,
/// and the tensor library says what hardware it found: so this is asked once, before any screen
/// is up.
pub fn measure() -> Vec<Runtime> {
    let Some(said) = c::waifu_json(c::waifu_devices_json()) else {
        return vec![Runtime::on(Device::Cpu)];
    };
    let mut available = Vec::new();
    for row in said["devices"].as_array().into_iter().flatten() {
        let Some(runtime) = row["name"].as_str().and_then(Runtime::named) else {
            continue;
        };
        if row["available"].as_bool() == Some(true) {
            available.push(runtime);
        }
    }
    if available.is_empty() {
        available.push(Runtime::on(Device::Cpu));
    }
    available
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_is_the_fastest_device_this_machine_has_and_never_the_offload_one() {
        let cpu = Runtime::on(Device::Cpu);
        let cuda = Runtime::on(Device::Cuda);
        let metal = Runtime::on(Device::Metal);
        let vulkan = Runtime::on(Device::Vulkan);

        // However large the model: the card keeps what fits of it and lets weights go.
        assert_eq!(DeviceOption::Auto.resolve(&Runtime::ALL), cuda);
        assert_eq!(DeviceOption::Auto.resolve(&[cpu, vulkan]), vulkan);
        assert_eq!(DeviceOption::Auto.resolve(&[cpu, metal]), metal);
        assert_eq!(DeviceOption::Auto.resolve(&[cpu]), cpu);

        // A device that was named is the device, whether or not it is here.
        assert_eq!(
            DeviceOption::CudaCpuOffload.resolve(&[cpu]),
            Runtime::CUDA_CPU_OFFLOAD
        );
    }
}
