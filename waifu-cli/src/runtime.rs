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

//! Where a run goes: which device, where the weights wait, and how to choose for a model.
//!
//! What this machine can run on and how much room its cards have are libwaifu's to say, through
//! `waifu_devices_json`; which of them suits a model is this tool's, and it is all here.

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

/// Where the weights wait between the steps that read them: on the device, or -- for a model
/// larger than it -- on the host or the disk, each moved over at the step that reads it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Residency {
    Device,
    LowVram,
}

use crate::c::{self, WaifuDevice};

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
        residency: Residency::LowVram,
    };

    /// Metal, with the package left in its file and each weight read off the disk into the
    /// device's memory at the instruction that reads it. Metal's memory is the machine's, so this
    /// is for a model larger than the machine.
    pub const METAL_DISK_OFFLOAD: Runtime = Runtime {
        device: Device::Metal,
        residency: Residency::LowVram,
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

    /// What it is called on the command line and on screen.
    pub fn name(self) -> &'static str {
        match (self.device, self.residency) {
            (device, Residency::Device) => device.name(),
            (Device::Metal, Residency::LowVram) => "metal_disk_offload",
            (_, Residency::LowVram) => "cuda_cpu_offload",
        }
    }

    /// What the drawing screen's device list says beside the name of one this machine can use.
    pub fn about(self) -> &'static str {
        match (self.device, self.residency) {
            (_, Residency::Device) => "ready",
            // The rows that have to say more than that. They are the slow answers, and the slow
            // answer picked by someone who was not told is a run that looks broken.
            (Device::Metal, Residency::LowVram) => {
                "ready, slower: for a model larger than the memory"
            }
            (_, Residency::LowVram) => "ready, slower: for a model larger than the card",
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
    MetalDiskOffload,
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
            DeviceOption::MetalDiskOffload => Runtime::METAL_DISK_OFFLOAD,
            DeviceOption::Vulkan => Runtime::on(Device::Vulkan),
            // Never an offload one: with no model in hand there is nothing to measure against
            // the card. [`Runtime::best`] is the answer once there is.
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

/// How much of each card is free, where the card said. `None` is a card that was not asked or
/// did not answer, which is taken as room enough: that is what `auto` assumed before it measured.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Room {
    pub cuda: Option<u64>,
    pub vulkan: Option<u64>,
}

/// What a model of `weights` bytes wants of a card to run with its weights on it: the weights,
/// and room beside them for what a run makes -- the latents, the attention, the decoded picture.
///
/// A guess, and a generous one. Too small and the run aborts out of memory halfway through a
/// picture; too large and a model that would have fitted runs offloaded, which is slower but
/// finishes. SDXL's 7 GB comes to 9.7, and Qwen-Image's 15 GB of fp8 to 18.6, which is what puts
/// the one on a sixteen gigabyte card and the other beside it.
pub fn wanted_on_the_card(weights: u64) -> u64 {
    const GIB: u64 = 1 << 30;
    weights + weights / 10 + 2 * GIB
}

impl Runtime {
    /// What a model of `weights` bytes wants of this runtime's card and what the card has free,
    /// where the one is more than the other: a run there is likely to abort out of memory.
    ///
    /// Only a runtime that puts the weights on a card that was measured can be short. The offload
    /// one keeps them on the host, Metal's memory is the machine's, and a size or a card that is
    /// not known is taken as one that fits.
    pub fn short_of(self, weights: Option<u64>, room: Room) -> Option<(u64, u64)> {
        let free = match (self.device, self.residency) {
            (Device::Cuda, Residency::Device) => room.cuda,
            (Device::Vulkan, Residency::Device) => room.vulkan,
            _ => None,
        }?;
        let wanted = wanted_on_the_card(weights?);
        (wanted > free).then_some((wanted, free))
    }

    /// The fastest of `available` a model of `weights` bytes will run on, given `room`.
    ///
    /// CUDA where the model fits on the card, and CUDA with the weights on the host where it does
    /// not: slower, but far faster than the processor, and it finishes where the other aborts.
    /// Then Metal, whose memory is the machine's. Then Vulkan, which has no offload of its own and
    /// so is only the answer where the model fits. The processor otherwise. A size that is not
    /// known is taken as one that fits.
    pub fn best(weights: Option<u64>, available: &[Runtime], room: Room) -> Runtime {
        let cuda = Runtime::on(Device::Cuda);
        let metal = Runtime::on(Device::Metal);
        let vulkan = Runtime::on(Device::Vulkan);

        if available.contains(&cuda) {
            if cuda.short_of(weights, room).is_none() {
                return cuda;
            }
            if available.contains(&Runtime::CUDA_CPU_OFFLOAD) {
                return Runtime::CUDA_CPU_OFFLOAD;
            }
        }
        if available.contains(&metal) {
            return metal;
        }
        if available.contains(&vulkan) && vulkan.short_of(weights, room).is_none() {
            return vulkan;
        }
        Runtime::on(Device::Cpu)
    }
}

impl Runtime {
    /// The device libwaifu's engine is started on for this.
    pub fn waifu_device(self) -> WaifuDevice {
        match (self.device, self.residency) {
            (Device::Cpu, _) => WaifuDevice::WAIFU_DEVICE_CPU,
            (Device::Cuda, Residency::Device) => WaifuDevice::WAIFU_DEVICE_CUDA,
            (Device::Cuda, Residency::LowVram) => WaifuDevice::WAIFU_DEVICE_CUDA_CPU_OFFLOAD,
            (Device::Metal, Residency::Device) => WaifuDevice::WAIFU_DEVICE_METAL,
            (Device::Metal, Residency::LowVram) => WaifuDevice::WAIFU_DEVICE_METAL_DISK_OFFLOAD,
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

/// What this machine can run on and how much room its cards have, asked of libwaifu.
///
/// The first time CUDA is asked it builds a context on the card, a few hundred megabytes of it,
/// and the tensor library says what hardware it found: so this is asked once, before any screen
/// is up.
pub fn measure() -> (Vec<Runtime>, Room) {
    let Some(said) = c::waifu_json(c::waifu_devices_json()) else {
        return (vec![Runtime::on(Device::Cpu)], Room::default());
    };
    let mut available = Vec::new();
    let mut room = Room::default();
    for row in said["devices"].as_array().into_iter().flatten() {
        let Some(runtime) = row["name"].as_str().and_then(Runtime::named) else {
            continue;
        };
        if row["available"].as_bool() != Some(true) {
            continue;
        }
        available.push(runtime);
        let free = row["free"].as_u64();
        match runtime.device() {
            Device::Cuda if room.cuda.is_none() => room.cuda = free,
            Device::Vulkan => room.vulkan = free,
            _ => {}
        }
    }
    if available.is_empty() {
        available.push(Runtime::on(Device::Cpu));
    }
    (available, room)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    #[test]
    fn auto_puts_a_model_on_the_card_it_fits_and_offloads_one_it_does_not() {
        let everything = Runtime::ALL.to_vec();
        let cuda = Runtime::on(Device::Cuda);
        let sixteen = Room {
            cuda: Some(16 * GB),
            vulkan: Some(16 * GB),
        };

        // SDXL on a sixteen gigabyte card, and Qwen-Image's fp8 beside it.
        assert_eq!(Runtime::best(Some(7 * GB), &everything, sixteen), cuda);
        assert_eq!(
            Runtime::best(Some(15 * GB), &everything, sixteen),
            Runtime::CUDA_CPU_OFFLOAD
        );

        // A size or a card that did not say is the old answer, which is the card.
        assert_eq!(Runtime::best(None, &everything, sixteen), cuda);
        assert_eq!(
            Runtime::best(Some(40 * GB), &everything, Room::default()),
            cuda
        );
    }

    #[test]
    fn auto_without_cuda_goes_to_vulkan_only_where_the_model_fits() {
        let cpu = Runtime::on(Device::Cpu);
        let vulkan = Runtime::on(Device::Vulkan);
        let here = [cpu, vulkan];
        let eight = Room {
            cuda: None,
            vulkan: Some(8 * GB),
        };

        assert_eq!(Runtime::best(Some(3 * GB), &here, eight), vulkan);
        assert_eq!(Runtime::best(Some(7 * GB), &here, eight), cpu);

        // Metal has the machine's memory, so it has no card to be too big for -- and its offload
        // is never chosen for anyone, however large the model: it is asked for by name.
        let mac = [cpu, Runtime::on(Device::Metal), Runtime::METAL_DISK_OFFLOAD];
        assert_eq!(
            Runtime::best(Some(400 * GB), &mac, Room::default()),
            Runtime::on(Device::Metal)
        );
        assert_eq!(
            Runtime::METAL_DISK_OFFLOAD.short_of(Some(400 * GB), Room::default()),
            None
        );
        assert_eq!(Runtime::best(Some(1), &[cpu], Room::default()), cpu);
    }

    #[test]
    fn only_a_card_the_weights_go_on_can_be_short_of_room() {
        let eight = Room {
            cuda: Some(8 * GB),
            vulkan: Some(8 * GB),
        };
        let cuda = Runtime::on(Device::Cuda);

        assert_eq!(cuda.short_of(Some(3 * GB), eight), None);
        assert_eq!(
            cuda.short_of(Some(7 * GB), eight),
            Some((wanted_on_the_card(7 * GB), 8 * GB))
        );
        assert!(Runtime::on(Device::Vulkan)
            .short_of(Some(7 * GB), eight)
            .is_some());

        // The offload one keeps the weights on the host, and the processor has no card at all.
        assert_eq!(
            Runtime::CUDA_CPU_OFFLOAD.short_of(Some(7 * GB), eight),
            None
        );
        assert_eq!(Runtime::on(Device::Cpu).short_of(Some(7 * GB), eight), None);
        // Nor is anything short of what was never measured.
        assert_eq!(cuda.short_of(None, eight), None);
        assert_eq!(cuda.short_of(Some(7 * GB), Room::default()), None);
    }
}
