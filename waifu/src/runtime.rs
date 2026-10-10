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

use crate::flint::MemorySnapshot;
use crate::{Device, Residency};

/// Where a run goes: a device, and where the weights wait between the steps that read them.
///
/// One name rather than two answers. Where the weights wait is a question about a card the model
/// does not fit on, so it has one device it can be asked of and one reason to ask it, and asking
/// it beside the device made a second box whose rows were struck out three times out of four. As
/// a device name -- `cuda_cpu_offload`, cuda with the weights kept on the host -- the pair that
/// cannot be built cannot be said either, and there is one list on screen instead of two.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Runtime {
    device: Device,
    residency: Residency,
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
        residency: Residency::LowVram,
    };

    /// A device with the weights on it, which is what every name but the offload one means.
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
        match self.residency {
            Residency::Device => self.device.name(),
            Residency::LowVram => "cuda_cpu_offload",
        }
    }

    /// What the terminal's device list says beside the name of one this machine can use.
    pub fn about(self) -> &'static str {
        match self.residency {
            Residency::Device => "ready",
            // The one row that has to say more than that. It is the slow answer, and the slow
            // answer picked by someone who was not told is a run that looks broken.
            Residency::LowVram => "ready, slower: for a model larger than the card",
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
    /// The runtime this actually means, which is the one question `auto` leaves open.
    pub fn resolve(self) -> Runtime {
        match self {
            DeviceOption::Cpu => Runtime::on(Device::Cpu),
            DeviceOption::Cuda => Runtime::on(Device::Cuda),
            DeviceOption::CudaCpuOffload => Runtime::CUDA_CPU_OFFLOAD,
            DeviceOption::Metal => Runtime::on(Device::Metal),
            DeviceOption::Vulkan => Runtime::on(Device::Vulkan),
            // Never the offload one: with no model in hand there is nothing to measure against
            // the card. [`DeviceOption::resolve_for`] is the answer once there is.
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

    /// The runtime this means for a model of `weights` bytes: a named device is that device, and
    /// `auto` is [`Runtime::best`] for this machine as it is now.
    pub fn resolve_for(self, weights: Option<u64>) -> Runtime {
        match self {
            DeviceOption::Auto => {
                let available = Runtime::available();
                Runtime::best(weights, &available, Room::measure(&available))
            }
            named => named.resolve(),
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
    /// Every place a run can go that this build and this machine can take it.
    ///
    /// The first call starts the tensor library, which says what it found on stderr.
    pub fn available() -> Vec<Runtime> {
        Runtime::ALL
            .into_iter()
            .filter(|runtime| runtime.device().is_available())
            .collect()
    }

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

        // Metal has the machine's memory, so it has no card to be too big for.
        let mac = [cpu, Runtime::on(Device::Metal)];
        assert_eq!(
            Runtime::best(Some(40 * GB), &mac, Room::default()),
            Runtime::on(Device::Metal)
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
