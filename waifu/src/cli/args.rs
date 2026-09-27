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

//! The flags the commands share, parsed the way the Go tool parses them.

use std::fmt;

use crate::cli::task::Task;
use crate::flint::MemorySnapshot;
use crate::{Device, Residency};

/// What went wrong with what the user typed. Reported rather than exiting, so that the caller
/// prints the usage that goes with the command they were running.
#[derive(Debug)]
pub struct ArgError(String);

impl fmt::Display for ArgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ArgError {}

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
    const fn on(device: Device) -> Runtime {
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

    /// The fastest of `available` a model of `weights` bytes will run on, given `room`.
    ///
    /// CUDA where the model fits on the card, and CUDA with the weights on the host where it does
    /// not: slower, but far faster than the processor, and it finishes where the other aborts.
    /// Then Metal, whose memory is the machine's. Then Vulkan, which has no offload of its own and
    /// so is only the answer where the model fits. The processor otherwise. A size that is not
    /// known is taken as one that fits.
    pub fn best(weights: Option<u64>, available: &[Runtime], room: Room) -> Runtime {
        let fits = |free: Option<u64>| match (weights, free) {
            (Some(weights), Some(free)) => wanted_on_the_card(weights) <= free,
            _ => true,
        };
        let cuda = Runtime::on(Device::Cuda);
        let metal = Runtime::on(Device::Metal);
        let vulkan = Runtime::on(Device::Vulkan);

        if available.contains(&cuda) {
            if fits(room.cuda) {
                return cuda;
            }
            if available.contains(&Runtime::CUDA_CPU_OFFLOAD) {
                return Runtime::CUDA_CPU_OFFLOAD;
            }
        }
        if available.contains(&metal) {
            return metal;
        }
        if available.contains(&vulkan) && fits(room.vulkan) {
            return vulkan;
        }
        Runtime::on(Device::Cpu)
    }
}

/// The flags a command was given.
#[derive(Debug, Default)]
pub struct Args {
    models: Vec<String>,
    device: Option<String>,
    image: Option<String>,
    port: Option<String>,
    output: Option<String>,
    task: Option<String>,
    help: bool,
}

impl Args {
    /// Reads `-m model` and `-device name`, in either the `-flag value` or the `-flag=value`
    /// form, which is what the Go `flag` package accepts.
    pub fn parse(arguments: &[String]) -> Result<Args, ArgError> {
        let mut args = Args::default();
        let mut rest = arguments.iter();

        while let Some(argument) = rest.next() {
            let (name, inline_value) = match argument.split_once('=') {
                Some((name, value)) => (name, Some(value.to_string())),
                None => (argument.as_str(), None),
            };

            let mut value = |name: &str| -> Result<String, ArgError> {
                match inline_value.clone() {
                    Some(value) => Ok(value),
                    None => rest
                        .next()
                        .cloned()
                        .ok_or_else(|| ArgError(format!("flag needs an argument: {name}"))),
                }
            };

            match name {
                "-m" | "--m" => args.models.push(value("-m")?),
                "-device" | "--device" => args.device = Some(value("-device")?),
                "-i" | "--i" | "-image" | "--image" => args.image = Some(value("-i")?),
                "-port" | "--port" => args.port = Some(value("-port")?),
                "-output" | "--output" => args.output = Some(value("-output")?),
                "-task" | "--task" => args.task = Some(value("-task")?),
                "-h" | "--h" | "-help" | "--help" => args.help = true,
                other => return Err(ArgError(format!("flag provided but not defined: {other}"))),
            }
        }

        Ok(args)
    }

    pub fn wants_help(&self) -> bool {
        self.help
    }

    /// The one model file to work with, if one was named.
    ///
    /// None is not an error: without `-m` the terminal offers the published models and fetches the
    /// one that is picked. For text2speech it is the voice. Several `-m` flags is usually a stray comma in one of them, and that
    /// is an error, because guessing which of the two was meant is worse than saying so.
    pub fn model(&self) -> Result<Option<&str>, ArgError> {
        match self.models.len() {
            0 => Ok(None),
            1 => Ok(Some(&self.models[0])),
            _ => Err(ArgError(
                "only 1 model (-m) is expected, please check if there is any unexpected comma \
                 \",\" in model arg (-m)."
                    .to_string(),
            )),
        }
    }

    /// The picture a run should start from, if one was named.
    ///
    /// Not opened here: this only fills the box on the screen, which is where it can be changed
    /// between runs, and the file is read by the thread that owns the model when a run begins.
    pub fn image(&self) -> Option<&str> {
        self.image.as_deref()
    }

    /// The task the page is served for, if one was named.
    ///
    /// None is not an error: beside `-m` the task is what the model is -- a voice reads, anything
    /// else draws -- and without `-m` the terminal asks.
    pub fn task(&self) -> Result<Option<Task>, ArgError> {
        let Some(task) = &self.task else {
            return Ok(None);
        };

        Task::named(task).map(Some).ok_or_else(|| {
            ArgError(format!(
                "invalid task \"{task}\": must be one of {}",
                Task::ALL.map(Task::name).join(", ")
            ))
        })
    }

    /// The port to serve the page on, if one was named.
    ///
    /// None is not an error: without it the tool takes the usual port, or the first free one near
    /// it. Zero is refused rather than passed on -- the operating system reads it as "any port at
    /// all", which is a fine thing for a test to ask for and not something anybody types.
    /// The directory the pictures and clips are written under: each page's in a folder of its
    /// own inside it. Left out, the directory the program was started in, which is where they
    /// always went.
    pub fn output(&self) -> &str {
        self.output.as_deref().unwrap_or(".")
    }

    pub fn port(&self) -> Result<Option<u16>, ArgError> {
        let Some(port) = &self.port else {
            return Ok(None);
        };

        match port.trim().parse::<u16>() {
            Ok(0) | Err(_) => Err(ArgError(format!(
                "invalid port \"{port}\": it must be a number between 1 and 65535"
            ))),
            Ok(port) => Ok(Some(port)),
        }
    }

    pub fn device(&self) -> Result<DeviceOption, ArgError> {
        match self
            .device
            .as_deref()
            .unwrap_or("auto")
            .to_lowercase()
            .as_str()
        {
            "auto" => Ok(DeviceOption::Auto),
            "cpu" => Ok(DeviceOption::Cpu),
            "cuda" => Ok(DeviceOption::Cuda),
            // The one name with a hyphen in it as well, because a name with two words in it is
            // typed both ways and being told off over the punctuation helps nobody.
            "cuda_cpu_offload" | "cuda-cpu-offload" => Ok(DeviceOption::CudaCpuOffload),
            "metal" => Ok(DeviceOption::Metal),
            "vulkan" => Ok(DeviceOption::Vulkan),
            // Every name it would have taken, built from the list rather than written out
            // again: a name refused here is most often a name that was nearly right, and a list
            // that had drifted from the one above would be the worst possible thing to show
            // somebody looking for their typo.
            _ => Err(ArgError(format!(
                "invalid device name: must be one of {} or \"auto\"",
                Runtime::ALL
                    .iter()
                    .map(|runtime| format!("\"{}\"", runtime.name()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))),
        }
    }
}

/// The flags every command prints under `Options:`.
pub fn print_options() {
    eprintln!(
        "  -device string\n    \tinference device, one of cpu, cuda, cuda_cpu_offload, metal or \
         auto (default \"auto\"). auto picks the fastest one the model fits on: cuda where the \
         weights and room for a run fit in the card's free memory, cuda_cpu_offload where they do \
         not. Without -m it is where the terminal's device list starts. \
         cuda_cpu_offload keeps the weights in host memory and moves each \
         one onto the card as it is used, so that a model larger than the card can still draw; it \
         is slower, since the whole model crosses the bus once per step."
    );
    eprintln!(
        "  -m value\n    \tthe model to draw with, or the voice to read with: either a manifest \
         file, which has the suffix \".yaml\" and names the packages the weights are in, or the \
         name of a published model, which is fetched before the page opens. Left out, the \
         terminal asks for the task, the model and the device, and fetches what is picked. The \
         names are: {}.",
        crate::cli::hub::names().join(", ")
    );
    eprintln!(
        "  -i string\n    \ta picture to draw from rather than from noise, as a PNG or a JPEG. \
         It opens in the img2img tab, scaled to the size chosen there, and how far the run walks \
         away from it is what the denoising strength box says."
    );
    eprintln!(
        "  -task string\n    \twhat the page is for, one of {}. Beside -m it can be left out: a \
         voice reads -- a published one, or a manifest of IndexTTS-2.5 or Fun-CosyVoice3 -- and \
         a picture model draws, from the picture where -i names one.",
        Task::ALL.map(Task::name).join(", ")
    );
    eprintln!(
        "  -port int\n    \tthe port to serve the page on (default 7860). Left out, the first \
         free port from 7860 upwards is used, so that a second copy of this in another window \
         still starts. The page is served to this machine and no further."
    );
    eprintln!(
        "  -output string\n    \twhere pictures and clips are written (default \".\", the directory \
         this was started in). Each browser that opens the page writes into a folder of its own \
         inside it -- session-0001, session-0002 and on -- so that two people's pictures are not \
         one list."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(arguments: &[&str]) -> Result<Args, ArgError> {
        Args::parse(&arguments.iter().map(|a| a.to_string()).collect::<Vec<_>>())
    }

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
    fn a_named_device_is_not_second_guessed_by_the_size_of_the_model() {
        assert_eq!(
            DeviceOption::Cuda.resolve_for(Some(1000 * GB)),
            Runtime::on(Device::Cuda)
        );
        assert_eq!(
            DeviceOption::Cpu.resolve_for(Some(1)),
            Runtime::on(Device::Cpu)
        );
    }

    #[test]
    fn reads_the_picture_to_start_from() {
        assert_eq!(args(&["-i", "cat.png"]).unwrap().image(), Some("cat.png"));
        assert_eq!(args(&["--image=cat.png"]).unwrap().image(), Some("cat.png"));
        assert_eq!(args(&["-m", "x.waifupkg"]).unwrap().image(), None);
    }

    #[test]
    fn keeps_the_weights_on_the_card_unless_the_offload_device_is_named() {
        // Where the weights wait is part of the device name now, so it is not a question anyone
        // is asked twice, and every other name answers it the same way.
        for device in [DeviceOption::Auto, DeviceOption::Cpu, DeviceOption::Cuda] {
            assert_eq!(device.resolve().residency(), Residency::Device);
        }

        let asked = args(&["-device", "cuda_cpu_offload"]).unwrap();
        assert_eq!(asked.device().unwrap().resolve(), Runtime::CUDA_CPU_OFFLOAD);
        assert_eq!(
            asked.device().unwrap().resolve().residency(),
            Residency::LowVram
        );
        assert_eq!(
            asked.device().unwrap().resolve().device(),
            Device::Cuda,
            "the weights can only wait off a card there is a card for"
        );
    }

    #[test]
    fn the_offload_device_is_taken_with_a_hyphen_as_well_as_an_underscore() {
        // Two words in a device name is two ways to type it, and a run refused over the
        // punctuation between them would be a run refused for nothing.
        for typed in ["cuda_cpu_offload", "cuda-cpu-offload", "CUDA_CPU_OFFLOAD"] {
            assert_eq!(
                args(&["-device", typed]).unwrap().device().unwrap(),
                DeviceOption::CudaCpuOffload,
                "{typed}"
            );
        }
    }

    #[test]
    fn what_a_runtime_is_called_is_what_names_it() {
        // The list on the screens and the names -device takes are the same list: anything the
        // picker can be left on has to be something the command line can be given.
        for runtime in Runtime::ALL {
            let named = args(&["-device", runtime.name()]).unwrap();
            assert_eq!(
                named.device().unwrap().resolve(),
                runtime,
                "{}",
                runtime.name()
            );
        }
    }

    #[test]
    fn reads_a_flag_in_either_form() {
        assert_eq!(
            args(&["-m", "sdxl-base.waifupkg"])
                .unwrap()
                .model()
                .unwrap(),
            Some("sdxl-base.waifupkg")
        );
        assert_eq!(
            args(&["-m=sdxl-base.waifupkg"]).unwrap().model().unwrap(),
            Some("sdxl-base.waifupkg")
        );
        assert_eq!(
            args(&["-m", "x.waifupkg", "-device", "cuda"])
                .unwrap()
                .device()
                .unwrap(),
            DeviceOption::Cuda
        );
    }

    #[test]
    fn defaults_the_device_and_lowercases_it() {
        assert_eq!(args(&[]).unwrap().device().unwrap(), DeviceOption::Auto);
        assert_eq!(
            args(&["-device", "CUDA"]).unwrap().device().unwrap(),
            DeviceOption::Cuda
        );
        let error = args(&["-device", "tpu"]).unwrap().device().unwrap_err();

        // Every name it would have taken, since a name it will not take is most often a name
        // that was nearly right.
        let error = error.to_string();
        for name in Runtime::ALL.map(Runtime::name).iter().chain(&["auto"]) {
            assert!(error.contains(name), "{name} missing from {error}");
        }
    }

    #[test]
    fn takes_at_most_one_model() {
        // No -m at all is a picker rather than a mistake.
        assert_eq!(args(&[]).unwrap().model().unwrap(), None);

        // Two -m flags usually means a comma crept into one of them.
        let two = args(&["-m", "a.waifupkg", "-m", "b.waifupkg"]).unwrap();
        let error = two.model().unwrap_err().to_string();
        assert!(error.contains("only 1 model"), "{error}");
    }

    #[test]
    fn refuses_what_it_does_not_understand() {
        assert!(args(&["-nope"]).is_err());

        // A flag at the end with nothing after it would otherwise take the next flag as its value.
        let error = args(&["-m"]).unwrap_err().to_string();
        assert!(error.contains("needs an argument"), "{error}");
    }

    #[test]
    fn a_named_device_is_the_one_that_is_used() {
        assert_eq!(DeviceOption::Cpu.resolve().device(), Device::Cpu);
        assert_eq!(DeviceOption::Cuda.resolve().device(), Device::Cuda);
        assert_eq!(
            DeviceOption::CudaCpuOffload.resolve().device(),
            Device::Cuda
        );
    }

    #[test]
    fn reads_the_port_to_serve_on() {
        // Left out is not an error: the tool has a port it takes when nobody says.
        assert_eq!(args(&[]).unwrap().port().unwrap(), None);
        assert_eq!(
            args(&["-port", "8080"]).unwrap().port().unwrap(),
            Some(8080)
        );
        assert_eq!(args(&["--port=1"]).unwrap().port().unwrap(), Some(1));

        // Zero means "whichever one is free" to the operating system and nothing at all to the
        // person typing it, and the rest are not numbers a port can be.
        for typed in ["0", "70000", "-1", "http"] {
            let error = args(&["-port", typed]).unwrap().port().unwrap_err();
            assert!(error.to_string().contains("invalid port"), "{typed}");
        }
    }

    #[test]
    fn reads_where_to_write() {
        // Left out, where it has always been: the directory the program was started in.
        assert_eq!(args(&[]).unwrap().output(), ".");
        assert_eq!(args(&["-output", "out"]).unwrap().output(), "out");
        assert_eq!(args(&["--output=/tmp/w"]).unwrap().output(), "/tmp/w");
    }

    #[test]
    fn reads_the_task_and_refuses_one_there_is_not() {
        assert_eq!(args(&[]).unwrap().task().unwrap(), None);
        assert_eq!(
            args(&["-task", "img2img"]).unwrap().task().unwrap(),
            Some(Task::Img2Img)
        );
        assert_eq!(
            args(&["--task=text2speech"]).unwrap().task().unwrap(),
            Some(Task::Text2Speech)
        );

        // Every name it would have taken, for the reason the device's refusal lists them.
        let error = args(&["-task", "sing"]).unwrap().task().unwrap_err();
        for task in Task::ALL {
            assert!(error.to_string().contains(task.name()), "{error}");
        }
    }

    #[test]
    fn the_offload_device_says_what_it_costs() {
        // The one row in the terminal's device list anybody has to be told something about.
        assert!(Runtime::CUDA_CPU_OFFLOAD.about().contains("slower"));
        assert!(!Runtime::ALL[0].about().contains("slower"));
    }

    #[test]
    fn recognises_a_request_for_help() {
        assert!(args(&["-h"]).unwrap().wants_help());
        assert!(args(&["--help"]).unwrap().wants_help());
        assert!(!args(&["-m", "x"]).unwrap().wants_help());
    }
}
