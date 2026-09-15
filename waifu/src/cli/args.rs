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
    /// Every place a run can go, in the order the screens list them.
    pub const ALL: [Runtime; 4] = [
        Runtime::on(Device::Cpu),
        Runtime::on(Device::Cuda),
        Runtime::CUDA_CPU_OFFLOAD,
        Runtime::on(Device::Metal),
    ];

    /// Cuda, with the package page-locked on the host and each weight moved onto the card at the
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

    /// Whether this build can run it, which is a question about the device underneath: weights
    /// waiting on the host need the same cuda driver that weights on the card do.
    pub fn is_available(self) -> bool {
        self.device.is_available()
    }

    /// The words beside the name in the list, for a runtime this build can run.
    ///
    /// What it costs rather than what it is, since the name says what it is. Nothing here asks
    /// whether it is available: that is a call across the C boundary and this is read on every
    /// redraw, so the caller, which asked once at the start, says so instead.
    pub fn about(self) -> &'static str {
        match self.residency {
            Residency::Device => "ready",
            // The one row that has to say more than that. It is the slow answer, and the slow
            // answer picked by someone who was not told is a run that looks broken.
            Residency::LowVram => "ready, slower",
        }
    }
}

/// Where the model should run, as `-device` spells it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceOption {
    /// The first accelerator there is a device for -- CUDA, then Metal -- and the CPU otherwise.
    Auto,
    Cpu,
    Cuda,
    CudaCpuOffload,
    Metal,
}

impl DeviceOption {
    /// The runtime this actually means, which is the one question `auto` leaves open.
    pub fn resolve(self) -> Runtime {
        match self {
            DeviceOption::Cpu => Runtime::on(Device::Cpu),
            DeviceOption::Cuda => Runtime::on(Device::Cuda),
            DeviceOption::CudaCpuOffload => Runtime::CUDA_CPU_OFFLOAD,
            DeviceOption::Metal => Runtime::on(Device::Metal),
            // Never the offload one. It is slower, and what it is for is a card the model does
            // not fit on, which is not something to decide on someone's behalf.
            DeviceOption::Auto => {
                // At most one of the two is ever built, so the order between them only decides
                // which check runs first, not which machine gets which accelerator.
                if Device::Cuda.is_available() {
                    Runtime::on(Device::Cuda)
                } else if Device::Metal.is_available() {
                    Runtime::on(Device::Metal)
                } else {
                    Runtime::on(Device::Cpu)
                }
            }
        }
    }
}

/// The flags a command was given.
#[derive(Debug, Default)]
pub struct Args {
    models: Vec<String>,
    device: Option<String>,
    image: Option<String>,
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
    /// None is not an error: without `-m` the screen offers the published models and fetches the
    /// one that is picked. Several `-m` flags is usually a stray comma in one of them, and that
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
            _ => Err(ArgError(
                "invalid device name: must be one of \"cpu\", \"cuda\", \"cuda_cpu_offload\", \
                 \"metal\" or \"auto\""
                    .to_string(),
            )),
        }
    }
}

/// The flags every command prints under `Options:`.
pub fn print_options() {
    eprintln!(
        "  -device string\n    \tinference device, one of cpu, cuda, cuda_cpu_offload, metal or \
         auto (default \"auto\"). cuda_cpu_offload keeps the weights in host memory and moves each \
         one onto the card as it is used, so that a model larger than the card can still draw; it \
         is slower, since the whole model crosses the bus once per step."
    );
    eprintln!(
        "  -m value\n    \tthe model to draw with: either a manifest file, which has the suffix \
         \".yaml\" and names the packages the weights are in, or the name of a published model, \
         which is fetched on first use. Left out, the screen offers the published ones to pick \
         from. The names are: {}.",
        crate::cli::hub::names().join(", ")
    );
    eprintln!(
        "  -i string\n    \ta picture to draw from rather than from noise, as a PNG or a JPEG. \
         It is scaled to the size on the screen, and how far the run walks away from it is what \
         the strength box says."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(arguments: &[&str]) -> Result<Args, ArgError> {
        Args::parse(&arguments.iter().map(|a| a.to_string()).collect::<Vec<_>>())
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
    fn recognises_a_request_for_help() {
        assert!(args(&["-h"]).unwrap().wants_help());
        assert!(args(&["--help"]).unwrap().wants_help());
        assert!(!args(&["-m", "x"]).unwrap().wants_help());
    }
}
