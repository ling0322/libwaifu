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

#[allow(unused_imports)]
use crate::runtime::Device;
use crate::task::Task;

pub use crate::runtime::{DeviceOption, Runtime};

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

/// Where pictures are written when `-output` does not say.
pub const DEFAULT_OUTPUT: &str = "waifu-output";

/// The flags a command was given.
#[derive(Debug, Default)]
pub struct Args {
    models: Vec<String>,
    device: Option<String>,
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
    /// one that is picked. Several `-m` flags is usually a stray comma in one of them, and that is
    /// an error, because guessing which of the two was meant is worse than saying so.
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

    /// The task the drawing screen is opened for, if one was named.
    ///
    /// None is not an error: it is txt2img, the one task there is. The flag is still read so that
    /// a script that names it keeps working, and so that a task there is not is refused.
    pub fn task(&self) -> Result<Option<Task>, ArgError> {
        let Some(task) = &self.task else {
            return Ok(None);
        };

        let named = Task::named(task).ok_or_else(|| {
            ArgError(format!(
                "invalid task \"{task}\": must be one of {}",
                Task::names()
            ))
        })?;
        Ok(Some(named))
    }

    /// The directory pictures are written into. Left out, `waifu-output` in the directory the
    /// program was started in.
    pub fn output(&self) -> &str {
        self.output.as_deref().unwrap_or(DEFAULT_OUTPUT)
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
        "  -device string\n    \tinference device, one of cpu, cuda, metal, vulkan or auto \
         (default \"auto\"). auto picks the fastest one this machine has: cuda, then metal, then \
         vulkan. It is where the model is read first; the drawing screen's device row moves it. cuda, vulkan and metal keep as many of the weights on the GPU as fit, the \
         smallest first, and copy the rest onto it as they are used, so that a model larger than \
         the card can still draw; a run that runs short lets weights on the GPU go. metal reads \
         the rest off the disk, so that a model larger than the Mac's memory can still draw."
    );
    eprintln!(
        "  -m value\n    \tthe model to draw with: either a manifest file, which has the suffix \
         \".yaml\" and names the packages the weights are in, or the name of a published model, \
         which is fetched before the drawing screen opens. Left out, the terminal asks for the \
         model, and fetches what is picked. The names are: {}.",
        crate::manager::pictures()
            .into_iter()
            .map(|model| model.name)
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!(
        "  -task string\n    \twhat the drawing screen is for, one of {} (default \"txt2img\").",
        Task::names()
    );
    eprintln!(
        "  -output string\n    \twhere pictures are written (default \"waifu-output\"), as \
         waifu-NNNN.png with the prompt and the settings that drew it inside."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(arguments: &[&str]) -> Result<Args, ArgError> {
        Args::parse(&arguments.iter().map(|a| a.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn a_named_device_is_not_second_guessed_by_what_the_machine_has() {
        let cpu_only = [Runtime::on(Device::Cpu)];
        assert_eq!(
            DeviceOption::Cuda.resolve(&cpu_only),
            Runtime::on(Device::Cuda)
        );
        assert_eq!(
            DeviceOption::Auto.resolve(&cpu_only),
            Runtime::on(Device::Cpu)
        );
        assert_eq!(
            DeviceOption::Auto.resolve(&Runtime::ALL),
            Runtime::on(Device::Cuda)
        );
    }

    #[test]
    fn a_picture_to_start_from_is_not_a_flag() {
        // Drawing from a picture is the engine's and not the terminal's, so -i is refused the way
        // any flag it never had is, rather than read and then ignored.
        for typed in [&["-i", "cat.png"][..], &["--image=cat.png"]] {
            let error = args(typed).unwrap_err().to_string();
            assert!(error.contains("not defined"), "{error}");
        }
    }

    #[test]
    fn the_offload_device_is_gone() {
        // The card keeps what fits of a model and lets weights go when a run runs short, so there
        // is no second name for it -- and one typed is refused with the names there are.
        for typed in ["cuda_cpu_offload", "cuda-cpu-offload"] {
            let error = args(&["-device", typed]).unwrap().device().unwrap_err();
            assert!(error.to_string().contains("\"cuda\""), "{error}");
        }
    }

    #[test]
    fn what_a_runtime_is_called_is_what_names_it() {
        // The list on the screens and the names -device takes are the same list: anything the
        // picker can be left on has to be something the command line can be given.
        for runtime in Runtime::ALL {
            let named = args(&["-device", runtime.name()]).unwrap();
            assert_eq!(
                named.device().unwrap().resolve(&Runtime::ALL),
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
        assert_eq!(
            DeviceOption::Cpu.resolve(&Runtime::ALL).device(),
            Device::Cpu
        );
        assert_eq!(
            DeviceOption::Cuda.resolve(&Runtime::ALL).device(),
            Device::Cuda
        );
    }

    #[test]
    fn reads_where_to_write_what_is_drawn() {
        assert_eq!(args(&[]).unwrap().output(), "waifu-output");
        assert_eq!(args(&["-output", "out"]).unwrap().output(), "out");
        assert_eq!(args(&["--output=/tmp/w"]).unwrap().output(), "/tmp/w");

        // What the page had and the terminal does not: a port, and a limit on a store of jobs.
        assert!(args(&["-port", "7860"]).is_err());
        assert!(args(&["-output-limit", "1GB"]).is_err());
    }

    #[test]
    fn reads_the_task_and_refuses_one_there_is_not() {
        assert_eq!(args(&[]).unwrap().task().unwrap(), None);
        assert_eq!(
            args(&["-task", "txt2img"]).unwrap().task().unwrap(),
            Some(Task::Txt2Img)
        );
        assert_eq!(
            args(&["--task=txt2img"]).unwrap().task().unwrap(),
            Some(Task::Txt2Img)
        );

        // img2img is gone from the terminal, and naming it is a task there is not.
        let error = args(&["-task", "img2img"]).unwrap().task().unwrap_err();
        assert!(error.to_string().contains("txt2img"), "{error}");

        // Every name it would have taken, for the reason the device's refusal lists them.
        let error = args(&["-task", "sing"]).unwrap().task().unwrap_err();
        for task in Task::ALL {
            assert!(error.to_string().contains(task.name()), "{error}");
        }
    }

    #[test]
    fn recognises_a_request_for_help() {
        assert!(args(&["-h"]).unwrap().wants_help());
        assert!(args(&["--help"]).unwrap().wants_help());
        assert!(!args(&["-m", "x"]).unwrap().wants_help());
    }
}
