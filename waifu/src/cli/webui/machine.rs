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

//! What this is running on: the processor, the memory, the card, and how much of each is left.
//!
//! On the screen because most of what anybody asks about a run is really a question about this. A
//! model that will not load, a run that takes four minutes rather than forty seconds, a size that
//! works at 1024 and aborts at 1536 -- each of those is answered by the card and how much room is
//! on it, and the page used to say which device a run goes to and nothing at all about what that
//! device is.
//!
//! Two kinds of fact, gathered two ways. What the machine *is* -- the processor's name, how many
//! cores it has, how much memory is fitted, what the card is called -- does not change while the
//! program runs, so it is read once and kept. What is *left* changes by the second, so it is
//! measured on every request; that costs a read of one file under `/proc`, or one small command
//! on a Mac, which is cheap enough to do while a bar is moving.
//!
//! Nothing here is asked of a crate. What this wants is in a text file on Linux and behind
//! `sysctl` on a Mac, and a dependency that read them for us would bring a description of every
//! operating system this will never run on.

use std::sync::OnceLock;

use serde_json::{json, Value};

use crate::cli::args::Runtime;
use crate::flint::{Device, MemorySnapshot};

/// What the machine is, as opposed to what is left of it.
///
/// Every field is optional because every one of them is read off a file or a command that may not
/// answer -- a kernel that spells a field differently, a container with no `/proc` mounted, a
/// processor nobody has taught this to name. A missing field is a line the page leaves out; it is
/// not an error, and it is certainly not a reason to refuse to draw.
struct Parts {
    /// The processor, as it calls itself, tidied of the trademark furniture.
    cpu: Option<String>,
    /// Cores, and threads: two numbers, because they differ by a factor of two on most of the
    /// machines this runs on and because which one somebody wants depends on what they are about
    /// to blame.
    cores: Option<usize>,
    threads: Option<usize>,
    /// The memory fitted, in bytes.
    memory: Option<i64>,
    gpu: Option<Gpu>,
}

/// The card, named.
struct Gpu {
    name: String,
    /// Whether its memory is the machine's memory. True on Apple's, where a model that would not
    /// fit in eight gigabytes of VRAM on a PC has the whole machine's memory to fit in -- and
    /// where a page showing VRAM beside memory would be showing the same bytes twice.
    unified: bool,
}

/// Everything the column on the left shows, gathered now.
///
/// `runtime` is where runs are going, which decides one thing: whether the card is asked how much
/// room is on it. See [`vram`].
pub fn describe(runtime: Runtime) -> Value {
    let parts = parts();
    let (vram, why_no_vram) = vram(runtime);

    // The accelerators this build can actually use on this machine, which is a different question
    // from what is fitted: a card in a build with no CUDA in it is a card the page can name and
    // cannot send anything to.
    let accelerators: Vec<&str> = [Device::Cuda, Device::Metal, Device::Vulkan]
        .into_iter()
        .filter(|device| device.is_available())
        .map(|device| device.name())
        .collect();

    json!({
        "cpu": parts.cpu,
        "cores": parts.cores,
        "threads": parts.threads,
        "memory": {
            "total": parts.memory,
            // What is in use, which is the number a bar is drawn from. Null where the machine
            // will say what it has but not what is spare: half of it draws no bar.
            "used": match (parts.memory, spare_memory()) {
                (Some(total), Some(spare)) => Some((total - spare).max(0)),
                _ => None,
            },
        },
        "gpu": parts.gpu.as_ref().map(|gpu| json!({
            "name": gpu.name,
            "unified": gpu.unified,
        })),
        "vram": vram,
        "why_no_vram": why_no_vram,
        "accelerators": accelerators,
    })
}

/// How much room is on the card, and -- when that is nothing -- why.
///
/// Asked only while runs are going to CUDA or Vulkan. Measuring it means asking the driver, and
/// the first time a process does that the driver builds a context on the card: several hundred
/// megabytes of the very thing being measured. Somebody who chose the CPU chose not to touch the
/// card at all, and a sidebar that made a context behind their back would be spending their VRAM
/// on a picture of their VRAM.
///
/// Metal answers nothing for a different reason: that backend implements no snapshot, and asking
/// for one ends the process. Its card has the machine's memory anyway, which the line above
/// already shows -- that is what `unified` is for.
fn vram(runtime: Runtime) -> (Value, Value) {
    let device = runtime.device();
    if device != Device::Cuda && device != Device::Vulkan {
        return (Value::Null, json!("measured while runs go to the card"));
    }

    match MemorySnapshot::capture(device) {
        Ok(memory) if memory.total > 0 => (
            json!({
                "total": memory.total,
                // Everything on the card rather than only this program's tensors: what decides
                // whether the next run fits is what is free, and a desktop and a browser are
                // holding some of it.
                "used": memory.total - memory.free,
                // And this program's own, which is the half of it a size that aborted is about.
                "ours": memory.allocated,
            }),
            Value::Null,
        ),
        Ok(_) | Err(_) => (Value::Null, json!("the card did not say")),
    }
}

/// What the machine is, read once.
///
/// Once because none of it changes, and because the page asks several times a minute: a processor
/// does not get a new name while a picture is being drawn.
fn parts() -> &'static Parts {
    static PARTS: OnceLock<Parts> = OnceLock::new();
    PARTS.get_or_init(look)
}

/// How many threads the machine will run at once, which the standard library knows everywhere.
fn threads() -> Option<usize> {
    std::thread::available_parallelism().ok().map(Into::into)
}

/// Trims the trademark furniture off a processor's own name.
///
/// "Intel(R) Xeon(R) w5-2465X" is what the chip says it is called and "Intel Xeon w5-2465X" is
/// what it is called. The column this sits in is a couple of hundred pixels wide, so those twelve
/// characters are twelve characters of the actual name pushed onto a second line. The clock goes
/// with them: it is the base clock rather than the one anything runs at, and the cores are on the
/// line below.
fn tidy(name: &str) -> String {
    let name = name.split('@').next().unwrap_or(name);
    name.replace("(R)", "")
        .replace("(r)", "")
        .replace("(TM)", "")
        .replace("(tm)", "")
        .replace(" CPU", "")
        .replace(" Processor", "")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

// -- Linux ---------------------------------------------------------------------------------------

#[cfg(target_os = "linux")]
fn look() -> Parts {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();

    Parts {
        cpu: cpu_name(&cpuinfo),
        cores: cores(&cpuinfo),
        threads: threads(),
        memory: kilobytes(&meminfo, "MemTotal"),
        gpu: nvidia_card(),
    }
}

/// The memory nothing has a claim on, in bytes.
///
/// `MemAvailable` rather than `MemFree`. A Linux machine that has been up for an hour has almost
/// no free memory and most of it available -- what is neither is cache it hands back the moment
/// something asks -- so a bar drawn from the free figure shows every idle machine as full.
#[cfg(target_os = "linux")]
fn spare_memory() -> Option<i64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    kilobytes(&meminfo, "MemAvailable")
}

/// The processor's name out of `/proc/cpuinfo`.
///
/// `model name` is what x86 writes. An Arm kernel writes no such line -- there is no such string
/// in the hardware for it to write -- and what it writes instead is a set of implementer and part
/// numbers that mean nothing to a reader, so those are left out rather than turned into
/// "0x41 0xd0c".
#[cfg(target_os = "linux")]
fn cpu_name(cpuinfo: &str) -> Option<String> {
    let named = field(cpuinfo, "model name").or_else(|| field(cpuinfo, "Model name"))?;
    Some(tidy(named))
}

/// How many cores there are, as opposed to how many threads.
///
/// `cpu cores` is per package, so a machine with two sockets says sixteen twice and has
/// thirty-two. The sockets are counted by how many distinct `physical id` values there are, that
/// being the only place the file says how many there are.
#[cfg(target_os = "linux")]
fn cores(cpuinfo: &str) -> Option<usize> {
    let per_socket: usize = field(cpuinfo, "cpu cores")?.parse().ok()?;

    let mut sockets: Vec<&str> = cpuinfo
        .lines()
        .filter_map(|line| line.split_once(':'))
        .filter(|(name, _)| name.trim() == "physical id")
        .map(|(_, value)| value.trim())
        .collect();
    sockets.sort_unstable();
    sockets.dedup();

    Some(per_socket * sockets.len().max(1))
}

/// The first value of a `name : value` line, which is the shape of everything in `/proc/cpuinfo`
/// and `/proc/meminfo` both.
#[cfg(target_os = "linux")]
fn field<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    text.lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(field, _)| field.trim() == name)
        .map(|(_, value)| value.trim())
}

/// A `Name: 1234 kB` line of `/proc/meminfo`, in bytes.
#[cfg(target_os = "linux")]
fn kilobytes(meminfo: &str, name: &str) -> Option<i64> {
    let line = field(meminfo, name)?;
    let number: i64 = line.split_whitespace().next()?.parse().ok()?;

    Some(number * 1024)
}

/// The card's name, out of the file the NVIDIA driver writes for each one.
///
/// This rather than asking CUDA, because CUDA is asked by making a context on the card and this
/// is a file: the name is on the screen for somebody drawing on the CPU, and in a build with no
/// CUDA in it at all. One card, because the runtime uses device zero and nothing here can send a
/// run to a second -- a list would be a list of what cannot be chosen.
#[cfg(target_os = "linux")]
fn nvidia_card() -> Option<Gpu> {
    let mut written: Vec<_> = std::fs::read_dir("/proc/driver/nvidia/gpus")
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .collect();
    written.sort();

    let told = std::fs::read_to_string(written.first()?.join("information")).ok()?;
    let name = field(&told, "Model")?;

    Some(Gpu {
        name: tidy(name),
        unified: false,
    })
}

// -- macOS ---------------------------------------------------------------------------------------

/// What `sysctl` says about one name, which is where a Mac keeps all of this.
///
/// The command rather than the C call behind it: this crate links one native library and it is
/// the tensor one. A subprocess is a millisecond, three of them run while the program starts and
/// none after that, and what they buy is not having an `unsafe` block and a libc dependency in
/// the file that draws a sidebar.
#[cfg(target_os = "macos")]
fn sysctl(name: &str) -> Option<String> {
    let said = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", name])
        .output()
        .ok()?;
    if !said.status.success() {
        return None;
    }

    let said = String::from_utf8(said.stdout).ok()?.trim().to_string();
    (!said.is_empty()).then_some(said)
}

#[cfg(target_os = "macos")]
fn look() -> Parts {
    let cpu = sysctl("machdep.cpu.brand_string").map(|name| tidy(&name));

    Parts {
        cores: sysctl("hw.physicalcpu").and_then(|cores| cores.parse().ok()),
        threads: threads(),
        memory: sysctl("hw.memsize").and_then(|bytes| bytes.parse().ok()),
        gpu: apple_card(cpu.as_deref()),
        cpu,
    }
}

/// The GPU on an Apple chip, which is the chip.
///
/// Named from the processor rather than asked after on its own. What would answer properly is
/// `system_profiler`, which takes a second or more to start and prints a hundred kilobytes of
/// plist in order to say "Apple M3 Pro" somewhere in the middle of it; the chip's own name is
/// already here and says the same thing. An Intel Mac gets nothing rather than a guess -- its GPU
/// is a separate part with a name of its own, and this has no honest way to learn it.
#[cfg(target_os = "macos")]
fn apple_card(cpu: Option<&str>) -> Option<Gpu> {
    let chip = cpu?;
    chip.starts_with("Apple").then(|| Gpu {
        name: format!("{chip} GPU"),
        unified: true,
    })
}

/// The memory nothing has a claim on, from `vm_stat`.
///
/// Free, plus the two kinds of page a Mac hands over the moment something asks: inactive pages,
/// which hold what was read a while ago, and speculative ones, which hold what was read ahead of
/// being asked for. This is roughly what Activity Monitor draws when it says how much memory is
/// not under pressure, and like that figure it is an estimate rather than a promise -- which is
/// why what it feeds is a bar and not a decision.
#[cfg(target_os = "macos")]
fn spare_memory() -> Option<i64> {
    let said = std::process::Command::new("/usr/bin/vm_stat")
        .output()
        .ok()?;
    let said = String::from_utf8(said.stdout).ok()?;

    spare_pages(&said)
}

/// The same reading, over the text, so that it can be tested without a Mac in the room.
#[cfg(target_os = "macos")]
fn spare_pages(said: &str) -> Option<i64> {
    // The first line is "Mach Virtual Memory Statistics: (page size of 16384 bytes)", which is
    // the only place the page size is printed and is not the same on every Mac.
    let first = said.lines().next()?;
    let page: i64 = first
        .split("page size of ")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;

    let pages = |name: &str| -> i64 {
        said.lines()
            .filter_map(|line| line.split_once(':'))
            .find(|(field, _)| field.trim() == name)
            .and_then(|(_, count)| count.trim().trim_end_matches('.').parse::<i64>().ok())
            .unwrap_or(0)
    };

    Some((pages("Pages free") + pages("Pages inactive") + pages("Pages speculative")) * page)
}

// -- everywhere else -----------------------------------------------------------------------------

/// A system this has not been taught to read. It says what the standard library knows and leaves
/// the rest of the column out, which is what a Linux machine with no `/proc` mounted does too.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn look() -> Parts {
    Parts {
        cpu: None,
        cores: None,
        threads: threads(),
        memory: None,
        gpu: None,
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn spare_memory() -> Option<i64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_processor_is_named_without_its_trademarks() {
        assert_eq!(
            tidy("Intel(R) Xeon(R) w5-2465X"),
            "Intel Xeon w5-2465X",
            "the registered marks are twelve characters of a name that has to fit in a column"
        );
        assert_eq!(
            tidy("Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz"),
            "Intel Core i7-9750H",
            "the clock is the base one, which is not the speed anything runs at"
        );
        assert_eq!(
            tidy("AMD Ryzen 9 5950X 16-Core Processor"),
            "AMD Ryzen 9 5950X 16-Core"
        );
        assert_eq!(tidy("Apple M3 Pro"), "Apple M3 Pro");
    }

    #[test]
    fn what_the_machine_is_survives_being_asked_twice() {
        // The whole of it, on whatever this is being built on. What it must not do is panic, and
        // the second call must answer out of the first rather than reading anything again.
        let once = describe(crate::cli::args::DeviceOption::Cpu.resolve());
        let twice = describe(crate::cli::args::DeviceOption::Cpu.resolve());

        assert_eq!(once["cpu"], twice["cpu"]);
        assert_eq!(once["threads"], twice["threads"]);
        assert!(once["threads"].as_u64().unwrap_or(0) >= 1, "{once}");

        // Nothing was asked of the card, because runs are going to the processor.
        assert!(once["vram"].is_null(), "{once}");
        assert!(once["why_no_vram"].is_string(), "{once}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn two_sockets_have_twice_the_cores_one_socket_says() {
        let told = "\
processor\t: 0
model name\t: Intel(R) Xeon(R) Gold 6248
physical id\t: 0
cpu cores\t: 20
processor\t: 1
model name\t: Intel(R) Xeon(R) Gold 6248
physical id\t: 1
cpu cores\t: 20
";
        assert_eq!(cpu_name(told).as_deref(), Some("Intel Xeon Gold 6248"));
        assert_eq!(cores(told), Some(40), "the file says 20 once per package");

        let alone = "model name\t: Intel(R) Xeon(R) w5-2465X\nphysical id\t: 0\ncpu cores\t: 16\n";
        assert_eq!(cores(alone), Some(16));

        // An Arm kernel writes neither line, and there is nothing to say about that.
        assert_eq!(cpu_name("processor\t: 0\nBogoMIPS\t: 50.00\n"), None);
        assert_eq!(cores("processor\t: 0\n"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn memory_is_read_in_bytes_out_of_a_file_written_in_kilobytes() {
        let told = "MemTotal:       65055196 kB\nMemFree: 6880824 kB\nMemAvailable: 34773724 kB\n";

        assert_eq!(kilobytes(told, "MemTotal"), Some(65_055_196 * 1024));
        assert_eq!(kilobytes(told, "MemAvailable"), Some(34_773_724 * 1024));
        assert_eq!(kilobytes(told, "MemNothing"), None);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn what_a_mac_will_hand_back_is_more_than_what_is_free() {
        let told = "\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               12000.
Pages active:                            900000.
Pages inactive:                           40000.
Pages speculative:                         8000.
";
        assert_eq!(spare_pages(told), Some((12_000 + 40_000 + 8_000) * 16_384));
        assert_eq!(spare_pages("nothing like it"), None);
    }
}
