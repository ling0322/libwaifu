//! What one draw actually holds on the card, off the allocator's own high-water mark rather than
//! off a sampler.
//!
//! `usage: memprofile MODEL.yaml [lowvram]`
//!
//! The peak is `cudaMemPoolAttrUsedMemHigh`, which the driver maintains exactly: it is the
//! largest the pool's *used* bytes ever were, not the largest anything happened to observe. So it
//! catches a weight that lands and is freed between two polls of `nvidia-smi`, which is most of
//! them under a low-vram run. It also leaves out the CUDA context and the pool's reserved-but-idle
//! bytes, so what it reports is the model rather than the process.
//!
//! Measured in two stages, because they answer different questions: what building the model puts
//! on the card, and what one pass adds on top of it. Under `Residency::LowVram` the first should
//! be nearly nothing -- that is the whole claim of the mode -- and under `Residency::Device` it is
//! the model.

use std::time::Instant;

use waifu::flint::MemorySnapshot;
use waifu::{Device, GenerationOptions, Krea2, Manifest, Residency};

const PROMPT: &str = "a calico cat asleep on a stack of books, warm afternoon light";

fn mib(bytes: i64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() -> Result<(), waifu::Error> {
    let mut arguments: Vec<String> = std::env::args().skip(1).collect();

    let residency = match arguments.iter().position(|a| a == "lowvram") {
        Some(at) => {
            arguments.remove(at);
            Residency::LowVram
        }
        None => Residency::Device,
    };

    let Some(model_path) = arguments.first() else {
        eprintln!("usage: memprofile MODEL.yaml [lowvram]");
        std::process::exit(1);
    };

    let device = Device::Cuda;
    let manifest = Manifest::open(model_path)?;
    eprintln!("weights {residency:?}");

    let start = MemorySnapshot::capture(device)?;
    println!(
        "card         total {:9.1} MiB   allocated before anything {:7.1} MiB",
        mib(start.total),
        mib(start.allocated)
    );

    // What building the model costs. A low-vram run page-locks the weights on the host rather than
    // putting them on the card, so this is the number that says the model was never put there.
    MemorySnapshot::reset_peak_stats(device)?;
    let built = Instant::now();
    let model = Krea2::from_manifest(device, residency, &manifest)?;
    let build_time = built.elapsed();
    let after_build = MemorySnapshot::capture(device)?;
    println!(
        "after build  allocated {:9.1} MiB   peak {:9.1} MiB   ({:.1?})",
        mib(after_build.allocated),
        mib(after_build.peak_allocated),
        build_time
    );

    let suggested = manifest.suggested();
    let options = GenerationOptions {
        negative_prompt: suggested.avoid.clone().unwrap_or_default(),
        seed: Some(7),
        ..suggested.over(Krea2::DEFAULTS).options()
    };
    println!(
        "drawing      {} by {}, {} steps, guidance {}",
        options.width, options.height, options.num_steps, options.guidance_scale
    );

    // And what one whole draw adds on top of that: the encoder, every step of the denoiser, and
    // the decode, which is usually where the peak of the three is.
    MemorySnapshot::reset_peak_stats(device)?;
    let drawn = Instant::now();
    let image = model.generate(PROMPT, &options)?;
    let draw_time = drawn.elapsed();
    let after_draw = MemorySnapshot::capture(device)?;

    println!(
        "after draw   allocated {:9.1} MiB   PEAK {:9.1} MiB   ({:.1?})",
        mib(after_draw.allocated),
        mib(after_draw.peak_allocated),
        draw_time
    );
    println!(
        "so one draw held at most {:.1} MiB of a {:.1} MiB card",
        mib(after_draw.peak_allocated),
        mib(after_draw.total)
    );

    // Touched so the decode is not optimized away as an unused result.
    eprintln!("image {} by {}", image.shape_at(3)?, image.shape_at(2)?);
    Ok(())
}
