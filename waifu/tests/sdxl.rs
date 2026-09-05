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

//! The whole of SDXL: a package read from disk, a prompt in, an image out.
//!
//! Each piece has its own test against its own reference. This is the one that says they are put
//! together the way `StableDiffusionXLPipeline` puts them together -- the same latent denoised by
//! the same schedule with the same guidance lands in the same place.

use std::path::PathBuf;

use waifu::flint::{functional as F, Tensor};
use waifu::{to_rgb8, DType, Device, GenerationOptions, Sdxl, UnetCondition, VarBuilder, ZipFile};

/// What the reference denoising run in the exporter was written for.
const PROMPT: &str = "a photo of an astronaut riding a horse on mars";
const STEPS: i32 = 4;
const GUIDANCE: f32 = 5.0;
const SIZE: i32 = 256;

/// Read only by the runs that start from a picture; the rest walk the whole schedule.
const STRENGTH: f32 = 0.8;

/// How far torch's own half precision run lands from its float32 one, on this very case:
/// measured at 2.87e-2 by running `StableDiffusionXLPipeline` twice, once at each precision, with
/// TF32 turned off so that the float32 side really is float32.
///
/// This is the bar rather than a number chosen by hand. The reference is float32 and this runs in
/// half, so most of the distance between them is what half costs and not what the implementation
/// does; the question a test can answer is whether we are further from float32 than the reference
/// implementation is at the same precision. Comparing against a half reference instead would not
/// help: ours is 2.80e-2 from torch's half, which is as far as torch's half is from its own
/// float32, so at that precision the kernels' own differences are already the size of the
/// rounding. `docs/TODO.md` has the table.
const TORCH_HALF_GAP: f32 = 2.87e-2;

/// How much further than that this is willing to be: five percent.
///
/// Not slack for its own sake. Four steps at guidance five is an amplifier, and the amount it
/// amplifies was measured in float32, where nothing rounds: a per-step nudge of the U-Net's
/// answer by 5e-4 lands 7.4e-3 from the clean run, and a context off by 2.6e-3 lands 2.06e-2
/// away. Half precision costs the second text encoder about that much for us (2.60e-3) and for
/// torch (2.62e-3) alike, so roughly two thirds of the number below is a disturbance neither
/// implementation can avoid and only the direction of which differs between them.
///
/// What that leaves is a metric decided by which way a last bit fell. Where the two
/// implementations disagree at all, they disagree by that much and no more: at the conditioning
/// projection's shape, 11 of 1280 outputs differ between the vector kernel a single row takes and
/// the GEMM a batch takes, by at most two of half's last bits, with both inside one of them from
/// the float32 answer. Running one U-Net call as a batch of two rather than one at a time was
/// measured over eight prompts: better on three, worse on five, spread from -11% to +36%, and
/// this test's prompt drew the +36%.
///
/// So five percent is the width of the coin flip, not a place to hide a regression in. The
/// float32 CPU walk below is the sensitive test, at 1.1e-4 against a bar of 1e-3, and an
/// assembly mistake shows there by orders of magnitude rather than by percent.
const HALF_ALLOWANCE: f32 = 1.05;

/// What the same walk costs in float32 against a float32 reference, which is what the CPU runs.
/// It measured 9.1e-5, so this leaves an order of magnitude and is the sensitive test of the two.
const FLOAT32_TOLERANCE: f32 = 1e-3;

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models")
}

fn model() -> Sdxl {
    let package = ZipFile::open(models_dir().join("sdxl-base.waifupkg")).unwrap();
    Sdxl::from_package(Device::Cuda, &package).unwrap()
}

fn cases() -> VarBuilder {
    let package = ZipFile::open(models_dir().join("sdxl-base_test.waifupkg")).unwrap();
    VarBuilder::from_reader(
        &mut package.open_entry("test_case.bin").unwrap(),
        Device::Cpu,
        DType::Float,
    )
    .unwrap()
}

fn options() -> GenerationOptions {
    GenerationOptions {
        width: SIZE,
        height: SIZE,
        num_steps: STEPS,
        guidance_scale: GUIDANCE,
        negative_prompt: String::new(),
        seed: None,
        strength: STRENGTH,
    }
}

fn to_cuda(tensor: &Tensor) -> Tensor {
    tensor
        .to_device(Device::Cuda)
        .unwrap()
        .cast(DType::Float16)
        .unwrap()
}

fn relative_rmse(actual: &Tensor, reference: &Tensor) -> f32 {
    let a = actual
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    let b = reference
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    assert_eq!(a.len(), b.len(), "the shapes do not match");

    let error: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum();
    let scale: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (error / scale).sqrt() as f32
}

#[test]
#[ignore = "needs the sdxl package"]
fn reads_its_shape_out_of_the_package() {
    let model = model();
    let config = model.config();

    assert_eq!(config.text.hidden_size, 768);
    assert_eq!(config.text2.hidden_size, 1280);
    assert_eq!(config.unet.block_out_channels, vec![320, 640, 1280]);
    assert_eq!(config.vae.block_out_channels, vec![128, 256, 512, 512]);
    assert_eq!(config.sampler.num_train_timesteps, 1000);
    assert_eq!(model.device(), Device::Cuda);
}

#[test]
#[ignore = "needs the sdxl package"]
fn encodes_a_prompt_the_way_the_reference_does() {
    // The pieces of this are checked in sdxl_text_encoder.rs, which starts from the reference's
    // own token ids. This one starts from the text, so it also says the tokenizer, the markers
    // and the padding are what the encoders were given.
    let cases = cases();
    let model = model();

    let embedding = model.encode_prompt(PROMPT).unwrap();
    assert_eq!(embedding.context.shape(), vec![1, 77, 2048]);
    assert_eq!(embedding.pooled.shape(), vec![1, 1280]);

    let hidden = cases.get_unchecked("test_case.hidden").unwrap();
    let hidden2 = cases.get_unchecked("test_case.hidden2").unwrap();

    // Sliced where it lives: a half precision tensor on the host has no copy kernel behind it,
    // and making one contiguous is a copy.
    let context = &embedding.context;
    let first = context.slice(2, 0, 768).unwrap().contiguous().unwrap();
    let second = context.slice(2, 768, 2048).unwrap().contiguous().unwrap();

    assert!(relative_rmse(&first, &hidden) < 2e-2);
    assert!(relative_rmse(&second, &hidden2) < 2e-2);
    assert!(
        relative_rmse(
            &embedding.pooled,
            &cases.get_unchecked("test_case.pooled2").unwrap()
        ) < 2e-2
    );
}

#[test]
#[ignore = "needs the sdxl package"]
fn a_batch_of_two_is_two_batches_of_one() {
    // What guidance relies on. The prompt and its absence go through the U-Net as one batch of
    // two, and that is only the same answer if nothing on the way reads across the batch --
    // which would not fail loudly, it would quietly mix the two prompts and give an image that
    // is a little wrong. `batching.rs` asks this of each operator on its own; this asks it of
    // the whole U-Net with the real weights, which is where an operator used in a way the unit
    // test did not think of would show.
    let model = model();
    let cases = cases();

    let prompt = model.encode_prompt(PROMPT).unwrap();
    let negative = model.encode_prompt("").unwrap();
    let latent = to_cuda(&cases.get_unchecked("test_case.latent").unwrap());
    let time_ids = [SIZE as f32, SIZE as f32, 0.0, 0.0, SIZE as f32, SIZE as f32];

    let alone = |embedding: &waifu::PromptEmbedding| {
        model
            .unet()
            .forward(
                &latent,
                801.0,
                &UnetCondition {
                    context: &embedding.context,
                    pooled: &embedding.pooled,
                    time_ids,
                },
            )
            .unwrap()
    };
    let alone_negative = alone(&negative);
    let alone_prompt = alone(&prompt);

    let context = F::cat(&negative.context, &prompt.context, 0).unwrap();
    let pooled = F::cat(&negative.pooled, &prompt.pooled, 0).unwrap();
    let batched = model
        .unet()
        .forward(
            &F::cat(&latent, &latent, 0).unwrap(),
            801.0,
            &UnetCondition {
                context: &context,
                pooled: &pooled,
                time_ids,
            },
        )
        .unwrap();

    // Half precision holds about four significant digits, so a row that agrees to a thousandth
    // is agreeing as closely as the arithmetic allows. A row read from the wrong place would be
    // wrong by tens of percent, which is what this is really looking for.
    for (index, one) in [alone_negative, alone_prompt].iter().enumerate() {
        let row = batched
            .slice(0, index as i32, index as i32 + 1)
            .unwrap()
            .contiguous()
            .unwrap();
        let rmse = relative_rmse(&row, one);
        assert!(
            rmse < 2e-3,
            "row {index} batched differs by {rmse} from the same row alone"
        );
    }
}

#[test]
#[ignore = "needs the sdxl package"]
fn denoising_matches_the_reference_pipeline() {
    let cases = cases();
    let model = model();

    // The same unit noise the reference started from. Everything after it -- the schedule, the
    // guidance, the size the model is told about -- has to agree for four steps running.
    let latent = to_cuda(&cases.get_unchecked("test_case.latent").unwrap());

    let prompt = model.encode_prompt(PROMPT).unwrap();
    let negative = model.encode_prompt("").unwrap();

    let denoised = model
        .denoise(&latent, &prompt, &negative, &options())
        .unwrap();
    assert_eq!(denoised.shape(), latent.shape());

    let rmse = relative_rmse(
        &denoised,
        &cases.get_unchecked("test_case.denoised").unwrap(),
    );
    println!("denoise rmse = {rmse} (torch's own half precision is {TORCH_HALF_GAP} from float32)");
    assert!(
        rmse < TORCH_HALF_GAP * HALF_ALLOWANCE,
        "four steps drifted by {rmse}, which is more than five percent further from float32 than \
         torch's own half precision run gets ({TORCH_HALF_GAP})"
    );
}

#[test]
#[ignore = "needs the sdxl package"]
fn denoising_on_the_cpu_matches_the_reference_pipeline() {
    // The same walk on the host, which is the sensitive version of the test above. The CPU is in
    // float32 throughout on x64 -- it has no half kernels -- and the reference was made in
    // float32, so the two are doing the same arithmetic and what is left is the implementation.
    // That is 9.1e-5 against the 2.1e-2 the CUDA run lives with, which is what makes this the one
    // that would catch a mistake in the model rather than in the precision.
    //
    // It is the same model code either way: the U-Net, the sampler and the assembly do not know
    // which device they are on, so only a CUDA kernel's own mistake can hide from this.
    let cases = cases();
    let package = ZipFile::open(models_dir().join("sdxl-base.waifupkg")).unwrap();
    let model = Sdxl::from_package(Device::Cpu, &package).unwrap();

    let latent = cases.get_unchecked("test_case.latent").unwrap();
    let prompt = model.encode_prompt(PROMPT).unwrap();
    let negative = model.encode_prompt("").unwrap();

    let denoised = model
        .denoise(&latent, &prompt, &negative, &options())
        .unwrap();
    assert_eq!(denoised.shape(), latent.shape());

    let rmse = relative_rmse(
        &denoised,
        &cases.get_unchecked("test_case.denoised").unwrap(),
    );
    println!("cpu denoise rmse = {rmse}");
    assert!(rmse < FLOAT32_TOLERANCE, "four steps drifted by {rmse}");
}

#[test]
#[ignore = "needs the sdxl package"]
fn generates_a_latent_from_a_prompt() {
    let model = model();
    let mut options = options();
    options.seed = Some(7);

    let latent = model.generate_latent(PROMPT, &options).unwrap();
    assert_eq!(latent.shape(), vec![1, 4, SIZE / 8, SIZE / 8]);

    let values = latent
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    assert!(values.iter().all(|x| x.is_finite()));

    // Four steps is not enough for a picture, but a latent that had gone wrong somewhere would be
    // flat or enormous rather than the handful of units a real one covers.
    let largest = values.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        (0.5..50.0).contains(&largest),
        "the latent reaches {largest}"
    );
}

#[test]
#[ignore = "needs the sdxl package"]
fn decodes_a_denoised_latent_into_an_image() {
    // The latent the sampler really hands over, which is the case the autoencoder needs float32
    // for: in half its last up block passes 65504 and everything after it is a NaN. The decoder
    // is built in float32 and the half latent is cast on the way in, so this is also what says
    // the two precisions meet where they are supposed to.
    let cases = cases();
    let model = model();

    let denoised = to_cuda(&cases.get_unchecked("test_case.denoised").unwrap());
    let image = model.decode(&denoised).unwrap();
    assert_eq!(image.shape(), vec![1, 3, SIZE, SIZE]);
    assert_eq!(image.dtype(), DType::Float);

    let values = image
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    assert!(
        values.iter().all(|x| x.is_finite()),
        "the decoder produced a NaN or an infinity"
    );

    // A decoder ends in roughly [-1, 1]. Four steps is not enough for a good picture, but one
    // that had gone wrong somewhere would be flat or far outside that range rather than merely
    // ugly, and either would survive a check that only asked for finite numbers.
    let extreme = values.iter().filter(|x| x.abs() > 1.5).count();
    assert!(
        extreme * 100 < values.len(),
        "{extreme} of {} pixels are far outside [-1, 1]",
        values.len()
    );

    let largest = values.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        largest > 0.5,
        "the image is flat: it reaches only {largest}"
    );
}

#[test]
#[ignore = "needs the sdxl package"]
fn generates_an_image_from_a_prompt() {
    // The whole of it, from a string to pixels: tokenizer, both encoders, the sampler walking the
    // U-Net, and the autoencoder. Every piece is checked against its own reference elsewhere, so
    // what this adds is that they are wired together and that the last step now runs.
    let model = model();
    let mut options = options();
    options.seed = Some(7);

    let image = model.generate(PROMPT, &options).unwrap();
    assert_eq!(image.shape(), vec![1, 3, SIZE, SIZE]);

    let pixels = to_rgb8(&image).unwrap();
    assert_eq!(pixels.len(), (SIZE * SIZE * 3) as usize);

    // Three bytes a pixel of one value would be a black or a white rectangle, which is what a
    // decoder that had overflowed or a latent that never moved would give.
    let smallest = *pixels.iter().min().unwrap();
    let largest = *pixels.iter().max().unwrap();
    assert!(
        largest - smallest > 32,
        "the image covers only {smallest}..{largest}"
    );
}

#[test]
#[ignore = "needs the sdxl package"]
fn the_seed_is_the_only_thing_that_makes_a_run_differ() {
    let model = model();
    let mut options = options();
    // Two steps rather than four: this is about which noise a run starts from, and running the
    // U-Net twice as many times says nothing more about that.
    options.num_steps = 2;

    options.seed = Some(11);
    let first = model.generate_latent(PROMPT, &options).unwrap();
    let again = model.generate_latent(PROMPT, &options).unwrap();
    assert_eq!(
        relative_rmse(&first, &again),
        0.0,
        "one seed gave two latents"
    );

    options.seed = Some(12);
    let other = model.generate_latent(PROMPT, &options).unwrap();
    assert!(
        relative_rmse(&first, &other) > 0.1,
        "two seeds gave one latent"
    );
}

#[test]
#[ignore = "needs the sdxl package"]
fn guidance_of_one_is_the_unprompted_answer() {
    // At a guidance of one the negative prompt is not consulted at all, and the second U-Net pass
    // is skipped for it. Anything else would be running the model twice for nothing.
    let cases = cases();
    let model = model();
    let latent = to_cuda(&cases.get_unchecked("test_case.latent").unwrap());

    let prompt = model.encode_prompt(PROMPT).unwrap();
    let negative = model.encode_prompt("a completely different thing").unwrap();
    let unrelated = model.encode_prompt("something else again").unwrap();

    let mut options = options();
    options.guidance_scale = 1.0;
    options.num_steps = 2;

    let first = model
        .denoise(&latent, &prompt, &negative, &options)
        .unwrap();
    let second = model
        .denoise(&latent, &prompt, &unrelated, &options)
        .unwrap();
    assert_eq!(relative_rmse(&first, &second), 0.0);
}

#[test]
#[ignore = "needs the sdxl package"]
fn refuses_a_size_it_cannot_work_in() {
    let model = model();

    // Eight for the autoencoder and four more for the U-Net's two halvings.
    for (width, height) in [(256, 100), (250, 256), (0, 256), (-256, 256)] {
        let mut options = options();
        options.width = width;
        options.height = height;
        assert!(
            model.generate_latent(PROMPT, &options).is_err(),
            "{width} by {height} was allowed"
        );
    }
}

/// The picture the reference decoder made of the reference latent, which is a real image at the
/// size these tests work at and so the natural thing to hand an encoder.
///
/// Left on the host, in float32, which is where a picture read off the disk is: `from_rgb8` hands
/// one back on the CPU whatever the model is on. That is what the draw command passes and so it is
/// what these tests pass; the one below says a picture already on the device works too.
fn picture(cases: &VarBuilder) -> Tensor {
    cases.get_unchecked("test_case.decoded").unwrap()
}

#[test]
#[ignore = "needs the sdxl package with an encoder"]
fn starting_from_a_picture_gives_a_picture_of_the_same_size() {
    let cases = cases();
    let model = model();

    let image = model
        .generate_from_image(&picture(&cases), PROMPT, &options())
        .unwrap();

    // The picture's own size, not the one in the options: what was handed in already says.
    assert_eq!(image.shape(), vec![1, 3, SIZE, SIZE]);

    let values = image
        .to_device(Device::Cpu)
        .unwrap()
        .cast(DType::Float)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    assert!(
        values.iter().all(|x| x.is_finite()),
        "the run produced a NaN or an infinity"
    );
}

#[test]
#[ignore = "needs the sdxl package with an encoder"]
fn no_strength_is_the_picture_through_the_autoencoder_and_nothing_else() {
    // A strength of zero runs no steps at all, so the whole of the run is the encoder and the
    // decoder. That is a long way from the picture it was handed -- an autoencoder is lossy, and
    // this one is being asked to encode its own output, which puts it 0.226 away -- so what this
    // compares against is the reference making the same trip, not the picture.
    let cases = cases();
    let model = model();

    let mut options = options();
    options.strength = 0.0;

    let given = picture(&cases);
    let image = model.generate_from_image(&given, PROMPT, &options).unwrap();

    let rmse = relative_rmse(
        &image,
        &cases.get_unchecked("test_case.round_trip").unwrap(),
    );
    println!("round trip rmse against the reference = {rmse}");
    assert!(rmse < 2e-2, "a run of no steps drifted by {rmse}");
}

#[test]
#[ignore = "needs the sdxl package with an encoder"]
fn more_strength_leaves_less_of_the_picture() {
    // The property the knob is for. Neither number means anything on its own -- what an image to
    // image run is worth is not something a test can measure -- but the order between them is
    // what someone turning the knob is asking for, and it is what a start index computed the
    // wrong way round would get backwards.
    let cases = cases();
    let model = model();
    let given = picture(&cases);

    let mut drift = Vec::new();
    for strength in [0.25, 0.5, 1.0] {
        let mut options = options();
        options.strength = strength;
        options.num_steps = 8;
        options.seed = Some(11);

        let image = model.generate_from_image(&given, PROMPT, &options).unwrap();
        drift.push(relative_rmse(
            &image,
            &cases.get_unchecked("test_case.decoded").unwrap(),
        ));
    }

    println!("drift from the picture at 0.25, 0.5 and 1.0 strength = {drift:?}");
    assert!(drift[0] < drift[1], "{drift:?}");
    assert!(drift[1] < drift[2], "{drift:?}");
}

#[test]
#[ignore = "needs the sdxl package with an encoder"]
fn a_picture_is_taken_from_wherever_it_already_is() {
    // The two ends of what a caller might hand over: a picture off the disk, which is float32 on
    // the host, and one already on the device in the type the model runs in. Getting this wrong
    // is not a wrong answer but a dead process -- the convolution checks that its input and its
    // weights are on one device and ends the program where they are not.
    let cases = cases();
    let model = model();

    let mut options = options();
    options.strength = 0.0;
    options.seed = Some(3);

    let host = picture(&cases);
    let device = to_cuda(&host);

    let from_host = model.generate_from_image(&host, PROMPT, &options).unwrap();
    let from_device = model
        .generate_from_image(&device, PROMPT, &options)
        .unwrap();

    // Not the same to the last bit -- one went through the encoder in float32 and the other in
    // half -- but the same picture, which is what "wherever it already is" has to mean.
    let drift = relative_rmse(&from_host, &from_device);
    println!("host and device pictures differ by {drift}");
    assert!(drift < 2e-2, "the two disagree by {drift}");
}

#[test]
#[ignore = "needs the sdxl package with an encoder"]
fn refuses_a_picture_it_cannot_start_from() {
    let model = model();

    // Not a picture at all, and a picture at a size the U-Net cannot work in.
    let latent = F::rand(&[1, 4, 32, 32], DType::Float16, Device::Cuda).unwrap();
    assert!(model
        .generate_from_image(&latent, PROMPT, &options())
        .is_err());

    let ragged = F::rand(&[1, 3, 250, 256], DType::Float16, Device::Cuda).unwrap();
    assert!(model
        .generate_from_image(&ragged, PROMPT, &options())
        .is_err());

    // And a strength that is not a share of the walk.
    let mut options = options();
    options.strength = 1.5;
    let picture = F::rand(&[1, 3, SIZE, SIZE], DType::Float16, Device::Cuda).unwrap();
    assert!(model
        .generate_from_image(&picture, PROMPT, &options)
        .is_err());
}
