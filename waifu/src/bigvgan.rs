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

//! BigVGAN, which is where a speech model stops being a picture of sound and starts being sound.
//!
//! A mel spectrogram is a hundred-odd numbers per frame, one frame every 256 samples; a waveform
//! is those 256 samples. This is the model that writes them: `(1, mel, frames)` in,
//! `(1, 1, frames * 256)` out, at 22.05 kHz. It is the last thing IndexTTS-2.5 runs and the only
//! part of it whose output is audio.
//!
//! It is a stack of transposed convolutions that double or quadruple the length, with residual
//! blocks between them, and it is [`crate::audio`] all the way down -- no operator here is one
//! this library did not already have.
//!
//! # What makes it a BigVGAN rather than a HiFi-GAN
//!
//! Two things, and both are in the activation.
//!
//! The activation is a *snake*, `x + sin(alpha x)^2 / beta`, rather than a leaky ReLU. It is
//! periodic, and periodicity is the inductive bias a waveform wants: a vocoder built on a ReLU
//! has to learn that speech is made of repeating things, and this one is told.
//!
//! And every snake is *anti-aliased*. A periodic activation applied at the signal's own rate
//! folds everything it generates above half that rate back down into the audible band, which is
//! heard as roughness. So each one upsamples by two, applies the snake there, and comes back --
//! [`crate::audio::upsample1d`] and [`crate::audio::downsample1d`], a Kaiser-windowed sinc each
//! way. That is the "anti-aliased multi-periodicity" in the AMP block's name, and it is most of
//! what this model costs: eighteen of them per upsampling stage, each on a tensor that has
//! already been lengthened.
//!
//! # What it is not
//!
//! It is not the vocoder for everything. `resblock: "2"` -- the cheaper AMP block, with one
//! convolution per dilation instead of two -- is not written here, because no released BigVGAN v2
//! uses it. A configuration asking for it is refused rather than quietly run as the other kind.
//!
//! # The weights are the checkpoint's, with the weight norm folded out
//!
//! Every convolution in the reference is wrapped in `weight_norm`, which stores a direction and a
//! magnitude rather than a weight. `IndexTTS2.__init__` calls `remove_weight_norm()` on the
//! vocoder as it loads it, so what the reference actually runs is the product of those two, and
//! that product is what `tools/bigvgan_exporter.py` writes and what this reads. The names are
//! otherwise the checkpoint's own: `conv_pre`, `ups.0.0`, `resblocks.3.convs1.1`, `conv_post`.

use std::fmt;
use std::rc::Rc;

use crate::audio::{downsample1d, kaiser_sinc_filter, snake, upsample1d};
use crate::error::{Error, Result};
use crate::flint::{
    check_parameters, DType, Device, Graph, Ir, ParamSource, Preloaded, RunContext, Tensor, Value,
};
use crate::layers::{Conv1d, ConvTranspose1d};

/// The resampling either side of every snake, which `Activation1d` fixes rather than configures:
/// two, and twelve taps. Nothing in a released BigVGAN changes them, and the length arithmetic
/// here assumes the ratio is what the two convolutions were built for.
const ACTIVATION_RATIO: i32 = 2;
const ACTIVATION_TAPS: i32 = 12;

/// What the snake divides by when its own parameter has decayed to nothing. The reference calls
/// it `no_div_by_zero` and it is added to the divisor rather than guarding the division.
const SNAKE_EPS: f32 = 1e-9;

/// What the package records about the vocoder, which is `config.json` beside the checkpoint.
#[derive(Clone, Debug)]
pub struct BigVganConfig {
    /// How many mel bands come in. 80 for the 22 kHz release IndexTTS-2.5 uses, 100 for the 44
    /// kHz ones.
    pub mel_channels: i32,
    /// How much each stage lengthens the signal. Their product is the hop the mel was computed
    /// with -- 256 for `[4, 4, 2, 2, 2, 2]` -- and a mel frame becomes exactly that many samples.
    pub upsample_rates: Vec<i32>,
    /// The transposed convolution's kernel at each stage, which is twice its rate in every
    /// released configuration.
    pub upsample_kernel_sizes: Vec<i32>,
    /// How wide the model is where the mel enters it. It halves at every stage.
    pub upsample_initial_channel: i32,
    /// The AMP blocks at each stage: one per kernel here, all run on the same input and averaged.
    pub resblock_kernel_sizes: Vec<i32>,
    /// The dilations inside each of those blocks, one row per kernel above.
    pub resblock_dilation_sizes: Vec<Vec<i32>>,
    /// `"snakebeta"`, which gives the snake a second parameter for its magnitude. `false` is the
    /// plain snake, where one parameter does both.
    pub snake_beta: bool,
    /// Whether the snake's parameters are stored as logarithms, which is how every released
    /// BigVGAN was trained. See [`activation`].
    pub snake_logscale: bool,
    /// How the waveform is bounded: a `tanh`, or a clamp to `[-1, 1]`. v2 clamps.
    pub use_tanh_at_final: bool,
    /// Whether the last convolution has a bias. v2 does not.
    pub use_bias_at_final: bool,
}

impl BigVganConfig {
    /// `nvidia/bigvgan_v2_22khz_80band_256x`, the vocoder IndexTTS-2.5 fetches at first run.
    ///
    /// Written out here because it is the one configuration this runtime is checked against, and
    /// because a caller with the weights and no `config.json` should not have to guess.
    pub fn v2_22khz_80band_256x() -> BigVganConfig {
        BigVganConfig {
            mel_channels: 80,
            upsample_rates: vec![4, 4, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![8, 8, 4, 4, 4, 4],
            upsample_initial_channel: 1536,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            snake_beta: true,
            snake_logscale: true,
            use_tanh_at_final: false,
            use_bias_at_final: false,
        }
    }

    /// How many samples one mel frame becomes, which is the hop the mel was analysed with.
    pub fn upsample_factor(&self) -> i32 {
        self.upsample_rates.iter().product()
    }

    /// How wide the model is going into `stage`, which halves at every one.
    fn channels_before(&self, stage: usize) -> i32 {
        self.upsample_initial_channel / (1 << stage)
    }

    /// And coming out of it.
    fn channels_after(&self, stage: usize) -> i32 {
        self.channels_before(stage + 1)
    }
}

/// The padding that leaves a dilated convolution the length it started, which is what every
/// convolution inside an AMP block wants. `utils.get_padding` in the reference.
fn same_padding(kernel: i32, dilation: i32) -> i32 {
    (kernel * dilation - dilation) / 2
}

/// One snake, applied at twice the rate and brought back down.
///
/// `up` and `down` are the filters, made once by [`write`] and shared by every activation in the
/// model: they depend on the ratio and nothing else, and at this ratio the two are the same
/// twelve numbers. They stay two arguments because a BigVGAN that resampled by different amounts
/// in the two directions would need them to be.
///
/// # Why the parameters are exponentiated here
///
/// A BigVGAN trained with `alpha_logscale` -- all of them -- stores the logarithm of each
/// parameter, and the reference raises it every time it runs. Doing the same keeps the package a
/// straight copy of the checkpoint rather than something an exporter has quietly transformed, and
/// it costs one node over a vector as long as the channel count.
#[allow(clippy::too_many_arguments)]
fn activation(
    g: &Graph,
    config: &BigVganConfig,
    x: Value,
    channels: i32,
    up: Value,
    down: Value,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let act = g.subgraph("act");

    let exponentiate = |value: Value| match config.snake_logscale {
        true => g.exp(value),
        false => value,
    };

    let alpha = exponentiate(act.load("alpha", &[channels]));
    let beta = match config.snake_beta {
        true => Some(exponentiate(act.load("beta", &[channels]))),
        false => None,
    };

    let raised = upsample1d(
        g,
        x,
        up,
        channels,
        ACTIVATION_RATIO,
        ACTIVATION_TAPS,
        dtype,
        device,
    )?;
    let curled = snake(g, raised, alpha, beta, SNAKE_EPS, dtype, device)?;

    downsample1d(
        g,
        curled,
        down,
        channels,
        ACTIVATION_RATIO,
        ACTIVATION_TAPS,
        dtype,
        device,
    )
}

/// An AMP block: for each dilation, two convolutions with an activation in front of each, added
/// back to what came in.
///
/// The dilated convolution is the first of the pair and the second is undilated, which is what
/// gives the block a receptive field wider than its kernel without making every layer expensive.
#[allow(clippy::too_many_arguments)]
fn amp_block(
    g: &Graph,
    config: &BigVganConfig,
    input: Value,
    channels: i32,
    kernel: i32,
    dilations: &[i32],
    up: Value,
    down: Value,
    dtype: DType,
    device: Device,
) -> Result<Value> {
    let mut x = input;

    for (layer, &dilation) in dilations.iter().enumerate() {
        // The activations are one flat list of twice as many, taken in pairs: the first of each
        // pair goes in front of `convs1`, the second in front of `convs2`. That is the layout the
        // checkpoint has, and `acts1, acts2 = self.activations[::2], self.activations[1::2]` is
        // where the reference says so.
        let first = g.subgraph(&format!("activations.{}", 2 * layer));
        let second = g.subgraph(&format!("activations.{}", 2 * layer + 1));

        let xt = activation(&first, config, x, channels, up, down, dtype, device)?;
        let xt = Conv1d::graph(
            &g.subgraph(&format!("convs1.{layer}")),
            xt,
            channels,
            channels,
            kernel,
            1,
            same_padding(kernel, dilation),
            dilation,
            true,
            dtype,
            device,
        )?;

        let xt = activation(&second, config, xt, channels, up, down, dtype, device)?;
        let xt = Conv1d::graph(
            &g.subgraph(&format!("convs2.{layer}")),
            xt,
            channels,
            channels,
            kernel,
            1,
            same_padding(kernel, 1),
            1,
            true,
            dtype,
            device,
        )?;

        x = g.add(xt, x);
    }

    Ok(x)
}

/// `clamp(x, -1, 1)`, which is not an operator here and does not need to be.
///
/// `(|x + 1| - |x - 1|) / 2` is the same function: above one both absolute values grow together
/// and the difference stops changing, below minus one the same in reverse, and in between the
/// second term is `1 - x` and the whole thing is `x`. Four nodes, all of them everywhere.
fn clamp_unit(g: &Graph, x: Value, dtype: DType, device: Device) -> Result<Value> {
    let one = g.constant(
        Tensor::from_f32(&[1], &[1.0])?
            .to_device(device)?
            .cast(dtype)?,
    );

    Ok(g.mul_scalar(g.sub(g.abs(g.add(x, one)), g.abs(g.sub(x, one))), 0.5))
}

/// The whole pass, from the mel the graph is given to the waveform it names.
fn write(config: &BigVganConfig, dtype: DType, device: Device, g: &Graph) -> Result<()> {
    // One filter for the whole model. The cutoff and the transition width are the reference's and
    // depend only on the ratio, and the two directions are designed at the same ratio with the
    // same number of taps -- so the upsampling's filter and the downsampling's are the same
    // numbers, and one constant does for every activation. The released model would otherwise
    // carry two hundred and eighteen copies of these twelve taps.
    let ratio = f64::from(ACTIVATION_RATIO);
    let taps = ACTIVATION_TAPS as usize;
    let filter = g.constant(
        kaiser_sinc_filter(0.5 / ratio, 0.6 / ratio, taps)?
            .to_device(device)?
            .cast(dtype)?,
    );
    let (up, down) = (filter, filter);

    let mel = g.input("mel");
    let mel = g.cast(g.to_device(mel, device), dtype);

    let widest = config.upsample_initial_channel;
    let mut x = Conv1d::graph(
        &g.subgraph("conv_pre"),
        mel,
        config.mel_channels,
        widest,
        7,
        1,
        3,
        1,
        true,
        dtype,
        device,
    )?;

    let kinds = config.resblock_kernel_sizes.len();
    for stage in 0..config.upsample_rates.len() {
        let (before, after) = (config.channels_before(stage), config.channels_after(stage));
        let (rate, kernel) = (
            config.upsample_rates[stage],
            config.upsample_kernel_sizes[stage],
        );

        // The checkpoint holds each upsampling as a list of one, which is a shape left over from
        // an earlier BigVGAN that could have several. Hence the `.0`.
        x = ConvTranspose1d::graph(
            &g.subgraph(&format!("ups.{stage}.0")),
            x,
            before,
            after,
            kernel,
            rate,
            (kernel - rate) / 2,
            dtype,
            device,
        )?;

        // Every AMP block at this stage reads the same input, and what goes on is their mean --
        // a multi-receptive-field fusion, which is the one thing this inherits from HiFi-GAN
        // unchanged.
        let mut total: Option<Value> = None;
        for kind in 0..kinds {
            let block = amp_block(
                &g.subgraph(&format!("resblocks.{}", stage * kinds + kind)),
                config,
                x,
                after,
                config.resblock_kernel_sizes[kind],
                &config.resblock_dilation_sizes[kind],
                up,
                down,
                dtype,
                device,
            )?;

            total = Some(match total {
                None => block,
                Some(sofar) => g.add(sofar, block),
            });
        }

        x = g.div_scalar(total.expect("a stage has at least one block"), kinds as f32);
    }

    let narrowest = config.channels_after(config.upsample_rates.len() - 1);
    let x = activation(
        &g.subgraph("activation_post"),
        config,
        x,
        narrowest,
        up,
        down,
        dtype,
        device,
    )?;

    let x = Conv1d::graph(
        &g.subgraph("conv_post"),
        x,
        narrowest,
        1,
        7,
        1,
        3,
        1,
        config.use_bias_at_final,
        dtype,
        device,
    )?;

    let bounded = match config.use_tanh_at_final {
        true => g.tanh(x),
        false => clamp_unit(g, x, dtype, device)?,
    };

    g.output("waveform", bounded);

    Ok(())
}

/// Everything about a configuration that would otherwise turn up as a kernel refusing a shape.
///
/// A graph holds no shapes, so nothing below this point can check any of it: what a stage is
/// given and what it hands on are numbers this file works out, and they have to be right before
/// the first node is written.
fn check(config: &BigVganConfig) -> Result<()> {
    let stages = config.upsample_rates.len();
    if stages == 0 {
        return Err(Error::model("a vocoder needs at least one upsampling"));
    }

    if config.upsample_kernel_sizes.len() != stages {
        return Err(Error::model(format!(
            "{} upsampling rates and {} kernels are not a pair each",
            stages,
            config.upsample_kernel_sizes.len()
        )));
    }

    if config.resblock_kernel_sizes.len() != config.resblock_dilation_sizes.len() {
        return Err(Error::model(format!(
            "{} residual kernels and {} rows of dilations are not a pair each",
            config.resblock_kernel_sizes.len(),
            config.resblock_dilation_sizes.len()
        )));
    }

    if config.resblock_kernel_sizes.is_empty() {
        return Err(Error::model("a stage needs at least one residual block"));
    }

    if config.mel_channels < 1 {
        return Err(Error::model("a mel spectrogram has at least one band"));
    }

    for (stage, (&rate, &kernel)) in config
        .upsample_rates
        .iter()
        .zip(&config.upsample_kernel_sizes)
        .enumerate()
    {
        if rate < 1 {
            return Err(Error::model(format!(
                "stage {stage} upsamples by {rate}, which is not an upsampling"
            )));
        }

        // The reference pads by `(kernel - rate) / 2` in integer arithmetic, which lengthens the
        // signal by exactly `rate` only when that division is exact. Every released configuration
        // has an even difference; one that did not would leave a sample per stage unaccounted
        // for, and it is better to say so than to produce audio a frame longer than the mel.
        if (kernel - rate) % 2 != 0 || kernel < rate {
            return Err(Error::model(format!(
                "stage {stage} has a kernel of {kernel} against a rate of {rate}, which is not \
                 the even difference an exact upsampling needs"
            )));
        }
    }

    // Every stage halves the width, and the last one has to land on something a convolution can
    // still be written over.
    if config.channels_after(stages - 1) < 1 {
        return Err(Error::model(format!(
            "{} channels do not survive {stages} halvings",
            config.upsample_initial_channel
        )));
    }

    Ok(())
}

/// One of what a run handed back, by the name the graph gave it.
fn output(outputs: &[(String, Tensor)], name: &str) -> Result<Tensor> {
    outputs
        .iter()
        .find(|(other, _)| other == name)
        .map(|(_, tensor)| tensor.clone())
        .ok_or_else(|| Error::model(format!("a vocoder produces no {name:?}")))
}

/// The vocoder: a mel spectrogram in, a waveform out.
pub struct BigVgan {
    config: BigVganConfig,
    /// What its weights are in, which a mel of any float type is cast to on the way in.
    dtype: DType,
    /// Where its weights are, which is where a mel has to be brought to meet them.
    device: Device,
    ir: Ir,
    preloaded: Preloaded,
    weights: Rc<dyn ParamSource>,
}

impl fmt::Debug for BigVgan {
    /// How big it is rather than what is in it. [`BigVgan::ir`] prints the pass, and every weight
    /// is named in it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BigVgan({} instructions)", self.ir.len())
    }
}

impl BigVgan {
    /// `name` is the namespace the package holds the vocoder under, which its weights are named
    /// from.
    pub fn build(
        config: BigVganConfig,
        name: &str,
        weights: &Rc<dyn ParamSource>,
        device: Device,
        float_type: DType,
    ) -> Result<BigVgan> {
        check(&config)?;

        // Not `with_weights`: a vocoder is convolutions and has no matrix in it worth quantizing,
        // so the format a package names is for the halves that do.
        let graph = Graph::new();
        write(&config, float_type, device, &graph.subgraph(name))?;
        check_parameters(&graph, weights.as_ref())?;

        let ir = Ir::compile(&graph, weights.residency());
        let preloaded = ir.load(weights.as_ref())?;

        Ok(BigVgan {
            weights: Rc::clone(weights),
            preloaded,
            ir,
            dtype: float_type,
            device,
            config,
        })
    }

    pub fn config(&self) -> &BigVganConfig {
        &self.config
    }

    /// The pass this vocoder runs, for printing.
    pub fn ir(&self) -> &Ir {
        &self.ir
    }

    /// What this vocoder computes in. A mel of any float type is cast to it on the way in.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Where this vocoder is. A mel anywhere else is brought here on the way in.
    pub fn device(&self) -> Device {
        self.device
    }

    /// The waveform a mel spectrogram means, as `(N, 1, frames * upsample_factor)` in `[-1, 1]`.
    ///
    /// `mel` is `(N, bands, frames)`, in whatever float type and on whatever device -- the pass
    /// begins by bringing it here. It is the log-mel the model was trained on and not a linear
    /// one; nothing in this model would notice the difference, and everything downstream would.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let dim = mel.dim()?;
        if dim != 3 {
            return Err(Error::model(format!(
                "a mel spectrogram is (N, bands, frames), got a {dim}-D tensor"
            )));
        }

        let bands = mel.shape_at(1)?;
        if bands != self.config.mel_channels {
            return Err(Error::model(format!(
                "a mel of {bands} bands is not the {} this vocoder reads",
                self.config.mel_channels
            )));
        }

        let context = RunContext::new(&*self.weights)
            .preloaded(&self.preloaded)
            .input("mel", mel);

        output(&self.ir.run(&context)?, "waveform")
    }
}
