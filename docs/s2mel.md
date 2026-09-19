# S2Mel

The middle of IndexTTS-2.5. The GPT in front of it produces semantic tokens; this turns them into
the mel spectrogram a BigVGAN makes audio of.

```rust
use waifu::s2mel::{self, Config};

let direction = s2mel::graph(
    &g, x, prompt_x, cond, style, time, time2,
    &Config::indextts(), frames, cos, sin, dtype, device,
)?;
```

It is a **flow-matching** model, so the network does not produce a mel. It produces one step of a
trajectory — given a partly denoised mel and a time, the direction to move — and a sampler calls
it a handful of times. What is implemented here is the network. See *What is missing* below.

## What it is conditioned on

Four different things, which is worth spelling out because they arrive separately:

| | |
| --- | --- |
| `cond` | the semantic tokens, one 512-wide vector per output frame |
| `prompt_x` | the reference speaker's own mel where there is one and zeros after, which is what makes this *continue* a voice rather than invent one |
| `style` | 192 numbers from [`campplus`](campplus.md) — who is speaking |
| `t` | where along the trajectory this step is |

## The shape of it

A thirteen-layer transformer over the frame axis, then an eight-layer WaveNet over the same axis.

The transformer is `gpt_fast`'s rather than a diffusion model's: RMS normalization, rotary
positions, a SwiGLU feed forward. What makes it a *diffusion* transformer is that every
normalization is adaptive — the timestep projects to a weight and a bias that scale and shift it
— and that half the layers hand a skip connection forward to the other half, U-net fashion.

## Three things that are not what a reader would guess

### The rotary embedding is interleaved, not split

`flint`'s `rotary_embedding` rotates the NeoX way, pairing element `i` with element `i + D/2`.
This model pairs `2i` with `2i + 1` — the other convention entirely, and it gives different
numbers for the same weights. So `rotate` does it explicitly, out of cosine and sine tables the
caller passes in, the same way `anima`'s transformer does.

The tables are `(T, head_dim / 2)` and the tensor is laid out with heads before time, so they
line up against the last two axes and broadcast over the rest without anything being moved.

### The content is read as continuous, whatever the configuration says

`config.yaml` sets `content_type: 'discrete'` and the model builds a `cond_embedder` for it. But
the reference's `forward` assigns `cond_in_module = self.cond_projection` unconditionally, with
the line that would have chosen between them commented out just above it. So the embedding is
never reached and the projection always is. This follows the code, not the configuration.

### The final modulation is not the adaptive norm

Every adaptive normalization in the transformer applies the projected weight as it comes:
`weight * norm(x) + bias`. The `FinalLayer` at the end does `(1 + scale) * norm(x) + shift`
instead, over a `LayerNorm` with no affine of its own. Two different conventions in one model.

## What is checked

`waifu/tests/s2mel.rs`, in the fast suite, against IndexTTS's own implementation —
`tools/s2mel_reference.py` downloads the reference and runs it. No package and no download at
test time: every parameter is filled from its own name, by the same FNV-1a hash and the same sine
in both languages.

The model under test has **every width cut and every count kept** — 16 mel bands, a 64-wide
transformer, four heads, but still thirteen transformer layers and eight WaveNet layers. Depth is
what the skip connections turn on: half the layers hand one forward to the other half, taken off
the end of a list, so which layer receives which is a function of the depth and nothing else. At a
depth of two there is nothing to get wrong; at thirteen there is an off-by-one in three places.

Two tests, so a failure lands on one side or the other:

| test | what it covers |
| --- | --- |
| `the_transformer_trunk_is_the_reference_trunk` | the skips and the rotary convention |
| `the_denoiser_is_the_reference_denoiser` | end to end, including the WaveNet and the final modulation |

```bash
cargo test --manifest-path waifu/Cargo.toml --test s2mel
.venv/bin/python tools/s2mel_reference.py        # to regenerate the constants
```

## What is missing

**The sampler.** `solve_euler` in the reference is a host-side loop: take `n_timesteps` even
steps from 0 to 1, call the network at each, and move the mel by `dt` times what comes back. It
also does classifier-free guidance, running the network a second time with `prompt_x`, `style`
and `cond` zeroed and combining the two as `(1 + rate) * cond - rate * uncond`; and it holds the
prompt frames at zero throughout. None of that is written yet. It is perhaps sixty lines of
orchestration around the graph, and it is the difference between a model and a model you can run.

**The length regulator**, which turns the GPT's token sequence into one vector per mel frame, is
a separate module and is not written either.

**Padding masks.** The reference builds a mask from `x_lens` and passes it to the attention. For a
batch of one at full length that mask is all true, which is what this assumes. A batch of several
utterances of different lengths would need it.

**The exporter.** There is none yet; `tools/s2mel_reference.py` fetches the reference source but
never the 415 MB checkpoint. Note for whoever writes it: several layers are wrapped in
`weight_norm`, which the reference removes when it loads, so the folded weight is what to export
— the same thing `tools/bigvgan_exporter.py` does.
