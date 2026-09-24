# FP8 weights

A weight held in E4M3 with one scale per output channel, multiplied by an activation that stays at
full width.

```cpp
Fp8Operand w = op::cuda::quantizeFp8(weightFp16);   // once, at load
Tensor y = op::cuda::gemmFp8(xFp16, w);             // per layer: half in, half out
```

Needs `WITH_CUDA=ON` and an sm_80 or newer device; unlike the NVFP4 path, which needs sm_120a
exactly, that is everything from Ampere on. `isFp8GemmAvailable()` reports it.

**The card is the only device that has it.** Nothing about the format is the card's -- `Fp8Operand`
and the bytes it holds are device agnostic, and a package that stores a weight quantized says
nothing about where it will be multiplied -- but the kernels that read those bytes are CUDA's, so
`fl_fp8_available` answers no everywhere else and the C interface refuses an operand that is not on
a card. A processor path existed and was taken back out; see "What a processor would need" at the
end for what it was and what it was worth.

What the narrow weight buys, and does not:

| | |
| --- | --- |
| what the weight costs | half the bytes from global memory, half the shared memory |
| what the multiply costs | the same HMMA, so up to 1.85x at a small row count and 10% behind at a large one |

## From Rust

`functional::fp8_matmul` takes the weight as the two tensors it is made of. `flint::Fp8Tensor` is
what quantizing hands both back in:

```rust
let w = Fp8Tensor::quantize(&float16_weight)?;              // on CUDA
let y = F::fp8_matmul(&x, w.data(), w.channel_scale())?;    // float16 in, float16 out
```

Two arguments rather than one pair, because a weight a package stored quantized never becomes a
pair: it arrives as two ordinary tensors under two ordinary names. See "In a package" below. The
C interface has always been this shape -- `fl_fp8_matmul` takes two handles and puts them together
on the other side.

`Fp8Tensor::is_available(device)` answers whether that device can run it, which today is CUDA and
nothing else. `k` has to be a multiple of 16 and the weight's row count a multiple of 8, which the
C interface checks rather than leaving to the kernel.

The C interface is `fl_fp8_available`, `fl_fp8_quantize`, `fl_fp8_dequantize` and `fl_fp8_matmul`.
All four dispatch on the device the tensors are already on -- one entry point per operation rather
than one per backend, which is the shape to keep whether there is one backend or two.

The kernels assert their preconditions with `CHECK`, which reports a broken invariant as
`FL_ERROR_ABORTED`. The C interface checks device, type, contiguity and shape itself first even
so, and `makeFp8Operand` checks what a caller hands back: those are the caller's mistakes rather
than the library's, so they come back as `FL_ERROR_INVALID_ARG` naming what was wrong, instead of
as an internal failure with a stack trace behind it.

## In a package

A model whose weights were quantized before they were published stores each one as **two
tensors**: `"…weight"` as `F8_E4M3` `(out, in)`, and `"…weight.scale"` as `F32` `(out,)`.

```
proj.weight         F8_E4M3  [2560, 2048]
proj.weight.scale   F32      [2560]
```

Two tensors and not one, because that is what the format can say. A safetensors header holds a
dtype, a shape and two offsets per tensor, and the only free text in the file is one string map for
the whole of it -- so there is nowhere to write down that these two belong together, and the name
is the whole of the pairing. Reading a package is where that convention is enforced --
`read_safetensors` and `Weights::from_files` go through the same reader: an `F8_E4M3` tensor
with no scale beside it, or with one of the wrong shape or type, is refused when the file is read.
A weight scaled by nothing would otherwise load perfectly well and draw noise.

The reverse is deliberately not checked. A tensor whose name happens to end in `.scale` is an
ordinary tensor, since this suffix is a convention of this library rather than a word reserved in
the format.

### What the pass does with them

The graph loads two weights and hands three operands to one node:

```
%7  = load("proj.weight", [2560, 2048])
%8  = load("proj.weight.scale", [2560])
%9  = fp8_matmul(%3, %7, %8)
```

Two `load`s rather than one composite value, which is the whole point of storing it this way: a
scale is then an ordinary weight under an ordinary name, and everything that walks loads reaches it
without being taught that FP8 exists -- `resident` reading the package onto the device, the
liveness the `free` instructions come from, `Residency::LowVram` putting one weight at a time
across the bus. The scale costs `4/k` of what the weight costs, which at `k=3072` is 0.13%.

Note also what is *not* there: the `transpose` a float `matmul` needs before it. `fp8_matmul` reads
the weight in the `(out, in)` a package stores.

### The model says so, the reader does not guess

Which of the two a package holds comes from its manifest:

```yaml
config:
  sdxl:
    weight_format: fp8      # "float" when absent, which is every package so far
```

`Linear` builds the quantized multiply under `WeightFormat::Fp8` and the ordinary one otherwise.
A graph could instead go looking for the scales and decide from what it found, and deliberately
does not: then what a model *is* would depend on what a reader happened to see, two packages with
the same configuration could compile to different passes, and a package that lost half its scales
would quietly build a mixture instead of failing. Configuration is where a package says what is in
it; the tensors are what it says it with.

The autoencoders do not read this. An autoencoder is convolutions, its few projections are small,
and SDXL runs that half in float32 where the CUDA FP8 multiply takes float16 -- so the format a
package names is for the halves that have the matrices in them.

### Writing one

`tools/model_writer.py` has `WeightsWriter.write_fp8_tensor`, which quantizes and writes the pair,
and `Quantization.quantize_to_fp8`, which is the same arithmetic the runtime's own quantizer does.
It has to be the same: a package written there is read by this, and this is one format rather than
two only if every producer agrees on the bytes. Three details in it are load bearing -- the
rounding of the division above is one -- and the docstring says which.

`tools/krea2_exporter.py` and `tools/qwen_image_exporter.py` (which shares its `Converter`) no
longer write this one for `-fp8`; see "One scale for the whole tensor" below for the format they
write instead. `write_fp8_tensor` and `quantize_to_fp8` stay for a package that wants one scale per
row on purpose -- an exporter is free to call either.

## One scale for the whole tensor

`WeightFormat::Fp8TensorScale` is everything above with one difference: `"…weight.scale"` is a
single `<float>`, not one per row, and the graph reads it that way -- `Linear::graph` loads it at
shape `[1]` rather than `[out_dim]` and builds `fp8_matmul_tensor_scale` in place of `fp8_matmul`.
Same elements, same pairing, same suffix; only the scale's shape and which multiply reads it change.

```rust
let y = F::fp8_matmul_tensor_scale(&x, w.data(), &tensor_scale)?;   // tensor_scale is <float>[1]
```

`flint::gemmFp8TensorScale` is the CUDA side of it (`flint/cuda/gemm_fp8_cutlass.h`,
`fl_fp8_matmul_tensor_scale` in the C interface): the same CUTLASS 2.x mixed-input mainloop as
`gemmFp8`, with `VisitorScalarBroadcast` in the epilogue where `gemmFp8` has
`VisitorRowBroadcast` -- a value read once and broadcast over the whole tile rather than a row
vector. `check_fp8_pairs` accepts `<float>[rows]` or `<float>[1]` beside an E4M3 tensor for this
reason: which shape a package actually means is the manifest's `weight_format`, read once the graph
is built, not something the file-level check is in a position to enforce ahead of that.

`Quantization.quantize_to_fp8_tensor_scale` in `tools/model_writer.py` is the same arithmetic as
`quantize_to_fp8`, over the whole matrix instead of one row: `scale = amax(|x|) / 448` taken across
every element, not per row. `WeightsWriter.write_fp8_tensor_scale` writes the pair; a manifest that
wants this format says `weight_format: fp8_tensor_scale`.

What one scale buys over one per row: a smaller pair -- `<float>[1]` against `<float>[out_dim]` --
and nothing else; the E4M3 elements are the same size either way. What it costs is precision on a
weight whose rows vary widely in magnitude, since the whole tensor's largest element sets the scale
every row is divided by: a row far below that magnitude lands in E4M3's coarser codes near zero,
where the per-row format would have scaled it onto the format's full range on its own. Which is
worth it for a given weight is the exporter's call, the same way choosing FP8 at all is.

## Why the activation is not narrowed

Nothing here multiplies in FP8. The mainloop is CUTLASS's 2.x mixed input tensor op, reached by
tagging the operator `arch::OpMultiplyAddMixedInputUpcast`: it loads the E4M3 weight through
`ldmatrix`, shuffles the bytes into the order `mma.sync` wants and converts them to half in
registers, one K tile at a time. The instruction it then issues is the ordinary one --

```
$ cuobjdump -sass gemm_fp8_cutlass.cu.o
      96   HMMA.16816.F32
```

-- the same `HMMA.16816.F32` a half GEMM runs. So quantizing the activation as well would buy
nothing: it would be converted straight back to half before the multiply. What the narrow weight
saves is bandwidth -- half the bytes from global memory and half the shared memory -- and that is
the whole of what this path is for.

That is the difference from `docs/nvfp4.md`, and it is the reason both exist. NVFP4 changes the
instruction, so it buys arithmetic and needs both operands narrow; FP8 here keeps the instruction
and buys traffic, so it needs only the weight narrow and leaves the activation exact.

It is the same story on a processor with the ceiling lower, which is worth knowing before wanting
one: x64 and aarch64 have no FP8 arithmetic at all, so the weight would be widened to float32
between memory and the micro-kernel and the multiply would be the float32 one it was anyway.

## The quantizer

One CUDA block owns one row, which makes the scale a block reduction rather than a second launch:
a row's maximum is known before the same block quantizes it. The row is read twice, once for the
maximum and once for the elements, which is the cost of keeping it in one launch -- and a weight
is quantized once at load, so it is not a cost anything pays twice.

```
channelScale[r] = rowAmax[r] / 448      // 448 is E4M3's largest finite magnitude
data[r][j]      = e4m3(x[r][j] / channelScale[r])
```

so the largest element of each row lands exactly on the top of the format's range. A row that is
all zero gets a zero scale and quantizes to zeros rather than to a NaN.

### The division is round to nearest, on purpose

`gemm_fp8_cutlass.cu` is compiled with `--use_fast_math`, which turns `448.0f / amax` into a
reciprocal approximation. One ulp off is enough to move an element onto the next code -- it moved
two in 16896 -- so those divisions are `__fdiv_rn`.

It matters beyond this kernel. Anything else that produces these bytes has to produce the same
ones, or a weight is not one format but two: an exporter that writes a package (see "In a package"
above) is doing this arithmetic in float32 on a processor, and a quantizer written for a processor
here would be doing it again. A reciprocal approximation in any of them would disagree with the
others in a few elements per ten thousand, which is exactly the kind of difference nothing but an
exact comparison finds.

One scale per row rather than one per tensor, because a projection's output channels do not share
a magnitude: with a single scale, one outlier channel pushes every other channel down into E4M3's
subnormals. It is also the coarsest scaling the multiply can undo for free -- a scale constant
down a column of the result is a multiply in the epilogue, where a finer one would have to be
applied inside the mainloop.

## The epilogue, on the GPU

`D = accumulator * channelScale[n]`, as a CUTLASS 2.x EVT tree: `VisitorAccFetch` and a
`VisitorRowBroadcast` over the scale, multiplied by `VisitorCompute<cutlass::multiplies>` and
written by `VisitorAuxStore`. The scale is a row vector -- stride zero on m, one on n -- so it
costs no pass of its own and no second buffer.

The one thing it costs: `GemmWithEpilogueVisitor::Params` asserts against a split K, so unlike the
half GEMM beside it (`gemm_cutlass.cu`, where `splitKSlices` decides per call) this kernel cannot
fill an idle machine by splitting k. The flat tile below is what fills it instead, along n.

Header order matters in `gemm_fp8_cutlass.cu`: `visitors.hpp` reaches for things it does not
include itself -- `NumericArrayConverter` among them -- so `gemm/device/gemm_universal.h` has to
come first. CUTLASS's own example 47 has the same order, and getting it wrong produces a hundred
lines of errors inside CUTLASS headers that say nothing about the cause.

## Two tiles, picked by the row count (GPU)

At one row a 128 row tile computes 128 rows and throws 127 away, and since the arithmetic is the
same HMMA a half GEMM runs, that waste is the whole cost. The flat tile also has a narrow n, which
is the other half of the same problem: at one row the tile count is the n tile count, and 3072
output channels over a 128 wide tile is 24 CTAs on a 36 SM part.

Microseconds on an RTX 5060 Ti. The 16384 by 3072 weight is 50 MB in E4M3, larger than the 32 MB
L2, so those rows are against cold weights; the others are not.

| shape | 32x64x64 | 32x128x64 | 128x128x64 |
| --- | --- | --- | --- |
| 1x1280x1280 | **10.4** | 17.4 | 32.4 |
| 1x3072x3072 | **23.4** | 39.7 | 75.0 |
| 1x16384x3072 | 131.2 | **128.2** | 302.2 |
| 16x3072x3072 | **23.6** | 39.8 | 75.0 |
| 64x3072x3072 | 41.0 | **40.4** | 75.3 |
| 128x5120x3072 | **98.8** | 118.2 | 150.6 |
| 512x5120x3072 | 370.9 | **355.9** | 375.1 |
| 512x16384x3072 | 1158.6 | 1134.3 | **1122.2** |
| 1024x10240x1280 | 618.5 | 616.7 | **588.0** |
| 4096x640x2560 | 334.1 | 344.8 | **315.0** |

So the flat tile is ahead by up to three times until the rows fill a square one, and the square
tile is never more than 5% ahead beyond that. `gemmFp8` splits them at 512 rows, and there are two
instantiations rather than three because nothing in between was worth its compile time.

Two other tile facts, both forced rather than chosen:

- The K tile is 64 and cannot be more. A half operand's shared memory layout is
  `TensorOpMultiplicandCrosswise<16, kK>`, whose crosswise extent is a 128 byte run, so 64 halves
  is where it ends. The deeper K tile a narrow operand would otherwise want is out of reach while
  A stays half.
- Two stages does not compile. At two, CUTLASS builds the mainloop out of `MmaPipelined` rather
  than `MmaMultistage`, and the mixed input warp operator has no overload it can call. Three works
  and measures within half a percent of four everywhere.

## Measurements on the GPU

RTX 5060 Ti (sm_120a, 36 SMs, 448 GB/s), CUDA 12.9, against cuBLAS FP16 on the same shapes.

### Speed

| shape | FP16 (cuBLAS) | FP16 x FP8 | |
| --- | --- | --- | --- |
| 1x16384x3072 (cold) | 237.1 us | **130.6 us** | 1.82x |
| 16x16384x3072 (cold) | 244.6 us | **132.1 us** | 1.85x |
| 64x16384x3072 (cold) | 264.8 us | **170.9 us** | 1.55x |
| 512x16384x3072 | 1052.8 us | 1167.8 us | 0.90x |
| 512x5120x3072 | 337.7 us | 373.7 us | 0.90x |
| 512x3072x8192 | 548.6 us | 618.6 us | 0.89x |
| 1024x10240x1280 (SDXL) | 560.6 us | 587.2 us | 0.95x |
| 77x2560x2048 (SDXL) | 35.2 us | **31.4 us** | 1.12x |
| 1x3072x3072 | 10.8 us | 23.7 us | 0.46x |

The first three rows are the case this exists for: a weight too large for L2, read once and
multiplied by few rows. There FP8 moves half the bytes at the same bandwidth and takes half the
time -- 130.6 us for 50 MB is 385 GB/s of a 448 GB/s part, and the FP16 GEMM beside it is 237 us
for 100 MB, which is 425 GB/s. Same bus, half the traffic.

Where the rows are many the arithmetic is the ceiling, and since it is the same HMMA either way,
FP8 can only lose: 10% at 512 rows, 5% on SDXL's largest shape. That is the price of the memory,
and the memory is the point -- the weight is half the size on the card, and it stays half the size
whatever the row count is.

The last row is the one to read carefully. A 3072 by 3072 weight is 18.9 MB in half, which fits
the 32 MB L2, and the benchmark runs one shape twenty times over, so cuBLAS's 10.8 us is 1750 GB/s
-- an L2 number that no real decode step gets. Ours is 23.7 us at 397 GB/s, which is DRAM speed,
because at 48 CTAs of two warps each the kernel is latency bound rather than bandwidth bound and
never gets far enough ahead to live in L2. See the gaps below.

The prologue costs what a full pass over the weight costs: 380 us for a 16384 by 3072 weight
(100 MB read, 50 MB written), 22 us for 3072 by 3072. It is paid once, at load.

### Accuracy

Relative RMSE, `sqrt(sum((x - ref)^2) / sum(ref^2))`, of the FP8 path against the FP16 GEMM, which
is the thing it replaces. `weight` is the quantized weight against the one it was made from, and
`gemm` is the whole multiply.

| shape (m,n,k) | weight | gemm |
| --- | --- | --- |
| 512x5120x3072 | 2.647e-02 | 2.649e-02 |
| 512x3072x8192 | 2.647e-02 | 2.645e-02 |
| 1x5120x3072 | 2.647e-02 | 2.647e-02 |
| 1024x10240x1280 | 2.644e-02 | 2.644e-02 |

The two columns agree to three figures, which is the result worth having: the GEMM adds nothing of
its own to the format's error. Three mantissa bits is a step of one sixteenth at the top of a
binade, and 2.6e-2 is what rounding normally distributed data to that costs.

For scale: `docs/nvfp4.md` measures 9.5e-2 for a single NVFP4 operand and 1.34e-1 for two, and the
FP16 GEMM's own error against an FP32 one is 3.6e-4. So this sits between them -- about 3.6 times
more accurate than NVFP4 on one operand, and about seventy times less accurate than FP16.

## Known gaps

- **Small n is latency bound.** At 1x3072x3072 the kernel reaches 397 GB/s, which is DRAM speed,
  but the weight is small enough to be in L2 and it never benefits: 48 CTAs of two warps each is
  96 warps over 36 SMs, and the mainloop's 48 K iterations are a dependent chain that a four stage
  pipeline does not fully hide. Eight stages was tried and is worse (45.3 us against 23.4). A
  GEMV-shaped kernel, which is what cuBLAS reaches for here, is the real answer for one row.
- **No split K.** The SM80 EVT epilogue asserts against it, so a shape with too few n tiles to
  fill the machine has nothing to fall back on. `gemm_cutlass.cu`'s `splitKSlices` is what the
  half path does instead, and it is worth 25% to 55% on the shapes it fires for.
- **No model is published quantized.** A package *can* store E4M3 now -- see "In a package" above,
  which is read, checked and built into the pass -- and none of the models this library ships does.
  Until one does, every weight here is still quantized at load if it is quantized at all, so the
  file and the load time are what they always were and the saving starts once the weight is on the
  device.
- **The quantizer reads each row twice.** One pass would need either a second launch or a scale
  chosen before the data is seen. At 380 us for a 50 MB weight it is a load-time cost only.
- **Nothing is built out of it yet.** The diffusion model this crate runs is float16 throughout,
  and at its row counts -- 1024 and 4096 -- this path is 5% behind cuBLAS rather than ahead. What
  it is for is a weight read once per token, which is the language model shape.
- **The card is the only device with the kernels**, so a package written with a quantized weight
  loads nowhere else: `resident` will put the bytes on a processor perfectly well and the multiply
  is what refuses them. See "What a processor would need" below.
- **No CI runner exercises the multiply.** What CI can check is the half that needs no kernel --
  that the file reader takes `F8_E4M3`, that it refuses elements with no scales beside them, and
  that a quantized projection is built out of two loads -- and it does. Everything past that is
  `#[ignore]`d for want of a card. This used to be covered on both halves and is the coverage the
  CPU path took with it.
- `n` must be a multiple of 8, which is how wide the epilogue writes, and `k` a multiple of 16,
  which is how wide the mainloop reads the weight. The row count is free. Unlike the half GEMM,
  which picks its alignment per call from the operands it is handed, this one is instantiated at
  one alignment and refuses anything narrower by name -- the weight comes from our own quantizer,
  which already insists on a `k` that 16 divides. `can_implement` is asked before every launch
  regardless, since CUTLASS's `initialize` returns success on strides the kernel cannot read.

## What a processor would need

There was a CPU path here and it was taken back out, so this is what it was, what it measured, and
what bringing it back costs. None of it is lost work: the format is unchanged, and a weight
quantized for the card is the same bytes a processor would read.

The place for it already exists. `Gemm` in `cpu/kernel/gemm.h` takes the type B is stored in
separately from the type the micro-kernel computes in and converts in `Pack()` -- which is how
`gemmHalfWeightFloat`, the float-activation-by-half-weight path a model on the CPU runs today,
already works. An E4M3 weight is that arrangement with a narrower B, so the GEMM itself is `wgemm`
instantiated at `TB = Fp8E4M3` and the micro-kernel never learns the format exists. The channel
scale goes afterwards as a pass over C: on the GPU it is folded into the epilogue because the
epilogue is free there, but `M*N` against the multiply's `M*N*K` does not justify touching the
kernel here.

Three kernels beyond that were each worth a measured factor, and are the reason it is not a
weekend's work to redo:

| | without | with | on |
| --- | --- | --- | --- |
| a blocked transpose in the pack | 102.6 ms | 32.2 ms | 1024x10240x1280 |
| a GEMV path (`dot`, `axpy`) | 5.0 ms | 3.3 ms | 1x5120x3072 |
| a gathered lookup inside both | 3.3 ms | 0.22 ms | 1x5120x3072 |

The first is a reading pattern rather than arithmetic: a weight is stored one output channel per
row, so the pack reads it down its columns, and a generic strided loop touches a separate cache
line for every byte it wants -- at a 1280 byte stride, 64 bytes moved per byte used.

The third is the one worth remembering. A scalar E4M3 to float conversion has two branches in it,
subnormals and the NaN code, and eight of those per vector cost about twenty cycles an element in a
loop that should run at one. There is no `_mm256_cvtph_ps` for 8 bits, so what replaces them is a
256 entry table -- the whole format is 256 values -- gathered with `_mm256_i32gather_ps`. Tabulate
the scalar conversion rather than writing a second one, and the two cannot drift. aarch64 wants the
same thing as a NEON table lookup (`vqtbl4q_u8`, four of them for 256 entries) and never got it,
so it would take the scalar path and pay what the table above says.

What it measured, on a 32 thread AVX-512 machine, fastest of twenty iterations, in milliseconds:

| shape (m,n,k) | f32 x f32 | f32 x f16 | f32 x fp8 |
| --- | --- | --- | --- |
| 1x5120x3072 | 0.28 | 0.19 | 0.22 |
| 8x5120x3072 | 1.28 | 0.82 | **0.74** |
| 77x2560x2048 | 0.69 | 0.66 | 0.67 |
| 1024x1280x1280 | 2.46 | 2.46 | 2.55 |
| 1024x10240x1280 | 29.1 | 28.9 | 28.9 |
| 4096x1920x640 | 6.09 | 6.05 | 6.46 |

Parity within a few percent either way, and half the weight in memory. That is the result to
expect rather than a disappointing one: the arithmetic is float32 in all three columns, and the
only thing FP8 changes is how many bytes have to reach it. Which is also why it was the half to
take out first -- on the card the narrow weight buys time as well as memory, and here it buys only
memory.

It would be worth having again for one reason, and it is a good one: a model that does not fit
stops not fitting. SDXL is 6.97 GB on the processor and its weights are nearly all of that.
