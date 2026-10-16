# FP8 weights

A weight held in E4M3 with one scale for the whole of it, multiplied by an activation that stays at
full width.

```cpp
op::cuda::gemmFp8(xFp16, codes, scale, y);   // per layer: half in, half out
```

`codes` is `<fp8e4m3>(rows, k)`, `scale` a single `<float>` on the device, and `y`
`<half>(..., rows)` a view the caller allocated, as every operator writes into.

Needs `WITH_CUDA=ON` and an sm_80 or newer device, which is everything from Ampere on, or Metal.
`isFp8GemmAvailable()` reports the CUDA half of that.

Nothing here quantizes. A weight arrives quantized, in a package an exporter wrote (see "In a
package"), and the runtime only multiplies by it. There used to be a runtime quantizer and a
format with one scale per row; both are gone -- see "Why one scale for the whole weight".

What the narrow weight buys, and does not:

| | |
| --- | --- |
| what the weight costs | half the bytes from global memory, half the shared memory |
| what the multiply costs | the same HMMA, so up to 1.85x at a small row count and 10% behind at a large one |

## From Rust

`functional::fp8_matmul` takes the weight as the two tensors it is made of:

```rust
let y = F::fp8_matmul(&x, &codes, &scale)?;   // float16 in, float16 out; scale is <float>[1]
```

Two arguments rather than one pair, because a weight a package stored quantized never becomes a
pair: it arrives as two ordinary tensors under two ordinary names. See "In a package" below. The
C interface is the same shape -- `fl_fp8_matmul` takes the two as two views.

`flint::fp8_available(device)` answers whether that device can run it. On CUDA `k` has to be a
multiple of 16 and the weight's row count a multiple of 8, which the C interface checks rather than
leaving to the kernel.

The C interface is `fl_fp8_available` and `fl_fp8_matmul`. Both dispatch on the device the tensors
are already on -- one entry point per operation rather than one per backend.

The kernels assert their preconditions with `CHECK`, which reports a broken invariant as
`FL_ERROR_ABORTED`. The C interface checks device, type, contiguity and shape itself first even
so -- of the weight and its scale as much as of the activation and the result: those are the
caller's mistakes rather than the library's, so they come back as `FL_ERROR_INVALID_ARG` naming
what was wrong, instead of as an internal failure with a stack trace behind it.

## In a package

A model whose weights were quantized before they were published stores each one as **two
tensors**: `"…weight"` as `F8_E4M3` `(out, in)`, and `"…weight.scale"` as `F32` `(1,)`.

```
proj.weight         F8_E4M3  [2560, 2048]
proj.weight.scale   F32      [1]
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
%8  = load("proj.weight.scale", [1])
%9  = fp8_matmul(%3, %7, %8)
```

Two `load`s rather than one composite value, which is the whole point of storing it this way: a
scale is then an ordinary weight under an ordinary name, and everything that walks loads reaches it
without being taught that FP8 exists -- `resident` reading the package onto the device, the
liveness the `free` instructions come from, `Residency::Hybrid` putting one weight at a time
across the bus.

Note also what is *not* there: the `transpose` a float `matmul` needs before it. `fp8_matmul` reads
the weight in the `(out, in)` a package stores.

### The model says so, the reader does not guess

Whether a package holds its matrices quantized comes from its manifest:

```yaml
config:
  krea2:
    weight_format: fp8_tensor_scale   # "float" when absent
```

`Linear` builds the quantized multiply under `WeightFormat::Fp8TensorScale` and the ordinary one
otherwise. A graph could instead go looking for the scales and decide from what it found, and
deliberately does not: then what a model *is* would depend on what a reader happened to see, two
packages with the same configuration could compile to different passes, and a package that lost
half its scales would quietly build a mixture instead of failing. Configuration is where a package
says what is in it; the tensors are what it says it with.

`weight_format: fp8` was the format with one scale per row. It is no longer a name a manifest may
use, and a package that still has a scale per row beside its weights is refused when it is read,
with a message that says to export it again.

The autoencoders do not read this. An autoencoder is convolutions, its few projections are small,
and SDXL runs that half in float32 where the CUDA FP8 multiply takes float16 -- so the format a
package names is for the halves that have the matrices in them.

### Writing one

`tools/model_writer.py` has `WeightsWriter.write_fp8_tensor_scale`, which quantizes and writes the
pair, and `Quantization.quantize_to_fp8_tensor_scale`, which is the arithmetic:

```
scale      = amax(|x|) / 448      // 448 is E4M3's largest finite magnitude
data[i][j] = e4m3(x[i][j] / scale)
```

so the largest element lands exactly on the top of the format's range. An all zero weight gets a
zero scale and quantizes to zeros rather than to NaN. The docstring says which details of it are
load bearing. `tools/krea2_exporter.py` and `tools/qwen_image_exporter.py` call it for `-fp8`.

## Why one scale for the whole weight

One scale per row would be more accurate. A projection's output channels do not share a magnitude,
and with one scale for all of them the largest sets the scale every row is divided by: a row far
below it lands in E4M3's coarser codes near zero, where a scale of its own would have put it on the
format's full range. Which is worth it for a given weight is the exporter's call, the same way
choosing FP8 at all is.

What one scale buys is one format. A checkpoint quantized elsewhere usually arrives this way -- E4M3
codes and a single `weight_scale` -- and is multiplied as stored, and every package here is now
written the same way. The per-row format, the runtime quantizer that produced it and the kernels
that read it were removed rather than kept beside it.

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

FP8 here keeps the instruction and buys traffic, so it needs only the weight narrow and leaves the
activation exact.

It is the same story on a processor with the ceiling lower, which is worth knowing before wanting
one: x64 and aarch64 have no FP8 arithmetic at all, so the weight would be widened to float32
between memory and the micro-kernel and the multiply would be the float32 one it was anyway.

## The epilogue, on the GPU

`D = accumulator * scale`, as a CUTLASS 2.x EVT tree: `VisitorAccFetch` and a
`VisitorScalarBroadcast` over the scale, multiplied by `VisitorCompute<cutlass::multiplies>` and
written by `VisitorAuxStore`. The scale is read through a pointer as the epilogue starts, so a scale
that lives on the device never has to come back to the host to be launched with, and it costs no
pass of its own.

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

These were measured with one scale per row. The mainloop and the tiles are the same now, and the
epilogue reads one scale where it read a row of them, so it costs no more than it did.

### Accuracy

What one scale costs depends on the weight: rounding to three mantissa bits is a step of one
sixteenth at the top of a binade, and a row well below the weight's largest is further down the
format than that. The multiply adds nothing of its own to the format's error. What it comes to for
a whole model is in that model's document -- `docs/krea2.md` and `docs/qwen_image.md`.

## Known gaps

- **Small n is latency bound.** At 1x3072x3072 the kernel reaches 397 GB/s, which is DRAM speed,
  but the weight is small enough to be in L2 and it never benefits: 48 CTAs of two warps each is
  96 warps over 36 SMs, and the mainloop's 48 K iterations are a dependent chain that a four stage
  pipeline does not fully hide. Eight stages was tried and is worse (45.3 us against 23.4). A
  GEMV-shaped kernel, which is what cuBLAS reaches for here, is the real answer for one row.
- **No split K.** The SM80 EVT epilogue asserts against it, so a shape with too few n tiles to
  fill the machine has nothing to fall back on. `gemm_cutlass.cu`'s `splitKSlices` is what the
  half path does instead, and it is worth 25% to 55% on the shapes it fires for.
- **The processor has no kernels**, so a package written with a quantized weight runs on CUDA and
  Metal only: `resident` will put the bytes on a processor perfectly well and the multiply is what
  refuses them. See "What a processor would need" below.
- **No CI runner exercises the multiply.** What CI can check is the half that needs no kernel --
  that the file reader takes `F8_E4M3`, that it refuses elements with no scale beside them, and
  that a quantized projection is built out of two loads -- and it does. Everything past that is
  `#[ignore]`d for want of a card. This used to be covered on both halves and is the coverage the
  CPU path took with it.
- `n` must be a multiple of 8, which is how wide the epilogue writes, and `k` a multiple of 16,
  which is how wide the mainloop reads the weight. The row count is free. Unlike the half GEMM,
  which picks its alignment per call from the operands it is handed, this one is instantiated at
  one alignment and refuses anything narrower by name -- an exporter leaves a matrix whose `k` 16
  does not divide in float. `can_implement` is asked before every launch regardless, since
  CUTLASS's `initialize` returns success on strides the kernel cannot read.

## What a processor would need

There was a CPU path here and it was taken back out, so this is what it was, what it measured, and
what bringing it back costs. None of it is lost work: the format is unchanged, and a weight
quantized for the card is the same bytes a processor would read.

The place for it already exists. `Gemm` in `cpu/kernel/gemm.h` takes the type B is stored in
separately from the type the micro-kernel computes in and converts in `Pack()` -- which is how
`gemmHalfWeightFloat`, the float-activation-by-half-weight path a model on the CPU runs today,
already works. An E4M3 weight is that arrangement with a narrower B, so the GEMM itself is `wgemm`
instantiated at `TB = Fp8E4M3` and the micro-kernel never learns the format exists. The scale
goes afterwards as a pass over C: on the GPU it is folded into the epilogue because the epilogue
is free there, but `M*N` against the multiply's `M*N*K` does not justify touching the kernel
here.

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
