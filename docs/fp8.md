# FP8 weights on the half tensor cores

`op::cuda::gemmFp8` multiplies a half activation by an E4M3 weight and returns half. The weight
carries one scale per output channel; the activation is not quantized at all.

```cpp
Fp8Operand w = quantizeFp8(weightFp16);   // once, at load
Tensor y = gemmFp8(xFp16, w);             // per layer: half in, half out
```

Requires `WITH_CUDA=ON` and an sm_80 or newer device. `isFp8GemmAvailable()` reports whether this
build and this GPU can run it. Unlike the NVFP4 path, which needs sm_120a exactly, that is true
of everything from Ampere on: what it asks is only whether the tensor cores the kernel is written
against exist.

## From Rust

`flint::Fp8Tensor` holds the two pieces a quantized weight is made of, and
`functional::fp8_matmul` multiplies by one:

```rust
let weight = Fp8Tensor::quantize(&float16_weight)?;
let y = F::fp8_matmul(&x, &weight)?;   // float16 in, float16 out
```

`Fp8Tensor::is_available()` answers whether this build and this GPU can run it. `k` has to be a
multiple of 16 and the weight's row count a multiple of 8, which the C interface checks rather
than leaving to the kernel.

The C interface is `fl_fp8_available`, `fl_fp8_quantize`, `fl_fp8_dequantize` and `fl_fp8_matmul`.
The kernels assert their preconditions with `CHECK`, which reports a broken invariant as
`FL_ERROR_ABORTED`. The C interface checks device, type, contiguity and shape itself first even
so, and `makeFp8Operand` checks what a caller hands back: those are the caller's mistakes rather
than the library's, so they come back as `FL_ERROR_INVALID_ARG` naming what was wrong, instead of
as an internal failure with a stack trace behind it.

## Why the activation stays half

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

One scale per row rather than one per tensor, because a projection's output channels do not share
a magnitude: with a single scale, one outlier channel pushes every other channel down into E4M3's
subnormals. It is also the coarsest scaling the multiply can undo for free -- a scale constant
down a column of the result is a multiply in the epilogue, where a finer one would have to be
applied inside the mainloop.

## The epilogue

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

## Two tiles, picked by the row count

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

## Measurements

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
- **Weights are quantized at load rather than stored quantized**, so a package holds float16 and
  the memory saving only starts once the weight is on the device. Storing E4M3 in a `.waifupkg`
  would also halve the file and the load time.
- **The quantizer reads each row twice.** One pass would need either a second launch or a scale
  chosen before the data is seen. At 380 us for a 50 MB weight it is a load-time cost only.
- **Nothing is built out of it yet.** The diffusion model this crate runs is float16 throughout,
  and at its row counts -- 1024 and 4096 -- this path is 5% behind cuBLAS rather than ahead. What
  it is for is a weight read once per token, which is the language model shape.
- `n` must be a multiple of 8, which is how wide the epilogue writes, and `k` a multiple of 16,
  which is how wide the mainloop reads the weight. The row count is free. Unlike the half GEMM,
  which picks its alignment per call from the operands it is handed, this one is instantiated at
  one alignment and refuses anything narrower by name -- the weight comes from our own quantizer,
  which already insists on a `k` that 16 divides. `can_implement` is asked before every launch
  regardless, since CUTLASS's `initialize` returns success on strides the kernel cannot read.
