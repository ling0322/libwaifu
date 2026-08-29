# NVFP4 projections on the SM120 block scaled tensor cores

`op::cuda::gemmNvfp4` multiplies a half activation by an NVFP4 weight and returns half. The
activation is quantized inside the call; CUTLASS supplies the mainloop; the prologue and the
epilogue around it are ours.

```cpp
Nvfp4Operand w = quantizeNvfp4(weightFp16);   // once, at load
Tensor y = gemmNvfp4(xFp16, w);               // per layer: half in, half out
```

Requires `WITH_CUTLASS=ON`, CUDA 12.8 or newer, and an sm_120a target. `isNvfp4GemmAvailable()`
reports whether this build and this GPU can run it.

## From Rust

`flint::Nvfp4Tensor` holds the three pieces a quantized operand is made of, and
`functional::nvfp4_matmul` multiplies by one:

```rust
let weight = Nvfp4Tensor::quantize(&float16_weight)?;
let y = F::nvfp4_matmul(&x, &weight)?;   // float16 in, float16 out
```

`Nvfp4Tensor::is_available()` answers whether this build and this GPU can run it. `k` -- the
dimension the two share -- has to be a multiple of 32 and the weight's row count a multiple of 8,
which the quantizer checks rather than leaving to the kernel.

The C interface is `fl_nvfp4_available`, `fl_nvfp4_quantize`, `fl_nvfp4_dequantize` and
`fl_nvfp4_matmul`.

The kernels assert their preconditions with `CHECK`, which aborts, so the C interface checks
device, type, contiguity and shape itself first: a host side weight comes back as an error rather
than as a dead process.

## Why the activation has to be quantized

The block scaled instruction takes no other kind of operand. Disassembling the two kernels says it
plainly:

```
$ cuobjdump -sass gemm_nvfp4_cutlass.cu.o
     64   OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X
```

`E2M1.E2M1` -- both operands are FP4, `UE4M3` is the block scale type, and k is 64, four times an
HMMA's depth. A kernel that keeps the activation in half compiles to `HMMA.16816.F32` instead and
is then bounded by the half tensor cores, which is the same ceiling cuBLAS FP16 already reaches.
So FP4 as a storage format saves bandwidth; only FP4 as an operand format buys arithmetic.

## The prologue

Two kernels, no memset, because on a decode step every launch is visible.

1. `amaxHalfKernel` -- the tensor wide maximum, one partial per block. The partials are left for
   the second kernel to finish rather than reduced atomically, which is what avoids having to zero
   an accumulator first, and so avoids a third launch.
2. `quantizeNvfp4Kernel` -- one thread owns one 16 element scale block, so the block maximum needs
   no cross lane reduction. It writes the E2M1 elements, the E4M3 block scale, and the padding
   bytes the atom layout carries beyond the operand's own extent, which folds the memset in.

The scale arithmetic is

```
globalScale = amax / (6 * 448)
blockScale  = blockAmax * 448 / amax     // == blockAmax / (6 * globalScale), in [0, 448]
```

and the elements are divided by the block scale **as E4M3 rounded it**, not as it was computed,
since the rounded value is all the mainloop will ever see.

### Scale factor layout

Block scales are not row major. CUTLASS's `SfKMajorAtom` is
`((32,4),(16,4)):((16,4),(0,1))`: a 512 byte tile covering 128 rows and 4 scale blocks, indexed
within the tile as

```
(row % 32) * 16 + ((row / 32) % 4) * 4 + (kBlock % 4)
```

The prologue does not reimplement this. It passes the `tile_atom_to_shape_SFA` layout object into
the kernel by value and indexes it as `layoutSF(row, kBlock * 16, 0)`, so the layout the prologue
writes and the layout the mainloop reads are the same object by construction.

The atom's extent is a multiple of 128 rows and 4 blocks, so a `(rows, k)` operand always carries
padding; the quantize kernel writes zeros there.

## The epilogue

`ElementC = void` and `ElementD = half`: D is `alpha * accumulator`, with no source read. `alpha`
is `globalScaleA * globalScaleB`, and both of those were produced on the device by the prologue,
so the product is formed there too and passed as `alpha_ptr`. Reading them back to pass a host
side alpha would serialize every layer against the host.

## Measurements

RTX 5060 Ti (sm_120a, 36 SMs, 448 GB/s), CUDA 12.9, Llama 3.2 3B projection shapes.

### Speed

`gemm` is the mainloop with both operands already quantized. `gemm+prologue` adds the activation
quantization, which is what a layer actually pays, since a weight is quantized once but an
activation is quantized every time.

| shape | FP16 (cuBLAS) | NVFP4 gemm | NVFP4 gemm+prologue |
| --- | --- | --- | --- |
| prefill-512 qkv_proj | 333 us / 48.3 TF | 57.4 us / 280 TF | 66.6 us / 242 TF |
| prefill-512 out_proj | 204 us / 47.3 TF | 35.7 us / 271 TF | 44.3 us / 218 TF |
| prefill-512 gate_up_proj | 1048 us / 49.2 TF | 174 us / 296 TF | 193 us / 267 TF |
| prefill-512 down_proj | 540 us / 47.7 TF | 84.7 us / 304 TF | 103 us / 249 TF |
| decode-1 qkv_proj | 56.5 us | 25.4 us | 28.1 us |
| decode-1 out_proj | 23.3 us | 13.7 us | 16.8 us |
| decode-1 gate_up_proj | 219 us | 46.8 us | 51.3 us |
| decode-1 lm_head | 1825 us | 532 us | 531 us |

Prefill runs 5 to 6 times faster. Decode gains far less because it is bound by reading the weight
rather than by arithmetic: decode-1 out_proj moves 5.3 MB in 13.7 us, which is 380 GB/s against a
448 GB/s part.

The prologue costs about 3 us on a decode step, which is two kernel launches, and 9 to 19 us at
prefill-512, or roughly 9% of the GEMM.

### Accuracy

Relative RMSE, `sqrt(sum((x - ref)^2) / sum(ref^2))`. The FP16 GEMM is measured against an FP32
GEMM on the CPU; the NVFP4 path against the FP16 GEMM, which is the thing it replaces.

| shape (m,n,k) | fp16/fp32 | nvfp4/fp16 |
| --- | --- | --- |
| qkv_proj (512,5120,3072) | 3.60e-04 | 1.34e-01 |
| out_proj (512,3072,3072) | 3.59e-04 | 1.34e-01 |
| gate_up_proj (512,16384,3072) | 3.59e-04 | 1.34e-01 |
| down_proj (512,3072,8192) | 3.60e-04 | 1.34e-01 |
| decode qkv_proj (1,5120,3072) | 3.60e-04 | 1.34e-01 |

The FP16 GEMM's own error is two and a half orders of magnitude below the quantization error, so
it is a clean reference. Quantizing a single operand to NVFP4 costs 9.5e-2, measured separately;
`sqrt(0.0951^2 + 0.0951^2) = 0.1345` against the 1.34e-1 above, so the two operands' errors add in
quadrature and nothing systematic is coming from the GEMM itself.

## Measured against cuBLASLt

cuBLASLt has the same kernel (`CUDA_R_4F_E2M1` with `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`)
and reads the block scales in the same interleaved layout, so the prologue's output feeds it
unchanged. It was wired up, measured, and then removed; the numbers are kept here.

| shape | CUTLASS | cuBLASLt |
| --- | --- | --- |
| prefill-512 gate_up_proj | 169 us / 304 TF | 182 us / 283 TF |
| prefill-512 qkv_proj | 57.2 us | 57.0 us |
| prefill-512 down_proj | 83.9 us | 83.9 us |
| prefill-128 down_proj | 29.6 us | 34.4 us |
| decode-1 qkv_proj | 25.7 us | 29.2 us |
| decode-1 down_proj | 29.8 us | 35.3 us |
| decode-1 gate_up_proj | 46.8 us | 45.7 us |
| decode-1 lm_head | 528 us | 528 us |

At parity, with CUTLASS 7 to 16% ahead on a few shapes and never meaningfully behind. Both sides
paid their per-call host setup inside the timed region, and cuBLASLt got to pick an algorithm per
shape by heuristic while the CUTLASS side is one fixed 128x128x128 tile.

Accuracy was identical to three significant figures, which is the expected result: the error is
the format's, not the implementation's.

Two things cost time getting cuBLASLt to run, recorded in case anyone tries again:

- `CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO` returns `NOT_SUPPORTED` for this matmul.
  `CUBLASLT_POINTER_MODE_DEVICE` works, but then beta has to be a device pointer as well.
- With beta on the device, a null C is rejected: cuBLASLt cannot rule out a read of C on the host.
  Passing D as C works, and the kernel does test beta at run time -- an output buffer filled with
  NaN beforehand comes back clean.

## Known gaps

- The prologue's scale writes are not coalesced. Consecutive threads walk consecutive scale
  blocks, and `kBlock % 4` occupies only the low two bits of the offset within a tile, so a warp
  writes eight four byte runs 512 bytes apart -- eight 32 byte sectors for 32 bytes of data.
  Scales are a sixteenth of the element volume, so the amplified traffic is worth roughly 10% of
  the prologue, or 1 to 2 us at prefill-512. Mapping thread `t` within a tile to byte `t`
  (`row = t / 16 + 32 * ((t % 16) / 4)`, `kBlock = t % 4`) would make the run contiguous.
- The mainloop is one fixed tile shape, `128x128x128`, with the default tile scheduler raster
  order. `sm120_rr_smem_selector` picks the shared memory swizzle from the K tile: at 128 with 4
  bit elements it lands on `Layout_K_SW64_Atom` (`Swizzle<2,4,3>`); a 256 deep K tile would reach
  the 128 byte one. Neither has been tuned.
- The output is half, so the next layer quantizes it again. Folding the quantization into the
  preceding operator's epilogue, so that an RMS norm emits NVFP4 directly, would remove a full
  activation round trip per layer.
- Weights are quantized at load rather than stored quantized, so a package holds float16 and the
  memory saving only starts once the weight is on the device. Storing NVFP4 in a `.waifupkg` would
  also cut the file size and the load time.
- Nothing is built out of it yet: the diffusion model this crate runs is float16 throughout, and
  the layer that wrapped this went with the language model runtime.
- `n` must be a multiple of 8, which is how wide the epilogue writes. The row count is free and
  `k` must be a multiple of 32.
