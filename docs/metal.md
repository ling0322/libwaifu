# Metal on flint's own kernels

`flint/metal_native` is the Metal device without MLX: flint's operators on flint's own kernels,
built the way the Vulkan backend is. `-DWITH_METAL=ON` builds it; it and `-DWITH_MLX=ON` are the
same device, so a build has one or the other.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWITH_OPENMP=ON -DWITH_METAL=ON
cmake --build build -j$(sysctl -n hw.ncpu)
```

It runs on every Apple GPU from the M1 on (family Apple7, for simdgroup matrices) and macOS 14.0.
Where the GPU has neural accelerators and the system is macOS 26.2 or later -- an M5 -- the
products and the convolutions run on those instead.

## Why

MLX evaluates a graph it is handed; flint hands it one operator at a time and wants the answer
written into a tensor it has already allocated. So every operator on the MLX backend was an
`eval`, a wait for the GPU, and a copy of the result into place, and the GPU sat idle for half of
an SDXL run while the host waited on it. Here an operator encodes its kernel and returns, and the
work goes to the GPU in batches without anyone waiting.

## How it is put together

- **metal-cpp** (`third_party/metal-cpp`, Apache-2.0, the macOS 14 release) is how the C++ talks
  to Metal. `context.cc` is the only file that does.
- **One queue, in order.** Commands go into a concurrent compute encoder with a memory barrier
  before each, so they run in the order they were encoded; batches of 64 are committed, at most
  three in flight, with a fence from one to the next. Buffers are untracked and command buffers do
  not retain them: the order is what keeps work from racing, as on Vulkan.
- **Stream-ordered memory.** A tensor's bytes go back to the allocator the moment it is freed and
  are handed to the next tensor at once, since anything the device does with them comes after
  whatever it did before. The host waits only when it reads or writes a buffer, and then only for
  the batch that used it last.
- **The allocator** (`block_allocator.cc`) cuts tensors from 512 MiB chunks, best fit, and merges
  free neighbours, so the bytes one stage of a model frees serve the next whatever their sizes. A
  tensor larger than a chunk has one of its own; before one is made, the context synchronizes and
  hands every empty chunk back. `release_memory()` hands back every empty chunk, and the engine
  calls it after each image.
- **The kernels** (`kernels/*.metal`) are compiled when flint is built, for Metal 3.1 and macOS
  14.0, without fast math, into one library carried inside flint as bytes. Those in
  `kernels/nax/` are Metal 4, for macOS 26.2, in a second library that the context loads only where
  it can run; a toolchain with no Metal 4 builds an empty one.
- **Products**: a simdgroup-matrix GEMM, 64 x 64 tiles of four simdgroups, everywhere; `matmul2d`
  on the neural accelerators where there are any. Convolution is an implicit GEMM on either --
  `C (K, P*Q) = W (K, C*R*S) unfold(x)` -- with the unfolded input gathered a chunk at a time into
  threadgroup memory, the next chunk read into registers while the accelerators work on this one.
- **Attention**: a fused kernel (online softmax, nothing written but the output) for up to 256 keys
  -- a prompt's 77 tokens, attended to from every pixel. Past that, the composition (matmul,
  softmax, matmul), whose products are the GEMMs above, is the faster.

## Measurements

Apple M5 Pro (16 GPU cores, neural accelerators), macOS 26, 2026-10-10. SDXL base at 1024 x 1024,
guidance 5, three images in a row; the second and third are the ones quoted. "MLX" is `main` as of
`171687c`, built with `-DWITH_MLX=ON`: the MLX backend is to go, and these are its last numbers,
kept here as what this backend replaced.

### Whole images

| | MLX | Metal, simdgroup GEMMs only | + accelerators for matmul | **+ for convolution, + fused attention** |
|---|---|---|---|---|
| 4 steps | 16.4 s | 29.4 s | 23.3 s | **9.2 s** |
| 20 steps | 52.2 to 56.1 s | 80.5 s | 53.6 s | **31.0 s** |
| one U-Net step | 2.4 s | 3.2 s | 1.9 s | **1.36 s** |
| the rest: text encoders, VAE decoder | 6.7 s | 16.6 s | 15.7 s | **3.8 s** |

A step is the difference between the 20- and 4-step runs over 16; the rest is what is left of the
4-step run.

Memory, as the allocator counts it: the model holds 6.8 GB once read, a 1024 x 1024 image needs
9.8 GB of tensors at its peak, in the VAE decoder, and the process holds 10.2 GB from the device at
that point and 7.2 GB once the image is done.

### Operator by operator

Each node of the graph timed on its own -- its operands waited for before the clock starts and its
result after -- over a whole 4-step image, summed by operator. The waiting adds 2.5 s on Metal.

| operator | MLX | Metal |
|---|---|---|
| conv2d | 4030 ms | 4765 ms |
| matmul | 5209 ms | 1926 ms |
| attention | 628 ms | 1480 ms |
| contiguous | 1822 ms | 1332 ms |
| add | 1741 ms | 1133 ms |
| group_norm | 1152 ms | 263 ms |
| silu | 604 ms | 197 ms |
| geglu | 588 ms | 132 ms |
| layer_norm | 437 ms | 220 ms |
| upsample_nearest2d | 221 ms | 43 ms |
| **every node** | **16.6 s** | **11.7 s** |

Where MLX is still ahead:

| | MLX | Metal |
|---|---|---|
| VAE conv, float32, [1, 512, 256, 256] x 512, 3 x 3 | 21 ms | 98 ms |
| U-Net conv, half, [2, 320, 128, 128] x 320, 3 x 3 | 3.1 ms | 6.5 ms |
| self-attention, half, [2, 10, 4096, 64] | 4.0 ms | 15 ms (the composition) |

The convolution is held back by the gather: with the input's reads taken out, the same kernel
runs at 11 to 13 TFLOP/s rather than 8. Self-attention needs a fused kernel on the accelerators.

What was tried for the convolution and left out, on this machine: tiles of 128 x 128 (slower than
64 x 128), chunks of 64 rather than 32 along K (the same), and two threadgroup tiles in turn to
save a barrier a chunk (slower: the threadgroup memory it costs).

`docs/benchmarks/sdxl_metal_native_applem5pro.txt` is the `sdxl_metal` benchmark on this backend,
beside `sdxl_metal_applem5pro.txt`, the same on MLX.

### Accuracy

`waifu/tests/sdxl*.rs` with the device set to Metal. Relative RMSE against torch in float32; the
CUDA column is from the same tests on an RTX 5060 Ti.

| | MLX | Metal, simdgroup only | **Metal** | CUDA |
|---|---|---|---|---|
| four denoising steps | 0.0227 | 0.0231 | **0.0248** | 0.0210 |
| text encoder 1, hidden | 7.6e-4 | 7.3e-4 | 7.3e-4 | |
| text encoder 2, hidden | 4.5e-3 | 2.7e-3 | 2.6e-3 | |
| text encoder 2, pooled | 1.8e-3 | 1.1e-3 | 1.1e-3 | |
| one U-Net step | 4.7e-4 | 5.5e-4 | 4.8e-4 | |
| VAE decode | 2.9e-3 | 4.6e-4 | 4.6e-4 | |
| VAE encode | 2.12e-2 | 3.2e-3 | 3.2e-3 | |
| an image through the autoencoder and back | 2.34e-2 | 6.3e-4 | 6.3e-4 | |

48 of the 49 pass; the one that does not is `denoising_on_the_cpu_matches_the_reference_pipeline`,
which assumes a CPU that computes in float32, as x86 does and an Apple CPU does not.

**The four denoising steps** moved from 0.0231 to 0.0248 when the new kernels went in, against a
bar of 0.0301. Each of the three, turned off on its own, moved it further, not back:

| | four denoising steps |
|---|---|
| everything on (an M5) | 0.0248 |
| no accelerators, fused attention on (an M1 to M4) | 0.0284 |
| everything off | 0.0231 |
| only the accelerators' matmul off | 0.0328 |
| only the accelerators' convolution off | 0.0336 |
| only fused attention off | 0.0268 |

Which is the test's own account of itself (`waifu/tests/sdxl.rs`): four steps at guidance five
amplify a disturbance of a single bit of half precision to a hundredth or so, in a direction that
is a coin flip. The kernels themselves, against float32 on the same half inputs:

| kernel, half unless said | old kernel | new kernel | new against old |
|---|---|---|---|
| matmul 2048 x 5120 x 1280 | 2.070e-4 | 2.070e-4 | 1.9e-5, 0.32 % of outputs differ |
| conv 3 x 3, [2, 320, 64, 64] | 2.074e-4 | 2.074e-4 | 2.4e-5, 0.48 % differ |
| conv 1 x 1, [2, 640, 64, 64] | 2.080e-4 | 2.080e-4 | 1.7e-5, 0.26 % differ |
| cross-attention, [2, 20, 1024, 64] over 77 keys | 7.08e-4 | 2.47e-4 | 7.5e-4 |

The products and convolutions on the accelerators are as accurate as the ones they replace -- the
sum is in float32 there too, and what is left is the one rounding to half -- and differ only in
the last bit of the few outputs whose order of summation tipped them. The fused attention is three
times nearer float32 than the composition, which rounds the scores and the probabilities to half
on the way; it is also the largest change, and so most of what moved the four steps.
