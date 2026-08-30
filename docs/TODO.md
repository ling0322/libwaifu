# TODO

Known issues and deferred work. Each entry says what was observed, not what it might be, so
whoever picks it up starts from evidence rather than from a guess.

## ~~`tensor_functional` failed once and did not reproduce~~ (fixed 2026-08-27)

Seen once during `cargo test` on the branch that folded `flint-rs` into `waifu` (2026-08-25), and
again on 2026-08-27, which is when the name was finally captured:

```
test draws_reproducible_random_numbers ... FAILED
```

It never reproduced on its own -- ten runs of `--test tensor_functional` in a row all passed --
because it needs two tests running at once.

The cause: a device has one generator, and `CPUOperators::sample` draws from it at
`cpu_operators.cc:194` with no short circuit for a greedy or top-k-of-one sample. The two sampling
tests in that file therefore draw from the same generator as `draws_reproducible_random_numbers`,
and cargo runs the tests in a file on several threads. A draw landing between that test's second
`manual_seed` and its second `rand` makes the two halves disagree.

The test had a comment saying it must stay the only one in the file that draws, which was true and
was quietly broken by the sampling tests: `grep F::rand` does not find them.

Fixed by giving the three tests a mutex to share. The library is documented as single threaded per
device, so the tests were the ones out of contract, not the generator.


## The mma path's recurrent branch has a reproducible hump at 13 to 15 tokens

Sweeping the sequence length at 64 sequences, 48 value heads and D=128 on an RTX 5060 Ti, the
in-kernel recurrent path costs, in microseconds:

```
len   11     12     13     14     15     16
    1106   1115   1172   1205   1175   1145
```

which is not monotonic: 14 tokens costs 5% more than 16 does. It reproduced identically across
four runs, so it is not noise, and the chunk path measured over the same lengths does not have it
(1130 at 12, 1140 at 14, 1146 at 16 -- flat and rising slowly, as it should be).

Ruled out: loop unrolling of the per-token loop. Building the same sweep with `#pragma unroll 1`
on it moved nothing (1193 at 14 against 1205), so it is not the compiler choosing a different
schedule at those trip counts.

Not yet looked at: whether it survives at other head dimensions or sequence counts, and whether it
is visible in the memory counters. The case is bandwidth bound at 64 sequences -- the state alone
is 196MB read and written -- so a 5% hump is somewhere in the memory system rather than in the
arithmetic, which is what makes it worth a profile rather than a re-read of the kernel.

`kDefaultRecurrentLen` is 12 partly because of this: the band is on the other side of it. Raising
it is worth ~8% at small batch, so this is what stands in the way.

## The mma kernel costs ten more registers at a head dimension of 32

Adding the recurrent branch took the D=32 instantiation from 128 registers to 138, with no spill
either way. 128 by 256 threads is exactly half a Blackwell SM's register file, so the baseline fit
two CTAs per SM there and this one fits one. D=64 went the other way (162 to 156) and D=128, which
is the shape the model runs, did not move at all.

`__launch_bounds__(kThreads, 2)` on the D=32 instantiation does bring it back to 128, but it buys
that with 36 bytes of spill stores and 92 of spill loads, which is the worse trade -- so it is not
applied. Nothing measures D=32: the benchmarks are all at the Qwen3.5 head dimension, and it is
reached only by the tests. Whether the lost CTA costs anything there is unmeasured.

## ~~Operators the CPU backend still does not implement~~ (the ones a model needs, 2026-08-30)

SDXL runs on the CPU now, end to end: the two text encoders, the U-Net, the sampler and the
autoencoder. `layerNorm`, `groupNorm`, `upsampleNearest2d` and `conv2d` were what stood in the
way, and each is checked against its definition written out in the test rather than against the
CUDA kernel.

Four operators are still declared on `Operators` and still abort on the CPU -- `rotaryEmbedding`,
`pagedAttention`, `storeKVCache` and `matmulNarrowPrecision`. All four are the language model's,
and the language model was deleted; nothing reachable calls them on either device. They are dead
declarations rather than missing implementations, and the thing to do with them is probably to
take them out.

What the CPU costs, on a 32 thread machine: 512 by 512 at 20 steps is 2m30s, 256 by 256 is 4.0 s
a step, and the model takes what it does on disk -- 6.96 GB.

It took 13.74 GB, and 19.46 during the load, until the weights were left at the precision the
file holds them in. x64 has no half arithmetic, so they used to be widened as they were read,
which is the doubling; and `VarBuilder` holds the file's copy while the model builds its own, so
both were live at once, which is the rest. A weight is now handed over as it was stored, and only
two operators learned to take a mixed pair: the matrix multiply, where `gemmHalfWeightFloat` was
already written and only needed reaching, and the convolution, which uses the same kernel. Those
are the two where widening would cost something, because there the narrow operand is the weight.
Everywhere else the layer converts -- an embedding widens the rows that came out of the table
rather than the table -- so the operators still take one type at a time.

It bought no speed -- 151.8 s against 150 -- which says this workload is bound by arithmetic
rather than by memory at 32 threads, and it cost no accuracy, since the arithmetic was float32
either way and the widened copy was only ever a copy.

## The CUDA tests cannot be tightened, and the CPU ones can

Measured on the reference denoising case, four steps at 256 by 256, relative RMSE:

```
                        vs torch fp32    vs torch fp16
  torch fp16               2.87e-2          0
  ours, CUDA (fp16)        2.10e-2          2.80e-2
  ours, CPU (float32)      1.28e-4          2.87e-2
```

The first column says the 2.1e-2 the CUDA test lives with is the price of half precision and not
a fault in the implementation: torch's own fp16 is further from its fp32 than we are.

The second column is the one that settles how to test. Our fp16 is as far from torch's fp16 as
torch's fp16 is from float32 -- at that precision the kernels' own differences, the accumulation
order and the attention blocking, are already the size of the rounding. So generating a second
reference in fp16 would buy nothing: its threshold would have to be 3e-2 as well. It would also
not be reproducible, since it depends on which kernels torch picked on the machine that made it.

So the CUDA test can only ever be a loose bound, and the right one to hold it to is torch's own
fp16 gap rather than a number chosen by hand. The sensitive test is the CPU one: float32 against
a float32 reference lands at 1.28e-4, which leaves room for a threshold two orders tighter, and
the model code it exercises -- the U-Net, the sampler, the assembly -- is the same code either
device runs. Only a CUDA kernel's own mistake could hide from it.

Both are written that way now. The CUDA one is held to torch's own half precision gap rather than
to a number chosen by hand, and there is a CPU denoising test beside it at 1e-3.

What the difference is worth, measured by putting a fault in and seeing which one notices. The
sampler's step was multiplied by a constant slightly over one, which is the shape of mistake this
is meant to catch -- small enough to leave the picture looking fine:

```
  step error      CUDA (bar 2.87e-2)      CPU (bar 1e-3)
  none            2.10e-2   passes        9.1e-5   passes
  0.02%           2.12e-2   passes        6.8e-4   passes
  0.10%           2.24e-2   passes        3.4e-3   FAILS
  0.50%           2.96e-2   FAILS         1.65e-2  FAILS
```

So the CPU test catches a fault five times smaller, which is the ratio the baselines already
implied: the CUDA one starts 2.1e-2 from its reference and a fault has to grow past that before
it shows, while the CPU one starts at 9.1e-5 and has nothing to hide behind.

## Operators worth adding next

The element-wise set, `div(a, b)` and `min` are in. Still missing from `functional.h`, roughly in
order of how often an inference engine wants them:

- `argmax`. Greedy decoding goes through `sample` with a temperature of 0 today, which works but
  makes the caller build three parameter tensors to ask for the largest logit. Unlike the
  reductions already there it returns indices rather than values, so it needs its own kernel that
  carries an index through the block reduction rather than an extra `MapReduceType`.
- `layerNorm`. `rmsNorm` covers Llama-style models; a model family that normalises with a mean and
  a variance cannot run at all.
- comparisons other than `eq` (`gt`, `lt`, `ge`, `le`, `ne`). `eq` is also unusual in taking only
  `<uint8>` and answering in `<bool>`, which is narrower than it looks.
- `mean`, `clamp`, `pow`, `topk`, `where`/`masked_fill`.

## `eq` only compares uint8 tensors

`Operators::eq` is implemented for `<uint8>` on both backends and nothing else, which is narrow
enough to be surprising given the name — comparing two `<float>` tensors is the obvious use and it
aborts. Widening it means picking a rounding policy for float comparison, which is why it was left
as-is rather than extended along with the other element-wise work.

## Gated DeltaNet prefill is within 10% of FlashInfer

`F::gatedDeltaNetPrefill` has five CUDA implementations. `kAuto` picks `gdnmma`
(`flint/cuda/gated_delta_net_mma.cu`) wherever it fits, which is head dimensions of 32, 64 and 128
on Ampere or later. Behind it are `gdnwmma`, the same algorithm on the WMMA API for the head
dimensions the first cannot take, and three FP32 SIMT paths kept for what they measured.

Measured on an RTX 5060 Ti (36 SMs, ~24 TFLOP/s FP32, ~88 TFLOP/s FP16 tensor, ~448 GB/s) with
`./build/benchmark "[gated_delta_net]"` and `tools/bench_gdn_flashinfer.py`, one Qwen3.5 gated
DeltaNet layer over 4096 tokens (16 key heads, 48 value heads, D=128), in microseconds:

```
  sequences        1      2      4      8
  chunked       7258   6868   6530   6124
  fused         8838   6650   6649   6100
  fused-regs    8530   7786   6444   6148
  tensorcore    1751   1336   1332   1245     (WMMA)
  mma           1301   1007   1031    936
  flashinfer    1094    851    897    876
```

So 6.5x faster than the best FP32 path and within 7% of FlashInfer's own SM120 kernel at eight
sequences, 19% at one. At 14 GFMA per layer that is ~30 TFLOP/s against FlashInfer's ~32, on a card
whose FP32 peak is 24 -- which is the whole point, since none of the FP32 paths could have got here
by any amount of tuning.

### The two things that closed the gap

Both are about where a product's result lives, not about the arithmetic, which has been the same
since the first tensor core version.

- **`mma.sync` instead of WMMA, with the state transposed.** WMMA will not say where an accumulator
  element is, so anything a product computes has to be written to shared memory before another
  product can read it as an operand: the state once a chunk, the right hand side once, u once. With
  `mma.sync` two adjacent m16n8 accumulators, cast to half, *are* the A operand of the 16 by 16 tile
  they cover -- so each product feeds the next out of the registers it landed in. Holding the state
  as its transpose (value dimension by key dimension, which is also how FlashInfer holds it) is what
  puts every intermediate on the A side of the products that consume it. It also puts the token on
  the *column* of every accumulator, which turns the two decays the WMMA path applies as in-place
  passes over shared memory into a multiply on a register.
- **The head dimension as a template parameter.** This is worth as much as everything above it: the
  same kernel with a runtime head dimension is **1681 us**, with 256 bytes a thread of stack frame,
  because every accumulator array is indexed by a loop bound derived from it and ptxas will not keep
  a dynamically indexed array in registers. Instantiated per head dimension it is 956 us with no
  stack at all. A register-resident design is only register-resident if the compiler can prove the
  indices.

Prefetching the next chunk's K, Q and v -- a second key buffer, and copies issued as each of the
other two falls out of use -- took the last 2%.

### What it cost to get there, and what to check first if it breaks

The layout facts below are not documented guesses; each was verified against a reference GEMM on
this device (`ldmatrix` and `mma.sync` are ABI, but which register holds what is easy to get wrong
and silent when you do -- the shapes are identical and only the answer changes):

- `ldmatrix.x4` on a row-major tile, with lane l addressing row l % 16 and column half l / 16,
  gives the four registers as (rows 0-7, k 0-7), (rows 8-15, k 0-7), (rows 0-7, k 8-15),
  (rows 8-15, k 8-15). As an A operand that is exactly right in register order.
- As **B operands** it is not: the operand for the first eight columns is registers 0 and 2, and for
  the second eight registers 1 and 3. Pairing them 0-1 and 2-3 is the mistake that cost the longest
  debugging round here, and it is why `mmaB` exists and is the only place the pairing is written.
- With `.trans` it is the other way round -- the transpose makes the halves of the address pattern
  index n rather than k -- so there the operands *are* consecutive. `mmaBTrans` is the only place
  that one is written.
- A 16 by 16 accumulator pair cast to half is the A operand of that tile, in register order.

The debugging that found the pairing bug is worth repeating if this ever breaks: dump the staged
tensors and compare against the input (they were exact), compute one output tile with a scalar loop
out of the same shared memory (exact), then compare the mma result against that (wrong) -- which
localises it to the instruction's operands in three steps rather than by reading PTX.

### Measurements from the WMMA path, which still apply

The WMMA path is 1245 us and the same shape of kernel, so its ablations are the best picture of
where the time goes: the HMMAs themselves ~600 us, the barriers ~181, the state's trip to shared
memory ~59, the key and query loads ~53, the two in-place decay passes ~48, the output store ~33.
On the mma path the same ablations give: global loads 93 us, the output store 46, the inverse 21.

The kernel is latency bound, not issue bound: at eight warps it issued about a sixth of the
instruction slots it had, because one CTA is resident per SM at this footprint and the block's own
warps are all there is to hide anything behind. That is why the WMMA path runs sixteen warps -- the
largest count its tiles divide into evenly -- and why the mma path, which is pinned to eight by the
row partitioning, needed the register residency instead.

What paid on the WMMA path, in order: vectorising the epilogues (3428 to 1550 us), sixteen warps
(1306 to 1244), fusing A and the score matrix into one pass, skipping the six tiles above the
diagonal, probing the accumulator layout at launch rather than round-tripping through shared memory
(1377 to 1335), the inversion moving to four warps behind a named barrier (1336 to 1306), staging v
coalesced, `cp.async` for the keys and queries, and a barrier-free decay scan.

What did not pay anywhere: twelve warps (5% slower than eight -- 64 state tiles over 12 is six for
some and five for others); giving each warp consecutive tiles to share an operand (25% fewer
fragment loads, 9% slower, because the triangular phases then hand one warp the cheapest tiles and
another the dearest); hoisting the operand the fixed column block indexes out of the tile loop (also
25% fewer loads, 12% slower, since it takes the loads out of the shadow of each other's HMMAs);
balancing the triangular phases (no change, which is what says the barrier cost is exposed load
latency rather than warps waiting on each other); staging the mma path's output through shared
memory to coalesce it; and padding the chunk-square matrices by a whole tile.

One correctness note, since it is the kind of thing tile handouts invite: a warp's tiles all sit in
one column block only when the tile width divides the warp count, which is true at D of 32, 64 and
128 and false at 48, 80 and 112. Hoisting that column out of the tile loop as a per-warp constant is
wrong at those head dimensions, silently. The test that covered it was at 48 and went when the WMMA
path did -- the operator no longer takes a head dimension that is not 32, 64 or 128 -- so anyone
instantiating the mma path for one of them has that trap waiting and no test standing on it.

### What is left

- **The last 7%, and the 19% at one sequence.** Both kernels lose the same way at one sequence: 48
  CTAs against 36 SMs is 1.33 waves, a third of the machine idle in the second one. FlashInfer is
  1094 against 876 for the same reason. Splitting a long sequence into segments and passing the
  state between them, which is what the chunked path does across launches, would fill it.
- **Warp specialisation.** FlashInfer runs 384 threads as three warp groups, one of them doing
  nothing but loads. The mma path is pinned to eight math warps by the row partitioning, so adding a
  load group is the natural next step and there is shared memory for it.
- **More head dimensions.** The mma path is instantiated for 32, 64 and 128 and nothing else is
  left to catch the rest -- `gatedDeltaNetPrefill` refuses them now rather than routing them
  somewhere slower. Instantiating more costs compile time in proportion; read the correctness note
  above first, since 48, 80 and 112 are exactly the ones the tile handout is a trap at.

### The three FP32 paths, and what they measured

All three are deleted, along with the WMMA path and `cuda/triangular_solve.cu`, which nothing but
`kChunked` used. This is what they established, before any of them was the answer -- kept because
it is the argument for the shape the surviving kernel has, not because the code is coming back:

- `kChunked` is three launches -- build every chunk's system, solve the batch of them through
  `triangularSolveInplace`, scan each sequence's chunks in order. It moves several hundred megabytes
  of intermediates for a long prefill but has blocks to spare however short the batch is. Its
  scratch is proportional to the token count -- 4096 tokens at these head counts is about 150 MB --
  so a long prefill through it has to arrive in batches. Neither tensor core path has any scratch.
- `kFused` gives one CTA a (sequence, value head) and keeps the state in shared memory. It removes
  ~80% of that traffic and 22% of the multiply-adds, and it is not faster: the (D, D) state is 64 KB
  of the 99 KB a block may have, so one CTA is resident per SM, and the chunk had to shrink to 32.
- `kFusedRegisters` is the same kernel with the state in registers, so two CTAs are resident. It
  pays for that with a warp shuffle reduction and 56 bytes a thread of spill, and it is a wash.

Three arrangements of the same FP32 arithmetic landing within 15% of each other is what argued for
changing the arithmetic rather than arranging it again. Two other results from that period: padding
the staged tiles to an odd row stride took `buildChunkKernel` from 2950 to 1642 us, and carrying the
right hand sides in half rather than float took the solve from 1190 to 835.

### Lining up with FlashInfer, if you compare again

`tools/bench_gdn_flashinfer.py` times `flashinfer.chunk_gated_delta_rule` on exactly the shapes the
benchmark above uses, with the same 5 warmups and 20 timed iterations between CUDA events, and
`--check` first verifies it computes the same recurrence flint's CPU operator is tested against. It
does, to about 1e-4, once two conventions are lined up:

- FlashInfer's `g` is the decay, in (0, 1]; flint's is its log, at most zero. Passing a log decay
  gives all-NaN output, since the kernel takes a log of it.
- FlashInfer stores a head's state as (value dim, key dim), flint as (key dim, value dim). The state
  is square, so the shapes agree either way and getting it wrong is silent. The mma path holds the
  transpose in registers for exactly the reason FlashInfer does, and transposes it on the way to and
  from the state tensor, which keeps flint's own layout unchanged.

FlashInfer has the same state-pool indirection this operator has -- `state_indices` alongside a
pool-shaped `initial_state` -- but only on its SM100/SM103 kernel: on sm120 it raises
`NotImplementedError`. The comparison above is against its packed, sequence-ordered path.

## `CudaOperators::zeros` ignores the dtype it is asked for

`flint/cuda/cuda_operators.cc` builds the tensor with `createCudaTensorHalf` whatever `dtype`
says, so `zeros(shape, DType::kFloat)` hands back a `<half>` and the next operator to look at it
aborts on a dtype check. Found while writing the gated DeltaNet benchmark, which now builds its
FP32 state on the host and copies it over instead. `op::cuda::fill` is half-only, which is
presumably why it was written this way, so fixing it means giving `fill` the other types first.
## The CLIP tokenizer does not normalize text the way ftfy does

`CLIPTokenizer` runs its input through `ftfy.fix_text` before matching its pattern, and libwaifu
does not. Diffed over 1600 texts (2026-08-27): all 1271 ASCII ones agree exactly, and all 137 that
disagree are exactly the 137 that ftfy rewrites -- ligatures such as `ﬁ` expanded to `fi`,
fullwidth `ｆｕｌｌ` folded to `full`, `ǅ` put into NFC. None is a disagreement about how to merge,
which is what the diff was written to find out.

This only shows up on text that is not already normalized, which a danbooru style prompt never is
not. Closing most of the gap would take an NFC pass; closing all of it would take ftfy, which is a
mojibake repair library and not something to reimplement.

Worth knowing: `CLIPTokenizer` itself falls back to a `BasicTokenizer` when ftfy is absent, and
that one puts spaces between CJK characters, so huggingface's own output for `霧雨魔理沙` depends
on what is installed beside it. The reference above was generated with ftfy present, which is what
CLIP itself does.

## ~~The SDXL VAE decoder overflows in half precision~~ (fixed 2026-08-28)

Two entries stood here, one about latents an encoder would never produce and one about the
latents the sampler really hands over. Both were the same thing and both are closed by running
the decoder in float32, which is what it is now built in.

What was measured, on the reference's own four step latent: the activations grow through the
decoder -- about 84 at the mid block, then 570, then 4046 -- and one convolution of the last up
block passes 65504, which is as far as half goes. Everything after it was a NaN. It is a range
problem and not a precision one, which is why accumulating in float did not help: the value
overflowed as it was stored. `rand`, which is uniform over `[0, 1)` and so has a large DC offset
once the scaling factor is divided out, reached infinity the same way and for the same reason.

The autoencoder's own config says all of this: `force_upcast` is true, and diffusers reads that
flag and runs the decoder in float32 whatever the rest of the pipeline is in.
`madebyollin/sdxl-vae-fp16-fix` exists because the alternative is retrained weights.

Closing it took the float arm of five CUDA operators that were written against `half`:
`groupNorm` and the two norms it shares its kernels with (norm.cu), `softmax` (softmax.cu),
`matmul` by way of `sgemm` and `sgemmArray` on the cuBLAS backend (matmul.cc, gemm_cublas.cc),
`copy` (copy.cu), and `upsampleNearest2d` (upsample.cu). `conv2d` already took float32 and the
unary and binary operators already dispatched on it, so `attention` -- which is matmul, softmax
and the copies behind `cat` -- followed once those were in. `CudaOperators::tensor` and
`tensorLike` had to learn the type as well, since `cat` and `contiguous` allocate through them.

Only the autoencoder is in float32; the U-Net and both text encoders are still half. `VarBuilder`
grew `with_float_type` for that, and the exporter already wrote the VAE's weights unnarrowed.
The decode drifts 1.0e-3 from the reference image, against 2.1e-2 for the half decoder that
worked only on noise.

bfloat16 would have closed it too and more cheaply at run time, but flint has no bfloat16 at all,
which is the larger change of the two and is still worth having for its own sake.

## ~~CUTLASS is 12% behind cuBLAS on SDXL's GEMM mix~~ (closed 2026-08-29)

It was, and it is not any more: the two now measure the same on a whole image, 12.985 s against
13.050 s at 1024 by 1024 over thirty steps, which is inside a spread of 12.878 to 13.278 across
runs. What closed it was split K, decided per call from the CTA count -- `splitKSlices` in
gemm_cutlass.cu, and the comment there has the table of what it picks for each shape SDXL runs.
A third of a step's GEMM time splits and two thirds does not, and the third that splits is
exactly what was behind: the two 1024 by 1280 shapes by 25% and 24%, and the two with 77 rows by
31% and 55%.

Two things worth keeping from it.

The first is that the accumulator was `half_t`, which was a bug rather than a tuning choice and
is fixed in its own commit. It cost enough to fail the tests SDXL stands on.

The second is a warning about `./build/benchmark "[sdxl]"`, which is the benchmark this entry was
written from. It runs one shape fifty times over, so a weight stays in L2 between iterations,
which a real run never gets. It therefore prefers whatever moves the least data per
multiply-add: it says a 64 by 64 tile is the fastest thing here, and on a whole image that tile
is about a percent slower than the 128 by 128 one. Both numbers are in the comment on `Sm80Gemm`.
Tuning a tile against that benchmark alone will pick the wrong one.

What is left, and it is not much: the remaining shapes where CUTLASS is behind cuBLAS are the two
with 77 rows, and a 64 by 64 tile for those alone -- dispatched on M, kept off everything else --
measured 26.5 and 23.4 TFLOP/s against cuBLAS at 23.0 and 18.0. They are 2.5% of a step's GEMM
time, so winning all of it is worth about 0.3% of an image. It has not been written.

The backend is now complete rather than half of one: CUTLASS answers float as well as half, so
`LIBWAIFU_GEMM=cutlass` runs the whole model including the autoencoder and cuBLAS is no longer
reached at all. It stays the default, since `MatMul::create` still tries it first.

## The CUTLASS convolution does not do groups, and costs 3% for the layout

`WITH_CUTLASS=ON` now answers `Conv2d`, so cuDNN is preferred rather than required, and
`LIBWAIFU_CONV=cutlass` runs the whole of SDXL through it. Two things are left in it.

The first is grouped convolution, which it refuses. CUTLASS can do them -- `GroupMode` on the
fprop kernel -- but it is another instantiation and nothing in this repository convolves in
groups: SDXL's 138 convolutions are all one group, one dilation, a 1x1 or a 3x3, and one of three
stride and padding pairs. There is a test standing on the refusal rather than on the answer.

The second is the layout, and it is the 3%. A whole 1024 image is 12.990 s on cuDNN and 13.382 s
on CUTLASS. The kernel itself is not the problem -- measured shape by shape on what the U-Net
runs, the NHWC kernel beats cuDNN's NCHW one by 7% to 17%, because NHWC is what the tensor cores
want. What it pays back is the permute on either side, and the two convolutions at each end of
each model, where a four channel latent or a three channel image leaves a 128 wide tile mostly
empty and the analytic iterator is slow.

Where the 3% would go, in order:

- Keep the activations in NHWC between consecutive convolutions. Everything between them in a
  ResNet block -- the group norm, the SiLU, the residual add -- is elementwise or per channel and
  would read either layout, so a block could permute once at each end instead of twice per
  convolution. This is most of the 3% and it is also the change that reaches furthest.
- A narrower tile for the shapes at the ends. 320 to 4 measured 1.18 TFLOP/s against 40 for the
  aligned kernel, on a 128 by 128 tile that has one useful column of four.
- The permute itself is a tiled transpose reaching something over 900 GB/s where the library's
  generic strided copy manages 146, so it is not the obvious thing to improve next.

## `CUDA_ARCH_NATIVE=ON` does not build on an RTX 50 series card

The CUDA build README recommends fails on sm_120, in ptxas, sixteen times over:

```
cmake -S . -B build -DWITH_CUDA=ON -DCUDA_ARCH_NATIVE=ON -DWITH_CUTLASS=ON
...
ptxas .../dequant.ptx, line 348; error: Instruction 'cvt with .e2m1x2' not supported
                                        on .target 'sm_120'
```

Two things have to line up for it, and both are worth fixing separately.

**The architecture the native build asks for.** `CMAKE_CUDA_ARCHITECTURES native` resolves an
RTX 5090 to plain `120`. The default build does not go through that path and asks for `120a-real`,
with the comment `# 120a (not 120): the mxfp4 kernels use sm_120a-only PTX` next to it — so the
requirement is known, and only the native branch misses it. CMake leaves what it detected in
`CMAKE_CUDA_ARCHITECTURES_NATIVE` after `enable_language(CUDA)`, so rewriting `120` to `120a`
there is possible, but it means deciding the architecture after `enable_language` rather than
before it, which is not the order the file has now.

**The guard on the kernels.** `flint/cuda/dequant.cu` gates all four mxfp4 entry points on
`#if __CUDA_ARCH__ >= 1200`. That asks whether the compute capability is at least 12.0, but what
`cvtFp32x8ToFp4`'s inline PTX needs is whether the e2m1 conversion is available, and those two
answers differ on exactly one target:

| target | `__CUDA_ARCH__ >= 1200` | e2m1 available |
|---|---|---|
| sm_80, sm_90, sm_100 | no | no |
| sm_120a | yes | yes |
| **sm_120** | **yes** | **no** |

So the guard admits the one target the asm cannot be assembled for. `__CUDA_ARCH_FEAT_SM120_ALL`
is the predicate that matches — it is what `cuda_fp4.hpp` gates its own copy of this instruction
on. Compiling the same asm behind `#if defined(__CUDA_ARCH_FEAT_SM120_ALL)` (2026-08-29):

```
sm_80    compiles, 0 F2FP.E2M1 in the SASS
sm_90    compiles, 0
sm_100   compiles, 0
sm_120   compiles, 0
sm_120a  compiles, 4
```

which is the intent the `>= 1200` guard was written with.

Note that a software fallback is the wrong answer here even though CUDA ships one:
`__nv_cvt_float2_to_fp4x2` compiles on sm_120 by falling back through `__nv_cvt_double_to_fp4`,
and the kernel goes from 24 SASS instructions to 152, `F2F.F64` and `DADD` among them. sm_120 and
sm_120a are the same silicon, so that path would emulate in double on a card whose hardware does
the conversion in one instruction — it turns a loud build failure into a quiet slowdown.

Worked around while building on this machine by passing the architecture explicitly, which skips
the whole `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)` block:

```bash
cmake -S . -B build -DWITH_CUDA=ON -DWITH_CUTLASS=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real
```

That builds and runs, but produces cubins for sm_120a only.

### Unverified, found while reading the same file

`quantHalfToMxfp4` and `dequandMxfp4ToHalf` are host functions, and both are wrapped in
`#if __CUDA_ARCH__ >= 1200` with `NOT_IMPL()` in the `#else`. `__CUDA_ARCH__` is not defined in
the host pass, so on the face of it the condition is always false and both entry points always
abort. Nothing was run to confirm this.
