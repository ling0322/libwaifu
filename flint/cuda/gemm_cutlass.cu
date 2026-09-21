// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#define CUTLASS_DEBUG_TRACE_LEVEL 2

#include <cuda_fp16.h>

#include <algorithm>
#include <type_traits>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_array.h>

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "lutil/error.h"
#include "flint/cpu/common.h"
#include "flint/cpu/matmul.h"
#include "flint/cuda/common.h"
#include "flint/cuda/gemm_cutlass.h"
#include "flint/dtype.h"

#define CUTLASS_CHECK(x)                                                                     \
  {                                                                                          \
    cutlass::Status status = x;                                                              \
    if (status != cutlass::Status::kSuccess) {                                               \
      LOG(ERROR) << "Error while calling: " << #x << ": " << cutlassGetStatusString(status); \
      throw lut::AbortedError(cutlassGetStatusString(status));                               \
    }                                                                                        \
  }

namespace fl {
namespace op {
namespace cuda {

using namespace cute;

using cutlass::layout::ColumnMajor;
using cutlass::layout::RowMajor;

/// Half in, half out, accumulated in float, over a 128 by 128 tile that may split K.
///
/// The accumulator is the seventh argument and it was `half_t`, which is a mistake: a K of a few
/// thousand walks the running sum up to a magnitude where half's step is larger than the products
/// still being added to it. Nothing here disagreed -- the batched path below has always
/// accumulated in float, and cuBLAS runs these with CUBLAS_COMPUTE_32F -- so it was this one
/// instantiation on its own, and it was enough to fail the tests SDXL stands on.
///
/// SplitKSerial is on rather than a second instantiation. Asked for one slice it measures the
/// same as an instantiation that cannot split at all, to within the noise, so the capability
/// costs nothing where it is not used and there is one kernel here rather than two. What splits,
/// and by how much, is decided per call in `splitKSlices` below.
///
/// The tile stayed at 128 by 128, and a 64 by 64 one was tried. It makes four times the CTAs and
/// wins the GEMM benchmark in this repository -- 47.56 TFLOP/s a step against 47.40 -- and does
/// not win the model. Median of a whole 1024 image, thirty steps, in seconds:
///
///     cuBLAS                   12.985   (seven runs, 12.878 to 13.147)
///     128 by 128, split K      13.050   (three runs, 12.912 to 13.278)
///     64 by 64, split K        13.170   (three runs, 13.165 to 13.178)
///
/// The spread is as large as the differences, so the first two are the same speed as far as this
/// can tell and the third is about a percent behind. What is worth keeping is the direction: the
/// benchmark prefers the tile the model does not, so do not tune this against the benchmark
/// alone. Two reasons it is misled, and it hides both. The smaller tile halves the arithmetic
/// intensity -- 32 FLOP per element loaded against 64 -- which only shows once the weights are
/// cold rather than left in L2 by the previous iteration of the same shape. And it fills the
/// machine well enough on its own that the rule below almost never splits: at 64 by 64 a split
/// covers 2.5% of a step's GEMM time, at 128 by 128 it covers 35.4%. So the two tiles were never
/// really being compared -- one of them was being compared with split K and the other without.
///
/// # Alignment
///
/// `ALIGNMENT` is how many halves each global access reads at once, and it is a promise about
/// the operands rather than a preference: at eight the iterators issue 128-bit loads, and an
/// operand whose leading dimension is not a multiple of eight makes every row after the first
/// start off a 16-byte boundary. CUTLASS launches it anyway unless it is asked -- see the
/// `can_implement` in `hgemmT` -- and what comes back from the device is a misaligned access,
/// reported at whichever unrelated CUDA call next happens to ask for a status. Anima's patch
/// embedder is a K of 68, which is how this was found; `hgemm` below is what picks.
template<class LayoutA, class LayoutB, int ALIGNMENT>
struct Sm80Gemm {
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t,
      LayoutA,
      cutlass::half_t,
      LayoutB,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, ALIGNMENT, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      3,
      ALIGNMENT,
      ALIGNMENT,
      true>;
};

/// The alignments the tensor-core kernel has an instantiation for, widest first.
///
/// Three rather than one because a leading dimension is whatever the model made it: eight covers
/// every shape SDXL runs and all but one of Anima's, and four is what that one -- a K of 68 --
/// needs. Two is the floor and not a choice: this kernel stages through `cp_async`, which takes
/// 4, 8 or 16 bytes and nothing else, so one half at a time does not compile.
///
/// An odd leading dimension is therefore not on this ladder at all, and is copied into a buffer
/// that is on it before anything runs. See `hgemm` below.
constexpr int kAlignments[] = {8, 4, 2};

/// Whether `data` and `ld` can be read `alignment` halves at a time.
///
/// Both ends are asked. The leading dimension is where this goes wrong in practice -- a
/// device allocation is 256-byte aligned, so the base of a whole tensor always qualifies -- but a
/// GEMM may also be handed the middle of one, and a row that starts off the boundary reads as
/// misaligned whatever the stride says.
bool isAlignedTo(int alignment, const void *data, int ld) {
  size_t bytes = static_cast<size_t>(alignment) * sizeof(cutlass::half_t);
  return reinterpret_cast<uintptr_t>(data) % bytes == 0 && ld % alignment == 0;
}

constexpr int kTileM = 128;
constexpr int kTileN = 128;

/// Somewhere for a split to keep its semaphores, kept between calls rather than asked for on
/// every one.
///
/// It is a few hundred bytes -- one counter per tile -- so what it costs is the asking rather
/// than the memory: on the two shapes that split, keeping it took a step's GEMMs from 47.40
/// TFLOP/s to 47.57. Grown and never shrunk, and not thread safe, which matches what the library
/// says about a device having one of everything.
uint8_t *splitKWorkspace(size_t bytes) {
  static lut::c_ptr<uint8_t> buffer;
  static size_t capacity = 0;

  if (bytes == 0) return nullptr;
  if (bytes > capacity) {
    buffer = llynCudaAlloc<uint8_t>(bytes);
    capacity = bytes;
  }
  return buffer.get();
}

/// How many ways to split K, which is how a GEMM too small to fill the machine fills it anyway.
///
/// The tile count is the CTA count, so a small output leaves most of the card idle however fast
/// the kernel is: splitting K multiplies the CTAs by the number of slices and reduces the partial
/// sums afterwards. That reduction is not free, which is why this only splits when there is a
/// reason to -- below four waves -- and only up to what fills about nine.
///
/// The thresholds are in waves rather than in CTAs, and the SM count is read rather than written
/// down, so they mean the same thing on a card with a different number of them.
///
/// What it decides for the ten shapes SDXL's U-Net runs at 1024 by 1024, on a 36 SM part, next to
/// the share of a step's GEMM time each of them is:
///
///     1024x10240x1280   36.2%    640 CTAs   1 slice
///     1024x1280x5120    17.9%     80 CTAs   4 slices
///     1024x1280x1280    15.0%     80 CTAs   4 slices
///     1024x3840x1280    13.7%    240 CTAs   1 slice
///     4096x5120x640      6.2%   1280 CTAs   1 slice
///     4096x640x640       3.2%    160 CTAs   1 slice
///     4096x640x2560      3.0%    160 CTAs   1 slice
///     4096x1920x640      2.3%    480 CTAs   1 slice
///     77x2560x2048       2.3%     20 CTAs   8 slices
///     77x1280x2048       0.2%     10 CTAs   8 slices
///
/// So a third of the time splits and two thirds does not, and the four that split are exactly the
/// four that were behind cuBLAS before this: by 25% and 24% at 1024 by 1280, and by 31% and 55%
/// where there are 77 rows. Sweeping one, two, four and eight slices per shape agrees with what
/// the rule picks for each.
int splitKSlices(int m, int n, int k) {
  constexpr int kEnoughWaves = 4;
  constexpr int kTargetWaves = 9;
  constexpr int kMaxSlices = 8;

  // A slice that is too short spends more time being reduced than it saves.
  constexpr int kMinKPerSlice = 128;

  int multiprocessors = getCudaDeviceAttribute(cudaDevAttrMultiProcessorCount);
  int ctas = ((m + kTileM - 1) / kTileM) * ((n + kTileN - 1) / kTileN);
  if (ctas >= kEnoughWaves * multiprocessors) return 1;

  int slices = std::min(kMaxSlices, std::max(1, kTargetWaves * multiprocessors / ctas));
  while (slices > 1 && k / slices < kMinKPerSlice) slices /= 2;

  return slices;
}

template<class LayoutA, class LayoutB, class ArchTag, int ALIGNMENT>
void hgemmT(
    int m,
    int n,
    int k,
    cutlass::half_t alpha,
    const cutlass::half_t *A,
    int lda,
    const cutlass::half_t *B,
    int ldb,
    cutlass::half_t beta,
    cutlass::half_t *C,
    int ldc) {
  using Gemm = typename Sm80Gemm<LayoutA, LayoutB, ALIGNMENT>::Gemm;
  Gemm gemmOperator;

  // The epilogue computes in float now, so the scalars go in as float.
  typename Gemm::Arguments args{
      {m, n, k},
      {A, lda},
      {B, ldb},
      {C, ldc},
      {C, ldc},
      {float(alpha), float(beta)},
      splitKSlices(m, n, k)};

  // Asked rather than assumed. `initialize` does not call this -- it builds the params and
  // returns success -- so an operand no instantiation can read is otherwise launched anyway, and
  // what comes back is a misaligned access inside a kernel with no way to report it. On this
  // hardware that kills the CUDA context, and since every later call then fails too, the first
  // one to notice is a `cudaFreeAsync` in a tensor destructor -- where the `CHECK` on it throws,
  // a destructor is `noexcept`, and the process goes to `std::terminate` without printing
  // anything at all. Refusing it here costs one host-side check per GEMM and names the operand.
  CUTLASS_CHECK(Gemm::can_implement(args));

  // Only a split needs anywhere to put its partial sums; at one slice this is zero bytes and
  // nothing is asked for. CUTLASS zeroes what it is given, so nothing here has to.
  size_t workspaceSize = Gemm::get_workspace_size(args);
  CUTLASS_CHECK(gemmOperator.initialize(args, splitKWorkspace(workspaceSize)));
  CUTLASS_CHECK(gemmOperator());
}

/// A copy of a 2-D operand in a buffer the widest kernel can read.
///
/// Zeros in the margin are what make this exact rather than merely aligned. The caller grows `m`,
/// `n` and `k` to multiples of eight and runs the product at those, so an input buffer carries
/// its operand in the top-left corner and zeros everywhere else: a zero row of A contributes a
/// zero row of the answer, a zero column of B a zero column, and a zero tail on K contributes
/// nothing to any sum. What comes back in the corner is the product that was asked for, to the
/// bit. A fresh allocation is 256-byte aligned, so the base needs nothing done to it.
struct PaddedOperand {
  lut::c_ptr<cutlass::half_t> buffer;
  cutlass::half_t *data = nullptr;
  int ld = 0;
};

/// `n` rounded up to where the widest kernel can start every row.
constexpr int roundUpToAlignment(int n) {
  return (n + kAlignments[0] - 1) / kAlignments[0] * kAlignments[0];
}

/// `srcRows` rows of `srcCols` halves, stepping `srcLd`, in the corner of a zeroed `dstRows` by
/// `dstCols` buffer that steps `dstCols`.
///
/// `copyIn` false leaves the buffer untouched, which is what an output wants: the kernel defines
/// every element of it, and nothing is going to read it back into a sum first.
PaddedOperand padOperand(
    const cutlass::half_t *src,
    int srcRows,
    int srcCols,
    int srcLd,
    int dstRows,
    int dstCols,
    bool copyIn = true) {
  constexpr size_t kHalf = sizeof(cutlass::half_t);

  PaddedOperand out;
  out.ld = dstCols;
  out.buffer = llynCudaAlloc<cutlass::half_t>(int64_t(dstRows) * dstCols);
  out.data = out.buffer.get();

  // An output is left as it is found. The kernel writes every element of it -- the extents it is
  // given are the padded ones and the leading dimension is their width -- so there is nothing
  // here that it does not define, and zeroing it first would be writing the whole buffer twice.
  if (!copyIn) return out;

  // An input is zeroed only where the copy does not reach. Zeroing the whole buffer and then
  // overwriting the corner costs as much again as the copy, and on a 1024 by 1280 by 1280 shape
  // that was half of what this path spends. The margin is at most seven columns and seven rows.
  if (dstCols > srcCols) {
    LL_CHECK_CUDA_STATUS(cudaMemset2D(
        out.data + srcCols,
        size_t(dstCols) * kHalf,
        0,
        size_t(dstCols - srcCols) * kHalf,
        size_t(srcRows)));
  }
  if (dstRows > srcRows) {
    LL_CHECK_CUDA_STATUS(cudaMemset(
        out.data + int64_t(srcRows) * dstCols,
        0,
        size_t(int64_t(dstRows - srcRows) * dstCols) * kHalf));
  }

  LL_CHECK_CUDA_STATUS(cudaMemcpy2D(
      out.data,
      size_t(dstCols) * kHalf,
      src,
      size_t(srcLd) * kHalf,
      size_t(srcCols) * kHalf,
      size_t(srcRows),
      cudaMemcpyDeviceToDevice));

  return out;
}

/// How an operand of `r` by `c` in `Layout` sits in memory, as rows of contiguous halves.
///
/// Row-major steps between rows, so it is `r` rows of `c`. Column-major steps between columns --
/// element (i, j) is at i + j * ld -- so it is `c` rows of `r`. Both step by the leading
/// dimension, which is what has to come out a multiple of eight.
template<class Layout>
void operandExtent(int r, int c, int *rows, int *cols) {
  constexpr bool kRowMajor = std::is_same<Layout, RowMajor>::value;
  *rows = kRowMajor ? r : c;
  *cols = kRowMajor ? c : r;
}

/// The same call at the widest alignment the three operands can actually be read at.
template<class LayoutA, class LayoutB, class ArchTag>
void hgemm(
    int m,
    int n,
    int k,
    cutlass::half_t alpha,
    const cutlass::half_t *A,
    int lda,
    const cutlass::half_t *B,
    int ldb,
    cutlass::half_t beta,
    cutlass::half_t *C,
    int ldc) {
  auto fits = [&](int alignment) {
    return isAlignedTo(alignment, A, lda) && isAlignedTo(alignment, B, ldb) &&
           isAlignedTo(alignment, C, ldc);
  };

  if (fits(kAlignments[0])) {
    return hgemmT<LayoutA, LayoutB, ArchTag, kAlignments[0]>(
        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  if (fits(kAlignments[1])) {
    return hgemmT<LayoutA, LayoutB, ArchTag, kAlignments[1]>(
        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  if (fits(kAlignments[2])) {
    return hgemmT<LayoutA, LayoutB, ArchTag, kAlignments[2]>(
        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  // Nothing on the ladder can read these: a leading dimension that is odd has every second row
  // starting on an odd number of halves, and two halves is the narrowest access a staged mainloop
  // has. Refusing them is not an option -- a 1x1 convolution over an odd number of audio frames
  // reaches here, and `conv1d` with a kernel of one is exactly that, on an operand whose leading
  // dimension is the frame count. Nor is a SIMT kernel that reads an element at a time, which
  // answers them four times slower: 316us against 77us at 1024 by 1280 by 1280.
  //
  // So copy. Every extent grows to a multiple of eight and all three operands are copied into
  // zeroed buffers stepping their own width, which is the shape the widest kernel wants and the
  // one it is fastest on. The zeros are why this is exact: a zero row of A gives a zero row of
  // the answer, a zero column of B a zero column, and a zero tail on K adds nothing to any sum,
  // so the corner copied back is the product that was asked for.
  //
  // All three, including any that could have been read as they stand. Growing K means reading
  // past where the untouched operand's data ends, and past there is whatever the allocator last
  // left rather than a zero.
  //
  // What it costs is three copies and a fourth back, each an area against a product that is an
  // area times an extent, so the ratio falls away as the shapes grow. Measured at 1024 by 1281 by
  // 1280: 109us, against 76us for the even shape beside it and 316us for the SIMT kernel this
  // replaced. Call it half again as long on a shape that reaches here, and nothing at all on one
  // that does not -- an operand the ladder above can read never sees any of this.
  //
  // Copying only the operands that are misaligned would get most of that back. It is not done
  // because it cannot be decided per operand: growing an extent forces every operand that shares
  // it to be padded too, whether or not that one was readable, and the version that tried to be
  // clever about which is which is the version that read off the end of A.
  int paddedM = roundUpToAlignment(m);
  int paddedN = roundUpToAlignment(n);
  int paddedK = roundUpToAlignment(k);

  int aRows, aCols, aPaddedRows, aPaddedCols;
  operandExtent<LayoutA>(m, k, &aRows, &aCols);
  operandExtent<LayoutA>(paddedM, paddedK, &aPaddedRows, &aPaddedCols);

  int bRows, bCols, bPaddedRows, bPaddedCols;
  operandExtent<LayoutB>(k, n, &bRows, &bCols);
  operandExtent<LayoutB>(paddedK, paddedN, &bPaddedRows, &bPaddedCols);

  PaddedOperand paddedA = padOperand(A, aRows, aCols, lda, aPaddedRows, aPaddedCols);
  PaddedOperand paddedB = padOperand(B, bRows, bCols, ldb, bPaddedRows, bPaddedCols);

  // The output is row-major whatever the operands are. A non-zero beta reads it, so it is copied
  // in as well as out.
  PaddedOperand paddedC =
      padOperand(C, m, n, ldc, paddedM, paddedN, beta != cutlass::half_t(0.0f));

  hgemmT<LayoutA, LayoutB, ArchTag, kAlignments[0]>(
      paddedM,
      paddedN,
      paddedK,
      alpha,
      paddedA.data,
      paddedA.ld,
      paddedB.data,
      paddedB.ld,
      beta,
      paddedC.data,
      paddedC.ld);

  LL_CHECK_CUDA_STATUS(cudaMemcpy2D(
      C,
      size_t(ldc) * sizeof(cutlass::half_t),
      paddedC.data,
      size_t(paddedC.ld) * sizeof(cutlass::half_t),
      size_t(n) * sizeof(cutlass::half_t),
      size_t(m),
      cudaMemcpyDeviceToDevice));
}

template<class ArchTag>
void cutlassHgemmArch(
    bool transA,
    bool transB,
    int m,
    int n,
    int k,
    cutlass::half_t alpha,
    const cutlass::half_t *A,
    int lda,
    const cutlass::half_t *B,
    int ldb,
    cutlass::half_t beta,
    cutlass::half_t *C,
    int ldc) {
  if (transA == false && transB == false) {
    return hgemm<RowMajor, RowMajor, ArchTag>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  } else if (transA == true && transB == false) {
    return hgemm<ColumnMajor, RowMajor, ArchTag>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  } else if (transA == false && transB == true) {
    return hgemm<RowMajor, ColumnMajor, ArchTag>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  } else if (transA == true && transB == true) {
    return hgemm<ColumnMajor, ColumnMajor, ArchTag>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  } else {
    NOT_IMPL();
  }
}

void cutlassHgemm(
    bool transA,
    bool transB,
    int m,
    int n,
    int k,
    cutlass::half_t alpha,
    const cutlass::half_t *A,
    int lda,
    const cutlass::half_t *B,
    int ldb,
    cutlass::half_t beta,
    cutlass::half_t *C,
    int ldc) {
  cutlassHgemmArch<
      cutlass::arch::Sm90>(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<class LayoutA, class layoutB>
void hgemmArrayT(
    int m,
    int n,
    int k,
    cutlass::half_t alpha,
    const cutlass::half_t *const *A,
    int lda,
    const cutlass::half_t *const *B,
    int ldb,
    cutlass::half_t beta,
    cutlass::half_t *const *C,
    int ldc,
    int batchSize) {
  using Gemm = cutlass::gemm::device::GemmArray<
      cutlass::half_t,
      LayoutA,
      cutlass::half_t,
      layoutB,
      cutlass::half_t,
      RowMajor,
      float>;
  Gemm gemmOperator;

  typename Gemm::Arguments
      args({m, n, k}, A, lda, B, ldb, C, ldc, C, ldc, {alpha, beta}, batchSize);
  CUTLASS_CHECK(gemmOperator(args));
}

void cutlassHgemmArray(
    bool transA,
    bool transB,
    int m,
    int n,
    int k,
    cutlass::half_t alpha,
    const cutlass::half_t *const *A,
    int lda,
    const cutlass::half_t *const *B,
    int ldb,
    cutlass::half_t beta,
    cutlass::half_t *const *C,
    int ldc,
    int batchSize) {
  int bs = batchSize;
  if (transA == false && transB == false) {
    hgemmArrayT<RowMajor, RowMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bs);
  } else if (transA == true && transB == false) {
    hgemmArrayT<ColumnMajor, RowMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bs);
  } else if (transA == false && transB == true) {
    hgemmArrayT<RowMajor, ColumnMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bs);
  } else if (transA == true && transB == true) {
    hgemmArrayT<ColumnMajor, ColumnMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bs);
  } else {
    NOT_IMPL();
  }
}

/// Float in, float out, on the SIMT pipeline.
///
/// The autoencoder is the only thing here that runs in float32, and it is the only reason this
/// exists: without it a model loaded on this backend aborts the moment it reaches the decoder,
/// which is what kept cuBLAS from being optional rather than preferred.
///
/// SIMT rather than a tensor core path, which is a deliberate trade. A float GEMM on tensor cores
/// means TF32, whose eight exponent bits carry the range the autoencoder needs but whose ten
/// mantissa bits do not carry what float32 carries -- so it would answer a slightly different
/// question and the decoder's agreement with the reference would have to be established again.
/// SIMT is the same arithmetic cuBLAS does under CUBLAS_COMPUTE_32F, and what it costs is not
/// worth arguing about: the four GEMMs the decoder runs are 0.58 TFLOP of an image's 270, and a
/// float GEMM has no tensor cores to leave on the table in the first place.
template<class LayoutA, class LayoutB>
struct Sm80SimtGemm {
  using Gemm = cutlass::gemm::device::Gemm<
      float,
      LayoutA,
      float,
      LayoutB,
      float,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassSimt,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 8>,
      cutlass::gemm::GemmShape<32, 64, 8>,
      cutlass::gemm::GemmShape<1, 1, 1>,
      cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      1,
      1,
      true>;
};

template<class LayoutA, class LayoutB>
void sgemmT(
    int m,
    int n,
    int k,
    float alpha,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    float beta,
    float *C,
    int ldc) {
  using Gemm = typename Sm80SimtGemm<LayoutA, LayoutB>::Gemm;
  Gemm gemmOperator;

  // The same tile as the half path, so the same rule says how to split it.
  typename Gemm::Arguments args{
      {m, n, k},
      {A, lda},
      {B, ldb},
      {C, ldc},
      {C, ldc},
      {alpha, beta},
      splitKSlices(m, n, k)};

  size_t workspaceSize = Gemm::get_workspace_size(args);
  CUTLASS_CHECK(gemmOperator.initialize(args, splitKWorkspace(workspaceSize)));
  CUTLASS_CHECK(gemmOperator());
}

void cutlassSgemm(
    bool transA,
    bool transB,
    int m,
    int n,
    int k,
    float alpha,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    float beta,
    float *C,
    int ldc) {
  if (!transA && !transB) {
    return sgemmT<RowMajor, RowMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  } else if (transA && !transB) {
    return sgemmT<ColumnMajor, RowMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  } else if (!transA && transB) {
    return sgemmT<RowMajor, ColumnMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  } else {
    return sgemmT<ColumnMajor, ColumnMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
}

template<class LayoutA, class LayoutB>
void sgemmArrayT(
    int m,
    int n,
    int k,
    float alpha,
    const float *const *A,
    int lda,
    const float *const *B,
    int ldb,
    float beta,
    float *const *C,
    int ldc,
    int batchSize) {
  using Gemm = cutlass::gemm::device::
      GemmArray<float, LayoutA, float, LayoutB, float, RowMajor, float>;
  Gemm gemmOperator;

  typename Gemm::Arguments
      args({m, n, k}, A, lda, B, ldb, C, ldc, C, ldc, {alpha, beta}, batchSize);
  CUTLASS_CHECK(gemmOperator(args));
}

void cutlassSgemmArray(
    bool transA,
    bool transB,
    int m,
    int n,
    int k,
    float alpha,
    const float *const *A,
    int lda,
    const float *const *B,
    int ldb,
    float beta,
    float *const *C,
    int ldc,
    int batchSize) {
  if (!transA && !transB) {
    return sgemmArrayT<RowMajor, RowMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchSize);
  } else if (transA && !transB) {
    return sgemmArrayT<ColumnMajor, RowMajor>(
        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchSize);
  } else if (!transA && transB) {
    return sgemmArrayT<RowMajor, ColumnMajor>(
        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchSize);
  } else {
    return sgemmArrayT<ColumnMajor, ColumnMajor>(
        m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchSize);
  }
}

void CutlassGemm::hgemm(
    bool transA,
    bool transB,
    int m,
    int n,
    int k,
    __half alpha,
    const __half *A,
    int lda,
    const __half *B,
    int ldb,
    __half beta,
    __half *C,
    int ldc) {
  cutlass::half_t alphaH = *reinterpret_cast<cutlass::half_t *>(&alpha);
  cutlass::half_t betaH = *reinterpret_cast<cutlass::half_t *>(&beta);
  cutlassHgemm(
      transA,
      transB,
      m,
      n,
      k,
      alphaH,
      reinterpret_cast<const cutlass::half_t *>(A),
      lda,
      reinterpret_cast<const cutlass::half_t *>(B),
      ldb,
      betaH,
      reinterpret_cast<cutlass::half_t *>(C),
      ldc);
}

void CutlassGemm::hgemmArray(
    bool transA,
    bool transB,
    int m,
    int n,
    int k,
    __half alpha,
    const __half *const *arrayA,
    int lda,
    const __half *const *arrayB,
    int ldb,
    __half beta,
    __half *const *arrayC,
    int ldc,
    int batchSize) {
  cutlass::half_t alphaH = *reinterpret_cast<cutlass::half_t *>(&alpha);
  cutlass::half_t betaH = *reinterpret_cast<cutlass::half_t *>(&beta);
  cutlassHgemmArray(
      transA,
      transB,
      m,
      n,
      k,
      alphaH,
      reinterpret_cast<const cutlass::half_t *const *>(arrayA),
      lda,
      reinterpret_cast<const cutlass::half_t *const *>(arrayB),
      ldb,
      betaH,
      reinterpret_cast<cutlass::half_t *const *>(arrayC),
      ldc,
      batchSize);
}

std::shared_ptr<Gemm> CutlassGemm::create() {
  std::shared_ptr<CutlassGemm> mm = std::make_shared<CutlassGemm>();
  return mm;
}

void CutlassGemm::sgemm(
    bool transA,
    bool transB,
    int m,
    int n,
    int k,
    float alpha,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    float beta,
    float *C,
    int ldc) {
  cutlassSgemm(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void CutlassGemm::sgemmArray(
    bool transA,
    bool transB,
    int m,
    int n,
    int k,
    float alpha,
    const float *const *arrayA,
    int lda,
    const float *const *arrayB,
    int ldb,
    float beta,
    float *const *arrayC,
    int ldc,
    int batchSize) {
  cutlassSgemmArray(
      transA, transB, m, n, k, alpha, arrayA, lda, arrayB, ldb, beta, arrayC, ldc, batchSize);
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
