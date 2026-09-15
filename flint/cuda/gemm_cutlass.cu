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
template<class LayoutA, class LayoutB>
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
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      3,
      8,
      8,
      true>;
};

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

template<class LayoutA, class LayoutB, class ArchTag>
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
  using Gemm = typename Sm80Gemm<LayoutA, LayoutB>::Gemm;
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

  // Only a split needs anywhere to put its partial sums; at one slice this is zero bytes and
  // nothing is asked for. CUTLASS zeroes what it is given, so nothing here has to.
  size_t workspaceSize = Gemm::get_workspace_size(args);
  CUTLASS_CHECK(gemmOperator.initialize(args, splitKWorkspace(workspaceSize)));
  CUTLASS_CHECK(gemmOperator());
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
    return hgemmT<RowMajor, RowMajor, ArchTag>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  } else if (transA == true && transB == false) {
    return hgemmT<ColumnMajor, RowMajor, ArchTag>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  } else if (transA == false && transB == true) {
    return hgemmT<RowMajor, ColumnMajor, ArchTag>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  } else if (transA == true && transB == true) {
    return hgemmT<ColumnMajor, ColumnMajor, ArchTag>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
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
