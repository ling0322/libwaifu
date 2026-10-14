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

// Half times E4M3 on the ordinary half tensor cores. Nothing here multiplies in FP8: the MMA is
// the same HMMA a half GEMM runs, and the weight is widened to half between shared memory and the
// operand registers, one K tile at a time. So what the narrow weight saves is the traffic -- half
// the bytes from global memory, half the shared memory, and the same arithmetic -- which is what
// a projection at a small row count is bound by.
//
// CUTLASS supplies both halves of that. The mainloop is its 2.x mixed input tensor op, reached by
// tagging the operator `OpMultiplyAddMixedInputUpcast`; the epilogue is an EVT tree that reads
// the weight's one scale and multiplies the accumulator by it, so the scale costs no pass of its
// own.

#include <cuda_fp16.h>

#include <limits>

// The order matters: `visitors.hpp` reaches for things -- NumericArrayConverter among them --
// that it does not include itself, so `gemm_universal.h` has to come first. This is the order
// CUTLASS's own example 47 uses.
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "lutil/error.h"
#include "flint/cuda/common.h"
#include "flint/cuda/gemm_fp8_cutlass.h"

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
namespace {

using namespace cute;

// A is row major and B column major, which for a weight stored one output channel per row means B
// is that weight read as its own transpose.
using ElementA = cutlass::half_t;
using ElementB = cutlass::float_e4m3_t;
using ElementOutput = cutlass::half_t;
using ElementScale = float;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// 128 bits at each operand's own width: eight halves, sixteen E4M3 codes.
constexpr int kAlignmentA = 8;
constexpr int kAlignmentB = 16;
constexpr int kAlignmentC = 8;

/// One tile shape's worth of kernel: the mixed input mainloop, and an EVT epilogue that multiplies
/// the accumulator by the weight's scale before writing half.
///
/// The K tile is 64 in both instantiations and cannot be anything else. A half operand's shared
/// memory layout is `TensorOpMultiplicandCrosswise<16, kK>`, whose crosswise extent is a 128 byte
/// run, so 64 halves is where it ends -- the deeper K tile a narrow operand would otherwise want
/// is not reachable while A stays half.
///
/// Two stages is not reachable either: at two CUTLASS builds the mainloop out of `MmaPipelined`
/// rather than `MmaMultistage`, and the mixed input warp operator has no overload it can call.
/// Three works and measures within half a percent of four everywhere, so four it is.
template<class ThreadblockShape, class WarpShape, int Stages>
struct Fp8Gemm {
  static constexpr int kEpilogueStages = 1;

  using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
      ThreadblockShape,
      WarpShape,
      ElementOutput,
      kAlignmentC,
      kEpilogueStages>;

  using Accumulator = cutlass::epilogue::threadblock::VisitorAccFetch;

  // D = accumulator * scale. Read through a pointer by every thread as the epilogue starts, so a
  // scale that lives on the device never has to come back to the host to be launched with.
  using Scale = cutlass::epilogue::threadblock::VisitorScalarBroadcast<ElementScale>;

  using ApplyScale = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies,
      ElementOutput,
      ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EvtScale = cutlass::epilogue::threadblock::Sm80EVT<ApplyScale, Accumulator, Scale>;

  using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap,
      ElementOutput,
      cutlass::FloatRoundStyle::round_to_nearest,
      Stride<int64_t, _1, int64_t>>;

  using EvtD = cutlass::epilogue::threadblock::Sm80EVT<StoreD, EvtScale>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementA,
      LayoutA,
      cutlass::ComplexTransform::kNone,
      kAlignmentA,
      ElementB,
      LayoutB,
      cutlass::ComplexTransform::kNone,
      kAlignmentB,
      ElementOutput,
      LayoutC,
      kAlignmentC,
      ElementAccumulator,
      ElementCompute,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      ThreadblockShape,
      WarpShape,
      cutlass::gemm::GemmShape<16, 8, 16>,
      EvtD,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      Stages,
      cutlass::arch::OpMultiplyAddMixedInputUpcast,
      kEpilogueStages>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

/// The tile for a small row count, and what decides that there are two of these.
///
/// At one row a 128 row tile computes 128 rows and throws 127 away, and since the arithmetic is
/// the same HMMA a half GEMM runs, that waste is the whole cost. The flat tile also has a narrow
/// n, which is the other half of the same problem: at one row the tile count is the n tile count,
/// and 3072 output channels over a 128 wide tile is 24 CTAs on a 36 SM part.
///
/// Microseconds on an RTX 5060 Ti. The weights are cold where they are larger than the 32 MB L2,
/// which the 16384 by 3072 one is (50 MB in E4M3) and the others are not.
///
///     shape               32x64x64   32x128x64   128x128x64
///     1x1280x1280             10.4        17.4         32.4
///     1x3072x3072             23.4        39.7         75.0
///     1x16384x3072           131.2       128.2        302.2
///     16x3072x3072            23.6        39.8         75.0
///     64x3072x3072            41.0        40.4         75.3
///     128x5120x3072           98.8       118.2        150.6
///     512x5120x3072          370.9       355.9        375.1
///     512x16384x3072        1158.6      1134.3       1122.2
///     1024x10240x1280        618.5       616.7        588.0
///     4096x640x2560          334.1       344.8        315.0
///
/// So the flat tile is ahead by up to three times until the rows fill a square one, and the
/// square tile is never more than 5% ahead beyond that. `gemmFp8` splits them at 512 rows.
using FlatTile =
    Fp8Gemm<cutlass::gemm::GemmShape<32, 64, 64>, cutlass::gemm::GemmShape<32, 32, 64>, 4>;

using SquareTile =
    Fp8Gemm<cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 32, 64>, 4>;

constexpr int kFlatTileMaxRow = 512;

/// @param scale the weight's one scale, <float>(1), on the device.
template<class Config>
void runFp8Gemm(int m, int n, int k, const half *A, const Fp8E4M3 *B, const float *scale, half *D) {
  using Gemm = typename Config::Gemm;

  typename Config::Scale::Arguments scaleArgs;
  scaleArgs.scalar_ptrs[0] = scale;

  typename Config::EvtD::Arguments callbackArgs{
      {
          {},         // accumulator
          scaleArgs,  // scale
          {}          // multiply
      },
      {reinterpret_cast<ElementOutput *>(D), {int64_t(n), _1{}, int64_t(m) * n}}};

  // Batch count one: the SM80 EVT epilogue asserts against a split K, so unlike the half GEMM
  // beside it this one cannot fill the machine by splitting a small output. The flat tile is what
  // fills it instead, and it fills it along n rather than along k.
  typename Gemm::Arguments arguments(
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k},
      1,
      callbackArgs,
      reinterpret_cast<const ElementA *>(A),
      reinterpret_cast<const ElementB *>(B),
      nullptr,
      nullptr,
      int64_t(m) * k,
      int64_t(n) * k,
      0,
      0,
      int64_t(k),
      int64_t(k),
      int64_t(0),
      int64_t(0));

  Gemm gemm;
  size_t workspaceSize = Gemm::get_workspace_size(arguments);
  lut::c_ptr<uint8_t> workspace;
  if (workspaceSize) workspace = llynCudaAlloc<uint8_t>(workspaceSize);

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm.run());
}

}  // namespace

bool isFp8GemmAvailable() {
  // An sm_80 kernel with nothing arch conditional in it, so this asks only whether the tensor
  // cores it is written against exist.
  static const bool available = getCudaArch() >= 80;
  return available;
}

namespace {

/// A (..., k) times the transpose of B (n, k), times `scale`, once gemmFp8 has checked B and the
/// scale.
void gemmFp8Impl(TensorView A, TensorView B, TensorView scale, TensorView D) {
  CHECK(A.getDevice().getType() == Device::kCuda);
  CHECK(A.getDType() == DType::kFloat16);
  CHECK(A.isContiguous());
  CHECK(A.getDim() >= 2);
  CHECK(A.getShape(-1) == B.getShape(1));

  std::vector<int> shapeD = A.getShape();
  shapeD.back() = B.getShape(0);
  CHECK(D.getDevice().getType() == Device::kCuda && D.getDType() == DType::kFloat16);
  CHECK(D.isContiguous() && D.getShape() == shapeD);

  if (A.getDim() > 2) {
    gemmFp8Impl(A.view({-1, A.getShape(-1)}), B, scale, D.view({-1, B.getShape(0)}));
    return;
  }

  if (!isFp8GemmAvailable()) {
    throw lut::AbortedError("the fp8 mixed precision kernel needs an sm_80 or newer device.");
  }

  int m = A.getShape(0);
  int n = B.getShape(0);
  int k = B.getShape(1);

  // The epilogue writes D 128 bits at a time along n, and the mainloop reads the weight 128 bits
  // at a time along k. m is free.
  CHECK(n % kAlignmentC == 0);
  CHECK(k % kAlignmentB == 0);

  const half *ptrA = getDataPtrCuda<half>(A);
  const Fp8E4M3 *ptrB = getDataPtrCuda<Fp8E4M3>(B);
  const float *ptrScale = getDataPtrCuda<float>(scale);
  half *ptrD = getDataPtrCuda<half>(D);

  if (m <= kFlatTileMaxRow) {
    runFp8Gemm<FlatTile>(m, n, k, ptrA, ptrB, ptrScale, ptrD);
  } else {
    runFp8Gemm<SquareTile>(m, n, k, ptrA, ptrB, ptrScale, ptrD);
  }

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

}  // namespace

void gemmFp8(TensorView A, TensorView B, TensorView scale, TensorView D) {
  CHECK(B.getDevice().getType() == Device::kCuda);
  CHECK(B.getDType() == DType::kFp8E4M3);
  CHECK(B.getDim() == 2);
  CHECK(B.isContiguous());
  CHECK(B.getNumEl() < std::numeric_limits<int32_t>::max());

  // One element, whatever its rank: a checkpoint stores the scale as a 0-d tensor as often as as
  // a (1). What the kernel dereferences is the first float, on the device.
  CHECK(scale.getDevice().getType() == Device::kCuda);
  CHECK(scale.getDType() == DType::kFloat);
  CHECK(scale.getNumEl() == 1);

  gemmFp8Impl(A, B, scale, D);
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
