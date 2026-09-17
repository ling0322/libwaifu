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
// the per channel scale as a row vector and multiplies the accumulator by it, so the scale costs
// no pass of its own.

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
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
#include "flint/cuda/fp8.h"
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

/// Eight elements a thread: sixteen bytes in and eight bytes out, which is what makes both ends of
/// the quantizer a single vector access.
constexpr int kElementPerThread = 8;

__forceinline__ __device__ float blockReduceMax(float v) {
  __shared__ float warpMax[32];

  int lane = threadIdx.x % 32;
  int warp = threadIdx.x / 32;

#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
  }
  if (lane == 0) warpMax[warp] = v;
  __syncthreads();

  int numWarp = (int(blockDim.x) + 31) / 32;
  v = threadIdx.x < numWarp ? warpMax[threadIdx.x] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
  }

  return v;
}

/// @brief One block owns one row, which is what makes the scale a block reduction rather than a
///        second launch: a row's maximum is known before the same block quantizes it.
///
/// The row is read twice, once for the maximum and once for the elements. That is the cost of
/// keeping it in one launch, and a weight is quantized once at load, so it is not a cost anything
/// pays twice.
__global__ void quantizeFp8Kernel(
    const half *__restrict__ x,
    int k,
    uint8_t *__restrict__ q,
    float *__restrict__ scale) {
  int row = blockIdx.x;
  const PackedOWORD<half2> *xRow =
      reinterpret_cast<const PackedOWORD<half2> *>(x + static_cast<int64_t>(row) * k);
  uint2 *qRow = reinterpret_cast<uint2 *>(q + static_cast<int64_t>(row) * k);
  int numPack = k / kElementPerThread;

  float amax = 0.0f;
  for (int i = threadIdx.x; i < numPack; i += blockDim.x) {
    PackedOWORD<half2> po = xRow[i];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f2 = __half22float2(po.v[j]);
      amax = fmaxf(amax, fmaxf(fabsf(f2.x), fabsf(f2.y)));
    }
  }
  amax = blockReduceMax(amax);

  __shared__ float sRcpScale;
  if (threadIdx.x == 0) {
    // An all zero row has no scale to speak of. Zero is the one that makes the round trip exact
    // and leaves nothing to divide by later.
    //
    // __fdiv_rn rather than `/`, which this file is compiled with --use_fast_math and would
    // otherwise turn into a reciprocal approximation. One ulp on the scale moves the odd element
    // that sits on a rounding boundary to the next code, and then a weight quantized here and one
    // quantized on the processor are not the same bytes. Two of 16896 elements is what it was
    // costing when the test that compares them was first run.
    scale[row] = __fdiv_rn(amax, kFp8E4M3Max);
    sRcpScale = amax > 0.0f ? __fdiv_rn(kFp8E4M3Max, amax) : 0.0f;
  }
  __syncthreads();
  float rcpScale = sRcpScale;

  for (int i = threadIdx.x; i < numPack; i += blockDim.x) {
    PackedOWORD<half2> po = xRow[i];
    union {
      uint16_t pair[4];
      uint2 v;
    } out;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f2 = __half22float2(po.v[j]);
      f2.x *= rcpScale;
      f2.y *= rcpScale;
      out.pair[j] = __nv_cvt_float2_to_fp8x2(f2, __NV_SATFINITE, __NV_E4M3);
    }
    qRow[i] = out.v;
  }
}

__global__ void dequantizeFp8Kernel(
    const uint8_t *__restrict__ q,
    const float *__restrict__ scale,
    half *__restrict__ x,
    int rows,
    int k) {
  int numPack = k / kElementPerThread;
  int total = rows * numPack;

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    half2 rowScale = __float2half2_rn(scale[idx / numPack]);
    union {
      uint16_t pair[4];
      uint2 v;
    } in;
    in.v = reinterpret_cast<const uint2 *>(q)[idx];

    PackedOWORD<half2> out;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      half2 v = half2(__nv_cvt_fp8x2_to_halfraw2(in.pair[j], __NV_E4M3));
      out.v[j] = __hmul2(v, rowScale);
    }
    reinterpret_cast<PackedOWORD<half2> *>(x)[idx] = out;
  }
}

using namespace cute;

// A is row major and B column major, which for a weight stored one output channel per row means B
// is that weight read as its own transpose -- the same arrangement the NVFP4 path uses.
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
/// the accumulator by the per channel scale before writing half.
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

  // D = accumulator * channelScale[n]. The scale is constant down a column of the result, so it is
  // a row vector broadcast over m -- a stride of zero on m, one on n.
  using Accumulator = cutlass::epilogue::threadblock::VisitorAccFetch;

  using ChannelScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap,
      ElementScale,
      Stride<_0, _1, int32_t>>;

  using ApplyScale = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies,
      ElementOutput,
      ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EvtScale = cutlass::epilogue::threadblock::Sm80EVT<ApplyScale, Accumulator, ChannelScale>;

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
using FlatTile = Fp8Gemm<
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    4>;

using SquareTile = Fp8Gemm<
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 32, 64>,
    4>;

constexpr int kFlatTileMaxRow = 512;

template<class Config>
void runFp8Gemm(
    int m,
    int n,
    int k,
    const half *A,
    const Fp8E4M3 *B,
    const float *channelScale,
    half *D) {
  using Gemm = typename Config::Gemm;

  typename Config::EvtD::Arguments callbackArgs{
      {
          {},                                                                     // accumulator
          {channelScale, ElementScale(1), {_0{}, _1{}, int32_t(n)}},               // channel scale
          {}                                                                      // multiply
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
  // An sm_80 kernel with nothing arch conditional in it, so unlike the NVFP4 path this asks only
  // whether the tensor cores it is written against exist.
  static const bool available = getCudaArch() >= 80;
  return available;
}

Fp8Operand quantizeFp8(const Tensor &x) {
  CHECK(x.getDevice().getType() == Device::kCuda);
  CHECK(x.getDType() == DType::kFloat16);
  CHECK(x.getDim() == 2);
  LL_CHECK_CONTIGUOUS(x);

  int rows = x.getShape(0);
  int k = x.getShape(1);

  // Sixteen is what the mainloop reads the weight in; it also keeps every row of the packed data
  // 16 byte aligned, which is what the quantizer's own vector accesses need.
  CHECK(k % 16 == 0);
  CHECK(x.getNumEl() < std::numeric_limits<int32_t>::max());

  Fp8Operand operand;
  operand.rows = rows;
  operand.k = k;
  operand.data = createCudaTensorFp8E4M3({rows, k});
  operand.channelScale = createCudaTensorFloat({rows});

  // One block a row, capped where a row is shorter than the block would be wide.
  int blockSize = std::min(256, std::max(32, (k / kElementPerThread + 31) / 32 * 32));
  quantizeFp8Kernel<<<rows, blockSize>>>(
      getDataPtrCuda<half>(x),
      k,
      reinterpret_cast<uint8_t *>(getDataPtrCuda<Fp8E4M3>(operand.data)),
      getDataPtrCuda<float>(operand.channelScale));

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return operand;
}

Tensor dequantFp8ToHalf(const Fp8Operand &operand) {
  Tensor x = createCudaTensorHalf({operand.rows, operand.k});

  constexpr int kBlockSize = 256;
  int numPack = operand.rows * (operand.k / kElementPerThread);
  dim3 grid = getGrid1D(numPack, kBlockSize);

  dequantizeFp8Kernel<<<grid, kBlockSize>>>(
      reinterpret_cast<const uint8_t *>(getDataPtrCuda<Fp8E4M3>(operand.data)),
      getDataPtrCuda<float>(operand.channelScale),
      getDataPtrCuda<half>(x),
      operand.rows,
      operand.k);

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return x;
}

Tensor gemmFp8(const Tensor &A, const Fp8Operand &B) {
  CHECK(A.getDevice().getType() == Device::kCuda);
  CHECK(A.getDType() == DType::kFloat16);
  CHECK(A.isContiguous());
  CHECK(A.getDim() >= 2);
  CHECK(A.getShape(-1) == B.k);

  if (A.getDim() > 2) {
    std::vector<int> shape = A.getShape();
    Tensor D = gemmFp8(A.view({-1, A.getShape(-1)}), B);

    shape.back() = B.rows;
    return D.view(shape);
  }

  if (!isFp8GemmAvailable()) {
    throw lut::AbortedError("the fp8 mixed precision kernel needs an sm_80 or newer device.");
  }

  int m = A.getShape(0);
  int n = B.rows;
  int k = B.k;

  // The epilogue writes D 128 bits at a time along n, and the mainloop reads the weight 128 bits
  // at a time along k. m is free.
  CHECK(n % kAlignmentC == 0);
  CHECK(k % kAlignmentB == 0);

  Tensor D = createCudaTensorHalf({m, n});

  const half *ptrA = getDataPtrCuda<half>(A);
  const Fp8E4M3 *ptrB = getDataPtrCuda<Fp8E4M3>(B.data);
  const float *ptrScale = getDataPtrCuda<float>(B.channelScale);
  half *ptrD = getDataPtrCuda<half>(D);

  if (m <= kFlatTileMaxRow) {
    runFp8Gemm<FlatTile>(m, n, k, ptrA, ptrB, ptrScale, ptrD);
  } else {
    runFp8Gemm<SquareTile>(m, n, k, ptrA, ptrB, ptrScale, ptrD);
  }

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return D;
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
