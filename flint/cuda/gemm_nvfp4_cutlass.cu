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

// NVFP4 x NVFP4 on the SM120 block scaled tensor cores. CUTLASS supplies the mainloop; the two
// ends are ours. The prologue turns a half operand into the three things the MMA reads -- E2M1
// elements, E4M3 block scales in the interleaved atom layout, and the per-tensor scale that the
// MMA knows nothing about -- and the epilogue puts that per-tensor scale back, on the device, so
// that quantizing an activation never costs a round trip to the host.

#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "lutil/error.h"
#include "flint/cuda/common.h"
#include "flint/cuda/gemm_nvfp4_cutlass.h"

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

// The largest E2M1 magnitude and the largest E4M3 one. A block scale of blockAmax / 6 maps the
// block onto the whole FP4 range, and dividing that by amax / (6 * 448) -- the global scale --
// leaves a number in [0, 448], which is exactly what E4M3 holds.
constexpr float kFp4Max = 6.0f;
constexpr float kE4m3Max = 448.0f;
constexpr int kSfVecSize = 16;

// The scale factor atom covers 128 rows and 4 scale blocks, so an operand's scale array is padded
// out to those multiples whatever its own extent is.
constexpr int kSfBlockMN = 128;
constexpr int kSfBlockK = 4;

int divUp(int a, int b) {
  return (a - 1) / b + 1;
}

int roundUp(int a, int b) {
  return divUp(a, b) * b;
}

__forceinline__ __device__ float fp8e4m3ToFloat(uint8_t x) {
  return __half2float(__ushort_as_half(__nv_cvt_fp8_to_halfraw(x, __NV_E4M3).x));
}

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

/// @brief First half of the prologue: the tensor wide maximum, as one partial per block. Leaving
///        the partials for the second kernel to finish avoids having to zero an accumulator, and
///        so avoids a launch on a path where every launch is visible.
__global__ void amaxHalfKernel(
    const half2 *__restrict__ x,
    int numel2,
    float *__restrict__ partial) {
  float v = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel2; i += blockDim.x * gridDim.x) {
    half2 a = __habs2(x[i]);
    v = fmaxf(v, fmaxf(__half2float(a.x), __half2float(a.y)));
  }

  v = blockReduceMax(v);
  if (threadIdx.x == 0) partial[blockIdx.x] = v;
}

/// @brief Second half of the prologue. One thread owns one scale block: it needs no cross lane
///        reduction for the block maximum, and it writes one scale byte, which may be a padding
///        byte the atom layout has room for but the operand does not reach.
template<class LayoutSF>
__global__ void quantizeNvfp4Kernel(
    const half *__restrict__ x,
    int rows,
    int k,
    const float *__restrict__ partial,
    int numPartial,
    uint8_t *__restrict__ q,
    uint8_t *__restrict__ sf,
    LayoutSF layoutSF,
    float *__restrict__ globalScaleOut,
    int paddedRows,
    int paddedNumBlock) {
  __shared__ float sAmax;

  float v = 0.0f;
  for (int i = threadIdx.x; i < numPartial; i += blockDim.x) {
    v = fmaxf(v, partial[i]);
  }
  v = blockReduceMax(v);
  if (threadIdx.x == 0) sAmax = v;
  __syncthreads();

  float amax = sAmax;
  float globalScale = amax > 0.0f ? amax / (kFp4Max * kE4m3Max) : 1.0f;
  if (blockIdx.x == 0 && threadIdx.x == 0) *globalScaleOut = globalScale;

  int numBlock = k / kSfVecSize;
  int stride = blockDim.x * gridDim.x;

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < paddedRows * paddedNumBlock;
       idx += stride) {
    int row = idx / paddedNumBlock;
    int blockIdxK = idx % paddedNumBlock;

    uint8_t sfByte = 0;
    if (row < rows && blockIdxK < numBlock) {
      int offset = row * numBlock + blockIdxK;
      PackedOWORD<half2> po[2];
      po[0] = reinterpret_cast<const PackedOWORD<half2> *>(x)[offset * 2];
      po[1] = reinterpret_cast<const PackedOWORD<half2> *>(x)[offset * 2 + 1];

      half2 maxAbs2 = __habs2(po[0].v[0]);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          maxAbs2 = __hmax2(maxAbs2, __habs2(po[j].v[i]));
        }
      }
      float blockAmax = float(__hmax(maxAbs2.x, maxAbs2.y));

      // blockAmax / (kFp4Max * globalScale), which is in [0, 448] by the choice of globalScale.
      float scale = amax > 0.0f ? blockAmax * kE4m3Max / amax : 0.0f;
      sfByte = __nv_cvt_float_to_fp8(scale, __NV_SATFINITE, __NV_E4M3);

      // Quantize against the scale as E4M3 rounded it, not as it was computed: the rounded one is
      // all the mainloop will ever see.
      float dequantScale = fp8e4m3ToFloat(sfByte) * globalScale;
      float rcpScale = dequantScale > 0.0f ? 1.0f / dequantScale : 0.0f;

      union {
        uint8_t b[8];
        uint2 v;
      } out;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float2 f2 = __half22float2(po[j].v[i]);
          f2.x *= rcpScale;
          f2.y *= rcpScale;
          out.b[j * 4 + i] = __nv_cvt_float2_to_fp4x2(f2, __NV_E2M1, cudaRoundNearest);
        }
      }
      reinterpret_cast<uint2 *>(q)[offset] = out.v;
    }

    sf[layoutSF(row, blockIdxK * kSfVecSize, 0)] = sfByte;
  }
}

template<class LayoutSF>
__global__ void dequantizeNvfp4Kernel(
    const uint8_t *__restrict__ q,
    const uint8_t *__restrict__ sf,
    LayoutSF layoutSF,
    const float *__restrict__ globalScale,
    half *__restrict__ x,
    int rows,
    int k) {
  float gs = *globalScale;
  int numBlock = k / kSfVecSize;
  int stride = blockDim.x * gridDim.x;

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rows * numBlock; idx += stride) {
    int row = idx / numBlock;
    int blockIdxK = idx % numBlock;

    float blockScale = fp8e4m3ToFloat(sf[layoutSF(row, blockIdxK * kSfVecSize, 0)]);
    half2 scale = __float2half2_rn(blockScale * gs);
    uint2 vq = reinterpret_cast<const uint2 *>(q)[idx];
    const uint8_t *packed = reinterpret_cast<const uint8_t *>(&vq);

#pragma unroll
    for (int j = 0; j < kSfVecSize / 2; ++j) {
      half2 v = __nv_cvt_fp4x2_to_halfraw2(packed[j], __NV_E2M1);
      reinterpret_cast<half2 *>(x)[idx * (kSfVecSize / 2) + j] = __hmul2(v, scale);
    }
  }
}

/// @brief The epilogue's scalar. Both global scales were produced on the device by the prologue,
///        so their product is formed there as well and handed to the epilogue as a pointer.
__global__ void alphaKernel(
    const float *__restrict__ globalScaleA,
    const float *__restrict__ globalScaleB,
    float *__restrict__ alphaBeta) {
  alphaBeta[0] = *globalScaleA * *globalScaleB;
  alphaBeta[1] = 0.0f;
}


using namespace cute;

// The mainloop. A is row major and B column major, which for a K-major weight means B is the
// transposed weight laid out one output channel per row, the same as everywhere else here.
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int kAlignmentA = 32;
constexpr int kAlignmentB = 32;

// The epilogue writes half, and takes no source: D is alpha * accumulator, nothing else.
using ElementC = void;
using ElementD = cutlass::half_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int kAlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int kAlignmentC = kAlignmentD;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ThreadBlockShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementAccumulator,
    ElementC,
    LayoutCTag,
    kAlignmentC,
    ElementD,
    LayoutDTag,
    kAlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA,
    LayoutATag,
    kAlignmentA,
    ElementB,
    LayoutBTag,
    kAlignmentB,
    ElementAccumulator,
    ThreadBlockShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::
    GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

/// @brief cudaGetDeviceProperties costs hundreds of microseconds, which is more than a decode
///        step's GEMM, so the two device facts this path needs are read once.
int cachedSmCount() {
  static const int smCount = getCudaDeviceAttribute(cudaDevAttrMultiProcessorCount);
  return smCount;
}

/// @brief The layout of an operand's block scales. It depends on the operand's own extent only,
///        so a weight can be quantized once without knowing what it will be multiplied by.
auto makeLayoutSF(int rows, int k) {
  return Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(rows, 1, k, 1));
}

}  // namespace

bool isNvfp4GemmAvailable() {
  static const bool available = getCudaArch() == 120;
  return available;
}

Nvfp4Operand quantizeNvfp4(const Tensor &x) {
  CHECK(x.getDevice().getType() == Device::kCuda);
  CHECK(x.getDType() == DType::kFloat16);
  CHECK(x.getDim() == 2);
  LL_CHECK_CONTIGUOUS(x);

  int rows = x.getShape(0);
  int k = x.getShape(1);

  // 32 elements is the tensor core's access granularity, and it keeps every row of the packed
  // data 16 byte aligned.
  CHECK(k % (2 * kSfVecSize) == 0);

  int64_t numel64 = x.getNumEl();
  CHECK(numel64 < std::numeric_limits<int32_t>::max());

  auto layoutSF = makeLayoutSF(rows, k);
  int paddedRows = roundUp(rows, kSfBlockMN);
  int paddedNumBlock = roundUp(divUp(k, kSfVecSize), kSfBlockK);
  CHECK(static_cast<int>(cute::size(cute::filter_zeros(layoutSF))) == paddedRows * paddedNumBlock);

  Nvfp4Operand operand;
  operand.rows = rows;
  operand.k = k;
  operand.data = createCudaTensorFp4x2({rows, k / 2});
  operand.blockScale = createCudaTensorUInt8({paddedRows * paddedNumBlock});
  operand.globalScale = createCudaTensorFloat({1});

  constexpr int kBlockSize = 256;
  int numPartial = std::min(divUp(static_cast<int>(numel64 / 2), kBlockSize), 256);

  lut::c_ptr<float> partial = llynCudaAlloc<float>(numPartial);
  amaxHalfKernel<<<numPartial, kBlockSize>>>(
      reinterpret_cast<const half2 *>(getDataPtrCuda<half>(x)),
      static_cast<int>(numel64 / 2),
      partial.get());

  dim3 grid = getGrid1D(paddedRows * paddedNumBlock, kBlockSize);
  quantizeNvfp4Kernel<<<grid, kBlockSize>>>(
      getDataPtrCuda<half>(x),
      rows,
      k,
      partial.get(),
      numPartial,
      reinterpret_cast<uint8_t *>(getDataPtrCuda<Fp4E2M0x2>(operand.data)),
      reinterpret_cast<uint8_t *>(getDataPtrCuda<UInt8>(operand.blockScale)),
      layoutSF,
      getDataPtrCuda<float>(operand.globalScale),
      paddedRows,
      paddedNumBlock);

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return operand;
}

Tensor dequantNvfp4ToHalf(const Nvfp4Operand &operand) {
  Tensor x = createCudaTensorHalf({operand.rows, operand.k});

  constexpr int kBlockSize = 256;
  int numBlock = operand.rows * (operand.k / kSfVecSize);
  dim3 grid = getGrid1D(numBlock, kBlockSize);

  dequantizeNvfp4Kernel<<<grid, kBlockSize>>>(
      reinterpret_cast<const uint8_t *>(getDataPtrCuda<Fp4E2M0x2>(operand.data)),
      reinterpret_cast<const uint8_t *>(getDataPtrCuda<UInt8>(operand.blockScale)),
      makeLayoutSF(operand.rows, operand.k),
      getDataPtrCuda<float>(operand.globalScale),
      getDataPtrCuda<half>(x),
      operand.rows,
      operand.k);

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return x;
}

Nvfp4Operand makeNvfp4Operand(
    const Tensor &data,
    const Tensor &blockScale,
    const Tensor &globalScale) {
  // Reached from outside the library, so what a caller could get wrong is reported as a bad
  // argument rather than as the aborted-operation a failed CHECK reads as.
  if (data.getDType() != DType::kFp4E2M0x2 || data.getDim() != 2) {
    throw lut::InvalidArgError("nvfp4 operand: data is not <fp4>(rows, k / 2)");
  }
  if (blockScale.getDType() != DType::kUInt8 || blockScale.getDim() != 1) {
    throw lut::InvalidArgError("nvfp4 operand: block scale is not <uint8>(n)");
  }
  if (globalScale.getDType() != DType::kFloat || globalScale.getNumEl() != 1) {
    throw lut::InvalidArgError("nvfp4 operand: global scale is not a single <float>");
  }

  Nvfp4Operand operand;
  operand.data = data;
  operand.blockScale = blockScale;
  operand.globalScale = globalScale;
  operand.rows = data.getShape(0);
  operand.k = data.getShape(1) * 2;

  int paddedRows = roundUp(operand.rows, kSfBlockMN);
  int paddedNumBlock = roundUp(divUp(operand.k, kSfVecSize), kSfBlockK);
  if (blockScale.getShape(0) != paddedRows * paddedNumBlock) {
    throw lut::InvalidArgError("nvfp4 operand: block scale does not match the data it scales");
  }

  return operand;
}

Tensor nvfp4Alpha(const Nvfp4Operand &A, const Nvfp4Operand &B) {
  Tensor alpha = createCudaTensorFloat({2});
  alphaKernel<<<1, 1>>>(
      getDataPtrCuda<float>(A.globalScale),
      getDataPtrCuda<float>(B.globalScale),
      getDataPtrCuda<float>(alpha));
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return alpha;
}

Tensor gemmNvfp4(const Tensor &A, const Nvfp4Operand &B) {
  CHECK(A.getDType() == DType::kFloat16);

  if (A.getDim() == 2) {
    return gemmNvfp4(quantizeNvfp4(A), B);
  }

  CHECK(A.getDim() > 2 && A.isContiguous());
  std::vector<int> shape = A.getShape();
  Tensor xC = gemmNvfp4(quantizeNvfp4(A.view({-1, A.getShape(-1)})), B);

  shape.back() = B.rows;
  return xC.view(shape);
}

Tensor gemmNvfp4(const Nvfp4Operand &A, const Nvfp4Operand &B) {
  CHECK(A.k == B.k);
  if (!isNvfp4GemmAvailable()) {
    throw lut::AbortedError("the NVFP4 block scaled kernel needs an sm_120 device.");
  }

  int m = A.rows;
  int n = B.rows;
  int k = A.k;

  // The epilogue writes D a 128 bit vector at a time along its innermost axis, which is n. m has
  // no such constraint, and k was already fixed at a multiple of 32 by the prologue.
  CHECK(n % (128 / 16) == 0);

  Tensor D = createCudaTensorHalf({m, n});

  Tensor alpha = nvfp4Alpha(A, B);

  StrideA strideA = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB strideB = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  StrideD strideD = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {reinterpret_cast<const cutlass::float_e2m1_t *>(getDataPtrCuda<Fp4E2M0x2>(A.data)),
       strideA,
       reinterpret_cast<const cutlass::float_e2m1_t *>(getDataPtrCuda<Fp4E2M0x2>(B.data)),
       strideB,
       reinterpret_cast<const cutlass::float_ue4m3_t *>(getDataPtrCuda<UInt8>(A.blockScale)),
       makeLayoutSF(m, k),
       reinterpret_cast<const cutlass::float_ue4m3_t *>(getDataPtrCuda<UInt8>(B.blockScale)),
       makeLayoutSF(n, k)},
      {{}, nullptr, StrideC{}, reinterpret_cast<cutlass::half_t *>(getDataPtrCuda<half>(D)),
       strideD}};

  arguments.epilogue.thread.alpha_ptr = getDataPtrCuda<float>(alpha);
  arguments.epilogue.thread.beta = 0.0f;

  // Left at zero, the kernel looks the SM count up on every launch, and that lookup alone is
  // longer than the GEMM it is preparing.
  arguments.hw_info.device_id = 0;
  arguments.hw_info.sm_count = cachedSmCount();

  Gemm gemm;
  size_t workspaceSize = Gemm::get_workspace_size(arguments);
  lut::c_ptr<uint8_t> workspace;
  if (workspaceSize) workspace = llynCudaAlloc<uint8_t>(workspaceSize);

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm.run());

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return D;
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
