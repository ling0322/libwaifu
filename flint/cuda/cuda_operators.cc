// The MIT License (MIT)
//
// Copyright (c) 2023-2024 Xiaoyang Chen
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

#include "flint/cuda/cuda_operators.h"

#include <math.h>

#include "flint/cuda/arange.h"
#include "flint/cuda/binary.h"
#include "flint/cuda/binary_scalar.h"
#include "flint/cpu/all_close.h"
#include "flint/cuda/cast.h"
#include "flint/cuda/causal_mask.h"
#include "flint/cuda/conv1d.h"
#include "flint/cuda/conv2d.h"
#include "flint/cuda/cuda_host_tensor_data.h"
#include "flint/cuda/norm.h"
#include "flint/cuda/upsample.h"
#include "flint/cuda/copy.h"
#include "flint/cuda/fill.h"
#include "flint/cuda/gated_delta_net.h"
#include "flint/cuda/flash_attn.h"
#include "flint/cuda/lookup.h"
#include "flint/cuda/matmul.h"
#include "flint/cuda/print.h"
#include "flint/cuda/rand.h"
#include "flint/cuda/reduce.h"
#include "flint/cuda/scan.h"
#include "flint/cuda/repetition_penalty.h"
#include "flint/cuda/rotary_embedding.h"
#include "flint/cuda/sampling.h"
#include "flint/cuda/softmax.h"
#include "flint/cuda/store_kv_cache.h"
#include "flint/cuda/glu.h"
#include "flint/cuda/to_device.h"
#include "flint/cuda/unary.h"
#include "flint/functional.h"
#include "lutil/error.h"

namespace fl {
namespace op {
namespace cuda {

bool CudaOperators::isAvailable() {
  return getCudaDeviceCount() > 0;
}

std::shared_ptr<Operators> CudaOperators::create(int options) {
  std::shared_ptr<CudaOperators> op{new CudaOperators()};
  if (!isAvailable()) {
    LOG(INFO) << "No CUDA device available.";
    return nullptr;
  }

#ifdef LIBWAIFU_CUDA_MALLOC_ASYNC_ENABLED
  int memoryPoolsSupported = 0;
  LL_CHECK_CUDA_STATUS(
      cudaDeviceGetAttribute(&memoryPoolsSupported, cudaDevAttrMemoryPoolsSupported, 0));
  CHECK(memoryPoolsSupported) << "CUDA device does not support stream-ordered memory allocation";

  cudaMemPool_t memoryPool;
  uint64_t releaseThreshold = UINT64_MAX;
  LL_CHECK_CUDA_STATUS(cudaDeviceGetDefaultMemPool(&memoryPool, 0));
  LL_CHECK_CUDA_STATUS(
      cudaMemPoolSetAttribute(memoryPool, cudaMemPoolAttrReleaseThreshold, &releaseThreshold));
#endif

  if (options & OPT_CUBLAS_GEMM) {
    LOG(INFO) << "Create CUDA operators with CUBLAS only";
    op->_matmul = MatMul::createCublas();
  } else if (options & OPT_CUTLASS_GEMM) {
    LOG(INFO) << "Create CUDA operators with CUTLASS only";
    op->_matmul = MatMul::createCutlass();
  } else {
    op->_matmul = MatMul::create();
  }
  op->_rand = Rand::newRand();

  LOG(INFO) << "cuda numDevices = " << getCudaDeviceCount();
  LOG(INFO) << "cuda:0 maxThreadsPerMultiProcessor = "
            << getCudaDeviceAttribute(cudaDevAttrMaxThreadsPerMultiProcessor);
  LOG(INFO) << "cuda:0 multiProcessorCount = "
            << getCudaDeviceAttribute(cudaDevAttrMultiProcessorCount);

  return op;
}

namespace {

/// `input` itself if it is packed, otherwise a packed copy of it kept alive in `holder`. A
/// temporary of this backend's own, for the kernels that walk their input as a flat array.
TensorView contiguousCuda(const TensorView &input, Tensor &holder) {
  if (input.isContiguous()) return input;

  holder = F::emptyLike(input);
  op::cuda::copy(input, holder);
  return holder;
}

}  // namespace

void CudaOperators::fill(TensorView input, float value) {
  op::cuda::fill(input, value);
}

void CudaOperators::square(TensorView input, TensorView out) {
  op::cuda::applyUnaryOp(op::cuda::UnaryOp::SQUARE, input, out);
}

void CudaOperators::max(TensorView input, TensorView out) {
  op::cuda::reduceLastDim(input, MapReduceType::MAX, out);
}

void CudaOperators::min(TensorView input, TensorView out) {
  op::cuda::reduceLastDim(input, MapReduceType::MIN, out);
}

void CudaOperators::divTensor(TensorView input, TensorView other, TensorView out) {
  op::cuda::applyBinaryOp(BinaryOp::DIV, input, other, out);
}

void CudaOperators::neg(TensorView input, TensorView out) {
  op::cuda::applyUnaryOp(op::cuda::UnaryOp::NEG, input, out);
}

void CudaOperators::abs(TensorView input, TensorView out) {
  op::cuda::applyUnaryOp(op::cuda::UnaryOp::ABS, input, out);
}

void CudaOperators::exp(TensorView input, TensorView out) {
  op::cuda::applyUnaryOp(op::cuda::UnaryOp::EXP, input, out);
}

void CudaOperators::log(TensorView input, TensorView out) {
  op::cuda::applyUnaryOp(op::cuda::UnaryOp::LOG, input, out);
}

void CudaOperators::round(TensorView input, TensorView out) {
  op::cuda::applyUnaryOp(op::cuda::UnaryOp::ROUND, input, out);
}

void CudaOperators::sqrt(TensorView input, TensorView out) {
  op::cuda::applyUnaryOp(op::cuda::UnaryOp::SQRT, input, out);
}

void CudaOperators::rsqrt(TensorView input, TensorView out) {
  op::cuda::applyUnaryOp(op::cuda::UnaryOp::RSQRT, input, out);
}

void CudaOperators::sigmoid(TensorView input, TensorView out) {
  op::cuda::applyUnaryOp(op::cuda::UnaryOp::SIGMOID, input, out);
}

void CudaOperators::tanh(TensorView input, TensorView out) {
  op::cuda::applyUnaryOp(op::cuda::UnaryOp::TANH, input, out);
}

void CudaOperators::relu(TensorView input, TensorView out) {
  op::cuda::applyUnaryOp(op::cuda::UnaryOp::RELU, input, out);
}

void CudaOperators::gelu(TensorView input, TensorView out) {
  op::cuda::applyUnaryOp(op::cuda::UnaryOp::GELU, input, out);
}

void CudaOperators::silu(TensorView input, TensorView out) {
  op::cuda::applyUnaryOp(op::cuda::UnaryOp::SILU, input, out);
}

void CudaOperators::sin(TensorView input, TensorView out) {
  op::cuda::applyUnaryOp(op::cuda::UnaryOp::SIN, input, out);
}

void CudaOperators::cos(TensorView input, TensorView out) {
  op::cuda::applyUnaryOp(op::cuda::UnaryOp::COS, input, out);
}

void CudaOperators::quickGelu(TensorView input, TensorView out) {
  op::cuda::applyUnaryOp(op::cuda::UnaryOp::QUICK_GELU, input, out);
}

bool CudaOperators::allClose(TensorView A, TensorView B, float rtol, float atol) {
  // The comparison itself is a host-side reduction over both tensors, so bring them over and let
  // the CPU backend do it rather than growing a kernel that would only be used by tests. The cast
  // to float is what the CPU comparison expects; half tensors go through it on the device, where
  // the copy is cheaper.
  auto toHostFloat = [](const TensorView &x) {
    Tensor packed;
    TensorView source = contiguousCuda(x, packed);
    Tensor wide = F::empty(Device::getCpu(), x.getShape(), DType::kFloat);
    if (x.getDType() == DType::kFloat) {
      op::cuda::transfer(source, wide);
    } else {
      op::cuda::transfer(op::cuda::castTo(source, DType::kFloat), wide);
    }
    return wide;
  };

  return op::cpu::allClose(toHostFloat(A), toHostFloat(B), rtol, atol);
}

bool CudaOperators::all(TensorView A) {
  Tensor packed;
  TensorView source = contiguousCuda(A, packed);
  return op::cuda::elemBool(op::cuda::reduceAll(source, DType::kBool, MapReduceType::ALL));
}

void CudaOperators::sum(TensorView input, int dim, TensorView out) {
  int ndim = input.getDim();
  if (dim < 0) dim += ndim;
  CHECK(dim >= 0 && dim < ndim);
  CHECK(out.getDType() == input.getDType() && out.isContiguous());

  // The summed dimension has to be the last for the row reduction. (.., D, ..) is viewed as
  // (outer, D, inner) and turned to (outer, inner, D), so that every other dimension keeps its
  // place and the rows come out in `out`'s order.
  TensorView source = input;
  Tensor packed, moved;
  if (dim != ndim - 1) {
    std::vector<int> shape = input.getShape();
    int outer = 1, inner = 1;
    for (int d = 0; d < dim; ++d) outer *= shape[d];
    for (int d = dim + 1; d < ndim; ++d) inner *= shape[d];

    TensorView grouped = contiguousCuda(input, packed).view({outer, shape[dim], inner});
    moved = F::emptyLike(grouped.transpose(1, 2));
    op::cuda::copy(grouped.transpose(1, 2), moved);
    source = moved;
  }

  // Half is summed in float and narrowed once at the end, which is where the precision is wanted.
  if (input.getDType() == DType::kFloat16) {
    Tensor wide = F::empty(out.getDevice(), out.getShape(), DType::kFloat);
    op::cuda::reduceLastDim(source, MapReduceType::SUM, wide);
    op::cuda::cast(wide, out);
  } else {
    op::cuda::reduceLastDim(source, MapReduceType::SUM, out);
  }
}

void CudaOperators::cumsum(TensorView input, int dim, TensorView out) {
  int ndim = input.getDim();
  if (dim < 0) dim += ndim;
  CHECK(dim >= 0 && dim < ndim);

  Tensor packed;
  if (dim == ndim - 1) {
    op::cuda::cumsumLastDim(contiguousCuda(input, packed), out);
    return;
  }

  // Any other dimension is turned to the last, scanned, and turned back as it is written out.
  TensorView turned = input.transpose(dim, ndim - 1);
  Tensor transposed = F::emptyLike(turned);
  op::cuda::copy(turned, transposed);
  Tensor scanned = F::emptyLike(transposed);
  op::cuda::cumsumLastDim(transposed, scanned);
  op::cuda::copy(scanned.transpose(dim, ndim - 1), out);
}

void CudaOperators::lookup(TensorView table, TensorView indices, TensorView out) {
  cuda::lookup(table, indices, out);
}

void CudaOperators::rotaryEmbedding(
    TensorView positions,
    TensorView query,
    TensorView key,
    TensorView rotaryCache) {
  cuda::rotaryEmbedding(positions, query, key, rotaryCache);
}

void CudaOperators::matmul(TensorView A, TensorView B, TensorView out) {
  _matmul->apply(A, B, out);
}

void CudaOperators::layerNorm(
    TensorView input,
    TensorView weight,
    TensorView bias,
    float eps,
    TensorView out) {
  cuda::layerNorm(input, weight, bias, eps, out);
}

void CudaOperators::groupNorm(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int groups,
    float eps,
    TensorView out) {
  cuda::groupNorm(input, weight, bias, groups, eps, out);
}

void CudaOperators::upsampleNearest2d(TensorView input, int scale, TensorView out) {
  cuda::upsampleNearest2d(input, scale, out);
}

void CudaOperators::upsampleNearest1d(TensorView input, TensorView out) {
  cuda::upsampleNearest1d(input, out);
}

void CudaOperators::geglu(TensorView input, TensorView out) {
  cuda::geglu(input, out);
}

void CudaOperators::conv2d(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    TensorView out) {
  cuda::conv2d(input, weight, bias, {stride, padding, dilation, groups}, out);
}

void CudaOperators::conv1d(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    TensorView out) {
  cuda::conv1d(input, weight, bias, {stride, padding, dilation, groups}, out);
}

void CudaOperators::gatedDeltaNetPrefill(
    TensorView q,
    TensorView k,
    TensorView v,
    TensorView g,
    TensorView beta,
    TensorView cuSeqlens,
    TensorView stateSlots,
    TensorView state,
    TensorView out) {
  op::cuda::gatedDeltaNetPrefill(q, k, v, g, beta, cuSeqlens, stateSlots, state, out);
}

void CudaOperators::mul(TensorView input, float other, TensorView out) {
  op::cuda::applyBinaryScalarOp(BinaryScalarOp::MUL, input, other, out);
}

void CudaOperators::div(TensorView input, float other, TensorView out) {
  op::cuda::applyBinaryScalarOp(BinaryScalarOp::DIV, input, other, out);
}

void CudaOperators::mod(TensorView input, LongType other, TensorView out) {
  op::cuda::applyBinaryScalarOpLong(BinaryScalarOp::MOD, input, other, out);
}

void CudaOperators::mul(TensorView input, TensorView other, TensorView out) {
  op::cuda::applyBinaryOp(BinaryOp::MUL, input, other, out);
}

void CudaOperators::softmax(TensorView input, TensorView out) {
  op::cuda::softmax(input, out);
}

void CudaOperators::add(TensorView input, TensorView other, TensorView out) {
  op::cuda::applyBinaryOp(BinaryOp::ADD, input, other, out);
}

void CudaOperators::sub(TensorView input, TensorView other, TensorView out) {
  op::cuda::applyBinaryOp(BinaryOp::SUB, input, other, out);
}

void CudaOperators::repetitionPenalty(TensorView logits, TensorView history, float weight) {
  CHECK(history.getDType() == DType::kLong);

  op::cuda::repetitionPenalty(logits, history, weight);
}

void CudaOperators::rmsNorm(TensorView input, TensorView weight, float eps, TensorView out) {
  op::cuda::rmsNorm(input, weight, eps, out);
}

void CudaOperators::causalMask(TensorView out) {
  op::cuda::causalMask(out);
}

void CudaOperators::attention(
    TensorView q,
    TensorView k,
    TensorView v,
    bool causal,
    TensorView out) {
#ifdef LIBWAIFU_FLASH_ATTN_ENABLED
  if (op::cuda::flashAttention(q, k, v, causal, out)) return;
#endif

  F::composedAttention(this, q, k, v, causal, out);
}

void CudaOperators::storeKVCache(
    TensorView k,
    TensorView v,
    TensorView keyCache,
    TensorView valueCache,
    TensorView slotMapping) {
  op::cuda::storeKVCache(k, v, keyCache, valueCache, slotMapping);
}

void CudaOperators::pagedAttention(
    TensorView q,
    TensorView keyCache,
    TensorView valueCache,
    TensorView blockTable,
    TensorView cuSeqlensQ,
    TensorView seqlensK,
    int maxQLen,
    int maxKLen,
    bool causal,
    TensorView out) {
#ifdef LIBWAIFU_FLASH_ATTN_ENABLED
  bool done = op::cuda::pagedFlashAttention(
      q,
      keyCache,
      valueCache,
      blockTable,
      cuSeqlensQ,
      seqlensK,
      maxQLen,
      maxKLen,
      causal,
      out);
  if (done) return;

  // Unlike attention above, there is no portable paged attention to fall back to. Throw the
  // way conv2d.cc does rather than aborting the process.
  throw lut::AbortedError("pagedAttention: FlashAttention does not support these inputs");
#else
  throw lut::AbortedError("this build has no paged attention (needs WITH_FLASH_ATTN=ON)");
#endif
}

MemorySnapshot CudaOperators::captureMemorySnapshot() {
  size_t freeMemory = 0;
  size_t totalMemory = 0;
  LL_CHECK_CUDA_STATUS(cudaMemGetInfo(&freeMemory, &totalMemory));

  uint64_t allocatedMemory = 0;
  uint64_t peakAllocatedMemory = 0;
#ifdef LIBWAIFU_CUDA_MALLOC_ASYNC_ENABLED
  cudaMemPool_t memoryPool;
  LL_CHECK_CUDA_STATUS(cudaDeviceGetDefaultMemPool(&memoryPool, 0));
  LL_CHECK_CUDA_STATUS(
      cudaMemPoolGetAttribute(memoryPool, cudaMemPoolAttrUsedMemCurrent, &allocatedMemory));
  LL_CHECK_CUDA_STATUS(
      cudaMemPoolGetAttribute(memoryPool, cudaMemPoolAttrUsedMemHigh, &peakAllocatedMemory));
#endif

  return MemorySnapshot(
      static_cast<int64_t>(totalMemory),
      static_cast<int64_t>(freeMemory),
      static_cast<int64_t>(allocatedMemory),
      static_cast<int64_t>(peakAllocatedMemory));
}

void CudaOperators::resetPeakMemoryStats() {
#ifdef LIBWAIFU_CUDA_MALLOC_ASYNC_ENABLED
  // the high watermark of a memory pool can only be reset to zero.
  cudaMemPool_t memoryPool;
  uint64_t zero = 0;
  LL_CHECK_CUDA_STATUS(cudaDeviceGetDefaultMemPool(&memoryPool, 0));
  LL_CHECK_CUDA_STATUS(cudaMemPoolSetAttribute(memoryPool, cudaMemPoolAttrUsedMemHigh, &zero));
#endif
}

void CudaOperators::releaseUnusedMemory() {
#ifdef LIBWAIFU_CUDA_MALLOC_ASYNC_ENABLED
  // The pool is told at startup never to give anything back on its own (its release threshold is
  // UINT64_MAX), because a run that had to ask the driver for the blocks the last run just freed
  // would pay for the same memory twice. So a tensor that is gone is memory this process still
  // holds, and `nvidia-smi` still counts it -- and this is the one call that ends that.
  //
  // Synchronized first because a free is stream-ordered: `cudaFreeAsync` hands the block back to
  // the pool at the point the stream reaches it, not at the point it was called, and a trim that
  // ran ahead of the stream would walk past blocks that are about to be free and hand back
  // whatever it happened to find. There is nothing to wait for here anyway -- what this is called
  // after is the end of a run.
  LL_CHECK_CUDA_STATUS(cudaDeviceSynchronize());

  cudaMemPool_t memoryPool;
  LL_CHECK_CUDA_STATUS(cudaDeviceGetDefaultMemPool(&memoryPool, 0));
  LL_CHECK_CUDA_STATUS(cudaMemPoolTrimTo(memoryPool, 0));
#endif
}

void CudaOperators::copy(TensorView src, TensorView dest) {
  CHECK(src.getDevice().getType() == Device::kCuda);
  CHECK(dest.getDevice().getType() == Device::kCuda);
  CHECK(src.getDType() == dest.getDType());
  src.throwIfInvalidShape(dest.getShape(), "CudaOperators::copy");

  if (src.isContiguous() && dest.isContiguous()) {
    copyContig(src, dest);
  } else {
    op::cuda::copy(src, dest);
  }
}

void CudaOperators::transfer(TensorView src, TensorView dest) {
  op::cuda::transfer(src, dest);
}

void CudaOperators::print(TensorView tensor) {
  op::cuda::print(tensor);
}

void CudaOperators::swiglu(TensorView input, TensorView out) {
  op::cuda::swiglu(input, out);
}

void CudaOperators::sample(
    TensorView logits,
    TensorView temperatures,
    TensorView topKs,
    TensorView topPs,
    TensorView out) {
  CHECK(logits.getDim() == 2);
  int rows = logits.getShape(0);
  Tensor uniformNoise = F::empty(Device::getCuda(), {rows}, DType::kFloat);
  _rand->rand(uniformNoise);
  op::cuda::sample(logits, uniformNoise, temperatures, topKs, topPs, out);
}

void CudaOperators::cast(TensorView input, TensorView out) {
  CHECK(input.getDevice().getType() == Device::kCuda);
  Tensor packed;
  cuda::cast(contiguousCuda(input, packed), out);
}

DType CudaOperators::getDefaultFloatType() {
  return DType::kFloat16;
}

void CudaOperators::randNormal(TensorView out) {
  CHECK(out.getDevice().getType() == Device::kCuda);
  _rand->randNormal(out);
}

void CudaOperators::rand(TensorView out) {
  CHECK(out.getDevice().getType() == Device::kCuda);
  if (out.getDType() == DType::kFloat) {
    _rand->rand(out);
    return;
  }

  // The generator draws float32; any other type is that, converted.
  Tensor drawn = F::empty(out.getDevice(), out.getShape(), DType::kFloat);
  _rand->rand(drawn);
  cuda::cast(drawn, out);
}

void CudaOperators::arangeLong(LongType begin, LongType step, TensorView out) {
  cuda::arangeLong(begin, step, out);
}

float CudaOperators::elem(TensorView tensor) {
  return op::cuda::elem(tensor);
}

bool CudaOperators::elemBool(TensorView tensor) {
  return op::cuda::elemBool(tensor);
}

void CudaOperators::eq(TensorView input, TensorView other, TensorView out) {
  op::cuda::applyBinaryOp(BinaryOp::EQUAL, input, other, out);
}

void CudaOperators::manualSeed(uint64_t seed) {
  _rand->setSeed(seed);
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
