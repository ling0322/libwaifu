// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
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

#include "flint/operators.h"

#include <atomic>
#include <mutex>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "flint/cpu/cpu_operators.h"
#include "flint/cpu/kernel/interface.h"
#ifdef LIBWAIFU_CUDA_ENABLED
#include "flint/cuda/cuda_operators.h"
#endif
#include "flint/functional.h"
#ifdef LIBWAIFU_MLX_ENABLED
#include "flint/metal/metal_operators.h"
#endif
#ifdef LIBWAIFU_VULKAN_ENABLED
#include "flint/vulkan/vulkan_operators.h"
#endif

namespace fl {

// What a device that has no kernel for something says when asked for it: a NotImplementedError
// naming the operator, which the C API hands back as an error rather than letting it abort the
// process. A missing kernel is a fact about the device and not a broken invariant, so whoever is
// driving it -- a server running one job of many -- gets to fail that job and carry on.
#define NO_KERNEL() THROW(NotImplemented, std::string(__func__) + " has no kernel on this device")

void Operators::arangeLong(LongType begin, LongType step, TensorView out) {
  NO_KERNEL();
}

void Operators::lookup(TensorView table, TensorView indices, TensorView out) {
  NO_KERNEL();
}

void Operators::rotaryEmbedding(
    TensorView positions,
    TensorView query,
    TensorView key,
    TensorView rotaryCache) {
  NO_KERNEL();
}

void Operators::rmsNorm(TensorView input, TensorView weight, float eps, TensorView out) {
  NO_KERNEL();
}

void Operators::layerNorm(
    TensorView input,
    TensorView weight,
    TensorView bias,
    float eps,
    TensorView out) {
  THROW(NotImplemented, "layerNorm is only available on the CUDA device");
}

void Operators::groupNorm(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int groups,
    float eps,
    TensorView out) {
  THROW(NotImplemented, "groupNorm is only available on the CUDA device");
}

void Operators::upsampleNearest2d(TensorView input, int scale, TensorView out) {
  THROW(NotImplemented, "upsampleNearest2d is only available on the CUDA device");
}

void Operators::upsampleNearest1d(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::geglu(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::matmul(TensorView A, TensorView B, TensorView out) {
  NO_KERNEL();
}

void Operators::conv2d(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    TensorView out) {
  THROW(NotImplemented, "conv2d is only available on the CUDA device, in a build with cuDNN");
}

void Operators::conv1d(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int dilation,
    int groups,
    TensorView out) {
  THROW(
      NotImplemented,
      "conv1d is not available on this device; the CPU, CUDA and Metal all have it");
}

void Operators::convTranspose1d(
    TensorView input,
    TensorView weight,
    TensorView bias,
    int stride,
    int padding,
    int outputPadding,
    int groups,
    TensorView out) {
  THROW(
      NotImplemented,
      "convTranspose1d: no device has this kernel; waifu::audio::conv_transpose1d composes one");
}

void Operators::snake(
    TensorView input,
    TensorView alpha,
    TensorView beta,
    float eps,
    TensorView out) {
  THROW(NotImplemented, "snake: no device has this kernel; waifu::audio::snake composes one");
}

void Operators::stft(
    TensorView input,
    TensorView window,
    int nFft,
    int hop,
    bool centered,
    TensorView out) {
  THROW(NotImplemented, "stft: no device has this kernel; waifu::audio::stft composes one");
}

void Operators::istft(
    TensorView spectrum,
    TensorView window,
    int nFft,
    int hop,
    bool centered,
    TensorView out) {
  THROW(NotImplemented, "istft: no device has this kernel; waifu::audio::istft composes one");
}

void Operators::mul(TensorView input, float other, TensorView out) {
  NO_KERNEL();
}

void Operators::div(TensorView input, float other, TensorView out) {
  NO_KERNEL();
}

void Operators::mod(TensorView input, LongType other, TensorView out) {
  NO_KERNEL();
}

void Operators::mul(TensorView input, TensorView other, TensorView out) {
  NO_KERNEL();
}

void Operators::softmax(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::attention(TensorView q, TensorView k, TensorView v, bool causal, TensorView out) {
  F::composedAttention(this, q, k, v, causal, out);
}

void Operators::pagedAttention(
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
  NO_KERNEL();
}

void Operators::storeKVCache(
    TensorView k,
    TensorView v,
    TensorView keyCache,
    TensorView valueCache,
    TensorView slotMapping) {
  NO_KERNEL();
}

void Operators::gatedDeltaNetPrefill(
    TensorView q,
    TensorView k,
    TensorView v,
    TensorView g,
    TensorView beta,
    TensorView cuSeqlens,
    TensorView stateSlots,
    TensorView state,
    TensorView out) {
  NO_KERNEL();
}

void Operators::sample(
    TensorView logits,
    TensorView temperatures,
    TensorView topKs,
    TensorView topPs,
    TensorView out) {
  NO_KERNEL();
}

void Operators::add(TensorView input, TensorView other, TensorView out) {
  NO_KERNEL();
}

void Operators::sub(TensorView input, TensorView other, TensorView out) {
  NO_KERNEL();
}

void Operators::sum(TensorView input, int dim, TensorView out) {
  NO_KERNEL();
}

void Operators::cumsum(TensorView input, int dim, TensorView out) {
  NO_KERNEL();
}

void Operators::max(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::eq(TensorView input, TensorView other, TensorView out) {
  NO_KERNEL();
}

void Operators::square(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::min(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::divTensor(TensorView input, TensorView other, TensorView out) {
  NO_KERNEL();
}

void Operators::neg(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::abs(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::exp(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::log(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::round(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::sqrt(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::rsqrt(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::sigmoid(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::tanh(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::relu(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::gelu(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::silu(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::sin(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::cos(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::quickGelu(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::fill(TensorView input, float value) {
  NO_KERNEL();
}

bool Operators::all(TensorView A) {
  NO_KERNEL();
}

bool Operators::allClose(TensorView A, TensorView B, float rtol, float atol) {
  NO_KERNEL();
}

void Operators::print(TensorView tensor) {
  NO_KERNEL();
}

void Operators::causalMask(TensorView out) {
  NO_KERNEL();
}

void Operators::copy(TensorView src, TensorView dest) {
  NO_KERNEL();
}

void Operators::swiglu(TensorView A, TensorView out) {
  NO_KERNEL();
}

void Operators::transfer(TensorView src, TensorView dest) {
  NO_KERNEL();
}

float Operators::elem(TensorView tensor) {
  NO_KERNEL();
}

bool Operators::elemBool(TensorView tensor) {
  NO_KERNEL();
}

void Operators::repetitionPenalty(TensorView logits, TensorView history, float weight) {
  NO_KERNEL();
}

void Operators::cast(TensorView input, TensorView out) {
  NO_KERNEL();
}

void Operators::rand(TensorView out) {
  NO_KERNEL();
}

void Operators::randNormal(TensorView out) {
  NO_KERNEL();
}

void Operators::manualSeed(uint64_t seed) {
  NO_KERNEL();
}

MemorySnapshot Operators::captureMemorySnapshot() {
  NO_KERNEL();
}

void Operators::resetPeakMemoryStats() {
  NO_KERNEL();
}

// Not NO_KERNEL(): a device whose allocator holds nothing back has nothing to hand over, and has
// answered this question by already having done it.
void Operators::releaseUnusedMemory() {
}

DType Operators::getDefaultFloatType() {
  NO_KERNEL();
}

void Operators::synchronize() {
}

std::shared_ptr<Operators> gOperatorsForDevice[Device::NumDeviceType] = {
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr};

static std::atomic<bool> gInitialized{false};

void initOperators() {
  op::cpu::kernel::init();

#ifdef _OPENMP
  LOG(INFO) << "OMP max_threads = " << omp_get_max_threads();
#endif

  if (!gInitialized.exchange(true)) {
    CHECK(!gOperatorsForDevice[Device::kCpu]);
    gOperatorsForDevice[Device::kCpu] = std::make_shared<op::cpu::CPUOperators>();
#ifdef LIBWAIFU_CUDA_ENABLED
    CHECK(!gOperatorsForDevice[Device::kCuda]);
    gOperatorsForDevice[Device::kCuda] = op::cuda::CudaOperators::create();
#endif
#ifdef LIBWAIFU_MLX_ENABLED
    // Unlike CUDA, a build with MLX still has to cope with there being no GPU to talk to, so
    // the operators are only registered when MLX can actually reach one.
    if (op::metal::MetalOperators::isAvailable()) {
      CHECK(!gOperatorsForDevice[Device::kMetal]);
      gOperatorsForDevice[Device::kMetal] = op::metal::MetalOperators::create();
    }
#endif
#ifdef LIBWAIFU_VULKAN_ENABLED
    // Registered the same way, only where there is a device: a Vulkan build loads the loader
    // itself, and so runs on machines that have none.
    if (op::vulkan::VulkanOperators::isAvailable()) {
      CHECK(!gOperatorsForDevice[Device::kVulkan]);
      gOperatorsForDevice[Device::kVulkan] = op::vulkan::VulkanOperators::create();
    }
#endif
  }
}

Operators *getOperators(Device::Type deviceType) {
  if (!gInitialized) throw lut::AbortedError("call getOperators() before initialization");
  if (!gOperatorsForDevice[deviceType]) {
    std::string deviceName = Device(deviceType).getName();
    throw lut::NotImplementedError(lut::sprintf("%s operators not implemented", deviceName));
  }

  return gOperatorsForDevice[deviceType].get();
}

std::shared_ptr<Operators> getOperatorsSharedPtr(Device::Type deviceType) {
  if (!gInitialized) throw lut::AbortedError("call getOperators() before initialization");
  if (!gOperatorsForDevice[deviceType]) {
    std::string deviceName = Device(deviceType).getName();
    throw lut::NotImplementedError(lut::sprintf("%s operators not implemented", deviceName));
  }

  return gOperatorsForDevice[deviceType];
}

bool isOperatorsAvailable(Device::Type deviceType) {
  if (!gInitialized) throw lut::AbortedError("call isOperatorsAvailable() before initialization");
  if (!gOperatorsForDevice[deviceType]) {
    return false;
  } else {
    return true;
  }
}

void destroyOperators() {
  op::cpu::kernel::destroy();

  if (gInitialized.exchange(false)) {
    for (int i = 0; i < Device::NumDeviceType; ++i) {
      gOperatorsForDevice[i] = nullptr;
    }
  }
}

}  // namespace fl
