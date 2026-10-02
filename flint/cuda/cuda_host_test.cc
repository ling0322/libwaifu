// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <algorithm>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/functional.h"
#include "flint/device.h"
#include "flint/capi.h"
#include "flint/cuda/future_tensor.h"
#include "flint/cuda/to_device.h"
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {

namespace {

/// The CUDA operators, which the calls here that run on CUDA are asked of.
Operators *cudaOps() {
  return getOperators(Device::kCuda);
}

/// The CPU operators, which the calls here that run on CPU are asked of.
Operators *cpuOps() {
  return getOperators(Device::kCpu);
}

/// The elements themselves rather than `allClose`: nothing here computes, so a round trip that
/// is not exact is a bug rather than a rounding.
bool equalFloat(Tensor a, Tensor b) {
  a.throwIfInvalidShape(b.getShape(), "equalFloat");

  const float *pa = a.getInternalData()->getData<float>(a.getInternalOffset());
  const float *pb = b.getInternalData()->getData<float>(b.getInternalOffset());
  return std::equal(pa, pa + a.getNumEl(), pb);
}

}  // namespace

CATCH_TEST_CASE("cuda-host memory is host memory the CPU can read", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor host = F::rand(Device::getCpu(), {4, 8}, DType::kFloat);
  Tensor locked = F::toDevice(cudaOps(), host, Device::getCudaHost());

  CATCH_REQUIRE(locked.getDevice().getType() == Device::kCudaHost);
  CATCH_REQUIRE(locked.getDevice().isHost());
  CATCH_REQUIRE(Device::getCudaHost().getName() == "cuda-host");

  // The point of the whole device: these bytes are addressable from here, so the comparison below
  // reads them where they lie rather than copying them back first.
  CATCH_REQUIRE(equalFloat(host, locked));
}

CATCH_TEST_CASE("cuda-host memory can be allocated directly", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor locked = F::empty(Device(Device::kCudaHost), {2, 3}, DType::kFloat);
  CATCH_REQUIRE(locked.getDevice().getType() == Device::kCudaHost);
  CATCH_REQUIRE(locked.getNumEl() == 6);

  // Written by the CPU where it lies, which is what makes it usable as a staging buffer.
  Tensor source = F::rand(Device::getCpu(), {2, 3}, DType::kFloat);
  cpuOps()->copy(source, locked);
  CATCH_REQUIRE(equalFloat(source, locked));
}

CATCH_TEST_CASE("cuda-host memory makes the round trip to the GPU unchanged", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor host = F::rand(Device::getCpu(), {16, 16}, DType::kFloat);

  Tensor there =
      F::toDevice(cudaOps(), F::toDevice(cudaOps(), host, Device::getCudaHost()), Device::getCuda());
  CATCH_REQUIRE(there.getDevice().getType() == Device::kCuda);

  Tensor back =
      F::toDevice(cudaOps(), F::toDevice(cudaOps(), there, Device::getCudaHost()), Device::getCpu());
  CATCH_REQUIRE(back.getDevice().getType() == Device::kCpu);
  CATCH_REQUIRE(equalFloat(host, back));
}

CATCH_TEST_CASE("an asynchronous copy arrives, once taken", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor host = F::rand(Device::getCpu(), {32, 32}, DType::kFloat);
  Tensor locked = F::toDevice(cudaOps(), host, Device::getCudaHost());

  FutureTensor pending = op::cuda::toDeviceAsync(Device::getCuda(), locked);
  Tensor there = pending.take();

  CATCH_REQUIRE(there.getDevice().getType() == Device::kCuda);
  CATCH_REQUIRE_NOTHROW(there.getInternalData()->getRawData());
  CATCH_REQUIRE(equalFloat(host, F::toDevice(cudaOps(), there, Device::getCpu())));
}

CATCH_TEST_CASE("takeSync waits for the copy rather than ordering it", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor host = F::rand(Device::getCpu(), {32, 32}, DType::kFloat);
  Tensor locked = F::toDevice(cudaOps(), host, Device::getCudaHost());

  // The bytes are there when this returns, not merely ordered to be. Nothing here can tell the
  // two apart -- reading through cudaMemcpy would be correct either way -- so what this pins down
  // is that the path works at all, and the difference is the host's, not the result's.
  Tensor there = op::cuda::toDeviceAsync(Device::getCuda(), locked).takeSync();

  CATCH_REQUIRE_NOTHROW(there.getInternalData()->getRawData());
  CATCH_REQUIRE(equalFloat(host, F::toDevice(cudaOps(), there, Device::getCpu())));
}

CATCH_TEST_CASE("many copies in flight all land where they belong", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Many copies issued before any of them is taken, which is the shape a model's weights arrive
  // in: each carries an event of its own, and none of them is the one taken next.
  constexpr int kCount = 64;
  std::vector<Tensor> sources;
  std::vector<FutureTensor> flying;
  for (int index = 0; index < kCount; ++index) {
    Tensor host = F::rand(Device::getCpu(), {16, 16}, DType::kFloat);
    sources.push_back(host);
    flying.push_back(op::cuda::toDeviceAsync(
        Device::getCuda(),
        F::toDevice(cudaOps(), host, Device::getCudaHost())));
  }

  for (int index = 0; index < kCount; ++index) {
    Tensor landed = flying[index].take();
    CATCH_REQUIRE(equalFloat(sources[index], F::toDevice(cudaOps(), landed, Device::getCpu())));
  }
}

CATCH_TEST_CASE("an asynchronous copy nobody took is thrown away safely", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // What a fetch that turned out not to be wanted looks like: a FutureTensor that goes out of
  // scope. The memory goes back in the copy stream's order rather than the compute stream's, so
  // it cannot be handed to the next allocation while the copy engine is still writing it.
  Tensor host = F::rand(Device::getCpu(), {64, 64}, DType::kFloat);
  Tensor locked = F::toDevice(cudaOps(), host, Device::getCudaHost());
  for (int index = 0; index < 32; ++index) {
    FutureTensor discarded = op::cuda::toDeviceAsync(Device::getCuda(), locked);
  }

  // Whatever was reused afterwards is still correct.
  Tensor kept = op::cuda::toDeviceAsync(Device::getCuda(), locked).take();
  CATCH_REQUIRE(equalFloat(host, F::toDevice(cudaOps(), kept, Device::getCpu())));
}

CATCH_TEST_CASE("an asynchronous copy refuses every direction but one", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor host = F::rand(Device::getCpu(), {4, 4}, DType::kFloat);
  Tensor locked = F::toDevice(cudaOps(), host, Device::getCudaHost());
  Tensor there = F::toDevice(cudaOps(), host, Device::getCuda());

  // Pageable host memory is the one that matters: the driver would stage it through a buffer of
  // its own and the copy would be synchronous under an asynchronous name.
  CATCH_REQUIRE_THROWS(op::cuda::toDeviceAsync(Device::getCuda(), host));
  CATCH_REQUIRE_THROWS(op::cuda::toDeviceAsync(Device::getCudaHost(), there));
  CATCH_REQUIRE_THROWS(op::cuda::toDeviceAsync(Device::getCpu(), there));
  CATCH_REQUIRE_THROWS(op::cuda::toDeviceAsync(Device::getCuda(), there));
}

CATCH_TEST_CASE("the C interface hands the copy over to be waited on", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor host = F::rand(Device::getCpu(), {16, 16}, DType::kFloat);
  Tensor locked = F::toDevice(cudaOps(), host, Device::getCudaHost());

  TensorView lockedView(locked);
  fl_tensor_view_t source = reinterpret_cast<fl_tensor_view_t>(&lockedView);

  fl_tensor_data_t dest = nullptr;
  fl_transfer_t transfer = nullptr;
  CATCH_REQUIRE(fl_transfer_async(source, FL_DEVICE_CUDA, &dest, &transfer) == FL_OK);
  CATCH_REQUIRE(dest != nullptr);
  CATCH_REQUIRE(transfer != nullptr);
  CATCH_REQUIRE(fl_transfer_wait(transfer) == FL_OK);

  // The storage is the caller's: seen through a view of the source's shape, it holds its bytes.
  TensorView arrived(
      reinterpret_cast<TensorData *>(dest), std::make_shared<TensorShape>(locked.getShape()), 0);
  Tensor back = F::empty(Device::getCpu(), {16, 16}, DType::kFloat);
  cudaOps()->transfer(arrived, back);
  CATCH_REQUIRE(equalFloat(host, back));

  // Waiting does not free the handle, and destroying one is fine whether it was waited on or not.
  fl_transfer_destroy(transfer);
  fl_tensor_data_destroy(dest);

  // A copy nobody waited on, freed the same way.
  CATCH_REQUIRE(fl_transfer_async(source, FL_DEVICE_CUDA, &dest, &transfer) == FL_OK);
  fl_transfer_destroy(transfer);
  fl_tensor_data_destroy(dest);

  // The one pair that is allowed is still the only one.
  TensorView hostView(host);
  fl_tensor_data_t refused = nullptr;
  fl_tensor_view_t pageable = reinterpret_cast<fl_tensor_view_t>(&hostView);
  CATCH_REQUIRE(fl_transfer_async(pageable, FL_DEVICE_CUDA, &refused, &transfer) != FL_OK);
  CATCH_REQUIRE(refused == nullptr);
}

CATCH_TEST_CASE("cuda-host has no operators of its own", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  CATCH_REQUIRE_FALSE(isOperatorsAvailable(Device::kCudaHost));

  // Asking for its operators is refused rather than quietly answered with the CPU's: the bytes
  // would be right and the answer about which device holds them would not. Page-locked memory is
  // made by the CUDA operators, which is the one thing about it a device decides.
  CATCH_REQUIRE_THROWS(getOperators(Device::kCudaHost));
  CATCH_REQUIRE_NOTHROW(F::empty(Device(Device::kCudaHost), {2, 2}, DType::kFloat));
}

}  // namespace fl
