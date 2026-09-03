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
#include "flint/device.h"
#include "flint/capi.h"
#include "flint/cuda/future_tensor.h"
#include "flint/functional.h"
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {
namespace {

/// The elements themselves rather than `F::allClose`: nothing here computes, so a round trip that
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

  Tensor host = F::rand({4, 8}, DType::kFloat, Device::getCpu());
  Tensor locked = F::toDevice(Device::getCudaHost(), host);

  CATCH_REQUIRE(locked.getDevice().getType() == Device::kCudaHost);
  CATCH_REQUIRE(locked.getDevice().isHost());
  CATCH_REQUIRE(Device::getCudaHost().getName() == "cuda-host");

  // The point of the whole device: these bytes are addressable from here, so the comparison below
  // reads them where they lie rather than copying them back first.
  CATCH_REQUIRE(equalFloat(host, locked));
}

CATCH_TEST_CASE("cuda-host memory can be allocated directly", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor locked = F::tensor({2, 3}, DType::kFloat, Device::getCudaHost());
  CATCH_REQUIRE(locked.getDevice().getType() == Device::kCudaHost);
  CATCH_REQUIRE(locked.getNumEl() == 6);

  // Written by the CPU where it lies, which is what makes it usable as a staging buffer.
  Tensor source = F::rand({2, 3}, DType::kFloat, Device::getCpu());
  F::copy(source, locked);
  CATCH_REQUIRE(equalFloat(source, locked));
}

CATCH_TEST_CASE("cuda-host memory makes the round trip to the GPU unchanged", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor host = F::rand({16, 16}, DType::kFloat, Device::getCpu());

  Tensor there = F::toDevice(Device::getCuda(), F::toDevice(Device::getCudaHost(), host));
  CATCH_REQUIRE(there.getDevice().getType() == Device::kCuda);

  Tensor back = F::toDevice(Device::getCpu(), F::toDevice(Device::getCudaHost(), there));
  CATCH_REQUIRE(back.getDevice().getType() == Device::kCpu);
  CATCH_REQUIRE(equalFloat(host, back));
}

CATCH_TEST_CASE("an asynchronous copy arrives, once taken", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor host = F::rand({32, 32}, DType::kFloat, Device::getCpu());
  Tensor locked = F::toDevice(Device::getCudaHost(), host);

  FutureTensor pending = F::toDeviceAsync(Device::getCuda(), locked);
  Tensor there = pending.take();

  CATCH_REQUIRE(there.getDevice().getType() == Device::kCuda);
  CATCH_REQUIRE_NOTHROW(there.getInternalData()->getRawData());
  CATCH_REQUIRE(equalFloat(host, F::toDevice(Device::getCpu(), there)));
}

CATCH_TEST_CASE("takeSync waits for the copy rather than ordering it", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor host = F::rand({32, 32}, DType::kFloat, Device::getCpu());
  Tensor locked = F::toDevice(Device::getCudaHost(), host);

  // The bytes are there when this returns, not merely ordered to be. Nothing here can tell the
  // two apart -- reading through cudaMemcpy would be correct either way -- so what this pins down
  // is that the path works at all, and the difference is the host's, not the result's.
  Tensor there = F::toDeviceAsync(Device::getCuda(), locked).takeSync();

  CATCH_REQUIRE_NOTHROW(there.getInternalData()->getRawData());
  CATCH_REQUIRE(equalFloat(host, F::toDevice(Device::getCpu(), there)));
}

CATCH_TEST_CASE("many copies in flight all land where they belong", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Many copies issued before any of them is taken, which is the shape a model's weights arrive
  // in: each carries an event of its own, and none of them is the one taken next.
  constexpr int kCount = 64;
  std::vector<Tensor> sources;
  std::vector<FutureTensor> flying;
  for (int index = 0; index < kCount; ++index) {
    Tensor host = F::rand({16, 16}, DType::kFloat, Device::getCpu());
    sources.push_back(host);
    flying.push_back(F::toDeviceAsync(Device::getCuda(), F::toDevice(Device::getCudaHost(), host)));
  }

  for (int index = 0; index < kCount; ++index) {
    Tensor landed = flying[index].take();
    CATCH_REQUIRE(equalFloat(sources[index], F::toDevice(Device::getCpu(), landed)));
  }
}

CATCH_TEST_CASE("an asynchronous copy nobody took is thrown away safely", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // What a fetch that turned out not to be wanted looks like: a FutureTensor that goes out of
  // scope. The memory goes back in the copy stream's order rather than the compute stream's, so
  // it cannot be handed to the next allocation while the copy engine is still writing it.
  Tensor host = F::rand({64, 64}, DType::kFloat, Device::getCpu());
  Tensor locked = F::toDevice(Device::getCudaHost(), host);
  for (int index = 0; index < 32; ++index) {
    FutureTensor discarded = F::toDeviceAsync(Device::getCuda(), locked);
  }

  // Whatever was reused afterwards is still correct.
  Tensor kept = F::toDeviceAsync(Device::getCuda(), locked).take();
  CATCH_REQUIRE(equalFloat(host, F::toDevice(Device::getCpu(), kept)));
}

CATCH_TEST_CASE("an asynchronous copy refuses every direction but one", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor host = F::rand({4, 4}, DType::kFloat, Device::getCpu());
  Tensor locked = F::toDevice(Device::getCudaHost(), host);
  Tensor there = F::toDevice(Device::getCuda(), host);

  // Pageable host memory is the one that matters: the driver would stage it through a buffer of
  // its own and the copy would be synchronous under an asynchronous name.
  CATCH_REQUIRE_THROWS(F::toDeviceAsync(Device::getCuda(), host));
  CATCH_REQUIRE_THROWS(F::toDeviceAsync(Device::getCudaHost(), there));
  CATCH_REQUIRE_THROWS(F::toDeviceAsync(Device::getCpu(), there));
  CATCH_REQUIRE_THROWS(F::toDeviceAsync(Device::getCuda(), there));
}

CATCH_TEST_CASE("the C interface hands the copy over as a future", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor host = F::rand({16, 16}, DType::kFloat, Device::getCpu());
  Tensor locked = F::toDevice(Device::getCudaHost(), host);

  fl_tensor_t source = reinterpret_cast<fl_tensor_t>(&locked);

  fl_future_tensor_t future = nullptr;
  CATCH_REQUIRE(fl_tensor_to_device_async(source, FL_DEVICE_CUDA, &future) == FL_OK);
  CATCH_REQUIRE(future != nullptr);

  fl_tensor_t taken = nullptr;
  CATCH_REQUIRE(fl_future_tensor_take(future, &taken) == FL_OK);
  CATCH_REQUIRE(equalFloat(host, F::toDevice(Device::getCpu(), *reinterpret_cast<Tensor *>(taken))));

  // Taking does not free the future, and destroying one is fine whether it was taken or not.
  fl_tensor_destroy(taken);
  fl_future_tensor_destroy(future);

  // A future nobody took, freed the same way.
  fl_future_tensor_t dropped = nullptr;
  CATCH_REQUIRE(fl_tensor_to_device_async(source, FL_DEVICE_CUDA, &dropped) == FL_OK);
  fl_future_tensor_destroy(dropped);

  // The one pair that is allowed is still the only one.
  fl_future_tensor_t refused = nullptr;
  fl_tensor_t pageable = reinterpret_cast<fl_tensor_t>(&host);
  CATCH_REQUIRE(fl_tensor_to_device_async(pageable, FL_DEVICE_CUDA, &refused) != FL_OK);
  CATCH_REQUIRE(refused == nullptr);
}

CATCH_TEST_CASE("cuda-host has no operators of its own", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  CATCH_REQUIRE_FALSE(isOperatorsAvailable(Device::kCudaHost));

  // Arithmetic on it is refused rather than quietly run on the CPU: the bytes would be right and
  // the answer about which device holds them would not.
  Tensor locked = F::tensor({2, 2}, DType::kFloat, Device::getCudaHost());
  CATCH_REQUIRE_THROWS(F::mul(locked, locked));
}

}  // namespace fl
