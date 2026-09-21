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
#include <cstdint>

#include "catch2/catch_amalgamated.hpp"
#include "flint/cuda/common.h"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/memory.h"
#include "flint/operators.h"

namespace fl {

CATCH_TEST_CASE("test CUDA memory snapshot", "[fl][cuda][memory]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  MemorySnapshot::resetPeakStats(Device::getCuda());

  MemorySnapshot before = MemorySnapshot::capture(Device::getCuda());
  CATCH_REQUIRE(before.getTotalMemory() > 0);
  CATCH_REQUIRE(before.getFreeMemory() > 0);
  CATCH_REQUIRE(before.getFreeMemory() <= before.getTotalMemory());

  int64_t bytes = 0;
  {
    Tensor x = F::tensor({1024, 1024}, DType::kFloat16, Device::getCuda());
    bytes = x.getNumEl() * 2;

    MemorySnapshot allocated = MemorySnapshot::capture(Device::getCuda());
    CATCH_REQUIRE(allocated.getAllocatedMemory() >= before.getAllocatedMemory() + bytes);
  }

  // the tensor is gone, but its bytes stay in the pool and remain in the peak.
  MemorySnapshot after = MemorySnapshot::capture(Device::getCuda());
  CATCH_REQUIRE(after.getAllocatedMemory() <= before.getAllocatedMemory());
  CATCH_REQUIRE(after.getPeakAllocatedMemory() >= bytes);
}

CATCH_TEST_CASE("test CUDA memory release", "[fl][cuda][memory]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

#ifndef LIBWAIFU_CUDA_MALLOC_ASYNC_ENABLED
  CATCH_SKIP("built without the pool, where a free hands the memory straight back");
#else
  // Measured on the pool rather than on the card, because the card is a number every process on
  // the machine moves and this is a claim about what *this* process holds. It is also the number
  // the free memory of the card follows: bytes the pool has reserved are bytes no one else may
  // have, which is the whole reason there is a call to hand them over.
  cudaMemPool_t memoryPool;
  LL_CHECK_CUDA_STATUS(cudaDeviceGetDefaultMemPool(&memoryPool, 0));
  auto reserved = [&memoryPool]() {
    uint64_t bytes = 0;
    LL_CHECK_CUDA_STATUS(
        cudaMemPoolGetAttribute(memoryPool, cudaMemPoolAttrReservedMemCurrent, &bytes));
    return bytes;
  };

  // From a pool that is holding nothing, so that what the numbers below move by is this test's
  // own tensor and not what an earlier test left in there.
  MemorySnapshot::releaseUnused(Device::getCuda());
  uint64_t empty = reserved();

  // Large enough to be unmistakable against an allocator that rounds its blocks up.
  constexpr int64_t kBytes = 32 * 1024 * 1024;
  {
    Tensor x = F::tensor({4096, 4096}, DType::kFloat16, Device::getCuda());
    CATCH_REQUIRE(x.getNumEl() * 2 == kBytes);
  }

  // The tensor is gone and the pool still has its bytes, which is what the release threshold set
  // at startup asks for: the next allocation this size is to come out of here rather than out of
  // the driver.
  CATCH_REQUIRE(reserved() >= empty + kBytes);

  // And this is the one call that ends that. Somebody who stopped a run to free the card is
  // asking for this number to come down, not for the allocator to keep its options open.
  MemorySnapshot::releaseUnused(Device::getCuda());
  CATCH_REQUIRE(reserved() <= empty);
#endif
}

CATCH_TEST_CASE("test CUDA FastDivmod", "[fl][cuda]") {
  constexpr uint32_t divisors[] = {1, 2, 3, 7, 16, 255, 65535, INT32_MAX};

  for (uint32_t divisor : divisors) {
    op::cuda::FastDivmod divider(divisor);
    uint32_t dividends[] = {
        0, 1, divisor - 1, divisor, std::min(divisor + 1, uint32_t{INT32_MAX}), INT32_MAX};

    for (uint32_t dividend : dividends) {
      uint32_t quotient;
      uint32_t remainder;
      divider.divmod(dividend, quotient, remainder);
      CATCH_REQUIRE(quotient == dividend / divisor);
      CATCH_REQUIRE(remainder == dividend % divisor);
    }
  }
}

CATCH_TEST_CASE("test CUDA FastDivmod (powers of two)", "[fl][cuda]") {
  // The magic-number derivation shifts by ceil(log2(divisor)); an exact power of two is where
  // that shift lands on the boundary and the multiplier is at its smallest.
  for (int shift = 0; shift < 31; ++shift) {
    uint32_t divisor = uint32_t{1} << shift;
    op::cuda::FastDivmod divider(divisor);

    uint32_t dividends[] = {
        0,
        1,
        divisor - 1,
        divisor,
        divisor + 1,
        divisor * 2 - 1,
        INT32_MAX - 1,
        INT32_MAX};
    for (uint32_t dividend : dividends) {
      uint32_t quotient;
      uint32_t remainder;
      divider.divmod(dividend, quotient, remainder);
      CATCH_INFO("divisor = " << divisor << ", dividend = " << dividend);
      CATCH_REQUIRE(quotient == dividend / divisor);
      CATCH_REQUIRE(remainder == dividend % divisor);
    }
  }
}

CATCH_TEST_CASE("test CUDA FastDivmod (exhaustive small)", "[fl][cuda]") {
  // Small divisors are what the tensor accessors actually use (one per axis), so walk every
  // dividend/divisor pair in that range rather than sampling it.
  for (uint32_t divisor = 1; divisor <= 64; ++divisor) {
    op::cuda::FastDivmod divider(divisor);
    for (uint32_t dividend = 0; dividend < 512; ++dividend) {
      uint32_t quotient;
      uint32_t remainder;
      divider.divmod(dividend, quotient, remainder);
      CATCH_INFO("divisor = " << divisor << ", dividend = " << dividend);
      CATCH_REQUIRE(quotient == dividend / divisor);
      CATCH_REQUIRE(remainder == dividend % divisor);
    }
  }
}

}  // namespace fl
