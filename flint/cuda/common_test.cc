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
#include <memory>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/test_functional.h"
#include "flint/cuda/common.h"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/memory.h"
#include "flint/operators.h"
#include "lutil/error.h"

namespace fl {

namespace {

/// The CUDA operators, which the calls here that run on CUDA are asked of.
Operators *cudaOps() {
  return getOperators(Device::kCuda);
}

}  // namespace


CATCH_TEST_CASE("test CUDA memory snapshot", "[fl][cuda][memory]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  MemorySnapshot::resetPeakStats(Device::getCuda());

  MemorySnapshot before = MemorySnapshot::capture(Device::getCuda());
  CATCH_REQUIRE(before.getTotalMemory() > 0);
  CATCH_REQUIRE(before.getFreeMemory() > 0);
  CATCH_REQUIRE(before.getFreeMemory() <= before.getTotalMemory());

  int64_t bytes = 0;
  {
    Tensor x = F::empty(Device::getCuda(), {1024, 1024}, DType::kFloat16);
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
    Tensor x = F::empty(Device::getCuda(), {4096, 4096}, DType::kFloat16);
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

CATCH_TEST_CASE("test CUDA memory budget counts what flint allocates", "[fl][cuda][memory]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Without a limit, the card is the capacity, whoever else has some of it.
  MemoryBudget before = MemoryBudget::capture(Device::getCuda());
  CATCH_REQUIRE(before.capacity == MemorySnapshot::capture(Device::getCuda()).getTotalMemory());

  // The bytes asked for, exactly: what the pool rounds them up to is not counted.
  {
    Tensor x = F::empty(Device::getCuda(), {1000, 1001}, DType::kFloat16);
    MemoryBudget during = MemoryBudget::capture(Device::getCuda());
    CATCH_REQUIRE(during.allocated == before.allocated + 1000 * 1001 * 2);
    // What the pool holds is at least what is in use: the pool rounds up, and keeps what is freed.
    CATCH_REQUIRE(during.held >= during.allocated);
  }
  CATCH_REQUIRE(MemoryBudget::capture(Device::getCuda()).allocated == before.allocated);
}

CATCH_TEST_CASE(
    "test CUDA allocation past the memory limit is out of memory",
    "[fl][cuda][memory]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Lifted however the test ends, so that no other test runs under it.
  struct Unlimited {
    ~Unlimited() {
      MemorySnapshot::setLimit(Device::getCuda(), 0);
    }
  } unlimited;

  // Room for one 2 MB tensor and not two.
  constexpr int64_t kTwoMB = 2 << 20;
  int64_t held = MemoryBudget::capture(Device::getCuda()).allocated;
  MemorySnapshot::setLimit(Device::getCuda(), held + 3 * (1 << 20));
  CATCH_REQUIRE(MemoryBudget::capture(Device::getCuda()).capacity == held + 3 * (1 << 20));

  Tensor first = F::empty(Device::getCuda(), {int(kTwoMB)}, DType::kInt8);
  CATCH_REQUIRE_THROWS_AS(
      F::empty(Device::getCuda(), {int(kTwoMB)}, DType::kInt8),
      lut::OutOfMemoryError);
  // A workspace an operator asks for is held to the same limit.
  CATCH_REQUIRE_THROWS_AS(op::cuda::llynCudaAlloc<int8_t>(kTwoMB), lut::OutOfMemoryError);
  // Refused, nothing was taken.
  CATCH_REQUIRE(MemoryBudget::capture(Device::getCuda()).allocated == held + kTwoMB);

  // Given back, the bytes are there for the next tensor, at once and inside the limit.
  first = Tensor();
  Tensor again = F::empty(Device::getCuda(), {int(kTwoMB)}, DType::kInt8);
  again = Tensor();
  CATCH_REQUIRE(MemoryBudget::capture(Device::getCuda()).allocated == held);
}

CATCH_TEST_CASE("test CUDA running out of the card is out of memory", "[fl][cuda][memory]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // More than any card has: the driver refuses it, as itself and not as an abort, and leaves no
  // error behind for the next launch to find.
  int64_t held = MemoryBudget::capture(Device::getCuda()).allocated;
  int64_t tooMuch = MemorySnapshot::capture(Device::getCuda()).getTotalMemory() * 4;
  CATCH_REQUIRE_THROWS_AS(op::cuda::llynCudaMalloc(tooMuch), lut::OutOfMemoryError);
  CATCH_REQUIRE(cudaGetLastError() == cudaSuccess);
  CATCH_REQUIRE(MemoryBudget::capture(Device::getCuda()).allocated == held);
}

CATCH_TEST_CASE("test CUDA weights kept on the card have memory of their own", "[fl][cuda][memory]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

#ifndef LIBWAIFU_CUDA_MALLOC_ASYNC_ENABLED
  CATCH_SKIP("built without the pool, where every allocation is memory of its own");
#else
  cudaMemPool_t memoryPool;
  LL_CHECK_CUDA_STATUS(cudaDeviceGetDefaultMemPool(&memoryPool, 0));
  auto reserved = [&memoryPool]() {
    uint64_t bytes = 0;
    LL_CHECK_CUDA_STATUS(
        cudaMemPoolGetAttribute(memoryPool, cudaMemPoolAttrReservedMemCurrent, &bytes));
    return static_cast<int64_t>(bytes);
  };
  auto cardFree = []() {
    size_t free = 0;
    size_t total = 0;
    LL_CHECK_CUDA_STATUS(cudaMemGetInfo(&free, &total));
    return static_cast<int64_t>(free);
  };
  constexpr int64_t kFourMB = 4 << 20;
  constexpr int kCount = 64;
  MemorySnapshot::releaseUnused(Device::getCuda());

  // Large weights, with small ones and temporary copies in among them, as a model's are.
  int64_t poolBefore = reserved();
  MemoryBudget before = MemoryBudget::capture(Device::getCuda());
  std::vector<std::unique_ptr<TensorData>> large;
  std::vector<std::unique_ptr<TensorData>> small;
  for (int i = 0; i < kCount; ++i) {
    large.push_back(F::allocate(Device::getCuda(), kFourMB, DType::kInt8, AllocHint::kWeight));
    small.push_back(F::allocate(Device::getCuda(), 4096, DType::kInt8, AllocHint::kWeight));
    small.push_back(
        F::allocate(Device::getCuda(), kFourMB, DType::kInt8, AllocHint::kTemporaryWeight));
  }

  // The large weights are not the pool's, and every byte is counted either way.
  int64_t smallBytes = kCount * (4096 + kFourMB);
  MemoryBudget during = MemoryBudget::capture(Device::getCuda());
  CATCH_REQUIRE(during.allocated == before.allocated + kCount * kFourMB + smallBytes);
  // The pool grew by the small ones and the temporary copies, and the granule it rounds to: not by
  // another 256 MB of weights besides.
  CATCH_REQUIRE(reserved() - poolBefore < smallBytes + kCount * kFourMB / 2);
  CATCH_REQUIRE(during.held >= reserved() + kCount * kFourMB);

  // Let go of, the large weights' memory is the driver's again -- not the pool's to keep -- with
  // every small tensor beside them still held. Card-wide, so with room for what else moves it.
  int64_t freeWhileHeld = cardFree();
  large.clear();
  CATCH_REQUIRE(cardFree() - freeWhileHeld >= kCount * kFourMB * 3 / 4);
  CATCH_REQUIRE(
      MemoryBudget::capture(Device::getCuda()).allocated == before.allocated + smallBytes);
#endif
}

}  // namespace fl
