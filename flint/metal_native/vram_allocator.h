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

#pragma once

#include <stdint.h>

#include <functional>
#include <memory>
#include <mutex>

#include "flint/memory.h"
#include "flint/metal_native/block_allocator.h"

// metal-cpp's types, named and not included: only the .cc files talk to Metal.
namespace MTL {
class Buffer;
class Device;
}  // namespace MTL

namespace fl {
namespace op {
namespace metal_native {

/// One tensor's storage: a run of shared memory, which the GPU and the host both address, cut
/// from a larger buffer the allocator holds.
///
/// It remembers the last batch of work that was handed it, which is what the host waits for
/// before it touches the bytes -- and only that batch, not everything queued.
struct BufferState {
  MTL::Buffer *buffer = nullptr;  // the chunk's
  int64_t offset = 0;             // where the run starts in it
  int64_t size = 0;               // what was allocated, which is the size asked for rounded up
  uint64_t lastUse = 0;
  Block *block = nullptr;
};
using Buffer = std::shared_ptr<BufferState>;

/// The memory of the tensors on a Metal device: chunks of shared memory asked of the device, the
/// tensors cut from them by a BlockAllocator, a limit on what may be held, and the numbers a
/// snapshot reports.
///
/// A chunk may go back to the device only once the device is done with it, and the work that uses
/// it is the context's: `synchronize`, which the allocator is given, waits for all of it. The
/// allocator calls it with its own lock held, so whoever owns the work must never call into the
/// allocator while holding a lock of its own.
class VRamAllocator {
 public:
  /// Chunks of `device`'s, kept within `budgetBytes` -- what the system suggests one process keep
  /// on the GPU -- where it can be.
  VRamAllocator(MTL::Device *device, int64_t budgetBytes, std::function<void()> synchronize);
  VRamAllocator(const VRamAllocator &) = delete;
  VRamAllocator &operator=(const VRamAllocator &) = delete;

  /// A buffer of at least `bytes` bytes, cut from a chunk the allocator holds, or from a new one.
  /// @throw lut::OutOfMemoryError when the device is out of memory, or a new chunk would take the
  ///     allocator past its memory limit, even after the empty chunks have gone back. Nothing is
  ///     taken then, so a caller that frees something may ask again.
  Buffer allocate(int64_t bytes);

  /// Hand `buffer`'s bytes back, for the next tensor that fits in them. The chunk stays the
  /// allocator's even once nothing is cut from it: chunks go back to the device only in
  /// releaseUnusedMemory(), or when the budget or the limit would be passed.
  void free(const Buffer &buffer);

  /// Wait for the device and hand every empty chunk back. What a caller does when it is done with
  /// a piece of work -- a whole image, say -- so that what the work needed at its peak is not kept
  /// until the next.
  void releaseUnusedMemory();

  /// The most the chunks the allocator holds may come to, in bytes; 0, the default, for no limit
  /// but the device's. A tensor that fits in a chunk already held is cut from it whatever the
  /// limit; one that needs a new chunk past it is refused by allocate(). Lowering it below
  /// what is held lets nothing go: the chunks in use stay, and the limit holds from the next
  /// chunk on.
  void setMemoryLimit(int64_t bytes);
  int64_t getMemoryLimit();

  /// The bytes of the chunks the allocator holds -- in use or not -- which is what the limit
  /// counts.
  int64_t getHeldBytes();

  /// The bytes the tensors hold: each one's size rounded up to the allocator's alignment, and none
  /// of the free blocks or empty chunks beside them. What a snapshot calls allocated.
  int64_t getAllocatedBytes();

  MemorySnapshot captureMemorySnapshot();
  void resetPeakMemoryStats();

 private:
  std::mutex _mutex;

  MTL::Device *_device;
  int64_t _budgetBytes;
  std::function<void()> _synchronize;

  // The chunks the tensors are cut from, and what the tensors hold.
  BlockAllocator _blocks;
  int64_t _allocatedBytes = 0;
  int64_t _peakAllocatedBytes = 0;
  int64_t _memoryLimit = 0;  // 0 for none

  // Waits for the device, then hands every empty chunk back to it.
  void synchronizeAndReleaseEmptyChunks();
};

}  // namespace metal_native
}  // namespace op
}  // namespace fl
