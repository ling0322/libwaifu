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

class CommandStream;

/// How a buffer was had, which is how it goes back.
enum class BufferType {
  /// A run of a chunk the allocator keeps for reuse: an activation, or anything but a weight.
  kCached,
  /// A buffer of its own, which goes back to the device whole: a weight's.
  kWeight,
};

/// One tensor's storage: a run of shared memory, which the GPU and the host both address -- cut
/// from a larger buffer the allocator holds, or, for a weight, a buffer of its own.
///
/// It remembers the last batch of work that was handed it, which is what the host waits for
/// before it touches the bytes -- and only that batch, not everything queued.
struct BufferState {
  MTL::Buffer *buffer = nullptr;  // the chunk's, or its own
  int64_t offset = 0;             // where the run starts in it
  int64_t size = 0;               // what was allocated, which is the size asked for rounded up
  uint64_t lastUse = 0;
  Block *block = nullptr;  // the run of the chunk it is, for a cached buffer
  BufferType type = BufferType::kCached;
};
using Buffer = std::shared_ptr<BufferState>;

/// The memory of the tensors on a Metal device: chunks of shared memory asked of the device and
/// the activations cut from them by a BlockAllocator, the weights' buffers of their own, a limit
/// on what may be held, and the numbers a snapshot reports.
///
/// Memory may go back to the device only once the device is done with it, and the work that uses
/// it is the stream's: the allocator waits for it there, with its own lock held. The stream never
/// asks anything of the allocator.
class VramAllocator {
 public:
  /// Memory of `device`'s, kept within `budgetBytes` -- what the system suggests one process keep
  /// on the GPU -- where it can be, for the work `stream` runs.
  VramAllocator(MTL::Device *device, int64_t budgetBytes, std::shared_ptr<CommandStream> stream);
  VramAllocator(const VramAllocator &) = delete;
  VramAllocator &operator=(const VramAllocator &) = delete;

  /// A buffer of at least `bytes` bytes for what `hint` says it is for. An activation, a temporary
  /// copy of a weight, or anything else that is not a weight kept on the device, is cut from a
  /// chunk the allocator holds, or from a new one. A weight is given a buffer of its own, so that
  /// letting it go gives its memory back to the device whole rather than leaving a hole in a
  /// chunk.
  /// @throw lut::OutOfMemoryError when the device is out of memory, or the allocator would pass its
  ///     memory limit, even after the empty chunks have gone back.
  ///     Nothing is taken then, so a caller that frees something may ask again.
  Buffer allocate(int64_t bytes, AllocHint hint = AllocHint::kActivation);

  /// Hand `buffer`'s bytes back, for the next tensor that fits in them. The chunk stays the
  /// allocator's even once nothing is cut from it: chunks go back to the device only in
  /// releaseUnusedMemory(), or when the budget or the limit would be passed. A weight's buffer goes
  /// back to the device at once -- after the work that last used it, which this waits for.
  void free(const Buffer &buffer);

  /// Wait for the device and hand every empty chunk back. What a caller does when it is done with
  /// a piece of work -- a whole image, say -- so that what the work needed at its peak is not kept
  /// until the next.
  void releaseUnusedMemory();

  /// The most the allocator may hold, in bytes -- its chunks and the weights' buffers; 0, the
  /// default, for no limit but the device's. A tensor that fits in a chunk already held is cut
  /// from it whatever the limit; one that needs a new chunk or a buffer of its own past it is
  /// refused by allocate(). Lowering it below what is held lets nothing go: what is in use stays,
  /// and the limit holds from the next allocation on.
  void setMemoryLimit(int64_t bytes);
  int64_t getMemoryLimit();

  /// The bytes the allocator holds from the device -- its chunks, in use or not, and the weights'
  /// buffers -- which is what the limit counts.
  int64_t getHeldBytes();

  /// The bytes the tensors hold: each one's size rounded up to the allocator's alignment, and none
  /// of the free blocks or empty chunks beside them. What a snapshot calls allocated.
  int64_t getAllocatedBytes();

  /// The device's budget, and what it leaves after everything this process has on the GPU,
  /// whatever the limit.
  MemorySnapshot captureMemorySnapshot();

  /// The limit, or the budget without one, and what the tensors hold against it.
  MemoryBudget getMemoryBudget();
  void resetPeakMemoryStats();

 private:
  std::mutex _mutex;

  MTL::Device *_device;
  int64_t _budgetBytes;
  std::shared_ptr<CommandStream> _stream;

  // The chunks the tensors are cut from, and what the tensors hold.
  BlockAllocator _blocks;
  int64_t _allocatedBytes = 0;
  int64_t _peakAllocatedBytes = 0;
  int64_t _memoryLimit = 0;  // 0 for none

  // The bytes of the weights' buffers, which are not cut from chunks.
  int64_t _weightBytes = 0;

  // A weight's buffer of its own, of at least `bytes` bytes.
  Buffer allocateWeight(int64_t bytes);

  // Waits for the device, then hands every empty chunk back to it.
  void synchronizeAndReleaseEmptyChunks();

  // What the limit counts: the chunks and the weights' buffers.
  int64_t held() const;
};

}  // namespace metal_native
}  // namespace op
}  // namespace fl
