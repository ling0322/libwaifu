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

#include "flint/device.h"

namespace fl {

/// What a tensor's storage is for, which an allocator may treat each in its own way.
///
/// What tells them apart is how long the storage lives. A weight is kept until it is let go of --
/// to make room for another, say -- and is given memory that goes back to the device whole when it
/// is: on Metal a buffer of its own, on CUDA (from 2 MB) its own cudaMalloc rather than a block of
/// the pool, which a large allocation could not have back while small blocks beside it stayed in
/// use. Everything else lives for a pass at most and is cut from the memory kept for reuse. Vulkan
/// gives every tensor a buffer of its own and ignores it.
enum class AllocHint : int32_t {
  /// What a pass makes and drops again: the outputs of operators, and their workspace.
  kActivation = 0,
  /// A model's parameter kept on the device, until it is let go of.
  kWeight = 1,
  /// Anything else, treated as an activation.
  kOther = 2,
  /// A copy of a weight that waits off the device, made for the instructions of one pass that
  /// read it and dropped after the last of them: an activation, by how long it lives.
  kTemporaryWeight = 3,
};

/// @brief A snapshot of the memory usage of one device. Devices that do not report their memory
/// usage (the CPU backend) report zero for every value.
class MemorySnapshot {
 public:
  /// @brief Capture the current memory usage of a device.
  /// @param device The device to measure.
  /// @return The captured snapshot.
  static MemorySnapshot capture(Device device);

  /// @brief Set the peak allocated bytes of a device back to zero.
  /// @param device The device to reset.
  static void resetPeakStats(Device device);

  /// @brief Give every byte that no tensor is using back to the driver, so that another process
  /// may have it. Devices whose allocator already does this as each tensor goes (the CPU backend)
  /// do nothing. Worth calling only where something large has just been let go of and nothing is
  /// about to ask for it again; between two runs of one model it hands back the blocks the next
  /// run would have reused.
  /// @param device The device to hand memory back on.
  static void releaseUnused(Device device);

  /// @brief Limit the memory a device's allocator holds to `bytes`, 0 for no limit but the
  /// device's: an allocation that would pass it throws lut::OutOfMemoryError. Metal, CUDA and
  /// Vulkan have one; the CPU refuses. See MemoryBudget for what the allocator then has left.
  /// @param device The device to limit.
  /// @param bytes The limit, at least 0.
  static void setLimit(Device device, int64_t bytes);

  MemorySnapshot(
      int64_t totalMemory,
      int64_t freeMemory,
      int64_t allocatedMemory,
      int64_t peakAllocatedMemory);

  /// @brief Get the total memory of the device.
  /// @return The total memory in bytes.
  int64_t getTotalMemory() const;

  /// @brief Get the memory of the device that no process reserved yet. Memory that this process
  /// already took from the driver is not free even once its tensors are destroyed, because the
  /// allocator keeps it for reuse.
  /// @return The free memory in bytes.
  int64_t getFreeMemory() const;

  /// @brief Get the bytes that the tensors of this process currently hold.
  /// @return The allocated memory in bytes.
  int64_t getAllocatedMemory() const;

  /// @brief Get the largest value getAllocatedMemory() reached since the last resetPeakStats().
  /// @return The peak allocated memory in bytes.
  int64_t getPeakAllocatedMemory() const;

 private:
  int64_t _totalMemory;
  int64_t _freeMemory;
  int64_t _allocatedMemory;
  int64_t _peakAllocatedMemory;
};

/// @brief What one device's allocator may hold and what it holds now, which is what a model's
/// weights are sized against -- not a MemorySnapshot, which is the device as a person looking at
/// it would see it, other processes and memory kept for reuse included.
struct MemoryBudget {
  /// @brief Capture the budget of a device's allocator.
  static MemoryBudget capture(Device device);

  /// The most the allocator may hold: its memory limit where it has one, and otherwise what the
  /// device has -- a card's memory, or on Metal the share of the machine's the system suggests
  /// for the GPU. 0 for a device that does not say, which is what the CPU does.
  int64_t capacity = 0;

  /// What the tensors of this process hold against it now -- and on CUDA the operators'
  /// workspaces. Not what an allocator keeps for reuse: Vulkan's cache and Metal's empty chunks go
  /// back before anything past the limit is refused, and CUDA's pool is not counted at all.
  int64_t allocated = 0;

  /// What the allocator holds from the device: what `allocated` counts, and what it keeps for
  /// reuse besides -- CUDA's pool, Vulkan's cache, Metal's chunks, in use or not. Far more than
  /// `allocated` is memory the allocator has and cannot hand out, which is what to look at when
  /// an allocation fails with room to spare.
  int64_t held = 0;
};

}  // namespace fl
