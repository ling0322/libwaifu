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

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "volk.h"

#include "vk_mem_alloc.h"
#include "flint/memory.h"

namespace fl {
namespace op {
namespace vulkan {

/// Every kernel is handed its arguments as push constants, and this many bytes is what every
/// Vulkan implementation is required to allow.
constexpr int kPushConstantSize = 128;

/// A buffer on the device: what one tensor's storage is.
///
/// Kernels never see the VkBuffer. They are given `address`, the buffer's device address, and
/// read and write through it, which is what lets every kernel share one pipeline layout with no
/// descriptors in it.
struct Buffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation allocation = nullptr;
  VkDeviceAddress address = 0;
  int64_t size = 0;  // what was allocated, which is the size asked for rounded up
};

/// What was learned about the device when it was opened.
struct DeviceInfo {
  std::string name;
  uint32_t maxWorkGroupCount[3];
  uint32_t subgroupSize;
  int64_t deviceLocalBytes;
  bool memoryBudget;

  /// Whether the device multiplies 16 x 16 x 16 tiles of half precision into float through
  /// VK_KHR_cooperative_matrix -- tensor cores, on the cards that have them -- with the subgroup
  /// of 32 the kernels that use it are written for.
  bool cooperativeMatrix;
};

/// The Vulkan device, and everything that is shared by the tensors on it: the allocator, the
/// queue its work goes to and the kernels it has built.
///
/// Work is recorded into a command buffer rather than run as it is asked for, and handed to the
/// queue a batch at a time; every command waits for all of those recorded before it, so the
/// order work was asked for in is the order it happens in. The host only ever waits in
/// synchronize() and in download(), which is where a result has to exist.
///
/// That order is also what lets a tensor's buffer go back to the cache the moment the tensor is
/// destroyed, even though work that reads it may still be queued: anything that is later handed
/// the same buffer is recorded after that work, and so runs after it too.
class Context {
 public:
  /// Whether the Vulkan loader can be opened and there is a device this backend can run on.
  static bool isAvailable();

  /// The context, opened the first time it is asked for and kept open until the process exits.
  /// @throw lut::AbortedError if there is no usable device.
  static std::shared_ptr<Context> get();

  ~Context();

  const DeviceInfo &getInfo() const {
    return _info;
  }

  /// A device buffer of at least `bytes` bytes, from the cache when one there is close enough.
  /// @throw lut::AbortedError when the device is out of memory even after the cache is emptied.
  Buffer allocate(int64_t bytes);

  /// Hand `buffer` back to the cache. It is not destroyed, see the class comment.
  void free(const Buffer &buffer);

  /// Wait for the device and destroy every buffer in the cache.
  void releaseUnusedMemory();

  MemorySnapshot captureMemorySnapshot();
  void resetPeakMemoryStats();

  /// Record a dispatch of the kernel called `kernel` over the given number of workgroups, with
  /// `push` as its arguments.
  void dispatch(
      const char *kernel,
      const void *push,
      size_t pushSize,
      uint32_t groupsX,
      uint32_t groupsY = 1,
      uint32_t groupsZ = 1);

  /// Record a dispatch of `kernel` over `numThreads` threads in workgroups of `groupSize`, folding
  /// the workgroups into a second dimension when there are more than one dimension may hold. The
  /// kernel works out its thread from gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x.
  void dispatchLinear(
      const char *kernel,
      const void *push,
      size_t pushSize,
      int64_t numThreads,
      int groupSize = 256);

  /// Record a fill of `bytes` bytes at `offset` with the 32-bit pattern `value`. Both must be
  /// multiples of four.
  void fillBuffer(const Buffer &buffer, int64_t offset, int64_t bytes, uint32_t value);

  /// Record a copy between two device buffers.
  void copyBuffer(
      const Buffer &src,
      int64_t srcOffset,
      const Buffer &dest,
      int64_t destOffset,
      int64_t bytes);

  /// Copy `bytes` bytes of host memory into `dest` at `offset`. The host memory may be reused as
  /// soon as this returns: it is staged, and the copy out of the staging buffer is recorded like
  /// any other work.
  void upload(const void *src, const Buffer &dest, int64_t offset, int64_t bytes);

  /// Copy `bytes` bytes at `offset` in `src` out to host memory, after all work recorded so far.
  void download(const Buffer &src, int64_t offset, void *dest, int64_t bytes);

  /// Wait until all recorded work has finished.
  void synchronize();

 private:
  struct Batch {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    uint64_t serial = 0;  // which submission this was, counting from one

    // Only when profiling: a timestamp either side of every command, and what each command was.
    VkQueryPool timestamps = VK_NULL_HANDLE;
    std::vector<std::string> labels;
  };

  struct ProfileEntry {
    int64_t count = 0;
    double nanoseconds = 0.0;
  };

  struct Kernel {
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
  };

  // Everything below is guarded by _mutex. It is recursive because allocate() empties the cache
  // through synchronize() when the device runs out.
  std::recursive_mutex _mutex;

  VkInstance _instance = VK_NULL_HANDLE;
  VkPhysicalDevice _physicalDevice = VK_NULL_HANDLE;
  VkDevice _device = VK_NULL_HANDLE;
  VkQueue _queue = VK_NULL_HANDLE;
  uint32_t _queueFamily = 0;
  VmaAllocator _allocator = nullptr;
  VkCommandPool _commandPool = VK_NULL_HANDLE;
  VkPipelineLayout _pipelineLayout = VK_NULL_HANDLE;
  DeviceInfo _info;

  // FLINT_VULKAN_PROFILE=1: how long the device spent on each kernel, printed at exit.
  bool _profile = false;
  double _timestampPeriod = 1.0;
  std::map<std::string, ProfileEntry> _profileTotals;

  std::unordered_map<std::string, Kernel> _kernels;

  // The batch being recorded, the ones submitted and not yet known to be finished (oldest first),
  // and finished ones kept to be recorded into again.
  Batch _recording;
  int _numRecorded = 0;
  std::deque<Batch> _submitted;
  std::vector<Batch> _idle;
  uint64_t _submittedSerial = 0;
  uint64_t _completedSerial = 0;

  // The ring uploads are staged through: one host buffer, mapped for as long as the device is
  // open and cut into kStagingSegments segments, filled one after another. A segment is written
  // again only once the batch that last copied out of it has finished. Created on first upload.
  static constexpr int kStagingSegments = 4;
  static constexpr int64_t kStagingSegmentBytes = 32 << 20;
  Buffer _staging;
  std::byte *_stagingMemory = nullptr;
  int _stagingSegment = 0;
  int64_t _stagingOffset = 0;
  uint64_t _segmentSerial[kStagingSegments] = {};

  // The buffer cache, by size, and what the tensors hold.
  std::multimap<int64_t, Buffer> _cache;
  int64_t _cachedBytes = 0;
  int64_t _allocatedBytes = 0;
  int64_t _peakAllocatedBytes = 0;

  Context() = default;

  // Opens the device, or throws saying why not.
  void open();

  const Kernel &getKernel(const char *name);

  Buffer createBuffer(
      int64_t bytes,
      VkBufferUsageFlags usage,
      VmaMemoryUsage memoryUsage,
      VmaAllocationCreateFlags flags);
  void destroyBuffer(const Buffer &buffer);

  // Starts recording if nothing is, and orders what is recorded next after everything before it.
  // `label` is what the command is called in a profile.
  VkCommandBuffer beginCommand(const std::string &label);
  void endCommand();
  void submit();
  void retire(Batch &batch);

  // Moves the staging ring on to its next segment, waiting for the device to be done with it.
  void nextStagingSegment();
  void emptyCache();
  void printProfile();
};

}  // namespace vulkan
}  // namespace op
}  // namespace fl
