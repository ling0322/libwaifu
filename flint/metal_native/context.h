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

#include <condition_variable>
#include <deque>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>

#include "flint/metal_native/vram_allocator.h"

// metal-cpp's types, named and not included: only context.cc talks to Metal.
namespace MTL {
class CommandBuffer;
class CommandQueue;
class ComputeCommandEncoder;
class ComputePipelineState;
class Device;
class Fence;
class Library;
}  // namespace MTL

namespace fl {
namespace op {
namespace metal_native {

/// What a kernel is handed for one of its tensors: a buffer, and where in it the tensor starts.
struct Arg {
  const Buffer *buffer;
  int64_t offset;
};

/// What was learned about the device when it was opened.
struct DeviceInfo {
  std::string name;

  /// What the system suggests one process keep on the GPU at once. It is the budget: past it,
  /// the cache is emptied before anything new is asked for.
  int64_t workingSetBytes;
};

/// The Metal device, and everything that is shared by the tensors on it: the queue its work goes
/// to, the kernels it has built and the allocator its tensors' memory comes from. The Vulkan backend's Context,
/// in Metal's terms.
///
/// Work is encoded into a command buffer rather than run as it is asked for, and committed a
/// batch at a time. Every dispatch is preceded by a memory barrier, and each command buffer waits
/// on a fence the one before it updates, so the order work was asked for in is the order it
/// happens in and every result is there for whatever is asked for after it. Buffers are untracked:
/// Metal is told nothing about which dispatch reads what, because that order already says it.
///
/// That order is also what lets a tensor's buffer go back to the cache the moment the tensor is
/// destroyed, even though work that reads it may still be queued: anything that is later handed
/// the same buffer is encoded after that work, and so runs after it too. The host waits only where
/// it reads or writes the bytes itself, and in synchronize().
class Context {
 public:
  /// Whether there is a Metal device this backend can run on.
  static bool isAvailable();

  /// The context, opened the first time it is asked for and kept open until the process exits.
  /// @throw lut::AbortedError if there is no usable device.
  static std::shared_ptr<Context> get();

  /// Whether the kernels for the neural accelerators run here: a GPU that has them, macOS 26.2 or
  /// later, and a flint built with Metal 4. The operators that have such a kernel use it then.
  bool hasNax() const {
    return _naxLibrary && _naxEnabled;
  }

  /// For tests: whether the kernels for the neural accelerators are used where there are any, so
  /// that the ones they stand in for can be tested on a GPU that has them too.
  void setNaxEnabled(bool enabled) {
    _naxEnabled = enabled;
  }

  const DeviceInfo &getInfo() const {
    return _info;
  }

  /// Where the tensors on the device get their storage.
  VRamAllocator &getAllocator() {
    return *_allocator;
  }

  /// Encode a dispatch of the kernel called `kernel` over `threads` threads in threadgroups of
  /// `groupSize`, the last one cut short where `threads` is not a multiple of it. `args` are bound
  /// to buffers 0, 1, ... in order and `params`, when there are any, to the buffer after them.
  void dispatch(
      const char *kernel,
      std::initializer_list<Arg> args,
      const void *params,
      size_t paramsSize,
      int64_t threads,
      int groupSize = 256);

  /// The same over a grid of whole threadgroups, for a kernel whose threadgroups cooperate: a row
  /// reduced by a group of threads, a tile of a product.
  void dispatchGroups(
      const char *kernel,
      std::initializer_list<Arg> args,
      const void *params,
      size_t paramsSize,
      uint32_t groupsX,
      uint32_t groupsY,
      uint32_t groupsZ,
      uint32_t threadsPerGroup);

  /// Copy `bytes` bytes of host memory into `dest` at `offset`, once nothing queued still uses
  /// `dest`. The host memory may be reused as soon as this returns.
  void upload(const void *src, const Buffer &dest, int64_t offset, int64_t bytes);

  /// Copy `bytes` bytes at `offset` in `src` out to host memory, after the work that last used it.
  void download(const Buffer &src, int64_t offset, void *dest, int64_t bytes);

  /// Wait until all encoded work has finished.
  void synchronize();

 private:
  struct Submitted {
    MTL::CommandBuffer *commandBuffer = nullptr;
    uint64_t serial = 0;
  };

  // Guarded by _mutex. Nothing is asked of the allocator while it is held: the allocator calls
  // synchronize() with its own lock held.
  std::mutex _mutex;

  MTL::Device *_device = nullptr;
  MTL::CommandQueue *_queue = nullptr;
  MTL::Library *_library = nullptr;
  MTL::Library *_naxLibrary = nullptr;  // kernels/nax/, where they run
  bool _naxEnabled = true;
  MTL::Fence *_fence = nullptr;
  DeviceInfo _info;

  std::unordered_map<std::string, MTL::ComputePipelineState *> _kernels;

  // The command buffer being encoded and its encoder, and the ones committed and not yet known to
  // be finished, oldest first.
  MTL::CommandBuffer *_recording = nullptr;
  MTL::ComputeCommandEncoder *_encoder = nullptr;
  int _numRecorded = 0;
  bool _fenceUpdated = false;
  std::deque<Submitted> _submitted;
  uint64_t _submittedSerial = 0;

  // Written by the completion handlers, on a thread of Metal's: guarded by _completionMutex, not
  // _mutex, so that a handler never waits for whoever is encoding.
  std::mutex _completionMutex;
  std::condition_variable _completed;
  uint64_t _completedSerial = 0;
  std::set<uint64_t> _finished;  // finished, but after one that has not
  std::string _error;

  std::unique_ptr<VRamAllocator> _allocator;

  Context() = default;

  // Opens the device, or throws saying why not.
  void open();

  MTL::ComputePipelineState *getKernel(const char *name);

  // Starts encoding if nothing is, binds `args` and `params`, and leaves the encoder ready for
  // the dispatch itself. Every buffer bound is marked as used by the batch being encoded.
  MTL::ComputeCommandEncoder *beginCommand(
      MTL::ComputePipelineState *kernel,
      std::initializer_list<Arg> args,
      const void *params,
      size_t paramsSize);
  void endCommand();
  void submit();

  // Waits until the batch numbered `serial` has finished, committing it first if it is the one
  // being encoded. Throws what the device reported if a batch failed.
  void waitFor(uint64_t serial);
  // Forgets committed batches that have finished.
  void retireFinished();
};

}  // namespace metal_native
}  // namespace op
}  // namespace fl
