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
#include <mutex>
#include <set>
#include <string>

#include "flint/metal_native/vram_allocator.h"

// metal-cpp's types, named and not included: only the .cc files talk to Metal.
namespace MTL {
class CommandBuffer;
class CommandQueue;
class ComputeCommandEncoder;
class ComputePipelineState;
class Device;
class Fence;
}  // namespace MTL

namespace fl {
namespace op {
namespace metal_native {

/// The work asked of a Metal device, in the order it was asked for.
///
/// Work is encoded into a command buffer rather than run as it is asked for, and committed a
/// batch at a time. Every dispatch is preceded by a memory barrier, and each command buffer waits
/// on a fence the one before it updates, so the order work was asked for in is the order it
/// happens in and every result is there for whatever is asked for after it. Buffers are untracked:
/// Metal is told nothing about which dispatch reads what, because that order already says it.
///
/// That order is also what lets a tensor's buffer go back to the allocator the moment the tensor
/// is destroyed, even though work that reads it may still be queued: anything that is later handed
/// the same buffer is encoded after that work, and so runs after it too. The host waits only where
/// it reads or writes the bytes itself, and in synchronize().
///
/// Batches are numbered from 1 as they are committed. A buffer remembers the number of the last
/// batch that used it, and wait() waits for that batch alone.
class CommandStream {
 public:
  /// What a kernel is handed for one of its tensors: a buffer, and where in it the tensor starts.
  struct Arg {
    const Buffer *buffer;
    int64_t offset;
  };

  /// The work of `device`, on a queue of its own.
  /// @throw lut::AbortedError if the device will not give one.
  explicit CommandStream(MTL::Device *device);
  CommandStream(const CommandStream &) = delete;
  CommandStream &operator=(const CommandStream &) = delete;

  /// Encode a dispatch of `pipeline` over `threads` threads in threadgroups of `groupSize`, the
  /// last one cut short where `threads` is not a multiple of it. `args` are bound to buffers 0, 1,
  /// ... in order and `params`, when there are any, to the buffer after them.
  void dispatch(
      MTL::ComputePipelineState *pipeline,
      std::initializer_list<Arg> args,
      const void *params,
      size_t paramsSize,
      int64_t threads,
      int groupSize);

  /// The same over a grid of whole threadgroups, for a kernel whose threadgroups cooperate.
  void dispatchGroups(
      MTL::ComputePipelineState *pipeline,
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

  /// Wait until the batch numbered `serial` has finished -- committing it first, if it is the one
  /// being encoded -- and with it every one before it. Nothing to wait for where `serial` is 0.
  /// @throw lut::AbortedError if the device failed a batch.
  void wait(uint64_t serial);

 private:
  struct Submitted {
    MTL::CommandBuffer *commandBuffer = nullptr;
    uint64_t serial = 0;
  };

  // Guarded by _mutex: the queue, and what is being encoded and has been committed.
  std::mutex _mutex;

  MTL::CommandQueue *_queue = nullptr;
  MTL::Fence *_fence = nullptr;

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

  // Starts encoding if nothing is, binds `args` and `params`, and leaves the encoder ready for
  // the dispatch itself. Every buffer bound is marked as used by the batch being encoded.
  MTL::ComputeCommandEncoder *beginCommand(
      MTL::ComputePipelineState *pipeline,
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
  // The number of the last batch known to have finished, with every one before it.
  uint64_t getCompletedSerial();
};

}  // namespace metal_native
}  // namespace op
}  // namespace fl
