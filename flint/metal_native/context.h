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

#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "flint/metal_native/command_stream.h"
#include "flint/metal_native/vram_allocator.h"

// metal-cpp's types, named and not included: only the .cc files talk to Metal.
namespace MTL {
class ComputePipelineState;
class Device;
class Library;
}  // namespace MTL

namespace fl {
namespace op {
namespace metal_native {

/// What was learned about the device when it was opened.
struct DeviceInfo {
  std::string name;

  /// What the system suggests one process keep on the GPU at once. It is the budget: past it,
  /// the cache is emptied before anything new is asked for.
  int64_t workingSetBytes;
};

/// The Metal device, and everything that is shared by the tensors on it: the kernels it has built,
/// the stream its work goes to and the allocator its tensors' memory comes from. The Vulkan
/// backend's Context, in Metal's terms.
///
/// How work is ordered is the stream's; see CommandStream.
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

  /// Where the work on the device goes.
  CommandStream &getStream() {
    return *_stream;
  }

  /// Where the tensors on the device get their storage.
  VramAllocator &getAllocator() {
    return *_allocator;
  }

  /// Encode a dispatch of the kernel called `kernel` over `threads` threads in threadgroups of
  /// `groupSize`, the last one cut short where `threads` is not a multiple of it. `args` are bound
  /// to buffers 0, 1, ... in order and `params`, when there are any, to the buffer after them.
  void dispatch(
      const char *kernel,
      std::initializer_list<CommandStream::Arg> args,
      const void *params,
      size_t paramsSize,
      int64_t threads,
      int groupSize = 256);

  /// The same over a grid of whole threadgroups, for a kernel whose threadgroups cooperate: a row
  /// reduced by a group of threads, a tile of a product.
  void dispatchGroups(
      const char *kernel,
      std::initializer_list<CommandStream::Arg> args,
      const void *params,
      size_t paramsSize,
      uint32_t groupsX,
      uint32_t groupsY,
      uint32_t groupsZ,
      uint32_t threadsPerGroup);

  /// The stream's: see CommandStream.
  void upload(const void *src, const Buffer &dest, int64_t offset, int64_t bytes) {
    _stream->upload(src, dest, offset, bytes);
  }
  void download(const Buffer &src, int64_t offset, void *dest, int64_t bytes) {
    _stream->download(src, offset, dest, bytes);
  }
  void synchronize() {
    _stream->synchronize();
  }

 private:
  MTL::Device *_device = nullptr;
  MTL::Library *_library = nullptr;
  MTL::Library *_naxLibrary = nullptr;  // kernels/nax/, where they run
  bool _naxEnabled = true;
  DeviceInfo _info;

  // The kernels built so far, by name. Guarded by _kernelMutex, which is never held while the
  // stream or the allocator is asked anything.
  std::mutex _kernelMutex;
  std::unordered_map<std::string, MTL::ComputePipelineState *> _kernels;

  // The stream is the allocator's as well as the context's: it waits there for memory to be done
  // with.
  std::shared_ptr<CommandStream> _stream;
  std::unique_ptr<VramAllocator> _allocator;

  Context() = default;

  // Opens the device, or throws saying why not.
  void open();

  MTL::ComputePipelineState *getKernel(const char *name);
};

}  // namespace metal_native
}  // namespace op
}  // namespace fl
