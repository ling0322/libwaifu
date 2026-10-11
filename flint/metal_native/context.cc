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

#include "flint/metal_native/context.h"

#include <dispatch/dispatch.h>
#include <string.h>

#include <algorithm>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "flint/metal_native/kernels.h"

namespace fl {
namespace op {
namespace metal_native {

namespace {

// Committed batches are waited for once there are more than this many, which is what keeps the
// host from encoding arbitrarily far ahead of the device.
constexpr uint64_t kMaxBatchesInFlight = 3;

// A batch is committed once it holds this many commands, so the device starts on the first of a
// long run of work while the rest is still being encoded.
constexpr int kCommandsPerBatch = 64;

// The size of a chunk, which tensors are cut from; a tensor larger than that has a chunk of its
// own, rounded up to the granule.
constexpr int64_t kChunkSize = 512 << 20;
constexpr int64_t kChunkGranule = 2 << 20;

// Making a chunk larger than this first hands back the empty chunks larger than it. See
// Context::allocate.
constexpr int64_t kLargeChunk = 512 << 20;

// Every buffer is shared memory, which the host addresses directly, and untracked: the order
// work is encoded in is what keeps it from racing, see the class comment.
constexpr MTL::ResourceOptions kBufferOptions =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked;

// What metal-cpp hands back autoreleased has to be made inside a pool, or it is never released.
class Pool {
 public:
  Pool()
      : _pool(NS::AutoreleasePool::alloc()->init()) {
  }
  ~Pool() {
    _pool->release();
  }

 private:
  NS::AutoreleasePool *_pool;
};

// Whether `device` has neural accelerators that kernels/nax/ can run on: Metal 4's tensor
// operations, which are macOS 26.2's, and a GPU of the generation that added the accelerators --
// 17, or 18 for the variant whose architecture name ends in p -- read off a name such as
// applegpu_g17s. The same test MLX makes. And a library to load, which a flint built without
// Metal 4 does not have.
bool hasNeuralAccelerators(MTL::Device *device) {
  if (kNaxMetallibSize == 0) return false;
  if (__builtin_available(macOS 26.2, *)) {
  } else {
    return false;
  }

  std::string name = device->architecture()->name()->utf8String();
  size_t g = name.rfind('g');
  if (g == std::string::npos || g + 1 >= name.size()) return false;
  int generation = atoi(name.c_str() + g + 1);
  char variant = name.back();
  return generation >= (variant == 'p' ? 18 : 17);
}

std::string describe(NS::Error *error) {
  if (!error) return "no reason given";
  return error->localizedDescription()->utf8String();
}

// The context, once opened, and never destroyed: the operators live in a static array whose
// destructor runs at exit, after the driver may have gone through exit handlers of its own.
// Everything the device holds goes back when the process does anyway.
std::mutex gContextMutex;
std::shared_ptr<Context> *gContext = nullptr;

}  // namespace

bool Context::isAvailable() {
  static const bool available = []() {
    try {
      get();
      return true;
    } catch (const lut::Error &e) {
      LOG(INFO) << "Metal is not available: " << e.what();
      return false;
    }
  }();
  return available;
}

std::shared_ptr<Context> Context::get() {
  std::lock_guard<std::mutex> lock(gContextMutex);
  if (!gContext) {
    std::shared_ptr<Context> context(new Context());
    context->open();
    gContext = new std::shared_ptr<Context>(std::move(context));
  }
  return *gContext;
}

void Context::open() {
  Pool pool;

  _device = MTL::CreateSystemDefaultDevice();
  if (!_device) throw lut::AbortedError("there is no Metal device");

  // simdgroup matrices, which the products are written in, are what family 7 -- the M1 and
  // everything after it -- added.
  if (!_device->supportsFamily(MTL::GPUFamilyApple7)) {
    throw lut::AbortedError(lut::sprintf(
        "%s is not an Apple GPU of family 7 or later, which this backend needs",
        _device->name()->utf8String()));
  }

  _queue = _device->newCommandQueue();
  if (!_queue) throw lut::AbortedError("could not create a Metal command queue");

  // The kernels were compiled into one library when flint was built, and are carried in it.
  dispatch_data_t data = dispatch_data_create(
      kMetallib,
      kMetallibSize,
      nullptr,
      DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  NS::Error *error = nullptr;
  _library = _device->newLibrary(data, &error);
  dispatch_release(data);
  if (!_library) {
    throw lut::AbortedError("could not load flint's Metal kernels: " + describe(error));
  }

  // And those for the neural accelerators, where they can run. They are only faster, so a GPU
  // that cannot load them runs the kernels above as every other does.
  if (hasNeuralAccelerators(_device)) {
    dispatch_data_t naxData = dispatch_data_create(
        kNaxMetallib,
        kNaxMetallibSize,
        nullptr,
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NS::Error *naxError = nullptr;
    _naxLibrary = _device->newLibrary(naxData, &naxError);
    dispatch_release(naxData);
    if (!_naxLibrary) {
      LOG(WARN) << "could not load flint's Metal kernels for the neural accelerators: "
                << describe(naxError);
    }
  }

  _fence = _device->newFence();

  _info.name = _device->name()->utf8String();
  _info.workingSetBytes = static_cast<int64_t>(_device->recommendedMaxWorkingSetSize());
  LOG(INFO) << "Metal device: " << _info.name << ", "
            << _info.workingSetBytes / (1 << 20) << " MB working set"
            << (_naxLibrary ? ", neural accelerators" : "");
}

MTL::ComputePipelineState *Context::getKernel(const char *name) {
  auto it = _kernels.find(name);
  if (it != _kernels.end()) return it->second;

  Pool pool;
  MTL::Library *library = strncmp(name, "nax_", 4) == 0 ? _naxLibrary : _library;
  if (!library) throw lut::AbortedError(lut::sprintf("no Metal kernel called %s here", name));
  MTL::Function *function = library->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
  if (!function) throw lut::AbortedError(lut::sprintf("no Metal kernel called %s", name));

  NS::Error *error = nullptr;
  MTL::ComputePipelineState *pipeline = _device->newComputePipelineState(function, &error);
  function->release();
  if (!pipeline) {
    throw lut::AbortedError(
        lut::sprintf("could not build the Metal kernel %s: %s", name, describe(error)));
  }

  return _kernels.emplace(name, pipeline).first->second;
}

Buffer Context::allocate(int64_t bytes) {
  CHECK(bytes > 0);
  int64_t size = BlockAllocator::roundUp(bytes);

  std::lock_guard<std::recursive_mutex> lock(_mutex);
  Block *block = _blocks.take(size);
  if (!block) {
    // Nothing free is large enough: a new chunk, kChunkSize unless the tensor is larger.
    int64_t chunkSize = std::max(
        kChunkSize,
        (size + kChunkGranule - 1) / kChunkGranule * kChunkGranule);

    // A tensor larger than a chunk has one of its own, and the empty chunks there are -- every one
    // too small for it, or take() would have found it -- go back before it is made, once the
    // device is done with them, rather than sit beside it until the work is over. That is where
    // a model's stages change: the denoiser's chunks, say, when the decoder's first gigabyte
    // comes. A tensor that size that fits in an empty chunk takes it instead, and nothing waits.
    if (chunkSize > kLargeChunk) {
      synchronize();
      releaseEmptyChunks();
    }

    // Past the budget, every empty chunk goes back before more is asked for.
    if (static_cast<int64_t>(_device->currentAllocatedSize()) + chunkSize > _info.workingSetBytes) {
      synchronize();
      releaseEmptyChunks();
    }

    MTL::Buffer *created = _device->newBuffer(static_cast<NS::UInteger>(chunkSize), kBufferOptions);
    if (!created) {
      synchronize();
      releaseEmptyChunks();
      created = _device->newBuffer(static_cast<NS::UInteger>(chunkSize), kBufferOptions);
    }
    if (!created) {
      throw lut::AbortedError(lut::sprintf(
          "out of memory: the Metal device could not allocate %d bytes (%d held by tensors)",
          chunkSize,
          _allocatedBytes));
    }


    _blocks.addChunk(created, chunkSize);
    block = _blocks.take(size);
    CHECK(block);
  }

  Buffer buffer = std::make_shared<BufferState>();
  buffer->buffer = static_cast<MTL::Buffer *>(block->chunk->handle);
  buffer->offset = block->offset;
  buffer->size = block->size;
  // Bytes another tensor had, which work still queued may use: the host waits for it.
  buffer->lastUse = block->lastUse;
  buffer->block = block;

  _allocatedBytes += buffer->size;
  _peakAllocatedBytes = std::max(_peakAllocatedBytes, _allocatedBytes);
  return buffer;
}

void Context::free(const Buffer &buffer) {
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  _allocatedBytes -= buffer->size;
  _blocks.give(buffer->block, buffer->lastUse);
  buffer->block = nullptr;
}

void Context::releaseEmptyChunks(int64_t largerThan) {
  // The command buffers do not retain what they use, which is why this comes only after
  // synchronize(): the device has to be done with a chunk before it goes.
  for (Chunk *chunk : _blocks.emptyChunks()) {
    if (chunk->size <= largerThan) continue;
    MTL::Buffer *buffer = static_cast<MTL::Buffer *>(chunk->handle);
    _blocks.removeChunk(chunk);
    buffer->release();
  }
}

void Context::releaseUnusedMemory() {
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  synchronize();
  releaseEmptyChunks();
}

MemorySnapshot Context::captureMemorySnapshot() {
  std::lock_guard<std::recursive_mutex> lock(_mutex);

  // One pool of memory for the host and the device, so what is free is what the budget leaves
  // after everything this process has on the GPU, the cache included.
  int64_t total = _info.workingSetBytes;
  int64_t free = total - static_cast<int64_t>(_device->currentAllocatedSize());
  return MemorySnapshot(total, std::max<int64_t>(0, free), _allocatedBytes, _peakAllocatedBytes);
}

void Context::resetPeakMemoryStats() {
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  _peakAllocatedBytes = _allocatedBytes;
}

MTL::ComputeCommandEncoder *Context::beginCommand(
    MTL::ComputePipelineState *kernel,
    std::initializer_list<Arg> args,
    const void *params,
    size_t paramsSize) {
  if (!_recording) {
    Pool pool;
    _recording = _queue->commandBufferWithUnretainedReferences()->retain();
    _encoder = _recording->computeCommandEncoder(MTL::DispatchTypeConcurrent)->retain();

    // After everything committed before: the batch before this one updated the fence last.
    if (_fenceUpdated) _encoder->waitForFence(_fence);
  }

  // Every command waits for every one before it in this encoder, and writes nothing until what
  // they wrote is there to be read. Coarser than saying which buffers each one touches, and it is
  // what makes the order work was asked for in the order it runs in, which the buffer cache
  // depends on.
  _encoder->memoryBarrier(MTL::BarrierScopeBuffers);
  _encoder->setComputePipelineState(kernel);

  uint64_t serial = _submittedSerial + 1;
  NS::UInteger index = 0;
  for (const Arg &arg : args) {
    const Buffer &buffer = *arg.buffer;
    _encoder->setBuffer(
        buffer->buffer, static_cast<NS::UInteger>(buffer->offset + arg.offset), index++);
    buffer->lastUse = serial;
  }
  if (paramsSize) _encoder->setBytes(params, static_cast<NS::UInteger>(paramsSize), index);

  return _encoder;
}

void Context::endCommand() {
  if (++_numRecorded >= kCommandsPerBatch) submit();
}

void Context::dispatch(
    const char *kernel,
    std::initializer_list<Arg> args,
    const void *params,
    size_t paramsSize,
    int64_t threads,
    int groupSize) {
  if (threads <= 0) return;

  std::lock_guard<std::recursive_mutex> lock(_mutex);
  MTL::ComputePipelineState *pipeline = getKernel(kernel);
  NS::UInteger width = std::min<NS::UInteger>(
      static_cast<NS::UInteger>(groupSize),
      pipeline->maxTotalThreadsPerThreadgroup());

  MTL::ComputeCommandEncoder *encoder = beginCommand(pipeline, args, params, paramsSize);
  encoder->dispatchThreads(
      MTL::Size(static_cast<NS::UInteger>(threads), 1, 1),
      MTL::Size(width, 1, 1));
  endCommand();
}

void Context::dispatchGroups(
    const char *kernel,
    std::initializer_list<Arg> args,
    const void *params,
    size_t paramsSize,
    uint32_t groupsX,
    uint32_t groupsY,
    uint32_t groupsZ,
    uint32_t threadsPerGroup) {
  if (groupsX == 0 || groupsY == 0 || groupsZ == 0) return;

  std::lock_guard<std::recursive_mutex> lock(_mutex);
  MTL::ComputePipelineState *pipeline = getKernel(kernel);
  CHECK(threadsPerGroup <= pipeline->maxTotalThreadsPerThreadgroup());

  MTL::ComputeCommandEncoder *encoder = beginCommand(pipeline, args, params, paramsSize);
  encoder->dispatchThreadgroups(
      MTL::Size(groupsX, groupsY, groupsZ),
      MTL::Size(threadsPerGroup, 1, 1));
  endCommand();
}

void Context::submit() {
  if (!_recording) return;

  // The next batch waits on this: see beginCommand().
  _encoder->updateFence(_fence);
  _encoder->endEncoding();
  _encoder->release();
  _encoder = nullptr;
  _fenceUpdated = true;

  uint64_t serial = ++_submittedSerial;
  _recording->addCompletedHandler([this, serial](MTL::CommandBuffer *commandBuffer) {
    std::lock_guard<std::mutex> lock(_completionMutex);
    if (commandBuffer->status() == MTL::CommandBufferStatusError && _error.empty()) {
      _error = describe(commandBuffer->error());
    }

    // Batches finish in the order they were committed on one queue; the handlers are not promised
    // to be called in it. What has finished is everything up to the first that has not.
    _finished.insert(serial);
    while (_finished.count(_completedSerial + 1)) {
      _finished.erase(_completedSerial + 1);
      ++_completedSerial;
    }

    _completed.notify_all();
  });
  _recording->commit();

  _submitted.push_back(Submitted{_recording, serial});
  _recording = nullptr;
  _numRecorded = 0;

  retireFinished();
  if (_submittedSerial > kMaxBatchesInFlight) waitFor(_submittedSerial - kMaxBatchesInFlight);
}

void Context::waitFor(uint64_t serial) {
  if (serial == 0) return;
  if (serial > _submittedSerial) submit();  // it is the batch still being encoded

  std::unique_lock<std::mutex> lock(_completionMutex);
  _completed.wait(lock, [&]() { return _completedSerial >= serial || !_error.empty(); });
  if (!_error.empty()) {
    // Nothing encoded after a failed batch can be trusted, and there is no recovering it.
    throw lut::AbortedError("the Metal device failed: " + _error);
  }
}

void Context::retireFinished() {
  uint64_t completed;
  {
    std::lock_guard<std::mutex> lock(_completionMutex);
    completed = _completedSerial;
  }
  while (!_submitted.empty() && _submitted.front().serial <= completed) {
    _submitted.front().commandBuffer->release();
    _submitted.pop_front();
  }
}

void Context::synchronize() {
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  submit();
  waitFor(_submittedSerial);
  retireFinished();
}

void Context::upload(const void *src, const Buffer &dest, int64_t offset, int64_t bytes) {
  if (bytes == 0) return;
  CHECK(offset >= 0 && offset + bytes <= dest->size);

  std::lock_guard<std::recursive_mutex> lock(_mutex);
  waitFor(dest->lastUse);
  memcpy(static_cast<std::byte *>(dest->buffer->contents()) + dest->offset + offset, src, bytes);
}

void Context::download(const Buffer &src, int64_t offset, void *dest, int64_t bytes) {
  if (bytes == 0) return;
  CHECK(offset >= 0 && offset + bytes <= src->size);

  std::lock_guard<std::recursive_mutex> lock(_mutex);
  waitFor(src->lastUse);
  memcpy(
      dest, static_cast<const std::byte *>(src->buffer->contents()) + src->offset + offset, bytes);
}

}  // namespace metal_native
}  // namespace op
}  // namespace fl
