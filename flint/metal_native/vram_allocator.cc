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

#include "flint/metal_native/vram_allocator.h"

#include <algorithm>

#include <Metal/Metal.hpp>

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "flint/metal_native/command_stream.h"

namespace fl {
namespace op {
namespace metal_native {

namespace {

// The size of a chunk, which tensors are cut from; a tensor larger than that has a chunk of its
// own, rounded up to the granule.
constexpr int64_t kChunkSize = 512 << 20;
constexpr int64_t kChunkGranule = 2 << 20;

// What the device rounds a buffer of its own up to: a weight's is counted as this many bytes at
// least, before it is made and its real size can be asked.
constexpr int64_t kPage = 16 << 10;

// Making a chunk larger than this first hands back the empty chunks larger than it. See
// VramAllocator::allocate.
constexpr int64_t kLargeChunk = 512 << 20;

// Every buffer is shared memory, which the host addresses directly, and untracked: the order
// work is encoded in is what keeps it from racing, see Context.
constexpr MTL::ResourceOptions kBufferOptions =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked;

}  // namespace

VramAllocator::VramAllocator(
    MTL::Device *device,
    int64_t budgetBytes,
    std::shared_ptr<CommandStream> stream)
    : _device(device),
      _budgetBytes(budgetBytes),
      _stream(std::move(stream)) {
}

Buffer VramAllocator::allocate(int64_t bytes, AllocHint hint) {
  CHECK(bytes > 0);
  if (hint == AllocHint::kWeight) return allocateWeight(bytes);
  int64_t size = BlockAllocator::roundUp(bytes);

  std::lock_guard<std::mutex> lock(_mutex);
  // From what a chunk has free, or else from the chunk of its own a tensor larger than a chunk
  // once had -- if this one fills four fifths of it at least. Such a chunk is kept for one tensor
  // at a time, as large as the one it was made for or nearly: cut up by small ones, it could go
  // back to the device only once all of them had, and the next large tensor -- the same one in
  // the next step, as often as not -- would need another.
  Block *block = _blocks.take(size);
  if (!block) block = _blocks.takeDedicated(size, size * 5 / 4);
  if (!block) {
    // Nothing free is large enough: a new chunk, kChunkSize unless the tensor is larger, and then
    // the tensor's own.
    int64_t needed = (size + kChunkGranule - 1) / kChunkGranule * kChunkGranule;
    int64_t chunkSize = std::max(kChunkSize, needed);
    bool dedicated = needed > kChunkSize;

    // Within the limit, if there is one: the empty chunks go back first where a new chunk would
    // pass it. A whole chunk where there is room for one, and
    // otherwise what this tensor needs and no more: what is left of the limit is for what comes
    // next, the weights among it, which are not cut from chunks. Where even the tensor does not
    // fit, nothing is asked of the device.
    if (_memoryLimit > 0) {
      if (held() + chunkSize > _memoryLimit) synchronizeAndReleaseEmptyChunks();
      int64_t room = (_memoryLimit - held()) / kChunkGranule * kChunkGranule;
      if (needed > room) {
        throw lut::OutOfMemoryError(lut::sprintf(
            "%d bytes more would take the Metal device past its limit of %d bytes (%d held, %d "
            "by tensors)",
            needed,
            _memoryLimit,
            held(),
            _allocatedBytes));
      }
      if (chunkSize > room) chunkSize = needed;
    }

    // A tensor larger than a chunk has one of its own, and the empty chunks there are -- every one
    // too small for it or too large for it to fill, or it would have been taken -- go back before
    // it is made, once the device is done with them, rather than sit beside it until the work is
    // over. That is where a model's stages change: the denoiser's chunks, say, when the decoder's
    // first gigabyte comes. A tensor that size that fills an empty chunk of its own takes it
    // instead, and nothing waits.
    if (chunkSize > kLargeChunk) synchronizeAndReleaseEmptyChunks();

    // Past the budget, every empty chunk goes back before more is asked for.
    if (static_cast<int64_t>(_device->currentAllocatedSize()) + chunkSize > _budgetBytes) {
      synchronizeAndReleaseEmptyChunks();
    }

    MTL::Buffer *created = _device->newBuffer(static_cast<NS::UInteger>(chunkSize), kBufferOptions);
    if (!created) {
      synchronizeAndReleaseEmptyChunks();
      created = _device->newBuffer(static_cast<NS::UInteger>(chunkSize), kBufferOptions);
    }
    if (!created) {
      throw lut::OutOfMemoryError(lut::sprintf(
          "the Metal device could not allocate %d bytes (%d held by tensors)",
          chunkSize,
          _allocatedBytes));
    }

    _blocks.addChunk(created, chunkSize, dedicated);
    block = dedicated ? _blocks.takeDedicated(size, chunkSize) : _blocks.take(size);
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

Buffer VramAllocator::allocateWeight(int64_t bytes) {
  int64_t size = BlockAllocator::roundUp(bytes);
  int64_t pages = (size + kPage - 1) / kPage * kPage;

  std::lock_guard<std::mutex> lock(_mutex);

  // Within the limit: the empty chunks go back first, and past it, nothing is asked of the device.
  if (_memoryLimit > 0 && held() + pages > _memoryLimit) {
    synchronizeAndReleaseEmptyChunks();
    if (held() + pages > _memoryLimit) {
      throw lut::OutOfMemoryError(lut::sprintf(
          "a weight of %d bytes would take the Metal device past its limit of %d bytes (%d held, "
          "%d by tensors)",
          size,
          _memoryLimit,
          held(),
          _allocatedBytes));
    }
  }

  // Past the budget, every empty chunk goes back before more is asked for, as for a chunk.
  if (static_cast<int64_t>(_device->currentAllocatedSize()) + pages > _budgetBytes) {
    synchronizeAndReleaseEmptyChunks();
  }

  MTL::Buffer *created = _device->newBuffer(static_cast<NS::UInteger>(size), kBufferOptions);
  if (!created) {
    synchronizeAndReleaseEmptyChunks();
    created = _device->newBuffer(static_cast<NS::UInteger>(size), kBufferOptions);
  }
  if (!created) {
    throw lut::OutOfMemoryError(lut::sprintf(
        "the Metal device could not allocate a weight of %d bytes (%d held by tensors)",
        size,
        _allocatedBytes));
  }

  Buffer buffer = std::make_shared<BufferState>();
  buffer->buffer = created;
  buffer->size = size;
  buffer->type = BufferType::kWeight;
  _weightBytes += static_cast<int64_t>(created->allocatedSize());

  _allocatedBytes += buffer->size;
  _peakAllocatedBytes = std::max(_peakAllocatedBytes, _allocatedBytes);
  return buffer;
}

void VramAllocator::free(const Buffer &buffer) {
  std::lock_guard<std::mutex> lock(_mutex);
  _allocatedBytes -= buffer->size;

  switch (buffer->type) {
    case BufferType::kCached:
      _blocks.give(buffer->block, buffer->lastUse);
      buffer->block = nullptr;
      break;
    case BufferType::kWeight:
      // Back to the device whole -- but not before the work that last read it, which may still be
      // queued: the command buffers do not keep what they use. Waited for here, from a destructor,
      // so a device that has failed is said in the log rather than thrown: the buffer is kept
      // then, since nothing can say the device is done with it.
      try {
        _stream->wait(buffer->lastUse);
      } catch (const lut::Error &e) {
        LOG(ERROR) << "a weight's buffer is kept, the device having failed: " << e.what();
        buffer->buffer = nullptr;
        break;
      }
      _weightBytes -= static_cast<int64_t>(buffer->buffer->allocatedSize());
      buffer->buffer->release();
      buffer->buffer = nullptr;
      break;
  }
}

void VramAllocator::synchronizeAndReleaseEmptyChunks() {
  // The command buffers do not retain what they use, which is why the device has to be done with
  // a chunk before it goes.
  _stream->synchronize();
  for (Chunk *chunk : _blocks.emptyChunks()) {
    MTL::Buffer *buffer = static_cast<MTL::Buffer *>(chunk->handle);
    _blocks.removeChunk(chunk);
    buffer->release();
  }
}

int64_t VramAllocator::held() const {
  return _blocks.getHeldBytes() + _weightBytes;
}

void VramAllocator::releaseUnusedMemory() {
  std::lock_guard<std::mutex> lock(_mutex);
  synchronizeAndReleaseEmptyChunks();
}

void VramAllocator::setMemoryLimit(int64_t bytes) {
  CHECK(bytes >= 0);
  std::lock_guard<std::mutex> lock(_mutex);
  _memoryLimit = bytes;
}

int64_t VramAllocator::getMemoryLimit() {
  std::lock_guard<std::mutex> lock(_mutex);
  return _memoryLimit;
}

int64_t VramAllocator::getHeldBytes() {
  std::lock_guard<std::mutex> lock(_mutex);
  return held();
}

int64_t VramAllocator::getAllocatedBytes() {
  std::lock_guard<std::mutex> lock(_mutex);
  return _allocatedBytes;
}

MemorySnapshot VramAllocator::captureMemorySnapshot() {
  std::lock_guard<std::mutex> lock(_mutex);

  // The device's budget and everything this process has on the GPU: one pool of memory for the
  // host and the device. The limit is the allocator's and not the device's, and is
  // getMemoryBudget()'s to say.
  int64_t used = static_cast<int64_t>(_device->currentAllocatedSize());
  int64_t free = _budgetBytes - used;
  return MemorySnapshot(
      _budgetBytes,
      std::max<int64_t>(0, free),
      _allocatedBytes,
      _peakAllocatedBytes);
}

MemoryBudget VramAllocator::getMemoryBudget() {
  std::lock_guard<std::mutex> lock(_mutex);
  MemoryBudget budget;
  budget.capacity = _memoryLimit > 0 ? _memoryLimit : _budgetBytes;
  // What the tensors hold, weights among them, and not the empty chunks: those go back before
  // anything past the limit is refused.
  budget.allocated = _allocatedBytes;
  return budget;
}

void VramAllocator::resetPeakMemoryStats() {
  std::lock_guard<std::mutex> lock(_mutex);
  _peakAllocatedBytes = _allocatedBytes;
}

}  // namespace metal_native
}  // namespace op
}  // namespace fl
