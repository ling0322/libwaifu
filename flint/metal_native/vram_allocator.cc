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

namespace fl {
namespace op {
namespace metal_native {

namespace {

// The size of a chunk, which tensors are cut from; a tensor larger than that has a chunk of its
// own, rounded up to the granule.
constexpr int64_t kChunkSize = 512 << 20;
constexpr int64_t kChunkGranule = 2 << 20;

// Making a chunk larger than this first hands back the empty chunks larger than it. See
// VRamAllocator::allocate.
constexpr int64_t kLargeChunk = 512 << 20;

// Every buffer is shared memory, which the host addresses directly, and untracked: the order
// work is encoded in is what keeps it from racing, see Context.
constexpr MTL::ResourceOptions kBufferOptions =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked;

}  // namespace

VRamAllocator::VRamAllocator(
    MTL::Device *device,
    int64_t budgetBytes,
    std::function<void()> synchronize)
    : _device(device),
      _budgetBytes(budgetBytes),
      _synchronize(std::move(synchronize)) {
}

Buffer VRamAllocator::allocate(int64_t bytes) {
  CHECK(bytes > 0);
  int64_t size = BlockAllocator::roundUp(bytes);

  std::lock_guard<std::mutex> lock(_mutex);
  Block *block = _blocks.take(size);
  if (!block) {
    // Nothing free is large enough: a new chunk, kChunkSize unless the tensor is larger.
    int64_t needed = (size + kChunkGranule - 1) / kChunkGranule * kChunkGranule;
    int64_t chunkSize = std::max(kChunkSize, needed);

    // Within the limit, if there is one: the empty chunks go back first where a new one would pass
    // it, and the new one is no larger than the room that leaves -- a limit is not to be spent on
    // the part of a chunk no tensor asked for. Where even the tensor does not fit, nothing is
    // asked of the device.
    if (_memoryLimit > 0) {
      if (_blocks.getHeldBytes() + chunkSize > _memoryLimit) synchronizeAndReleaseEmptyChunks();
      int64_t room = (_memoryLimit - _blocks.getHeldBytes()) / kChunkGranule * kChunkGranule;
      if (needed > room) {
        throw lut::OutOfMemoryError(lut::sprintf(
            "%d bytes more would take the Metal device past its limit of %d bytes (%d held, %d "
            "by tensors)",
            needed,
            _memoryLimit,
            _blocks.getHeldBytes(),
            _allocatedBytes));
      }
      chunkSize = std::min(chunkSize, room);
    }

    // A tensor larger than a chunk has one of its own, and the empty chunks there are -- every one
    // too small for it, or take() would have found it -- go back before it is made, once the
    // device is done with them, rather than sit beside it until the work is over. That is where
    // a model's stages change: the denoiser's chunks, say, when the decoder's first gigabyte
    // comes. A tensor that size that fits in an empty chunk takes it instead, and nothing waits.
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

void VRamAllocator::free(const Buffer &buffer) {
  std::lock_guard<std::mutex> lock(_mutex);
  _allocatedBytes -= buffer->size;
  _blocks.give(buffer->block, buffer->lastUse);
  buffer->block = nullptr;
}

void VRamAllocator::synchronizeAndReleaseEmptyChunks() {
  // The command buffers do not retain what they use, which is why the device has to be done with
  // a chunk before it goes.
  _synchronize();
  for (Chunk *chunk : _blocks.emptyChunks()) {
    MTL::Buffer *buffer = static_cast<MTL::Buffer *>(chunk->handle);
    _blocks.removeChunk(chunk);
    buffer->release();
  }
}

void VRamAllocator::releaseUnusedMemory() {
  std::lock_guard<std::mutex> lock(_mutex);
  synchronizeAndReleaseEmptyChunks();
}

void VRamAllocator::setMemoryLimit(int64_t bytes) {
  CHECK(bytes >= 0);
  std::lock_guard<std::mutex> lock(_mutex);
  _memoryLimit = bytes;
}

int64_t VRamAllocator::getMemoryLimit() {
  std::lock_guard<std::mutex> lock(_mutex);
  return _memoryLimit;
}

int64_t VRamAllocator::getHeldBytes() {
  std::lock_guard<std::mutex> lock(_mutex);
  return _blocks.getHeldBytes();
}

int64_t VRamAllocator::getAllocatedBytes() {
  std::lock_guard<std::mutex> lock(_mutex);
  return _allocatedBytes;
}

MemorySnapshot VRamAllocator::captureMemorySnapshot() {
  std::lock_guard<std::mutex> lock(_mutex);

  // One pool of memory for the host and the device, so what is free is what the budget leaves
  // after everything this process has on the GPU, the cache included.
  int64_t total = _budgetBytes;
  int64_t free = total - static_cast<int64_t>(_device->currentAllocatedSize());
  return MemorySnapshot(total, std::max<int64_t>(0, free), _allocatedBytes, _peakAllocatedBytes);
}

void VRamAllocator::resetPeakMemoryStats() {
  std::lock_guard<std::mutex> lock(_mutex);
  _peakAllocatedBytes = _allocatedBytes;
}

}  // namespace metal_native
}  // namespace op
}  // namespace fl
