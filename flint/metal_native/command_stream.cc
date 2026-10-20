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

#include "flint/metal_native/command_stream.h"

#include <string.h>

#include <algorithm>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "lutil/error.h"
#include "lutil/log.h"
#include "flint/metal_native/metal_cpp.h"

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

}  // namespace

CommandStream::CommandStream(MTL::Device *device) {
  _queue = device->newCommandQueue();
  if (!_queue) throw lut::AbortedError("could not create a Metal command queue");
  _fence = device->newFence();
}

MTL::ComputeCommandEncoder *CommandStream::beginCommand(
    MTL::ComputePipelineState *pipeline,
    std::initializer_list<Arg> args,
    const void *params,
    size_t paramsSize) {
  if (!_recording) {
    AutoreleasePool pool;
    _recording = _queue->commandBufferWithUnretainedReferences()->retain();
    _encoder = _recording->computeCommandEncoder(MTL::DispatchTypeConcurrent)->retain();

    // After everything committed before: the batch before this one updated the fence last.
    if (_fenceUpdated) _encoder->waitForFence(_fence);
  }

  // Every command waits for every one before it in this encoder, and writes nothing until what
  // they wrote is there to be read. Coarser than saying which buffers each one touches, and it is
  // what makes the order work was asked for in the order it runs in, which the allocator depends
  // on.
  _encoder->memoryBarrier(MTL::BarrierScopeBuffers);
  _encoder->setComputePipelineState(pipeline);

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

void CommandStream::endCommand() {
  if (++_numRecorded >= kCommandsPerBatch) submit();
}

void CommandStream::dispatch(
    MTL::ComputePipelineState *pipeline,
    std::initializer_list<Arg> args,
    const void *params,
    size_t paramsSize,
    int64_t threads,
    int groupSize) {
  if (threads <= 0) return;

  NS::UInteger width = std::min<NS::UInteger>(
      static_cast<NS::UInteger>(groupSize),
      pipeline->maxTotalThreadsPerThreadgroup());

  std::lock_guard<std::mutex> lock(_mutex);
  MTL::ComputeCommandEncoder *encoder = beginCommand(pipeline, args, params, paramsSize);
  encoder->dispatchThreads(
      MTL::Size(static_cast<NS::UInteger>(threads), 1, 1),
      MTL::Size(width, 1, 1));
  endCommand();
}

void CommandStream::dispatchGroups(
    MTL::ComputePipelineState *pipeline,
    std::initializer_list<Arg> args,
    const void *params,
    size_t paramsSize,
    uint32_t groupsX,
    uint32_t groupsY,
    uint32_t groupsZ,
    uint32_t threadsPerGroup) {
  if (groupsX == 0 || groupsY == 0 || groupsZ == 0) return;
  CHECK(threadsPerGroup <= pipeline->maxTotalThreadsPerThreadgroup());

  std::lock_guard<std::mutex> lock(_mutex);
  MTL::ComputeCommandEncoder *encoder = beginCommand(pipeline, args, params, paramsSize);
  encoder->dispatchThreadgroups(
      MTL::Size(groupsX, groupsY, groupsZ),
      MTL::Size(threadsPerGroup, 1, 1));
  endCommand();
}

void CommandStream::submit() {
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

void CommandStream::waitFor(uint64_t serial) {
  if (serial == 0) return;
  if (serial > _submittedSerial) submit();  // it is the batch still being encoded

  std::unique_lock<std::mutex> lock(_completionMutex);
  _completed.wait(lock, [&]() { return _completedSerial >= serial || !_error.empty(); });
  if (!_error.empty()) {
    // Nothing encoded after a failed batch can be trusted, and there is no recovering it.
    throw lut::AbortedError("the Metal device failed: " + _error);
  }
}

void CommandStream::retireFinished() {
  uint64_t completed = getCompletedSerial();
  while (!_submitted.empty() && _submitted.front().serial <= completed) {
    _submitted.front().commandBuffer->release();
    _submitted.pop_front();
  }
}

uint64_t CommandStream::getCompletedSerial() {
  std::lock_guard<std::mutex> lock(_completionMutex);
  return _completedSerial;
}

void CommandStream::wait(uint64_t serial) {
  std::lock_guard<std::mutex> lock(_mutex);
  waitFor(serial);
  retireFinished();
}

void CommandStream::synchronize() {
  std::lock_guard<std::mutex> lock(_mutex);
  submit();
  waitFor(_submittedSerial);
  retireFinished();
}

void CommandStream::upload(const void *src, const Buffer &dest, int64_t offset, int64_t bytes) {
  if (bytes == 0) return;
  CHECK(offset >= 0 && offset + bytes <= dest->size);

  std::lock_guard<std::mutex> lock(_mutex);
  waitFor(dest->lastUse);
  memcpy(static_cast<std::byte *>(dest->buffer->contents()) + dest->offset + offset, src, bytes);
}

void CommandStream::download(const Buffer &src, int64_t offset, void *dest, int64_t bytes) {
  if (bytes == 0) return;
  CHECK(offset >= 0 && offset + bytes <= src->size);

  std::lock_guard<std::mutex> lock(_mutex);
  waitFor(src->lastUse);
  memcpy(
      dest, static_cast<const std::byte *>(src->buffer->contents()) + src->offset + offset, bytes);
}

}  // namespace metal_native
}  // namespace op
}  // namespace fl
