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

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <mutex>

namespace fl {
namespace op {
namespace cuda {

/// @brief Copies pageable host memory to the GPU through two small page-locked buffers that
/// several threads fill.
///
/// What the driver does for a pageable cudaMemcpy is the same thing with one thread: it copies
/// the source into a page-locked buffer of its own and sends that. One thread cannot keep up with
/// the bus. On a Xeon w5-2465X a thread copies 12.9 GiB/s, so the driver's pageable copy runs at
/// 9.7 GiB/s against the 24.9 GiB/s of a page-locked source on PCIe 5.0 x8. Four threads copy
/// 25.9 GiB/s, and staging through two buffers with them sends 24.5 GiB/s -- the page-locked
/// rate, with 32 MiB locked rather than the whole source.
///
/// Two buffers, so that the threads fill one while the other is on the bus: with one the two
/// take turns and the rate is 15.8 GiB/s. Not three, and not larger: the pair is sized to stay in
/// the last-level cache (33.8 MiB there), where the copy engine reads what the threads just wrote
/// without a trip to memory. Two of 64 MiB send 19.9 GiB/s, three of 16 MiB 22.7 GiB/s.
///
/// Never destroyed, for the reason CopyStream is not: freeing page-locked memory drains the
/// device, and a static destructor may run after the CUDA context is gone.
class StagedUpload {
 public:
  /// @brief Size of each of the two staging buffers.
  static constexpr int64_t ChunkBytes = int64_t(16) << 20;

  /// @brief Threads that fill a buffer.
  static constexpr int NumThreads = 4;

  /// @brief Copies shorter than this are left to cudaMemcpy: the threads cost more to wake than
  /// a buffer of this size takes to fill, and a copy this short is a latency, not a rate.
  static constexpr int64_t MinBytes = int64_t(1) << 20;

  /// @brief The instance, created on first use.
  static StagedUpload *getInstance();

  /// @brief Enqueue the copy of `n` bytes from pageable `src` to device `dest` on `stream`.
  ///
  /// Returns once every byte of `src` has been read, so the caller may free or overwrite it; the
  /// last buffers may still be on the bus, and whoever reads `dest` has to be ordered after
  /// `stream` to see them. Safe to call from several threads, which take turns.
  void copy(void *dest, const void *src, int64_t n, cudaStream_t stream);

 private:
  StagedUpload();

  std::mutex _mutex;
  std::byte *_buffers[2];

  /// Recorded after each buffer's send, so that a buffer is not refilled while the copy engine
  /// is still reading it -- including by the next call, which starts where this one left off.
  cudaEvent_t _sent[2];

  /// The buffer the next chunk goes into.
  int _next;
};

}  // namespace cuda
}  // namespace op
}  // namespace fl
