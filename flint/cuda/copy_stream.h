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

namespace fl {
namespace op {
namespace cuda {

/// @brief The one stream asynchronous copies run on.
///
/// One stream rather than one per copy, because the order copies are issued in is the order they
/// are wanted in: a queue is exactly the right shape. It is created with cudaStreamNonBlocking,
/// which is the whole point -- everything else in this library runs on the legacy default stream,
/// and a stream without that flag synchronizes with the legacy stream at every operation. It
/// would still be correct. It would simply never overlap with anything, and nothing would say so.
///
/// Those two are the whole arrangement: the legacy default stream for every kernel, this one for
/// every copy. Code elsewhere is written to that count -- see CudaTensorData::waitForReady(),
/// which hands out its dependency exactly once because there is exactly one consumer to hand it
/// to -- so a third stream is a change to more than this file.
///
/// Never destroyed. A tensor in static storage may outlive any singleton this could be held in,
/// and destroying the stream out from under a pending copy is worse than letting the CUDA context
/// teardown reclaim it, which it does.
class CopyStream {
 public:
  /// @brief The instance, created on first use.
  static CopyStream *getInstance();

  cudaStream_t getStream() const {
    return _stream;
  }

 private:
  CopyStream();

  cudaStream_t _stream;
};

}  // namespace cuda
}  // namespace op
}  // namespace fl
