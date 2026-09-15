// The MIT License (MIT)
//
// Copyright (c) 2025 Xiaoyang Chen
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

#include <memory>

#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cuda {

class Rand {
 public:
  ~Rand() = default;
  static std::shared_ptr<Rand> newRand();

  Tensor randNormal(lut::Span<const int> shape);
  Tensor rand(lut::Span<const int> shape);
  void setSeed(uint64_t seed);

 private:
  class Impl;
  std::unique_ptr<Impl> _impl;

  Rand() = default;
};

/// Runs the Philox4x32-10 bijection the draws above are made of: ten rounds over `ctr`, keyed by
/// `key`, into `out`. It is reachable from outside so the test can hold it against the vectors
/// Random123 publishes for it. The distributions the other tests measure would come out right for
/// any decent generator; this is the one that says which generator it is, and it is what a port to
/// another card has to agree with.
void philox4x32_10ForTest(const uint32_t ctr[4], const uint32_t key[2], uint32_t out[4]);

}  // namespace cuda
}  // namespace op
}  // namespace fl
