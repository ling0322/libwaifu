// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
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

#include "flint/cpu/rand.h"

#include <math.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <vector>

#include "lutil/half.h"
#include "lutil/random.h"
#include "lutil/time.h"
#include "flint/cpu/cast.h"
#include "flint/cpu/common.h"
#include "flint/functional.h"

namespace fl {
namespace op {
namespace cpu {

namespace {

void randFp32(const TensorView &out, lut::Random *generator, float min, float max) {
  CHECK(out.isContiguous());
  lut::Span<float> tensorData(getDataPtrCpu<float>(out), out.getNumEl());

  if (generator) {
    generator->fill(tensorData, min, max);
  } else {
    // if no generator specified, we could go parallel.
    std::vector<lut::Random> rs;
    lut::Random rseed;
    int numThreads = 1;
#ifdef _OPENMP
    numThreads = omp_get_max_threads();
#endif
    for (int i = 0; i < numThreads; ++i) {
      rs.emplace_back(rseed.nextInt());
    }

    int blockSize = 1024;
    int nb = static_cast<int>((tensorData.size() + blockSize - 1) / blockSize);
#pragma omp parallel for schedule(dynamic, 1)
    for (int b = 0; b < nb; ++b) {
      int threadIdx = 0;
#ifdef _OPENMP
      threadIdx = omp_get_thread_num();
#endif
      int64_t begin = static_cast<int64_t>(b) * blockSize;
      int64_t end = std::min(begin + blockSize, static_cast<int64_t>(tensorData.size()));
      for (int64_t i = begin; i < end; ++i) {
        float nextR = rs[threadIdx].nextFloat();
        tensorData[i] = min + (max - min) * nextR;
      }
    }
  }
}

}  // namespace

void rand(const TensorView &out, lut::Random *generator, float min, float max) {
  switch (int16_t(out.getDType())) {
    case DType::kFloat:
      return randFp32(out, generator, min, max);
    case DType::kFloat16: {
      // Drawn in float and narrowed, so that a seed draws the same numbers in either type.
      Tensor wide = F::empty(Device::getCpu(), out.getShape(), DType::kFloat);
      randFp32(wide, generator, min, max);
      return castFp32ToFp16(wide, out);
    }
    default:
      NOT_IMPL();
  }
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
