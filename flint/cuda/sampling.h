#pragma once

#include "flint/tensor.h"
#include "flint/tensor_view.h"

namespace fl {
namespace op {
namespace cuda {

/// Draw one label per row of `logits` into `out` <int64>(rows).
void sample(
    const TensorView &logits,
    const TensorView &uniformNoise,
    const TensorView &temperatures,
    const TensorView &topKs,
    const TensorView &topPs,
    const TensorView &out);

}  // namespace cuda
}  // namespace op
}  // namespace fl