// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include <cuda_fp16.h>

#include "flint/cuda/accessor.h"
#include "flint/cuda/common.h"
#include "flint/cuda/repetition_penalty.h"

namespace fl {
namespace op {
namespace cuda {

// Two kernels, a gather and then a scatter, which is what the processor does on the host and what
// HuggingFace's `RepetitionPenaltyLogitsProcessor` does.
//
// This used to be one kernel, a thread per history entry reading a score, penalizing it and
// writing it straight back. That had three problems. It was written for half logits only, and the
// only caller casts its logits to float first, so on a card it never ran at all. It launched one
// block of sixty-four threads per row and refused a history longer than that, where a speech model
// says hundreds of tokens and penalizes every one. And a token said twice was two threads reading
// and writing the same score with nothing ordering them -- penalized once or twice depending on
// which read landed first.
//
// Splitting the read from the write settles the last of those: every score is gathered from the
// logits as they were before any of them changed, so a token said five times writes the same
// once-penalized value five times.
template<typename T>
__global__ void gatherPenalizedKernel(
    PackedTensorAccessor<const T, 2> logits,
    PackedTensorAccessor<const LongType, 2> history,
    PackedTensorAccessor<T, 2> scores,
    float weight) {
  int batchIdx = blockIdx.y;
  int historyIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (historyIdx >= history.getShape(1)) return;

  LongType logitsIdx = history[batchIdx][historyIdx];
  assert(logitsIdx >= 0 && logitsIdx < logits.getShape(1));

  float score = logits[batchIdx][logitsIdx];
  if (score > 0) {
    score /= weight;
  } else if (score < 0) {
    score *= weight;
  }

  scores[batchIdx][historyIdx] = score;
}

template<typename T>
__global__ void scatterPenalizedKernel(
    PackedTensorAccessor<T, 2> logits,
    PackedTensorAccessor<const LongType, 2> history,
    PackedTensorAccessor<const T, 2> scores) {
  int batchIdx = blockIdx.y;
  int historyIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (historyIdx >= history.getShape(1)) return;

  logits[batchIdx][history[batchIdx][historyIdx]] = scores[batchIdx][historyIdx];
}

template<typename T>
void repetitionPenalty2DImpl(Tensor logits, Tensor history, float weight) {
  CHECK(logits.getShape(0) == history.getShape(0));

  int rows = history.getShape(0);
  int length = history.getShape(1);
  if (rows == 0 || length == 0) return;

  Tensor scores = createCudaTensor<T>({rows, length});

  constexpr int blockSize = 256;
  dim3 grid((length + blockSize - 1) / blockSize, rows);

  gatherPenalizedKernel<T><<<grid, blockSize>>>(logits, history, scores, weight);
  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  scatterPenalizedKernel<T><<<grid, blockSize>>>(logits, history, scores);
  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

void repetitionPenalty2D(Tensor logits, Tensor history, float weight) {
  if (logits.getDType() == DType::kFloat) return repetitionPenalty2DImpl<float>(logits, history, weight);
  if (logits.getDType() == DType::kFloat16) return repetitionPenalty2DImpl<half>(logits, history, weight);

  NOT_IMPL();
}

void repetitionPenalty1D(const Tensor &logits, const Tensor &history, float weight) {
  repetitionPenalty2D(logits.unsqueeze(0), history.unsqueeze(0), weight);
}

void repetitionPenalty(Tensor logits, Tensor history, float weight) {
  if (logits.getDim() == 2)
    repetitionPenalty2D(logits, history, weight);
  else if (logits.getDim() == 1)
    repetitionPenalty1D(logits, history, weight);
  else
    NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
