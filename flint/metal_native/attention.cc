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

#include <math.h>

#include <string>

#include "lutil/strings.h"
#include "flint/metal_native/common.h"
#include "flint/metal_native/ops.h"

namespace fl {
namespace op {
namespace metal_native {

namespace {

// In step with AttentionParams in kernels/attention.metal.
struct AttentionParams {
  uint32_t numHeads;
  uint32_t headsPerKeyValueHead;
  uint32_t queryLength;
  uint32_t keyLength;
  uint32_t qStrideB;
  uint32_t qStrideH;
  uint32_t qStrideL;
  uint32_t kStrideB;
  uint32_t kStrideH;
  uint32_t kStrideL;
  uint32_t vStrideB;
  uint32_t vStrideH;
  uint32_t vStrideL;
  float scale;
  uint32_t maskType;
  uint32_t blockSize;
};

// The queries each threadgroup of kernels/attention.metal takes, and its threads.
constexpr int kBlockQ = 32;
constexpr uint32_t kThreads = 128;

// The most keys this kernel is used for. See attention().
constexpr int kMaxFusedKeys = 256;

// Whether the kernel was built for a head this wide in this type: half to 128, float to 64,
// which is what fits its threadgroup memory.
bool hasKernelFor(DType dtype, int headDim) {
  if (dtype == DType::kFloat16) return headDim == 32 || headDim == 64 || headDim == 128;
  if (dtype == DType::kFloat) return headDim == 32 || headDim == 64;
  return false;
}

// The queries each threadgroup of kernels/nax/attention.metal takes: sixteen to each of its four
// simdgroups. In step with kBlockQ there.
constexpr int kNaxBlockQ = 64;

// Whether the kernel on the neural accelerators was built for a head this wide in this type: half
// alone, since its products take float at about half's precision. See kernels/nax/attention.metal.
bool hasNaxKernelFor(DType dtype, int headDim) {
  if (dtype != DType::kFloat16) return false;
  return headDim == 32 || headDim == 64 || headDim == 128;
}

}  // namespace

bool attention(
    const TensorView &q,
    const TensorView &k,
    const TensorView &v,
    MaskType maskType,
    int blockSize,
    const TensorView &out) {
  if (q.getDim() != 4 || k.getDim() != 4 || v.getDim() != 4) return false;
  DType dtype = q.getDType();
  int batch = q.getShape(0);
  int numHeads = q.getShape(1);
  int queryLength = q.getShape(2);
  int headDim = q.getShape(3);
  int numKeyValueHeads = k.getShape(1);
  int keyLength = k.getShape(2);

  // What the kernel does not take goes to the composition, which takes everything.
  bool nax = getContext(q)->hasNax() && hasNaxKernelFor(dtype, headDim);
  if (!nax && !hasKernelFor(dtype, headDim)) return false;
  if (k.getDType() != dtype || v.getDType() != dtype || out.getDType() != dtype) return false;
  if (k.getShape() != v.getShape() || k.getShape(0) != batch || k.getShape(3) != headDim) {
    return false;
  }
  if (numKeyValueHeads == 0 || numHeads % numKeyValueHeads != 0) return false;
  if (q.getStride(3) != 1 || k.getStride(3) != 1 || v.getStride(3) != 1) return false;
  if (!out.isContiguous() || out.getShape() != q.getShape()) return false;
  if (queryLength == 0 || keyLength == 0) return false;
  // The mask is the caller's to have checked; what is left is what this kernel was not built
  // for. A causal mask over fewer keys than queries leaves rows that see nothing, which the
  // composition has always been the one to answer.
  bool blockMask = maskType == MaskType::kBlockCausal || maskType == MaskType::kBlockDiagonal;
  if (blockMask && blockSize <= 0) return false;
  if (maskType != MaskType::kNone && keyLength < queryLength) return false;
  if (maskType == MaskType::kBlockDiagonal && keyLength != queryLength) return false;

  // Over many keys the composition is the faster: its products are the GEMM kernels at their
  // best, and this kernel reads every key again for each kBlockQ queries. Over few -- a prompt's
  // 77 tokens, attended to from every pixel -- this saves the score matrix's trip through memory
  // and is the faster by twice. Measured on an M5 Pro over SDXL's attentions. A block mask has no
  // composition to be faster, so it is this kernel at every length -- which also skips the blocks
  // of keys a threadgroup's queries do not see.
  if (!nax && !blockMask && keyLength > kMaxFusedKeys) return false;

  AttentionParams params{};
  params.numHeads = static_cast<uint32_t>(numHeads);
  params.headsPerKeyValueHead = static_cast<uint32_t>(numHeads / numKeyValueHeads);
  params.queryLength = static_cast<uint32_t>(queryLength);
  params.keyLength = static_cast<uint32_t>(keyLength);
  params.qStrideB = static_cast<uint32_t>(q.getStride(0));
  params.qStrideH = static_cast<uint32_t>(q.getStride(1));
  params.qStrideL = static_cast<uint32_t>(q.getStride(2));
  params.kStrideB = static_cast<uint32_t>(k.getStride(0));
  params.kStrideH = static_cast<uint32_t>(k.getStride(1));
  params.kStrideL = static_cast<uint32_t>(k.getStride(2));
  params.vStrideB = static_cast<uint32_t>(v.getStride(0));
  params.vStrideH = static_cast<uint32_t>(v.getStride(1));
  params.vStrideL = static_cast<uint32_t>(v.getStride(2));
  params.scale = 1.0f / sqrtf(static_cast<float>(headDim));
  params.maskType = static_cast<uint32_t>(maskType);
  params.blockSize = static_cast<uint32_t>(blockMask ? blockSize : 0);

  int blockQ = nax ? kNaxBlockQ : kBlockQ;
  std::string kernel =
      kernelName(lut::sprintf(nax ? "nax_attention_%d" : "attention_%d", headDim).c_str(), dtype);
  getContext(q)->dispatchGroups(
      kernel.c_str(),
      {getArg(out), getArg(q), getArg(k), getArg(v)},
      &params,
      sizeof(params),
      static_cast<uint32_t>((queryLength + blockQ - 1) / blockQ),
      static_cast<uint32_t>(numHeads),
      static_cast<uint32_t>(batch),
      kThreads);
  return true;
}

}  // namespace metal_native
}  // namespace op
}  // namespace fl
