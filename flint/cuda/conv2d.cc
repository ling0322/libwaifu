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

// Conv2d through cuDNN. The library is resolved by name at the first call, the same way cuBLAS is,
// so a build that has cuDNN still runs on a machine that does not.
//
// This uses the convolution API that cuDNN 9 marks deprecated rather than the backend graph API.
// It is a dozen calls against several hundred, it is what cuDNN 9.20 still ships, and the graph
// API is where to go when this stops being true or when fusing the surrounding normalization and
// activation starts to matter.

#include <cstdlib>
#include <string>

#include "flint/cuda/conv2d.h"

#include "flint/cuda/conv2d_cutlass.h"

#include <cuda_runtime.h>

#include <unordered_map>

#include "lutil/error.h"
#include "lutil/shared_library.h"
#include "lutil/strings.h"
#include "flint/cuda/common.h"

#ifdef LIBWAIFU_CUDNN_ENABLED
#include <cudnn.h>
#endif

namespace fl {
namespace op {
namespace cuda {

#ifdef LIBWAIFU_CUDNN_ENABLED

namespace {

typedef cudnnStatus_t (*cudnnCreateFunc_t)(cudnnHandle_t *);
typedef cudnnStatus_t (*cudnnDestroyFunc_t)(cudnnHandle_t);
typedef const char *(*cudnnGetErrorStringFunc_t)(cudnnStatus_t);
typedef cudnnStatus_t (*cudnnCreateTensorDescriptorFunc_t)(cudnnTensorDescriptor_t *);
typedef cudnnStatus_t (*cudnnDestroyTensorDescriptorFunc_t)(cudnnTensorDescriptor_t);
typedef cudnnStatus_t (*cudnnSetTensor4dDescriptorFunc_t)(
    cudnnTensorDescriptor_t,
    cudnnTensorFormat_t,
    cudnnDataType_t,
    int,
    int,
    int,
    int);
typedef cudnnStatus_t (*cudnnCreateFilterDescriptorFunc_t)(cudnnFilterDescriptor_t *);
typedef cudnnStatus_t (*cudnnDestroyFilterDescriptorFunc_t)(cudnnFilterDescriptor_t);
typedef cudnnStatus_t (*cudnnSetFilter4dDescriptorFunc_t)(
    cudnnFilterDescriptor_t,
    cudnnDataType_t,
    cudnnTensorFormat_t,
    int,
    int,
    int,
    int);
typedef cudnnStatus_t (*cudnnCreateConvolutionDescriptorFunc_t)(cudnnConvolutionDescriptor_t *);
typedef cudnnStatus_t (*cudnnDestroyConvolutionDescriptorFunc_t)(cudnnConvolutionDescriptor_t);
typedef cudnnStatus_t (*cudnnSetConvolution2dDescriptorFunc_t)(
    cudnnConvolutionDescriptor_t,
    int,
    int,
    int,
    int,
    int,
    int,
    cudnnConvolutionMode_t,
    cudnnDataType_t);
typedef cudnnStatus_t (*cudnnSetConvolutionGroupCountFunc_t)(cudnnConvolutionDescriptor_t, int);
typedef cudnnStatus_t (*cudnnSetConvolutionMathTypeFunc_t)(
    cudnnConvolutionDescriptor_t,
    cudnnMathType_t);
typedef cudnnStatus_t (*cudnnGetConvolution2dForwardOutputDimFunc_t)(
    const cudnnConvolutionDescriptor_t,
    const cudnnTensorDescriptor_t,
    const cudnnFilterDescriptor_t,
    int *,
    int *,
    int *,
    int *);
typedef cudnnStatus_t (*cudnnGetConvolutionForwardAlgorithmFunc_t)(
    cudnnHandle_t,
    const cudnnTensorDescriptor_t,
    const cudnnFilterDescriptor_t,
    const cudnnConvolutionDescriptor_t,
    const cudnnTensorDescriptor_t,
    const int,
    int *,
    cudnnConvolutionFwdAlgoPerf_t *);
typedef cudnnStatus_t (*cudnnGetConvolutionForwardWorkspaceSizeFunc_t)(
    cudnnHandle_t,
    const cudnnTensorDescriptor_t,
    const cudnnFilterDescriptor_t,
    const cudnnConvolutionDescriptor_t,
    const cudnnTensorDescriptor_t,
    cudnnConvolutionFwdAlgo_t,
    size_t *);
typedef cudnnStatus_t (*cudnnConvolutionForwardFunc_t)(
    cudnnHandle_t,
    const void *,
    const cudnnTensorDescriptor_t,
    const void *,
    const cudnnFilterDescriptor_t,
    const void *,
    const cudnnConvolutionDescriptor_t,
    cudnnConvolutionFwdAlgo_t,
    void *,
    size_t,
    const void *,
    const cudnnTensorDescriptor_t,
    void *);
typedef cudnnStatus_t (*cudnnAddTensorFunc_t)(
    cudnnHandle_t,
    const void *,
    const cudnnTensorDescriptor_t,
    const void *,
    const void *,
    const cudnnTensorDescriptor_t,
    void *);

/// libcudnn, opened once. Everything the convolution needs is resolved by name so that a build
/// with cuDNN still links, and still runs, where the library is absent.
class Cudnn {
 public:
  static Cudnn *get() {
    static Cudnn *instance = create();
    return instance;
  }

  cudnnHandle_t handle;
  cudnnGetErrorStringFunc_t getErrorString;
  cudnnCreateTensorDescriptorFunc_t createTensorDescriptor;
  cudnnDestroyTensorDescriptorFunc_t destroyTensorDescriptor;
  cudnnSetTensor4dDescriptorFunc_t setTensor4dDescriptor;
  cudnnCreateFilterDescriptorFunc_t createFilterDescriptor;
  cudnnDestroyFilterDescriptorFunc_t destroyFilterDescriptor;
  cudnnSetFilter4dDescriptorFunc_t setFilter4dDescriptor;
  cudnnCreateConvolutionDescriptorFunc_t createConvolutionDescriptor;
  cudnnDestroyConvolutionDescriptorFunc_t destroyConvolutionDescriptor;
  cudnnSetConvolution2dDescriptorFunc_t setConvolution2dDescriptor;
  cudnnSetConvolutionGroupCountFunc_t setConvolutionGroupCount;
  cudnnSetConvolutionMathTypeFunc_t setConvolutionMathType;
  cudnnGetConvolution2dForwardOutputDimFunc_t getForwardOutputDim;
  cudnnGetConvolutionForwardAlgorithmFunc_t getForwardAlgorithm;
  cudnnGetConvolutionForwardWorkspaceSizeFunc_t getForwardWorkspaceSize;
  cudnnConvolutionForwardFunc_t convolutionForward;
  cudnnAddTensorFunc_t addTensor;

 private:
  std::unique_ptr<lut::SharedLibrary> _lib;

  /// pip ships cuDNN as libcudnn.so.9 with no unversioned name beside it, and a system install
  /// has the unversioned one, so both are worth a try before giving up.
  static std::unique_ptr<lut::SharedLibrary> openLibrary() {
    try {
      return lut::SharedLibrary::open("cudnn");
    } catch (const lut::Error &e) {
      LOG(DEBUG) << "libcudnn.so did not load: " << e.what();
    }

    return lut::SharedLibrary::openFile(
        lut::sprintf("libcudnn.so.%d", int(CUDNN_MAJOR)));
  }

  static Cudnn *create() {
    try {
      std::unique_ptr<Cudnn> cudnn = std::make_unique<Cudnn>();
      cudnn->_lib = openLibrary();

      auto create = cudnn->_lib->getFunc<cudnnCreateFunc_t>("cudnnCreate");
      cudnn->getErrorString = cudnn->_lib->getFunc<cudnnGetErrorStringFunc_t>(
          "cudnnGetErrorString");
      cudnn->createTensorDescriptor = cudnn->_lib->getFunc<cudnnCreateTensorDescriptorFunc_t>(
          "cudnnCreateTensorDescriptor");
      cudnn->destroyTensorDescriptor = cudnn->_lib->getFunc<cudnnDestroyTensorDescriptorFunc_t>(
          "cudnnDestroyTensorDescriptor");
      cudnn->setTensor4dDescriptor = cudnn->_lib->getFunc<cudnnSetTensor4dDescriptorFunc_t>(
          "cudnnSetTensor4dDescriptor");
      cudnn->createFilterDescriptor = cudnn->_lib->getFunc<cudnnCreateFilterDescriptorFunc_t>(
          "cudnnCreateFilterDescriptor");
      cudnn->destroyFilterDescriptor = cudnn->_lib->getFunc<cudnnDestroyFilterDescriptorFunc_t>(
          "cudnnDestroyFilterDescriptor");
      cudnn->setFilter4dDescriptor = cudnn->_lib->getFunc<cudnnSetFilter4dDescriptorFunc_t>(
          "cudnnSetFilter4dDescriptor");
      cudnn->createConvolutionDescriptor =
          cudnn->_lib->getFunc<cudnnCreateConvolutionDescriptorFunc_t>(
              "cudnnCreateConvolutionDescriptor");
      cudnn->destroyConvolutionDescriptor =
          cudnn->_lib->getFunc<cudnnDestroyConvolutionDescriptorFunc_t>(
              "cudnnDestroyConvolutionDescriptor");
      cudnn->setConvolution2dDescriptor =
          cudnn->_lib->getFunc<cudnnSetConvolution2dDescriptorFunc_t>(
              "cudnnSetConvolution2dDescriptor");
      cudnn->setConvolutionGroupCount = cudnn->_lib->getFunc<cudnnSetConvolutionGroupCountFunc_t>(
          "cudnnSetConvolutionGroupCount");
      cudnn->setConvolutionMathType = cudnn->_lib->getFunc<cudnnSetConvolutionMathTypeFunc_t>(
          "cudnnSetConvolutionMathType");
      cudnn->getForwardOutputDim =
          cudnn->_lib->getFunc<cudnnGetConvolution2dForwardOutputDimFunc_t>(
              "cudnnGetConvolution2dForwardOutputDim");
      cudnn->getForwardAlgorithm = cudnn->_lib->getFunc<cudnnGetConvolutionForwardAlgorithmFunc_t>(
          "cudnnGetConvolutionForwardAlgorithm_v7");
      cudnn->getForwardWorkspaceSize =
          cudnn->_lib->getFunc<cudnnGetConvolutionForwardWorkspaceSizeFunc_t>(
              "cudnnGetConvolutionForwardWorkspaceSize");
      cudnn->convolutionForward = cudnn->_lib->getFunc<cudnnConvolutionForwardFunc_t>(
          "cudnnConvolutionForward");
      cudnn->addTensor = cudnn->_lib->getFunc<cudnnAddTensorFunc_t>("cudnnAddTensor");

      if (!create || !cudnn->convolutionForward || !cudnn->getForwardAlgorithm) {
        LOG(WARN) << "libcudnn loaded but does not have the convolution entry points.";
        return nullptr;
      }
      if (create(&cudnn->handle) != CUDNN_STATUS_SUCCESS) {
        LOG(WARN) << "cudnnCreate failed.";
        return nullptr;
      }

      return cudnn.release();
    } catch (const lut::Error &e) {
      LOG(DEBUG) << "unable to load cudnn: " << e.what();
      return nullptr;
    }
  }
};


cudnnDataType_t toCudnnDataType(DType dtype) {
  if (dtype == DType::kFloat16) return CUDNN_DATA_HALF;
  if (dtype == DType::kFloat) return CUDNN_DATA_FLOAT;

  THROW(InvalidArg, "conv2d takes a float16 or float tensor");
  return CUDNN_DATA_FLOAT;
}

/// Everything about a convolution that decides how it is carried out. Two calls that agree on all
/// of it can share the descriptors and the algorithm choice.
struct PlanKey {
  int n, c, h, w;
  int k, filterC, r, s;
  int stride, padding, dilation, groups;
  int dtype;

  bool operator==(const PlanKey &rhs) const {
    return n == rhs.n && c == rhs.c && h == rhs.h && w == rhs.w && k == rhs.k &&
           filterC == rhs.filterC && r == rhs.r && s == rhs.s && stride == rhs.stride &&
           padding == rhs.padding && dilation == rhs.dilation && groups == rhs.groups &&
           dtype == rhs.dtype;
  }
};

struct PlanKeyHash {
  size_t operator()(const PlanKey &key) const {
    size_t hash = 1469598103934665603ull;
    for (int value : {key.n, key.c, key.h, key.w, key.k, key.filterC, key.r, key.s, key.stride,
                      key.padding, key.dilation, key.groups, key.dtype}) {
      hash = (hash ^ static_cast<size_t>(value)) * 1099511628211ull;
    }
    return hash;
  }
};

/// The descriptors and the algorithm choice, worked out once.
///
/// Asking cuDNN which algorithm to use, and building the descriptors to ask with, costs about a
/// millisecond -- more than most of these convolutions take to run, and a diffusion model walks
/// the same few dozen shapes thousands of times. The descriptors live as long as the process,
/// which is affordable because that few dozen is the whole set and it is known by the second step.
struct Plan {
  cudnnTensorDescriptor_t inputDesc = nullptr;
  cudnnTensorDescriptor_t outputDesc = nullptr;
  cudnnTensorDescriptor_t biasDesc = nullptr;
  cudnnFilterDescriptor_t filterDesc = nullptr;
  cudnnConvolutionDescriptor_t convDesc = nullptr;
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  size_t workspaceSize = 0;
  int outN = 0, outC = 0, outH = 0, outW = 0;
};

}  // namespace

#define LL_CHECK_CUDNN(x)                                                                   \
  {                                                                                         \
    cudnnStatus_t status = x;                                                               \
    if (status != CUDNN_STATUS_SUCCESS) {                                                   \
      LOG(ERROR) << "Error while calling: " << #x << ": " << cudnn->getErrorString(status);  \
      throw lut::AbortedError(cudnn->getErrorString(status));                                \
    }                                                                                       \
  }

namespace {

/// How much scratch a convolution may ask for before a cheaper algorithm is preferred.
constexpr size_t kMaxWorkspace = size_t(512) << 20;

Plan buildPlan(Cudnn *cudnn, const PlanKey &key) {
  Plan plan;
  cudnnDataType_t dataType = static_cast<cudnnDataType_t>(key.dtype);

  LL_CHECK_CUDNN(cudnn->createTensorDescriptor(&plan.inputDesc));
  LL_CHECK_CUDNN(cudnn->createTensorDescriptor(&plan.outputDesc));
  LL_CHECK_CUDNN(cudnn->createTensorDescriptor(&plan.biasDesc));
  LL_CHECK_CUDNN(cudnn->createFilterDescriptor(&plan.filterDesc));
  LL_CHECK_CUDNN(cudnn->createConvolutionDescriptor(&plan.convDesc));

  LL_CHECK_CUDNN(cudnn->setTensor4dDescriptor(
      plan.inputDesc, CUDNN_TENSOR_NCHW, dataType, key.n, key.c, key.h, key.w));
  LL_CHECK_CUDNN(cudnn->setFilter4dDescriptor(
      plan.filterDesc, dataType, CUDNN_TENSOR_NCHW, key.k, key.filterC, key.r, key.s));

  // Half data still accumulates in float, which is what keeps a 3x3 over hundreds of channels from
  // losing its tail, and the math type is what lets the tensor cores take it.
  LL_CHECK_CUDNN(cudnn->setConvolution2dDescriptor(
      plan.convDesc,
      key.padding,
      key.padding,
      key.stride,
      key.stride,
      key.dilation,
      key.dilation,
      CUDNN_CROSS_CORRELATION,
      CUDNN_DATA_FLOAT));
  LL_CHECK_CUDNN(cudnn->setConvolutionGroupCount(plan.convDesc, key.groups));
  LL_CHECK_CUDNN(cudnn->setConvolutionMathType(plan.convDesc, CUDNN_TENSOR_OP_MATH));

  LL_CHECK_CUDNN(cudnn->getForwardOutputDim(
      plan.convDesc,
      plan.inputDesc,
      plan.filterDesc,
      &plan.outN,
      &plan.outC,
      &plan.outH,
      &plan.outW));
  if (plan.outH <= 0 || plan.outW <= 0) {
    THROW(InvalidArg, "conv2d: the kernel and the padding leave nothing of the input");
  }

  LL_CHECK_CUDNN(cudnn->setTensor4dDescriptor(
      plan.outputDesc, CUDNN_TENSOR_NCHW, dataType, plan.outN, plan.outC, plan.outH, plan.outW));
  LL_CHECK_CUDNN(cudnn->setTensor4dDescriptor(
      plan.biasDesc, CUDNN_TENSOR_NCHW, dataType, 1, plan.outC, 1, 1));

  constexpr int kMaxAlgo = 8;
  cudnnConvolutionFwdAlgoPerf_t algos[kMaxAlgo];
  int numAlgo = 0;
  LL_CHECK_CUDNN(cudnn->getForwardAlgorithm(
      cudnn->handle,
      plan.inputDesc,
      plan.filterDesc,
      plan.convDesc,
      plan.outputDesc,
      kMaxAlgo,
      &numAlgo,
      algos));

  // The list comes back fastest first, with the ones this build cannot run marked. Take the first
  // that works and whose scratch is one we are willing to hand it; failing that, take whichever
  // works and asks for the least, because a convolution that is slower than it might have been
  // still beats one that will not run.
  const cudnnConvolutionFwdAlgoPerf_t *smallest = nullptr;
  bool chosen = false;
  for (int i = 0; i < numAlgo; ++i) {
    if (algos[i].status != CUDNN_STATUS_SUCCESS) continue;
    if (!smallest || algos[i].memory < smallest->memory) smallest = &algos[i];
    if (algos[i].memory > kMaxWorkspace) continue;

    plan.algo = algos[i].algo;
    plan.workspaceSize = algos[i].memory;
    chosen = true;
    break;
  }
  if (!chosen) {
    if (!smallest) THROW(InvalidArg, "conv2d: cuDNN has no algorithm for this convolution");

    LOG(WARN) << "conv2d: every algorithm wants more than " << (kMaxWorkspace >> 20)
              << "MB of scratch; taking the one that wants least, " << (smallest->memory >> 20)
              << "MB.";
    plan.algo = smallest->algo;
    plan.workspaceSize = smallest->memory;
  }

  return plan;
}

const Plan &getPlan(Cudnn *cudnn, const PlanKey &key) {
  static std::unordered_map<PlanKey, Plan, PlanKeyHash> cache;

  auto it = cache.find(key);
  if (it != cache.end()) return it->second;

  return cache.emplace(key, buildPlan(cudnn, key)).first->second;
}

}  // namespace

bool isConv2dAvailable() {
  static const bool available = Cudnn::get() != nullptr || isConv2dCutlassAvailable();
  return available;
}

/// Which library convolves, from LIBWAIFU_CONV. Unset means cuDNN wherever it loaded, which is
/// what a run that does not ask gets. Asked for once and remembered, so that a run cannot change
/// its mind partway and be measured as though it had not.
bool wantsCutlass() {
  static const bool wanted = [] {
    const char *choice = std::getenv("LIBWAIFU_CONV");
    if (!choice) return false;

    std::string name = choice;
    if (name == "cutlass") {
      LOG(INFO) << "LIBWAIFU_CONV=cutlass";
      return true;
    }
    if (name == "cudnn") return false;

    throw lut::AbortedError(
        lut::sprintf("LIBWAIFU_CONV is \"%s\", which is neither cudnn nor cutlass", choice));
  }();

  return wanted;
}

bool convolvesOnCutlass() {
  // The same question conv2d asks itself below, and it has to stay the same question: a caller
  // that guards a grouped convolution on this has to be told what will actually run, not what
  // was asked for. Being pointed at CUTLASS is only one of the two ways of getting it; the other
  // is cuDNN not loading, which no environment variable says.
  return wantsCutlass() || Cudnn::get() == nullptr;
}

Tensor conv2d(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    const Conv2dOptions &options) {
  Cudnn *cudnn = wantsCutlass() ? nullptr : Cudnn::get();

  // cuDNN is preferred where it is there, and it is looked for by name at the first call, so a
  // machine without it takes the other path rather than the build having to know. Said once
  // rather than on every convolution, and at a level that is on by default: which library the
  // convolutions go through is not a debugging detail.
  if (!cudnn) {
    static const bool reported = [] {
      if (!wantsCutlass()) LOG(WARN) << "cuDNN is not available, convolving on CUTLASS instead";
      return true;
    }();
    (void)reported;

    if (!isConv2dCutlassAvailable()) throw lut::AbortedError("neither cuDNN nor CUTLASS is here");
    return conv2dCutlass(input, weight, bias, options);
  }

  if (input.getDim() != 4) THROW(InvalidArg, "conv2d takes a 4-D input, as (N, C, H, W)");
  if (weight.getDim() != 4) THROW(InvalidArg, "conv2d takes a 4-D weight, as (K, C, R, S)");
  if (input.getDType() != weight.getDType()) {
    THROW(InvalidArg, "conv2d: the input and the weight are of different types");
  }
  if (options.groups < 1) THROW(InvalidArg, "conv2d: the group count is below one");
  if (options.stride < 1 || options.dilation < 1) {
    THROW(InvalidArg, "conv2d: the stride and the dilation are below one");
  }
  if (options.padding < 0) THROW(InvalidArg, "conv2d: the padding is negative");
  LL_CHECK_CONTIGUOUS(input);
  LL_CHECK_CONTIGUOUS(weight);

  if (input.getShape(1) != weight.getShape(1) * options.groups) {
    THROW(
        InvalidArg,
        lut::sprintf(
            "conv2d: an input of %d channels does not match a weight of %d by %d groups",
            input.getShape(1),
            weight.getShape(1),
            options.groups));
  }

  PlanKey key{
      input.getShape(0),
      input.getShape(1),
      input.getShape(2),
      input.getShape(3),
      weight.getShape(0),
      weight.getShape(1),
      weight.getShape(2),
      weight.getShape(3),
      options.stride,
      options.padding,
      options.dilation,
      options.groups,
      static_cast<int>(toCudnnDataType(input.getDType()))};
  const Plan &plan = getPlan(cudnn, key);

  Tensor output = input.getDType() == DType::kFloat16
                      ? createCudaTensorHalf({plan.outN, plan.outC, plan.outH, plan.outW})
                      : createCudaTensorFloat({plan.outN, plan.outC, plan.outH, plan.outW});

  lut::c_ptr<int8_t> workspace;
  if (plan.workspaceSize) workspace = llynCudaAlloc<int8_t>(plan.workspaceSize);

  float alpha = 1.0f;
  float beta = 0.0f;
  LL_CHECK_CUDNN(cudnn->convolutionForward(
      cudnn->handle,
      &alpha,
      plan.inputDesc,
      input.getInternalData()->getData<void>(input.getInternalOffset()),
      plan.filterDesc,
      weight.getInternalData()->getData<void>(weight.getInternalOffset()),
      plan.convDesc,
      plan.algo,
      workspace.get(),
      plan.workspaceSize,
      &beta,
      plan.outputDesc,
      output.getInternalData()->getData<void>(output.getInternalOffset())));

  if (!bias.empty()) {
    if (bias.getNumEl() != plan.outC) {
      THROW(
          InvalidArg,
          lut::sprintf(
              "conv2d: a bias of %d does not match %d output channels",
              int(bias.getNumEl()),
              plan.outC));
    }
    if (bias.getDType() != input.getDType()) {
      THROW(InvalidArg, "conv2d: the bias and the input are of different types");
    }

    // One value per channel, which cuDNN spreads over the batch and the two spatial axes.
    float one = 1.0f;
    LL_CHECK_CUDNN(cudnn->addTensor(
        cudnn->handle,
        &one,
        plan.biasDesc,
        bias.getInternalData()->getData<void>(bias.getInternalOffset()),
        &one,
        plan.outputDesc,
        output.getInternalData()->getData<void>(output.getInternalOffset())));
  }

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return output;
}

#else  // LIBWAIFU_CUDNN_ENABLED

bool isConv2dAvailable() {
  return isConv2dCutlassAvailable();
}

bool convolvesOnCutlass() {
  // Nothing else was built, so whatever convolves here is CUTLASS.
  return true;
}

Tensor conv2d(
    const Tensor &input,
    const Tensor &weight,
    const Tensor &bias,
    const Conv2dOptions &options) {
  if (!isConv2dCutlassAvailable()) {
    throw lut::AbortedError("this build has no Conv2d (needs WITH_CUDNN=ON or WITH_CUTLASS=ON)");
  }

  return conv2dCutlass(input, weight, bias, options);
}

#endif  // LIBWAIFU_CUDNN_ENABLED

}  // namespace cuda
}  // namespace op
}  // namespace fl
