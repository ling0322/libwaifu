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

#include "flint/capi.h"

#include <string.h>

#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "flint/functional.h"
#include "flint/memory.h"
#include "flint/operators.h"
#ifdef LIBWAIFU_CUTLASS_ENABLED
#include "flint/cuda/gemm_nvfp4_cutlass.h"
#include "flint/cuda/nvfp4.h"
#endif
#include "flint/tensor.h"
#include "lutil/error.h"

namespace {

thread_local int32_t gErrorCode = 0;
thread_local std::string gErrorMessage;

std::once_flag gInitOnce;

int32_t setError(int32_t code, const std::string &message) {
  gErrorCode = code;
  gErrorMessage = message;
  return code;
}

int32_t clearError() {
  gErrorCode = 0;
  gErrorMessage.clear();
  return FL_OK;
}

fl::DType toDType(fl_dtype_t dtype) {
  switch (dtype) {
    case FL_DTYPE_FLOAT:
    case FL_DTYPE_LONG:
    case FL_DTYPE_UINT8:
    case FL_DTYPE_FLOAT16:
    case FL_DTYPE_INT8:
    case FL_DTYPE_FP4E2M0X2:
    case FL_DTYPE_BOOL:
    case FL_DTYPE_INT32:
      return fl::DType(static_cast<int16_t>(dtype));
    default:
      throw lut::InvalidArgError("invalid dtype");
  }
}

fl::Device toDevice(fl_device_type_t device) {
  switch (device) {
    case FL_DEVICE_CPU:
      return fl::Device(fl::Device::kCpu);
    case FL_DEVICE_CUDA:
      return fl::Device(fl::Device::kCuda);
    default:
      throw lut::InvalidArgError("invalid device");
  }
}

fl_device_type_t fromDevice(fl::Device device) {
  switch (device.getType()) {
    case fl::Device::kCpu:
      return FL_DEVICE_CPU;
    case fl::Device::kCuda:
      return FL_DEVICE_CUDA;
    default:
      return FL_DEVICE_UNKNOWN;
  }
}

std::vector<int> toShape(const int32_t *shape, int32_t ndim) {
  if (ndim < 0) throw lut::InvalidArgError("ndim must not be negative");
  if (ndim > 0 && !shape) throw lut::InvalidArgError("shape is null");

  std::vector<int> result;
  result.reserve(static_cast<size_t>(ndim));
  for (int32_t i = 0; i < ndim; ++i) {
    if (shape[i] < 0) throw lut::InvalidArgError("shape must not be negative");
    result.push_back(shape[i]);
  }
  return result;
}

fl::Tensor &deref(fl_tensor_t tensor) {
  if (!tensor) throw lut::InvalidArgError("tensor is null");
  return *reinterpret_cast<fl::Tensor *>(tensor);
}

/// Number of bytes the elements of `tensor` occupy once packed together.
int64_t getPackedSize(const fl::Tensor &tensor) {
  return tensor.getDType().getTotalSize(tensor.getNumEl());
}

/// Hand a freshly made tensor back as a handle. Only called once the operation itself succeeded,
/// so `out` is left untouched on every failure path.
int32_t publish(fl::Tensor tensor, fl_tensor_t *out) {
  if (!out) throw lut::InvalidArgError("out is null");

  *out = reinterpret_cast<fl_tensor_t>(new fl::Tensor(std::move(tensor)));
  return clearError();
}

/// Runs `body` and turns whatever it throws into an error code, so that no exception crosses the
/// C boundary.
template<typename Body>
int32_t guard(Body &&body) {
  try {
    return body();
  } catch (const lut::InvalidArgError &error) {
    return setError(FL_ERROR_INVALID_ARG, error.what());
  } catch (const std::bad_alloc &) {
    return setError(FL_ERROR_ABORTED, "out of memory");
  } catch (const std::exception &error) {
    return setError(FL_ERROR_ABORTED, error.what());
  } catch (...) {
    return setError(FL_ERROR_ABORTED, "unknown exception");
  }
}

}  // namespace

void fl_init() {
  guard([]() {
    std::call_once(gInitOnce, []() { fl::initOperators(); });
    return clearError();
  });
}

int32_t fl_is_device_available(fl_device_type_t device, int32_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = fl::isOperatorsAvailable(toDevice(device).getType()) ? 1 : 0;
    return clearError();
  });
}

int32_t fl_get_last_error_code() {
  return gErrorCode;
}

const char *fl_get_last_error_message() {
  return gErrorMessage.c_str();
}

int32_t fl_tensor_zeros(
    const int32_t *shape,
    int32_t ndim,
    fl_dtype_t dtype,
    fl_device_type_t device,
    fl_tensor_t *out) {
  return guard([&]() {
    std::vector<int> dims = toShape(shape, ndim);
    return publish(fl::F::zeros(dims, toDType(dtype), toDevice(device)), out);
  });
}

int32_t fl_tensor_empty(
    const int32_t *shape,
    int32_t ndim,
    fl_dtype_t dtype,
    fl_device_type_t device,
    fl_tensor_t *out) {
  return guard([&]() {
    std::vector<int> dims = toShape(shape, ndim);
    return publish(fl::F::tensor(dims, toDType(dtype), toDevice(device)), out);
  });
}

int32_t fl_tensor_from_data(
    const int32_t *shape,
    int32_t ndim,
    fl_dtype_t dtype,
    const void *data,
    int64_t data_size,
    fl_tensor_t *out) {
  return guard([&]() {
    if (!data && data_size != 0) throw lut::InvalidArgError("data is null");
    if (data_size < 0) throw lut::InvalidArgError("data_size must not be negative");

    std::vector<int> dims = toShape(shape, ndim);
    fl::Tensor tensor = fl::F::tensor(dims, toDType(dtype), fl::Device::getCpu());
    int64_t expected = getPackedSize(tensor);
    if (data_size != expected) {
      throw lut::InvalidArgError(
          "data_size does not match the shape and dtype: expected " + std::to_string(expected) +
          " bytes, got " + std::to_string(data_size));
    }

    // Freshly created, so it is contiguous and starts at offset zero.
    if (expected > 0) {
      memcpy(tensor.getInternalData()->getData<void>(0), data, static_cast<size_t>(expected));
    }
    return publish(std::move(tensor), out);
  });
}

int32_t fl_tensor_clone(fl_tensor_t tensor, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(tensor), out); });
}

void fl_tensor_destroy(fl_tensor_t tensor) {
  delete reinterpret_cast<fl::Tensor *>(tensor);
}

int32_t fl_tensor_get_dim(fl_tensor_t tensor, int32_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = deref(tensor).getDim();
    return clearError();
  });
}

int32_t fl_tensor_get_shape(fl_tensor_t tensor, int32_t dim, int32_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = deref(tensor).getShape(dim);
    return clearError();
  });
}

int32_t fl_tensor_get_stride(fl_tensor_t tensor, int32_t dim, int32_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = deref(tensor).getStride(dim);
    return clearError();
  });
}

int32_t fl_tensor_get_numel(fl_tensor_t tensor, int64_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = deref(tensor).getNumEl();
    return clearError();
  });
}

int32_t fl_tensor_get_dtype(fl_tensor_t tensor, fl_dtype_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = static_cast<fl_dtype_t>(static_cast<int16_t>(deref(tensor).getDType()));
    return clearError();
  });
}

int32_t fl_tensor_get_device(fl_tensor_t tensor, fl_device_type_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = fromDevice(deref(tensor).getDevice());
    return clearError();
  });
}

int32_t fl_tensor_is_contiguous(fl_tensor_t tensor, int32_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = deref(tensor).isContiguous() ? 1 : 0;
    return clearError();
  });
}

int32_t fl_tensor_view(
    fl_tensor_t tensor,
    const int32_t *shape,
    int32_t ndim,
    fl_tensor_t *out) {
  return guard([&]() {
    std::vector<int> dims = toShape(shape, ndim);
    return publish(deref(tensor).view(dims), out);
  });
}

int32_t fl_tensor_transpose(
    fl_tensor_t tensor,
    int32_t dim0,
    int32_t dim1,
    fl_tensor_t *out) {
  return guard([&]() { return publish(deref(tensor).transpose(dim0, dim1), out); });
}

int32_t fl_tensor_slice(
    fl_tensor_t tensor,
    int32_t dim,
    int32_t begin,
    int32_t end,
    fl_tensor_t *out) {
  return guard([&]() {
    return publish(deref(tensor).slice(dim, {begin, end}), out);
  });
}

int32_t fl_tensor_subtensor(fl_tensor_t tensor, int32_t index, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(tensor).subtensor(index), out); });
}

int32_t fl_tensor_unsqueeze(fl_tensor_t tensor, int32_t dim, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(tensor).unsqueeze(dim), out); });
}

int32_t fl_tensor_squeeze(fl_tensor_t tensor, int32_t dim, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(tensor).squeeze(dim), out); });
}

int32_t fl_tensor_contiguous(fl_tensor_t tensor, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::contiguous(deref(tensor)), out); });
}

int32_t fl_tensor_to_device(fl_tensor_t tensor, fl_device_type_t device, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::to(toDevice(device), deref(tensor)), out); });
}

int32_t fl_tensor_cast(fl_tensor_t tensor, fl_dtype_t dtype, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::cast(deref(tensor), toDType(dtype)), out); });
}

int32_t fl_tensor_get_nbytes(fl_tensor_t tensor, int64_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = getPackedSize(deref(tensor));
    return clearError();
  });
}

int32_t fl_tensor_copy_to_host(fl_tensor_t tensor, void *buffer, int64_t buffer_size) {
  return guard([&]() {
    fl::Tensor source = deref(tensor);
    int64_t nbytes = getPackedSize(source);
    if (!buffer && nbytes != 0) throw lut::InvalidArgError("buffer is null");
    if (buffer_size < nbytes) {
      throw lut::InvalidArgError(
          "buffer is too small: need " + std::to_string(nbytes) + " bytes, got " +
          std::to_string(buffer_size));
    }
    if (nbytes == 0) return clearError();

    // Both steps copy, so do the device transfer first and pack the smaller result on the host.
    if (source.getDevice().getType() != fl::Device::kCpu) {
      source = fl::F::to(fl::Device::getCpu(), source);
    }
    if (!source.isContiguous()) source = fl::F::contiguous(source);

    const void *data = source.getInternalData()->getData<void>(source.getInternalOffset());
    memcpy(buffer, data, static_cast<size_t>(nbytes));
    return clearError();
  });
}

int32_t fl_arange(
    int64_t begin,
    int64_t end,
    int64_t step,
    fl_device_type_t device,
    fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::arange(begin, end, step, toDevice(device)), out); });
}

int32_t fl_rand(
    const int32_t *shape,
    int32_t ndim,
    fl_dtype_t dtype,
    fl_device_type_t device,
    fl_tensor_t *out) {
  return guard([&]() {
    std::vector<int> dims = toShape(shape, ndim);
    return publish(fl::F::rand(dims, toDType(dtype), toDevice(device)), out);
  });
}

int32_t fl_randn(
    const int32_t *shape,
    int32_t ndim,
    fl_device_type_t device,
    fl_tensor_t *out) {
  return guard([&]() {
    std::vector<int> dims = toShape(shape, ndim);
    return publish(fl::F::randn(dims, toDevice(device)), out);
  });
}

int32_t fl_manual_seed(fl_device_type_t device, uint64_t seed) {
  return guard([&]() {
    fl::F::manualSeed(toDevice(device), seed);
    return clearError();
  });
}

int32_t fl_lookup(fl_tensor_t table, fl_tensor_t indices, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::lookup(deref(table), deref(indices)), out); });
}

int32_t fl_rotary_embedding(
    fl_tensor_t positions,
    fl_tensor_t query,
    fl_tensor_t key,
    fl_tensor_t rotary_cache) {
  return guard([&]() {
    fl::F::rotaryEmbedding(deref(positions), deref(query), deref(key), deref(rotary_cache));
    return clearError();
  });
}

int32_t fl_rms_norm(fl_tensor_t input, fl_tensor_t weight, float eps, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::rmsNorm(deref(input), deref(weight), eps), out); });
}

int32_t fl_conv2d(
    fl_tensor_t input,
    fl_tensor_t weight,
    fl_tensor_t bias,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t groups,
    fl_tensor_t *out) {
  return guard([&]() {
    fl::Tensor emptyTensor;
    const fl::Tensor &b = bias ? deref(bias) : emptyTensor;

    return publish(
        fl::F::conv2d(deref(input), deref(weight), b, stride, padding, dilation, groups),
        out);
  });
}

int32_t fl_group_norm(
    fl_tensor_t input,
    fl_tensor_t weight,
    fl_tensor_t bias,
    int32_t groups,
    float eps,
    fl_tensor_t *out) {
  return guard([&]() {
    fl::Tensor emptyTensor;
    const fl::Tensor &w = weight ? deref(weight) : emptyTensor;
    const fl::Tensor &b = bias ? deref(bias) : emptyTensor;

    return publish(fl::F::groupNorm(deref(input), w, b, groups, eps), out);
  });
}

int32_t fl_upsample_nearest2d(fl_tensor_t input, int32_t scale, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::upsampleNearest2d(deref(input), scale), out); });
}

int32_t fl_layer_norm(
    fl_tensor_t input,
    fl_tensor_t weight,
    fl_tensor_t bias,
    float eps,
    fl_tensor_t *out) {
  return guard([&]() {
    // An absent weight or bias is a null handle rather than an empty tensor, which is what a
    // caller with nothing to pass has.
    fl::Tensor emptyTensor;
    const fl::Tensor &w = weight ? deref(weight) : emptyTensor;
    const fl::Tensor &b = bias ? deref(bias) : emptyTensor;

    return publish(fl::F::layerNorm(deref(input), w, b, eps), out);
  });
}

int32_t fl_quick_gelu(fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::quickGelu(deref(input)), out); });
}

int32_t fl_matmul(fl_tensor_t a, fl_tensor_t b, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::matmul(deref(a), deref(b)), out); });
}

#ifdef LIBWAIFU_CUTLASS_ENABLED

namespace {

/// The kernels assert their preconditions with CHECK, which aborts, and nothing may abort across
/// this boundary. Whatever a caller could get wrong is checked here first, where it can be named
/// as a bad argument in the caller's own terms.
void checkNvfp4Half(const fl::Tensor &x, const char *what) {
  if (x.getDevice().getType() != fl::Device::kCuda) {
    throw lut::InvalidArgError(std::string(what) + " is not on a CUDA device");
  }
  if (x.getDType() != fl::DType::kFloat16) {
    throw lut::InvalidArgError(std::string(what) + " is not <float16>");
  }
  if (!x.isContiguous()) {
    throw lut::InvalidArgError(std::string(what) + " is not contiguous");
  }
  if (x.getDim() < 2) {
    throw lut::InvalidArgError(std::string(what) + " has fewer than two dimensions");
  }
  if (x.getShape(-1) % 32 != 0) {
    throw lut::InvalidArgError(std::string(what) + ": the last dimension is not a multiple of 32");
  }
}

}  // namespace

int32_t fl_nvfp4_available(int32_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = fl::op::cuda::isNvfp4GemmAvailable() ? 1 : 0;
    return clearError();
  });
}

int32_t fl_nvfp4_quantize(
    fl_tensor_t x,
    fl_tensor_t *data,
    fl_tensor_t *block_scale,
    fl_tensor_t *global_scale) {
  return guard([&]() {
    if (!data || !block_scale || !global_scale) throw lut::InvalidArgError("out is null");

    checkNvfp4Half(deref(x), "the tensor to quantize");
    if (deref(x).getDim() != 2) {
      throw lut::InvalidArgError("the tensor to quantize is not two dimensional");
    }

    fl::op::cuda::Nvfp4Operand operand = fl::op::cuda::quantizeNvfp4(deref(x));

    // Three handles have to appear together or not at all, and publish() is what allocates, so
    // they are made here and only handed over once all three exist.
    std::unique_ptr<fl::Tensor> ownedData{new fl::Tensor(std::move(operand.data))};
    std::unique_ptr<fl::Tensor> ownedScale{new fl::Tensor(std::move(operand.blockScale))};
    std::unique_ptr<fl::Tensor> ownedGlobal{new fl::Tensor(std::move(operand.globalScale))};

    *data = reinterpret_cast<fl_tensor_t>(ownedData.release());
    *block_scale = reinterpret_cast<fl_tensor_t>(ownedScale.release());
    *global_scale = reinterpret_cast<fl_tensor_t>(ownedGlobal.release());
    return clearError();
  });
}

int32_t fl_nvfp4_dequantize(
    fl_tensor_t data,
    fl_tensor_t block_scale,
    fl_tensor_t global_scale,
    fl_tensor_t *out) {
  return guard([&]() {
    fl::op::cuda::Nvfp4Operand operand = fl::op::cuda::makeNvfp4Operand(
        deref(data),
        deref(block_scale),
        deref(global_scale));
    return publish(fl::op::cuda::dequantNvfp4ToHalf(operand), out);
  });
}

int32_t fl_nvfp4_matmul(
    fl_tensor_t a,
    fl_tensor_t data,
    fl_tensor_t block_scale,
    fl_tensor_t global_scale,
    fl_tensor_t *out) {
  return guard([&]() {
    fl::op::cuda::Nvfp4Operand operand = fl::op::cuda::makeNvfp4Operand(
        deref(data),
        deref(block_scale),
        deref(global_scale));

    checkNvfp4Half(deref(a), "the left operand");
    if (deref(a).getShape(-1) != operand.k) {
      throw lut::InvalidArgError("the two operands disagree about k");
    }
    if (operand.rows % 8 != 0) {
      throw lut::InvalidArgError("the nvfp4 operand's row count is not a multiple of 8");
    }

    return publish(fl::op::cuda::gemmNvfp4(deref(a), operand), out);
  });
}

#else

int32_t fl_nvfp4_available(int32_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = 0;
    return clearError();
  });
}

static int32_t nvfp4Unavailable() {
  return setError(FL_ERROR_ABORTED, "this build has no NVFP4 support (needs WITH_CUTLASS=ON)");
}

int32_t fl_nvfp4_quantize(fl_tensor_t, fl_tensor_t *, fl_tensor_t *, fl_tensor_t *) {
  return nvfp4Unavailable();
}

int32_t fl_nvfp4_dequantize(fl_tensor_t, fl_tensor_t, fl_tensor_t, fl_tensor_t *) {
  return nvfp4Unavailable();
}

int32_t fl_nvfp4_matmul(fl_tensor_t, fl_tensor_t, fl_tensor_t, fl_tensor_t, fl_tensor_t *) {
  return nvfp4Unavailable();
}

#endif  // LIBWAIFU_CUTLASS_ENABLED

int32_t fl_mul(fl_tensor_t a, fl_tensor_t b, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::mul(deref(a), deref(b)), out); });
}

int32_t fl_add(fl_tensor_t a, fl_tensor_t b, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::add(deref(a), deref(b)), out); });
}

int32_t fl_sub(fl_tensor_t a, fl_tensor_t b, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::sub(deref(a), deref(b)), out); });
}

int32_t fl_eq(fl_tensor_t a, fl_tensor_t b, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::eq(deref(a), deref(b)), out); });
}

int32_t fl_div(fl_tensor_t a, fl_tensor_t b, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::div(deref(a), deref(b)), out); });
}

int32_t fl_mul_scalar(fl_tensor_t input, float other, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::mul(deref(input), other), out); });
}

int32_t fl_div_scalar(fl_tensor_t input, float other, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::div(deref(input), other), out); });
}

int32_t fl_mod_scalar(fl_tensor_t input, int64_t other, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::mod(deref(input), other), out); });
}

int32_t fl_square(fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::square(deref(input)), out); });
}

int32_t fl_neg(fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::neg(deref(input)), out); });
}

int32_t fl_abs(fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::abs(deref(input)), out); });
}

int32_t fl_exp(fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::exp(deref(input)), out); });
}

int32_t fl_sqrt(fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::sqrt(deref(input)), out); });
}

int32_t fl_rsqrt(fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::rsqrt(deref(input)), out); });
}

int32_t fl_sigmoid(fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::sigmoid(deref(input)), out); });
}

int32_t fl_tanh(fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::tanh(deref(input)), out); });
}

int32_t fl_relu(fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::relu(deref(input)), out); });
}

int32_t fl_gelu(fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::gelu(deref(input)), out); });
}

int32_t fl_silu(fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::silu(deref(input)), out); });
}

int32_t fl_softmax(fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::softmax(deref(input)), out); });
}

int32_t fl_swiglu(fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::swiglu(deref(input)), out); });
}

int32_t fl_geglu(fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::geglu(deref(input)), out); });
}

int32_t fl_sum(fl_tensor_t input, int32_t dim, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::sum(deref(input), dim), out); });
}

int32_t fl_max(fl_tensor_t input, int32_t dim, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::max(deref(input), dim), out); });
}

int32_t fl_min(fl_tensor_t input, int32_t dim, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::min(deref(input), dim), out); });
}

int32_t fl_cat(fl_tensor_t a, fl_tensor_t b, int32_t dim, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::cat(deref(a), deref(b), dim), out); });
}

int32_t fl_causal_mask(int32_t max_len, fl_device_type_t device, fl_tensor_t *out) {
  return guard([&]() { return publish(fl::F::causalMask(max_len, toDevice(device)), out); });
}

int32_t fl_attention(
    fl_tensor_t q,
    fl_tensor_t k,
    fl_tensor_t v,
    int32_t causal,
    fl_tensor_t *out) {
  return guard([&]() {
    return publish(fl::F::attention(deref(q), deref(k), deref(v), causal != 0), out);
  });
}

int32_t fl_paged_attention(
    fl_tensor_t q,
    fl_tensor_t key_cache,
    fl_tensor_t value_cache,
    fl_tensor_t block_table,
    fl_tensor_t cu_seqlens_q,
    fl_tensor_t seqlens_k,
    int32_t max_q_len,
    int32_t max_k_len,
    int32_t causal,
    fl_tensor_t *out) {
  return guard([&]() {
    return publish(
        fl::F::pagedAttention(
            deref(q),
            deref(key_cache),
            deref(value_cache),
            deref(block_table),
            deref(cu_seqlens_q),
            deref(seqlens_k),
            max_q_len,
            max_k_len,
            causal != 0),
        out);
  });
}

int32_t fl_store_kv_cache(
    fl_tensor_t k,
    fl_tensor_t v,
    fl_tensor_t key_cache,
    fl_tensor_t value_cache,
    fl_tensor_t slot_mapping) {
  return guard([&]() {
    fl::F::storeKVCache(
        deref(k),
        deref(v),
        deref(key_cache),
        deref(value_cache),
        deref(slot_mapping));
    return clearError();
  });
}

int32_t fl_sample_with_params(
    fl_tensor_t logits,
    fl_tensor_t temperatures,
    fl_tensor_t top_ks,
    fl_tensor_t top_ps,
    fl_tensor_t *out) {
  return guard([&]() {
    return publish(
        fl::F::sample(deref(logits), deref(temperatures), deref(top_ks), deref(top_ps)),
        out);
  });
}

int32_t fl_repetition_penalty(fl_tensor_t logits, fl_tensor_t history, float weight) {
  return guard([&]() {
    fl::F::repetitionPenalty(deref(logits), deref(history), weight);
    return clearError();
  });
}

int32_t fl_copy(fl_tensor_t src, fl_tensor_t dest) {
  return guard([&]() {
    fl::F::copy(deref(src), deref(dest));
    return clearError();
  });
}

int32_t fl_fill(fl_tensor_t tensor, float value) {
  return guard([&]() {
    fl::F::fill(deref(tensor), value);
    return clearError();
  });
}

int32_t fl_all_close(fl_tensor_t a, fl_tensor_t b, float rtol, float atol, int32_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = fl::F::allClose(deref(a), deref(b), rtol, atol) ? 1 : 0;
    return clearError();
  });
}

int32_t fl_all(fl_tensor_t tensor, int32_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = fl::F::all(deref(tensor)) ? 1 : 0;
    return clearError();
  });
}

int32_t fl_elem(fl_tensor_t tensor, float *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = fl::F::elem(deref(tensor));
    return clearError();
  });
}

int32_t fl_get_default_float_type(fl_device_type_t device, fl_dtype_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    fl::DType dtype = fl::F::getDefaultFloatType(toDevice(device));
    *out = static_cast<fl_dtype_t>(static_cast<int16_t>(dtype));
    return clearError();
  });
}

int32_t fl_print(fl_tensor_t tensor) {
  return guard([&]() {
    fl::F::print(deref(tensor));
    return clearError();
  });
}

int32_t fl_memory_capture(fl_device_type_t device, fl_memory_snapshot_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    fl::MemorySnapshot snapshot = fl::MemorySnapshot::capture(toDevice(device));
    out->total = snapshot.getTotalMemory();
    out->free = snapshot.getFreeMemory();
    out->allocated = snapshot.getAllocatedMemory();
    out->peak_allocated = snapshot.getPeakAllocatedMemory();
    return clearError();
  });
}

int32_t fl_memory_reset_peak_stats(fl_device_type_t device) {
  return guard([&]() {
    fl::MemorySnapshot::resetPeakStats(toDevice(device));
    return clearError();
  });
}
