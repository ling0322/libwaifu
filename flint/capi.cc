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

#include "flint/fp8.h"
#include "flint/memory.h"
#include "flint/operators.h"
#include "lutil/internal/log.h"
#ifdef LIBWAIFU_CUDA_ENABLED
#include "flint/cuda/future_tensor.h"
#include "flint/cuda/to_device.h"
#endif  // LIBWAIFU_CUDA_ENABLED

#ifdef LIBWAIFU_CUDA_ENABLED
#include "flint/cuda/fp8.h"
#include "flint/cuda/gemm_fp8_cutlass.h"
#include "flint/cuda/gemm_nvfp4_cutlass.h"
#include "flint/cuda/nvfp4.h"
#endif  // LIBWAIFU_CUDA_ENABLED
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
    // A caller may name this one because a package may hold it: the elements of a weight stored
    // quantized arrive as bytes and a dtype like any other tensor's, and only the scales beside
    // them say what they are worth. See fl_fp8_matmul().
    case FL_DTYPE_FP8E4M3:
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
    case FL_DEVICE_CUDA_HOST:
      return fl::Device(fl::Device::kCudaHost);
    case FL_DEVICE_METAL:
      return fl::Device(fl::Device::kMetal);
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
    case fl::Device::kCudaHost:
      return FL_DEVICE_CUDA_HOST;
    case fl::Device::kMetal:
      return FL_DEVICE_METAL;
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

/// What an fl_operators_t points at: the operators of one device, held by shared pointer so that
/// a handle keeps them alive, and the device they were asked for, so that a handle can say what
/// it is without the operators having to carry the answer.
struct OperatorsHandle {
  std::shared_ptr<fl::Operators> operators;
  fl_device_type_t device;
};

OperatorsHandle &derefHandle(fl_operators_t operators) {
  if (!operators) throw lut::InvalidArgError("operators is null");
  return *reinterpret_cast<OperatorsHandle *>(operators);
}

fl::Operators *deref(fl_operators_t operators) {
  return derefHandle(operators).operators.get();
}

/// Every tensor an operation reads has to be on the operators' own device. The kernels check that
/// with the library's fatal check, which ends the process; asked here first, a caller that mixed
/// two devices is told about it instead.
void checkOnDevice(const fl::Tensor &tensor, fl_operators_t operators, const char *what) {
  fl::Device::Type device = toDevice(derefHandle(operators).device).getType();
  if (tensor.getDevice().getType() != device) {
    throw lut::InvalidArgError(
        std::string(what) + " is on " + tensor.getDevice().getName() + ", and these are the " +
        fl::Device(device).getName() + " operators");
  }
}

/// The same for a tensor an operation may be given or not, which is a null handle when it is not.
void checkOptionalOnDevice(fl_tensor_t tensor, fl_operators_t operators, const char *what) {
  if (tensor) checkOnDevice(deref(tensor), operators, what);
}

/// An operation with nothing to say about a tensor that holds no elements.
void checkNotEmpty(const fl::Tensor &tensor, const char *what) {
  if (tensor.empty()) throw lut::InvalidArgError(std::string(what) + " is empty");
}

/// max() and min() reduce the last dimension and no other, so a caller that named a different one
/// is told rather than quietly handed the last.
void checkLastDim(const fl::Tensor &tensor, int32_t dim, const char *what) {
  if (dim != -1 && dim != tensor.getDim() - 1) {
    throw lut::InvalidArgError(
        std::string(what) + " reduces the last dimension, not dimension " + std::to_string(dim));
  }
}

#ifdef LIBWAIFU_CUDA_ENABLED
fl::FutureTensor &deref(fl_future_tensor_t future) {
  if (!future) throw lut::InvalidArgError("future is null");
  return *reinterpret_cast<fl::FutureTensor *>(future);
}
#endif  // LIBWAIFU_CUDA_ENABLED

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

int32_t fl_operators_create(fl_device_type_t device, fl_operators_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");

    std::shared_ptr<fl::Operators> operators =
        fl::getOperatorsSharedPtr(toDevice(device).getType());
    *out = reinterpret_cast<fl_operators_t>(new OperatorsHandle{std::move(operators), device});
    return clearError();
  });
}

void fl_operators_destroy(fl_operators_t operators) {
  delete reinterpret_cast<OperatorsHandle *>(operators);
}

int32_t fl_operators_get_device(fl_operators_t operators, fl_device_type_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = derefHandle(operators).device;
    return clearError();
  });
}

int32_t fl_get_last_error_code() {
  return gErrorCode;
}

const char *fl_get_last_error_message() {
  return gErrorMessage.c_str();
}

void fl_set_fatal_handler(fl_fatal_handler_t handler) {
  lut::internal::setFatalHandler(handler);
}

int32_t fl_tensor_zeros(
    fl_operators_t operators,
    const int32_t *shape,
    int32_t ndim,
    fl_dtype_t dtype,
    fl_tensor_t *out) {
  return guard([&]() {
    std::vector<int> dims = toShape(shape, ndim);
    return publish(deref(operators)->zeros(dims, toDType(dtype)), out);
  });
}

int32_t fl_tensor_empty(
    fl_operators_t operators,
    const int32_t *shape,
    int32_t ndim,
    fl_dtype_t dtype,
    fl_tensor_t *out) {
  return guard([&]() {
    std::vector<int> dims = toShape(shape, ndim);
    return publish(deref(operators)->tensor(dims, toDType(dtype)), out);
  });
}

int32_t fl_tensor_host_empty(
    fl_operators_t operators,
    const int32_t *shape,
    int32_t ndim,
    fl_dtype_t dtype,
    fl_tensor_t *out) {
  return guard([&]() {
    std::vector<int> dims = toShape(shape, ndim);
    return publish(deref(operators)->hostTensor(dims, toDType(dtype)), out);
  });
}

int32_t fl_tensor_from_data(
    fl_operators_t operators,
    const int32_t *shape,
    int32_t ndim,
    fl_dtype_t dtype,
    const void *data,
    int64_t data_size,
    fl_tensor_t *out) {
  return guard([&]() {
    if (!data && data_size != 0) throw lut::InvalidArgError("data is null");
    if (data_size < 0) throw lut::InvalidArgError("data_size must not be negative");

    if (derefHandle(operators).device != FL_DEVICE_CPU) {
      throw lut::InvalidArgError(
          "the bytes are copied in here and now, so this takes the CPU operators");
    }

    std::vector<int> dims = toShape(shape, ndim);
    fl::Tensor tensor = deref(operators)->tensor(dims, toDType(dtype));
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

int32_t fl_tensor_host_data(fl_tensor_t tensor, void **out, int64_t *nbytes) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    if (!nbytes) throw lut::InvalidArgError("nbytes is null");

    const fl::Tensor &x = deref(tensor);
    if (!x.getDevice().isHost()) {
      throw lut::InvalidArgError(
          "the bytes of a tensor on " + x.getDevice().getName() +
          " have no address the caller may touch");
    }
    if (!x.isContiguous()) {
      throw lut::InvalidArgError("a non-contiguous tensor's bytes are not one run");
    }

    *out = x.getInternalData()->getData<void>(x.getInternalOffset());
    *nbytes = getPackedSize(x);
    return FL_OK;
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

int32_t fl_tensor_contiguous(fl_operators_t operators, fl_tensor_t tensor, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->contiguous(deref(tensor)), out); });
}

int32_t fl_tensor_to_device(
    fl_operators_t operators,
    fl_tensor_t tensor,
    fl_device_type_t device,
    fl_tensor_t *out) {
  return guard([&]() {
    const fl::Tensor &source = deref(tensor);

    // Already there, so nothing is copied and no operator is asked anything. This is a question
    // about the two ends rather than about a backend.
    if (source.getDevice().getType() == toDevice(device).getType()) return publish(source, out);

    return publish(deref(operators)->toDevice(toDevice(device), source), out);
  });
}

#ifdef LIBWAIFU_CUDA_ENABLED

int32_t fl_tensor_to_device_async(
    fl_tensor_t tensor,
    fl_device_type_t device,
    fl_future_tensor_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");

    fl::FutureTensor future = fl::op::cuda::toDeviceAsync(toDevice(device), deref(tensor));
    *out = reinterpret_cast<fl_future_tensor_t>(new fl::FutureTensor(std::move(future)));
    return clearError();
  });
}

int32_t fl_future_tensor_take(fl_future_tensor_t future, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(future).take(), out); });
}

int32_t fl_future_tensor_take_sync(fl_future_tensor_t future, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(future).takeSync(), out); });
}

void fl_future_tensor_destroy(fl_future_tensor_t future) {
  delete reinterpret_cast<fl::FutureTensor *>(future);
}

#else  // LIBWAIFU_CUDA_ENABLED

namespace {

/// The one thing every entry point below has to say. Kept in one place so that a build without
/// CUDA says it the same way each time.
int32_t noCuda() {
  return setError(FL_ERROR_ABORTED, "this build has no CUDA support (needs WITH_CUDA=ON)");
}

}  // namespace

int32_t fl_tensor_to_device_async(fl_tensor_t, fl_device_type_t, fl_future_tensor_t *) {
  return noCuda();
}

int32_t fl_future_tensor_take(fl_future_tensor_t, fl_tensor_t *) {
  return noCuda();
}

int32_t fl_future_tensor_take_sync(fl_future_tensor_t, fl_tensor_t *) {
  return noCuda();
}

void fl_future_tensor_destroy(fl_future_tensor_t) {
}

#endif  // LIBWAIFU_CUDA_ENABLED

int32_t fl_tensor_cast(
    fl_operators_t operators,
    fl_tensor_t tensor,
    fl_dtype_t dtype,
    fl_tensor_t *out) {
  return guard(
      [&]() { return publish(deref(operators)->cast(deref(tensor), toDType(dtype)), out); });
}

int32_t fl_tensor_get_nbytes(fl_tensor_t tensor, int64_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = getPackedSize(deref(tensor));
    return clearError();
  });
}

int32_t fl_tensor_copy_to_host(
    fl_operators_t operators,
    fl_tensor_t tensor,
    void *buffer,
    int64_t buffer_size) {
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

    // Packed where it lies and moved afterwards, rather than the other way round: a transfer
    // reads one contiguous run, so a strided tensor has to be packed on the device it is on --
    // which is the device whose operators were handed in.
    if (!source.isContiguous()) source = deref(operators)->contiguous(source);

    // Only what is not already host memory crosses the bus. Page-locked host memory is read
    // where it lies, and the operators for it are the CPU's rather than the card's.
    if (!source.getDevice().isHost()) {
      source = deref(operators)->toDevice(fl::Device::getCpu(), source);
    }

    const void *data = source.getInternalData()->getData<void>(source.getInternalOffset());
    memcpy(buffer, data, static_cast<size_t>(nbytes));
    return clearError();
  });
}

int32_t fl_arange(
    fl_operators_t operators,
    int64_t begin,
    int64_t end,
    int64_t step,
    fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->arangeLong(begin, end, step), out); });
}

int32_t fl_rand(
    fl_operators_t operators,
    const int32_t *shape,
    int32_t ndim,
    fl_dtype_t dtype,
    fl_tensor_t *out) {
  return guard([&]() {
    std::vector<int> dims = toShape(shape, ndim);
    return publish(deref(operators)->rand(dims, toDType(dtype)), out);
  });
}

int32_t fl_randn(fl_operators_t operators, const int32_t *shape, int32_t ndim, fl_tensor_t *out) {
  return guard([&]() {
    std::vector<int> dims = toShape(shape, ndim);
    return publish(deref(operators)->randNormal(dims), out);
  });
}

int32_t fl_manual_seed(fl_operators_t operators, uint64_t seed) {
  return guard([&]() {
    deref(operators)->manualSeed(seed);
    return clearError();
  });
}

int32_t fl_lookup(
    fl_operators_t operators,
    fl_tensor_t table,
    fl_tensor_t indices,
    fl_tensor_t *out) {
  return guard(
      [&]() { return publish(deref(operators)->lookup(deref(table), deref(indices)), out); });
}

int32_t fl_rotary_embedding(
    fl_operators_t operators,
    fl_tensor_t positions,
    fl_tensor_t query,
    fl_tensor_t key,
    fl_tensor_t rotary_cache) {
  return guard([&]() {
    deref(operators)->rotaryEmbedding(
        deref(positions),
        deref(query),
        deref(key),
        deref(rotary_cache));
    return clearError();
  });
}

int32_t fl_rms_norm(
    fl_operators_t operators,
    fl_tensor_t input,
    fl_tensor_t weight,
    float eps,
    fl_tensor_t *out) {
  return guard(
      [&]() { return publish(deref(operators)->rmsNorm(deref(input), deref(weight), eps), out); });
}

int32_t fl_conv2d(
    fl_operators_t operators,
    fl_tensor_t input,
    fl_tensor_t weight,
    fl_tensor_t bias,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t groups,
    fl_tensor_t *out) {
  return guard([&]() {
    checkNotEmpty(deref(input), "the input");
    checkNotEmpty(deref(weight), "the weight");
    checkOnDevice(deref(input), operators, "the input");
    checkOnDevice(deref(weight), operators, "the weight");
    checkOptionalOnDevice(bias, operators, "the bias");

    fl::Tensor emptyTensor;
    const fl::Tensor &b = bias ? deref(bias) : emptyTensor;

    return publish(
        deref(operators)->conv2d(deref(input), deref(weight), b, stride, padding, dilation, groups),
        out);
  });
}

int32_t fl_conv1d(
    fl_operators_t operators,
    fl_tensor_t input,
    fl_tensor_t weight,
    fl_tensor_t bias,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t groups,
    fl_tensor_t *out) {
  return guard([&]() {
    checkNotEmpty(deref(input), "the input");
    checkNotEmpty(deref(weight), "the weight");
    checkOnDevice(deref(input), operators, "the input");
    checkOnDevice(deref(weight), operators, "the weight");
    checkOptionalOnDevice(bias, operators, "the bias");

    fl::Tensor emptyTensor;
    const fl::Tensor &b = bias ? deref(bias) : emptyTensor;

    return publish(
        deref(operators)->conv1d(deref(input), deref(weight), b, stride, padding, dilation, groups),
        out);
  });
}

int32_t fl_conv_transpose1d(
    fl_operators_t operators,
    fl_tensor_t input,
    fl_tensor_t weight,
    fl_tensor_t bias,
    int32_t stride,
    int32_t padding,
    int32_t output_padding,
    int32_t groups,
    fl_tensor_t *out) {
  return guard([&]() {
    checkNotEmpty(deref(input), "the input");
    checkNotEmpty(deref(weight), "the weight");
    checkOnDevice(deref(input), operators, "the input");
    checkOnDevice(deref(weight), operators, "the weight");
    checkOptionalOnDevice(bias, operators, "the bias");

    fl::Tensor emptyTensor;
    const fl::Tensor &b = bias ? deref(bias) : emptyTensor;

    return publish(
        deref(operators)->convTranspose1d(
            deref(input),
            deref(weight),
            b,
            stride,
            padding,
            output_padding,
            groups),
        out);
  });
}

int32_t fl_snake(
    fl_operators_t operators,
    fl_tensor_t input,
    fl_tensor_t alpha,
    fl_tensor_t beta,
    float eps,
    fl_tensor_t *out) {
  return guard([&]() {
    checkNotEmpty(deref(input), "the input");
    checkNotEmpty(deref(alpha), "alpha");
    checkOnDevice(deref(input), operators, "the input");
    checkOnDevice(deref(alpha), operators, "alpha");
    checkOptionalOnDevice(beta, operators, "beta");

    fl::Tensor emptyTensor;
    const fl::Tensor &b = beta ? deref(beta) : emptyTensor;

    return publish(deref(operators)->snake(deref(input), deref(alpha), b, eps), out);
  });
}

int32_t fl_stft(
    fl_operators_t operators,
    fl_tensor_t input,
    fl_tensor_t window,
    int32_t n_fft,
    int32_t hop,
    int32_t centered,
    fl_tensor_t *out) {
  return guard([&]() {
    checkNotEmpty(deref(input), "the input");
    checkNotEmpty(deref(window), "the window");
    checkOnDevice(deref(input), operators, "the input");
    checkOnDevice(deref(window), operators, "the window");

    return publish(
        deref(operators)->stft(deref(input), deref(window), n_fft, hop, centered != 0),
        out);
  });
}

int32_t fl_istft(
    fl_operators_t operators,
    fl_tensor_t spectrum,
    fl_tensor_t window,
    int32_t n_fft,
    int32_t hop,
    int32_t centered,
    fl_tensor_t *out) {
  return guard([&]() {
    checkNotEmpty(deref(spectrum), "the spectrum");
    checkNotEmpty(deref(window), "the window");
    checkOnDevice(deref(spectrum), operators, "the spectrum");
    checkOnDevice(deref(window), operators, "the window");

    return publish(
        deref(operators)->istft(deref(spectrum), deref(window), n_fft, hop, centered != 0),
        out);
  });
}

int32_t fl_group_norm(
    fl_operators_t operators,
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

    return publish(deref(operators)->groupNorm(deref(input), w, b, groups, eps), out);
  });
}

int32_t fl_upsample_nearest2d(
    fl_operators_t operators,
    fl_tensor_t input,
    int32_t scale,
    fl_tensor_t *out) {
  return guard(
      [&]() { return publish(deref(operators)->upsampleNearest2d(deref(input), scale), out); });
}

int32_t fl_layer_norm(
    fl_operators_t operators,
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

    return publish(deref(operators)->layerNorm(deref(input), w, b, eps), out);
  });
}

int32_t fl_quick_gelu(fl_operators_t operators, fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->quickGelu(deref(input)), out); });
}

int32_t fl_matmul(fl_operators_t operators, fl_tensor_t a, fl_tensor_t b, fl_tensor_t *out) {
  return guard([&]() {
    checkNotEmpty(deref(a), "the left operand");
    checkNotEmpty(deref(b), "the right operand");
    checkOnDevice(deref(a), operators, "the left operand");
    checkOnDevice(deref(b), operators, "the right operand");

    return publish(deref(operators)->matmul(deref(a), deref(b)), out);
  });
}

#ifdef LIBWAIFU_CUDA_ENABLED

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
  return setError(FL_ERROR_ABORTED, "this build has no NVFP4 support (needs WITH_CUDA=ON)");
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

#endif  // LIBWAIFU_CUDA_ENABLED

namespace {

/// Whether `device` has the kernels, asked without touching one. On CUDA `isFp8GemmAvailable()`
/// reads the card's architecture, which fails where there is no card, and answering that without
/// failing is this call's whole job.
///
/// CUDA is the only device that answers yes. Nothing about the format is specific to it -- the
/// bytes would mean the same on a processor, and `Fp8Operand` is device agnostic for that reason --
/// but the kernels that read them are the card's, so every other device is asked and says no here
/// rather than further in.
bool fp8Available(fl::Device::Type device) {
  if (!fl::isOperatorsAvailable(device)) return false;

#ifdef LIBWAIFU_CUDA_ENABLED
  if (device == fl::Device::kCuda) return fl::op::cuda::isFp8GemmAvailable();
#endif

  return false;
}

/// The kernels assert their preconditions with CHECK, which reports a broken invariant of ours.
/// What a caller could get wrong is checked here first, where it can be named as a bad argument.
void checkFp8Activation(const fl::Tensor &x, fl::Device::Type device, const char *what) {
  if (x.getDevice().getType() != device) {
    throw lut::InvalidArgError(std::string(what) + " is not on the same device as the weight");
  }
  if (!x.isContiguous()) {
    throw lut::InvalidArgError(std::string(what) + " is not contiguous");
  }
  if (x.getDim() < 2) {
    throw lut::InvalidArgError(std::string(what) + " has fewer than two dimensions");
  }

  // The activation is not narrowed on the way in: only the weight is narrow, and the multiply
  // happens in what the device computes in.
  if (x.getDType() != fl::DType::kFloat16) {
    throw lut::InvalidArgError(
        std::string(what) + " is not <float16>, which is what this device multiplies in");
  }
}

}  // namespace

int32_t fl_fp8_available(fl_device_type_t device, int32_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = fp8Available(toDevice(device).getType()) ? 1 : 0;
    return clearError();
  });
}

int32_t fl_fp8_quantize(fl_tensor_t x, fl_tensor_t *data, fl_tensor_t *channel_scale) {
  return guard([&]() {
    if (!data || !channel_scale) throw lut::InvalidArgError("out is null");

    const fl::Tensor &tensor = deref(x);
    if (!tensor.isContiguous()) {
      throw lut::InvalidArgError("the tensor to quantize is not contiguous");
    }
    if (tensor.getDim() != 2) {
      throw lut::InvalidArgError("the tensor to quantize is not two dimensional");
    }

    fl::Fp8Operand operand;
    if (tensor.getDevice().getType() == fl::Device::kCuda) {
#ifdef LIBWAIFU_CUDA_ENABLED
      if (tensor.getDType() != fl::DType::kFloat16) {
        throw lut::InvalidArgError("the tensor to quantize is not <float16>");
      }
      if (tensor.getShape(-1) % 16 != 0) {
        throw lut::InvalidArgError("the tensor to quantize: k is not a multiple of 16");
      }
      operand = fl::op::cuda::quantizeFp8(tensor);
#else
      throw lut::InvalidArgError("this build has no CUDA support (needs WITH_CUDA=ON)");
#endif
    } else {
      throw lut::InvalidArgError("the tensor to quantize is on a device with no FP8 kernels");
    }

    // Two handles have to appear together or not at all, and publish() is what allocates, so they
    // are made here and only handed over once both exist.
    std::unique_ptr<fl::Tensor> ownedData{new fl::Tensor(std::move(operand.data))};
    std::unique_ptr<fl::Tensor> ownedScale{new fl::Tensor(std::move(operand.channelScale))};

    *data = reinterpret_cast<fl_tensor_t>(ownedData.release());
    *channel_scale = reinterpret_cast<fl_tensor_t>(ownedScale.release());
    return clearError();
  });
}

int32_t fl_fp8_dequantize(fl_tensor_t data, fl_tensor_t channel_scale, fl_tensor_t *out) {
  // The return type is written down because a build without CUDA has nothing but the throw left
  // in here, and a lambda that only throws deduces void.
  return guard([&]() -> int32_t {
    fl::Fp8Operand operand = fl::makeFp8Operand(deref(data), deref(channel_scale));

#ifdef LIBWAIFU_CUDA_ENABLED
    if (operand.data.getDevice().getType() == fl::Device::kCuda) {
      return publish(fl::op::cuda::dequantFp8ToHalf(operand), out);
    }
#endif
    throw lut::InvalidArgError("the fp8 operand is on a device with no FP8 kernels");
  });
}

int32_t fl_fp8_matmul(
    fl_tensor_t a,
    fl_tensor_t data,
    fl_tensor_t channel_scale,
    fl_tensor_t *out) {
  // As in fl_fp8_dequantize: without CUDA every path out of here throws.
  return guard([&]() -> int32_t {
    fl::Fp8Operand operand = fl::makeFp8Operand(deref(data), deref(channel_scale));
    fl::Device::Type device = operand.data.getDevice().getType();

    checkFp8Activation(deref(a), device, "the left operand");
    if (deref(a).getShape(-1) != operand.k) {
      throw lut::InvalidArgError("the two operands disagree about k");
    }

#ifdef LIBWAIFU_CUDA_ENABLED
    if (device == fl::Device::kCuda) {
      // What the CUTLASS instantiation can read, rather than what the format can hold.
      if (operand.rows % 8 != 0) {
        throw lut::InvalidArgError("the fp8 operand's row count is not a multiple of 8");
      }
      if (operand.k % 16 != 0) {
        throw lut::InvalidArgError("the fp8 operand's k is not a multiple of 16");
      }
      return publish(fl::op::cuda::gemmFp8(deref(a), operand), out);
    }
#endif
    throw lut::InvalidArgError("the fp8 operand is on a device with no FP8 kernels");
  });
}

int32_t fl_mul(fl_operators_t operators, fl_tensor_t a, fl_tensor_t b, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->mul(deref(a), deref(b)), out); });
}

int32_t fl_add(fl_operators_t operators, fl_tensor_t a, fl_tensor_t b, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->add(deref(a), deref(b)), out); });
}

int32_t fl_sub(fl_operators_t operators, fl_tensor_t a, fl_tensor_t b, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->sub(deref(a), deref(b)), out); });
}

int32_t fl_eq(fl_operators_t operators, fl_tensor_t a, fl_tensor_t b, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->eq(deref(a), deref(b)), out); });
}

int32_t fl_div(fl_operators_t operators, fl_tensor_t a, fl_tensor_t b, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->divTensor(deref(a), deref(b)), out); });
}

int32_t fl_mul_scalar(fl_operators_t operators, fl_tensor_t input, float other, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->mul(deref(input), other), out); });
}

int32_t fl_div_scalar(fl_operators_t operators, fl_tensor_t input, float other, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->div(deref(input), other), out); });
}

int32_t fl_mod_scalar(
    fl_operators_t operators,
    fl_tensor_t input,
    int64_t other,
    fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->mod(deref(input), other), out); });
}

int32_t fl_square(fl_operators_t operators, fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->square(deref(input)), out); });
}

int32_t fl_neg(fl_operators_t operators, fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->neg(deref(input)), out); });
}

int32_t fl_abs(fl_operators_t operators, fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->abs(deref(input)), out); });
}

int32_t fl_exp(fl_operators_t operators, fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->exp(deref(input)), out); });
}

int32_t fl_sqrt(fl_operators_t operators, fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->sqrt(deref(input)), out); });
}

int32_t fl_rsqrt(fl_operators_t operators, fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->rsqrt(deref(input)), out); });
}

int32_t fl_sigmoid(fl_operators_t operators, fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->sigmoid(deref(input)), out); });
}

int32_t fl_tanh(fl_operators_t operators, fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->tanh(deref(input)), out); });
}

int32_t fl_relu(fl_operators_t operators, fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->relu(deref(input)), out); });
}

int32_t fl_gelu(fl_operators_t operators, fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->gelu(deref(input)), out); });
}

int32_t fl_silu(fl_operators_t operators, fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->silu(deref(input)), out); });
}

int32_t fl_sin(fl_operators_t operators, fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->sin(deref(input)), out); });
}

int32_t fl_cos(fl_operators_t operators, fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->cos(deref(input)), out); });
}

int32_t fl_softmax(fl_operators_t operators, fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->softmax(deref(input)), out); });
}

int32_t fl_swiglu(fl_operators_t operators, fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->swiglu(deref(input)), out); });
}

int32_t fl_geglu(fl_operators_t operators, fl_tensor_t input, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->geglu(deref(input)), out); });
}

int32_t fl_sum(fl_operators_t operators, fl_tensor_t input, int32_t dim, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->sum(deref(input), dim), out); });
}

int32_t fl_max(fl_operators_t operators, fl_tensor_t input, int32_t dim, fl_tensor_t *out) {
  return guard([&]() {
    checkLastDim(deref(input), dim, "fl_max");
    return publish(deref(operators)->max(deref(input)), out);
  });
}

int32_t fl_min(fl_operators_t operators, fl_tensor_t input, int32_t dim, fl_tensor_t *out) {
  return guard([&]() {
    checkLastDim(deref(input), dim, "fl_min");
    return publish(deref(operators)->min(deref(input)), out);
  });
}

int32_t fl_cat(
    fl_operators_t operators,
    fl_tensor_t a,
    fl_tensor_t b,
    int32_t dim,
    fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->cat(deref(a), deref(b), dim), out); });
}

int32_t fl_causal_mask(fl_operators_t operators, int32_t max_len, fl_tensor_t *out) {
  return guard([&]() { return publish(deref(operators)->causalMask(max_len), out); });
}

int32_t fl_attention(
    fl_operators_t operators,
    fl_tensor_t q,
    fl_tensor_t k,
    fl_tensor_t v,
    int32_t causal,
    fl_tensor_t *out) {
  return guard([&]() {
    return publish(deref(operators)->attention(deref(q), deref(k), deref(v), causal != 0), out);
  });
}

int32_t fl_paged_attention_available(int32_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
#ifdef LIBWAIFU_FLASH_ATTN_ENABLED
    *out = 1;
#else
    *out = 0;
#endif
    return clearError();
  });
}

int32_t fl_paged_attention(
    fl_operators_t operators,
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
        deref(operators)->pagedAttention(
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
    fl_operators_t operators,
    fl_tensor_t k,
    fl_tensor_t v,
    fl_tensor_t key_cache,
    fl_tensor_t value_cache,
    fl_tensor_t slot_mapping) {
  return guard([&]() {
    deref(operators)->storeKVCache(
        deref(k),
        deref(v),
        deref(key_cache),
        deref(value_cache),
        deref(slot_mapping));
    return clearError();
  });
}

int32_t fl_sample_with_params(
    fl_operators_t operators,
    fl_tensor_t logits,
    fl_tensor_t temperatures,
    fl_tensor_t top_ks,
    fl_tensor_t top_ps,
    fl_tensor_t *out) {
  return guard([&]() {
    checkOnDevice(deref(logits), operators, "the logits");
    checkOnDevice(deref(temperatures), operators, "the temperatures");
    checkOnDevice(deref(top_ks), operators, "the top-k values");
    checkOnDevice(deref(top_ps), operators, "the top-p values");

    return publish(
        deref(operators)->sample(deref(logits), deref(temperatures), deref(top_ks), deref(top_ps)),
        out);
  });
}

int32_t fl_repetition_penalty(
    fl_operators_t operators,
    fl_tensor_t logits,
    fl_tensor_t history,
    float weight) {
  return guard([&]() {
    deref(operators)->repetitionPenalty(deref(logits), deref(history), weight);
    return clearError();
  });
}

int32_t fl_copy(fl_operators_t operators, fl_tensor_t src, fl_tensor_t dest) {
  return guard([&]() {
    if (deref(src).getDType() != deref(dest).getDType()) {
      throw lut::InvalidArgError("the source and the destination hold different types");
    }
    deref(src).throwIfInvalidShape(deref(dest).getShape(), "fl_copy");

    deref(operators)->copy(deref(src), deref(dest));
    return clearError();
  });
}

int32_t fl_fill(fl_operators_t operators, fl_tensor_t tensor, float value) {
  return guard([&]() {
    deref(operators)->fill(deref(tensor), value);
    return clearError();
  });
}

int32_t fl_all_close(
    fl_operators_t operators,
    fl_tensor_t a,
    fl_tensor_t b,
    float rtol,
    float atol,
    int32_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = deref(operators)->allClose(deref(a), deref(b), rtol, atol) ? 1 : 0;
    return clearError();
  });
}

int32_t fl_all(fl_operators_t operators, fl_tensor_t tensor, int32_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = deref(operators)->all(deref(tensor)) ? 1 : 0;
    return clearError();
  });
}

int32_t fl_elem(fl_operators_t operators, fl_tensor_t tensor, float *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = deref(operators)->elem(deref(tensor));
    return clearError();
  });
}

int32_t fl_get_default_float_type(fl_operators_t operators, fl_dtype_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    fl::DType dtype = deref(operators)->getDefaultFloatType();
    *out = static_cast<fl_dtype_t>(static_cast<int16_t>(dtype));
    return clearError();
  });
}

int32_t fl_print(fl_operators_t operators, fl_tensor_t tensor) {
  return guard([&]() {
    deref(operators)->print(deref(tensor));
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

int32_t fl_memory_release_unused(fl_device_type_t device) {
  return guard([&]() {
    fl::MemorySnapshot::releaseUnused(toDevice(device));
    return clearError();
  });
}
