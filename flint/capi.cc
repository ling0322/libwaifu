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

#include <algorithm>
#include <atomic>
#include <exception>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "flint/cpu/external_tensor_data.h"
#include "flint/functional.h"
#include "flint/memory.h"
#include "flint/operators.h"
#include "flint/tensor_view.h"
#include "lutil/error.h"
#include "lutil/internal/log.h"
#ifdef LIBWAIFU_CUDA_ENABLED
#include "flint/cuda/fp8.h"
#include "flint/cuda/gemm_fp8_cutlass.h"
#include "flint/cuda/to_device.h"
#endif  // LIBWAIFU_CUDA_ENABLED
#ifdef LIBWAIFU_MLX_ENABLED
#include "mlx/memory.h"
#endif  // LIBWAIFU_MLX_ENABLED

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

fl_dtype_t fromDType(fl::DType dtype) {
  return static_cast<fl_dtype_t>(static_cast<int16_t>(dtype));
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
    case FL_DEVICE_VULKAN:
      return fl::Device(fl::Device::kVulkan);
    default:
      throw lut::InvalidArgError("invalid device");
  }
}

fl::TensorData &deref(fl_tensor_data_t data) {
  if (!data) throw lut::InvalidArgError("data is null");
  return *reinterpret_cast<fl::TensorData *>(data);
}

const fl::TensorView &deref(fl_tensor_view_t view) {
  if (!view) throw lut::InvalidArgError("view is null");
  return *reinterpret_cast<fl::TensorView *>(view);
}

/// A view an operation may be given or not: NULL is the empty view, which is how the operators
/// are told there is none.
fl::TensorView optional(fl_tensor_view_t view) {
  return view ? deref(view) : fl::TensorView();
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

/// Every view an operation reads or writes has to be on the operators' own device. The kernels
/// check that with the library's fatal check, which ends the process; asked here first, a caller
/// that mixed two devices is told about it instead.
void checkOnDevice(const fl::TensorView &view, fl_operators_t operators, const char *what) {
  fl::Device::Type device = toDevice(derefHandle(operators).device).getType();
  if (view.getDevice().getType() != device) {
    throw lut::InvalidArgError(
        std::string(what) + " is on " + view.getDevice().getName() + ", and these are the " +
        fl::Device(device).getName() + " operators");
  }
}

/// The same for a view an operation may be given or not.
void checkOptionalOnDevice(fl_tensor_view_t view, fl_operators_t operators, const char *what) {
  if (view) checkOnDevice(deref(view), operators, what);
}

/// An operation with nothing to say about a view that holds no elements.
void checkNotEmpty(const fl::TensorView &view, const char *what) {
  if (view.getNumEl() == 0) throw lut::InvalidArgError(std::string(what) + " is empty");
}

/// `view` has to have `shape`. What an elementwise operation writes is as large as what it reads,
/// so an `out` of any other shape is a write past its end rather than a wrong answer.
void checkShape(const fl::TensorView &view, const std::vector<int> &shape, const char *what) {
  if (view.getShape() != shape) {
    throw lut::InvalidArgError(
        std::string(what) + " is " + view.getShapeString() + " where " +
        fl::TensorShape(lut::makeConstSpan(shape)).toString() + " was expected");
  }
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

/// An operation of `operators` that reads `input` and writes `out` of the same shape.
template<typename Body>
int32_t elementwise(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t out,
    Body &&body) {
  return guard([&]() {
    const fl::TensorView &x = deref(input);
    const fl::TensorView &y = deref(out);
    checkOnDevice(x, operators, "the input");
    checkOnDevice(y, operators, "out");
    checkShape(y, x.getShape(), "out");
    body(deref(operators), x, y);
    return clearError();
  });
}

/// An operation of `operators` that reads `a` and `b`, both of the same shape, and writes `out` of
/// that shape too.
template<typename Body>
int32_t binary(
    fl_operators_t operators,
    fl_tensor_view_t a,
    fl_tensor_view_t b,
    fl_tensor_view_t out,
    Body &&body) {
  return guard([&]() {
    const fl::TensorView &x = deref(a);
    const fl::TensorView &y = deref(b);
    const fl::TensorView &z = deref(out);
    checkOnDevice(x, operators, "the left operand");
    checkOnDevice(y, operators, "the right operand");
    checkOnDevice(z, operators, "out");
    if (x.getDType() != y.getDType()) {
      throw lut::InvalidArgError("the two operands differ in type");
    }
    checkShape(y, x.getShape(), "the right operand");
    checkShape(z, x.getShape(), "out");
    body(deref(operators), x, y, z);
    return clearError();
  });
}

/// An operation of `operators` on the views it names, checked to be on its device, with nothing
/// else to check here.
template<typename Body>
int32_t onDevice(
    fl_operators_t operators,
    std::initializer_list<std::pair<fl_tensor_view_t, const char *>> views,
    Body &&body) {
  return guard([&]() {
    for (const auto &[view, what] : views) checkOnDevice(deref(view), operators, what);
    body(deref(operators));
    return clearError();
  });
}

}  // namespace

// --- Setup and errors -------------------------------------------------------------------------

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

int32_t fl_get_default_float_type(fl_operators_t operators, fl_dtype_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = fromDType(deref(operators)->getDefaultFloatType());
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

// The C API's levels are LogSeverity's, in the same order; a trampoline turns one into the other
// rather than a cast of the function pointer, which is not a thing C++ allows to be called.
static std::atomic<fl_log_sink_t> gCLogSink{nullptr};

static void toCLogSink(lut::LogSeverity severity, const char *source, const char *message) {
  fl_log_sink_t sink = gCLogSink.load(std::memory_order_acquire);
  if (sink) sink(static_cast<int32_t>(severity), source, message);
}

void fl_set_log_sink(fl_log_sink_t sink) {
  gCLogSink.store(sink, std::memory_order_release);
  lut::setLogSink(sink ? toCLogSink : nullptr);
}

void fl_set_log_level(int32_t level) {
  if (level < 0) level = 0;
  if (level > static_cast<int32_t>(lut::LogSeverity::kFATAL)) {
    level = static_cast<int32_t>(lut::LogSeverity::kFATAL);
  }
  lut::setLogLevel(static_cast<lut::LogSeverity>(level));
}

void fl_release_memory() {
#ifdef LIBWAIFU_MLX_ENABLED
  // MLX keeps a freed buffer for the next array of its size, up to the whole of memory: a model
  // that has been dropped is still the process's until this hands it back.
  try {
    mlx::core::clear_cache();
  } catch (const std::exception &e) {
    LOG(WARN) << "could not release the Metal buffer cache: " << e.what();
  }
#endif  // LIBWAIFU_MLX_ENABLED
}

// --- Storage ----------------------------------------------------------------------------------

int32_t fl_tensor_data_create(
    fl_device_type_t device,
    fl_dtype_t dtype,
    int64_t numel,
    fl_tensor_data_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    if (numel < 1) throw lut::InvalidArgError("storage holds at least one element");

    *out = reinterpret_cast<fl_tensor_data_t>(
        fl::F::allocate(toDevice(device), numel, toDType(dtype)).release());
    return clearError();
  });
}

int32_t fl_tensor_data_borrow(
    const void *data,
    fl_dtype_t dtype,
    int64_t numel,
    fl_tensor_data_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    if (!data) throw lut::InvalidArgError("data is null");
    if (numel < 1) throw lut::InvalidArgError("storage holds at least one element");
    if (numel > fl::TensorData::MaxNumEl) throw lut::InvalidArgError("too many elements");

    // The CPU kernels read elements as their own type. Storage the allocator made is always
    // aligned for that; bytes at an arbitrary offset into a file need not be, and are refused
    // rather than read misaligned.
    fl::DType type = toDType(dtype);
    int64_t alignment = std::max<int64_t>(1, type.getTotalSize(1));
    if (reinterpret_cast<uintptr_t>(data) % alignment != 0) {
      throw lut::InvalidArgError(
          "data is not aligned to its " + std::to_string(alignment) + "-byte elements");
    }

    *out = reinterpret_cast<fl_tensor_data_t>(
        fl::op::cpu::ExternalTensorData::create(data, numel, type).release());
    return clearError();
  });
}

void fl_tensor_data_destroy(fl_tensor_data_t data) {
  delete reinterpret_cast<fl::TensorData *>(data);
}

int32_t fl_tensor_data_get_host_ptr(fl_tensor_data_t data, void **out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    fl::TensorData &storage = deref(data);
    if (!storage.getDevice().isHost()) {
      throw lut::InvalidArgError(
          "storage on " + storage.getDevice().getName() + " has no address the caller may touch");
    }

    *out = storage.getRawData();
    return clearError();
  });
}

// --- Views ------------------------------------------------------------------------------------

int32_t fl_tensor_view_create(
    fl_tensor_data_t data,
    const int32_t *shape,
    const int32_t *stride,
    int32_t ndim,
    int64_t offset,
    fl_tensor_view_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    fl::TensorData *storage = &deref(data);
    if (ndim < 0) throw lut::InvalidArgError("ndim must not be negative");
    if (ndim > 0 && (!shape || !stride)) throw lut::InvalidArgError("shape or stride is null");
    if (offset < 0) throw lut::InvalidArgError("offset must not be negative");

    // The furthest element the view reaches, which has to be inside the storage. Worked out in 64
    // bits: a large stride times a large size is exactly where a 32-bit product would wrap.
    std::vector<fl::TensorShape::Elem> elems(static_cast<size_t>(ndim));
    int64_t last = offset;
    bool empty = false;
    for (int32_t d = 0; d < ndim; ++d) {
      if (shape[d] < 0) throw lut::InvalidArgError("shape must not be negative");
      if (stride[d] < 0) throw lut::InvalidArgError("stride must not be negative");
      elems[d].shape = shape[d];
      elems[d].stride = stride[d];
      if (shape[d] == 0) empty = true;
      last += static_cast<int64_t>(std::max(shape[d] - 1, 0)) * stride[d];
    }
    if (!empty && last >= storage->getNumEl()) {
      throw lut::InvalidArgError(
          "the view reaches element " + std::to_string(last) + " of storage holding " +
          std::to_string(storage->getNumEl()));
    }

    auto tensorShape = std::make_shared<fl::TensorShape>(lut::makeConstSpan(elems));
    *out = reinterpret_cast<fl_tensor_view_t>(new fl::TensorView(storage, tensorShape, offset));
    return clearError();
  });
}

void fl_tensor_view_destroy(fl_tensor_view_t view) {
  delete reinterpret_cast<fl::TensorView *>(view);
}

// --- Copies and conversions -------------------------------------------------------------------

int32_t fl_copy(fl_operators_t operators, fl_tensor_view_t src, fl_tensor_view_t dest) {
  return guard([&]() {
    const fl::TensorView &from = deref(src);
    const fl::TensorView &to = deref(dest);

    // The CPU operators copy between any two kinds of host memory, page-locked or not; every
    // other device copies within itself.
    if (derefHandle(operators).device == FL_DEVICE_CPU) {
      if (!from.getDevice().isHost() || !to.getDevice().isHost()) {
        throw lut::InvalidArgError("the CPU operators copy between host memory only");
      }
    } else {
      checkOnDevice(from, operators, "the source");
      checkOnDevice(to, operators, "the destination");
    }
    if (from.getDType() != to.getDType()) {
      throw lut::InvalidArgError("the source and the destination hold different types");
    }
    checkShape(to, from.getShape(), "the destination");

    deref(operators)->copy(from, to);
    return clearError();
  });
}

int32_t fl_cast(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out) {
  return elementwise(operators, input, out, [](fl::Operators *op, auto x, auto y) {
    op->cast(x, y);
  });
}

int32_t fl_transfer(fl_operators_t operators, fl_tensor_view_t src, fl_tensor_view_t dest) {
  return guard([&]() {
    const fl::TensorView &from = deref(src);
    const fl::TensorView &to = deref(dest);
    if (!from.isContiguous() || !to.isContiguous()) {
      throw lut::InvalidArgError("a transfer copies one contiguous run into another");
    }
    if (from.getDType() != to.getDType()) {
      throw lut::InvalidArgError("the source and the destination hold different types");
    }
    checkShape(to, from.getShape(), "the destination");

    deref(operators)->transfer(from, to);
    return clearError();
  });
}

#ifdef LIBWAIFU_CUDA_ENABLED

namespace {

/// What an fl_transfer_t points at: the event marking the end of the copy, cleared once it has
/// been seen through, and the storage the copy fills, which the caller owns.
struct TransferHandle {
  cudaEvent_t event;
  fl::TensorData *dest;
};

TransferHandle &deref(fl_transfer_t transfer) {
  if (!transfer) throw lut::InvalidArgError("transfer is null");
  return *reinterpret_cast<TransferHandle *>(transfer);
}

int32_t waitTransfer(fl_transfer_t transfer, bool sync) {
  return guard([&]() {
    TransferHandle &handle = deref(transfer);
    if (handle.event) {
      fl::op::cuda::completeTransfer(handle.dest, handle.event, sync);
      handle.event = nullptr;
    }
    return clearError();
  });
}

}  // namespace

int32_t fl_transfer_async(
    fl_tensor_view_t src,
    fl_device_type_t device,
    fl_tensor_data_t *dest,
    fl_transfer_t *out) {
  return guard([&]() {
    if (!dest || !out) throw lut::InvalidArgError("out is null");

    fl::op::cuda::PendingTransfer pending =
        fl::op::cuda::startTransferAsync(toDevice(device), deref(src));
    auto handle = std::make_unique<TransferHandle>(TransferHandle{pending.event, pending.dest.get()});
    *dest = reinterpret_cast<fl_tensor_data_t>(pending.dest.release());
    *out = reinterpret_cast<fl_transfer_t>(handle.release());
    return clearError();
  });
}

int32_t fl_transfer_wait(fl_transfer_t transfer) {
  return waitTransfer(transfer, false);
}

int32_t fl_transfer_wait_sync(fl_transfer_t transfer) {
  return waitTransfer(transfer, true);
}

void fl_transfer_destroy(fl_transfer_t transfer) {
  auto *handle = reinterpret_cast<TransferHandle *>(transfer);
  if (!handle) return;

  // A fetch nobody waited on. Destroying a pending event is allowed: the call returns at once and
  // the event goes once the device has reached it. The storage the copy writes is the caller's,
  // and goes back in the copy stream's order whenever the caller lets go of it.
  if (handle->event) cudaEventDestroy(handle->event);
  delete handle;
}

#else  // LIBWAIFU_CUDA_ENABLED

namespace {

int32_t noCuda() {
  return setError(FL_ERROR_ABORTED, "this build has no CUDA support (needs WITH_CUDA=ON)");
}

}  // namespace

int32_t fl_transfer_async(fl_tensor_view_t, fl_device_type_t, fl_tensor_data_t *, fl_transfer_t *) {
  return noCuda();
}

int32_t fl_transfer_wait(fl_transfer_t) {
  return noCuda();
}

int32_t fl_transfer_wait_sync(fl_transfer_t) {
  return noCuda();
}

void fl_transfer_destroy(fl_transfer_t) {
}

#endif  // LIBWAIFU_CUDA_ENABLED

// --- Filling ----------------------------------------------------------------------------------

int32_t fl_fill(fl_operators_t operators, fl_tensor_view_t tensor, float value) {
  return onDevice(operators, {{tensor, "the tensor"}}, [&](fl::Operators *op) {
    op->fill(deref(tensor), value);
  });
}

int32_t fl_rand(fl_operators_t operators, fl_tensor_view_t out) {
  return onDevice(operators, {{out, "out"}}, [&](fl::Operators *op) { op->rand(deref(out)); });
}

int32_t fl_randn(fl_operators_t operators, fl_tensor_view_t out) {
  return onDevice(operators, {{out, "out"}}, [&](fl::Operators *op) {
    if (deref(out).getDType() != fl::DType::kFloat) throw lut::InvalidArgError("out is not <float>");
    op->randNormal(deref(out));
  });
}

int32_t fl_manual_seed(fl_operators_t operators, uint64_t seed) {
  return guard([&]() {
    deref(operators)->manualSeed(seed);
    return clearError();
  });
}

int32_t fl_arange(fl_operators_t operators, int64_t begin, int64_t step, fl_tensor_view_t out) {
  return onDevice(operators, {{out, "out"}}, [&](fl::Operators *op) {
    const fl::TensorView &y = deref(out);
    if (y.getDType() != fl::DType::kLong || y.getDim() != 1) {
      throw lut::InvalidArgError("out is not a <long> vector");
    }
    if (y.getNumEl() > 0) op->arangeLong(begin, step, y);
  });
}

int32_t fl_causal_mask(fl_operators_t operators, fl_tensor_view_t out) {
  return onDevice(operators, {{out, "out"}}, [&](fl::Operators *op) {
    const fl::TensorView &y = deref(out);
    if (y.getDim() != 2 || y.getShape(0) != y.getShape(1) || y.getShape(0) < 1) {
      throw lut::InvalidArgError("out is not a square matrix");
    }
    op->causalMask(y);
  });
}

// --- Layers -----------------------------------------------------------------------------------

int32_t fl_lookup(
    fl_operators_t operators,
    fl_tensor_view_t table,
    fl_tensor_view_t indices,
    fl_tensor_view_t out) {
  return onDevice(
      operators,
      {{table, "the table"}, {indices, "the indices"}, {out, "out"}},
      [&](fl::Operators *op) { op->lookup(deref(table), deref(indices), deref(out)); });
}

int32_t fl_rotary_embedding(
    fl_operators_t operators,
    fl_tensor_view_t positions,
    fl_tensor_view_t query,
    fl_tensor_view_t key,
    fl_tensor_view_t rotary_cache) {
  return onDevice(
      operators,
      {{positions, "the positions"}, {query, "the query"}, {key, "the key"},
       {rotary_cache, "the rotary cache"}},
      [&](fl::Operators *op) {
        op->rotaryEmbedding(deref(positions), deref(query), deref(key), deref(rotary_cache));
      });
}

int32_t fl_rms_norm(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t weight,
    float eps,
    fl_tensor_view_t out) {
  return elementwise(operators, input, out, [&](fl::Operators *op, auto x, auto y) {
    checkOnDevice(deref(weight), operators, "the weight");
    op->rmsNorm(x, deref(weight), eps, y);
  });
}

int32_t fl_layer_norm(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t weight,
    fl_tensor_view_t bias,
    float eps,
    fl_tensor_view_t out) {
  return elementwise(operators, input, out, [&](fl::Operators *op, auto x, auto y) {
    checkOptionalOnDevice(weight, operators, "the weight");
    checkOptionalOnDevice(bias, operators, "the bias");
    op->layerNorm(x, optional(weight), optional(bias), eps, y);
  });
}

int32_t fl_group_norm(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t weight,
    fl_tensor_view_t bias,
    int32_t groups,
    float eps,
    fl_tensor_view_t out) {
  return elementwise(operators, input, out, [&](fl::Operators *op, auto x, auto y) {
    checkOptionalOnDevice(weight, operators, "the weight");
    checkOptionalOnDevice(bias, operators, "the bias");
    op->groupNorm(x, optional(weight), optional(bias), groups, eps, y);
  });
}

namespace {

/// What every convolution checks before it reaches a kernel.
void checkConvolutionViews(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t weight,
    fl_tensor_view_t bias,
    fl_tensor_view_t out) {
  checkNotEmpty(deref(input), "the input");
  checkNotEmpty(deref(weight), "the weight");
  checkOnDevice(deref(input), operators, "the input");
  checkOnDevice(deref(weight), operators, "the weight");
  checkOptionalOnDevice(bias, operators, "the bias");
  checkOnDevice(deref(out), operators, "out");
}

}  // namespace

int32_t fl_conv2d(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t weight,
    fl_tensor_view_t bias,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t groups,
    fl_tensor_view_t out) {
  return guard([&]() {
    checkConvolutionViews(operators, input, weight, bias, out);
    deref(operators)->conv2d(
        deref(input), deref(weight), optional(bias), stride, padding, dilation, groups, deref(out));
    return clearError();
  });
}

int32_t fl_conv1d(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t weight,
    fl_tensor_view_t bias,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t groups,
    fl_tensor_view_t out) {
  return guard([&]() {
    checkConvolutionViews(operators, input, weight, bias, out);
    deref(operators)->conv1d(
        deref(input), deref(weight), optional(bias), stride, padding, dilation, groups, deref(out));
    return clearError();
  });
}

int32_t fl_conv_transpose1d(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t weight,
    fl_tensor_view_t bias,
    int32_t stride,
    int32_t padding,
    int32_t output_padding,
    int32_t groups,
    fl_tensor_view_t out) {
  return guard([&]() {
    checkConvolutionViews(operators, input, weight, bias, out);
    deref(operators)->convTranspose1d(
        deref(input),
        deref(weight),
        optional(bias),
        stride,
        padding,
        output_padding,
        groups,
        deref(out));
    return clearError();
  });
}

int32_t fl_snake(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t alpha,
    fl_tensor_view_t beta,
    float eps,
    fl_tensor_view_t out) {
  return elementwise(operators, input, out, [&](fl::Operators *op, auto x, auto y) {
    checkNotEmpty(x, "the input");
    checkNotEmpty(deref(alpha), "alpha");
    checkOnDevice(deref(alpha), operators, "alpha");
    checkOptionalOnDevice(beta, operators, "beta");
    op->snake(x, deref(alpha), optional(beta), eps, y);
  });
}

int32_t fl_stft(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t window,
    int32_t n_fft,
    int32_t hop,
    int32_t centered,
    fl_tensor_view_t out) {
  return guard([&]() {
    checkNotEmpty(deref(input), "the input");
    checkNotEmpty(deref(window), "the window");
    checkOnDevice(deref(input), operators, "the input");
    checkOnDevice(deref(window), operators, "the window");
    checkOnDevice(deref(out), operators, "out");
    deref(operators)->stft(deref(input), deref(window), n_fft, hop, centered != 0, deref(out));
    return clearError();
  });
}

int32_t fl_istft(
    fl_operators_t operators,
    fl_tensor_view_t spectrum,
    fl_tensor_view_t window,
    int32_t n_fft,
    int32_t hop,
    int32_t centered,
    fl_tensor_view_t out) {
  return guard([&]() {
    checkNotEmpty(deref(spectrum), "the spectrum");
    checkNotEmpty(deref(window), "the window");
    checkOnDevice(deref(spectrum), operators, "the spectrum");
    checkOnDevice(deref(window), operators, "the window");
    checkOnDevice(deref(out), operators, "out");
    deref(operators)->istft(deref(spectrum), deref(window), n_fft, hop, centered != 0, deref(out));
    return clearError();
  });
}

int32_t fl_upsample_nearest2d(
    fl_operators_t operators,
    fl_tensor_view_t input,
    int32_t scale,
    fl_tensor_view_t out) {
  return onDevice(operators, {{input, "the input"}, {out, "out"}}, [&](fl::Operators *op) {
    op->upsampleNearest2d(deref(input), scale, deref(out));
  });
}

int32_t fl_upsample_nearest1d(
    fl_operators_t operators,
    fl_tensor_view_t input,
    fl_tensor_view_t out) {
  return onDevice(operators, {{input, "the input"}, {out, "out"}}, [&](fl::Operators *op) {
    op->upsampleNearest1d(deref(input), deref(out));
  });
}

int32_t fl_matmul(
    fl_operators_t operators,
    fl_tensor_view_t a,
    fl_tensor_view_t b,
    fl_tensor_view_t out) {
  return guard([&]() {
    checkNotEmpty(deref(a), "the left operand");
    checkNotEmpty(deref(b), "the right operand");
    checkOnDevice(deref(a), operators, "the left operand");
    checkOnDevice(deref(b), operators, "the right operand");
    checkOnDevice(deref(out), operators, "out");
    deref(operators)->matmul(deref(a), deref(b), deref(out));
    return clearError();
  });
}

// --- FP8 --------------------------------------------------------------------------------------

namespace {

/// Whether `device` has the kernels, asked without touching one. On CUDA `isFp8GemmAvailable()`
/// reads the card's architecture, which fails where there is no card, and answering that without
/// failing is this call's whole job.
bool fp8Available(fl::Device::Type device) {
  if (!fl::isOperatorsAvailable(device)) return false;

#ifdef LIBWAIFU_CUDA_ENABLED
  if (device == fl::Device::kCuda) return fl::op::cuda::isFp8GemmAvailable();
#endif

  return false;
}

/// The kernels assert their preconditions with CHECK, which reports a broken invariant of ours.
/// What a caller could get wrong is checked here first, where it can be named as a bad argument.
void checkFp8Weight(const fl::TensorView &data, const fl::TensorView &scale, bool perRow) {
  if (data.getDType() != fl::DType::kFp8E4M3 || data.getDim() != 2) {
    throw lut::InvalidArgError("fp8 operand: data is not <fp8e4m3>(rows, k)");
  }
  if (perRow) {
    if (scale.getDType() != fl::DType::kFloat || scale.getDim() != 1 ||
        scale.getShape(0) != data.getShape(0)) {
      throw lut::InvalidArgError("fp8 operand: channel scale is not <float>(rows)");
    }
  } else if (scale.getDType() != fl::DType::kFloat || scale.getNumEl() != 1) {
    throw lut::InvalidArgError("fp8 operand: the tensor scale is not a single <float>");
  }
  // A weight and its scales on different devices would reach a kernel as one pointer it may touch
  // and one it may not, and nothing would say so.
  if (data.getDevice().getType() != scale.getDevice().getType()) {
    throw lut::InvalidArgError("fp8 operand: the data and the scale are on different devices");
  }
  if (!data.isContiguous() || !scale.isContiguous()) {
    throw lut::InvalidArgError("fp8 operand: not contiguous");
  }
  // The kernels index the operand in int.
  if (data.getNumEl() >= std::numeric_limits<int32_t>::max()) {
    throw lut::InvalidArgError("fp8 operand: more elements than the kernels can index");
  }
}

void checkFp8Activation(const fl::TensorView &x, fl::Device::Type device, const char *what) {
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

/// `out` (..., rows) <float16>, contiguous, on `device`: what a product of `a` (..., k) by a weight
/// of `rows` rows writes.
void checkFp8Result(const fl::TensorView &out, const fl::TensorView &a, int rows, fl::Device::Type device) {
  if (out.getDevice().getType() != device || out.getDType() != fl::DType::kFloat16) {
    throw lut::InvalidArgError("out is not <float16> on the weight's device");
  }
  if (!out.isContiguous()) throw lut::InvalidArgError("out is not contiguous");
  std::vector<int> shape = a.getShape();
  shape.back() = rows;
  checkShape(out, shape, "out");
}

/// What the CUTLASS instantiation can read, rather than what the format can hold.
void checkFp8Tiles(int rows, int k) {
  if (rows % 8 != 0) throw lut::InvalidArgError("the fp8 operand's row count is not a multiple of 8");
  if (k % 16 != 0) throw lut::InvalidArgError("the fp8 operand's k is not a multiple of 16");
}

/// The product of `a` and the FP8 weight `data` with `scale`, per row or for the whole tensor.
int32_t fp8Matmul(
    fl_tensor_view_t a,
    fl_tensor_view_t data,
    fl_tensor_view_t scale,
    fl_tensor_view_t out,
    bool perRow) {
  // The return type is written down because a build without CUDA has nothing but the throw left
  // in here, and a lambda that only throws deduces void.
  return guard([&]() -> int32_t {
    const fl::TensorView &weight = deref(data);
    checkFp8Weight(weight, deref(scale), perRow);
    fl::Device::Type device = weight.getDevice().getType();
    int rows = weight.getShape(0);
    int k = weight.getShape(1);

    checkFp8Activation(deref(a), device, "the left operand");
    if (deref(a).getShape(-1) != k) throw lut::InvalidArgError("the two operands disagree about k");
    checkFp8Result(deref(out), deref(a), rows, device);

#ifdef LIBWAIFU_CUDA_ENABLED
    if (device == fl::Device::kCuda) {
      checkFp8Tiles(rows, k);
      if (perRow) {
        fl::op::cuda::gemmFp8(deref(a), weight, deref(scale), deref(out));
      } else {
        fl::op::cuda::gemmFp8TensorScale(deref(a), weight, deref(scale), deref(out));
      }
      return clearError();
    }
#endif
    throw lut::InvalidArgError("the fp8 operand is on a device with no FP8 kernels");
  });
}

}  // namespace

int32_t fl_fp8_available(fl_device_type_t device, int32_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = fp8Available(toDevice(device).getType()) ? 1 : 0;
    return clearError();
  });
}

int32_t fl_fp8_quantize(fl_tensor_view_t x, fl_tensor_view_t data, fl_tensor_view_t channel_scale) {
  return guard([&]() -> int32_t {
    const fl::TensorView &tensor = deref(x);
    if (!tensor.isContiguous()) throw lut::InvalidArgError("the tensor to quantize is not contiguous");
    if (tensor.getDim() != 2) {
      throw lut::InvalidArgError("the tensor to quantize is not two dimensional");
    }
    if (tensor.getDevice().getType() != fl::Device::kCuda) {
      throw lut::InvalidArgError("the tensor to quantize is on a device with no FP8 kernels");
    }

#ifdef LIBWAIFU_CUDA_ENABLED
    if (tensor.getDType() != fl::DType::kFloat16) {
      throw lut::InvalidArgError("the tensor to quantize is not <float16>");
    }
    if (tensor.getShape(-1) % 16 != 0) {
      throw lut::InvalidArgError("the tensor to quantize: k is not a multiple of 16");
    }

    const fl::TensorView &codes = deref(data);
    const fl::TensorView &scales = deref(channel_scale);
    checkFp8Weight(codes, scales, true);
    if (codes.getDevice().getType() != fl::Device::kCuda) {
      throw lut::InvalidArgError("the codes are not on the tensor's device");
    }
    checkShape(codes, tensor.getShape(), "the codes");

    fl::op::cuda::quantizeFp8(tensor, codes, scales);
    return clearError();
#else
    throw lut::InvalidArgError("this build has no CUDA support (needs WITH_CUDA=ON)");
#endif
  });
}

int32_t fl_fp8_dequantize(fl_tensor_view_t data, fl_tensor_view_t channel_scale, fl_tensor_view_t out) {
  return guard([&]() -> int32_t {
    const fl::TensorView &codes = deref(data);
    checkFp8Weight(codes, deref(channel_scale), true);

#ifdef LIBWAIFU_CUDA_ENABLED
    if (codes.getDevice().getType() == fl::Device::kCuda) {
      const fl::TensorView &y = deref(out);
      if (y.getDevice().getType() != fl::Device::kCuda || y.getDType() != fl::DType::kFloat16 ||
          !y.isContiguous()) {
        throw lut::InvalidArgError("out is not contiguous <float16> on the device");
      }
      checkShape(y, codes.getShape(), "out");

      fl::op::cuda::dequantFp8ToHalf(codes, deref(channel_scale), y);
      return clearError();
    }
#endif
    throw lut::InvalidArgError("the fp8 operand is on a device with no FP8 kernels");
  });
}

int32_t fl_fp8_matmul(
    fl_tensor_view_t a,
    fl_tensor_view_t data,
    fl_tensor_view_t channel_scale,
    fl_tensor_view_t out) {
  return fp8Matmul(a, data, channel_scale, out, true);
}

int32_t fl_fp8_matmul_tensor_scale(
    fl_tensor_view_t a,
    fl_tensor_view_t data,
    fl_tensor_view_t scale,
    fl_tensor_view_t out) {
  return fp8Matmul(a, data, scale, out, false);
}

// --- Elementwise ------------------------------------------------------------------------------

#define FL_CAPI_BINARY(name, method)                                                             \
  int32_t name(fl_operators_t operators, fl_tensor_view_t a, fl_tensor_view_t b, fl_tensor_view_t out) { \
    return binary(operators, a, b, out, [](fl::Operators *op, auto x, auto y, auto z) {         \
      op->method(x, y, z);                                                                      \
    });                                                                                         \
  }

FL_CAPI_BINARY(fl_add, add)
FL_CAPI_BINARY(fl_sub, sub)
FL_CAPI_BINARY(fl_mul, mul)
FL_CAPI_BINARY(fl_div, divTensor)
FL_CAPI_BINARY(fl_eq, eq)

#undef FL_CAPI_BINARY

int32_t fl_mul_scalar(fl_operators_t operators, fl_tensor_view_t input, float other, fl_tensor_view_t out) {
  return elementwise(operators, input, out, [&](fl::Operators *op, auto x, auto y) {
    op->mul(x, other, y);
  });
}

int32_t fl_div_scalar(fl_operators_t operators, fl_tensor_view_t input, float other, fl_tensor_view_t out) {
  return elementwise(operators, input, out, [&](fl::Operators *op, auto x, auto y) {
    op->div(x, other, y);
  });
}

int32_t fl_mod_scalar(fl_operators_t operators, fl_tensor_view_t input, int64_t other, fl_tensor_view_t out) {
  return elementwise(operators, input, out, [&](fl::Operators *op, auto x, auto y) {
    if (other == 0) throw lut::InvalidArgError("mod: by zero");
    op->mod(x, other, y);
  });
}

#define FL_CAPI_UNARY(name, method)                                                           \
  int32_t name(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out) {     \
    return elementwise(operators, input, out, [](fl::Operators *op, auto x, auto y) {        \
      op->method(x, y);                                                                       \
    });                                                                                       \
  }

FL_CAPI_UNARY(fl_square, square)
FL_CAPI_UNARY(fl_neg, neg)
FL_CAPI_UNARY(fl_abs, abs)
FL_CAPI_UNARY(fl_exp, exp)
FL_CAPI_UNARY(fl_log, log)
FL_CAPI_UNARY(fl_round, round)
FL_CAPI_UNARY(fl_sqrt, sqrt)
FL_CAPI_UNARY(fl_rsqrt, rsqrt)
FL_CAPI_UNARY(fl_sigmoid, sigmoid)
FL_CAPI_UNARY(fl_tanh, tanh)
FL_CAPI_UNARY(fl_relu, relu)
FL_CAPI_UNARY(fl_gelu, gelu)
FL_CAPI_UNARY(fl_silu, silu)
FL_CAPI_UNARY(fl_sin, sin)
FL_CAPI_UNARY(fl_cos, cos)
FL_CAPI_UNARY(fl_quick_gelu, quickGelu)
FL_CAPI_UNARY(fl_softmax, softmax)

#undef FL_CAPI_UNARY

int32_t fl_swiglu(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out) {
  return onDevice(operators, {{input, "the input"}, {out, "out"}}, [&](fl::Operators *op) {
    op->swiglu(deref(input), deref(out));
  });
}

int32_t fl_geglu(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out) {
  return onDevice(operators, {{input, "the input"}, {out, "out"}}, [&](fl::Operators *op) {
    op->geglu(deref(input), deref(out));
  });
}

// --- Reductions -------------------------------------------------------------------------------

namespace {

void checkDim(const fl::TensorView &input, int32_t dim, const char *what) {
  if (dim < 0 || dim >= input.getDim()) {
    throw lut::InvalidArgError(
        std::string(what) + ": no dimension " + std::to_string(dim) + " in a " +
        std::to_string(input.getDim()) + "-D tensor");
  }
}

}  // namespace

int32_t fl_sum(fl_operators_t operators, fl_tensor_view_t input, int32_t dim, fl_tensor_view_t out) {
  return onDevice(operators, {{input, "the input"}, {out, "out"}}, [&](fl::Operators *op) {
    checkDim(deref(input), dim, "fl_sum");
    op->sum(deref(input), dim, deref(out));
  });
}

int32_t fl_cumsum(fl_operators_t operators, fl_tensor_view_t input, int32_t dim, fl_tensor_view_t out) {
  return elementwise(operators, input, out, [&](fl::Operators *op, auto x, auto y) {
    checkDim(x, dim, "fl_cumsum");
    op->cumsum(x, dim, y);
  });
}

int32_t fl_max(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out) {
  return onDevice(operators, {{input, "the input"}, {out, "out"}}, [&](fl::Operators *op) {
    op->max(deref(input), deref(out));
  });
}

int32_t fl_min(fl_operators_t operators, fl_tensor_view_t input, fl_tensor_view_t out) {
  return onDevice(operators, {{input, "the input"}, {out, "out"}}, [&](fl::Operators *op) {
    op->min(deref(input), deref(out));
  });
}

// --- Attention and sampling -------------------------------------------------------------------

int32_t fl_attention(
    fl_operators_t operators,
    fl_tensor_view_t q,
    fl_tensor_view_t k,
    fl_tensor_view_t v,
    int32_t causal,
    fl_tensor_view_t out) {
  return onDevice(
      operators,
      {{q, "q"}, {k, "k"}, {v, "v"}, {out, "out"}},
      [&](fl::Operators *op) { op->attention(deref(q), deref(k), deref(v), causal != 0, deref(out)); });
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
    fl_tensor_view_t q,
    fl_tensor_view_t key_cache,
    fl_tensor_view_t value_cache,
    fl_tensor_view_t block_table,
    fl_tensor_view_t cu_seqlens_q,
    fl_tensor_view_t seqlens_k,
    int32_t max_q_len,
    int32_t max_k_len,
    int32_t causal,
    fl_tensor_view_t out) {
  return elementwise(operators, q, out, [&](fl::Operators *op, auto x, auto y) {
    op->pagedAttention(
        x,
        deref(key_cache),
        deref(value_cache),
        deref(block_table),
        deref(cu_seqlens_q),
        deref(seqlens_k),
        max_q_len,
        max_k_len,
        causal != 0,
        y);
  });
}

int32_t fl_store_kv_cache(
    fl_operators_t operators,
    fl_tensor_view_t k,
    fl_tensor_view_t v,
    fl_tensor_view_t key_cache,
    fl_tensor_view_t value_cache,
    fl_tensor_view_t slot_mapping) {
  return guard([&]() {
    deref(operators)->storeKVCache(
        deref(k), deref(v), deref(key_cache), deref(value_cache), deref(slot_mapping));
    return clearError();
  });
}

int32_t fl_sample_with_params(
    fl_operators_t operators,
    fl_tensor_view_t logits,
    fl_tensor_view_t temperatures,
    fl_tensor_view_t top_ks,
    fl_tensor_view_t top_ps,
    fl_tensor_view_t out) {
  return onDevice(
      operators,
      {{logits, "the logits"}, {temperatures, "the temperatures"}, {top_ks, "the top-k values"},
       {top_ps, "the top-p values"}, {out, "out"}},
      [&](fl::Operators *op) {
        op->sample(deref(logits), deref(temperatures), deref(top_ks), deref(top_ps), deref(out));
      });
}

int32_t fl_repetition_penalty(
    fl_operators_t operators,
    fl_tensor_view_t logits,
    fl_tensor_view_t history,
    float weight) {
  return guard([&]() {
    deref(operators)->repetitionPenalty(deref(logits), deref(history), weight);
    return clearError();
  });
}

// --- Reading results --------------------------------------------------------------------------

int32_t fl_all_close(
    fl_operators_t operators,
    fl_tensor_view_t a,
    fl_tensor_view_t b,
    float rtol,
    float atol,
    int32_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = deref(operators)->allClose(deref(a), deref(b), rtol, atol) ? 1 : 0;
    return clearError();
  });
}

int32_t fl_all(fl_operators_t operators, fl_tensor_view_t tensor, int32_t *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = deref(operators)->all(deref(tensor)) ? 1 : 0;
    return clearError();
  });
}

int32_t fl_elem(fl_operators_t operators, fl_tensor_view_t tensor, float *out) {
  return guard([&]() {
    if (!out) throw lut::InvalidArgError("out is null");
    *out = deref(operators)->elem(deref(tensor));
    return clearError();
  });
}

int32_t fl_print(fl_operators_t operators, fl_tensor_view_t tensor) {
  return guard([&]() {
    deref(operators)->print(deref(tensor));
    return clearError();
  });
}

// --- Memory -----------------------------------------------------------------------------------

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
