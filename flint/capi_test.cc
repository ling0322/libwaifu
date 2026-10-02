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

#include <string>
#include <vector>

#include "catch2/catch_amalgamated.hpp"

namespace {

/// Destroys a handle however the test leaves the scope, so a failed assertion does not leak.
class ScopedTensor {
 public:
  ScopedTensor() = default;
  ScopedTensor(const ScopedTensor &) = delete;
  ScopedTensor &operator=(const ScopedTensor &) = delete;

  ~ScopedTensor() {
    fl_tensor_destroy(_tensor);
  }

  fl_tensor_t *operator&() {
    return &_tensor;
  }

  operator fl_tensor_t() const {
    return _tensor;
  }

 private:
  fl_tensor_t _tensor = nullptr;
};

/// The same for the operators every operation is asked of.
class ScopedOperators {
 public:
  ScopedOperators() = default;
  ScopedOperators(const ScopedOperators &) = delete;
  ScopedOperators &operator=(const ScopedOperators &) = delete;

  ~ScopedOperators() {
    fl_operators_destroy(_operators);
  }

  fl_operators_t *operator&() {
    return &_operators;
  }

  operator fl_operators_t() const {
    return _operators;
  }

 private:
  fl_operators_t _operators = nullptr;
};

/// The CPU operators, which every test here starts by asking for.
void makeCpu(fl_operators_t *out) {
  fl_init();
  CATCH_REQUIRE(fl_operators_create(FL_DEVICE_CPU, out) == FL_OK);
}

/// Creates a CPU float tensor holding `data`, the input every operator test starts from.
void makeFloats(fl_operators_t operators, const std::vector<int32_t> &shape,
                const std::vector<float> &data, fl_tensor_t *out) {
  CATCH_REQUIRE(
      fl_tensor_from_data(
          operators,
          shape.data(),
          static_cast<int32_t>(shape.size()),
          FL_DTYPE_FLOAT,
          data.data(),
          static_cast<int64_t>(data.size() * sizeof(float)),
          out) == FL_OK);
}

std::vector<float> readFloats(fl_operators_t operators, fl_tensor_t tensor) {
  int64_t nbytes = 0;
  CATCH_REQUIRE(fl_tensor_get_nbytes(tensor, &nbytes) == FL_OK);

  std::vector<float> values(static_cast<size_t>(nbytes) / sizeof(float));
  CATCH_REQUIRE(fl_tensor_copy_to_host(operators, tensor, values.data(), nbytes) == FL_OK);
  return values;
}

}  // namespace

CATCH_TEST_CASE("flint C API hands out operators per device", "[core][flint][capi]") {
  fl_init();
  CATCH_REQUIRE(fl_get_last_error_code() == FL_OK);

  ScopedOperators cpu;
  CATCH_REQUIRE(fl_operators_create(FL_DEVICE_CPU, &cpu) == FL_OK);

  fl_device_type_t device = FL_DEVICE_UNKNOWN;
  CATCH_REQUIRE(fl_operators_get_device(cpu, &device) == FL_OK);
  CATCH_REQUIRE(device == FL_DEVICE_CPU);

  // A second handle on the same device is another reference rather than another backend.
  ScopedOperators again;
  CATCH_REQUIRE(fl_operators_create(FL_DEVICE_CPU, &again) == FL_OK);
  CATCH_REQUIRE(fl_operators_get_device(again, &device) == FL_OK);
  CATCH_REQUIRE(device == FL_DEVICE_CPU);

  // Page-locked host memory names memory rather than a processor, so it has no operators of its
  // own; nor does a device this build does not know about.
  ScopedOperators none;
  CATCH_REQUIRE(fl_operators_create(FL_DEVICE_CUDA_HOST, &none) != FL_OK);
  CATCH_REQUIRE(fl_operators_create(FL_DEVICE_UNKNOWN, &none) == FL_ERROR_INVALID_ARG);

  // Destroying a null handle is allowed, which is what lets callers clean up unconditionally.
  fl_operators_destroy(nullptr);
  CATCH_REQUIRE(fl_operators_get_device(nullptr, &device) == FL_ERROR_INVALID_ARG);
}

CATCH_TEST_CASE("flint C API reports tensor metadata", "[core][flint][capi]") {
  ScopedOperators cpu;
  makeCpu(&cpu);

  const std::vector<int32_t> shape{2, 3};
  ScopedTensor tensor;
  CATCH_REQUIRE(fl_tensor_zeros(cpu, shape.data(), 2, FL_DTYPE_FLOAT, &tensor) == FL_OK);

  int32_t dim = 0;
  CATCH_REQUIRE(fl_tensor_get_dim(tensor, &dim) == FL_OK);
  CATCH_REQUIRE(dim == 2);

  int32_t size = 0;
  CATCH_REQUIRE(fl_tensor_get_shape(tensor, 0, &size) == FL_OK);
  CATCH_REQUIRE(size == 2);
  CATCH_REQUIRE(fl_tensor_get_shape(tensor, -1, &size) == FL_OK);
  CATCH_REQUIRE(size == 3);

  int64_t numel = 0;
  CATCH_REQUIRE(fl_tensor_get_numel(tensor, &numel) == FL_OK);
  CATCH_REQUIRE(numel == 6);

  fl_dtype_t dtype = FL_DTYPE_UNKNOWN;
  CATCH_REQUIRE(fl_tensor_get_dtype(tensor, &dtype) == FL_OK);
  CATCH_REQUIRE(dtype == FL_DTYPE_FLOAT);

  fl_device_type_t device = FL_DEVICE_UNKNOWN;
  CATCH_REQUIRE(fl_tensor_get_device(tensor, &device) == FL_OK);
  CATCH_REQUIRE(device == FL_DEVICE_CPU);

  int32_t contiguous = 0;
  CATCH_REQUIRE(fl_tensor_is_contiguous(tensor, &contiguous) == FL_OK);
  CATCH_REQUIRE(contiguous == 1);

  CATCH_REQUIRE(readFloats(cpu, tensor) == std::vector<float>(6, 0.0f));
}

CATCH_TEST_CASE("flint C API round-trips element data", "[core][flint][capi]") {
  ScopedOperators cpu;
  makeCpu(&cpu);

  const std::vector<int32_t> shape{2, 2};
  const std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
  ScopedTensor tensor;
  CATCH_REQUIRE(
      fl_tensor_from_data(
          cpu,
          shape.data(),
          2,
          FL_DTYPE_FLOAT,
          data.data(),
          static_cast<int64_t>(data.size() * sizeof(float)),
          &tensor) == FL_OK);

  CATCH_REQUIRE(readFloats(cpu, tensor) == data);
}

CATCH_TEST_CASE("flint C API packs a transposed tensor on copy", "[core][flint][capi]") {
  ScopedOperators cpu;
  makeCpu(&cpu);

  const std::vector<int32_t> shape{2, 2};
  const std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};
  ScopedTensor tensor;
  makeFloats(cpu, shape, data, &tensor);

  ScopedTensor transposed;
  CATCH_REQUIRE(fl_tensor_transpose(tensor, 0, 1, &transposed) == FL_OK);

  int32_t contiguous = 1;
  CATCH_REQUIRE(fl_tensor_is_contiguous(transposed, &contiguous) == FL_OK);
  CATCH_REQUIRE(contiguous == 0);

  // The copy has to make it contiguous, so the caller sees the transposed order.
  CATCH_REQUIRE(readFloats(cpu, transposed) == std::vector<float>{1.0f, 3.0f, 2.0f, 4.0f});
}

CATCH_TEST_CASE("flint C API reshapes and slices a tensor", "[core][flint][capi]") {
  ScopedOperators cpu;
  makeCpu(&cpu);

  const std::vector<float> data{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  ScopedTensor tensor;
  makeFloats(cpu, {6}, data, &tensor);

  const std::vector<int32_t> viewShape{2, 3};
  ScopedTensor view;
  CATCH_REQUIRE(fl_tensor_view(tensor, viewShape.data(), 2, &view) == FL_OK);
  int32_t dim = 0;
  CATCH_REQUIRE(fl_tensor_get_dim(view, &dim) == FL_OK);
  CATCH_REQUIRE(dim == 2);

  ScopedTensor row;
  CATCH_REQUIRE(fl_tensor_subtensor(view, 1, &row) == FL_OK);
  CATCH_REQUIRE(readFloats(cpu, row) == std::vector<float>{3.0f, 4.0f, 5.0f});

  // FL_NONE leaves that end of the range where it is.
  ScopedTensor tail;
  CATCH_REQUIRE(fl_tensor_slice(tensor, 0, 4, FL_NONE, &tail) == FL_OK);
  CATCH_REQUIRE(readFloats(cpu, tail) == std::vector<float>{4.0f, 5.0f});

  ScopedTensor unsqueezed;
  CATCH_REQUIRE(fl_tensor_unsqueeze(tensor, 0, &unsqueezed) == FL_OK);
  CATCH_REQUIRE(fl_tensor_get_dim(unsqueezed, &dim) == FL_OK);
  CATCH_REQUIRE(dim == 2);

  ScopedTensor squeezed;
  CATCH_REQUIRE(fl_tensor_squeeze(unsqueezed, 0, &squeezed) == FL_OK);
  CATCH_REQUIRE(fl_tensor_get_dim(squeezed, &dim) == FL_OK);
  CATCH_REQUIRE(dim == 1);
}

CATCH_TEST_CASE("flint C API shares storage between handles", "[core][flint][capi]") {
  ScopedOperators cpu;
  makeCpu(&cpu);

  const std::vector<float> data{7.0f, 8.0f};
  ScopedTensor tensor;
  makeFloats(cpu, {2}, data, &tensor);

  ScopedTensor clone;
  CATCH_REQUIRE(fl_tensor_clone(tensor, &clone) == FL_OK);
  CATCH_REQUIRE(readFloats(cpu, clone) == data);

  // Destroying one handle leaves the storage alive for the other.
  fl_tensor_destroy(tensor);
  *(&tensor) = nullptr;
  CATCH_REQUIRE(readFloats(cpu, clone) == data);
}

CATCH_TEST_CASE("flint C API reports errors instead of throwing", "[core][flint][capi]") {
  ScopedOperators cpu;
  makeCpu(&cpu);

  const std::vector<int32_t> shape{2, 2};
  const std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f};

  // A size that does not match the shape is rejected before anything is allocated.
  ScopedTensor mismatched;
  CATCH_REQUIRE(
      fl_tensor_from_data(cpu, shape.data(), 2, FL_DTYPE_FLOAT, data.data(), 8, &mismatched) ==
      FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(fl_get_last_error_code() == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(std::string(fl_get_last_error_message()).find("data_size") != std::string::npos);

  ScopedTensor invalidDtype;
  CATCH_REQUIRE(
      fl_tensor_zeros(cpu, shape.data(), 2, FL_DTYPE_UNKNOWN, &invalidDtype) ==
      FL_ERROR_INVALID_ARG);

  int32_t dim = 0;
  CATCH_REQUIRE(fl_tensor_get_dim(nullptr, &dim) == FL_ERROR_INVALID_ARG);

  // An operation with no operators to run on is a bad argument like any other.
  ScopedTensor noOperators;
  CATCH_REQUIRE(
      fl_tensor_zeros(nullptr, shape.data(), 2, FL_DTYPE_FLOAT, &noOperators) ==
      FL_ERROR_INVALID_ARG);

  ScopedTensor tensor;
  CATCH_REQUIRE(fl_tensor_zeros(cpu, shape.data(), 2, FL_DTYPE_FLOAT, &tensor) == FL_OK);
  CATCH_REQUIRE(fl_get_last_error_code() == FL_OK);

  std::vector<float> tooSmall(2);
  CATCH_REQUIRE(
      fl_tensor_copy_to_host(cpu, tensor, tooSmall.data(), 8) == FL_ERROR_INVALID_ARG);

  // Destroying a null handle is allowed, which is what lets callers clean up unconditionally.
  fl_tensor_destroy(nullptr);
}

CATCH_TEST_CASE("flint C API runs the operators it is handed", "[core][flint][capi]") {
  ScopedOperators cpu;
  makeCpu(&cpu);

  ScopedTensor a;
  ScopedTensor b;
  makeFloats(cpu, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, &a);
  makeFloats(cpu, {2, 2}, {10.0f, 20.0f, 30.0f, 40.0f}, &b);

  ScopedTensor sum;
  CATCH_REQUIRE(fl_add(cpu, a, b, &sum) == FL_OK);
  CATCH_REQUIRE(readFloats(cpu, sum) == std::vector<float>{11.0f, 22.0f, 33.0f, 44.0f});

  ScopedTensor scaled;
  CATCH_REQUIRE(fl_mul_scalar(cpu, a, 2.0f, &scaled) == FL_OK);
  CATCH_REQUIRE(readFloats(cpu, scaled) == std::vector<float>{2.0f, 4.0f, 6.0f, 8.0f});

  ScopedTensor identity;
  makeFloats(cpu, {2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}, &identity);
  ScopedTensor product;
  CATCH_REQUIRE(fl_matmul(cpu, a, identity, &product) == FL_OK);
  CATCH_REQUIRE(readFloats(cpu, product) == readFloats(cpu, a));

  // The rows of a softmax sum to one, which is what fl_sum() should find.
  ScopedTensor probabilities;
  ScopedTensor rowSums;
  ScopedTensor ones;
  CATCH_REQUIRE(fl_softmax(cpu, a, &probabilities) == FL_OK);
  CATCH_REQUIRE(fl_sum(cpu, probabilities, -1, &rowSums) == FL_OK);
  makeFloats(cpu, {2}, {1.0f, 1.0f}, &ones);

  int32_t close = 0;
  CATCH_REQUIRE(fl_all_close(cpu, rowSums, ones, 1e-3f, 1e-5f, &close) == FL_OK);
  CATCH_REQUIRE(close == 1);

  ScopedTensor filled;
  CATCH_REQUIRE(
      fl_tensor_zeros(cpu, std::vector<int32_t>{2}.data(), 1, FL_DTYPE_FLOAT, &filled) == FL_OK);
  CATCH_REQUIRE(fl_fill(cpu, filled, 3.0f) == FL_OK);
  CATCH_REQUIRE(readFloats(cpu, filled) == std::vector<float>{3.0f, 3.0f});

  // A null handle is still reported rather than dereferenced.
  ScopedTensor unused;
  CATCH_REQUIRE(fl_add(cpu, nullptr, b, &unused) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(fl_fill(cpu, nullptr, 1.0f) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(fl_causal_mask(nullptr, 4, &unused) == FL_ERROR_INVALID_ARG);
}

CATCH_TEST_CASE("flint C API refuses tensors from another device", "[core][flint][capi]") {
  fl_init();

  int32_t available = 0;
  CATCH_REQUIRE(fl_is_device_available(FL_DEVICE_CUDA, &available) == FL_OK);
  if (!available) CATCH_SKIP("cuda device not available");

  ScopedOperators cpu;
  makeCpu(&cpu);
  ScopedOperators cuda;
  CATCH_REQUIRE(fl_operators_create(FL_DEVICE_CUDA, &cuda) == FL_OK);

  ScopedTensor host;
  makeFloats(cpu, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, &host);

  ScopedTensor there;
  CATCH_REQUIRE(fl_tensor_to_device(cuda, host, FL_DEVICE_CUDA, &there) == FL_OK);

  // The kernels check this with a fatal check, so it is checked here first and reported.
  ScopedTensor mixed;
  CATCH_REQUIRE(fl_matmul(cuda, host, there, &mixed) == FL_ERROR_INVALID_ARG);

  // Page-locked host memory comes from the CUDA operators, which are the ones that know how.
  ScopedTensor locked;
  CATCH_REQUIRE(
      fl_tensor_host_empty(cuda, std::vector<int32_t>{2, 2}.data(), 2, FL_DTYPE_FLOAT, &locked) ==
      FL_OK);
  fl_device_type_t device = FL_DEVICE_UNKNOWN;
  CATCH_REQUIRE(fl_tensor_get_device(locked, &device) == FL_OK);
  CATCH_REQUIRE(device == FL_DEVICE_CUDA_HOST);

  // And the bytes are the host's to write, whoever page-locked them.
  CATCH_REQUIRE(fl_copy(cpu, host, locked) == FL_OK);
  CATCH_REQUIRE(readFloats(cpu, locked) == std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
}

namespace {

/// Counts how often fl_tensor_from_external() gave its memory back.
void countRelease(void *context) {
  *static_cast<int *>(context) += 1;
}

}  // namespace

CATCH_TEST_CASE("flint C API borrows external bytes without copying", "[core][flint][capi]") {
  ScopedOperators cpu;
  makeCpu(&cpu);

  alignas(16) float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<int32_t> shape{2, 3};
  int released = 0;

  fl_tensor_t tensor = nullptr;
  CATCH_REQUIRE(
      fl_tensor_from_external(
          shape.data(),
          2,
          FL_DTYPE_FLOAT,
          data,
          sizeof(data),
          countRelease,
          &released,
          &tensor) == FL_OK);

  // Borrowed, not copied: a change to the lender's bytes is a change to the tensor.
  data[0] = 10.0f;
  CATCH_REQUIRE(readFloats(cpu, tensor) == std::vector<float>{10.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  // An operator reads it like any CPU tensor, and what it makes is its own.
  ScopedTensor sum;
  CATCH_REQUIRE(fl_add(cpu, tensor, tensor, &sum) == FL_OK);
  CATCH_REQUIRE(readFloats(cpu, sum) == std::vector<float>{20.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f});

  // Read-only: there is no writable pointer to be had.
  void *bytes = nullptr;
  int64_t nbytes = 0;
  CATCH_REQUIRE(fl_tensor_host_data(tensor, &bytes, &nbytes) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(std::string(fl_get_last_error_message()).find("read-only") != std::string::npos);

  // But its bytes may be read where they lie: they are the lender's.
  const void *readable = nullptr;
  CATCH_REQUIRE(fl_tensor_host_bytes(tensor, &readable, &nbytes) == FL_OK);
  CATCH_REQUIRE(readable == data);
  CATCH_REQUIRE(nbytes == static_cast<int64_t>(sizeof(data)));

  // Given back once, when the last of the tensor, a clone and a slice goes -- not before.
  fl_tensor_t clone = nullptr;
  fl_tensor_t slice = nullptr;
  CATCH_REQUIRE(fl_tensor_clone(tensor, &clone) == FL_OK);
  CATCH_REQUIRE(fl_tensor_slice(tensor, 1, 1, 3, &slice) == FL_OK);
  fl_tensor_destroy(tensor);
  fl_tensor_destroy(clone);
  CATCH_REQUIRE(released == 0);
  CATCH_REQUIRE(readFloats(cpu, slice) == std::vector<float>{2.0f, 3.0f, 5.0f, 6.0f});
  fl_tensor_destroy(slice);
  CATCH_REQUIRE(released == 1);
}

CATCH_TEST_CASE("flint C API gives borrowed bytes back when it refuses them", "[core][flint][capi]") {
  alignas(16) unsigned char data[32] = {};
  const std::vector<int32_t> shape{2, 2};
  int released = 0;

  // The wrong size.
  ScopedTensor mismatched;
  CATCH_REQUIRE(
      fl_tensor_from_external(
          shape.data(),
          2,
          FL_DTYPE_FLOAT,
          data,
          12,
          countRelease,
          &released,
          &mismatched) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(std::string(fl_get_last_error_message()).find("data_size") != std::string::npos);
  CATCH_REQUIRE(released == 1);

  // Floats that do not start on a float.
  ScopedTensor misaligned;
  CATCH_REQUIRE(
      fl_tensor_from_external(
          shape.data(),
          2,
          FL_DTYPE_FLOAT,
          data + 2,
          16,
          countRelease,
          &released,
          &misaligned) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(std::string(fl_get_last_error_message()).find("aligned") != std::string::npos);
  CATCH_REQUIRE(released == 2);

  // Bytes are aligned wherever they start.
  fl_tensor_t bytes = nullptr;
  CATCH_REQUIRE(
      fl_tensor_from_external(
          shape.data(),
          2,
          FL_DTYPE_UINT8,
          data + 3,
          4,
          countRelease,
          &released,
          &bytes) == FL_OK);
  CATCH_REQUIRE(released == 2);
  fl_tensor_destroy(bytes);
  CATCH_REQUIRE(released == 3);
}

CATCH_TEST_CASE("flint C API makes storage and views over it", "[core][flint][capi]") {
  fl_init();

  fl_tensor_data_t data = nullptr;
  CATCH_REQUIRE(fl_tensor_data_create(FL_DEVICE_CPU, FL_DTYPE_FLOAT, 24, &data) == FL_OK);

  int64_t numel = 0;
  fl_dtype_t dtype;
  fl_device_type_t device;
  CATCH_REQUIRE(fl_tensor_data_get_numel(data, &numel) == FL_OK);
  CATCH_REQUIRE(fl_tensor_data_get_dtype(data, &dtype) == FL_OK);
  CATCH_REQUIRE(fl_tensor_data_get_device(data, &device) == FL_OK);
  CATCH_REQUIRE(numel == 24);
  CATCH_REQUIRE(dtype == FL_DTYPE_FLOAT);
  CATCH_REQUIRE(device == FL_DEVICE_CPU);

  // A (3, 4) block of a (4, 5) layout, transposed: shape (4, 3), strides (1, 5), starting at 2.
  const int32_t shape[] = {4, 3};
  const int32_t stride[] = {1, 5};
  fl_tensor_view_t view = nullptr;
  CATCH_REQUIRE(fl_tensor_view_create(data, shape, stride, 2, 2, &view) == FL_OK);

  int32_t value = 0;
  int64_t offset = 0;
  CATCH_REQUIRE(fl_tensor_view_get_dim(view, &value) == FL_OK);
  CATCH_REQUIRE(value == 2);
  CATCH_REQUIRE(fl_tensor_view_get_shape(view, -1, &value) == FL_OK);
  CATCH_REQUIRE(value == 3);
  CATCH_REQUIRE(fl_tensor_view_get_stride(view, 1, &value) == FL_OK);
  CATCH_REQUIRE(value == 5);
  CATCH_REQUIRE(fl_tensor_view_get_offset(view, &offset) == FL_OK);
  CATCH_REQUIRE(offset == 2);
  CATCH_REQUIRE(fl_tensor_view_get_dtype(view, &dtype) == FL_OK);
  CATCH_REQUIRE(dtype == FL_DTYPE_FLOAT);
  CATCH_REQUIRE(fl_tensor_view_get_device(view, &device) == FL_OK);
  CATCH_REQUIRE(device == FL_DEVICE_CPU);
  CATCH_REQUIRE(fl_tensor_view_is_contiguous(view, &value) == FL_OK);
  CATCH_REQUIRE(value == 0);
  fl_tensor_view_destroy(view);

  const int32_t flat[] = {24};
  const int32_t one[] = {1};
  CATCH_REQUIRE(fl_tensor_view_create(data, flat, one, 1, 0, &view) == FL_OK);
  CATCH_REQUIRE(fl_tensor_view_is_contiguous(view, &value) == FL_OK);
  CATCH_REQUIRE(value == 1);
  fl_tensor_view_destroy(view);

  // A zero stride repeats one element, and reaches no further than it.
  const int32_t wide[] = {1000};
  const int32_t zero[] = {0};
  CATCH_REQUIRE(fl_tensor_view_create(data, wide, zero, 1, 23, &view) == FL_OK);
  fl_tensor_view_destroy(view);

  // The view spans 3 * 1 + 2 * 5 = 13 elements past its start: offset 10 ends on the last
  // element, 11 one past it.
  CATCH_REQUIRE(fl_tensor_view_create(data, shape, stride, 2, 10, &view) == FL_OK);
  fl_tensor_view_destroy(view);
  fl_tensor_view_t refused = nullptr;
  CATCH_REQUIRE(fl_tensor_view_create(data, shape, stride, 2, 11, &refused) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(fl_tensor_view_create(data, flat, one, 0, 0, &refused) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(fl_tensor_view_create(data, flat, one, 1, -1, &refused) == FL_ERROR_INVALID_ARG);
  const int32_t negative[] = {-1};
  CATCH_REQUIRE(fl_tensor_view_create(data, flat, negative, 1, 0, &refused) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(fl_tensor_view_create(nullptr, flat, one, 1, 0, &refused) == FL_ERROR_INVALID_ARG);

  // A size and stride whose product overflows 32 bits is still caught.
  const int32_t big[] = {65537};
  const int32_t bigStride[] = {65536};
  CATCH_REQUIRE(fl_tensor_view_create(data, big, bigStride, 1, 0, &refused) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(refused == nullptr);

  fl_tensor_data_destroy(data);
  fl_tensor_data_destroy(nullptr);
  fl_tensor_view_destroy(nullptr);

  fl_tensor_data_t none = nullptr;
  CATCH_REQUIRE(fl_tensor_data_create(FL_DEVICE_CPU, FL_DTYPE_FLOAT, 0, &none) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(none == nullptr);
}

CATCH_TEST_CASE("flint C API makes storage on the cuda device", "[core][flint][capi]") {
  fl_init();

  int32_t available = 0;
  CATCH_REQUIRE(fl_is_device_available(FL_DEVICE_CUDA, &available) == FL_OK);
  if (!available) CATCH_SKIP("cuda device not available");

  for (fl_device_type_t type : {FL_DEVICE_CUDA, FL_DEVICE_CUDA_HOST}) {
    fl_tensor_data_t data = nullptr;
    CATCH_REQUIRE(fl_tensor_data_create(type, FL_DTYPE_FLOAT16, 1024, &data) == FL_OK);
    fl_device_type_t device;
    CATCH_REQUIRE(fl_tensor_data_get_device(data, &device) == FL_OK);
    CATCH_REQUIRE(device == type);

    const int32_t shape[] = {32, 32};
    const int32_t stride[] = {32, 1};
    fl_tensor_view_t view = nullptr;
    CATCH_REQUIRE(fl_tensor_view_create(data, shape, stride, 2, 0, &view) == FL_OK);
    CATCH_REQUIRE(fl_tensor_view_get_device(view, &device) == FL_OK);
    CATCH_REQUIRE(device == type);
    fl_tensor_view_destroy(view);
    fl_tensor_data_destroy(data);
  }
}
