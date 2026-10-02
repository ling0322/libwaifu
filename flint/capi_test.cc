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

#include <string>
#include <vector>

#include "catch2/catch_amalgamated.hpp"

namespace {

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

/// What a binding builds a tensor out of: storage it owns and a view over it. Destroys both
/// however the test leaves the scope, the view first.
class Tensor {
 public:
  Tensor() = default;
  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;

  ~Tensor() {
    fl_tensor_view_destroy(view);
    fl_tensor_data_destroy(data);
  }

  operator fl_tensor_view_t() const {
    return view;
  }

  fl_tensor_data_t data = nullptr;
  fl_tensor_view_t view = nullptr;
};

/// Row-major strides for `shape`.
std::vector<int32_t> stridesOf(const std::vector<int32_t> &shape) {
  std::vector<int32_t> stride(shape.size());
  int32_t step = 1;
  for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
    stride[d] = step;
    step *= shape[d];
  }
  return stride;
}

int64_t numelOf(const std::vector<int32_t> &shape) {
  int64_t numel = 1;
  for (int32_t n : shape) numel *= n;
  return numel;
}

/// Fresh storage on `device` with a contiguous view of `shape` over it.
void makeEmpty(
    fl_device_type_t device,
    const std::vector<int32_t> &shape,
    fl_dtype_t dtype,
    Tensor *out) {
  CATCH_REQUIRE(fl_tensor_data_create(device, dtype, numelOf(shape), &out->data) == FL_OK);
  std::vector<int32_t> stride = stridesOf(shape);
  CATCH_REQUIRE(
      fl_tensor_view_create(
          out->data,
          shape.data(),
          stride.data(),
          static_cast<int32_t>(shape.size()),
          0,
          &out->view) == FL_OK);
}

/// The CPU operators, which every test here starts by asking for.
void makeCpu(fl_operators_t *out) {
  fl_init();
  CATCH_REQUIRE(fl_operators_create(FL_DEVICE_CPU, out) == FL_OK);
}

/// A CPU float tensor holding `values`, written in through the storage's own address.
void makeFloats(const std::vector<int32_t> &shape, const std::vector<float> &values, Tensor *out) {
  makeEmpty(FL_DEVICE_CPU, shape, FL_DTYPE_FLOAT, out);
  void *bytes = nullptr;
  CATCH_REQUIRE(fl_tensor_data_get_host_ptr(out->data, &bytes) == FL_OK);
  memcpy(bytes, values.data(), values.size() * sizeof(float));
}

/// The elements of a contiguous float view of host storage, read where they lie.
std::vector<float> readFloats(const Tensor &tensor) {
  int32_t contiguous = 0;
  CATCH_REQUIRE(fl_tensor_view_is_contiguous(tensor, &contiguous) == FL_OK);
  CATCH_REQUIRE(contiguous == 1);

  int32_t dim = 0;
  int64_t numel = 1;
  CATCH_REQUIRE(fl_tensor_view_get_dim(tensor, &dim) == FL_OK);
  for (int32_t d = 0; d < dim; ++d) {
    int32_t size = 0;
    CATCH_REQUIRE(fl_tensor_view_get_shape(tensor, d, &size) == FL_OK);
    numel *= size;
  }
  int64_t offset = 0;
  CATCH_REQUIRE(fl_tensor_view_get_offset(tensor, &offset) == FL_OK);

  void *bytes = nullptr;
  CATCH_REQUIRE(fl_tensor_data_get_host_ptr(tensor.data, &bytes) == FL_OK);
  const float *first = static_cast<const float *>(bytes) + offset;
  return std::vector<float>(first, first + numel);
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

  // A view of no dimensions, which needs no shape to be handed over.
  CATCH_REQUIRE(fl_tensor_view_create(data, nullptr, nullptr, 0, 0, &view) == FL_OK);
  CATCH_REQUIRE(fl_tensor_view_get_dim(view, &value) == FL_OK);
  CATCH_REQUIRE(value == 0);
  fl_tensor_view_destroy(view);

  // The view spans 3 * 1 + 2 * 5 = 13 elements past its start: offset 10 ends on the last
  // element, 11 one past it.
  CATCH_REQUIRE(fl_tensor_view_create(data, shape, stride, 2, 10, &view) == FL_OK);
  fl_tensor_view_destroy(view);
  fl_tensor_view_t refused = nullptr;
  CATCH_REQUIRE(fl_tensor_view_create(data, shape, stride, 2, 11, &refused) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(fl_tensor_view_create(data, flat, one, -1, 0, &refused) == FL_ERROR_INVALID_ARG);
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
  CATCH_REQUIRE(fl_tensor_data_create(FL_DEVICE_CPU, FL_DTYPE_UNKNOWN, 4, &none) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(none == nullptr);
}

CATCH_TEST_CASE("flint C API runs the operators it is handed", "[core][flint][capi]") {
  ScopedOperators cpu;
  makeCpu(&cpu);

  Tensor a, b;
  makeFloats({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, &a);
  makeFloats({2, 2}, {10.0f, 20.0f, 30.0f, 40.0f}, &b);

  Tensor sum;
  makeEmpty(FL_DEVICE_CPU, {2, 2}, FL_DTYPE_FLOAT, &sum);
  CATCH_REQUIRE(fl_add(cpu, a, b, sum) == FL_OK);
  CATCH_REQUIRE(readFloats(sum) == std::vector<float>{11.0f, 22.0f, 33.0f, 44.0f});

  Tensor scaled;
  makeEmpty(FL_DEVICE_CPU, {2, 2}, FL_DTYPE_FLOAT, &scaled);
  CATCH_REQUIRE(fl_mul_scalar(cpu, a, 2.0f, scaled) == FL_OK);
  CATCH_REQUIRE(readFloats(scaled) == std::vector<float>{2.0f, 4.0f, 6.0f, 8.0f});

  Tensor identity, product;
  makeFloats({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f}, &identity);
  makeEmpty(FL_DEVICE_CPU, {2, 2}, FL_DTYPE_FLOAT, &product);
  CATCH_REQUIRE(fl_matmul(cpu, a, identity, product) == FL_OK);
  CATCH_REQUIRE(readFloats(product) == readFloats(a));

  // The rows of a softmax sum to one, which is what fl_sum() should find.
  Tensor probabilities, rowSums, ones;
  makeEmpty(FL_DEVICE_CPU, {2, 2}, FL_DTYPE_FLOAT, &probabilities);
  makeEmpty(FL_DEVICE_CPU, {2}, FL_DTYPE_FLOAT, &rowSums);
  makeFloats({2}, {1.0f, 1.0f}, &ones);
  CATCH_REQUIRE(fl_softmax(cpu, a, probabilities) == FL_OK);
  CATCH_REQUIRE(fl_sum(cpu, probabilities, 1, rowSums) == FL_OK);

  int32_t close = 0;
  CATCH_REQUIRE(fl_all_close(cpu, rowSums, ones, 1e-3f, 1e-5f, &close) == FL_OK);
  CATCH_REQUIRE(close == 1);

  CATCH_REQUIRE(fl_fill(cpu, ones, 3.0f) == FL_OK);
  CATCH_REQUIRE(readFloats(ones) == std::vector<float>{3.0f, 3.0f});

  // A broadcast operand is a view the caller expanded: (2) seen as (2, 2) by a zero stride.
  const int32_t shape[] = {2, 2};
  const int32_t stride[] = {0, 1};
  fl_tensor_view_t row = nullptr;
  CATCH_REQUIRE(fl_tensor_view_create(ones.data, shape, stride, 2, 0, &row) == FL_OK);
  CATCH_REQUIRE(fl_add(cpu, a, row, sum) == FL_OK);
  fl_tensor_view_destroy(row);
  CATCH_REQUIRE(readFloats(sum) == std::vector<float>{4.0f, 5.0f, 6.0f, 7.0f});
}

CATCH_TEST_CASE("flint C API packs a transposed view on copy", "[core][flint][capi]") {
  ScopedOperators cpu;
  makeCpu(&cpu);

  Tensor tensor;
  makeFloats({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, &tensor);

  const int32_t shape[] = {2, 2};
  const int32_t stride[] = {1, 2};
  fl_tensor_view_t transposed = nullptr;
  CATCH_REQUIRE(fl_tensor_view_create(tensor.data, shape, stride, 2, 0, &transposed) == FL_OK);

  Tensor packed;
  makeEmpty(FL_DEVICE_CPU, {2, 2}, FL_DTYPE_FLOAT, &packed);
  CATCH_REQUIRE(fl_copy(cpu, transposed, packed) == FL_OK);
  fl_tensor_view_destroy(transposed);
  CATCH_REQUIRE(readFloats(packed) == std::vector<float>{1.0f, 3.0f, 2.0f, 4.0f});

  // A slice of a row is a view at an offset, and is read from there.
  const int32_t tail[] = {1};
  const int32_t unit[] = {1};
  fl_tensor_view_t last = nullptr;
  CATCH_REQUIRE(fl_tensor_view_create(tensor.data, tail, unit, 1, 3, &last) == FL_OK);
  Tensor copied;
  makeEmpty(FL_DEVICE_CPU, {1}, FL_DTYPE_FLOAT, &copied);
  CATCH_REQUIRE(fl_copy(cpu, last, copied) == FL_OK);
  fl_tensor_view_destroy(last);
  CATCH_REQUIRE(readFloats(copied) == std::vector<float>{4.0f});
}

CATCH_TEST_CASE("flint C API reports errors instead of throwing", "[core][flint][capi]") {
  ScopedOperators cpu;
  makeCpu(&cpu);

  Tensor a, b, wrong;
  makeFloats({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, &a);
  makeFloats({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, &b);
  makeEmpty(FL_DEVICE_CPU, {3}, FL_DTYPE_FLOAT, &wrong);

  // An out of another shape would be written past its end, so it is refused.
  CATCH_REQUIRE(fl_add(cpu, a, b, wrong) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(fl_get_last_error_code() == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(std::string(fl_get_last_error_message()).find("out") != std::string::npos);
  CATCH_REQUIRE(fl_exp(cpu, a, wrong) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(fl_copy(cpu, a, wrong) == FL_ERROR_INVALID_ARG);

  // So is an operand that was not broadcast to the other's shape.
  CATCH_REQUIRE(fl_add(cpu, a, wrong, b) == FL_ERROR_INVALID_ARG);

  // Null handles are reported rather than dereferenced.
  CATCH_REQUIRE(fl_add(cpu, nullptr, b, a) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(fl_add(nullptr, a, b, a) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(fl_fill(cpu, nullptr, 1.0f) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(fl_causal_mask(nullptr, a) == FL_ERROR_INVALID_ARG);
  int32_t dim = 0;
  CATCH_REQUIRE(fl_tensor_view_get_dim(nullptr, &dim) == FL_ERROR_INVALID_ARG);

  // An operation that succeeds clears what the last one left.
  CATCH_REQUIRE(fl_add(cpu, a, b, a) == FL_OK);
  CATCH_REQUIRE(fl_get_last_error_code() == FL_OK);
  CATCH_REQUIRE(readFloats(a) == std::vector<float>{2.0f, 4.0f, 6.0f, 8.0f});
}

CATCH_TEST_CASE("flint C API borrows external bytes without copying", "[core][flint][capi]") {
  ScopedOperators cpu;
  makeCpu(&cpu);

  alignas(16) float values[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor tensor;
  CATCH_REQUIRE(fl_tensor_data_borrow(values, FL_DTYPE_FLOAT, 6, &tensor.data) == FL_OK);
  const int32_t shape[] = {2, 3};
  const int32_t stride[] = {3, 1};
  CATCH_REQUIRE(fl_tensor_view_create(tensor.data, shape, stride, 2, 0, &tensor.view) == FL_OK);

  // Borrowed, not copied: the storage's address is the lender's, and a change to the lender's
  // bytes is a change to what an operator reads.
  void *bytes = nullptr;
  CATCH_REQUIRE(fl_tensor_data_get_host_ptr(tensor.data, &bytes) == FL_OK);
  CATCH_REQUIRE(bytes == values);
  values[0] = 10.0f;

  Tensor sum;
  makeEmpty(FL_DEVICE_CPU, {2, 3}, FL_DTYPE_FLOAT, &sum);
  CATCH_REQUIRE(fl_add(cpu, tensor, tensor, sum) == FL_OK);
  CATCH_REQUIRE(readFloats(sum) == std::vector<float>{20.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f});

  // Floats that do not start on a float are refused; bytes are aligned wherever they start.
  alignas(16) unsigned char raw[32] = {};
  fl_tensor_data_t refused = nullptr;
  CATCH_REQUIRE(fl_tensor_data_borrow(raw + 2, FL_DTYPE_FLOAT, 4, &refused) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(std::string(fl_get_last_error_message()).find("aligned") != std::string::npos);
  CATCH_REQUIRE(fl_tensor_data_borrow(nullptr, FL_DTYPE_FLOAT, 4, &refused) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(refused == nullptr);

  fl_tensor_data_t unaligned = nullptr;
  CATCH_REQUIRE(fl_tensor_data_borrow(raw + 3, FL_DTYPE_UINT8, 4, &unaligned) == FL_OK);
  fl_tensor_data_destroy(unaligned);
}

CATCH_TEST_CASE("flint C API makes storage on the cuda device", "[core][flint][capi]") {
  fl_init();

  int32_t available = 0;
  CATCH_REQUIRE(fl_is_device_available(FL_DEVICE_CUDA, &available) == FL_OK);
  if (!available) CATCH_SKIP("cuda device not available");

  for (fl_device_type_t type : {FL_DEVICE_CUDA, FL_DEVICE_CUDA_HOST}) {
    Tensor tensor;
    makeEmpty(type, {32, 32}, FL_DTYPE_FLOAT16, &tensor);
    fl_device_type_t device;
    CATCH_REQUIRE(fl_tensor_data_get_device(tensor.data, &device) == FL_OK);
    CATCH_REQUIRE(device == type);
    CATCH_REQUIRE(fl_tensor_view_get_device(tensor, &device) == FL_OK);
    CATCH_REQUIRE(device == type);

    // Page-locked memory is the host's to touch; the card's is not.
    void *bytes = nullptr;
    int32_t status = fl_tensor_data_get_host_ptr(tensor.data, &bytes);
    CATCH_REQUIRE(status == (type == FL_DEVICE_CUDA_HOST ? FL_OK : FL_ERROR_INVALID_ARG));
  }
}

CATCH_TEST_CASE("flint C API moves views between devices", "[core][flint][capi]") {
  fl_init();

  int32_t available = 0;
  CATCH_REQUIRE(fl_is_device_available(FL_DEVICE_CUDA, &available) == FL_OK);
  if (!available) CATCH_SKIP("cuda device not available");

  ScopedOperators cpu;
  makeCpu(&cpu);
  ScopedOperators cuda;
  CATCH_REQUIRE(fl_operators_create(FL_DEVICE_CUDA, &cuda) == FL_OK);

  const std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
  Tensor host, there, sum, back;
  makeFloats({2, 2}, values, &host);
  makeEmpty(FL_DEVICE_CUDA, {2, 2}, FL_DTYPE_FLOAT, &there);
  makeEmpty(FL_DEVICE_CUDA, {2, 2}, FL_DTYPE_FLOAT, &sum);
  makeEmpty(FL_DEVICE_CPU, {2, 2}, FL_DTYPE_FLOAT, &back);

  CATCH_REQUIRE(fl_transfer(cuda, host, there) == FL_OK);
  CATCH_REQUIRE(fl_add(cuda, there, there, sum) == FL_OK);
  CATCH_REQUIRE(fl_transfer(cuda, sum, back) == FL_OK);
  CATCH_REQUIRE(readFloats(back) == std::vector<float>{2.0f, 4.0f, 6.0f, 8.0f});

  // The kernels check this with a fatal check, so it is checked here first and reported.
  CATCH_REQUIRE(fl_matmul(cuda, host, there, sum) == FL_ERROR_INVALID_ARG);

  // And the bytes of page-locked memory are the host's to write, whoever page-locked them.
  Tensor locked;
  makeEmpty(FL_DEVICE_CUDA_HOST, {2, 2}, FL_DTYPE_FLOAT, &locked);
  CATCH_REQUIRE(fl_copy(cpu, host, locked) == FL_OK);
  CATCH_REQUIRE(readFloats(locked) == values);

  // From there a copy can be started and waited on later; what it fills is the caller's.
  Tensor arrived;
  fl_transfer_t transfer = nullptr;
  CATCH_REQUIRE(fl_transfer_async(locked, FL_DEVICE_CUDA, &arrived.data, &transfer) == FL_OK);
  const int32_t shape[] = {2, 2};
  const int32_t stride[] = {2, 1};
  CATCH_REQUIRE(fl_tensor_view_create(arrived.data, shape, stride, 2, 0, &arrived.view) == FL_OK);
  CATCH_REQUIRE(fl_transfer_wait(transfer) == FL_OK);
  CATCH_REQUIRE(fl_transfer_wait_sync(transfer) == FL_OK);
  fl_transfer_destroy(transfer);
  CATCH_REQUIRE(fl_transfer(cuda, arrived, back) == FL_OK);
  CATCH_REQUIRE(readFloats(back) == values);

  // Only page-locked memory to the card goes asynchronously; a pageable source is refused.
  fl_tensor_data_t refused = nullptr;
  CATCH_REQUIRE(fl_transfer_async(host, FL_DEVICE_CUDA, &refused, &transfer) == FL_ERROR_INVALID_ARG);
  CATCH_REQUIRE(refused == nullptr);
  fl_transfer_destroy(nullptr);
}
