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

#include "flint/tensor.h"

#include <stdlib.h>

#include <limits>

#include "lutil/error.h"
#include "lutil/strings.h"
#include "flint/cpu/common.h"
#include "flint/cpu/cpu_tensor_data.h"
#include "flint/tensor_view.h"
#include "flint/operators.h"

namespace fl {

template<typename T>
Tensor Tensor::create(std::initializer_list<int> shape, lut::Span<const T> data) {
  Tensor tensor;

  tensor._shape = std::make_shared<TensorShape>(shape);
  int64_t numel = tensor._shape->getNumEl();

  DType dtype = DType::getType<T>();
  tensor._data = op::cpu::CpuTensorData::create(numel, dtype);
  tensor._offset = 0;

  // fill data
  if (numel != static_cast<int64_t>(data.size())) {
    THROW(
        InvalidArg,
        lut::sprintf("a shape of %d elements was given %d", numel, int(data.size())));
  }
  std::copy(data.begin(), data.end(), op::cpu::getDataPtrCpu<T>(tensor));

  return tensor;
}

template Tensor Tensor::create(std::initializer_list<int> shape, lut::Span<const float> data);
template Tensor Tensor::create(std::initializer_list<int> shape, lut::Span<const IntType> data);
template Tensor Tensor::create(std::initializer_list<int> shape, lut::Span<const LongType> data);

Tensor Tensor::create(
    std::shared_ptr<TensorShape> shape,
    std::shared_ptr<TensorData> data,
    int64_t offset) {
  Tensor tensor;
  tensor._shape = shape;
  tensor._data = data;
  tensor._offset = offset;

  return tensor;
}

Tensor::Tensor()
    : _offset(0) {
}
Tensor::~Tensor() {
}

Tensor::Tensor(const Tensor &tensor) {
  _data = tensor._data;
  _shape = tensor._shape;
  _offset = tensor._offset;
}

Tensor &Tensor::operator=(const Tensor &tensor) {
  _data = tensor._data;
  _shape = tensor._shape;
  _offset = tensor._offset;

  return *this;
}

Tensor::Tensor(Tensor &&tensor) noexcept {
  _data = tensor._data;
  _shape = std::move(tensor._shape);
  _offset = tensor._offset;
}

Tensor &Tensor::operator=(Tensor &&tensor) {
  _data = tensor._data;
  _shape = std::move(tensor._shape);
  _offset = tensor._offset;

  return *this;
}

void Tensor::read(lut::Reader *fp) {
  std::string s = fp->readString(4);
  if (s != "tnsr") {
    throw lut::AbortedError("bad tensor format");
  }

  _shape = TensorShape::read(fp);
  _data = op::cpu::CpuTensorData::read(fp);
  _offset = 0;

  // check
  if (_shape->getNumEl() != _data->getNumEl())
    throw lut::AbortedError("tensor data and shape mismatch.");
}

namespace {

/// The tensor a view of `data` is, which shares the storage with whatever it was taken of.
Tensor fromView(std::shared_ptr<TensorData> data, const TensorView &view) {
  return Tensor::create(view.getInternalShape(), std::move(data), view.getInternalOffset());
}

}  // namespace

// Every view is worked out by TensorView, which is where the rules for shapes and strides live;
// a tensor only adds the reference that keeps the storage alive.

Tensor Tensor::view(lut::Span<const int> shape) const {
  return fromView(_data, TensorView(*this).view(shape));
}

Tensor Tensor::expand(lut::Span<const int> shape) const {
  return fromView(_data, TensorView(*this).expand(shape));
}

std::vector<int> Tensor::getShape() const {
  return TensorView(*this).getShape();
}

std::string Tensor::getShapeString() const {
  return _shape->toString();
}

bool Tensor::isContiguous() const {
  return TensorView(*this).isContiguous();
}

Tensor Tensor::slice(int dim, std::pair<int, int> range) const {
  return fromView(_data, TensorView(*this).slice(dim, range));
}

Tensor Tensor::slice(std::pair<int, int> range) const {
  return fromView(_data, TensorView(*this).slice(range));
}

Tensor Tensor::subtensor(int index) const {
  return fromView(_data, TensorView(*this).subtensor(index));
}

Tensor Tensor::transpose(int dim0, int dim1) const {
  return fromView(_data, TensorView(*this).transpose(dim0, dim1));
}

Tensor Tensor::unsqueeze(int dim) const {
  return fromView(_data, TensorView(*this).unsqueeze(dim));
}

Tensor Tensor::squeeze(int dim) const {
  return fromView(_data, TensorView(*this).squeeze(dim));
}

void Tensor::throwIfInvalidShape(lut::Span<const int> shape, const std::string &name) const {
  TensorView(*this).throwIfInvalidShape(shape, name);
}

int Tensor::getDim() const {
  return _shape->getDim();
}

int Tensor::getShape(int d) const {
  return _shape->getShape(d);
}

std::shared_ptr<TensorShape> Tensor::getInternalShape() const {
  return _shape;
}

bool Tensor::empty() const {
  return !_shape;
}

Device Tensor::getDevice() const {
  return _data->getDevice();
}

Tensor::ShapeType Tensor::getStride(int d) const {
  return _shape->getStride(d);
}

int64_t Tensor::getNumEl() const {
  return _shape->getNumEl();
}

int64_t Tensor::getInternalOffset() const {
  return _offset;
}

std::shared_ptr<TensorData> Tensor::getInternalData() const {
  return _data;
}

Operators *Tensor::getOperators() const {
  return fl::getOperators(getDevice().getType());
}

TensorShape::TensorShape(const TensorShape &size)
    : _data(size._data.copy()) {
}
TensorShape::TensorShape(TensorShape &&size) noexcept
    : _data(std::move(size._data)) {
}
TensorShape &TensorShape::operator=(const TensorShape &size) {
  _data = size._data.copy();
  return *this;
}
TensorShape &TensorShape::operator=(TensorShape &&size) noexcept {
  _data = std::move(size._data);
  return *this;
}

TensorShape::TensorShape(lut::Span<const ShapeType> shape) {
  _data = lut::FixedArray<Elem>(shape.size());
  lut::FixedArray<Elem>::iterator it = _data.begin();
  for (int n : shape) {
    it->shape = n;
    ++it;
  }

  int64_t stride = 1;
  for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
    CHECK(stride < std::numeric_limits<ShapeType>::max());
    _data[d].stride = static_cast<ShapeType>(stride);
    stride *= _data[d].shape;
  }
}

TensorShape::TensorShape(lut::Span<const Elem> shape) {
  _data = lut::FixedArray<Elem>(shape.size());
  std::copy(shape.begin(), shape.end(), _data.begin());
}

std::shared_ptr<TensorShape> TensorShape::subsize(int d) const {
  CHECK(d < getDim());

  std::shared_ptr<TensorShape> subsize{new TensorShape()};
  subsize->_data = lut::FixedArray<Elem>(getDim() - d);
  std::copy(_data.begin() + d, _data.end(), subsize->_data.begin());

  return subsize;
}

std::shared_ptr<TensorShape> TensorShape::read(lut::Reader *fp) {
  // rank
  int16_t rank = fp->readValue<int16_t>();
  if (rank > 16 || rank < 0) {
    throw lut::AbortedError("invalid rank.");
  }

  // shape
  std::vector<ShapeType> shape;
  for (int16_t d = 0; d < rank; ++d) {
    int32_t size = fp->readValue<int32_t>();
    if (size >= 1048576 || size <= 0) throw lut::AbortedError("invalid size in shape.");

    shape.push_back(size);
  }

  return std::make_shared<TensorShape>(lut::makeConstSpan(shape));
}

std::shared_ptr<TensorShape> TensorShape::transpose(int dim0, int dim1) const {
  dim0 = getRealDim(dim0);
  dim1 = getRealDim(dim1);

  std::shared_ptr<TensorShape> size = std::make_shared<TensorShape>(*this);
  Elem dim0_elem = size->_data[dim0];
  size->_data[dim0] = size->_data[dim1];
  size->_data[dim1] = dim0_elem;

  return size;
}

std::shared_ptr<TensorShape> TensorShape::squeeze(int dim) const {
  CHECK(getShape(dim) == 1);

  dim = getRealDim(dim);
  std::shared_ptr<TensorShape> size{new TensorShape()};
  size->_data = lut::FixedArray<Elem>(getDim() - 1);
  for (int d = 0; d < dim; ++d) {
    size->_data[d] = _data[d];
  }
  for (int d = dim + 1; d < getDim(); ++d) {
    size->_data[d - 1] = _data[d];
  }

  return size;
}

std::shared_ptr<TensorShape> TensorShape::unsqueeze(int dim) const {
  if (dim != getDim()) dim = getRealDim(dim);

  std::shared_ptr<TensorShape> size{new TensorShape()};
  size->_data = lut::FixedArray<Elem>(getDim() + 1);
  for (int d = 0; d < dim; ++d) {
    size->_data[d] = _data[d];
  }
  size->_data[dim].shape = 1;
  size->_data[dim].stride = dim == 0 ? getStride(0) * getShape(0) : getStride(dim - 1);
  for (int d = dim; d < getDim(); ++d) {
    size->_data[d + 1] = _data[d];
  }

  return size;
}

int TensorShape::getRealDim(int d) const {
  CHECK(!empty());
  int rank = getDim();
  if (d < 0) {
    d = rank + d;
  }

  if (d < 0 || d >= rank) {
    THROW(InvalidArg, lut::sprintf("no dimension %d in a %d-D tensor", d, rank));
  }
  return d;
}

int TensorShape::getRealIndex(int dim, int index) const {
  CHECK(!empty());
  dim = getRealDim(dim);

  int shape = _data[dim].shape;
  index = index >= 0 ? index : shape + index;

  if (index < 0 || index > shape) {
    THROW(InvalidArg, lut::sprintf("index %d is outside a dimension of %d", index, shape));
  }
  return index;
}

int TensorShape::getDim() const {
  return static_cast<int>(_data.size());
}

bool TensorShape::empty() const {
  return _data.empty();
}

int TensorShape::getShape(int d) const {
  return _data[getRealDim(d)].shape;
}

int TensorShape::getStride(int d) const {
  return _data[getRealDim(d)].stride;
}

int64_t TensorShape::getNumEl() const {
  if (empty()) {
    return 0;
  }

  int64_t n = 1;
  for (const Elem &elem : _data) {
    n *= elem.shape;
  }
  return n;
}

void TensorShape::setShape(int dim, ShapeType shape) {
  dim = getRealDim(dim);
  CHECK(dim >= 0 && dim <= this->getDim());
  CHECK(shape <= _data[dim].shape);

  _data[dim].shape = shape;
}

std::shared_ptr<TensorShape> TensorShape::expand(lut::Span<const int> shape) const {
  CHECK(getDim() == shape.size());
  std::shared_ptr<TensorShape> view = std::make_shared<TensorShape>(lut::makeConstSpan(_data));
  int dim = getDim();
  for (int d = 0; d < dim; ++d) {
    if (shape[d] != getShape(d)) {
      if (getShape(d) != 1) {
        THROW(
            InvalidArg,
            lut::sprintf(
                "expand: dimension %d holds %d elements, and only a single one can grow",
                d,
                getShape(d)));
      }
      view->_data[d].shape = shape[d];
      view->_data[d].stride = 0;
    }
  }

  return view;
}

std::string TensorShape::toString() const {
  std::ostringstream os;
  bool first = true;

  os << "(";
  for (Elem elem : _data) {
    if (first) {
      first = false;
    } else {
      os << ", ";
    }
    os << elem.shape;
  }
  os << ")";
  return os.str();
}

}  // namespace fl
