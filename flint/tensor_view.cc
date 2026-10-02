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

#include "flint/tensor_view.h"

#include <algorithm>
#include <sstream>

#include "lutil/error.h"
#include "lutil/strings.h"
#include "flint/operators.h"

namespace fl {

namespace {

/// The shape a view asks for with its one inferred dimension (-1) resolved.
std::vector<TensorShape::ShapeType> getRealShape(int64_t numEl, lut::Span<const int> viewShape) {
  std::vector<TensorShape::ShapeType> shape{viewShape.begin(), viewShape.end()};
  auto inferDim = shape.end();
  int64_t viewNumEl = 1;
  for (auto it = shape.begin(); it != shape.end(); ++it) {
    if (*it < 0) {
      if (inferDim != shape.end()) THROW(InvalidArg, "a view infers more than one dimension");
      inferDim = it;
    } else {
      viewNumEl *= *it;
    }
  }

  if (inferDim != shape.end()) {
    if (viewNumEl == 0 || numEl % viewNumEl != 0) {
      THROW(
          InvalidArg,
          lut::sprintf("a view of %d elements cannot infer a dimension of %d", numEl, viewNumEl));
    }
    *inferDim = static_cast<TensorShape::ShapeType>(numEl / viewNumEl);
  } else if (numEl != viewNumEl) {
    THROW(
        InvalidArg,
        lut::sprintf("invalid view: %d elements cannot be seen as %d", numEl, viewNumEl));
  }

  return shape;
}

/// The dimensions of `src` with every run that is contiguous in memory merged into one.
std::vector<TensorShape::Elem> mergeContigShape(const TensorView &src) {
  std::vector<TensorShape::Elem> mergedShape;
  for (int d = src.getDim() - 1; d >= 0; --d) {
    if (src.getStride(d) == 0) {
      THROW(InvalidArg, "unable to change the view of an expanded tensor");
    }

    if (d == src.getDim() - 1 ||
        src.getStride(d + 1) * src.getShape(d + 1) != src.getStride(d)) {
      TensorShape::Elem s;
      s.shape = src.getShape(d);
      s.stride = src.getStride(d);
      mergedShape.push_back(s);
    } else {
      mergedShape.back().shape *= src.getShape(d);
    }
  }

  std::reverse(mergedShape.begin(), mergedShape.end());
  return mergedShape;
}

/// The shape and strides of `view` over a non-contiguous `src`, splitting only within its
/// contiguous runs.
std::vector<TensorShape::Elem> getViewShapeStride(
    const TensorView &src,
    lut::Span<const int> view) {
  std::vector<TensorShape::Elem> mergedShape = mergeContigShape(src);
  std::vector<TensorShape::Elem> viewShape;
  auto vi = view.rbegin();
  for (int64_t d = mergedShape.size() - 1; d >= 0; --d) {
    TensorShape::Elem ms = mergedShape[d];
    int numel = 1;
    while (vi != view.rend() && *vi * numel <= ms.shape) {
      TensorShape::Elem s;
      s.shape = *vi;
      s.stride = numel * ms.stride;
      viewShape.push_back(s);

      numel *= *vi;
      ++vi;
    }

    if (numel != ms.shape) THROW(InvalidArg, "unable to get this view of the tensor");
  }

  std::reverse(viewShape.begin(), viewShape.end());
  return viewShape;
}

}  // namespace

TensorView::TensorView()
    : _data(nullptr),
      _offset(0),
      _dtype(DType::kUnknown) {
}

TensorView::TensorView(const Tensor &tensor)
    : _data(tensor.getInternalData().get()),
      _shape(tensor.getInternalShape()),
      _offset(tensor.getInternalOffset()),
      _dtype(tensor.getDType()),
      _device(_data ? _data->getDevice() : Device()) {
}

TensorView::TensorView(TensorData *data, std::shared_ptr<TensorShape> shape, int64_t offset)
    : _data(data),
      _shape(std::move(shape)),
      _offset(offset),
      _dtype(data ? data->getDType() : DType(DType::kUnknown)),
      _device(data ? data->getDevice() : Device()) {
}

int TensorView::getDim() const {
  return _shape->getDim();
}

TensorView::ShapeType TensorView::getShape(int d) const {
  return _shape->getShape(d);
}

std::vector<int> TensorView::getShape() const {
  std::vector<int> shape;
  for (int d = 0; d < getDim(); ++d) shape.push_back(getShape(d));
  return shape;
}

std::string TensorView::getShapeString() const {
  return _shape->toString();
}

TensorView::ShapeType TensorView::getStride(int d) const {
  return _shape->getStride(d);
}

int64_t TensorView::getNumEl() const {
  return _shape->getNumEl();
}

bool TensorView::isContiguous() const {
  int64_t numel = 1;
  for (int i = getDim() - 1; i >= 0; --i) {
    if (numel != getStride(i) && getShape(i) != 1) return false;
    numel *= getShape(i);
  }
  return true;
}

Operators *TensorView::getOperators() const {
  return fl::getOperators(getDevice().getType());
}

void TensorView::throwIfInvalidShape(lut::Span<const int> shape, const std::string &name) const {
  if (shape.size() != getDim()) {
    throw lut::AbortedError(
        lut::sprintf(
            "%s: invalid shape. dim=%d expected, but %d got.",
            name,
            shape.size(),
            getDim()));
  }

  bool correct = true;
  int i = 0;
  for (int s : shape) {
    if (getShape(i++) != s) correct = false;
  }
  if (correct) return;

  std::ostringstream actual;
  actual << "(";
  for (int d = 0; d < getDim(); ++d) {
    if (d) actual << ", ";
    actual << getShape(d);
  }
  actual << ")";

  std::ostringstream expected;
  expected << "(";
  bool first = true;
  for (int s : shape) {
    if (!first) expected << ", ";
    expected << s;
    first = false;
  }
  expected << ")";

  throw lut::AbortedError(
      lut::sprintf(
          "%s: invalid shape: %s expected, but %s found.",
          name,
          expected.str(),
          actual.str()));
}

TensorView TensorView::view(lut::Span<const int> view) const {
  std::vector<ShapeType> shape = getRealShape(getNumEl(), view);
  if (isContiguous()) {
    return TensorView(_data, std::make_shared<TensorShape>(lut::makeConstSpan(shape)), _offset);
  }

  // `shape`, not `view`: an inferred -1 has already been resolved into it. Passing the raw request
  // through would have the stride walk try to match a dimension of -1.
  std::vector<TensorShape::Elem> viewShape = getViewShapeStride(*this, shape);
  return TensorView(_data, std::make_shared<TensorShape>(lut::makeConstSpan(viewShape)), _offset);
}

TensorView TensorView::expand(lut::Span<const int> shape) const {
  CHECK(!getDType().isQuantized());
  return TensorView(_data, _shape->expand(shape), _offset);
}

TensorView TensorView::slice(int dim, std::pair<int, int> range) const {
  CHECK(!getDType().isQuantized());

  dim = _shape->getRealDim(dim);
  if (dim < 0 || dim >= getDim()) {
    THROW(InvalidArg, lut::sprintf("slice: no dimension %d in a %d-D tensor", dim, getDim()));
  }

  int begin = range.first;
  int end = range.second;
  if (begin == None) begin = 0;
  if (end == None) end = getShape(dim);

  begin = _shape->getRealIndex(dim, begin);
  end = _shape->getRealIndex(dim, end);
  if (begin < 0 || begin >= end || end > getShape(dim)) {
    THROW(
        InvalidArg,
        lut::sprintf(
            "slice: [%d, %d) is not within a dimension of %d",
            begin,
            end,
            getShape(dim)));
  }

  auto shape = std::make_shared<TensorShape>(*_shape);
  shape->setShape(dim, end - begin);
  return TensorView(_data, shape, _offset + _shape->getStride(dim) * begin);
}

TensorView TensorView::slice(std::pair<int, int> range) const {
  return slice(0, range);
}

TensorView TensorView::subtensor(int index) const {
  CHECK(!getDType().isQuantized());

  index = _shape->getRealIndex(0, index);
  if (index < 0 || index >= getShape(0)) {
    THROW(
        InvalidArg,
        lut::sprintf("subtensor: %d is not within a dimension of %d", index, getShape(0)));
  }

  return TensorView(_data, _shape->subsize(1), _offset + _shape->getStride(0) * index);
}

TensorView TensorView::unsqueeze(int dim) const {
  return TensorView(_data, _shape->unsqueeze(dim), _offset);
}

TensorView TensorView::squeeze(int dim) const {
  return TensorView(_data, _shape->squeeze(dim), _offset);
}

TensorView TensorView::transpose(int dim0, int dim1) const {
  return TensorView(_data, _shape->transpose(dim0, dim1), _offset);
}

}  // namespace fl
