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

#pragma once

#include <stdint.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lutil/span.h"
#include "flint/device.h"
#include "flint/dtype.h"
#include "flint/tensor.h"

namespace fl {

class Operators;

/// @brief A tensor's elements as an operator sees them: which storage, where in it, and how its
/// dimensions are laid out -- and nothing that keeps any of it alive.
///
/// What every operator takes, for its inputs and for the output it writes. The storage is a plain
/// pointer: whoever hands a view to an operator keeps the tensor it came from alive for the call,
/// which is all an operator needs, since none of them holds on to a tensor after it returns. That
/// is what lets a caller hand an operator memory it does not own -- a weight still in its file's
/// mapping -- without lending it ownership as well.
///
/// The shape is shared rather than copied, as it is between a tensor and its views: it is never
/// changed after it is made, and a view of a view only makes a new one.
///
/// Made from a Tensor implicitly, so that anything that takes a view takes a tensor as well.
class TensorView {
 public:
  typedef TensorShape::ShapeType ShapeType;

  /// @brief A view of nothing, which every query but empty() refuses.
  TensorView();

  /// @brief The view of all of `tensor`. Borrows its storage: `tensor` outlives the view.
  TensorView(const Tensor &tensor);

  TensorView(TensorData *data, std::shared_ptr<TensorShape> shape, int64_t offset);

  bool empty() const {
    return !_shape;
  }

  int getDim() const;
  ShapeType getShape(int d) const;
  std::vector<int> getShape() const;
  std::string getShapeString() const;
  ShapeType getStride(int d) const;
  int64_t getNumEl() const;
  bool isContiguous() const;

  DType getDType() const {
    return _dtype;
  }

  Device getDevice() const {
    return _device;
  }

  /// @brief The operators of the device the elements are on.
  Operators *getOperators() const;

  /// @brief Throw lut::AbortedError, naming `name`, unless the shape is exactly `shape`.
  void throwIfInvalidShape(lut::Span<const int> shape, const std::string &name) const;

  // The same views a Tensor has, of the same storage. Each is a new view; none changes this one.
  TensorView view(lut::Span<const int> shape) const;
  TensorView expand(lut::Span<const int> shape) const;
  TensorView slice(int dim, std::pair<int, int> range) const;
  TensorView slice(std::pair<int, int> range) const;
  TensorView subtensor(int index) const;
  TensorView unsqueeze(int dim) const;
  TensorView squeeze(int dim) const;
  TensorView transpose(int dim0, int dim1) const;

  /// @brief The storage, which this view does not own.
  TensorData *getInternalData() const {
    return _data;
  }

  /// @brief Where the first element is in the storage, in elements.
  int64_t getInternalOffset() const {
    return _offset;
  }

  std::shared_ptr<TensorShape> getInternalShape() const {
    return _shape;
  }

 private:
  TensorData *_data;
  std::shared_ptr<TensorShape> _shape;
  int64_t _offset;
  DType _dtype;
  Device _device;
};

}  // namespace fl
