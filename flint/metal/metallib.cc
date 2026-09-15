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

#include "flint/metal/metallib.h"

#include <mach-o/getsect.h>

#include "lutil/error.h"
#include "mlx/backend/metal/metal.h"

// The Mach-O header of whichever image this object file ended up in. Going through
// __dso_handle rather than _mh_execute_header keeps the lookup correct if libflint.a is
// ever linked into a dylib instead of straight into the executable.
extern "C" const char __dso_handle[];

namespace fl {
namespace op {
namespace metal {

namespace {

const uint8_t *findEmbeddedMetallib(unsigned long *size) {
  *size = 0;
  return getsectiondata(reinterpret_cast<const struct mach_header_64 *>(&__dso_handle),
                        "__TEXT",
                        "__metallib",
                        size);
}

}  // namespace

void useEmbeddedMetallib() {
  unsigned long size = 0;
  const uint8_t *data = findEmbeddedMetallib(&size);
  if (data == nullptr) {
    throw lut::AbortedError(
        "no __TEXT,__metallib section in this binary: it was linked without "
        "-sectcreate __TEXT __metallib <path to mlx.metallib>");
  }

  // Borrowed, not copied. The section is mapped from the binary and lives as long as the
  // process, which is exactly the lifetime MLX requires of this pointer.
  mlx::core::metal::set_metallib_data(data, size);
}

size_t getEmbeddedMetallibSize() {
  unsigned long size = 0;
  findEmbeddedMetallib(&size);
  return size;
}

}  // namespace metal
}  // namespace op
}  // namespace fl
