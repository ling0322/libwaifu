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

#include <string>

#include "../../third_party/catch2/catch_amalgamated.hpp"
#include "lutil/error.h"
#include "lutil/log.h"

namespace lut {
namespace {

/// A CHECK in a destructor, which is where the throw has to be held back: the destructor runs
/// while another exception is on its way out, and throwing a second one ends the process.
struct CheckingDestructor {
  ~CheckingDestructor() {
    CHECK(false) << "checked on the way out";
  }
};

}  // namespace

CATCH_TEST_CASE("a failed check throws rather than ending the process", "[lut][log]") {
  CATCH_REQUIRE_THROWS_AS([] { CHECK(1 == 2); }(), lut::AbortedError);
  CATCH_REQUIRE_NOTHROW([] { CHECK(1 == 1); }());
}

CATCH_TEST_CASE("a failed check says what failed and where", "[lut][log]") {
  try {
    [] { CHECK(1 == 2) << "one is not two"; }();
    CATCH_FAIL("the check did not throw");
  } catch (const lut::Error &error) {
    std::string what = error.what();

    // The condition, whatever was streamed after it, and the line it was on: the stack trace that
    // the aborting version printed is gone, so the message has to carry the location itself.
    CATCH_REQUIRE(what.find("1 == 2") != std::string::npos);
    CATCH_REQUIRE(what.find("one is not two") != std::string::npos);
    CATCH_REQUIRE(what.find("log_test.cc") != std::string::npos);
  }
}

CATCH_TEST_CASE("a check that fails while unwinding does not replace the exception", "[lut][log]") {
  // If the second throw were let out, this would end the process rather than fail.
  CATCH_REQUIRE_THROWS_AS(
      [] {
        CheckingDestructor onTheWayOut;
        throw lut::InvalidArgError("the failure that started it");
      }(),
      lut::InvalidArgError);
}

}  // namespace lut
