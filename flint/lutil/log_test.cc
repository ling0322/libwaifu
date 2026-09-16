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

#include "lutil/log.h"

#include <string>

#include "../../third_party/catch2/catch_amalgamated.hpp"
#include "lutil/error.h"
#include "lutil/internal/log.h"

namespace lut {
namespace {

/// Puts the log level back however the test leaves, since a failed assertion leaves early.
class ScopedLogLevel {
 public:
  explicit ScopedLogLevel(LogSeverity level)
      : previous_(internal::gLogLevel) {
    setLogLevel(level);
  }

  ~ScopedLogLevel() {
    setLogLevel(previous_);
  }

 private:
  LogSeverity previous_;
};

/// A CHECK has to be reached through a call: the macro is a statement, and catch2 wants an
/// expression to wrap.
void failingCheck() {
  CHECK(1 + 1 == 3) << "arithmetic is broken";
}

void failingCheckWithoutMessage() {
  CHECK(false);
}

int gConditionEvaluations = 0;

bool countAndReturn(bool value) {
  ++gConditionEvaluations;
  return value;
}

}  // namespace

// These run at whatever level the binary was given -- kFATAL unless LIBWAIFU_LOG says otherwise,
// which is what keeps a green run quiet. `LIBWAIFU_LOG=error` is then what covers the reporting
// path here, printing the ERROR line and the stack trace that go with a failure. Only the case
// that is about the gate pins the level itself.

CATCH_TEST_CASE("a failing CHECK throws instead of ending the process", "[core][util][log]") {
  CATCH_REQUIRE_THROWS_AS(failingCheck(), AbortedError);
}

CATCH_TEST_CASE("a failing CHECK hands the caller its message", "[core][util][log]") {
  try {
    failingCheck();
    CATCH_FAIL("CHECK(1 + 1 == 3) returned");
  } catch (const AbortedError &e) {
    // lut::Error names the code in what(), so a caller reading only the message still sees
    // which kind of failure it was.
    CATCH_REQUIRE(std::string(e.what()) == "Aborted: arithmetic is broken");
    CATCH_REQUIRE(e.getCode() == ErrorCode::Aborted);
  }
}

CATCH_TEST_CASE("a failing CHECK with nothing streamed reports its condition", "[core][util][log]") {
  try {
    failingCheckWithoutMessage();
    CATCH_FAIL("CHECK(false) returned");
  } catch (const AbortedError &e) {
    CATCH_REQUIRE(std::string(e.what()) == "Aborted: Check false failed.");
  }
}

CATCH_TEST_CASE("silencing the log does not silence the failure", "[core][util][log]") {
  // The level decides what is printed. What the caller is told is the exception, and no level
  // turns that off -- a check that failed unheard would be the worst of both.
  ScopedLogLevel quiet(LogSeverity::kFATAL);

  CATCH_REQUIRE_THROWS_AS(failingCheck(), AbortedError);
}

CATCH_TEST_CASE("a CHECK that holds evaluates its condition once", "[core][util][log]") {
  gConditionEvaluations = 0;

  CHECK(countAndReturn(true)) << "never built";

  CATCH_REQUIRE(gConditionEvaluations == 1);
}

CATCH_TEST_CASE("a failing CHECK unwinds what it passes", "[core][util][log]") {
  // The point of throwing rather than aborting: the destructors between the check and whoever
  // handles it get to run.
  bool released = false;
  struct Guard {
    bool *flag;
    ~Guard() {
      *flag = true;
    }
  };

  try {
    Guard guard{&released};
    failingCheck();
  } catch (const AbortedError &) {
  }

  CATCH_REQUIRE(released);
}

}  // namespace lut
