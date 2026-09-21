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

#include "lutil/env.h"

#include <stdlib.h>

#include "../../third_party/catch2/catch_amalgamated.hpp"

namespace lut {

namespace {

constexpr char kName[] = "LIBWAIFU_ENV_FLAG_TEST";

/// Set the variable, or remove it where `value` is null. Windows has no setenv, and its _putenv_s
/// removes a name by assigning the empty string to it, so there the empty value below is the same
/// thing as being unset -- which both sides of this test expect to read as off regardless.
void setFlag(const char *value) {
#if defined(_WIN32)
  _putenv_s(kName, value ? value : "");
#else
  if (value) {
    setenv(kName, value, 1);
  } else {
    unsetenv(kName);
  }
#endif
}

}  // namespace

CATCH_TEST_CASE("isEnvFlagSet reads a flag", "[core][util]") {
  setFlag(nullptr);
  CATCH_REQUIRE(isEnvFlagSet(kName) == false);

  // An empty export is a shell with nothing to say, and the words for off are off whatever their
  // case. Everything else is the flag being asked for.
  for (const char *off : {"", " ", "0", "false", "FALSE", "no", "Off", " off "}) {
    setFlag(off);
    CATCH_REQUIRE(isEnvFlagSet(kName) == false);
  }

  for (const char *on : {"1", "true", "TRUE", "yes", "on", " 1 ", "cublas"}) {
    setFlag(on);
    CATCH_REQUIRE(isEnvFlagSet(kName) == true);
  }

  setFlag(nullptr);
}

}  // namespace lut
