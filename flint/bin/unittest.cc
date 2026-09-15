// The MIT License (MIT)
//
// Copyright (c) 2023-2025 Xiaoyang Chen
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

#include <stdio.h>
#include <stdlib.h>

#include <string>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/platform.h"
#include "flint/cpu/kernel/interface.h"
#include "flint/operators.h"

/// The level the tests run at, from LIBWAIFU_LOG, or quiet.
///
/// Quiet by default because most of what the library logs during a test run is the test working:
/// every case that checks a bad argument is refused leaves an ERROR behind it, since THROW logs
/// before it throws. A green run that prints a page of errors teaches people to skim past errors.
/// What a failing test says, it says through Catch2.
lut::LogSeverity levelFromEnvironment() {
  const char *asked = std::getenv("LIBWAIFU_LOG");
  if (!asked) return lut::LogSeverity::kFATAL;

  std::string level = asked;
  if (level == "debug") return lut::LogSeverity::kDEBUG;
  if (level == "info") return lut::LogSeverity::kINFO;
  if (level == "warn") return lut::LogSeverity::kWARN;
  if (level == "error") return lut::LogSeverity::kERROR;
  if (level == "fatal") return lut::LogSeverity::kFATAL;

  // Said through the level it is about to be denied by, so it cannot itself be swallowed.
  fprintf(stderr, "LIBWAIFU_LOG is \"%s\", which is none of debug/info/warn/error/fatal\n", asked);
  return lut::LogSeverity::kFATAL;
}

int main(int argc, char **argv) {
  // lut::enablePrintStackOnError();

  // Before anything that logs, which the operators do as they start.
  lut::setLogLevel(levelFromEnvironment());

  fl::initOperators();

  // enable some slow kernels for reference.
  fl::op::cpu::kernel::setAllowSlowKernel(true);

  int result = Catch::Session().run(argc, argv);

  fl::destroyOperators();

  return result;
}
