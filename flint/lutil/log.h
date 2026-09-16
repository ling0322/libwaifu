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

#pragma once

#include <sstream>

#include "lutil/time.h"

#define LOG(severity)                                             \
  if (lut::internal::gLogLevel > lut::LogSeverity::k##severity) { \
  } else                                                          \
    lut::internal::LogWrapperk##severity(__FILE__, __LINE__)
#define NOT_IMPL()                   \
  {                                  \
    LOG(FATAL) << "not implemented"; \
    abort();                         \
  }

#define LUT_CONCAT2(l, r) l##r
#define LUT_CONCAT(l, r) LUT_CONCAT2(l, r)

#define LOG_TIME(stmt, message)                 \
  double LUT_CONCAT(t0, __LINE__) = lut::now(); \
  stmt;                                         \
  LOG(INFO) << message << ": " << (lut::now() - LUT_CONCAT(t0, __LINE__)) * 1000 << "ms";

// CHECK macro conflicts with catch2, which is what CATCH_CONFIG_PREFIX_ALL in the build is for:
// catch2's own assertions are all CATCH_-prefixed, so the bare name is this one everywhere,
// inside test code included.
//
// CHECK is for what must never happen: a broken invariant, where the code that hit it has nothing
// sensible left to do and a stack trace at the point of failure is the only thing worth having.
// It logs the message and that trace at ERROR, and then throws lut::AbortedError.
//
// It does not end the process. flint runs inside someone else's -- a CLI, a test harness, an
// application that loaded the shared library -- and aborting takes down the one party who could
// have done something about it, before it has been told anything. So it is told instead: through
// the C interface the throw arrives as FL_ERROR_ABORTED carrying the message, and what to do
// about a library with a broken invariant is the host's call, not ours.
//
// A caller getting an argument wrong is not a broken invariant. Those are ordinary and
// recoverable, and are thrown with THROW(InvalidArg, ...) instead: a code the caller can tell
// apart, and no stack trace, since there is no bug of ours in one to go looking for.
//
// The shape is glog's. `&` binds looser than `<<`, so the streamed message is complete before
// operator& is reached, and the throw happens there -- inside an ordinary call, which is a place
// a throw is allowed. A destructor is not: one that throws while another exception is already
// unwinding ends the process, which is the single thing this must never do.
//
// Which is also the one rule for using it: do not CHECK anywhere a destructor can reach. A
// destructor is implicitly noexcept, so the throw does not unwind out of it -- it goes straight
// to std::terminate, and takes the message with it, leaving a process that exits with nothing on
// either stream. That is strictly worse than what a destructor should do with a failure it cannot
// return, which is to LOG(ERROR) it and carry on; see llynCudaFree and the three tensor
// destructors in flint/cuda for the shape of it.
#define CHECK(cond)                \
  if (cond) {                      \
  } else                           \
    lut::internal::CheckRaiser() & \
        lut::internal::CheckFailure(__FILE__, __LINE__, #cond)

namespace lut {

/// @brief How bad a message is, and how much of it a level lets through: LOG(x) prints when the
///        level set is no higher than x.
///
/// In order, which they were not: kERROR used to be 4 and kFATAL 3, so a level of kERROR -- asked
/// for by someone who wanted errors and nothing else -- was above kFATAL and swallowed the one
/// message that cannot be missed.
enum class LogSeverity { kDEBUG = 0, kINFO = 1, kWARN = 2, kERROR = 3, kFATAL = 4 };

void setLogLevel(LogSeverity level);

}  // namespace lut

#include "lutil/internal/log.h"
