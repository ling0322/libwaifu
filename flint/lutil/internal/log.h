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

#include "lutil/log.h"

namespace lut {
namespace internal {

extern LogSeverity gLogLevel;

/// What a fatal log calls just before it takes the process down, or nullptr for none.
///
/// A program that has taken the screen over -- a full screen terminal application -- has nowhere
/// for a message to land until it gives the screen back, and by the time the process is dying it
/// has no way to be told. This is where it is told. It runs before anything is printed, since the
/// first line printed is already too late for one.
typedef void (*FatalHandler)();

/// Sets it, and returns the one that was there.
FatalHandler setFatalHandler(FatalHandler handler);

class LogWrapper {
 public:
  LogWrapper(LogSeverity severity, const char *source_file, int source_line);
  ~LogWrapper();

  LogWrapper(LogWrapper &) = delete;
  LogWrapper &operator=(LogWrapper &) = delete;

  template<typename T>
  LogWrapper &operator<<(const T &value) {
    os_ << value;
    return *this;
  }

 private:
  std::ostringstream os_;

  LogSeverity severity_;
  const char *source_file_;
  int source_line_;
  char time_[200];

  const char *Time();
  const char *Severity() const;
};

// log wrappers for each severity
class LogWrapperkDEBUG : public LogWrapper {
 public:
  LogWrapperkDEBUG(const char *source_file, int source_line)
      : LogWrapper(LogSeverity::kDEBUG, source_file, source_line) {
  }
};
class LogWrapperkINFO : public LogWrapper {
 public:
  LogWrapperkINFO(const char *source_file, int source_line)
      : LogWrapper(LogSeverity::kINFO, source_file, source_line) {
  }
};
class LogWrapperkWARN : public LogWrapper {
 public:
  LogWrapperkWARN(const char *source_file, int source_line)
      : LogWrapper(LogSeverity::kWARN, source_file, source_line) {
  }
};
class LogWrapperkERROR : public LogWrapper {
 public:
  LogWrapperkERROR(const char *source_file, int source_line)
      : LogWrapper(LogSeverity::kERROR, source_file, source_line) {
  }
};
class LogWrapperkFATAL : public LogWrapper {
 public:
  LogWrapperkFATAL(const char *source_file, int source_line)
      : LogWrapper(LogSeverity::kFATAL, source_file, source_line) {
  }
};

/// The failure path of CHECK(), built only when the condition did not hold. It collects whatever
/// is streamed into it, and raise() then reports it and throws it.
///
/// The throw is in raise() rather than in the destructor on purpose. A destructor that throws
/// while another exception is already unwinding ends the process, and a CHECK that fails on the
/// way out of a frame -- during cleanup after an unrelated error -- is exactly when that would
/// happen. Not ending the process is the whole point of this class.
class CheckFailure {
 public:
  CheckFailure(const char *source_file, int source_line, const char *condition);

  CheckFailure(CheckFailure &) = delete;
  CheckFailure &operator=(CheckFailure &) = delete;

  template<typename T>
  CheckFailure &operator<<(const T &value) {
    os_ << value;
    return *this;
  }

  /// Logs the message and a stack trace at ERROR, then throws lut::AbortedError carrying that
  /// same message. A CHECK() with nothing streamed into it reports its own condition instead.
  ///
  /// Const so that it can be reached through CheckRaiser, which has to bind to the temporary a
  /// CHECK() with no message appended leaves behind.
  [[noreturn]] void raise() const;

 private:
  std::ostringstream os_;
  const char *source_file_;
  int source_line_;
  const char *condition_;
};

/// Calls CheckFailure::raise() from an operator that binds looser than `<<`, so that it runs once
/// the streamed message is complete rather than before it starts. It is nothing but somewhere for
/// that precedence to hang.
struct CheckRaiser {
  [[noreturn]] void operator&(const CheckFailure &failure) const {
    failure.raise();
  }
};

}  // namespace internal
}  // namespace lut
