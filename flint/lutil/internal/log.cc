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

#include "lutil/internal/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ctime>
#include <exception>
#include <string>

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/platform.h"

namespace lut {
namespace internal {

LogSeverity gLogLevel = LogSeverity::kINFO;

LogWrapper::LogWrapper(LogSeverity severity, const char *source_file, int source_line)
    : severity_(severity),
      source_line_(source_line) {
  const char *s = strrchr(source_file, '/');
  if (!s) {
    s = strrchr(source_file, '\\');
  }

  if (s) {
    source_file_ = s + 1;
  } else {
    source_file_ = s;
  }
}

LogWrapper::~LogWrapper() {
  std::string message = os_.str();
  if (message.empty()) message = default_message_;

  printf("%s %s %s:%d] %s\n", Severity(), Time(), source_file_, source_line_, message.c_str());

  if (severity_ == LogSeverity::kFATAL) {
    printStackTrace();
    abort();
  }
}

const char *LogWrapper::Time() {
  time_t now = time(nullptr);

  std::strftime(time_, sizeof(time_), "%FT%TZ", std::gmtime(&now));
  return time_;
}

const char *LogWrapper::Severity() const {
  switch (severity_) {
    case LogSeverity::kDEBUG:
      return "DEBUG";
    case LogSeverity::kINFO:
      return "INFO";
    case LogSeverity::kWARN:
      return "WARNING";
    case LogSeverity::kERROR:
      return "ERROR";
    case LogSeverity::kFATAL:
      return "FATAL";
    default:
      fputs("invalid log severity.", stderr);
      abort();
  }
}

LogWrapper &LogWrapper::DefaultMessage(const char *message) {
  default_message_ = message;
  return *this;
}

CheckFailure::CheckFailure(const char *source_file, int source_line, const char *condition)
    : source_file_(source_file),
      source_line_(source_line),
      condition_(condition) {
}

CheckFailure::~CheckFailure() noexcept(false) {
  std::string message = std::string(source_file_) + ":" + std::to_string(source_line_) +
                        "] Check " + condition_ + " failed";

  std::string detail = os_.str();
  if (!detail.empty()) message += ": " + detail;

  // Throwing while another exception is on its way out ends the process, which is the outcome
  // this whole class exists to avoid. That happens when a destructor runs a CHECK during
  // unwinding, and there the message is all that can be salvaged.
  if (std::uncaught_exceptions() > 0) {
    LOG(ERROR) << message << " (during unwinding, so it is reported rather than thrown)";
    return;
  }

  throw lut::AbortedError(message);
}

}  // namespace internal
}  // namespace lut

namespace lut {

void setLogLevel(LogSeverity level) {
  internal::gLogLevel = level;
}

}  // namespace lut
