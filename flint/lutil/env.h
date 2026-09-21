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

namespace lut {

/// @brief Whether an environment variable asks for something to be switched on.
///
/// True when `name` is set to anything but a word for off -- an empty value, "0", "false", "no"
/// or "off", in any case and around any whitespace. Exporting a variable as empty is what a
/// shell does when it has nothing to say, and `FOO=0` reading as on is a surprise no flag should
/// hold. Unlike parseBool this takes whatever it is given rather than throwing on it: a flag that
/// aborts the process over the word in it is worse than one that reads "yes" as yes.
/// @param name the variable to read.
bool isEnvFlagSet(const char *name);

}  // namespace lut
