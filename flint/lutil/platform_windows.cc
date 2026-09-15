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

#include <windows.h>

// dbghelp.h reaches for types that windows.h defines, so it cannot come first.
#include <dbghelp.h>
#include <stdio.h>

#include <mutex>

#include "lutil/platform.h"

namespace lut {

void *alloc32ByteAlignedMem(int64_t size) {
  return _aligned_malloc(size, 32);
}

void free32ByteAlignedMem(void *ptr) {
  _aligned_free(ptr);
}

const char *getPathDelim() {
  return "\\";
}

// What libunwind does everywhere else, DbgHelp does here: walk the stack, then put a name to
// each return address. The format is the one platform_linux.cc prints, so a trace read off a
// Windows run lines up with a trace read off a Linux one.
//
// The symbol handler belongs to the process rather than the call, and initializing it twice is
// an error, so the first caller sets it up and the rest inherit it. SYMOPT_UNDNAME asks DbgHelp
// to undecorate C++ names on the way out, which is why nothing here answers to the
// __cxa_demangle call its Linux counterpart makes.
void printStackTrace() {
  puts("Stack trace:");

  HANDLE process = GetCurrentProcess();
  static std::once_flag symbolsInitialized;
  std::call_once(symbolsInitialized, [process] {
    SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
    SymInitialize(process, nullptr, TRUE);
  });

  // Skipping one frame drops printStackTrace itself, the way the unw_step that runs before the
  // Linux loop reads anything drops it there.
  constexpr int kMaxFrames = 62;
  void *frames[kMaxFrames];
  USHORT captured = CaptureStackBackTrace(1, kMaxFrames, frames, nullptr);

  // SYMBOL_INFO keeps the name past the end of the struct, so it is handed room to do that.
  constexpr int kMaxNameLen = 512;
  char symbolBuffer[sizeof(SYMBOL_INFO) + kMaxNameLen] = {0};
  SYMBOL_INFO *symbol = reinterpret_cast<SYMBOL_INFO *>(symbolBuffer);
  symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
  symbol->MaxNameLen = kMaxNameLen;

  for (USHORT frame = 0; frame < captured; ++frame) {
    DWORD64 ip = reinterpret_cast<DWORD64>(frames[frame]);

    DWORD64 off = 0;
    const char *name = SymFromAddr(process, ip, &off, symbol) ? symbol->Name : "???";

    IMAGEHLP_MODULE64 module = {0};
    module.SizeOfStruct = sizeof(module);
    const char *obj = SymGetModuleInfo64(process, ip, &module) ? module.LoadedImageName : "";

    fprintf(
        stderr,
        "#%-2d  %p  %s + 0x%llx  (%s)\n",
        static_cast<int>(frame),
        frames[frame],
        name,
        static_cast<unsigned long long>(off),
        obj);
  }
}

}  // namespace lut
