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

#include "flint/bench.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#ifdef LIBWAIFU_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace fl {
namespace bench {
namespace {

/// Where a kept copy goes, or null when none was asked for. Never closed on purpose: it is open
/// for the length of the run and the process ending flushes it.
FILE *gSaving = nullptr;

struct Entry {
  Group group;
  std::string name;
  void (*body)();
};

/// The list, built as the translation units are loaded. A function rather than a global, because
/// a global in one translation unit and a Registration in another have no order between them.
std::vector<Entry> &entries() {
  static std::vector<Entry> list;
  return list;
}


/// Put the list in an order that does not depend on how the linker happened to order the static
/// initialisers that built it: by group, then by name.
void sortEntries() {
  std::sort(entries().begin(), entries().end(), [](const Entry &a, const Entry &b) {
    if (a.group != b.group) return a.group < b.group;
    return a.name < b.name;
  });
}

/// Whether a group's numbers are the card's or the processor's, which is what the kept file is
/// named after.
bool runsOnCuda(Group group) {
  return group == Group::kSdxlCuda;
}

/// The first line of /proc that begins with `key`, minus the key and the colon. Empty where there
/// is no /proc, which is every platform but Linux: the report says what it knows and leaves out
/// what it does not, rather than guessing.
std::string fromProc(const char *path, const char *key) {
  std::ifstream file(path);
  std::string line;
  std::string prefix = key;
  while (std::getline(file, line)) {
    if (line.compare(0, prefix.size(), prefix) != 0) continue;

    size_t colon = line.find(':');
    if (colon == std::string::npos) return std::string();

    size_t start = line.find_first_not_of(" \t", colon + 1);
    if (start == std::string::npos) return std::string();
    return line.substr(start);
  }
  return std::string();
}

/// A model name cut down to something that belongs in a file name: lowercase, without the words
/// every part of a make carries, and with the punctuation dropped except the hyphens inside a
/// model number, which are part of how the part is written. "NVIDIA GeForce RTX 5060 Ti" comes
/// out rtx5060ti and "Intel(R) Xeon(R) w5-2465X" comes out w5-2465x.
std::string slug(const std::string &model) {
  // The maker and the umbrella brand go; the product line stays, because "rtx5060ti" is what
  // someone would have called the file and "5060ti" is what they would have had to work out was
  // the same card.
  static const char *skip[] = {"nvidia", "geforce",   "intel", "amd",   "core",
                               "xeon",   "processor", "cpu",   "gpu",   "laptop"};

  std::string lowered;
  for (char c : model) lowered += char(tolower(c));

  // Before anything is split on: "intel(r)" would otherwise reach the list below as "intelr" and
  // match nothing, which is how a Xeon ended up called intelrxeonrw52465x.
  for (const char *mark : {"(r)", "(tm)"}) {
    for (size_t at = lowered.find(mark); at != std::string::npos; at = lowered.find(mark)) {
      lowered.erase(at, strlen(mark));
    }
  }

  std::string result;
  size_t at = 0;
  while (at < lowered.size()) {
    size_t end = lowered.find_first_of(" \t", at);
    if (end == std::string::npos) end = lowered.size();

    std::string word = lowered.substr(at, end - at);
    at = end + 1;

    bool wanted = true;
    for (const char *dull : skip) {
      if (word == dull) wanted = false;
    }
    if (!wanted) continue;

    for (char c : word) {
      if (isalnum(static_cast<unsigned char>(c))) {
        result += c;
      } else if (c == '-' && !result.empty() && result.back() != '-') {
        result += c;
      }
    }
  }
  return result;
}

std::string gpuSlug() {
#ifdef LIBWAIFU_CUDA_ENABLED
  int devices = 0;
  cudaDeviceProp properties;
  if (cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0 &&
      cudaGetDeviceProperties(&properties, 0) == cudaSuccess) {
    return slug(properties.name);
  }
#endif  // LIBWAIFU_CUDA_ENABLED
  return std::string();
}

/// The chip or processor name, from /proc/cpuinfo on Linux and sysctl on macOS.
std::string cpuModel() {
#ifdef __APPLE__
  char buf[256] = {};
  size_t len = sizeof(buf);
  if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0) return buf;
#endif
  return fromProc("/proc/cpuinfo", "model name");
}

/// Total physical memory in bytes, or zero when it cannot be read.
double totalMemoryBytes() {
#ifdef __APPLE__
  int64_t mem = 0;
  size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) return double(mem);
#endif
  std::string line = fromProc("/proc/meminfo", "MemTotal");
  if (!line.empty()) return atof(line.c_str()) * 1024.0;  // /proc reports kilobytes
  return 0;
}

std::string cpuSlug() {
  return slug(cpuModel());
}



/// What to call this machine in a file name, decided by what was run rather than by what the
/// machine has: a CPU benchmark on a machine with a card is about the processor, and a name that
/// says otherwise is a name that will be believed later.
std::string machineSlug(const std::vector<const Entry *> &chosen) {
  bool cuda = false;
  bool cpu = false;
  for (const Entry *entry : chosen) {
    if (runsOnCuda(entry->group)) {
      cuda = true;
    } else {
      cpu = true;
    }
  }

  std::string name;
  if (cuda) name = gpuSlug();
  if (cpu || name.empty()) {
    std::string processor = cpuSlug();
    if (!processor.empty()) {
      if (!name.empty()) name += "_";
      name += processor;
    }
  }
  return name.empty() ? "machine" : name;
}

/// The name a kept copy gets when none was given: what was run, then what it ran on.
std::string savedName(const char *group, const std::vector<const Entry *> &chosen) {
  return std::string(group) + "_" + machineSlug(chosen) + ".txt";
}

void reportCpu() {
  std::string model = cpuModel();
  unsigned threads = std::thread::hardware_concurrency();
  if (model.empty() && threads == 0) return;

  print("  cpu       %s", model.empty() ? "unknown" : model.c_str());
  if (threads > 0) print("  (%u threads)", threads);
  print("\n");

  double mem = totalMemoryBytes();
  if (mem > 0) print("  memory    %.1f GB\n", mem / 1e9);
}

void reportGpu() {
#ifdef LIBWAIFU_CUDA_ENABLED
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    print("  gpu       none the CUDA runtime can see\n");
    return;
  }

  for (int device = 0; device < devices; ++device) {
    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) continue;

    print(
        "  gpu:%d     %s  (sm_%d%d, %d SMs, %.1f GB, %.1f GB/s)\n",
        device,
        properties.name,
        properties.major,
        properties.minor,
        properties.multiProcessorCount,
        double(properties.totalGlobalMem) / 1e9,
        // What the bus can carry, which is the ceiling every memory-bound number below is up
        // against: the clock is in kilohertz and the bus is double-pumped.
        2.0 * properties.memoryClockRate * (properties.memoryBusWidth / 8) / 1e6);
  }

  int runtime = 0;
  int driver = 0;
  cudaRuntimeGetVersion(&runtime);
  cudaDriverGetVersion(&driver);
  print(
      "  cuda      runtime %d.%d, driver %d.%d\n",
      runtime / 1000,
      (runtime % 1000) / 10,
      driver / 1000,
      (driver % 1000) / 10);
#endif  // LIBWAIFU_CUDA_ENABLED
}

/// What was compiled in, which decides what the numbers above could have come from. A build
/// without CUTLASS and one with it measure different kernels under the same benchmark name.
void reportBuild() {
  std::string parts;
  auto add = [&parts](const char *what) {
    if (!parts.empty()) parts += ", ";
    parts += what;
  };

#ifdef LIBWAIFU_CUDA_ENABLED
  add("CUDA");
#endif
#ifdef LIBWAIFU_CUTLASS_ENABLED
  add("CUTLASS");
#endif
#ifdef LIBWAIFU_CUDNN_ENABLED
  add("cuDNN");
#endif
#ifdef LIBWAIFU_CUDA_MALLOC_ASYNC_ENABLED
  add("stream-ordered alloc");
#endif
#ifdef LIBWAIFU_CUDA_SYNC_ENABLED
  add("CUDA sync (slow)");
#endif
#ifdef _OPENMP
  add("OpenMP");
#endif

  print("  build     %s\n", parts.empty() ? "nothing optional" : parts.c_str());
}

void reportMachine() {
  // No blank line before it. This is the first thing in a kept file, and a file that opens on an
  // empty line reads as one that lost something. What separates it on the terminal is the line
  // above, which only the terminal gets.
  print("Running on\n");
  reportCpu();
  reportGpu();
  reportBuild();
}

void list() {
  sortEntries();
  for (Group group : {Group::kSdxlCuda, Group::kSdxlCpu, Group::kSdxlMetal}) {
    print("%s\n", nameOf(group));
    for (const Entry &entry : entries()) {
      if (entry.group == group) print("    %s\n", entry.name.c_str());
    }
  }
  print("\nName one of them to run it. --help says the rest.\n");
}

}  // namespace

void print(const char *format, ...) {
  va_list arguments;
  va_start(arguments, format);
  vprintf(format, arguments);
  va_end(arguments);

  if (gSaving) {
    va_start(arguments, format);
    vfprintf(gSaving, format, arguments);
    va_end(arguments);
  }
}

const char *nameOf(Group group) {
  switch (group) {
    case Group::kSdxlCuda:
      return "sdxl_cuda";
    case Group::kSdxlCpu:
      return "sdxl_cpu";
    case Group::kSdxlMetal:
      return "sdxl_metal";
  }
  return "unknown";
}

Registration::Registration(Group group, const char *name, void (*body)()) {
  entries().push_back(Entry{group, name, body});
}

int run(int argc, char **argv) {
  const char *wanted = nullptr;
  bool saving = false;
  std::string saved;
  for (int i = 1; i < argc; ++i) {
    std::string argument = argv[i];
    if (argument == "--list") {
      list();
      return 0;
    }
    if (argument == "--save" || argument.compare(0, 7, "--save=") == 0) {
      saving = true;
      if (argument.size() > 7) saved = argument.substr(7);
      continue;
    }
    if (argument == "--help" || argument == "-h") {
      print("Usage: benchmark GROUP [--save[=FILE]]\n\n");
      print("  GROUP       sdxl_cuda, sdxl_cpu, or sdxl_metal\n");
      print("  --list      print the groups and what is in them, and run nothing\n");
      print("  --save[=F]  keep a copy in a text file, named after the group and the machine\n");
      print("              when no name is given\n");
      return 0;
    }

    if (wanted) {
      print("one group at a time. --list says what there is.\n");
      return 1;
    }
    wanted = argv[i];
  }

  // Nothing named is a question rather than an instruction. Every group takes minutes, and a
  // command that starts all of them because it was typed bare is one nobody types twice.
  if (!wanted) {
    list();
    return 0;
  }

  Group group = Group::kSdxlCuda;
  bool found = false;
  for (Group candidate : {Group::kSdxlCuda, Group::kSdxlCpu, Group::kSdxlMetal}) {
    if (std::string(wanted) == nameOf(candidate)) {
      group = candidate;
      found = true;
    }
  }
  if (!found) {
    print("there is no group called \"%s\". --list says what there is.\n", wanted);
    return 1;
  }

  // Registration happens as the translation units load, in an order the standard does not fix,
  // and it showed: the two SDXL benchmarks came out one way on CUDA, where both live in one
  // file, and the other way on the processor, where they live in two.
  sortEntries();

  // Chosen before any of them runs, so that a group with nothing in it is answered at once
  // rather than after a page about the machine it would have run on.
  std::vector<const Entry *> chosen;
  for (const Entry &entry : entries()) {
    if (entry.group == group) chosen.push_back(&entry);
  }

  if (chosen.empty()) {
    print("%s has nothing in it in this build.\n", nameOf(group));
    return 1;
  }

  if (saving) {
    if (saved.empty()) saved = savedName(nameOf(group), chosen);

    FILE *file = fopen(saved.c_str(), "w");
    if (!file) {
      print("cannot write %s\n", saved.c_str());
      return 1;
    }

    // Said to the terminal and not into the file: where the copy went is not part of the copy.
    fprintf(stdout, "keeping a copy in %s\n\n", saved.c_str());
    gSaving = file;
  }

  // First, not last. Every number below is only worth something next to the machine it was taken
  // on, and a run long enough to walk away from is one whose top is still on screen when the
  // bottom is not. Printed even where every benchmark goes on to skip: what the machine has and
  // what the build has are most of the reason they would.
  reportMachine();

  for (const Entry *entry : chosen) {
    // The framework's job, not each benchmark's. One of them had gone without a heading for as
    // long as it had existed, which is what happens to anything every author has to remember on
    // their own. Underlined so that it outranks the section titles a body prints for itself.
    print("\n%s\n", entry->name.c_str());
    print("%s\n", std::string(entry->name.size(), '-').c_str());

    try {
      entry->body();
    } catch (const Skipped &skipped) {
      print("\n%s: skipped, %s\n", entry->name.c_str(), skipped.why().c_str());
    } catch (const std::exception &error) {
      // Reported and moved past rather than thrown out of. One benchmark that cannot allocate
      // says nothing about the twenty after it, and a run that stops at the first is a run that
      // has to be repeated with the first one filtered out.
      print("\n%s: failed, %s\n", entry->name.c_str(), error.what());
    }
  }

  return 0;
}

}  // namespace bench
}  // namespace fl
