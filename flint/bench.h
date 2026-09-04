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

#include <string>

namespace fl {
namespace bench {

/// @brief The sets a benchmark belongs to. Two, and adding a third is a decision made here rather
///        than by whoever writes the next benchmark.
///
/// It was a list of tags once, matched by substring, and what that bought was a benchmark tagged
/// [cpu_kernel] that [cpu] did not select and a CPU run whose file was named after the card. A
/// closed set of four says what a benchmark is with no room to say it two ways.
enum class Group {
  /// What SDXL runs on the GPU, in the shapes and the proportions it runs them.
  kSdxlCuda,
  /// The same, on the processor.
  kSdxlCpu,
};

/// @brief What a group is called on the command line and in a saved file's name.
const char *nameOf(Group group);

/// @brief Thrown by LL_BENCHMARK_SKIP to leave a benchmark unrun and say why.
///
/// A benchmark that cannot run is not a failure -- no CUDA device, a build without the library it
/// measures -- but it must not print a number either, and it must say which of the two happened.
class Skipped {
 public:
  explicit Skipped(std::string why)
      : _why(std::move(why)) {
  }

  const std::string &why() const {
    return _why;
  }

 private:
  std::string _why;
};

/// @brief Puts one benchmark in the list at static-initialisation time. Made by LL_BENCHMARK.
class Registration {
 public:
  Registration(Group group, const char *name, void (*body)());
};

/// @brief Define a benchmark in one of the groups.
///
/// The body prints its own table: what a benchmark has to say is a shape and a number, and no
/// framework knows how to lay that out better than the code that measured it does. The heading
/// above it is the framework's, so that no benchmark can be written without one.
#define LL_BENCHMARK(group, name) LL_BENCHMARK_AT(group, name, __LINE__)
#define LL_BENCHMARK_AT(group, name, line) LL_BENCHMARK_JOINED(group, name, line)
#define LL_BENCHMARK_JOINED(group, name, line)                        \
  static void llBenchmarkBody##line();                                \
  static const ::fl::bench::Registration llBenchmarkReg##line(        \
      group,                                                          \
      name,                                                           \
      llBenchmarkBody##line);                                         \
  static void llBenchmarkBody##line()

/// @brief Leave this benchmark unrun, saying why.
#define LL_BENCHMARK_SKIP(why) throw ::fl::bench::Skipped(why)

/// @brief Print a line of the report, to the terminal and to the file --save asked for.
///
/// Every benchmark says what it found through this rather than through printf, which is what lets
/// a run be kept: the numbers are worth nothing a week later without the machine they came from,
/// and a terminal scrollback is not where that survives.
#ifdef __GNUC__
__attribute__((format(printf, 1, 2)))
#endif
void print(const char *format, ...);

/// @brief Run the group the arguments name and report the machine. What main returns.
///
/// With no group every benchmark runs. `--list` prints the groups and what is in them. `--save`
/// keeps a copy in a text file, named after the group and the machine -- sdxl_cuda on a 5060 Ti
/// becomes sdxl_cuda_rtx5060ti.txt -- unless a name is given as `--save=FILE`.
int run(int argc, char **argv);

}  // namespace bench
}  // namespace fl
