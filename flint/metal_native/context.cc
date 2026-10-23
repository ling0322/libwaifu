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

#include "flint/metal_native/context.h"

#include <dispatch/dispatch.h>
#include <string.h>

#include <algorithm>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "flint/metal_native/kernels.h"
#include "flint/metal_native/metal_cpp.h"

namespace fl {
namespace op {
namespace metal_native {

namespace {

// Whether `device` has neural accelerators that kernels/nax/ can run on: Metal 4's tensor
// operations, which are macOS 26.2's, and a GPU of the generation that added the accelerators --
// 17, or 18 for the variant whose architecture name ends in p -- read off a name such as
// applegpu_g17s. The same test MLX makes. And a library to load, which a flint built without
// Metal 4 does not have.
bool hasNeuralAccelerators(MTL::Device *device) {
  if (kNaxMetallibSize == 0) return false;
  if (__builtin_available(macOS 26.2, *)) {
  } else {
    return false;
  }

  std::string name = device->architecture()->name()->utf8String();
  size_t g = name.rfind('g');
  if (g == std::string::npos || g + 1 >= name.size()) return false;
  int generation = atoi(name.c_str() + g + 1);
  char variant = name.back();
  return generation >= (variant == 'p' ? 18 : 17);
}

// The context, once opened, and never destroyed: the operators live in a static array whose
// destructor runs at exit, after the driver may have gone through exit handlers of its own.
// Everything the device holds goes back when the process does anyway.
std::mutex gContextMutex;
std::shared_ptr<Context> *gContext = nullptr;

}  // namespace

bool Context::isAvailable() {
  static const bool available = []() {
    try {
      get();
      return true;
    } catch (const lut::Error &e) {
      LOG(INFO) << "Metal is not available: " << e.what();
      return false;
    }
  }();
  return available;
}

std::shared_ptr<Context> Context::get() {
  std::lock_guard<std::mutex> lock(gContextMutex);
  if (!gContext) {
    std::shared_ptr<Context> context(new Context());
    context->open();
    gContext = new std::shared_ptr<Context>(std::move(context));
  }
  return *gContext;
}

void Context::open() {
  AutoreleasePool pool;

  _device = MTL::CreateSystemDefaultDevice();
  if (!_device) throw lut::AbortedError("there is no Metal device");

  // simdgroup matrices, which the products are written in, are what family 7 -- the M1 and
  // everything after it -- added.
  if (!_device->supportsFamily(MTL::GPUFamilyApple7)) {
    throw lut::AbortedError(lut::sprintf(
        "%s is not an Apple GPU of family 7 or later, which this backend needs",
        _device->name()->utf8String()));
  }

  // The kernels were compiled into one library when flint was built, and are carried in it.
  dispatch_data_t data = dispatch_data_create(
      kMetallib,
      kMetallibSize,
      nullptr,
      DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  NS::Error *error = nullptr;
  _library = _device->newLibrary(data, &error);
  dispatch_release(data);
  if (!_library) {
    throw lut::AbortedError("could not load flint's Metal kernels: " + describe(error));
  }

  // And those for the neural accelerators, where they can run. They are only faster, so a GPU
  // that cannot load them runs the kernels above as every other does.
  if (hasNeuralAccelerators(_device)) {
    dispatch_data_t naxData = dispatch_data_create(
        kNaxMetallib,
        kNaxMetallibSize,
        nullptr,
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NS::Error *naxError = nullptr;
    _naxLibrary = _device->newLibrary(naxData, &naxError);
    dispatch_release(naxData);
    if (!_naxLibrary) {
      LOG(WARN) << "could not load flint's Metal kernels for the neural accelerators: "
                << describe(naxError);
    }
  }

  _info.name = _device->name()->utf8String();
  _info.workingSetBytes = static_cast<int64_t>(_device->recommendedMaxWorkingSetSize());
  LOG(INFO) << "Metal device: " << _info.name << ", "
            << _info.workingSetBytes / (1 << 20) << " MB working set"
            << (_naxLibrary ? ", neural accelerators" : "");

  _stream = std::make_shared<CommandStream>(_device);
  _allocator = std::make_unique<VramAllocator>(_device, _info.workingSetBytes, _stream);
}

MTL::ComputePipelineState *Context::getKernel(const char *name) {
  std::lock_guard<std::mutex> lock(_kernelMutex);
  auto it = _kernels.find(name);
  if (it != _kernels.end()) return it->second;

  AutoreleasePool pool;
  MTL::Library *library = strncmp(name, "nax_", 4) == 0 ? _naxLibrary : _library;
  if (!library) throw lut::AbortedError(lut::sprintf("no Metal kernel called %s here", name));
  MTL::Function *function = library->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
  if (!function) throw lut::AbortedError(lut::sprintf("no Metal kernel called %s", name));

  NS::Error *error = nullptr;
  MTL::ComputePipelineState *pipeline = _device->newComputePipelineState(function, &error);
  function->release();
  if (!pipeline) {
    throw lut::AbortedError(
        lut::sprintf("could not build the Metal kernel %s: %s", name, describe(error)));
  }

  return _kernels.emplace(name, pipeline).first->second;
}

void Context::dispatch(
    const char *kernel,
    std::initializer_list<CommandStream::Arg> args,
    const void *params,
    size_t paramsSize,
    int64_t threads,
    int groupSize) {
  if (threads <= 0) return;
  _stream->dispatch(getKernel(kernel), args, params, paramsSize, threads, groupSize);
}

void Context::dispatchGroups(
    const char *kernel,
    std::initializer_list<CommandStream::Arg> args,
    const void *params,
    size_t paramsSize,
    uint32_t groupsX,
    uint32_t groupsY,
    uint32_t groupsZ,
    uint32_t threadsPerGroup) {
  if (groupsX == 0 || groupsY == 0 || groupsZ == 0) return;
  _stream->dispatchGroups(
      getKernel(kernel), args, params, paramsSize, groupsX, groupsY, groupsZ, threadsPerGroup);
}

}  // namespace metal_native
}  // namespace op
}  // namespace fl
