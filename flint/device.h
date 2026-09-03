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

#include <string>

namespace fl {

// storage device for tensor data.
// Note: once the Device type is increased, we should also change the initialization of
// gOperatorsForDevice.
class Device {
 public:
  enum Type {
    kCpu,
    kCuda,
    kMetal,
    /// @brief Host memory that the CUDA driver allocated and page-locked, so that its copy
    /// engine can read it directly instead of staging it through a buffer of its own. Named
    /// after the cudaHost* family of calls that makes it, because that family is what owns it:
    /// the CPU may read and write it like any other memory, but only the CUDA operators know how
    /// to create it or move it across the bus.
    ///
    /// It carries no operators of its own. Asking for one says so rather than quietly running
    /// the CPU's, which would be right for the arithmetic and wrong about which device this
    /// memory belongs to.
    kCudaHost,
    NumDeviceType,  // number of device types
    kUnknown
  };

  /// @brief Return true if cuda device is available.
  /// @return availability of cuda device.
  static bool isCudaAvailable();

  /// @brief Return true if the Metal device is available, which needs both a build with MLX and
  ///        a machine with a Metal GPU.
  /// @return availability of the Metal device.
  static bool isMetalAvailable();

  static Device getCpu();
  static Device getCuda();
  static Device getMetal();
  static Device getCudaHost();

  // construct device by device type
  Device();
  Device(Type type);

  // get type of the device
  Type getType() const {
    return _type;
  }

  /// @brief Whether this device's memory is host memory, which the CPU may address directly.
  /// True of both the CPU and the page-locked host memory CUDA hands out.
  /// @return whether the memory lives on the host.
  bool isHost() const {
    return _type == kCpu || _type == kCudaHost;
  }

  /// @brief Get the name of device.
  /// @return name of the device.
  std::string getName() const;

 private:
  Type _type;
};

}  // namespace fl
