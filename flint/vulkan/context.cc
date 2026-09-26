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

#include "flint/vulkan/context.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/strings.h"
#include "flint/vulkan/shaders.h"

namespace fl {
namespace op {
namespace vulkan {

namespace {

// Submitted batches are waited for once there are more than this many, which is what keeps the
// host from recording arbitrarily far ahead of the device.
constexpr size_t kMaxBatchesInFlight = 3;

// A batch goes to the queue once it holds this many commands, so the device starts on the first
// of a long run of work while the rest is still being recorded.
constexpr int kCommandsPerBatch = 64;

// Downloads are staged a chunk at a time.
constexpr int64_t kStagingChunk = 64 << 20;

const char *toString(VkResult result) {
  switch (result) {
    case VK_SUCCESS:
      return "VK_SUCCESS";
    case VK_NOT_READY:
      return "VK_NOT_READY";
    case VK_TIMEOUT:
      return "VK_TIMEOUT";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
      return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
      return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST:
      return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
      return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT:
      return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:
      return "VK_ERROR_INCOMPATIBLE_DRIVER";
    default:
      return "VkResult";
  }
}

void check(VkResult result, const char *call) {
  if (result != VK_SUCCESS) {
    throw lut::AbortedError(lut::sprintf("%s failed: %s (%d)", call, toString(result), result));
  }
}

#define VK_CHECK(call) check((call), #call)

// What this backend needs of a device. Buffer device addresses are how every kernel reaches its
// tensors; 8- and 16-bit storage is how it reads bool and half tensors without widening them;
// and 64-bit integers are the index type the rest of flint uses.
struct Features {
  VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  VkPhysicalDeviceVulkan11Features vulkan11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceVulkan12Features vulkan12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};

  Features() {
    features2.pNext = &vulkan11;
    vulkan11.pNext = &vulkan12;
  }

  // Why `device` cannot run this backend, or an empty string when it can.
  std::string query(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(device, &properties);
    if (properties.apiVersion < VK_API_VERSION_1_2) return "it does not support Vulkan 1.2";

    vkGetPhysicalDeviceFeatures2(device, &features2);
    if (!vulkan12.bufferDeviceAddress) return "it has no buffer device addresses";
    if (!vulkan11.storageBuffer16BitAccess) return "it cannot store 16-bit values";
    if (!vulkan12.storageBuffer8BitAccess) return "it cannot store 8-bit values";
    if (!features2.features.shaderInt64) return "it has no 64-bit integers in shaders";
    return "";
  }

  // Only what query() asked for -- and cooperative matrices when `cooperativeMatrix`, which also
  // need the memory model their scopes are written in -- so that nothing else is switched on by
  // accident.
  void keepRequired(bool cooperativeMatrix) {
    VkPhysicalDeviceFeatures2 all = features2;
    features2.features = VkPhysicalDeviceFeatures{};
    features2.features.shaderInt64 = all.features.shaderInt64;

    VkPhysicalDeviceVulkan11Features all11 = vulkan11;
    vulkan11 = VkPhysicalDeviceVulkan11Features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    vulkan11.pNext = &vulkan12;
    vulkan11.storageBuffer16BitAccess = all11.storageBuffer16BitAccess;

    VkPhysicalDeviceVulkan12Features all12 = vulkan12;
    vulkan12 = VkPhysicalDeviceVulkan12Features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    vulkan12.bufferDeviceAddress = all12.bufferDeviceAddress;
    vulkan12.storageBuffer8BitAccess = all12.storageBuffer8BitAccess;
    vulkan12.shaderFloat16 = all12.shaderFloat16;
    vulkan12.shaderInt8 = all12.shaderInt8;
    if (cooperativeMatrix) {
      vulkan12.vulkanMemoryModel = VK_TRUE;
      cooperativeMatrixFeatures.cooperativeMatrix = VK_TRUE;
      vulkan12.pNext = &cooperativeMatrixFeatures;
    }
  }

  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperativeMatrixFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
};

bool hasExtension(VkPhysicalDevice device, const char *name) {
  uint32_t count = 0;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> extensions(count);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data());
  for (const VkExtensionProperties &extension : extensions) {
    if (strcmp(extension.extensionName, name) == 0) return true;
  }
  return false;
}

// Whether `device` offers the one cooperative matrix shape the kernels are written for: 16 x 16 x 16
// with half precision operands, a float accumulator and result, over a subgroup -- and runs a
// subgroup of 32, which is how those kernels share out a tile.
bool hasCooperativeMatrix(VkPhysicalDevice device, uint32_t subgroupSize) {
  if (subgroupSize != 32 || !hasExtension(device, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
    return false;
  }
  if (!vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) return false;

  VkPhysicalDeviceVulkan12Features vulkan12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  vulkan12.pNext = &cooperative;
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features.pNext = &vulkan12;
  vkGetPhysicalDeviceFeatures2(device, &features);
  if (!cooperative.cooperativeMatrix || !vulkan12.vulkanMemoryModel || !vulkan12.shaderFloat16) {
    return false;
  }

  uint32_t count = 0;
  vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device, &count, nullptr);
  std::vector<VkCooperativeMatrixPropertiesKHR> shapes(
      count,
      VkCooperativeMatrixPropertiesKHR{VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
  vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(device, &count, shapes.data());
  for (const VkCooperativeMatrixPropertiesKHR &shape : shapes) {
    if (shape.MSize == 16 && shape.NSize == 16 && shape.KSize == 16 &&
        shape.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
        shape.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
        shape.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
        shape.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR && shape.scope == VK_SCOPE_SUBGROUP_KHR) {
      return true;
    }
  }
  return false;
}

bool hasLayer(const char *name) {
  uint32_t count = 0;
  vkEnumerateInstanceLayerProperties(&count, nullptr);
  std::vector<VkLayerProperties> layers(count);
  vkEnumerateInstanceLayerProperties(&count, layers.data());
  for (const VkLayerProperties &layer : layers) {
    if (strcmp(layer.layerName, name) == 0) return true;
  }
  return false;
}

// A software implementation of Vulkan, such as lavapipe, is slower at this than flint's own CPU
// operators, so it is only chosen when asked for by name.
int preference(VkPhysicalDeviceType type) {
  switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return 3;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return 2;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return 1;
    default:
      return -1;
  }
}

// Picks the device: the one FLINT_VULKAN_DEVICE names -- by its index among the devices, or by a
// part of its name -- when it is set, and otherwise the most capable kind there is, the first of
// that kind winning. Returns null, with the reason in `whyNot`, when there is nothing to pick.
VkPhysicalDevice pickDevice(VkInstance instance, std::string *whyNot) {
  uint32_t count = 0;
  vkEnumeratePhysicalDevices(instance, &count, nullptr);
  std::vector<VkPhysicalDevice> devices(count);
  vkEnumeratePhysicalDevices(instance, &count, devices.data());
  if (devices.empty()) {
    *whyNot = "the loader lists no devices";
    return VK_NULL_HANDLE;
  }

  const char *wanted = getenv("FLINT_VULKAN_DEVICE");
  VkPhysicalDevice best = VK_NULL_HANDLE;
  int bestPreference = -1;
  std::string reasons;
  for (uint32_t i = 0; i < count; ++i) {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(devices[i], &properties);

    Features features;
    std::string problem = features.query(devices[i]);
    if (wanted && *wanted) {
      bool named = std::to_string(i) == wanted || strstr(properties.deviceName, wanted);
      if (!named) continue;
      if (!problem.empty()) {
        *whyNot = lut::sprintf("FLINT_VULKAN_DEVICE names %s, but %s", properties.deviceName,
                               problem);
        return VK_NULL_HANDLE;
      }
      return devices[i];
    }

    if (!problem.empty()) {
      reasons += lut::sprintf("%s%s: %s", reasons.empty() ? "" : "; ", properties.deviceName,
                              problem);
      continue;
    }
    int score = preference(properties.deviceType);
    if (score > bestPreference) {
      best = devices[i];
      bestPreference = score;
    }
  }

  if (wanted && *wanted) {
    *whyNot = lut::sprintf("FLINT_VULKAN_DEVICE=%s names none of the devices", wanted);
  } else if (!best) {
    *whyNot = reasons.empty() ? "only software implementations were found" : reasons;
  }
  return best;
}

// The context, once opened, and never destroyed. The operators live in a static array whose
// destructor runs at exit, and by then the driver has been through exit handlers of its own:
// destroying the device from there crashed inside the driver. Everything the device holds goes
// back when the process does anyway.
std::mutex gContextMutex;
std::shared_ptr<Context> *gContext = nullptr;

}  // namespace

bool Context::isAvailable() {
  static const bool available = []() {
    try {
      get();
      return true;
    } catch (const lut::Error &e) {
      LOG(INFO) << "Vulkan is not available: " << e.what();
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
  if (volkInitialize() != VK_SUCCESS) throw lut::AbortedError("unable to load the Vulkan loader");
  if (volkGetInstanceVersion() < VK_API_VERSION_1_2) {
    throw lut::AbortedError("the Vulkan loader is older than 1.2");
  }

  // FLINT_VULKAN_VALIDATION=1 turns on the Khronos validation layer, where it is installed. It
  // is for working on the kernels: it slows everything down a great deal.
  std::vector<const char *> layers;
  const char *validation = getenv("FLINT_VULKAN_VALIDATION");
  if (validation && strcmp(validation, "1") == 0) {
    if (hasLayer("VK_LAYER_KHRONOS_validation")) {
      layers.push_back("VK_LAYER_KHRONOS_validation");
    } else {
      LOG(WARN) << "FLINT_VULKAN_VALIDATION is set, but VK_LAYER_KHRONOS_validation is missing";
    }
  }

  VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  application.pApplicationName = "flint";
  application.pEngineName = "flint";
  application.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instanceInfo.pApplicationInfo = &application;
  instanceInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
  instanceInfo.ppEnabledLayerNames = layers.data();
  VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &_instance));
  volkLoadInstanceOnly(_instance);

  std::string whyNot;
  _physicalDevice = pickDevice(_instance, &whyNot);
  if (!_physicalDevice) throw lut::AbortedError("no usable Vulkan device: " + whyNot);

  VkPhysicalDeviceSubgroupProperties subgroup{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  properties.pNext = &subgroup;
  vkGetPhysicalDeviceProperties2(_physicalDevice, &properties);
  _info.name = properties.properties.deviceName;
  for (int i = 0; i < 3; ++i) {
    _info.maxWorkGroupCount[i] = properties.properties.limits.maxComputeWorkGroupCount[i];
  }
  _info.subgroupSize = subgroup.subgroupSize;
  if (properties.properties.limits.maxPushConstantsSize < kPushConstantSize) {
    throw lut::AbortedError("the device allows fewer push constants than Vulkan requires");
  }

  VkPhysicalDeviceMemoryProperties memory;
  vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &memory);
  _info.deviceLocalBytes = 0;
  for (uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
    if (memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      _info.deviceLocalBytes = std::max<int64_t>(_info.deviceLocalBytes, memory.memoryHeaps[i].size);
    }
  }

  // Any family that can compute. The first is usually the one that can do everything, which is
  // also the one with the most of the device behind it.
  uint32_t numFamilies = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(_physicalDevice, &numFamilies, nullptr);
  std::vector<VkQueueFamilyProperties> families(numFamilies);
  vkGetPhysicalDeviceQueueFamilyProperties(_physicalDevice, &numFamilies, families.data());
  _queueFamily = numFamilies;
  for (uint32_t i = 0; i < numFamilies; ++i) {
    if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      _queueFamily = i;
      break;
    }
  }
  if (_queueFamily == numFamilies) throw lut::AbortedError("the device has no compute queue");

  float priority = 1.0f;
  VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  queueInfo.queueFamilyIndex = _queueFamily;
  queueInfo.queueCount = 1;
  queueInfo.pQueuePriorities = &priority;

  std::vector<const char *> extensions;
  _info.memoryBudget = hasExtension(_physicalDevice, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
  if (_info.memoryBudget) extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

  // FLINT_VULKAN_COOPERATIVE_MATRIX=0 turns the tensor core kernels off, to compare against the
  // ones every device runs.
  const char *cooperative = getenv("FLINT_VULKAN_COOPERATIVE_MATRIX");
  _info.cooperativeMatrix = !(cooperative && strcmp(cooperative, "0") == 0) &&
                            hasCooperativeMatrix(_physicalDevice, _info.subgroupSize);
  if (_info.cooperativeMatrix) extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

  Features features;
  features.query(_physicalDevice);
  features.keepRequired(_info.cooperativeMatrix);

  VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  deviceInfo.pNext = &features.features2;
  deviceInfo.queueCreateInfoCount = 1;
  deviceInfo.pQueueCreateInfos = &queueInfo;
  deviceInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  deviceInfo.ppEnabledExtensionNames = extensions.data();
  VK_CHECK(vkCreateDevice(_physicalDevice, &deviceInfo, nullptr, &_device));
  volkLoadDevice(_device);
  vkGetDeviceQueue(_device, _queueFamily, 0, &_queue);

  VmaVulkanFunctions functions{};
  functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

  VmaAllocatorCreateInfo allocatorInfo{};
  allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  if (_info.memoryBudget) allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
  allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_2;
  allocatorInfo.physicalDevice = _physicalDevice;
  allocatorInfo.device = _device;
  allocatorInfo.instance = _instance;
  allocatorInfo.pVulkanFunctions = &functions;
  VK_CHECK(vmaCreateAllocator(&allocatorInfo, &_allocator));

  VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = _queueFamily;
  VK_CHECK(vkCreateCommandPool(_device, &poolInfo, nullptr, &_commandPool));

  VkPushConstantRange pushRange{};
  pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushRange.offset = 0;
  pushRange.size = kPushConstantSize;

  VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layoutInfo.pushConstantRangeCount = 1;
  layoutInfo.pPushConstantRanges = &pushRange;
  VK_CHECK(vkCreatePipelineLayout(_device, &layoutInfo, nullptr, &_pipelineLayout));

  const char *profile = getenv("FLINT_VULKAN_PROFILE");
  if (profile && strcmp(profile, "1") == 0) {
    if (families[_queueFamily].timestampValidBits == 0) {
      LOG(WARN) << "FLINT_VULKAN_PROFILE is set, but the queue has no timestamps";
    } else {
      _profile = true;
      _timestampPeriod = properties.properties.limits.timestampPeriod;
      std::atexit([]() {
        if (gContext) (*gContext)->printProfile();
      });
    }
  }

  LOG(INFO) << "Vulkan device: " << _info.name
            << (_info.cooperativeMatrix ? ", with cooperative matrices" : "");
}

Context::~Context() {
  if (!_device) {
    if (_instance) vkDestroyInstance(_instance, nullptr);
    return;
  }

  try {
    synchronize();
  } catch (const lut::Error &e) {
    LOG(ERROR) << "while closing the Vulkan device: " << e.what();
    vkDeviceWaitIdle(_device);
  }
  emptyCache();
  if (_staging.buffer) destroyBuffer(_staging);
  if (_allocatedBytes) {
    LOG(WARN) << _allocatedBytes << " bytes of Vulkan tensors outlived the device";
  }

  for (Batch &batch : _idle) {
    vkDestroyFence(_device, batch.fence, nullptr);
    if (batch.timestamps) vkDestroyQueryPool(_device, batch.timestamps, nullptr);
  }
  for (auto &entry : _kernels) {
    vkDestroyPipeline(_device, entry.second.pipeline, nullptr);
    vkDestroyShaderModule(_device, entry.second.module, nullptr);
  }
  vkDestroyPipelineLayout(_device, _pipelineLayout, nullptr);
  vkDestroyCommandPool(_device, _commandPool, nullptr);
  vmaDestroyAllocator(_allocator);
  vkDestroyDevice(_device, nullptr);
  vkDestroyInstance(_instance, nullptr);
}

const Context::Kernel &Context::getKernel(const char *name) {
  auto it = _kernels.find(name);
  if (it != _kernels.end()) return it->second;

  const Spirv *spirv = findSpirv(name);
  if (!spirv) throw lut::AbortedError(lut::sprintf("no Vulkan kernel called %s", name));

  Kernel kernel;
  VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  moduleInfo.codeSize = spirv->size;
  moduleInfo.pCode = spirv->code;
  VK_CHECK(vkCreateShaderModule(_device, &moduleInfo, nullptr, &kernel.module));

  VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineInfo.stage.module = kernel.module;
  pipelineInfo.stage.pName = "main";
  pipelineInfo.layout = _pipelineLayout;
  VkResult result = vkCreateComputePipelines(
      _device,
      VK_NULL_HANDLE,
      1,
      &pipelineInfo,
      nullptr,
      &kernel.pipeline);
  if (result != VK_SUCCESS) {
    vkDestroyShaderModule(_device, kernel.module, nullptr);
    check(result, "vkCreateComputePipelines");
  }

  return _kernels.emplace(name, kernel).first->second;
}

Buffer Context::createBuffer(
    int64_t bytes,
    VkBufferUsageFlags usage,
    VmaMemoryUsage memoryUsage,
    VmaAllocationCreateFlags flags) {
  VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bufferInfo.size = static_cast<VkDeviceSize>(bytes);
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo allocationInfo{};
  allocationInfo.usage = memoryUsage;
  allocationInfo.flags = flags;

  Buffer buffer;
  buffer.size = bytes;
  VkResult result = vmaCreateBuffer(
      _allocator,
      &bufferInfo,
      &allocationInfo,
      &buffer.buffer,
      &buffer.allocation,
      nullptr);
  if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY) {
    return Buffer();
  }
  check(result, "vmaCreateBuffer");

  if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
    VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addressInfo.buffer = buffer.buffer;
    buffer.address = vkGetBufferDeviceAddress(_device, &addressInfo);
  }
  return buffer;
}

void Context::destroyBuffer(const Buffer &buffer) {
  vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
}

Buffer Context::allocate(int64_t bytes) {
  CHECK(bytes > 0);

  // Rounded up to a size class an eighth of a power of two wide, which wastes at most an eighth
  // of any buffer and makes the sizes a model asks for over and over fall into a few classes.
  int64_t granule = 256;
  while (granule * 16 <= bytes) granule *= 2;
  int64_t size = (bytes + granule - 1) / granule * granule;

  std::lock_guard<std::recursive_mutex> lock(_mutex);
  auto it = _cache.lower_bound(size);
  if (it != _cache.end() && it->first <= size + size / 4) {
    Buffer buffer = it->second;
    _cache.erase(it);
    _cachedBytes -= buffer.size;
    _allocatedBytes += buffer.size;
    _peakAllocatedBytes = std::max(_peakAllocatedBytes, _allocatedBytes);
    return buffer;
  }

  VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  VmaAllocationCreateFlags flags = _info.memoryBudget ? VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT : 0;
  Buffer buffer = createBuffer(size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, flags);
  if (!buffer.buffer) {
    // Whatever the cache holds is memory nothing is using, so it goes back before giving up.
    synchronize();
    emptyCache();
    buffer = createBuffer(size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, flags);
  }
  if (!buffer.buffer) {
    throw lut::AbortedError(lut::sprintf(
        "out of memory: the Vulkan device could not allocate %d bytes (%d held by tensors)",
        size,
        _allocatedBytes));
  }

  _allocatedBytes += buffer.size;
  _peakAllocatedBytes = std::max(_peakAllocatedBytes, _allocatedBytes);
  return buffer;
}

void Context::free(const Buffer &buffer) {
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  _allocatedBytes -= buffer.size;
  _cachedBytes += buffer.size;
  _cache.emplace(buffer.size, buffer);
}

void Context::emptyCache() {
  for (auto &entry : _cache) destroyBuffer(entry.second);
  _cache.clear();
  _cachedBytes = 0;
}

void Context::releaseUnusedMemory() {
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  synchronize();
  emptyCache();
}

MemorySnapshot Context::captureMemorySnapshot() {
  std::lock_guard<std::recursive_mutex> lock(_mutex);

  int64_t total = _info.deviceLocalBytes;
  int64_t free = total - _allocatedBytes - _cachedBytes;
  if (_info.memoryBudget) {
    // The budget is what this process may use of each heap, and usage what it already does, so
    // their difference is what nobody has taken yet -- other processes included.
    std::vector<VmaBudget> budgets(VK_MAX_MEMORY_HEAPS);
    vmaGetHeapBudgets(_allocator, budgets.data());
    const VkPhysicalDeviceMemoryProperties *memory = nullptr;
    vmaGetMemoryProperties(_allocator, &memory);
    free = 0;
    for (uint32_t i = 0; i < memory->memoryHeapCount; ++i) {
      if (!(memory->memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
      if (static_cast<int64_t>(memory->memoryHeaps[i].size) != total) continue;
      free = std::max<int64_t>(0, budgets[i].budget - budgets[i].usage);
    }
  }

  return MemorySnapshot(total, std::max<int64_t>(0, free), _allocatedBytes, _peakAllocatedBytes);
}

void Context::resetPeakMemoryStats() {
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  _peakAllocatedBytes = _allocatedBytes;
}

VkCommandBuffer Context::beginCommand(const std::string &label) {
  if (!_recording.commandBuffer) {
    if (!_idle.empty()) {
      _recording = std::move(_idle.back());
      _idle.pop_back();
    } else {
      VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      allocateInfo.commandPool = _commandPool;
      allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      allocateInfo.commandBufferCount = 1;
      VK_CHECK(vkAllocateCommandBuffers(_device, &allocateInfo, &_recording.commandBuffer));

      VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      VK_CHECK(vkCreateFence(_device, &fenceInfo, nullptr, &_recording.fence));

      if (_profile) {
        VkQueryPoolCreateInfo queryInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryInfo.queryCount = 2 * kCommandsPerBatch;
        VK_CHECK(vkCreateQueryPool(_device, &queryInfo, nullptr, &_recording.timestamps));
      }
    }

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(_recording.commandBuffer, &beginInfo));
    if (_profile) {
      vkCmdResetQueryPool(_recording.commandBuffer, _recording.timestamps, 0,
                          2 * kCommandsPerBatch);
    }
  }

  // Every command waits for every one before it, including those of batches already submitted --
  // a barrier's first scope is everything earlier in submission order on the queue. That is
  // coarser than tracking which tensors each command reads, and it is what makes the order work
  // was asked for in the order it runs in, which the buffer cache depends on.
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                          VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                VK_PIPELINE_STAGE_TRANSFER_BIT;
  vkCmdPipelineBarrier(
      _recording.commandBuffer,
      stages,
      stages,
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr);

  // After the barrier, so that what is timed is this command and not the wait for the last one.
  if (_profile) {
    vkCmdWriteTimestamp(
        _recording.commandBuffer,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        _recording.timestamps,
        2 * _numRecorded);
    _recording.labels.push_back(label);
  }

  return _recording.commandBuffer;
}

void Context::endCommand() {
  if (_profile) {
    vkCmdWriteTimestamp(
        _recording.commandBuffer,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        _recording.timestamps,
        2 * _numRecorded + 1);
  }
  if (++_numRecorded >= kCommandsPerBatch) submit();
}

void Context::submit() {
  if (!_recording.commandBuffer) return;

  VK_CHECK(vkEndCommandBuffer(_recording.commandBuffer));

  VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &_recording.commandBuffer;
  VK_CHECK(vkQueueSubmit(_queue, 1, &submitInfo, _recording.fence));
  _recording.serial = ++_submittedSerial;

  _submitted.push_back(std::move(_recording));
  _recording = Batch();
  _numRecorded = 0;

  while (_submitted.size() > kMaxBatchesInFlight) {
    retire(_submitted.front());
    _submitted.pop_front();
  }
}

void Context::retire(Batch &batch) {
  VkResult result = vkWaitForFences(_device, 1, &batch.fence, VK_TRUE, UINT64_MAX);
  if (result == VK_ERROR_DEVICE_LOST) {
    // Nothing recorded after this can be trusted, and there is no recovering the device.
    LOG(FATAL) << "the Vulkan device was lost";
    abort();
  }
  check(result, "vkWaitForFences");
  VK_CHECK(vkResetFences(_device, 1, &batch.fence));
  VK_CHECK(vkResetCommandBuffer(batch.commandBuffer, 0));

  _completedSerial = std::max(_completedSerial, batch.serial);

  if (_profile && !batch.labels.empty()) {
    std::vector<uint64_t> ticks(2 * batch.labels.size());
    VK_CHECK(vkGetQueryPoolResults(
        _device,
        batch.timestamps,
        0,
        static_cast<uint32_t>(ticks.size()),
        ticks.size() * sizeof(uint64_t),
        ticks.data(),
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    for (size_t i = 0; i < batch.labels.size(); ++i) {
      ProfileEntry &entry = _profileTotals[batch.labels[i]];
      entry.count += 1;
      entry.nanoseconds += (ticks[2 * i + 1] - ticks[2 * i]) * _timestampPeriod;
    }
    batch.labels.clear();
  }

  _idle.push_back(std::move(batch));
}

void Context::synchronize() {
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  submit();
  while (!_submitted.empty()) {
    retire(_submitted.front());
    _submitted.pop_front();
  }
}

void Context::dispatch(
    const char *kernel,
    const void *push,
    size_t pushSize,
    uint32_t groupsX,
    uint32_t groupsY,
    uint32_t groupsZ) {
  CHECK(pushSize <= kPushConstantSize);
  if (groupsX == 0 || groupsY == 0 || groupsZ == 0) return;
  CHECK(groupsX <= _info.maxWorkGroupCount[0] && groupsY <= _info.maxWorkGroupCount[1] &&
        groupsZ <= _info.maxWorkGroupCount[2]);

  std::lock_guard<std::recursive_mutex> lock(_mutex);
  const Kernel &compiled = getKernel(kernel);

  VkCommandBuffer commandBuffer = beginCommand(kernel);
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, compiled.pipeline);
  vkCmdPushConstants(
      commandBuffer,
      _pipelineLayout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      static_cast<uint32_t>(pushSize),
      push);
  vkCmdDispatch(commandBuffer, groupsX, groupsY, groupsZ);
  endCommand();
}

void Context::dispatchLinear(
    const char *kernel,
    const void *push,
    size_t pushSize,
    int64_t numThreads,
    int groupSize) {
  if (numThreads <= 0) return;

  int64_t numGroups = (numThreads + groupSize - 1) / groupSize;
  int64_t groupsX = std::min<int64_t>(numGroups, _info.maxWorkGroupCount[0]);
  int64_t groupsY = (numGroups + groupsX - 1) / groupsX;
  dispatch(
      kernel,
      push,
      pushSize,
      static_cast<uint32_t>(groupsX),
      static_cast<uint32_t>(groupsY));
}

void Context::fillBuffer(const Buffer &buffer, int64_t offset, int64_t bytes, uint32_t value) {
  CHECK(offset % 4 == 0 && bytes % 4 == 0);
  if (bytes == 0) return;

  std::lock_guard<std::recursive_mutex> lock(_mutex);
  VkCommandBuffer commandBuffer = beginCommand("fillBuffer");
  vkCmdFillBuffer(commandBuffer, buffer.buffer, offset, bytes, value);
  endCommand();
}

void Context::copyBuffer(
    const Buffer &src,
    int64_t srcOffset,
    const Buffer &dest,
    int64_t destOffset,
    int64_t bytes) {
  if (bytes == 0) return;

  std::lock_guard<std::recursive_mutex> lock(_mutex);
  VkBufferCopy region{};
  region.srcOffset = srcOffset;
  region.dstOffset = destOffset;
  region.size = bytes;

  VkCommandBuffer commandBuffer = beginCommand("copyBuffer");
  vkCmdCopyBuffer(commandBuffer, src.buffer, dest.buffer, 1, &region);
  endCommand();
}

void Context::nextStagingSegment() {
  _stagingSegment = (_stagingSegment + 1) % kStagingSegments;
  _stagingOffset = 0;

  uint64_t lastUse = _segmentSerial[_stagingSegment];
  if (lastUse <= _completedSerial) return;
  if (lastUse > _submittedSerial) submit();  // it is the batch still being recorded
  while (_completedSerial < lastUse) {
    retire(_submitted.front());
    _submitted.pop_front();
  }
}

void Context::upload(const void *src, const Buffer &dest, int64_t offset, int64_t bytes) {
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  const std::byte *from = reinterpret_cast<const std::byte *>(src);

  if (!_staging.buffer) {
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = kStagingSegments * kStagingSegmentBytes;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    allocationInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                           VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    VK_CHECK(vmaCreateBuffer(
        _allocator,
        &bufferInfo,
        &allocationInfo,
        &_staging.buffer,
        &_staging.allocation,
        &info));
    _staging.size = static_cast<int64_t>(bufferInfo.size);
    _stagingMemory = reinterpret_cast<std::byte *>(info.pMappedData);
  }

  for (int64_t done = 0; done < bytes;) {
    if (_stagingOffset == kStagingSegmentBytes) nextStagingSegment();
    int64_t chunk = std::min(kStagingSegmentBytes - _stagingOffset, bytes - done);
    int64_t stagingOffset = _stagingSegment * kStagingSegmentBytes + _stagingOffset;

    memcpy(_stagingMemory + stagingOffset, from + done, chunk);
    VK_CHECK(vmaFlushAllocation(_allocator, _staging.allocation, stagingOffset, chunk));

    VkBufferCopy region{};
    region.srcOffset = stagingOffset;
    region.dstOffset = offset + done;
    region.size = chunk;

    VkCommandBuffer commandBuffer = beginCommand("upload");
    vkCmdCopyBuffer(commandBuffer, _staging.buffer, dest.buffer, 1, &region);
    _segmentSerial[_stagingSegment] = _submittedSerial + 1;
    endCommand();

    // The next chunk starts on sixteen bytes, which is what copies are fastest from.
    _stagingOffset = std::min(kStagingSegmentBytes, (_stagingOffset + chunk + 15) / 16 * 16);
    done += chunk;
  }
}

void Context::download(const Buffer &src, int64_t offset, void *dest, int64_t bytes) {
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  std::byte *to = reinterpret_cast<std::byte *>(dest);

  for (int64_t done = 0; done < bytes; done += kStagingChunk) {
    int64_t chunk = std::min(kStagingChunk, bytes - done);
    Buffer staging = createBuffer(
        chunk,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
    if (!staging.buffer) throw lut::AbortedError("out of host memory for a Vulkan download");

    VkBufferCopy region{};
    region.srcOffset = offset + done;
    region.dstOffset = 0;
    region.size = chunk;

    VkCommandBuffer commandBuffer = beginCommand("download");
    vkCmdCopyBuffer(commandBuffer, src.buffer, staging.buffer, 1, &region);

    // What the copy wrote has to be made visible to the host before it reads it.
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1,
        &barrier,
        0,
        nullptr,
        0,
        nullptr);
    endCommand();

    try {
      synchronize();
      check(vmaCopyAllocationToMemory(_allocator, staging.allocation, 0, to + done, chunk),
            "vmaCopyAllocationToMemory");
    } catch (...) {
      destroyBuffer(staging);
      throw;
    }
    destroyBuffer(staging);
  }
}

void Context::printProfile() {
  std::lock_guard<std::recursive_mutex> lock(_mutex);

  std::vector<std::pair<std::string, ProfileEntry>> entries(
      _profileTotals.begin(),
      _profileTotals.end());
  std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b) {
    return a.second.nanoseconds > b.second.nanoseconds;
  });

  double total = 0.0;
  for (const auto &entry : entries) total += entry.second.nanoseconds;

  fprintf(stderr, "flint Vulkan profile: %.1f ms on the device\n", total / 1e6);
  fprintf(stderr, "  %-24s %10s %12s %10s %7s\n", "kernel", "calls", "total ms", "avg us", "share");
  for (const auto &entry : entries) {
    const ProfileEntry &e = entry.second;
    fprintf(stderr, "  %-24s %10lld %12.2f %10.1f %6.1f%%\n",
            entry.first.c_str(),
            static_cast<long long>(e.count),
            e.nanoseconds / 1e6,
            e.nanoseconds / 1e3 / e.count,
            100.0 * e.nanoseconds / total);
  }
}

}  // namespace vulkan
}  // namespace op
}  // namespace fl
