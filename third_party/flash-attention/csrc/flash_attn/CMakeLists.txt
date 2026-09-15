cmake_minimum_required(VERSION 3.22)
project(flash_attn LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

# Built as a separate project so its pinned CUTLASS never reaches the libllm build.
set(FLASH_ATTN_CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cutlass")

if(NOT EXISTS "${FLASH_ATTN_CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR
        "FlashAttention CUTLASS was not found at ${FLASH_ATTN_CUTLASS_DIR}. "
        "Run third_party/install_flash_attn.sh.")
endif()

# Chosen before the CUDA language is enabled, which is why this project takes CUDA through
# enable_language rather than through project(). Enabling it is what gives
# CMAKE_CUDA_ARCHITECTURES a default of its own -- 52, on CMake 4 -- and a test for DEFINED made
# after that never fires: the list below was skipped and the kernels were built for a card from
# 2014, in fourteen seconds, to no complaint from anything.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Real code for every target: sm80 PTX JITs into silently wrong kernels on sm120.
    set(CMAKE_CUDA_ARCHITECTURES 80-real 86-real 89-real 90-real 120-real)
endif()
message(STATUS "flash_attn CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The same runtime the top-level build picks. This archive is built by a project of its own and
# inherits none of that build's settings, so left alone MSVC would give it the dynamic runtime
# and the static one to everything it links beside -- which the linker reports as an LNK2038
# mismatch rather than anything about a runtime. The two have to agree, and agreeing is a
# property of this archive rather than of whoever configures it, so it is settled here.
# Release unless told otherwise, and said before the runtime below is chosen, because that line
# reads the configuration to decide. install_flash_attn.sh names no build type, and with none
# named MSVC took the debug runtime and the debug iterators with it -- which the linker reports
# against whatever links this archive, as an LNK2038 about MTd_StaticDebug not matching
# MT_StaticRelease, naming a file in this project rather than the one that chose wrongly.
#
# It matters on Linux too, where nothing complains: nvcc optimises device code on its own, and
# the host half of these kernels was being compiled with no -O at all.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "flash_attn build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# And the same answer about the standard. cl reports __cplusplus as 199711L whatever /std it was
# given unless told otherwise; CUTLASS gates its C++17 aliases on that macro, so platform::
# is_unsigned_v goes undeclared and cutlass/exmy_base.h -- which uses it unconditionally -- takes
# every kernel here down with it. Set twice because it has to arrive twice: cl takes the flag
# directly, and nvcc would read a bare /Zc as the name of a file to compile.
#
# The top-level CMakeLists says both of these for itself. They are repeated rather than shared
# because this is a separate project by design -- it has to be, to keep its pinned CUTLASS away
# from the main build -- and a separate project inherits nothing.
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:__cplusplus")

    # M_LOG2E and the rest of the M_ constants are a glibc and BSD convention rather than
    # anything the C or C++ standards promise, and MSVC's <math.h> hands them over only when
    # asked first. src/softmax.h spells M_LOG2E straight out, so ask.
    add_compile_definitions(_USE_MATH_DEFINES)
endif()

file(GLOB FLASH_ATTN_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/flash_fwd*.cu")

add_library(flash_attn STATIC
    ${FLASH_ATTN_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/src/flash.cc")

target_compile_features(flash_attn PRIVATE cuda_std_17)
target_compile_definitions(flash_attn PRIVATE FLASHATTENTION_DISABLE_DROPOUT)
target_compile_options(flash_attn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        # 221 is "floating-point value does not fit in required floating-point type", and every
        # one of them is src/softmax.h comparing against -INFINITY. MSVC spells that macro as
        # ((float)(1e+300 * 1e+300)) -- a value that overflows to infinity on purpose, which is
        # the entire point of it -- and nvcc remarks on the overflow once per kernel per
        # architecture. Sixty of those buried everything worth reading. Suppressed for every
        # host rather than for Windows, so the two builds print the same thing.
        --diag-suppress=221
        --expt-relaxed-constexpr
        --expt-extended-lambda
        --use_fast_math
        -U__CUDA_NO_HALF_OPERATORS__
        -U__CUDA_NO_HALF_CONVERSIONS__
        -U__CUDA_NO_HALF2_OPERATORS__
        -U__CUDA_NO_BFLOAT16_CONVERSIONS__>)
target_include_directories(flash_attn PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${FLASH_ATTN_CUTLASS_DIR}/include"
    ${CUDAToolkit_INCLUDE_DIRS})

set_target_properties(flash_attn PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
