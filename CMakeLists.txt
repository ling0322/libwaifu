cmake_minimum_required(VERSION 3.22)

# Default to macOS 14.0 (Sonoma) so the binary runs on all Apple Silicon machines.
# Override with -DCMAKE_OSX_DEPLOYMENT_TARGET=... if needed.
if(APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.2" CACHE STRING "macOS deployment target")
endif()

project(libwaifu LANGUAGES CXX C)
include(CheckTypeSize)

# Lets an editor read the real compile flags instead of guessing them.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WITH_CUDA "CUDA compute support." OFF)
option(WITH_CUDA_SYNC "Synchronize after each CUDA operator." OFF)
option(WITH_CUDA_MALLOC_ASYNC "Use the CUDA stream-ordered memory allocator." ON)
option(WITH_MKL "MKL support (only for reference)." OFF)
option(WITH_OPENMP "Build with OpenMP." ON)
option(WITH_CUDNN "Build the benchmark's cuDNN Conv2d reference (needs FLINT_ENABLE_CUDNN to run)." OFF)
# FlashAttention costs minutes of nvcc per architecture, so it is opt-in: a CUDA build
# without it still has attention, through the block-wise fallback in operators.cc.
option(WITH_FLASH_ATTN "Link the prebuilt FlashAttention forward kernels (slow to build)." OFF)
option(WITH_MLX "Metal compute support, through MLX." OFF)
# On by default where there is a Vulkan to run on. The build needs no SDK and the binary loads the
# loader itself, so turning it on costs a machine without Vulkan nothing but build time -- about
# half a minute, most of it compiling glslang once. macOS has Metal instead, through MLX.
if(WIN32 OR (UNIX AND NOT APPLE))
    set(_libwaifu_vulkan_default ON)
else()
    set(_libwaifu_vulkan_default OFF)
endif()
option(WITH_VULKAN "Vulkan compute support." ${_libwaifu_vulkan_default})
option(CUDA_ARCH_NATIVE "Build CUDA kernels only for the GPUs in this machine." OFF)
option(MKL_PREFIX "Prefix for MKL headers and libraries." "/opt/intel/mkl")

# AppleClang implements the OpenMP pragmas but ships neither omp.h nor a runtime, so
# FindOpenMP always comes up empty here. Point it at Homebrew's libomp, and at the static
# archive specifically: a binary that needs a Homebrew dylib beside it is not one you can
# hand to anybody else.
if(APPLE AND WITH_OPENMP AND NOT DEFINED OpenMP_CXX_FLAGS)
    if(NOT LIBOMP_ROOT)
        find_program(BREW_EXECUTABLE brew)
        if(BREW_EXECUTABLE)
            execute_process(COMMAND "${BREW_EXECUTABLE}" --prefix libomp
                            OUTPUT_VARIABLE LIBOMP_ROOT
                            OUTPUT_STRIP_TRAILING_WHITESPACE
                            ERROR_QUIET)
        endif()
    endif()
    if(EXISTS "${LIBOMP_ROOT}/lib/libomp.a")
        set(OpenMP_C_FLAGS "-Xclang -fopenmp -I${LIBOMP_ROOT}/include")
        set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I${LIBOMP_ROOT}/include")
        set(OpenMP_C_LIB_NAMES omp)
        set(OpenMP_CXX_LIB_NAMES omp)
        set(OpenMP_omp_LIBRARY "${LIBOMP_ROOT}/lib/libomp.a")
    else()
        message(FATAL_ERROR
          "WITH_OPENMP=ON needs libomp on macOS: run 'brew install libomp', or point "
          "LIBOMP_ROOT at a prefix that has lib/libomp.a, or configure with "
          "-DWITH_OPENMP=OFF.")
    endif()
endif()

find_package(OpenMP)

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

if(WITH_CUDA)
    add_definitions("-DLIBWAIFU_CUDA_ENABLED")

    # CUTLASS is not optional: it is the only Conv2d the library has, and the GEMM the whole
    # model runs on. It is a header library, but not a vendored one -- install_cutlass.sh in
    # third_party clones it.
    set(CUTLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass")
    if(NOT EXISTS ${CUTLASS_ROOT}/include/cutlass/blas3.h)
        message(FATAL_ERROR
          "invalid CUTLASS_ROOT \"${CUTLASS_ROOT}\": unable to find file "
          "${CUTLASS_ROOT}/include/cutlass/blas3.h, please "
          "call install_cutlass.sh in third_party first.")
    endif()
    include_directories(${CUTLASS_ROOT}/include)

    if(WITH_CUDA_SYNC)
        add_compile_definitions(LIBWAIFU_CUDA_SYNC_ENABLED)
    endif()
    if(WITH_CUDA_MALLOC_ASYNC)
        add_compile_definitions(LIBWAIFU_CUDA_MALLOC_ASYNC_ENABLED)
    endif()
    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(CUDA_ARCH_NATIVE)
            set(CMAKE_CUDA_ARCHITECTURES native)
        else()
            # Volta through Ada. Kernels needing newer hardware are guarded by __CUDA_ARCH__.
            set(CMAKE_CUDA_ARCHITECTURES 70-real 75-real 80-real 86-real 89-real)
            # PTX for the newest non arch-conditional target, so future GPUs can JIT.
            set(_libwaifu_ptx_arch 89-virtual)
            if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.8)
                list(APPEND CMAKE_CUDA_ARCHITECTURES 90-real)
                set(_libwaifu_ptx_arch 90-virtual)
            endif()
            if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
                # 120a (not 120): the mxfp4 kernels use sm_120a-only PTX.
                list(APPEND CMAKE_CUDA_ARCHITECTURES 100-real 120a-real)
                set(_libwaifu_ptx_arch 100-virtual)
            endif()
            list(APPEND CMAKE_CUDA_ARCHITECTURES ${_libwaifu_ptx_arch})
        endif()
    endif()
    message(STATUS "libwaifu CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    # Two of these are kept away from MSVC, for reasons that have nothing to do with each other.
    #
    # CUTE_ARCH_TCGEN05_TMEM_ENABLED is CUTLASS's own, and CUTLASS raises it only inside a device
    # pass for an sm_100a-class target -- its condition reads __CUDA_ARCH__. Setting it on the
    # command line raises it for the host pass as well, and cute/arch/copy_sm100.hpp then shows
    # the host compiler a GCC inline-asm statement. gcc and clang parse it and move on; nvcc's
    # frontend in MSVC mode stops at `expected a "("`. Windows is therefore left to CUTLASS's
    # detection, which means the tcgen05 paths there follow the architectures actually asked for
    # rather than this flag.
    #
    # -O3 is nvcc's, but nvcc hands it to the host compiler too, and on MSVC it arrives as /O2.
    # CMake tests a new compiler in the Debug configuration rather than this build's own, and
    # Debug there carries /RTC1, which /O2 is not allowed to accompany -- so enable_language()
    # below fails before the project ever configures. MSVC's Release flags already say /O2, so
    # the only host left to tell is the one whose CMAKE_CUDA_FLAGS_RELEASE does not.
    if(NOT MSVC)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUTE_ARCH_TCGEN05_TMEM_ENABLED -O3")
    endif()
    enable_language(CUDA)
endif()

if(WITH_CUDNN)
    if(NOT WITH_CUDA)
        message(FATAL_ERROR "WITH_CUDNN requires WITH_CUDA=ON")
    endif()

    # Only the headers are needed to build: the library itself is loaded by name at run time, the
    # same way cuBLAS is, so a build with cuDNN still runs on a machine without it. And it is only
    # looked for where FLINT_ENABLE_CUDNN asks, so a build with cuDNN is not a run with it either.
    if(NOT CUDNN_ROOT)
        message(FATAL_ERROR
          "WITH_CUDNN=ON needs CUDNN_ROOT set to a directory holding include/cudnn.h. "
          "A pip install of nvidia-cudnn-cu12 puts one under "
          "<venv>/lib/pythonX.Y/site-packages/nvidia/cudnn.")
    endif()
    if(NOT EXISTS "${CUDNN_ROOT}/include/cudnn.h")
        message(FATAL_ERROR "unable to find \"${CUDNN_ROOT}/include/cudnn.h\"")
    endif()

    add_definitions("-DLIBWAIFU_CUDNN_ENABLED")
    include_directories("${CUDNN_ROOT}/include")
endif()

if(WITH_MKL)
    add_definitions("-DMKL_ENABLED")
    include_directories(${MKL_PREFIX}/include)
    link_directories(${MKL_PREFIX}/lib/intel64)
    if (NOT EXISTS ${MKL_PREFIX}/include/mkl.h)
        message(FATAL_ERROR 
        "invalid MKL_PREFIX \"${MKL_PREFIX}\": unable to find file "
        "${MKL_PREFIX}/include/mkl.h, please "
        "enusre MKL_PREFIX contains the MKL \"include\" directory.")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#add_compile_options(-fsanitize=address)
#add_link_options(-fsanitize=address)

if(WIN32)
    # Named, rather than written out as /D flags. nvcc reads an argument that begins with a
    # slash as a file name rather than an option, so the MSVC spelling reached it as three more
    # inputs and it refused the compile -- "a single input file is required for a non-link
    # phase". cl accepts either, which is why only the .cu files minded. Letting CMake write the
    # definitions gives each language the spelling it actually parses.
    #
    # _USE_MATH_DEFINES is what makes MSVC's <math.h> part with M_LOG2E and the other M_
    # constants: they are a glibc and BSD convention rather than anything standard, and
    # flint/cuda/flash_attn.cc uses one. That file is compiled only when WITH_FLASH_ATTN is on,
    # so CI -- which builds without it -- would never have said so.
    add_compile_definitions(
        _CRT_SECURE_NO_WARNINGS
        _USE_MATH_DEFINES
        CATCH_AMALGAMATED_CUSTOM_MAIN
        CATCH_CONFIG_PREFIX_ALL)
endif(WIN32)
if(MSVC)
    # cl answers __cplusplus with 199711L no matter what /std it was given, unless it is asked
    # for the truth. CUTLASS gates its C++17 aliases on that macro, so platform::is_unsigned_v
    # was never declared and cutlass/exmy_base.h -- which uses it unconditionally -- could not
    # compile. Three kernels here include it.
    #
    # Said twice because it has to arrive twice: cl takes the flag directly, and nvcc would read
    # a bare /Zc as the name of a file to compile, so its host compiler is handed it explicitly.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:__cplusplus")
endif(MSVC)
if(UNIX)
    add_definitions( "-DCATCH_AMALGAMATED_CUSTOM_MAIN -DCATCH_CONFIG_PREFIX_ALL -D_FILE_OFFSET_BITS=64" )
    set(CMAKE_CXX_FLAGS "-O3 -g")
    set(CMAKE_C_FLAGS "-O3 -g")
endif(UNIX)


# printStackTrace() is the only thing that wants a stack walker, and each host hands over a
# different one. macOS already ships libunwind in libSystem, and its header covers the handful
# of unw_* calls that code makes, so there is nothing to build and nothing to link there.
# Windows has DbgHelp, which likewise comes with the OS and only needs naming on the link line.
# Linux gets the NonGNU libunwind, built from its release tarball by the external project below
# -- that one is ELF-only, which is why neither of the others can be given it.
set(STACKTRACE_LIBRARIES "")
if(WIN32)
  set(STACKTRACE_LIBRARIES dbghelp)
endif()
if(UNIX AND NOT APPLE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(UNWIND_ARCH_LIB_NAME unwind-x86_64)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(UNWIND_ARCH_LIB_NAME unwind-aarch64)
  else()
    message(FATAL_ERROR "Unknown arch '${CMAKE_SYSTEM_PROCESSOR}', no libunwind to build for it.")
  endif()

  include(ExternalProject)
  include(ProcessorCount)

  set(UNWIND_VERSION "1.6.2")
  # The tarball is cached in the source tree rather than the build directory, so a fresh
  # build directory unpacks what is already there instead of fetching it again.
  set(UNWIND_DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libunwind")
  set(UNWIND_PREFIX "${CMAKE_BINARY_DIR}/libunwind")
  set(UNWIND_INCLUDE_DIR "${UNWIND_PREFIX}/include")
  set(UNWIND_LIBRARY_DIR "${UNWIND_PREFIX}/lib")
  set(UNWIND_LIB "${UNWIND_LIBRARY_DIR}/libunwind.a")
  set(UNWIND_ARCH_LIB "${UNWIND_LIBRARY_DIR}/lib${UNWIND_ARCH_LIB_NAME}.a")

  ProcessorCount(UNWIND_BUILD_JOBS)
  if(UNWIND_BUILD_JOBS EQUAL 0)
    set(UNWIND_BUILD_JOBS 4)
  endif()

  # An autotools project, so this is a configure/make/make install of its own rather than
  # anything CMake can drive directly. -fPIC because the archives end up inside libflint.a,
  # which rustc links into a position-independent executable.
  ExternalProject_Add(unwind_external
      URL               "https://github.com/libunwind/libunwind/releases/download/v${UNWIND_VERSION}/libunwind-${UNWIND_VERSION}.tar.gz"
      URL_HASH          SHA256=4a6aec666991fb45d0889c44aede8ad6eb108071c3554fcdff671f9c94794976
      # Keep the tarball's own mtimes, which is also what CMake does when CMP0135 is unset --
      # saying so here is what stops it warning that the choice was never made. An autotools
      # tree wants the released order kept, with configure newer than configure.ac, or make
      # goes looking for autoconf; and the staleness the policy guards against cannot happen
      # here, since the version is in both the URL and SOURCE_DIR, so a bump extracts into a
      # directory of its own.
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      DOWNLOAD_DIR      "${UNWIND_DOWNLOAD_DIR}"
      SOURCE_DIR        "${UNWIND_DOWNLOAD_DIR}/libunwind-${UNWIND_VERSION}"
      BINARY_DIR        "${CMAKE_BINARY_DIR}/libunwind-build"
      INSTALL_DIR       "${UNWIND_PREFIX}"
      CONFIGURE_COMMAND <SOURCE_DIR>/configure
                            --prefix=<INSTALL_DIR>
                            --disable-minidebuginfo
                            --enable-static
                            --disable-shared
                            "CC=${CMAKE_C_COMPILER}"
                            "CFLAGS=-O2 -fPIC"
                            "CXXFLAGS=-O2 -fPIC"
      BUILD_COMMAND     make "-j${UNWIND_BUILD_JOBS}"
      INSTALL_COMMAND   make install
      BUILD_BYPRODUCTS  "${UNWIND_LIB}" "${UNWIND_ARCH_LIB}"
      LOG_DOWNLOAD      ON
      LOG_CONFIGURE     ON
      LOG_BUILD         ON
      LOG_INSTALL       ON
      LOG_OUTPUT_ON_FAILURE ON)

  # The external project runs at build time, but CMake validates include paths at configure
  # time. Create the directory so the check passes; `make install` fills it in.
  file(MAKE_DIRECTORY "${UNWIND_INCLUDE_DIR}")

  add_library(unwind_arch STATIC IMPORTED)
  set_target_properties(unwind_arch PROPERTIES IMPORTED_LOCATION "${UNWIND_ARCH_LIB}")

  add_library(unwind STATIC IMPORTED)
  set_target_properties(unwind PROPERTIES
      IMPORTED_LOCATION "${UNWIND_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${UNWIND_INCLUDE_DIR}")

  # The arch-specific half first: it draws on the generic one.
  set(STACKTRACE_LIBRARIES unwind_arch unwind)
endif()

message("CMAKE_HOST_SYSTEM_PROCESSOR=" ${CMAKE_HOST_SYSTEM_PROCESSOR})

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(LIBWAIFU_KERNEL_X86_64 ON)
endif()
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)")
    set(LIBWAIFU_KERNEL_AARCH64 ON)
endif()

check_type_size(__fp16 FP16 BUILTIN_TYPES_ONLY LANGUAGE CXX)
if(HAVE_FP16)
    message("__fp16 exists.")
    add_compile_definitions(LIBWAIFU_HAVE_FP16)
endif()

if(WITH_FLASH_ATTN)
    if(NOT WITH_CUDA)
        message(FATAL_ERROR "WITH_FLASH_ATTN requires WITH_CUDA=ON")
    endif()

    set(FLASH_ATTN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/flash-attention")
    set(FLASH_ATTN_LIB
        "${FLASH_ATTN_ROOT}/build/${CMAKE_STATIC_LIBRARY_PREFIX}flash_attn${CMAKE_STATIC_LIBRARY_SUFFIX}")
    if(NOT EXISTS "${FLASH_ATTN_LIB}")
        message(FATAL_ERROR
          "unable to find \"${FLASH_ATTN_LIB}\", please "
          "call install_flash_attn.sh in third_party first.")
    endif()

    add_definitions("-DLIBWAIFU_FLASH_ATTN_ENABLED")
    add_library(flash_attn STATIC IMPORTED)
    set_target_properties(flash_attn PROPERTIES
        IMPORTED_LOCATION "${FLASH_ATTN_LIB}"
        INTERFACE_COMPILE_DEFINITIONS FLASHATTENTION_DISABLE_DROPOUT
        INTERFACE_INCLUDE_DIRECTORIES "${FLASH_ATTN_ROOT}/csrc/flash_attn/src")
endif()

if(WITH_MLX)
    if(NOT APPLE)
        message(FATAL_ERROR "WITH_MLX requires macOS: MLX's Metal backend has no other host")
    endif()

    include(ExternalProject)

    set(MLX_VERSION "v0.32.2")
    set(MLX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mlx")
    set(MLX_BUILD_ROOT "${CMAKE_BINARY_DIR}/mlx-build")
    set(MLX_LIB "${MLX_BUILD_ROOT}/libmlx.a")
    set(MLX_METALLIB "${MLX_BUILD_ROOT}/mlx/backend/metal/kernels/mlx.metallib")
    set(MLX_PATCH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mlx_metallib_from_memory.patch")

    ExternalProject_Add(mlx_external
        GIT_REPOSITORY  https://github.com/ml-explore/mlx.git
        GIT_TAG         "${MLX_VERSION}"
        GIT_SHALLOW     ON
        SOURCE_DIR      "${MLX_SOURCE_DIR}"
        BINARY_DIR      "${MLX_BUILD_ROOT}"
        PATCH_COMMAND   ${CMAKE_COMMAND}
                            -DMLX_DIR=${MLX_SOURCE_DIR}
                            -DPATCH_FILE=${MLX_PATCH}
                            -P ${CMAKE_CURRENT_SOURCE_DIR}/third_party/apply_mlx_patch.cmake
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DMLX_BUILD_METAL=ON
            -DMLX_BUILD_CPU=ON
            -DMLX_BUILD_TESTS=OFF
            -DMLX_BUILD_EXAMPLES=OFF
            -DMLX_BUILD_BENCHMARKS=OFF
            -DMLX_BUILD_PYTHON_BINDINGS=OFF
            -DMLX_BUILD_PYTHON_STUBS=OFF
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -DCMAKE_MESSAGE_LOG_LEVEL=ERROR
        BUILD_COMMAND   ${CMAKE_COMMAND} --build <BINARY_DIR> --target mlx --parallel
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS "${MLX_LIB}" "${MLX_BUILD_ROOT}/mlx/io/libgguflib.a"
    )

    add_definitions("-DLIBWAIFU_MLX_ENABLED")

    add_library(mlx_gguflib STATIC IMPORTED)
    set_target_properties(mlx_gguflib PROPERTIES
        IMPORTED_LOCATION "${MLX_BUILD_ROOT}/mlx/io/libgguflib.a")

    set(MLX_INCDIRS
        "${MLX_SOURCE_DIR}"
        "${MLX_BUILD_ROOT}/_deps/fmt-src/include"
        "${MLX_BUILD_ROOT}/_deps/json-src/single_include"
        "${MLX_BUILD_ROOT}/_deps/metal_cpp-src")

    # ExternalProject runs at build time, but CMake validates include paths at
    # configure time.  Create the directories so the check passes; they will be
    # populated when mlx_external fetches its dependencies.
    foreach(_dir ${MLX_INCDIRS})
        file(MAKE_DIRECTORY "${_dir}")
    endforeach()

    add_library(mlx STATIC IMPORTED)
    set_target_properties(mlx PROPERTIES
        IMPORTED_LOCATION "${MLX_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${MLX_INCDIRS}"
        INTERFACE_LINK_LIBRARIES
            "mlx_gguflib;-framework Metal;-framework Foundation;-framework QuartzCore;-framework Accelerate")
    add_dependencies(mlx mlx_external)

    set(MLX_METALLIB_LINK_OPTION "-Wl,-sectcreate,__TEXT,__metallib,${MLX_METALLIB}")
    set_property(TARGET mlx APPEND PROPERTY
        INTERFACE_LINK_OPTIONS "${MLX_METALLIB_LINK_OPTION}")
endif()

if(WITH_VULKAN)
    # Nothing Vulkan is linked. volk opens the loader with dlopen when the operators are first
    # created, so a build with Vulkan still starts on a machine that has no loader, and simply
    # reports the device as unavailable -- the same bargain an MLX build makes with a Mac that
    # has no Metal GPU. That is also why nothing here asks for a Vulkan SDK: the headers, volk
    # and the memory allocator are fetched source, and the one tool, the GLSL compiler, is built.
    include(FetchContent)
    include(ExternalProject)

    set(VULKAN_SDK_TAG "vulkan-sdk-1.4.350.0")
    set(VULKAN_THIRD_PARTY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vulkan")

    # SOURCE_SUBDIR names a directory none of them has, which is what keeps FetchContent from
    # adding their own CMake projects: only the headers and volk.c are wanted.
    FetchContent_Declare(vulkan_headers
        GIT_REPOSITORY  https://github.com/KhronosGroup/Vulkan-Headers.git
        GIT_TAG         "${VULKAN_SDK_TAG}"
        GIT_SHALLOW     TRUE
        SOURCE_DIR      "${VULKAN_THIRD_PARTY}/Vulkan-Headers"
        SOURCE_SUBDIR   "_not_a_project")
    FetchContent_Declare(volk
        GIT_REPOSITORY  https://github.com/zeux/volk.git
        GIT_TAG         "${VULKAN_SDK_TAG}"
        GIT_SHALLOW     TRUE
        SOURCE_DIR      "${VULKAN_THIRD_PARTY}/volk"
        SOURCE_SUBDIR   "_not_a_project")
    FetchContent_Declare(vma
        GIT_REPOSITORY  https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG         v3.3.0
        GIT_SHALLOW     TRUE
        SOURCE_DIR      "${VULKAN_THIRD_PARTY}/VulkanMemoryAllocator"
        SOURCE_SUBDIR   "_not_a_project")
    FetchContent_MakeAvailable(vulkan_headers volk vma)

    set(VULKAN_INCDIRS
        "${VULKAN_THIRD_PARTY}/Vulkan-Headers/include"
        "${VULKAN_THIRD_PARTY}/volk"
        "${VULKAN_THIRD_PARTY}/VulkanMemoryAllocator/include")

    # The shaders are GLSL compiled to SPIR-V at build time and embedded in the library. A
    # glslangValidator on the PATH is used as it is; otherwise glslang's own is built, without
    # its optimizer -- that would need SPIRV-Tools, and every driver optimizes the SPIR-V it is
    # handed anyway.
    find_program(GLSLANG_VALIDATOR NAMES glslangValidator glslang)
    set(GLSLANG_DEPENDS "")
    if(NOT GLSLANG_VALIDATOR)
        set(GLSLANG_BUILD_ROOT "${CMAKE_BINARY_DIR}/glslang-build")
        # Always Release, and named as such: a multi-config generator -- Visual Studio, say -- puts
        # the binary in a directory of the configuration's name, and builds Debug when not told.
        get_property(_libwaifu_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
        if(_libwaifu_multi_config)
            set(GLSLANG_VALIDATOR
                "${GLSLANG_BUILD_ROOT}/StandAlone/Release/glslang${CMAKE_EXECUTABLE_SUFFIX}")
        else()
            set(GLSLANG_VALIDATOR
                "${GLSLANG_BUILD_ROOT}/StandAlone/glslang${CMAKE_EXECUTABLE_SUFFIX}")
        endif()
        ExternalProject_Add(glslang_external
            GIT_REPOSITORY  https://github.com/KhronosGroup/glslang.git
            GIT_TAG         "${VULKAN_SDK_TAG}"
            GIT_SHALLOW     TRUE
            SOURCE_DIR      "${VULKAN_THIRD_PARTY}/glslang"
            BINARY_DIR      "${GLSLANG_BUILD_ROOT}"
            CMAKE_ARGS
                -DCMAKE_BUILD_TYPE=Release
                -DENABLE_OPT=OFF
                -DENABLE_HLSL=OFF
                -DGLSLANG_TESTS=OFF
                -DGLSLANG_ENABLE_INSTALL=OFF
                -DBUILD_EXTERNAL=OFF
                -DENABLE_GLSLANG_BINARIES=ON
            BUILD_COMMAND   ${CMAKE_COMMAND} --build <BINARY_DIR> --target glslang-standalone
                                --config Release --parallel
            INSTALL_COMMAND ""
            BUILD_BYPRODUCTS "${GLSLANG_VALIDATOR}"
            LOG_CONFIGURE   ON
            LOG_BUILD       ON
            LOG_OUTPUT_ON_FAILURE ON)
        set(GLSLANG_DEPENDS glslang_external)
    endif()
    message(STATUS "libwaifu GLSL compiler: ${GLSLANG_VALIDATOR}")

    add_definitions("-DLIBWAIFU_VULKAN_ENABLED")
endif()

add_subdirectory("flint")

set(libwaifu_LIBADD lutil flint_static ${CMAKE_DL_LIBS} ${STACKTRACE_LIBRARIES})

if(WITH_FLASH_ATTN)
    set(libwaifu_LIBADD ${libwaifu_LIBADD} flash_attn)
endif()

if(WITH_MLX)
    set(libwaifu_LIBADD ${libwaifu_LIBADD} mlx)
endif()

if(WITH_OPENMP)
    set(libwaifu_LIBADD ${libwaifu_LIBADD} OpenMP::OpenMP_CXX)
endif()

if(WITH_CUDA)
    # cudart is the only CUDA library this links, and it is linked static, so the binary asks the
    # loader for nothing but the driver. cuRAND used to be here too and was the one library with
    # no static form on Windows: the import of curand64_*.dll is resolved before main() runs, so a
    # machine with a driver and no CUDA toolkit could not start the binary at all. flint/cuda/
    # rand.cu carries the one generator that was used from it, so there is nothing left to name.
    set(libwaifu_LIBADD ${libwaifu_LIBADD} CUDA::cudart_static)
endif()

# libflint.a is what the Rust `flint` crate links against. It bundles every object this project
# compiles -- the kernels and the lutil helpers they use -- so a consumer needs exactly one
# archive from the CMake build. The model runtime that used to sit on top of it now lives in the
# Rust `waifu` crate.
add_library(flint STATIC $<TARGET_OBJECTS:flint_static> $<TARGET_OBJECTS:lutil>)

# A static archive carries no record of what it still needs, and Cargo cannot read CMake's
# dependency graph, so write the rest of the link line out as cargo directives for the flint
# crate's build script to echo. Order matters: dependents come before what they depend on.
set(flint_CARGO_FLAGS
    "cargo:rustc-link-search=native=${CMAKE_BINARY_DIR}"
    "cargo:rustc-link-lib=static=flint")

# Some of those archives are built by an external project during this build, and rustc needs them
# on disk by the time it links. Depending on `flint` does not arrange that: compiling flint only
# reads their headers, so it finishes long before the archive beside those headers is written --
# which is how a parallel build reached `cargo build` while libmlx.a was still several minutes of
# Metal kernels away. Each block below names the project that produces what it asks rustc to link.
set(flint_CARGO_DEPENDS "")

if(UNIX AND NOT APPLE)
    list(APPEND flint_CARGO_FLAGS
        "cargo:rustc-link-search=native=${UNWIND_LIBRARY_DIR}"
        "cargo:rustc-link-lib=static=${UNWIND_ARCH_LIB_NAME}"
        "cargo:rustc-link-lib=static=unwind")
    list(APPEND flint_CARGO_DEPENDS unwind_external)
elseif(WIN32)
    list(APPEND flint_CARGO_FLAGS "cargo:rustc-link-lib=dylib=dbghelp")
endif()

if(WITH_FLASH_ATTN)
    get_filename_component(flint_FLASH_ATTN_DIR "${FLASH_ATTN_LIB}" DIRECTORY)
    list(APPEND flint_CARGO_FLAGS
        "cargo:rustc-link-search=native=${flint_FLASH_ATTN_DIR}"
        "cargo:rustc-link-lib=static=flash_attn")
endif()

if(WITH_CUDA)
    list(APPEND flint_CARGO_FLAGS
        "cargo:rustc-link-search=native=${CUDAToolkit_LIBRARY_DIR}")
    list(APPEND flint_CARGO_FLAGS "cargo:rustc-link-lib=static=cudart_static")

    # culibos is the shim the static CUDA runtime calls into for threads and files. It ships with
    # the static Linux libraries and exists nowhere else, so it is named only where the toolkit
    # actually has it -- which is the thing that varies -- and last, since cudart_static above
    # draws on it. Windows needs no counterpart: its static runtime calls the C runtime directly.
    if(TARGET CUDA::culibos)
        list(APPEND flint_CARGO_FLAGS "cargo:rustc-link-lib=static=culibos")
    endif()
endif()

if(WITH_MLX)
    # The metallib section is a property of the final link, which for the `waifu` binary is
    # rustc's, so it has to be passed through as a link arg rather than inherited from the
    # `mlx` target the way the CMake executables inherit it.
    list(APPEND flint_CARGO_FLAGS
        "cargo:rustc-link-search=native=${MLX_BUILD_ROOT}"
        "cargo:rustc-link-lib=static=mlx"
        "cargo:rustc-link-search=native=${MLX_BUILD_ROOT}/mlx/io"
        "cargo:rustc-link-lib=static=gguflib"
        "cargo:rustc-link-lib=framework=Metal"
        "cargo:rustc-link-lib=framework=Foundation"
        "cargo:rustc-link-lib=framework=QuartzCore"
        "cargo:rustc-link-lib=framework=Accelerate"
        "cargo:rustc-link-arg=${MLX_METALLIB_LINK_OPTION}")
    list(APPEND flint_CARGO_DEPENDS mlx_external)
endif()

if(WITH_OPENMP)
    if(APPLE)
        # Static, so the binary carries its own OpenMP runtime rather than depending on a
        # Homebrew dylib that only exists on the machine that built it.
        list(APPEND flint_CARGO_FLAGS
            "cargo:rustc-link-search=native=${LIBOMP_ROOT}/lib"
            "cargo:rustc-link-lib=static=omp")
    else()
        foreach(flint_OMP_LIB ${OpenMP_CXX_LIB_NAMES})
            list(APPEND flint_CARGO_FLAGS "cargo:rustc-link-lib=dylib=${flint_OMP_LIB}")
        endforeach()
    endif()
endif()

# The C++ runtime last: everything above draws on it.
if(APPLE)
    # libSystem already provides dl, rt and m, and the C++ runtime here is libc++.
    list(APPEND flint_CARGO_FLAGS "cargo:rustc-link-lib=dylib=c++")
elseif(WIN32)
    # MSVC writes its own runtime into the object files it produces, as the /MT that
    # CMAKE_MSVC_RUNTIME_LIBRARY asks for, so the linker finds it without being told. None of
    # dl, rt or m exists here either -- the C runtime covers all three.
else()
    list(APPEND flint_CARGO_FLAGS
        "cargo:rustc-link-lib=dylib=stdc++"
        "cargo:rustc-link-lib=dylib=dl"
        "cargo:rustc-link-lib=dylib=rt"
        "cargo:rustc-link-lib=dylib=m")
endif()

string(REPLACE ";" "\n" flint_CARGO_FLAGS_TEXT "${flint_CARGO_FLAGS}")
file(GENERATE
    OUTPUT "${CMAKE_BINARY_DIR}/flint_link_flags.txt"
    CONTENT "${flint_CARGO_FLAGS_TEXT}\n")

# CMake drives the Rust build too, so a plain `cmake --build build` is the whole build: no
# separate `cargo build` step. LIBWAIFU_LIB_DIR is set explicitly so this works regardless of where
# the build directory was configured, matching what waifu/build.rs falls back to on its own.
#
# CARGO_TARGET_DIR points inside the build directory as well, so that a build writes nothing
# outside it and two build directories configured differently do not fight over one cache. Cargo
# still puts the binary under a profile directory of its own, so the last step copies it up beside
# the C++ executables, where someone who built in `build` expects to find it.
find_program(CARGO_EXECUTABLE cargo REQUIRED)
set(waifu_CARGO_DIR "${CMAKE_BINARY_DIR}/cargo")
set(waifu_CLI "${waifu_CARGO_DIR}/release/waifu${CMAKE_EXECUTABLE_SUFFIX}")
# `--features cli` is what builds the `waifu` binary: it is off by default so that using the crate
# as a library does not drag in a terminal library.
add_custom_target(waifu-cli ALL
    COMMAND ${CMAKE_COMMAND} -E env "LIBWAIFU_LIB_DIR=${CMAKE_BINARY_DIR}"
            "CARGO_TARGET_DIR=${waifu_CARGO_DIR}"
            ${CARGO_EXECUTABLE} build -p waifu --features cli --release
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${waifu_CLI}" "${CMAKE_BINARY_DIR}/waifu${CMAKE_EXECUTABLE_SUFFIX}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS flint "${CMAKE_BINARY_DIR}/flint_link_flags.txt" ${flint_CARGO_DEPENDS}
    COMMENT "cargo build -p waifu --features cli --release"
    VERBATIM)

set(unittest_LIBADD
    flint_test
    lutil_test
    ${libwaifu_LIBADD}
    catch2)
if (WITH_MKL)
    set(unittest_LIBADD
        ${unittest_LIBADD}
        ${LIBDWARF_LIB}
        mkl_intel_lp64
        mkl_intel_thread
        mkl_core
        iomp5)
endif()

add_library(catch2 STATIC  "third_party/catch2/catch_amalgamated.cpp")

add_executable(unittest "flint/bin/unittest.cc")
target_include_directories(unittest PRIVATE "." "flint" "third_party")

target_link_libraries(unittest ${unittest_LIBADD})

add_executable(benchmark "flint/bin/benchmark.cc")
target_include_directories(benchmark PRIVATE "." "flint" "third_party")
target_link_libraries(benchmark flint_benchmark ${libwaifu_LIBADD})

enable_testing()
add_test(NAME unittest COMMAND $<TARGET_FILE:unittest>)
