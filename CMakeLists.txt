cmake_minimum_required(VERSION 3.22)

# The macOS a build runs on, unless -DCMAKE_OSX_DEPLOYMENT_TARGET=... says otherwise: 14.0
# (Sonoma), every Apple Silicon machine there is. The Metal backend decides at run time whether the
# neural accelerators' kernels, which want 26.2, can load -- a decision a later target would make at
# build time instead, since __builtin_available is then known to hold.
if(APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "macOS deployment target")
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
# Metal (flint/metal_native), the Vulkan backend's structure in Metal's terms: on by default on
# macOS. Its kernels are compiled with the toolchain Xcode has and carried inside flint, so it
# needs nothing to download and checks for a GPU when the program starts.
if(APPLE)
    set(_libwaifu_metal_default ON)
else()
    set(_libwaifu_metal_default OFF)
endif()
option(WITH_METAL "Metal compute support." ${_libwaifu_metal_default})
# On by default where there is a Vulkan to run on. The build needs no SDK and the binary loads the
# loader itself, so turning it on costs a machine without Vulkan nothing but build time -- about
# half a minute, most of it compiling glslang once. macOS has Metal instead.
if(WIN32 OR (UNIX AND NOT APPLE))
    set(_libwaifu_vulkan_default ON)
else()
    set(_libwaifu_vulkan_default OFF)
endif()
option(WITH_VULKAN "Vulkan compute support." ${_libwaifu_vulkan_default})
option(CUDA_ARCH_NATIVE "Build CUDA kernels only for the GPUs in this machine." OFF)
option(MKL_PREFIX "Prefix for MKL headers and libraries." "/opt/intel/mkl")

# AppleClang implements the OpenMP pragmas but ships neither omp.h nor a runtime, so FindOpenMP
# always comes up empty here. The runtime is linked in static, since a binary that needs a dylib
# beside it is not one you can hand to anybody else -- and it is built from LLVM's source for this
# build's deployment target, since a prebuilt one, Homebrew's, is built for the macOS it was
# installed on, and a binary that carries it runs on no older one. LIBOMP_ROOT names a prefix
# with lib/libomp.a and include/omp.h to use instead, built for whatever it was built for.
set(libwaifu_LIBOMP_FROM_SOURCE OFF)
if(APPLE AND WITH_OPENMP AND NOT DEFINED OpenMP_CXX_FLAGS)
    if(LIBOMP_ROOT)
        if(NOT EXISTS "${LIBOMP_ROOT}/lib/libomp.a")
            message(FATAL_ERROR "LIBOMP_ROOT=${LIBOMP_ROOT} has no lib/libomp.a")
        endif()
        set(LIBOMP_INCLUDE_DIR "${LIBOMP_ROOT}/include")
        set(LIBOMP_LIBRARY_DIR "${LIBOMP_ROOT}/lib")
        set(OpenMP_C_FLAGS "-Xclang -fopenmp -I${LIBOMP_INCLUDE_DIR}")
        set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I${LIBOMP_INCLUDE_DIR}")
        set(OpenMP_C_LIB_NAMES omp)
        set(OpenMP_CXX_LIB_NAMES omp)
        set(OpenMP_omp_LIBRARY "${LIBOMP_LIBRARY_DIR}/libomp.a")
    else()
        # LLVM's OpenMP runtime (Apache 2.0 with LLVM exceptions) on its own: the openmp
        # directory of a release and the CMake modules it reads from beside it, a few megabytes
        # rather than the whole of llvm-project, which configure and build in a dozen seconds.
        include(ExternalProject)
        set(LIBOMP_VERSION "21.1.0")
        set(LIBOMP_RELEASE
            "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LIBOMP_VERSION}")
        set(LIBOMP_SOURCE_ROOT "${CMAKE_BINARY_DIR}/libomp/src")
        set(LIBOMP_BINARY_DIR "${CMAKE_BINARY_DIR}/libomp/build")
        set(LIBOMP_INCLUDE_DIR "${LIBOMP_BINARY_DIR}/runtime/src")
        set(LIBOMP_LIBRARY_DIR "${LIBOMP_BINARY_DIR}/runtime/src")

        ExternalProject_Add(libomp_cmake_modules
            URL             "${LIBOMP_RELEASE}/cmake-${LIBOMP_VERSION}.src.tar.xz"
            URL_HASH        SHA256=528347c84c3571d9d387b825ef8b07c7ad93e9437243c32173838439c3b6028f
            DOWNLOAD_EXTRACT_TIMESTAMP ON
            SOURCE_DIR      "${LIBOMP_SOURCE_ROOT}/cmake"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND   ""
            INSTALL_COMMAND "")
        ExternalProject_Add(libomp_external
            URL             "${LIBOMP_RELEASE}/openmp-${LIBOMP_VERSION}.src.tar.xz"
            URL_HASH        SHA256=60d6c4d2019d546aefc029c9b2c4006dc78419dba4cfd0a376ae025be8ff9778
            DOWNLOAD_EXTRACT_TIMESTAMP ON
            SOURCE_DIR      "${LIBOMP_SOURCE_ROOT}/openmp"
            BINARY_DIR      "${LIBOMP_BINARY_DIR}"
            DEPENDS         libomp_cmake_modules
            CMAKE_ARGS
                -DCMAKE_BUILD_TYPE=Release
                -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
                -DCMAKE_OSX_ARCHITECTURES=arm64
                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                -DOPENMP_STANDALONE_BUILD=ON
                -DLIBOMP_ENABLE_SHARED=OFF
                -DOPENMP_ENABLE_LIBOMPTARGET=OFF
                -DOPENMP_ENABLE_OMPT_TOOLS=OFF
                -DLIBOMP_OMPT_SUPPORT=OFF
                -DLIBOMP_OMPD_SUPPORT=OFF
                -DLIBOMP_USE_HWLOC=OFF
                -DCMAKE_MESSAGE_LOG_LEVEL=ERROR
            BUILD_COMMAND   ${CMAKE_COMMAND} --build <BINARY_DIR> --target omp --parallel
            INSTALL_COMMAND ""
            BUILD_BYPRODUCTS "${LIBOMP_LIBRARY_DIR}/libomp.a")

        # What FindOpenMP would have said, had there been a runtime for it to try at configure
        # time: there is none until the build has made it.
        set(libwaifu_LIBOMP_FROM_SOURCE ON)
        set(OPENMP_FOUND TRUE)
        set(OpenMP_C_FLAGS "-Xclang -fopenmp -I${LIBOMP_INCLUDE_DIR}")
        set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I${LIBOMP_INCLUDE_DIR}")
        set(OpenMP_CXX_INCLUDE_DIRS "${LIBOMP_INCLUDE_DIR}")
        add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
            INTERFACE_LINK_LIBRARIES "${LIBOMP_LIBRARY_DIR}/libomp.a")
        message(STATUS "libwaifu OpenMP: LLVM ${LIBOMP_VERSION}'s, built for macOS "
                       "${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif()
endif()

if(NOT libwaifu_LIBOMP_FROM_SOURCE)
    find_package(OpenMP)
endif()

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
            # One cubin per major version: every kernel is sm_80 class (mma.sync, cp.async), and a
            # cubin runs on any later minor of its major, so 80 covers 8.6 and 8.9 too. Nothing
            # older than 8.0: CUTLASS's Sm80 MMA compiles to an empty assert there, and a 7.x card
            # is better told "no kernel image" than handed a GEMM that returns garbage.
            set(CMAKE_CUDA_ARCHITECTURES 80-real)
            # PTX for the newest non arch-conditional target, so future GPUs can JIT.
            set(_libwaifu_ptx_arch 80-virtual)
            if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.8)
                list(APPEND CMAKE_CUDA_ARCHITECTURES 90-real)
                set(_libwaifu_ptx_arch 90-virtual)
            endif()
            if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
                list(APPEND CMAKE_CUDA_ARCHITECTURES 100-real 120-real)
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

if(WITH_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "WITH_METAL requires macOS")
    endif()
    add_definitions("-DLIBWAIFU_METAL_ENABLED")
endif()

if(WITH_VULKAN)
    # Nothing Vulkan is linked. volk opens the loader with dlopen when the operators are first
    # created, so a build with Vulkan still starts on a machine that has no loader, and simply
    # reports the device as unavailable -- the same bargain a Metal build makes with a Mac that
    # has no Metal GPU it can run on. That is also why nothing here asks for a Vulkan SDK: the headers, volk
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

if(WITH_METAL)
    set(libwaifu_LIBADD ${libwaifu_LIBADD} "-framework Metal" "-framework Foundation")
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
# which is how a parallel build once reached `cargo build` while an archive it named was still
# minutes from being written. Each block below names the project that produces what it asks rustc to link.
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


if(WITH_METAL)
    # The kernels are inside flint already; Metal and Foundation are all there is to link.
    list(APPEND flint_CARGO_FLAGS
        "cargo:rustc-link-lib=framework=Metal"
        "cargo:rustc-link-lib=framework=Foundation")
endif()

if(WITH_OPENMP)
    if(APPLE)
        # Static, so the binary carries its own OpenMP runtime rather than depending on a dylib
        # that only exists on the machine that built it. Where libomp.a is: the one built from
        # source or under LIBOMP_ROOT, as set above -- or, for an OpenMP configuration given by
        # hand with -DOpenMP_CXX_FLAGS, which skips all that, beside the library it names.
        if(LIBOMP_LIBRARY_DIR)
            set(flint_OMP_LIBRARY_DIR "${LIBOMP_LIBRARY_DIR}")
        elseif(OpenMP_omp_LIBRARY)
            get_filename_component(flint_OMP_LIBRARY_DIR "${OpenMP_omp_LIBRARY}" DIRECTORY)
        elseif(LIBOMP_ROOT)
            set(flint_OMP_LIBRARY_DIR "${LIBOMP_ROOT}/lib")
        else()
            message(FATAL_ERROR
              "WITH_OPENMP=ON with OpenMP_CXX_FLAGS given by hand: name the runtime as well, with "
              "-DOpenMP_omp_LIBRARY=<path to libomp.a> or -DLIBOMP_ROOT=<prefix>, so that the Rust "
              "link can find it.")
        endif()
        list(APPEND flint_CARGO_FLAGS
            "cargo:rustc-link-search=native=${flint_OMP_LIBRARY_DIR}"
            "cargo:rustc-link-lib=static=omp")
        if(libwaifu_LIBOMP_FROM_SOURCE)
            list(APPEND flint_CARGO_DEPENDS libomp_external)
        endif()
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
# the build directory was configured, matching what the build scripts fall back to on their own.
#
# CARGO_TARGET_DIR points inside the build directory as well, so that a build writes nothing
# outside it and two build directories configured differently do not fight over one cache. Cargo
# still puts what it builds under a profile directory of its own, so each target copies it up
# beside the C++ executables, where someone who built in `build` expects to find it.
find_program(CARGO_EXECUTABLE cargo REQUIRED)
set(waifu_CARGO_DIR "${CMAKE_BINARY_DIR}/cargo")
set(waifu_CARGO_OUT "${waifu_CARGO_DIR}/release")

# The two libraries, and the command line built on their C APIs. See docs/ffi.md.
#
#   libwaifu               the engine: libwaifu.dylib, libwaifu.so or waifu.dll
#   libwaifu_modelmanager  the catalogue and fetching, with no engine and no flint behind it
#
# Each is copied beside the executables with its header under include/. The two share nothing,
# headers included. On Windows the import library goes too, which is what a program linking the
# DLL is linked against.
if (APPLE)
    set(waifu_LIB_PREFIX "lib")
    set(waifu_LIB_SUFFIX ".dylib")
elseif (WIN32)
    set(waifu_LIB_PREFIX "")
    set(waifu_LIB_SUFFIX ".dll")
else()
    set(waifu_LIB_PREFIX "lib")
    set(waifu_LIB_SUFFIX ".so")
endif()

# waifu_copy_commands(<out> <library> <header dir> <header>): the commands that copy a library
# cargo built, and its header, up beside the executables.
function(waifu_copy_commands out library header_dir header)
    set(commands
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${waifu_CARGO_OUT}/${waifu_LIB_PREFIX}${library}${waifu_LIB_SUFFIX}"
                "${CMAKE_BINARY_DIR}/${waifu_LIB_PREFIX}${library}${waifu_LIB_SUFFIX}")
    if (WIN32)
        list(APPEND commands
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${waifu_CARGO_OUT}/${library}.dll.lib" "${CMAKE_BINARY_DIR}/${library}.dll.lib")
    endif()
    list(APPEND commands
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/include"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/${header_dir}/${header}" "${CMAKE_BINARY_DIR}/include/${header}")
    set(${out} ${commands} PARENT_SCOPE)
endfunction()

# The engine. Its own build script reads flint_link_flags.txt, puts the Metal kernels into it and
# writes the header.
waifu_copy_commands(waifu_ENGINE_COPY waifu waifu/include waifu.h)
add_custom_target(waifu-engine ALL
    COMMAND ${CMAKE_COMMAND} -E env "LIBWAIFU_LIB_DIR=${CMAKE_BINARY_DIR}"
            "CARGO_TARGET_DIR=${waifu_CARGO_DIR}"
            ${CARGO_EXECUTABLE} build -p waifu --lib --release
    ${waifu_ENGINE_COPY}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS flint "${CMAKE_BINARY_DIR}/flint_link_flags.txt" ${flint_CARGO_DEPENDS}
    COMMENT "cargo build -p waifu --lib --release"
    VERBATIM)

# The model manager. It links nothing native, so it needs nothing CMake built.
waifu_copy_commands(waifu_MODELMANAGER_COPY waifu_modelmanager waifu-modelmanager/include waifu_modelmanager.h)
add_custom_target(waifu-modelmanager ALL
    COMMAND ${CMAKE_COMMAND} -E env "CARGO_TARGET_DIR=${waifu_CARGO_DIR}"
            ${CARGO_EXECUTABLE} build -p waifu-modelmanager --lib --release
    ${waifu_MODELMANAGER_COPY}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "cargo build -p waifu-modelmanager --lib --release"
    VERBATIM)

# The command line, `waifu`, which calls the two libraries above through nothing but their C APIs,
# with both linked into it statically: one file, which needs neither library beside it.
set(waifu_CLI "${waifu_CARGO_OUT}/waifu${CMAKE_EXECUTABLE_SUFFIX}")
add_custom_target(waifu-cli ALL
    COMMAND ${CMAKE_COMMAND} -E env "LIBWAIFU_LIB_DIR=${CMAKE_BINARY_DIR}"
            "CARGO_TARGET_DIR=${waifu_CARGO_DIR}"
            ${CARGO_EXECUTABLE} build -p waifu-cli --release
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${waifu_CLI}" "${CMAKE_BINARY_DIR}/waifu${CMAKE_EXECUTABLE_SUFFIX}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "cargo build -p waifu-cli --release"
    VERBATIM)

# One cargo at a time in one target directory: two would wait on each other's lock anyway, and
# this way each reuses what the one before built.
add_dependencies(waifu-modelmanager waifu-engine)
add_dependencies(waifu-cli waifu-engine waifu-modelmanager)

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
