cmake_minimum_required(VERSION 3.22)

# Match the deployment target to the running system so that object files built by vendored
# libraries (MLX in particular) do not trigger linker warnings about a newer macOS version.
if(APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
    execute_process(COMMAND sw_vers -productVersion
                    OUTPUT_VARIABLE _macos_ver OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "${_macos_ver}" CACHE STRING "macOS deployment target")
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
option(WITH_CUTLASS "build MatMul operators with CUTLASS." OFF)
option(WITH_CUDNN "build the Conv2d operator with cuDNN." OFF)
option(WITH_FLASH_ATTN "Link the prebuilt FlashAttention forward kernels." ${WITH_CUDA})
option(WITH_MLX "Metal compute support, through MLX." OFF)
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
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUTE_ARCH_TCGEN05_TMEM_ENABLED -O3")
    enable_language(CUDA)
endif()

if(WITH_CUTLASS)
    add_definitions("-DLIBWAIFU_CUTLASS_ENABLED")
    include_directories(${CUTLASS_ROOT}/include)

    set(CUTLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass")
    if (NOT EXISTS ${CUTLASS_ROOT}/include/cutlass/blas3.h)
        message(FATAL_ERROR 
          "invalid CUTLASS_ROOT \"${CUTLASS_ROOT}\": unable to find file "
          "${CUTLASS_ROOT}/include/cutlass/blas3.h, please "
          "call install_cutlass.sh in third_party first.")
    endif()
endif()

if(WITH_CUDNN)
    if(NOT WITH_CUDA)
        message(FATAL_ERROR "WITH_CUDNN requires WITH_CUDA=ON")
    endif()

    # Only the headers are needed to build: the library itself is loaded by name at run time, the
    # same way cuBLAS is, so a build with cuDNN still runs on a machine without it.
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
    add_definitions( "/D_CRT_SECURE_NO_WARNINGS /DCATCH_AMALGAMATED_CUSTOM_MAIN /DCATCH_CONFIG_PREFIX_ALL" )
endif(WIN32)
if(UNIX)
    add_definitions( "-DCATCH_AMALGAMATED_CUSTOM_MAIN -DCATCH_CONFIG_PREFIX_ALL -D_FILE_OFFSET_BITS=64" )
    set(CMAKE_CXX_FLAGS "-O3 -g")
    set(CMAKE_C_FLAGS "-O3 -g")
endif(UNIX)


# printStackTrace() is the only thing that wants libunwind. macOS already ships one in
# libSystem, and its header covers the handful of unw_* calls that code makes, so there is
# nothing to find and nothing to link there. Everywhere else it comes from the vendored
# NonGNU build -- which is ELF-only and cannot be built on Darwin at all, hence the split.
set(UNWIND_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libunwind/libunwind-1.6.2")
set(UNWIND_ARCH_LIB_NAME "")
if(NOT APPLE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(UNWIND_ARCH_LIB_NAME unwind-x86_64)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(UNWIND_ARCH_LIB_NAME unwind-aarch64)
  else()
    message(FATAL_ERROR "Unknown arch '${CMAKE_SYSTEM_PROCESSOR}', will try generic libunwind only.")
  endif()
  find_library(UNWIND_LIB NAMES libunwind.a REQUIRED PATHS "${UNWIND_ROOT}/build/src/.libs/"
               NO_DEFAULT_PATH)
  find_library(UNWIND_ARCH_LIB NAMES lib${UNWIND_ARCH_LIB_NAME}.a REQUIRED PATHS "${UNWIND_ROOT}/build/src/.libs/"
               NO_DEFAULT_PATH)

  message("UNWIND_LIB=" ${UNWIND_LIB})
  message("UNWIND_ARCH_LIB=" ${UNWIND_ARCH_LIB})
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

    set(MLX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mlx")
    set(MLX_BUILD_ROOT "${MLX_ROOT}/build")
    set(MLX_LIB "${MLX_BUILD_ROOT}/libmlx.a")
    set(MLX_METALLIB "${MLX_BUILD_ROOT}/mlx/backend/metal/kernels/mlx.metallib")
    if(NOT EXISTS "${MLX_LIB}" OR NOT EXISTS "${MLX_METALLIB}")
        message(FATAL_ERROR
          "unable to find \"${MLX_LIB}\" and \"${MLX_METALLIB}\", please "
          "call install_mlx.sh in third_party first.")
    endif()

    add_definitions("-DLIBWAIFU_MLX_ENABLED")

    # MLX fetches these two at configure time and leaves them beside libmlx.a rather than
    # folding them in, so a consumer of the static build has to name them itself.
    add_library(mlx_jaccl STATIC IMPORTED)
    set_target_properties(mlx_jaccl PROPERTIES
        IMPORTED_LOCATION "${MLX_BUILD_ROOT}/jaccl/libjaccl.a")
    add_library(mlx_gguflib STATIC IMPORTED)
    set_target_properties(mlx_gguflib PROPERTIES
        IMPORTED_LOCATION "${MLX_BUILD_ROOT}/mlx/io/libgguflib.a")

    # fmt, json and metal-cpp are header-only here, but MLX's own headers include them, so
    # anything that includes mlx/mlx.h needs them on the search path too.
    set(MLX_INCDIRS
        "${MLX_ROOT}"
        "${MLX_BUILD_ROOT}/_deps/fmt-src/include"
        "${MLX_BUILD_ROOT}/_deps/json-src/single_include"
        "${MLX_BUILD_ROOT}/_deps/metal_cpp-src")

    add_library(mlx STATIC IMPORTED)
    set_target_properties(mlx PROPERTIES
        IMPORTED_LOCATION "${MLX_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${MLX_INCDIRS}"
        INTERFACE_LINK_LIBRARIES
            "mlx_jaccl;mlx_gguflib;-framework Metal;-framework Foundation;-framework QuartzCore;-framework Accelerate")

    # Carry the Metal kernels inside the binary, the way a CUDA build carries its fatbin,
    # instead of leaving them as a file that has to be found at run time. Metal has no
    # toolchain support for this, so it takes two halves: a Mach-O section put in here, and
    # fl::metal::useEmbeddedMetallib() handing it to MLX at startup. As an INTERFACE option
    # on the imported target, every executable that links MLX gets the section for free.
    set(MLX_METALLIB_LINK_OPTION "-Wl,-sectcreate,__TEXT,__metallib,${MLX_METALLIB}")
    set_property(TARGET mlx APPEND PROPERTY
        INTERFACE_LINK_OPTIONS "${MLX_METALLIB_LINK_OPTION}")
endif()

add_subdirectory("flint")

set(libwaifu_LIBADD lutil flint_static ${CMAKE_DL_LIBS} ${UNWIND_ARCH_LIB} ${UNWIND_LIB} )

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
    set(libwaifu_LIBADD ${libwaifu_LIBADD} CUDA::cudart_static CUDA::curand_static)
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

if(NOT APPLE)
    get_filename_component(flint_UNWIND_DIR "${UNWIND_LIB}" DIRECTORY)
    list(APPEND flint_CARGO_FLAGS
        "cargo:rustc-link-search=native=${flint_UNWIND_DIR}"
        "cargo:rustc-link-lib=static=${UNWIND_ARCH_LIB_NAME}"
        "cargo:rustc-link-lib=static=unwind")
endif()

if(WITH_FLASH_ATTN)
    get_filename_component(flint_FLASH_ATTN_DIR "${FLASH_ATTN_LIB}" DIRECTORY)
    list(APPEND flint_CARGO_FLAGS
        "cargo:rustc-link-search=native=${flint_FLASH_ATTN_DIR}"
        "cargo:rustc-link-lib=static=flash_attn")
endif()

if(WITH_CUDA)
    list(APPEND flint_CARGO_FLAGS
        "cargo:rustc-link-search=native=${CUDAToolkit_LIBRARY_DIR}"
        "cargo:rustc-link-lib=static=curand_static"
        "cargo:rustc-link-lib=static=cudart_static"
        "cargo:rustc-link-lib=static=culibos")
endif()

if(WITH_MLX)
    # The metallib section is a property of the final link, which for the `waifu` binary is
    # rustc's, so it has to be passed through as a link arg rather than inherited from the
    # `mlx` target the way the CMake executables inherit it.
    list(APPEND flint_CARGO_FLAGS
        "cargo:rustc-link-search=native=${MLX_BUILD_ROOT}"
        "cargo:rustc-link-lib=static=mlx"
        "cargo:rustc-link-search=native=${MLX_BUILD_ROOT}/jaccl"
        "cargo:rustc-link-lib=static=jaccl"
        "cargo:rustc-link-search=native=${MLX_BUILD_ROOT}/mlx/io"
        "cargo:rustc-link-lib=static=gguflib"
        "cargo:rustc-link-lib=framework=Metal"
        "cargo:rustc-link-lib=framework=Foundation"
        "cargo:rustc-link-lib=framework=QuartzCore"
        "cargo:rustc-link-lib=framework=Accelerate"
        "cargo:rustc-link-arg=${MLX_METALLIB_LINK_OPTION}")
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
    DEPENDS flint "${CMAKE_BINARY_DIR}/flint_link_flags.txt"
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
