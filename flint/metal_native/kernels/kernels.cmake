# Builds flint's Metal kernels into one library, and that library into a source file of flint's.
#
# Every kernels/*.metal is compiled to AIR and the lot linked into flint.metallib, which
# embed.cmake writes out as a byte array (metallib.cc). Carried in flint like that, the kernels
# need no file beside whatever links flint -- a CMake executable, the waifu binary, libwaifu.dylib
# -- and no link option of their own either.
#
# Metal 3.1 and macOS 14.0 are the deployment target's: a library built for anything newer does not
# load on a Mac that is not. -fno-fast-math, because Metal assumes fast math otherwise, and that
# rewrites what the kernels say about infinity, NaN and rounding into whatever is quickest.

set(FLINT_METAL_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/metal_native/kernels")
set(FLINT_METAL_KERNEL_OUT "${CMAKE_CURRENT_BINARY_DIR}/metal_kernels")
file(MAKE_DIRECTORY "${FLINT_METAL_KERNEL_OUT}")

file(GLOB FLINT_METAL_KERNEL_SOURCES "${FLINT_METAL_KERNEL_DIR}/*.metal")
file(GLOB FLINT_METAL_KERNEL_HEADERS "${FLINT_METAL_KERNEL_DIR}/*.h")

set(FLINT_METAL_AIR "")
foreach(source ${FLINT_METAL_KERNEL_SOURCES})
    get_filename_component(name "${source}" NAME_WE)
    set(air "${FLINT_METAL_KERNEL_OUT}/${name}.air")
    add_custom_command(
        OUTPUT "${air}"
        COMMAND xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fno-fast-math
                -Wall -Werror "-I${FLINT_METAL_KERNEL_DIR}" -c "${source}" -o "${air}"
        DEPENDS "${source}" ${FLINT_METAL_KERNEL_HEADERS}
        COMMENT "metal ${name}.metal"
        VERBATIM)
    list(APPEND FLINT_METAL_AIR "${air}")
endforeach()

set(FLINT_METALLIB "${FLINT_METAL_KERNEL_OUT}/flint.metallib")
add_custom_command(
    OUTPUT "${FLINT_METALLIB}"
    COMMAND xcrun -sdk macosx metallib ${FLINT_METAL_AIR} -o "${FLINT_METALLIB}"
    DEPENDS ${FLINT_METAL_AIR}
    COMMENT "metallib flint.metallib"
    VERBATIM)

set(FLINT_METALLIB_SOURCE "${FLINT_METAL_KERNEL_OUT}/metallib.cc")
add_custom_command(
    OUTPUT "${FLINT_METALLIB_SOURCE}"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${FLINT_METALLIB}" "-DOUTPUT=${FLINT_METALLIB_SOURCE}"
            -P "${FLINT_METAL_KERNEL_DIR}/embed.cmake"
    DEPENDS "${FLINT_METALLIB}" "${FLINT_METAL_KERNEL_DIR}/embed.cmake"
    COMMENT "embed flint.metallib"
    VERBATIM)

# The kernels in nax/, for the neural accelerators of the GPUs that have them: Metal 4 and its
# tensor operations, so built for macOS 26.2, into a library of their own that the context loads
# only where it can run. A toolchain with no Metal 4 builds an empty one, and those GPUs run the
# kernels above instead.
set(FLINT_METAL_NAX_DIR "${FLINT_METAL_KERNEL_DIR}/nax")
set(FLINT_METAL_NAX_FLAGS -std=metal4.0 -mmacosx-version-min=26.2 -fno-fast-math -Wall -Werror
    "-I${FLINT_METAL_KERNEL_DIR}")

set(probe "${FLINT_METAL_KERNEL_OUT}/nax_probe.metal")
file(WRITE "${probe}" "#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
kernel void probe(device half *a [[buffer(0)]]) { a[0] = 0; }
")
execute_process(
    COMMAND xcrun -sdk macosx metal ${FLINT_METAL_NAX_FLAGS} -c "${probe}"
            -o "${FLINT_METAL_KERNEL_OUT}/nax_probe.air"
    RESULT_VARIABLE probe_result
    OUTPUT_QUIET ERROR_QUIET)

set(FLINT_NAX_METALLIB_SOURCE "${FLINT_METAL_KERNEL_OUT}/nax_metallib.cc")
if(probe_result EQUAL 0)
    message(STATUS "Metal kernels for the neural accelerators: built")
    file(GLOB FLINT_METAL_NAX_SOURCES "${FLINT_METAL_NAX_DIR}/*.metal")
    file(GLOB FLINT_METAL_NAX_HEADERS "${FLINT_METAL_NAX_DIR}/*.h")

    set(FLINT_METAL_NAX_AIR "")
    foreach(source ${FLINT_METAL_NAX_SOURCES})
        get_filename_component(name "${source}" NAME_WE)
        set(air "${FLINT_METAL_KERNEL_OUT}/nax_${name}.air")
        add_custom_command(
            OUTPUT "${air}"
            COMMAND xcrun -sdk macosx metal ${FLINT_METAL_NAX_FLAGS} -c "${source}" -o "${air}"
            DEPENDS "${source}" ${FLINT_METAL_NAX_HEADERS} ${FLINT_METAL_KERNEL_HEADERS}
            COMMENT "metal nax/${name}.metal"
            VERBATIM)
        list(APPEND FLINT_METAL_NAX_AIR "${air}")
    endforeach()

    set(FLINT_NAX_METALLIB "${FLINT_METAL_KERNEL_OUT}/flint_nax.metallib")
    add_custom_command(
        OUTPUT "${FLINT_NAX_METALLIB}"
        COMMAND xcrun -sdk macosx metallib ${FLINT_METAL_NAX_AIR} -o "${FLINT_NAX_METALLIB}"
        DEPENDS ${FLINT_METAL_NAX_AIR}
        COMMENT "metallib flint_nax.metallib"
        VERBATIM)
    add_custom_command(
        OUTPUT "${FLINT_NAX_METALLIB_SOURCE}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${FLINT_NAX_METALLIB}"
                "-DOUTPUT=${FLINT_NAX_METALLIB_SOURCE}" -DNAME=kNaxMetallib
                -P "${FLINT_METAL_KERNEL_DIR}/embed.cmake"
        DEPENDS "${FLINT_NAX_METALLIB}" "${FLINT_METAL_KERNEL_DIR}/embed.cmake"
        COMMENT "embed flint_nax.metallib"
        VERBATIM)
else()
    message(STATUS "Metal kernels for the neural accelerators: not built, no Metal 4 here")
    add_custom_command(
        OUTPUT "${FLINT_NAX_METALLIB_SOURCE}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=" "-DOUTPUT=${FLINT_NAX_METALLIB_SOURCE}"
                -DNAME=kNaxMetallib -P "${FLINT_METAL_KERNEL_DIR}/embed.cmake"
        DEPENDS "${FLINT_METAL_KERNEL_DIR}/embed.cmake"
        COMMENT "embed an empty flint_nax.metallib"
        VERBATIM)
endif()
