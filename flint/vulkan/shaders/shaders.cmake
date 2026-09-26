# Compiles the GLSL compute shaders to SPIR-V and embeds them in the library.
#
# One source is usually several kernels: the element types a kernel reads and writes are fixed
# when it is compiled, so each (source, types) pair it is needed for is a variant with a name of
# its own. flint_vulkan_shader() adds one; kernels are then asked for by that name at run time,
# through the table this file writes out as shaders.cc.

set(FLINT_VULKAN_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders")
set(FLINT_VULKAN_SHADER_OUT "${CMAKE_CURRENT_BINARY_DIR}/vulkan_shaders")
file(MAKE_DIRECTORY "${FLINT_VULKAN_SHADER_OUT}")
file(GLOB FLINT_VULKAN_SHADER_INCLUDES "${FLINT_VULKAN_SHADER_DIR}/*.glsl")

set(FLINT_VULKAN_SHADER_NAMES "")
set(FLINT_VULKAN_SHADER_HEADERS "")

# flint_vulkan_shader(<name> <source> [DEFINE=VALUE ...])
function(flint_vulkan_shader name source)
    set(header "${FLINT_VULKAN_SHADER_OUT}/${name}.h")
    set(defines "")
    foreach(define ${ARGN})
        list(APPEND defines "-D${define}")
    endforeach()

    add_custom_command(
        OUTPUT "${header}"
        COMMAND "${GLSLANG_VALIDATOR}" -V --target-env vulkan1.2 --quiet
                "-I${FLINT_VULKAN_SHADER_DIR}" ${defines}
                --vn "kSpirv_${name}" -o "${header}"
                "${FLINT_VULKAN_SHADER_DIR}/${source}"
        DEPENDS "${FLINT_VULKAN_SHADER_DIR}/${source}" ${FLINT_VULKAN_SHADER_INCLUDES}
                ${GLSLANG_DEPENDS}
        COMMENT "glslang ${source} -> ${name}"
        VERBATIM)

    set(FLINT_VULKAN_SHADER_NAMES ${FLINT_VULKAN_SHADER_NAMES} "${name}" PARENT_SCOPE)
    set(FLINT_VULKAN_SHADER_HEADERS ${FLINT_VULKAN_SHADER_HEADERS} "${header}" PARENT_SCOPE)
endfunction()

# The element types, by the suffix a variant's name carries for them: the GLSL type a buffer holds
# it as, and its alignment in bytes.
set(FLINT_VULKAN_TYPE_f32 "float;4")
set(FLINT_VULKAN_TYPE_f16 "float16_t;2")
set(FLINT_VULKAN_TYPE_i64 "int64_t;8")
set(FLINT_VULKAN_TYPE_i32 "int;4")
set(FLINT_VULKAN_TYPE_u8 "uint8_t;1")
set(FLINT_VULKAN_TYPE_i8 "int8_t;1")
set(FLINT_VULKAN_TYPE_bool "uint8_t;1")

# The defines that make slot `slot` (A, B, C, ...) hold elements of type `suffix`.
function(flint_vulkan_type_defines out slot suffix)
    list(GET FLINT_VULKAN_TYPE_${suffix} 0 glsl_type)
    list(GET FLINT_VULKAN_TYPE_${suffix} 1 align)
    set(defines "${slot}_T=${glsl_type}" "${slot}_ALIGN=${align}")
    if(suffix STREQUAL "bool")
        list(APPEND defines "${slot}_BOOL=1")
    endif()
    if(suffix STREQUAL "i64" OR suffix STREQUAL "i32" OR suffix STREQUAL "u8" OR
       suffix STREQUAL "i8" OR suffix STREQUAL "bool")
        list(APPEND defines "${slot}_INTEGER=1")
    endif()
    set(${out} ${defines} PARENT_SCOPE)
endfunction()

# copy: strided copy with a conversion, which is also what cast and contiguous are.
foreach(pair
        f32_f32 f16_f16 f32_f16 f16_f32 i64_i64 i32_i32 u8_u8 i8_i8 bool_bool
        i64_f32 i64_f16 i32_f32 i32_f16 i32_i64 i64_i32 bool_f32 bool_f16 f32_bool f16_bool
        u8_f32 u8_f16 i8_f32 i8_f16 f32_i64 f16_i64)
    string(REPLACE "_" ";" types "${pair}")
    list(GET types 0 src)
    list(GET types 1 dst)
    flint_vulkan_type_defines(src_defines A ${src})
    flint_vulkan_type_defines(dst_defines C ${dst})
    flint_vulkan_shader("copy_${pair}" "copy.comp" ${src_defines} ${dst_defines})
endforeach()

foreach(type f32 f16 i64 i32 u8 i8 bool)
    flint_vulkan_type_defines(defines C ${type})
    flint_vulkan_shader("fill_${type}" "fill.comp" ${defines})
endforeach()

foreach(type f32 f16)
    flint_vulkan_type_defines(defines_a A ${type})
    flint_vulkan_type_defines(defines_c C ${type})
    flint_vulkan_shader("unary_${type}" "unary.comp" ${defines_a} ${defines_c})
    flint_vulkan_type_defines(defines_b B ${type})
    flint_vulkan_shader("binary_${type}" "binary.comp" ${defines_a} ${defines_b} ${defines_c})
    flint_vulkan_type_defines(defines_bool C bool)
    flint_vulkan_shader("equal_${type}" "binary.comp" ${defines_a} ${defines_b} ${defines_bool}
                        EQUAL_ONLY=1)
    flint_vulkan_shader("softmax_${type}" "softmax.comp" ${defines_a} ${defines_c})
    flint_vulkan_shader("reduce_${type}" "reduce.comp" ${defines_a} ${defines_c})
    flint_vulkan_shader("layer_norm_${type}" "layer_norm.comp" ${defines_a} ${defines_c})
    flint_vulkan_shader("rms_norm_${type}" "layer_norm.comp" ${defines_a} ${defines_c} RMS_NORM=1)
    flint_vulkan_shader("group_norm_${type}" "group_norm.comp" ${defines_a} ${defines_c})
    flint_vulkan_shader("glu_${type}" "glu.comp" ${defines_a} ${defines_c})
    flint_vulkan_shader("upsample_${type}" "upsample.comp" ${defines_a} ${defines_c})
    flint_vulkan_shader("lookup_${type}" "lookup.comp" ${defines_a} ${defines_c})
    flint_vulkan_shader("rotary_${type}" "rotary.comp" ${defines_a})
    flint_vulkan_shader("gemm_${type}" "gemm.comp" ${defines_a} ${defines_c})
    flint_vulkan_shader("conv2d_${type}" "conv2d.comp" ${defines_a} ${defines_c})
endforeach()

# The same two products on cooperative matrices, for devices that have them. Half precision only:
# that is what the tensor cores multiply, and rounding a float model's operands to half would
# change its answer.
flint_vulkan_type_defines(defines_a A f16)
flint_vulkan_type_defines(defines_c C f16)
flint_vulkan_shader("gemm_coopmat_f16" "gemm_coopmat.comp" ${defines_a} ${defines_c})
flint_vulkan_shader("conv2d_coopmat_f16" "conv2d_coopmat.comp" ${defines_a} ${defines_c})

flint_vulkan_type_defines(defines_i64 A i64)
flint_vulkan_shader("equal_i64" "binary.comp" ${defines_i64} B_T=int64_t B_ALIGN=8 B_INTEGER=1
                    C_T=uint8_t C_ALIGN=1 C_BOOL=1 C_INTEGER=1 EQUAL_ONLY=1)
flint_vulkan_shader("binary_i64" "binary.comp" ${defines_i64} B_T=int64_t B_ALIGN=8 B_INTEGER=1
                    C_T=int64_t C_ALIGN=8 C_INTEGER=1)
flint_vulkan_shader("mod_i64" "mod.comp")
flint_vulkan_shader("arange_i64" "arange.comp")
flint_vulkan_shader("reduce_bool" "reduce.comp" A_T=uint8_t A_ALIGN=1 A_BOOL=1 A_INTEGER=1
                    C_T=uint8_t C_ALIGN=1 C_BOOL=1 C_INTEGER=1)
flint_vulkan_shader("equal_u8" "binary.comp" A_T=uint8_t A_ALIGN=1 A_INTEGER=1
                    B_T=uint8_t B_ALIGN=1 B_INTEGER=1 C_T=uint8_t C_ALIGN=1 C_BOOL=1 C_INTEGER=1
                    EQUAL_ONLY=1)
flint_vulkan_shader("rand_f32" "rand.comp")

# The table the runtime finds a kernel in, by name.
set(_includes "")
set(_entries "")
foreach(name ${FLINT_VULKAN_SHADER_NAMES})
    string(APPEND _includes "#include \"${name}.h\"\n")
    string(APPEND _entries "    {\"${name}\", kSpirv_${name}, sizeof(kSpirv_${name})},\n")
endforeach()
file(WRITE "${FLINT_VULKAN_SHADER_OUT}/shaders.cc.in" "\
// Generated by flint/vulkan/shaders/shaders.cmake. Do not edit.

#include <stdint.h>
#include <string.h>

#include \"flint/vulkan/shaders.h\"

${_includes}
namespace fl {
namespace op {
namespace vulkan {

namespace {

const Spirv kShaders[] = {
${_entries}};

}  // namespace

const Spirv *findSpirv(const char *name) {
  for (const Spirv &shader : kShaders) {
    if (strcmp(shader.name, name) == 0) return &shader;
  }
  return nullptr;
}

}  // namespace vulkan
}  // namespace op
}  // namespace fl
")
# Written through a copy so that an unchanged table leaves shaders.cc, and so the library,
# untouched on a reconfigure.
file(COPY_FILE "${FLINT_VULKAN_SHADER_OUT}/shaders.cc.in" "${FLINT_VULKAN_SHADER_OUT}/shaders.cc"
     ONLY_IF_DIFFERENT)
set_source_files_properties("${FLINT_VULKAN_SHADER_OUT}/shaders.cc"
    PROPERTIES OBJECT_DEPENDS "${FLINT_VULKAN_SHADER_HEADERS}")
add_custom_target(flint_vulkan_shaders DEPENDS ${FLINT_VULKAN_SHADER_HEADERS})
