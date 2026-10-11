# cmake -DINPUT=<flint.metallib> -DOUTPUT=<metallib.cc> [-DNAME=<symbol>] -P embed.cmake
#
# Writes INPUT out as the bytes of NAME, kMetallib unless given, with its size in NAME##Size:
# both of which flint/metal_native/kernels.h declares. An empty INPUT writes a library of no
# bytes, for one the toolchain could not build.

if(NOT DEFINED NAME)
    set(NAME kMetallib)
endif()

if("${INPUT}" STREQUAL "")
    set(bytes "0x00,")
    set(size 0)
else()
    file(READ "${INPUT}" hex HEX)
    string(LENGTH "${hex}" length)
    math(EXPR size "${length} / 2")

    # Two hex digits a byte, as 0xNN, sixteen to a line.
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
    string(REGEX REPLACE "((0x[0-9a-f][0-9a-f],){16})" "\\1\n" bytes "${bytes}")
endif()

file(WRITE "${OUTPUT}" "// Written by flint/metal_native/kernels/embed.cmake. Do not edit.

#include \"flint/metal_native/kernels.h\"

namespace fl {
namespace op {
namespace metal_native {

alignas(16) const unsigned char ${NAME}[] = {
${bytes}
};
const size_t ${NAME}Size = ${size};

}  // namespace metal_native
}  // namespace op
}  // namespace fl
")
