set(lutil_SOURCES
    "internal/log.cc"
    "internal/sprintf.cc"
    "error.cc"
    "half.cc"
    "is_debug.cc"
    "path.cc"
    "random.cc"
    "reader.cc"
    "strings.cc"
    "time.cc")

set(lut_test_SOURCES
    "log_test.cc"
    "path_test.cc"
    "strings_test.cc")

# OS specific code
if(WIN32)
    set(lutil_SOURCES
        ${lutil_SOURCES}
        "path_windows.cc"
        "platform_windows.cc"
        "shared_library_windows.cc")
endif()
if(UNIX)
    set(lutil_SOURCES
        ${lutil_SOURCES}
        "platform_linux.cc"
        "shared_library_linux.cc")
endif()
if(UNIX AND APPLE)
    set(lutil_SOURCES ${lutil_SOURCES} "path_darwin.cc")
endif()
if(UNIX AND NOT APPLE)
    set(lutil_SOURCES ${lutil_SOURCES} "path_linux.cc")
endif()

add_library(lutil STATIC ${lutil_SOURCES})
set_target_properties(lutil PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_include_directories(lutil PRIVATE ".." ${LIBDWARF_INCLUDE_DIR})
if(UNIX AND NOT APPLE)
    # On macOS <libunwind.h> comes from the SDK, and on Windows there is no such header at all
    # -- platform_windows.cc walks the stack with DbgHelp instead. Only Linux takes it from the
    # NonGNU build the top-level CMakeLists drives. That one runs at build time, so say so:
    # lutil is the only thing that includes the header, and every binary reaches the archives
    # through it.
    target_include_directories(lutil PRIVATE "${UNWIND_INCLUDE_DIR}")
    add_dependencies(lutil unwind_external)
endif()

add_library(lutil_test OBJECT ${lut_test_SOURCES})
target_include_directories(lutil_test PRIVATE "..")
