# Apply the metallib-from-memory patch to MLX, idempotently.
# Called by ExternalProject_Add's PATCH_COMMAND via:
#   cmake -DMLX_DIR=... -DPATCH_FILE=... -P apply_mlx_patch.cmake

execute_process(
    COMMAND git -C "${MLX_DIR}" apply --reverse --check "${PATCH_FILE}"
    RESULT_VARIABLE already_applied
    OUTPUT_QUIET ERROR_QUIET)

if(already_applied EQUAL 0)
    message(STATUS "Patch already applied")
else()
    execute_process(
        COMMAND git -C "${MLX_DIR}" apply "${PATCH_FILE}"
        RESULT_VARIABLE rc)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "Failed to apply ${PATCH_FILE}")
    endif()
    message(STATUS "Applied ${PATCH_FILE}")
endif()
