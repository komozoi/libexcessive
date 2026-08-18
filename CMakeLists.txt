cmake_minimum_required(VERSION 3.15)

project(LibExcessive LANGUAGES C CXX)

file(GLOB LIBEXCESSIVE_SOURCES CONFIGURE_DEPENDS
        src/*.cpp
        src/alloc/*.cpp
        src/ds/*.cpp
        src/fs/*.cpp
        src/parallel/*.cpp
)

add_library(excessive ${LIBEXCESSIVE_SOURCES})
add_library(LibExcessive::excessive ALIAS excessive)

# LibExcessive should be compiled with -Wall -Werror -Wextra
# There are a few ignored errors/warnings that I think are a bit silly in our case.
# Currently c++17 is required for string_view, but this can be fixed in
# a future release.
target_compile_features(excessive PUBLIC cxx_std_17)
target_compile_options(excessive
        PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:
        -Wall
        -Wextra
        -Werror
        -Wno-format
        -Wno-unused-parameter
        -Wno-unused-result
        -fexec-charset=UTF-8
        -finput-charset=UTF-8
        -fextended-identifiers
        >
        $<$<CXX_COMPILER_ID:MSVC>:
        /W4
        /WX
        >
)

# Public include directory (for consumers)
target_include_directories(excessive
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# INT3 mul/div (NDArray.cpp) and packed score kernels (ndarray_score.cpp):
# compile with the best ISA the toolchain can emit. Runtime dispatch is
# compile-time via __AVX512F__ / __AVX2__ / __SSE2__ / __ARM_NEON.
include(CheckCXXCompilerFlag)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i.86")
    check_cxx_compiler_flag("-mavx512f"             LIBEXCESSIVE_HAS_AVX512F)
    check_cxx_compiler_flag("-mavx512bw"            LIBEXCESSIVE_HAS_AVX512BW)
    check_cxx_compiler_flag("-mavx512vbmi"          LIBEXCESSIVE_HAS_AVX512VBMI)
    check_cxx_compiler_flag("-mavx512dq"            LIBEXCESSIVE_HAS_AVX512DQ)
    check_cxx_compiler_flag("-mavx512vpopcntdq"     LIBEXCESSIVE_HAS_AVX512VPOPCNT)
    check_cxx_compiler_flag("-mavx2"                LIBEXCESSIVE_HAS_AVX2)
    check_cxx_compiler_flag("-mfma"                 LIBEXCESSIVE_HAS_FMA)
    set(_ndscore_flags "")
    if(LIBEXCESSIVE_HAS_AVX512F AND LIBEXCESSIVE_HAS_AVX512BW AND LIBEXCESSIVE_HAS_AVX512VBMI)
        set(_ndscore_flags "-mavx512f -mavx512bw -mavx512vbmi")
        if(LIBEXCESSIVE_HAS_AVX512DQ)
            set(_ndscore_flags "${_ndscore_flags} -mavx512dq")
        endif()
        if(LIBEXCESSIVE_HAS_AVX512VPOPCNT)
            set(_ndscore_flags "${_ndscore_flags} -mavx512vpopcntdq")
        endif()
        set_source_files_properties(src/NDArray.cpp PROPERTIES COMPILE_FLAGS "${_ndscore_flags}")
        set_source_files_properties(src/ndarray_matmul.cpp PROPERTIES COMPILE_FLAGS "${_ndscore_flags}")
    elseif(LIBEXCESSIVE_HAS_AVX2)
        set(_ndscore_flags "-mavx2")
        if(LIBEXCESSIVE_HAS_FMA)
            set(_ndscore_flags "${_ndscore_flags} -mfma")
        endif()
    endif()
    if(_ndscore_flags)
        set_source_files_properties(src/ndarray_score.cpp PROPERTIES COMPILE_FLAGS "${_ndscore_flags}")
        set_source_files_properties(src/ndarray_matmul.cpp PROPERTIES COMPILE_FLAGS "${_ndscore_flags}")
    endif()
endif()

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    option(LIBEXCESSIVE_BUILD_TESTS "Build libexcessive tests" ON)
else()
    option(LIBEXCESSIVE_BUILD_TESTS "Build libexcessive tests" OFF)
endif()

if(LIBEXCESSIVE_BUILD_TESTS)
    find_package(GTest QUIET)

    if(GTest_FOUND)
        file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
                src/tests/*.cpp
        )

        add_executable(libexcessive_tests src/tests/testmain.cpp ${TEST_SOURCES})
        target_link_libraries(libexcessive_tests PRIVATE LibExcessive::excessive ${GTEST_LIBRARIES})
        # Same diagnostics as the library so consumer -Werror issues (e.g. bool set) fail here.
        target_compile_options(libexcessive_tests PRIVATE
                $<$<CXX_COMPILER_ID:GNU,Clang>:
                -Wall
                -Wextra
                -Werror
                -Wno-format
                -Wno-unused-parameter
                -Wno-unused-result
                >
        )
    endif()
endif()
