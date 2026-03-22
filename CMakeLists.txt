cmake_minimum_required(VERSION 3.15)

project(LibExcessive LANGUAGES C CXX)

file(GLOB_RECURSE LIBEXCESSIVE_SOURCES CONFIGURE_DEPENDS
        src/*.cpp
        src/alloc/*.cpp
        src/ds/*.cpp
        src/fs/*.cpp
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

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    option(LIBEXCESSIVE_BUILD_TESTS "Build libexcessive tests" ON)
else()
    option(LIBEXCESSIVE_BUILD_TESTS "Build libexcessive tests" OFF)
endif()

if(LIBEXCESSIVE_BUILD_TESTS)
    find_package(GTest QUIET)

    if(GTest_FOUND)
        file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
                src/tests/*.cpp
        )

        add_executable(libexcessive_tests src/testmain.cpp ${TEST_SOURCES})
        target_link_libraries(libexcessive_tests PRIVATE LibExcessive::excessive ${GTEST_LIBRARIES})
    endif()
endif()
