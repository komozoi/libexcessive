cmake_minimum_required(VERSION 3.0)

project(LibExcessive)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D LIBUS_NO_SSL -Wall -Wextra -Werror -Wno-format -Wno-unused-parameter -Wno-unused-result")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fexec-charset=UTF-8 -fextended-identifiers -finput-charset=UTF-8") # -mavx512f -mavx512dq
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D LIBUS_NO_SSL -Wno-format -Wno-unused-parameter -Wno-unused-result")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

file(GLOB SOURCES
        "src/*.cpp"
        "src/alloc/*.cpp"
        "src/ds/*.cpp"
        "src/fs/*.cpp"
)

add_library(excessive ${SOURCES})
target_include_directories(excessive PUBLIC "include")

find_package(GTest)
if(GTest_FOUND)
    include_directories(${GTEST_INCLUDE_DIRS})
    file(GLOB_RECURSE TEST_SOURCES
            "src/tests/*.cpp"
    )
    add_executable(libexcessive_tests src/testmain.cpp ${TEST_SOURCES} ${SOURCES})
    target_link_libraries(libexcessive_tests excessive ${GTEST_LIBRARIES})
endif()