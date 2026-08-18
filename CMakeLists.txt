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

# libmvec / auto-vec of expf in the unary TU (does not affect errno policy elsewhere).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
	set_source_files_properties(src/ndarray_unary.cpp PROPERTIES
		COMPILE_OPTIONS "-ffast-math;-fopenmp-simd")
endif()

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

# Which SIMD binary to build. auto = this machine's CPU (not "gcc -mavx512f").
# Explicit values are for a later multi-.so release:
#   cmake -DLIBEXCESSIVE_ISA=avx512
#   cmake -DLIBEXCESSIVE_ISA=avx2
#   cmake -DLIBEXCESSIVE_ISA=sse2
#   cmake -DLIBEXCESSIVE_ISA=neon
#   cmake -DLIBEXCESSIVE_ISA=vanilla
set(LIBEXCESSIVE_ISA "auto" CACHE STRING "ISA binary: auto, vanilla, sse2, avx2, avx512, neon")
set_property(CACHE LIBEXCESSIVE_ISA PROPERTY STRINGS auto vanilla sse2 avx2 avx512 neon)

include(CheckCXXCompilerFlag)

set(_nd_cpuinfo "")
set(_nd_leaf7 "")
set(_nd_cpu_avx2 FALSE)
set(_nd_cpu_avx512 FALSE)
set(_nd_cpu_f16c FALSE)
set(_nd_cpu_avx512fp16 FALSE)
set(_nd_cpu_avx512bf16 FALSE)
set(_nd_cpu_sse2 FALSE)
set(_nd_cpu_neon FALSE)
set(_nd_cpu_neon_fp16 FALSE)
set(_nd_cpu_neon_bf16 FALSE)
set(_nd_is_x86 FALSE)
set(_nd_is_arm64 FALSE)
set(_nd_is_arm32 FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
	set(_nd_is_arm64 TRUE)
	set(_nd_cpu_neon TRUE)
	if(APPLE)
		set(_nd_cpu_neon_fp16 TRUE)
	elseif(EXISTS "/proc/cpuinfo")
		file(READ "/proc/cpuinfo" _nd_cpuinfo)
		if(_nd_cpuinfo MATCHES "fphp" OR _nd_cpuinfo MATCHES "asimdhp")
			set(_nd_cpu_neon_fp16 TRUE)
		endif()
		if(_nd_cpuinfo MATCHES "asimdbf16" OR _nd_cpuinfo MATCHES " bf16")
			set(_nd_cpu_neon_bf16 TRUE)
		endif()
	endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|armv6|arm")
	set(_nd_is_arm32 TRUE)
	if(EXISTS "/proc/cpuinfo")
		file(READ "/proc/cpuinfo" _nd_cpuinfo)
		if(_nd_cpuinfo MATCHES " neon" OR _nd_cpuinfo MATCHES " asimd")
			set(_nd_cpu_neon TRUE)
		endif()
	else()
		set(_nd_cpu_neon TRUE)
	endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i.86")
	set(_nd_is_x86 TRUE)
	set(_nd_cpu_sse2 TRUE)
	if(EXISTS "/proc/cpuinfo")
		file(READ "/proc/cpuinfo" _nd_cpuinfo)
		if(_nd_cpuinfo MATCHES " avx2")
			set(_nd_cpu_avx2 TRUE)
		endif()
		if(_nd_cpuinfo MATCHES " avx512f" AND _nd_cpuinfo MATCHES " avx512bw" AND _nd_cpuinfo MATCHES " avx512vbmi")
			set(_nd_cpu_avx512 TRUE)
		endif()
		if(_nd_cpuinfo MATCHES " f16c")
			set(_nd_cpu_f16c TRUE)
		endif()
		if(_nd_cpuinfo MATCHES "avx512_fp16" OR _nd_cpuinfo MATCHES "avx512fp16")
			set(_nd_cpu_avx512fp16 TRUE)
		endif()
		if(_nd_cpuinfo MATCHES "avx512_bf16" OR _nd_cpuinfo MATCHES "avx512bf16")
			set(_nd_cpu_avx512bf16 TRUE)
		endif()
	elseif(APPLE)
		execute_process(COMMAND sysctl -n machdep.cpu.leaf7_features
			OUTPUT_VARIABLE _nd_leaf7 OUTPUT_STRIP_TRAILING_WHITESPACE
			ERROR_QUIET)
		if(_nd_leaf7 MATCHES "AVX2")
			set(_nd_cpu_avx2 TRUE)
		endif()
		if(_nd_leaf7 MATCHES "AVX512F" AND _nd_leaf7 MATCHES "AVX512BW" AND _nd_leaf7 MATCHES "AVX512VBMI")
			set(_nd_cpu_avx512 TRUE)
		endif()
	endif()
endif()

string(TOLOWER "${LIBEXCESSIVE_ISA}" _nd_isa_in)
if(_nd_isa_in STREQUAL "auto")
	if(_nd_cpu_avx512)
		set(LIBEXCESSIVE_ISA_RESOLVED "avx512")
	elseif(_nd_cpu_avx2)
		set(LIBEXCESSIVE_ISA_RESOLVED "avx2")
	elseif(_nd_cpu_neon)
		set(LIBEXCESSIVE_ISA_RESOLVED "neon")
	elseif(_nd_is_x86)
		set(LIBEXCESSIVE_ISA_RESOLVED "sse2")
	else()
		set(LIBEXCESSIVE_ISA_RESOLVED "vanilla")
	endif()
elseif(_nd_isa_in STREQUAL "vanilla" OR _nd_isa_in STREQUAL "sse2" OR _nd_isa_in STREQUAL "avx2" OR _nd_isa_in STREQUAL "avx512" OR _nd_isa_in STREQUAL "neon")
	set(LIBEXCESSIVE_ISA_RESOLVED "${_nd_isa_in}")
else()
	message(FATAL_ERROR "LIBEXCESSIVE_ISA must be auto, vanilla, sse2, avx2, avx512, or neon (got '${LIBEXCESSIVE_ISA}')")
endif()

if(NOT _nd_is_x86)
	if(LIBEXCESSIVE_ISA_RESOLVED STREQUAL "sse2" OR LIBEXCESSIVE_ISA_RESOLVED STREQUAL "avx2" OR LIBEXCESSIVE_ISA_RESOLVED STREQUAL "avx512")
		message(FATAL_ERROR "LIBEXCESSIVE_ISA=${LIBEXCESSIVE_ISA_RESOLVED} is x86-only")
	endif()
endif()
if(_nd_is_x86 AND LIBEXCESSIVE_ISA_RESOLVED STREQUAL "neon")
	message(FATAL_ERROR "LIBEXCESSIVE_ISA=neon is ARM-only")
endif()

set(_nd_isa_flags "")
if(LIBEXCESSIVE_ISA_RESOLVED STREQUAL "vanilla")
	check_cxx_compiler_flag("-fno-tree-vectorize" LIBEXCESSIVE_HAS_NO_TREE_VEC)
	check_cxx_compiler_flag("-fno-slp-vectorize" LIBEXCESSIVE_HAS_NO_SLP_VEC)
	if(LIBEXCESSIVE_HAS_NO_TREE_VEC)
		set(_nd_isa_flags "-fno-tree-vectorize")
	endif()
	if(LIBEXCESSIVE_HAS_NO_SLP_VEC)
		set(_nd_isa_flags "${_nd_isa_flags};-fno-slp-vectorize")
	endif()
elseif(LIBEXCESSIVE_ISA_RESOLVED STREQUAL "neon")
	if(_nd_is_arm64)
		# auto: fp16 only if the CPU has it. explicit neon: modern ARM / Apple Silicon.
		set(_nd_want_fp16 FALSE)
		if(_nd_isa_in STREQUAL "auto")
			if(_nd_cpu_neon_fp16)
				set(_nd_want_fp16 TRUE)
			endif()
		else()
			set(_nd_want_fp16 TRUE)
		endif()
		if(_nd_want_fp16)
			if(_nd_cpu_neon_bf16 OR (NOT _nd_isa_in STREQUAL "auto"))
				check_cxx_compiler_flag("-march=armv8.2-a+fp16+bf16" LIBEXCESSIVE_HAS_ARM_FP16BF16)
			endif()
			check_cxx_compiler_flag("-march=armv8.2-a+fp16" LIBEXCESSIVE_HAS_ARM_FP16)
		endif()
		if(_nd_want_fp16 AND LIBEXCESSIVE_HAS_ARM_FP16BF16 AND (_nd_cpu_neon_bf16 OR (NOT _nd_isa_in STREQUAL "auto")))
			set(_nd_isa_flags "-march=armv8.2-a+fp16+bf16")
		elseif(_nd_want_fp16 AND LIBEXCESSIVE_HAS_ARM_FP16)
			set(_nd_isa_flags "-march=armv8.2-a+fp16")
		else()
			check_cxx_compiler_flag("-march=armv8-a+simd" LIBEXCESSIVE_HAS_ARMV8_SIMD)
			if(LIBEXCESSIVE_HAS_ARMV8_SIMD)
				set(_nd_isa_flags "-march=armv8-a+simd")
			endif()
		endif()
	else()
		check_cxx_compiler_flag("-mfpu=neon" LIBEXCESSIVE_HAS_MFPU_NEON)
		if(NOT LIBEXCESSIVE_HAS_MFPU_NEON)
			message(FATAL_ERROR "LIBEXCESSIVE_ISA=neon but the compiler rejects -mfpu=neon")
		endif()
		set(_nd_isa_flags "-mfpu=neon")
	endif()
elseif(LIBEXCESSIVE_ISA_RESOLVED STREQUAL "sse2")
	if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
		set(_nd_isa_flags "-msse2")
	endif()
elseif(LIBEXCESSIVE_ISA_RESOLVED STREQUAL "avx2")
	check_cxx_compiler_flag("-mavx2" LIBEXCESSIVE_HAS_AVX2)
	check_cxx_compiler_flag("-mfma" LIBEXCESSIVE_HAS_FMA)
	check_cxx_compiler_flag("-mf16c" LIBEXCESSIVE_HAS_F16C)
	if(NOT LIBEXCESSIVE_HAS_AVX2)
		message(FATAL_ERROR "LIBEXCESSIVE_ISA=avx2 but the compiler rejects -mavx2")
	endif()
	set(_nd_isa_flags "-mavx2")
	if(LIBEXCESSIVE_HAS_FMA)
		set(_nd_isa_flags "${_nd_isa_flags};-mfma")
	endif()
	# F16C is on every AVX2 Intel/AMD we care about. auto still honors /proc.
	if(LIBEXCESSIVE_HAS_F16C AND (_nd_isa_in STREQUAL "avx2" OR _nd_cpu_f16c OR NOT EXISTS "/proc/cpuinfo"))
		set(_nd_isa_flags "${_nd_isa_flags};-mf16c")
	endif()
elseif(LIBEXCESSIVE_ISA_RESOLVED STREQUAL "avx512")
	check_cxx_compiler_flag("-mavx512f" LIBEXCESSIVE_HAS_AVX512F)
	check_cxx_compiler_flag("-mavx512bw" LIBEXCESSIVE_HAS_AVX512BW)
	check_cxx_compiler_flag("-mavx512vbmi" LIBEXCESSIVE_HAS_AVX512VBMI)
	if(NOT (LIBEXCESSIVE_HAS_AVX512F AND LIBEXCESSIVE_HAS_AVX512BW AND LIBEXCESSIVE_HAS_AVX512VBMI))
		message(FATAL_ERROR "LIBEXCESSIVE_ISA=avx512 but the compiler rejects required -mavx512* flags")
	endif()
	set(_nd_isa_flags "-mavx512f;-mavx512bw;-mavx512vbmi")
	check_cxx_compiler_flag("-mavx512dq" LIBEXCESSIVE_HAS_AVX512DQ)
	check_cxx_compiler_flag("-mavx512vpopcntdq" LIBEXCESSIVE_HAS_AVX512VPOPCNT)
	check_cxx_compiler_flag("-mavx512vnni" LIBEXCESSIVE_HAS_AVX512VNNI)
	check_cxx_compiler_flag("-mf16c" LIBEXCESSIVE_HAS_F16C)
	check_cxx_compiler_flag("-mavx512fp16" LIBEXCESSIVE_HAS_AVX512FP16)
	check_cxx_compiler_flag("-mavx512bf16" LIBEXCESSIVE_HAS_AVX512BF16)
	if(LIBEXCESSIVE_HAS_AVX512DQ)
		set(_nd_isa_flags "${_nd_isa_flags};-mavx512dq")
	endif()
	if(LIBEXCESSIVE_HAS_AVX512VPOPCNT)
		set(_nd_isa_flags "${_nd_isa_flags};-mavx512vpopcntdq")
	endif()
	if(LIBEXCESSIVE_HAS_AVX512VNNI)
		set(_nd_isa_flags "${_nd_isa_flags};-mavx512vnni")
	endif()
	if(LIBEXCESSIVE_HAS_F16C)
		set(_nd_isa_flags "${_nd_isa_flags};-mf16c")
	endif()
	check_cxx_compiler_flag("-mfma" LIBEXCESSIVE_HAS_FMA_AVX512)
	if(LIBEXCESSIVE_HAS_FMA_AVX512)
		set(_nd_isa_flags "${_nd_isa_flags};-mfma")
	endif()
	# Native half ALU only when this CPU has it (explicit avx512 stays portable).
	if(_nd_isa_in STREQUAL "auto" AND _nd_cpu_avx512fp16 AND LIBEXCESSIVE_HAS_AVX512FP16)
		set(_nd_isa_flags "${_nd_isa_flags};-mavx512fp16")
	endif()
	if(_nd_isa_in STREQUAL "auto" AND _nd_cpu_avx512bf16 AND LIBEXCESSIVE_HAS_AVX512BF16)
		set(_nd_isa_flags "${_nd_isa_flags};-mavx512bf16")
	endif()
endif()

if(_nd_isa_flags)
	target_compile_options(excessive PRIVATE ${_nd_isa_flags})
endif()
target_compile_definitions(excessive PRIVATE
	LIBEXCESSIVE_ISA_${LIBEXCESSIVE_ISA_RESOLVED}=1
)
if(NOT _nd_isa_in STREQUAL "auto")
	set_target_properties(excessive PROPERTIES OUTPUT_NAME "excessive_${LIBEXCESSIVE_ISA_RESOLVED}")
endif()
message(STATUS "LIBEXCESSIVE_ISA=${LIBEXCESSIVE_ISA} -> ${LIBEXCESSIVE_ISA_RESOLVED} (${_nd_isa_flags})")

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
