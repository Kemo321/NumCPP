cmake_minimum_required(VERSION 3.20)
project(NumCPP VERSION 1.0 LANGUAGES CXX)

# Set C++23 standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable testing with CTest
enable_testing()

# Fetch GoogleTest with custom configuration to avoid -Werror,-Wdeprecated-copy
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
)
# Prevent GoogleTest from overriding compiler/linker options
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# Suppress -Wdeprecated-copy for GoogleTest build
set(CMAKE_CXX_FLAGS_BACKUP "${CMAKE_CXX_FLAGS}")
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy -Wno-error=deprecated-copy")
endif()
FetchContent_MakeAvailable(googletest)
# Restore original CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BACKUP}")

# Source and include directories
set(NUMCPP_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(NUMCPP_TEST_DIR ${CMAKE_SOURCE_DIR}/test/Array)

# Test sources
file(GLOB_RECURSE TEST_SOURCES
    ${NUMCPP_TEST_DIR}/*.cpp
)

# Add test executable
add_executable(tests ${TEST_SOURCES})

# Link GoogleTest and GoogleMock
target_link_libraries(tests PRIVATE GTest::gtest_main GTest::gmock_main)

# Include directories
target_include_directories(tests PRIVATE ${NUMCPP_INCLUDE_DIR})

# MSVC runtime library configuration
if(MSVC)
    # Use static CRT for consistency (MultiThreaded for Release, MultiThreadedDebug for Debug)
    target_compile_options(tests PRIVATE
        $<$<CONFIG:Debug>:/MTd>
        $<$<CONFIG:Release>:/MT>
    )
endif()

# Compiler-specific options for the main project
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(tests PRIVATE
        -Wall -Wextra -Wpedantic
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(tests PRIVATE
        /W4
        /wd4996 # Suppress deprecated function warnings
    )
endif()

# Enable test discovery
include(GoogleTest)
gtest_discover_tests(tests)

# Install rules (optional)
install(TARGETS tests DESTINATION bin)
install(DIRECTORY ${NUMCPP_INCLUDE_DIR}/ DESTINATION include/NumCPP
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.tpp")

# Diagnostic output
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "GoogleTest source dir: ${googletest_SOURCE_DIR}")