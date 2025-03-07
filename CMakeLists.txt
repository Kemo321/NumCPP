# Set the minimum required version for CMake
cmake_minimum_required(VERSION 3.14)

# Set the project name and version
project(NumCPP VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable testing
enable_testing()

# GoogleTest setup via FetchContent for testing
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0  # Specify the version you want
)

# Fetch GoogleTest
FetchContent_MakeAvailable(googletest)

# Set test sources
set(TEST_SOURCES
    test/test_main.cpp
    test/NDArray/ConDes.cpp
    test/NDArray/Prop.cpp
    test/NDArray/Modify.cpp
    test/NDArray/Access.cpp
)

# Add test executable
add_executable(tests ${TEST_SOURCES})

# Ensure runtime library consistency for MSVC (static or dynamic)
if(MSVC)
  # Set to static CRT for Debug and Release
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")  # Static CRT for Debug
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")  # Static CRT for Release
  # or to dynamic CRT if you prefer:
  # set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")  # Dynamic CRT for Debug
  # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")  # Dynamic CRT for Release
endif()

# Set runtime library to be consistent
set_target_properties(tests PROPERTIES
  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
)

# Link GoogleTest
target_link_libraries(tests gtest_main)

# Include directories
target_include_directories(tests PRIVATE include)

# Add test
add_test(NAME tests COMMAND tests)

