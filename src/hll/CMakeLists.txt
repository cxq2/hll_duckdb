cmake_minimum_required(VERSION 3.14)
project(hll VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Library target ---
add_library(hll STATIC src/hll.cpp)
target_include_directories(hll PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# --- Tests ---
option(HLL_BUILD_TESTS "Build HLL unit tests" ON)
if(HLL_BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.2
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(test_hll tests/test_hll.cpp)
    target_link_libraries(test_hll PRIVATE hll Catch2::Catch2WithMain)
    add_test(NAME test_hll COMMAND test_hll)
endif()
