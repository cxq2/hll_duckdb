cmake_minimum_required(VERSION 3.5)

# Set extension name here
set(TARGET_NAME quack)

# DuckDB's extension distribution supports vcpkg. As such, dependencies can be added in ./vcpkg.json and then
# used in cmake with find_package. Feel free to remove or replace with other dependencies.
# Note that it should also be removed from vcpkg.json to prevent needlessly installing it..
find_package(OpenSSL REQUIRED)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(src/include src/hll/include)

set(EXTENSION_SOURCES src/quack_extension.cpp)

# Prefer newly added HLL .cc sources; fall back to existing .cpp layout when needed.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/hll/hll.cc")
  list(APPEND EXTENSION_SOURCES src/hll/hll.cc)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/hll/src/hll.cpp")
  list(APPEND EXTENSION_SOURCES src/hll/src/hll.cpp)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/hll/murmur3.cc")
  list(APPEND EXTENSION_SOURCES src/hll/murmur3.cc)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/hll/src/murmur3.cpp")
  list(APPEND EXTENSION_SOURCES src/hll/src/murmur3.cpp)
endif()

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Link OpenSSL in both the static library as the loadable extension
target_link_libraries(${EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto)
target_link_libraries(${LOADABLE_EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto)

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
