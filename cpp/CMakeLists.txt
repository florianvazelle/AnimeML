cmake_minimum_required(VERSION 3.8)

# ---- Project ----

project(cpp VERSION 1.2.3.4 LANGUAGES CXX)
set(LIBRARY_NAME example)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# ---- Options ----

option(ENABLE_TESTING "Whether to generate unit test program and test" ON)

# ---- Add source files ----

file(GLOB_RECURSE PROJECT_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE PROJECT_HEADERS "${CMAKE_SOURCE_DIR}/include/*.hpp")

include_directories("${CMAKE_SOURCE_DIR}/include")

# ---- Conan package ----

if(CONAN_EXPORTED)
    include(${CMAKE_CURRENT_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup()
else()
    include("${CMAKE_SOURCE_DIR}/cmake/conan.cmake")
    conan_cmake_run(CONANFILE conanfile.txt BASIC_SETUP BUILD missing)
endif()

# ---- Create object library ----

add_library(${LIBRARY_NAME}Obj OBJECT ${PROJECT_SOURCES} ${PROJECT_HEADERS})
target_link_libraries(${LIBRARY_NAME}Obj PUBLIC ${CONAN_LIBS})

# ---- Create shared library ----

add_library(${LIBRARY_NAME} SHARED ${PROJECT_SOURCES} ${PROJECT_HEADERS})
target_link_libraries(${LIBRARY_NAME} PUBLIC ${CONAN_LIBS})

# ---- Create Entry Point ----

set_target_properties(
    ${LIBRARY_NAME}
      PROPERTIES 
        CXX_STANDARD ${CMAKE_CXX_STANDARD} 
        CXX_STANDARD_REQUIRED YES 
        CXX_EXTENSIONS NO
)

add_compile_definitions(DATA_PATH="${PROJECT_SOURCE_DIR}/../Data")

# Set up tests (see test/CMakeLists.txt).
if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()