cmake_minimum_required(VERSION 3.8)

# ---- Project ----

project(cpp VERSION 1.2.3.4 LANGUAGES CXX)
set(LIBRARY_NAME example)
set(CMAKE_CXX_STANDARD 17)

# ---- Add source files ----
file(GLOB_RECURSE PROJECT_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE PROJECT_HEADERS "${CMAKE_SOURCE_DIR}/include/*.hpp")

include_directories("${CMAKE_SOURCE_DIR}/include")

# ---- Create shared library ----
add_library(${LIBRARY_NAME} SHARED ${PROJECT_SOURCES} ${PROJECT_HEADERS})

# ---- Create Entry Point ----
set_target_properties(
    ${LIBRARY_NAME}
      PROPERTIES 
        CXX_STANDARD ${CMAKE_CXX_STANDARD} 
        CXX_STANDARD_REQUIRED YES 
        CXX_EXTENSIONS NO
)

add_compile_definitions(DATA_PATH="${PROJECT_SOURCE_DIR}/../Data")