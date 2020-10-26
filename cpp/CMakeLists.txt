cmake_minimum_required(VERSION 3.8)

# ---- Project ----

project(cpp VERSION 1.2.3.4 LANGUAGES CXX)
set(LIBRARY_NAME example)
set(PROJECT_CXX_STANDARD 11)

# ---- Add source files ----
file(GLOB_RECURSE PROJECT_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

# ---- Create shared library ----
add_library(${LIBRARY_NAME} SHARED ${PROJECT_SOURCES})

# ---- Create Entry Point ----
set_target_properties(
    ${LIBRARY_NAME}
      PROPERTIES 
        CXX_STANDARD ${PROJECT_CXX_STANDARD} 
        CXX_STANDARD_REQUIRED YES 
        CXX_EXTENSIONS NO
)
