cmake_minimum_required(VERSION 3.14)
project(sentinel LANGUAGES CXX)

# ============================================================
# Compiler and Build Settings
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# For macOS M1 / Apple Silicon
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
endif()

# ============================================================
# Dependencies
# ============================================================

# ---- PCL ----
find_package(PCL REQUIRED COMPONENTS common io filters segmentation features)
if(NOT PCL_FOUND)
    message(FATAL_ERROR "PCL not found. Install via: brew install pcl")
endif()

# ---- LibTorch ----
set(Torch_DIR "/usr/local/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
if(NOT Torch_FOUND)
    message(FATAL_ERROR "LibTorch not found. Download from pytorch.org and extract to /usr/local/libtorch")
endif()

# ---- Eigen ----
find_package(Eigen3 REQUIRED)
if(NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 not found. Install via: brew install eigen")
endif()

# ============================================================
# Include Directories
# ============================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/include
    ${PCL_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# ============================================================
# Source Files
# ============================================================
set(SOURCES
    src/cpp/src/main_pcl.cpp
    src/cpp/src/geometric_refinement_pcl.cpp
)

# ============================================================
# Executable Target
# ============================================================
add_executable(sentinel ${SOURCES})

# ============================================================
# Link Libraries
# ============================================================
target_link_libraries(sentinel
    ${PCL_LIBRARIES}
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
)

# ============================================================
# Post-Build Info
# ============================================================
message(STATUS "===================================================")
message(STATUS " Sentinel Build Configuration")
message(STATUS " C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS " Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS " Architecture: ${CMAKE_OSX_ARCHITECTURES}")
message(STATUS "===================================================")
message(STATUS " Torch Include: ${TORCH_INCLUDE_DIRS}")
message(STATUS " PCL Include: ${PCL_INCLUDE_DIRS}")
message(STATUS " Eigen Include: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "===================================================")
