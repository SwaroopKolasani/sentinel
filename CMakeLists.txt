# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(sentinel_perception VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find Local & System Libraries ---

# Point directly to the local LibTorch directory
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vendor/libtorch")

# Point directly to the Open3D installation inside your Python venv
# This is the key change to fix the Open3D issue
set(Open3D_DIR "${CMAKE_CURRENT_SOURCE_DIR}/venv/lib/python3.12/site-packages/open3d/cmake")

# Find the packages
find_package(Torch REQUIRED)
find_package(Open3D REQUIRED)
find_package(PCL REQUIRED) # PCL is still found via Homebrew

# Set C++ flags for macOS
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION")
endif()

# --- Configure Executable ---

# Include directories
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${Open3D_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

# Add executable
add_executable(sentinel_app
    src/cpp/main.cpp
    src/cpp/core/geometric_refinement.cpp
    src/cpp/inference/model_loader.cpp
)

# Link libraries
target_link_libraries(sentinel_app
    PRIVATE
    ${TORCH_LIBRARIES}
    Open3D::Open3D
    PCL::common
    PCL::io
)