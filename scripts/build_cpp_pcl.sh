#!/bin/bash

# Build C++ components with PCL for Project SENTINEL
echo "=========================================="
echo "Building C++ Components with PCL"
echo "=========================================="

# Check dependencies
echo "Checking dependencies..."

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake is not installed"
    echo "Install with: brew install cmake"
    exit 1
fi

# Check for PCL
if ! pkg-config --exists pcl_common-1.10 2>/dev/null; then
    echo "Warning: PCL might not be installed or detected"
    echo "Install with: brew install pcl"
fi

# Check for Eigen
if ! pkg-config --exists eigen3 2>/dev/null; then
    echo "Warning: Eigen3 might not be installed"
    echo "Install with: brew install eigen"
fi

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Download LibTorch if not present
LIBTORCH_PATH="/usr/local/libtorch"
if [ ! -d "$LIBTORCH_PATH" ]; then
    echo "LibTorch not found. Downloading for macOS ARM64..."
    cd /tmp
    wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.0.1.zip
    unzip -q libtorch-macos-arm64-2.0.1.zip
    sudo mv libtorch /usr/local/
    cd -
fi

# Configure with CMake
echo "Configuring with CMake..."
cmake ../src/cpp \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_CXX_STANDARD=17

# Check if configuration was successful
if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed"
    exit 1
fi

# Build
echo "Building..."
make -j$(sysctl -n hw.ncpu)

# Check if build was successful
if [ $? -ne 0 ]; then
    echo "Error: Build failed"
    exit 1
fi

echo "=========================================="
echo "Build complete!"
echo "Executable: build/sentinel"
echo "=========================================="

# Create run script
cat > ../run_sentinel.sh << 'EOF'
#!/bin/bash
# Run SENTINEL C++ application
if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_path> <pointcloud_path>"
    echo "Example: $0 models/sentinel_model.pt data/sample.bin"
    exit 1
fi

./build/sentinel "$1" "$2"
EOF

chmod +x ../run_sentinel.sh
echo "Created run script: ./run_sentinel.sh"