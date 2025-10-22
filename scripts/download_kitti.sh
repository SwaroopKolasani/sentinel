#!/bin/bash

# Download KITTI odometry dataset
echo "=========================================="
echo "Downloading KITTI Odometry Dataset"
echo "=========================================="

# Create data directory
DATA_DIR="./data/kitti"
mkdir -p $DATA_DIR
cd $DATA_DIR

# Function to download with wget
download_file() {
    local url=$1
    local filename=$2
    
    if [ -f "$filename" ]; then
        echo "$filename already exists, skipping..."
    else
        echo "Downloading $filename..."
        wget -c "$url" -O "$filename"
    fi
}

# Note: You need to register at http://www.cvlibs.net/datasets/kitti/eval_odometry.php
# and get the actual download links

echo "Please ensure you have registered at KITTI website and have the download links."
echo "Update this script with the actual URLs from KITTI website."

# Example structure (replace with actual URLs after registration)
# download_file "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip" "data_odometry_velodyne.zip"
# download_file "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_labels.zip" "data_odometry_labels.zip"

# Extract files
echo "Extracting datasets..."
for zip_file in *.zip; do
    if [ -f "$zip_file" ]; then
        echo "Extracting $zip_file..."
        unzip -q "$zip_file"
    fi
done

echo "=========================================="
echo "Download complete!"
echo "Data location: $DATA_DIR"
echo "=========================================="