#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <torch/torch.h>
#include <torch/script.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include "geometric_refinement_pcl.h"

namespace sentinel {

class SentinelSystem {
private:
    torch::jit::script::Module model_;
    torch::Device device_;
    GeometricRefinementModule refiner_;
    bool model_loaded_ = false;

public:
    SentinelSystem() 
        : device_(torch::kCPU),
          refiner_(GeometricConstraints()) {
        // Check CUDA availability (though on M1 Mac it won't be available)
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available, using GPU" << std::endl;
            device_ = torch::Device(torch::kCUDA);
        } else {
            std::cout << "Using CPU (Apple M1)" << std::endl;
        }
    }

    bool loadModel(const std::string& model_path) {
        try {
            std::cout << "Loading model from: " << model_path << std::endl;
            model_ = torch::jit::load(model_path);
            model_.to(device_);
            model_.eval();
            model_loaded_ = true;
            std::cout << "Model loaded successfully" << std::endl;
            return true;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.msg() << std::endl;
            return false;
        }
    }

    bool loadPointCloudKITTI(const std::string& filepath, 
                            sentinel::PointCloudPtr& cloud) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            return false;
        }

        // Get file size
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        size_t num_points = file_size / (4 * sizeof(float)); // XYZI format
        file.seekg(0, std::ios::beg);

        // Read binary data
        std::vector<float> buffer(num_points * 4);
        file.read(reinterpret_cast<char*>(buffer.data()), file_size);
        file.close();

        // Convert to PCL point cloud
        cloud->width = num_points;
        cloud->height = 1;
        cloud->is_dense = false;
        cloud->points.resize(num_points);

        for (size_t i = 0; i < num_points; ++i) {
            cloud->points[i].x = buffer[i * 4 + 0];
            cloud->points[i].y = buffer[i * 4 + 1];
            cloud->points[i].z = buffer[i * 4 + 2];
            // Intensity at buffer[i * 4 + 3] is ignored for now
        }

        std::cout << "Loaded " << num_points << " points from KITTI file" << std::endl;
        return true;
    }

    std::vector<int> runInference(const sentinel::PointCloudPtr& cloud) {
        if (!model_loaded_) {
            throw std::runtime_error("Model not loaded");
        }

        int num_points = cloud->points.size();
        
        // Convert point cloud to tensor
        torch::Tensor points_tensor = torch::zeros({1, num_points, 4});
        
        for (int i = 0; i < num_points; ++i) {
            points_tensor[0][i][0] = cloud->points[i].x;
            points_tensor[0][i][1] = cloud->points[i].y;
            points_tensor[0][i][2] = cloud->points[i].z;
            points_tensor[0][i][3] = 1.0;  // Dummy intensity
        }
        
        points_tensor = points_tensor.to(device_);

        // Run inference
        std::cout << "Running PointNet++ inference..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(points_tensor);
        
        torch::NoGradGuard no_grad;
        auto output = model_.forward(inputs).toTensor();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
        
        // Get predictions
        auto predictions = output.argmax(-1);  // [1, N]
        predictions = predictions.to(torch::kCPU);
        
        // Convert to vector
        std::vector<int> labels(num_points);
        auto pred_accessor = predictions.accessor<int64_t, 2>();
        for (int i = 0; i < num_points; ++i) {
            labels[i] = static_cast<int>(pred_accessor[0][i]);
        }
        
        return labels;
    }

    std::vector<int> refineLabels(const sentinel::PointCloudPtr& cloud,
                                  const std::vector<int>& initial_labels) {
        std::cout << "Applying geometric refinement..." << std::endl;
        
        // For demonstration, create dummy instance IDs
        // In practice, you would use clustering to get instance segmentation
        std::vector<int> instance_ids(initial_labels.size(), 0);
        
        // Simple instance assignment based on connected components
        // This is a placeholder - implement proper clustering in production
        int current_instance = 1;
        for (size_t i = 0; i < initial_labels.size(); ++i) {
            if (initial_labels[i] > 0 && initial_labels[i] < 9) {  // Object classes
                instance_ids[i] = current_instance++;
                if (current_instance > 1000) current_instance = 1;  // Reset
            }
        }
        
        auto refined = refiner_.refineLabels(cloud, initial_labels, instance_ids);
        
        // Count changes
        int num_changed = 0;
        for (size_t i = 0; i < initial_labels.size(); ++i) {
            if (initial_labels[i] != refined[i]) {
                num_changed++;
            }
        }
        
        std::cout << "Refinement changed " << num_changed 
                  << " labels (" << (100.0 * num_changed / initial_labels.size()) 
                  << "%)" << std::endl;
        
        return refined;
    }

    void saveColoredPointCloud(const sentinel::PointCloudPtr& cloud,
                               const std::vector<int>& labels,
                               const std::string& output_path) {
        // Create colored point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(
            new pcl::PointCloud<pcl::PointXYZRGB>
        );
        
        colored_cloud->points.resize(cloud->points.size());
        colored_cloud->width = cloud->width;
        colored_cloud->height = cloud->height;
        colored_cloud->is_dense = cloud->is_dense;
        
        // Color map for 20 classes
        const uint8_t color_map[20][3] = {
            {0, 0, 0},        // 0: unlabeled - black
            {255, 0, 0},      // 1: car - red
            {0, 255, 0},      // 2: bicycle - green
            {0, 0, 255},      // 3: motorcycle - blue
            {255, 255, 0},    // 4: truck - yellow
            {255, 0, 255},    // 5: other-vehicle - magenta
            {0, 255, 255},    // 6: person - cyan
            {128, 0, 0},      // 7: bicyclist - dark red
            {0, 128, 0},      // 8: motorcyclist - dark green
            {128, 128, 128},  // 9: road - gray
            {64, 64, 64},     // 10: parking - dark gray
            {192, 192, 192},  // 11: sidewalk - light gray
            {128, 64, 0},     // 12: other-ground - brown
            {128, 0, 128},    // 13: building - purple
            {64, 64, 128},    // 14: fence - blue-gray
            {0, 128, 64},     // 15: vegetation - green
            {64, 128, 0},     // 16: trunk - olive
            {128, 128, 0},    // 17: terrain - olive
            {255, 128, 0},    // 18: pole - orange
            {255, 255, 128},  // 19: traffic-sign - light yellow
        };
        
        for (size_t i = 0; i < cloud->points.size(); ++i) {
            colored_cloud->points[i].x = cloud->points[i].x;
            colored_cloud->points[i].y = cloud->points[i].y;
            colored_cloud->points[i].z = cloud->points[i].z;
            
            int label = labels[i] % 20;  // Ensure within range
            colored_cloud->points[i].r = color_map[label][0];
            colored_cloud->points[i].g = color_map[label][1];
            colored_cloud->points[i].b = color_map[label][2];
        }
        
        // Save as PLY
        pcl::io::savePLYFileBinary(output_path, *colored_cloud);
        std::cout << "Saved colored point cloud to: " << output_path << std::endl;
    }

    void printStatistics(const std::vector<int>& labels) {
        // Count points per class
        std::vector<int> class_counts(20, 0);
        for (int label : labels) {
            if (label >= 0 && label < 20) {
                class_counts[label]++;
            }
        }
        
        const char* class_names[20] = {
            "unlabeled", "car", "bicycle", "motorcycle", "truck",
            "other-vehicle", "person", "bicyclist", "motorcyclist", "road",
            "parking", "sidewalk", "other-ground", "building", "fence",
            "vegetation", "trunk", "terrain", "pole", "traffic-sign"
        };
        
        std::cout << "\nClass Distribution:" << std::endl;
        std::cout << "==================" << std::endl;
        for (int i = 0; i < 20; ++i) {
            if (class_counts[i] > 0) {
                double percentage = 100.0 * class_counts[i] / labels.size();
                std::cout << std::setw(15) << class_names[i] << ": " 
                         << std::setw(8) << class_counts[i] 
                         << " points (" << std::fixed << std::setprecision(1) 
                         << percentage << "%)" << std::endl;
            }
        }
    }
};

} // namespace sentinel

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <pointcloud_path>" << std::endl;
        std::cerr << "Example: " << argv[0] << " models/sentinel_model.pt data/000000.bin" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string pointcloud_path = argv[2];

    try {
        std::cout << "\n========================================" << std::endl;
        std::cout << "       PROJECT SENTINEL v1.0" << std::endl;
        std::cout << "========================================\n" << std::endl;

        sentinel::SentinelSystem system;
        
        // Load model
        if (!system.loadModel(model_path)) {
            return -1;
        }
        
        // Load point cloud
        sentinel::PointCloudPtr cloud(new sentinel::PointCloud);
        
        std::cout << "\nLoading point cloud from: " << pointcloud_path << std::endl;
        
        // Try different formats
        bool loaded = false;
        if (pointcloud_path.find(".bin") != std::string::npos) {
            // KITTI binary format
            loaded = system.loadPointCloudKITTI(pointcloud_path, cloud);
        } else if (pcl::io::loadPCDFile<sentinel::PointT>(pointcloud_path, *cloud) == 0) {
            loaded = true;
            std::cout << "Loaded PCD file with " << cloud->points.size() << " points" << std::endl;
        } else if (pcl::io::loadPLYFile<sentinel::PointT>(pointcloud_path, *cloud) == 0) {
            loaded = true;
            std::cout << "Loaded PLY file with " << cloud->points.size() << " points" << std::endl;
        }
        
        if (!loaded || cloud->points.empty()) {
            std::cerr << "Failed to load point cloud or empty point cloud" << std::endl;
            return -1;
        }
        
        // Run inference
        std::cout << "\n--- Stage 1: Deep Learning Inference ---" << std::endl;
        auto initial_labels = system.runInference(cloud);
        
        // Apply geometric refinement
        std::cout << "\n--- Stage 2: Geometric Refinement ---" << std::endl;
        auto refined_labels = system.refineLabels(cloud, initial_labels);
        
        // Print statistics
        system.printStatistics(refined_labels);
        
        // Save results
        std::string output_path = "sentinel_output.ply";
        system.saveColoredPointCloud(cloud, refined_labels, output_path);
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "       PROCESSING COMPLETE" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Output saved to: " << output_path << std::endl;
        
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch error: " << e.msg() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}