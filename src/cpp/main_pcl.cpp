#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include "geometric_refinement_pcl.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <pointcloud_path>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string pointcloud_path = argv[2];

    try {
        // Load PyTorch model
        std::cout << "Loading model from: " << model_path << std::endl;
        torch::jit::script::Module model;
        model = torch::jit::load(model_path);
        model.eval();
        
        // Check device availability
        torch::Device device(torch::kCPU);  // Use CPU for M1 Mac
        model.to(device);

        // Load point cloud using PCL
        std::cout << "Loading point cloud from: " << pointcloud_path << std::endl;
        sentinel::PointCloudPtr pointcloud(new sentinel::PointCloud);
        
        // Try different formats
        if (pcl::io::loadPCDFile<sentinel::PointT>(pointcloud_path, *pointcloud) == -1 &&
            pcl::io::loadPLYFile<sentinel::PointT>(pointcloud_path, *pointcloud) == -1) {
            
            // If PCD/PLY fails, try loading as binary KITTI format
            std::ifstream file(pointcloud_path, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open point cloud file" << std::endl;
                return -1;
            }
            
            // Read KITTI binary format
            file.seekg(0, std::ios::end);
            size_t file_size = file.tellg();
            size_t num_points = file_size / (4 * sizeof(float)); // XYZI
            file.seekg(0, std::ios::beg);
            
            std::vector<float> buffer(num_points * 4);
            file.read(reinterpret_cast<char*>(buffer.data()), file_size);
            file.close();
            
            // Convert to PCL point cloud
            pointcloud->width = num_points;
            pointcloud->height = 1;
            pointcloud->is_dense = false;
            pointcloud->points.resize(num_points);
            
            for (size_t i = 0; i < num_points; ++i) {
                pointcloud->points[i].x = buffer[i * 4 + 0];
                pointcloud->points[i].y = buffer[i * 4 + 1];
                pointcloud->points[i].z = buffer[i * 4 + 2];
                // Ignore intensity (buffer[i * 4 + 3])
            }
        }
        
        std::cout << "Loaded " << pointcloud->points.size() << " points" << std::endl;

        // Convert to tensor
        int num_points = pointcloud->points.size();
        torch::Tensor points_tensor = torch::zeros({1, num_points, 3});
        for (int i = 0; i < num_points; ++i) {
            points_tensor[0][i][0] = pointcloud->points[i].x;
            points_tensor[0][i][1] = pointcloud->points[i].y;
            points_tensor[0][i][2] = pointcloud->points[i].z;
        }
        points_tensor = points_tensor.to(device);

        // Run inference
        std::cout << "Running PointNet++ inference..." << std::endl;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(points_tensor);
        
        torch::NoGradGuard no_grad;
        auto output = model.forward(inputs).toTensor();
        
        // Get predictions
        auto predictions = output.argmax(-1);
        predictions = predictions.to(torch::kCPU);

        // Convert to vector
        std::vector<int> initial_labels(num_points);
        auto pred_accessor = predictions.accessor<int64_t, 2>();
        for (int i = 0; i < num_points; ++i) {
            initial_labels[i] = pred_accessor[0][i];
        }

        // Apply geometric refinement
        std::cout << "Applying geometric refinement..." << std::endl;
        sentinel::GeometricConstraints constraints;
        sentinel::GeometricRefinementModule refiner(constraints);
        
        // For now, create dummy instance IDs
        std::vector<int> instance_ids(num_points, 0);
        
        auto refined_labels = refiner.refineLabels(
            pointcloud, initial_labels, instance_ids
        );

        // Count changes
        int num_changed = 0;
        for (size_t i = 0; i < initial_labels.size(); ++i) {
            if (initial_labels[i] != refined_labels[i]) {
                num_changed++;
            }
        }
        std::cout << "Refinement changed " << num_changed << " labels" << std::endl;

        // Save results with colors based on refined labels
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(
            new pcl::PointCloud<pcl::PointXYZRGB>
        );
        colored_cloud->points.resize(pointcloud->points.size());
        
        for (size_t i = 0; i < pointcloud->points.size(); ++i) {
            colored_cloud->points[i].x = pointcloud->points[i].x;
            colored_cloud->points[i].y = pointcloud->points[i].y;
            colored_cloud->points[i].z = pointcloud->points[i].z;
            
            // Color by label
            int label = refined_labels[i];
            colored_cloud->points[i].r = (label * 50) % 255;
            colored_cloud->points[i].g = (label * 100) % 255;
            colored_cloud->points[i].b = (label * 150) % 255;
        }
        
        // Save output
        std::string output_path = "refined_pointcloud.ply";
        pcl::io::savePLYFileBinary(output_path, *colored_cloud);
        std::cout << "Saved refined point cloud to: " << output_path << std::endl;

        std::cout << "SENTINEL processing complete!" << std::endl;
        
    } catch (const c10::Error& e) {
        std::cerr << "PyTorch error: " << e.msg() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}