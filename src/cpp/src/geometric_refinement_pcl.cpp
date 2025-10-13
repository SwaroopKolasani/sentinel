#include "geometric_refinement_pcl.h"
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/centroid.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <algorithm>
#include <unordered_map>

namespace sentinel {

GeometricRefinementModule::GeometricRefinementModule(
    const GeometricConstraints& constraints)
    : constraints_(constraints) {}

std::vector<int> GeometricRefinementModule::refineLabels(
    const PointCloudPtr& pointcloud,
    const std::vector<int>& initial_labels,
    const std::vector<int>& instance_ids) {
    
    std::vector<int> refined_labels = initial_labels;
    
    // Group points by instance
    std::unordered_map<int, std::vector<int>> instance_indices;
    for (size_t i = 0; i < instance_ids.size(); ++i) {
        if (instance_ids[i] > 0) {  // Skip background (0)
            instance_indices[instance_ids[i]].push_back(i);
        }
    }
    
    // Process each instance
    for (const auto& [instance_id, indices] : instance_indices) {
        // Extract cluster
        PointCloudPtr cluster = extractCluster(pointcloud, indices);
        
        // Get the semantic class (assuming consistent within instance)
        int semantic_class = initial_labels[indices[0]];
        
        bool is_valid = false;
        
        // Validate based on semantic class
        switch (semantic_class) {
            case 1:  // Car
            case 4:  // Truck
            case 5:  // Other vehicle
                is_valid = validateVehicle(cluster);
                break;
            
            case 6:  // Person
                is_valid = validatePedestrian(cluster);
                break;
            
            case 7:  // Bicyclist
            case 8:  // Motorcyclist
                is_valid = validateCyclist(cluster);
                break;
            
            default:
                is_valid = true;  // Don't validate other classes
                break;
        }
        
        // If validation failed, relabel as unclassified
        if (!is_valid) {
            for (int idx : indices) {
                refined_labels[idx] = 0;  // 0 = unclassified/clutter
            }
        }
    }
    
    return refined_labels;
}

PointCloudPtr GeometricRefinementModule::extractCluster(
    const PointCloudPtr& pointcloud,
    const std::vector<int>& indices) {
    
    PointCloudPtr cluster(new PointCloud);
    pcl::PointIndices::Ptr cluster_indices(new pcl::PointIndices);
    cluster_indices->indices = indices;
    
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(pointcloud);
    extract.setIndices(cluster_indices);
    extract.setNegative(false);
    extract.filter(*cluster);
    
    return cluster;
}

bool GeometricRefinementModule::validateVehicle(const PointCloudPtr& cluster) {
    
    // Check minimum point count
    if (cluster->points.size() < constraints_.vehicle_min_points) {
        return false;
    }
    
    // Compute bounding box using PCL
    pcl::MomentOfInertiaEstimation<PointT> feature_extractor;
    feature_extractor.setInputCloud(cluster);
    feature_extractor.compute();
    
    PointT min_point_OBB, max_point_OBB, position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    
    feature_extractor.getOBB(min_point_OBB, max_point_OBB, 
                             position_OBB, rotational_matrix_OBB);
    
    // Calculate dimensions
    float width = std::abs(max_point_OBB.x - min_point_OBB.x);
    float length = std::abs(max_point_OBB.y - min_point_OBB.y);
    float height = std::abs(max_point_OBB.z - min_point_OBB.z);
    
    // Ensure width < length
    if (width > length) std::swap(width, length);
    
    // Check dimensions
    if (width < constraints_.vehicle_min_width || 
        width > constraints_.vehicle_max_width ||
        length < constraints_.vehicle_min_length || 
        length > constraints_.vehicle_max_length) {
        return false;
    }
    
    // Extract bottom points (lower 30% of height)
    PointCloudPtr bottom_cluster(new PointCloud);
    float min_z = min_point_OBB.z;
    float height_threshold = min_z + 0.3f * height;
    
    for (const auto& point : cluster->points) {
        if (point.z <= height_threshold) {
            bottom_cluster->points.push_back(point);
        }
    }
    bottom_cluster->width = bottom_cluster->points.size();
    bottom_cluster->height = 1;
    bottom_cluster->is_dense = true;
    
    // Fit plane to bottom points using RANSAC
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    
    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(constraints_.ransac_iterations);
    seg.setDistanceThreshold(constraints_.ransac_distance_threshold);
    seg.setInputCloud(bottom_cluster);
    seg.segment(*inliers, *coefficients);
    
    if (inliers->indices.empty()) {
        return false;
    }
    
    // Check if plane is horizontal (normal should be close to vertical)
    Eigen::Vector3f normal(coefficients->values[0], 
                          coefficients->values[1], 
                          coefficients->values[2]);
    normal.normalize();
    
    float vertical_alignment = std::abs(normal.dot(Eigen::Vector3f(0, 0, 1)));
    
    if (vertical_alignment < 0.8f) {  // Threshold for horizontal plane
        return false;
    }
    
    // Check inlier ratio
    float inlier_ratio = static_cast<float>(inliers->indices.size()) / 
                        bottom_cluster->points.size();
    
    if (inlier_ratio < 0.6f) {  // At least 60% should fit the plane
        return false;
    }
    
    return true;
}

bool GeometricRefinementModule::validatePedestrian(const PointCloudPtr& cluster) {
    
    // Check minimum point count
    if (cluster->points.size() < constraints_.pedestrian_min_points) {
        return false;
    }
    
    // Compute bounding box
    pcl::MomentOfInertiaEstimation<PointT> feature_extractor;
    feature_extractor.setInputCloud(cluster);
    feature_extractor.compute();
    
    PointT min_point_AABB, max_point_AABB;
    feature_extractor.getAABB(min_point_AABB, max_point_AABB);
    
    // Check dimensions
    float width = std::min(max_point_AABB.x - min_point_AABB.x,
                           max_point_AABB.y - min_point_AABB.y);
    float height = max_point_AABB.z - min_point_AABB.z;
    
    if (width > constraints_.pedestrian_max_width ||
        height < constraints_.pedestrian_min_height ||
        height > constraints_.pedestrian_max_height) {
        return false;
    }
    
    // Check vertical elongation
    if (height < 1.5f * width) {
        return false;
    }
    
    // Compute compactness
    double compactness = computeCompactness(cluster);
    if (compactness < constraints_.pedestrian_compactness_threshold) {
        return false;
    }
    
    return true;
}

bool GeometricRefinementModule::validateCyclist(const PointCloudPtr& cluster) {
    // Similar to pedestrian but with relaxed constraints
    
    if (cluster->points.size() < constraints_.pedestrian_min_points) {
        return false;
    }
    
    pcl::MomentOfInertiaEstimation<PointT> feature_extractor;
    feature_extractor.setInputCloud(cluster);
    feature_extractor.compute();
    
    PointT min_point, max_point;
    feature_extractor.getAABB(min_point, max_point);
    
    float width = std::min(max_point.x - min_point.x,
                           max_point.y - min_point.y);
    float height = max_point.z - min_point.z;
    
    // Cyclists can be wider than pedestrians
    if (width > constraints_.pedestrian_max_width * 1.5f ||
        height < constraints_.pedestrian_min_height * 0.8f) {
        return false;
    }
    
    return height > width;  // Still should be taller than wide
}

double GeometricRefinementModule::computeCompactness(const PointCloudPtr& cluster) {
    
    // Compute convex hull
    pcl::ConvexHull<PointT> hull;
    hull.setInputCloud(cluster);
    hull.setDimension(3);
    
    PointCloudPtr hull_points(new PointCloud);
    hull.reconstruct(*hull_points);
    
    // Simplified compactness: ratio of points to hull points
    return static_cast<double>(cluster->size()) / 
           (hull_points->size() + 1e-6);
}

double GeometricRefinementModule::computePointDensity(const PointCloudPtr& cluster) {
    
    PointT min_pt, max_pt;
    pcl::getMinMax3D(*cluster, min_pt, max_pt);
    
    double volume = (max_pt.x - min_pt.x) * 
                   (max_pt.y - min_pt.y) * 
                   (max_pt.z - min_pt.z);
    
    return static_cast<double>(cluster->size()) / (volume + 1e-6);
}

} // namespace sentinel