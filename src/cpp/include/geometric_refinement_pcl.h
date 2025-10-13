#ifndef GEOMETRIC_REFINEMENT_PCL_H
#define GEOMETRIC_REFINEMENT_PCL_H

#include <vector>
#include <memory>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <Eigen/Dense>

namespace sentinel {

using PointT = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointT>;
using PointCloudPtr = PointCloud::Ptr;

struct GeometricConstraints {
    // Vehicle constraints
    double vehicle_min_points = 100;
    double vehicle_plane_threshold = 0.1;
    double vehicle_min_width = 1.0;
    double vehicle_max_width = 3.0;
    double vehicle_min_length = 2.0;
    double vehicle_max_length = 10.0;
    
    // Pedestrian constraints
    double pedestrian_min_points = 50;
    double pedestrian_max_width = 1.0;
    double pedestrian_min_height = 1.0;
    double pedestrian_max_height = 2.5;
    double pedestrian_compactness_threshold = 0.7;
    
    // RANSAC parameters
    int ransac_iterations = 1000;
    double ransac_distance_threshold = 0.05;
    int ransac_min_inliers = 30;
};

class GeometricRefinementModule {
public:
    GeometricRefinementModule(const GeometricConstraints& constraints);
    
    // Main refinement function
    std::vector<int> refineLabels(
        const PointCloudPtr& pointcloud,
        const std::vector<int>& initial_labels,
        const std::vector<int>& instance_ids
    );
    
    // Individual validation functions
    bool validateVehicle(const PointCloudPtr& cluster);
    bool validatePedestrian(const PointCloudPtr& cluster);
    bool validateCyclist(const PointCloudPtr& cluster);
    
private:
    GeometricConstraints constraints_;
    
    // Helper functions
    PointCloudPtr extractCluster(
        const PointCloudPtr& pointcloud,
        const std::vector<int>& indices
    );
    
    std::tuple<pcl::ModelCoefficients, pcl::PointIndices> fitPlaneRANSAC(
        const PointCloudPtr& cluster
    );
    
    void computeBoundingBox(
        const PointCloudPtr& cluster,
        Eigen::Vector3f& min_point,
        Eigen::Vector3f& max_point,
        Eigen::Vector3f& position,
        Eigen::Matrix3f& rotation
    );
    
    double computeCompactness(const PointCloudPtr& cluster);
    double computePointDensity(const PointCloudPtr& cluster);
};

} // namespace sentinel

#endif // GEOMETRIC_REFINEMENT_PCL_H