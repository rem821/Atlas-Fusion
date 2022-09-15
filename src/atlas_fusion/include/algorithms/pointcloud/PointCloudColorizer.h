#pragma once

#include <memory>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <Context.h>
#include "algorithms/Projector.h"
#include "data_models/camera/CameraFrameDataModel.h"
#include "data_models/camera/CameraIrFrameDataModel.h"
#include "data_loader/DataLoaderIdentifiers.h"
#include "data_models/local_map/PointCloudBatch.h"

namespace AtlasFusion::Algorithms {

    class PointCloudColorizer {

    public:

        PointCloudColorizer() = delete;
        PointCloudColorizer(Context& context);

        void add_projector(std::shared_ptr<Projector> projector, DataLoader::CameraIndentifier id);
        void update_rgb_camera_frame(std::shared_ptr<DataModels::CameraFrameDataModel>);
        void update_ir_camera_frame(std::shared_ptr<DataModels::CameraIrFrameDataModel>);

        std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> colorize_point_cloud_rgb(std::vector<std::shared_ptr<DataModels::PointCloudBatch>> batches, const rtl::RigidTf3D<double>& origin_to_imu);
        std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> colorize_point_cloud_ir(std::vector<std::shared_ptr<DataModels::PointCloudBatch>> batches, const rtl::RigidTf3D<double>& origin_to_imu);

    protected:

        Context& context_;
        std::map<DataLoader::CameraIndentifier, std::shared_ptr<Projector>> projectors_{};
        std::map<DataLoader::CameraIndentifier, std::shared_ptr<DataModels::CameraFrameDataModel>> rgb_frames_;
        std::map<DataLoader::CameraIndentifier, std::shared_ptr<DataModels::CameraIrFrameDataModel>> ir_frames_;

        std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> colorize_point_cloud(std::vector<std::shared_ptr<DataModels::PointCloudBatch>> batches, const rtl::RigidTf3D<double>& origin_to_imu, const DataLoader::CameraIndentifier& camera_id);

        std::string cameraIdentifierToFrame(DataLoader::CameraIndentifier id);
    };
}

