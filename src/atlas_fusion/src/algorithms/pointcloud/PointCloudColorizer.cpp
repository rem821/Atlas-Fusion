#include "algorithms/pointcloud/PointCloudColorizer.h"
#include "RustStyle.h"
#include "local_map/Frames.h"

namespace AtlasFusion::Algorithms {

    PointCloudColorizer::PointCloudColorizer(Context& context) :
            context_{context} {

    }


    void PointCloudColorizer::add_projector(std::shared_ptr<Projector> projector, DataLoader::CameraIndentifier id) {
        projectors_[id] = projector;
    }


    void PointCloudColorizer::update_rgb_camera_frame(std::shared_ptr<DataModels::CameraFrameDataModel> frame) {
        if (rgb_frames_.count(frame->getCameraIdentifier()) == 0) {
            rgb_frames_.insert({frame->getCameraIdentifier(), frame});
        } else {
            rgb_frames_[frame->getCameraIdentifier()] = frame;
        }
    }


    void PointCloudColorizer::update_ir_camera_frame(std::shared_ptr<DataModels::CameraIrFrameDataModel> frame) {
        if (ir_frames_.count(frame->getCameraIdentifier()) == 0) {
            ir_frames_.insert({frame->getCameraIdentifier(), frame});
        } else {
            ir_frames_[frame->getCameraIdentifier()] = frame;
        }
    }



    std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> PointCloudColorizer::colorize_point_cloud_rgb(std::vector<std::shared_ptr<DataModels::PointCloudBatch>> batches, const rtl::RigidTf3D<double>& origin_to_imu) {
        let camera_id = DataLoader::CameraIndentifier::kCameraLeftFront;
        if (rgb_frames_.count(camera_id) == 0) {
            return std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        }
        return colorize_point_cloud(batches, origin_to_imu, camera_id);
    }


    std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> PointCloudColorizer::colorize_point_cloud_ir(std::vector<std::shared_ptr<DataModels::PointCloudBatch>> batches, const rtl::RigidTf3D<double>& origin_to_imu) {
        let camera_id = DataLoader::CameraIndentifier::kCameraIr;
        if (ir_frames_.count(camera_id) == 0) {
            return std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        }
        return colorize_point_cloud(batches, origin_to_imu, camera_id);
    }


    std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> PointCloudColorizer::colorize_point_cloud(std::vector<std::shared_ptr<DataModels::PointCloudBatch>> batches,
                                                                                                 const rtl::RigidTf3D<double>& origin_to_imu,
                                                                                                 const DataLoader::CameraIndentifier& camera_id) {

        mut colorized_pc = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

        auto projector = projectors_[camera_id];
        auto cameraFrame = cameraIdentifierToFrame(camera_id);

        auto imuToCamera = context_.tfTree_.getTransformationForFrame(cameraFrame);
        rtl::RigidTf3D<double> originToCameraTf = (imuToCamera.inverted()(origin_to_imu.inverted()));


        pcl::PointCloud<pcl::PointXYZ> pc_in_origin_frame;
        pcl::PointCloud<pcl::PointXYZ> pc_in_camera_frame;
        for(const auto& batch : batches) {
            pc_in_camera_frame += *(batch->getTransformedPointsWithAnotherTF(originToCameraTf));
            pc_in_origin_frame += *(batch->getTransformedPoints());
        }

        std::vector<cv::Point3f> points3D;
        std::vector<cv::Point2f> points2D;
        std::vector<cv::Point3f> points3D_global_frame;
        points3D.reserve(pc_in_camera_frame.width * pc_in_camera_frame.height);
        size_t index = 0;
        for(const auto& pnt : pc_in_camera_frame ){
            if(pnt.z > 0 ) {
                points3D.push_back({pnt.x, pnt.y, pnt.z});
            }
            let point_in_origin = pc_in_origin_frame.at(index);
            points3D_global_frame.push_back({point_in_origin.x, point_in_origin.y, point_in_origin.z});
            index += 1;
        }

        let useDistMat = true;
        projector->projectPoints(points3D, points2D, useDistMat);

        if(points2D.size() != points3D.size()) {
            context_.logger_.error("Number of projected points does not corresponds with number of input points!");
            return colorized_pc;
        }


        std::shared_ptr<DataModels::CameraFrameDataModel> camera_frame;
        if (camera_id == DataLoader::CameraIndentifier::kCameraIr) {
            camera_frame = ir_frames_.at(camera_id)->asCameraFrame();
        } else {
            camera_frame = rgb_frames_.at(camera_id);
        }


        for (size_t i = 0 ; i < points2D.size() ; i++) {
            let point2D = points2D.at(i);
            if ( static_cast<int>(point2D.x) >= 0 &&
                 static_cast<int>(point2D.x) < camera_frame->getImage().cols &&
                 static_cast<int>(point2D.y) >= 0 &&
                 static_cast<int>(point2D.y) < camera_frame->getImage().rows ) {

                let point3D = rtl::Vector3D<double>{points3D.at(i).x, points3D.at(i).y, points3D.at(i).z};
                let point3D_in_origin = point3D.transformed(originToCameraTf.inverted());
                let pixel = camera_frame->getImage().at<cv::Vec3b>(cv::Point(point2D.x,point2D.y));
                mut p = pcl::PointXYZRGB{};
                p.x = point3D_in_origin.x();
                p.y = point3D_in_origin.y();
                p.z = point3D_in_origin.z();
                p.r = pixel[2];
                p.g = pixel[1];
                p.b = pixel[0];
                colorized_pc->push_back(p);
            }
        }
        return colorized_pc;
    }


    std::string PointCloudColorizer::cameraIdentifierToFrame(DataLoader::CameraIndentifier id) {
        switch(id){
            case DataLoader::CameraIndentifier::kCameraLeftFront:
                return LocalMap::Frames::kCameraLeftFront;
            case DataLoader::CameraIndentifier::kCameraLeftSide:
                return LocalMap::Frames::kCameraLeftSide;
            case DataLoader::CameraIndentifier::kCameraRightFront:
                return LocalMap::Frames::kCameraRightFront;
            case DataLoader::CameraIndentifier::kCameraRightSide:
                return LocalMap::Frames::kCameraRightSide;
            case DataLoader::CameraIndentifier::kCameraIr:
                return LocalMap::Frames::kCameraIr;
            case DataLoader::CameraIndentifier::kCameraVirtual:
                return LocalMap::Frames::kCameraVirtual;
            default:
                context_.logger_.warning("Unexpected camera ID type in depth map");
                return LocalMap::Frames::kOrigin;
        }
    }
}