#include "algorithms/pointcloud/PointCloudExtrapolator.h"
#include "local_map/Frames.h"

namespace AutoDrive::Algorithms {

    std::vector<std::shared_ptr<DataModels::PointCloudBatch>> PointCloudExtrapolator::splitPointCloudToBatches(
            std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> scan,
            DataModels::LocalPosition startPose,
            DataModels::LocalPosition poseDiff,
            rtl::RigidTf3D<double> sensorOffsetTf) {


        std::vector<std::shared_ptr<DataModels::PointCloudBatch>> output;

        auto singleBatchSize = static_cast<size_t>(std::ceil(static_cast<float>(scan->width) / noOfBatches_));
        auto timeOffset = startPose.getTimestamp();
        auto duration = poseDiff.getTimestamp();
        DataModels::LocalPosition endPose {startPose.getPosition() + poseDiff.getPosition(),
                                           startPose.getOrientation() * poseDiff.getOrientation(),
                                           startPose.getTimestamp() + poseDiff.getTimestamp()};

        size_t pointCnt = 0;
        for(size_t i = 0 ; i < noOfBatches_ ; i++) {

            double ratio = static_cast<double>(i) / noOfBatches_;
            auto pose = AutoDrive::DataModels::LocalPosition {
                    {poseDiff.getPosition().x() * (ratio), poseDiff.getPosition().y() * (ratio), poseDiff.getPosition().z() * (ratio)},
                    {poseDiff.getOrientation().slerp(rtl::Quaternion<double>::identity(), (float)(1-ratio))},
                    uint64_t(duration * (ratio))
            };

            rtl::RigidTf3D<double> movementCompensationTF{pose.getOrientation(), pose.getPosition()};
            uint64_t ts = timeOffset + static_cast<uint64_t>(ratio * duration);

            pcl::PointCloud<pcl::PointXYZ> points;
            points.reserve(singleBatchSize);

            // TODO: Avoid point copying
            for(size_t j = 0 ; j < singleBatchSize ; j++) {
                if(pointCnt < scan->width) {
                    points.push_back(scan->at(pointCnt++));
                }
            }

            rtl::RigidTf3D<double> toGlobelFrameTf{endPose.getOrientation(), endPose.getPosition()};
            rtl::RigidTf3D<double> startToEndTf{poseDiff.getOrientation(), poseDiff.getPosition()};
            auto finalTF = toGlobelFrameTf(startToEndTf.inverted()(movementCompensationTF(sensorOffsetTf)));

            // TODO: Avoid point copying
            auto batch = std::make_shared<DataModels::PointCloudBatch>(ts, points, LocalMap::Frames::kOrigin, finalTF);
            output.push_back(batch);
        }

        return output;
    }
}
