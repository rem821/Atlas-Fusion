#include "algorithms/pointcloud/PointCloudAggregator.h"

namespace AutoDrive::Algorithms {

    void PointCloudAggregator::addPointCloudBatches(std::vector<std::shared_ptr<DataModels::PointCloudBatch>> batches) {
        for (const auto batch : batches) {
            batchQueue_.push_back(batch);
        }
    }


    void PointCloudAggregator::filterOutBatches(uint64_t currentTime) {
        while(batchQueue_.size() > 0) {
            auto timeDiff = static_cast<double>((currentTime - batchQueue_.front()->getTimestamp()))*1e-9;
            if ( (timeDiff > aggregationTime_ )) {
                batchQueue_.pop_front();
            } else {
                break;
            }
        }
        std::cout << "batchSize: " << batchQueue_.size() << std::endl;
    }


    std::vector<std::shared_ptr<DataModels::PointCloudBatch>> PointCloudAggregator::getAllBatches() {
        std::vector<std::shared_ptr<DataModels::PointCloudBatch>> output;
        output.reserve(batchQueue_.size());

        for(auto it = batchQueue_.begin(); it < batchQueue_.end() ; it++) {
            output.push_back(*it);
        }

        return output;
    }


    std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> PointCloudAggregator::getAggregatedPointCloud() {

        auto output = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        for (const auto& batch : batchQueue_) {
            // TODO: Avoid using + operator
            *output += batch->getTransformedPoints();
        }

        return output;
    }

}
