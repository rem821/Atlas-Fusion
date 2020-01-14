#pragma once

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include "Context.h"

namespace AutoDrive::Algorithms {

    class PointCloudProcessor {

    public:

        PointCloudProcessor(Context& context, float leafSize)
        : context_{context}
        , leafSize_{leafSize} {

        }

        std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> downsamplePointCloud(std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> input);

    private:

        Context& context_;

        float leafSize_;
    };

}