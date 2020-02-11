#include <gtest/gtest.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include "Context.h"
#include "algorithms/pointcloud/LaserAggregator.h"

#define N 40
#define LASERS 1
#define NOISE_AMP 0.00
//#define VISUALIZE


float getRandFloat() {
    return static_cast<float>( ( ( ( rand()%200 ) - 100 ) / 100.0 ) * NOISE_AMP );
}

pcl::PointCloud<pcl::PointXYZ> getTestData() {
    pcl::PointCloud<pcl::PointXYZ> output;

    for(int i = 0 ; i < N ; i++) {
        double step = 2*M_PI / N;
        double angle = step * i;
        output.push_back({(float)std::cos(angle) + getRandFloat() + 2, (float)-sin(angle) + getRandFloat() , getRandFloat()});
    }
    return output;
}


class VisualizerAgg {

public:
    VisualizerAgg() {
        int argc = 0;
        char* argv[] = {0};

        ros::init(argc, argv, "autodrive_localmap");
        node_ = new ros::NodeHandle();
        pc_publisher_ = node_->advertise<sensor_msgs::PointCloud2>("pointcloud_aggregation", 0);
    }

    void publishPointcloud(pcl::PointCloud<pcl::PointXYZ>& pc) {
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(pc, msg);

        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = "origin";
        msg.header.seq = 0;
        pc_publisher_.publish(msg);
    }
private:

    ros::NodeHandle* node_;
    ros::Publisher pc_publisher_;
};


#ifdef VISUALIZE
VisualizerAgg visualizerLaser;
#endif


TEST(laser_aggregator, init) {
    auto context = AutoDrive::Context::getEmptyContext();
    auto aggregator = AutoDrive::Algorithms::LaserAggregator(context, LASERS, N);
}



TEST(laser_aggregator, forward_movement) {
    auto context = AutoDrive::Context::getEmptyContext();
    auto aggregator = AutoDrive::Algorithms::LaserAggregator(context, LASERS, N);

    auto data = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>(getTestData());

    AutoDrive::DataModels::LocalPosition startPose {{0,0,0}, {}, 0};
    AutoDrive::DataModels::LocalPosition endPose {{1,0,0}, {}, uint64_t(0.1e9)};
    auto poseDiff = endPose - startPose;

    aggregator.onNewLaserData(data, startPose, poseDiff, {});

#ifdef VISUALIZE
    auto result = aggregator.getAllAggregatedLasers();
    visualizerLaser.publishPointcloud(*result);
#endif
}



int main(int argc, char **argv){

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}