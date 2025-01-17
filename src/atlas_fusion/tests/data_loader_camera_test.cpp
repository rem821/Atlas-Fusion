/*
 * Copyright 2020 Brno University of Technology
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
 * OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "gtest/gtest.h"
#include <iostream>
#include <istream>

#include "data_loader/DataLoader.h"
#include "data_loader/CameraDataLoader.h"
#include "Context.h"

#define DATA_FOLDER TEST_FOLDER"test_data/"
#define TEST_ERR_TOLERANCE 1e-16

TEST(data_loader_camera_test, load_camera_rgb)
{
    auto context = AtlasFusion::Context::getEmptyContext();
    AtlasFusion::DataLoader::CameraDataLoader dataLoader(context, AtlasFusion::DataLoader::CameraIndentifier::kCameraLeftFront, "");
    dataLoader.loadData(DATA_FOLDER);
    EXPECT_EQ(dataLoader.getDataSize(),500);

    auto data = std::dynamic_pointer_cast<AtlasFusion::DataModels::CameraFrameDataModel>(dataLoader.getNextData());
    EXPECT_EQ(data->getType(), AtlasFusion::DataModels::DataModelTypes::kCameraDataModelType);
    EXPECT_EQ(data->getTimestamp(), 1568186745453388439);
    auto frame = data->getImage();
    EXPECT_EQ(frame.rows, 1200);
    EXPECT_EQ(frame.cols, 1920);

    data = std::dynamic_pointer_cast<AtlasFusion::DataModels::CameraFrameDataModel>(dataLoader.getNextData());
    EXPECT_EQ(data->getType(), AtlasFusion::DataModels::DataModelTypes::kCameraDataModelType);
    EXPECT_EQ(data->getTimestamp(), 1568186745553488239);
    frame = data->getImage();
    EXPECT_EQ(frame.rows, 1200);
    EXPECT_EQ(frame.cols, 1920);
}


TEST(data_loader_camera_test, load_ir_camera_ir)
{
    auto context = AtlasFusion::Context::getEmptyContext();
    AtlasFusion::DataLoader::CameraDataLoader dataLoader(context, AtlasFusion::DataLoader::CameraIndentifier::kCameraIr, "");
    dataLoader.loadData(DATA_FOLDER);
    EXPECT_EQ(dataLoader.getDataSize(),500);

    auto data = std::dynamic_pointer_cast<AtlasFusion::DataModels::CameraIrFrameDataModel>(dataLoader.getNextData());
    EXPECT_EQ(data->getType(), AtlasFusion::DataModels::DataModelTypes::kCameraIrDataModelType);
    EXPECT_EQ(data->getTimestamp(), 1568186745426675402);
    EXPECT_NEAR(data->getTemp().first, -29.65, TEST_ERR_TOLERANCE);
    EXPECT_NEAR(data->getTemp().second, 20.74, TEST_ERR_TOLERANCE);
    auto frame = data->getImage();
    EXPECT_EQ(frame.rows, 512);
    EXPECT_EQ(frame.cols, 640);

    data = std::dynamic_pointer_cast<AtlasFusion::DataModels::CameraIrFrameDataModel>(dataLoader.getNextData());
    EXPECT_EQ(data->getType(), AtlasFusion::DataModels::DataModelTypes::kCameraIrDataModelType);
    EXPECT_EQ(data->getTimestamp(), 1568186745463015536);
    EXPECT_NEAR(data->getTemp().first, -29.65, TEST_ERR_TOLERANCE);
    EXPECT_NEAR(data->getTemp().second, 20.74, TEST_ERR_TOLERANCE);
    frame = data->getImage();
    EXPECT_EQ(frame.rows, 512);
    EXPECT_EQ(frame.cols, 640);
}


TEST(data_loader_camera_test, load_camera_get_timestamp) {

    auto context = AtlasFusion::Context::getEmptyContext();
    AtlasFusion::DataLoader::CameraDataLoader dataLoader(context, AtlasFusion::DataLoader::CameraIndentifier::kCameraLeftFront, "");
    dataLoader.loadData(DATA_FOLDER);

    EXPECT_EQ(dataLoader.isOnEnd(), false);
    EXPECT_EQ(dataLoader.getLowestTimestamp(), 1568186745453388439);
    auto data = dataLoader.getNextData();
    EXPECT_EQ(dataLoader.getLowestTimestamp(), 1568186745553488239);
    data = dataLoader.getNextData();
    EXPECT_EQ(dataLoader.getLowestTimestamp(), 1568186745653588039);

    EXPECT_EQ(dataLoader.isOnEnd(), false);

    while(!dataLoader.isOnEnd()) {
        data = dataLoader.getNextData();
    }
    EXPECT_EQ(dataLoader.isOnEnd(), true);
}


TEST(data_loader_camera_test, load_camera_set_pose) {

    auto context = AtlasFusion::Context::getEmptyContext();
    AtlasFusion::DataLoader::CameraDataLoader dataLoader(context, AtlasFusion::DataLoader::CameraIndentifier::kCameraLeftFront, "");
    dataLoader.loadData(DATA_FOLDER);

    EXPECT_EQ(dataLoader.isOnEnd(), false);

    dataLoader.setPose(1568186758466362438);
    EXPECT_EQ(dataLoader.getLowestTimestamp(), 1568186758466362439);
    auto data = dataLoader.getNextData();
    EXPECT_EQ(dataLoader.getLowestTimestamp(), 1568186758566462239);
    data = dataLoader.getNextData();
    EXPECT_EQ(dataLoader.getLowestTimestamp(), 1568186758666562039);

    EXPECT_EQ(dataLoader.isOnEnd(), false);

    while(!dataLoader.isOnEnd()) {
        data = dataLoader.getNextData();
    }
    EXPECT_EQ(dataLoader.isOnEnd(), true);
}



TEST(data_loader_camera_test, load_camera_yolo_detections) {

    auto context = AtlasFusion::Context::getEmptyContext();
    AtlasFusion::DataLoader::CameraDataLoader dataLoader(context, AtlasFusion::DataLoader::CameraIndentifier::kCameraLeftFront, "");
    dataLoader.loadData(DATA_FOLDER);


    auto data = std::dynamic_pointer_cast<AtlasFusion::DataModels::CameraFrameDataModel>(dataLoader.getNextData());
    EXPECT_EQ(data->getType(), AtlasFusion::DataModels::DataModelTypes::kCameraDataModelType);
    EXPECT_EQ(data->getTimestamp(), 1568186745453388439);
    data = std::dynamic_pointer_cast<AtlasFusion::DataModels::CameraFrameDataModel>(dataLoader.getNextData());
    EXPECT_EQ(data->getType(), AtlasFusion::DataModels::DataModelTypes::kCameraDataModelType);
    EXPECT_EQ(data->getTimestamp(), 1568186745553488239);

    data = std::dynamic_pointer_cast<AtlasFusion::DataModels::CameraFrameDataModel>(dataLoader.getNextData());
    EXPECT_EQ(data->getType(), AtlasFusion::DataModels::DataModelTypes::kCameraDataModelType);
    EXPECT_EQ(data->getTimestamp(), 1568186745653588039);
    auto yoloDet = data->getYoloDetections();
    EXPECT_EQ(yoloDet.size(), 1);
    EXPECT_EQ(yoloDet.at(0)->getBoundingBox().x1_, 310);
    EXPECT_EQ(yoloDet.at(0)->getBoundingBox().y1_, 550);
    EXPECT_EQ(yoloDet.at(0)->getBoundingBox().x2_, 379);
    EXPECT_EQ(yoloDet.at(0)->getBoundingBox().y2_, 582);
    EXPECT_NEAR(yoloDet.at(0)->getDetectionConfidence(), 0.81889998912811279, TEST_ERR_TOLERANCE);
    EXPECT_NEAR(yoloDet.at(0)->getDetectionConfidence(), 0.9966999888420105, TEST_ERR_TOLERANCE);

    data = std::dynamic_pointer_cast<AtlasFusion::DataModels::CameraFrameDataModel>(dataLoader.getNextData());
    EXPECT_EQ(data->getType(), AtlasFusion::DataModels::DataModelTypes::kCameraDataModelType);
    EXPECT_EQ(data->getTimestamp(), 1568186745753687839);
    yoloDet = data->getYoloDetections();
    EXPECT_EQ(yoloDet.size(), 1);
    EXPECT_EQ(yoloDet.at(0)->getBoundingBox().x1_, 305);
    EXPECT_EQ(yoloDet.at(0)->getBoundingBox().y1_, 548);
    EXPECT_EQ(yoloDet.at(0)->getBoundingBox().x2_, 369);
    EXPECT_EQ(yoloDet.at(0)->getBoundingBox().y2_, 580);
    EXPECT_NEAR(yoloDet.at(0)->getDetectionConfidence(), 0.83819997310638428, TEST_ERR_TOLERANCE);
    EXPECT_NEAR(yoloDet.at(0)->getDetectionConfidence(), 0.99409997463226318, TEST_ERR_TOLERANCE);

    data = std::dynamic_pointer_cast<AtlasFusion::DataModels::CameraFrameDataModel>(dataLoader.getNextData());
    EXPECT_EQ(data->getType(), AtlasFusion::DataModels::DataModelTypes::kCameraDataModelType);
    EXPECT_EQ(data->getTimestamp(), 1568186745853787639);
    yoloDet = data->getYoloDetections();
    EXPECT_EQ(yoloDet.size(), 1);
    EXPECT_EQ(yoloDet.at(0)->getBoundingBox().x1_, 282);
    EXPECT_EQ(yoloDet.at(0)->getBoundingBox().y1_, 548);
    EXPECT_EQ(yoloDet.at(0)->getBoundingBox().x2_, 352);
    EXPECT_EQ(yoloDet.at(0)->getBoundingBox().y2_, 583);
    EXPECT_NEAR(yoloDet.at(0)->getDetectionConfidence(), 0.80430001020431519, TEST_ERR_TOLERANCE);
    EXPECT_NEAR(yoloDet.at(0)->getDetectionConfidence(), 0.99750000238418579, TEST_ERR_TOLERANCE);

    while(!dataLoader.isOnEnd()) {
        data = std::dynamic_pointer_cast<AtlasFusion::DataModels::CameraFrameDataModel>(dataLoader.getNextData());
        EXPECT_EQ(data->getType(), AtlasFusion::DataModels::DataModelTypes::kCameraDataModelType);

        yoloDet = data->getYoloDetections();
        if(yoloDet.size() > 1) {
            EXPECT_EQ(data->getTimestamp(), 1568186748756681839); // first time there are two detection at frame no. 58
            return;
        }
    }
    EXPECT_ANY_THROW("Error when loading multiple yolo detections for single frame");
}



int main(int argc, char **argv){

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}