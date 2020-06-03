#pragma once

#include <iostream>
#include <fstream>

#include "Context.h"

#include "data_models/yolo/YoloDetection.h"
#include "data_models/camera/CameraIrFrameDataModel.h"

namespace AutoDrive::DataWriters {

    /**
     * Yolo Detection Writer is used to store reprojected neural networks detections from one camera to another on the
     * local disk storage
     */
    class YoloDetectionWriter {

    public:

        YoloDetectionWriter() = delete;

        /**
         * Constructor
         * @param context global services container (timestamp, logging, etc.)
         * @param destinationDir folder in which the reprojected detection will be stored
         * @param destinationFile the file name into which the writer will store the data
         */
        YoloDetectionWriter(Context& context, std::string destinationDir, std::string destinationFile)
        : context_{context}
        , destinationDir_{std::move(destinationDir)}
        , destinationFile_{std::move(destinationFile)} {

        }

        /**
         * Method writes given detections into the file
         * @param detections vector of the detection to be writen
         * @param frameNo frame number that detections corresponds to
         */
        void writeDetections(std::shared_ptr<std::vector<DataModels::YoloDetection>> detections, size_t frameNo);

        /**
         * Method writes data in the format COCO dataset, so it is compatible with YOLO input
         * @param detections detections to be writen
         * @param frameNo frame number that detections corresponds to
         * @param image_width image width
         * @param image_height image height
         */
        void writeDetectionsAsTrainData(std::shared_ptr<std::vector<DataModels::YoloDetection>> detections, size_t frameNo, int image_width, int image_height);

        /**
         * Method stores IR image on the dist in the format that is compatible with YOLO training data
         * @param frame IR image frame
         * @param frameNo IR image frame number
         */
        void writeIRImageAsTrainData(std::shared_ptr<DataModels::CameraIrFrameDataModel> frame, size_t frameNo);

        /**
         * Changes location where the data should be stored
         * @param destinationDir storage folder
         * @param destinationFile storage file name
         */
        void changeDestinationFile(std::string destinationDir, std::string destinationFile);

    private:

        Context& context_;
        std::string destinationDir_;
        std::string destinationFile_;

        std::ofstream outputFile_;

        void openFile(std::string path);
    };

}