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

#include <gtest/gtest.h>
#include "algorithms/Kalman1D.h"


TEST(kalman_d1_test, init) {

    AtlasFusion::Algorithms::Kalman1D kalman(0.1, 0.5);
    EXPECT_EQ(kalman.getPosition(), 0);
    EXPECT_EQ(kalman.getVelocity(), 0);
}


TEST(kalman_d1_test, iteration) {

    AtlasFusion::Algorithms::Kalman1D kalman{0.1, 0.5};
    double measured_pose = 0;
    double dt = 0.1;
    for (int i = 0 ; i < 20 ; i++) {
        if(i*dt >= 1) {
            measured_pose = 1.0;
        }
        std::cout << "Iteration " << i+1 << std::endl;
        std::cout << "Prediciton, states: " << kalman.getPosition() << " " << kalman.getVelocity() << std::endl;
        kalman.predict(dt, 0);
        cv::Mat measurement = (cv::Mat_<double>(2, 1) <<
                                                      measured_pose,  // p
                0);
        std::cout << "Measurement, states: " << kalman.getPosition() << " " << kalman.getVelocity() << std::endl;
        kalman.correct(measurement);
    }
}


TEST(kalman_d1_test, fixed_pose) {

    AtlasFusion::Algorithms::Kalman1D kalman{0.1, 0.5};
    double measured_pose = 10;
    double dt = 0.1;
    for (int i = 0 ; i < 1000 ; i++) {
        kalman.predict(dt, 0);
        cv::Mat measurement = (cv::Mat_<double>(2, 1) <<
                measured_pose,
                0);
        kalman.correct(measurement);
    }

    EXPECT_NEAR(kalman.getPosition(), measured_pose, 0.1);
    EXPECT_NEAR(kalman.getVelocity(), 0, 0.1);
}


TEST(kalman_d1_test, fixed_speed) {

    AtlasFusion::Algorithms::Kalman1D kalman{0.1, 0.5};
    double dt = 0.1;
    double speed = 1;
    double measured_pose = 0;
    for (int i = 0 ; i < 1000 ; i++) {

        measured_pose = dt * i * speed;
        kalman.predict(dt, 0);
        cv::Mat measurement = (cv::Mat_<double>(2, 1) <<
                measured_pose,
                speed);
        kalman.correct(measurement);
    }

    EXPECT_NEAR(kalman.getPosition(), measured_pose, 0.1);
    EXPECT_NEAR(kalman.getVelocity(), speed, 0.1);
}

int main(int argc, char **argv){

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


