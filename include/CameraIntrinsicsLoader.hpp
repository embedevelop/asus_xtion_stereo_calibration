//
// Created by sean on 15/01/16.
//

#ifndef PCL_CLOUD_REGISTRATION_CAMERA_INTRINSICS_LOADER_HPP
#define PCL_CLOUD_REGISTRATION_CAMERA_INTRINSICS_LOADER_HPP

#include <pcl/io/openni2_grabber.h>
#include <opencv2/core/mat.hpp>

namespace CameraIntrinsicsLoader {
    bool getIntrinsics(int camera_num, cv::Mat& camera_matrix, cv::Mat& dist_coeffs);
};

#endif //PCL_CLOUD_REGISTRATION_CAMERAINTRINSICSLOADER_HPP
