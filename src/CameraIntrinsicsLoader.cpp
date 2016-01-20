//
// Created by sean on 15/01/16.
//

#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include "CameraIntrinsicsLoader.hpp"

bool CameraIntrinsicsLoader::getIntrinsics(int camera_num, cv::Mat& camera_matrix, cv::Mat& dist_coeffs) {
    std::string path = "/home/sean/Documents/cameraparams/";
    std::stringstream ss;
    ss << path << camera_num << "rgb.yml";
    std::string rgb_filename = ss.str();

    cv::FileStorage fs_rgb;
    if(!fs_rgb.open(rgb_filename.c_str(), cv::FileStorage::READ))
        return false;

    camera_matrix = cv::Mat_<double>::zeros(3, 3);
    dist_coeffs = cv::Mat_<double>::zeros(5, 1);

    fs_rgb["camera_matrix"] >> camera_matrix;
    fs_rgb["distortion_coefficients"] >> dist_coeffs;

    return true;
}
