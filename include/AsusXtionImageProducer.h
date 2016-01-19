//
// Created by sean on 19/01/16.
//

#ifndef ASUS_XTION_STEREO_CALIBRATION_ASUSXTION_IMAGE_PRODUCER_H
#define ASUS_XTION_STEREO_CALIBRATION_ASUS_XTION_IMAGE_PRODUCER_H

#include <pcl/io/openni2_grabber.h>
#include <opencv2/core/mat.hpp>
#include <mutex>
#include <condition_variable>

typedef pcl::io::OpenNI2Grabber NI2Grabber;
typedef pcl::io::openni2::OpenNI2DeviceManager NI2DeviceManager;
typedef pcl::io::openni2::Image NI2Image;
typedef NI2Image::Ptr NI2ImagePtr;
typedef pcl::io::openni2::IRImage NI2IRImage;
typedef NI2IRImage::Ptr NI2IRImagePtr;

class AsusXtionImageProducer {

public:
    AsusXtionImageProducer(bool ir = false) : ir_ (ir), image_ready_ (false) {};
    ~AsusXtionImageProducer();
    bool open(int camera_id = -1);
    bool isOpened();
    cv::Mat getImage();

private:
    void rgb_callback(const NI2ImagePtr& image);
    void ir_callback(const NI2IRImagePtr& image);
    cv::Mat getRGBImage();
    cv::Mat getIRImage();
    std::shared_ptr<NI2Grabber> grabber_;
    std::mutex image_mutex_;
    std::once_flag ready_flag_;
    std::condition_variable condition_variable_;
    bool image_ready_;
    NI2ImagePtr rgb_image_;
    NI2IRImagePtr ir_image_;
    bool ir_;
};

#endif //ASUS_XTION_STEREO_CALIBRATION_ASUSXTIONIMAGEPRODUCER_H
