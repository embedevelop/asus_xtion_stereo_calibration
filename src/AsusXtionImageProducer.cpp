//
// Created by sean on 19/01/16.
//


#include <thread>
#include <opencv2/imgproc.hpp>
#include "AsusXtionImageProducer.h"

bool AsusXtionImageProducer::open(int camera_id) {

    auto device_manager = NI2DeviceManager::getInstance();
    size_t num_devices = device_manager->getNumOfConnectedDevices();
    if (num_devices > 0) {
        if(camera_id == -1) {
            grabber_ = std::make_shared<NI2Grabber>();
        }
        else if( camera_id <= num_devices ) {
            std::stringstream ss;
            ss << '#' << camera_id;
            grabber_ = std::make_shared<NI2Grabber>(ss.str());
        }

        if(ir_) {
            if(grabber_->providesCallback<void (const NI2IRImagePtr&)> () ) {
                boost::function<void (const NI2IRImagePtr&)> ir_callback =
                        boost::bind(&AsusXtionImageProducer::ir_callback, this, _1);
                grabber_->registerCallback(ir_callback);
            }
        }
        else {
            if(grabber_->providesCallback<void (const NI2ImagePtr&)> () ) {
                boost::function<void (const NI2ImagePtr&)> rgb_callback =
                        boost::bind(&AsusXtionImageProducer::rgb_callback, this, _1);
                grabber_->registerCallback(rgb_callback);
            }
        }

        grabber_->start();
        return true;
    }
    else {
        std::cout << "No devices found!" << std::endl;
        return false;
    }
}

bool AsusXtionImageProducer::isOpened() {
    return grabber_->isRunning();
}

void AsusXtionImageProducer::rgb_callback(const NI2ImagePtr& rgb_image) {
    std::lock_guard<std::mutex> lock(image_mutex_);
    rgb_image_ = rgb_image;

    std::call_once(ready_flag_, [&] {
        image_ready_ = true;
        condition_variable_.notify_one();
    });
}

void AsusXtionImageProducer::ir_callback(const NI2IRImagePtr& ir_image) {
    std::lock_guard<std::mutex> lock(image_mutex_);
    ir_image_ = ir_image;
    std::call_once(ready_flag_, [&] {
        image_ready_ = true;
        condition_variable_.notify_one();
    });
}

cv::Mat AsusXtionImageProducer::getImage() {
    return ir_ ? getIRImage() : getRGBImage();
}

cv::Mat AsusXtionImageProducer::getRGBImage() {
    NI2ImagePtr temp_image;
    {
        std::unique_lock<std::mutex> lock(image_mutex_);
        condition_variable_.wait(lock, [&]{ return image_ready_; });
        temp_image = rgb_image_;
    }

    cv::Mat result(temp_image->getHeight(), temp_image->getWidth(), CV_8UC3, (void *)temp_image->getData());
    cv::cvtColor(result, result, cv::COLOR_RGB2BGR);
    return result;
}

cv::Mat AsusXtionImageProducer::getIRImage() {
    NI2IRImagePtr temp_image;
    {
        std::unique_lock<std::mutex> lock(image_mutex_);
        condition_variable_.wait(lock, [&]{ return image_ready_; });
        temp_image = ir_image_;
    }

    // Do conversion
    cv::Mat result(temp_image->getHeight(), temp_image->getWidth(), CV_16SC1, (void *)temp_image->getData());
    cv::Mat gray_result;
    // TODO: Correct scaling?
    result.convertTo(gray_result, CV_8U, 1.0);

    return gray_result;
}

AsusXtionImageProducer::~AsusXtionImageProducer() {
    grabber_->stop();
}