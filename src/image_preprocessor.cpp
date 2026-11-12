#include "image_preprocessor.h"
#include <iostream>
#include <vector>

std::vector<float> ImagePreprocessor::preprocess(const cv::Mat& image, bool debug) {
    const int target_width = 112;
    const int target_height = 112;
    const int channels = 3;

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(target_width, target_height));

    if (debug) {
        std::cout << "Width: " << resized_image.cols << std::endl;
        std::cout << "Height: " << resized_image.rows << std::endl;
    }

    cv::Mat rgb_image;
    cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);

    // (x - 127.5) / 127.5 -> [-1, 1], the range insightface ArcFace was trained on
    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32FC3, 1.0 / 127.5, -1.0);

    if (debug) {
        double min_val, max_val;
        cv::minMaxLoc(float_image.reshape(1), &min_val, &max_val);
        std::cout << "Min: " << min_val << std::endl;
        std::cout << "Max: " << max_val << std::endl;
    }

    // HWC (OpenCV, interleaved) -> CHW (ONNX, one plane per channel)
    std::vector<float> tensor_buffer(channels * target_height * target_width);

    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < target_height; ++y) {
            for (int x = 0; x < target_width; ++x) {
                tensor_buffer[c * target_height * target_width + y * target_width + x] =
                    float_image.at<cv::Vec3f>(y, x)[c];
            }
        }
    }

    if (debug) {
        std::cout << "Tensor size: " << tensor_buffer.size() << std::endl;
    }

    return tensor_buffer;
}
