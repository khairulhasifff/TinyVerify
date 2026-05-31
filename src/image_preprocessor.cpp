#include "image_preprocessor.h"
#include <iostream>
#include <vector>

std::vector<float> ImagePreprocessor::preprocess(const cv::Mat& image, bool debug) {
    /**
     * @brief MEMORY LAYOUT TRANSFORMATIONS
     * Step 1: [cv::resize]       --> Enforce exact spatial grid matrix boundaries
     * Step 2: [cv::cvtColor]     --> Correct chromatic channel arrangement swaps
     * Step 3: [convertTo]        --> Project 8-bit integer bits to float32 domain
     * Step 4: [HWC -> CHW]       --> Convert OpenCV interleaved image memory to ONNX NCHW-compatible tensor layout
     */

    const int target_width = 112;
    const int target_height = 112;
    const int channels = 3;

    // 1. Structural Conformance: ArcFace expects a rigid 112x112 layout
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(target_width, target_height));

    if (debug) {
        std::cout << "Width: " << resized_image.cols << std::endl;
        std::cout << "Height: " << resized_image.rows << std::endl;
    }

    // 2. Chromatic Alignment: Enforce RGB channel order (OpenCV default is BGR)
    cv::Mat rgb_image;
    cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);

    // 3. Precision Scaling & Normalization: Scale into explicit [0.0, 1.0] float bounds
    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    // Observability Logging: Scan matrix memory blocks for min/max values
    double min_val, max_val;
    cv::minMaxLoc(float_image.reshape(1), &min_val, &max_val);

    if (debug) {
        std::cout << "Min: " << min_val << std::endl;
        std::cout << "Max: " << max_val << std::endl;
    }

    // 4. Tensor Memory Layout Conversion: HWC -> CHW
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