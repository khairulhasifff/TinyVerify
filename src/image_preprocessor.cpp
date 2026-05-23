#include "image_preprocessor.h"
#include <iostream>

std::vector<float> ImagePreprocessor::preprocess(const cv::Mat& image) {
    /**
     * @brief MEMORY LAYOUT TRANSFORMATIONS
     * * Step 1: [cv::resize]       --> Enforce exact spatial grid matrix boundaries
     * Step 2: [cv::cvtColor]     --> Correct chromatic channel arrangement swaps
     * Step 3: [convertTo]        --> Project 8-bit integer bits to float32 domain
     * Step 4: [Memory Flatten]   --> Serialize matrix nodes into a raw contiguous 1D array
     */

     // 1. Structural Conformance: ArcFace expects a rigid 112x112 layout
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(112, 112));

    std::cout << "Width: " << resized_image.cols << std::endl;
    std::cout << "Height: " << resized_image.rows << std::endl;

    // 2. Chromatic Alignment: Enforce RGB channel order (OpenCV default is BGR)
    cv::Mat rgb_image;
    cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);

    // 3. Precision Scaling & 4. Normalization: Scale into explicit [0.0, 1.0] float bounds
    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    // Observability Logging Step B: Scan matrix memory blocks for min/max values
    double min_val, max_val;
    cv::minMaxLoc(float_image.reshape(1), &min_val, &max_val);
    std::cout << "Min: " << min_val << std::endl;
    std::cout << "Max: " << max_val << std::endl;

    // 5. Contiguous Memory Flattening (HWC Layout Transformation)
    std::vector<float> tensor_buffer;
    tensor_buffer.assign((float*)float_image.datastart, (float*)float_image.dataend);

    std::cout << "Tensor size: " << tensor_buffer.size() << std::endl;

    return tensor_buffer;
} 