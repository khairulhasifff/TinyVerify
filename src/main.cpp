#include <iostream>
#include <opencv2/opencv.hpp>
#include "image_preprocessor.h"

/**
 * @brief APPLICATION PIPELINE EXECUTION
 * * [Disk I/O Ingestion] ---> [Production Safeguard] ---> [Preprocessing Infra Layer]
 * (test_face.jpeg)            (Mat::empty())            (Model Contract Fit)
 */
int main() {
    std::cout << "TinyVerify initialized successfully" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    // Ingest the asset using our Phase 2 filename fix
    cv::Mat input_image = cv::imread("data/test_face.jpeg");

    // Dynamic Production Safeguard
    if (input_image.empty()) {
        std::cerr << "[SYSTEM-ERROR] Pipeline broken. Could not read image asset." << std::endl;
        return -1;
    }

    // Initialize our deterministic model contract engine
    ImagePreprocessor preprocessor;
    std::vector<float> model_input = preprocessor.preprocess(input_image);

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Preprocessing completed" << std::endl;

    return 0;
}