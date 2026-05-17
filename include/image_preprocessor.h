#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief ARCHITECTURAL VISUALIZATION: INFERENCE CONTRACT ENFORCEMENT
 * * +-------------------+
 * | Raw cv::Mat (BGR) |  <-- Variable disk asset inputs
 * +-------------------+
 * │
 * ▼ [ImagePreprocessor::preprocess]
 * +----------------------------------------+
 * | 1. Structural Conformance (112x112)    |
 * | 2. Chromatic Realignment (BGR -> RGB)  |
 * | 3. Precision Scaling (uint8 -> float32)|
 * | 4. Normalization Range ([0.0, 1.0])    |
 * +----------------------------------------+
 * │
 * ▼
 * +-------------------+
 * | std::vector<float>|  <-- Contiguous 1D HWC Tensor Buffer (Size: 37632)
 * +-------------------+
 */
class ImagePreprocessor {
public:
    // Enforces the rigid data contract required by ArcFace
    std::vector<float> preprocess(const cv::Mat& image);
};