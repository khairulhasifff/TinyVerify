#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// BGR image -> 112x112 RGB, scaled to [-1, 1], laid out CHW (37632 floats) for ArcFace.
class ImagePreprocessor {
public:
    std::vector<float> preprocess(const cv::Mat& image, bool debug = false);
};
