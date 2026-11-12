#pragma once

#include <opencv2/opencv.hpp>
#include <string>

// Finds the main face in an image with an OpenCV Haar cascade.
class FaceDetector {
public:
    // Throws if the cascade XML cannot be loaded.
    FaceDetector(const std::string& cascade_path);

    // Returns the largest face, or an empty rect if none is found.
    cv::Rect detect(const cv::Mat& image);

    // Returns a deep copy so the crop does not share memory with the source image.
    cv::Mat crop_face(
        const cv::Mat& image,
        const cv::Rect& face_box
    );

private:
    cv::CascadeClassifier face_cascade_;
};
