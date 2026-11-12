#include "face_detector.h"
#include <algorithm>
#include <stdexcept>

FaceDetector::FaceDetector(const std::string& cascade_path) {
    // Fail fast: an empty classifier would only throw later inside detect().
    if (!face_cascade_.load(cascade_path)) {
        throw std::runtime_error(
            "Failed to load cascade classifier: " + cascade_path
        );
    }
}

cv::Rect FaceDetector::detect(const cv::Mat& image) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> faces;
    face_cascade_.detectMultiScale(gray, faces);

    if (faces.empty()) {
        return cv::Rect();
    }

    // Boxes come back in no set order and Haar adds small background false
    // positives, so the largest box is the best guess for the subject.
    return *std::max_element(
        faces.begin(),
        faces.end(),
        [](const cv::Rect& a, const cv::Rect& b) { return a.area() < b.area(); }
    );
}

cv::Mat FaceDetector::crop_face(
    const cv::Mat& image,
    const cv::Rect& face_box
) {
    return image(face_box).clone();
}
