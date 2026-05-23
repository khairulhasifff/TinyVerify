#pragma once

#include <opencv2/opencv.hpp>
#include <string>

/**
 * ============================================================================
 * MODULE: Face Detection Infrastructure Layer
 * ============================================================================
 *
 * Responsibility:
 * Detect the spatial coordinates of a human face inside an image.
 *
 * Pipeline Role:
 *
 * Raw Image
 *     ↓
 * FaceDetector::detect()
 *     ↓
 * Face Bounding Box (cv::Rect)
 *     ↓
 * Future Face Crop
 *     ↓
 * Preprocessing Pipeline
 *
 * Why This Exists:
 *
 * Face verification models should process only the face region,
 * not the entire image background.
 *
 * This module isolates the facial region before inference.
 *
 * ============================================================================
 */

class FaceDetector {
public:

    /**
     * @brief Constructor
     *
     * Loads the Haar Cascade XML classifier into memory.
     *
     * The classifier contains pre-trained facial pattern data used by
     * OpenCV's classical detection engine.
     */
    FaceDetector(const std::string& cascade_path);

    /**
     * @brief Detect face location inside an image
     *
     * Input:
     * - Full image matrix
     *
     * Output:
     * - Bounding rectangle containing detected face coordinates
     *
     * Returns:
     * - Empty rectangle if no face is detected
     */
    cv::Rect detect(const cv::Mat& image);

private:

    /**
     * OpenCV cascade classifier engine
     *
     * Internally stores the Haar feature cascade used for
     * classical face detection.
     */
    cv::CascadeClassifier face_cascade_;
};