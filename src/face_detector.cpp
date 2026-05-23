#include "face_detector.h"
#include <iostream>

/**
 * ============================================================================
 * CONSTRUCTOR: Cascade Initialization Phase
 * ============================================================================
 *
 * System Flow:
 *
 * XML Cascade File
 *     ↓
 * OpenCV Cascade Loader
 *     ↓
 * FaceDetector Memory State
 *
 * The XML file contains pre-trained Haar feature patterns used to
 * identify facial structures.
 * ============================================================================
 */
FaceDetector::FaceDetector(const std::string& cascade_path) {

    // Load Haar Cascade classifier into memory
    if (!face_cascade_.load(cascade_path)) {
        std::cerr << "[SYSTEM-ERROR] Failed to load cascade classifier: "
            << cascade_path << std::endl;
    }
}

/**
 * ============================================================================
 * FACE DETECTION EXECUTION PIPELINE
 * ============================================================================
 *
 * Full Image
 *     ↓
 * Grayscale Conversion
 *     ↓
 * Haar Cascade Scan
 *     ↓
 * Face Coordinate Extraction
 *     ↓
 * Return Bounding Rectangle
 *
 * Why Grayscale?
 *
 * Haar cascades operate on luminance/intensity patterns rather than
 * RGB color information.
 *
 * This reduces computational complexity.
 * ============================================================================
 */
cv::Rect FaceDetector::detect(const cv::Mat& image) {

    // Container storing all detected face regions
    std::vector<cv::Rect> faces;

    /**
     * STEP 1: Chromatic Reduction
     *
     * Convert BGR image into grayscale intensity matrix.
     *
     * OpenCV Haar cascades expect single-channel grayscale input.
     */
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    /**
     * STEP 2: Multi-Scale Face Detection
     *
     * Internally scans the image at multiple scales searching for
     * Haar facial feature patterns.
     */
    face_cascade_.detectMultiScale(gray, faces);

    /**
     * STEP 3: Defensive Safeguard
     *
     * Return empty rectangle if no face is detected.
     */
    if (faces.empty()) {
        return cv::Rect();
    }

    /**
     * STEP 4: Primary Face Selection
     *
     * Currently returns the first detected face.
     *
     * Future improvement:
     * - Select largest face
     * - Multi-face handling
     * - Confidence ranking
     */
    return faces[0];
}