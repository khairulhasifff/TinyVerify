#include "face_detector.h"
#include <algorithm>
#include <stdexcept>

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
 *
 * ============================================================================
 */
FaceDetector::FaceDetector(const std::string& cascade_path) {

    // Load Haar Cascade classifier into memory
    // Fail fast: an empty classifier would only throw later inside detect().
    if (!face_cascade_.load(cascade_path)) {
        throw std::runtime_error(
            "Failed to load cascade classifier: " + cascade_path
        );
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
 *
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

    cv::cvtColor(
        image,
        gray,
        cv::COLOR_BGR2GRAY
    );

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
     * Current implementation:
     * - returns the largest detected face
     *
     * Haar cascades often add small false positives in the background,
     * and detectMultiScale returns boxes in no particular order, so
     * faces[0] is not guaranteed to be the subject.
     *
     * Future improvements:
     * - confidence ranking
     * - multi-face handling
     */
    return *std::max_element(
        faces.begin(),
        faces.end(),
        [](const cv::Rect& a, const cv::Rect& b) { return a.area() < b.area(); }
    );
}

/**
 * ============================================================================
 * FACE REGION EXTRACTION PIPELINE
 * ============================================================================
 *
 * Pipeline Flow:
 *
 * Original Image
 *     ↓
 * Bounding Rectangle Coordinates
 *     ↓
 * OpenCV Region Extraction
 *     ↓
 * Independent Face Matrix (clone)
 *
 * Why clone() Matters:
 *
 * OpenCV ROI slicing can reference the original image memory buffer.
 *
 * clone() creates a deep independent copy to ensure:
 * - memory safety
 * - preprocessing isolation
 * - stable downstream tensor preparation
 *
 * This prevents accidental coupling between:
 * - original image memory
 * - cropped face memory
 *
 * ============================================================================
 */
cv::Mat FaceDetector::crop_face(
    const cv::Mat& image,
    const cv::Rect& face_box
) {

    return image(face_box).clone();
}