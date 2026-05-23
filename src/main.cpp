#include <iostream>
#include <opencv2/opencv.hpp>

#include "image_preprocessor.h"
#include "face_detector.h"

/**
 * ============================================================================
 * TINYVERIFY — PIPELINE ORCHESTRATION ENTRYPOINT
 * ============================================================================
 *
 * Current System Flow:
 *
 * Disk Asset
 *     ↓
 * Image Ingestion
 *     ↓
 * Defensive Validation
 *     ↓
 * Face Detection
 *     ↓
 * Debug Visualization
 *     ↓
 * Future Face Crop
 *     ↓
 * Preprocessing Pipeline
 *
 * ============================================================================
 */

int main() {

    std::cout << "TinyVerify initialized successfully" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    /**
     * =========================================================================
     * STAGE 1 — IMAGE INGESTION
     * =========================================================================
     *
     * Flow:
     * Disk Asset → OpenCV Decoder → cv::Mat Pixel Matrix
     *
     * OpenCV loads compressed image bytes into RAM as a matrix structure.
     * =========================================================================
     */
    cv::Mat input_image = cv::imread("data/test_face.jpeg");

    /**
     * =========================================================================
     * STAGE 2 — DEFENSIVE VALIDATION GATE
     * =========================================================================
     *
     * Prevent downstream pipeline failures if the image cannot be loaded.
     * =========================================================================
     */
    if (input_image.empty()) {
        std::cerr << "[SYSTEM-ERROR] Could not read image asset." << std::endl;
        return -1;
    }

    /**
     * =========================================================================
     * STAGE 3 — FACE DETECTION INITIALIZATION
     * =========================================================================
     *
     * Load Haar Cascade classifier into memory.
     *
     * The cascade contains pre-trained facial feature patterns used by
     * OpenCV's classical computer vision engine.
     * =========================================================================
     */
    FaceDetector detector(
        "data/haarcascade_frontalface_default.xml"
    );

    /**
     * =========================================================================
     * STAGE 4 — FACE LOCALIZATION PIPELINE
     * =========================================================================
     *
     * Full Image
     *     ↓
     * Grayscale Conversion
     *     ↓
     * Haar Cascade Scan
     *     ↓
     * Face Coordinate Extraction
     *
     * Output:
     * Bounding rectangle containing detected face coordinates.
     * =========================================================================
     */
    cv::Rect face_box = detector.detect(input_image);

    /**
     * =========================================================================
     * STAGE 5 — FACE DETECTION VALIDATION
     * =========================================================================
     *
     * Ensure at least one face region was detected.
     * =========================================================================
     */
    if (face_box.empty()) {
        std::cerr << "[SYSTEM-ERROR] No face detected." << std::endl;
        return -1;
    }

    /**
     * =========================================================================
     * STAGE 6 — DEBUG VISUALIZATION LAYER
     * =========================================================================
     *
     * Draw bounding rectangle around detected face region.
     *
     * Purpose:
     * - visual verification
     * - debugging
     * - pipeline observability
     * =========================================================================
     */
    cv::rectangle(
        input_image,
        face_box,
        cv::Scalar(0, 255, 0),
        2
    );

    /**
     * =========================================================================
     * STAGE 7 — DEBUG OUTPUT EXPORT
     * =========================================================================
     *
     * Persist debug image for inspection.
     * =========================================================================
     */
    cv::imwrite("data/detected_face.jpg", input_image);

    std::cout << "Face detected successfully" << std::endl;
    std::cout << "x: " << face_box.x << std::endl;
    std::cout << "y: " << face_box.y << std::endl;
    std::cout << "width: " << face_box.width << std::endl;
    std::cout << "height: " << face_box.height << std::endl;

    std::cout << "Saved debug image: data/detected_face.jpg" << std::endl;

    /**
 * =========================================================================
 * STAGE 8 — FACE REGION EXTRACTION
 * =========================================================================
 *
 * Extract only the detected facial region from the image.
 *
 * Why This Matters:
 *
 * Face verification models should process only the face region,
 * not the surrounding background.
 *
 * clone() creates an independent memory copy of the cropped region.
 * =========================================================================
 */
    cv::Mat cropped_face = input_image(face_box).clone();

    cv::imwrite("data/cropped_face.jpg", cropped_face);

    std::cout << "Saved cropped face: data/cropped_face.jpg" << std::endl;

    /**
     * =========================================================================
     * STAGE 9 — PREPROCESSING PIPELINE
     * =========================================================================
     *
     * Full Face Region
     *     ↓
     * Resize to 112x112
     *     ↓
     * BGR → RGB conversion
     *     ↓
     * Float normalization
     *     ↓
     * Tensor flattening
     *
     * Output:
     * Model-ready tensor buffer for future ONNX Runtime inference.
     * =========================================================================
     */
    ImagePreprocessor preprocessor;

    std::vector<float> model_input =
        preprocessor.preprocess(cropped_face);

    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Preprocessing completed" << std::endl;

    /**
     * =========================================================================
     * FUTURE PIPELINE STAGE
     * =========================================================================
     *
     * Detected Face
     *     ↓
     * Face Crop
     *     ↓
     * Preprocessing Pipeline
     *     ↓
     * Tensor Conversion
     *     ↓
     * Future ONNX Runtime Inference
     * =========================================================================
     */

    return 0;
}