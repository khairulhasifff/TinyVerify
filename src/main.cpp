#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <onnxruntime_cxx_api.h>
#include "image_preprocessor.h"
#include "face_detector.h"
#include "face_verifier.h"

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

    // Mute OpenCV INFO logs (Only show Warnings and Errors)
    cv::utils::logging::setLogLevel(
        cv::utils::logging::LOG_LEVEL_ERROR
    );

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
    cv::imwrite("output/detected_face.jpg", input_image);

    std::cout << "Face detected successfully" << std::endl;
    std::cout << "x: " << face_box.x << std::endl;
    std::cout << "y: " << face_box.y << std::endl;
    std::cout << "width: " << face_box.width << std::endl;
    std::cout << "height: " << face_box.height << std::endl;

    std::cout << "Saved debug image: output/detected_face.jpg" << std::endl;

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
    cv::Mat cropped_face =
        detector.crop_face(input_image, face_box);

    cv::imwrite("output/cropped_face.jpg", cropped_face);

    std::cout << "Saved cropped face: output/cropped_face.jpg" << std::endl;

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
 * STAGE 10 — FACE VERIFICATION PLACEHOLDER
 * =========================================================================
 *
 * This stage connects the preprocessing output to the future inference layer.
 *
 * Current behavior:
 * - initializes FaceVerifier
 * - sends preprocessed tensor into placeholder embedding generator
 *
 * Future behavior:
 * - load ArcFace ONNX model
 * - run ONNX Runtime inference
 * - return real facial embedding vector
 * =========================================================================
 */
 /** * ========================================================================= * STAGE 10 — REAL ONNX RUNTIME TENSOR TETHERING * ========================================================================= * * This stage binds the preprocessed floating-point vector into a real * ONNX Runtime memory allocation. * * Tensor Dimensions for ArcFace: * - Batch Size: 1 * - Channels: 3 (RGB) * - Height: 112 * - Width: 112 * * Total Flat Features: 1 * 3 * 112 * 112 = 37,632 elements * ========================================================================= */
     // 1. Initialize our FaceVerifier with the active ArcFace model
    FaceVerifier verifier("models/arcface_buffalo_1.onnx");

    // 2. Define the input shape required by ArcFace Buffalo (1, 3, 112, 112)
    std::vector<int64_t> input_shape = { 1, 3, 112, 112 };

    // 3. Allocate a memory info block for CPU usage allocation
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtDeviceAllocator,
        OrtMemTypeCPU
    );

    // 4. Create the real input tensor wrapper using our preprocessed float data pointer
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(model_input.data()), // Points to your preprocessed data
        model_input.size(),
        input_shape.data(),
        input_shape.size()
    );

    std::cout << "ONNX Input Tensor bound successfully!" << std::endl;
    std::cout << "Tensor Type: Float32 | Shape: [1, 3, 112, 112]" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    // 5. Send your wrapped tensor into the model context
    std::vector<float> embedding = verifier.generate_embedding(input_tensor); // <-- Change model_input to input_tensor

    std::cout << "Generated placeholder embedding size: "
        << embedding.size()
        << std::endl;

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