#include <iostream>
#include <vector>
#include <string>

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
 * Project Role:
 *
 * This file is the top-level orchestration layer for TinyVerify.
 *
 * It connects:
 *
 * - OpenCV image loading
 * - OpenCV / Haar Cascade face detection
 * - face cropping
 * - ImagePreprocessor tensor preparation
 * - ONNX Runtime tensor binding
 * - ArcFace ONNX inference
 * - embedding extraction
 * - cosine similarity comparison
 *
 * Current Engineering Goal:
 *
 * Move from a one-image self-similarity sanity test into a two-image
 * embedding comparison test.
 *
 * Previous behavior:
 *
 * image A
 *     ↓
 * embedding A
 *     ↓
 * compare embedding A against itself
 *
 * New behavior:
 *
 * image A
 *     ↓
 * embedding A
 *
 * image B
 *     ↓
 * embedding B
 *
 * embedding A + embedding B
 *     ↓
 * cosine similarity
 *
 * Frameworks Used:
 *
 * - OpenCV:
 *   Used for image loading, face detection, drawing debug rectangles,
 *   image cropping, and saving debug output images.
 *
 * - ONNX Runtime:
 *   Used to create input tensors and execute the ArcFace ONNX model.
 *
 * - ArcFace:
 *   Deep face-recognition model that converts a face image into a
 *   512-dimensional embedding vector.
 *
 * - TinyVerify custom modules:
 *   FaceDetector       → detects and crops faces
 *   ImagePreprocessor  → converts cropped face image into CHW float tensor
 *   FaceVerifier       → runs ONNX inference and computes cosine similarity
 *
 * ============================================================================
 */


 /**
  * ============================================================================
  * FUNCTION: generate_embedding_from_image
  * ============================================================================
  *
  * Responsibility:
  *
  * Convert one image file into one ArcFace embedding.
  *
  * This helper function exists because the same full pipeline must be run for
  * both person_a.jpg and person_b.jpg.
  *
  * Without this helper, main() would need to duplicate the same image loading,
  * face detection, cropping, preprocessing, tensor binding, and inference code
  * twice.
  *
  * Full Internal Flow:
  *
  * Image Path
  *     ↓
  * cv::imread
  *     ↓
  * cv::Mat image
  *     ↓
  * FaceDetector::detect
  *     ↓
  * Face bounding box
  *     ↓
  * Debug rectangle image export
  *     ↓
  * FaceDetector::crop_face
  *     ↓
  * Cropped face image
  *     ↓
  * ImagePreprocessor::preprocess
  *     ↓
  * CHW float tensor buffer
  *     ↓
  * Ort::Value input tensor
  *     ↓
  * FaceVerifier::generate_embedding
  *     ↓
  * 512-dimensional ArcFace embedding
  *
  * Parameters:
  *
  * image_path:
  *     Path to the input image on disk.
  *
  * debug_prefix:
  *     Prefix used when saving debug images to output/.
  *     Example:
  *     - person_a_detected_face.jpg
  *     - person_a_cropped_face.jpg
  *
  * detector:
  *     Existing FaceDetector instance.
  *
  * preprocessor:
  *     Existing ImagePreprocessor instance.
  *
  * verifier:
  *     Existing FaceVerifier instance holding the ArcFace ONNX Runtime session.
  *
  * memory_info:
  *     ONNX Runtime CPU memory descriptor used when creating input tensors.
  *
  * Return:
  *
  * std::vector<float>
  *     A real ArcFace embedding vector.
  *
  * Failure Behavior:
  *
  * Returns an empty vector if:
  * - the image cannot be loaded
  * - no face is detected
  *
  * ============================================================================
  */
std::vector<float> generate_embedding_from_image(
    const std::string& image_path,
    const std::string& debug_prefix,
    const std::string& display_label,
    FaceDetector& detector,
    ImagePreprocessor& preprocessor,
    FaceVerifier& verifier,
    const Ort::MemoryInfo& memory_info
) {
    /**
     * ========================================================================
     * STAGE A1 — IMAGE INGESTION
     * ========================================================================
     *
     * Framework:
     * OpenCV
     *
     * cv::imread loads an image file from disk and decodes it into a cv::Mat.
     *
     * Important:
     * OpenCV loads images in BGR channel order by default.
     * TinyVerify's ImagePreprocessor is responsible for converting BGR to RGB.
     * ========================================================================
     */
    cv::Mat input_image = cv::imread(image_path);
    std::cout << std::endl;
    std::cout << "[" << display_label << "]" << std::endl;
    std::cout << "Input image: " << image_path << std::endl;

    /**
     * ========================================================================
     * STAGE A2 — IMAGE LOAD VALIDATION
     * ========================================================================
     *
     * Defensive check to prevent the rest of the pipeline from running on an
     * invalid or missing image.
     * ========================================================================
     */
    if (input_image.empty()) {
        std::cerr << "[SYSTEM-ERROR] Could not read image asset: "
            << image_path
            << std::endl;

        return {};
    }

    /**
     * ========================================================================
     * STAGE A3 — FACE DETECTION
     * ========================================================================
     *
     * Framework:
     * OpenCV Haar Cascade through TinyVerify FaceDetector.
     *
     * The detector scans the image and returns a cv::Rect describing the
     * detected face location.
     *
     * Output:
     * - x coordinate
     * - y coordinate
     * - width
     * - height
     * ========================================================================
     */
    cv::Rect face_box = detector.detect(input_image);

    /**
     * ========================================================================
     * STAGE A4 — FACE DETECTION VALIDATION
     * ========================================================================
     *
     * If no face is found, this image cannot produce a valid ArcFace embedding.
     * ========================================================================
     */
    if (face_box.empty()) {
        std::cerr << "[SYSTEM-ERROR] No face detected in image: "
            << image_path
            << std::endl;

        return {};
    }

    /**
     * ========================================================================
     * STAGE A5 — DEBUG VISUALIZATION
     * ========================================================================
     *
     * Framework:
     * OpenCV
     *
     * clone() is used so the debug rectangle does not modify the original image
     * that will later be used for cropping.
     *
     * This keeps the inference input clean.
     * ========================================================================
     */
    cv::Mat debug_image = input_image.clone();

    cv::rectangle(
        debug_image,
        face_box,
        cv::Scalar(0, 255, 0),
        2
    );

    std::string detected_output =
        "output/" + debug_prefix + "_detected_face.jpg";

    cv::imwrite(detected_output, debug_image);

    /**
     * ========================================================================
     * STAGE A6 — FACE REGION EXTRACTION
     * ========================================================================
     *
     * Framework:
     * OpenCV through TinyVerify FaceDetector.
     *
     * The model should receive the face region, not the full background image.
     * ========================================================================
     */
    cv::Mat cropped_face =
        detector.crop_face(input_image, face_box);

    std::string cropped_output =
        "output/" + debug_prefix + "_cropped_face.jpg";

    cv::imwrite(cropped_output, cropped_face);

    /**
     * ========================================================================
     * STAGE A7 — FACE DETECTION DEBUG LOGGING
     * ========================================================================
     *
     * These logs make it clear which image was processed and what face region
     * was detected.
     * ========================================================================
     */
    std::cout << "Face detected successfully" << std::endl;

    std::cout << "Bounding box: "
        << "x=" << face_box.x
        << ", y=" << face_box.y
        << ", width=" << face_box.width
        << ", height=" << face_box.height
        << std::endl;

    std::cout << "Saved debug image: "
        << detected_output
        << std::endl;

    std::cout << "Saved cropped face: "
        << cropped_output
        << std::endl;

    /**
     * ========================================================================
     * STAGE A8 — IMAGE PREPROCESSING
     * ========================================================================
     *
     * Framework:
     * OpenCV inside TinyVerify ImagePreprocessor.
     *
     * Expected preprocessing responsibilities:
     *
     * Cropped Face
     *     ↓
     * Resize to 112x112
     *     ↓
     * BGR to RGB conversion
     *     ↓
     * Float normalization
     *     ↓
     * HWC to CHW layout conversion
     *
     * Output:
     * Flat std::vector<float> containing:
     *
     * 1 * 3 * 112 * 112 = 37,632 float values
     *
     * This is the tensor buffer that ONNX Runtime will bind.
     * ========================================================================
     */
    std::vector<float> model_input =
        preprocessor.preprocess(cropped_face);

    /**
     * ========================================================================
     * STAGE A9 — ONNX INPUT SHAPE DECLARATION
     * ========================================================================
     *
     * Framework:
     * ONNX Runtime
     *
     * ArcFace Buffalo expects input shape:
     *
     * [batch, channels, height, width]
     *
     * [1, 3, 112, 112]
     *
     * Meaning:
     * - 1 image
     * - 3 RGB channels
     * - 112 pixel height
     * - 112 pixel width
     * ========================================================================
     */
    std::vector<int64_t> input_shape = { 1, 3, 112, 112 };

    /**
     * ========================================================================
     * STAGE A10 — ONNX RUNTIME TENSOR BINDING
     * ========================================================================
     *
     * Framework:
     * ONNX Runtime C++ API
     *
     * Ort::Value::CreateTensor does not copy the data.
     * It wraps the existing model_input vector memory.
     *
     * Important:
     * model_input must remain alive while generate_embedding() runs.
     * That is true here because model_input exists in this function scope
     * until after inference completes.
     * ========================================================================
     */
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        model_input.data(),
        model_input.size(),
        input_shape.data(),
        input_shape.size()
    );

    std::cout << "Preprocessed tensor shape: [1, 3, 112, 112]" << std::endl;
    std::cout << "Tensor size: " << model_input.size() << std::endl;

    /**
     * ========================================================================
     * STAGE A11 — ARCFACE ONNX INFERENCE
     * ========================================================================
     *
     * Framework:
     * ONNX Runtime + ArcFace ONNX model
     *
     * FaceVerifier::generate_embedding() is responsible for:
     *
     * - passing the tensor into session_.Run(...)
     * - reading the output tensor
     * - converting the output to std::vector<float>
     *
     * Expected output:
     * 512-dimensional ArcFace embedding.
     * ========================================================================
     */
    std::vector<float> embedding =
        verifier.generate_embedding(input_tensor);

    std::cout << "ArcFace embedding size: "
        << embedding.size()
        << std::endl;

    std::cout << "-----------------------------------" << std::endl;

    return embedding;
}


/**
 * ============================================================================
 * FUNCTION: main
 * ============================================================================
 *
 * Responsibility:
 *
 * Run the current TinyVerify two-image verification experiment.
 *
 * Current Test:
 *
 * data/person_a.jpg
 *     ↓
 * embedding A
 *
 * data/person_b.jpg
 *     ↓
 * embedding B
 *
 * embedding A + embedding B
 *     ↓
 * cosine similarity score
 *
 * Important:
 *
 * This stage does not apply a final verification threshold yet.
 *
 * The current purpose is only to prove that TinyVerify can:
 *
 * 1. Load two separate images.
 * 2. Detect a face in each image.
 * 3. Generate one real ArcFace embedding for each image.
 * 4. Compute cosine similarity between the two embeddings.
 *
 * Threshold-based same/different identity decisions should come after this
 * two-image pipeline is verified.
 *
 * ============================================================================
 */
int main() {

    /**
     * ========================================================================
     * STAGE 1 — OPENCV LOG CONFIGURATION
     * ========================================================================
     *
     * Framework:
     * OpenCV
     *
     * This suppresses noisy OpenCV INFO logs so TinyVerify's own pipeline logs
     * are easier to read.
     * ========================================================================
     */
    cv::utils::logging::setLogLevel(
        cv::utils::logging::LOG_LEVEL_ERROR
    );

    std::cout << "TinyVerify initialized successfully" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    /**
     * ========================================================================
     * STAGE 2 — FACE DETECTOR INITIALIZATION
     * ========================================================================
     *
     * Framework:
     * OpenCV Haar Cascade
     *
     * The Haar Cascade XML file contains pre-trained classical vision patterns
     * used to locate frontal faces.
     * ========================================================================
     */
    FaceDetector detector(
        "data/haarcascade_frontalface_default.xml"
    );

    /**
     * ========================================================================
     * STAGE 3 — IMAGE PREPROCESSOR INITIALIZATION
     * ========================================================================
     *
     * Framework:
     * OpenCV + TinyVerify preprocessing layer
     *
     * This object prepares cropped face images for ArcFace.
     * ========================================================================
     */
    ImagePreprocessor preprocessor;

    /**
     * ========================================================================
     * STAGE 4 — FACE VERIFIER INITIALIZATION
     * ========================================================================
     *
     * Framework:
     * ONNX Runtime + ArcFace ONNX model
     *
     * FaceVerifier owns the ONNX Runtime session and exposes:
     *
     * - generate_embedding()
     * - compute_similarity()
     * ========================================================================
     */
    FaceVerifier verifier(
        "models/arcface_buffalo_1.onnx"
    );

    /**
     * ========================================================================
     * STAGE 5 — ONNX RUNTIME MEMORY DESCRIPTOR
     * ========================================================================
     *
     * Framework:
     * ONNX Runtime
     *
     * This tells ONNX Runtime that the input tensor data lives in normal CPU
     * memory.
     * ========================================================================
     */
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtDeviceAllocator,
        OrtMemTypeCPU
    );

    /**
     * ========================================================================
     * STAGE 6 — GENERATE EMBEDDING A
     * ========================================================================
     *
     * Input:
     * data/person_a.jpg
     *
     * Output:
     * embedding_a
     *
     * This should be a real 512-dimensional ArcFace embedding.
     * ========================================================================
     */
    std::vector<float> embedding_a =
        generate_embedding_from_image(
            "data/person_a.jpg",
            "person_a",
            "Image A",
            detector,
            preprocessor,
            verifier,
            memory_info
        );

    if (embedding_a.empty()) {
        std::cerr << "[SYSTEM-ERROR] Failed to generate embedding A." << std::endl;
        return -1;
    }

    /**
     * ========================================================================
     * STAGE 7 — GENERATE EMBEDDING B
     * ========================================================================
     *
     * Input:
     * data/person_b.jpg
     *
     * Output:
     * embedding_b
     *
     * This should also be a real 512-dimensional ArcFace embedding.
     * ========================================================================
     */
    std::vector<float> embedding_b =
        generate_embedding_from_image(
            "data/person_b.jpg",
            "person_b",
            "Image B",
            detector,
            preprocessor,
            verifier,
            memory_info
        );

    if (embedding_b.empty()) {
        std::cerr << "[SYSTEM-ERROR] Failed to generate embedding B." << std::endl;
        return -1;
    }

    /**
     * ========================================================================
     * STAGE 8 — TWO-IMAGE COSINE SIMILARITY
     * ========================================================================
     *
     * Framework:
     * TinyVerify FaceVerifier math layer
     *
     * This compares two real ArcFace embeddings:
     *
     * similarity = dot(A, B) / (norm(A) * norm(B))
     *
     * Expected behavior:
     *
     * - Same or very similar face images should produce a higher score.
     * - Different identities should usually produce a lower score.
     *
     * Important:
     * 
     * This stage now applies a temporary, uncalibrated threshold to convert
     * the cosine similarity score into a basic SAME / DIFFERENT identity result.
     *
     * This is still not a production-grade verification decision because the
     * threshold has not been calibrated with a real evaluation dataset.
     * ========================================================================
     */
    std::cout << std::endl;
    std::cout << "[Comparison]" << std::endl;

    float similarity =
        verifier.compute_similarity(embedding_a, embedding_b);

    /**
     * Temporary verification threshold.
     *
     * This value is not calibrated yet.
     * It exists only to turn the current similarity score into a basic
     * SAME / DIFFERENT identity decision for the current end-to-end prototype.
     *
     * Future work:
     * - test more same-person image pairs
     * - test more different-person image pairs
     * - tune this threshold using real evaluation data
     */
    constexpr float verification_threshold = 0.60f;

    const bool is_same_identity =
        similarity >= verification_threshold;

    std::cout << "Two-image cosine similarity: "
        << similarity
        << std::endl;

    std::cout << "Verification threshold: "
        << verification_threshold
        << std::endl;

    std::cout << "Verification result: "
        << (is_same_identity ? "SAME identity" : "DIFFERENT identity")
        << std::endl;

    /**
     * ========================================================================
     * STAGE 9 — SUCCESSFUL PROGRAM EXIT
     * ========================================================================
     */
    return 0;
}