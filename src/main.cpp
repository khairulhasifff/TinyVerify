// TinyVerify: two face photos -> two ArcFace embeddings -> SAME / DIFFERENT.
// Pipeline per image: imread -> FaceDetector -> crop -> ImagePreprocessor -> FaceVerifier.

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "image_preprocessor.h"
#include "face_detector.h"
#include "face_verifier.h"

// Runs the full single-image pipeline. Returns an empty vector if the image
// cannot be read or has no face. Saves debug images to output/<debug_prefix>_*.jpg.
std::vector<float> process_image_to_embedding(
    const std::string& image_path,
    const std::string& debug_prefix,
    const std::string& display_label,
    FaceDetector& detector,
    ImagePreprocessor& preprocessor,
    FaceVerifier& verifier
) {
    cv::Mat input_image = cv::imread(image_path);  // OpenCV loads BGR
    std::cout << std::endl;
    std::cout << "[" << display_label << "]" << std::endl;
    std::cout << "Input image: " << image_path << std::endl;

    if (input_image.empty()) {
        std::cerr << "[SYSTEM-ERROR] Could not read image asset: "
            << image_path
            << std::endl;
        return {};
    }

    cv::Rect face_box = detector.detect(input_image);

    if (face_box.empty()) {
        std::cerr << "[SYSTEM-ERROR] No face detected in image: "
            << image_path
            << std::endl;
        return {};
    }

    // Draw on a copy so the box never reaches the crop that goes to the model.
    cv::Mat debug_image = input_image.clone();
    cv::rectangle(debug_image, face_box, cv::Scalar(0, 255, 0), 2);

    std::string detected_output =
        "output/" + debug_prefix + "_detected_face.jpg";
    cv::imwrite(detected_output, debug_image);

    cv::Mat cropped_face = detector.crop_face(input_image, face_box);

    std::string cropped_output =
        "output/" + debug_prefix + "_cropped_face.jpg";
    cv::imwrite(cropped_output, cropped_face);

    std::cout << "Face detected successfully" << std::endl;
    std::cout << "Bounding box: "
        << "x=" << face_box.x
        << ", y=" << face_box.y
        << ", width=" << face_box.width
        << ", height=" << face_box.height
        << std::endl;
    std::cout << "Saved debug image: " << detected_output << std::endl;
    std::cout << "Saved cropped face: " << cropped_output << std::endl;

    std::vector<float> model_input = preprocessor.preprocess(cropped_face);

    std::cout << "Preprocessed tensor shape: [1, 3, 112, 112]" << std::endl;
    std::cout << "Tensor size: " << model_input.size() << std::endl;

    std::vector<float> embedding = verifier.generate_embedding(model_input);

    std::cout << "ArcFace embedding size: " << embedding.size() << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    return embedding;
}

// Self-check for BGR -> RGB and HWC -> CHW: a solid red BGR image must come out
// as R=1, G=-1, B=-1 in the first pixel of each CHW plane.
bool validate_preprocessing_color_layout(ImagePreprocessor& preprocessor)
{
    constexpr int image_width = 112;
    constexpr int image_height = 112;
    constexpr int channel_size = image_width * image_height;

    cv::Mat red_bgr_image(
        image_height,
        image_width,
        CV_8UC3,
        cv::Scalar(0, 0, 255)
    );

    std::vector<float> tensor = preprocessor.preprocess(red_bgr_image, true);

    if (tensor.size() != channel_size * 3) {
        std::cerr << "[PREPROCESS-VALIDATION-ERROR] Unexpected tensor size: "
            << tensor.size()
            << std::endl;
        return false;
    }

    const float r = tensor[0];
    const float g = tensor[channel_size];
    const float b = tensor[channel_size * 2];

    std::cout << std::endl;
    std::cout << "[Preprocessing Validation]" << std::endl;
    std::cout << "Controlled input: OpenCV BGR red pixel cv::Scalar(0, 0, 255)" << std::endl;
    std::cout << "Expected RGB CHW first pixel: R=1, G=-1, B=-1" << std::endl;
    std::cout << "Observed RGB CHW first pixel: "
        << "R=" << r
        << ", G=" << g
        << ", B=" << b
        << std::endl;

    const bool is_valid =
        // 255 * (1/127.5) - 1 lands a float ULP below 1.0, so compare with a tolerance
        (std::abs(r - 1.0f) < 1e-5f) &&
        (std::abs(g + 1.0f) < 1e-5f) &&
        (std::abs(b + 1.0f) < 1e-5f);

    std::cout << "Preprocessing color/layout validation: "
        << (is_valid ? "PASSED" : "FAILED")
        << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    return is_valid;
}

// Function-try-block: ONNX Runtime, OpenCV and FaceVerifier report failures
// (missing model, bad tensor size, missing cascade) by throwing. Catch them
// here so the user sees the reason instead of an unexplained abort.
int main() try {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    // cv::imwrite does not create folders, so debug images were silently dropped.
    std::filesystem::create_directories("output");

    std::cout << "TinyVerify initialized successfully" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    // shortcut: demo inputs are hardcoded; move to command-line args when testing more pairs.
    constexpr const char* cascade_path = "data/haarcascade_frontalface_default.xml";
    constexpr const char* arcface_model_path = "models/arcface_buffalo_1.onnx";

    constexpr const char* image_a_path = "data/person_a.jpg";
    constexpr const char* image_b_path = "data/person_b.jpg";

    constexpr bool run_preprocessing_validation = true;

    FaceDetector detector(cascade_path);
    ImagePreprocessor preprocessor;

    if (run_preprocessing_validation &&
        !validate_preprocessing_color_layout(preprocessor)) {
        std::cerr << "[SYSTEM-ERROR] Preprocessing color/layout validation failed." << std::endl;
        return -1;
    }

    FaceVerifier verifier(arcface_model_path);

    std::vector<float> embedding_a = process_image_to_embedding(
        image_a_path, "person_a", "Image A", detector, preprocessor, verifier);

    if (embedding_a.empty()) {
        std::cerr << "[SYSTEM-ERROR] Failed to generate embedding A." << std::endl;
        return -1;
    }

    std::vector<float> embedding_b = process_image_to_embedding(
        image_b_path, "person_b", "Image B", detector, preprocessor, verifier);

    if (embedding_b.empty()) {
        std::cerr << "[SYSTEM-ERROR] Failed to generate embedding B." << std::endl;
        return -1;
    }

    std::cout << std::endl;
    std::cout << "[Comparison]" << std::endl;

    VerificationResult result = verifier.verify_pair(embedding_a, embedding_b);

    std::cout << "Two-image cosine similarity: " << result.similarity << std::endl;
    std::cout << "Verification threshold: " << result.threshold << std::endl;
    std::cout << "Verification result: "
        << (result.is_same_identity ? "SAME identity" : "DIFFERENT identity")
        << std::endl;

    std::cout << "Program exited successfully with code 0." << std::endl;
    return 0;
}
catch (const std::exception& e) {
    std::cerr << "[SYSTEM-ERROR] " << e.what() << std::endl;
    return -1;
}
