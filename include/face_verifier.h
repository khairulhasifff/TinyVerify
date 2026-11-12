#pragma once

#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>

struct VerificationResult {
    float similarity;
    float threshold;
    bool is_same_identity;
};

// Runs ArcFace through ONNX Runtime and compares the resulting embeddings.
class FaceVerifier {
public:
    FaceVerifier(const std::string& model_path);

    // Input: CHW float buffer, shape [1, 3, 112, 112], values in [-1, 1]. Output: 512 floats.
    std::vector<float> generate_embedding(const std::vector<float>& input_tensor_data);

    float compute_similarity(
        const std::vector<float>& embedding_a,
        const std::vector<float>& embedding_b
    ) const;

    // shortcut: 0.60 is a guess, not calibrated; tune it on real same/different pairs.
    VerificationResult verify_pair(
        const std::vector<float>& embedding_a,
        const std::vector<float>& embedding_b,
        float threshold = 0.60f
    ) const;

private:
    Ort::Env env_;  // must outlive session_, so it is declared first
    Ort::Session session_;
};
