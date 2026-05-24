#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/**
 * ============================================================================
 * MODULE: Face Verification Inference Layer
 * ============================================================================
 *
 * Responsibility:
 *
 * Execute identity verification inference using facial embeddings.
 *
 * Future Pipeline Role:
 *
 * Cropped Face
 *     ↓
 * ImagePreprocessor
 *     ↓
 * Tensor Buffer
 *     ↓
 * FaceVerifier
 *     ↓
 * ONNX Runtime Inference
 *     ↓
 * ArcFace Embedding Vector
 *     ↓
 * Cosine Similarity
 *     ↓
 * Verification Decision
 *
 * Why This Exists:
 *
 * This module isolates all model inference responsibilities from:
 * - preprocessing
 * - face detection
 * - orchestration logic
 *
 * Future Responsibilities:
 * - load ArcFace ONNX model
 * - execute ONNX Runtime inference
 * - extract facial embeddings
 * - compute cosine similarity
 * - return verification confidence
 *
 * ============================================================================
 */

class FaceVerifier {
public:

    /**
     * @brief Constructor
     *
     * Future responsibility:
     * - initialize ONNX Runtime session
     * - load ArcFace model into memory
     */
    FaceVerifier(const std::string& model_path);

    /**
     * @brief Generate embedding vector from preprocessed tensor
     *
     * Future Input:
     * - normalized tensor buffer
     *
     * Future Output:
     * - facial embedding vector
     */
    std::vector<float> generate_embedding(
        const std::vector<float>& input_tensor
    );

    /**
     * @brief Compare two facial embeddings
     *
     * Future Output:
     * - cosine similarity score
     */
    float compute_similarity(
        const std::vector<float>& embedding_a,
        const std::vector<float>& embedding_b
    );

private:

    /**
     * Future ONNX Runtime session ownership
     *
     * Will later contain:
     * - Ort::Session
     * - Ort::Env
     * - runtime inference state
     */
};