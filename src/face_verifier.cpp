#include "face_verifier.h"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <array>
#include <filesystem>
#include <string>

FaceVerifier::FaceVerifier(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "TinyVerify"),
      session_(nullptr)
{
    // Intra-op thread count is left at ONNX Runtime's default (one per
    // physical core); forcing 1 thread made every inference single-core.
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );

    // std::filesystem::path converts to the platform's native ORTCHAR_T
    // string (wchar_t on Windows) correctly, including non-ASCII paths.
    const std::filesystem::path native_model_path(model_path);

    session_ = Ort::Session(
        env_,
        native_model_path.c_str(),
        session_options
    );

    std::cout << "FaceVerifier ONNX session loaded successfully" << std::endl;
    std::cout << "Model path: " << model_path << std::endl;
}

std::vector<float> FaceVerifier::generate_embedding(
    const std::vector<float>& input_tensor_data
)
{
    const std::array<int64_t, 4> input_shape = { 1, 3, 112, 112 };
    constexpr size_t EXPECTED_TENSOR_SIZE = 1 * 3 * 112 * 112;

    // Fail here with a clear message instead of deep inside ONNX Runtime.
    if (input_tensor_data.size() != EXPECTED_TENSOR_SIZE)
    {
        throw std::runtime_error(
            "Invalid ArcFace input tensor size. Expected 37632 floats, but received " +
            std::to_string(input_tensor_data.size())
        );
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault
    );

    // ONNX Runtime only reads the input buffer, but its API takes a non-const pointer.
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input_tensor_data.data()),
        input_tensor_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    // Ask the model for its node names instead of hardcoding "input.1" etc.,
    // so other ArcFace exports work too.
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name_allocated = session_.GetInputNameAllocated(0, allocator);
    auto output_name_allocated = session_.GetOutputNameAllocated(0, allocator);

    const char* input_name = input_name_allocated.get();
    const char* output_name = output_name_allocated.get();

    std::cout << "ONNX Input Name: " << input_name << std::endl;
    std::cout << "ONNX Output Name: " << output_name << std::endl;

    const char* input_names[] = { input_name };
    const char* output_names[] = { output_name };

    std::vector<Ort::Value> output_tensors = session_.Run(
        Ort::RunOptions{ nullptr },
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    if (output_tensors.empty())
    {
        throw std::runtime_error("ONNX Runtime returned no output tensors");
    }

    Ort::Value& output_tensor = output_tensors.front();
    size_t output_element_count =
        output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();

    constexpr size_t EXPECTED_ARCFACE_EMBEDDING_SIZE = 512;

    if (output_element_count != EXPECTED_ARCFACE_EMBEDDING_SIZE)
    {
        throw std::runtime_error(
            "Invalid ArcFace output embedding size. Expected 512 floats, but received " +
            std::to_string(output_element_count)
        );
    }

    // Copy out so callers never depend on ONNX Runtime's tensor lifetime.
    float* output_data = output_tensor.GetTensorMutableData<float>();

    std::vector<float> embedding(
        output_data,
        output_data + output_element_count
    );

    std::cout << "Real ONNX inference completed successfully" << std::endl;
    std::cout << "Embedding size: " << embedding.size() << std::endl;

    return embedding;
}

float FaceVerifier::compute_similarity(
    const std::vector<float>& embedding_a,
    const std::vector<float>& embedding_b
) const {
    if (embedding_a.empty() || embedding_b.empty()) {
        throw std::runtime_error("Cannot compute similarity: embedding is empty.");
    }

    if (embedding_a.size() != embedding_b.size()) {
        throw std::runtime_error("Cannot compute similarity: embedding sizes do not match.");
    }

    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < embedding_a.size(); ++i) {
        dot_product += embedding_a[i] * embedding_b[i];
        norm_a += embedding_a[i] * embedding_a[i];
        norm_b += embedding_b[i] * embedding_b[i];
    }

    if (norm_a == 0.0f || norm_b == 0.0f) {
        throw std::runtime_error("Cannot compute similarity: embedding norm is zero.");
    }

    return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

VerificationResult FaceVerifier::verify_pair(
    const std::vector<float>& embedding_a,
    const std::vector<float>& embedding_b,
    float threshold
) const {
    const float similarity = compute_similarity(embedding_a, embedding_b);
    return { similarity, threshold, similarity >= threshold };
}
