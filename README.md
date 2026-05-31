# TinyVerify

TinyVerify is a C++ learning project focused on understanding 
face verification preprocessing and deployment-oriented AI engineering.

The project uses OpenCV for image preprocessing and ONNX Runtime inference
using the InsightFace ArcFace model.

Built as a portfolio project inspired by real-world eKYC and fintech
verification systems.

---

## Problem Statement

Modern eKYC systems require fast and reliable face verification pipelines.
While many AI workflows are prototyped in Python, deployment-oriented
inference systems are often implemented in C++ for performance and
system-level control.

TinyVerify explores the preprocessing and inference pipeline behind
face verification systems using OpenCV and ONNX Runtime.

---

## Engineering Focus

TinyVerify focuses on understanding the deployment side of AI systems,
including preprocessing pipelines, runtime inference orchestration,
tensor preparation, and scalable verification architecture.

---

## Planned Architecture

```text
Image Input
    ↓
Face Detection
    ↓
Face Crop
    ↓
Image Preprocessing
    ↓
Tensor Buffer Generation
    ↓
FaceVerifier
    ↓
ONNX Runtime
    ↓
ArcFace ONNX Model
    ↓
Embedding Vector
    ↓
Cosine Similarity
    ↓
Verification Result
```

---

## Current Progress

| Component | Status |
|---|---|
| Image loading (OpenCV) | ✅ Complete |
| Image preprocessing pipeline | ✅ Working with controlled validation |
| Resize to 112x112 | ✅ Complete |
| BGR → RGB conversion | ✅ Validated with controlled color test |
| Pixel normalization | ✅ Complete |
| Tensor buffer generation | ✅ Complete |
| CHW tensor buffer generation | ✅ Complete |
| Face detection | ✅ Complete |
| Face cropping pipeline | ✅ Complete |
| Runtime debug artifact generation | ✅ Complete |
| FaceVerifier module scaffold | ✅ Complete |
| ONNX Runtime dependency integration | ✅ Complete |
| Placeholder embedding pipeline | ✅ Replaced by real ONNX inference |
| ONNX Runtime session initialization | ✅ Complete |
| ArcFace ONNX model loading | 🔄 Works when model file exists |
| ONNX input tensor binding | ✅ Complete |
| Real ONNX Runtime inference execution | ✅ Complete |
| ArcFace embedding extraction | ✅ Complete |
| Cosine similarity computation | ✅ Complete |
| Two-image embedding comparison | ✅ Complete |
| Temporary threshold-based verification result | ✅ Complete |
| `verify_pair()` verification workflow | ✅ Complete |
| pybind11 Python bindings | ⏳ Planned / not importable yet |
| Benchmarking | ⏳ Planned |
| IC card / MyKad document scanner | ⏳ Not started |

---

## Current Preprocessing Pipeline

The current preprocessing pipeline performs:

1. Image loading using OpenCV
2. Resize image to 112x112
3. Convert BGR color space to RGB
4. Normalize pixel values from 0–255 to 0.0–1.0
5. Convert OpenCV HWC image memory into CHW tensor layout
6. Store the result in a contiguous tensor buffer

This prepares image data for ONNX Runtime input tensor binding.

---

## Current Pipeline Status

The current TinyVerify pipeline successfully performs:

1. Image loading using OpenCV
2. Face detection using Haar Cascade classifiers
3. Face region extraction (cropping)
4. Image preprocessing:
   - resize to 112x112
   - BGR → RGB conversion
   - normalization to 0.0–1.0
5. CHW tensor buffer preparation for ONNX Runtime inference
6. ONNX Runtime input tensor binding
7. ArcFace inference execution
8. Real 512-dimensional embedding extraction for image A
9. Real 512-dimensional embedding extraction for image B
10. Cosine similarity computation between two real embeddings
11. Temporary threshold comparison
12. SAME / DIFFERENT identity result output

The project currently outputs:
- detected face visualizations for both input images
- cropped face artifacts for both input images
- preprocessed tensor-ready image buffers
- real ArcFace embedding size confirmation for both images
- cosine similarity score between two real ArcFace embeddings
- temporary verification threshold
- SAME / DIFFERENT identity result

---

## Current Inference Architecture

TinyVerify now includes a dedicated `FaceVerifier` module responsible for
ONNX Runtime inference orchestration.

The top-level `main.cpp` orchestration has also been cleaned up with a reusable
`process_image_to_embedding()` helper. This helper runs the full single-image
pipeline for each input image: image loading, face detection, debug artifact
export, face cropping, preprocessing, ONNX tensor binding, and ArcFace embedding
generation.

The current inference layer supports:

- ONNX Runtime SDK integration
- modular inference ownership separation
- ONNX Runtime session initialization
- ArcFace ONNX model loading when the model file exists
- ONNX input tensor binding
- real ONNX Runtime `session.Run(...)` execution
- ArcFace output tensor extraction
- real 512-dimensional embedding generation
- cosine similarity computation
- temporary threshold-based verification through `FaceVerifier::verify_pair()`
- structured `VerificationResult` output

Current embedding behavior now uses real ONNX Runtime inference. A preprocessed
CHW float tensor is bound to an ONNX Runtime input tensor, executed through the
ArcFace ONNX model using `session.Run(...)`, and copied into a real
512-dimensional embedding vector.

The preprocessing tensor layout has been updated from OpenCV-style HWC memory
to ONNX-compatible CHW layout for the input shape `[1, 3, 112, 112]`.

The current verified pipeline is:

```text
Image loading
    ↓
Face detection
    ↓
Face crop
    ↓
Image preprocessing
    ↓
Tensor Buffer Generation
    ↓
FaceVerifier
    ↓
ONNX Runtime session setup
    ↓
ONNX input tensor binding
    ↓
ONNX Runtime session.Run(...)
    ↓
ArcFace output tensor extraction
    ↓
Real 512-dimensional embedding generation
    ↓
Cosine similarity computation
    ↓
FaceVerifier::verify_pair(...)
    ↓
VerificationResult
```

The current verified two-image comparison pipeline is:

```text
Image A
    ↓
Face detection
    ↓
Face crop
    ↓
Preprocessing
    ↓
ArcFace embedding A

Image B
    ↓
Face detection
    ↓
Face crop
    ↓
Preprocessing
    ↓
ArcFace embedding B
    
Embedding A + Embedding B
    ↓
FaceVerifier::verify_pair(...)
    ↓
Cosine similarity score
    ↓
Temporary threshold comparison
    ↓
VerificationResult
    ↓
SAME / DIFFERENT identity result
```

The current threshold-based result is temporary and uncalibrated, but the decision logic has now been moved into a dedicated `FaceVerifier::verify_pair()` workflow.

---

## Known Gaps

The following parts are not yet complete or not fully validated:

- Threshold-based verification now lives in `FaceVerifier::verify_pair()`, but the threshold is still temporary and uncalibrated.
- BGR → RGB preprocessing has a basic controlled validation, but broader preprocessing tests are still needed.
- Python bindings are not currently importable.
- Benchmarking scripts are not implemented yet.
- IC card / MyKad document scanning has not started.
- A larger real sample image dataset is still needed for testing.

---

## Example Runtime Output

The example below shows TinyVerify processing two input images:

- `data/person_a.jpg`
- `data/person_b.jpg`

For each image, TinyVerify detects a face, saves debug artifacts, preprocesses the cropped face, binds the tensor for ONNX Runtime inference, runs the ArcFace model, and extracts a 512-dimensional embedding.

Finally, TinyVerify compares both embeddings through `FaceVerifier::verify_pair()`, which computes cosine similarity and applies a temporary threshold.

```text
TinyVerify initialized successfully
-----------------------------------
Width: 112
Height: 112
Min: 0
Max: 1
Tensor size: 37632

[Preprocessing Validation]
Controlled input: OpenCV BGR red pixel cv::Scalar(0, 0, 255)
Expected RGB CHW first pixel: R=1, G=0, B=0
Observed RGB CHW first pixel: R=1, G=0, B=0
Preprocessing color/layout validation: PASSED
-----------------------------------
FaceVerifier ONNX session loaded successfully
Model path: models/arcface_buffalo_1.onnx

[Image A]
Input image: data/person_a.jpg
Face detected successfully
Bounding box: x=105, y=169, width=249, height=249
Saved debug image: output/person_a_detected_face.jpg
Saved cropped face: output/person_a_cropped_face.jpg
Preprocessed tensor shape: [1, 3, 112, 112]
Tensor size: 37632
ONNX Input Name: input.1
ONNX Output Name: 683
Real ONNX inference completed successfully
Embedding size: 512
ArcFace embedding size: 512
-----------------------------------

[Image B]
Input image: data/person_b.jpg
Face detected successfully
Bounding box: x=108, y=171, width=243, height=243
Saved debug image: output/person_b_detected_face.jpg
Saved cropped face: output/person_b_cropped_face.jpg
Preprocessed tensor shape: [1, 3, 112, 112]
Tensor size: 37632
ONNX Input Name: input.1
ONNX Output Name: 683
Real ONNX inference completed successfully
Embedding size: 512
ArcFace embedding size: 512
-----------------------------------

[Comparison]
Two-image cosine similarity: 0.9829
Verification threshold: 0.6
Verification result: SAME identity
Program exited successfully with code 0.
```

In this run, TinyVerify generated two real ArcFace embeddings, computed a cosine similarity score of `0.9829`, compared it against a temporary threshold of `0.6`, and returned `SAME identity`.

A higher cosine similarity score means the two embeddings are more similar. The current SAME / DIFFERENT result is only a prototype decision because the threshold has not yet been calibrated with a larger evaluation dataset.

---

## Installation

### Dependencies

TinyVerify currently integrates:

- OpenCV 4.x
- ONNX Runtime 1.x

### Prerequisites

- Windows 10/11
- Visual Studio 2022
- CMake 3.10+
- OpenCV 4.x

---

## Learning Goals

This project is being used to learn:

- OpenCV image preprocessing
- C++ inference pipelines
- ONNX Runtime integration
- Deployment-oriented AI engineering
- Face verification system architecture

---

## Future Goals

Planned future improvements include:

- Calibrate the `FaceVerifier::verify_pair()` threshold using more same-person and different-person image pairs
- pybind11 Python bindings
- Benchmarking against Python implementations
- Lightweight web-based demonstration interface

---

## Connection to Malaysian eKYC Systems

This project is inspired by real-world digital identity verification
systems used in fintech and eKYC workflows.

The goal is to better understand the engineering layers behind
face verification pipelines used in modern financial platforms.