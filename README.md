# TinyVerify

TinyVerify is a C++ learning project focused on understanding 
face verification preprocessing and deployment-oriented AI engineering.

The project uses OpenCV for image preprocessing and is being developed
toward ONNX Runtime inference using the InsightFace ArcFace model.

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
| Image preprocessing pipeline | ✅ Complete |
| Resize to 112x112 | ✅ Complete |
| BGR → RGB conversion | ✅ Complete |
| Pixel normalization | ✅ Complete |
| Tensor flattening | ✅ Complete |
| Face detection | ✅ Complete |
| Face cropping pipeline | ✅ Complete |
| Runtime debug artifact generation | ✅ Complete |
| ONNX Runtime integration | ⏳ Planned |
| ArcFace embedding extraction | ⏳ Planned |
| Cosine similarity verification | ⏳ Planned |
| pybind11 Python bindings | ⏳ Planned |
| Benchmarking | ⏳ Planned |

---

## Current Preprocessing Pipeline

The current preprocessing pipeline performs:

1. Image loading using OpenCV
2. Resize image to 112x112
3. Convert BGR color space to RGB
4. Normalize pixel values from 0–255 to 0.0–1.0
5. Flatten image memory into a contiguous tensor buffer

This prepares image data for future ONNX Runtime inference.

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
5. Tensor buffer preparation for future ONNX Runtime inference

The project currently outputs:
- detected face visualization
- cropped face artifacts
- preprocessed tensor-ready image buffers

---

## Installation

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

- ONNX Runtime integration
- ArcFace embedding extraction
- Cosine similarity verification
- pybind11 Python bindings
- Benchmarking against Python implementations
- Lightweight web-based demonstration interface

---

## Connection to Malaysian eKYC Systems

This project is inspired by real-world digital identity verification
systems used in fintech and eKYC workflows.

The goal is to better understand the engineering layers behind
face verification pipelines used in modern financial platforms.