# TinyVerify

A C++ learning project for face verification: two photos in, SAME or DIFFERENT
identity out. It uses OpenCV for detection and preprocessing, and ONNX Runtime
to run the InsightFace ArcFace model.

It's built as a portfolio project, inspired by eKYC and fintech identity checks.
Production inference is often written in C++ for speed and control. This
project explores that deployment side: preprocessing, tensor preparation and
running the model.

---

## Pipeline

```text
Image A / Image B
    ↓
Haar cascade face detection (largest face)
    ↓
Face crop
    ↓
Preprocess: resize 112x112 → BGR→RGB → (x - 127.5) / 127.5 → HWC→CHW
    ↓
FaceVerifier::generate_embedding  (ONNX Runtime, input [1,3,112,112] → 512 floats)
    ↓
Cosine similarity
    ↓
FaceVerifier::verify_pair  (threshold 0.60) → VerificationResult
```

| File | Job |
|---|---|
| `src/main.cpp` | Runs the two-image demo and the preprocessing self-check |
| `src/face_detector.cpp` | Haar cascade detection and cropping |
| `src/image_preprocessor.cpp` | Builds the 37,632-float CHW tensor |
| `src/face_verifier.cpp` | ONNX session, embedding, similarity, decision |

`FaceVerifier` validates both sides of the model contract: the input must be
37,632 floats and the output must be 512 floats. Otherwise it throws with a clear
message.

---

## Status

| Part | Status |
|---|---|
| Detection, cropping, debug images in `output/` | ✅ |
| Preprocessing + red-pixel color/layout self-check | ✅ |
| Real ArcFace inference, 512-d embedding | ✅ |
| Cosine similarity + SAME / DIFFERENT result | ✅ (threshold not calibrated) |
| pybind11 Python bindings | ⏳ Planned |
| Benchmarking | ⏳ Planned |
| IC card / MyKad document scanner | ⏳ Not started |

## Known gaps

- The 0.60 threshold is a guess. It needs calibration on real same-person and different-person pairs.
- Faces are not aligned to ArcFace's 5-point landmark template before inference.
- Preprocessing has one controlled self-check, not a full test suite.
- Inputs are hardcoded (`data/person_a.jpg`, `data/person_b.jpg`).

---

## Build

You'll need Windows 10/11, Visual Studio 2022 and CMake 3.10+.

These folders are git-ignored. Put them in place before building:

```text
third_party/opencv/build/                                     OpenCV 4.13 (opencv_world4130.dll)
third_party/onnxruntime-win-x64-1.26.0/onnxruntime-win-x64-1.26.0/   ONNX Runtime 1.26
models/arcface_buffalo_1.onnx                                 InsightFace ArcFace (buffalo_l)
```

The build copies the DLLs, `data/` and `models/` next to the executable. Run it
from that folder.

## Example output

```text
TinyVerify initialized successfully
-----------------------------------
Width: 112
Height: 112
Min: -1
Max: 1
Tensor size: 37632

[Preprocessing Validation]
Controlled input: OpenCV BGR red pixel cv::Scalar(0, 0, 255)
Expected RGB CHW first pixel: R=1, G=-1, B=-1
Observed RGB CHW first pixel: R=1, G=-1, B=-1
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
...same steps for data/person_b.jpg...

[Comparison]
Two-image cosine similarity: 0.982955
Verification threshold: 0.6
Verification result: SAME identity
Program exited successfully with code 0.
```

A higher cosine similarity means the two faces are more alike. The SAME /
DIFFERENT call is a prototype until the threshold is calibrated.

---

## Next

- Calibrate the threshold on more image pairs
- Align faces with landmarks before inference
- pybind11 bindings and a benchmark against Python
- A small web demo
