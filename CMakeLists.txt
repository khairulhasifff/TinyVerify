cmake_minimum_required(VERSION 3.10)

# ============================================================================
# PROJECT INITIALIZATION LAYER
# ============================================================================
#
# Defines:
# - project identity
# - semantic version
# - global build scope
#
# ============================================================================
project(TinyVerify VERSION 1.0)

# ============================================================================
# C++ LANGUAGE CONFIGURATION
# ============================================================================
#
# Enforces:
# - C++17 standard
# - modern compiler feature availability
# - deterministic compilation behavior
#
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# LOCAL THIRD-PARTY DEPENDENCY PATHS
# ============================================================================
#
# These variables define where external native SDK dependencies
# physically exist inside the project repository.
#
# Why This Matters:
#
# TinyVerify intentionally vendors dependencies locally to:
# - improve reproducibility
# - avoid global machine dependency issues
# - simplify onboarding
# - preserve deterministic build environments
#
# ============================================================================
set(OpenCV_DIR
    "${CMAKE_SOURCE_DIR}/third_party/opencv/build"
)

set(ONNXRUNTIME_DIR
    "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-win-x64-1.26.0/onnxruntime-win-x64-1.26.0"
)

# ============================================================================
# OPENCV PACKAGE DISCOVERY
# ============================================================================
#
# Parses OpenCVConfig.cmake to automatically discover:
# - include paths
# - library binaries
# - compiler definitions
#
# ============================================================================
find_package(OpenCV REQUIRED)

# ============================================================================
# HEADER SEARCH ENVIRONMENT
# ============================================================================
#
# Exposes internal and external header directories to the compiler.
#
# Compiler Visibility:
#
# TinyVerify Internal Headers
#     ↓
# include/
#
# OpenCV SDK Headers
#     ↓
# OpenCV_INCLUDE_DIRS
#
# ONNX Runtime SDK Headers
#     ↓
# ONNXRUNTIME_DIR/include
#
# ============================================================================
include_directories(${CMAKE_SOURCE_DIR}/include)

include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("${ONNXRUNTIME_DIR}/include")

# ============================================================================
# EXECUTABLE SYSTEM COMPOSITION
# ============================================================================
#
# Formally composes all C++ modules into:
#
# TinyVerify.exe
#
# Current Architecture:
#
# main.cpp
#     ↓
# orchestration layer
#
# image_loader.cpp
#     ↓
# image ingestion layer
#
# image_preprocessor.cpp
#     ↓
# preprocessing pipeline layer
#
# face_detector.cpp
#     ↓
# detection + cropping layer
#
# face_verifier.cpp
#     ↓
# future inference + similarity layer
#
# ============================================================================
add_executable(TinyVerify

    src/main.cpp

    src/image_loader.cpp

    src/image_preprocessor.cpp

    src/face_detector.cpp

    src/face_verifier.cpp
)

# ============================================================================
# BINARY LINKING PHASE
# ============================================================================
#
# Binds external precompiled native libraries into TinyVerify.exe.
#
# Linked Systems:
#
# OpenCV
# - image processing
# - matrix operations
# - classical computer vision
#
# ONNX Runtime
# - neural network inference execution
# - ArcFace model runtime engine
#
# ============================================================================
target_link_libraries(

    TinyVerify

    ${OpenCV_LIBS}

    "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
)

# ============================================================================
# OPENCV RUNTIME DLL MIGRATION
# ============================================================================
#
# Automatically copies the correct OpenCV runtime DLL into the
# executable directory after every build.
#
# Why This Exists:
#
# Windows executables dynamically resolve DLL dependencies at runtime.
#
# Without this:
# - TinyVerify.exe would fail to launch
#
# Adaptive Behavior:
#
# Debug Build
#     ↓
# opencv_world4130d.dll
#
# Release Build
#     ↓
# opencv_world4130.dll
#
# ============================================================================
add_custom_command(TARGET TinyVerify POST_BUILD

    COMMAND ${CMAKE_COMMAND} -E copy_if_different

    "$<$<CONFIG:Debug>:${CMAKE_SOURCE_DIR}/third_party/opencv/build/x64/vc16/bin/opencv_world4130d.dll>$<$<NOT:$<CONFIG:Debug>>:${CMAKE_SOURCE_DIR}/third_party/opencv/build/x64/vc16/bin/opencv_world4130.dll>"

    $<TARGET_FILE_DIR:TinyVerify>
)

# ============================================================================
# ONNX RUNTIME DLL MIGRATION
# ============================================================================
#
# Copies ONNX Runtime execution engine DLL into the executable directory.
#
# Syntax Note: 
# Target file expressions must use the '$<TARGET_FILE_DIR:TargetName>' colon 
# syntax. Using parentheses will cause Ninja generation failures.
#
# Future Responsibility:
#
# TinyVerify.exe
#     ↓
# dynamically loads
#     ↓
# onnxruntime.dll
#     ↓
# executes ArcFace neural network inference
#
# ============================================================================
add_custom_command(TARGET TinyVerify POST_BUILD

    COMMAND ${CMAKE_COMMAND} -E copy_if_different

    "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"

    $<TARGET_FILE_DIR:TinyVerify>
)

# ============================================================================
# DATA ASSET RUNTIME MIGRATION
# ============================================================================
#
# Mirrors runtime assets into the executable directory.
#
# Current Runtime Assets:
# - test images
# - Haar cascade XML classifiers
#
# Future Runtime Assets:
# - ArcFace ONNX model
# - verification samples
# - benchmark datasets
#
# Why This Exists:
#
# Relative runtime paths should work automatically without requiring
# manual file copying after every build.
#
# ============================================================================
add_custom_command(TARGET TinyVerify POST_BUILD

    COMMAND ${CMAKE_COMMAND} -E copy_directory

    "${CMAKE_SOURCE_DIR}/data"

    "$<TARGET_FILE_DIR:TinyVerify>/data"
)