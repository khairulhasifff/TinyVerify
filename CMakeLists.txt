cmake_minimum_required(VERSION 3.10)
project(TinyVerify VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies are vendored locally (git-ignored); see README "Build".
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/third_party/opencv/build")
set(ONNXRUNTIME_DIR
    "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-win-x64-1.26.0/onnxruntime-win-x64-1.26.0"
)

find_package(OpenCV REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("${ONNXRUNTIME_DIR}/include")

add_executable(TinyVerify
    src/main.cpp
    src/image_preprocessor.cpp
    src/face_detector.cpp
    src/face_verifier.cpp
)

target_link_libraries(
    TinyVerify
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
)

# Windows resolves DLLs next to the .exe, so copy the runtime DLLs and the
# data/models folders there after every build.
add_custom_command(TARGET TinyVerify POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<$<CONFIG:Debug>:${CMAKE_SOURCE_DIR}/third_party/opencv/build/x64/vc16/bin/opencv_world4130d.dll>$<$<NOT:$<CONFIG:Debug>>:${CMAKE_SOURCE_DIR}/third_party/opencv/build/x64/vc16/bin/opencv_world4130.dll>"
    $<TARGET_FILE_DIR:TinyVerify>
)

# Use the $<TARGET_FILE_DIR:Target> colon syntax; parentheses break Ninja generation.
add_custom_command(TARGET TinyVerify POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
    $<TARGET_FILE_DIR:TinyVerify>
)

add_custom_command(TARGET TinyVerify POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/data"
    "$<TARGET_FILE_DIR:TinyVerify>/data"
)

add_custom_command(TARGET TinyVerify POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/models"
    "$<TARGET_FILE_DIR:TinyVerify>/models"
)
