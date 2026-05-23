cmake_minimum_required(VERSION 3.10)
project(TinyVerify VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 🛠️ ARCHITECTURAL LINKING: Step 1
# Tell CMake exactly where our un-tracked local OpenCV package config lives.
# ${CMAKE_SOURCE_DIR} points directly to C:/Users/khair/Projects/TinyVerify
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/third_party/opencv/build")

# 🛠️ ARCHITECTURAL LINKING: Step 2
# Instruct CMake to parse OpenCVConfig.cmake to extract include paths and library names.
find_package(OpenCV REQUIRED)

# 🛠️ ARCHITECTURAL LINKING: Step 3
# Append OpenCV's header locations and our internal include directory to the compiler's search environment path.
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# 🛠️ ARCHITECTURAL LINKING: Step 4
# Formally compile ALL C++ modules together into our unified system binary executable.
# FIX: Removed the literal quotes around image_preprocessor.cpp
add_executable(TinyVerify 
    src/main.cpp 
    src/image_loader.cpp
    src/image_preprocessor.cpp
    src/face_detector.cpp
)

# 🛠️ ARCHITECTURAL LINKING: Step 5
# Instruct the linker engine to bind OpenCV's pre-compiled binary files (.lib) to TinyVerify.exe
target_link_libraries(TinyVerify ${OpenCV_LIBS})

# 🛠️ ARCHITECTURAL RUNTIME FIX: Step 6 (Adaptive Configuration)
# Automatically chooses between copying the debug ('4130d') or release ('4130') DLL
# based entirely on your Visual Studio active build configuration selector.
add_custom_command(TARGET TinyVerify POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<$<CONFIG:Debug>:${CMAKE_SOURCE_DIR}/third_party/opencv/build/x64/vc16/bin/opencv_world4130d.dll>$<$<NOT:$<CONFIG:Debug>>:${CMAKE_SOURCE_DIR}/third_party/opencv/build/x64/vc16/bin/opencv_world4130.dll>"
    $<TARGET_FILE_DIR:TinyVerify>
)

# 🛠️ DATA ASSET MIGRATION: Step 7
# Automate mirroring the data directory inside our build target execution space 
# so relative paths like "data/test_face.jpg" resolve out-of-the-box.
add_custom_command(TARGET TinyVerify POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/data"
    "$<TARGET_FILE_DIR:TinyVerify>/data"
)