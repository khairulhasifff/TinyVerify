#include <opencv2/opencv.hpp>
#include <iostream>

/* ===================================================================================
ARCHITECTURE PROFILE: Image Processing Infrastructure Layer
===================================================================================
      [ Disk / Data Layer ]               ---> Reads binary image format (.jpg/.png)
                │ (File Path String)
                ▼
  ┌───────────────────────────┐
  │     cv::imread()          │           ---> Decode compressed bytes to Matrix
  └─────────────┬─────────────┘
                ▼
  ┌───────────────────────────┐
  │     cv::Mat Object        │           ---> Allocates raw pixel buffer in RAM
  └─────────────┬─────────────┘
                ▼
  ┌───────────────────────────┐
  │   Input Validation Gate   │           ---> Production Shield (Prevents segmentation faults)
  └─────────────┬─────────────┘
                ├─► [FAIL] ──► std::cerr (Safe system exit out of pipeline)
                └─► [PASS] ──► Extract Metadata (Cols, Rows, Channels)
===================================================================================
*/

void load_and_print_metadata(const std::string& image_path) {

    // ----------------------------------------------------------------------------
    // PHASE 1: DATA INGESTION & DECODING
    // ----------------------------------------------------------------------------
    // Flow: Takes relative string path -> queries operating system file handle ->
    //       passes binary stream to OpenCV's decoding codecs -> parses data to grid.
    // Memory: cv::Mat manages its own reference-counted dynamic pointer matrix allocation.
    cv::Mat image = cv::imread(image_path);

    // ----------------------------------------------------------------------------
    // PHASE 2: DEFENSIVE ENGINEERING GUARD
    // ----------------------------------------------------------------------------
    // Vibe: Production systems never trust the filesystem. 
    // Flow Interrupt: If path is mistyped, file corrupted, or permissions denied, 
    //                 image.data becomes `nullptr`. Handling it here prevents downstream crashes.
    if (image.empty()) {
        std::cerr << "[SYSTEM-ERROR] Pipeline broken. Could not ingest or decode asset at: " << image_path << "\n";
        return; // Graceful unwind. Return control to main pipeline loop safely.
    }

    // ----------------------------------------------------------------------------
    // PHASE 3: METADATA EXTRACTION & DIAGNOSTICS
    // ----------------------------------------------------------------------------
    // Structure:
    // .cols     -> Matrix width (X-axis count)
    // .rows     -> Matrix height (Y-axis count)
    // .channels()-> Depth matrix dimensions (e.g., 3 channels for BGR color arrays)
    std::cout << "[PIPELINE-SUCCESS] Image loaded successfully\n";
    std::cout << "Width   : " << image.cols << " pixels\n";
    std::cout << "Height  : " << image.rows << " pixels\n";
    std::cout << "Channels: " << image.channels() << " (BGR Multi-Channel Layout)\n";
}