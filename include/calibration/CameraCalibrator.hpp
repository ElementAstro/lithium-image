#pragma once

#include <atomic>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>

// Modern C++20 calibration class
class CameraCalibrator {
public:
  struct Settings {
    cv::Size patternSize{9, 6};          // Checkerboard pattern size
    float squareSize{25.0f};             // Physical square size in mm
    int minImages{10};                   // Minimum images required
    double maxRMS{1.0};                  // Maximum acceptable RMS error
    int flags{cv::CALIB_RATIONAL_MODEL}; // Calibration flags
    std::string outputDir{"calibration_output/"};

    // C++20 designated initializers support
    static Settings createDefault() {
      return Settings{.patternSize = {9, 6},
                      .squareSize = 25.0f,
                      .minImages = 10,
                      .maxRMS = 1.0,
                      .flags = cv::CALIB_RATIONAL_MODEL,
                      .outputDir = "calibration_output/"};
    }

    // Flag helpers
    Settings &withFixedAspectRatio(bool enable = true) {
      if (enable)
        flags |= cv::CALIB_FIX_ASPECT_RATIO;
      else
        flags &= ~cv::CALIB_FIX_ASPECT_RATIO;
      return *this;
    }

    Settings &withZeroTangentialDistortion(bool enable = true) {
      if (enable)
        flags |= cv::CALIB_ZERO_TANGENT_DIST;
      else
        flags &= ~cv::CALIB_ZERO_TANGENT_DIST;
      return *this;
    }

    Settings &withFixedPrincipalPoint(bool enable = true) {
      if (enable)
        flags |= cv::CALIB_FIX_PRINCIPAL_POINT;
      else
        flags &= ~cv::CALIB_FIX_PRINCIPAL_POINT;
      return *this;
    }
  };

  struct Results {
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    double totalRMS{0.0};
    std::vector<double> perViewErrors;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    double avgReprError{0.0};
    double maxReprError{0.0};
    double fovX{0.0}, fovY{0.0};
    cv::Point2d principalPoint;
    cv::Point2d focalLength;
    double aspectRatio{0.0};

    bool isValid() const noexcept {
      return !cameraMatrix.empty() && !distCoeffs.empty() && totalRMS >= 0;
    }

    // Export/import functions
    bool saveToFile(const std::string &filename) const;
    bool loadFromFile(const std::string &filename);
  };

private:
  Settings settings;
  Results results;
  cv::Size imageSize;
  std::vector<std::vector<cv::Point3f>> objectPoints;
  std::vector<std::vector<cv::Point2f>> imagePoints;
  std::vector<cv::Mat> calibrationImages;
  std::atomic<bool> isCalibrated{false};
  std::shared_mutex mutex;

  // Create a grid of 3D points for the calibration pattern
  [[nodiscard]] std::vector<cv::Point3f> createObjectPoints() const;

public:
  explicit CameraCalibrator(
      const Settings &settings = Settings::createDefault());

  // Enhanced pattern detection with modern C++ exception handling
  [[nodiscard]] std::optional<std::vector<cv::Point2f>>
  detectPattern(const cv::Mat &image, bool drawCorners = false) noexcept;

  // Process a batch of images
  bool processImages(const std::vector<std::string> &imageFiles);

  // Modern calibration method with C++20 features
  [[nodiscard]] std::optional<Results> calibrate();

  void calculateCalibrationResults();
  void generateReport(const std::string &filename = "calibration_report.txt");
  void saveCalibrationData(const std::string &filename = "calibration.yml");
  cv::Mat undistortImage(const cv::Mat &input) const;
  void saveCalibrationVisualization();

  // Getters
  [[nodiscard]] const Results &getResults() const noexcept { return results; }
  [[nodiscard]] bool isCalibrationValid() const noexcept;
  [[nodiscard]] const cv::Mat &getCameraMatrix() const noexcept;
  [[nodiscard]] const cv::Mat &getDistCoeffs() const noexcept;
};