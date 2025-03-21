#pragma once

#include <atomic>
#include <opencv2/opencv.hpp>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>

/**
 * @brief A class for calibrating a camera using a set of images containing a
 * known pattern.
 */
class CameraCalibrator {
public:
  /**
   * @brief Struct containing the settings for the camera calibration process.
   */
  struct Settings {
    cv::Size patternSize{9, 6};          ///< Checkerboard pattern size
    float squareSize{25.0f};             ///< Physical square size in mm
    int minImages{10};                   ///< Minimum images required
    double maxRMS{1.0};                  ///< Maximum acceptable RMS error
    int flags{cv::CALIB_RATIONAL_MODEL}; ///< Calibration flags
    std::string outputDir{"calibration_output/"}; ///< Output directory for
                                                  ///< calibration results

    /**
     * @brief Creates a default set of calibration settings.
     * @return A Settings struct with default values.
     */
    static Settings createDefault() {
      return Settings{.patternSize = {9, 6},
                      .squareSize = 25.0f,
                      .minImages = 10,
                      .maxRMS = 1.0,
                      .flags = cv::CALIB_RATIONAL_MODEL,
                      .outputDir = "calibration_output/"};
    }

    /**
     * @brief Enables or disables the fixed aspect ratio flag.
     * @param enable True to enable, false to disable.
     * @return A reference to the Settings object.
     */
    Settings &withFixedAspectRatio(bool enable = true) {
      if (enable)
        flags |= cv::CALIB_FIX_ASPECT_RATIO;
      else
        flags &= ~cv::CALIB_FIX_ASPECT_RATIO;
      return *this;
    }

    /**
     * @brief Enables or disables the zero tangential distortion flag.
     * @param enable True to enable, false to disable.
     * @return A reference to the Settings object.
     */
    Settings &withZeroTangentialDistortion(bool enable = true) {
      if (enable)
        flags |= cv::CALIB_ZERO_TANGENT_DIST;
      else
        flags &= ~cv::CALIB_ZERO_TANGENT_DIST;
      return *this;
    }

    /**
     * @brief Enables or disables the fixed principal point flag.
     * @param enable True to enable, false to disable.
     * @return A reference to the Settings object.
     */
    Settings &withFixedPrincipalPoint(bool enable = true) {
      if (enable)
        flags |= cv::CALIB_FIX_PRINCIPAL_POINT;
      else
        flags &= ~cv::CALIB_FIX_PRINCIPAL_POINT;
      return *this;
    }
  };

  /**
   * @brief Struct containing the results of the camera calibration process.
   */
  struct Results {
    cv::Mat cameraMatrix;              ///< The camera matrix
    cv::Mat distCoeffs;                ///< The distortion coefficients
    double totalRMS{0.0};              ///< The total RMS error
    std::vector<double> perViewErrors; ///< The per-view errors
    std::vector<cv::Mat> rvecs;        ///< The rotation vectors
    std::vector<cv::Mat> tvecs;        ///< The translation vectors
    double avgReprError{0.0};          ///< The average reprojection error
    double maxReprError{0.0};          ///< The maximum reprojection error
    double fovX{0.0}, fovY{0.0};       ///< The field of view in X and Y
    cv::Point2d principalPoint;        ///< The principal point
    cv::Point2d focalLength;           ///< The focal length
    double aspectRatio{0.0};           ///< The aspect ratio

    /**
     * @brief Checks if the calibration results are valid.
     * @return True if the results are valid, false otherwise.
     */
    bool isValid() const noexcept {
      return !cameraMatrix.empty() && !distCoeffs.empty() && totalRMS >= 0;
    }

    /**
     * @brief Saves the calibration results to a file.
     * @param filename The name of the file to save to.
     * @return True if the results were saved successfully, false otherwise.
     */
    bool saveToFile(const std::string &filename) const;
    /**
     * @brief Loads the calibration results from a file.
     * @param filename The name of the file to load from.
     * @return True if the results were loaded successfully, false otherwise.
     */
    bool loadFromFile(const std::string &filename);
  };

private:
  Settings settings;  ///< The calibration settings
  Results results;    ///< The calibration results
  cv::Size imageSize; ///< The size of the images used
                      ///< for calibration
  std::vector<std::vector<cv::Point3f>> objectPoints; ///< The 3D object points
  std::vector<std::vector<cv::Point2f>> imagePoints;  ///< The 2D image points
  std::vector<cv::Mat> calibrationImages;             ///< The images used for
                                                      ///< calibration
  std::atomic<bool> isCalibrated{false}; ///< Flag indicating if the camera
                                         ///< is calibrated
  std::shared_mutex mutex;               ///< Mutex for thread-safe access
                                         ///< to calibration data

  /**
   * @brief Creates a grid of 3D points for the calibration pattern.
   * @return A vector of 3D points.
   */
  [[nodiscard]] std::vector<cv::Point3f> createObjectPoints() const;

public:
  /**
   * @brief Constructor for the CameraCalibrator class.
   * @param settings The calibration settings to use.
   */
  explicit CameraCalibrator(
      const Settings &settings = Settings::createDefault());

  /**
   * @brief Detects the calibration pattern in an image.
   * @param image The image to detect the pattern in.
   * @param drawCorners True to draw the corners on the image, false otherwise.
   * @return An optional vector of 2D points representing the corners of the
   * pattern, or std::nullopt if the pattern was not found.
   */
  [[nodiscard]] std::optional<std::vector<cv::Point2f>>
  detectPattern(const cv::Mat &image, bool drawCorners = false) noexcept;

  /**
   * @brief Processes a batch of images to find calibration patterns.
   * @param imageFiles A vector of image file paths.
   * @return True if all images were processed successfully, false otherwise.
   */
  bool processImages(const std::vector<std::string> &imageFiles);

  /**
   * @brief Performs the camera calibration.
   * @return An optional Results struct containing the calibration results, or
   * std::nullopt if the calibration failed.
   */
  [[nodiscard]] std::optional<Results> calibrate();

  /**
   * @brief Calculates the calibration results.
   */
  void calculateCalibrationResults();
  /**
   * @brief Generates a report of the calibration results.
   * @param filename The name of the file to save the report to.
   */
  void generateReport(const std::string &filename = "calibration_report.txt");
  /**
   * @brief Saves the calibration data to a file.
   * @param filename The name of the file to save the data to.
   */
  void saveCalibrationData(const std::string &filename = "calibration.yml");
  /**
   * @brief Undistorts an image using the calibration results.
   * @param input The image to undistort.
   * @return The undistorted image.
   */
  cv::Mat undistortImage(const cv::Mat &input) const;
  /**
   * @brief Saves a visualization of the calibration results.
   */
  void saveCalibrationVisualization();

  /**
   * @brief Gets the calibration results.
   * @return The calibration results.
   */
  [[nodiscard]] const Results &getResults() const noexcept { return results; }
  /**
   * @brief Checks if the calibration is valid.
   * @return True if the calibration is valid, false otherwise.
   */
  [[nodiscard]] bool isCalibrationValid() const noexcept;
  /**
   * @brief Gets the camera matrix.
   * @return The camera matrix.
   */
  [[nodiscard]] const cv::Mat &getCameraMatrix() const noexcept;
  /**
   * @brief Gets the distortion coefficients.
   * @return The distortion coefficients.
   */
  [[nodiscard]] const cv::Mat &getDistCoeffs() const noexcept;
};