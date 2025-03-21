#include "calibration/CameraCalibrator.hpp"
#include "Logging.hpp"

#include <algorithm>
#include <chrono>
#include <future>
#include <fstream>

CameraCalibrator::CameraCalibrator(const Settings &settings)
    : settings(settings) {
  std::filesystem::create_directories(settings.outputDir);
}

std::vector<cv::Point3f> CameraCalibrator::createObjectPoints() const {
  std::vector<cv::Point3f> points;
  points.reserve(settings.patternSize.width * settings.patternSize.height);

  for (int i = 0; i < settings.patternSize.height; i++) {
    for (int j = 0; j < settings.patternSize.width; j++) {
      points.emplace_back(j * settings.squareSize, i * settings.squareSize, 0);
    }
  }
  return points;
}

std::optional<std::vector<cv::Point2f>>
CameraCalibrator::detectPattern(const cv::Mat &image,
                                bool drawCorners) noexcept {
  if (image.empty())
    return std::nullopt;

  try {
    std::vector<cv::Point2f> corners;
    cv::Mat gray;

    // Convert to grayscale if needed
    if (image.channels() == 3 || image.channels() == 4)
      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
      gray = image.clone();

    // Try different detection methods
    bool found = cv::findChessboardCorners(gray, settings.patternSize, corners,
                                           cv::CALIB_CB_ADAPTIVE_THRESH +
                                               cv::CALIB_CB_NORMALIZE_IMAGE +
                                               cv::CALIB_CB_FAST_CHECK);

    if (!found) {
      found = cv::findCirclesGrid(gray, settings.patternSize, corners,
                                  cv::CALIB_CB_ASYMMETRIC_GRID);
    }

    if (found) {
      // Refine corner locations
      cv::cornerSubPix(
          gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30,
                           0.1));

      if (drawCorners) {
        cv::Mat display = image.clone();
        cv::drawChessboardCorners(display, settings.patternSize, corners,
                                  found);
        cv::imshow("Pattern Detection", display);
        cv::waitKey(100);
      }

      return corners;
    }

    return std::nullopt;
  } catch (const cv::Exception &e) {
    const auto &logger = Logger::getInstance();
    logger->error("OpenCV error during pattern detection: {}", e.what());
    return std::nullopt;
  } catch (const std::exception &e) {
    const auto &logger = Logger::getInstance();
    logger->error("Error during pattern detection: {}", e.what());
    return std::nullopt;
  }
}

bool CameraCalibrator::processImages(
    const std::vector<std::string> &imageFiles) {
  const auto &logger = Logger::getInstance();
  logger->info("Processing {} calibration images", imageFiles.size());

  calibrationImages.clear();
  imagePoints.clear();
  objectPoints.clear();

  std::vector<
      std::future<std::optional<std::pair<cv::Mat, std::vector<cv::Point2f>>>>>
      futures;

  for (const auto &file : imageFiles) {
    futures.push_back(std::async(
        std::launch::async,
        [file, this]()
            -> std::optional<std::pair<cv::Mat, std::vector<cv::Point2f>>> {
          try {
            cv::Mat image = cv::imread(file);
            if (image.empty()) {
              const auto &logger = Logger::getInstance();
              logger->warn("Failed to load image: {}", file);
              return std::nullopt;
            }

            auto corners = detectPattern(image, true);
            if (!corners)
              return std::nullopt;

            // Set image size from first valid image
            if (imageSize.empty()) {
              std::unique_lock lock(mutex);
              if (imageSize.empty()) {
                imageSize = image.size();
              }
            }

            return std::make_pair(image, *corners);
          } catch (...) {
            return std::nullopt;
          }
        }));
  }

  int validCount = 0;
  for (auto &future : futures) {
    auto result = future.get();
    if (result) {
      std::unique_lock lock(mutex);
      calibrationImages.push_back(result->first);
      imagePoints.push_back(result->second);
      objectPoints.push_back(createObjectPoints());
      validCount++;
    }
  }

  logger->info("Found valid patterns in {}/{} images", validCount,
               imageFiles.size());
  return validCount >= settings.minImages;
}

std::optional<CameraCalibrator::Results> CameraCalibrator::calibrate() {
  const auto &logger = Logger::getInstance();

  std::shared_lock readLock(mutex);
  if (imagePoints.empty() || imagePoints.size() < settings.minImages) {
    logger->error(
        "Insufficient valid images for calibration. Found {}, need {}",
        imagePoints.size(), settings.minImages);
    return std::nullopt;
  }

  if (imageSize.empty()) {
    logger->error("Image size not determined");
    return std::nullopt;
  }
  readLock.unlock();

  try {
    // Need exclusive lock for calibration
    std::unique_lock writeLock(mutex);

    // Initialize output matrices
    results.cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    results.distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

    // Perform calibration
    results.totalRMS = cv::calibrateCamera(
        objectPoints, imagePoints, imageSize, results.cameraMatrix,
        results.distCoeffs, results.rvecs, results.tvecs, settings.flags);

    isCalibrated = true;

    // Calculate detailed results
    calculateCalibrationResults();

    logger->info("Calibration completed with RMS error: {}", results.totalRMS);

    // Save calibration parameters
    saveCalibrationData();

    return results;
  } catch (const cv::Exception &e) {
    logger->error("OpenCV error during calibration: {}", e.what());
    return std::nullopt;
  } catch (const std::exception &e) {
    logger->error("Error during calibration: {}", e.what());
    return std::nullopt;
  }
}

void CameraCalibrator::calculateCalibrationResults() {
  const auto &logger = Logger::getInstance();

  if (!isCalibrated) {
    logger->warn("Cannot calculate results: not calibrated yet");
    return;
  }

  try {
    // Calculate reprojection errors
    results.perViewErrors.clear();
    results.avgReprError = 0;
    results.maxReprError = 0;

    for (size_t i = 0; i < objectPoints.size(); i++) {
      std::vector<cv::Point2f> projectedPoints;
      cv::projectPoints(objectPoints[i], results.rvecs[i], results.tvecs[i],
                        results.cameraMatrix, results.distCoeffs,
                        projectedPoints);

      double err = cv::norm(imagePoints[i], projectedPoints, cv::NORM_L2);
      err /= projectedPoints.size();
      results.perViewErrors.push_back(err);

      results.avgReprError += err;
      results.maxReprError = std::max(results.maxReprError, err);
    }
    results.avgReprError /= objectPoints.size();

    // Calculate FOV and other parameters
    cv::calibrationMatrixValues(results.cameraMatrix, imageSize, 0.0,
                                0.0, // Assume sensor size unknown
                                results.fovX, results.fovY,
                                results.focalLength.x, results.principalPoint,
                                results.aspectRatio);

    logger->info("Average reprojection error: {}", results.avgReprError);
  } catch (const std::exception &e) {
    logger->error("Error calculating calibration results: {}", e.what());
  }
}

void CameraCalibrator::generateReport(const std::string &filename) {
  const auto &logger = Logger::getInstance();

  if (!isCalibrated) {
    logger->warn("Cannot generate report: not calibrated yet");
    return;
  }

  try {
    std::ofstream report(settings.outputDir + filename);
    if (!report.is_open()) {
      logger->error("Failed to open report file: {}",
                    settings.outputDir + filename);
      return;
    }

    report << "Camera Calibration Report\n";
    report << "========================\n\n";

    // Get current date and time
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::string datetime = std::ctime(&time_t);
    datetime.pop_back(); // Remove trailing newline

    report << "Calibration Date: " << datetime << "\n\n";

    report << "Settings:\n";
    report << "- Pattern Size: " << settings.patternSize.width << "x"
           << settings.patternSize.height << "\n";
    report << "- Square Size: " << settings.squareSize << "mm\n";
    report << "- Number of images: " << calibrationImages.size() << "\n\n";

    report << "Results:\n";
    report << "- RMS Error: " << results.totalRMS << "\n";
    report << "- Average Reprojection Error: " << results.avgReprError << "\n";
    report << "- Maximum Reprojection Error: " << results.maxReprError << "\n";
    report << "- FOV: " << results.fovX << "x" << results.fovY << " degrees\n";
    report << "- Principal Point: (" << results.principalPoint.x << ", "
           << results.principalPoint.y << ")\n";
    report << "- Focal Length: (" << results.focalLength.x << ", "
           << results.focalLength.y << ")\n";
    report << "- Aspect Ratio: " << results.aspectRatio << "\n\n";

    report << "Camera Matrix:\n" << results.cameraMatrix << "\n\n";
    report << "Distortion Coefficients:\n" << results.distCoeffs << "\n";

    report.close();
    logger->info("Calibration report generated: {}",
                 settings.outputDir + filename);
  } catch (const std::exception &e) {
    logger->error("Error generating report: {}", e.what());
  }
}

void CameraCalibrator::saveCalibrationData(const std::string &filename) {
  if (!isCalibrated)
    return;

  try {
    results.saveToFile(settings.outputDir + filename);
  } catch (...) {
    const auto &logger = Logger::getInstance();
    logger->error("Failed to save calibration data");
  }
}

cv::Mat CameraCalibrator::undistortImage(const cv::Mat &input) const {
  if (!isCalibrated || input.empty()) {
    return input.clone();
  }

  cv::Mat output;
  cv::undistort(input, output, results.cameraMatrix, results.distCoeffs);
  return output;
}

void CameraCalibrator::saveCalibrationVisualization() {
  const auto &logger = Logger::getInstance();

  if (!isCalibrated) {
    logger->warn("Cannot visualize: not calibrated yet");
    return;
  }

  std::shared_lock lock(mutex);
  auto images = calibrationImages; // Make a copy to safely unlock
  lock.unlock();

  try {
    for (size_t i = 0; i < images.size(); i++) {
      // Create undistorted image
      cv::Mat undistorted = undistortImage(images[i]);

      // Create side-by-side comparison
      cv::Mat comparison;
      cv::hconcat(images[i], undistorted, comparison);

      // Add error information
      std::string text =
          "RMS Error: " + (i < results.perViewErrors.size()
                               ? std::to_string(results.perViewErrors[i])
                               : "N/A");

      cv::putText(comparison, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                  1.0, cv::Scalar(0, 255, 0), 2);

      // Save comparison image
      std::string filename =
          settings.outputDir + "comparison_" + std::to_string(i) + ".jpg";
      cv::imwrite(filename, comparison);
    }
    logger->info("Saved {} calibration visualizations", images.size());
  } catch (const std::exception &e) {
    logger->error("Error saving visualizations: {}", e.what());
  }
}

bool CameraCalibrator::isCalibrationValid() const noexcept {
  return isCalibrated && results.isValid();
}

const cv::Mat &CameraCalibrator::getCameraMatrix() const noexcept {
  return results.cameraMatrix;
}

const cv::Mat &CameraCalibrator::getDistCoeffs() const noexcept {
  return results.distCoeffs;
}

bool CameraCalibrator::Results::saveToFile(const std::string &filename) const {
  try {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened())
      return false;

    fs << "camera_matrix" << cameraMatrix;
    fs << "dist_coeffs" << distCoeffs;
    fs << "rms" << totalRMS;
    fs << "avg_error" << avgReprError;
    fs << "max_error" << maxReprError;
    fs << "fov_x" << fovX;
    fs << "fov_y" << fovY;

    fs.release();
    return true;
  } catch (...) {
    return false;
  }
}

bool CameraCalibrator::Results::loadFromFile(const std::string &filename) {
  try {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
      return false;

    fs["camera_matrix"] >> cameraMatrix;
    fs["dist_coeffs"] >> distCoeffs;
    fs["rms"] >> totalRMS;
    fs["avg_error"] >> avgReprError;
    fs["max_error"] >> maxReprError;
    fs["fov_x"] >> fovX;
    fs["fov_y"] >> fovY;

    fs.release();
    return true;
  } catch (...) {
    return false;
  }
}