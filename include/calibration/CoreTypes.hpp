#pragma once

#include <concepts>
#include <future>
#include <opencv2/opencv.hpp>

/**
 * @brief C++20 concept to check if a type is an image container.
 *
 * This concept requires the type to have members `cols`, `rows`, `channels()`,
 * and `type()` that are convertible to `int`.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept ImageContainer = requires(T a) {
  { a.cols } -> std::convertible_to<int>;
  { a.rows } -> std::convertible_to<int>;
  { a.channels() } -> std::convertible_to<int>;
  { a.type() } -> std::convertible_to<int>;
};

/**
 * @brief Base exception class for calibration errors.
 */
class CalibrationError : public std::runtime_error {
public:
  /**
   * @brief Constructor for CalibrationError.
   * @param message The error message.
   */
  explicit CalibrationError(const std::string &message)
      : std::runtime_error(message) {}
};

/**
 * @brief Exception class for invalid parameter errors during calibration.
 */
class InvalidParameterError : public CalibrationError {
public:
  /**
   * @brief Constructor for InvalidParameterError.
   * @param message The error message.
   */
  explicit InvalidParameterError(const std::string &message)
      : CalibrationError(message) {}
};

/**
 * @brief Exception class for processing errors during calibration.
 */
class ProcessingError : public CalibrationError {
public:
  /**
   * @brief Constructor for ProcessingError.
   * @param message The error message.
   */
  explicit ProcessingError(const std::string &message)
      : CalibrationError(message) {}
};

/**
 * @brief Structure to hold the results of a flux calibration.
 */
struct FluxCalibrationResult {
  cv::Mat image;        ///< The calibrated image.
  double min_value;     ///< The minimum pixel value in the image.
  double range_value;   ///< The range of pixel values in the image.
  double flx2dn_factor; ///< The flux to digital number conversion factor.
};

/**
 * @brief Helper function to run a function asynchronously.
 *
 * This function uses `std::async` to run the given function in a separate
 * thread.
 *
 * @tparam Func The type of the function to run.
 * @tparam Args The types of the arguments to the function.
 * @param func The function to run.
 * @param args The arguments to pass to the function.
 * @return A `std::future` that will hold the result of the function.
 */
template <typename Func, typename... Args>
[[nodiscard]] auto run_async(Func &&func, Args &&...args) {
  return std::async(std::launch::async, std::forward<Func>(func),
                    std::forward<Args>(args)...);
}