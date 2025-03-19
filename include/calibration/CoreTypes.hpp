#pragma once

#include <concepts>
#include <functional>
#include <future>
#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>

// C++20 concepts for image types
template <typename T>
concept ImageContainer = requires(T a) {
  { a.cols } -> std::convertible_to<int>;
  { a.rows } -> std::convertible_to<int>;
  { a.channels() } -> std::convertible_to<int>;
  { a.type() } -> std::convertible_to<int>;
};

// Exception types
class CalibrationError : public std::runtime_error {
public:
  explicit CalibrationError(const std::string &message)
      : std::runtime_error(message) {}
};

class InvalidParameterError : public CalibrationError {
public:
  explicit InvalidParameterError(const std::string &message)
      : CalibrationError(message) {}
};

class ProcessingError : public CalibrationError {
public:
  explicit ProcessingError(const std::string &message)
      : CalibrationError(message) {}
};

// Enhanced result type with std::optional for error handling
struct FluxCalibrationResult {
  cv::Mat image;
  double min_value;
  double range_value;
  double flx2dn_factor;
};

// Helper for async operations
template <typename Func, typename... Args>
[[nodiscard]] auto run_async(Func &&func, Args &&...args) {
  return std::async(std::launch::async, std::forward<Func>(func),
                    std::forward<Args>(args)...);
}