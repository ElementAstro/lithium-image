#pragma once

#include "Parameters.hpp"
#include <opencv2/opencv.hpp>

// Optimized versions with explicit optimization parameters
cv::Mat
instrument_response_correction_optimized(cv::InputArray &image,
                                         cv::InputArray &response_function,
                                         const OptimizationParams &params);

cv::Mat background_noise_correction_optimized(
    cv::InputArray &image, const OptimizationParams &params) noexcept;