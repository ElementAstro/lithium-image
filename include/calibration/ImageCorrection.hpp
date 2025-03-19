#pragma once

#include "CoreTypes.hpp"
#include "Parameters.hpp"
#include <opencv2/opencv.hpp>

// Function declarations for basic image correction
cv::Mat instrument_response_correction(cv::InputArray &image,
                                       cv::InputArray &response_function);

cv::Mat background_noise_correction(cv::InputArray &image) noexcept;

cv::Mat apply_flat_field_correction(cv::InputArray &image,
                                    cv::InputArray &flat_field);

cv::Mat apply_dark_frame_subtraction(cv::InputArray &image,
                                     cv::InputArray &dark_frame);

[[nodiscard]] double compute_flx2dn(const CalibrationParams &params);

// Using std::optional for potential failure
[[nodiscard]] std::optional<FluxCalibrationResult>
flux_calibration_ex(const cv::Mat &image, const CalibrationParams &params,
                    const cv::Mat *response_function = nullptr,
                    const cv::Mat *flat_field = nullptr,
                    const cv::Mat *dark_frame = nullptr,
                    bool enable_optimization = false);