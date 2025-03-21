#pragma once

#include "CoreTypes.hpp"
#include "Parameters.hpp"
#include <opencv2/opencv.hpp>

/**
 * @brief Corrects an image for instrument response using a provided response
 * function.
 *
 * This function applies a correction to the input image based on the
 * instrument's response function, which maps pixel values to corrected values.
 *
 * @param image The input image to be corrected.
 * @param response_function The instrument response function.
 * @return The corrected image.
 */
cv::Mat instrument_response_correction(cv::InputArray &image,
                                       cv::InputArray &response_function);

/**
 * @brief Corrects an image for background noise.
 *
 * This function attempts to reduce background noise in the image.
 *
 * @param image The input image to be corrected.
 * @return The corrected image.
 */
cv::Mat background_noise_correction(cv::InputArray &image) noexcept;

/**
 * @brief Applies flat-field correction to an image.
 *
 * This function corrects for variations in pixel sensitivity using a flat-field
 * image.
 *
 * @param image The input image to be corrected.
 * @param flat_field The flat-field image.
 * @return The corrected image.
 */
cv::Mat apply_flat_field_correction(cv::InputArray &image,
                                    cv::InputArray &flat_field);

/**
 * @brief Applies dark frame subtraction to an image.
 *
 * This function subtracts a dark frame from the input image to correct for
 * thermal noise.
 *
 * @param image The input image to be corrected.
 * @param dark_frame The dark frame image.
 * @return The corrected image.
 */
cv::Mat apply_dark_frame_subtraction(cv::InputArray &image,
                                     cv::InputArray &dark_frame);

/**
 * @brief Computes the flux to digital number (FLX2DN) conversion factor.
 *
 * This function calculates the FLX2DN factor based on the provided calibration
 * parameters.
 *
 * @param params The calibration parameters.
 * @return The computed FLX2DN factor.
 */
[[nodiscard]] double compute_flx2dn(const CalibrationParams &params);

/**
 * @brief Performs flux calibration on an image.
 *
 * This function performs a full flux calibration on the input image, including
 * optional instrument response correction, flat-field correction, and dark
 * frame subtraction.
 *
 * @param image The input image to be calibrated.
 * @param params The calibration parameters.
 * @param response_function Optional instrument response function.
 * @param flat_field Optional flat-field image.
 * @param dark_frame Optional dark frame image.
 * @param enable_optimization Flag to enable optimization during calibration.
 * @return An optional FluxCalibrationResult struct containing the calibration
 * results, or std::nullopt if the calibration failed.
 */
[[nodiscard]] std::optional<FluxCalibrationResult>
flux_calibration_ex(const cv::Mat &image, const CalibrationParams &params,
                    const cv::Mat *response_function = nullptr,
                    const cv::Mat *flat_field = nullptr,
                    const cv::Mat *dark_frame = nullptr,
                    bool enable_optimization = false);