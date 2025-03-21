#pragma once

#include "Parameter.hpp"
#include <opencv2/opencv.hpp>


/**
 * @brief Class for image denoising.
 */
class ImageDenoiser {
public:
  /**
   * @brief Constructor for ImageDenoiser.
   */
  explicit ImageDenoiser();

  /**
   * @brief Denoise an image using specified parameters.
   * @param input Input image
   * @param params Denoising parameters
   * @return Denoised image
   */
  cv::Mat denoise(const cv::Mat &input, const DenoiseParameters &params);

  /**
   * @brief Analyze noise pattern in the image
   * @param input Input image
   * @return Noise analysis results
   */
  NoiseAnalysis analyzeNoise(const cv::Mat &input);

private:
  void process_median(const cv::Mat &src, cv::Mat &dst,
                      const DenoiseParameters &params);
  void process_gaussian(const cv::Mat &src, cv::Mat &dst,
                        const DenoiseParameters &params);
  void process_bilateral(const cv::Mat &src, cv::Mat &dst,
                         const DenoiseParameters &params);
  void process_nlm(const cv::Mat &src, cv::Mat &dst,
                   const DenoiseParameters &params);

  void validate_median(const DenoiseParameters &params);
  void validate_gaussian(const DenoiseParameters &params);
  void validate_bilateral(const DenoiseParameters &params);

  const char *method_to_string(DenoiseMethod method);

  // 频域滤波方法
  cv::Mat frequency_domain_filter(const cv::Mat &channel);
  cv::Mat create_bandstop_filter(const cv::Size &size, double sigma);
  void apply_filter(cv::Mat &complexImg, const cv::Mat &filter);
};