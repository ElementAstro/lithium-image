#pragma once

#include "Parameter.hpp"
#include <functional>
#include <opencv2/opencv.hpp>


/**
 * @brief Class for wavelet-based denoising.
 */
class WaveletDenoiser {
public:
  /**
   * @brief Denoise an image using wavelet transform.
   * @param src Source image
   * @param dst Destination image
   * @param levels Number of decomposition levels
   * @param threshold Threshold for denoising
   */
  static void denoise(const cv::Mat &src, cv::Mat &dst, int levels,
                      float threshold);

  /**
   * @brief Denoise an image using specified parameters.
   * @param src Source image
   * @param dst Destination image
   * @param params Denoising parameters
   */
  static void denoise(const cv::Mat &src, cv::Mat &dst,
                      const DenoiseParameters &params);

private:
  static void wavelet_process_single_channel(const cv::Mat &src, cv::Mat &dst,
                                             int levels, float threshold);
  static cv::Mat decompose_one_level(const cv::Mat &src);
  static cv::Mat recompose_one_level(const cv::Mat &waveCoeffs,
                                     const cv::Size &originalSize);
  static void process_blocks(cv::Mat &img, int block_size,
                             const std::function<void(cv::Mat &)> &process_fn);
  static void wavelet_transform_simd(cv::Mat &data);
  static float compute_adaptive_threshold(const cv::Mat &coeffs,
                                          double noise_estimate);

  // 优化方法
  static void process_tile_simd(cv::Mat &tile);
  static void parallel_wavelet_transform(cv::Mat &data);
  static void optimize_memory_layout(cv::Mat &data);
  static void stream_process(const cv::Mat &src, cv::Mat &dst,
                             const std::function<void(cv::Mat &)> &process_fn);
};