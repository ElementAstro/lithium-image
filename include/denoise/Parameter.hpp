#pragma once

#include "Enums.hpp"
#include <map>
#include <opencv2/opencv.hpp>


/**
 * @brief Structure to hold denoising parameters.
 */
struct DenoiseParameters {
  DenoiseMethod method = DenoiseMethod::Auto; ///< Denoising method
  int threads = 4;                            ///< Number of parallel threads

  // Median filter parameters
  int median_kernel = 5; ///< Kernel size for median filter

  // Gaussian filter parameters
  cv::Size gaussian_kernel = {5, 5}; ///< Kernel size for Gaussian filter
  double sigma_x = 1.5;              ///< Sigma X for Gaussian filter
  double sigma_y = 1.5;              ///< Sigma Y for Gaussian filter

  // Bilateral filter parameters
  int bilateral_d = 9;       ///< Diameter of each pixel neighborhood
  double sigma_color = 75.0; ///< Filter sigma in the color space
  double sigma_space = 75.0; ///< Filter sigma in the coordinate space

  // NLM parameters
  float nlm_h = 3.0f;        ///< Parameter regulating filter strength
  int nlm_template_size = 7; ///< Size of the template patch
  int nlm_search_size = 21;  ///< Size of the window search area

  // Wavelet parameters
  int wavelet_level = 3;           ///< Number of decomposition levels
  float wavelet_threshold = 15.0f; ///< Threshold for wavelet denoising
  WaveletType wavelet_type = WaveletType::Haar; ///< Type of wavelet
  bool use_adaptive_threshold = true;           ///< Use adaptive thresholding
  double noise_estimate = 0.0;                  ///< Estimated noise level
  int block_size = 32;                          ///< Block size for processing

  // Optimization parameters
  bool use_simd = true;    ///< 使用SIMD优化
  bool use_opencl = false; ///< 使用OpenCL GPU加速
  int tile_size = 256;     ///< 分块大小
  bool use_stream = true;  ///< 使用流水线处理
};

/**
 * @brief Structure to hold noise analysis results.
 */
struct NoiseAnalysis {
  NoiseType type = NoiseType::Unknown;       ///< 检测到的噪声类型
  double intensity = 0.0;                    ///< 噪声强度 [0,1]
  double snr = 0.0;                          ///< 信噪比
  cv::Mat noiseMask;                         ///< 噪声分布掩码
  std::map<NoiseType, double> probabilities; ///< 各种噪声类型的概率
};