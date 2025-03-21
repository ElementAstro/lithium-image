#pragma once

/**
 * @brief Enumeration of denoising methods.
 */
enum class DenoiseMethod {
  Auto,      ///< Automatically select based on noise analysis
  Median,    ///< Median filter for salt-and-pepper noise
  Gaussian,  ///< Gaussian filter for Gaussian noise
  Bilateral, ///< Bilateral filter to preserve edges
  NLM,       ///< Non-Local Means for uniform noise
  Wavelet    ///< Wavelet transform denoising
};

/**
 * @brief Enumeration of wavelet types.
 */
enum class WaveletType { Haar, Daubechies4, Coiflet, Biorthogonal };

/**
 * @brief Enumeration of noise types.
 */
enum class NoiseType {
  Unknown,       ///< 未知噪声类型
  Gaussian,      ///< 高斯噪声
  SaltAndPepper, ///< 椒盐噪声
  Speckle,       ///< 散斑噪声
  Poisson,       ///< 泊松噪声
  Periodic,      ///< 周期性噪声
  Mixed          ///< 混合噪声
};