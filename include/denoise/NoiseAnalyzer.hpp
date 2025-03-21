#pragma once

#include "Parameter.hpp"
#include <opencv2/opencv.hpp>

/**
 * @brief Class for analyzing noise in images.
 */
class NoiseAnalyzer {
public:
  /**
   * @brief Analyze noise pattern in the image
   * @param input Input image
   * @return Noise analysis results
   */
  static NoiseAnalysis analyzeNoise(const cv::Mat &input);

  /**
   * @brief Determine best denoising method based on image analysis.
   * @param img Input image
   * @return Recommended denoising method
   */
  static DenoiseMethod recommendMethod(const cv::Mat &img);

  /**
   * @brief Update denoising parameters based on noise analysis.
   * @param params Parameters to update
   * @param analysis Noise analysis results
   */
  static void updateDenoiseParams(DenoiseParameters &params,
                                  const NoiseAnalysis &analysis);

private:
  static double detect_salt_pepper(const cv::Mat &gray);
  static double detect_gaussian(const cv::Mat &gray);
  static double estimateNoiseLevel(const cv::Mat &img);
  static double estimatePeriodicNoise(const cv::Mat &img);
  static cv::Mat detectNoiseDistribution(const cv::Mat &img);
  static std::vector<double> computeImageStatistics(const cv::Mat &img);
  static double calculateLocalVariance(const cv::Mat &img, int x, int y,
                                       int windowSize);
  static cv::Mat computeNoiseSpectrum(const cv::Mat &img);
};