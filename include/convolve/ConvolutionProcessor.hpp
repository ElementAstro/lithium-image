#pragma once

#include "ImageProcessingConfig.hpp"
#include <opencv2/opencv.hpp>

#include "utils/Expected.hpp"


/**
 * @class ConvolutionProcessor
 * @brief 处理卷积相关操作的类
 */
class ConvolutionProcessor {
public:
  /**
   * @brief 使用卷积配置处理图像
   */
  static atom::type::expected<cv::Mat, ProcessError>
  process(const cv::Mat &input, const ConvolutionConfig &cfg);

  /**
   * @brief 对单通道图像执行卷积
   */
  static atom::type::expected<cv::Mat, ProcessError>
  convolveSingleChannel(const cv::Mat &input, const ConvolutionConfig &cfg);

private:
  /**
   * @brief 验证卷积配置
   */
  static atom::type::expected<void, ProcessError>
  validateConfig(const cv::Mat &input, const ConvolutionConfig &cfg);

  /**
   * @brief 准备卷积核
   */
  static atom::type::expected<cv::Mat, ProcessError>
  prepareKernel(const ConvolutionConfig &cfg);

  /**
   * @brief 使用AVX优化的卷积
   */
  static atom::type::expected<void, ProcessError>
  optimizedConvolveAVX(const cv::Mat &input, cv::Mat &output,
                       const cv::Mat &kernel, const ConvolutionConfig &cfg);

  /**
   * @brief 使用FFT的卷积
   */
  static atom::type::expected<void, ProcessError>
  fftConvolve(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);

  /**
   * @brief 块处理器 - 分块处理大型图像
   */
  static atom::type::expected<void, ProcessError>
  blockProcessing(const cv::Mat &input, cv::Mat &output,
                  const std::function<atom::type::expected<void, ProcessError>(
                      const cv::Mat &, cv::Mat &)> &processor,
                  int blockSize);
};