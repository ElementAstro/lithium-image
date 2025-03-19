// DeconvolutionProcessor.hpp
#pragma once

#include "ImageProcessingConfig.hpp"
#include "ProcessError.hpp"
#include "MemoryPool.hpp"
#include <expected>
#include <opencv2/opencv.hpp>

/**
 * @class DeconvolutionProcessor
 * @brief 处理反卷积相关操作的类
 */
class DeconvolutionProcessor {
public:
  /**
   * @brief 使用反卷积配置处理图像
   */
  static std::expected<cv::Mat, ProcessError> 
  process(const cv::Mat &input, const DeconvolutionConfig &cfg);

  /**
   * @brief 对单通道图像执行反卷积
   */
  static std::expected<cv::Mat, ProcessError>
  deconvolveSingleChannel(const cv::Mat &input, const DeconvolutionConfig &cfg);

private:
  /**
   * @brief 验证反卷积配置
   */
  static std::expected<void, ProcessError>
  validateConfig(const cv::Mat &input, const DeconvolutionConfig &cfg);

  /**
   * @brief 估算点扩散函数(PSF)
   */
  static std::expected<cv::Mat, ProcessError> 
  estimatePSF(cv::Size imgSize);

  /**
   * @brief 执行Richardson-Lucy反卷积
   */
  static std::expected<void, ProcessError>
  richardsonLucyDeconv(const cv::Mat &input, const cv::Mat &psf,
                     cv::Mat &output, const DeconvolutionConfig &cfg);

  /**
   * @brief 执行Wiener反卷积
   */
  static std::expected<void, ProcessError>
  wienerDeconv(const cv::Mat &input, const cv::Mat &psf, cv::Mat &output,
             const DeconvolutionConfig &cfg);

  /**
   * @brief 执行Tikhonov正则化反卷积
   */
  static std::expected<void, ProcessError>
  tikhonovDeconv(const cv::Mat &input, const cv::Mat &psf, cv::Mat &output,
               const DeconvolutionConfig &cfg);
};