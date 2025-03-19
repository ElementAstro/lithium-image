// Convolve.hpp
#pragma once

#include <chrono>
#include <concepts>
#include <expected>
#include <future>
#include <opencv2/opencv.hpp>
#include <variant>

#include "ConvolutionProcessor.hpp"
#include "DeconvolutionProcessor.hpp"
#include "ImageProcessingConfig.hpp"
#include "MemoryPool.hpp"
#include "ProcessError.hpp"


/**
 * @class Convolve
 * @brief 用于执行图像处理操作如卷积和反卷积的类。
 */
class Convolve {
public:
  /**
   * @brief 使用指定的配置处理图像
   * @param input 输入图像
   * @param config 处理配置（卷积或反卷积）
   * @return 处理后的图像或错误
   */
  static std::expected<cv::Mat, ProcessError>
  process(const cv::Mat &input,
          const std::variant<ConvolutionConfig, DeconvolutionConfig> &config);

  /**
   * @brief process的异步版本，返回协程
   */
  static std::future<std::expected<cv::Mat, ProcessError>> processAsync(
      const cv::Mat &input,
      const std::variant<ConvolutionConfig, DeconvolutionConfig> &config);

  // Convolve.hpp (继续)
  /**
   * @brief 清理Convolve类使用的资源
   */
  static void cleanup();

private:
  /**
   * @brief 根据输入验证配置类型
   */
  template <ConfigType T>
  static std::expected<void, ProcessError> validateConfig(const cv::Mat &input,
                                                          const T &config);
};