// ImageProcessingUtils.hpp
#pragma once

#include "ImageProcessingConfig.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <string_view>
#include <spdlog/spdlog.h>

/**
 * @namespace IPUtils
 * @brief 图像处理工具命名空间
 */
namespace IPUtils {
  /**
   * @brief 检测系统是否支持AVX
   */
  bool hasAVXSupport() noexcept;

  /**
   * @brief 安全检查矩阵是否有效
   */
  bool isValidMatrix(const cv::Mat &mat) noexcept;

  /**
   * @brief 将值限制在一个范围内
   */
  template <typename T>
  constexpr T clamp(T value, T min_val, T max_val) noexcept {
    return std::min(std::max(value, min_val), max_val);
  }

  /**
   * @brief 转换边界模式到OpenCV边界类型
   */
  int getOpenCVBorderType(BorderMode mode) noexcept;

  /**
   * @struct ScopedTimer
   * @brief 用于测量操作持续时间的工具结构
   */
  struct ScopedTimer {
    std::string operation; ///< 正在计时的操作名称
    std::chrono::steady_clock::time_point start; ///< 操作的开始时间

    /**
     * @brief ScopedTimer的构造函数
     * @param op 操作的名称
     */
    explicit ScopedTimer(std::string_view op);

    /**
     * @brief ScopedTimer的析构函数
     */
    ~ScopedTimer();

    // 禁止复制和移动
    ScopedTimer(const ScopedTimer &) = delete;
    ScopedTimer &operator=(const ScopedTimer &) = delete;
  };

  /**
   * @brief 多通道处理器
   */
  std::expected<cv::Mat, ProcessError> processMultiChannel(
      const cv::Mat &input,
      const std::function<std::expected<cv::Mat, ProcessError>(const cv::Mat &)>
          &processor);
}