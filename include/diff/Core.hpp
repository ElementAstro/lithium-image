#pragma once

#include <concepts>
#include <stdexcept>
#include <string>
#include <vector>

#include "ImageType.hpp"
#include "Parameter.hpp"
#include "utils/Async.hpp"

/**
 * @brief 表示图像比较结果的结构体
 */
struct ComparisonResult {
  Image differenceImage;                    ///< 显示差异的图像
  double similarityPercent = 0.0;           ///< 图像相似度百分比
  std::vector<Rectangle> differenceRegions; ///< 发现差异的区域
  std::chrono::milliseconds duration;       ///< 比较过程的持续时间

  // 添加有效性检查
  [[nodiscard]] bool isValid() const noexcept {
    return !differenceImage.isNull() && similarityPercent >= 0.0 &&
           similarityPercent <= 100.0;
  }
};

/**
 * @brief 定义比较策略的概念
 */
template <typename T>
concept ComparisonStrategy = requires(T s, const Image &a, const Image &b,
                                      Promise<ComparisonResult> &p) {
  { s.compare(a, b, p) } -> std::same_as<ComparisonResult>;
  { s.name() } -> std::convertible_to<std::string>;
  {
    s.name()
  } noexcept -> std::convertible_to<std::string>; // 确保name()是noexcept
};

namespace ColorSpace {
/**
 * @brief 表示CIELAB色彩空间中颜色的结构体
 */
struct CIELAB {
  double L; ///< 亮度分量
  double a; ///< 绿-红分量
  double b; ///< 蓝-黄分量

  // 添加相等比较
  bool operator==(const CIELAB &other) const noexcept {
    constexpr double epsilon = 1e-6;
    return std::abs(L - other.L) < epsilon && std::abs(a - other.a) < epsilon &&
           std::abs(b - other.b) < epsilon;
  }

  // 添加不相等比较
  bool operator!=(const CIELAB &other) const noexcept {
    return !(*this == other);
  }
};

/**
 * @brief 将RGB颜色转换为CIELAB色彩空间
 *
 * @param rgb RGB颜色
 * @return 相应的CIELAB颜色
 */
[[nodiscard]] CIELAB RGB2LAB(const RGB &rgb) noexcept;
} // namespace ColorSpace

/**
 * @brief 将RGB值分解为其各个组成部分
 *
 * @param rgb RGB值
 * @return 包含红色、绿色和蓝色分量的元组
 */
[[nodiscard]] constexpr std::tuple<int, int, int>
unpackRGB(const RGB &rgb) noexcept {
  return {rgb.r, rgb.g, rgb.b};
}

/**
 * @brief 处理图像的行
 *
 * @param img 要处理的图像
 * @param height 图像高度
 * @param fn 应用于每一行的函数
 * @throws std::invalid_argument 如果图像无效
 */
void processRows(const Image &img, int height,
                 const std::function<void(int)> &fn);

/**
 * @brief 图像差异计算类
 */
class ImageDiff {
public:
  /**
   * @brief 使用指定策略比较两个图像
   *
   * @tparam Strategy 使用的比较策略
   * @param img1 第一个图像
   * @param img2 第二个图像
   * @param strategy 比较策略
   * @param promise 用于报告比较结果的promise
   * @return 比较的结果
   * @throws std::invalid_argument 如果图像无效
   * @throws std::runtime_error 如果比较失败
   */
  template <ComparisonStrategy Strategy>
  ComparisonResult compare(const Image &img1, const Image &img2,
                           Strategy &&strategy,
                           Promise<ComparisonResult> &promise) {
    if (!validateImages(img1, img2)) {
      promise.cancel();
      throw std::invalid_argument("Invalid images for comparison");
    }

    try {
      const auto start = std::chrono::high_resolution_clock::now();

      Image converted1 = img1.convertToFormat(Image::ARGB32);
      Image converted2 = img2.convertToFormat(Image::ARGB32);

      auto result = strategy.compare(converted1, converted2, promise);
      result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - start);

      postProcessResult(result);
      return result;
    } catch (const std::exception &e) {
      promise.cancel();
      throw std::runtime_error(std::string("Comparison failed: ") + e.what());
    }
  }

  // C++20协程任务生成器，用于异步比较
  template <ComparisonStrategy Strategy>
  [[nodiscard]] Task<ComparisonResult>
  compareAsync(const Image &img1, const Image &img2, Strategy &&strategy,
               Promise<ComparisonResult> &promise) {
    co_return compare(img1, img2, std::forward<Strategy>(strategy), promise);
  }

private:
  /**
   * @brief 验证输入图像
   *
   * @param img1 第一个图像
   * @param img2 第二个图像
   * @return 如果图像对比较有效则为true，否则为false
   */
  [[nodiscard]] bool validateImages(const Image &img1,
                                    const Image &img2) noexcept;

  /**
   * @brief 后处理比较结果
   *
   * @param result 要后处理的比较结果
   */
  void postProcessResult(ComparisonResult &result) const noexcept;
};