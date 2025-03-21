#ifndef IMAGESHARPENER_HPP
#define IMAGESHARPENER_HPP

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <type_traits>
#include <variant>


#include "Parameters.hpp"

// 前向声明
class SharpenerAlgorithm;

// ImageSharpener类提供多种图像锐化算法
class ImageSharpener {
public:
  enum class Method {
    Laplace,
    UnsharpMask,
    HighBoost,
    LaplacianOfGaussian,
    BilateralSharp,
    FrequencyDomain,
    AdaptiveUnsharp,
    EdgePreserving,
    CustomKernel
  };

  // 构造函数设置锐化方法
  explicit ImageSharpener(Method method = Method::UnsharpMask);

  // 析构函数
  ~ImageSharpener();

  // 设置锐化方法
  void setMethod(Method method);

  // 主处理函数，应用锐化算法
  cv::Mat sharpen(const cv::Mat &input);

  // 设置参数的模板函数(仅支持定义的参数类型)
  template <typename T> void setParameters(const T &params) {
    static_assert(std::is_same_v<T, LaplaceParams> ||
                      std::is_same_v<T, UnsharpMaskParams> ||
                      std::is_same_v<T, HighBoostParams> ||
                      std::is_same_v<T, LaplacianOfGaussianParams> ||
                      std::is_same_v<T, BilateralSharpParams> ||
                      std::is_same_v<T, FrequencyDomainParams> ||
                      std::is_same_v<T, AdaptiveUnsharpParams> ||
                      std::is_same_v<T, EdgePreservingParams> ||
                      std::is_same_v<T, CustomKernelParams>,
                  "不支持的参数类型");
    m_params = params;
  }

  // 获取当前方法的字符串表示
  std::string getMethodName() const;

private:
  Method m_method;
  std::unique_ptr<SharpenerAlgorithm> m_algorithm;
  std::variant<LaplaceParams, UnsharpMaskParams, HighBoostParams,
               LaplacianOfGaussianParams, BilateralSharpParams,
               FrequencyDomainParams, AdaptiveUnsharpParams,
               EdgePreservingParams, CustomKernelParams>
      m_params;

  // 根据方法创建适当的算法
  void createAlgorithm();
};

#endif // IMAGESHARPENER_HPP