#include "sharpen/Sharpener.hpp"
#include "sharpen/Algorithm.hpp"

ImageSharpener::ImageSharpener(Method method) : m_method(method) {
  // 根据选定方法初始化默认参数
  switch (method) {
  case Method::Laplace:
    m_params = LaplaceParams{};
    break;
  case Method::UnsharpMask:
    m_params = UnsharpMaskParams{};
    break;
  case Method::HighBoost:
    m_params = HighBoostParams{};
    break;
  case Method::LaplacianOfGaussian:
    m_params = LaplacianOfGaussianParams{};
    break;
  case Method::BilateralSharp:
    m_params = BilateralSharpParams{};
    break;
  case Method::FrequencyDomain:
    m_params = FrequencyDomainParams{};
    break;
  case Method::AdaptiveUnsharp:
    m_params = AdaptiveUnsharpParams{};
    break;
  case Method::EdgePreserving:
    m_params = EdgePreservingParams{};
    break;
  case Method::CustomKernel:
    m_params = CustomKernelParams{};
    break;
  }

  createAlgorithm();
}

ImageSharpener::~ImageSharpener() = default;

void ImageSharpener::setMethod(Method method) {
  m_method = method;
  createAlgorithm();
}

cv::Mat ImageSharpener::sharpen(const cv::Mat &input) {
  if (!m_algorithm) {
    createAlgorithm();
  }

  return m_algorithm->process(input);
}

std::string ImageSharpener::getMethodName() const {
  static const std::unordered_map<Method, std::string> methodNames = {
      {Method::Laplace, "拉普拉斯锐化"},
      {Method::UnsharpMask, "USM锐化"},
      {Method::HighBoost, "高增益锐化"},
      {Method::LaplacianOfGaussian, "高斯拉普拉斯锐化"},
      {Method::BilateralSharp, "双边滤波锐化"},
      {Method::FrequencyDomain, "频域锐化"},
      {Method::AdaptiveUnsharp, "自适应USM锐化"},
      {Method::EdgePreserving, "边缘保留锐化"},
      {Method::CustomKernel, "自定义核锐化"}};

  auto it = methodNames.find(m_method);
  if (it != methodNames.end()) {
    return it->second;
  }
  return "未知方法";
}

void ImageSharpener::createAlgorithm() {
  switch (m_method) {
  case Method::Laplace:
    m_algorithm =
        std::make_unique<LaplaceSharpener>(std::get<LaplaceParams>(m_params));
    break;
  case Method::UnsharpMask:
    m_algorithm = std::make_unique<UnsharpMaskSharpener>(
        std::get<UnsharpMaskParams>(m_params));
    break;
  case Method::HighBoost:
    m_algorithm = std::make_unique<HighBoostSharpener>(
        std::get<HighBoostParams>(m_params));
    break;
  case Method::LaplacianOfGaussian:
    m_algorithm = std::make_unique<LaplacianOfGaussianSharpener>(
        std::get<LaplacianOfGaussianParams>(m_params));
    break;
  case Method::BilateralSharp:
    m_algorithm = std::make_unique<BilateralSharpener>(
        std::get<BilateralSharpParams>(m_params));
    break;
  case Method::FrequencyDomain:
    m_algorithm = std::make_unique<FrequencyDomainSharpener>(
        std::get<FrequencyDomainParams>(m_params));
    break;
  case Method::AdaptiveUnsharp:
    m_algorithm = std::make_unique<AdaptiveUnsharpSharpener>(
        std::get<AdaptiveUnsharpParams>(m_params));
    break;
  case Method::EdgePreserving:
    m_algorithm = std::make_unique<EdgePreservingSharpener>(
        std::get<EdgePreservingParams>(m_params));
    break;
  case Method::CustomKernel:
    m_algorithm = std::make_unique<CustomKernelSharpener>(
        std::get<CustomKernelParams>(m_params));
    break;
  }
}