#ifndef SHARPENERALGORITHM_HPP
#define SHARPENERALGORITHM_HPP

#include "Parameters.hpp"
#include <opencv2/opencv.hpp>


// 所有锐化算法的抽象基类
class SharpenerAlgorithm {
public:
  virtual ~SharpenerAlgorithm() = default;
  virtual cv::Mat process(const cv::Mat &input) = 0;
  virtual void validateInput(const cv::Mat &input);
};

// 具体算法实现
class LaplaceSharpener : public SharpenerAlgorithm {
public:
  explicit LaplaceSharpener(const LaplaceParams &params);
  cv::Mat process(const cv::Mat &input) override;

private:
  LaplaceParams m_params;
};

class UnsharpMaskSharpener : public SharpenerAlgorithm {
public:
  explicit UnsharpMaskSharpener(const UnsharpMaskParams &params);
  cv::Mat process(const cv::Mat &input) override;

private:
  UnsharpMaskParams m_params;
  void validateParams();
};

class HighBoostSharpener : public SharpenerAlgorithm {
public:
  explicit HighBoostSharpener(const HighBoostParams &params);
  cv::Mat process(const cv::Mat &input) override;

private:
  HighBoostParams m_params;
  void validateParams();
};

class LaplacianOfGaussianSharpener : public SharpenerAlgorithm {
public:
  explicit LaplacianOfGaussianSharpener(
      const LaplacianOfGaussianParams &params);
  cv::Mat process(const cv::Mat &input) override;

private:
  LaplacianOfGaussianParams m_params;
  void validateParams();
  cv::Mat createLoGKernel(int size, double sigma);
};

class BilateralSharpener : public SharpenerAlgorithm {
public:
  explicit BilateralSharpener(const BilateralSharpParams &params);
  cv::Mat process(const cv::Mat &input) override;

private:
  BilateralSharpParams m_params;
};

class FrequencyDomainSharpener : public SharpenerAlgorithm {
public:
  explicit FrequencyDomainSharpener(const FrequencyDomainParams &params);
  cv::Mat process(const cv::Mat &input) override;

private:
  FrequencyDomainParams m_params;
  cv::Mat createFilterMask(int rows, int cols);
};

class AdaptiveUnsharpSharpener : public SharpenerAlgorithm {
public:
  explicit AdaptiveUnsharpSharpener(const AdaptiveUnsharpParams &params);
  cv::Mat process(const cv::Mat &input) override;

private:
  AdaptiveUnsharpParams m_params;
};

class EdgePreservingSharpener : public SharpenerAlgorithm {
public:
  explicit EdgePreservingSharpener(const EdgePreservingParams &params);
  cv::Mat process(const cv::Mat &input) override;

private:
  EdgePreservingParams m_params;
};

class CustomKernelSharpener : public SharpenerAlgorithm {
public:
  explicit CustomKernelSharpener(const CustomKernelParams &params);
  cv::Mat process(const cv::Mat &input) override;

private:
  CustomKernelParams m_params;
  void validateKernel();
};

#endif // SHARPENERALGORITHM_HPP