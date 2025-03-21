#ifndef SHARPENERPARARMETERS_HPP
#define SHARPENERPARARMETERS_HPP

#include <opencv2/opencv.hpp>

// 拉普拉斯滤波参数
struct LaplaceParams {
  double scale = 1.0;
  double delta = 0.0;
  int ddepth = CV_16S;
  int kernelSize = 3; // 可以是1, 3, 5, 7
};

// USM锐化参数
struct UnsharpMaskParams {
  double sigma = 1.0;
  double amount = 1.0;
  int radius = 5; // 必须是奇数
};

// 高增益锐化参数
struct HighBoostParams {
  double sigma = 1.0;
  double boostFactor = 1.5; // 必须 >= 1.0
  int radius = 5;           // 必须是奇数
};

// 高斯拉普拉斯(LoG)滤波参数
struct LaplacianOfGaussianParams {
  double sigma = 1.5;
  int kernelSize = 9; // 必须是奇数
  double weight = 1.0;
};

// 双边滤波锐化参数
struct BilateralSharpParams {
  int diameter = 9;
  double sigmaColor = 75.0;
  double sigmaSpace = 75.0;
  double amount = 1.0;
};

// 频域锐化参数
struct FrequencyDomainParams {
  double highFreqGain = 1.5;
  double lowFreqGain = 1.0;
  double radius = 30.0;     // 截止频率
  bool butterworth = true;  // 使用巴特沃斯滤波而非高斯
  int butterworthOrder = 2; // 巴特沃斯滤波器阶数
};

// 自适应USM锐化参数
struct AdaptiveUnsharpParams {
  double globalSigma = 1.0;
  double localBlockSize = 16;  // 局部块大小
  double edgeThreshold = 30.0; // 边缘检测阈值
  double edgeSigma = 0.5;      // 边缘区域的sigma
  double flatSigma = 1.5;      // 平坦区域的sigma
  double amount = 1.2;
};

// 边缘保留锐化参数
struct EdgePreservingParams {
  int flags = cv::NORMCONV_FILTER; // 边缘保留滤波器类型
  double sigma_s = 60;             // 空间标准差
  double sigma_r = 0.4;            // 颜色标准差
  double amount = 1.0;
};

// 自定义核参数
struct CustomKernelParams {
  cv::Mat kernel = (cv::Mat_<float>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1,
                    -1); // 默认是简单锐化核
  double delta = 0.0;
};

#endif // SHARPENERPARARMETERS_HPP