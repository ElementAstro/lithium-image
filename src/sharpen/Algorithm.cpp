#include "sharpen/Algorithm.hpp"
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// 基类实现
void SharpenerAlgorithm::validateInput(const cv::Mat &input) {
  if (input.empty()) {
    throw std::invalid_argument("输入图像为空");
  }
  if (input.channels() > 4) {
    throw std::invalid_argument("不支持的通道数");
  }
}

// LaplaceSharpener实现
LaplaceSharpener::LaplaceSharpener(const LaplaceParams &params)
    : m_params(params) {}

cv::Mat LaplaceSharpener::process(const cv::Mat &input) {
  validateInput(input);

  cv::Mat laplace;
  cv::Laplacian(input, laplace, m_params.ddepth, m_params.kernelSize,
                m_params.scale, m_params.delta, cv::BORDER_DEFAULT);

  cv::Mat result;
  laplace.convertTo(result, input.type());
  cv::addWeighted(input, 1.0, result, 1.0, 0.0, result);

  return result;
}

// UnsharpMaskSharpener实现
UnsharpMaskSharpener::UnsharpMaskSharpener(const UnsharpMaskParams &params)
    : m_params(params) {
  validateParams();
}

void UnsharpMaskSharpener::validateParams() {
  if (m_params.radius % 2 == 0) {
    throw std::invalid_argument("USM锐化的半径必须是奇数");
  }
  if (m_params.sigma <= 0) {
    throw std::invalid_argument("USM锐化的sigma必须为正数");
  }
}

cv::Mat UnsharpMaskSharpener::process(const cv::Mat &input) {
  validateInput(input);

  cv::Mat blurred;
  cv::GaussianBlur(input, blurred, cv::Size(m_params.radius, m_params.radius),
                   m_params.sigma);

  cv::Mat mask;
  cv::subtract(input, blurred, mask);

  cv::Mat result;
  cv::addWeighted(input, 1.0 + m_params.amount, mask, m_params.amount, 0,
                  result);

  return result;
}

// HighBoostSharpener实现
HighBoostSharpener::HighBoostSharpener(const HighBoostParams &params)
    : m_params(params) {
  validateParams();
}

void HighBoostSharpener::validateParams() {
  if (m_params.radius % 2 == 0) {
    throw std::invalid_argument("高增益锐化的半径必须是奇数");
  }
  if (m_params.sigma <= 0) {
    throw std::invalid_argument("高增益锐化的sigma必须为正数");
  }
  if (m_params.boostFactor < 1.0) {
    throw std::invalid_argument("高增益因子必须至少为1.0");
  }
}

cv::Mat HighBoostSharpener::process(const cv::Mat &input) {
  validateInput(input);

  cv::Mat blurred;
  cv::GaussianBlur(input, blurred, cv::Size(m_params.radius, m_params.radius),
                   m_params.sigma);

  cv::Mat result;
  cv::addWeighted(input, m_params.boostFactor, blurred,
                  1.0 - m_params.boostFactor, 0.0, result);

  return result;
}

// LaplacianOfGaussian实现
LaplacianOfGaussianSharpener::LaplacianOfGaussianSharpener(
    const LaplacianOfGaussianParams &params)
    : m_params(params) {
  validateParams();
}

void LaplacianOfGaussianSharpener::validateParams() {
  if (m_params.kernelSize % 2 == 0) {
    throw std::invalid_argument("LoG锐化的核大小必须是奇数");
  }
  if (m_params.sigma <= 0) {
    throw std::invalid_argument("LoG锐化的sigma必须为正数");
  }
}

cv::Mat LaplacianOfGaussianSharpener::createLoGKernel(int size, double sigma) {
  cv::Mat kernel(size, size, CV_32F);
  int center = size / 2;
  double sigma2 = sigma * sigma;

  for (int y = 0; y < size; y++) {
    for (int x = 0; x < size; x++) {
      double dx = x - center;
      double dy = y - center;
      double r2 = dx * dx + dy * dy;

      // LoG方程: -1/(pi*sigma^4) * [1 - r^2/(2*sigma^2)] * e^(-r^2/(2*sigma^2))
      double value = -1.0 / (M_PI * sigma2 * sigma2) *
                     (1.0 - r2 / (2.0 * sigma2)) *
                     std::exp(-r2 / (2.0 * sigma2));

      kernel.at<float>(y, x) = static_cast<float>(value);
    }
  }

  // 归一化核以确保其和为零
  double sum = cv::sum(kernel)[0];
  kernel = kernel - (sum / (size * size));

  return kernel;
}

cv::Mat LaplacianOfGaussianSharpener::process(const cv::Mat &input) {
  validateInput(input);

  // 创建LoG核
  cv::Mat logKernel = createLoGKernel(m_params.kernelSize, m_params.sigma);

  // 应用LoG滤波
  cv::Mat filtered;
  cv::filter2D(input, filtered, -1, logKernel, cv::Point(-1, -1), 0,
               cv::BORDER_DEFAULT);

  // 将滤波结果添加到原始图像
  cv::Mat result;
  cv::addWeighted(input, 1.0, filtered, m_params.weight, 0, result);

  return result;
}

// BilateralSharpener实现
BilateralSharpener::BilateralSharpener(const BilateralSharpParams &params)
    : m_params(params) {}

cv::Mat BilateralSharpener::process(const cv::Mat &input) {
  validateInput(input);

  // 应用双边滤波平滑图像同时保留边缘
  cv::Mat smoothed;
  cv::bilateralFilter(input, smoothed, m_params.diameter, m_params.sigmaColor,
                      m_params.sigmaSpace);

  // 计算细节层(原始图像 - 平滑图像)
  cv::Mat detail;
  cv::subtract(input, smoothed, detail);

  // 增强细节并添加回平滑图像
  cv::Mat result;
  cv::addWeighted(input, 1.0, detail, m_params.amount, 0, result);

  return result;
}

// FrequencyDomainSharpener实现
FrequencyDomainSharpener::FrequencyDomainSharpener(
    const FrequencyDomainParams &params)
    : m_params(params) {}

cv::Mat FrequencyDomainSharpener::createFilterMask(int rows, int cols) {
  cv::Mat mask(rows, cols, CV_32F, cv::Scalar(0));
  int centerX = cols / 2;
  int centerY = rows / 2;

  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < cols; x++) {
      // 计算与中心的距离
      double dx = x - centerX;
      double dy = y - centerY;
      double distance = std::sqrt(dx * dx + dy * dy);

      // 计算滤波器值
      double value;
      if (m_params.butterworth) {
        // 巴特沃斯高通滤波器
        value = 1.0 / (1.0 + std::pow(m_params.radius / (distance + 1e-5),
                                      2 * m_params.butterworthOrder));
        // 调整增益
        value = m_params.lowFreqGain +
                (m_params.highFreqGain - m_params.lowFreqGain) * value;
      } else {
        // 高斯高通滤波器
        double radius2 = m_params.radius * m_params.radius;
        value = 1.0 - std::exp(-(distance * distance) / (2.0 * radius2));
        // 调整增益
        value = m_params.lowFreqGain +
                (m_params.highFreqGain - m_params.lowFreqGain) * value;
      }

      mask.at<float>(y, x) = static_cast<float>(value);
    }
  }

  return mask;
}

cv::Mat FrequencyDomainSharpener::process(const cv::Mat &input) {
  validateInput(input);

  // 分离输入通道
  std::vector<cv::Mat> channels;
  cv::split(input, channels);

  // 在频域中处理每个通道
  for (auto &channel : channels) {
    // 为DFT准备图像
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(channel.rows);
    int n = cv::getOptimalDFTSize(channel.cols);
    cv::copyMakeBorder(channel, padded, 0, m - channel.rows, 0,
                       n - channel.cols, cv::BORDER_CONSTANT,
                       cv::Scalar::all(0));

    // 转换为浮点型并变换到频域
    cv::Mat paddedF;
    padded.convertTo(paddedF, CV_32F);

    // 执行DFT
    cv::Mat dft;
    cv::dft(paddedF, dft, cv::DFT_COMPLEX_OUTPUT);

    // 将DFT移动到中心以使低频位于中心
    int cx = dft.cols / 2;
    int cy = dft.rows / 2;

    cv::Mat q0(dft, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(dft, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(dft, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(dft, cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    // 创建高通滤波器掩码
    cv::Mat filterMask = createFilterMask(dft.rows, dft.cols / 2);
    cv::Mat filterComplex;
    cv::Mat zeros = cv::Mat::zeros(filterMask.size(), CV_32F);
    std::vector<cv::Mat> filterPlanes = {filterMask, zeros};
    cv::merge(filterPlanes, filterComplex);

    // 在频域应用滤波器
    cv::mulSpectrums(dft, filterComplex, dft, 0);

    // 移回
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    // 反DFT
    cv::Mat idft;
    cv::dft(dft, idft, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    // 归一化并裁剪结果
    cv::normalize(idft, idft, 0, 1, cv::NORM_MINMAX);
    idft = idft(cv::Rect(0, 0, channel.cols, channel.rows));
    idft.convertTo(channel, channel.type(), 255);
  }

  // 合并通道
  cv::Mat result;
  cv::merge(channels, result);

  return result;
}

// AdaptiveUnsharpSharpener实现
AdaptiveUnsharpSharpener::AdaptiveUnsharpSharpener(
    const AdaptiveUnsharpParams &params)
    : m_params(params) {}

cv::Mat AdaptiveUnsharpSharpener::process(const cv::Mat &input) {
  validateInput(input);

  // 计算梯度幅度以识别边缘
  cv::Mat gray;
  if (input.channels() > 1) {
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = input.clone();
  }

  // 计算x和y梯度
  cv::Mat gradX, gradY;
  cv::Sobel(gray, gradX, CV_32F, 1, 0, 3);
  cv::Sobel(gray, gradY, CV_32F, 0, 1, 3);

  // 计算梯度幅度
  cv::Mat gradMagnitude;
  cv::magnitude(gradX, gradY, gradMagnitude);

  // 归一化到[0,1]
  cv::normalize(gradMagnitude, gradMagnitude, 0, 1, cv::NORM_MINMAX);

  // 创建边缘区域的掩码
  cv::Mat edgeMask;
  cv::threshold(gradMagnitude, edgeMask, m_params.edgeThreshold, 1.0,
                cv::THRESH_BINARY);

  // 使用不同的sigma值对边缘和非边缘区域进行模糊处理
  cv::Mat blurredEdge, blurredFlat;
  cv::GaussianBlur(input, blurredEdge, cv::Size(0, 0), m_params.edgeSigma);
  cv::GaussianBlur(input, blurredFlat, cv::Size(0, 0), m_params.flatSigma);

  // 使用边缘掩码组合不同的模糊图像
  cv::Mat blurred;
  blurredEdge.copyTo(blurred, edgeMask);
  blurredFlat.copyTo(blurred, 1.0 - edgeMask);

  // 计算掩码(细节层)
  cv::Mat mask;
  cv::subtract(input, blurred, mask);

  // 自适应地增强细节并添加回原始图像
  cv::Mat result;
  cv::addWeighted(input, 1.0 + m_params.amount, mask, m_params.amount, 0,
                  result);

  return result;
}

// EdgePreservingSharpener实现
EdgePreservingSharpener::EdgePreservingSharpener(
    const EdgePreservingParams &params)
    : m_params(params) {}

cv::Mat EdgePreservingSharpener::process(const cv::Mat &input) {
  validateInput(input);

  // 应用边缘保留滤波
  cv::Mat filtered;
  cv::edgePreservingFilter(input, filtered, m_params.flags, m_params.sigma_s,
                           m_params.sigma_r);

  // 计算细节层
  cv::Mat detail;
  cv::subtract(input, filtered, detail);

  // 增强细节并添加回滤波后的图像
  cv::Mat result;
  cv::addWeighted(input, 1.0, detail, m_params.amount, 0, result);

  return result;
}

// CustomKernelSharpener实现
CustomKernelSharpener::CustomKernelSharpener(const CustomKernelParams &params)
    : m_params(params) {
  validateKernel();
}

void CustomKernelSharpener::validateKernel() {
  if (m_params.kernel.empty()) {
    throw std::invalid_argument("核为空");
  }
  if (m_params.kernel.cols % 2 == 0 || m_params.kernel.rows % 2 == 0) {
    throw std::invalid_argument("核尺寸必须是奇数");
  }
}

cv::Mat CustomKernelSharpener::process(const cv::Mat &input) {
  validateInput(input);

  cv::Mat result;
  cv::filter2D(input, result, input.depth(), m_params.kernel, cv::Point(-1, -1),
               m_params.delta, cv::BORDER_DEFAULT);

  return result;
}