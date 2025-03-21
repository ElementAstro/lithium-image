#include "denoise/NoiseAnalyzer.hpp"
#include "Logging.hpp"

NoiseAnalysis NoiseAnalyzer::analyzeNoise(const cv::Mat &input) {
  auto logger = Logger::getInstance();
  logger->info("Starting comprehensive noise analysis");
  NoiseAnalysis analysis;

  cv::Mat gray;
  if (input.channels() > 1) {
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = input.clone();
  }

  // 计算图像统计特征
  auto stats = computeImageStatistics(gray);
  double variance = stats[1];
  // 保留 skewness 变量，但在适当的地方使用它
  double skewness = stats[2];
  double kurtosis = stats[3];

  // 噪声水平估计
  analysis.intensity = estimateNoiseLevel(gray);

  // 计算信噪比
  cv::Mat smoothed;
  cv::GaussianBlur(gray, smoothed, cv::Size(5, 5), 1.5);
  cv::Mat noise;
  cv::absdiff(gray, smoothed, noise);
  cv::Scalar mean, stddev;
  cv::meanStdDev(gray, mean, stddev);
  analysis.snr = mean[0] / stddev[0];

  // 生成噪声分布掩码
  analysis.noiseMask = detectNoiseDistribution(gray);

  // 检测周期性噪声
  double periodicNoiseStrength = estimatePeriodicNoise(gray);

  // 基于特征进行噪声类型概率计算
  std::map<NoiseType, double> &probs = analysis.probabilities;

  // 高斯噪声特征：kurtosis接近3
  probs[NoiseType::Gaussian] = std::exp(-std::abs(kurtosis - 3.0) / 2.0);

  // 椒盐噪声特征：极值点比例
  double saltPepperRatio = detect_salt_pepper(gray);
  probs[NoiseType::SaltAndPepper] = saltPepperRatio;

  // 散斑噪声特征：方差与均值的关系
  double speckleProb = variance / (mean[0] * mean[0]);
  probs[NoiseType::Speckle] = std::min(speckleProb, 1.0);

  // 周期性噪声
  probs[NoiseType::Periodic] = periodicNoiseStrength;

  // 确定主要噪声类型
  auto maxProb = std::max_element(
      probs.begin(), probs.end(),
      [](const auto &a, const auto &b) { return a.second < b.second; });

  if (maxProb->second > 0.5) {
    analysis.type = maxProb->first;
  } else {
    analysis.type = NoiseType::Mixed;
  }

  logger->info(
      "Noise analysis completed. Type: {}, Intensity: {:.2f}, SNR: {:.2f}",
      static_cast<int>(analysis.type), analysis.intensity, analysis.snr);

  return analysis;
}

DenoiseMethod NoiseAnalyzer::recommendMethod(const cv::Mat &img) {
  auto logger = Logger::getInstance();
  logger->info("Starting noise analysis for method recommendation");
  cv::Mat gray;
  if (img.channels() > 1) {
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = img.clone();
  }

  // 分析噪声类型
  double salt_pepper_ratio = detect_salt_pepper(gray);
  double gaussian_likelihood = detect_gaussian(gray);

  logger->debug("Salt and pepper ratio: {}, Gaussian likelihood: {}",
                salt_pepper_ratio, gaussian_likelihood);
  if (salt_pepper_ratio > 0.1) {
    logger->info("Detected salt and pepper noise, using Median filter");
    return DenoiseMethod::Median;
  } else if (gaussian_likelihood > 0.7) {
    logger->info("Detected Gaussian noise, using Gaussian filter");
    return DenoiseMethod::Gaussian;
  } else {
    // 检测周期噪声可能性
    double periodic_noise = estimatePeriodicNoise(gray);
    if (periodic_noise > 0.5) {
      logger->info("Detected periodic noise, using Wavelet filter");
      return DenoiseMethod::Wavelet;
    }

    // 检测边缘信息
    // 声明未定义的变量
    cv::Mat img1 = gray.clone();
    cv::Mat img2;
    cv::GaussianBlur(img1, img2, cv::Size(5, 5), 1.5);

    // 声明SSIM计算所需变量
    cv::Mat mu1, mu2;
    cv::GaussianBlur(img1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    // 创建平方图像
    cv::Mat I1_2 = img1.mul(img1);
    cv::Mat I2_2 = img2.mul(img2);
    cv::Mat I1_I2 = img1.mul(img2);

    // 定义常量C1, C2
    const double C1 = 6.5025;  // (0.01 * 255)^2
    const double C2 = 58.5225; // (0.03 * 255)^2

    cv::Mat sigma1_2, sigma2_2, sigma12;
    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);

    cv::Mat ssim;
    divide(t3, t1, ssim);
    double ssim_val = mean(ssim)[0];

    logger->debug("SSIM result: {:.4f}", ssim_val);

    // 根据SSIM值返回适当的降噪方法
    if (ssim_val > 0.8) {
      return DenoiseMethod::NonLocalMeans;
    } else {
      return DenoiseMethod::BilateralFilter;
    }
  }
}

// 将QualityEvaluator类的方法移出NoiseAnalyzer类
double QualityEvaluator::noiseReductionRatio(const cv::Mat &orig,
                                             const cv::Mat &processed) {
  auto logger = Logger::getInstance();
  logger->debug("Calculating noise reduction ratio");

  cv::Mat gray_orig, gray_proc;
  if (orig.channels() > 1) {
    cv::cvtColor(orig, gray_orig, cv::COLOR_BGR2GRAY);
    cv::cvtColor(processed, gray_proc, cv::COLOR_BGR2GRAY);
  } else {
    gray_orig = orig;
    gray_proc = processed;
  }

  cv::Scalar mean_orig, stddev_orig, mean_proc, stddev_proc;
  cv::meanStdDev(gray_orig, mean_orig, stddev_orig);
  cv::meanStdDev(gray_proc, mean_proc, stddev_proc);

  double noise_reduction = (1.0 - stddev_proc[0] / stddev_orig[0]) * 100.0;
  double result = std::max(0.0, noise_reduction);

  logger->debug("Noise reduction ratio: {:.2f}%", result);
  return result;
}

void QualityEvaluator::generateQualityReport(const cv::Mat &orig,
                                             const cv::Mat &processed,
                                             const std::string &output_path) {
  auto logger = Logger::getInstance();
  logger->info("Generating quality report at: {}", output_path);

  std::ofstream report(output_path);
  if (!report.is_open()) {
    logger->error("Failed to create quality report at: {}", output_path);
    return;
  }

  double psnr = calculatePSNR(orig, processed);
  double ssim = calculateSSIM(orig, processed);
  double noise_red = noiseReductionRatio(orig, processed);

  report << "=== Image Quality Report ===\n"
         << "PSNR: " << psnr << " dB\n"
         << "SSIM: " << ssim << "\n"
         << "Noise Reduction: " << noise_red << "%\n";

  logger->info(
      "Quality report generated: PSNR={:.2f}dB, SSIM={:.4f}, NR={:.1f}%", psnr,
      ssim, noise_red);

  report.close();
}