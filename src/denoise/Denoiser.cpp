#include "denoise/Denoiser.hpp"
#include "Logging.hpp"
#include "denoise/NoiseAnalyzer.hpp"
#include "denoise/WaveletDenoiser.hpp"
#include <stdexcept>

ImageDenoiser::ImageDenoiser() {
  // 初始化
}

cv::Mat ImageDenoiser::denoise(const cv::Mat &input,
                               const DenoiseParameters &params) {
  auto logger = Logger::getInstance();
  logger->info("Starting image denoise");
  if (input.empty()) {
    logger->error("Input image is empty");
    throw std::invalid_argument("Empty input image");
  }

  // 支持8位单通道或三通道图像
  if (input.depth() != CV_8U ||
      (input.channels() != 1 && input.channels() != 3)) {
    logger->error("Unsupported format: depth={} channels={}", input.depth(),
                  input.channels());
    throw std::invalid_argument("Unsupported image format");
  }

  try {
    cv::Mat processed;
    // 如果是Auto，则先根据噪声分析选择合适的去噪方法
    const auto method = (params.method == DenoiseMethod::Auto)
                            ? NoiseAnalyzer::recommendMethod(input)
                            : params.method;

    logger->debug("Using denoise method: {}", method_to_string(method));
    switch (method) {
    case DenoiseMethod::Median:
      validate_median(params);
      process_median(input, processed, params);
      break;
    case DenoiseMethod::Gaussian:
      validate_gaussian(params);
      process_gaussian(input, processed, params);
      break;
    case DenoiseMethod::Bilateral:
      validate_bilateral(params);
      process_bilateral(input, processed, params);
      break;
    case DenoiseMethod::NLM:
      process_nlm(input, processed, params);
      break;
    case DenoiseMethod::Wavelet:
      // Wavelet去噪处理
      WaveletDenoiser::denoise(input, processed, params);
      break;
    default:
      logger->error("Unsupported denoising method");
      throw std::runtime_error("Unsupported denoising method");
    }

    logger->info("Denoising completed using {}", method_to_string(method));
    return processed;
  } catch (const cv::Exception &e) {
    logger->error("OpenCV error: {}", e.what());
    throw;
  }
}

NoiseAnalysis ImageDenoiser::analyzeNoise(const cv::Mat &input) {
  return NoiseAnalyzer::analyzeNoise(input);
}

void ImageDenoiser::process_median(const cv::Mat &src, cv::Mat &dst,
                                   const DenoiseParameters &params) {
  auto logger = Logger::getInstance();
  logger->debug("Processing with median filter, kernel: {}",
                params.median_kernel);
  cv::medianBlur(src, dst, params.median_kernel);
}

void ImageDenoiser::process_gaussian(const cv::Mat &src, cv::Mat &dst,
                                     const DenoiseParameters &params) {
  auto logger = Logger::getInstance();
  logger->debug("Processing with Gaussian filter, sigma_x: {}, sigma_y: {}",
                params.sigma_x, params.sigma_y);
  cv::GaussianBlur(src, dst, params.gaussian_kernel, params.sigma_x,
                   params.sigma_y);
}

void ImageDenoiser::process_bilateral(const cv::Mat &src, cv::Mat &dst,
                                      const DenoiseParameters &params) {
  auto logger = Logger::getInstance();
  logger->debug("Processing bilateral filter");
  if (src.channels() == 3) {
    // 转成Lab，先对亮度通道进行双边滤波，再转换回来
    cv::cvtColor(src, dst, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> channels;
    cv::split(dst, channels);

    // 并行处理亮度通道
    cv::parallel_for_(
        cv::Range(0, 1),
        [&]([[maybe_unused]] const cv::Range &range) {
          cv::bilateralFilter(channels[0], channels[0], params.bilateral_d,
                              params.sigma_color, params.sigma_space);
        },
        params.threads);

    cv::merge(channels, dst);
    cv::cvtColor(dst, dst, cv::COLOR_Lab2BGR);
  } else {
    cv::bilateralFilter(src, dst, params.bilateral_d, params.sigma_color,
                        params.sigma_space);
  }
  logger->debug("Bilateral filter completed");
}

void ImageDenoiser::process_nlm(const cv::Mat &src, cv::Mat &dst,
                                const DenoiseParameters &params) {
  auto logger = Logger::getInstance();
  logger->debug("Processing NLM denoise");
  if (src.channels() == 3) {
    cv::fastNlMeansDenoisingColored(src, dst, params.nlm_h, params.nlm_h,
                                    params.nlm_template_size,
                                    params.nlm_search_size);
  } else {
    cv::fastNlMeansDenoising(src, dst, params.nlm_h, params.nlm_template_size,
                             params.nlm_search_size);
  }
  logger->debug("NLM denoise completed");
}

// 参数校验
void ImageDenoiser::validate_median(const DenoiseParameters &params) {
  auto logger = Logger::getInstance();
  if (params.median_kernel % 2 == 0 || params.median_kernel < 3) {
    logger->error("Median kernel size must be odd and ≥3");
    throw std::invalid_argument("Median kernel size must be odd and ≥3");
  }
  logger->debug("Median parameters validated");
}

void ImageDenoiser::validate_gaussian(const DenoiseParameters &params) {
  auto logger = Logger::getInstance();
  if (params.gaussian_kernel.width % 2 == 0 ||
      params.gaussian_kernel.height % 2 == 0) {
    logger->error("Gaussian kernel size must be odd");
    throw std::invalid_argument("Gaussian kernel size must be odd");
  }
  logger->debug("Gaussian parameters validated");
}

void ImageDenoiser::validate_bilateral(const DenoiseParameters &params) {
  auto logger = Logger::getInstance();
  if (params.bilateral_d <= 0) {
    logger->error("Bilateral d must be positive");
    throw std::invalid_argument("Bilateral d must be positive");
  }
  logger->debug("Bilateral parameters validated");
}

const char *ImageDenoiser::method_to_string(DenoiseMethod method) {
  switch (method) {
  case DenoiseMethod::Median:
    return "Median";
  case DenoiseMethod::Gaussian:
    return "Gaussian";
  case DenoiseMethod::Bilateral:
    return "Bilateral";
  case DenoiseMethod::NLM:
    return "Non-Local Means";
  case DenoiseMethod::Wavelet:
    return "Wavelet";
  default:
    return "Unknown";
  }
}

cv::Mat ImageDenoiser::frequency_domain_filter(const cv::Mat &channel) {
  cv::Mat padded;
  int m = cv::getOptimalDFTSize(channel.rows);
  int n = cv::getOptimalDFTSize(channel.cols);
  cv::copyMakeBorder(channel, padded, 0, m - channel.rows, 0, n - channel.cols,
                     cv::BORDER_CONSTANT, cv::Scalar::all(0));

  cv::Mat planes[] = {cv::Mat_<float>(padded),
                      cv::Mat::zeros(padded.size(), CV_32F)};
  cv::Mat complexImg;
  cv::merge(planes, 2, complexImg);
  cv::dft(complexImg, complexImg);

  cv::Mat filter = create_bandstop_filter(padded.size(), 30.0);
  apply_filter(complexImg, filter);

  cv::idft(complexImg, complexImg);
  cv::split(complexImg, planes);
  cv::normalize(planes[0], planes[0], 0, 255, cv::NORM_MINMAX, CV_8U);

  return planes[0](cv::Rect(0, 0, channel.cols, channel.rows));
}

cv::Mat ImageDenoiser::create_bandstop_filter(const cv::Size &size,
                                              double sigma) {
  cv::Mat filter = cv::Mat::zeros(size, CV_32F);
  cv::Point center(size.width / 2, size.height / 2);
  double D0 = sigma * 10;

  cv::parallel_for_(cv::Range(0, size.height), [&](const cv::Range &range) {
    for (int i = range.start; i < range.end; ++i) {
      float *p = filter.ptr<float>(i);
      for (int j = 0; j < size.width; ++j) {
        double d = cv::norm(cv::Point(j, i) - center);
        p[j] = 1 - std::exp(-(d * d) / (2 * D0 * D0));
      }
    }
  });
  return filter;
}

void ImageDenoiser::apply_filter(cv::Mat &complexImg, const cv::Mat &filter) {
  cv::Mat planes[2];
  cv::split(complexImg, planes);
  cv::multiply(planes[0], filter, planes[0]);
  cv::multiply(planes[1], filter, planes[1]);
  cv::merge(planes, 2, complexImg);
}
