// ConvolutionProcessor.cpp
#include "ConvolutionProcessor.hpp"
#include "ImageProcessingUtils.hpp"
#include <algorithm>
#include <execution>
#include <fmt/format.h>
#include <numeric>
#include <spdlog/spdlog.h>


namespace {
std::shared_ptr<spdlog::logger> conv_logger = spdlog::basic_logger_mt(
    "ConvolutionProcessorLogger", "logs/convolution.log");
}

std::expected<cv::Mat, ProcessError>
ConvolutionProcessor::process(const cv::Mat &input,
                              const ConvolutionConfig &cfg) {
  try {
    conv_logger->info("Starting convolution process");

    // 验证输入
    auto validate_result = validateConfig(input, cfg);
    if (!validate_result.has_value()) {
      return std::unexpected(validate_result.error());
    }

    if (input.channels() > 1 && cfg.per_channel) {
      conv_logger->debug("Processing convolution per channel");
      return IPUtils::processMultiChannel(input, [&](const cv::Mat &channel) {
        return convolveSingleChannel(channel, cfg);
      });
    }

    conv_logger->debug("Processing convolution on entire image");
    return convolveSingleChannel(input, cfg);
  } catch (const std::exception &e) {
    conv_logger->error("Exception in convolve: {}", e.what());
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     fmt::format("Exception in convolve: {}", e.what())});
  } catch (...) {
    conv_logger->error("Unknown exception in convolve");
    return std::unexpected(ProcessError{ProcessError::Code::PROCESSING_FAILED,
                                        "Unknown exception in convolve"});
  }
}

std::expected<cv::Mat, ProcessError>
ConvolutionProcessor::convolveSingleChannel(const cv::Mat &input,
                                            const ConvolutionConfig &cfg) {
  try {
    IPUtils::ScopedTimer timer("Optimized Convolution");

    auto kernelResult = prepareKernel(cfg);
    if (!kernelResult.has_value()) {
      return std::unexpected(kernelResult.error());
    }

    cv::Mat kernel = kernelResult.value();
    cv::Mat output =
        cfg.use_memory_pool
            ? MemoryPool::allocate(input.rows, input.cols, input.type())
            : cv::Mat(input.rows, input.cols, input.type());

    // 根据核大小和配置选择适当的算法
    std::expected<void, ProcessError> result;
    if (cfg.use_fft && kernel.rows > 15) {
      conv_logger->debug("Using FFT-based convolution for large kernel");
      result = fftConvolve(input, output, kernel);
    } else if (cfg.use_avx && IPUtils::hasAVXSupport()) {
      conv_logger->debug("Using AVX-optimized convolution");
      result = optimizedConvolveAVX(input, output, kernel, cfg);
    } else {
      conv_logger->debug("Using block-based convolution");
      result = blockProcessing(
          input, output,
          [&](const cv::Mat &inBlock,
              cv::Mat &outBlock) -> std::expected<void, ProcessError> {
            // ConvolutionProcessor.cpp (继续)
            try {
              cv::filter2D(inBlock, outBlock, -1, kernel);
              return {};
            } catch (const std::exception &e) {
              return std::unexpected(
                  ProcessError{ProcessError::Code::PROCESSING_FAILED,
                               fmt::format("Filter2D failed: {}", e.what())});
            }
          },
          cfg.block_size);
    }

    if (!result.has_value()) {
      return std::unexpected(result.error());
    }

    return output;
  } catch (const cv::Exception &e) {
    conv_logger->error("OpenCV exception in convolveSingleChannel: {}",
                       e.what());
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     fmt::format("OpenCV exception: {}", e.what())});
  } catch (const std::exception &e) {
    conv_logger->error("Exception in convolveSingleChannel: {}", e.what());
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     fmt::format("Exception: {}", e.what())});
  } catch (...) {
    conv_logger->error("Unknown exception in convolveSingleChannel");
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     "Unknown exception in convolveSingleChannel"});
  }
}

std::expected<void, ProcessError>
ConvolutionProcessor::validateConfig(const cv::Mat &input,
                                     const ConvolutionConfig &cfg) {
  try {
    if (!IPUtils::isValidMatrix(input)) {
      conv_logger->error("Input image is empty or invalid");
      return std::unexpected(ProcessError{ProcessError::Code::INVALID_INPUT,
                                          "Empty or invalid input image"});
    }

    if (!cfg.per_channel && input.channels() > 1) {
      conv_logger->warn("Multi-channel image will be processed as a whole");
    }

    if (cfg.kernel_size % 2 == 0 || cfg.kernel_size < 3) {
      conv_logger->error("Invalid kernel size: {}", cfg.kernel_size);
      return std::unexpected(
          ProcessError{ProcessError::Code::INVALID_CONFIG,
                       fmt::format("Kernel size must be odd and >=3, got {}",
                                   cfg.kernel_size)});
    }

    if (cfg.kernel.size() !=
        static_cast<size_t>(cfg.kernel_size * cfg.kernel_size)) {
      conv_logger->error("Kernel size mismatch: expected {}, got {}",
                         cfg.kernel_size * cfg.kernel_size, cfg.kernel.size());
      return std::unexpected(ProcessError{
          ProcessError::Code::INVALID_CONFIG,
          fmt::format("Kernel dimensions mismatch: expected {}, got {}",
                      cfg.kernel_size * cfg.kernel_size, cfg.kernel.size())});
    }

    // 验证线程数
    if (cfg.thread_count < 0) {
      conv_logger->warn("Negative thread count specified ({}), using auto",
                        cfg.thread_count);
    }

    // 验证块大小
    if (cfg.block_size <= 0) {
      conv_logger->warn("Invalid block size: {}, using default",
                        cfg.block_size);
    }

    conv_logger->debug("Convolution config validated");
    return {};
  } catch (const std::exception &e) {
    conv_logger->error("Exception in validateConvolutionConfig: {}", e.what());
    return std::unexpected(ProcessError{
        ProcessError::Code::INVALID_CONFIG,
        fmt::format("Configuration validation failed: {}", e.what())});
  } catch (...) {
    conv_logger->error("Unknown exception in validateConvolutionConfig");
    return std::unexpected(
        ProcessError{ProcessError::Code::INVALID_CONFIG,
                     "Unknown exception in configuration validation"});
  }
}

std::expected<cv::Mat, ProcessError>
ConvolutionProcessor::prepareKernel(const ConvolutionConfig &cfg) {
  try {
    conv_logger->debug("Preparing convolution kernel");

    // 验证核大小
    if (cfg.kernel_size % 2 == 0 || cfg.kernel_size < 3) {
      return std::unexpected(
          ProcessError{ProcessError::Code::INVALID_CONFIG,
                       fmt::format("Kernel size must be odd and >=3, got {}",
                                   cfg.kernel_size)});
    }

    // 验证核数据大小
    if (cfg.kernel.size() !=
        static_cast<size_t>(cfg.kernel_size * cfg.kernel_size)) {
      return std::unexpected(ProcessError{
          ProcessError::Code::INVALID_CONFIG,
          fmt::format("Kernel dimensions mismatch: expected {}, got {}",
                      cfg.kernel_size * cfg.kernel_size, cfg.kernel.size())});
    }

    // 检查核中的NaN值
    for (const auto &val : cfg.kernel) {
      if (std::isnan(val) || std::isinf(val)) {
        return std::unexpected(
            ProcessError{ProcessError::Code::INVALID_CONFIG,
                         "Kernel contains NaN or Inf values"});
      }
    }

    cv::Mat kernel(cfg.kernel_size, cfg.kernel_size, CV_32F);
    std::copy(cfg.kernel.begin(), cfg.kernel.end(), kernel.ptr<float>());

    if (cfg.normalize_kernel) {
      double sum = cv::sum(kernel)[0];
      if (std::abs(sum) < 1e-10) {
        conv_logger->warn(
            "Kernel sum is nearly zero ({}), normalization skipped", sum);
      } else {
        kernel /= sum;
        conv_logger->debug("Kernel normalized with sum {}", sum);
      }
    }

    conv_logger->debug("Kernel prepared successfully");
    return kernel;
  } catch (const std::exception &e) {
    conv_logger->error("Exception in prepareKernel: {}", e.what());
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     fmt::format("Failed to prepare kernel: {}", e.what())});
  } catch (...) {
    conv_logger->error("Unknown exception in prepareKernel");
    return std::unexpected(
        ProcessError{ProcessError::Code::PROCESSING_FAILED,
                     "Unknown exception in kernel preparation"});
  }
}
