// DeconvolutionProcessor.cpp
#include "DeconvolutionProcessor.hpp"
#include "ImageProcessingUtils.hpp"
#include "ThreadPool.hpp" 
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <algorithm>
#include <execution>

namespace {
    std::shared_ptr<spdlog::logger> deconv_logger =
        spdlog::basic_logger_mt("DeconvolutionProcessorLogger", "logs/deconvolution.log");
}

std::expected<cv::Mat, ProcessError>
DeconvolutionProcessor::process(const cv::Mat &input, const DeconvolutionConfig &cfg) {
    try {
        IPUtils::ScopedTimer timer("Deconvolution");
        deconv_logger->info("Starting deconvolution process");

        // 验证配置
        auto validate_result = validateConfig(input, cfg);
        if (!validate_result.has_value()) {
            return std::unexpected(validate_result.error());
        }

        // 如果是多通道图像且请求了按通道处理
        if (input.channels() > 1 && cfg.per_channel) {
            deconv_logger->debug("Processing deconvolution per channel");
            return IPUtils::processMultiChannel(input, [&](const cv::Mat &channel) {
                return deconvolveSingleChannel(channel, cfg);
            });
        }

        return deconvolveSingleChannel(input, cfg);
    } catch (const std::exception &e) {
        deconv_logger->error("Exception in deconvolve: {}", e.what());
        return std::unexpected(
            ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       fmt::format("Deconvolution failed: {}", e.what())});
    } catch (...) {
        deconv_logger->error("Unknown exception in deconvolve");
        return std::unexpected(ProcessError{ProcessError::Code::PROCESSING_FAILED,
                                        "Unknown exception in deconvolve"});
    }
}

std::expected<cv::Mat, ProcessError>
DeconvolutionProcessor::deconvolveSingleChannel(const cv::Mat &input,
                                           const DeconvolutionConfig &cfg) {
    try {
        deconv_logger->info("Starting single channel deconvolution");

        auto validate_result = validateConfig(input, cfg);
        if (!validate_result.has_value()) {
            return std::unexpected(validate_result.error());
        }

        auto psf_result = estimatePSF(input.size());
        if (!psf_result.has_value()) {
            return std::unexpected(psf_result.error());
        }

        cv::Mat psf = psf_result.value();
        cv::Mat output =
            cfg.use_memory_pool
                ? MemoryPool::allocate(input.rows, input.cols, input.type())
                : cv::Mat(input.rows, input.cols, input.type());

        std::expected<void, ProcessError> result;

        switch (cfg.method) {
        case DeconvMethod::RICHARDSON_LUCY:
            deconv_logger->debug("Using Richardson-Lucy deconvolution method");
            result = richardsonLucyDeconv(input, psf, output, cfg);
            break;
        case DeconvMethod::WIENER:
            deconv_logger->debug("Using Wiener deconvolution method");
            result = wienerDeconv(input, psf, output, cfg);
            break;
        case DeconvMethod::TIKHONOV:
            deconv_logger->debug("Using Tikhonov deconvolution method");
            result = tikhonovDeconv(input, psf, output, cfg);
            break;
        default:
            deconv_logger->error("Unsupported deconvolution method");
            return std::unexpected(ProcessError{ProcessError::Code::INVALID_CONFIG,
                                          "Unsupported deconvolution method"});
        }

        if (!result.has_value()) {
            return std::unexpected(result.error());
        }

        // 正规化输出
        cv::normalize(output, output, 0, 255, cv::NORM_MINMAX, CV_8U);
        deconv_logger->info("Single channel deconvolution completed");

        return output;
    } catch (const cv::Exception &e) {
        deconv_logger->error("OpenCV exception in deconvolveSingleChannel: {}", e.what());
        return std::unexpected(
            ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       fmt::format("OpenCV exception: {}", e.what())});
    } catch (const std::exception &e) {
        deconv_logger->error("Exception in deconvolveSingleChannel: {}", e.what());
        return std::unexpected(
            ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       fmt::format("Exception: {}", e.what())});
    } catch (...) {
        deconv_logger->error("Unknown exception in deconvolveSingleChannel");
        return std::unexpected(
            ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       "Unknown exception in deconvolveSingleChannel"});
    }
}

std::expected<cv::Mat, ProcessError> 
DeconvolutionProcessor::estimatePSF(cv::Size imgSize) {
    try {
        IPUtils::ScopedTimer timer("PSF Estimation");
        deconv_logger->debug("Estimating PSF for image size: width={}, height={}",
                          imgSize.width, imgSize.height);

        // 验证图像大小
        if (imgSize.width <= 0 || imgSize.height <= 0) {
            return std::unexpected(
                ProcessError{ProcessError::Code::INVALID_INPUT,
                           fmt::format("Invalid image dimensions: {}x{}",
                                     imgSize.width, imgSize.height)});
        }

        // 计算FFT效率的最佳大小
        const int optimal_size =
            cv::getOptimalDFTSize(std::max(imgSize.width, imgSize.height) * 2 - 1);

        if (optimal_size <= 0) {
            return std::unexpected(
                ProcessError{ProcessError::Code::PROCESSING_FAILED,
                           "Failed to determine optimal size for PSF"});
        }

        // 创建PSF作为高斯核
        cv::Mat psf = cv::Mat::zeros(optimal_size, optimal_size, CV_32F);

        // 根据图像大小计算核参数
        const int kernelSize = std::min(imgSize.width, imgSize.height) / 20;
        const double sigma = kernelSize / 6.0;
        const cv::Point center(optimal_size / 2, optimal_size / 2);

        // 使用并行计算生成PSF
        std::vector<int> rows(optimal_size);
        std::iota(rows.begin(), rows.end(), 0);

        std::for_each(std::execution::par, rows.begin(), rows.end(), [&](int i) {
            for (int j = 0; j < optimal_size; j++) {
                const double dx = j - center.x;
                const double dy = i - center.y;
                const double r2 = dx * dx + dy * dy;
                psf.at<float>(i, j) =
                    static_cast<float>(std::exp(-r2 / (2 * sigma * sigma)));
            }
        });

        // 正规化PSF以确保能量守恒
        cv::normalize(psf, psf, 1.0, 0.0, cv::NORM_L1);

        // 剪裁PSF以匹配输入图像大小
        if (imgSize.width > psf.cols || imgSize.height > psf.rows) {
            return std::unexpected(
                ProcessError{ProcessError::Code::PROCESSING_FAILED,
                           "PSF dimensions smaller than requested size"});
        }

        cv::Mat croppedPsf =
            psf(cv::Rect(0, 0, imgSize.width, imgSize.height)).clone();

        deconv_logger->info("PSF estimated successfully");
        return croppedPsf;
    } catch (const std::exception &e) {
        deconv_logger->error("Exception in estimatePSF: {}", e.what());
        return std::unexpected(
            ProcessError{ProcessError::Code::PROCESSING_FAILED,
                       fmt::format("PSF estimation failed: {}", e.what())});
    } catch (...) {
        deconv_logger->error("Unknown exception in estimatePSF");
        return std::unexpected(ProcessError{ProcessError::Code::PROCESSING_FAILED,
                                        "Unknown exception in PSF estimation"});
    }
}