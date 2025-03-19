// Convolve.cpp
#include "Convolve.hpp"
#include <spdlog/sinks/basic_file_sink.h>

namespace {
    std::shared_ptr<spdlog::logger> convolve_logger =
        spdlog::basic_logger_mt("ConvolveLogger", "logs/convolve.log");
}

std::expected<cv::Mat, ProcessError> Convolve::process(
    const cv::Mat &input,
    const std::variant<ConvolutionConfig, DeconvolutionConfig> &config) {
    if (!IPUtils::isValidMatrix(input)) {
        return std::unexpected(ProcessError{ProcessError::Code::INVALID_INPUT,
                                  "Input matrix is empty or invalid"});
    }

    return std::visit(
        [&](auto &&cfg) -> std::expected<cv::Mat, ProcessError> {
            using T = std::decay_t<decltype(cfg)>;

            if constexpr (std::is_same_v<T, ConvolutionConfig>) {
                return ConvolutionProcessor::process(input, cfg);
            } else if constexpr (std::is_same_v<T, DeconvolutionConfig>) {
                return DeconvolutionProcessor::process(input, cfg);
            } else {
                return std::unexpected(
                    ProcessError{ProcessError::Code::UNSUPPORTED_OPERATION,
                               "Unsupported configuration type"});
            }
        },
        config);
}

std::future<std::expected<cv::Mat, ProcessError>> Convolve::processAsync(
    const cv::Mat &input,
    const std::variant<ConvolutionConfig, DeconvolutionConfig> &config) {
    return std::async(std::launch::async, [input = input.clone(), config]() {
        return process(input, config);
    });
}

void Convolve::cleanup() {
    convolve_logger->info("Cleaning up Convolve resources");
    MemoryPool::clear();
}

template <ConfigType T>
std::expected<void, ProcessError>
Convolve::validateConfig(const cv::Mat &input, const T &config) {
    if constexpr (std::is_same_v<T, ConvolutionConfig>) {
        return ConvolutionProcessor::validateConfig(input, config);
    } else if constexpr (std::is_same_v<T, DeconvolutionConfig>) {
        return DeconvolutionProcessor::validateConfig(input, config);
    }
}