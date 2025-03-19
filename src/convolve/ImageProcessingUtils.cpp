// ImageProcessingUtils.cpp
#include "ImageProcessingUtils.hpp"
#include <spdlog/sinks/basic_file_sink.h>
#include <mutex>
#include <thread>
#include <counting_semaphore>
#include <fmt/format.h>

namespace {
    std::shared_ptr<spdlog::logger> utils_logger =
        spdlog::basic_logger_mt("UtilsLogger", "logs/image_processing.log");
}

namespace IPUtils {

bool hasAVXSupport() noexcept {
#if defined(__AVX__) || defined(__AVX2__)
    return true;
#else
    return false;
#endif
}

bool isValidMatrix(const cv::Mat &mat) noexcept {
    return !mat.empty() && mat.data != nullptr;
}

int getOpenCVBorderType(BorderMode mode) noexcept {
    switch (mode) {
    case BorderMode::ZERO_PADDING:
        return cv::BORDER_CONSTANT;
    case BorderMode::MIRROR_REFLECT:
        return cv::BORDER_REFLECT101;
    case BorderMode::REPLICATE:
        return cv::BORDER_REPLICATE;
    case BorderMode::CIRCULAR:
        return cv::BORDER_WRAP;
    default:
        return cv::BORDER_DEFAULT;
    }
}

ScopedTimer::ScopedTimer(std::string_view op)
    : operation(op), start(std::chrono::steady_clock::now()) {
    utils_logger->debug("Starting operation: {}", operation);
}

ScopedTimer::~ScopedTimer() {
    auto duration = std::chrono::steady_clock::now() - start;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    utils_logger->debug("{} took {} ms", operation, ms.count());
}

std::expected<cv::Mat, ProcessError> processMultiChannel(
    const cv::Mat &input,
    const std::function<std::expected<cv::Mat, ProcessError>(const cv::Mat &)>
        &processor) {
    try {
        ScopedTimer timer("Multi-channel Processing");
        utils_logger->info("Processing multi-channel image with {} channels",
                           input.channels());

        if (!isValidMatrix(input)) {
            return std::unexpected(ProcessError{ProcessError::Code::INVALID_INPUT,
                                      "Input matrix is empty or invalid"});
        }

        std::vector<cv::Mat> channels;
        cv::split(input, channels);

        // 使用并行处理每个通道
        std::vector<std::expected<cv::Mat, ProcessError>> results(channels.size());
        std::counting_semaphore<> completion(0);
        std::mutex error_mutex;
        std::optional<ProcessError> first_error;

        for (size_t i = 0; i < channels.size(); i++) {
            std::thread([&, i]() {
                try {
                    results[i] = processor(channels[i]);
                    if (!results[i].has_value()) {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (!first_error) {
                            first_error = results[i].error();
                        }
                    }
                } catch (const std::exception &e) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!first_error) {
                        first_error = ProcessError{
                            ProcessError::Code::PROCESSING_FAILED,
                            fmt::format("Exception in channel processing: {}", e.what())};
                    }
                }
                completion.release();
            }).detach();
        }

        // 等待所有线程
        for (size_t i = 0; i < channels.size(); i++) {
            completion.acquire();
        }

        // 如果任何通道出错，返回第一个错误
        if (first_error) {
            return std::unexpected(*first_error);
        }

        // 检查所有结果
        for (size_t i = 0; i < results.size(); i++) {
            if (!results[i].has_value()) {
                return std::unexpected(results[i].error());
            }
            channels[i] = results[i].value();
        }

        cv::Mat result;
        cv::merge(channels, result);
        utils_logger->info("Channels merged back into a single image");
        return result;
    } catch (const cv::Exception &e) {
        utils_logger->error("OpenCV exception: {}", e.what());
        return std::unexpected(
            ProcessError{ProcessError::Code::PROCESSING_FAILED,
                         fmt::format("OpenCV exception: {}", e.what())});
    } catch (const std::exception &e) {
        utils_logger->error("Standard exception: {}", e.what());
        return std::unexpected(
            ProcessError{ProcessError::Code::PROCESSING_FAILED,
                         fmt::format("Standard exception: {}", e.what())});
    } catch (...) {
        utils_logger->error("Unknown exception in processMultiChannel");
        return std::unexpected(
            ProcessError{ProcessError::Code::PROCESSING_FAILED,
                         "Unknown exception in processMultiChannel"});
    }
}

} // namespace IPUtils