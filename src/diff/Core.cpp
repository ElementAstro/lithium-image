#include "diff_core.h"
#include "logger.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>
#include <vector>

namespace {
// 初始化日志记录器
std::shared_ptr<Logger> diffLogger = getLogger("DiffLogger");

struct LoggerInitializer {
  LoggerInitializer() noexcept {
    diffLogger->setLevel(LogLevel::Debug);
    diffLogger->setOutputFile("logs/diff.log");
  }
} loggerInit;
} // namespace

namespace ColorSpace {
CIELAB RGB2LAB(const RGB &rgb) noexcept {
  const auto [r, g, b] = unpackRGB(rgb);
  return {0.2126 * r + 0.7152 * g + 0.0722 * b, static_cast<double>(r - g),
          static_cast<double>(g - b)};
}
} // namespace ColorSpace

bool ImageDiff::validateImages(const Image &img1, const Image &img2) noexcept {
  if (img1.isNull() || img2.isNull()) {
    diffLogger->error("One or both images are null");
    return false;
  }

  if (img1.width() != img2.width() || img1.height() != img2.height()) {
    diffLogger->error("Image sizes don't match: {}x{} vs {}x{}", img1.width(),
                      img1.height(), img2.width(), img2.height());
    return false;
  }

  // 增强对像素格式的验证
  const auto validFormat = [](const Image &img) {
    return img.format() == Image::ARGB32 || img.format() == Image::RGB32;
  };

  if (!validFormat(img1)) {
    diffLogger->warning("First image format is not optimal: {}",
                        static_cast<int>(img1.format()));
  }

  if (!validFormat(img2)) {
    diffLogger->warning("Second image format is not optimal: {}",
                        static_cast<int>(img2.format()));
  }

  return true;
}

void ImageDiff::postProcessResult(ComparisonResult &result) const noexcept {
  if (result.differenceImage.isNull()) {
    diffLogger->warning("Difference image is null, skipping post-processing");
    return;
  }

  // 规范化差异图像
  try {
    unsigned char *bits = result.differenceImage.bits();
    const size_t size = result.differenceImage.sizeInBytes();

    // 使用并行算法查找最小/最大值
    std::vector<unsigned char> pixelData(bits, bits + size);
    const auto [minIter, maxIter] =
        std::minmax_element(pixelData.begin(), pixelData.end());

    if (minIter == pixelData.end() || maxIter == pixelData.end()) {
      diffLogger->error("Failed to find min/max pixel values");
      return;
    }

    const unsigned char minVal = *minIter;
    const unsigned char maxVal = *maxIter;

    // 避免除以零
    if (maxVal == minVal) {
      diffLogger->warning("No variance in difference image");
      return;
    }

    // 应用对比度拉伸
    const float scale = 255.0f / (maxVal - minVal);

#pragma omp parallel for simd
    for (size_t i = 0; i < size; ++i) {
      bits[i] = static_cast<unsigned char>((bits[i] - minVal) * scale);
    }

    // 记录统计信息
    diffLogger->debug("Post-process stats - Min: {}, Max: {}, Scale: {:.4f}",
                      static_cast<int>(minVal), static_cast<int>(maxVal),
                      scale);
  } catch (const std::exception &e) {
    diffLogger->error("Error in postProcessResult: {}", e.what());
  }
}

void processRows(const Image &img, int height,
                 const std::function<void(int)> &fn) {
  if (img.isNull() || height <= 0 || !fn) {
    throw std::invalid_argument("Invalid parameters in processRows");
  }

  try {
    auto threadCount = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    threads.reserve(threadCount);

    for (unsigned t = 0; t < threadCount; ++t) {
      threads.emplace_back([t, threadCount, height, &fn]() {
        for (int row = t; row < height; row += threadCount) {
          fn(row);
        }
      });
    }

    for (auto &thread : threads) {
      thread.join();
    }
  } catch (const std::exception &e) {
    diffLogger->error("Error processing rows: {}", e.what());
    throw;
  }
}