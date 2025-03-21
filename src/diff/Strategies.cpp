#include "diff/Strategies.hpp"
#include "Logging.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <numeric>
#include <thread>

namespace {
std::shared_ptr<spdlog::logger> strategyLogger = Logger::getInstance();

// 计算灰度值的助手函数
unsigned char grayValue(const Image &img, int x, int y) {
  RGB pixel = img.pixelAt(x, y);
  return ColorUtils::gray(pixel);
}
} // namespace

//////////////////////////////////////////////////////////////
// ComparisonStrategyBase method implementations
//////////////////////////////////////////////////////////////

Image ComparisonStrategyBase::preprocessImage(const Image &img) const {
  if (img.isNull()) {
    throw std::invalid_argument("Cannot preprocess null image");
  }

  try {
    if (SUBSAMPLE_FACTOR > 1) {
      return img.scaled(img.width() / SUBSAMPLE_FACTOR,
                        img.height() / SUBSAMPLE_FACTOR);
    }
    return img;
  } catch (const std::exception &e) {
    strategyLogger->error("Image preprocessing failed: {}", e.what());
    throw std::runtime_error(std::string("Image preprocessing failed: ") +
                             e.what());
  }
}

void ComparisonStrategyBase::compareBlockSIMD(
    std::span<const unsigned char> block1,
    std::span<const unsigned char> block2,
    std::span<unsigned char> dest) const noexcept {

  if (block1.size() != block2.size() || block1.size() != dest.size() ||
      block1.empty()) {
    strategyLogger->error("Invalid block sizes in SIMD comparison");
    return;
  }

  const size_t size = block1.size();
  size_t i = 0;

#ifdef __AVX2__
  // 当可用时使用AVX2指令
  for (; i + 32 <= size; i += 32) {
    __m256i a = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(block1.data() + i));
    __m256i b = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>(block2.data() + i));
    __m256i diff =
        _mm256_sub_epi8(_mm256_max_epu8(a, b), _mm256_min_epu8(a, b));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dest.data() + i), diff);
  }
#elif defined(__SSE4_1__)
  // 如果没有AVX2，则使用SSE4.1
  for (; i + 16 <= size; i += 16) {
    __m128i a =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(block1.data() + i));
    __m128i b =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(block2.data() + i));
    __m128i diff = _mm_sub_epi8(_mm_max_epu8(a, b), _mm_min_epu8(a, b));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dest.data() + i), diff);
  }
#endif

  // 处理剩余元素
  for (; i < size; ++i) {
    dest[i] = static_cast<unsigned char>(
        std::abs(static_cast<int>(block1[i]) - static_cast<int>(block2[i])));
  }
}

std::vector<Rectangle>
ComparisonStrategyBase::findDifferenceRegions(const Image &diffImg) const {
  if (diffImg.isNull()) {
    strategyLogger->error("Cannot find difference regions in null image");
    return {};
  }

  try {
    const int width = diffImg.width();
    const int height = diffImg.height();
    const int threshold = 32; // 考虑为差异的最小差值

    // 使用更高效的并查集结构
    DisjointSet ds(width * height);
    std::unordered_map<int, Rectangle> regionMap;

    // 确定差异区域的连通分量
    std::atomic<int> progress{0};

#pragma omp parallel for
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (grayValue(diffImg, x, y) > threshold) {
          const int idx = y * width + x;

          // 使用方向数组获得更清晰的代码
          constexpr std::array<std::pair<int, int>, 4> directions = {
              std::make_pair(-1, 0), std::make_pair(0, -1),
              std::make_pair(1, 0), std::make_pair(0, 1)};

          for (const auto &[dx, dy] : directions) {
            const int nx = x + dx;
            const int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
                grayValue(diffImg, nx, ny) > threshold) {
#pragma omp critical
              {
                ds.unite(idx, ny * width + nx);
              }
            }
          }
        }
      }

      // 更新进度
      ++progress;
      if (progress % (height / 10) == 0) {
        strategyLogger->debug("Finding regions: {}% complete",
                              (progress * 100) / height);
      }
    }

    // 从连通分量构建矩形区域
    std::mutex regionMutex;

#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (grayValue(diffImg, x, y) > threshold) {
          int setId = ds.find(y * width + x);

#pragma omp critical
          {
            auto &rect = regionMap[setId];
            if (rect.isNull()) {
              rect = Rectangle(x, y, 1, 1);
            } else {
              rect = rect.united(Rectangle(x, y, 1, 1));
            }
          }
        }
      }
    }

    // 提取结果
    std::vector<Rectangle> result;
    result.reserve(regionMap.size());
    for (const auto &[_, rect] : regionMap) {
      result.push_back(rect);
    }

    // 通过合并重叠或相近区域来优化
    if (result.size() > 1) {
      bool merged = true;
      while (merged) {
        merged = false;
        for (size_t i = 0; i < result.size() && !merged; ++i) {
          for (size_t j = i + 1; j < result.size() && !merged; ++j) {
            const int distance = 10; // 区域合并的最大距离
            Rectangle expandedRect =
                result[i].adjusted(-distance, -distance, distance, distance);
            if (expandedRect.intersects(result[j])) {
              result[i] = result[i].united(result[j]);
              result.erase(result.begin() + j);
              merged = true;
            }
          }
        }
      }
    }

    strategyLogger->debug("Found {} difference regions", result.size());
    return result;
  } catch (const std::exception &e) {
    strategyLogger->error("Error finding difference regions: {}", e.what());
    return {};
  }
}

ComparisonStrategyBase::DisjointSet::DisjointSet(int size)
    : parent(size), rank(size, 0) {
  if (size <= 0) {
    throw std::invalid_argument("DisjointSet size must be positive");
  }
  std::iota(parent.begin(), parent.end(), 0);
}

int ComparisonStrategyBase::DisjointSet::find(int x) noexcept {
  // 路径压缩以提高性能
  if (x < 0 || x >= static_cast<int>(parent.size())) {
    return -1; // 无效索引
  }

  if (parent[x] != x) {
    parent[x] = find(parent[x]);
  }
  return parent[x];
}

void ComparisonStrategyBase::DisjointSet::unite(int x, int y) noexcept {
  if (x < 0 || x >= static_cast<int>(parent.size()) || y < 0 ||
      y >= static_cast<int>(parent.size())) {
    return; // 无效索引
  }

  x = find(x);
  y = find(y);

  if (x != y) {
    // 按秩合并以提高性能
    if (rank[x] < rank[y]) {
      std::swap(x, y);
    }
    parent[y] = x;
    if (rank[x] == rank[y]) {
      ++rank[x];
    }
  }
}

//////////////////////////////////////////////////////////////
// PixelDifferenceStrategy method implementations
//////////////////////////////////////////////////////////////

ComparisonResult
PixelDifferenceStrategy::compare(const Image &img1, const Image &img2,
                                 Promise<ComparisonResult> &promise) const {
  try {
    Image processed1 = preprocessImage(img1);
    Image processed2 = preprocessImage(img2);

    const int height = processed1.height();
    const int width = processed1.width();

    if (height <= 0 || width <= 0) {
      throw std::runtime_error("Invalid image dimensions after preprocessing");
    }

    Image diffImg(processed1.width(), processed1.height(), Image::ARGB32);
    std::atomic_uint64_t totalDiff{0};
    std::atomic_int progress{0};

    // 确定最佳线程数和块大小
    const unsigned int threadCount = std::thread::hardware_concurrency();
    const int optimalBlockSize =
        std::max(BLOCK_SIZE, static_cast<int>(32 / sizeof(unsigned char)));

// 使用OpenMP进行并行化
#pragma omp parallel for collapse(2) reduction(+ : totalDiff) schedule(dynamic)
    for (int y = 0; y < height; y += optimalBlockSize) {
      for (int x = 0; x < width; x += optimalBlockSize) {
        if (promise.isCanceled())
          continue;

        const int blockHeight = std::min(optimalBlockSize, height - y);
        const int blockWidth = std::min(optimalBlockSize, width - x);

        for (int by = 0; by < blockHeight; ++by) {
          const unsigned char *line1 = processed1.scanLine(y + by) + x * 4;
          const unsigned char *line2 = processed2.scanLine(y + by) + x * 4;
          unsigned char *dest = diffImg.scanLine(y + by) + x * 4;

          // 使用span进行更安全的内存访问
          std::span<const unsigned char> block1(line1, blockWidth * 4);
          std::span<const unsigned char> block2(line2, blockWidth * 4);
          std::span<unsigned char> destSpan(dest, blockWidth * 4);

          compareBlockSIMD(block1, block2, destSpan);

          // 使用SIMD计算总差异
          uint64_t localDiff = 0;
          for (int bx = 0; bx < blockWidth * 4; ++bx) {
            localDiff += dest[bx];
          }
          totalDiff += localDiff;
        }

#pragma omp atomic
        progress += blockHeight;
      }

      // 每10%更新一次进度
      if (progress % (height / 10) == 0) {
        promise.setProgressValue(static_cast<int>(progress * 100.0 / height));
      }
    }

    // 计算相似度，并进行适当的归一化
    const double maxPossibleDiff = 255.0 * width * height * 4;
    const double mse = static_cast<double>(totalDiff) / (width * height * 4);
    const double similarity = 100.0 * (1.0 - std::sqrt(mse) / 255.0);

    strategyLogger->debug(
        "Pixel difference comparison - MSE: {:.4f}, Similarity: {:.2f}%", mse,
        similarity);

    return {diffImg, similarity, findDifferenceRegions(diffImg)};
  } catch (const std::exception &e) {
    strategyLogger->error("Error in pixel difference comparison: {}", e.what());
    promise.cancel();
    throw;
  }
}

//////////////////////////////////////////////////////////////
// SSIMStrategy method implementations
//////////////////////////////////////////////////////////////

ComparisonResult
SSIMStrategy::compare(const Image &img1, const Image &img2,
                      Promise<ComparisonResult> &promise) const {
  try {
    Image processed1 = preprocessImage(img1);
    Image processed2 = preprocessImage(img2);

    const int width = processed1.width();
    const int height = processed1.height();

    if (width < WINDOW_SIZE || height < WINDOW_SIZE) {
      throw std::runtime_error("Images too small for SSIM comparison");
    }

    Image diffImg(width, height, Image::ARGB32);
    diffImg.fill(RGB(255, 255, 255)); // 初始化为白色
    std::atomic<double> totalSSIM{0.0};
    std::atomic<int> processedWindows{0};
    const int totalWindows = ((width - WINDOW_SIZE) / WINDOW_SIZE) *
                             ((height - WINDOW_SIZE) / WINDOW_SIZE);

// 窗口的并行处理
#pragma omp parallel for collapse(2) reduction(+ : totalSSIM) schedule(dynamic)
    for (int y = 0; y <= height - WINDOW_SIZE; y += WINDOW_SIZE) {
      for (int x = 0; x <= width - WINDOW_SIZE; x += WINDOW_SIZE) {
        if (promise.isCanceled())
          continue;

        double ssim = computeSSIM(processed1, processed2, x, y);
        totalSSIM += ssim;

        // 将SSIM转换为颜色（1.0 = 白色，0.0 = 黑色）
        int color = static_cast<int>((1.0 - ssim) * 255);
        color = std::clamp(color, 0, 255);
        RGB value = RGB(color, color, color);

        // 在差异图像中填充窗口
        for (int wy = 0; wy < WINDOW_SIZE && y + wy < height; ++wy) {
          for (int wx = 0; wx < WINDOW_SIZE && x + wx < width; ++wx) {
            diffImg.setPixelAt(x + wx, y + wy, value);
          }
        }

#pragma omp atomic
        ++processedWindows;

        // 更新进度
        if (processedWindows % std::max(1, totalWindows / 20) == 0) {
          double progress =
              static_cast<double>(processedWindows) / totalWindows;
          promise.setProgressValue(static_cast<int>(progress * 100));
          strategyLogger->debug("SSIM progress: {:.1f}%", progress * 100);
        }
      }
    }

    // 计算最终相似度百分比
    double numWindows = totalWindows > 0 ? totalWindows : 1.0;
    double similarity = (totalSSIM * 100.0) / numWindows;
    similarity = std::clamp(similarity, 0.0, 100.0);

    strategyLogger->debug(
        "SSIM comparison - Average SSIM: {:.4f}, Similarity: {:.2f}%",
        totalSSIM / numWindows, similarity);

    return {diffImg, similarity, findDifferenceRegions(diffImg)};
  } catch (const std::exception &e) {
    strategyLogger->error("Error in SSIM comparison: {}", e.what());
    promise.cancel();
    throw;
  }
}

double SSIMStrategy::computeSSIM(const Image &img1, const Image &img2, int x,
                                 int y) const noexcept {
  if (img1.isNull() || img2.isNull() || x < 0 || y < 0 ||
      x + WINDOW_SIZE > img1.width() || y + WINDOW_SIZE > img1.height() ||
      x + WINDOW_SIZE > img2.width() || y + WINDOW_SIZE > img2.height()) {
    return 0.0; // 无效输入
  }

  double mean1 = 0, mean2 = 0, variance1 = 0, variance2 = 0, covariance = 0;
  std::array<double, WINDOW_SIZE * WINDOW_SIZE> values1;
  std::array<double, WINDOW_SIZE * WINDOW_SIZE> values2;

  // 第一遍：计算均值
  int idx = 0;
  for (int wy = 0; wy < WINDOW_SIZE; ++wy) {
    for (int wx = 0; wx < WINDOW_SIZE; ++wx) {
      values1[idx] = grayValue(img1, x + wx, y + wy);
      values2[idx] = grayValue(img2, x + wx, y + wy);
      mean1 += values1[idx];
      mean2 += values2[idx];
      ++idx;
    }
  }

  const double windowSize = WINDOW_SIZE * WINDOW_SIZE;
  mean1 /= windowSize;
  mean2 /= windowSize;

  // 第二遍：计算方差和协方差
  for (size_t i = 0; i < values1.size(); ++i) {
    double diff1 = values1[i] - mean1;
    double diff2 = values2[i] - mean2;
    variance1 += diff1 * diff1;
    variance2 += diff2 * diff2;
    covariance += diff1 * diff2;
  }

  variance1 /= (windowSize - 1);
  variance2 /= (windowSize - 1);
  covariance /= (windowSize - 1);

  // 常数以稳定除法
  const double C1 = (K1 * 255) * (K1 * 255);
  const double C2 = (K2 * 255) * (K2 * 255);

  // SSIM公式
  double numerator = (2 * mean1 * mean2 + C1) * (2 * covariance + C2);
  double denominator =
      (mean1 * mean1 + mean2 * mean2 + C1) * (variance1 + variance2 + C2);

  if (denominator < 1e-10) {
    return 0.0; // 避免除以零
  }

  double ssim = numerator / denominator;
  return std::clamp(ssim, 0.0, 1.0); // 确保值在有效范围内
}

//////////////////////////////////////////////////////////////
// PerceptualHashStrategy method implementations
//////////////////////////////////////////////////////////////

ComparisonResult
PerceptualHashStrategy::compare(const Image &img1, const Image &img2,
                                Promise<ComparisonResult> &promise) const {
  try {
    promise.setProgressValue(10);
    uint64_t hash1 = computeHash(img1);

    promise.setProgressValue(50);
    uint64_t hash2 = computeHash(img2);

    promise.setProgressValue(70);

    int distance = hammingDistance(hash1, hash2);
    double similarity =
        100.0 * (1.0 - static_cast<double>(distance) / HASH_SIZE);

    // 创建差异可视化
    Image diffImg(std::max(img1.width(), img2.width()),
                  std::max(img1.height(), img2.height()), Image::ARGB32);
    diffImg.fill(RGB(255, 255, 255));

    if (distance > 0) {
      // 可视化不同的位
      const int blockWidth = diffImg.width() / 8;
      const int blockHeight = diffImg.height() / 8;

      for (int i = 0; i < 64; ++i) {
        bool bit1 = (hash1 & (1ULL << i)) != 0;
        bool bit2 = (hash2 & (1ULL << i)) != 0;

        if (bit1 != bit2) {
          int x = (i % 8) * blockWidth;
          int y = (i / 8) * blockHeight;
          diffImg.fillRect(Rectangle(x, y, blockWidth, blockHeight),
                           RGB(255, 0, 0, 127));
        }
      }
    }

    promise.setProgressValue(90);

    strategyLogger->debug(
        "Perceptual hash comparison - Distance: {}/64, Similarity: {:.2f}%",
        distance, similarity);

    // 创建通用差异区域，因为pHash无法捕获确切的像素差异
    std::vector<Rectangle> regions;
    if (distance > 0) {
      const int regionSize = 50;                             // 近似区域大小
      const int numRegions = std::min(5, 1 + distance / 10); // 限制区域数量

      for (int i = 0; i < numRegions; ++i) {
        int x = (diffImg.width() - regionSize) * (i + 1) / (numRegions + 1);
        int y = (diffImg.height() - regionSize) / 2;
        regions.push_back(Rectangle(x, y, regionSize, regionSize));
      }
    }

    promise.setProgressValue(100);
    return {diffImg, similarity, regions};
  } catch (const std::exception &e) {
    strategyLogger->error("Error in perceptual hash comparison: {}", e.what());
    promise.cancel();
    throw;
  }
}

uint64_t PerceptualHashStrategy::computeHash(const Image &img) const {
  if (img.isNull()) {
    throw std::invalid_argument("Cannot compute hash of null image");
  }

  try {
    // 调整为8x8灰度图像
    Image scaled = img.scaled(8, 8).convertToFormat(Image::Grayscale8);

    // 计算平均像素值
    int sum = 0;
    std::array<int, 64> pixels;
    int idx = 0;

    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 8; ++x) {
        pixels[idx] = grayValue(scaled, x, y);
        sum += pixels[idx];
        ++idx;
      }
    }

    int avg = sum / 64;

    // 计算哈希：如果像素>=平均值，则设置相应的位
    uint64_t hash = 0;
    for (int i = 0; i < 64; ++i) {
      hash = (hash << 1) | (pixels[i] >= avg ? 1 : 0);
    }

    return hash;
  } catch (const std::exception &e) {
    strategyLogger->error("Error computing perceptual hash: {}", e.what());
    throw std::runtime_error(std::string("Error computing perceptual hash: ") +
                             e.what());
  }
}

int PerceptualHashStrategy::hammingDistance(uint64_t hash1,
                                            uint64_t hash2) const noexcept {
  // 使用C++20的std::popcount进行高效位计数
  return std::popcount(hash1 ^ hash2);
}

//////////////////////////////////////////////////////////////
// HistogramStrategy method implementations
//////////////////////////////////////////////////////////////

ComparisonResult
HistogramStrategy::compare(const Image &img1, const Image &img2,
                           Promise<ComparisonResult> &promise) const {
  try {
    promise.setProgressValue(10);
    auto hist1 = computeHistogram(img1);

    promise.setProgressValue(40);
    auto hist2 = computeHistogram(img2);

    promise.setProgressValue(70);
    double similarity = compareHistograms(hist1, hist2);

    // 创建差异可视化作为直方图图表
    Image diffImg(std::max(img1.width(), img2.width()),
                  std::max(img1.height(), img2.height()), Image::ARGB32);
    diffImg.fill(RGB(255, 255, 255));

    // 绘制直方图背景网格
    for (int i = 0; i < 4; ++i) {
      int y = diffImg.height() * i / 4;
      diffImg.drawLine(0, y, diffImg.width(), y, RGB(200, 200, 200));
    }

    // 计算显示的归一化因子
    int maxHistValue1 = *std::max_element(hist1.begin(), hist1.end());
    int maxHistValue2 = *std::max_element(hist2.begin(), hist2.end());
    double normFactor = static_cast<double>(diffImg.height()) /
                        (std::max(maxHistValue1, maxHistValue2) * 1.1);

    // 绘制第一个直方图（蓝色）
    for (int i = 0; i < HIST_BINS - 1; ++i) {
      int x1 = i * diffImg.width() / HIST_BINS;
      int x2 = (i + 1) * diffImg.width() / HIST_BINS;
      int y1 = diffImg.height() - static_cast<int>(hist1[i] * normFactor);
      int y2 = diffImg.height() - static_cast<int>(hist1[i + 1] * normFactor);
      diffImg.drawLine(x1, y1, x2, y2, RGB(0, 0, 255, 200), 2);
    }

    // 绘制第二个直方图（红色）
    for (int i = 0; i < HIST_BINS - 1; ++i) {
      int x1 = i * diffImg.width() / HIST_BINS;
      int x2 = (i + 1) * diffImg.width() / HIST_BINS;
      int y1 = diffImg.height() - static_cast<int>(hist2[i] * normFactor);
      int y2 = diffImg.height() - static_cast<int>(hist2[i + 1] * normFactor);
      diffImg.drawLine(x1, y1, x2, y2, RGB(255, 0, 0, 200), 2);
    }

    // 绘制图例
    Rectangle legendRect(10, 10, 200, 40);
    diffImg.fillRect(legendRect, RGB(255, 255, 255, 200));
    diffImg.drawRect(legendRect, RGB(0, 0, 0));

    diffImg.drawText(20, 30, "Image 1", RGB(0, 0, 255));
    diffImg.fillRect(Rectangle(100, 22, 20, 10), RGB(0, 0, 255, 200));

    diffImg.drawText(130, 30, "Image 2", RGB(255, 0, 0));
    diffImg.fillRect(Rectangle(190, 22, 20, 10), RGB(255, 0, 0, 200));

    promise.setProgressValue(100);

    // 将相似度转换为百分比
    double similarityPercent = similarity * 100.0;

    strategyLogger->debug(
        "Histogram comparison - Correlation: {:.4f}, Similarity: {:.2f}%",
        similarity, similarityPercent);

    return {diffImg, similarityPercent, {}};
  } catch (const std::exception &e) {
    strategyLogger->error("Error in histogram comparison: {}", e.what());
    promise.cancel();
    throw;
  }
}

std::vector<int> HistogramStrategy::computeHistogram(const Image &img) const {
  if (img.isNull()) {
    throw std::invalid_argument("Cannot compute histogram of null image");
  }

  try {
    std::vector<int> histogram(HIST_BINS, 0);
    std::vector<std::thread> threads;
    const unsigned int threadCount = std::thread::hardware_concurrency();
    std::vector<std::vector<int>> threadHistograms(
        threadCount, std::vector<int>(HIST_BINS, 0));

    // 在线程之间分配工作
    for (unsigned int t = 0; t < threadCount; ++t) {
      threads.emplace_back([&img, t, threadCount, &threadHistograms]() {
        for (int y = t; y < img.height(); y += threadCount) {
          for (int x = 0; x < img.width(); ++x) {
            int gray = grayValue(img, x, y);
            if (gray >= 0 && gray < HIST_BINS) {
              ++threadHistograms[t][gray];
            }
          }
        }
      });
    }

    // 合并线程并组合结果
    for (auto &thread : threads) {
      thread.join();
    }

    // 合并线程局部直方图
    for (const auto &threadHist : threadHistograms) {
      for (int i = 0; i < HIST_BINS; ++i) {
        histogram[i] += threadHist[i];
      }
    }

    return histogram;
  } catch (const std::exception &e) {
    strategyLogger->error("Error computing histogram: {}", e.what());
    throw std::runtime_error(std::string("Error computing histogram: ") +
                             e.what());
  }
}

double HistogramStrategy::compareHistograms(
    const std::vector<int> &hist1,
    const std::vector<int> &hist2) const noexcept {
  if (hist1.size() != hist2.size() || hist1.empty()) {
    return 0.0;
  }

  try {
    // 使用优化算法计算相关性
    double correlation = 0;
    double norm1 = 0, norm2 = 0;

#pragma omp parallel for simd reduction(+ : correlation, norm1, norm2)
    for (size_t i = 0; i < hist1.size(); ++i) {
      correlation += static_cast<double>(hist1[i]) * hist2[i];
      norm1 += static_cast<double>(hist1[i]) * hist1[i];
      norm2 += static_cast<double>(hist2[i]) * hist2[i];
    }

    if (norm1 < 1e-10 || norm2 < 1e-10) {
      return 0.0; // 避免除以零
    }

    return correlation / (std::sqrt(norm1) * std::sqrt(norm2));
  } catch (...) {
    return 0.0; // 安全处理任何意外错误
  }
}