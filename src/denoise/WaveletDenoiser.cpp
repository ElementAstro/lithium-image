#include "denoise/WaveletDenoiser.hpp"
#include "Logging.hpp"
#include <algorithm>
#include <future>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <xmmintrin.h>
#endif

void WaveletDenoiser::denoise(const cv::Mat &src, cv::Mat &dst, int levels,
                              float threshold) {
  auto logger = Logger::getInstance();
  logger->debug("Starting wavelet denoise with levels: {}, threshold: {}",
                levels, threshold);

  // 为简单起见，仅对单通道进行演示，彩色可拆分通道分别处理
  if (src.channels() > 1) {
    logger->debug("Processing multi-channel image");
    // 转换为Lab或YUV后处理亮度通道，再合并，还可并行处理
    cv::Mat copyImg;
    src.copyTo(copyImg);

    std::vector<cv::Mat> bgr;
    cv::split(copyImg, bgr);

    // 仅处理第一个通道(蓝色通道)做演示，实际可处理所有通道
    cv::Mat denoisedChannel;
    wavelet_process_single_channel(bgr[0], denoisedChannel, levels, threshold);
    bgr[0] = denoisedChannel;

    cv::merge(bgr, dst);
    logger->debug("Multi-channel wavelet denoise completed");
  } else {
    logger->debug("Processing single-channel image");
    wavelet_process_single_channel(src, dst, levels, threshold);
    logger->debug("Single-channel wavelet denoise completed");
  }
}

void WaveletDenoiser::wavelet_process_single_channel(const cv::Mat &src,
                                                     cv::Mat &dst, int levels,
                                                     float threshold) {
  auto logger = Logger::getInstance();
  logger->debug("Starting wavelet process for single channel with "
                "levels: {}, threshold: {}",
                levels, threshold);
  // 转成float类型，便于处理
  cv::Mat floatSrc;
  src.convertTo(floatSrc, CV_32F);
  logger->debug("Converted source to float type");

  // 小波分解
  cv::Mat waveCoeffs = floatSrc.clone();
  for (int i = 0; i < levels; i++) {
    waveCoeffs = decompose_one_level(waveCoeffs);
    logger->debug("Decomposed level {}", i + 1);
  }

  // 去噪（简单阈值处理）
  cv::threshold(waveCoeffs, waveCoeffs, threshold, 0, cv::THRESH_TOZERO);
  logger->debug("Applied thresholding");

  // 逆变换
  for (int i = 0; i < levels; i++) {
    waveCoeffs = recompose_one_level(waveCoeffs, floatSrc.size());
    logger->debug("Recomposed level {}", i + 1);
  }

  // 转回原类型
  waveCoeffs.convertTo(dst, src.type());
  logger->debug("Converted back to original type");
}

// 单层离散小波分解(示例性拆分，不以真实小波为准)
cv::Mat WaveletDenoiser::decompose_one_level(const cv::Mat &src) {
  auto logger = Logger::getInstance();
  logger->debug("Starting decompose one level");
  cv::Mat dst = src.clone();
  // 此处可进行实际小波分解，此处仅演示简化方法：
  // 例如：将图像分块(低频 + 高频)
  // 这里直接用高通滤波模拟高频，低通滤波模拟低频
  cv::Mat lowFreq, highFreq;
  cv::blur(dst, lowFreq, cv::Size(3, 3));
  highFreq = dst - lowFreq;
  // 将低频和高频拼接在同一Mat中返回(仅示意)
  // 为了安全，在行方向拼接（可根据需求改变）
  cv::Mat combined;
  cv::vconcat(lowFreq, highFreq, combined);
  logger->debug("Decompose one level completed");
  return combined;
}

// 单层离散小波重构(示例性的逆过程)
cv::Mat WaveletDenoiser::recompose_one_level(const cv::Mat &waveCoeffs,
                                             const cv::Size &originalSize) {
  auto logger = Logger::getInstance();
  logger->debug("Starting recompose one level");
  // 假设waveCoeffs是上下拼接的
  int rowCount = waveCoeffs.rows / 2;
  cv::Mat lowFreq = waveCoeffs(cv::Rect(0, 0, waveCoeffs.cols, rowCount));
  cv::Mat highFreq =
      waveCoeffs(cv::Rect(0, rowCount, waveCoeffs.cols, rowCount));

  // 简化逆过程：dst = lowFreq + highFreq
  cv::Mat combined = lowFreq + highFreq;

  // 保证输出大小与原图一致(多层变换后可能需要特别处理)
  if (combined.size() != originalSize) {
    cv::resize(combined, combined, originalSize, 0, 0, cv::INTER_LINEAR);
    logger->debug("Resized combined image to original size");
  }
  logger->debug("Recompose one level completed");
  return combined;
}

// SIMD优化的小波变换
void WaveletDenoiser::wavelet_transform_simd(cv::Mat &data) {
  const int n = data.cols;
  float *ptr = data.ptr<float>();

#if defined(__AVX2__)
  for (int i = 0; i < n / 8; ++i) {
    __m256 vec = _mm256_loadu_ps(ptr + i * 8);
    __m256 result = _mm256_mul_ps(vec, _mm256_set1_ps(0.707106781f));
    _mm256_storeu_ps(ptr + i * 8, result);
  }
#elif defined(__SSE2__)
  for (int i = 0; i < n / 4; ++i) {
    __m128 vec = _mm_loadu_ps(ptr + i * 4);
    __m128 result = _mm_mul_ps(vec, _mm_set1_ps(0.707106781f));
    _mm_storeu_ps(ptr + i * 4, result);
  }
#else
#pragma omp simd
  for (int i = 0; i < n; ++i) {
    ptr[i] *= 0.707106781f;
  }
#endif
}

void WaveletDenoiser::process_blocks(
    cv::Mat &img, int block_size,
    const std::function<void(cv::Mat &)> &process_fn) {
  const int rows = img.rows;
  const int cols = img.cols;

// 使用动态调度以更好地平衡负载
#pragma omp parallel for collapse(2) schedule(dynamic, 1)
  for (int i = 0; i < rows; i += block_size) {
    for (int j = 0; j < cols; j += block_size) {
      const int current_block_rows = std::min(block_size, rows - i);
      const int current_block_cols = std::min(block_size, cols - j);

      // 使用连续内存块提高缓存命中率
      cv::Mat block;
      img(cv::Range(i, i + current_block_rows),
          cv::Range(j, j + current_block_cols))
          .copyTo(block);

      process_fn(block);

      // 写回结果
      block.copyTo(img(cv::Range(i, i + current_block_rows),
                       cv::Range(j, j + current_block_cols)));
    }
  }
}

void WaveletDenoiser::stream_process(
    const cv::Mat &src, cv::Mat &dst,
    const std::function<void(cv::Mat &)> &process_fn) {
  const int pipeline_stages = 3;
  const int tile_rows = src.rows / pipeline_stages;

  std::vector<cv::Mat> tiles(pipeline_stages);
  std::vector<std::future<void>> futures(pipeline_stages);

  // 创建流水线
  for (int i = 0; i < pipeline_stages; ++i) {
    cv::Mat tile = src(cv::Range(i * tile_rows, (i + 1) * tile_rows),
                       cv::Range(0, src.cols))
                       .clone();
    futures[i] = std::async(std::launch::async,
                            [&process_fn, &tile]() { process_fn(tile); });
    tiles[i] = tile;
  }

  // 等待所有处理完成
  for (int i = 0; i < pipeline_stages; ++i) {
    futures[i].wait();
    tiles[i].copyTo(dst(cv::Range(i * tile_rows, (i + 1) * tile_rows),
                        cv::Range(0, src.cols)));
  }
}

// 优化内存访问模式
void WaveletDenoiser::optimize_memory_layout(cv::Mat &data) {
  auto logger = Logger::getInstance();
  // 确保数据是连续的
  if (!data.isContinuous()) {
    data = data.clone();
  }

  // 内存对齐
  const size_t alignment = 32; // AVX2需要32字节对齐
  uchar *ptr = data.data;
  size_t space = data.total() * data.elemSize();
  void *aligned_ptr = nullptr;

#if defined(_WIN32)
  aligned_ptr = _aligned_malloc(space, alignment);
  if (aligned_ptr) {
    memcpy(aligned_ptr, ptr, space);
    data = cv::Mat(data.rows, data.cols, data.type(), aligned_ptr);
  }
#else
  if (posix_memalign(&aligned_ptr, alignment, space) == 0) {
    memcpy(aligned_ptr, ptr, space);
    data = cv::Mat(data.rows, data.cols, data.type(), aligned_ptr);
  }
#endif
}

// 使用SIMD优化的tile处理
void WaveletDenoiser::process_tile_simd(cv::Mat &tile) {
  optimize_memory_layout(tile);

  float *ptr = tile.ptr<float>();
  const int size = tile.total();

#if defined(__AVX2__)
  const int vec_size = 8;
  const int vec_count = size / vec_size;

#pragma omp parallel for
  for (int i = 0; i < vec_count; ++i) {
    __m256 vec = _mm256_load_ps(ptr + i * vec_size);
    // 进行SIMD运算 - 简单示例
    __m256 result = _mm256_mul_ps(vec, _mm256_set1_ps(0.9f));
    _mm256_store_ps(ptr + i * vec_size, result);
  }
#endif

  // 处理剩余元素
  for (int i = (size / 8) * 8; i < size; ++i) {
    ptr[i] *= 0.9f;
  }
}

float WaveletDenoiser::compute_adaptive_threshold(const cv::Mat &coeffs,
                                                  double noise_estimate) {
  auto logger = Logger::getInstance();
  logger->debug("Starting compute adaptive threshold with noise "
                "estimate: {}",
                noise_estimate);
  cv::Mat abs_coeffs;
  cv::absdiff(coeffs, cv::Scalar(0), abs_coeffs);

  double median = 0.0;
#pragma omp parallel
  {
    std::vector<float> local_data;
#pragma omp for nowait
    for (int i = 0; i < coeffs.rows; ++i) {
      for (int j = 0; j < coeffs.cols; ++j) {
        local_data.push_back(abs_coeffs.at<float>(i, j));
      }
    }

#pragma omp critical
    {
      std::sort(local_data.begin(), local_data.end());
      median = local_data[local_data.size() / 2];
    }
  }

  logger->debug("Adaptive threshold computed: {}", median * noise_estimate);
  return static_cast<float>(median * noise_estimate);
}

void WaveletDenoiser::denoise(const cv::Mat &src, cv::Mat &dst,
                              const DenoiseParameters &params) {
  auto logger = Logger::getInstance();
  logger->info("Starting wavelet denoise with parameters");
  cv::Mat working;
  src.convertTo(working, CV_32F);
  logger->debug("Converted source to float type");

  // 分块并行处理
  process_blocks(working, params.block_size, [&params](cv::Mat &block) {
    // 小波变换
    for (int level = 0; level < params.wavelet_level; ++level) {
      // 行变换
      for (int i = 0; i < block.rows; ++i) {
        cv::Mat row = block.row(i);
        wavelet_transform_simd(row);
      }

      // 列变换
      cv::Mat block_t = block.t();
      for (int i = 0; i < block_t.rows; ++i) {
        cv::Mat row = block_t.row(i);
        wavelet_transform_simd(row);
      }
      block = block_t.t();
    }

    // 自适应阈值去噪
    if (params.use_adaptive_threshold) {
      float thresh = compute_adaptive_threshold(block, params.noise_estimate);
      cv::threshold(block, block, thresh, 0, cv::THRESH_TOZERO);
    } else {
      cv::threshold(block, block, params.wavelet_threshold, 0,
                    cv::THRESH_TOZERO);
    }

    // 逆变换过程类似
  });

  working.convertTo(dst, src.type());
  logger->info("Wavelet denoise completed");
}

void WaveletDenoiser::parallel_wavelet_transform(cv::Mat &data) {
  const int rows = data.rows;
  const int cols = data.cols;
  const int tile_size = 32; // 分块大小

// 并行处理每个块
#pragma omp parallel for collapse(2)
  for (int i = 0; i < rows; i += tile_size) {
    for (int j = 0; j < cols; j += tile_size) {
      // 计算当前块的实际大小
      int current_rows = std::min(tile_size, rows - i);
      int current_cols = std::min(tile_size, cols - j);

      // 获取当前块
      cv::Mat tile =
          data(cv::Range(i, i + current_rows), cv::Range(j, j + current_cols));

      // 对当前块应用SIMD优化的小波变换
      process_tile_simd(tile);
    }
  }

// 同步所有线程
#pragma omp barrier
}