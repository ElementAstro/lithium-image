// ImageProcessingConfig.hpp
#pragma once

#include <vector>

/**
 * @enum BorderMode
 * @brief 表示不同边界处理模式的枚举
 */
enum class BorderMode { ZERO_PADDING, MIRROR_REFLECT, REPLICATE, CIRCULAR };

/**
 * @enum DeconvMethod
 * @brief 表示不同反卷积方法的枚举
 */
enum class DeconvMethod { RICHARDSON_LUCY, WIENER, TIKHONOV };

/**
 * @struct ConvolutionConfig
 * @brief 卷积操作的配置结构
 */
struct ConvolutionConfig {
  std::vector<float> kernel;                      ///< 卷积核
  int kernel_size;                                ///< 卷积核大小
  BorderMode border_mode = BorderMode::REPLICATE; ///< 边界处理模式
  bool normalize_kernel = true;                   ///< 是否归一化卷积核
  bool parallel_execution = true;                 ///< 是否启用并行执行
  bool per_channel = false;                       ///< 是否分通道处理
  bool use_simd = true;                           ///< 启用SIMD优化
  bool use_memory_pool = true;                    ///< 使用内存池
  int tile_size = 256;                            ///< 缓存优化的块大小
  bool use_fft = false;                           ///< 对大型核使用FFT
  int thread_count = 0;                           ///< 线程数（0表示自动）
  bool use_avx = true;                            ///< 使用AVX指令集
  int block_size = 32;                            ///< 缓存块大小
};

/**
 * @struct DeconvolutionConfig
 * @brief 反卷积操作的配置结构
 */
struct DeconvolutionConfig {
  DeconvMethod method = DeconvMethod::RICHARDSON_LUCY; ///< 反卷积方法
  int iterations = 30;                                 ///< 迭代方法的迭代次数
  double noise_power = 0.0;     ///< Wiener反卷积的噪声功率
  double regularization = 1e-6; ///< Tikhonov反卷积的正则化参数
  BorderMode border_mode = BorderMode::REPLICATE; ///< 边界处理模式
  bool per_channel = false;                       ///< 是否分通道处理
  bool use_simd = true;                           ///< 启用SIMD优化
  bool use_memory_pool = true;                    ///< 使用内存池
  int tile_size = 256;                            ///< 缓存优化的块大小
  bool use_fft = true;                            ///< 使用FFT加速
  int thread_count = 0;                           ///< 线程数（0表示自动）
  bool use_avx = true;                            ///< 使用AVX指令集
  int block_size = 32;                            ///< 缓存块大小
};

// C++20 concept用于验证配置类型
template <typename T>
concept ConfigType =
    std::same_as<T, ConvolutionConfig> || std::same_as<T, DeconvolutionConfig>;