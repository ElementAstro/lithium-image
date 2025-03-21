#include "calibration/OptimizedCorrection.hpp"
#include "Logging.hpp"
#include "calibration/CoreTypes.hpp"
#include "calibration/ImageCorrection.hpp"
#include "utils/HasSimdSupport.hpp"
#include "utils/ImageCache.hpp"
#include <opencv2/core/ocl.hpp>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

// SIMD support
#ifdef __AVX512F__
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__AVX__)
#include <immintrin.h>
#elif defined(__SSE4_2__) || defined(__SSE4_1__) || defined(__SSSE3__) ||      \
    defined(__SSE3__) || defined(__SSE2__) || defined(__SSE__)
#include <emmintrin.h> // SSE2
#include <xmmintrin.h> // SSE

#ifdef __SSE3__
#include <pmmintrin.h> // SSE3
#endif
#ifdef __SSSE3__
#include <tmmintrin.h> // SSSE3
#endif
#if defined(__SSE4_1__) || defined(__SSE4_2__)
#include <smmintrin.h> // SSE4.1
#endif
#ifdef __SSE4_2__
#include <nmmintrin.h> // SSE4.2
#endif
#endif

cv::Mat
instrument_response_correction_optimized(cv::InputArray &image,
                                         cv::InputArray &response_function,
                                         const OptimizationParams &params) {
  const auto &logger = Logger::getInstance();
  logger->debug("Applying optimized instrument response correction.");

  if (!params.isValid()) {
    logger->error("Invalid optimization parameters");
    throw InvalidParameterError("Invalid optimization parameters");
  }

  try {
    // Check cache first if enabled
    if (params.use_cache) {
      auto cache = getGlobalCache(params.cache_size);
      cv::Mat img = image.getMat();
      cv::Mat resp = response_function.getMat();

      auto cachedResult = cache->get(img);
      if (cachedResult) {
        logger->debug("Cache hit for image");
        return cachedResult.value();
      }
    }

    // GPU optimization path
    if (params.use_gpu && cv::ocl::haveOpenCL()) {
      logger->debug("Using GPU acceleration");
      cv::UMat uImage = image.getUMat();
      cv::UMat uResponse = response_function.getUMat();
      cv::UMat uResult;
      cv::multiply(uImage, uResponse, uResult);
      cv::Mat result = uResult.getMat(cv::ACCESS_READ);

      // Store in cache if enabled
      if (params.use_cache) {
        auto cache = getGlobalCache(params.cache_size);
        cache->put(image.getMat(), result);
      }

      return result;
    }

    // SIMD optimization path
    if (params.use_simd && utils::hasSIMDSupport()) {
      logger->debug("Using SIMD acceleration");
      cv::Mat img = image.getMat();
      cv::Mat resp = response_function.getMat();
      cv::Mat result = cv::Mat::zeros(img.size(), img.type());

      // Ensure single-channel float type for SIMD processing
      if (img.channels() == 1 && img.depth() == CV_32F) {
#ifdef __AVX512F__
        logger->debug("Using AVX512 instructions");
        // Process 16 floats at a time with AVX512
        for (int i = 0; i < img.rows; i++) {
          float *imgPtr = img.ptr<float>(i);
          float *respPtr = resp.ptr<float>(i);
          float *resultPtr = result.ptr<float>(i);

          int j = 0;
          for (; j <= img.cols - 16; j += 16) {
            __m512 imgVec = _mm512_loadu_ps(imgPtr + j);
            __m512 respVec = _mm512_loadu_ps(respPtr + j);
            __m512 resultVec = _mm512_mul_ps(imgVec, respVec);
            _mm512_storeu_ps(resultPtr + j, resultVec);
          }

          // Process remaining elements
          for (; j < img.cols; j++) {
            resultPtr[j] = imgPtr[j] * respPtr[j];
          }
        }
#elif defined(__AVX2__)
        logger->debug("Using AVX2 instructions");
        // Process 8 floats at a time with AVX2
        for (int i = 0; i < img.rows; i++) {
          float *imgPtr = img.ptr<float>(i);
          float *respPtr = resp.ptr<float>(i);
          float *resultPtr = result.ptr<float>(i);

          int j = 0;
          for (; j <= img.cols - 8; j += 8) {
            __m256 imgVec = _mm256_loadu_ps(imgPtr + j);
            __m256 respVec = _mm256_loadu_ps(respPtr + j);
            __m256 resultVec = _mm256_mul_ps(imgVec, respVec);
            _mm256_storeu_ps(resultPtr + j, resultVec);
          }

          // Process remaining elements
          for (; j < img.cols; j++) {
            resultPtr[j] = imgPtr[j] * respPtr[j];
          }
        }
#elif defined(__AVX__)
        logger->debug("Using AVX instructions");
        // Process 8 floats at a time with AVX
        for (int i = 0; i < img.rows; i++) {
          float *imgPtr = img.ptr<float>(i);
          float *respPtr = resp.ptr<float>(i);
          float *resultPtr = result.ptr<float>(i);

          int j = 0;
          for (; j <= img.cols - 8; j += 8) {
            __m256 imgVec = _mm256_loadu_ps(imgPtr + j);
            __m256 respVec = _mm256_loadu_ps(respPtr + j);
            __m256 resultVec = _mm256_mul_ps(imgVec, respVec);
            _mm256_storeu_ps(resultPtr + j, resultVec);
          }

          // Process remaining elements
          for (; j < img.cols; j++) {
            resultPtr[j] = imgPtr[j] * respPtr[j];
          }
        }
#elif defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__) ||            \
    defined(__SSSE3__) || defined(__SSE4_1__) || defined(__SSE4_2__)
        logger->debug("Using SSE instructions");
        // Process 4 floats at a time with SSE
        for (int i = 0; i < img.rows; i++) {
          float *imgPtr = img.ptr<float>(i);
          float *respPtr = resp.ptr<float>(i);
          float *resultPtr = result.ptr<float>(i);

          int j = 0;
          for (; j <= img.cols - 4; j += 4) {
            __m128 imgVec = _mm_loadu_ps(imgPtr + j);
            __m128 respVec = _mm_loadu_ps(respPtr + j);
            __m128 resultVec = _mm_mul_ps(imgVec, respVec);
            _mm_storeu_ps(resultPtr + j, resultVec);
          }

          // Process remaining elements
          for (; j < img.cols; j++) {
            resultPtr[j] = imgPtr[j] * respPtr[j];
          }
        }
#else
        logger->debug(
            "SIMD support detected but no specific implementation available");
        // Fall back to standard implementation if no SIMD instruction set is
        // available
        cv::multiply(img, resp, result);
#endif
        // Store in cache if enabled
        if (params.use_cache) {
          auto cache = getGlobalCache(params.cache_size);
          cache->put(img, result);
        }

        return result;
      } else {
        logger->debug("Image format not suitable for SIMD, using standard "
                      "implementation");
        cv::multiply(img, resp, result);
        return result;
      }
    }

    // Parallel optimization path
    if (params.use_parallel) {
      logger->debug("Using parallel processing with {} threads",
                    params.num_threads);
      cv::Mat img = image.getMat();
      cv::Mat resp = response_function.getMat();
      cv::Mat result = cv::Mat::zeros(img.size(), img.type());

      // Set number of threads if specified
      int oldThreads = 0;
      if (params.num_threads > 0) {
        oldThreads = tbb::global_control::active_value(
            tbb::global_control::max_allowed_parallelism);
        tbb::global_control control(
            tbb::global_control::max_allowed_parallelism, params.num_threads);
      }

      tbb::parallel_for(tbb::blocked_range<int>(0, img.rows),
                        [&](const tbb::blocked_range<int> &range) {
                          for (int i = range.begin(); i < range.end(); ++i) {
                            auto *img_ptr = img.ptr<float>(i);
                            auto *resp_ptr = resp.ptr<float>(i);
                            auto *result_ptr = result.ptr<float>(i);
                            for (int j = 0; j < img.cols; ++j) {
                              result_ptr[j] = img_ptr[j] * resp_ptr[j];
                            }
                          }
                        });

      // Store in cache if enabled
      if (params.use_cache) {
        auto cache = getGlobalCache(params.cache_size);
        cache->put(img, result);
      }

      return result;
    }

    // Default path - call standard implementation
    return instrument_response_correction(image, response_function);

  } catch (const cv::Exception &e) {
    logger->error("OpenCV error during optimized response correction: {}",
                  e.what());
    throw ProcessingError(std::string("OpenCV error: ") + e.what());
  } catch (const std::exception &e) {
    logger->error("Error during optimized instrument response correction: {}",
                  e.what());
    throw;
  }
}

cv::Mat background_noise_correction_optimized(
    cv::InputArray &image, const OptimizationParams &params) noexcept {
  const auto &logger = Logger::getInstance();
  logger->debug("Applying optimized background noise correction.");

  try {
    if (params.use_gpu && cv::ocl::haveOpenCL()) {
      logger->debug("Using GPU acceleration for background noise correction");
      cv::UMat uImage = image.getUMat();
      double medianValue = cv::mean(uImage)[0];
      cv::UMat uCorrected;
      cv::subtract(uImage, medianValue, uCorrected);
      return uCorrected.getMat(cv::ACCESS_READ);
    } else if (params.use_simd && utils::hasSIMDSupport()) {
      cv::Mat imgMat = image.getMat();
      double medianValue = cv::mean(imgMat)[0];
      cv::Mat corrected = cv::Mat::zeros(imgMat.size(), imgMat.type());

      if (imgMat.depth() == CV_32F && imgMat.channels() == 1) {
#ifdef __AVX512F__
        logger->debug(
            "Using AVX512 instructions for background noise correction");
        __m512 medianVec = _mm512_set1_ps(static_cast<float>(medianValue));

        for (int i = 0; i < imgMat.rows; i++) {
          float *src = imgMat.ptr<float>(i);
          float *dst = corrected.ptr<float>(i);

          int j = 0;
          for (; j <= imgMat.cols - 16; j += 16) {
            __m512 srcVec = _mm512_loadu_ps(src + j);
            __m512 result = _mm512_sub_ps(srcVec, medianVec);
            _mm512_storeu_ps(dst + j, result);
          }

          // Process remaining elements
          for (; j < imgMat.cols; j++) {
            dst[j] = src[j] - static_cast<float>(medianValue);
          }
        }
        return corrected;
#elif defined(__AVX2__)
        logger->debug(
            "Using AVX2 instructions for background noise correction");
        __m256 medianVec = _mm256_set1_ps(static_cast<float>(medianValue));

        for (int i = 0; i < imgMat.rows; i++) {
          float *src = imgMat.ptr<float>(i);
          float *dst = corrected.ptr<float>(i);

          int j = 0;
          for (; j <= imgMat.cols - 8; j += 8) {
            __m256 srcVec = _mm256_loadu_ps(src + j);
            __m256 result = _mm256_sub_ps(srcVec, medianVec);
            _mm256_storeu_ps(dst + j, result);
          }

          // Process remaining elements
          for (; j < imgMat.cols; j++) {
            dst[j] = src[j] - static_cast<float>(medianValue);
          }
        }
        return corrected;
#elif defined(__AVX__)
        logger->debug("Using AVX instructions for background noise correction");
        __m256 medianVec = _mm256_set1_ps(static_cast<float>(medianValue));

        for (int i = 0; i < imgMat.rows; i++) {
          float *src = imgMat.ptr<float>(i);
          float *dst = corrected.ptr<float>(i);

          int j = 0;
          for (; j <= imgMat.cols - 8; j += 8) {
            __m256 srcVec = _mm256_loadu_ps(src + j);
            __m256 result = _mm256_sub_ps(srcVec, medianVec);
            _mm256_storeu_ps(dst + j, result);
          }

          // Process remaining elements
          for (; j < imgMat.cols; j++) {
            dst[j] = src[j] - static_cast<float>(medianValue);
          }
        }
        return corrected;
#elif defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__) ||            \
    defined(__SSSE3__) || defined(__SSE4_1__) || defined(__SSE4_2__)
        logger->debug("Using SSE instructions for background noise correction");
        __m128 medianVec = _mm_set1_ps(static_cast<float>(medianValue));

        for (int i = 0; i < imgMat.rows; i++) {
          float *src = imgMat.ptr<float>(i);
          float *dst = corrected.ptr<float>(i);

          int j = 0;
          for (; j <= imgMat.cols - 4; j += 4) {
            __m128 srcVec = _mm_loadu_ps(src + j);
            __m128 result = _mm_sub_ps(srcVec, medianVec);
            _mm_storeu_ps(dst + j, result);
          }

          // Process remaining elements
          for (; j < imgMat.cols; j++) {
            dst[j] = src[j] - static_cast<float>(medianValue);
          }
        }
        return corrected;
#else
        logger->debug(
            "SIMD support detected but no specific implementation available");
        // Fall back to standard implementation
        cv::subtract(imgMat, medianValue, corrected);
        return corrected;
#endif
      } else {
        logger->debug("Image format not suitable for SIMD, using standard "
                      "implementation");
        cv::subtract(imgMat, medianValue, corrected);
        return corrected;
      }
    }

    // Fall back to standard implementation
    logger->debug(
        "Using standard implementation for background noise correction");
    return background_noise_correction(image);
  } catch (const std::exception &e) {
    logger->error("Error in optimized background noise correction: {}",
                  e.what());
    return image.getMat().clone(); // Return original image in case of failure
  }
}