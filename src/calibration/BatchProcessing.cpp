#include "calibration/BatchProcessing.hpp"

#include <algorithm>
#include <execution>
#include <numeric>
#include <spdlog/spdlog.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <opencv2/core/ocl.hpp>

// Implement batch processing using C++20 ranges and parallelism
std::vector<cv::Mat>
batch_process_images(const std::vector<cv::Mat> &images,
                     const std::function<cv::Mat(const cv::Mat &)> &processor,
                     const OptimizationParams &params) {

  spdlog::debug("Starting batch processing of {} images", images.size());

  std::vector<cv::Mat> results(images.size());

  try {
    if (images.empty()) {
      return results;
    }

    if (params.use_parallel) {
#if __cplusplus >= 202002L
      // Using C++20 ranges and views
      std::vector<size_t> indices(images.size());
      std::iota(indices.begin(), indices.end(), 0);

      if (params.use_gpu && cv::ocl::haveOpenCL()) {
        std::for_each(
            std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
              try {
                cv::UMat uimage = images[i].getUMat(cv::ACCESS_READ);
                cv::Mat result = processor(uimage.getMat(cv::ACCESS_READ));
                results[i] = result;
              } catch (const std::exception &e) {
                spdlog::error("Error processing image {}: {}", i, e.what());
                results[i] = images[i].clone(); // Return original on error
              }
            });
      } else {
        std::for_each(
            std::execution::par, indices.begin(), indices.end(), [&](size_t i) {
              try {
                results[i] = processor(images[i]);
              } catch (const std::exception &e) {
                spdlog::error("Error processing image {}: {}", i, e.what());
                results[i] = images[i].clone(); // Return original on error
              }
            });
      }
#else
      // Fallback for older C++ standards
      tbb::parallel_for(tbb::blocked_range<size_t>(0, images.size()),
                        [&](const tbb::blocked_range<size_t> &range) {
                          for (size_t i = range.begin(); i < range.end(); ++i) {
                            try {
                              results[i] = processor(images[i]);
                            } catch (const std::exception &e) {
                              spdlog::error("Error processing image {}: {}", i,
                                            e.what());
                              results[i] = images[i].clone();
                            }
                          }
                        });
#endif
    } else {
      // Sequential processing
      for (size_t i = 0; i < images.size(); ++i) {
        try {
          results[i] = processor(images[i]);
        } catch (const std::exception &e) {
          spdlog::error("Error processing image {}: {}", i, e.what());
          results[i] = images[i].clone();
        }
      }
    }

    spdlog::info("Batch processing completed for {} images", images.size());
    return results;
  } catch (const std::exception &e) {
    spdlog::error("Error during batch processing: {}", e.what());
    return images; // Return original images on error
  }
}