#pragma once

#include "CoreTypes.hpp"
#include "Parameters.hpp"
#include <functional>
#include <opencv2/opencv.hpp>
#include <vector>

// Batch processing capability
std::vector<cv::Mat>
batch_process_images(const std::vector<cv::Mat> &images,
                     const std::function<cv::Mat(const cv::Mat &)> &processor,
                     const OptimizationParams &params = {});