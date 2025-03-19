// MemoryPool.hpp
#pragma once

#include <mutex>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

/**
 * @class MemoryPool
 * @brief 管理CV::Mat对象的内存池
 */
class MemoryPool {
public:
  static cv::Mat allocate(int rows, int cols, int type);
  static void deallocate(cv::Mat &mat);
  static void clear();

private:
  static std::vector<std::shared_ptr<cv::Mat>> pool_;
  static std::mutex pool_mutex_;
  static const int max_pool_size_ = 100;
};