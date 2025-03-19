// MemoryPool.cpp
#include "MemoryPool.hpp"
#include <spdlog/spdlog.h>

std::vector<std::shared_ptr<cv::Mat>> MemoryPool::pool_;
std::mutex MemoryPool::pool_mutex_;
const int MemoryPool::max_pool_size_;

namespace {
    std::shared_ptr<spdlog::logger> pool_logger =
        spdlog::basic_logger_mt("PoolLogger", "logs/memory_pool.log");
}

cv::Mat MemoryPool::allocate(int rows, int cols, int type) {
    try {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        pool_logger->debug(
            "Allocating memory from pool: rows={}, cols={}, type={}", rows, cols,
            type);

        for (auto it = pool_.begin(); it != pool_.end();) {
            auto &mat_ptr = *it;
            // 检查shared_ptr是否唯一（未在其他地方使用）
            if (mat_ptr.use_count() == 1 && mat_ptr->rows == rows &&
                mat_ptr->cols == cols && mat_ptr->type() == type) {

                cv::Mat mat = *mat_ptr;
                // 将矩阵置零以避免数据泄露
                mat.setTo(cv::Scalar::all(0));

                it = pool_.erase(it);
                pool_logger->debug("Memory allocated from pool");
                return mat;
            } else {
                ++it;
            }
        }

        pool_logger->debug("No suitable memory found in pool, allocating new");
        cv::Mat newMat(rows, cols, type);
        return newMat;
    } catch (const cv::Exception &e) {
        pool_logger->error("OpenCV exception in memory allocation: {}", e.what());
        throw; // 重新抛出以让调用者处理
    } catch (const std::exception &e) {
        pool_logger->error("Exception in memory allocation: {}", e.what());
        throw;
    } catch (...) {
        pool_logger->error("Unknown exception in memory allocation");
        throw;
    }
}

void MemoryPool::deallocate(cv::Mat &mat) {
    if (!mat.empty()) {
        try {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            if (pool_.size() < max_pool_size_) {
                pool_logger->debug("Deallocating matrix {}x{} type {} to pool",
                                  mat.rows, mat.cols, mat.type());

                // 创建指向mat数据的shared_ptr
                auto mat_ptr = std::make_shared<cv::Mat>(mat);
                pool_.push_back(mat_ptr);
            } else {
                pool_logger->warn("Memory pool is full (size={}), discarding matrix",
                                 pool_.size());
            }
            // 将mat设为空以表示已被回收
            mat = cv::Mat();
        } catch (const std::exception &e) {
            pool_logger->error("Exception in deallocate: {}", e.what());
        } catch (...) {
            pool_logger->error("Unknown exception in deallocate");
        }
    }
}

void MemoryPool::clear() {
    try {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        size_t count = pool_.size();
        pool_.clear();
        pool_logger->info("Memory pool cleared, {} matrices released", count);
    } catch (const std::exception &e) {
        pool_logger->error("Exception in clear: {}", e.what());
    } catch (...) {
        pool_logger->error("Unknown exception in clear");
    }
}