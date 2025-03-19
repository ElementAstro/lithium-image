#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <shared_mutex>
#include <unordered_map>


// Thread-local cache for optimization
class ImageCache {
private:
  std::unordered_map<size_t, cv::Mat> cache;
  std::shared_mutex mutex;
  size_t max_size;
  std::atomic<size_t> current_size{0};

  // Hash function for cv::Mat
  static size_t hashMat(const cv::Mat &mat) {
    size_t hash = 0;
    auto dataPtr = mat.data;
    auto dataSize = mat.total() * mat.elemSize();

    for (size_t i = 0; i < dataSize; i += sizeof(size_t)) {
      size_t value = 0;
      std::memcpy(&value, dataPtr + i, std::min(sizeof(size_t), dataSize - i));
      hash ^= value + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }

public:
  explicit ImageCache(size_t max_size_mb)
      : max_size(max_size_mb * 1024 * 1024) {}

  std::optional<cv::Mat> get(const cv::Mat &key) {
    size_t hash = hashMat(key);
    std::shared_lock lock(mutex);
    auto it = cache.find(hash);
    if (it != cache.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  void put(const cv::Mat &key, const cv::Mat &value) {
    size_t newSize = value.total() * value.elemSize();
    if (newSize > max_size)
      return;

    size_t hash = hashMat(key);
    std::unique_lock lock(mutex);

    // Make space if needed
    while (current_size + newSize > max_size && !cache.empty()) {
      auto it = cache.begin();
      current_size -= it->second.total() * it->second.elemSize();
      cache.erase(it);
    }

    cache[hash] = value.clone();
    current_size += newSize;
  }

  void clear() {
    std::unique_lock lock(mutex);
    cache.clear();
    current_size = 0;
  }
};

// Singleton image cache
std::unique_ptr<ImageCache> getGlobalCache(size_t size_mb) {
  static std::mutex mutex;
  static std::weak_ptr<ImageCache> weakCache;

  std::lock_guard lock(mutex);
  auto cache = weakCache.lock();

  if (!cache) {
    cache = std::make_shared<ImageCache>(size_mb);
    weakCache = cache;
  }

  return std::make_unique<ImageCache>(size_mb);
}