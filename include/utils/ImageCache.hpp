#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <shared_mutex>
#include <unordered_map>


/**
 * Thread-safe image cache for storing processed OpenCV images.
 * Uses content-based hashing for lookups and implements various eviction
 * policies.
 */
class ImageCache {
private:
  // Internal storage for cached images
  std::unordered_map<size_t, cv::Mat> cache;

  // Cache entry metadata for statistics and eviction
  struct CacheMetadata {
    std::chrono::steady_clock::time_point lastAccess;
    size_t size;
    unsigned int accessCount;
  };

  std::unordered_map<size_t, CacheMetadata> metadata;

  // Thread synchronization
  std::shared_mutex mutex;

  // Cache configuration
  size_t max_size;
  std::atomic<size_t> current_size{0};
  std::atomic<unsigned int> hits{0};
  std::atomic<unsigned int> misses{0};

  // Cache eviction policy
  enum class EvictionPolicy {
    LRU, // Least Recently Used
    LFU, // Least Frequently Used
    FIFO // First In First Out
  };
  EvictionPolicy policy;

  // Hash function for cv::Mat
  static size_t hashMat(const cv::Mat &mat);

  // Internal method to free space in cache
  void makeSpace(size_t requiredSize);

public:
  /**
   * Creates an image cache with specified maximum size in MB
   * @param max_size_mb Maximum cache size in megabytes
   * @param policy Eviction policy to use
   */
  explicit ImageCache(size_t max_size_mb,
                      EvictionPolicy policy = EvictionPolicy::LRU);

  /**
   * Retrieves a cached image if available
   * @param key Source image to use as lookup key
   * @return Cached image if found, std::nullopt otherwise
   */
  std::optional<cv::Mat> get(const cv::Mat &key);

  /**
   * Stores an image in the cache
   * @param key Source image to use as lookup key
   * @param value Image to store in cache
   */
  void put(const cv::Mat &key, const cv::Mat &value);

  /**
   * Clears all cached images
   */
  void clear();

  /**
   * Sets the eviction policy
   * @param newPolicy The policy to use for cache eviction
   */
  void setEvictionPolicy(EvictionPolicy newPolicy);

  /**
   * Returns cache hit rate statistics
   * @return Cache hit rate (0.0 to 1.0)
   */
  double getHitRate() const;

  /**
   * Returns current cache memory usage
   * @return Current size in bytes
   */
  size_t getCurrentSize() const;

  /**
   * Returns maximum cache size
   * @return Maximum size in bytes
   */
  size_t getMaxSize() const;

  /**
   * Sets new maximum cache size
   * @param size_mb New maximum size in megabytes
   */
  void setMaxSize(size_t size_mb);
};

/**
 * Gets or creates the global image cache singleton
 * @param size_mb Cache size in megabytes
 * @return Shared pointer to the global cache
 */
std::shared_ptr<ImageCache> getGlobalCache(size_t size_mb = 100);
