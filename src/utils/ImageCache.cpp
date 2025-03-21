#include "utils/ImageCache.hpp"

#include <algorithm>
#include <limits>


// Hash function implementation for cv::Mat
size_t ImageCache::hashMat(const cv::Mat &mat) {
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

ImageCache::ImageCache(size_t max_size_mb, EvictionPolicy policy)
    : max_size(max_size_mb * 1024 * 1024), policy(policy) {}

std::optional<cv::Mat> ImageCache::get(const cv::Mat &key) {
  size_t hash = hashMat(key);
  std::shared_lock lock(mutex);

  auto it = cache.find(hash);
  if (it != cache.end()) {
    // Update access metadata for hit
    auto now = std::chrono::steady_clock::now();
    metadata[hash].lastAccess = now;
    metadata[hash].accessCount++;
    hits++;

    return it->second;
  }

  misses++;
  return std::nullopt;
}

void ImageCache::makeSpace(size_t requiredSize) {
  // Skip if we already have enough space
  if (current_size + requiredSize <= max_size) {
    return;
  }

  while (current_size + requiredSize > max_size && !cache.empty()) {
    // Choose entry to remove based on policy
    size_t hashToRemove = 0;

    if (policy == EvictionPolicy::LRU) {
      // Find least recently used entry
      auto oldestTime = std::chrono::steady_clock::now();
      for (const auto &entry : metadata) {
        if (entry.second.lastAccess < oldestTime) {
          oldestTime = entry.second.lastAccess;
          hashToRemove = entry.first;
        }
      }
    } else if (policy == EvictionPolicy::LFU) {
      // Find least frequently used entry
      unsigned int lowestCount = std::numeric_limits<unsigned int>::max();
      for (const auto &entry : metadata) {
        if (entry.second.accessCount < lowestCount) {
          lowestCount = entry.second.accessCount;
          hashToRemove = entry.first;
        }
      }
    } else { // FIFO - just take the first entry in the map
      hashToRemove = cache.begin()->first;
    }

    // Remove the selected entry
    current_size -= metadata[hashToRemove].size;
    cache.erase(hashToRemove);
    metadata.erase(hashToRemove);
  }
}

void ImageCache::put(const cv::Mat &key, const cv::Mat &value) {
  size_t newSize = value.total() * value.elemSize();
  if (newSize > max_size)
    return; // Skip if image is larger than entire cache

  size_t hash = hashMat(key);
  std::unique_lock lock(mutex);

  // Check if this key already exists
  auto it = cache.find(hash);
  if (it != cache.end()) {
    // Update existing entry
    current_size -= metadata[hash].size;
    current_size += newSize;

    cache[hash] = value.clone();
    metadata[hash].size = newSize;
    metadata[hash].lastAccess = std::chrono::steady_clock::now();
    metadata[hash].accessCount++;
    return;
  }

  // Make space if needed
  makeSpace(newSize);

  // Store image and metadata
  cache[hash] = value.clone();
  metadata[hash] = {
      std::chrono::steady_clock::now(), newSize,
      1 // Initial access count
  };

  current_size += newSize;
}

void ImageCache::clear() {
  std::unique_lock lock(mutex);
  cache.clear();
  metadata.clear();
  current_size = 0;
  hits = 0;
  misses = 0;
}

void ImageCache::setEvictionPolicy(EvictionPolicy newPolicy) {
  std::unique_lock lock(mutex);
  policy = newPolicy;
}

double ImageCache::getHitRate() const {
  unsigned int totalAccesses = hits + misses;
  if (totalAccesses == 0)
    return 0.0;
  return static_cast<double>(hits) / totalAccesses;
}

size_t ImageCache::getCurrentSize() const { return current_size; }

size_t ImageCache::getMaxSize() const { return max_size; }

void ImageCache::setMaxSize(size_t size_mb) {
  std::unique_lock lock(mutex);
  max_size = size_mb * 1024 * 1024;

  // If new size is smaller than current usage, evict entries
  if (current_size > max_size) {
    makeSpace(0);
  }
}

// Global cache implementation
std::shared_ptr<ImageCache> getGlobalCache(size_t size_mb) {
  static std::mutex mutex;
  static std::weak_ptr<ImageCache> weakCache;

  std::lock_guard<std::mutex> lock(mutex);
  auto cache = weakCache.lock();

  if (!cache) {
    cache = std::make_shared<ImageCache>(size_mb);
    weakCache = cache;
  }

  return cache;
}