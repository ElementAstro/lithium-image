#pragma once

#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <utility>

/**
 * @brief Defines the size of a cache line in bytes.
 */
constexpr size_t CACHE_LINE_SIZE = 64;

/**
 * @brief Defines the default number of blocks per chunk in the memory pool.
 */
constexpr size_t DEFAULT_BLOCKS_PER_CHUNK = 8192;

/**
 * @brief Calculates the aligned size of a type `T` to the nearest multiple of
 * `std::max_align_t`.
 *
 * This ensures that objects of type `T` are properly aligned in memory.
 *
 * @tparam T The type to calculate the aligned size for.
 */
template <typename T>
static constexpr size_t aligned_size =
    ((sizeof(T) + alignof(std::max_align_t) - 1) &
     ~(alignof(std::max_align_t) - 1));

/**
 * @brief Concept that checks if a type `T` is suitable for use with the memory
 * pool.
 *
 * Requires that `T` is trivially destructible and that its size is at least
 * the size of an `int`.
 *
 * @tparam T The type to check.
 */
template <typename T>
concept ValidPoolType = requires {
  requires std::is_trivially_destructible_v<T>;
  requires sizeof(T) >= sizeof(int);
};

/**
 * @brief A simple memory pool implementation for fixed-size blocks.
 *
 * This class provides a fast and efficient way to allocate and deallocate
 * memory blocks of a fixed size. It is designed to reduce memory fragmentation
 * and improve performance compared to using `new` and `delete` directly.
 *
 * @tparam BlockSize The size of each memory block in bytes.
 * @tparam BlocksPerChunk The number of blocks to allocate in each chunk.
 */
template <size_t BlockSize, size_t BlocksPerChunk = DEFAULT_BLOCKS_PER_CHUNK>
class MemoryPool {
public:
  /**
   * @brief Default constructor.
   */
  MemoryPool() noexcept = default;

  /**
   * @brief Deleted copy constructor.
   */
  MemoryPool(const MemoryPool &) = delete;

  /**
   * @brief Deleted copy assignment operator.
   */
  MemoryPool &operator=(const MemoryPool &) = delete;

  /**
   * @brief Move constructor.
   *
   * Moves the internal state of `other` to this object, leaving `other` in a
   * valid but empty state.
   *
   * @param other The `MemoryPool` to move from.
   */
  MemoryPool(MemoryPool &&other) noexcept {
    std::scoped_lock lock(other.mutex);
    free_list = std::exchange(other.free_list, nullptr);
    chunks = std::exchange(other.chunks, nullptr);
  }

  /**
   * @brief Destructor.
   *
   * Deallocates all memory chunks managed by the pool.
   */
  ~MemoryPool() {
    while (chunks) {
      Chunk *next = chunks->next;
      std::destroy_at(chunks);
      ::operator delete(chunks);
      chunks = next;
    }
  }

  /**
   * @brief Allocates a memory block from the pool.
   *
   * @return A pointer to the allocated memory block, or `nullptr` if allocation
   * fails.
   */
  [[nodiscard]] void *allocate() {
    std::scoped_lock lock(mutex);

    if (!free_list) {
      allocate_chunk();
    }

    Block *block = free_list;
    free_list = block->next;
    return block;
  }

  /**
   * @brief Deallocates a memory block back to the pool.
   *
   * @param ptr A pointer to the memory block to deallocate.
   */
  void deallocate(void *ptr) noexcept {
    if (!ptr)
      return;

    std::scoped_lock lock(mutex);
    Block *block = static_cast<Block *>(ptr);
    block->next = free_list;
    free_list = block;
  }

  /**
   * @brief Structure for holding memory pool statistics.
   */
  struct Stats {
    size_t total_chunks; ///< The total number of chunks allocated.
    size_t total_blocks; ///< The total number of blocks allocated.
    size_t free_blocks;  ///< The number of free blocks in the pool.
  };

  /**
   * @brief Gets statistics about the memory pool.
   *
   * @return A `Stats` struct containing information about the pool's memory
   * usage.
   */
  [[nodiscard]] Stats get_stats() const noexcept {
    std::scoped_lock lock(mutex);

    Stats stats{0, 0, 0};
    Chunk *current = chunks;
    while (current) {
      ++stats.total_chunks;
      current = current->next;
    }

    stats.total_blocks = stats.total_chunks * BlocksPerChunk;

    Block *current_free = free_list;
    while (current_free) {
      ++stats.free_blocks;
      current_free = current_free->next;
    }

    return stats;
  }

#ifdef MEMORY_POOL_DEBUG
  /**
   * @brief Dumps memory pool statistics to `std::cout`.
   *
   * This function is only available when `MEMORY_POOL_DEBUG` is defined.
   */
  void dump_stats() const {
    auto stats = get_stats();
    std::cout << "Memory Pool Stats:\n"
              << "Total chunks: " << stats.total_chunks << '\n'
              << "Total blocks: " << stats.total_blocks << '\n'
              << "Free blocks: " << stats.free_blocks << '\n'
              << "Block size: " << BlockSize << '\n'
              << "Blocks per chunk: " << BlocksPerChunk << '\n';
  }
#endif

private:
  /**
   * @brief Represents a memory block in the pool.
   *
   * Each block contains a pointer to the next free block (when the block is
   * free) or user data (when the block is allocated).
   */
  struct alignas(std::max_align_t) Block {
    union {
      Block *next;               ///< Pointer to the next free block.
      std::byte data[BlockSize]; ///< User data.
    };
  };

  /**
   * @brief Represents a chunk of memory in the pool.
   *
   * Each chunk contains an array of blocks and a pointer to the next chunk.
   */
  struct Chunk {
    std::array<Block, BlocksPerChunk> blocks; ///< Array of memory blocks.
    Chunk *next = nullptr;                    ///< Pointer to the next chunk.
  };

  /**
   * @brief Allocates a new chunk of memory and adds it to the pool.
   *
   * This function allocates a new `Chunk` and initializes the free list with
   * the blocks in the chunk.
   */
  void allocate_chunk() {
    Chunk *new_chunk = new Chunk();

    // Initialize free list
    for (size_t i = 0; i < BlocksPerChunk - 1; ++i) {
      new_chunk->blocks[i].next = &new_chunk->blocks[i + 1];
    }
    new_chunk->blocks[BlocksPerChunk - 1].next = free_list;

    free_list = &new_chunk->blocks[0];
    new_chunk->next = chunks;
    chunks = new_chunk;
  }

  Block *free_list = nullptr; ///< Pointer to the first free block in the pool.
  Chunk *chunks = nullptr;    ///< Pointer to the first chunk in the pool.
  mutable std::mutex mutex;   ///< Mutex to protect the pool from concurrent
                              ///< access.
};

/**
 * @brief A pool allocator adapter for STL containers.
 *
 * This class allows you to use the `MemoryPool` with STL containers like
 * `std::vector` and `std::list`.
 *
 * @tparam T The type of object to allocate.
 * @tparam BlocksPerChunk The number of blocks per chunk in the underlying
 * `MemoryPool`.
 */
template <typename T, size_t BlocksPerChunk = DEFAULT_BLOCKS_PER_CHUNK>
class PoolAllocator {
public:
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using propagate_on_container_move_assignment = std::true_type;

  template <typename U> struct rebind {
    using other = PoolAllocator<U, BlocksPerChunk>;
  };

  /**
   * @brief Default constructor.
   */
  PoolAllocator() noexcept = default;

  /**
   * @brief Template constructor to allow construction from allocators of other
   * types.
   *
   * @tparam U The type of object the other allocator allocates.
   * @param other The other allocator.
   */
  template <typename U> PoolAllocator(const PoolAllocator<U> &) noexcept {}

  /**
   * @brief Allocates memory for `n` objects of type `T`.
   *
   * @param n The number of objects to allocate memory for.
   * @return A pointer to the allocated memory.
   * @throws std::bad_alloc If the allocation fails.
   */
  [[nodiscard]] T *allocate(size_t n) {
    // 修改为支持多个对象的分配
    if (n > BlocksPerChunk) {
      throw std::bad_alloc();
    }

    // 为n个对象分配连续内存
    T *result = nullptr;
    void *memory = nullptr;

    try {
      memory = ::operator new(n * sizeof(T));
      result = static_cast<T *>(memory);
    } catch (...) {
      throw std::bad_alloc();
    }

    return result;
  }

  /**
   * @brief Deallocates memory pointed to by `p`.
   *
   * @param p The pointer to the memory to deallocate.
   * @param n The number of objects that were allocated (unused).
   */
  void deallocate(T *p, [[maybe_unused]] size_t n) noexcept {
    // 修改为支持多个对象的释放
    if (p) {
      ::operator delete(p);
    }
  }

  /**
   * @brief Equality operator.
   *
   * All `PoolAllocator` objects are considered equal.
   *
   * @tparam U The type of object the other allocator allocates.
   * @param other The other allocator.
   * @return `true`.
   */
  template <typename U>
  bool operator==(const PoolAllocator<U> &) const noexcept {
    return true;
  }

  /**
   * @brief Inequality operator.
   *
   * All `PoolAllocator` objects are considered equal.
   *
   * @tparam U The type of object the other allocator allocates.
   * @param other The other allocator.
   * @return `false`.
   */
  template <typename U>
  bool operator!=(const PoolAllocator<U> &) const noexcept {
    return false;
  }

private:
  static MemoryPool<sizeof(T), BlocksPerChunk> pool;
};

template <typename T, size_t B>
MemoryPool<sizeof(T), B> PoolAllocator<T, B>::pool;