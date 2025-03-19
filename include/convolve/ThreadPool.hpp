// ThreadPool.hpp
#pragma once

#include <future>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <vector>
#include <expected>
#include "ProcessError.hpp"

/**
 * @class DynamicThreadPool
 * @brief 用于并行执行任务的线程池
 */
class DynamicThreadPool {
public:
  /**
   * @brief 构造函数
   * @param num_threads 线程池中的线程数量
   */
  explicit DynamicThreadPool(size_t num_threads);

  /**
   * @brief 析构函数，终止所有线程
   */
  ~DynamicThreadPool();

  /**
   * @brief 将任务添加到队列并获取future结果
   */
  template<class F, class... Args>
  auto enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::invoke_result<F, Args...>::type>;

private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_;
};