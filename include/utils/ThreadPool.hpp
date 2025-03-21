#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <coroutine>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <vector>

// Concept for invocable tasks
template <typename F, typename... Args>
concept TaskInvocable = requires(F &&f, Args &&...args) {
  { std::invoke(std::forward<F>(f), std::forward<Args>(args)...) };
};

class TaskGroup;

class DynamicThreadPool {
public:
  enum class Priority { High, Normal, Low };

  explicit DynamicThreadPool(
      int min_threads = std::thread::hardware_concurrency(),
      int max_threads = 4 * std::thread::hardware_concurrency(),
      std::chrono::milliseconds idle_timeout = std::chrono::seconds(30));

  ~DynamicThreadPool();

  // Improved template signature with concepts
  template <TaskInvocable F, typename... Args>
  auto enqueue(F &&func, Args &&...args)
      -> std::future<std::invoke_result_t<F, Args...>>;

  // Improved template signature with concepts
  template <TaskInvocable F, typename... Args>
  auto enqueueWithPriority(Priority priority, F &&func, Args &&...args)
      -> std::future<std::invoke_result_t<F, Args...>>;

  // Delayed task scheduling
  template <TaskInvocable F, typename... Args>
  auto enqueueDelayed(std::chrono::milliseconds delay, F &&func, Args &&...args)
      -> std::future<std::invoke_result_t<F, Args...>>;

  struct TaskHandle {
    std::shared_ptr<std::atomic<bool>> cancelled =
        std::make_shared<std::atomic<bool>>(false);
    std::shared_ptr<std::promise<void>> cancel_promise;

    // Move constructors and assignment
    TaskHandle() = default;
    TaskHandle(TaskHandle &&) noexcept = default;
    TaskHandle &operator=(TaskHandle &&) noexcept = default;
    // Delete copy
    TaskHandle(const TaskHandle &) = delete;
    TaskHandle &operator=(const TaskHandle &) = delete;

    // Cancel the task
    void cancel() {
      cancelled->store(true);
      if (cancel_promise) {
        try {
          cancel_promise->set_value();
        } catch (const std::future_error &) {
          // Promise might already be satisfied
        }
      }
    }
  };

  template <TaskInvocable F, typename... Args>
  std::pair<TaskHandle, std::future<std::invoke_result_t<F, Args...>>>
  enqueueCancelable(F &&func, Args &&...args);

  void waitAll();
  void cancelAll();

  // Configuration methods
  void setMinThreads(int min);
  void setMaxThreads(int max);
  void setIdleTimeout(std::chrono::milliseconds timeout);

  // Status queries
  int activeThreads() const { return active_workers_.load(); }
  int pendingTasks() const;
  int currentThreadCount() const { return current_threads_.load(); }

  // C++20 coroutine support
  struct Task {
    struct promise_type {
      std::suspend_never initial_suspend() noexcept { return {}; }
      std::suspend_never final_suspend() noexcept { return {}; }
      void return_void() {}
      Task get_return_object() { return {}; }
      void unhandled_exception() { std::terminate(); }
    };
  };

  template <typename F, typename... Args>
    requires std::invocable<F, Args...>
  Task schedule(F &&func, Args &&...args);

  // Task group support
  std::shared_ptr<TaskGroup> create_task_group();

  template <TaskInvocable F, typename... Args>
  auto enqueue_in_group(std::shared_ptr<TaskGroup> group, F &&func,
                        Args &&...args)
      -> std::future<std::invoke_result_t<F, Args...>>;

  // Task with timeout support
  template <TaskInvocable F, typename... Args>
  auto enqueue_with_timeout(std::chrono::milliseconds timeout, F &&func,
                            Args &&...args)
      -> std::future<std::invoke_result_t<F, Args...>>;

  // Thread affinity control
  void set_thread_affinity(bool enable);

  // Performance metrics
  struct PerformanceMetrics {
    std::atomic<uint64_t> tasks_completed{0};
    std::atomic<uint64_t> tasks_failed{0};
    std::atomic<uint64_t> tasks_cancelled{0};
    std::chrono::nanoseconds total_execution_time{0};
    std::atomic<uint64_t> task_queue_max_size{0};
  };

  const PerformanceMetrics &get_metrics() const { return metrics_; }
  void reset_metrics();

  // 使用回调函数替代Qt信号
  using TaskCallback = std::function<void(uint64_t)>;
  using TaskErrorCallback = std::function<void(uint64_t, const std::string &)>;
  using ThreadCountCallback = std::function<void(int)>;

  void setTaskStartedCallback(TaskCallback cb) {
    taskStartedCb = std::move(cb);
  }
  void setTaskFinishedCallback(TaskCallback cb) {
    taskFinishedCb = std::move(cb);
  }
  void setTaskFailedCallback(TaskErrorCallback cb) {
    taskFailedCb = std::move(cb);
  }
  void setTaskCancelledCallback(TaskCallback cb) {
    taskCancelledCb = std::move(cb);
  }
  void setThreadCountChangedCallback(ThreadCountCallback cb) {
    threadCountChangedCb = std::move(cb);
  }

private:
  struct PrioritizedTask {
    std::packaged_task<void()> task;
    Priority priority;
    uint64_t task_id;
    std::chrono::steady_clock::time_point enqueue_time;
    std::weak_ptr<TaskHandle> handle;

    PrioritizedTask(std::packaged_task<void()> &&t, Priority p, uint64_t id,
                    std::chrono::steady_clock::time_point time,
                    std::weak_ptr<TaskHandle> h)
        : task(std::move(t)), priority(p), task_id(id), enqueue_time(time),
          handle(std::move(h)) {}

    PrioritizedTask(PrioritizedTask &&other) noexcept = default;
    PrioritizedTask &operator=(PrioritizedTask &&other) noexcept = default;

    // Delete copy
    PrioritizedTask(const PrioritizedTask &) = delete;
    PrioritizedTask &operator=(const PrioritizedTask &) = delete;

    bool operator<(const PrioritizedTask &other) const {
      if (priority == other.priority) {
        return enqueue_time > other.enqueue_time; // earlier time has priority
      }
      return priority < other.priority;
    }
  };

  void workerRoutine(std::stop_token st);
  void managerRoutine();
  void expandIfNeeded();
  void shrinkIfPossible();
  void processDelayedTasks();
  bool stealTask(PrioritizedTask &stolen_task);

  // SIMD helper for batch processing when possible
  template <typename Container, typename Func>
  void processBatchSIMD(Container &container, Func &&func);

  // Configuration parameters (atomic for dynamic modification)
  std::atomic<int> min_threads_;
  std::atomic<int> max_threads_;
  std::atomic<std::chrono::milliseconds> idle_timeout_;

  // Synchronization primitives
  mutable std::shared_mutex queue_mutex_;
  std::condition_variable_any queue_cv_;
  std::mutex control_mutex_;
  std::mutex delay_mutex_;

  // Status management
  std::atomic<int> current_threads_{0};
  std::atomic<int> active_workers_{0};
  std::atomic<bool> shutdown_{false};
  std::atomic<uint64_t> next_task_id_{1};

  // Task queues - improved with better data structure
  using TaskQueue = std::priority_queue<PrioritizedTask>;
  std::unique_ptr<TaskQueue> task_queue_;
  std::unordered_map<uint64_t, PrioritizedTask> delayed_queue_;
  std::vector<std::jthread> workers_;
  std::jthread manager_thread_;

  // Work stealing related
  static constexpr size_t WORK_STEAL_ATTEMPTS = 2;
  std::atomic<size_t> steal_index_{0};

  // Performance tracking
  PerformanceMetrics metrics_;
  bool enable_thread_affinity_ = false;

  // Set thread affinity for worker threads
  void set_worker_affinity(std::thread::native_handle_type handle,
                           size_t thread_index);

  // Enhanced work stealing using work-first principle
  bool try_steal_batch(std::vector<PrioritizedTask> &target,
                       size_t max_tasks = 3);

  // Track task execution time
  void track_task_execution(std::chrono::nanoseconds execution_time);

  // Task groups tracking
  std::unordered_map<uint64_t, std::weak_ptr<TaskGroup>> task_groups_;
  std::mutex task_groups_mutex_;

  // 回调函数
  TaskCallback taskStartedCb;
  TaskCallback taskFinishedCb;
  TaskErrorCallback taskFailedCb;
  TaskCallback taskCancelledCb;
  ThreadCountCallback threadCountChangedCb;

  // 替换 QMetaObject::invokeMethod
  template <typename F> void postCallback(F &&f) {
    enqueue(std::forward<F>(f));
  }
};

template <TaskInvocable F, typename... Args>
auto DynamicThreadPool::enqueue(F &&func, Args &&...args)
    -> std::future<std::invoke_result_t<F, Args...>> {
  return enqueueWithPriority(Priority::Normal, std::forward<F>(func),
                             std::forward<Args>(args)...);
}

template <TaskInvocable F, typename... Args>
auto DynamicThreadPool::enqueueWithPriority(Priority priority, F &&func,
                                            Args &&...args)
    -> std::future<std::invoke_result_t<F, Args...>> {
  using ReturnType = std::invoke_result_t<F, Args...>;

  // Input validation
  if (shutdown_.load()) {
    throw std::runtime_error("Cannot enqueue tasks after shutdown");
  }

  auto task = std::packaged_task<ReturnType()>(
      std::bind(std::forward<F>(func), std::forward<Args>(args)...));
  auto future = task.get_future();
  const auto task_id = next_task_id_.fetch_add(1);

  try {
    std::unique_lock lock(queue_mutex_);
    if (!task_queue_) {
      task_queue_ = std::make_unique<TaskQueue>();
    }
    task_queue_->push(PrioritizedTask(
        std::packaged_task<void()>(std::move(task)), priority, task_id,
        std::chrono::steady_clock::now(), std::weak_ptr<TaskHandle>()));
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("Failed to enqueue task: ") +
                             e.what());
  }

  postCallback([this, task_id] {
    if (taskStartedCb)
      taskStartedCb(task_id);
  });
  queue_cv_.notify_one();
  return future;
}

template <TaskInvocable F, typename... Args>
auto DynamicThreadPool::enqueueDelayed(std::chrono::milliseconds delay,
                                       F &&func, Args &&...args)
    -> std::future<std::invoke_result_t<F, Args...>> {
  using ReturnType = std::invoke_result_t<F, Args...>;

  // Input validation
  if (shutdown_.load()) {
    throw std::runtime_error("Cannot enqueue tasks after shutdown");
  }
  if (delay < std::chrono::milliseconds(0)) {
    throw std::invalid_argument("Delay cannot be negative");
  }

  auto task = std::packaged_task<ReturnType()>(
      std::bind(std::forward<F>(func), std::forward<Args>(args)...));
  auto future = task.get_future();
  const auto task_id = next_task_id_.fetch_add(1);
  auto execution_time = std::chrono::steady_clock::now() + delay;

  try {
    std::lock_guard<std::mutex> lock(delay_mutex_);
    delayed_queue_.emplace(
        task_id, PrioritizedTask(std::packaged_task<void()>(std::move(task)),
                                 Priority::Normal, task_id, execution_time,
                                 std::weak_ptr<TaskHandle>()));
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("Failed to enqueue delayed task: ") +
                             e.what());
  }

  postCallback([this, task_id] {
    if (taskStartedCb)
      taskStartedCb(task_id);
  });
  return future;
}

template <TaskInvocable F, typename... Args>
std::pair<DynamicThreadPool::TaskHandle,
          std::future<std::invoke_result_t<F, Args...>>>
DynamicThreadPool::enqueueCancelable(F &&func, Args &&...args) {
  using ReturnType = std::invoke_result_t<F, Args...>;

  // Input validation
  if (shutdown_.load()) {
    throw std::runtime_error("Cannot enqueue tasks after shutdown");
  }

  auto handle_ptr = std::make_shared<TaskHandle>();
  TaskHandle handle;
  handle.cancelled = handle_ptr->cancelled;
  auto promise = std::make_shared<std::promise<void>>();
  handle.cancel_promise = promise;
  handle_ptr->cancel_promise = promise;

  auto task = std::packaged_task<ReturnType()>(
      [func = std::forward<F>(func),
       args = std::make_tuple(std::forward<Args>(args)...),
       handle = handle_ptr]() mutable {
        if (handle->cancelled->load()) {
          throw std::runtime_error("Task cancelled");
        }
        return std::apply(func, args);
      });

  auto future = task.get_future();
  const auto task_id = next_task_id_.fetch_add(1);

  try {
    std::unique_lock lock(queue_mutex_);
    if (!task_queue_) {
      task_queue_ = std::make_unique<TaskQueue>();
    }
    task_queue_->push(PrioritizedTask(
        std::packaged_task<void()>(std::move(task)), Priority::Normal, task_id,
        std::chrono::steady_clock::now(), handle_ptr));
  } catch (const std::exception &e) {
    throw std::runtime_error(
        std::string("Failed to enqueue cancelable task: ") + e.what());
  }

  postCallback([this, task_id] {
    if (taskStartedCb)
      taskStartedCb(task_id);
  });
  queue_cv_.notify_one();
  return {std::move(handle), std::move(future)};
}

// C++20 coroutine support implementation
template <typename F, typename... Args>
  requires std::invocable<F, Args...>
DynamicThreadPool::Task DynamicThreadPool::schedule(F &&func, Args &&...args) {
  enqueue(std::forward<F>(func), std::forward<Args>(args)...);
  co_return;
}

// SIMD batch processing helper
template <typename Container, typename Func>
void DynamicThreadPool::processBatchSIMD(Container &container, Func &&func) {
  // Implementation would depend on specific SIMD instruction sets
  // This is a placeholder for the actual SIMD implementation
  if constexpr (std::is_arithmetic_v<typename Container::value_type>) {
    // For arithmetic types, we can use SIMD
    constexpr size_t batch_size = 4; // Adjust based on SIMD width
    for (size_t i = 0; i < container.size(); i += batch_size) {
      // Process elements in batches
      auto end = std::min(i + batch_size, container.size());
      for (size_t j = i; j < end; ++j) {
        func(container[j]);
      }
    }
  } else {
    // For non-arithmetic types, fall back to standard processing
    for (auto &item : container) {
      func(item);
    }
  }
}

// Task grouping support
class TaskGroup {
  friend class DynamicThreadPool;

private:
  std::atomic<size_t> pending_tasks_{0};
  std::promise<void> completion_promise_;
  std::shared_ptr<std::atomic<bool>> cancelled_ =
      std::make_shared<std::atomic<bool>>(false);

public:
  std::future<void> get_future() { return completion_promise_.get_future(); }
  void cancel() { cancelled_->store(true); }
  bool is_cancelled() const { return cancelled_->load(); }
};

template <TaskInvocable F, typename... Args>
auto DynamicThreadPool::enqueue_in_group(std::shared_ptr<TaskGroup> group,
                                         F &&func, Args &&...args)
    -> std::future<std::invoke_result_t<F, Args...>> {
  using ReturnType = std::invoke_result_t<F, Args...>;

  if (!group) {
    throw std::invalid_argument("Task group cannot be null");
  }

  if (shutdown_.load()) {
    throw std::runtime_error("Cannot enqueue tasks after shutdown");
  }

  // Increment pending tasks counter
  group->pending_tasks_++;

  // Create wrapper function that decrements counter on completion
  auto wrapper = [func = std::forward<F>(func),
                  args = std::make_tuple(std::forward<Args>(args)...), group,
                  start_time = std::chrono::steady_clock::now()]() mutable {
    // Check cancellation
    if (group->is_cancelled()) {
      throw std::runtime_error("Task group cancelled");
    }

    // Execute actual function
    if constexpr (std::is_void_v<ReturnType>) {
      std::apply(func, args);

      // Decrement counter and check if all tasks completed
      if (--group->pending_tasks_ == 0) {
        group->completion_promise_.set_value();
      }

      return; // void return type
    } else {
      auto result = std::apply(func, args);

      // Decrement counter and check if all tasks completed
      if (--group->pending_tasks_ == 0) {
        group->completion_promise_.set_value();
      }

      return result;
    }
  };

  // Enqueue the wrapped task
  return enqueue(std::move(wrapper));
}

template <TaskInvocable F, typename... Args>
auto DynamicThreadPool::enqueue_with_timeout(std::chrono::milliseconds timeout,
                                             F &&func, Args &&...args)
    -> std::future<std::invoke_result_t<F, Args...>> {
  using ReturnType = std::invoke_result_t<F, Args...>;

  if (shutdown_.load()) {
    throw std::runtime_error("Cannot enqueue tasks after shutdown");
  }

  if (timeout < std::chrono::milliseconds(0)) {
    throw std::invalid_argument("Timeout cannot be negative");
  }

  // Create a shared state to track timeout
  struct SharedState {
    std::mutex mutex;
    std::atomic<bool> timed_out{false};
    std::atomic<bool> started{false};
  };

  auto state = std::make_shared<SharedState>();

  // Create wrapper with timeout logic
  auto wrapper = [func = std::forward<F>(func),
                  args = std::make_tuple(std::forward<Args>(args)...), state,
                  timeout]() mutable -> ReturnType {
    // Mark as started
    state->started.store(true);

    // Check if already timed out
    if (state->timed_out.load()) {
      throw std::runtime_error("Task timed out before execution");
    }

    // Execute with timeout monitoring
    if constexpr (std::is_void_v<ReturnType>) {
      std::apply(func, args);
      return;
    } else {
      return std::apply(func, args);
    }
  };

  // Enqueue the task
  auto future = enqueue(std::move(wrapper));

  // Create and enqueue the timeout task
  auto timeout_task = [state, timeout]() {
    std::this_thread::sleep_for(timeout);

    // Only mark as timed out if the task hasn't started yet
    if (!state->started.load()) {
      state->timed_out.store(true);
    }
  };

  // Enqueue timeout task with low priority
  enqueueWithPriority(Priority::Low, std::move(timeout_task));

  return future;
}
