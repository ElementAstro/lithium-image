#pragma once

#include <atomic>
#include <concepts>
#include <coroutine>
#include <functional>
#include <future>
#include <mutex>

/**
 * @brief 表示异步操作的进度和结果，替代QPromise
 */
template <typename T> class Promise {
public:
  Promise() : isCancelled_(false), progressValue_(0) {}

  void setProgressValue(int value) {
    progressValue_ = value;
    if (progressCallback_) {
      progressCallback_(value);
    }
  }

  bool isCanceled() const { return isCancelled_; }

  void cancel() { isCancelled_ = true; }

  void setProgressCallback(std::function<void(int)> callback) {
    progressCallback_ = std::move(callback);
  }

  // 设置结果并通知等待者
  void setResult(T result) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      result_ = std::move(result);
    }
    resultReady_.set_value();
  }

  // 获取异步操作的Future对象
  std::future<T> future() { return resultReady_.get_future(); }

private:
  std::atomic<bool> isCancelled_;
  std::atomic<int> progressValue_;
  std::function<void(int)> progressCallback_;
  std::promise<T> resultReady_;
  T result_;
  std::mutex mutex_;
};

/**
 * @brief 替代QFuture的Future类
 */
template <typename T> class Future {
public:
  Future(std::future<T> future) : future_(std::move(future)) {}

  T result() { return future_.get(); }

  bool isRunning() const {
    return future_.valid() && future_.wait_for(std::chrono::seconds(0)) !=
                                  std::future_status::ready;
  }

  bool isFinished() const {
    return future_.valid() && future_.wait_for(std::chrono::seconds(0)) ==
                                  std::future_status::ready;
  }

  void wait() { future_.wait(); }

private:
  std::future<T> future_;
};

/**
 * @brief 协程任务模板，替代原始Task模板
 */
template <typename T> struct Task {
  struct promise_type {
    T result;
    std::exception_ptr exception;

    Task get_return_object() noexcept {
      return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
    }

    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }

    void unhandled_exception() noexcept {
      exception = std::current_exception();
    }

    template <std::convertible_to<T> U> void return_value(U &&value) noexcept {
      result = std::forward<U>(value);
    }
  };

  std::coroutine_handle<promise_type> handle;

  Task(std::coroutine_handle<promise_type> h) : handle(h) {}
  Task(Task &&t) noexcept : handle(t.handle) { t.handle = nullptr; }
  ~Task() {
    if (handle)
      handle.destroy();
  }

  T result() const {
    if (handle.promise().exception)
      std::rethrow_exception(handle.promise().exception);
    return handle.promise().result;
  }
};