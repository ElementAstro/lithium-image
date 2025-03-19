// ThreadPool.cpp
#include "ThreadPool.hpp"
#include <spdlog/spdlog.h>

namespace {
    std::shared_ptr<spdlog::logger> thread_logger =
        spdlog::basic_logger_mt("ThreadPoolLogger", "logs/thread_pool.log");
}

DynamicThreadPool::DynamicThreadPool(size_t num_threads) : stop_(false) {
    thread_logger->info("Initializing thread pool with {} threads", num_threads);
    
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    
                    condition_.wait(lock, [this] { 
                        return stop_ || !tasks_.empty(); 
                    });
                    
                    if (stop_ && tasks_.empty()) {
                        return;
                    }
                    
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                
                try {
                    task();
                } catch (const std::exception& e) {
                    thread_logger->error("Exception in thread pool task: {}", e.what());
                } catch (...) {
                    thread_logger->error("Unknown exception in thread pool task");
                }
            }
        });
    }
}

DynamicThreadPool::~DynamicThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    
    condition_.notify_all();
    
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    
    thread_logger->info("Thread pool destroyed");
}

template<class F, class... Args>
auto DynamicThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result<F, Args...>::type> {
    
    using return_type = typename std::invoke_result<F, Args...>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future();
    
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        if (stop_) {
            throw std::runtime_error("Cannot enqueue on stopped ThreadPool");
        }
        
        tasks_.emplace([task]() { (*task)(); });
    }
    
    condition_.notify_one();
    return res;
}
